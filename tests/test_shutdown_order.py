"""Lifespan shutdown order: the off-machine audit sinks' final flush runs
from the STILL-OPEN audit database, after every request finalizer wrote its
END row, and before the audit database and then the vault close — bounded,
so a hanging sink never keeps the databases open (docs/resilience.md,
"Shutdown order").

The concrete sinks and the write-ahead audit log are llm-redact-pro; these
tests drive the Free lifespan with fakes registered through the plugin
registry (the test_audit_required.py pattern).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest
import uvicorn

import llm_redact.proxy as proxy_mod
import llm_redact.registry as registry_mod
import llm_redact.serving as serving_mod
from llm_redact.audit import AuditRecord
from llm_redact.config import AuditConfig, Config, ProviderConfig
from llm_redact.proxy import create_app
from llm_redact.registry import Registry

UPSTREAM = "http://upstream.test"


class SpoolAudit:
    """A write-ahead audit log holding its rows like a database would.

    Reading it once closed raises, like a closed sqlite connection."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.closed = False
        self.rows: list[tuple[str, object, int | None]] = []

    def _write(self, kind: str, token: object, status: int | None) -> None:
        if self.closed:
            raise RuntimeError("audit database closed")
        self.rows.append((kind, token, status))

    def begin(self, entry: AuditRecord) -> object:
        token = len(self.rows) + 1
        self._write("start", token, None)
        return token

    def finalize(self, token: object, entry: AuditRecord) -> None:
        self._write("end", token, entry.status)

    def record(self, entry: AuditRecord) -> None:
        self._write("row", None, entry.status)

    def pending(self) -> list[tuple[str, object, int | None]]:
        if self.closed:
            raise RuntimeError("audit database closed")
        return list(self.rows)

    def recent(self, limit: int) -> list[dict[str, object]]:
        return []

    def count(self) -> int:
        return len(self.rows)

    def close(self) -> None:
        self.events.append("audit.close")
        self.closed = True


class SpoolSink:
    """A sink whose final flush reads the audit database (pro's spool)."""

    batches_uploaded = 0
    rows_dropped = 0

    def __init__(self, name: str, spool: SpoolAudit, events: list[str]) -> None:
        self.name = name
        self.spool = spool
        self.events = events
        self.shipped: list[tuple[str, object, int | None]] = []
        self.run_cancelled = False

    def add(self, row: dict[str, Any]) -> None:
        pass  # spool mode: rows come from the database

    async def run(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.run_cancelled = True
            raise

    async def aclose(self) -> None:
        assert self.run_cancelled, "the flush loop must be stopped before the final flush"
        self.events.append(f"{self.name}.aclose")
        self.shipped = self.spool.pending()


class HangingSink(SpoolSink):
    """A final flush that never finishes (a store that never answers)."""

    def __init__(self, *args: Any, ignore_cancel: bool = False) -> None:
        super().__init__(*args)
        self.ignore_cancel = ignore_cancel
        self.release = asyncio.Event()
        self.ended = False

    async def aclose(self) -> None:
        self.events.append(f"{self.name}.aclose")
        try:
            while True:
                try:
                    await self.release.wait()
                    return
                except asyncio.CancelledError:
                    if not self.ignore_cancel:
                        raise
        finally:
            self.ended = True


class FailingSink(SpoolSink):
    async def aclose(self) -> None:
        self.events.append(f"{self.name}.aclose")
        raise OSError("https://bucket.example/?sig=SECRET")


def _register(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    make_sinks: Any = None,
) -> tuple[SpoolAudit, tuple[SpoolSink, SpoolSink]]:
    audit = SpoolAudit(events)
    sinks = (
        make_sinks(audit)
        if make_sinks is not None
        else (SpoolSink("s3", audit, events), SpoolSink("azure", audit, events))
    )
    reg = Registry()
    reg.build_audit = lambda cfg: audit
    reg.build_audit_sinks = lambda cfg: sinks
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return audit, sinks


def _app(events: list[str], transport: httpx.AsyncBaseTransport) -> Any:
    app = create_app(
        Config(
            providers={"anthropic": ProviderConfig(upstream_base_url=UPSTREAM)},
            audit=AuditConfig(enabled=True, required=True),
        ),
        upstream_transport=transport,
    )
    manager = app.state.proxy.vault_manager
    original = manager.close

    def close() -> None:
        events.append("vault.close")
        original()

    manager.close = close
    return app


def _answer(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200, json={"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
    )


_BODY = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "email jane.doe@corp.example please"}],
}


async def test_sinks_flush_from_the_open_database_before_it_and_the_vault_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    audit, sinks = _register(monkeypatch, events)
    app = _app(events, httpx.MockTransport(_answer))
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0)  # the flush loops start
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
            for _ in range(2):
                assert (await client.post("/v1/messages", json=_BODY)).status_code == 200
    # Both sinks read the database while it was open, and the last
    # request's END row was in it.
    expected = [("start", 1, None), ("end", 1, 200), ("start", 3, None), ("end", 3, 200)]
    assert audit.rows == expected
    for sink in sinks:
        assert sink.shipped == expected
    assert sorted(events[:2]) == ["azure.aclose", "s3.aclose"]
    assert events[2:] == ["audit.close", "vault.close"]


async def test_a_request_in_flight_at_shutdown_finalizes_before_the_sinks_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Through the real server: shutdown starts while a request waits on its
    # upstream. uvicorn stops accepting, lets the request finish (its END row
    # committed by the proxy's finalizer), and only then runs the lifespan
    # shutdown, whose sinks read that END row from the open database.
    events: list[str] = []
    audit, sinks = _register(monkeypatch, events)
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        events.append("upstream.answered")
        return _answer(request)

    app = _app(events, httpx.MockTransport(slow))
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="on", log_level="warning")
    )
    serving = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        reply = asyncio.create_task(client.post("/v1/messages", json=_BODY))
        await asyncio.wait_for(entered.wait(), 5)
        server.should_exit = True
        await asyncio.sleep(0.3)
        assert "audit.close" not in events  # the lifespan waits for the request
        release.set()
        assert (await asyncio.wait_for(reply, 5)).status_code == 200
    await asyncio.wait_for(serving, 5)
    assert audit.rows == [("start", 1, None), ("end", 1, 200)]
    for sink in sinks:
        assert sink.shipped == [("start", 1, None), ("end", 1, 200)]
    assert events[0] == "upstream.answered"
    assert sorted(events[1:3]) == ["azure.aclose", "s3.aclose"]
    assert events[3:] == ["audit.close", "vault.close"]


async def _serve_one_request(app: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0)  # the flush loops start
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://proxy.test"
            ) as client:
                assert (await client.post("/v1/messages", json=_BODY)).status_code == 200


@pytest.mark.parametrize("ignore_cancel", [False, True])
async def test_a_hanging_sink_never_keeps_the_databases_open(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, ignore_cancel: bool
) -> None:
    events: list[str] = []

    def sinks(audit: SpoolAudit) -> tuple[SpoolSink, SpoolSink]:
        return (
            HangingSink("s3", audit, events, ignore_cancel=ignore_cancel),
            SpoolSink("azure", audit, events),
        )

    audit, (hanging, healthy) = _register(monkeypatch, events, sinks)
    monkeypatch.setattr(proxy_mod, "_SINK_CLOSE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(proxy_mod, "_SINK_CANCEL_GRACE_SECONDS", 0.05)
    app = _app(events, httpx.MockTransport(_answer))
    try:
        await _serve_one_request(app, caplog)
    finally:
        # Never leave an uncancellable task behind for the loop teardown.
        hanging.release.set()
    # The healthy sink still flushed, the databases still closed.
    assert healthy.shipped == [("start", 1, None), ("end", 1, 200)]
    assert events[-2:] == ["audit.close", "vault.close"]
    messages = [record.getMessage() for record in caplog.records]
    assert (
        "1 audit sink(s) did not finish the final flush within 0.1 s; cancelled"
        " (unshipped spooled rows upload at the next start)"
    ) in messages
    abandoned = "1 audit sink(s) ignored the cancellation; abandoned"
    assert (abandoned in messages) is ignore_cancel
    assert hanging.ended or ignore_cancel
    if ignore_cancel:
        # The abandoned flush's outcome is still retrieved when it ends.
        for _ in range(100):
            if hanging.ended:
                break
            await asyncio.sleep(0.01)
        assert hanging.ended


async def test_a_failing_final_flush_is_contained_and_logged_by_type(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    events: list[str] = []

    def sinks(audit: SpoolAudit) -> tuple[SpoolSink, SpoolSink]:
        return FailingSink("s3", audit, events), SpoolSink("azure", audit, events)

    _register(monkeypatch, events, sinks)
    app = _app(events, httpx.MockTransport(_answer))
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0)  # the flush loops start
    assert events[-2:] == ["audit.close", "vault.close"]
    assert "an audit sink's final flush failed (OSError)" in caplog.text
    assert "SECRET" not in caplog.text


async def test_a_failing_cancelled_sink_flush_is_logged_by_type(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A flush that answers its cancellation with an error: the outcome is
    # retrieved (no "exception never retrieved"), logged by type.
    events: list[str] = []

    class RaisesOnCancel(HangingSink):
        async def aclose(self) -> None:
            try:
                await super().aclose()
            except asyncio.CancelledError:
                raise OSError("https://bucket.example/?sig=SECRET") from None

    def sinks(audit: SpoolAudit) -> tuple[SpoolSink, SpoolSink]:
        return RaisesOnCancel("s3", audit, events), SpoolSink("azure", audit, events)

    _register(monkeypatch, events, sinks)
    monkeypatch.setattr(proxy_mod, "_SINK_CLOSE_TIMEOUT_SECONDS", 0.05)
    app = _app(events, httpx.MockTransport(_answer))
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0)  # the flush loops start
    assert "an audit sink's final flush failed (OSError)" in caplog.text
    assert "SECRET" not in caplog.text
    assert events[-2:] == ["audit.close", "vault.close"]


async def test_a_dead_flush_loop_never_cuts_the_shutdown_short(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    events: list[str] = []

    class DeadLoop(SpoolSink):
        async def run(self) -> None:
            self.run_cancelled = True
            raise ValueError("https://bucket.example/?sig=SECRET")

    def sinks(audit: SpoolAudit) -> tuple[SpoolSink, SpoolSink]:
        return DeadLoop("s3", audit, events), SpoolSink("azure", audit, events)

    _register(monkeypatch, events, sinks)
    app = _app(events, httpx.MockTransport(_answer))
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0)
    assert "a background task had failed (ValueError)" in caplog.text
    assert "SECRET" not in caplog.text
    assert sorted(events[:2]) == ["azure.aclose", "s3.aclose"]
    assert events[2:] == ["audit.close", "vault.close"]


async def test_no_sinks_still_closes_the_audit_database_then_the_vault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _register(monkeypatch, events, lambda audit: (None, None))
    app = _app(events, httpx.MockTransport(_answer))
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0)
    assert events == ["audit.close", "vault.close"]


async def _start(server: uvicorn.Server) -> tuple[asyncio.Task[None], int]:
    serving = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    return serving, server.servers[0].sockets[0].getsockname()[1]


async def test_an_open_events_stream_never_holds_the_shutdown(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The dashboard's /__llm-redact/events stream never ends on its own, and
    # uvicorn waits for every open response before the lifespan shutdown:
    # an open dashboard kept a SIGTERM from ever reaching the sinks' final
    # flush and the database closes (the supervisor killed the process).
    # ProxyServer ends the streams first; the client sees a clean end.
    events: list[str] = []
    audit, sinks = _register(monkeypatch, events)
    app = _app(events, httpx.MockTransport(_answer))
    server = serving_mod.ProxyServer(
        uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="on", log_level="warning")
    )
    serving, port = await _start(server)
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        async with (
            httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client,
            client.stream("GET", "/__llm-redact/events") as stream,
        ):
            assert stream.status_code == 200
            lines = stream.aiter_lines()
            assert await asyncio.wait_for(anext(lines), 5) == ": connected"
            server.should_exit = True
            # The stream ends (no revocation logged) and the server exits.
            remaining = [line async for line in lines]
            assert all(line.startswith(":") or not line for line in remaining)
        await asyncio.wait_for(serving, 5)
    assert sorted(events[:2]) == ["azure.aclose", "s3.aclose"]
    assert events[2:] == ["audit.close", "vault.close"]
    assert "ended 1 events stream(s): the server is shutting down" in caplog.text
    assert "its access was revoked" not in caplog.text
    assert app.state.proxy.connections.closed == {}  # not a revocation


def test_run_server_runs_the_proxy_server_like_uvicorn_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[serving_mod.ProxyServer] = []

    def started(self: serving_mod.ProxyServer, sockets: object = None) -> None:
        ran.append(self)
        self.started = True
        raise KeyboardInterrupt  # uvicorn.run swallows it too

    monkeypatch.setattr(serving_mod.ProxyServer, "run", started)
    app = create_app(Config())
    serving_mod.run_server(app, host="127.0.0.1", port=0, access_log=False)
    assert ran[0].config.app is app and ran[0].config.access_log is False

    # A server that never started exits with uvicorn's startup-failure code.
    monkeypatch.setattr(serving_mod.ProxyServer, "run", lambda self, sockets=None: None)
    with pytest.raises(SystemExit) as excinfo:
        serving_mod.run_server(app, host="127.0.0.1", port=0)
    assert excinfo.value.code == serving_mod.STARTUP_FAILURE == 3


async def test_proxy_server_shutdown_without_a_proxy_app() -> None:
    # Any other ASGI app: nothing to end, uvicorn's shutdown unchanged.
    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        pass

    server = serving_mod.ProxyServer(
        uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="off", log_level="warning")
    )
    serving, _port = await _start(server)
    server.should_exit = True
    await asyncio.wait_for(serving, 5)


def test_the_shutdown_deadline_never_cuts_one_working_upload_short() -> None:
    # The final flush uploads the sink's in-memory START/AMEND rows first;
    # rows it cannot ship then are lost from the off-machine copy. A slow
    # but working store (one upload within the sink's own 30 s timeout)
    # must finish: only a stuck sink or a long backlog drain (whose spooled
    # rows wait in the database) may reach the core's deadline.
    assert proxy_mod._SINK_UPLOAD_TIMEOUT_SECONDS == 30.0  # the pro sinks' httpx timeout
    assert proxy_mod._SINK_CLOSE_TIMEOUT_SECONDS >= proxy_mod._SINK_UPLOAD_TIMEOUT_SECONDS + 10
