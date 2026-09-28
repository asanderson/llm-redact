"""The WebSocket relay (realtime.py): real sockets end to end.

ASGITransport cannot carry WebSocket upgrades, so — like the mTLS suite —
these tests run uvicorn on port 0 in a thread and a fake `websockets`
upstream in the test's own event loop. P11.2 pins the PLUMBING contract:
byte-identical pass-through, header/subprotocol forwarding, close-code
mirroring, refusal paths, and the per-connection record_request row.
"""

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import uvicorn
import websockets
import websockets.asyncio.server

from llm_redact import realtime
from llm_redact.config import Config, ProviderConfig
from llm_redact.proxy import create_app

pytestmark = pytest.mark.asyncio


class FakeUpstream:
    """A websockets server that records requests and echoes frames."""

    def __init__(self) -> None:
        self.server: websockets.asyncio.server.Server | None = None
        self.port = 0
        self.headers: list[dict[str, str]] = []
        self.paths: list[str] = []
        self.received: list[str | bytes] = []
        self.close_with: tuple[int, str] | None = None
        self.greeting: str | None = None

    async def _handler(self, connection: Any) -> None:
        self.paths.append(connection.request.path)
        self.headers.append({k.lower(): v for k, v in connection.request.headers.items()})
        if self.greeting is not None:
            await connection.send(self.greeting)
        if self.close_with is not None:
            await connection.close(*self.close_with)
            return
        async for message in connection:
            self.received.append(message)
            await connection.send(message)  # echo

    async def __aenter__(self) -> "FakeUpstream":
        self.server = await websockets.serve(
            self._handler, "127.0.0.1", 0, subprotocols=None, select_subprotocol=self._select
        )
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    @staticmethod
    def _select(connection: Any, subprotocols: Any) -> Any:
        return subprotocols[0] if subprotocols else None

    async def __aexit__(self, *exc: Any) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()


@contextlib.contextmanager
def _proxy(config: Config) -> Iterator[str]:
    server = uvicorn.Server(
        uvicorn.Config(create_app(config), host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _config(upstream_port: int, provider: str = "openai", **provider_kwargs: Any) -> Config:
    return Config(
        providers={
            **Config().providers,
            provider: ProviderConfig(f"http://127.0.0.1:{upstream_port}", **provider_kwargs),
        }
    )


@contextlib.asynccontextmanager
async def _relay_setup(
    provider: str = "openai", **provider_kwargs: Any
) -> AsyncIterator[tuple[FakeUpstream, str]]:
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, provider, **provider_kwargs)) as proxy_host:
            yield fake, proxy_host


async def test_passthrough_round_trip_headers_and_query() -> None:
    async with (
        _relay_setup() as (fake, proxy_host),
        websockets.connect(
            f"ws://{proxy_host}/v1/realtime?model=gpt-realtime",
            additional_headers={"Authorization": "Bearer sk-test-123"},
        ) as client,
    ):
        # JSON events are parsed and re-serialized (redaction walks them);
        # non-JSON text and opaque binary must stay byte-identical.
        await client.send('{"type":"session.update","session":{}}')
        assert json.loads(await client.recv()) == {"type": "session.update", "session": {}}
        await client.send("not-json ping")
        assert await client.recv() == "not-json ping"
        await client.send(b"\x00\x01binary-opaque")
        assert await client.recv() == b"\x00\x01binary-opaque"
    # The upstream saw the same path AND the raw query, and auth passed
    # through untouched.
    assert fake.paths == ["/v1/realtime?model=gpt-realtime"]
    assert fake.headers[0]["authorization"] == "Bearer sk-test-123"
    assert json.loads(fake.received[0]) == {"type": "session.update", "session": {}}
    assert fake.received[1:] == ["not-json ping", b"\x00\x01binary-opaque"]


async def test_subprotocol_negotiation_forwarded() -> None:
    async with (
        _relay_setup() as (fake, proxy_host),
        websockets.connect(
            f"ws://{proxy_host}/v1/realtime",
            subprotocols=[
                websockets.Subprotocol("realtime"),
                websockets.Subprotocol("openai-insecure-api-key.sk-test"),
            ],
        ) as client,
    ):
        assert client.subprotocol == "realtime"
    assert "realtime" in fake.headers[0]["sec-websocket-protocol"]


async def test_upstream_close_code_mirrored() -> None:
    async with _relay_setup() as (fake, proxy_host):
        fake.close_with = (4002, "policy violation")
        fake.greeting = '{"type":"error"}'
        async with websockets.connect(f"ws://{proxy_host}/v1/realtime") as client:
            assert await client.recv() == '{"type":"error"}'
            with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                await client.recv()
            assert closed.value.rcvd is not None
            assert closed.value.rcvd.code == 4002
            assert closed.value.rcvd.reason == "policy violation"


async def test_unknown_path_and_reserved_path_refused() -> None:
    async with _relay_setup() as (fake, proxy_host):
        for path in ("/v1/other", "/__llm-redact/status"):
            async with websockets.connect(f"ws://{proxy_host}{path}") as client:
                with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                    await client.recv()
                assert closed.value.rcvd is not None
                assert closed.value.rcvd.code == 1011
    assert fake.paths == []  # nothing ever reached the upstream


async def test_disabled_provider_refused_fail_closed() -> None:
    async with (
        _relay_setup(enabled=False) as (fake, proxy_host),
        websockets.connect(f"ws://{proxy_host}/v1/realtime") as client,
    ):
        with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
            await client.recv()
        assert closed.value.rcvd is not None
        assert closed.value.rcvd.code == 1011
        assert "disabled" in closed.value.rcvd.reason
    assert fake.paths == []


async def test_missing_websockets_package_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(realtime, "websockets_available", lambda: False)
    async with (
        _relay_setup() as (fake, proxy_host),
        websockets.connect(f"ws://{proxy_host}/v1/realtime") as client,
    ):
        with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
            await client.recv()
        assert closed.value.rcvd is not None
        assert "realtime" in closed.value.rcvd.reason
    assert fake.paths == []


async def test_connection_recorded_metadata_only() -> None:
    async with _relay_setup() as (fake, proxy_host):
        async with websockets.connect(f"ws://{proxy_host}/v1/realtime") as client:
            await client.send('{"type":"noop"}')
            await client.recv()
        # The relay records at close; give the finally block a beat.
        row = None
        async with httpx.AsyncClient(base_url=f"http://{proxy_host}") as http:
            for _ in range(50):
                entries = (await http.get("/__llm-redact/recent")).json()["entries"]
                if entries:
                    row = entries[0]
                    break
                await asyncio.sleep(0.05)
    assert row is not None
    assert row["method"] == "WS"
    assert row["path"] == "/v1/realtime"
    assert row["provider"] == "openai"
    assert row["streamed"] is True
    assert row["status"] == 101


async def test_access_gate_on_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    # The access seam on the WS surface: a refusal is accept-then-close with
    # the gate's message before any upstream contact; an admitted connection
    # is attributed, and no x-llm-redact-* header ever reaches the upstream.
    import llm_redact.registry as registry_mod
    from license_fixtures import resolved
    from llm_redact.registry import Registry
    from test_access_seam import FakeGate

    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("team")
    reg.build_access_gate = lambda config, license: FakeGate(strict=True)
    monkeypatch.setattr(registry_mod, "_registry", reg)
    async with _relay_setup() as (fake, proxy_host):
        async with websockets.connect(f"ws://{proxy_host}/v1/realtime") as client:
            with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                await client.recv()
            assert closed.value.rcvd is not None
            assert closed.value.rcvd.code == 1011
            assert "a test key is required" in closed.value.rcvd.reason
        assert fake.paths == []
        async with websockets.connect(
            f"ws://{proxy_host}/v1/realtime",
            additional_headers={"x-test-key": "good", "x-llm-redact-other": "v"},
        ) as client:
            await client.send('{"type": "noop"}')
            assert json.loads(await client.recv()) == {"type": "noop"}
        async with httpx.AsyncClient() as http:
            rows: list[dict[str, Any]] = []
            for _ in range(50):
                rows = (await http.get(f"http://{proxy_host}/__llm-redact/recent")).json()[
                    "entries"
                ]
                if rows:
                    break
                await asyncio.sleep(0.05)
    assert fake.paths == ["/v1/realtime"]
    assert "x-test-key" not in fake.headers[0]
    assert "x-llm-redact-other" not in fake.headers[0]
    assert rows[0]["method"] == "WS" and rows[0]["user"] == "ada via websocket"


@pytest.mark.parametrize("refusal", ["a key is required", None])
async def test_identity_path_on_websocket_is_refused_and_never_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, refusal: str | None
) -> None:
    # A WS upgrade under /u/<key>/ (an SDK deriving its realtime URL from a
    # /u/<key> base URL) that the gate leaves in place: refused, never
    # forwarded, and the key never reaches a log line — whether the gate
    # admitted or refused the connection.
    import logging

    import llm_redact.registry as registry_mod
    from license_fixtures import resolved
    from llm_redact.plugin_api import Admission
    from llm_redact.registry import Registry
    from test_access_seam import FakeGate

    class PathKeepingGate(FakeGate):
        def admit(self, conn: Any, surface: str) -> Admission:
            return Admission(refusal=refusal)

    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("team")
    reg.build_access_gate = lambda config, license: PathKeepingGate()
    monkeypatch.setattr(registry_mod, "_registry", reg)
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _relay_setup() as (fake, proxy_host):
        async with websockets.connect(f"ws://{proxy_host}/u/lrk_SUPERSECRET/v1/realtime") as client:
            with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                await client.recv()
            assert closed.value.rcvd is not None and closed.value.rcvd.code == 1011
        assert fake.paths == []
    assert "SUPERSECRET" not in caplog.text


async def test_identity_auth_provider_refused_never_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[providers.azure] auth = "identity": the realtime relay cannot sign a
    WebSocket handshake with the proxy's cloud identity, and forwarding the
    client's credential (or none) would break the configured contract —
    accept-then-close 1011 with the reason, the upstream never dialled."""
    import llm_redact.registry as registry_mod
    from llm_redact.registry import Registry

    class _Auth:
        async def authorize(
            self, method: str, url: str, headers: list[tuple[str, str]], body: bytes
        ) -> list[tuple[str, str]]:
            raise AssertionError("never called for a WebSocket")

        def close(self) -> None:
            pass

    reg = Registry()
    reg.build_upstream_auth = lambda name, provider: (
        _Auth() if provider.auth == "identity" else None
    )
    monkeypatch.setattr(registry_mod, "_registry", reg)
    async with (
        _relay_setup("azure", auth="identity") as (fake, proxy_host),
        websockets.connect(
            f"ws://{proxy_host}/openai/realtime?api-version=2025-04-01&deployment=d",
            additional_headers={"api-key": "client-key"},
        ) as client,
    ):
        with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
            await client.recv()
        assert closed.value.rcvd is not None
        assert closed.value.rcvd.code == 1011
        assert 'auth = "identity"' in closed.value.rcvd.reason
    assert fake.paths == []
