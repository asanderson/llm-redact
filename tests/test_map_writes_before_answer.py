"""``[vault] map_writes``: when the durable maps' background writes must land.

The durable map writes (Responses chains, stored-object owner records, Live
resumption handles) run on the vault manager's background writer thread
(``vault_writer.MapWriter``). THIS process answers lookups from the writer's
overlay at once; another replica sharing the vault reads a record only once
its write landed. ``map_writes = "before_answer"`` (the default for the
shared-database backends) restores CROSS-REPLICA read-your-writes: the
proxy waits — off the event loop, bounded — for the writes an answer caused
before it sends the bytes that carry the recorded id:

- a buffered answer before it is returned;
- a streamed answer before the event (SSE), line (NDJSON) or frame
  (eventstream) whose bookkeeping queued the write — so the id never
  reaches the client before its record landed;
- realtime: a server frame whose observation queued a write (a Live handle)
  is held until it landed.

Pinned through the real app: two replicas (two ``create_app`` instances)
sharing ONE RDBMS vault (the stdlib sqlite3-as-DB-API backend the battery
uses). ``background`` keeps the old behaviour exactly — pinned by the same
scenario, which then reaches the other replica before its write.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import pytest
import uvicorn
import websockets

import llm_redact.registry as registry_mod
from llm_redact.config import (
    MAP_WRITES_MODES,
    RDBMS_BACKENDS,
    Config,
    ConfigError,
    ProviderConfig,
    RdbmsConfig,
    VaultConfig,
    map_writes_mode,
    parse_config,
)
from llm_redact.config_write import emit_config_toml as emit_config
from llm_redact.proxy import MAP_WRITE_WAIT_STAGE, create_app
from llm_redact.registry import Registry
from llm_redact.vault import SqliteVaultManager, build_vault_manager
from llm_redact.vault_rdbms import RdbmsStore, RdbmsVaultManager
from llm_redact.vault_writer import MISS, MapWrite, MapWriter, awaited_writes
from test_realtime_identity import GEMINI_LIVE
from test_realtime_server_frame import Replying
from test_vault_writer import ChainRouter, Responses, _gate, _paused

UPSTREAM = "https://upstream.test"
EMAIL_A = "ada@corp.example"
EMAIL_B = "bob@corp.example"
WAIT_MESSAGE = "sent before its vault map writes landed"


# --- configuration -------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["memory", "sqlite", *RDBMS_BACKENDS])
def test_the_default_follows_the_backend(backend: str) -> None:
    expected = "before_answer" if backend in RDBMS_BACKENDS else "background"
    assert map_writes_mode(VaultConfig(backend=backend)) == expected
    for mode in MAP_WRITES_MODES:  # an explicit choice always wins
        assert map_writes_mode(VaultConfig(backend=backend, map_writes=mode)) == mode


def test_the_setting_parses_and_round_trips() -> None:
    for mode in MAP_WRITES_MODES:
        config = parse_config({"vault": {"backend": "sqlite", "map_writes": mode}}, "t")
        assert config.vault.map_writes == mode
        assert f'map_writes = "{mode}"' in emit_config(config)
        assert parse_config_text(emit_config(config)).vault.map_writes == mode
    unset = parse_config({"vault": {"backend": "sqlite"}}, "t")
    assert unset.vault.map_writes is None
    # Unset stays unset (it keeps following the backend), never pinned.
    assert "map_writes" not in emit_config(unset)
    with pytest.raises(ConfigError, match=r"\[vault\] map_writes must be one of"):
        parse_config({"vault": {"map_writes": "eventually"}}, "t")


def parse_config_text(text: str) -> Config:
    import tomllib

    return parse_config(tomllib.loads(text), "emitted")


def test_a_reload_reports_the_setting_restart_only(tmp_path: Path) -> None:
    app = create_app(Config(vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db"))))
    state = app.state.proxy
    changed = Config(
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db"), map_writes="before_answer")
    )
    assert "vault" in state.apply_config(changed)
    assert state.map_writes == "background"  # unchanged until a restart
    state.vault_manager.close()


@pytest.mark.parametrize("mode", [None, *MAP_WRITES_MODES])
async def test_status_reports_synchronous_without_a_background_writer(mode: str | None) -> None:
    # The in-memory manager (like a third-party one without
    # write_maps_in_background) writes its maps synchronously: nothing is
    # ever awaited, so /status never claims a mode the proxy does not run.
    app = create_app(Config(vault=VaultConfig(map_writes=mode)))
    state = app.state.proxy
    assert state.awaits_map_writes is False
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["vault"]["map_writes"] == "synchronous"


def test_doctor_shows_the_effective_mode(tmp_path: Path) -> None:
    from llm_redact.doctor_cli import _check_vault, _Report

    lines: dict[str, list[str]] = {}
    for name, vault in {
        "default": VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
        "set": VaultConfig(
            backend="sqlite", path=str(tmp_path / "v.db"), map_writes="before_answer"
        ),
        "memory": VaultConfig(),
    }.items():
        report = _Report(json_mode=True)
        _check_vault(report, Config(vault=vault))
        lines[name] = [
            f"{row['level']} {row['message']}"
            for row in report.rows
            if "map_writes" in row["message"]
        ]
    assert lines["default"] == [
        "PASS map_writes = background (default): durable map writes land after the answer:"
        " this process answers at once, another replica sharing the vault reads the"
        " record once its write landed"
    ]
    assert lines["set"] == [
        "PASS map_writes = before_answer (set): an answer waits (bounded) for its durable"
        " map writes, so a follow-up reaching any replica sharing the vault finds the record"
    ]
    assert lines["memory"] == []  # the in-memory vault keeps no durable map


@pytest.mark.parametrize(
    ("vault", "line"),
    [
        (
            {"backend": "postgresql", "entries": 3, "map_writes": "before_answer"},
            "vault: postgresql (3 entries, map writes: before_answer)",
        ),
        # An older proxy omits it: nothing invented.
        ({"backend": "sqlite", "entries": 1}, "vault: sqlite (1 entries)"),
    ],
)
def test_status_prints_the_mode_when_reported(
    vault: dict[str, Any],
    line: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from test_cli_status import _full_status_payload, _status_run

    payload = _full_status_payload()
    payload["vault"] = vault
    _status_run(monkeypatch, payload)
    assert line in capsys.readouterr().out


# --- the writer's completion futures ----------------------------------------------------


class _Faults:
    def __init__(self) -> None:
        self.counter: Counter[str] | None = Counter()
        self.failing = False
        self.stage = "response_id"
        self.errors: list[Exception] = []

    def failed(self, exc: Exception) -> None:
        self.errors.append(exc)

    def succeeded(self) -> None:
        pass


def _write(
    rows: list[str], value: str = "row", *, fails: bool = False, faults: _Faults | None = None
) -> MapWrite:
    def run(conn: Any) -> None:
        if fails:
            raise OSError("disk I/O error")
        rows.append(value)

    def erase(conn: Any) -> None:
        rows.remove(value)

    return MapWrite("s", run, erase, faults or _Faults())


def _writer(**kwargs: Any) -> MapWriter:
    return MapWriter(lambda: object(), **kwargs)


def test_a_write_queued_inside_awaited_writes_carries_a_future_set_once_it_landed() -> None:
    writer = _writer()
    rows: list[str] = []
    seen: list[tuple[list[str], bool]] = []
    released = threading.Event()

    def on_release(done: Future[None]) -> None:
        # What a waiter woken now finds: the row written and the overlay
        # settled (the database answers for the key).
        seen.append((list(rows), writer.verdict(("row", "r1")) is MISS))
        released.set()

    with _paused(writer):
        with awaited_writes() as pending:
            write = _write(rows)
            writer.submit(write, sets=(("row", "r1"),))
        assert pending == [write.landed]
        future = pending[0]
        assert not future.done()
        future.add_done_callback(on_release)
    assert released.wait(10)
    assert future.result(0) is None
    assert seen == [(["row"], True)]
    writer.close()


def test_outside_awaited_writes_no_future_is_made() -> None:
    writer = _writer()
    rows: list[str] = []
    with awaited_writes() as pending:
        pass
    write = _write(rows)
    writer.submit(write)
    assert write.landed is None and pending == []
    assert writer.drain(10) == 0
    writer.close()


def test_a_failed_write_releases_its_waiter() -> None:
    writer = _writer()
    faults = _Faults()
    with awaited_writes() as pending:
        writer.submit(_write([], fails=True, faults=faults))
    assert pending[0].result(10) is None
    assert writer.drain(10) == 0
    assert [type(exc) for exc in faults.errors] == [OSError]
    writer.close()


def test_a_write_not_queued_carries_nothing_to_wait_for() -> None:
    writer = _writer(max_pending=1)
    with _paused(writer), awaited_writes() as pending:
        writer.submit(_write([], "a"))
        writer.submit(_write([], "b"))  # the queue is full: overflow
    assert len(pending) == 1
    writer.close()
    with awaited_writes() as after_close:
        writer.submit(_write([], "c"))  # closed: counted, not queued
    assert after_close == []


def test_close_releases_what_it_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_redact.vault_writer as writer_mod

    monkeypatch.setattr(writer_mod, "STOP_JOIN_SECONDS", 0.05)
    writer = _writer()
    gate = _gate(writer)  # the first write hangs in flight
    try:
        with awaited_writes() as pending:
            writer.submit(_write([], "a"))
            deadline = time.monotonic() + 10
            while writer._in_flight is None:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            writer.submit(_write([], "b"))
        assert writer.close(0.05) == 2
        assert all(future.done() for future in pending)
    finally:
        gate.set()


def test_a_dying_writer_thread_releases_the_write_it_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    writer = _writer()

    class Died(BaseException):
        pass

    def dies(conn: Any) -> None:
        raise Died

    with awaited_writes() as pending:
        writer.submit(MapWrite("s", dies, dies, _Faults()))
    assert pending[0].result(10) is None
    writer.close()


def test_writes_queued_behind_a_dying_writer_thread_land_without_another_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The thread dies (a BaseException) while more writes are queued: it
    # starts its successor, so their waiters are released when they land —
    # not left pending until the wait bound, or a submit that may never come.
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    writer = _writer()
    rows: list[str] = []
    running = threading.Event()
    release = threading.Event()

    class Died(BaseException):
        pass

    def dies(conn: Any) -> None:
        running.set()
        release.wait(10)
        raise Died

    with awaited_writes() as pending:
        writer.submit(MapWrite("s", dies, dies, _Faults()))
        assert running.wait(10)
        writer.submit(_write(rows, "after"))
    release.set()
    assert pending[1].result(10) is None
    assert rows == ["after"]
    writer.close()


def test_a_waiter_that_gave_up_never_breaks_the_writer() -> None:
    writer = _writer()
    rows: list[str] = []
    with _paused(writer), awaited_writes() as pending:
        write = _write(rows)
        writer.submit(write)
        assert pending[0].cancel()  # the proxy stopped waiting (its timeout)
    assert writer.drain(10) == 0
    assert rows == ["row"]
    write.release()  # a second release is a no-op too
    writer.close()


# --- two replicas sharing one RDBMS vault ------------------------------------------------


class RecordingChain(ChainRouter):
    """The chain router, remembering every session it resolved."""

    def __init__(self, durable_lookup: Callable[[str], str | None] | None) -> None:
        super().__init__(durable_lookup)
        self.resolved: list[str] = []

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        session = super().resolve(adapter_name, method, path, body)
        self.resolved.append(session)
        return session


def _install(
    monkeypatch: pytest.MonkeyPatch, router: Callable[..., Any] = RecordingChain
) -> list[Any]:
    """A registry building, per app, its OWN vault manager over the configured
    database (an RDBMS one for the DB-API backend: the replicas' shared
    vault) and a chain router reading that manager's durable lookup."""
    routers: list[Any] = []
    reg = Registry()

    def build_router(config: VaultConfig, **kw: Any) -> Any:
        made = router(kw.get("durable_lookup"))
        routers.append(made)
        return made

    def build_manager(config: VaultConfig) -> Any:
        if config.backend == "dbapi":
            return RdbmsVaultManager(RdbmsStore(config, None))
        return build_vault_manager(config)

    reg.build_session_router = build_router
    reg.build_vault_manager = build_manager
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return routers


def _shared_vault(tmp_path: Path, map_writes: str | None = None) -> VaultConfig:
    return VaultConfig(
        backend="dbapi",
        session="main",
        rdbms=RdbmsConfig(dsn=str(tmp_path / "shared.db"), module="sqlite3"),
        map_writes=map_writes,
    )


def _replica(
    vault: VaultConfig, upstream: Callable[[httpx.Request], httpx.Response], **providers: str
) -> Any:
    configured = {name: ProviderConfig(url) for name, url in providers.items()}
    config = Config(
        providers={**Config().providers, "openai": ProviderConfig(UPSTREAM), **configured},
        vault=vault,
    )
    return create_app(config, upstream_transport=httpx.MockTransport(upstream))


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


@contextlib.contextmanager
def _replicas(count: int, *args: Any, **kwargs: Any) -> Iterator[list[Any]]:
    apps = [_replica(*args, **kwargs) for _ in range(count)]
    try:
        yield apps
    finally:
        for app in apps:
            manager = app.state.proxy.vault_manager
            manager.drain_map_writes(10)
            manager.close()


@contextlib.contextmanager
def _reader(vault: VaultConfig) -> Iterator[RdbmsVaultManager]:
    """A third instance over the shared vault, no writer of its own: what
    any other replica's database read answers."""
    manager = RdbmsVaultManager(RdbmsStore(vault, None))
    try:
        yield manager
    finally:
        manager.close()


def _first(body: str) -> dict[str, Any]:
    return {"model": "m", "input": body}


def _next(previous: str, body: str) -> dict[str, Any]:
    return {"model": "m", "previous_response_id": previous, "input": body}


@pytest.mark.parametrize("mode", ["before_answer", "background"])
async def test_a_chain_continued_on_another_replica_right_after_a_buffered_answer(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    routers = _install(monkeypatch)
    upstream = Responses()
    # Unset is "before_answer" on a shared-database backend.
    vault = _shared_vault(tmp_path, None if mode == "before_answer" else "background")
    with _replicas(2, vault, upstream) as (a, b), _reader(vault) as reader:
        assert a.state.proxy.map_writes == mode
        gate = _gate(a.state.proxy.vault_manager._maps)  # replica A's database is slow
        if mode == "before_answer":
            threading.Timer(0.3, gate.set).start()
        try:
            async with _client(a) as client_a, _client(b) as client_b:
                first = await client_a.post("/v1/responses", json=_first(f"mail {EMAIL_A}"))
                assert first.json()["id"] == "resp_1"
                landed = reader.lookup_response_session("resp_1")
                second = await client_b.post(
                    "/v1/responses", json=_next("resp_1", f"and {EMAIL_B}")
                )
                assert second.status_code == 200
        finally:
            gate.set()
    assert "«EMAIL_001»" in upstream.bodies[0]
    if mode == "before_answer":
        # The answer left replica A only once its record landed: replica B
        # continues the chain in its own session — bob is that session's
        # SECOND address.
        assert landed == "conv-a"
        assert routers[1].resolved == ["conv-a"]
        assert "«EMAIL_002»" in upstream.bodies[1]
    else:
        # Background: the answer left at once; replica B did not find the
        # chain yet (an orphan here; refused or sealed by a real router).
        assert landed is None
        assert routers[1].resolved == ["orphan:resp_1"]
        assert "«EMAIL_001»" in upstream.bodies[1]
    assert EMAIL_B not in upstream.bodies[1]
    assert MAP_WRITE_WAIT_STAGE not in a.state.proxy.bookkeeping_errors


class _Raw(NamedTuple):
    status: int
    chunks: list[bytes]


async def _post_raw(app: Any, path: str, body: Any, on_body: Callable[[bytes], None]) -> _Raw:
    """POST through the app's raw ASGI interface, calling ``on_body`` for
    each body message AS IT IS SENT (ASGITransport would join them)."""
    data = json.dumps(body).encode()
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8787),
    }
    received = False
    status = 0
    chunks: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": data, "more_body": False}
        await asyncio.Event().wait()  # never disconnects
        raise AssertionError  # pragma: no cover

    async def send(message: dict[str, Any]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body" and message.get("body"):
            chunks.append(message["body"])
            on_body(message["body"])

    await asyncio.wait_for(app(scope, receive, send), 30)
    return _Raw(status, chunks)


def _sse(*events: tuple[str, dict[str, Any]]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


class StreamedResponses(Responses):
    """The Responses upstream answering as a stream: the id rides the FIRST
    event (``response.created``), the text after it."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content.decode())
        rid = f"resp_{len(self.bodies)}"
        body = _sse(
            ("response.created", {"type": "response.created", "response": {"id": rid}}),
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "item_id": "i", "delta": "ok"},
            ),
            ("response.completed", {"type": "response.completed", "response": {"id": rid}}),
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


@pytest.mark.parametrize("mode", ["before_answer", "background"])
async def test_a_streamed_answer_names_its_id_only_once_the_record_landed(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    routers = _install(monkeypatch)
    upstream = StreamedResponses()
    vault = _shared_vault(tmp_path, mode)
    with _replicas(2, vault, upstream) as (a, b), _reader(vault) as reader:
        gate = _gate(a.state.proxy.vault_manager._maps)
        if mode == "before_answer":
            threading.Timer(0.3, gate.set).start()
        seen: list[str | None] = []

        def on_body(chunk: bytes) -> None:
            # The moment the id reaches the client: does another replica
            # already read its record?
            if b"resp_1" in chunk and not seen:
                assert chunk.startswith(b"event: response.created")
                seen.append(reader.lookup_response_session("resp_1"))

        try:
            answer = await _post_raw(a, "/v1/responses", _first(f"mail {EMAIL_A}"), on_body)
            assert answer.status == 200 and len(answer.chunks) >= 2
            async with _client(b) as client_b:
                second = await client_b.post(
                    "/v1/responses", json=_next("resp_1", f"and {EMAIL_B}")
                )
                assert second.status_code == 200
        finally:
            gate.set()
    if mode == "before_answer":
        assert seen == ["conv-a"]
        assert routers[1].resolved == ["conv-a"]
        assert "«EMAIL_002»" in upstream.bodies[1]
    else:
        assert seen == [None]
        assert routers[1].resolved == ["orphan:resp_1"]


class EchoingChat(Responses):
    """Responses as before; a chat completion answers with the user's
    (redacted) message — a placeholder the proxy restores."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/chat/completions":
            return super().__call__(request)
        text = json.loads(request.content)["messages"][-1]["content"]
        message = {"role": "assistant", "content": text}
        return httpx.Response(200, json={"id": "chatcmpl-1", "choices": [{"message": message}]})


async def test_a_held_writer_delays_only_the_answers_that_recorded_something(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch)
    upstream = EchoingChat()
    with _replicas(1, _shared_vault(tmp_path), upstream) as (a,):
        gate = _gate(a.state.proxy.vault_manager._maps)  # held until released below
        try:
            async with _client(a) as client:
                chain = asyncio.create_task(client.post("/v1/responses", json=_first("hi")))
                await asyncio.sleep(0.2)
                assert not chain.done()  # waits for its record
                other = await asyncio.wait_for(
                    client.post(
                        "/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": EMAIL_A}]},
                    ),
                    5,
                )
                # Recorded nothing: never held — and restored while the
                # other answer waited.
                assert other.status_code == 200
                assert other.json()["choices"][0]["message"]["content"] == EMAIL_A
                assert not chain.done()
                gate.set()
                answer = await asyncio.wait_for(chain, 10)
                assert answer.status_code == 200
                recent = (await client.get("/__llm-redact/recent")).json()
        finally:
            gate.set()
        assert a.state.proxy.bookkeeping_errors == {}
    # Each row counts its OWN restorations: the waiting answer's share was
    # taken before its wait, not diffed across the other request's.
    rows = {row["path"]: row["rehydrations"] for row in recent["entries"]}
    assert rows == {"/v1/chat/completions": {"EMAIL": 1}, "/v1/responses": {}}


async def test_a_wait_past_its_bound_sends_anyway_and_is_counted_once_per_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _install(monkeypatch)
    upstream = Responses()
    caplog.set_level(logging.INFO, logger="llm_redact")
    with _replicas(1, _shared_vault(tmp_path), upstream) as (a,):
        state = a.state.proxy
        state.map_write_wait_seconds = 0.05
        gate = _gate(state.vault_manager._maps)
        try:
            async with _client(a) as client:
                for text in ("one", "two"):
                    answer = await asyncio.wait_for(
                        client.post("/v1/responses", json=_first(text)), 5
                    )
                    assert answer.status_code == 200
                assert state.bookkeeping_errors == {MAP_WRITE_WAIT_STAGE: 2}
                status = (await client.get("/__llm-redact/status")).json()
                assert status["bookkeeping_errors_total"] == {MAP_WRITE_WAIT_STAGE: 2}
                assert status["vault"]["map_writes"] == "before_answer"
                gate.set()
                assert await asyncio.to_thread(state.vault_manager.drain_map_writes, 10) == 0
                state.map_write_wait_seconds = 10
                assert (await client.post("/v1/responses", json=_first("three"))).status_code == 200
        finally:
            gate.set()
    # Once per episode, then recovery; never an id.
    assert caplog.text.count(WAIT_MESSAGE) == 1
    assert "POST /v1/responses -> sent before its vault map writes landed" in caplog.text
    assert "vault map writes land in time again" in caplog.text
    assert "resp_" not in caplog.text
    assert state.bookkeeping_errors == {MAP_WRITE_WAIT_STAGE: 2}


async def test_a_stuck_writer_costs_one_wait_per_episode_not_one_per_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A hung database (a network blackhole: no driver timeout) holds the
    # writer on one write and queues every later one behind it. The first
    # answer pays the bound; while no write has left the writer since,
    # waiting again is futile (one writer, in submission order), so later
    # answers are sent at once — still counted. A write leaving the writer
    # ends the episode: the next answer waits again.
    _install(monkeypatch)
    caplog.set_level(logging.INFO, logger="llm_redact")
    with _replicas(1, _shared_vault(tmp_path), Responses()) as (a,):
        state = a.state.proxy
        state.map_write_wait_seconds = 1.0
        gate = _gate(state.vault_manager._maps)
        took: list[float] = []
        try:
            async with _client(a) as client:
                for text in ("one", "two", "three"):
                    started = time.monotonic()
                    answer = await asyncio.wait_for(
                        client.post("/v1/responses", json=_first(text)), 10
                    )
                    assert answer.status_code == 200
                    took.append(time.monotonic() - started)
                assert state.bookkeeping_errors == {MAP_WRITE_WAIT_STAGE: 3}
                gate.set()
                assert await asyncio.to_thread(state.vault_manager.drain_map_writes, 10) == 0
                # The episode ended: the next answer waits for its write again.
                state.map_write_wait_seconds = 10
                assert (await client.post("/v1/responses", json=_first("four"))).status_code == 200
                assert state.vault_manager.lookup_response_session("resp_4") is not None
        finally:
            gate.set()
    assert took[0] >= 0.95
    assert max(took[1:]) < 0.5, took
    assert caplog.text.count(WAIT_MESSAGE) == 1
    assert "vault map writes land in time again" in caplog.text
    assert state.bookkeeping_errors == {MAP_WRITE_WAIT_STAGE: 3}


async def test_a_failed_write_releases_the_answer_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch)
    upstream = Responses()
    with _replicas(1, _shared_vault(tmp_path), upstream) as (a,):
        state = a.state.proxy
        state.map_write_wait_seconds = 60

        def broken() -> Any:
            raise OSError("database unreachable")

        state.vault_manager._maps._open_connection = broken
        async with _client(a) as client:
            answer = await asyncio.wait_for(client.post("/v1/responses", json=_first("hi")), 5)
        assert answer.status_code == 200
        deadline = time.monotonic() + 10
        while state.bookkeeping_errors != {"response_id": 1}:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        # Unknown, as a failed synchronous write read; the wait was released.
        assert state.vault_manager.lookup_response_session("resp_1") is None


class ObjectObserver(RecordingChain):
    """The chain router plus a response observer that records, from a
    streamed NDJSON answer's final line, the object it names (as
    llm-redact-pro records a compaction item) in the durable map."""

    def __init__(self, durable_lookup: Callable[[str], str | None] | None) -> None:
        super().__init__(durable_lookup)
        self.manager = getattr(durable_lookup, "__self__", None)

    def response_observer(self, context: Any) -> Callable[[Any], None]:
        session_id = context.session_id

        def observe(payload: Any) -> None:
            if isinstance(payload, dict) and payload.get("done") is True:
                self.manager.record_object_session(payload["object"], session_id)

        return observe


def _ollama(request: httpx.Request) -> httpx.Response:
    lines = [
        {"model": "m", "message": {"role": "assistant", "content": "ok"}, "done": False},
        {
            "model": "m",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "object": "obj-7",
        },
    ]
    body = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
    return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})


async def test_an_ndjson_line_whose_observation_recorded_waits_for_the_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, ObjectObserver)
    vault = _shared_vault(tmp_path)
    with _replicas(1, vault, _ollama, ollama=UPSTREAM) as (a,), _reader(vault) as reader:
        gate = _gate(a.state.proxy.vault_manager._maps)
        threading.Timer(0.3, gate.set).start()
        seen: list[tuple[bool, str | None]] = []

        def on_body(chunk: bytes) -> None:
            seen.append((b"obj-7" in chunk, reader.lookup_response_session("obj-7")))

        try:
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            answer = await _post_raw(a, "/api/chat", body, on_body)
        finally:
            gate.set()
    assert answer.status == 200
    # The first line recorded nothing and went out at once; the line naming
    # the object only once its record landed.
    assert seen == [(False, None), (True, "conv-a")]


# --- realtime -------------------------------------------------------------------------

HANDLE_DIGEST = "handle-digest:h1"
RESUMPTION = json.dumps(
    {"sessionResumptionUpdate": {"newHandle": "h1", "resumable": True}}
).encode()


class HandleRouter:
    """A static-mode router recording each Live resumption handle in the
    vault's durable handle map from the server frame that issues it (as
    llm-redact-pro does, by a digest)."""

    mode = "static"

    def __init__(self, durable_lookup: Callable[[str], str | None] | None) -> None:
        self.manager = getattr(durable_lookup, "__self__", None)

    def resolve(self, adapter_name: Any, method: str, path: str, body: Any) -> str:
        return "default"

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def realtime_server_frame(
        self, adapter_name: str, path: str, frame: Any, *, identity: bool, session_id: str
    ) -> None:
        if isinstance(frame, dict) and "sessionResumptionUpdate" in frame:
            self.manager.record_handle_session(HANDLE_DIGEST, session_id)


class _Served(NamedTuple):
    host: str
    state: Any


@contextlib.contextmanager
def _serve(config: Config) -> Iterator[_Served]:
    """uvicorn on port 0 in a thread, the app (and its vault connection)
    BUILT in that thread: a sqlite connection is never shared across
    threads."""
    holder: dict[str, Any] = {}
    ready = threading.Event()

    def run() -> None:
        app = create_app(config)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        holder.update(app=app, server=server)
        ready.set()
        server.run()
        app.state.proxy.vault_manager.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(15)
    server = holder["server"]
    deadline = time.time() + 15
    while not server.started:
        assert time.time() < deadline, "uvicorn did not start"
        time.sleep(0.01)
    try:
        port = server.servers[0].sockets[0].getsockname()[1]
        yield _Served(f"127.0.0.1:{port}", holder["app"].state.proxy)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.mark.parametrize("mode", ["before_answer", "background"])
async def test_a_live_handle_frame_is_forwarded_only_once_its_record_landed(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, HandleRouter)
    path = tmp_path / "vault.db"
    async with Replying([RESUMPTION]) as fake:
        raw = {
            "providers": {"gemini": {"upstream_base_url": fake.url()}},
            "vault": {"backend": "sqlite", "path": str(path), "map_writes": mode},
        }
        with _serve(parse_config(raw, "t")) as proxy:
            assert proxy.state.awaits_map_writes is (mode == "before_answer")
            gate = _gate(proxy.state.vault_manager._maps)
            if mode == "before_answer":
                threading.Timer(0.3, gate.set).start()
            try:
                async with websockets.connect(f"ws://{proxy.host}{GEMINI_LIVE}") as client:
                    frame = await asyncio.wait_for(client.recv(), 10)
                    other = SqliteVaultManager(path)  # another replica's read
                    try:
                        landed = other.lookup_handle_session(HANDLE_DIGEST)
                    finally:
                        other.close()
            finally:
                gate.set()
    assert json.loads(frame) == json.loads(RESUMPTION)
    assert landed == ("default" if mode == "before_answer" else None)


def test_the_future_type_is_the_stdlib_one() -> None:
    # The proxy wraps each with asyncio.wrap_future: a thread-safe future.
    with awaited_writes() as pending:
        writer = _writer()
        writer.submit(_write([]))
    assert isinstance(pending[0], Future)
    assert pending[0].result(10) is None
    writer.close()
