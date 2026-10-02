"""The per-request vault transaction through the real app (sqlite vault).

Every redacting path wraps its synchronous redaction in ``run_batched``: a
JSON body (``prepare_request``, the Bedrock count-tokens blob included), a
multipart upload (``redact_multipart``: form fields, file names, JSONL
lines) and each realtime client frame (``redact_message``). One COMMIT per
request — not per value — and it lands BEFORE anything is forwarded: a
refusal after some values were issued (block mode, max_body_strings) rolls
them back, and a COMMIT that fails refuses the request with nothing sent: a
recorded, provider-shaped 503 (a realtime frame: the connection closes
1011), counted as the "vault" bookkeeping stage — never a bare 500.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
import websockets

from llm_redact.config import Config, DetectionConfig, ProviderConfig, VaultConfig
from llm_redact.proxy import create_app
from local_refusals import refused_once
from test_vault_faults import _FlakyConn


class _FlakyAnyCase(_FlakyConn):
    """``_FlakyConn`` matching its SQL in any case, as the engine reads it:
    a statement spelled in another case is the same statement and fails
    the same way."""

    def execute(self, sql: str, *args: object) -> object:
        if self._fail_on.lower() in sql.lower() and self._times > 0:
            self._times -= 1
            raise sqlite3.OperationalError("database or disk is full")
        return self._real.execute(sql, *args)


EMAILS = [f"user{i}@corp.example" for i in range(12)]


def _chat_upstream(sent: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append({"path": request.url.path, "body": request.content})
        if request.url.path == "/v1/files":
            return httpx.Response(200, json={"id": "file_abc", "object": "file"})
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            },
        )

    return httpx.MockTransport(handler)


def _app(tmp_path: Path, sent: list[dict[str, Any]], **config: Any) -> tuple[Any, list[str]]:
    app = create_app(
        Config(
            providers={
                **Config().providers,
                "anthropic": ProviderConfig("http://upstream.test"),
                "openai": ProviderConfig("http://upstream.test"),
            },
            vault=VaultConfig(backend="sqlite", path=str(tmp_path / "vault.db")),
            **config,
        ),
        upstream_transport=_chat_upstream(sent),
    )
    seen: list[str] = []
    app.state.proxy.vault_manager._conn.set_trace_callback(
        lambda sql: seen.append(sql) if sql in ("BEGIN IMMEDIATE", "COMMIT", "ROLLBACK") else None
    )
    return app, seen


def _messages(*texts: str) -> dict[str, Any]:
    return {
        "model": "m",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": text} for text in texts],
    }


async def _post(app: Any, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        return await client.post(path, **kwargs)


def _sent_text(sent: list[dict[str, Any]]) -> str:
    return sent[-1]["body"].decode("utf-8")


async def test_a_json_body_writes_its_new_values_in_one_commit(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    app, seen = _app(tmp_path, sent)
    body = _messages(" ".join(EMAILS[:6]), " ".join(EMAILS[6:]))
    response = await _post(app, "/v1/messages", json=body)
    assert response.status_code == 200
    assert seen == ["BEGIN IMMEDIATE", "COMMIT"]  # one fsync for twelve values
    forwarded = _sent_text(sent)
    assert all(f"«EMAIL_{n:03d}»" in forwarded for n in range(1, 13))
    assert not any(email in forwarded for email in EMAILS)
    # A request whose values are all known takes no write lock at all.
    seen.clear()
    assert (await _post(app, "/v1/messages", json=body)).status_code == 200
    assert seen == []
    app.state.proxy.vault_manager.close()


async def test_a_blocked_value_rolls_back_the_requests_earlier_values(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    app, seen = _app(tmp_path, sent, detection=DetectionConfig(modes=(("us_ssn", "block"),)))
    body = _messages(" ".join(EMAILS[:3]), "ssn 078-05-1120")
    response = await _post(app, "/v1/messages", json=body)
    assert response.status_code == 400
    assert sent == []  # nothing forwarded...
    assert seen == ["BEGIN IMMEDIATE", "ROLLBACK"]
    manager = app.state.proxy.vault_manager
    assert manager.total_entries() == 0  # ...and nothing kept
    assert app.state.proxy.vault.original_for("«EMAIL_001»") is None
    # The next request numbers from 001 again: the rolled-back numbers were
    # never forwarded, and are issued again only to the same values.
    assert (await _post(app, "/v1/messages", json=_messages(EMAILS[0]))).status_code == 200
    assert "«EMAIL_001»" in _sent_text(sent)
    manager.close()


async def test_a_body_over_max_body_strings_rolls_back(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    app, seen = _app(tmp_path, sent, max_body_strings=3)
    body = _messages(*EMAILS[:5])
    response = await _post(app, "/v1/messages", json=body)
    assert response.status_code == 413
    assert sent == [] and app.state.proxy.vault_manager.total_entries() == 0
    assert seen == ["BEGIN IMMEDIATE", "ROLLBACK"]
    app.state.proxy.vault_manager.close()


async def test_a_failed_commit_refuses_the_request_with_nothing_forwarded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sent: list[dict[str, Any]] = []
    app, _ = _app(tmp_path, sent)
    manager = app.state.proxy.vault_manager
    real = manager._conn
    manager._conn = _FlakyConn(real, "COMMIT")
    body = _messages(" ".join(EMAILS[:4]))
    with caplog.at_level("ERROR", logger="llm_redact"):
        response = await _post(app, "/v1/messages", json=body)
    # Failed closed: a recorded, provider-shaped 503 — never a bare 500.
    assert response.status_code == 503
    refusal = response.json()
    assert refusal["type"] == "error"  # the Anthropic shape (the route's adapter)
    assert "vault could not record" in refusal["error"]["message"]
    assert sent == []  # the upstream never saw the request
    state = app.state.proxy
    assert state.bookkeeping_errors == {"vault": 1}
    (row,) = state.recent
    assert row["status"] == 503 and row["provider"] == "anthropic"
    refused_once(state, "vault_fault", "anthropic")
    # Logged by exception TYPE only: never a value, never the SQL error text.
    assert "OperationalError" in caplog.text
    assert "disk is full" not in caplog.text
    assert not any(email in caplog.text for email in EMAILS)
    manager._conn = real
    assert manager.total_entries() == 0
    assert not real.in_transaction
    assert app.state.proxy.vault.original_for("«EMAIL_001»") is None  # nothing cached
    # The retry issues the very same tokens.
    assert (await _post(app, "/v1/messages", json=body)).status_code == 200
    assert all(f"«EMAIL_{n:03d}»" in _sent_text(sent) for n in range(1, 5))
    manager.close()


async def test_a_failed_commit_refuses_an_upload_with_nothing_forwarded(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    app, _ = _app(tmp_path, sent)
    manager = app.state.proxy.vault_manager
    real = manager._conn
    manager._conn = _FlakyConn(real, "COMMIT")
    line = json.dumps({"custom_id": "r1", "body": {"input": f"mail {EMAILS[0]}"}})
    response = await _post(
        app,
        "/v1/files",
        content=_upload(line),
        headers={"content-type": "multipart/form-data; boundary=b0undary"},
    )
    assert response.status_code == 503
    assert "vault could not record" in response.json()["error"]["message"]
    assert sent == [] and app.state.proxy.bookkeeping_errors == {"vault": 1}
    manager._conn = real
    assert manager.total_entries() == 0
    manager.close()


def test_the_vault_fault_types_are_sqlites_plus_the_managers_own() -> None:
    from llm_redact.proxy import vault_fault_types

    class Declared(Exception):
        pass

    class Manager:
        fault_types = (Declared,)

    assert vault_fault_types(object()) == (sqlite3.Error,)
    assert vault_fault_types(Manager()) == (sqlite3.Error, Declared)


def _upload(*lines: str) -> bytes:
    jsonl = "".join(f"{line}\n" for line in lines).encode()
    return (
        b"--b0undary\r\n"
        b'Content-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
        b"--b0undary\r\n"
        b'Content-Disposition: form-data; name="file"; filename="input.jsonl"\r\n'
        b"Content-Type: application/jsonl\r\n\r\n" + jsonl + b"\r\n"
        b"--b0undary--\r\n"
    )


async def test_a_multipart_upload_writes_its_new_values_in_one_commit(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    app, seen = _app(tmp_path, sent)
    lines = [
        json.dumps(
            {
                "custom_id": f"r{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "m", "messages": [{"role": "user", "content": f"mail {email}"}]},
            }
        )
        for i, email in enumerate(EMAILS[:5])
    ]
    response = await _post(
        app,
        "/v1/files",
        content=_upload(*lines),
        headers={"content-type": "multipart/form-data; boundary=b0undary"},
    )
    assert response.status_code == 200
    assert seen == ["BEGIN IMMEDIATE", "COMMIT"]
    forwarded = sent[-1]["body"].decode("utf-8")
    assert all(f"«EMAIL_{n:03d}»" in forwarded for n in range(1, 6))
    assert not any(email in forwarded for email in EMAILS[:5])
    app.state.proxy.vault_manager.close()


# --- realtime: one transaction per client frame -----------------------------------


class _EchoUpstream:
    def __init__(self) -> None:
        self.received: list[str] = []
        self.port = 0
        self.server: Any = None

    async def _handler(self, connection: Any) -> None:
        async for message in connection:
            self.received.append(message)
            # The server's echo of a created item (restored whole).
            item = json.loads(message)["item"]
            await connection.send(json.dumps({"type": "conversation.item.created", "item": item}))

    async def __aenter__(self) -> _EchoUpstream:
        self.server = await websockets.serve(self._handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.server.close()
        await self.server.wait_closed()


@contextlib.contextmanager
def _relay(
    config: Config,
    seen: list[str],
    *,
    fail_on: str | None = None,
    flaky: type[_FlakyConn] = _FlakyConn,
    apps: list[Any] | None = None,
) -> Iterator[str]:
    """The proxy in uvicorn's own thread — built there too (a factory):
    sqlite connections are per-thread. ``fail_on``: the vault's SQL that
    fails once (a disk-full write); ``apps`` collects the built app."""

    def factory() -> Any:
        app = create_app(config)
        manager = app.state.proxy.vault_manager
        manager._conn.set_trace_callback(
            lambda sql: seen.append(sql) if sql in ("BEGIN IMMEDIATE", "COMMIT") else None
        )
        if fail_on is not None:
            manager._conn = flaky(manager._conn, fail_on)
        if apps is not None:
            apps.append(app)
        return app

    server = uvicorn.Server(
        uvicorn.Config(factory, factory=True, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    try:
        yield f"127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.mark.asyncio
async def test_each_realtime_frame_writes_its_new_values_in_one_commit(tmp_path: Path) -> None:
    seen: list[str] = []
    async with _EchoUpstream() as upstream:
        config = Config(
            providers={
                **Config().providers,
                "openai": ProviderConfig(f"http://127.0.0.1:{upstream.port}"),
            },
            vault=VaultConfig(backend="sqlite", path=str(tmp_path / "vault.db")),
        )
        with _relay(config, seen) as host:
            async with websockets.connect(f"ws://{host}/v1/realtime") as client:
                frame = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": " ".join(EMAILS[:4])}],
                    },
                }
                await client.send(json.dumps(frame))
                echoed = json.loads(await asyncio.wait_for(client.recv(), 10))
    forwarded = upstream.received[0]
    assert all(f"«EMAIL_{n:03d}»" in forwarded for n in range(1, 5))
    assert seen == ["BEGIN IMMEDIATE", "COMMIT"]  # one frame, one fsync
    # The echo comes back restored.
    assert echoed["item"]["content"][0]["text"] == " ".join(EMAILS[:4])
    # Committed before the frame was sent: a fresh connection sees the rows.
    conn = sqlite3.connect(tmp_path / "vault.db")
    assert conn.execute("SELECT COUNT(*) FROM mappings").fetchone() == (4,)
    conn.close()


@pytest.mark.asyncio
async def test_a_failed_commit_closes_the_realtime_connection_1011(tmp_path: Path) -> None:
    """A frame whose placeholders the vault cannot record is never sent: the
    connection closes 1011, recorded as the HTTP path's 503 and counted as
    the "vault" bookkeeping stage."""
    seen: list[str] = []
    apps: list[Any] = []
    async with _EchoUpstream() as upstream:
        config = Config(
            providers={
                **Config().providers,
                "openai": ProviderConfig(f"http://127.0.0.1:{upstream.port}"),
            },
            vault=VaultConfig(backend="sqlite", path=str(tmp_path / "vault.db")),
        )
        with _relay(config, seen, fail_on="COMMIT", apps=apps) as host:
            async with websockets.connect(f"ws://{host}/v1/realtime") as client:
                frame = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": " ".join(EMAILS[:2])}],
                    },
                }
                await client.send(json.dumps(frame))
                with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                    await asyncio.wait_for(client.recv(), 10)
    assert closed.value.rcvd is not None and closed.value.rcvd.code == 1011
    assert "could not record" in closed.value.rcvd.reason
    assert upstream.received == []  # the frame never left
    state = apps[0].state.proxy
    assert state.bookkeeping_errors == {"vault": 1}
    (row,) = state.recent
    assert row["method"] == "WS" and row["status"] == 503
    conn = sqlite3.connect(tmp_path / "vault.db")
    assert conn.execute("SELECT COUNT(*) FROM mappings").fetchone() == (0,)  # rolled back
    conn.close()


# --- a vault fault while a request's session is opened ------------------------------


class _NewSessionRouter:
    """A session router (llm-redact-pro's seam) resolving every request to a
    session this process has not opened yet: its vault view is built — and
    reads the database — while the request's context is made."""

    mode = "per-conversation"

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return "conv-new"

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None


def _new_session_router(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_session_ownership_seams import _registry

    router = _NewSessionRouter()
    _registry(monkeypatch, build_session_router=lambda config, **kw: router)


@pytest.mark.parametrize(
    ("path", "provider", "shape"),
    [("/v1/messages", "anthropic", "error"), ("/v1/moderations", "openai", None)],
    ids=["matched", "pass-through"],
)
async def test_a_vault_fault_opening_the_requests_session_refuses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    path: str,
    provider: str,
    shape: str | None,
) -> None:
    """Opening a new session's view reads its rows: a database fault there
    is a recorded, provider-shaped 503 before any upstream contact (labelled
    with the configured session: the request has none yet) — never a bare,
    unrecorded 500."""
    _new_session_router(monkeypatch)
    sent: list[dict[str, Any]] = []
    app, _ = _app(tmp_path, sent)
    manager = app.state.proxy.vault_manager
    real = manager._conn
    manager._conn = _FlakyAnyCase(real, "FROM retired_numbers")
    headers = {"authorization": "Bearer sk-proj-FAKE"} if provider == "openai" else {}
    with caplog.at_level("ERROR", logger="llm_redact"):
        response = await _post(app, path, json=_messages(EMAILS[0]), headers=headers)
    assert response.status_code == 503, response.text
    refusal = response.json()
    assert refusal.get("type") == shape  # the route's adapter shape, else a generic one
    assert "vault could not" in json.dumps(refusal)
    assert sent == []
    state = app.state.proxy
    assert state.bookkeeping_errors == {"vault": 1}
    (row,) = state.recent
    assert row["status"] == 503 and row["provider"] == provider
    assert row["session"] == state.config.vault.session
    assert "OperationalError" in caplog.text and "disk is full" not in caplog.text
    assert EMAILS[0] not in caplog.text
    # The fault passed: the next request opens the session and is forwarded.
    manager._conn = real
    assert (await _post(app, path, json=_messages(EMAILS[0]), headers=headers)).status_code == 200
    manager.close()


@pytest.mark.asyncio
async def test_a_vault_fault_opening_a_realtime_session_closes_1011(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _new_session_router(monkeypatch)
    seen: list[str] = []
    apps: list[Any] = []
    async with _EchoUpstream() as upstream:
        config = Config(
            providers={
                **Config().providers,
                "openai": ProviderConfig(f"http://127.0.0.1:{upstream.port}"),
            },
            vault=VaultConfig(backend="sqlite", path=str(tmp_path / "vault.db")),
        )
        with _relay(
            config, seen, fail_on="FROM retired_numbers", flaky=_FlakyAnyCase, apps=apps
        ) as host:
            async with websockets.connect(f"ws://{host}/v1/realtime") as client:
                with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                    await asyncio.wait_for(client.recv(), 10)
    assert closed.value.rcvd is not None and closed.value.rcvd.code == 1011
    assert "vault" in closed.value.rcvd.reason
    assert upstream.received == []
    state = apps[0].state.proxy
    assert state.bookkeeping_errors == {"vault": 1}
    (row,) = state.recent
    assert row["method"] == "WS" and row["status"] == 503
