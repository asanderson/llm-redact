"""The /__llm-redact/sessions browser and its guarded prune endpoint."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig, VaultConfig
from llm_redact.proxy import CSRF_HEADER, create_app

EMAIL = "jane.doe@corp.example"

received: dict[str, Any] = {}


def _fake_upstream() -> Starlette:
    async def messages(request: Request) -> JSONResponse:
        received["anthropic"] = await request.json()
        return JSONResponse({"content": [{"type": "text", "text": "ok"}], "role": "assistant"})

    return Starlette(routes=[Route("/v1/messages", messages, methods=["POST"])])


def _make_client(
    vault: VaultConfig | None = None, *, base_url: str = "http://127.0.0.1:8787"
) -> httpx.AsyncClient:
    received.clear()
    config = Config(
        providers={**Config().providers, "anthropic": ProviderConfig("http://upstream")},
        vault=vault if vault is not None else VaultConfig(),
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)
    _tokens[id(client)] = app.state.proxy.csrf_token
    return client


# The per-process CSRF token. The pro dashboard hands it to same-origin
# pages via GET /config; the free core has no page, so tests read it off
# the live ProxyState.
_tokens: dict[int, str] = {}


async def _token(client: httpx.AsyncClient) -> str:
    return _tokens[id(client)]


def _age_session(db: Path, session_id: str, days: int) -> None:
    """Backdate a session's rows so it counts as idle (WAL allows the
    second writer while the proxy holds the DB open)."""
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute(
        "UPDATE mappings SET created_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?)"
        " WHERE session_id = ?",
        (f"-{days} days", session_id),
    )
    conn.close()


def _insert_session(db: Path, session_id: str, days_old: int) -> None:
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute(
        "INSERT INTO mappings (session_id, detector_type, original, placeholder, n, created_at)"
        " VALUES (?, 'EMAIL', 'old@corp.example', '«EMAIL_001»', 1,"
        " strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?))",
        (session_id, f"-{days_old} days"),
    )
    conn.close()


async def test_sessions_get_memory_backend() -> None:
    client = _make_client()
    body = {"model": "m", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]}
    await client.post("/v1/messages", json=body)

    payload = (await client.get("/__llm-redact/sessions")).json()
    assert payload["backend"] == "memory"
    assert payload["active_session"] == "default"
    (entry,) = payload["sessions"]
    assert entry["session"] == "default"
    assert entry["entries"] == 1
    # Metadata only — never the redacted values.
    assert EMAIL not in json.dumps(payload)


async def test_prune_memory_backend_rejected() -> None:
    client = _make_client()
    token = await _token(client)
    response = await client.post(
        "/__llm-redact/sessions/prune",
        headers={CSRF_HEADER: token},
        json={"older_than_days": 30},
    )
    assert response.status_code == 400
    assert "sqlite" in response.json()["error"]


async def test_prune_deletes_idle_sessions_not_active(tmp_path: Path) -> None:
    db = tmp_path / "vault.db"
    client = _make_client(VaultConfig(backend="sqlite", path=str(db)))
    body = {"model": "m", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]}
    await client.post("/v1/messages", json=body)

    _insert_session(db, "conv-idle", days_old=100)
    _age_session(db, "default", days=100)  # active session is old too

    sessions = (await client.get("/__llm-redact/sessions")).json()["sessions"]
    assert {s["session"] for s in sessions} == {"default", "conv-idle"}

    token = await _token(client)
    response = await client.post(
        "/__llm-redact/sessions/prune",
        headers={CSRF_HEADER: token},
        json={"older_than_days": 30},
    )
    assert response.status_code == 200
    assert response.json() == {"pruned": 1}

    remaining = (await client.get("/__llm-redact/sessions")).json()["sessions"]
    # The idle conversation is gone; the active session survives despite
    # being idle, because the live process never prunes its own namespace.
    assert {s["session"] for s in remaining} == {"default"}

    again = await client.post(
        "/__llm-redact/sessions/prune",
        headers={CSRF_HEADER: token},
        json={"older_than_days": 30},
    )
    assert again.json() == {"pruned": 0}


async def test_prune_guard_stack() -> None:
    client = _make_client()
    token = await _token(client)

    # No CSRF header.
    response = await client.post("/__llm-redact/sessions/prune", json={"older_than_days": 30})
    assert response.status_code == 403

    # Wrong content type.
    response = await client.post(
        "/__llm-redact/sessions/prune",
        headers={CSRF_HEADER: token, "content-type": "text/plain"},
        content=b'{"older_than_days": 30}',
    )
    assert response.status_code == 415

    # Bad payloads.
    for bad in ({"older_than_days": "30"}, {"older_than_days": -1}, {"older_than_days": True}, {}):
        response = await client.post(
            "/__llm-redact/sessions/prune", headers={CSRF_HEADER: token}, json=bad
        )
        assert response.status_code == 400, bad

    # Methods: GET on /prune and POST on the browser are both 405.
    assert (await client.get("/__llm-redact/sessions/prune")).status_code == 405
    assert (
        await client.post("/__llm-redact/sessions", headers={CSRF_HEADER: token}, json={})
    ).status_code == 405


async def test_sessions_reject_foreign_host() -> None:
    # DNS-rebinding defense applies to the browser and the prune endpoint.
    client = _make_client(base_url="http://evil.example")
    assert (await client.get("/__llm-redact/sessions")).status_code == 403
    assert (await client.post("/__llm-redact/sessions/prune", json={})).status_code == 403


class _DurableRouter:
    """A router (like llm-redact-pro's per-user wrapper) that marks some
    sessions as holding provider-side state the live process must keep."""

    mode = "static"

    def __init__(self, fallback: str) -> None:
        self._fallback = fallback

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return self._fallback

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def is_durable(self, session_id: str) -> bool:
        return session_id.startswith("user:") and session_id.endswith(":" + self._fallback)


async def test_prune_keeps_sessions_the_router_marks_durable(tmp_path: Path) -> None:
    # A router's durable sessions (a named user's copy of the static
    # session) survive a prune like the static session itself; other idle
    # sessions still go, and a router without the hook behaves as before.
    from llm_redact import registry as registry_mod

    db = tmp_path / "vault.db"
    registry = registry_mod.get_registry()
    original = registry.build_session_router
    registry.build_session_router = lambda cfg, **kw: _DurableRouter(cfg.session)
    try:
        client = _make_client(VaultConfig(backend="sqlite", path=str(db)))
    finally:
        registry.build_session_router = original
    for session in ("user:7:default", "user:8:default", "user:7:conv-idle", "conv-idle"):
        _insert_session(db, session, days_old=100)
    _insert_session(db, "default", days_old=100)
    token = await _token(client)
    response = await client.post(
        "/__llm-redact/sessions/prune", headers={CSRF_HEADER: token}, json={"older_than_days": 30}
    )
    assert response.json() == {"pruned": 2}
    remaining = (await client.get("/__llm-redact/sessions")).json()["sessions"]
    assert {s["session"] for s in remaining} == {"default", "user:7:default", "user:8:default"}


class _NoHookRouter(_DurableRouter):
    is_durable = None  # type: ignore[assignment]  # an older router without the member


class _RaisingRouter(_DurableRouter):
    def is_durable(self, session_id: str) -> bool:
        raise RuntimeError("registry closed")


class _SloppyRouter(_DurableRouter):
    def is_durable(self, session_id: str) -> bool:
        return None  # type: ignore[return-value]  # a buggy router's "don't know"


async def _prune_with(router_cls: type[_DurableRouter], tmp_path: Path) -> set[str]:
    from llm_redact import registry as registry_mod

    db = tmp_path / "vault.db"
    registry = registry_mod.get_registry()
    original = registry.build_session_router
    registry.build_session_router = lambda cfg, **kw: router_cls(cfg.session)
    try:
        client = _make_client(VaultConfig(backend="sqlite", path=str(db)))
    finally:
        registry.build_session_router = original
    for session in ("user:7:default", "conv-idle", "default"):
        _insert_session(db, session, days_old=100)
    token = await _token(client)
    response = await client.post(
        "/__llm-redact/sessions/prune", headers={CSRF_HEADER: token}, json={"older_than_days": 30}
    )
    assert response.status_code == 200
    return {s["session"] for s in (await client.get("/__llm-redact/sessions")).json()["sessions"]}


async def test_prune_without_the_hook_keeps_only_the_static_session(tmp_path: Path) -> None:
    assert await _prune_with(_NoHookRouter, tmp_path) == {"default"}


async def test_prune_keeps_sessions_when_is_durable_fails(tmp_path: Path) -> None:
    # A raising router is a 200 that keeps everything, never a 500 or a loss.
    assert await _prune_with(_RaisingRouter, tmp_path) == {"default", "user:7:default", "conv-idle"}


async def test_prune_keeps_sessions_on_a_non_bool_answer(tmp_path: Path) -> None:
    # Only an explicit False releases a session.
    assert await _prune_with(_SloppyRouter, tmp_path) == {"default", "user:7:default", "conv-idle"}


def test_the_router_can_veto_the_durable_response_map(tmp_path: Path) -> None:
    from llm_redact import registry as registry_mod
    from llm_redact.config import Config
    from llm_redact.proxy import create_app

    class Router:
        mode = "per-conversation"

        def __init__(self, answer: bool | None) -> None:
            self.answer = answer

        def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
            return "default"

        def record_response_id(self, response_id: str, session_id: str) -> bool | None:
            return self.answer

    registry = registry_mod.get_registry()
    original = registry.build_session_router
    results = {}
    try:
        for answer in (False, None, True):
            registry.build_session_router = lambda cfg, a=answer, **kw: Router(a)
            db = tmp_path / f"vault-{answer}.db"
            state = create_app(
                Config(vault=VaultConfig(backend="sqlite", path=str(db)))
            ).state.proxy
            state.record_response_id("resp_1", "conv-a")
            results[answer] = state.vault_manager.lookup_response_session("resp_1")
    finally:
        registry.build_session_router = original
    assert results == {False: None, None: "conv-a", True: "conv-a"}


def test_the_memory_vault_hands_the_router_no_durable_lookup(tmp_path: Path) -> None:
    # Its lookup always answers "unknown", which a router must never read
    # as "that session was pruned" (every Responses chain would orphan).
    from llm_redact import registry as registry_mod
    from llm_redact.config import Config
    from llm_redact.proxy import create_app

    seen: dict[str, Any] = {}
    registry = registry_mod.get_registry()
    original = registry.build_session_router

    def capture(cfg: Any, **kw: Any) -> _DurableRouter:
        seen.update(kw)
        return _DurableRouter(cfg.session)

    registry.build_session_router = capture
    try:
        create_app(Config(vault=VaultConfig(backend="memory")))
        assert seen["durable_lookup"] is None
        create_app(Config(vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db"))))
        assert seen["durable_lookup"] is not None
    finally:
        registry.build_session_router = original
