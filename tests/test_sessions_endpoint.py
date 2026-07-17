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
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


async def _token(client: httpx.AsyncClient) -> str:
    return str((await client.get("/__llm-redact/config")).json()["csrf_token"])


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
