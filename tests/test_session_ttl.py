"""[vault] session_ttl_days: scheduled background prune of idle sessions."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

import llm_redact.proxy as proxy_mod
from llm_redact.config import Config, VaultConfig
from llm_redact.proxy import create_app
from llm_redact.vault import open_sqlite_vault


def _seed_and_age(db: Path, idle_session: str, active_session: str) -> None:
    v_idle = open_sqlite_vault(db, idle_session)
    v_idle.placeholder_for("EMAIL", "old@corp.example")
    v_idle.close()
    v_active = open_sqlite_vault(db, active_session)
    v_active.placeholder_for("EMAIL", "new@corp.example")
    v_active.close()
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE mappings SET created_at ="
        " strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-100 days') WHERE session_id = ?",
        (idle_session,),
    )
    conn.commit()
    conn.close()


def _sessions(db: Path) -> set[str]:
    conn = sqlite3.connect(db)
    seen = {row[0] for row in conn.execute("SELECT DISTINCT session_id FROM mappings")}
    conn.close()
    return seen


def _config(db: Path, ttl: int) -> Config:
    return Config(
        vault=VaultConfig(backend="sqlite", path=str(db), session="default", session_ttl_days=ttl)
    )


async def test_ttl_loop_prunes_idle_keeps_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "vault.db"
    _seed_and_age(db, "conv-old", "default")
    monkeypatch.setattr(proxy_mod, "_TTL_PRUNE_INTERVAL_SECONDS", 0.05)
    app = create_app(_config(db, ttl=1))
    async with app.router.lifespan_context(app):
        for _ in range(40):  # up to ~2s for the sleep-first loop to fire
            await asyncio.sleep(0.05)
            if _sessions(db) == {"default"}:
                break
    assert _sessions(db) == {"default"}  # idle pruned, active never pruned


async def test_ttl_loop_absent_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "vault.db"
    _seed_and_age(db, "conv-old", "default")
    monkeypatch.setattr(proxy_mod, "_TTL_PRUNE_INTERVAL_SECONDS", 0.05)
    app = create_app(_config(db, ttl=0))  # disabled: no task created
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.2)
    assert _sessions(db) == {"conv-old", "default"}  # nothing pruned


async def test_ttl_loop_keeps_sessions_the_router_marks_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact import registry as registry_mod

    class DurableRouter:
        mode = "static"

        def resolve(self, adapter_name: str | None, method: str, path: str, body: object) -> str:
            return "default"

        def record_response_id(self, response_id: str, session_id: str) -> None:
            return None

        def is_durable(self, session_id: str) -> bool:
            return session_id.endswith(":default")

    db = tmp_path / "vault.db"
    _seed_and_age(db, "conv-old", "default")
    _seed_and_age(db, "user:3:default", "default")
    monkeypatch.setattr(proxy_mod, "_TTL_PRUNE_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(
        registry_mod.get_registry(), "build_session_router", lambda cfg, **kw: DurableRouter()
    )
    app = create_app(_config(db, ttl=1))
    async with app.router.lifespan_context(app):
        for _ in range(40):
            await asyncio.sleep(0.05)
            if "conv-old" not in _sessions(db):
                break
        await asyncio.sleep(0.1)  # a further pass must not take the durable session either
    assert _sessions(db) == {"default", "user:3:default"}
