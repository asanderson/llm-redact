"""compaction_forks counts only genuine history-compaction forks: a session
first seen by this process whose history carries placeholders it cannot own.
A persisted session resumed after a restart owns its tokens, and a session
the router marks durable is not anchored on a first message."""

from pathlib import Path
from typing import Any

import pytest

from llm_redact import registry as registry_mod
from llm_redact.config import Config, VaultConfig
from llm_redact.proxy import ProxyState, create_app
from llm_redact.vault import open_sqlite_vault

_TOKEN_BODY = {"messages": [{"role": "user", "content": "mail «EMAIL_001» again"}]}
_PLAIN_BODY = {"messages": [{"role": "user", "content": "hello"}]}


class _BodyRouter:
    """Resolves the session named in the body; ``user:`` sessions are durable."""

    mode = "per-conversation"

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return str(body["session"])

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def is_durable(self, session_id: str) -> bool:
        return session_id.startswith("user:")


class _NoDurableRouter(_BodyRouter):
    is_durable = None  # type: ignore[assignment]  # an older router without the member


def _state(
    monkeypatch: pytest.MonkeyPatch, db: Path, router: type[_BodyRouter] = _BodyRouter
) -> ProxyState:
    monkeypatch.setattr(
        registry_mod.get_registry(), "build_session_router", lambda cfg, **kw: router()
    )
    app = create_app(Config(vault=VaultConfig(backend="sqlite", path=str(db), session="default")))
    state: ProxyState = app.state.proxy
    return state


def _see(state: ProxyState, session: str, body: dict[str, Any]) -> None:
    state.context_for(None, "POST", "/v1/messages", {**body, "session": session})


def test_new_empty_session_with_placeholders_is_a_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(monkeypatch, tmp_path / "vault.db")
    _see(state, "conv-new", _TOKEN_BODY)
    _see(state, "conv-new", _TOKEN_BODY)  # counted once per session
    _see(state, "conv-plain", _PLAIN_BODY)
    assert state.compaction_forks == 1


def test_persisted_session_resumed_after_restart_is_not_a_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "vault.db"
    seeded = open_sqlite_vault(db, "conv-resumed")
    seeded.placeholder_for("EMAIL", "a@corp.example")
    seeded.close()
    state = _state(monkeypatch, db)
    _see(state, "conv-resumed", _TOKEN_BODY)
    assert state.compaction_forks == 0


def test_durable_session_is_not_a_fork(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(monkeypatch, tmp_path / "vault.db")
    _see(state, "user:3:default", _TOKEN_BODY)
    assert state.compaction_forks == 0


def test_router_without_is_durable_still_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(monkeypatch, tmp_path / "vault.db", _NoDurableRouter)
    _see(state, "user:3:default", _TOKEN_BODY)
    assert state.compaction_forks == 1


class _RaisingRouter(_BodyRouter):
    def is_durable(self, session_id: str) -> bool:
        raise RuntimeError("registry closed")


class _SloppyRouter(_BodyRouter):
    def is_durable(self, session_id: str) -> bool:
        return None  # type: ignore[return-value]  # a buggy router's "don't know"


@pytest.mark.parametrize("router", [_RaisingRouter, _SloppyRouter])
def test_a_misbehaving_is_durable_never_fails_the_request(
    router: type[_BodyRouter], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The prune's contract: only an explicit False means "not durable".
    state = _state(monkeypatch, tmp_path / "vault.db", router)
    _see(state, "conv-new", _TOKEN_BODY)
    assert state.compaction_forks == 0
