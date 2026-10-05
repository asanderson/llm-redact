"""The access gate's optional policy reload (``plugin_api.AccessGate``'s
``reload`` and ``validate_reload``).

``ProxyState.apply_config`` — SIGHUP's ``reload()`` and the dashboard
editor's POST alike — calls ``reload(config)`` once per apply, after the hot
swap is live and open relays were revoked, even when nothing changed. A gate
that keeps its previous policy (a reason), fails or answers nonsense is
logged (the reason escaped and cut, a fault by TYPE) and counted under the
bookkeeping stage ``gate_reload``; nothing propagates. ``validate_config``
(the editor's dry run) asks ``validate_reload(candidate)``: a refusal is a
ConfigError. Without the members, nothing is asked.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from pathlib import Path
from typing import Any

import pytest

from llm_redact.authorization import GATE_RELOAD_CHARS, GATE_RELOAD_STAGE
from llm_redact.config import Config, ConfigError, load_config
from llm_redact.detection.engine import DetectionConfig
from llm_redact.proxy import ProxyState
from test_access_seam import FakeGate
from test_authorization_seam import _registry

SECRET = "jane.doe@corp.example"


class ReloadingGate(FakeGate):
    """A gate declaring ``reload`` and ``validate_reload``: records each
    call (and what the live state held when it was made)."""

    def __init__(self, answer: Any = None, verdict: Any = None) -> None:
        super().__init__()
        self.answer = answer
        self.verdict = verdict
        self.state: ProxyState | None = None
        self.reloaded: list[Config] = []
        self.live: list[bool] = []
        self.validated: list[Config] = []

    def reload(self, config: Config) -> Any:
        self.reloaded.append(config)
        assert self.state is not None
        # The swap is live: the configuration handed over is the running one.
        self.live.append(self.state.config is config)
        answer = self.answer
        if isinstance(answer, BaseException):
            raise answer
        return answer() if callable(answer) else answer

    def validate_reload(self, candidate: Config) -> Any:
        self.validated.append(candidate)
        verdict = self.verdict
        if isinstance(verdict, BaseException):
            raise verdict
        return verdict() if callable(verdict) else verdict


def _state(monkeypatch: pytest.MonkeyPatch, gate: Any, config: Config | None = None) -> ProxyState:
    _registry(monkeypatch, gate)
    state = ProxyState(config or Config(), None)
    if isinstance(gate, ReloadingGate):
        gate.state = state
    return state


def test_reload_is_called_once_per_apply_with_the_live_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = ReloadingGate()
    state = _state(monkeypatch, gate)
    assert state.authorization.reloads is True
    fresh = Config(detection=DetectionConfig(enabled=("email",)))
    state.apply_config(fresh)
    # Even an apply that changes nothing reloads the gate's policy.
    state.apply_config(fresh)
    assert len(gate.reloaded) == 2 and gate.live == [True, True]
    assert gate.reloaded[0].detection.enabled == ("email",)
    assert state.bookkeeping_errors[GATE_RELOAD_STAGE] == 0
    assert gate.validated == []  # the dry run is the editor's alone


def test_reload_follows_the_revocation_of_stale_relays(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    gate = ReloadingGate(lambda: order.append("reload"))
    state = _state(monkeypatch, gate)
    revoke = state._revoke_stale_relays
    monkeypatch.setattr(state, "_revoke_stale_relays", lambda: (order.append("revoke"), revoke()))
    state.apply_config(Config())
    assert order == ["revoke", "reload"]


def test_a_kept_policy_is_logged_counted_and_the_rest_still_applies(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gate = ReloadingGate("policy.cedar line 3: unexpected token\x1b[31m " + "x" * 400)
    state = _state(monkeypatch, gate)
    caplog.set_level(logging.WARNING, logger="llm_redact")
    state.apply_config(Config(detection=DetectionConfig(enabled=("email", "ipv4"))))
    assert len(state.detectors) == 2  # the reload itself applied
    assert state.bookkeeping_errors[GATE_RELOAD_STAGE] == 1
    (record,) = [r for r in caplog.records if "kept its previous policy" in r.getMessage()]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "access gate kept its previous policy: policy.cedar line 3" in message
    # Escaped, never a raw control character; cut to the bound.
    assert "\x1b" not in message and "\\x1b[31m" in message
    reason = message.split("kept its previous policy: ", 1)[1]
    assert len(reason) == GATE_RELOAD_CHARS and reason.endswith("…")


async def _later() -> None:
    await asyncio.sleep(0)


@pytest.mark.parametrize(
    ("answer", "logged"),
    [
        (RuntimeError(f"bad file holding {SECRET}"), "RuntimeError"),
        ("", "answered str"),
        (True, "answered bool"),
        (["reason"], "answered list"),
    ],
    ids=["raises", "empty", "true", "list"],
)
def test_a_failed_reload_is_contained(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, answer: Any, logged: str
) -> None:
    gate = ReloadingGate(answer)
    state = _state(monkeypatch, gate)
    caplog.set_level(logging.WARNING, logger="llm_redact")
    state.apply_config(Config(detection=DetectionConfig(enabled=("email",))))
    assert len(state.detectors) == 1
    assert state.bookkeeping_errors[GATE_RELOAD_STAGE] == 1
    assert f"policy reload failed ({logged})" in caplog.text
    assert SECRET not in caplog.text


def test_an_awaitable_answer_is_closed_unrun(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ran: list[bool] = []

    async def reload_later() -> None:
        ran.append(True)

    coroutine = reload_later()
    gate = ReloadingGate(lambda: coroutine)
    state = _state(monkeypatch, gate)
    caplog.set_level(logging.WARNING, logger="llm_redact")
    state.apply_config(Config())
    assert ran == [] and coroutine.cr_frame is None  # closed, never run
    assert state.bookkeeping_errors[GATE_RELOAD_STAGE] == 1
    assert "answered an awaitable" in caplog.text


def test_a_sighup_reload_reaches_the_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[detection]\nenabled = ["email"]\n')
    gate = ReloadingGate()
    _registry(monkeypatch, gate)
    state = ProxyState(load_config(config_path), None, config_path=config_path)
    gate.state = state
    config_path.write_text('[detection]\nenabled = ["email", "ipv4"]\n')
    state.reload()
    assert len(gate.reloaded) == 1 and gate.live == [True]
    assert gate.reloaded[0].detection.enabled == ("email", "ipv4")
    # A reload that fails before the swap never reaches the gate.
    config_path.write_text("this is not toml [ [")
    state.reload()
    assert len(gate.reloaded) == 1


def test_validate_reload_gates_the_editors_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = ReloadingGate(verdict="policy.cedar: unknown entity type\x07")
    state = _state(monkeypatch, gate)
    candidate = Config(detection=DetectionConfig(enabled=("email",)))
    with pytest.raises(ConfigError) as refused:
        state.validate_config(candidate)
    assert str(refused.value) == (
        "the access gate refuses this configuration: policy.cedar: unknown entity type\\x07"
    )
    assert gate.validated == [candidate]
    gate.verdict = None
    state.validate_config(candidate)  # allowed: nothing raised
    assert gate.reloaded == []  # a dry run changes nothing


@pytest.mark.parametrize(
    ("verdict", "named"),
    [
        (ValueError(f"cannot parse {SECRET}"), "ValueError"),
        ("", "answered str"),
        (False, "answered bool"),
    ],
    ids=["raises", "empty", "false"],
)
def test_a_failing_validate_reload_refuses_naming_the_type(
    monkeypatch: pytest.MonkeyPatch, verdict: Any, named: str
) -> None:
    state = _state(monkeypatch, ReloadingGate(verdict=verdict))
    with pytest.raises(ConfigError) as refused:
        state.validate_config(Config())
    assert named in str(refused.value) and SECRET not in str(refused.value)


def test_an_awaitable_validate_reload_is_closed_and_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coroutine = _later()
    state = _state(monkeypatch, ReloadingGate(verdict=lambda: coroutine))
    with pytest.raises(ConfigError, match="an awaitable"):
        state.validate_config(Config())
    assert coroutine.cr_frame is None


def test_without_the_members_nothing_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(monkeypatch, FakeGate())
    assert state.authorization.reloads is False
    state.apply_config(Config(detection=DetectionConfig(enabled=("email",))))
    state.validate_config(Config())
    assert state.bookkeeping_errors[GATE_RELOAD_STAGE] == 0


def test_no_gate_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    state = ProxyState(Config(), None)
    state.apply_config(dataclasses.replace(Config(), inject_system_note=False))
    state.validate_config(Config())
    assert state.authorization.reloads is False
