"""Refusal overrides and the access gate's optional members.

- ``approves_overrides`` and ``override_subject`` are asked only once a
  refusal is being decided — never per request or per realtime frame (a
  gate may answer from a database on the event loop).
- ``override_subject`` binds a named user's records to a STABLE id: a
  renamed user keeps their approvals, and a user who later takes the old
  name inherits none of them. A gate that cannot name the requester gives no
  override and no code (never the subject instead).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import websockets

import llm_redact.registry as registry_mod
from llm_redact.config import Config, OverridesConfig, ProviderConfig
from llm_redact.detection.engine import DetectionConfig
from llm_redact.proxy import CSRF_HEADER
from llm_redact.registry import Registry
from test_overrides import (
    DASHBOARD_HINT,
    EMAIL,
    KEY,
    SubjectGate,
    Upstream,
    _app,
    _chat,
    _client,
    _store,
)
from test_realtime_relay import FakeUpstream, _proxy


class StableGate(SubjectGate):
    """Counts every question; names each user by a stable id (``ids``,
    editable: a rename moves a name to another id)."""

    def __init__(self) -> None:
        self.approves: list[str] = []
        self.stable: list[str] = []
        self.ids: dict[str, Any] = {"alice": "u1", "alicia": "u1"}

    def approves_overrides(self, subject: str) -> bool:
        self.approves.append(subject)
        return True

    def override_subject(self, subject: str) -> Any:
        self.stable.append(subject)
        value = self.ids.get(subject, subject)
        if isinstance(value, Exception):
            raise value
        return value


def _install(monkeypatch: pytest.MonkeyPatch, gate: StableGate) -> None:
    reg = Registry()
    reg.build_access_gate = lambda config, license: gate  # type: ignore[method-assign]
    monkeypatch.setattr(registry_mod, "_registry", reg)


def _as(user: str) -> dict[str, str]:
    return {**KEY, "x-test-user": user}


async def test_the_gate_is_asked_only_on_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = StableGate()
    _install(monkeypatch, gate)
    app = _app(tmp_path, Upstream(), detection=DetectionConfig(modes=(("phone_number", "block"),)))
    async with _client(app) as client:
        for text in ("hello", "hello again", f"mail {EMAIL}"):  # the email is redacted
            ok = await client.post("/v1/chat/completions", json=_chat(text), headers=_as("ada"))
            assert ok.status_code == 200
        assert gate.approves == [] and gate.stable == []
        refused = await client.post(
            "/v1/chat/completions", json=_chat("call +1 415 555 0132"), headers=_as("ada")
        )
    assert refused.status_code == 400 and refused.json()["error"]["message"].endswith(
        DASHBOARD_HINT
    )
    assert gate.approves == ["ada"] and gate.stable == ["ada"]


async def test_a_realtime_frame_asks_the_gate_nothing_until_it_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = StableGate()
    _install(monkeypatch, gate)
    async with FakeUpstream() as fake:
        config = Config(
            providers={
                **Config().providers,
                "openai": ProviderConfig(f"http://127.0.0.1:{fake.port}"),
            },
            detection=DetectionConfig(modes=(("email", "block"),)),
            overrides=OverridesConfig(path=str(tmp_path / "overrides.db")),
        )
        with _proxy(config) as host:
            async with websockets.connect(
                f"ws://{host}/v1/realtime", additional_headers={"x-test-user": "ada"}
            ) as ws:
                for _ in range(20):
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": "AA"}))
                    await ws.recv()
                assert gate.approves == [] and gate.stable == []
                await ws.send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {"content": [{"type": "input_text", "text": EMAIL}]},
                        }
                    )
                )
                with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                    await ws.recv()
    assert closed.value.rcvd is not None and closed.value.rcvd.code == 1008
    assert gate.approves == ["ada"] and gate.stable == ["ada"]


async def test_records_follow_the_stable_id_not_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = StableGate()
    _install(monkeypatch, gate)
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    csrf = {CSRF_HEADER: app.state.proxy.csrf_token}
    async with _client(app) as client:
        await client.post("/v1/chat/completions", json=_chat(EMAIL), headers=_as("alice"))
        (pending,) = _store(tmp_path).entries()
        assert pending.subject == "id:u1"
        listed = (await client.get("/__llm-redact/overrides", headers=_as("alice"))).json()
        assert listed["subject"] == "alice" and [e["id"] for e in listed["entries"]] == [pending.id]
        approved = await client.post(
            "/__llm-redact/overrides/approve",
            json={"id": pending.id, "scope": "always"},
            headers={"x-test-user": "alice", "cookie": "test-session=alice", **csrf},
        )
        assert approved.status_code == 200, approved.text
        # alice is renamed alicia (same id); another user takes "alice".
        gate.ids["alice"] = "u2"
        renamed = await client.post(
            "/v1/chat/completions", json=_chat(EMAIL), headers=_as("alicia")
        )
        assert renamed.status_code == 200
        newcomer = await client.post(
            "/v1/chat/completions", json=_chat(EMAIL), headers=_as("alice")
        )
        assert newcomer.status_code == 400
        theirs = (await client.get("/__llm-redact/overrides", headers=_as("alice"))).json()
    # The newcomer sees only their own fresh pending code, not alicia's rule.
    assert [e["state"] for e in theirs["entries"]] == ["pending"]
    assert {e.subject for e in _store(tmp_path).entries()} == {"id:u1", "id:u2"}


@pytest.mark.parametrize("answer", [RuntimeError("directory down"), "", None, 7])
async def test_a_gate_that_cannot_name_the_requester_gives_no_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: Any
) -> None:
    gate = StableGate()
    _install(monkeypatch, gate)
    # An every-time rule kept under the plain name (as before the member):
    # never used in its place.
    store = _store(tmp_path)
    code = store.record_pending("block", "ada", "openai", "POST", "/x", [("EMAIL", EMAIL)])
    store.approve("always", approver="ada", code=code)
    gate.ids["ada"] = answer
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    async with _client(app) as client:
        refused = await client.post("/v1/chat/completions", json=_chat(EMAIL), headers=_as("ada"))
        listing = await client.get("/__llm-redact/overrides", headers=_as("ada"))
    assert refused.status_code == 400 and "to allow" not in refused.text
    assert listing.status_code == 503 and "could not name" in listing.text
    assert not upstream.requests
    assert [e.state for e in _store(tmp_path).entries()] == ["always"]
