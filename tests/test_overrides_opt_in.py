"""Refusal overrides are OFF by default (owner decision, 10-01): an operator
opts in with ``[overrides] enabled = true``. End to end through the real
app: under the default config no refusal of any overridable kind carries a
code or a hint (HTTP and a realtime close reason, the local operator and a
named user alike), nothing is written to — or even created as — the store,
the override endpoints answer a local 404 naming the setting, and /status
says ``enabled: false``; opting in restores codes and approvals; approvals a
store already holds are INERT while off (the values stay refused, nothing in
the file changes) and apply again once overrides are turned back on."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets

import llm_redact.registry as registry_mod
from llm_redact.config import Config, ConfigError, OverridesConfig, parse_config
from llm_redact.overrides import OverrideStore, default_overrides_path
from llm_redact.proxy import CSRF_HEADER, create_app
from llm_redact.registry import Registry
from test_overrides import (
    CODE_RE,
    EMAIL,
    FORM,
    KEY,
    OTHER,
    PDF,
    SubjectGate,
    Upstream,
    _chat,
    _form,
)

PLAIN = {**KEY, "content-type": "text/plain"}
SUFFIX_BODY = {"model": "gpt-4o-mini", "training_file": "file-abc", "suffix": EMAIL}
NO_HINT = ("llm-redact override", "to allow", "Refusal overrides")


def _raw(**overrides: Any) -> dict[str, Any]:
    """A config file's contents: block-mode emails, binary uploads refused,
    and (only when given) an [overrides] table."""
    raw: dict[str, Any] = {
        "providers": {"openai": {"upstream_base_url": "http://upstream"}},
        "detection": {"modes": {"email": "block"}, "binary_uploads": "refuse"},
    }
    if overrides:
        raw["overrides"] = overrides
    return raw


def _app(raw: dict[str, Any], upstream: Upstream) -> Any:
    return create_app(parse_config(raw, "<test>"), upstream_transport=httpx.MockTransport(upstream))


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_no_hint(response: httpx.Response) -> None:
    assert response.status_code == 400, response.text
    message = response.json()["error"]["message"]
    for hint in NO_HINT:
        assert hint not in message, message


def test_overrides_are_off_by_default() -> None:
    assert Config().overrides.enabled is False
    assert OverridesConfig().enabled is False
    assert parse_config({}, "<test>").overrides.enabled is False
    assert parse_config({"overrides": {"ttl_minutes": 5}}, "<test>").overrides.enabled is False
    assert parse_config({"overrides": {"enabled": True}}, "<test>").overrides.enabled is True
    with pytest.raises(ConfigError, match=r"\[overrides\]"):
        parse_config({"overrides": {"enabled": "yes"}}, "<test>")


async def test_the_default_config_refuses_every_kind_with_no_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every overridable refusal kind under the default config: refused,
    nothing forwarded, no code and no hint; the default store path is never
    created; the endpoints 404 naming the setting; /status says off."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    upstream = Upstream()
    app = _app(_raw(), upstream)
    assert app.state.proxy.overrides is None
    async with _client(app) as client:
        refusals = [
            # block
            await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY),
            # unscanned_body
            await client.post("/v1/chat/completions", content=b"hello there", headers=PLAIN),
            # binary_upload (binary_uploads = "refuse")
            await client.post("/v1/files", content=_form(PDF), headers=FORM),
            # verbatim_field
            await client.post("/v1/fine_tuning/jobs", json=SUFFIX_BODY, headers=KEY),
        ]
        listing = await client.get("/__llm-redact/overrides")
        token = app.state.proxy.csrf_token
        guarded = {"content-type": "application/json", CSRF_HEADER: token}
        approve = await client.post(
            "/__llm-redact/overrides/approve",
            json={"id": "p1", "scope": "always"},
            headers=guarded,
        )
        revoke = await client.post(
            "/__llm-redact/overrides/revoke", json={"id": "r1"}, headers=guarded
        )
        status = (await client.get("/__llm-redact/status")).json()
        recent = (await client.get("/__llm-redact/recent")).json()["entries"]
    for refused in refusals:
        _assert_no_hint(refused)
    assert not upstream.requests
    for answer in (listing, approve, revoke):
        assert answer.status_code == 404
        assert "[overrides] enabled = true" in answer.json()["error"]
    assert status["overrides"] == {"enabled": False}
    assert all(row["override"] is None for row in recent)
    assert not default_overrides_path().exists()
    assert not (tmp_path / "data").exists()


async def test_values_in_an_inspected_upload_carry_no_code_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """binary_values: an inspector finds a (redact-mode) value in the
    extracted text of a binary part — refused with no code."""
    from test_upload_inspection import FakeInspector, reads

    reg = Registry()
    reg.build_upload_inspector = lambda config, tier: FakeInspector(  # type: ignore[method-assign]
        reads(f"contact {OTHER} for details")
    )
    monkeypatch.setattr(registry_mod, "_registry", reg)
    raw = _raw()
    raw["detection"] = {"binary_uploads": "refuse"}
    upstream = Upstream()
    async with _client(_app(raw, upstream)) as client:
        refused = await client.post("/v1/files", content=_form(PDF), headers=FORM)
    _assert_no_hint(refused)
    assert "EMAIL" in refused.text and not upstream.requests


async def test_a_named_user_gets_no_dashboard_hint_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A named user who could approve in the dashboard (with overrides on:
    DASHBOARD_HINT) gets no hint while they are off — and the access gate
    is never asked whether they could."""
    asked: list[str] = []

    class Gate(SubjectGate):
        def approves_overrides(self, subject: str) -> bool:
            asked.append(subject)
            return True

    reg = Registry()
    reg.build_access_gate = lambda config, license: Gate()  # type: ignore[method-assign]
    monkeypatch.setattr(registry_mod, "_registry", reg)
    upstream = Upstream()
    async with _client(_app(_raw(), upstream)) as client:
        refused = await client.post(
            "/v1/chat/completions",
            json=_chat(f"mail {EMAIL}"),
            headers={**KEY, "x-test-user": "alice"},
        )
    _assert_no_hint(refused)
    assert asked == []


async def test_opting_in_restores_codes_and_approvals(tmp_path: Path) -> None:
    store_path = tmp_path / "overrides.db"
    upstream = Upstream()
    app = _app(_raw(enabled=True, path=str(store_path)), upstream)
    async with _client(app) as client:
        refused = await client.post(
            "/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY
        )
        match = CODE_RE.search(refused.json()["error"]["message"])
        assert match is not None
        plain = await client.post("/v1/chat/completions", content=b"hello there", headers=PLAIN)
        assert CODE_RE.search(plain.json()["error"]["message"]) is not None
        OverrideStore(store_path).approve("once", approver=None, code=match.group(1))
        passed = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY)
        listing = await client.get("/__llm-redact/overrides")
        status = (await client.get("/__llm-redact/status")).json()
    assert passed.status_code == 200 and EMAIL in upstream.requests[-1].content.decode()
    assert listing.status_code == 200 and [e["state"] for e in listing.json()["entries"]] == [
        "pending"
    ]
    assert status["overrides"]["enabled"] is True
    assert status["overrides"]["used_total"] == {"once": 1}


def _kept_approvals(store_path: Path) -> None:
    """A store holding what an earlier opt-in approved: an every-time rule
    for EMAIL, a one-time grant for OTHER, and an every-time route rule for
    plain-text chat bodies."""
    writer = OverrideStore(store_path)
    for scope, kind, values in (
        ("always", "block", [("EMAIL", EMAIL)]),
        ("once", "block", [("EMAIL", OTHER)]),
        ("always", "unscanned_body", []),
    ):
        code = writer.record_pending(kind, "", "openai", "POST", "/v1/chat/completions", values)
        writer.approve(scope, approver=None, code=code)
    writer.close()


async def test_approvals_kept_in_the_store_are_inert_while_off(tmp_path: Path) -> None:
    store_path = tmp_path / "overrides.db"
    _kept_approvals(store_path)
    before = _digest(store_path)

    upstream = Upstream()
    off = _app(_raw(enabled=False, path=str(store_path)), upstream)
    async with _client(off) as client:
        for _ in range(2):
            _assert_no_hint(
                await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY)
            )
        _assert_no_hint(
            await client.post("/v1/chat/completions", json=_chat(f"mail {OTHER}"), headers=KEY)
        )
        _assert_no_hint(
            await client.post("/v1/chat/completions", content=b"hello there", headers=PLAIN)
        )
        status = (await client.get("/__llm-redact/status")).json()
    assert not upstream.requests
    assert status["overrides"] == {"enabled": False}
    # Never read, never written: the file is byte-identical, every approval
    # still there and unused.
    assert _digest(store_path) == before
    kept = OverrideStore(store_path, read_only=True).entries()
    assert sorted((e.state, e.kind, e.uses) for e in kept) == [
        ("always", "block", 0),
        ("always", "unscanned_body", 0),
        ("once", "block", 0),
    ]

    # Turned back on (a restart with enabled = true): they apply again.
    on = _app(_raw(enabled=True, path=str(store_path)), upstream)
    async with _client(on) as client:
        always = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY)
        once = await client.post("/v1/chat/completions", json=_chat(f"mail {OTHER}"), headers=KEY)
        once_again = await client.post(
            "/v1/chat/completions", json=_chat(f"mail {OTHER}"), headers=KEY
        )
        body = await client.post("/v1/chat/completions", content=b"hello there", headers=PLAIN)
    assert always.status_code == 200 and once.status_code == 200 and body.status_code == 200
    assert once_again.status_code == 400 and CODE_RE.search(once_again.text) is not None
    sent = [request.content for request in upstream.requests]
    assert EMAIL.encode() in sent[0] and OTHER.encode() in sent[1] and sent[2] == b"hello there"


async def test_a_realtime_close_reason_carries_no_code_by_default(tmp_path: Path) -> None:
    from test_realtime_relay import FakeUpstream, _proxy

    async with FakeUpstream() as fake:
        raw = _raw()
        raw["providers"] = {"openai": {"upstream_base_url": f"http://127.0.0.1:{fake.port}"}}
        with _proxy(parse_config(raw, "<test>")) as host:
            url = f"ws://{host}/v1/realtime?model=gpt-realtime"
            frame = {
                "type": "conversation.item.create",
                "item": {"type": "message", "content": [{"type": "input_text", "text": EMAIL}]},
            }
            async with websockets.connect(url) as client:
                await client.send(json.dumps(frame))
                with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                    await client.recv()
    assert closed.value.rcvd is not None and closed.value.rcvd.code == 1008
    assert closed.value.rcvd.reason == "blocked by llm-redact policy (EMAIL)"
    assert fake.received == []
