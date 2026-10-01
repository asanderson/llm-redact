"""Refusal overrides (overrides.py) end to end through the real app.

Every overridable refusal kind — a block-mode value, a verbatim field that
would be redacted, values found in an inspected binary upload, a binary part
under ``binary_uploads = "refuse"``, a body that is not a JSON object — is
driven: refused with a code → approved once → the next request passes once
(the value FORWARDED as sent) → the one after is refused again; approved
always → passes every time, only for that value and type. Pinned too: the
requester binding (another subject can neither use nor approve it), expired
and used codes, the atomic one-time use under concurrency, non-overridable
kinds carry no code, and values, digests and codes never reach the store
file, the logs, /status, /recent or the listing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.requests import HTTPConnection

import llm_redact.registry as registry_mod
from llm_redact.config import Config, ProviderConfig
from llm_redact.detection.engine import DetectionConfig
from llm_redact.overrides import (
    CODE_LENGTH,
    OverrideError,
    OverrideStore,
    allow_hint,
    normalize_code,
)
from llm_redact.plugin_api import Admission
from llm_redact.proxy import CSRF_HEADER, create_app
from llm_redact.registry import Registry

EMAIL = "jane.doe@corp.example"
OTHER = "john.roe@corp.example"
KEY = {"authorization": "Bearer sk-client"}
CODE_RE = re.compile(r"llm-redact override ([0-9A-Z]{12}) --once \| --always")


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"id": "file-1", "object": "file"})
        if "fine_tuning" in request.url.path:
            return httpx.Response(200, json={"id": "ftjob-1", "object": "fine_tuning.job"})
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )


class SubjectGate:
    """A stand-in access gate: the subject is the ``x-test-user`` header
    (scrubbed), on the API and the dashboard alike."""

    guards_dashboard = True

    def admit(self, conn: HTTPConnection, surface: str) -> Admission:
        user = None
        kept = []
        for name, value in conn.scope["headers"]:
            if name == b"x-test-user":
                user = value.decode()
            else:
                kept.append((name, value))
        conn.scope["headers"] = kept
        for attr in ("_headers",):
            if hasattr(conn, attr):
                delattr(conn, attr)
        return Admission(subject=user)

    def public_origin(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {}

    async def handle(self, request: Any, host: Any) -> Any:
        raise AssertionError("not used")

    def close(self) -> None:
        pass


def _config(tmp_path: Path, **kwargs: Any) -> Config:
    detection = kwargs.pop("detection", DetectionConfig(modes=(("email", "block"),)))
    overrides = kwargs.pop("overrides_path", tmp_path / "overrides.db")
    from llm_redact.config import OverridesConfig

    return Config(
        providers={**Config().providers, "openai": ProviderConfig("http://upstream")},
        detection=detection,
        overrides=kwargs.pop("overrides", OverridesConfig(path=str(overrides))),
        **kwargs,
    )


def _app(tmp_path: Path, upstream: Upstream, **kwargs: Any) -> Any:
    return create_app(_config(tmp_path, **kwargs), upstream_transport=httpx.MockTransport(upstream))


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _chat(text: str) -> dict[str, Any]:
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": text}]}


def _code(response: httpx.Response) -> str:
    message = response.json()["error"]["message"]
    match = CODE_RE.search(message)
    assert match is not None, message
    return match.group(1)


def _store(tmp_path: Path) -> OverrideStore:
    """The CLI's view of the same file (another connection)."""
    return OverrideStore(tmp_path / "overrides.db")


def _sent(upstream: Upstream) -> str:
    return upstream.requests[-1].content.decode()


# --- block mode -----------------------------------------------------------------------


async def test_block_once_passes_exactly_the_next_request(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    async with _client(app) as client:
        refused = await client.post(
            "/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY
        )
        assert refused.status_code == 400
        code = _code(refused)
        assert EMAIL not in refused.text
        assert not upstream.requests

        approved = _store(tmp_path).approve("once", approver=None, code=code)
        assert approved.state == "once" and approved.types == ("EMAIL",)

        passed = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY)
        assert passed.status_code == 200
        # Forwarded as sent, exactly like a warn-mode value.
        assert EMAIL in _sent(upstream)

        again = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY)
        assert again.status_code == 400
        assert _code(again) != code
        assert len(upstream.requests) == 1

        status = (await client.get("/__llm-redact/status")).json()
        recent = (await client.get("/__llm-redact/recent")).json()["entries"]
        metrics = (await client.get("/__llm-redact/metrics")).text
    assert status["overrides"]["used_total"] == {"once": 1}
    assert status["overrides"]["once"] == 0 and status["overrides"]["always"] == 0
    assert [row["override"] for row in recent] == [None, "once", None]  # newest first
    assert 'llm_redact_overrides_used_total{kind="once"} 1' in metrics
    # Never a value, a code or a digest in logs, status or rows.
    everything = caplog.text + json.dumps(status) + json.dumps(recent) + metrics
    assert EMAIL not in everything and code not in everything
    stored = (tmp_path / "overrides.db").read_bytes()
    assert EMAIL.encode() not in stored and code.encode() not in stored


async def test_block_always_covers_only_that_value(tmp_path: Path) -> None:
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    async with _client(app) as client:
        refused = await client.post(
            "/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY
        )
        _store(tmp_path).approve("always", approver=None, code=_code(refused))
        for _ in range(3):
            reply = await client.post(
                "/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY
            )
            assert reply.status_code == 200
        other = await client.post("/v1/chat/completions", json=_chat(f"mail {OTHER}"), headers=KEY)
        assert other.status_code == 400
        both = await client.post(
            "/v1/chat/completions", json=_chat(f"{EMAIL} and {OTHER}"), headers=KEY
        )
        assert both.status_code == 400
        status = (await client.get("/__llm-redact/status")).json()
    assert status["overrides"]["always"] == 1
    assert status["overrides"]["used_total"] == {"always": 3}
    entries = _store(tmp_path).entries()
    rule = next(e for e in entries if e.state == "always")
    assert rule.uses == 3 and rule.route == "any route" and rule.types == ("EMAIL",)
    # Revoked: refused again.
    _store(tmp_path).revoke(rule.id)
    async with _client(app) as client:
        reply = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY)
    assert reply.status_code == 400


async def test_a_once_grant_is_used_by_exactly_one_of_two_parallel_requests(
    tmp_path: Path,
) -> None:
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    async with _client(app) as client:
        refused = await client.post(
            "/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY
        )
        _store(tmp_path).approve("once", approver=None, code=_code(refused))
        replies = await asyncio.gather(
            *(
                client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY)
                for _ in range(2)
            )
        )
    assert sorted(r.status_code for r in replies) == [200, 400]
    assert len(upstream.requests) == 1


def test_a_consumed_grant_fails_a_stale_scope(tmp_path: Path) -> None:
    """Two requests that both read the grant before either used it (an
    await between, or two processes): the second commit fails."""
    from llm_redact.overrides import OverrideScope

    store = _store(tmp_path)
    code = store.record_pending("block", "", "openai", "POST", "/v1/x", [("EMAIL", EMAIL)])
    store.approve("once", approver=None, code=code)
    first, second = OverrideScope(store, ""), OverrideScope(store, "")
    assert first.allows("EMAIL", EMAIL) and second.allows("EMAIL", EMAIL)
    assert first.commit() == (True, "once")
    assert second.commit() == (False, None)


async def test_the_race_is_refused_through_the_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    store = app.state.proxy.overrides
    monkeypatch.setattr(store, "consume", lambda once, always: False)
    async with _client(app) as client:
        refused = await client.post(
            "/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY
        )
        _store(tmp_path).approve("once", approver=None, code=_code(refused))
        raced = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY)
    assert raced.status_code == 400
    assert "already used" in raced.json()["error"]["message"]
    assert not upstream.requests


# --- codes ----------------------------------------------------------------------------


def test_codes_are_single_use_and_expire(tmp_path: Path) -> None:
    now = [1000.0]
    store = OverrideStore(tmp_path / "o.db", ttl_seconds=60, clock=lambda: now[0])
    code = store.record_pending("block", "", "openai", "POST", "/v1/x", [("EMAIL", EMAIL)])
    assert len(code) == CODE_LENGTH and normalize_code(code.lower()) == code
    assert normalize_code(f"{code[:4]}-{code[4:]}") == code
    assert normalize_code("short") is None and normalize_code("U" * CODE_LENGTH) is None
    store.approve("once", approver=None, code=code)
    with pytest.raises(OverrideError, match="unknown, used or expired"):
        store.approve("once", approver=None, code=code)
    late = store.record_pending("block", "", "openai", "POST", "/v1/x", [("EMAIL", EMAIL)])
    now[0] += 61
    with pytest.raises(OverrideError, match="unknown, used or expired"):
        store.describe(late)
    with pytest.raises(OverrideError, match="unknown, used or expired"):
        store.approve("always", approver=None, code=late)
    # The approved one-time grant expired too.
    assert store.snapshot("").once_values == []
    with pytest.raises(OverrideError, match="once or always"):
        store.approve("forever", approver=None, code=late)


def test_the_store_file_is_private_and_bounded(tmp_path: Path) -> None:
    store = OverrideStore(tmp_path / "d" / "o.db", max_pending=3)
    assert not (tmp_path / "d").exists()  # nothing written before a code
    assert store.entries() == [] and store.counts() == {"pending": 0, "once": 0, "always": 0}
    for _ in range(5):
        store.record_pending("block", "", "openai", "POST", "/v1/x", [("EMAIL", EMAIL)])
    assert (tmp_path / "d").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "d" / "o.db").stat().st_mode & 0o777 == 0o600
    assert store.counts()["pending"] == 3  # oldest dropped
    store.close()
    assert EMAIL.encode() not in (tmp_path / "d" / "o.db").read_bytes()


def test_revoke_and_listing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    code = store.record_pending("block", "alice", "openai", "POST", "/v1/x", [("EMAIL", EMAIL)])
    (pending,) = store.entries()
    assert pending.state == "pending" and pending.subject == "alice"
    assert pending.route == "POST openai /v1/x"
    assert store.entries("bob") == []
    with pytest.raises(OverrideError, match="another requester"):
        store.approve("once", approver="bob", code=code)
    with pytest.raises(OverrideError, match="to revoke"):
        store.revoke(pending.id, subject="bob")
    store.revoke(pending.id, subject="alice")
    with pytest.raises(OverrideError, match="looks like"):
        store.revoke("x9")
    with pytest.raises(OverrideError, match="looks like"):
        store.approve("once", approver=None, pending_id="r1")
    with pytest.raises(OverrideError, match="to revoke"):
        store.revoke("r99")
    assert json.dumps([e.as_dict() for e in store.entries()]) == "[]"


def test_a_store_fault_never_overrides(tmp_path: Path) -> None:
    """A store that cannot be read is no override (the refusal stands), and
    one that cannot mint a code leaves the refusal without one."""
    from llm_redact.overrides import OverrideScope

    bad = tmp_path / "not-a-dir"
    bad.write_text("x")
    scope = OverrideScope(OverrideStore(bad / "o.db"), "")
    assert scope.allows("EMAIL", EMAIL) is False
    assert scope.refusal_code("block", "openai", "POST", "/v1/x") is None
    assert scope.commit() == (True, None)
    broken = OverrideStore(bad / "o.db")
    broken.path = tmp_path  # a directory: exists, cannot be opened
    scope = OverrideScope(broken, "")
    assert scope.allows("EMAIL", EMAIL) is False
    scope._once.add(1)  # a grant it relied on: the use cannot be recorded
    assert scope.commit() == (False, None)


# --- the requester binding ------------------------------------------------------------


def _gated(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = Registry()
    reg.build_access_gate = lambda config, license: SubjectGate()  # type: ignore[method-assign]
    monkeypatch.setattr(registry_mod, "_registry", reg)


async def test_only_the_requester_uses_and_approves_its_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gated(monkeypatch)
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    alice = {**KEY, "x-test-user": "alice"}
    bob = {**KEY, "x-test-user": "bob"}
    async with _client(app) as client:
        refused = await client.post(
            "/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=alice
        )
        code = _code(refused)
        # The operator (CLI, no subject) and another user cannot approve it.
        with pytest.raises(OverrideError, match="another requester"):
            _store(tmp_path).approve("always", approver=None, code=code)
        with pytest.raises(OverrideError, match="another requester"):
            _store(tmp_path).approve("always", approver="bob", code=code)
        _store(tmp_path).approve("always", approver="alice", code=code)
        assert (
            await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=alice)
        ).status_code == 200
        assert (
            await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=bob)
        ).status_code == 400
        # Unattributed requests are the operator's: not alice's rule either.
        assert (
            await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY)
        ).status_code == 400


async def test_the_dashboard_endpoints_serve_the_admitted_subject_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gated(monkeypatch)
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    csrf = {CSRF_HEADER: app.state.proxy.csrf_token}
    async with _client(app) as client:
        await client.post(
            "/v1/chat/completions",
            json=_chat(f"mail {EMAIL}"),
            headers={**KEY, "x-test-user": "alice"},
        )
        await client.post(
            "/v1/chat/completions",
            json=_chat(f"mail {OTHER}"),
            headers={**KEY, "x-test-user": "bob"},
        )
        listed = (
            await client.get("/__llm-redact/overrides", headers={"x-test-user": "alice"})
        ).json()
        (entry,) = listed["entries"]
        assert listed["subject"] == "alice" and entry["types"] == ["EMAIL"]
        assert listed["can_approve"] is True
        assert entry["state"] == "pending" and EMAIL not in json.dumps(listed)
        # No CSRF token: refused. Bob cannot approve alice's.
        no_csrf = await client.post(
            "/__llm-redact/overrides/approve",
            json={"id": entry["id"], "scope": "once"},
            headers={"x-test-user": "alice"},
        )
        assert no_csrf.status_code == 403
        as_bob = await client.post(
            "/__llm-redact/overrides/approve",
            json={"id": entry["id"], "scope": "once"},
            headers={"x-test-user": "bob", **csrf},
        )
        assert as_bob.status_code == 400 and "another requester" in as_bob.json()["error"]
        ok = await client.post(
            "/__llm-redact/overrides/approve",
            json={"id": entry["id"], "scope": "always"},
            headers={"x-test-user": "alice", **csrf},
        )
        assert ok.status_code == 200 and ok.json()["approved"]["state"] == "always"
        passed = await client.post(
            "/v1/chat/completions",
            json=_chat(f"mail {EMAIL}"),
            headers={**KEY, "x-test-user": "alice"},
        )
        assert passed.status_code == 200
        rules = (
            await client.get("/__llm-redact/overrides", headers={"x-test-user": "alice"})
        ).json()["entries"]
        (rule,) = rules
        assert rule["state"] == "always" and rule["uses"] == 1
        bad = await client.post(
            "/__llm-redact/overrides/revoke",
            json={"id": 7},
            headers={"x-test-user": "alice", **csrf},
        )
        assert bad.status_code == 400
        not_bobs = await client.post(
            "/__llm-redact/overrides/revoke",
            json={"id": rule["id"]},
            headers={"x-test-user": "bob", **csrf},
        )
        assert not_bobs.status_code == 400
        revoked = await client.post(
            "/__llm-redact/overrides/revoke",
            json={"id": rule["id"]},
            headers={"x-test-user": "alice", **csrf},
        )
        assert revoked.json() == {"revoked": rule["id"]}
        assert (await client.post("/__llm-redact/overrides", json={})).status_code == 405
        assert (await client.get("/__llm-redact/overrides/approve")).status_code == 405
        rebinding = await client.get("/__llm-redact/overrides", headers={"host": "evil.example"})
        assert rebinding.status_code == 403
        cross = await client.get(
            "/__llm-redact/overrides", headers={"origin": "http://evil.example"}
        )
        assert cross.status_code == 403


@pytest.mark.parametrize("gated", [False, True])
async def test_the_dashboard_endpoints_approve_nothing_without_a_sign_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gated: bool
) -> None:
    """Without a subject an access gate signed in to the dashboard, the
    CSRF token proves no person (any local client — an agent with curl —
    can read it from the dashboard): the approve and revoke POSTs are
    refused 403, pointing to the CLI, and the listing says so."""
    if gated:
        _gated(monkeypatch)  # guards the dashboard; no x-test-user: no subject
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    csrf = {CSRF_HEADER: app.state.proxy.csrf_token}
    async with _client(app) as client:
        refused = await client.post(
            "/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY
        )
        assert refused.status_code == 400
        listed = (await client.get("/__llm-redact/overrides")).json()
        assert listed["can_approve"] is False
        (entry,) = listed["entries"]
        for action, body in (
            ("approve", {"id": entry["id"], "scope": "always"}),
            ("revoke", {"id": entry["id"]}),
        ):
            reply = await client.post(f"/__llm-redact/overrides/{action}", json=body, headers=csrf)
            assert reply.status_code == 403, reply.text
            assert "llm-redact override CODE" in reply.json()["error"]
        again = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY)
    assert again.status_code == 400
    assert not upstream.requests
    assert [e.state for e in _store(tmp_path).entries()] == ["pending", "pending"]


# --- verbatim fields ------------------------------------------------------------------


async def test_a_verbatim_field_override(tmp_path: Path) -> None:
    upstream = Upstream()
    app = _app(tmp_path, upstream, detection=DetectionConfig())
    body = {"model": "gpt-4o-mini", "training_file": "file-abc", "suffix": EMAIL}
    async with _client(app) as client:
        refused = await client.post("/v1/fine_tuning/jobs", json=body, headers=KEY)
        assert refused.status_code == 400 and "`suffix`" in refused.text
        _store(tmp_path).approve("once", approver=None, code=_code(refused))
        passed = await client.post("/v1/fine_tuning/jobs", json=body, headers=KEY)
        assert passed.status_code == 200
        assert json.loads(_sent(upstream))["suffix"] == EMAIL
        again = await client.post("/v1/fine_tuning/jobs", json=body, headers=KEY)
        assert again.status_code == 400


# --- uploads --------------------------------------------------------------------------

PDF = b"%PDF-1.7\n1 0 obj << >> endobj\n%%EOF\n"
FORM = {**KEY, "content-type": "multipart/form-data; boundary=b"}


def _form(content: bytes) -> bytes:
    return (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="r.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n" + content + b"\r\n--b--\r\n"
    )


async def test_a_refused_binary_upload_override(tmp_path: Path) -> None:
    upstream = Upstream()
    app = _app(tmp_path, upstream, detection=DetectionConfig(binary_uploads="refuse"))
    async with _client(app) as client:
        refused = await client.post("/v1/files", content=_form(PDF), headers=FORM)
        assert refused.status_code == 400 and "binary" in refused.text
        code = _code(refused)
        (pending,) = _store(tmp_path).entries()
        assert pending.kind == "binary_upload" and pending.route == "POST openai /v1/files"
        _store(tmp_path).approve("once", approver=None, code=code)
        passed = await client.post("/v1/files", content=_form(PDF), headers=FORM)
        assert passed.status_code == 200
        assert PDF in upstream.requests[-1].content
        again = await client.post("/v1/files", content=_form(PDF), headers=FORM)
        assert again.status_code == 400
        _store(tmp_path).approve("always", approver=None, code=_code(again))
        for _ in range(2):
            assert (
                await client.post("/v1/files", content=_form(PDF), headers=FORM)
            ).status_code == 200
        # A rule on this route only: another route's upload stays refused.
        other = await client.post("/custom/nope/v1/files", content=_form(PDF), headers=FORM)
        assert other.status_code != 200
        status = (await client.get("/__llm-redact/status")).json()
    assert status["unscanned_uploads_total"] == {"openai": 3}
    assert status["overrides"]["used_total"] == {"once": 1, "always": 2}


async def test_an_override_rule_is_unused_by_an_upload_without_binary_parts(
    tmp_path: Path,
) -> None:
    upstream = Upstream()
    app = _app(tmp_path, upstream, detection=DetectionConfig(binary_uploads="refuse"))
    text_form = _form(PDF).replace(b"application/pdf", b"text/plain").replace(PDF, b"plain notes")
    async with _client(app) as client:
        refused = await client.post("/v1/files", content=_form(PDF), headers=FORM)
        _store(tmp_path).approve("once", approver=None, code=_code(refused))
        text = await client.post("/v1/files", content=text_form, headers=FORM)
        assert text.status_code == 200
        # The grant is still there for the binary upload.
        assert (await client.post("/v1/files", content=_form(PDF), headers=FORM)).status_code == 200


async def test_values_in_an_inspected_binary_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_upload_inspection import FakeInspector, reads

    inspector = FakeInspector(reads(f"contact {EMAIL} for details"))
    reg = Registry()
    reg.build_upload_inspector = lambda config, tier: inspector  # type: ignore[method-assign]
    monkeypatch.setattr(registry_mod, "_registry", reg)
    upstream = Upstream()
    app = _app(tmp_path, upstream, detection=DetectionConfig(binary_uploads="refuse"))
    async with _client(app) as client:
        refused = await client.post("/v1/files", content=_form(PDF), headers=FORM)
        assert refused.status_code == 400 and "EMAIL" in refused.text
        code = _code(refused)
        (pending,) = _store(tmp_path).entries()
        assert pending.kind == "binary_values" and pending.types == ("EMAIL",)
        _store(tmp_path).approve("once", approver=None, code=code)
        passed = await client.post("/v1/files", content=_form(PDF), headers=FORM)
        assert passed.status_code == 200
        assert PDF in upstream.requests[-1].content
        status = (await client.get("/__llm-redact/status")).json()
        again = await client.post("/v1/files", content=_form(PDF), headers=FORM)
        assert again.status_code == 400
    # Cleared by its (overridden) clean scan: not counted unscanned.
    assert status["unscanned_uploads_total"] == {}


async def test_a_block_value_in_an_upload(tmp_path: Path) -> None:
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    jsonl = (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="d.jsonl"\r\n'
        b"Content-Type: application/jsonl\r\n\r\n"
        + json.dumps({"note": EMAIL}).encode()
        + b"\r\n--b--\r\n"
    )
    async with _client(app) as client:
        refused = await client.post("/v1/files", content=jsonl, headers=FORM)
        assert refused.status_code == 400
        _store(tmp_path).approve("once", approver=None, code=_code(refused))
        passed = await client.post("/v1/files", content=jsonl, headers=FORM)
        assert passed.status_code == 200
        assert EMAIL.encode() in upstream.requests[-1].content


# --- bodies the proxy cannot read -----------------------------------------------------


async def test_a_body_that_is_not_json(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    plain = {**KEY, "content-type": "text/plain"}
    async with _client(app) as client:
        refused = await client.post("/v1/chat/completions", content=b"hello there", headers=plain)
        assert refused.status_code == 400 and "not a JSON object" in refused.text
        _store(tmp_path).approve("once", approver=None, code=_code(refused))
        passed = await client.post("/v1/chat/completions", content=b"hello there", headers=plain)
        assert passed.status_code == 200
        assert upstream.requests[-1].content == b"hello there"
        # A top-level scalar is the same kind: its own route rule applies.
        again = await client.post("/v1/chat/completions", content=b"42", headers=KEY)
        assert again.status_code == 400
        _store(tmp_path).approve("always", approver=None, code=_code(again))
        for _ in range(2):
            reply = await client.post("/v1/chat/completions", content=b"42", headers=KEY)
            assert reply.status_code == 200
    assert "forwarded unscanned" in caplog.text


async def test_an_unscanned_body_rule_never_lets_json_past_redaction(tmp_path: Path) -> None:
    """An every-time approval of a plain-text body on a route must not
    reach a body a first-value or lenient reader (Go encoding/json: Ollama,
    many gateways) reads as a chat request: JSON with one trailing byte, or
    with one invalid UTF-8 byte, is refused with no code and never sent."""
    upstream = Upstream()
    app = _app(tmp_path, upstream, detection=DetectionConfig())
    chat = json.dumps(_chat(f"mail {EMAIL}")).encode()
    variants = [
        chat + b"\x00",
        chat.replace(b"jane", b"jan\xffe"),
        b"\xef\xbb\xbf \x00" + chat + b"x",
        b"\x0b\x0c" + chat + b"x",
        chat.decode().encode("utf-16-be") + b"\x00x",
        b"[" + chat + b"]",
        b"   ",
    ]
    async with _client(app) as client:
        refused = await client.post(
            "/v1/chat/completions", content=b"hello", headers={**KEY, "content-type": "text/plain"}
        )
        _store(tmp_path).approve("always", approver=None, code=_code(refused))
        for body in variants:
            reply = await client.post(
                "/v1/chat/completions",
                content=body,
                headers={**KEY, "content-type": "application/json"},
            )
            assert reply.status_code == 400, body
            assert "llm-redact override" not in reply.text
        # A plain-text body still passes on the approval.
        plain = await client.post(
            "/v1/chat/completions", content=b"hello", headers={**KEY, "content-type": "text/plain"}
        )
        assert plain.status_code == 200
    assert all(EMAIL.encode() not in sent.content for sent in upstream.requests)
    assert [sent.content for sent in upstream.requests] == [b"hello"]


@pytest.mark.parametrize(
    "content_type",
    [
        "multipart/form-data; boundary=a; boundary=b",
        "multipart/form-data",
        "application/x-www-form-urlencoded",
    ],
)
async def test_a_multipart_or_form_body_carries_no_code(tmp_path: Path, content_type: str) -> None:
    """Multipart framing two readers could disagree on (a repeated boundary
    makes the proxy's reading None) is final: no code, nothing sent."""
    upstream = Upstream()
    app = _app(tmp_path, upstream, detection=DetectionConfig())
    body = (
        b'--a\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nassistants\r\n'
        b'--a\r\nContent-Disposition: form-data; name="file"; filename="x.txt"\r\n\r\n'
        b"mail " + EMAIL.encode() + b"\r\n--a--\r\n"
    )
    async with _client(app) as client:
        reply = await client.post(
            "/v1/files", content=body, headers={**KEY, "content-type": content_type}
        )
    assert reply.status_code == 400 and "llm-redact override" not in reply.text
    assert not upstream.requests


@pytest.mark.parametrize(
    ("headers", "body", "status"),
    [
        ({"content-encoding": "gzip", "content-type": "application/json"}, b"\x1f\x8b..", 415),
        ({"content-type": "application/json"}, b"[" * 200 + b"]" * 200, 400),
        ({"content-type": "multipart/form-data; boundary=b"}, b"--b\r\nbroken", 400),
    ],
)
async def test_framing_refusals_carry_no_code(
    tmp_path: Path, headers: dict[str, str], body: bytes, status: int
) -> None:
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    async with _client(app) as client:
        reply = await client.post("/v1/chat/completions", content=body, headers={**KEY, **headers})
    assert reply.status_code == status
    assert "llm-redact override" not in reply.text
    assert not (tmp_path / "overrides.db").exists()


async def test_a_repeated_content_type_carries_no_code(tmp_path: Path) -> None:
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    headers = [
        ("authorization", "Bearer sk-client"),
        ("content-type", "text/plain"),
        ("content-type", "application/json"),
    ]
    async with _client(app) as client:
        reply = await client.post("/v1/chat/completions", content=b"hello", headers=headers)
    assert reply.status_code == 400 and "llm-redact override" not in reply.text


async def test_overrides_disabled_mint_no_code(tmp_path: Path) -> None:
    from llm_redact.config import OverridesConfig

    upstream = Upstream()
    app = _app(tmp_path, upstream, overrides=OverridesConfig(enabled=False))
    async with _client(app) as client:
        blocked = await client.post(
            "/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY
        )
        plain = await client.post(
            "/v1/chat/completions", content=b"hi", headers={**KEY, "content-type": "text/plain"}
        )
        listing = await client.get("/__llm-redact/overrides")
        status = (await client.get("/__llm-redact/status")).json()
        metrics = (await client.get("/__llm-redact/metrics")).text
    assert blocked.status_code == 400 and plain.status_code == 400
    assert "llm-redact override" not in blocked.text + plain.text
    assert listing.status_code == 404
    assert status["overrides"] == {"enabled": False}
    assert "llm_redact_overrides_used_total" in metrics
    assert not (tmp_path / "overrides.db").exists()


async def test_status_survives_an_unreadable_store(tmp_path: Path) -> None:
    upstream = Upstream()
    app = _app(tmp_path, upstream)
    app.state.proxy.overrides.path = tmp_path  # a directory: exists, cannot be opened
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
        blocked = await client.post(
            "/v1/chat/completions", json=_chat(f"mail {EMAIL}"), headers=KEY
        )
    assert status["overrides"]["enabled"] is True and "always" not in status["overrides"]
    assert blocked.status_code == 400 and "llm-redact override" not in blocked.text


def test_the_hint_text() -> None:
    assert allow_hint("ABC") == "to allow: llm-redact override ABC --once | --always"


# --- config ---------------------------------------------------------------------------


def test_the_overrides_section_parses_and_round_trips(tmp_path: Path) -> None:
    from llm_redact.config import ConfigError, OverridesConfig, parse_config
    from llm_redact.config_write import emit_config_toml as emit_config

    parsed = parse_config(
        {"overrides": {"enabled": False, "ttl_minutes": 5, "path": "/x/o.db"}}, "t"
    )
    assert parsed.overrides == OverridesConfig(enabled=False, ttl_minutes=5, path="/x/o.db")
    assert "[overrides]" not in emit_config(Config())
    emitted = emit_config(parsed)
    assert "[overrides]" in emitted and 'path = "/x/o.db"' in emitted
    for bad in ({"ttl_minutes": 0}, {"ttl_minutes": 99999}, {"nope": 1}, {"enabled": "yes"}):
        with pytest.raises(ConfigError, match=r"\[overrides\]"):
            parse_config({"overrides": bad}, "t")
