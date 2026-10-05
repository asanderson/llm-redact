"""The access gate's optional ``authorize_content`` over HTTP.

Asked once for every forwarded request AFTER its redaction (the vault batch
and a refusal override's use committed) and BEFORE the ``[audit] required``
START row, the upstream authorizer, a routed plan's ``begin()`` and any
upstream contact, with the SAME ``AuthorizationRequest`` that
``authorize_request`` got (built alike without it) and the request's OWN
``ContentFacts`` — value types and counts, never a value. Refusals are
recorded 403s (kind ``authorization``); a fault or timeout refuses with the
core's fixed text (bookkeeping stage ``authorization``). Keyless: scripted
fakes on a bare Registry stand in for llm-redact-pro.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from llm_redact.authorization import AUTHORIZATION_FAULT, AUTHORIZATION_STAGE, GateAuthorization
from llm_redact.config import AuditConfig, OverridesConfig, ProviderConfig
from llm_redact.detection.engine import DetectionConfig
from llm_redact.overrides import OverrideStore
from llm_redact.plugin_api import Admission, AuthorizationRequest, ContentFacts
from llm_redact.registry import Registry
from local_refusals import refused_once
from test_access_seam import FakeGate
from test_audit_required import FakeAudit
from test_authorization_seam import (
    ANTHROPIC,
    EMAIL,
    KEY,
    Upstream,
    _app,
    _client,
    _messages,
    _never,
    _post,
    _registry,
)

REASON = "llm-redact: your role may not send credentials, even redacted"
OTHER = "john.roe@corp.example"
THIRD = "mary.major@corp.example"


class ContentGate(FakeGate):
    """A gate declaring ``authorize_content`` (and, with ``requests``,
    ``authorize_request``): records every question and answers
    ``decide(request, content)``."""

    def __init__(
        self,
        decide: Callable[[AuthorizationRequest, ContentFacts], Any] | None = None,
        *,
        subject: str | None = "ada",
    ) -> None:
        super().__init__()
        self.decide = decide or (lambda request, content: None)
        self.subject = subject
        self.asked: list[tuple[AuthorizationRequest, ContentFacts]] = []

    def admit(self, conn: Any, surface: str) -> Admission:
        return Admission(subject=self.subject)

    def authorize_content(self, request: AuthorizationRequest, content: ContentFacts) -> Any:
        self.asked.append((request, content))
        return self.decide(request, content)


class BothGate(ContentGate):
    """``authorize_request`` too: the content check gets its very object."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.requests: list[AuthorizationRequest] = []

    def authorize_request(self, request: AuthorizationRequest) -> None:
        self.requests.append(request)


def _facts(**kwargs: Any) -> ContentFacts:
    return ContentFacts(**{"scanned": True, **kwargs})


# --- the facts ------------------------------------------------------------------------


async def test_a_matched_route_is_asked_with_its_redactions_and_the_same_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = BothGate()
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream)
    body = _messages(f"mail {EMAIL} and {OTHER}")
    assert (await _post(app, "/v1/messages", body, ANTHROPIC)).status_code == 200
    ((request, content),) = gate.asked
    # The very object authorize_request was asked with.
    assert gate.requests == [request] and request is gate.requests[0]
    assert content == _facts(detected=(("EMAIL", 2),))
    assert EMAIL not in upstream.sent()


async def test_without_authorize_request_the_facts_are_built_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = ContentGate()
    _registry(monkeypatch, gate)
    app = _app(Upstream())
    assert (await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)).is_success
    ((request, content),) = gate.asked
    assert request == AuthorizationRequest(
        surface="http",
        provider="anthropic",
        adapter="anthropic",
        kind="chat",
        method="POST",
        path="/v1/messages",
        model="claude-sonnet-4-5",
        identity=False,
    )
    assert content == _facts(detected=(("EMAIL", 1),))


async def test_counts_are_sorted_and_carry_no_value(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = ContentGate()
    _registry(monkeypatch, gate)
    app = _app(Upstream())
    text = f"mail {EMAIL} from 10.1.2.3 and {OTHER}"
    assert (await _post(app, "/v1/messages", _messages(text), ANTHROPIC)).is_success
    ((_request, content),) = gate.asked
    assert content.detected == (("EMAIL", 2), ("IPV4", 1))
    assert EMAIL not in repr(content) and "«" not in repr(content)


async def test_warn_mode_values_are_reported_as_warned(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = ContentGate()
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream, detection=DetectionConfig(modes=(("email", "warn"),)))
    assert (await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)).is_success
    ((_request, content),) = gate.asked
    assert content == _facts(warned=(("EMAIL", 1),))
    assert EMAIL in upstream.sent()  # warn mode forwards it, honestly reported


async def test_detection_off_is_asked_as_unscanned(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = ContentGate()
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(
        upstream, providers={"anthropic": ProviderConfig("https://upstream.test", detection=False)}
    )
    assert (await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)).is_success
    ((request, content),) = gate.asked
    assert request.adapter == "anthropic"
    assert content == ContentFacts(scanned=False)
    assert EMAIL in upstream.sent()


async def test_a_pass_through_request_is_asked_as_unscanned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = BothGate()
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream)
    body = {"content": f"mail {EMAIL}"}
    response = await _post(app, "/v1/threads/thread_abc/messages", body, KEY)
    assert response.status_code == 200
    ((request, content),) = gate.asked
    assert request is gate.requests[0]
    assert (request.adapter, request.kind, request.model) == (None, "none", None)
    assert content == ContentFacts(scanned=False)


async def test_a_body_less_request_is_asked_as_unscanned(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = ContentGate()
    _registry(monkeypatch, gate)
    app = _app(Upstream())
    async with _client(app) as client:
        assert (await client.get("/v1/models", headers=KEY)).status_code == 200
    ((request, content),) = gate.asked
    assert request.method == "GET" and content == ContentFacts(scanned=False)


PDF = b"%PDF-1.7\n1 0 obj << >> endobj\n%%EOF\n"


def _form(content: bytes, filename: str = "a.pdf", content_type: str = "application/pdf") -> bytes:
    return (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="'
        + filename.encode()
        + b'"\r\nContent-Type: '
        + content_type.encode()
        + b"\r\n\r\n"
        + content
        + b"\r\n--b--\r\n"
    )


async def test_an_upload_reports_its_unscanned_binary_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = ContentGate(lambda request, content: REASON if content.unscanned_parts else None)
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream)
    form = {**KEY, "content-type": "multipart/form-data; boundary=b"}
    async with _client(app) as client:
        text = await client.post(
            "/v1/files",
            content=_form(f"mail {EMAIL}".encode(), "a.txt", "text/plain"),
            headers=form,
        )
        binary = await client.post("/v1/files", content=_form(PDF), headers=form)
    assert text.status_code == 200 and binary.status_code == 403
    assert [content for _request, content in gate.asked] == [
        _facts(detected=(("EMAIL", 1),)),
        _facts(unscanned_parts=1),
    ]
    assert len(upstream.requests) == 1
    state = app.state.proxy
    # The refused upload never counts its binary part as forwarded.
    assert state.unscanned_uploads == {}
    refused_once(state, "authorization", "openai")


# --- concurrency: each request sees only its own counts ---------------------------------


async def test_concurrent_requests_each_see_only_their_own_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    both_in = asyncio.Event()
    entered: list[ContentFacts] = []

    async def decide(request: AuthorizationRequest, content: ContentFacts) -> None:
        entered.append(content)
        if len(entered) == 2:
            both_in.set()
        # Yields while the other request redacts (and is asked) meanwhile.
        await asyncio.wait_for(both_in.wait(), 5)

    gate = ContentGate(decide)
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream, detection=DetectionConfig(modes=(("ipv4", "warn"),)))
    one = _messages(f"mail {EMAIL}")
    three = _messages(f"mail {EMAIL} {OTHER} {THIRD} from 10.1.2.3")
    first, second = await asyncio.gather(
        _post(app, "/v1/messages", one, ANTHROPIC), _post(app, "/v1/messages", three, ANTHROPIC)
    )
    assert first.status_code == second.status_code == 200
    by_count = sorted((content for _request, content in gate.asked), key=lambda c: c.detected)
    assert by_count == [
        _facts(detected=(("EMAIL", 1),)),
        _facts(detected=(("EMAIL", 3),), warned=(("IPV4", 1),)),
    ]


# --- refusals and faults ------------------------------------------------------------------


async def test_a_refusal_is_a_recorded_403_before_the_audit_start_row_and_upstream(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    audit = FakeAudit()
    reg = Registry()
    reg.build_audit = lambda cfg: audit if cfg.enabled else None
    gate = ContentGate(lambda request, content: REASON if content.detected else None)
    _registry(monkeypatch, gate, reg)
    upstream = Upstream()
    app = _app(upstream, audit=AuditConfig(enabled=True, required=True))
    state = app.state.proxy
    caplog.set_level(logging.INFO, logger="llm_redact")
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    assert response.status_code == 403
    assert response.json()["type"] == "error" and response.json()["error"]["message"] == REASON
    # No START row, nothing sent; the END-only row of a local refusal.
    assert audit.begun == [] and upstream.requests == []
    (row,) = state.recent
    assert (row["status"], row["user"], row["detections"]) == (403, "ada", {"EMAIL": 1})
    assert REASON not in caplog.text and EMAIL not in caplog.text
    assert "403 refused by the access gate (authorization)" in caplog.text
    refused_once(state, "authorization", "anthropic")
    # Placeholders the redaction issued stay (nothing was forwarded).
    assert len(state.vault) == 1
    # A clean request passes and writes its START row.
    assert (await _post(app, "/v1/messages", _messages("hello"), ANTHROPIC)).is_success
    assert len(audit.begun) == 1


@pytest.mark.parametrize(
    ("decide", "logged"),
    [
        (lambda request, content: (_ for _ in ()).throw(KeyError(EMAIL)), "KeyError"),
        (lambda request, content: True, "answered bool"),
        (lambda request, content: "", "answered str"),
    ],
    ids=["raises", "true", "empty"],
)
async def test_a_failing_or_nonsense_answer_refuses_with_the_core_text(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, decide: Any, logged: str
) -> None:
    _registry(monkeypatch, ContentGate(decide))
    upstream = Upstream()
    app = _app(upstream)
    caplog.set_level(logging.INFO, logger="llm_redact")
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    assert response.status_code == 403
    assert response.json()["error"]["message"] == AUTHORIZATION_FAULT
    assert upstream.requests == []
    state = app.state.proxy
    assert state.bookkeeping_errors[AUTHORIZATION_STAGE] == 1
    assert logged in caplog.text and EMAIL not in caplog.text
    refused_once(state, "authorization", "anthropic")


async def test_an_awaitable_answer_is_awaited(monkeypatch: pytest.MonkeyPatch) -> None:
    async def decide(request: AuthorizationRequest, content: ContentFacts) -> str | None:
        await asyncio.sleep(0)
        return REASON if content.detected else None

    _registry(monkeypatch, ContentGate(decide))
    upstream = Upstream()
    app = _app(upstream)
    refused = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    allowed = await _post(app, "/v1/messages", _messages("hello"), ANTHROPIC)
    assert (refused.status_code, allowed.status_code) == (403, 200)
    assert refused.json()["error"]["message"] == REASON
    assert len(upstream.requests) == 1


async def test_a_slow_answer_times_out_and_is_cancelled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cancelled = asyncio.Event()

    async def decide(request: AuthorizationRequest, content: ContentFacts) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    _registry(monkeypatch, ContentGate(decide))
    upstream = Upstream()
    app = _app(upstream)
    app.state.proxy.authorization.timeout = 0.05
    caplog.set_level(logging.WARNING, logger="llm_redact")
    response = await _post(app, "/v1/messages", _messages("hi"), ANTHROPIC)
    assert response.status_code == 403
    assert response.json()["error"]["message"] == AUTHORIZATION_FAULT
    await asyncio.wait_for(cancelled.wait(), 5)
    assert "timed out" in caplog.text and upstream.requests == []
    assert app.state.proxy.bookkeeping_errors[AUTHORIZATION_STAGE] == 1


async def test_a_member_that_cannot_be_called_refuses_every_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = FakeGate()
    gate.authorize_content = "allow everything"  # type: ignore[attr-defined]
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream)
    response = await _post(app, "/v1/messages", _messages("hi"), ANTHROPIC)
    assert response.status_code == 403
    assert response.json()["error"]["message"] == AUTHORIZATION_FAULT
    assert upstream.requests == []


async def test_an_earlier_refusal_is_never_put_to_the_content_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = ContentGate()
    _registry(monkeypatch, gate)
    app = _app(Upstream(), detection=DetectionConfig(modes=(("email", "block"),)))
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    assert response.status_code == 400  # the block, not the gate
    assert gate.asked == []


# --- overrides ----------------------------------------------------------------------------


def _chat(text: str) -> dict[str, Any]:
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": text}]}


async def test_a_refusal_hands_a_one_time_override_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from test_overrides import _code

    refuse = [True]
    # The local operator (no subject): its refusals carry the CLI's code.
    gate = ContentGate(
        lambda request, content: REASON if refuse[0] and content.overridden else None,
        subject=None,
    )
    _registry(monkeypatch, gate)
    upstream = Upstream()
    store_path = tmp_path / "overrides.db"
    app = _app(
        upstream,
        detection=DetectionConfig(modes=(("email", "block"),)),
        overrides=OverridesConfig(enabled=True, path=str(store_path)),
    )
    chat = _chat(f"mail {EMAIL}")
    async with _client(app) as client:
        blocked = await client.post("/v1/chat/completions", json=chat, headers=KEY)
        assert blocked.status_code == 400
        OverrideStore(store_path).approve("once", approver=None, code=_code(blocked))
        refused = await client.post("/v1/chat/completions", json=chat, headers=KEY)
        assert refused.status_code == 403 and upstream.requests == []
        # Handed back, unused: the next request may still use it.
        assert [e.state for e in OverrideStore(store_path).entries()] == ["once"]
        refuse[0] = False
        passed = await client.post("/v1/chat/completions", json=chat, headers=KEY)
        assert passed.status_code == 200 and EMAIL in upstream.sent()
        recent = (await client.get("/__llm-redact/recent")).json()["entries"]
    assert [content.overridden for _request, content in gate.asked] == [True, True]
    assert app.state.proxy.overrides.used == {"once": 1}
    # Newest first: the refused request's row says no override.
    assert [row["override"] for row in recent] == ["once", None, None]


# --- absent member, routing -----------------------------------------------------------------


async def test_without_the_member_nothing_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    _registry(monkeypatch, FakeGate())
    monkeypatch.setattr(GateAuthorization, "content_refusal", _never)
    upstream = Upstream()
    app = _app(upstream)
    assert app.state.proxy.authorization.checks_content is False
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    assert response.status_code == 200 and "«EMAIL_001»" in upstream.sent()
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["access"]["authorizes_content"] is False


async def test_a_routed_request_is_refused_before_its_plan_begins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = FakeRouter({"r": [Hop("op", "http://op.example/v1/messages"), Stop()]})
    reg, _calls = install(monkeypatch, router)
    gate = ContentGate(lambda request, content: REASON if content.detected else None)
    _registry(monkeypatch, gate, reg)
    upstream = Upstream()
    app = _routed_app(upstream)
    headers = {**ANTHROPIC, ROUTE_HEADER: "r"}
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), headers)
    assert response.status_code == 403
    (plan,) = router.plans
    assert plan.begun == []  # never begun: no budget spent, no hop
    assert upstream.requests == []
    refused_once(app.state.proxy, "authorization", "anthropic")


def _routed_app(upstream: Upstream) -> Any:
    from llm_redact.proxy import create_app

    return create_app(routed_config(), upstream_transport=httpx.MockTransport(upstream))


async def test_status_reports_the_member(monkeypatch: pytest.MonkeyPatch) -> None:
    _registry(monkeypatch, ContentGate())
    app = _app(Upstream())
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["access"] == {
        "authorizes_requests": False,
        "detection_overlays": False,
        "authorizes_content": True,
        "reloads_policy": False,
    }
