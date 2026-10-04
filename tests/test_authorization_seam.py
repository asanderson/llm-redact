"""The access gate's authorization seams (``plugin_api.AccessGate``'s optional
``authorize_request`` and ``detection_overlay``), over HTTP.

The core holds no role logic: it hands a gate that declares
``authorize_request`` the FACTS of every forwarded request (matched or
pass-through, under the client's key or a credential the proxy holds) after
everything else that may refuse it, and BEFORE the session, redaction, the
``[audit] required`` START row, the upstream authorizer and any upstream
contact; and it applies a gate's ``detection_overlay`` — tighten-only — to
everything that request's redaction does. Both fail closed (403, kind
``authorization``, bookkeeping stage ``authorization``), never logging a
reason, a deny string or a user. Keyless: scripted fakes on a bare Registry
stand in for llm-redact-pro.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from license_fixtures import resolved
from llm_redact import authorization as authorization_mod
from llm_redact.authorization import (
    AUTHORIZATION_FAULT,
    AUTHORIZATION_STAGE,
    OVERLAY_CACHE_SIZE,
    OVERLAY_FAULT,
    GateAuthorization,
    OverlayBuilds,
    OverlayError,
)
from llm_redact.config import AuditConfig, Config, ProviderConfig
from llm_redact.detection.engine import DetectionConfig, build_modes
from llm_redact.plugin_api import Admission, AuthorizationRequest, DetectionOverlay
from llm_redact.proxy import create_app
from llm_redact.registry import Registry
from local_refusals import refused_once
from test_access_seam import FakeGate
from test_audit_required import FakeAudit
from test_upload_inspection import FakeInspector, reads
from test_upstream_auth import AZURE, AZURE_PATH, _identity

EMAIL = "jane.doe@corp.example"
DENIED = "Project Zebra"
REASON = "llm-redact: your role does not grant this model"
KEY = {"authorization": "Bearer sk-client"}
ANTHROPIC = {"x-api-key": "sk-ant-client", "anthropic-version": "2023-06-01"}
UPSTREAM = "https://upstream.test"
# What the gate below sets for the request it admits (a stand-in for
# llm-redact-pro's per-request user).
_ADMITTED: ContextVar[str | None] = ContextVar("test_authz_admitted", default=None)


def _messages(text: str, model: str = "claude-sonnet-4-5") -> dict[str, Any]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": text}]}


class AuthorizingGate(FakeGate):
    """A gate declaring only ``authorize_request``: records every request
    (with what ``admit`` set for it) and answers ``decide(request)``."""

    def __init__(self, decide: Callable[[AuthorizationRequest], Any] | None = None) -> None:
        super().__init__()
        self.decide = decide or (lambda request: None)
        self.requests: list[AuthorizationRequest] = []
        self.users: list[str | None] = []

    def admit(self, conn: Any, surface: str) -> Admission:
        _ADMITTED.set("ada")
        return Admission(subject="ada")

    def authorize_request(self, request: AuthorizationRequest) -> Any:
        self.requests.append(request)
        self.users.append(_ADMITTED.get())
        return self.decide(request)


class OverlayGate(FakeGate):
    """A gate declaring only ``detection_overlay``."""

    def __init__(self, overlay: Callable[[], Any] | Any = None) -> None:
        super().__init__()
        self.overlay = overlay
        self.asked = 0

    def detection_overlay(self) -> Any:
        self.asked += 1
        return self.overlay() if callable(self.overlay) else self.overlay


class BothGate(AuthorizingGate, OverlayGate):
    def __init__(self, decide: Any = None, overlay: Any = None) -> None:
        AuthorizingGate.__init__(self, decide)
        self.overlay = overlay
        self.asked = 0


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200,
            json={"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            headers={"content-type": "application/json"},
        )

    def sent(self) -> str:
        (request,) = self.requests
        return request.content.decode()


def _registry(monkeypatch: pytest.MonkeyPatch, gate: Any, reg: Registry | None = None) -> Registry:
    reg = reg or Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("team")
    reg.build_access_gate = lambda config, license: gate
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return reg


def _app(upstream: Upstream, **config: Any) -> Any:
    providers = {
        **Config().providers,
        "anthropic": ProviderConfig(UPSTREAM),
        "openai": ProviderConfig(UPSTREAM),
    }
    providers.update(config.pop("providers", {}))
    return create_app(
        Config(providers=providers, **config), upstream_transport=httpx.MockTransport(upstream)
    )


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


async def _post(app: Any, path: str, body: Any, headers: dict[str, str]) -> httpx.Response:
    async with _client(app) as client:
        return await client.post(path, json=body, headers=headers)


def _never(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("reached after a refusal")


# --- the request facts, on every kind of route ---------------------------------------


async def test_a_matched_chat_route_is_asked_with_its_facts_and_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = AuthorizingGate()
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream)
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    assert response.status_code == 200
    assert gate.requests == [
        AuthorizationRequest(
            surface="http",
            provider="anthropic",
            adapter="anthropic",
            kind="chat",
            method="POST",
            path="/v1/messages",
            model="claude-sonnet-4-5",
            identity=False,
        )
    ]
    # In the request's own context: what admit set is visible.
    assert gate.users == ["ada"]
    assert EMAIL not in upstream.sent() and "«EMAIL_001»" in upstream.sent()
    # Never shown: the dataclass's repr leaves the path out.
    assert "/v1/messages" not in repr(gate.requests[0])


async def test_a_redact_only_route_reports_its_kind_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = AuthorizingGate()
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream)
    body = {"model": "text-embedding-3-small", "input": f"mail {EMAIL}"}
    assert (await _post(app, "/v1/embeddings", body, KEY)).status_code == 200
    (request,) = gate.requests
    assert (request.provider, request.adapter, request.kind, request.model) == (
        "openai",
        "openai",
        "redact_only",
        "text-embedding-3-small",
    )


async def test_a_pass_through_route_is_asked_with_no_adapter_and_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = AuthorizingGate(lambda request: REASON if request.adapter is None else None)
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream)
    body = {"model": "gpt-4o", "content": "hi"}
    response = await _post(app, "/v1/threads/thread_abc/messages", body, KEY)
    assert response.status_code == 403
    assert response.json() == {"error": REASON}
    (request,) = gate.requests
    assert (request.adapter, request.kind, request.model, request.provider) == (
        None,
        "none",
        None,  # a pass-through body is never read
        "openai",
    )
    assert upstream.requests == []
    refused_once(app.state.proxy, "authorization", "openai")


def _upload(text: bytes, filename: str = "notes.txt", content_type: str = "text/plain") -> bytes:
    return (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="'
        + filename.encode()
        + b'"\r\nContent-Type: '
        + content_type.encode()
        + b"\r\n\r\n"
        + text
        + b"\r\n--b--\r\n"
    )


FORM = {**KEY, "content-type": "multipart/form-data; boundary=b"}


async def test_an_upload_is_asked_and_refused_before_anything_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = AuthorizingGate(lambda request: REASON)
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream)
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=_upload(f"mail {EMAIL}".encode()), headers=FORM
        )
    assert response.status_code == 403 and REASON in response.text
    (request,) = gate.requests
    assert (request.adapter, request.kind, request.model) == ("openai", "chat", None)
    assert upstream.requests == []
    assert len(app.state.proxy.vault) == 0
    refused_once(app.state.proxy, "authorization", "openai")


async def test_identity_requests_say_so_and_a_refusal_never_reaches_the_authorizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_upstream_auth import _install as install_auth

    reg, built = install_auth(monkeypatch)
    # The deployment the path names is the model it runs (the body's is not).
    gate = AuthorizingGate(lambda request: REASON if request.model == "refuse-me" else None)
    _registry(monkeypatch, gate, reg)
    upstream = Upstream()
    app = _app(upstream, providers={"azure": _identity(AZURE)})
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]}
    query = "?api-version=2024-10-21"
    refused_path = AZURE_PATH.replace("/gpt/", "/refuse-me/") + query
    refused = await _post(app, refused_path, body, {"api-key": "client-key"})
    assert refused.status_code == 403
    assert built[0].calls == [] and upstream.requests == []
    allowed = await _post(app, AZURE_PATH + query, body, {"api-key": "client-key"})
    assert allowed.status_code == 200
    assert [r.identity for r in gate.requests] == [True, True]
    assert [r.adapter for r in gate.requests] == ["azure", "azure"]
    assert [r.model for r in gate.requests] == ["refuse-me", "gpt"]
    assert len(built[0].calls) == 1


async def test_a_routed_plan_lending_an_operator_key_is_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = FakeRouter(
        {"r": [Hop("op", "http://op.example/v1/messages"), Stop()]},
        plan_kwargs={"r": {"proxy_credential": True}},
    )
    reg, _calls = install(monkeypatch, router)
    gate = AuthorizingGate(lambda request: REASON)
    _registry(monkeypatch, gate, reg)
    upstream = Upstream()
    app = create_app(routed_config(), upstream_transport=httpx.MockTransport(upstream))
    headers = {**ANTHROPIC, ROUTE_HEADER: "r"}
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), headers)
    assert response.status_code == 403
    (request,) = gate.requests
    assert request.identity is True
    (plan,) = router.plans
    assert plan.begun == []  # never begun: no budget spent, no hop
    assert upstream.requests == []


# --- ordering: nothing happens before the verdict --------------------------------------


async def test_a_refusal_precedes_the_session_redaction_audit_and_upstream(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    audit = FakeAudit()
    gate = AuthorizingGate(lambda request: REASON)
    reg = Registry()
    reg.build_audit = lambda cfg: audit if cfg.enabled else None
    _registry(monkeypatch, gate, reg)
    upstream = Upstream()
    app = _app(upstream, audit=AuditConfig(enabled=True, required=True))
    state = app.state.proxy
    monkeypatch.setattr(state, "context_for", _never)
    caplog.set_level(logging.INFO, logger="llm_redact")
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    assert response.status_code == 403
    # Provider-shaped: the Anthropic error envelope, the gate's reason.
    assert response.json()["type"] == "error" and response.json()["error"]["message"] == REASON
    assert audit.begun == [] and upstream.requests == []
    assert len(state.vault) == 0
    # Recorded (the END-only row of a local refusal), the reason never logged.
    (row,) = state.recent
    assert (row["status"], row["user"]) == (403, "ada")
    assert REASON not in caplog.text and EMAIL not in caplog.text
    assert "403 refused by the access gate (authorization)" in caplog.text
    refused_once(state, "authorization", "anthropic")


async def test_earlier_refusals_win_and_are_never_put_to_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = AuthorizingGate()
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream)
    async with _client(app) as client:
        # The scanned-body rule: not a JSON object.
        response = await client.post(
            "/v1/messages", content=b"[1]", headers={**ANTHROPIC, "content-type": "text/plain"}
        )
    assert response.status_code == 400
    assert gate.requests == []


# --- the gate's answers -----------------------------------------------------------------


async def _sleeps(request: AuthorizationRequest) -> None:
    await asyncio.sleep(3600)


@pytest.mark.parametrize(
    ("decide", "logged"),
    [
        (lambda request: (_ for _ in ()).throw(RuntimeError(f"no {EMAIL}")), "RuntimeError"),
        (lambda request: True, "answered bool"),
        (lambda request: "", "answered str"),
        (lambda request: 0, "answered int"),
        (lambda request: [REASON], "answered list"),
    ],
    ids=["raises", "true", "empty", "zero", "list"],
)
async def test_a_failing_or_nonsense_answer_refuses_with_the_core_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    decide: Any,
    logged: str,
) -> None:
    _registry(monkeypatch, AuthorizingGate(decide))
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


async def test_an_awaitable_answer_is_awaited_in_the_requests_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str | None] = []

    async def decide(request: AuthorizationRequest) -> str | None:
        await asyncio.sleep(0)
        seen.append(_ADMITTED.get())
        return REASON if request.model == "claude-opus" else None

    _registry(monkeypatch, AuthorizingGate(decide))
    upstream = Upstream()
    app = _app(upstream)
    allowed = await _post(app, "/v1/messages", _messages("hi"), ANTHROPIC)
    refused = await _post(app, "/v1/messages", _messages("hi", "claude-opus"), ANTHROPIC)
    assert (allowed.status_code, refused.status_code) == (200, 403)
    assert refused.json()["error"]["message"] == REASON
    assert seen == ["ada", "ada"]
    assert len(upstream.requests) == 1


@pytest.mark.parametrize("fault", ["raises", "cancels", "nonsense"])
async def test_an_awaitable_that_fails_refuses(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fault: str
) -> None:
    async def decide(request: AuthorizationRequest) -> Any:
        await asyncio.sleep(0)
        if fault == "raises":
            raise LookupError("directory down")
        if fault == "cancels":
            raise asyncio.CancelledError
        return 42

    _registry(monkeypatch, AuthorizingGate(decide))
    upstream = Upstream()
    app = _app(upstream)
    caplog.set_level(logging.WARNING, logger="llm_redact")
    response = await _post(app, "/v1/messages", _messages("hi"), ANTHROPIC)
    assert response.status_code == 403
    assert response.json()["error"]["message"] == AUTHORIZATION_FAULT
    assert app.state.proxy.bookkeeping_errors[AUTHORIZATION_STAGE] == 1
    expected = {"raises": "LookupError", "cancels": "cancelled", "nonsense": "answered int"}
    assert expected[fault] in caplog.text and "directory down" not in caplog.text


async def test_a_slow_awaitable_times_out_and_is_cancelled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cancelled = asyncio.Event()

    async def decide(request: AuthorizationRequest) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    _registry(monkeypatch, AuthorizingGate(decide))
    upstream = Upstream()
    app = _app(upstream)
    app.state.proxy.authorization.timeout = 0.05
    caplog.set_level(logging.WARNING, logger="llm_redact")
    response = await _post(app, "/v1/messages", _messages("hi"), ANTHROPIC)
    assert response.status_code == 403
    assert response.json()["error"]["message"] == AUTHORIZATION_FAULT
    await asyncio.wait_for(cancelled.wait(), 5)
    assert "timed out" in caplog.text
    assert upstream.requests == []


async def test_a_task_that_ignores_cancellation_is_abandoned_not_awaited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished = asyncio.Event()

    async def stubborn(request: AuthorizationRequest) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)  # keeps running after the refusal
            finished.set()
            raise RuntimeError("late") from None

    _registry(monkeypatch, AuthorizingGate(stubborn))
    upstream = Upstream()
    app = _app(upstream)
    app.state.proxy.authorization.timeout = 0.05
    response = await _post(app, "/v1/messages", _messages("hi"), ANTHROPIC)
    assert response.status_code == 403
    assert not finished.is_set()  # answered without waiting for it
    await asyncio.wait_for(finished.wait(), 5)
    await asyncio.sleep(0)  # its late exception is retrieved, never unraised


async def test_without_the_members_nothing_is_asked_and_no_await_is_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _registry(monkeypatch, FakeGate())
    monkeypatch.setattr(GateAuthorization, "refusal", _never)
    monkeypatch.setattr(GateAuthorization, "overlay", _never)
    monkeypatch.setattr(authorization_mod.asyncio, "ensure_future", _never)
    upstream = Upstream()
    app = _app(upstream)
    state = app.state.proxy
    assert (state.authorization.authorizes, state.authorization.overlays) == (False, False)
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    assert response.status_code == 200
    # The shared static context: no overlay copy.
    assert state.context_for(None, "POST", "/v1/messages", None) is state._static_context
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["access"] == {"authorizes_requests": False, "detection_overlays": False}


async def test_a_synchronous_answer_costs_no_await(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = AuthorizingGate()
    _registry(monkeypatch, gate)
    monkeypatch.setattr(authorization_mod.asyncio, "ensure_future", _never)
    app = _app(Upstream())
    assert (await _post(app, "/v1/messages", _messages("hi"), ANTHROPIC)).status_code == 200
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["access"] == {"authorizes_requests": True, "detection_overlays": False}


async def test_no_gate_at_all_reports_neither_seam() -> None:
    app = _app(Upstream())
    assert app.state.proxy.authorization.authorizes is False
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["access"] == {"authorizes_requests": False, "detection_overlays": False}


# --- the detection overlay ----------------------------------------------------------------


def _overlay_app(
    monkeypatch: pytest.MonkeyPatch,
    overlay: Any,
    detection: DetectionConfig | None = None,
    **config: Any,
) -> tuple[Any, Upstream, OverlayGate]:
    gate = OverlayGate(overlay)
    _registry(monkeypatch, gate)
    upstream = Upstream()
    app = _app(upstream, detection=detection or DetectionConfig(), **config)
    return app, upstream, gate


async def test_warn_is_tightened_to_redact(monkeypatch: pytest.MonkeyPatch) -> None:
    app, upstream, gate = _overlay_app(
        monkeypatch,
        DetectionOverlay(modes=(("email", "redact"),)),
        DetectionConfig(modes=(("email", "warn"),)),
    )
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    assert response.status_code == 200
    assert EMAIL not in upstream.sent() and "«EMAIL_001»" in upstream.sent()
    assert gate.asked == 1
    state = app.state.proxy
    assert state.warn_counts["EMAIL"] == 0 and state.detection_counts["EMAIL"] == 1


async def test_redact_is_tightened_to_block(monkeypatch: pytest.MonkeyPatch) -> None:
    app, upstream, _gate = _overlay_app(monkeypatch, DetectionOverlay(modes=(("email", "block"),)))
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    assert response.status_code == 400 and "EMAIL" in response.text
    assert upstream.requests == []
    refused_once(app.state.proxy, "blocked_value", "anthropic")


# An email the configured policy redacts, holding the overlay's deny string.
OVERLAPPED = "bob.private@acme-corp.example"


async def test_extra_deny_strings_only_tighten_the_configured_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Redacted wherever they occur — and where one overlaps a value the
    # configured policy redacts, that value keeps its own token over their
    # union: never a placeholder cut into it, the rest sent as it was.
    app, upstream, _gate = _overlay_app(
        monkeypatch, DetectionOverlay(deny=(DENIED.lower(), DENIED.lower(), "acme"))
    )
    text = f"about {DENIED}, mail {OVERLAPPED} or {EMAIL}, ask acme"
    response = await _post(app, "/v1/messages", _messages(text), ANTHROPIC)
    assert response.status_code == 200
    sent = upstream.sent()
    assert "about «DENY_001», mail «EMAIL_001» or «EMAIL_002», ask «DENY_002»" in sent
    assert DENIED not in sent and "bob.private" not in sent and "acme" not in sent
    # Only this requester's: the configured policy is untouched.
    assert app.state.proxy.redactor.redact_text(DENIED) == DENIED


async def test_an_extra_deny_string_never_lets_a_configured_block_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, upstream, _gate = _overlay_app(
        monkeypatch,
        DetectionOverlay(deny=("acme",)),
        DetectionConfig(modes=(("email", "block"),)),
    )
    response = await _post(app, "/v1/messages", _messages(f"mail {OVERLAPPED}"), ANTHROPIC)
    assert response.status_code == 400 and "EMAIL" in response.text
    assert upstream.requests == [] and len(app.state.proxy.vault) == 0
    refused_once(app.state.proxy, "blocked_value", "anthropic")


async def test_an_extra_deny_string_beats_only_a_value_forwarded_as_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # warn mode forwards the email as sent: the deny string inside it is
    # redacted, the rest goes as the configured policy sends it (honest:
    # warn mode protects nothing).
    app, upstream, _gate = _overlay_app(
        monkeypatch,
        DetectionOverlay(deny=("acme",)),
        DetectionConfig(modes=(("email", "warn"),)),
    )
    response = await _post(app, "/v1/messages", _messages(f"mail {OVERLAPPED}"), ANTHROPIC)
    assert response.status_code == 200
    assert "mail bob.private@«DENY_001»-corp.example" in upstream.sent()


async def test_an_extra_deny_string_and_a_configured_one_are_redacted_as_their_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_redact.detection.deny import DenyEntry

    app, upstream, _gate = _overlay_app(
        monkeypatch,
        DetectionOverlay(deny=("acme corp",)),
        DetectionConfig(deny_strings=(DenyEntry("corp secret plan"),)),
    )
    text = "status of acme corp secret plan today"
    response = await _post(app, "/v1/messages", _messages(text), ANTHROPIC)
    assert response.status_code == 200
    assert "status of «DENY_001» today" in upstream.sent()
    assert app.state.proxy.vault.original_for("«DENY_001»") == "acme corp secret plan"


async def test_an_overlay_applies_to_an_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    app, upstream, _gate = _overlay_app(monkeypatch, DetectionOverlay(deny=(DENIED, "acme")))
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            content=_upload(f"notes on {DENIED} for {OVERLAPPED}".encode()),
            headers=FORM,
        )
    assert response.status_code == 200
    sent = upstream.requests[0].content
    assert DENIED.encode() not in sent and b"bob.private" not in sent
    assert "notes on «DENY_001» for «EMAIL_001»".encode() in sent


async def test_an_uploaded_value_a_configured_block_refuses_stays_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, upstream, _gate = _overlay_app(
        monkeypatch,
        DetectionOverlay(deny=("acme",)),
        DetectionConfig(modes=(("email", "block"),)),
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=_upload(f"notes for {OVERLAPPED}".encode()), headers=FORM
        )
    assert response.status_code == 400 and "EMAIL" in response.text
    assert upstream.requests == []
    refused_once(app.state.proxy, "blocked_value", "openai")


async def test_a_converted_document_keeps_the_configured_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Convert mode scans the extracted text and sends the document as its
    # redacted text: an extra deny string inside the email is redacted with
    # it, never cut into it.
    from document_fixtures import pdf
    from llm_redact.config import parse_config
    from test_extraction_convert import _file_part
    from test_extraction_convert import _upload as _document

    _registry(monkeypatch, OverlayGate(DetectionOverlay(deny=("acme",))))
    upstream = Upstream()
    config = parse_config(
        {
            "extraction": {"enabled": True, "convert": True},
            "providers": {"openai": {"upstream_base_url": UPSTREAM}},
        },
        "t",
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    document = _document(pdf([f"payroll contact {OVERLAPPED}"]))
    async with _client(app) as client:
        response = await client.post("/v1/files", content=document, headers=FORM)
    assert response.status_code == 200
    _headers, sent = _file_part(upstream.requests[0])
    assert b"bob.private" not in sent and b"acme" not in sent
    assert "payroll contact «EMAIL_001»".encode() in sent


async def test_an_overlay_applies_to_the_inspected_text_of_a_binary_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = OverlayGate(DetectionOverlay(modes=(("email", "block"),)))
    reg = Registry()
    inspector = FakeInspector(reads(f"contact {EMAIL}"))
    reg.build_upload_inspector = lambda config, tier: inspector
    _registry(monkeypatch, gate, reg)
    upstream = Upstream()
    app = _app(upstream)
    pdf = b"%PDF-1.7\n1 0 obj << >> endobj\n%%EOF\n"
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            content=_upload(pdf, "report.pdf", "application/pdf"),
            headers=FORM,
        )
    # The extracted text holds an email: under the overlay it BLOCKS.
    assert response.status_code == 400 and "EMAIL" in response.text
    assert len(inspector.parts) == 1 and upstream.requests == []


@pytest.mark.parametrize(
    "overlay",
    [
        DetectionOverlay(modes=(("email", "redact"),)),  # the configured block, relaxed
        DetectionOverlay(modes=(("no_such_rule", "block"),)),
        DetectionOverlay(modes=(("email", "warn"),)),
        DetectionOverlay(modes=(("email", "drop"),)),
        DetectionOverlay(deny=("",)),
        DetectionOverlay(deny=("«EMAIL_001»",)),
        DetectionOverlay(modes=[("email", "block")]),  # type: ignore[arg-type]
        DetectionOverlay(modes=(("email",),)),  # type: ignore[arg-type]
        DetectionOverlay(deny=[DENIED]),  # type: ignore[arg-type]
        DetectionOverlay(deny=(7,)),  # type: ignore[arg-type]
        {"modes": ()},
    ],
    ids=[
        "relaxation",
        "unknown-rule",
        "warn",
        "unknown-mode",
        "empty-deny",
        "guillemet-deny",
        "list-modes",
        "short-pair",
        "list-deny",
        "non-str-deny",
        "not-an-overlay",
    ],
)
async def test_an_overlay_the_core_cannot_apply_refuses(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, overlay: Any
) -> None:
    app, upstream, _gate = _overlay_app(
        monkeypatch, overlay, DetectionConfig(modes=(("email", "block"),))
    )
    caplog.set_level(logging.INFO, logger="llm_redact")
    response = await _post(app, "/v1/messages", _messages("hello"), ANTHROPIC)
    assert response.status_code == 403
    assert response.json()["error"]["message"] == OVERLAY_FAULT
    assert upstream.requests == []
    assert app.state.proxy.bookkeeping_errors[AUTHORIZATION_STAGE] == 1
    assert "«" not in caplog.text and DENIED not in caplog.text
    refused_once(app.state.proxy, "authorization", "anthropic")


async def test_an_overlay_that_raises_or_awaits_refuses(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def later() -> DetectionOverlay:
        return DetectionOverlay()

    answers: list[Any] = []

    def overlay() -> Any:
        if not answers:
            answers.append(later())
            return answers[0]
        raise KeyError(EMAIL)

    app, upstream, _gate = _overlay_app(monkeypatch, overlay)
    caplog.set_level(logging.WARNING, logger="llm_redact")
    first = await _post(app, "/v1/messages", _messages("hello"), ANTHROPIC)
    second = await _post(app, "/v1/messages", _messages("hello"), ANTHROPIC)
    assert (first.status_code, second.status_code) == (403, 403)
    assert first.json()["error"]["message"] == OVERLAY_FAULT
    import inspect

    assert inspect.getcoroutinestate(answers[0]) == inspect.CORO_CLOSED  # closed unrun
    assert "answered an awaitable" in caplog.text and "KeyError" in caplog.text
    assert EMAIL not in caplog.text
    assert upstream.requests == []


async def test_a_rule_that_is_not_built_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    # email disabled; aadhaar scoped out by languages: both no-ops.
    detection = DetectionConfig(
        enabled=tuple(name for name in DetectionConfig().enabled if name != "email"),
        languages=("en",),
    )
    overlay = DetectionOverlay(modes=(("email", "block"), ("aadhaar", "block")))
    app, upstream, _gate = _overlay_app(monkeypatch, overlay, detection)
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    assert response.status_code == 200
    assert EMAIL in upstream.sent()
    # Nothing tightened: the shared static context serves the request.
    assert len(app.state.proxy.overlay_builds) == 1


@pytest.mark.parametrize("overlay", [None, DetectionOverlay()], ids=["none", "empty"])
async def test_no_overlay_keeps_the_configured_policy(
    monkeypatch: pytest.MonkeyPatch, overlay: Any
) -> None:
    app, upstream, gate = _overlay_app(monkeypatch, overlay)
    response = await _post(app, "/v1/messages", _messages(f"mail {EMAIL}"), ANTHROPIC)
    assert response.status_code == 200 and "«EMAIL_001»" in upstream.sent()
    assert gate.asked == 1
    assert len(app.state.proxy.overlay_builds) == 0


async def test_the_overlay_follows_the_authorization_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = BothGate(lambda request: REASON, DetectionOverlay(deny=(DENIED,)))
    _registry(monkeypatch, gate)
    app = _app(Upstream())
    response = await _post(app, "/v1/messages", _messages(DENIED), ANTHROPIC)
    assert response.status_code == 403 and gate.asked == 0  # refused first
    gate.decide = lambda request: None
    upstream = Upstream()
    app = _app(upstream)
    response = await _post(app, "/v1/messages", _messages(DENIED), ANTHROPIC)
    assert response.status_code == 200 and DENIED not in upstream.sent()
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["access"] == {"authorizes_requests": True, "detection_overlays": True}


class ConversationRouter:
    """A session router resolving every request to its own conversation
    (the non-static path of ``context_for``)."""

    mode = "per-conversation"

    def resolve(self, adapter_name: Any, method: str, path: str, body: Any) -> str:
        return "conv-1"

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None


async def test_the_overlay_reaches_a_per_conversation_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = OverlayGate(DetectionOverlay(deny=(DENIED,)))
    reg = Registry()
    reg.build_session_router = lambda config, **kw: ConversationRouter()
    _registry(monkeypatch, gate, reg)
    upstream = Upstream()
    app = _app(upstream)
    response = await _post(app, "/v1/messages", _messages(DENIED), ANTHROPIC)
    assert response.status_code == 200
    assert DENIED not in upstream.sent() and "«DENY_001»" in upstream.sent()
    assert len(app.state.proxy.vault_manager.get("conv-1")) == 1


async def test_builds_are_cached_bounded_and_dropped_by_a_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlays = [DetectionOverlay(deny=(f"secret-{n}",)) for n in range(OVERLAY_CACHE_SIZE + 6)]
    current: list[DetectionOverlay] = [overlays[0]]
    app, upstream, _gate = _overlay_app(monkeypatch, lambda: current[0])
    state = app.state.proxy
    await _post(app, "/v1/messages", _messages("secret-0"), ANTHROPIC)
    first = state.overlay_builds.build(overlays[0])
    for overlay in overlays:
        current[0] = overlay
        assert (await _post(app, "/v1/messages", _messages("x"), ANTHROPIC)).status_code == 200
    assert len(state.overlay_builds) == OVERLAY_CACHE_SIZE
    # The oldest was dropped: a fresh build replaces it.
    assert state.overlay_builds.build(overlays[0]) is not first
    assert state.overlay_builds.build(overlays[-1]) is state.overlay_builds.build(overlays[-1])
    # A reload that changes [detection] drops every build with the objects.
    builds = state.overlay_builds
    state.apply_config(
        Config(providers=state.config.providers, detection=DetectionConfig(allowlist=("x",)))
    )
    assert state.overlay_builds is not builds and len(state.overlay_builds) == 0
    unchanged = state.overlay_builds
    state.apply_config(state.config)
    assert state.overlay_builds is unchanged
    assert "secret-0" not in upstream.requests[0].content.decode()


# --- the building blocks --------------------------------------------------------------------


def _builds(config: DetectionConfig, size: int = OVERLAY_CACHE_SIZE) -> OverlayBuilds:
    return OverlayBuilds(config, build_modes(config), size=size)


def test_tightening_is_per_type_and_takes_the_strictest() -> None:
    config = DetectionConfig(modes=(("github_token", "warn"),))
    builds = _builds(config)
    # github_token and github_fine_grained_pat share GITHUB_TOKEN.
    built = builds.build(
        DetectionOverlay(
            modes=(
                ("github_fine_grained_pat", "block"),
                ("github_token", "redact"),
                ("email", "redact"),
            )
        )
    )
    assert built is not None and built.modes == {"GITHUB_TOKEN": "block"}
    assert built.deny is None  # modes only: no deny strings added
    redacted = builds.build(DetectionOverlay(modes=(("github_token", "redact"),)))
    assert redacted is not None and redacted.modes == {}
    assert builds.modes == {"GITHUB_TOKEN": "warn"}  # never mutated


def test_a_custom_rule_is_known_by_name() -> None:
    from llm_redact.detection.engine import CustomRule

    config = DetectionConfig(
        custom_rules=(CustomRule(name="ticket", pattern=r"TCK-\d+", detector_type="TICKET"),)
    )
    built = _builds(config).build(DetectionOverlay(modes=(("ticket", "block"),)))
    assert built is not None and built.modes == {"TICKET": "block"}


def test_a_relaxation_is_refused_even_for_a_rule_that_is_not_built() -> None:
    config = DetectionConfig(
        enabled=tuple(n for n in DetectionConfig().enabled if n != "email"),
        modes=(("email", "block"),),
    )
    with pytest.raises(OverlayError, match="relaxation"):
        _builds(config).build(DetectionOverlay(modes=(("email", "redact"),)))


def test_a_failed_build_is_not_kept() -> None:
    builds = _builds(DetectionConfig(), size=2)
    with pytest.raises(OverlayError, match="unknown rule"):
        builds.build(DetectionOverlay(modes=(("nope", "block"),)))
    assert len(builds) == 0
