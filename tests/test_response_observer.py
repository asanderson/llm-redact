"""The optional ``SessionRouter.response_observer`` seam: a plugin observes
upstream answers, read-only, as the PROVIDER sent them.

Pinned here, keyless (scripted routers on a bare Registry stand in for
llm-redact-pro):

- what the observer is told (``ResponseContext``: adapter, method, path,
  status, content type, the request body as sent upstream, the session,
  whether a proxy-held credential was spent) — on the unrouted and the
  routed path, under the client's key and the proxy's identity;
- what it sees: its own parse of each JSON value of the answer before
  rehydration (placeholders, never a restored value) — a buffered JSON
  body once, an SSE stream per event, an NDJSON stream per line, an AWS
  eventstream per frame — and that nothing it does changes what the
  client receives;
- faults are contained (bookkeeping stage ``response_observer``, logged by
  exception type only, the answer delivered unchanged, observed no
  further);
- a router without the member costs one attribute test per answer: the
  observation code never runs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

import llm_redact.proxy as proxy_mod
from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from license_fixtures import resolved
from llm_redact.config import Config, ProviderConfig
from llm_redact.eventstream import EventStreamMessage, string_header
from llm_redact.eventstream import serialize as serialize_eventstream
from llm_redact.jsonwalk import MAX_JSON_DEPTH
from llm_redact.plugin_api import ResponseContext
from llm_redact.proxy import ProxyState, create_app
from llm_redact.registry import Registry
from test_object_access_seams import FakeAuth
from test_object_access_seams import _registry as install_session_router

EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"  # the first email the static session sees
OPENAI = "https://upstream.test"
AZURE = "https://res.openai.azure.com"
BEDROCK = "https://bedrock-runtime.us-east-1.amazonaws.com"
OLLAMA = "http://127.0.0.1:11434"


class ObservingRouter:
    """A static-mode router with the optional member, scripted."""

    mode = "static"

    def __init__(
        self,
        *,
        decline: bool = False,
        factory_error: Exception | None = None,
        observe_error: Exception | None = None,
        mutate: bool = False,
    ) -> None:
        self.decline = decline
        self.factory_error = factory_error
        self.observe_error = observe_error
        self.mutate = mutate
        self.contexts: list[ResponseContext] = []
        self.seen: list[Any] = []

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        raise AssertionError("never resolved in static mode")

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def response_observer(self, context: ResponseContext) -> Any:
        self.contexts.append(context)
        if self.factory_error is not None:
            raise self.factory_error
        if self.decline:
            return None
        return self._observe

    def _observe(self, payload: Any) -> None:
        self.seen.append(json.loads(json.dumps(payload)))  # a snapshot
        if self.mutate and isinstance(payload, dict):
            payload.clear()
            payload["tampered"] = True
        if self.observe_error is not None:
            raise self.observe_error


class Upstream:
    def __init__(self, content: bytes, content_type: str, status: int = 200) -> None:
        self.content = content
        self.content_type = content_type
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self.status, content=self.content, headers={"content-type": self.content_type}
        )


def _providers() -> dict[str, ProviderConfig]:
    return {
        **Config().providers,
        "openai": ProviderConfig(OPENAI),
        "bedrock": ProviderConfig(BEDROCK),
        "ollama": ProviderConfig(OLLAMA),
    }


def _app(router: Any, upstream: Upstream, monkeypatch: pytest.MonkeyPatch, **config: Any) -> Any:
    install_session_router(monkeypatch, router)
    return create_app(
        Config(providers=_providers(), **config), upstream_transport=httpx.MockTransport(upstream)
    )


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _chat(text: str, **extra: Any) -> dict[str, Any]:
    return {"model": "m", "messages": [{"role": "user", "content": text}], **extra}


def _completion(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
    }


def _sse(*payloads: Any) -> bytes:
    events = [f"data: {json.dumps(payload)}\n\n" for payload in payloads]
    return ("".join(events) + "data: [DONE]\n\n").encode()


def _chunk(text: str, finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
    }


def _frame(event_type: str, payload: Any) -> bytes:
    return serialize_eventstream(
        EventStreamMessage(
            headers=[
                string_header(":message-type", "event"),
                string_header(":event-type", event_type),
                string_header(":content-type", "application/json"),
            ],
            payload=json.dumps(payload).encode(),
        )
    )


# --- what is observed -----------------------------------------------------------------


async def test_a_buffered_answer_is_observed_as_the_provider_sent_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ObservingRouter(mutate=True)
    answer = _completion(f"wrote to {TOKEN}")
    upstream = Upstream(json.dumps(answer).encode(), "application/json")
    app = _app(router, upstream, monkeypatch)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"))
    assert response.status_code == 200
    # The client's answer is restored — and untouched by what the observer
    # did to its own copy.
    assert response.json()["choices"][0]["message"]["content"] == f"wrote to {EMAIL}"
    (context,) = router.contexts
    assert context.adapter_name == "openai"
    assert (context.method, context.path, context.status) == ("POST", "/v1/chat/completions", 200)
    assert context.content_type == "application/json"
    assert context.session_id == "default" and context.identity is False
    # The request body AS SENT UPSTREAM: redacted, never the client's value.
    assert context.request_body["messages"][-1]["content"] == f"mail {TOKEN}"
    assert EMAIL not in json.dumps(context.request_body)
    # The answer as the provider sent it: placeholders only, observed once.
    assert router.seen == [answer]


async def test_an_sse_stream_is_observed_event_by_event(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ObservingRouter(mutate=True)
    chunks = [_chunk("to «EMA"), _chunk("IL_001»"), _chunk("", finish="stop")]
    upstream = Upstream(_sse(*chunks) + b": keepalive\n\n", "text/event-stream")
    app = _app(router, upstream, monkeypatch)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions", json=_chat(f"mail {EMAIL}", stream=True)
        )
    assert response.status_code == 200
    assert EMAIL in response.text and "tampered" not in response.text
    (context,) = router.contexts
    assert context.content_type == "text/event-stream"
    # Every JSON event, split tokens and all, exactly as sent; the [DONE]
    # sentinel and the comment are no JSON and are skipped.
    assert router.seen == chunks


async def test_an_ndjson_stream_is_observed_line_by_line(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ObservingRouter()
    lines = [
        {"model": "m", "message": {"role": "assistant", "content": f"to {TOKEN}"}},
        {"model": "m", "message": {"role": "assistant", "content": ""}, "done": True},
    ]
    stream = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
    # The last line without its newline (the parser's tail), and one that
    # is not JSON (forwarded as it came, skipped here).
    upstream = Upstream(b"not json\n" + stream.rstrip(b"\n"), "application/x-ndjson")
    app = _app(router, upstream, monkeypatch)
    async with _client(app) as client:
        response = await client.post("/api/chat", json=_chat(f"mail {EMAIL}"))
    assert response.status_code == 200 and EMAIL in response.text
    (context,) = router.contexts
    assert context.adapter_name == "ollama" and context.content_type == "application/x-ndjson"
    assert router.seen == lines


async def test_an_eventstream_is_observed_frame_by_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ObservingRouter()
    payloads = [
        {"contentBlockIndex": 0, "delta": {"text": f"to {TOKEN}"}},
        {"contentBlockIndex": 0},
    ]
    stream = _frame("contentBlockDelta", payloads[0]) + _frame("contentBlockStop", payloads[1])
    upstream = Upstream(stream, "application/vnd.amazon.eventstream")
    app = _app(router, upstream, monkeypatch)
    body = {"messages": [{"role": "user", "content": [{"text": f"mail {EMAIL}"}]}]}
    async with _client(app) as client:
        response = await client.post(
            "/model/m/converse-stream", json=body, headers={"authorization": "Bearer k"}
        )
    assert response.status_code == 200 and EMAIL.encode() in response.content
    (context,) = router.contexts
    assert context.adapter_name == "bedrock"
    assert router.seen == payloads


async def test_a_pass_through_answer_and_an_error_status_are_observed_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ObservingRouter()
    error = {"error": {"message": "nope"}}
    upstream = Upstream(json.dumps(error).encode(), "application/json", status=404)
    app = _app(router, upstream, monkeypatch)
    async with _client(app) as client:
        response = await client.get(
            "/v1/assistants/asst_1", headers={"authorization": "Bearer sk-proj-FAKE"}
        )
    assert response.status_code == 404
    (context,) = router.contexts
    assert context.adapter_name is None and context.status == 404
    assert context.request_body is None
    assert router.seen == [error]


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        (b"plain text", "text/plain"),  # not JSON by type: never parsed
        (b"{not json", "application/json"),  # JSON by type, unparseable
        (b"[" * (MAX_JSON_DEPTH + 1) + b"]" * (MAX_JSON_DEPTH + 1), "application/json"),
        (b"", "application/json"),  # no body
    ],
)
async def test_what_is_no_json_value_is_never_observed(
    content: bytes, content_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = ObservingRouter()
    app = _app(router, Upstream(content, content_type), monkeypatch)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat("hi"))
    assert response.status_code == 200 and response.content == content
    assert len(router.contexts) == 1 and router.seen == []
    assert app.state.proxy.bookkeeping_errors == {}  # skipped, not a fault


async def test_a_declining_router_observes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ObservingRouter(decline=True)
    upstream = Upstream(_sse(_chunk(TOKEN, finish="stop")), "text/event-stream")
    app = _app(router, upstream, monkeypatch)
    monkeypatch.setattr(proxy_mod, "loads_bounded", _never_parsed)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat("hi", stream=True))
    assert response.status_code == 200
    assert len(router.contexts) == 1 and router.seen == []


def _never_parsed(data: Any) -> Any:
    raise AssertionError("parsed for an observer that declined")


# --- whose credential, which body: routed and identity paths ----------------------------


@pytest.mark.parametrize(
    ("plan_kwargs", "identity"),
    [
        pytest.param({"proxy_credential": True}, True, id="proxy-credential"),
        pytest.param({"proxy_credential": False}, False, id="client-credential"),
    ],
)
async def test_a_routed_answer_is_observed_with_the_plan_credential(
    plan_kwargs: dict[str, Any], identity: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = {"type": "message", "content": [{"type": "text", "text": f"hi {TOKEN}"}]}
    fake = FakeRouter(
        {"r": [Hop("x", "http://x.example/v1/messages"), Stop()]}, plan_kwargs={"r": plan_kwargs}
    )
    reg, _ = install(monkeypatch, fake)
    router = ObservingRouter()
    install_session_router(monkeypatch, router, reg)
    upstream = Upstream(json.dumps(answer).encode(), "application/json")
    app = create_app(routed_config(), upstream_transport=httpx.MockTransport(upstream))
    body = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": EMAIL}]}
    async with _client(app) as client:
        response = await client.post("/v1/messages", json=body, headers={ROUTE_HEADER: "r"})
    assert response.status_code == 200 and EMAIL in response.text
    (context,) = router.contexts
    assert context.adapter_name == "anthropic" and context.identity is identity
    assert context.request_body["messages"][0]["content"] == TOKEN
    # The route's own delivery hook marks the CLIENT's payload; the
    # observer's copy is the provider's.
    assert router.seen == [answer]


async def test_an_identity_authorized_answer_reports_the_proxy_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = FakeAuth()
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("pro")
    reg.build_access_gate = lambda cfg, lic: None
    reg.build_upstream_auth = lambda name, provider: auth if provider.auth == "identity" else None
    router = ObservingRouter()
    install_session_router(monkeypatch, router, reg)
    upstream = Upstream(json.dumps(_completion("ok")).encode(), "application/json")
    providers = {**_providers(), "azure": ProviderConfig(AZURE, auth="identity")}
    app = create_app(Config(providers=providers), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post("/openai/v1/chat/completions", json=_chat("hi"))
    assert response.status_code == 200 and auth.calls == 1
    (context,) = router.contexts
    assert context.adapter_name == "azure" and context.identity is True


# --- faults are contained -------------------------------------------------------------


async def test_a_failing_factory_is_contained(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    router = ObservingRouter(factory_error=LookupError("resp-secret-id"))
    upstream = Upstream(json.dumps(_completion(TOKEN)).encode(), "application/json")
    app = _app(router, upstream, monkeypatch)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with _client(app) as client:
            response = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"))
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == EMAIL
    assert app.state.proxy.bookkeeping_errors == {"response_observer": 1}
    assert "LookupError" in caplog.text and "resp-secret-id" not in caplog.text
    (row,) = app.state.proxy.recent
    assert row["status"] == 200


@pytest.mark.parametrize("stream", [True, False])
async def test_a_failing_observer_is_contained_and_observes_no_further(
    stream: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    router = ObservingRouter(observe_error=ValueError("cmp-secret"))
    if stream:
        upstream = Upstream(
            _sse(_chunk("to "), _chunk(TOKEN), _chunk("", finish="stop")), "text/event-stream"
        )
    else:
        upstream = Upstream(json.dumps(_completion(TOKEN)).encode(), "application/json")
    app = _app(router, upstream, monkeypatch)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with _client(app) as client:
            response = await client.post(
                "/v1/chat/completions", json=_chat(f"mail {EMAIL}", stream=stream)
            )
    assert response.status_code == 200 and EMAIL in response.text  # the answer, whole
    assert len(router.seen) == 1  # dropped after its first fault
    assert app.state.proxy.bookkeeping_errors == {"response_observer": 1}
    assert "ValueError" in caplog.text and "cmp-secret" not in caplog.text


# --- zero cost without the member ---------------------------------------------------


class OlderRouter:
    """A router from before the member."""

    mode = "static"

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        raise AssertionError("never resolved in static mode")

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None


@pytest.mark.parametrize("router", [None, OlderRouter()], ids=["free-default", "older-router"])
async def test_without_the_member_the_observation_code_never_runs(
    router: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("observation code ran without an observer")

    monkeypatch.setattr(ProxyState, "response_observer", never)
    monkeypatch.setattr(proxy_mod, "_observe", never)
    answers = [
        ("/v1/chat/completions", _chat(EMAIL), json.dumps(_completion(TOKEN)), "application/json"),
        (
            "/v1/chat/completions",
            _chat(EMAIL, stream=True),
            _sse(_chunk(TOKEN, finish="stop")).decode(),
            "text/event-stream",
        ),
        (
            "/api/chat",
            _chat(EMAIL),
            json.dumps({"message": {"content": TOKEN}, "done": True}) + "\n",
            "application/x-ndjson",
        ),
    ]
    for path, body, content, content_type in answers:
        upstream = Upstream(content.encode(), content_type)
        if router is None:
            app = create_app(
                Config(providers=_providers()), upstream_transport=httpx.MockTransport(upstream)
            )
        else:
            app = _app(router, upstream, monkeypatch)
        assert app.state.proxy.observes_responses is False
        async with _client(app) as client:
            response = await client.post(path, json=body)
        assert response.status_code == 200 and EMAIL in response.text
