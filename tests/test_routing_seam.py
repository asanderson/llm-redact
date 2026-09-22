"""The routing SEAM in the Free core: the policy-free driver of a routed
request and every surface around it, driven by the scripted fakes in
tests/fake_router.py — keyless, without the llm-redact-pro package.

Two claims are load-bearing and pinned here:

* the UNROUTED path is byte-identical and never consults the router
  (``test_unrouted_path_never_consults_router``, ``test_plan_none_takes_legacy_path``);
* every arm of the driver — refusals, the audit ordering, hops, faults,
  waits, re-issues, the delivery hooks on all three delivery branches,
  the mid-stream fault, status, reload/close — executes on facts the
  ``plugin_api`` contract carries, and the log lines/rows the core writes
  carry names, ids, modes and classes only.
"""

from __future__ import annotations

import json
import time
import tomllib
from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx
import pytest

import llm_redact.proxy as proxy_mod
from fake_router import (
    ROUTE_HEADER,
    FakeDelivery,
    FakeRouter,
    Hop,
    Refuse402,
    Refuse404,
    Stop,
    install,
    routed_config,
)
from llm_redact.audit import AuditRecord, AuditWriteError
from llm_redact.config import AuditConfig, Config, ProviderConfig, parse_config
from llm_redact.plugin_api import HopResult, LocalAnswer, RouteKind
from llm_redact.providers import AnthropicAdapter, OllamaAdapter
from llm_redact.proxy import (
    ProxyState,
    RequestMeta,
    _hop_timeout,
    _route_log_suffix,
    _stream_rehydrated,
    _stream_rehydrated_ndjson,
    create_app,
)

EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"  # the first email the static session sees
URL_A = "http://a.example/v1/messages"
URL_B = "http://b.example/v1/messages"
JSON_REPLY = b'{"role":"assistant","content":[{"type":"text","text":"hi"}]}'


def _messages(text: str, **extra: Any) -> dict[str, Any]:
    return {
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": text}],
        **extra,
    }


def _sse(text: str) -> bytes:
    events = [
        ("message_start", {"type": "message_start", "message": {"id": "msg_1", "model": "m"}}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(
        f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        for name, payload in events
    ).encode("utf-8")


def _ndjson(text: str) -> bytes:
    lines = [
        {"model": "m", "message": {"role": "assistant", "content": text}, "done": False},
        {"model": "m", "message": {"role": "assistant", "content": ""}, "done": True},
    ]
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode("utf-8")


class _Upstream(httpx.AsyncBaseTransport):
    """Records every request; answers by URL from ``replies`` (bytes or a
    callable building the response), JSON 200 by default."""

    def __init__(self, replies: dict[str, Any] | None = None) -> None:
        self.calls: list[httpx.Request] = []
        self.responses: list[httpx.Response] = []
        self.replies = replies or {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        reply = self.replies.get(str(request.url), JSON_REPLY)
        if callable(reply):
            response = reply(request)
        else:
            response = httpx.Response(
                200, headers={"content-type": "application/json"}, content=reply, request=request
            )
        self.responses.append(response)
        return response


class _Exploding(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)


class FakeAudit:
    """The write-ahead audit log of tests/test_audit_required.py, trimmed."""

    def __init__(self, *, fail_begin: bool = False) -> None:
        self.fail_begin = fail_begin
        self.begun: list[AuditRecord] = []
        self.finalized: list[tuple[object, AuditRecord]] = []
        self.recorded: list[AuditRecord] = []

    def record(self, entry: AuditRecord) -> None:
        self.recorded.append(entry)

    def begin(self, entry: AuditRecord) -> object | None:
        if self.fail_begin:
            raise AuditWriteError("injected write fault")
        self.begun.append(entry)
        return len(self.begun)

    def finalize(self, token: object, entry: AuditRecord) -> None:
        self.finalized.append((token, entry))

    def recent(self, limit: int) -> list[dict[str, object]]:
        return []

    def count(self) -> int:
        return len(self.begun) + len(self.recorded)

    def close(self) -> None:
        pass


def _app(
    monkeypatch: pytest.MonkeyPatch,
    router: FakeRouter,
    transport: httpx.AsyncBaseTransport,
    *,
    config: Config | None = None,
    audit: FakeAudit | None = None,
) -> tuple[ProxyState, httpx.AsyncClient]:
    install(monkeypatch, router, audit=audit)
    app = create_app(config or routed_config(), upstream_transport=transport)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")
    state: ProxyState = app.state.proxy
    assert state.router is router
    return state, client


def _required_audit_config() -> Config:
    return routed_config(audit=AuditConfig(enabled=True, required=True))


# --- refusals and local answers -------------------------------------------------


async def test_plan_refusal_no_route(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = _Upstream()
    router = FakeRouter()
    state, client = _app(monkeypatch, router, upstream)
    with caplog.at_level("INFO", logger="llm_redact"):
        response = await client.post(
            "/v1/messages", json=_messages(f"mail {EMAIL}"), headers={ROUTE_HEADER: "refuse"}
        )
    assert response.status_code == 502
    # Provider-shaped (the anthropic adapter's api_error), no route headers.
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "fake no_route"},
    }
    assert "x-llm-redact-reissue" not in response.headers
    assert upstream.calls == []
    row = state.recent[-1]
    assert row["status"] == 502 and row["provider"] == "anthropic"
    assert row["route"] == {
        "rule": None,
        "upstream": None,
        "hops": 0,
        "auth": "none",
        "class": "no_route",
        "reissue": "no",
    }
    # Refused BEFORE redaction (decision 2): nothing was redacted or minted.
    assert row["detections"] == {} and state.detection_counts == Counter()
    assert (
        "POST /v1/messages -> 502 rule=- upstream=- hops=0 auth=none class=no_route reissue=no"
        in caplog.text
    )
    assert EMAIL not in caplog.text


async def test_pass_through_refusal_with_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = _Upstream()
    router = FakeRouter(refusal_headers={"x-llm-redact-reissue": "skipped; reason=no-candidate"})
    state, client = _app(monkeypatch, router, upstream)
    response = await client.get("/v1/models/x", headers={ROUTE_HEADER: "refuse"})
    assert response.status_code == 502
    assert response.json() == {"error": "fake no_route"}  # adapter None: the bare shape
    assert response.headers["x-llm-redact-reissue"] == "skipped; reason=no-candidate"
    assert upstream.calls == []
    row = state.recent[-1]
    assert row["provider"] is None and row["route"]["class"] == "no_route"
    inbound = router.inbounds[0]
    assert inbound.adapter_name is None and inbound.provider_name == "openai"
    assert inbound.method == "GET" and inbound.path == "/v1/models/x" and inbound.model is None


async def test_begin_refusal_carries_audit_token(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = _Upstream()
    audit = FakeAudit()
    router = FakeRouter(
        {"budget": [Refuse402(headers={"x-llm-redact-reissue": "skipped; reason=stateful"})]}
    )
    state, client = _app(
        monkeypatch, router, upstream, config=_required_audit_config(), audit=audit
    )
    response = await client.post(
        "/v1/messages", json=_messages(f"mail {EMAIL}"), headers={ROUTE_HEADER: "budget"}
    )
    assert response.status_code == 402
    assert response.json()["error"]["type"] == "billing_error"
    assert response.headers["x-llm-redact-reissue"] == "skipped; reason=stateful"
    assert upstream.calls == []
    # The 402 is an ATTEMPT: the write-ahead START row exists and the refusal
    # finalized it with the token (never an orphaned START row).
    assert len(audit.begun) == 1 and audit.begun[0].detections == {"EMAIL": 1}
    assert [token for token, _entry in audit.finalized] == [1]
    assert audit.finalized[0][1].status == 402
    row = state.recent[-1]["route"]
    assert row["class"] == "budget_exhausted" and row["reissue"] == "skipped:stateful"
    # begin() saw the REDACTED body, its decoded form and the forwardable
    # headers (hop-by-hop dropped, accept-encoding: identity added).
    outbound, outbound_obj, forward_headers = router.plans[0].begun[0]
    assert TOKEN.encode() in outbound and EMAIL.encode() not in outbound
    assert isinstance(outbound_obj, dict) and TOKEN in json.dumps(outbound_obj, ensure_ascii=False)
    names = [name for name, _value in forward_headers]
    assert "host" not in names and ("accept-encoding", "identity") in list(forward_headers)


async def test_local_answer_is_recorded_before_body_read(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = _Upstream()
    answer = LocalAnswer(
        200,
        {"object": "list", "data": [{"id": "m", "object": "model"}]},
        provider="routing",
        reason="routing.expose_models",
    )
    router = FakeRouter(local={"/v1/models": answer})
    state, client = _app(monkeypatch, router, upstream)
    with caplog.at_level("INFO", logger="llm_redact"):
        response = await client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == dict(answer.body)
    assert upstream.calls == [] and router.inbounds == []  # never planned, never forwarded
    row = state.recent[-1]
    assert row["provider"] == "routing" and row["method"] == "GET"
    assert row["path"] == "/v1/models" and row["status"] == 200 and row["route"] is None
    assert "GET /v1/models -> 200 answered locally (routing.expose_models)" in caplog.text
    assert router.local_calls == [("GET", "/v1/models")]


async def test_local_answer_sits_behind_the_disabled_provider_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # /v1/models infers provider openai: with that provider disabled the
    # 502 answers first and the router is never asked.
    router = FakeRouter(local={"/v1/models": LocalAnswer(200, {}, provider="routing", reason="x")})
    config = routed_config(
        providers={
            **Config().providers,
            "openai": ProviderConfig(upstream_base_url="http://up", enabled=False),
        }
    )
    _state, client = _app(monkeypatch, router, _Upstream(), config=config)
    response = await client.get("/v1/models")
    assert response.status_code == 502
    assert router.local_calls == []


# --- issuing hops ---------------------------------------------------------------


async def test_unavailable_hop_counts_without_sending(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = _Upstream()
    router = FakeRouter(
        {"go": [Hop("a", URL_A, unavailable="FAKE_KEY_VAR"), Stop()]},
        plan_kwargs={"go": {"delivery_headers": {"x-llm-redact-upstream": "a"}}},
    )
    state, client = _app(monkeypatch, router, upstream)
    with caplog.at_level("WARNING", logger="llm_redact"):
        response = await client.post(
            "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "go"}
        )
    assert response.status_code == 502
    assert response.headers["x-llm-redact-upstream"] == "a"
    assert upstream.calls == []  # nothing sent
    plan = router.plans[0]
    assert plan.results == [HopResult("a", None, {}, "MissingCredential")]
    assert state.upstream_errors["a"] == 1
    assert state.metrics.routed[("a", "r")] == 1
    assert "POST /v1/messages upstream a unavailable: FAKE_KEY_VAR" in caplog.text


async def test_send_fault_is_a_hop_result(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    router = FakeRouter({"go": [Hop("a", URL_A), Stop()]})
    state, client = _app(monkeypatch, router, _Exploding())
    with caplog.at_level("WARNING", logger="llm_redact"):
        response = await client.post(
            "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "go"}
        )
    assert response.status_code == 502
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "llm-redact: upstream request failed"},
    }
    result = router.plans[0].results[0]
    assert result.upstream == "a" and result.status is None and result.fault == "ConnectError"
    assert state.upstream_errors["a"] == 1
    assert "POST /v1/messages upstream a fault (ConnectError)" in caplog.text


async def test_buffered_read_fault_is_a_hop_result(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def dropping(request: httpx.Request) -> httpx.Response:
        async def body() -> Any:
            yield b'{"partial":'
            raise httpx.ReadError("upstream dropped mid-body", request=request)

        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=body(), request=request
        )

    upstream = _Upstream({URL_A: dropping})
    router = FakeRouter({"go": [Hop("a", URL_A), Stop()]})
    state, client = _app(monkeypatch, router, upstream)
    with caplog.at_level("WARNING", logger="llm_redact"):
        response = await client.post(
            "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "go"}
        )
    # A body _deliver would BUFFER is pre-read: the drop surfaces as a hop
    # fault while no byte reached the client (R-17), never as a partial body.
    assert response.status_code == 502
    assert router.plans[0].results[0].fault == "ReadError"
    assert state.upstream_errors["a"] == 1
    assert "POST /v1/messages upstream a fault while reading the body (ReadError)" in caplog.text
    assert upstream.responses[0].is_closed


async def test_hop_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = _Upstream()
    router = FakeRouter({"go": [Hop("a", URL_A), Stop()]})
    state, client = _app(monkeypatch, router, upstream)
    response = await client.post(
        "/v1/messages", json=_messages(f"mail {EMAIL}"), headers={ROUTE_HEADER: "go"}
    )
    assert response.status_code == 200 and response.content == JSON_REPLY
    result = router.plans[0].results[0]
    assert result.upstream == "a" and result.status == 200 and result.fault is None
    assert result.headers["content-type"] == "application/json"
    # The hop's own URL, headers and (redacted) body were sent — the core added nothing.
    sent = upstream.calls[0]
    assert str(sent.url) == URL_A
    assert TOKEN.encode() in sent.content and EMAIL.encode() not in sent.content
    assert sent.headers["accept-encoding"] == "identity"
    assert state.metrics.routed[("a", "r")] == 1


async def test_streamed_bodies_are_not_pre_read(monkeypatch: pytest.MonkeyPatch) -> None:
    class _CountingRead(httpx.Response):
        reads = 0

        async def aread(self) -> bytes:
            self.reads += 1
            return await super().aread()

    def sse(request: httpx.Request) -> httpx.Response:
        return _CountingRead(
            200, headers={"content-type": "text/event-stream"}, content=_sse("hi"), request=request
        )

    def plain(request: httpx.Request) -> httpx.Response:
        return _CountingRead(
            200, headers={"content-type": "application/json"}, content=JSON_REPLY, request=request
        )

    upstream = _Upstream({URL_A: sse, URL_B: plain})
    router = FakeRouter({"sse": [Hop("a", URL_A), Stop()], "json": [Hop("b", URL_B), Stop()]})
    _state, client = _app(monkeypatch, router, upstream)
    response = await client.post(
        "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "sse"}
    )
    assert response.status_code == 200 and b"event: message_stop" in response.content
    assert upstream.responses[0].reads == 0  # streamed: first byte reaches the client as it arrives
    response = await client.post(
        "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "json"}
    )
    assert response.status_code == 200
    assert upstream.responses[1].reads == 2  # the pre-read, then the delivery branch's (idempotent)


def test_hop_timeout_bounds() -> None:
    expired = _hop_timeout(-1.0)
    assert (expired.connect, expired.read, expired.write, expired.pool) == (0.001,) * 4
    wide = _hop_timeout(1000.0)
    assert (
        wide.read == 600.0 and wide.write == 600.0 and wide.pool == 600.0 and wide.connect == 10.0
    )
    short = _hop_timeout(5.0)
    assert short.read == 5.0 and short.connect == 5.0


def test_route_log_suffix_is_fixed_keys_only() -> None:
    assert _route_log_suffix({}) == " rule=- upstream=- hops=0 auth=- class=- reissue=-"
    assert (
        _route_log_suffix(
            {
                "rule": "r",
                "upstream": "a",
                "hops": 2,
                "auth": "oauth",
                "class": "ok",
                "reissue": "yes",
            }
        )
        == " rule=r upstream=a hops=2 auth=oauth class=ok reissue=yes"
    )


# --- the driver's ordering and loop ---------------------------------------------


async def test_local_refusal_precedes_audit_start(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = _Upstream()
    audit = FakeAudit()
    router = FakeRouter({"ct": [Refuse404()]})
    state, client = _app(
        monkeypatch, router, upstream, config=_required_audit_config(), audit=audit
    )
    response = await client.post(
        "/v1/messages/count_tokens", json=_messages("hi"), headers={ROUTE_HEADER: "ct"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"
    assert upstream.calls == []
    # Never an attempt: no START row was written; the classic row records it.
    assert audit.begun == [] and audit.finalized == []
    assert len(audit.recorded) == 1 and audit.recorded[0].status == 404
    assert state.recent[-1]["route"]["class"] == "404"
    assert router.plans[0].begun == []


async def test_audit_refusal_503_on_routed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = _Upstream()
    router = FakeRouter({"go": [Hop("a", URL_A), Stop()]})
    _state, client = _app(
        monkeypatch,
        router,
        upstream,
        config=_required_audit_config(),
        audit=FakeAudit(fail_begin=True),
    )
    response = await client.post("/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "go"})
    assert response.status_code == 503
    assert response.json()["error"]["message"] == (
        "llm-redact: audit log unavailable and [audit] required is enabled"
    )
    assert upstream.calls == [] and router.plans[0].begun == []  # no upstream contact, no hop


async def test_loop_waits_then_reissues(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = _Upstream({URL_A: b'{"which":"a"}', URL_B: b'{"which":"b"}'})
    router = FakeRouter(
        {"chain": [Hop("a", URL_A), Hop("b", URL_B, wait=0.01, reissued_from="a"), Stop()]}
    )
    state, client = _app(monkeypatch, router, upstream)
    response = await client.post(
        "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "chain"}
    )
    assert response.status_code == 200 and response.json() == {"which": "b"}
    assert [str(call.url) for call in upstream.calls] == [URL_A, URL_B]
    plan = router.plans[0]
    assert plan.waits == [0.01]  # awaited through the plan, not the loop's own sleep
    assert [(r.upstream, r.status, r.fault) for r in plan.results] == [
        ("a", 200, None),
        ("b", 200, None),
    ]
    assert upstream.responses[0].is_closed  # the undelivered first response was discarded
    assert state.metrics.reissues[("a", "b")] == 1
    assert state.metrics.routed[("b", "r")] == 1
    assert state.recent[-1]["route"]["upstream"] == "b"


async def test_undelivered_streamed_hop_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A streamed (SSE) response is NOT pre-read by _issue_hop, so only the
    # loop's own discard can close it when the router moves to another hop;
    # a buffered first hop would be closed by its pre-read either way.
    # A real async stream: httpx reads a bytes body eagerly (closing it at
    # construction), which would make this assertion hold with no discard.
    async def chunks() -> Any:
        yield _sse("from a")

    def streamed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=chunks(),
            request=request,
        )

    upstream = _Upstream({URL_A: streamed, URL_B: b'{"which":"b"}'})
    router = FakeRouter({"chain": [Hop("a", URL_A), Hop("b", URL_B, reissued_from="a"), Stop()]})
    _state, client = _app(monkeypatch, router, upstream)
    response = await client.post(
        "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "chain"}
    )
    assert response.json() == {"which": "b"}
    assert upstream.responses[0].headers["content-type"] == "text/event-stream"
    assert upstream.responses[0].is_closed


async def test_hop_timeout_is_derived_from_the_plan_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The per-hop httpx timeout comes from the plan's deadline, not the
    # shared client's 600 s: with 5 s left the sent request carries <= 5 s.
    upstream = _Upstream()
    router = FakeRouter(
        {"short": [Hop("a", URL_A), Stop()]}, plan_kwargs={"short": {"deadline_seconds": 5.0}}
    )
    _state, client = _app(monkeypatch, router, upstream)
    response = await client.post(
        "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "short"}
    )
    assert response.status_code == 200
    timeout = upstream.calls[0].extensions["timeout"]
    assert 4.0 < timeout["read"] <= 5.0
    assert 4.0 < timeout["connect"] <= 5.0


async def test_retry_same_waits_without_reissue_count(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = _Upstream()
    router = FakeRouter(
        {
            "retry": [Hop("a", URL_A), Hop("a", URL_A, wait=2.0), Stop()],
            "nowait": [Hop("a", URL_A), Hop("b", URL_B, reissued_from="a"), Stop()],
        }
    )
    state, client = _app(monkeypatch, router, upstream)
    response = await client.post(
        "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "retry"}
    )
    assert response.status_code == 200
    assert [str(call.url) for call in upstream.calls] == [URL_A, URL_A]
    assert router.plans[0].waits == [2.0]
    assert state.metrics.reissues == Counter()  # retry-same is not a re-issue
    response = await client.post(
        "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "nowait"}
    )
    assert response.status_code == 200
    assert router.plans[1].waits == []  # wait_seconds 0 never calls wait()
    assert state.metrics.reissues[("a", "b")] == 1


async def test_all_hops_fail_is_502_with_route_headers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    router = FakeRouter(
        {"go": [Hop("a", URL_A), Stop()]},
        plan_kwargs={
            "go": {
                "rule": None,
                "delivery_headers": {"x-llm-redact-upstream": "a", "x-llm-redact-hops": "1"},
            }
        },
    )
    state, client = _app(monkeypatch, router, _Exploding())
    with caplog.at_level("INFO", logger="llm_redact"):
        response = await client.post(
            "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "go"}
        )
    assert response.status_code == 502
    assert response.headers["x-llm-redact-upstream"] == "a"
    assert response.headers["x-llm-redact-hops"] == "1"
    assert state.metrics.routed[("a", "-")] == 1  # every routed outcome, 502s included
    assert state.upstream_errors["a"] == 1
    delivered = router.plans[0].delivered
    assert delivered is not None and delivered.finished == [502]
    assert state.recent[-1]["route"] == delivered.row()
    assert "POST /v1/messages -> 502 rule=- upstream=a hops=1 auth=none class=ok reissue=no" in (
        caplog.text
    )


async def test_deliver_after_stop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = _Upstream()
    router = FakeRouter(
        {"go": [Hop("a", URL_A), Stop()]},
        plan_kwargs={"go": {"delivery_headers": {"x-llm-redact-upstream": "a"}}},
    )
    state, client = _app(monkeypatch, router, upstream)
    with caplog.at_level("INFO", logger="llm_redact"):
        response = await client.post(
            "/v1/messages", json=_messages(f"mail {EMAIL}"), headers={ROUTE_HEADER: "go"}
        )
    assert response.status_code == 200 and response.content == JSON_REPLY
    assert response.headers["x-llm-redact-upstream"] == "a"
    assert response.headers["content-type"] == "application/json"
    delivered = router.plans[0].delivered
    assert delivered is not None and delivered.finished == [200]
    row = state.recent[-1]
    assert row["route"] == delivered.row() and row["streamed"] is False
    assert row["detections"] == {"EMAIL": 1}
    assert state.metrics.routed[("a", "r")] == 1
    assert (
        "POST /v1/messages -> 200 rule=r upstream=a hops=1 auth=none class=ok reissue=no"
        " redacted: EMAIL×1" in caplog.text
    )
    assert EMAIL not in caplog.text


# --- the delivery hooks on every branch -------------------------------------------


async def test_delivery_hooks_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    def sse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(f"mail {TOKEN} ok"),
            request=request,
        )

    upstream = _Upstream({URL_A: sse})
    router = FakeRouter(
        {"go": [Hop("a", URL_A), Stop()]},
        plan_kwargs={"go": {"mutate": True, "delivery_headers": {"x-llm-redact-upstream": "a"}}},
    )
    state, client = _app(monkeypatch, router, upstream)
    response = await client.post(
        "/v1/messages", json=_messages(f"mail {EMAIL}"), headers={ROUTE_HEADER: "go"}
    )
    assert response.status_code == 200
    assert response.headers["x-llm-redact-upstream"] == "a"
    text = response.text
    # The hook ran AFTER the adapter's rehydration and BEFORE serialization:
    # the restored value and the hook's marker both reached the client.
    assert EMAIL in text and TOKEN not in text
    assert text.count('"observed": true') == 5
    delivered = router.plans[0].delivered
    assert delivered is not None and len(delivered.events) == 5
    assert delivered.finished == [200]
    assert state.metrics.routed[("a", "r")] == 1
    row = state.recent[-1]
    assert row["streamed"] is True and row["rehydrations"] == {"EMAIL": 1}
    assert row["route"] == delivered.row()


async def test_delivery_hooks_ndjson(monkeypatch: pytest.MonkeyPatch) -> None:
    def ndjson(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content=_ndjson(f"saw {TOKEN} end"),
            request=request,
        )

    url = "http://a.example/api/chat"
    upstream = _Upstream({url: ndjson})
    router = FakeRouter({"go": [Hop("a", url), Stop()]}, plan_kwargs={"go": {"mutate": True}})
    state, client = _app(monkeypatch, router, upstream)
    body = {"model": "m", "messages": [{"role": "user", "content": f"mail {EMAIL} now"}]}
    response = await client.post("/api/chat", json=body, headers={ROUTE_HEADER: "go"})
    assert response.status_code == 200
    lines = [json.loads(line) for line in response.content.splitlines() if line.strip()]
    assert [line["observed"] for line in lines] == [True, True]
    assert lines[0]["message"]["content"] == f"saw {EMAIL} end"
    delivered = router.plans[0].delivered
    assert delivered is not None and len(delivered.lines) == 2
    assert delivered.finished == [200]
    assert state.metrics.routed[("a", "r")] == 1
    assert state.recent[-1]["streamed"] is True


async def test_observe_payload_reserializes(monkeypatch: pytest.MonkeyPatch) -> None:
    distinctive = b'{"role" :  "assistant","content":[{"type":"text","text":"plain"}]}'
    upstream = _Upstream({URL_A: distinctive})
    router = FakeRouter(
        {"plain": [Hop("a", URL_A), Stop()], "mut": [Hop("a", URL_A), Stop()]},
        plan_kwargs={"mut": {"mutate": True}},
    )
    _state, client = _app(monkeypatch, router, upstream)
    response = await client.post(
        "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "plain"}
    )
    # Observed but unchanged: the no-op short-circuit still forwards the
    # upstream's exact bytes.
    assert response.content == distinctive
    assert router.plans[0].delivered is not None
    assert router.plans[0].delivered.payloads[0][1] is RouteKind.CHAT
    response = await client.post(
        "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "mut"}
    )
    assert response.json()["observed"] is True  # the hook changed it: re-serialized
    assert response.json()["content"][0]["text"] == "plain"


async def test_redact_only_payload_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "http://a.example/v1/embeddings"
    upstream = _Upstream({url: b'{"object":"list","data":[],"usage":{"prompt_tokens":3}}'})
    router = FakeRouter({"go": [Hop("a", url), Stop()]}, plan_kwargs={"go": {"mutate": True}})
    _state, client = _app(monkeypatch, router, upstream)
    response = await client.post(
        "/v1/embeddings",
        json={"model": "m", "input": f"mail {EMAIL}"},
        headers={ROUTE_HEADER: "go"},
    )
    assert response.status_code == 200
    assert response.json()["observed"] is True
    assert (
        TOKEN.encode() in upstream.calls[0].content
        and EMAIL.encode() not in upstream.calls[0].content
    )
    delivered = router.plans[0].delivered
    assert delivered is not None
    assert delivered.wants_payload_calls == [RouteKind.REDACT_ONLY]
    assert delivered.payloads[0][1] is RouteKind.REDACT_ONLY


async def test_passthrough_payload_not_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "http://a.example/v1/models/x"
    upstream = _Upstream({url: b'{"id" : "x"}'})
    router = FakeRouter(
        {"skip": [Hop("a", url), Stop()], "want": [Hop("a", url), Stop()]},
        plan_kwargs={"skip": {"wants_payload": False}},
    )
    _state, client = _app(monkeypatch, router, upstream)
    loads_calls: list[Any] = []
    real_loads = json.loads

    def counting_loads(*args: Any, **kwargs: Any) -> Any:
        loads_calls.append(args[0])
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(proxy_mod.json, "loads", counting_loads)
    response = await client.get("/v1/models/x", headers={ROUTE_HEADER: "skip"})
    assert response.status_code == 200 and response.content == b'{"id" : "x"}'
    assert loads_calls == []  # a pass-through body the router does not want is never parsed
    delivered = router.plans[0].delivered
    assert delivered is not None
    assert delivered.wants_payload_calls == [RouteKind.NONE] and delivered.payloads == []

    response = await client.get("/v1/models/x", headers={ROUTE_HEADER: "want"})
    assert response.status_code == 200 and response.content == b'{"id" : "x"}'
    assert loads_calls == [b'{"id" : "x"}']  # wanted: parsed once, observed, left unchanged
    delivered = router.plans[1].delivered
    assert delivered is not None and delivered.payloads[0][1] is RouteKind.NONE


async def test_buffered_fault_in_deliver_closes_route(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _FlakyRead(httpx.Response):
        """aread() succeeds once (the driver's pre-read) and raises on the
        next call (the delivery branch's): a body gone between the two."""

        reads = 0

        async def aread(self) -> bytes:
            self.reads += 1
            if self.reads >= 2:
                raise httpx.ReadError("dropped after the pre-read", request=self.request)
            return await super().aread()

    def flaky(request: httpx.Request) -> httpx.Response:
        return _FlakyRead(
            200, headers={"content-type": "application/json"}, content=JSON_REPLY, request=request
        )

    upstream = _Upstream({URL_A: flaky})
    router = FakeRouter(
        {"go": [Hop("a", URL_A), Stop()]},
        plan_kwargs={"go": {"delivery_headers": {"x-llm-redact-upstream": "a"}}},
    )
    state, client = _app(monkeypatch, router, upstream)
    with caplog.at_level("WARNING", logger="llm_redact"):
        response = await client.post(
            "/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "go"}
        )
    assert response.status_code == 502
    assert response.headers["x-llm-redact-upstream"] == "a"
    assert router.plans[0].results[0].fault is None  # the pre-read had succeeded
    delivered = router.plans[0].delivered
    assert delivered is not None
    assert delivered.failed == ["transport"] and delivered.finished == [502]
    assert state.upstream_errors["a"] == 1
    assert state.metrics.routed[("a", "r")] == 1
    assert state.recent[-1]["route"]["class"] == "transport"
    assert (
        "POST /v1/messages -> 502 upstream fault (ReadError) rule=r upstream=a hops=1"
        " auth=none class=transport reissue=no" in caplog.text
    )


class _FaultyBody:
    def __init__(self, chunks: list[bytes], exc: BaseException) -> None:
        self.status_code = 200
        self._chunks = chunks
        self._exc = exc
        self.closed = False

    async def aiter_bytes(self) -> Any:
        for chunk in self._chunks:
            yield chunk
        raise self._exc

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("codec", ["sse", "ndjson"])
async def test_stream_fault_after_first_byte_marks_stream_error(
    codec: str, caplog: pytest.LogCaptureFixture
) -> None:
    app = create_app(Config())
    state: ProxyState = app.state.proxy
    ctx = state._static_context
    token = ctx.vault.placeholder_for("EMAIL", EMAIL)
    adapter: Any
    generator: Callable[..., Any]
    if codec == "sse":
        adapter = AnthropicAdapter()
        chunk = _sse(f"hi {token}")
        generator = _stream_rehydrated
    else:
        adapter = OllamaAdapter()
        chunk = _ndjson(f"hi {token}")
        generator = _stream_rehydrated_ndjson
    delivery = FakeDelivery("a", "r", {"x-llm-redact-upstream": "a"})
    upstream = _FaultyBody([chunk], httpx.ReadError("upstream dropped mid-stream"))
    gen = generator(
        upstream,
        adapter,
        state,
        ctx,
        request_meta=RequestMeta("POST", "/v1/x", time.perf_counter(), {}, {}),
        route=delivery,
    )
    out = bytearray()
    with caplog.at_level("WARNING", logger="llm_redact"), pytest.raises(httpx.TransportError):
        async for piece in gen:
            out += piece
    assert EMAIL.encode() in bytes(out) and token.encode() not in bytes(out)
    # After the first byte a fault is never re-issued: classified stream_error,
    # logged by the core with the fixed-key suffix, finalized once.
    assert delivery.failed == ["stream_error"] and delivery.finished == [200]
    assert (
        "POST /v1/x stream failed after first byte (ReadError) rule=r upstream=a hops=1"
        " auth=none class=stream_error reissue=no" in caplog.text
    )
    assert state.metrics.routed[("a", "r")] == 1
    assert state.recent[-1]["route"]["class"] == "stream_error"
    assert upstream.closed


# --- the unrouted path is byte-identical and free ---------------------------------------


def _snapshot(state: ProxyState, upstream: _Upstream, response: httpx.Response) -> dict[str, Any]:
    row = {k: v for k, v in state.recent[-1].items() if k not in ("ts", "duration_ms")}
    return {
        "sent": [(str(c.url), c.content, list(c.headers.items())) for c in upstream.calls],
        "status": response.status_code,
        "headers": dict(response.headers),
        "content": response.content,
        "row": row,
    }


async def _fixture_traffic(
    state: ProxyState, client: httpx.AsyncClient, upstream: _Upstream
) -> list[dict[str, Any]]:
    snapshots = []
    response = await client.post("/v1/messages", json=_messages(f"mail {EMAIL}"))
    snapshots.append(_snapshot(state, upstream, response))
    response = await client.post("/v1/messages", json=_messages(f"again {EMAIL}", stream=True))
    snapshots.append(_snapshot(state, upstream, response))
    response = await client.get("/v1/models/x")
    snapshots.append(_snapshot(state, upstream, response))
    return snapshots


def _fixture_upstream() -> _Upstream:
    def sse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(f"mail {TOKEN} ok"),
            request=request,
        )

    def by_body(request: httpx.Request) -> httpx.Response:
        if b'"stream": true' in request.content:
            return sse(request)
        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=JSON_REPLY, request=request
        )

    return _Upstream(
        {
            "https://api.anthropic.com/v1/messages": by_body,
            "https://api.openai.com/v1/models/x": b'{"id":"x"}',
        }
    )


async def test_unrouted_path_never_consults_router(monkeypatch: pytest.MonkeyPatch) -> None:
    # Baseline: the default registry, routing absent.
    upstream = _fixture_upstream()
    app = create_app(Config(), upstream_transport=upstream)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")
    baseline = await _fixture_traffic(app.state.proxy, client, upstream)
    assert all(snapshot["row"]["route"] is None for snapshot in baseline)

    # An every-method-raises router is registered. With [routing] absent the
    # factory is consulted once, sees routing disabled and returns None, so
    # the proxy never HOLDS a router: this pins that the factory decision is
    # driven by the config (routing off => None) and that the same traffic
    # then round-trips byte-identical. The router-held-but-declining case is
    # test_plan_none_takes_legacy_path below.
    router = FakeRouter(raise_everything=True)
    _reg, calls = install(monkeypatch, router)
    upstream = _fixture_upstream()
    app = create_app(Config(), upstream_transport=upstream)
    state: ProxyState = app.state.proxy
    assert state.router is None and len(calls) == 1
    assert calls[0][0].routing.enabled is False
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")
    assert await _fixture_traffic(state, client, upstream) == baseline
    # /status reads the router attribute only.
    assert (await client.get("/__llm-redact/status")).json()["routing"] == {"enabled": False}


async def test_plan_none_takes_legacy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = _fixture_upstream()
    app = create_app(Config(), upstream_transport=upstream)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")
    baseline = await _fixture_traffic(app.state.proxy, client, upstream)

    # A router is held, but plan() returns None for every request without
    # the scripted header (a provider outside its protocols): the legacy
    # tail runs with route=None and the bytes are the baseline's.
    router = FakeRouter()
    upstream = _fixture_upstream()
    state, client = _app(monkeypatch, router, upstream)
    assert await _fixture_traffic(state, client, upstream) == baseline
    assert len(router.inbounds) == 3  # planned (and declined) once per request
    first = router.inbounds[0]
    assert first.adapter_name == "anthropic" and first.provider_name == "anthropic"
    assert first.method == "POST" and first.path == "/v1/messages"
    assert first.raw_path == "/v1/messages" and first.query == "" and first.model == "m"
    assert router.inbounds[2].adapter_name is None and router.inbounds[2].model is None


async def test_plan_sees_inbound_facts_and_governs_note_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _Upstream()
    router = FakeRouter(
        {"note": [Hop("a", URL_A), Stop()], "nonote": [Hop("a", URL_A), Stop()]},
        plan_kwargs={"nonote": {"inject_system_note": False}},
    )
    _state, client = _app(monkeypatch, router, upstream)
    await client.post(
        "/v1/messages?beta=true",
        json={**_messages(f"mail {EMAIL}"), "model": 7},
        headers={ROUTE_HEADER: "note"},
    )
    inbound = router.inbounds[0]
    assert inbound.query == "beta=true" and inbound.raw_path == "/v1/messages"
    assert inbound.model is None  # a non-string model is never handed over
    assert inbound.headers[ROUTE_HEADER] == "note"
    # Decision 4: the plan's inject_system_note governs the prepared body.
    assert "system" in json.loads(upstream.calls[0].content)
    await client.post(
        "/v1/messages", json=_messages(f"mail {EMAIL}"), headers={ROUTE_HEADER: "nonote"}
    )
    assert "system" not in json.loads(upstream.calls[1].content)


def test_upstreams_alone_logs_inert_warning_without_router(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = parse_config(
        tomllib.loads('[upstreams.x]\nprotocol = "anthropic"\nbase_url = "http://x.example"\n'),
        "<test>",
    )
    with caplog.at_level("WARNING", logger="llm_redact"):
        state = ProxyState(config, None)
    assert state.router is None
    assert (
        "routing: [upstreams] is configured but there is no [routing] table: upstreams are inert"
        in caplog.text
    )


# --- status, lifespan, finish_route -------------------------------------------------------


async def test_status_carries_router_block(monkeypatch: pytest.MonkeyPatch) -> None:
    router = FakeRouter(status={"enabled": True, "rules": 2, "upstreams": {}})
    _state, client = _app(monkeypatch, router, _Upstream())
    payload = (await client.get("/__llm-redact/status")).json()
    assert payload["routing"] == {"enabled": True, "rules": 2, "upstreams": {}}
    assert list(payload)[-1] == "routing"  # the last top-level key


async def test_status_reports_enabled_false_without_router() -> None:
    app = create_app(Config())
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")
    payload = (await client.get("/__llm-redact/status")).json()
    assert payload["routing"] == {"enabled": False}
    assert list(payload)[-1] == "routing"


async def test_lifespan_closes_router(monkeypatch: pytest.MonkeyPatch) -> None:
    router = FakeRouter()
    install(monkeypatch, router)
    app = create_app(routed_config())
    async with app.router.lifespan_context(app):
        assert router.closed == 0
    assert router.closed == 1


def test_finish_route_counts_every_outcome() -> None:
    state: ProxyState = create_app(Config()).state.proxy
    delivery = FakeDelivery("a", None, {})
    assert state.finish_route(delivery, 502) == delivery.row()
    delivery.mark_failed("transport")
    assert state.finish_route(delivery, None)["class"] == "transport"
    assert state.metrics.routed[("a", "-")] == 2  # rule None renders as "-"
    assert delivery.finished == [502, None]
