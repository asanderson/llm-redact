"""Faults in the bookkeeping that runs after the provider answered.

After the upstream responds, the proxy restores listed objects in their
owners' sessions (vault reads), reports stored objects and response ids to
the session router (a router call and a durable-map write). None of that
was contained: a vault or router fault turned a delivered answer into
Starlette's bare 500 — unrecorded, the ``[audit] required`` START row never
finalized, a billed ``store: true`` completion thrown away — and killed a
streamed completion before its first byte while recording a 200 success.

Now each is contained and counted (``bookkeeping_errors_total`` in /status,
``llm_redact_bookkeeping_errors_total{stage}``, a WARNING naming the stage
and exception TYPE only), and the answer is delivered: a listed item whose
owner's session cannot be read is delivered exactly as the provider sent it
(placeholders in place — never another session's value), a lost ownership
record or response-id mapping leaves the object unattributed (the router's
unknown-object handling, never a wrong value). Anything else failing while
the answer is restored is a recorded, provider-shaped 502 (never a bare
500), its audit END row finalized.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

import httpx
import pytest

from license_fixtures import resolved
from llm_redact.config import AuditConfig, ProviderConfig
from llm_redact.registry import Registry
from local_refusals import refused_once
from test_audit_required import FakeAudit
from test_object_access_seams import ScriptedRouter, Upstream, _app, _client

EMAIL = "jane.doe@corp.example"
OTHER = "sam.roe@corp.example"
TOKEN = "«EMAIL_001»"
SECRET_ID = "file-secretid"


class TrackingRouter(ScriptedRouter):
    """Reports stored objects and response ids; either can be made to fail."""

    def __init__(self, *, object_error: Exception | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.object_error = object_error
        self.response_error: Exception | None = None
        self.objects: list[tuple[str, str]] = []

    def record_object_id(self, object_id: str, session_id: str) -> bool:
        if self.object_error is not None:
            raise self.object_error
        self.objects.append((object_id, session_id))
        return True

    def record_response_id(self, response_id: str, session_id: str) -> None:
        if self.response_error is not None:
            raise self.response_error


def _audited(monkeypatch: pytest.MonkeyPatch) -> tuple[Registry, FakeAudit]:
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("pro")
    reg.build_access_gate = lambda cfg, lic: None
    audit = FakeAudit()
    reg.build_audit = lambda cfg: audit if cfg.enabled else None
    return reg, audit


def _completion(object_id: str = "chatcmpl-1") -> dict[str, Any]:
    return {
        "id": object_id,
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": f"hi {TOKEN}"}}],
    }


def _assert_one_row(app: Any, audit: FakeAudit, status: int, *, streamed: bool = False) -> None:
    (row,) = app.state.proxy.recent
    assert (row["status"], row["streamed"]) == (status, streamed)
    assert len(audit.begun) == 1
    ((token, end),) = audit.finalized  # the START row got its END row
    assert token == 1 and end.status == status


# --- stored objects ---------------------------------------------------------------


@pytest.mark.parametrize("where", ["router", "durable map"])
async def test_a_lost_ownership_record_still_delivers_the_answer(
    where: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fault = sqlite3.OperationalError("database is locked: " + SECRET_ID)
    router = TrackingRouter(object_error=fault if where == "router" else None)
    reg, audit = _audited(monkeypatch)
    upstream = Upstream(_completion())
    app = _app(
        monkeypatch, router, upstream, registry=reg, audit=AuditConfig(enabled=True, required=True)
    )
    state = app.state.proxy
    state.vault_manager.get(router.session).placeholder_for("EMAIL", EMAIL)
    if where == "durable map":

        def locked(object_id: str, session_id: str) -> None:
            raise fault

        # A stored object's owner record (bounded apart from Responses rows).
        monkeypatch.setattr(state.vault_manager, "record_object_session", locked)
    caplog.set_level(logging.WARNING, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "store": True, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == f"hi {EMAIL}"
    _assert_one_row(app, audit, 200)
    assert state.bookkeeping_errors == {"object_ids": 1}
    assert "OperationalError" in caplog.text and SECRET_ID not in caplog.text
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
        metrics = (await client.get("/__llm-redact/metrics")).text
    assert status["bookkeeping_errors_total"] == {"object_ids": 1}
    assert 'llm_redact_bookkeeping_errors_total{stage="object_ids"} 1' in metrics


async def test_a_streamed_stored_completion_survives_a_lost_ownership_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = TrackingRouter(object_error=sqlite3.OperationalError("database is locked"))
    reg, audit = _audited(monkeypatch)
    chunks = [
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"content": "hi "}}]},
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"content": TOKEN}}]},
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    sse = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
    sse += b"data: [DONE]\n\n"

    class Streaming(Upstream):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    app = _app(
        monkeypatch,
        router,
        Streaming(),
        registry=reg,
        audit=AuditConfig(enabled=True, required=True),
    )
    state = app.state.proxy
    state.vault_manager.get(router.session).placeholder_for("EMAIL", EMAIL)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "store": True,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    text = "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )
    assert text == f"hi {EMAIL}"
    _assert_one_row(app, audit, 200, streamed=True)
    assert state.bookkeeping_errors == {"object_ids": 1}


async def test_a_lost_response_id_mapping_still_delivers_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = TrackingRouter()
    router.response_error = RuntimeError("router down")
    upstream = Upstream({"id": "resp_1", "object": "response", "output": []})
    app = _app(monkeypatch, router, upstream)
    async with _client(app) as client:
        response = await client.post("/v1/responses", json={"model": "m", "input": "hi"})
    assert response.status_code == 200 and response.json()["id"] == "resp_1"
    assert app.state.proxy.bookkeeping_errors == {"response_id": 1}
    assert app.state.proxy.recent[-1]["status"] == 200


# --- listings ------------------------------------------------------------------


async def test_an_unreadable_owner_session_delivers_the_item_as_the_provider_sent_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The listing is read in a POPULATED session ("shared" holds OTHER as
    # «EMAIL_001»): an item named to a session that cannot be read must keep
    # the provider's placeholder — never the request session's value.
    router = ScriptedRouter(session="shared", owners={"video_a": "user:n1:main"})
    reg, audit = _audited(monkeypatch)
    listing = {
        "object": "list",
        "data": [
            {"id": "video_a", "object": "video", "prompt": f"film {TOKEN}"},
            {"id": "video_b", "object": "video", "prompt": f"film {TOKEN}"},
        ],
    }
    app = _app(
        monkeypatch,
        router,
        Upstream(listing),
        registry=reg,
        audit=AuditConfig(enabled=True, required=True),
    )
    manager = app.state.proxy.vault_manager
    manager.get("shared").placeholder_for("EMAIL", OTHER)
    real_get = manager.get

    def failing_get(session_id: str) -> Any:
        if session_id == "user:n1:main":
            raise sqlite3.OperationalError("disk I/O error")
        return real_get(session_id)

    monkeypatch.setattr(manager, "get", failing_get)
    monkeypatch.setattr(manager, "has_session", lambda session_id: True)
    async with _client(app) as client:
        response = await client.get("/v1/videos")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data[0]["prompt"] == f"film {TOKEN}"  # unrestored, never OTHER
    assert data[1]["prompt"] == f"film {OTHER}"  # the request's own session, as before
    _assert_one_row(app, audit, 200)
    assert app.state.proxy.bookkeeping_errors == {"listing": 1}


# --- the backstop -------------------------------------------------------------------


async def test_a_fault_restoring_the_answer_is_a_recorded_502(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    router = ScriptedRouter(mode="static")
    reg, audit = _audited(monkeypatch)
    app = _app(
        monkeypatch,
        router,
        Upstream(_completion()),
        registry=reg,
        audit=AuditConfig(enabled=True, required=True),
        providers={"openai": ProviderConfig("https://up.test")},
    )
    vault = app.state.proxy._static_context.vault  # noqa: SLF001

    def unreadable(placeholder: str) -> str | None:
        raise ValueError("ciphertext for " + SECRET_ID + " does not decrypt")

    monkeypatch.setattr(vault, "original_for", unreadable)
    caplog.set_level(logging.WARNING, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )
    assert response.status_code == 502
    assert response.json()["error"]["message"].startswith("llm-redact:")
    assert TOKEN not in response.text  # nothing of the unrestored answer
    _assert_one_row(app, audit, 502)
    assert app.state.proxy.bookkeeping_errors == {"delivery": 1}
    assert "ValueError" in caplog.text and SECRET_ID not in caplog.text
    refused_once(app.state.proxy, "delivery_fault", "openai")


# --- streams: a restore fault cuts the stream, counted, booked as a 502 ---------


class _Body:
    """An httpx.Response stand-in streaming ``chunks`` (no transport fault)."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.status_code = 200
        self._chunks = chunks
        self.closed = False

    async def aiter_bytes(self) -> Any:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("codec", ["sse", "eventstream", "ndjson"])
async def test_a_stream_restore_fault_is_counted_and_the_row_finalized(
    codec: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import time

    from llm_redact.config import Config
    from llm_redact.providers import BedrockAdapter, OllamaAdapter, OpenAIAdapter
    from llm_redact.proxy import (
        RequestMeta,
        _stream_rehydrated,
        _stream_rehydrated_eventstream,
        _stream_rehydrated_ndjson,
        create_app,
    )
    from test_upstream_faults import _eventstream_delta

    app = create_app(Config())
    state = app.state.proxy
    ctx = state._static_context  # noqa: SLF001

    def unreadable(placeholder: str) -> str | None:
        raise ValueError("ciphertext for " + SECRET_ID + " does not decrypt")

    monkeypatch.setattr(ctx.vault, "original_for", unreadable)
    adapter: Any
    if codec == "sse":
        adapter, gen_fn = OpenAIAdapter(), _stream_rehydrated
        delta = {"choices": [{"index": 0, "delta": {"content": f"hi {TOKEN}"}}]}
        chunk = b"data: " + json.dumps(delta).encode() + b"\n\n"
    elif codec == "eventstream":
        adapter, gen_fn = BedrockAdapter(), _stream_rehydrated_eventstream
        chunk = _eventstream_delta(TOKEN)
    else:
        adapter, gen_fn = OllamaAdapter(), _stream_rehydrated_ndjson
        line = {"message": {"content": f"hi {TOKEN}"}, "done": True}
        chunk = json.dumps(line).encode() + b"\n"
    upstream = _Body([chunk])
    caplog.set_level(logging.WARNING, logger="llm_redact")
    with pytest.raises(ValueError):
        async for _piece in gen_fn(
            upstream,  # type: ignore[arg-type]
            adapter,
            state,
            ctx,
            request_meta=RequestMeta("POST", "/v1/x", time.perf_counter(), {}, {}),
        ):
            pass
    assert upstream.closed
    assert state.bookkeeping_errors == {"delivery": 1}
    (row,) = state.recent
    # The proxy cut the stream: booked as its failure, never the upstream's 200.
    assert (row["streamed"], row["status"]) == (True, 502)
    assert "ValueError" in caplog.text and SECRET_ID not in caplog.text


# --- routed requests: the route is closed as the proxy's 502 --------------------


async def test_a_routed_restore_fault_closes_the_route_as_a_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop
    from test_routing_seam import URL_A, _app, _messages, _Upstream

    reply = {"role": "assistant", "content": [{"type": "text", "text": f"hi {TOKEN}"}]}
    upstream = _Upstream({URL_A: json.dumps(reply).encode()})
    router = FakeRouter(
        {"go": [Hop("a", URL_A), Stop()]},
        plan_kwargs={"go": {"delivery_headers": {"x-llm-redact-upstream": "a"}}},
    )
    state, client = _app(monkeypatch, router, upstream)

    def unreadable(placeholder: str) -> str | None:
        raise ValueError("does not decrypt")

    monkeypatch.setattr(state._static_context.vault, "original_for", unreadable)  # noqa: SLF001
    response = await client.post("/v1/messages", json=_messages("hi"), headers={ROUTE_HEADER: "go"})
    assert response.status_code == 502
    assert response.headers["x-llm-redact-upstream"] == "a"
    delivered = router.plans[0].delivered
    assert delivered is not None
    # The upstream did not fail: no fault class, no upstream error.
    assert delivered.failed == [] and delivered.finished == [502]
    assert state.upstream_errors == {} and state.bookkeeping_errors == {"delivery": 1}
    assert state.recent[-1]["status"] == 502 and state.recent[-1]["route"] is not None


@pytest.mark.parametrize("codec", ["sse", "ndjson"])
async def test_a_routed_stream_cut_by_the_proxy_is_a_stream_error(
    codec: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from fake_router import FakeDelivery
    from llm_redact.config import Config
    from llm_redact.providers import OllamaAdapter, OpenAIAdapter
    from llm_redact.proxy import (
        RequestMeta,
        _stream_rehydrated,
        _stream_rehydrated_ndjson,
        create_app,
    )

    app = create_app(Config())
    state = app.state.proxy
    ctx = state._static_context  # noqa: SLF001

    def unreadable(placeholder: str) -> str | None:
        raise ValueError("does not decrypt")

    monkeypatch.setattr(ctx.vault, "original_for", unreadable)
    adapter: Any
    if codec == "sse":
        adapter, gen_fn = OpenAIAdapter(), _stream_rehydrated
        delta = {"choices": [{"index": 0, "delta": {"content": f"hi {TOKEN}"}}]}
        chunk = b"data: " + json.dumps(delta).encode() + b"\n\n"
    else:
        adapter, gen_fn = OllamaAdapter(), _stream_rehydrated_ndjson
        chunk = json.dumps({"message": {"content": f"hi {TOKEN}"}, "done": True}).encode() + b"\n"
    route = FakeDelivery("a", "r", {})
    with pytest.raises(ValueError):
        async for _piece in gen_fn(
            _Body([chunk]),  # type: ignore[arg-type]
            adapter,
            state,
            ctx,
            request_meta=RequestMeta("POST", "/v1/x", time.perf_counter(), {}, {}),
            route=route,
        ):
            pass
    assert route.failed == ["stream_error"] and route.finished == [502]
    assert state.recent[-1]["status"] == 502
