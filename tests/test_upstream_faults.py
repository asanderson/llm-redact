"""Upstream-fault battery: the proxy sits in the live request path, so it
must fail closed and never emit a wrong value when the upstream refuses,
times out, 5xxs, or drops mid-body.

Two layers:

* generator level — the three streaming rehydrators (SSE / eventstream /
  ndjson) each run their `finally` even when the body raises mid-stream, so
  a dropped connection still closes the upstream and records the request, and
  the bytes emitted before the drop carry restored values, never a partial
  placeholder or a leaked token.
* handler level — a connect fault (no response body) and a mid-body drop on a
  buffered response both fail closed with a provider-shaped 502 and a recorded
  request, instead of leaking the upstream connection or a bare 500.

Faults are injected with fake transports/bodies — no real sockets, no sleeps.
"""

import json
import time
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig
from llm_redact.eventstream import (
    EventStreamMessage,
    string_header,
)
from llm_redact.eventstream import (
    serialize as serialize_eventstream,
)
from llm_redact.providers import BedrockAdapter, OllamaAdapter, OpenAIAdapter
from llm_redact.proxy import (
    RequestMeta,
    _stream_rehydrated,
    _stream_rehydrated_eventstream,
    _stream_rehydrated_ndjson,
    create_app,
)

EMAIL = "jane.doe@corp.example"


# --------------------------------------------------------------------------
# Generator level: the streaming finally finalizes even on a mid-stream drop.
# --------------------------------------------------------------------------


class _FaultyBody:
    """An httpx.Response stand-in whose body yields `chunks`, then raises
    `exc` — an upstream that streams a bit and then drops the connection."""

    def __init__(self, chunks: list[bytes], exc: BaseException, status_code: int = 200) -> None:
        self.status_code = status_code
        self._chunks = chunks
        self._exc = exc
        self.closed = False

    async def aiter_bytes(self) -> Any:
        for chunk in self._chunks:
            yield chunk
        raise self._exc

    async def aclose(self) -> None:
        self.closed = True


async def _drain(gen: Any) -> bytes:
    """Consume a rehydrating generator that will raise mid-stream; return the
    bytes emitted before the drop."""
    out = bytearray()
    with pytest.raises(httpx.TransportError):
        async for piece in gen:
            out += piece
    return bytes(out)


def _eventstream_delta(token: str) -> bytes:
    frame = EventStreamMessage(
        headers=[
            string_header(":message-type", "event"),
            string_header(":event-type", "contentBlockDelta"),
            string_header(":content-type", "application/json"),
        ],
        payload=json.dumps(
            {"contentBlockIndex": 0, "delta": {"text": f"hi {token}"}}, ensure_ascii=False
        ).encode(),
    )
    return serialize_eventstream(frame)


@pytest.mark.anyio
@pytest.mark.parametrize("codec", ["sse", "eventstream", "ndjson"])
async def test_stream_drop_finalizes_and_never_leaks(codec: str) -> None:
    app = create_app(Config())
    state = app.state.proxy
    ctx = state._static_context
    token = ctx.vault.placeholder_for("EMAIL", EMAIL)

    if codec == "sse":
        adapter: Any = OpenAIAdapter()
        chunk = (
            "data: "
            + json.dumps(
                {"choices": [{"index": 0, "delta": {"content": f"hello {token}"}}]},
                ensure_ascii=False,
            )
            + "\n\n"
        ).encode()
        gen_fn = _stream_rehydrated
    elif codec == "eventstream":
        adapter = BedrockAdapter()
        chunk = _eventstream_delta(token)
        gen_fn = _stream_rehydrated_eventstream
    else:
        adapter = OllamaAdapter()
        chunk = (
            json.dumps({"message": {"content": f"hi {token}"}, "done": False}, ensure_ascii=False)
            + "\n"
        ).encode()
        gen_fn = _stream_rehydrated_ndjson

    upstream = _FaultyBody([chunk], httpx.ReadError("upstream dropped mid-stream"))
    gen = gen_fn(
        upstream,  # type: ignore[arg-type]
        adapter,
        state,
        ctx,
        request_meta=RequestMeta("POST", "/v1/x", time.perf_counter(), {}, {}),
    )
    out = await _drain(gen)

    # never-wrong-value: the restored value rode the pre-drop bytes; the raw
    # placeholder token never appears in what the tool received.
    assert EMAIL.encode() in out
    assert token.encode() not in out
    # no leak: the upstream connection was closed even though the body raised.
    assert upstream.closed
    # finalized: exactly one streamed request row was recorded despite the drop.
    assert len(state.recent) == 1
    row = state.recent[0]
    assert row["streamed"] is True
    assert row["status"] == 200


@pytest.mark.anyio
async def test_stream_close_failure_still_finalizes() -> None:
    """If aclose() itself raises on an already-broken stream, the request is
    still recorded — the finally suppresses the close error, never the row."""
    app = create_app(Config())
    state = app.state.proxy
    ctx = state._static_context

    class _CloseExplodes(_FaultyBody):
        async def aclose(self) -> None:
            raise httpx.CloseError("connection already gone")

    upstream = _CloseExplodes([], httpx.ReadError("drop"))
    gen = _stream_rehydrated(
        upstream,  # type: ignore[arg-type]
        OpenAIAdapter(),
        state,
        ctx,
        request_meta=RequestMeta("POST", "/v1/x", time.perf_counter(), {}, {}),
    )
    await _drain(gen)
    assert len(state.recent) == 1


# --------------------------------------------------------------------------
# Handler level: connect fault and buffered mid-body drop fail closed with a
# recorded provider-shaped 502.
# --------------------------------------------------------------------------


class _ConnectFaultTransport(httpx.AsyncBaseTransport):
    """Refuses every connection — models an unreachable upstream."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)


class _MidBodyDropTransport(httpx.AsyncBaseTransport):
    """Sends response headers, then drops mid-body — models a truncated
    buffered response."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async def body() -> Any:
            yield b'{"partial":'
            raise httpx.ReadError("upstream dropped mid-body", request=request)

        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=body(), request=request
        )


def _openai_client(transport: httpx.AsyncBaseTransport) -> tuple[Any, httpx.AsyncClient]:
    config = Config(
        providers={**Config().providers, "openai": ProviderConfig(upstream_base_url="http://up")}
    )
    app = create_app(config, upstream_transport=transport)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    return app, client


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_connect_fault_fails_closed_502(stream: bool) -> None:
    app, client = _openai_client(_ConnectFaultTransport())
    body = {"model": "gpt-4o", "stream": stream, "messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 502
    # Provider-shaped error body, no partial/wrong content.
    assert "error" in response.json()
    # Recorded: the fault is visible in metrics/recent, not swallowed.
    recent = app.state.proxy.recent
    assert len(recent) == 1
    assert recent[0]["status"] == 502
    assert recent[0]["streamed"] is False
    # The fault is counted per provider and surfaced in /metrics and /status.
    assert app.state.proxy.upstream_errors["openai"] == 1
    metrics = (await client.get("/__llm-redact/metrics")).text
    assert 'llm_redact_upstream_errors_total{provider="openai"} 1' in metrics
    status = (await client.get("/__llm-redact/status")).json()
    assert status["upstream_errors_total"] == {"openai": 1}
    await client.aclose()


@pytest.mark.anyio
async def test_buffered_mid_body_drop_fails_closed_502() -> None:
    app, client = _openai_client(_MidBodyDropTransport())
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 502
    assert "error" in response.json()
    recent = app.state.proxy.recent
    assert len(recent) == 1
    assert recent[0]["status"] == 502
    await client.aclose()
