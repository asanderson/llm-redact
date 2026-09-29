"""A lone surrogate never fails a request or its answer.

JSON can carry a lone UTF-16 surrogate — as a ``\\ud800``-style escape, and
``json.loads`` of bytes also accepts one encoded in them — and such a code
point has no UTF-8 form. Re-serialized with ``ensure_ascii=False`` it came
out raw, and the encode after it (the request body, an SSE event, an NDJSON
line, an event-stream frame, a WebSocket frame) raised: an unrecorded bare
500 on the request path, a 502 or a broken stream on the answer path.
Every re-serialization now goes through ``jsonwalk.json_text`` /
``json_bytes``: the same JSON value, each lone surrogate written back as
its escape, every other character exactly as before.

The request and answer properties run through the real app (one app and
one event loop for all examples: building the app per example would
dominate the run); the streaming codecs are swept at every chunk split of
the upstream bytes, the convention of test_rehydrate/test_sse.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
import websockets
from hypothesis import given, settings
from hypothesis import strategies as st

from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from llm_redact.config import Config, ProviderConfig
from llm_redact.eventstream import EventStreamMessage, EventStreamParser, string_header
from llm_redact.eventstream import serialize as serialize_eventstream
from llm_redact.proxy import create_app
from llm_redact.realtime import GeminiLiveWs, OpenAIRealtimeWs
from llm_redact.rehydrate import RehydratorPool
from llm_redact.sse import SSEParser
from test_object_access_seams import ScriptedRouter
from test_object_access_seams import _registry as listing_registry
from test_realtime_relay import _relay_setup

EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"  # the first email the static session sees
HIGH = "\ud800"  # a high surrogate with no low one after it
LOW = "\udfff"  # a low surrogate with no high one before it
BEDROCK = "https://bedrock-runtime.us-east-1.amazonaws.com"
CONFIG = Config(providers={**Config().providers, "bedrock": ProviderConfig(BEDROCK)})

# Text no detector matches (no digits, no ASCII punctuation, no uppercase),
# with lone surrogates, non-ASCII letters, symbols and emoji mixed in.
_TEXT = st.text(
    alphabet=st.one_of(
        st.characters(categories=("Ll", "Lo", "So", "Zs")),
        st.integers(0xD800, 0xDFFF).map(chr),
    ),
    max_size=10,
)


def _json_reading(value: Any) -> Any:
    """``value`` as JSON reads it back: an escaped high+low surrogate pair
    is one character (RFC 8259), whatever code points the tree held."""
    return json.loads(json.dumps(value))


def _replace(value: Any, old: str, new: str) -> Any:
    """``value`` with ``old`` replaced in every string VALUE (keys kept)."""
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, old, new) for key, item in value.items()}
    return value


class _Upstream(httpx.AsyncBaseTransport):
    """The fake provider: records every request and answers with ``reply``
    — (content type, chunks) — streamed chunk by chunk (ASGITransport on
    the upstream side would join them)."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.reply: tuple[str, list[bytes]] = ("application/json", [b"{}"])

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        content_type, chunks = self.reply

        async def body() -> Any:
            for chunk in chunks:
                yield chunk

        return httpx.Response(
            200, headers={"content-type": content_type}, content=body(), request=request
        )


class _Proxy:
    """One proxy app and one event loop, shared by many examples."""

    def __init__(self, config: Config = CONFIG) -> None:
        self.loop = asyncio.new_event_loop()
        self.upstream = _Upstream()
        self.app = create_app(config, upstream_transport=self.upstream)

    def send(
        self,
        method: str,
        path: str,
        *,
        content: bytes = b"",
        reply: tuple[str, list[bytes]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        if reply is not None:
            self.upstream.reply = reply

        async def go() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app), base_url="http://127.0.0.1"
            ) as client:
                return await client.request(
                    method,
                    path,
                    content=content,
                    headers={"content-type": "application/json", **(headers or {})},
                )

        return self.loop.run_until_complete(go())

    def issue_token(self) -> None:
        """Make «EMAIL_001» the static session's token for EMAIL."""
        body = {"model": "m", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]}
        response = self.send("POST", "/v1/chat/completions", content=json.dumps(body).encode())
        assert response.status_code == 200


@pytest.fixture(scope="module")
def proxy() -> Iterator[_Proxy]:
    app = _Proxy()
    app.issue_token()
    yield app
    app.loop.close()


# --- the serializer ---------------------------------------------------------------


def test_json_text_escapes_exactly_the_lone_surrogates() -> None:
    from llm_redact.jsonwalk import json_bytes, json_text

    value = {"k\udc80": [f"a{HIGH}b", "«é» 中", "\U0001f600", '\n"'], "n": 1.5, "t": True}
    text = json_text(value)
    assert text == '{"k\\udc80": ["a\\ud800b", "«é» 中", "😀", "\\n\\""], "n": 1.5, "t": true}'
    assert json_bytes(value) == text.encode("utf-8")
    assert json.loads(text) == value


_ANY_TEXT = st.text(
    alphabet=st.one_of(st.characters(), st.integers(0xD800, 0xDFFF).map(chr)), max_size=12
)
_ANY_JSON = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | _ANY_TEXT,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(_ANY_TEXT, children, max_size=4)
    ),
    max_leaves=12,
)


@settings(deadline=None)
@given(value=_ANY_JSON)
def test_json_text_is_always_utf8_and_the_same_json_value(value: Any) -> None:
    from llm_redact.jsonwalk import json_bytes, json_text

    text = json_text(value)
    raw = json_bytes(value)
    assert raw == text.encode("utf-8")  # never raises
    assert json.loads(raw) == _json_reading(value)
    plain = json.dumps(value, ensure_ascii=False)
    try:
        plain.encode("utf-8")
    except UnicodeEncodeError:
        return
    # Every value without a lone surrogate keeps the historical form, byte
    # for byte.
    assert text == plain


# --- request bodies through the app -----------------------------------------------


def _b64_json(value: Any) -> str:
    return base64.b64encode(json.dumps(value).encode()).decode("ascii")


# route -> (path, build a body around the user text, read that text back)
_ROUTES: dict[str, tuple[str, Callable[[str], Any], Callable[[Any], str]]] = {
    "openai-chat": (
        "/v1/chat/completions",
        lambda t: {"model": "m", "messages": [{"role": "user", "content": t}]},
        lambda b: b["messages"][-1]["content"],
    ),
    "anthropic": (
        "/v1/messages",
        lambda t: {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": t}]},
        lambda b: b["messages"][-1]["content"],
    ),
    "gemini": (
        "/v1beta/models/m:generateContent",
        lambda t: {"contents": [{"role": "user", "parts": [{"text": t}]}]},
        lambda b: b["contents"][-1]["parts"][0]["text"],
    ),
    "ollama": (
        "/api/chat",
        lambda t: {"model": "m", "stream": False, "messages": [{"role": "user", "content": t}]},
        lambda b: b["messages"][-1]["content"],
    ),
    "cohere": (
        "/v2/chat",
        lambda t: {"model": "m", "messages": [{"role": "user", "content": t}]},
        lambda b: b["messages"][-1]["content"],
    ),
    "responses": (
        "/v1/responses",
        lambda t: {"model": "m", "input": t},
        lambda b: b["input"],
    ),
    "embeddings": (
        "/v1/embeddings",
        lambda t: {"model": "m", "input": t},
        lambda b: b["input"],
    ),
    "bedrock-converse": (
        "/model/m/converse",
        lambda t: {"messages": [{"role": "user", "content": [{"text": t}]}]},
        lambda b: b["messages"][-1]["content"][0]["text"],
    ),
    # The base64 invoke body inside a count-tokens envelope is decoded,
    # redacted and re-encoded (bedrock.py).
    "bedrock-count-tokens": (
        "/model/m/count-tokens",
        lambda t: {
            "input": {
                "invokeModel": {
                    "body": _b64_json(
                        {
                            "anthropic_version": "bedrock-2023-05-31",
                            "messages": [{"role": "user", "content": t}],
                        }
                    )
                }
            }
        },
        lambda b: json.loads(base64.b64decode(b["input"]["invokeModel"]["body"]))["messages"][-1][
            "content"
        ],
    ),
}


@pytest.mark.parametrize("route", sorted(_ROUTES))
@settings(deadline=None, max_examples=25)
@given(before=_TEXT, after=_TEXT, raw=st.booleans())
def test_a_request_holding_lone_surrogates_is_redacted_and_sent(
    proxy: _Proxy, route: str, before: str, after: str, raw: bool
) -> None:
    path, build, read = _ROUTES[route]
    body = build(f"{before}{HIGH} {EMAIL} {LOW}{after}")
    # As JSON escapes, or as surrogate-encoded bytes (json.loads accepts both).
    content = (
        json.dumps(body, ensure_ascii=False).encode("utf-8", "surrogatepass")
        if raw
        else json.dumps(body).encode()
    )
    sent_before = len(proxy.upstream.requests)
    response = proxy.send("POST", path, content=content)
    assert response.status_code == 200
    assert len(proxy.upstream.requests) == sent_before + 1
    sent = proxy.upstream.requests[-1].content
    sent.decode("utf-8")  # valid UTF-8, whichever form the client used
    assert EMAIL.encode() not in sent
    # The same JSON value, redaction applied: the text the proxy read, with
    # the email's token in its place.
    assert read(json.loads(sent)) == read(_json_reading(json.loads(content))).replace(EMAIL, TOKEN)


def test_a_request_with_nothing_to_redact_is_still_forwarded_byte_identical(
    proxy: _Proxy,
) -> None:
    content = b'{"model": "m", "messages": [{"role": "user", "content": "\\ud800 \xed\xa0\x80"}]}'
    response = proxy.send("POST", "/v1/chat/completions", content=content)
    assert response.status_code == 200
    assert proxy.upstream.requests[-1].content == content


def test_a_repeated_key_holding_a_lone_surrogate_is_resent_on_a_detection_off_route() -> None:
    # detection = false forwards the body unredacted, but a repeated key is
    # re-serialized (the ownership check read its LAST occurrence).
    config = Config(
        providers={
            **Config().providers,
            "openai": ProviderConfig("https://api.openai.com", detection=False),
        }
    )
    app = _Proxy(config)
    try:
        content = (
            b'{"messages": [], "messages": [{"role": "user", "content": "\\ud800 '
            + EMAIL.encode()
            + b'"}]}'
        )
        response = app.send("POST", "/v1/chat/completions", content=content)
    finally:
        app.loop.close()
    assert response.status_code == 200
    sent = app.upstream.requests[-1].content
    assert json.loads(sent) == {"messages": [{"role": "user", "content": f"{HIGH} {EMAIL}"}]}


# --- buffered answers through the app ---------------------------------------------


@pytest.mark.parametrize(
    "route", ["openai-chat", "anthropic", "gemini", "ollama", "cohere", "responses"]
)
@settings(deadline=None, max_examples=25)
@given(before=_TEXT, after=_TEXT)
def test_a_buffered_answer_holding_lone_surrogates_is_restored(
    proxy: _Proxy, route: str, before: str, after: str
) -> None:
    path, build, _ = _ROUTES[route]
    answer = {
        "reply": f"{before}{HIGH} {TOKEN} {LOW}{after}",
        "more": [LOW, {"k\udc00": TOKEN}],
    }
    upstream_bytes = json.dumps(answer).encode()
    response = proxy.send(
        "POST",
        path,
        content=json.dumps(build(f"mail {EMAIL}")).encode(),
        reply=("application/json", [upstream_bytes]),
    )
    assert response.status_code == 200
    assert response.json() == _replace(json.loads(upstream_bytes), TOKEN, EMAIL)


def test_a_stored_file_download_holding_lone_surrogates_is_restored(proxy: _Proxy) -> None:
    lines = [
        {"custom_id": "a", "response": {"body": {"text": f"{HIGH} {TOKEN}"}}},
        {"custom_id": f"b{LOW}", "response": {"body": {"text": "none"}}},
    ]
    upstream_bytes = b"\n".join(json.dumps(line).encode() for line in lines) + b"\n"
    response = proxy.send(
        "GET", "/v1/files/file-1/content", reply=("application/octet-stream", [upstream_bytes])
    )
    assert response.status_code == 200
    out = [json.loads(line) for line in response.content.splitlines() if line]
    assert out == _replace(lines, TOKEN, EMAIL)


# --- streaming answers: every chunk split of the upstream bytes --------------------

_PARTS = (f"{HIGH}a «EMA", f"IL_001» {LOW}中")
_RESTORED = f"{HIGH}a {EMAIL} {LOW}中"


def _sse(events: list[tuple[str | None, Any]]) -> bytes:
    out = b""
    for name, payload in events:
        if name is not None:
            out += f"event: {name}\n".encode()
        out += b"data: " + json.dumps(payload).encode() + b"\n\n"
    return out


def _sse_payloads(content: bytes) -> list[tuple[str | None, Any]]:
    parser = SSEParser()
    events = [*parser.feed(content), *parser.close()]
    return [(e.event, json.loads(e.data)) for e in events if e.data and e.data != "[DONE]"]


def _openai_stream() -> bytes:
    chunks: list[tuple[str | None, Any]] = [
        (None, {"choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}]})
        for part in _PARTS
    ]
    chunks.append((None, {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    return _sse(chunks) + b"data: [DONE]\n\n"


def _openai_text(content: bytes) -> str:
    return "".join(
        choice["delta"].get("content") or ""
        for _, payload in _sse_payloads(content)
        for choice in payload["choices"]
    )


def _anthropic_stream() -> bytes:
    events: list[tuple[str | None, Any]] = [
        ("message_start", {"type": "message_start", "message": {"id": "msg_1"}}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        ),
    ]
    events += [
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": p}},
        )
        for p in _PARTS
    ]
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return _sse(events)


def _anthropic_text(content: bytes) -> str:
    return "".join(
        payload["delta"]["text"]
        for name, payload in _sse_payloads(content)
        if name == "content_block_delta"
    )


def _responses_stream() -> bytes:
    item = {"item_id": "i", "output_index": 0, "content_index": 0}
    full = "".join(_PARTS)
    events: list[tuple[str | None, Any]] = [
        ("response.created", {"type": "response.created", "response": {"id": "r"}}),
    ]
    events += [
        ("response.output_text.delta", {"type": "response.output_text.delta", **item, "delta": p})
        for p in _PARTS
    ]
    events += [
        ("response.output_text.done", {"type": "response.output_text.done", **item, "text": full}),
        (
            "response.completed",
            {
                "type": "response.completed",
                "response": {"id": "r", "output": [{"content": [{"text": full}]}]},
            },
        ),
    ]
    return _sse(events)


def _responses_text(content: bytes) -> str:
    payloads = _sse_payloads(content)
    done = [p["text"] for name, p in payloads if name == "response.output_text.done"]
    completed = [
        p["response"]["output"][0]["content"][0]["text"]
        for name, p in payloads
        if name == "response.completed"
    ]
    assert done == completed == [_RESTORED]  # the repeated full values, restored
    return "".join(p["delta"] for name, p in payloads if name == "response.output_text.delta")


def _gemini_stream() -> bytes:
    return _sse(
        [
            (None, {"candidates": [{"index": 0, "content": {"parts": [{"text": _PARTS[0]}]}}]}),
            (
                None,
                {
                    "candidates": [
                        {
                            "index": 0,
                            "content": {"parts": [{"text": _PARTS[1]}]},
                            "finishReason": "STOP",
                        }
                    ]
                },
            ),
        ]
    )


def _gemini_text(content: bytes) -> str:
    return "".join(
        part.get("text", "")
        for _, payload in _sse_payloads(content)
        for candidate in payload["candidates"]
        for part in candidate["content"]["parts"]
    )


def _cohere_stream() -> bytes:
    events: list[tuple[str | None, Any]] = [
        (
            "content-delta",
            {"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": p}}}},
        )
        for p in _PARTS
    ]
    events += [
        ("content-end", {"type": "content-end", "index": 0}),
        ("message-end", {"type": "message-end"}),
    ]
    return _sse(events)


def _cohere_text(content: bytes) -> str:
    return "".join(
        payload["delta"]["message"]["content"]["text"]
        for _, payload in _sse_payloads(content)
        if payload.get("type") == "content-delta"
    )


def _ollama_stream() -> bytes:
    lines = [{"model": "m", "message": {"role": "assistant", "content": p}} for p in _PARTS]
    lines.append({"model": "m", "message": {"role": "assistant", "content": ""}, "done": True})
    return b"".join(json.dumps(line).encode() + b"\n" for line in lines)


def _ollama_text(content: bytes) -> str:
    return "".join(json.loads(line)["message"]["content"] for line in content.splitlines())


def _generate_stream() -> bytes:
    lines = [{"model": "m", "response": p} for p in _PARTS]
    lines.append({"model": "m", "response": "", "done": True})
    return b"".join(json.dumps(line).encode() + b"\n" for line in lines)


def _generate_text(content: bytes) -> str:
    return "".join(json.loads(line)["response"] for line in content.splitlines())


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


def _eventstream_payloads(content: bytes) -> list[tuple[str | None, Any]]:
    return [
        (message.event_type, json.loads(message.payload))
        for message in EventStreamParser().feed(content)
    ]


def _converse_stream() -> bytes:
    frames = [
        _frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": p}}) for p in _PARTS
    ]
    frames.append(_frame("contentBlockStop", {"contentBlockIndex": 0}))
    frames.append(_frame("messageStop", {"stopReason": "end_turn"}))
    return b"".join(frames)


def _converse_text(content: bytes) -> str:
    return "".join(
        payload["delta"]["text"]
        for name, payload in _eventstream_payloads(content)
        if name == "contentBlockDelta"
    )


def _invoke_stream() -> bytes:
    def chunk(inner: Any) -> bytes:
        return _frame("chunk", {"bytes": _b64_json(inner)})

    frames = [
        chunk(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": p}}
        )
        for p in _PARTS
    ]
    frames.append(chunk({"type": "content_block_stop", "index": 0}))
    return b"".join(frames)


def _invoke_text(content: bytes) -> str:
    inner = [
        json.loads(base64.b64decode(payload["bytes"]))
        for _, payload in _eventstream_payloads(content)
    ]
    return "".join(event["delta"]["text"] for event in inner if "delta" in event)


# name -> (method, path, content type, upstream bytes, client text reader)
_STREAMS: dict[str, tuple[str, str, str, bytes, Callable[[bytes], str]]] = {
    "openai-chat": (
        "POST",
        "/v1/chat/completions",
        "text/event-stream",
        _openai_stream(),
        _openai_text,
    ),
    "anthropic": (
        "POST",
        "/v1/messages",
        "text/event-stream",
        _anthropic_stream(),
        _anthropic_text,
    ),
    "responses": (
        "POST",
        "/v1/responses",
        "text/event-stream",
        _responses_stream(),
        _responses_text,
    ),
    "gemini": (
        "POST",
        "/v1beta/models/m:streamGenerateContent?alt=sse",
        "text/event-stream",
        _gemini_stream(),
        _gemini_text,
    ),
    "cohere": ("POST", "/v2/chat", "text/event-stream", _cohere_stream(), _cohere_text),
    "ollama-chat": ("POST", "/api/chat", "application/x-ndjson", _ollama_stream(), _ollama_text),
    "ollama-generate": (
        "POST",
        "/api/generate",
        "application/x-ndjson",
        _generate_stream(),
        _generate_text,
    ),
    "bedrock-converse": (
        "POST",
        "/model/m/converse-stream",
        "application/vnd.amazon.eventstream",
        _converse_stream(),
        _converse_text,
    ),
    "bedrock-invoke": (
        "POST",
        "/model/m/invoke-with-response-stream",
        "application/vnd.amazon.eventstream",
        _invoke_stream(),
        _invoke_text,
    ),
}


@pytest.mark.parametrize("name", sorted(_STREAMS))
def test_a_stream_holding_lone_surrogates_is_restored_at_every_split(
    proxy: _Proxy, name: str
) -> None:
    method, path, content_type, stream, read = _STREAMS[name]
    # The request body carries no text of its own: the answer's token is the
    # one the module fixture issued.
    request = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    for split in range(len(stream) + 1):
        response = proxy.send(
            method,
            path,
            content=request.encode(),
            reply=(content_type, [stream[:split], stream[split:]]),
        )
        assert response.status_code == 200, split
        assert read(response.content) == _RESTORED, split


def test_batch_results_holding_lone_surrogates_are_restored(proxy: _Proxy) -> None:
    lines = [
        {
            "custom_id": f"a{HIGH}",
            "result": {"type": "succeeded", "message": {"content": [{"text": f"{TOKEN} {LOW}"}]}},
        },
        {"custom_id": "b", "result": {"type": "errored"}},
    ]
    stream = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
    for split in range(len(stream) + 1):
        response = proxy.send(
            "GET",
            "/v1/messages/batches/msgbatch_1/results",
            headers={"anthropic-version": "2023-06-01"},
            reply=("application/x-jsonl", [stream[:split], stream[split:]]),
        )
        assert response.status_code == 200, split
        out = [json.loads(line) for line in response.content.splitlines() if line]
        assert out == _replace(lines, TOKEN, EMAIL), split


# --- routed answers and listings ----------------------------------------------------


async def test_a_routed_answer_the_router_rewrote_keeps_its_lone_surrogates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # REDACT_ONLY (embeddings): nothing to restore, but the router observes
    # the payload and rewrites it — the core re-serializes what it rewrote.
    router = FakeRouter(
        {"r": [Hop("a", "http://a.example/v1/embeddings"), Stop()]},
        plan_kwargs={"r": {"mutate": True}},
    )
    install(monkeypatch, router)
    upstream = _Upstream()
    answer = {"object": "list", "data": [{"embedding": [0.5]}], "note": f"{HIGH} x"}
    upstream.reply = ("application/json", [json.dumps(answer).encode()])
    app = create_app(routed_config(), upstream_transport=upstream)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/embeddings",
            json={"model": "m", "input": f"mail {EMAIL}"},
            headers={ROUTE_HEADER: "r"},
        )
    assert response.status_code == 200
    assert response.json() == {**answer, "observed": True}


async def test_a_listing_restored_in_its_owners_session_keeps_its_lone_surrogates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ScriptedRouter(session="read-empty", owners={"file-own": "user:n1:main"})
    listing_registry(monkeypatch, router)
    listing = {
        "object": "list",
        "data": [{"id": "file-own", "object": "file", "filename": f"notes {TOKEN} {HIGH}"}],
        "first_id": LOW,
    }
    upstream = _Upstream()
    upstream.reply = ("application/json", [json.dumps(listing).encode()])
    app = create_app(Config(), upstream_transport=upstream)
    manager = app.state.proxy.vault_manager
    manager.get("user:n1:main").placeholder_for("EMAIL", EMAIL)
    manager.record_response_session("file-own", "user:n1:main")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get("/v1/files")
    assert response.status_code == 200
    assert response.json() == _replace(listing, TOKEN, EMAIL)


# --- realtime frames ------------------------------------------------------------------


class _Ctx:
    def __init__(self, proxy: _Proxy) -> None:
        self.redactor = proxy.app.state.proxy._static_context.redactor


# adapter -> (a client frame around a text, read it back, a server frame
# around a text, the text the client receives across the returned frames)
_REALTIME: dict[
    str,
    tuple[
        Any,
        Callable[[str], Any],
        Callable[[Any], str],
        Callable[[str], Any],
        Callable[[list[Any]], str],
    ],
] = {
    "openai": (
        OpenAIRealtimeWs(),
        lambda t: {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": t}],
            },
            "k\udc00": LOW,
        },
        lambda p: p["item"]["content"][0]["text"],
        lambda t: {
            "type": "conversation.item.created",
            "item": {"id": "i", "content": [{"type": "input_text", "text": t}]},
            "k\udc00": LOW,
        },
        lambda ps: "".join(p["item"]["content"][0]["text"] for p in ps),
    ),
    "gemini": (
        GeminiLiveWs(),
        lambda t: {
            "clientContent": {"turns": [{"role": "user", "parts": [{"text": t}]}]},
            "k\udc00": LOW,
        },
        lambda p: p["clientContent"]["turns"][0]["parts"][0]["text"],
        lambda t: {
            "serverContent": {"modelTurn": {"parts": [{"text": t}]}, "turnComplete": True},
            "k\udc00": LOW,
        },
        lambda ps: "".join(
            part["text"]
            for p in ps
            for part in p["serverContent"].get("modelTurn", {}).get("parts", [])
        ),
    ),
}


@pytest.mark.parametrize("name", sorted(_REALTIME))
@pytest.mark.parametrize("binary", [False, True], ids=["text", "binary"])
@settings(deadline=None, max_examples=25)
@given(before=_TEXT, after=_TEXT)
def test_a_realtime_frame_holding_lone_surrogates_is_redacted_and_restored(
    proxy: _Proxy, name: str, binary: bool, before: str, after: str
) -> None:
    adapter, client_frame, read_client, server_frame, read_server = _REALTIME[name]

    def frame(value: Any) -> str | bytes:
        text = json.dumps(value)
        return text.encode() if binary else text

    def payloads(frames: list[str | bytes]) -> list[Any]:
        assert frames and all(isinstance(f, bytes) is binary for f in frames)
        # A text frame goes out as UTF-8: encoding it must never fail.
        return [json.loads(f if isinstance(f, bytes) else f.encode("utf-8")) for f in frames]

    text = f"{before}{HIGH} {EMAIL} {LOW}{after}"
    [redacted] = payloads([adapter.redact_message(frame(client_frame(text)), _Ctx(proxy))])
    assert read_client(redacted) == _json_reading(text).replace(EMAIL, TOKEN)
    assert redacted["k\udc00"] == LOW
    vault = proxy.app.state.proxy._static_context.vault
    echoed = frame(server_frame(read_client(redacted)))
    restored = payloads(adapter.rehydrate_message(echoed, RehydratorPool(vault)))
    assert read_server(restored) == _json_reading(text)


async def test_the_relay_carries_lone_surrogates_both_ways() -> None:
    # The fake upstream echoes every frame verbatim; an item event is what the
    # inbound side restores whole, so the client sends one (outbound, every
    # client event is walked whatever its type).
    frame = json.dumps(
        {
            "type": "conversation.item.created",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"{HIGH} mail {EMAIL}"}],
            },
        }
    )
    async with (
        _relay_setup() as (fake, proxy_host),
        websockets.connect(f"ws://{proxy_host}/v1/realtime") as client,
    ):
        await client.send(frame)
        echoed = json.loads(await client.recv())  # the upstream echoes the redacted frame
    [sent] = fake.received
    assert isinstance(sent, str)
    upstream_text = json.loads(sent)["item"]["content"][0]["text"]
    assert upstream_text == f"{HIGH} mail {TOKEN}"
    assert echoed["item"]["content"][0]["text"] == f"{HIGH} mail {EMAIL}"
