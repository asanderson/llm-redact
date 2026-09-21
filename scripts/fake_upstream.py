"""Runnable fake LLM upstream for manual end-to-end verification — and the
programmable upstream the routing walkthrough tests drive in-process.

Serves Anthropic-shaped /v1/messages (+ count_tokens), OpenAI-shaped
/v1/chat/completions (streaming and non-streaming, with usage blocks),
Bedrock-shaped /model/{id}/converse[-stream] (real binary event-stream
frames), Ollama's native /api/chat + /api/generate, the batch/files
endpoints and a minimal Realtime WebSocket. It logs every request body it
receives — so you can verify that only «TYPE_NNN» placeholders arrive — and
echoes any placeholders back in its reply, split across SSE chunks (or
binary frames) to exercise the proxy's streaming rehydration.

SCENARIOS (docs/routing.md): ``build_app(scenarios)`` takes a per-upstream
table keyed by the request's ``Host`` (or an ``x-fake-upstream`` header) —
one app therefore serves ``http://oauth``, ``http://key`` and
``http://ollama`` at once through a single ``httpx.ASGITransport`` — and
each :class:`Scenario` programs that upstream's behaviour: ``status``
(a non-2xx answered instead of a reply), ``plan_limit`` (the answer carries
Anthropic's unified rate-limit headers marked ``rejected``), ``retry_after``,
``usage`` (always emitted in JSON bodies, as ``message_start``/
``message_delta`` usage on Anthropic SSE, and as OpenAI's final chunk when
``stream_options.include_usage`` is set), ``abort_mid_stream`` (the SSE
body ends after the first content event, and the response carries
``x-fake-abort-after-first-event: 1`` so a wrapping transport can turn that
into a real connection drop — ASGITransport itself buffers whole bodies),
``echo_model`` (the reply's ``model`` echoes the request's, so a proxy
model rewrite/restore is testable) and ``fail_count`` (the first N requests
fail with ``status``, later ones succeed). Every request an upstream saw is
appended to its ``Scenario.received`` (headers + decoded body) for
assertions — received bodies are printed only by the CLI (``--quiet`` off).

--mangle makes the echo imitate an LLM that rewrites placeholders
(lowercased, hyphens, stripped zero-padding): with [rehydration] fuzzy = true
the proxy still restores them; with fuzzy = false they come back verbatim.

Usage:
    uv run python scripts/fake_upstream.py --port 9999 [--mangle]
        [--status 429 --plan-limit --retry-after 5 --abort-mid-stream
         --echo-model --fail-count 2]
"""

import argparse
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute

from llm_redact.eventstream import EventStreamMessage, serialize, string_header

MANGLE = False
VERBOSE = False

# Anthropic's unified rate-limit headers a plan-limit 429 carries (the
# proxy's DEFAULT_PLAN_LIMIT_HEADERS): every one marked rejected here.
PLAN_LIMIT_HEADERS = {
    "anthropic-ratelimit-unified-status": "rejected",
    "anthropic-ratelimit-unified-5h-status": "rejected",
    "anthropic-ratelimit-unified-7d-status": "rejected",
    "anthropic-ratelimit-unified-representative-claim": "five_hour",
}
ABORT_HEADER = "x-fake-abort-after-first-event"
SCENARIO_HEADER = "x-fake-upstream"
TOKEN_RE = re.compile("«[A-Z0-9_]+»")


@dataclass
class Received:
    """One request as the fake upstream saw it."""

    path: str
    headers: dict[str, str]
    body: Any


@dataclass
class Scenario:
    """Programmable behaviour of one fake upstream (see the module docstring)."""

    status: int = 200
    plan_limit: bool = False
    retry_after: float | None = None
    # Token counts, rendered in each protocol's own usage shape.
    usage: dict[str, int] = field(
        default_factory=lambda: {"input": 10, "output": 5, "cache_read": 0, "cache_write": 0}
    )
    abort_mid_stream: bool = False
    echo_model: bool = False
    fail_count: int = 0
    received: list[Received] = field(default_factory=list)

    def failing(self) -> bool:
        """Whether the request just recorded should be answered with
        ``status``: always when it is not a success status, and for the
        first ``fail_count`` requests otherwise."""
        seen = len(self.received)
        if self.fail_count and seen <= self.fail_count:
            return True
        return not 200 <= self.status < 300

    def error_status(self) -> int:
        return self.status if not 200 <= self.status < 300 else 503

    def error_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.plan_limit:
            headers.update(PLAN_LIMIT_HEADERS)
        if self.retry_after is not None:
            headers["retry-after"] = str(int(self.retry_after))
        return headers

    def anthropic_usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.usage["input"],
            "output_tokens": self.usage["output"],
            "cache_creation_input_tokens": self.usage.get("cache_write", 0),
            "cache_read_input_tokens": self.usage.get("cache_read", 0),
        }

    def openai_usage(self) -> dict[str, Any]:
        cached = self.usage.get("cache_read", 0)
        return {
            "prompt_tokens": self.usage["input"] + cached,
            "completion_tokens": self.usage["output"],
            "total_tokens": self.usage["input"] + cached + self.usage["output"],
            "prompt_tokens_details": {"cached_tokens": cached},
        }


Scenarios = dict[str, Scenario]


def _log(message: str) -> None:
    if VERBOSE:
        print(message)


def _mangle(token: str) -> str:
    # «EMAIL_001» -> «email-1»
    body = token.strip("«»").lower().replace("_", "-")
    body = re.sub(r"-0+(\d)", r"-\1", body)
    return f"«{body}»"


def _echo(flat: str) -> str:
    tokens = TOKEN_RE.findall(flat)
    if MANGLE:
        tokens = [_mangle(t) for t in tokens]
    echoed = " and ".join(tokens) if tokens else "no placeholders"
    return f"Upstream saw {echoed} in your message."


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode()
    return f"event: {event}\ndata: ".encode() + data + b"\n\n"


class _ScenarioApp:
    """The scenario-aware handlers, bound to one scenario table."""

    def __init__(self, scenarios: Scenarios) -> None:
        self.scenarios = scenarios

    def scenario_for(self, request: Request, body: Any) -> Scenario:
        """The scenario for this request's upstream identity (the
        ``x-fake-upstream`` header, else ``Host`` with and without its port,
        else the ``*`` wildcard); an unknown upstream gets a fresh default
        scenario registered under its host so its traffic is inspectable."""
        host = request.headers.get("host", "")
        keys = [request.headers.get(SCENARIO_HEADER, ""), host, host.rsplit(":", 1)[0], "*"]
        scenario: Scenario | None = None
        for key in keys:
            if key and key in self.scenarios:
                scenario = self.scenarios[key]
                break
        if scenario is None:
            scenario = self.scenarios.setdefault(host or "*", Scenario())
        scenario.received.append(Received(request.url.path, dict(request.headers), body))
        _log(f"--- request body received by upstream {host} ({request.url.path}) ---")
        _log(json.dumps(body, indent=2, ensure_ascii=False))
        return scenario

    async def messages(self, request: Request) -> Response:
        body = await request.json()
        sc = self.scenario_for(request, body)
        if sc.failing():
            status = sc.error_status()
            error_type = "rate_limit_error" if status == 429 else "api_error"
            return JSONResponse(
                {"type": "error", "error": {"type": error_type, "message": "fake upstream"}},
                status_code=status,
                headers=sc.error_headers(),
            )
        reply = _echo(json.dumps(body.get("messages", []), ensure_ascii=False))
        model = body.get("model", "fake") if sc.echo_model else "fake"
        usage = sc.anthropic_usage()
        # Headers a real Anthropic answer carries; the proxy relays them.
        headers = {"anthropic-ratelimit-unified-status": "allowed", "request-id": "req_fake"}

        if not body.get("stream"):
            return JSONResponse(
                {
                    "id": "msg_fake",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": reply}],
                    "stop_reason": "end_turn",
                    "usage": usage,
                },
                headers=headers,
            )

        async def stream() -> AsyncIterator[bytes]:
            yield _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_fake",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "usage": {k: v for k, v in usage.items() if k != "output_tokens"},
                    },
                },
            )
            # A keep-alive comment, relayed verbatim by the proxy (R-13).
            yield b": ping\n\n"
            yield _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            )
            # Deliberately cut the reply into 7-char chunks so placeholders are
            # split across events; the proxy must reassemble them.
            for i in range(0, len(reply), 7):
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": reply[i : i + 7]},
                    },
                )
                if sc.abort_mid_stream:
                    # Truncated after the first content event; a wrapping
                    # transport turns the marker header into a real drop.
                    return
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": usage["output_tokens"]},
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})

        if sc.abort_mid_stream:
            headers[ABORT_HEADER] = "1"
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    async def count_tokens(self, request: Request) -> Response:
        body = await request.json()
        sc = self.scenario_for(request, body)
        if sc.failing():
            return JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": "fake upstream"}},
                status_code=sc.error_status(),
                headers=sc.error_headers(),
            )
        return JSONResponse({"input_tokens": sc.usage["input"]})

    async def chat_completions(self, request: Request) -> Response:
        body = await request.json()
        sc = self.scenario_for(request, body)
        if sc.failing():
            status = sc.error_status()
            return JSONResponse(
                {
                    "error": {
                        "message": "fake upstream",
                        "type": "rate_limit_error" if status == 429 else "server_error",
                        "code": None,
                    }
                },
                status_code=status,
                headers=sc.error_headers(),
            )
        reply = _echo(json.dumps(body.get("messages", []), ensure_ascii=False))
        model = body.get("model", "fake") if sc.echo_model else "fake"
        usage = sc.openai_usage()

        if not body.get("stream"):
            return JSONResponse(
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            )

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def chunk(delta: dict[str, Any], finish: str | None, extra: dict[str, Any]) -> bytes:
            payload = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **extra,
            }
            return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"

        async def stream() -> AsyncIterator[bytes]:
            yield chunk({"role": "assistant", "content": ""}, None, {})
            for i in range(0, len(reply), 7):
                yield chunk({"content": reply[i : i + 7]}, None, {})
                if sc.abort_mid_stream:
                    return
            yield chunk({}, "stop", {})
            if include_usage:
                yield (
                    b"data: "
                    + json.dumps(
                        {
                            "id": "chatcmpl-fake",
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [],
                            "usage": usage,
                        }
                    ).encode()
                    + b"\n\n"
                )
            yield b"data: [DONE]\n\n"

        headers = {ABORT_HEADER: "1"} if sc.abort_mid_stream else {}
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    async def gemini_generate(self, request: Request) -> Response:
        """Gemini-shaped generateContent: the model id comes from the PATH
        (models/{m}:generateContent) and is echoed as `modelVersion`."""
        body = await request.json()
        sc = self.scenario_for(request, body)
        if sc.failing():
            status = sc.error_status()
            return JSONResponse(
                {"error": {"code": status, "message": "fake upstream", "status": "UNAVAILABLE"}},
                status_code=status,
                headers=sc.error_headers(),
            )
        reply = _echo(json.dumps(body.get("contents", []), ensure_ascii=False))
        model = request.path_params.get("model", "fake") if sc.echo_model else "fake"
        return JSONResponse(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": reply}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": sc.usage["input"],
                    "candidatesTokenCount": sc.usage["output"],
                    "totalTokenCount": sc.usage["input"] + sc.usage["output"],
                },
                "modelVersion": model,
            }
        )

    async def ollama_chat(self, request: Request) -> Response:
        """Ollama's native /api/chat + /api/generate: NDJSON streams (7-char
        pieces split placeholders across lines) with the token counts on the
        done:true line; stream:false answers one JSON object."""
        body = await request.json()
        sc = self.scenario_for(request, body)
        if sc.failing():
            return JSONResponse(
                {"error": "fake upstream"},
                status_code=sc.error_status(),
                headers=sc.error_headers(),
            )
        reply = _echo(json.dumps(body.get("messages", body.get("prompt", "")), ensure_ascii=False))
        generate = request.url.path.endswith("/generate")
        model = body.get("model", "fake") if sc.echo_model else "fake"

        def chunk(text: str, done: bool) -> bytes:
            payload: dict[str, Any] = {"model": model, "done": done}
            if generate:
                payload["response"] = text
            else:
                payload["message"] = {"role": "assistant", "content": text}
            if done:
                payload["prompt_eval_count"] = sc.usage["input"]
                payload["eval_count"] = sc.usage["output"]
            return json.dumps(payload, ensure_ascii=False).encode() + b"\n"

        if body.get("stream") is False:
            return Response(chunk(reply, True), media_type="application/json")

        async def stream() -> AsyncIterator[bytes]:
            for i in range(0, len(reply), 7):
                yield chunk(reply[i : i + 7], False)
            yield chunk("", True)

        return StreamingResponse(stream(), media_type="application/x-ndjson")


def _bedrock_frame(event_type: str, payload: dict[str, Any]) -> bytes:
    return serialize(
        EventStreamMessage(
            headers=[
                string_header(":message-type", "event"),
                string_header(":event-type", event_type),
                string_header(":content-type", "application/json"),
            ],
            payload=json.dumps(payload, ensure_ascii=False).encode(),
        )
    )


async def bedrock_converse(request: Request) -> Response:
    body = await request.json()
    _log(f"--- request body received by upstream ({request.url.path}) ---")
    _log(json.dumps(body, indent=2, ensure_ascii=False))
    reply = _echo(json.dumps(body.get("messages", []), ensure_ascii=False))

    if request.url.path.endswith("/converse"):
        return JSONResponse(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": reply}]}},
                "stopReason": "end_turn",
            }
        )

    async def stream() -> AsyncIterator[bytes]:
        yield _bedrock_frame("messageStart", {"role": "assistant"})
        # 7-char text deltas split placeholders across frames; the proxy
        # must reassemble them from the binary stream.
        for i in range(0, len(reply), 7):
            yield _bedrock_frame(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"text": reply[i : i + 7]}},
            )
        yield _bedrock_frame("contentBlockStop", {"contentBlockIndex": 0})
        yield _bedrock_frame("messageStop", {"stopReason": "end_turn"})

    return StreamingResponse(stream(), media_type="application/vnd.amazon.eventstream")


async def realtime_ws(websocket: Any) -> None:
    """A minimal OpenAI-Realtime-shaped upstream: prints what it received
    (placeholders only, if the proxy did its job) and streams the text back
    as output_text deltas split mid-token, then a done pair."""
    await websocket.accept()
    while True:
        try:
            raw = await websocket.receive_text()
        except Exception:
            return
        event = json.loads(raw)
        _log(f"[fake realtime] <- {event.get('type')}: {json.dumps(event)[:200]}")
        if event.get("type") != "conversation.item.create":
            continue
        parts = event.get("item", {}).get("content", [])
        text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
        reply = f"Upstream saw {text} in your session."
        step = 7
        for i in range(0, len(reply), step):
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "response.output_text.delta",
                        "item_id": "item_fake",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": reply[i : i + step],
                    }
                )
            )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "response.output_text.done",
                    "item_id": "item_fake",
                    "output_index": 0,
                    "content_index": 0,
                    "text": reply,
                }
            )
        )
        await websocket.send_text(json.dumps({"type": "response.done", "response": {"id": "r1"}}))


_uploaded_files: dict[str, bytes] = {}
_created_batches: dict[str, dict[str, Any]] = {}


async def anthropic_batch_create(request: Request) -> Response:
    body = await request.json()
    flat = json.dumps(body, ensure_ascii=False)
    tokens = TOKEN_RE.findall(flat)
    _created_batches["msgbatch_fake"] = {"tokens": list(dict.fromkeys(tokens))}
    _log(f"[fake] batch create carrying tokens: {tokens}")
    return JSONResponse({"id": "msgbatch_fake", "processing_status": "ended"})


async def anthropic_batch_poll(request: Request) -> Response:
    return JSONResponse({"id": "msgbatch_fake", "processing_status": "ended"})


async def anthropic_batch_results(request: Request) -> Response:
    batch = _created_batches.get("msgbatch_fake", {"tokens": []})
    lines = b"".join(
        json.dumps(
            {
                "custom_id": f"req_{i}",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"echoing {token} back"}],
                    },
                },
            },
            ensure_ascii=False,
        ).encode()
        + b"\n"
        for i, token in enumerate(batch["tokens"])
    )
    return Response(content=lines, media_type="application/x-jsonl")


async def openai_file_upload(request: Request) -> Response:
    from llm_redact.multipart import parse as parse_multipart
    from llm_redact.multipart import parse_boundary

    raw = await request.body()
    boundary = parse_boundary(request.headers.get("content-type", ""))
    parsed = parse_multipart(raw, boundary) if boundary else None
    if parsed is not None:
        for part in parsed.parts:
            if part.filename is not None:
                _uploaded_files["file_fake"] = part.content
                _log(f"[fake] stored upload ({len(part.content)} bytes)")
    return JSONResponse({"id": "file_fake", "object": "file", "purpose": "batch"})


async def openai_file_content(request: Request) -> Response:
    return Response(
        content=_uploaded_files.get("file_fake", b""),
        media_type="application/octet-stream",
    )


async def openai_batch_create(request: Request) -> Response:
    return JSONResponse({"id": "batch_fake", "status": "validating"})


def build_app(scenarios: Scenarios | None = None) -> Starlette:
    """The fake upstream app. ``scenarios`` programs per-upstream behaviour
    (keyed by Host or ``x-fake-upstream``); None serves plain successes."""
    handlers = _ScenarioApp(scenarios if scenarios is not None else {})
    return Starlette(
        routes=[
            Route("/v1/messages", handlers.messages, methods=["POST"]),
            Route("/v1/messages/count_tokens", handlers.count_tokens, methods=["POST"]),
            Route("/v1/chat/completions", handlers.chat_completions, methods=["POST"]),
            # OpenAI-compatible surfaces under their own base paths (the
            # proxy folds the inbound /v1 into the upstream's base URL).
            Route("/v1beta/openai/chat/completions", handlers.chat_completions, methods=["POST"]),
            Route("/api/v1/chat/completions", handlers.chat_completions, methods=["POST"]),
            Route(
                "/v1beta/models/{model}:generateContent", handlers.gemini_generate, methods=["POST"]
            ),
            Route("/v1/messages/batches", anthropic_batch_create, methods=["POST"]),
            Route("/v1/messages/batches/msgbatch_fake", anthropic_batch_poll, methods=["GET"]),
            Route(
                "/v1/messages/batches/msgbatch_fake/results",
                anthropic_batch_results,
                methods=["GET"],
            ),
            Route("/v1/files", openai_file_upload, methods=["POST"]),
            Route("/v1/files/file_fake/content", openai_file_content, methods=["GET"]),
            Route("/v1/batches", openai_batch_create, methods=["POST"]),
            Route("/model/{model_id:path}/converse", bedrock_converse, methods=["POST"]),
            Route("/model/{model_id:path}/converse-stream", bedrock_converse, methods=["POST"]),
            Route("/api/chat", handlers.ollama_chat, methods=["POST"]),
            Route("/api/generate", handlers.ollama_chat, methods=["POST"]),
            WebSocketRoute("/v1/realtime", realtime_ws),
        ]
    )


def main() -> None:
    global MANGLE, VERBOSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 in containers)")
    parser.add_argument(
        "--mangle", action="store_true", help="echo placeholders mangled (lowercase/hyphens)"
    )
    parser.add_argument("--status", type=int, default=200, help="answer every request with this")
    parser.add_argument(
        "--plan-limit",
        action="store_true",
        help="error answers carry Anthropic's unified rate-limit headers marked rejected",
    )
    parser.add_argument("--retry-after", type=float, default=None, help="retry-after seconds")
    parser.add_argument(
        "--abort-mid-stream",
        action="store_true",
        help="SSE replies end after the first content event",
    )
    parser.add_argument(
        "--echo-model", action="store_true", help="replies echo the request's model id"
    )
    parser.add_argument(
        "--fail-count", type=int, default=0, help="the first N requests fail, later ones succeed"
    )
    parser.add_argument("--quiet", action="store_true", help="do not print received bodies")
    args = parser.parse_args()
    MANGLE = args.mangle
    VERBOSE = not args.quiet
    scenario = Scenario(
        status=args.status,
        plan_limit=args.plan_limit,
        retry_after=args.retry_after,
        abort_mid_stream=args.abort_mid_stream,
        echo_model=args.echo_model,
        fail_count=args.fail_count,
    )
    app = build_app({"*": scenario})
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
