"""Runnable fake LLM upstream for manual end-to-end verification.

Serves Anthropic-shaped /v1/messages (streaming and non-streaming) and
Bedrock-shaped /model/{id}/converse[-stream] — the streaming form emits
real binary event-stream frames. It logs every request body it receives —
so you can verify that only «TYPE_NNN» placeholders arrive — and echoes
any placeholders back in its reply, split across SSE chunks (or binary
frames) to exercise the proxy's streaming rehydration.

--mangle makes the echo imitate an LLM that rewrites placeholders
(lowercased, hyphens, stripped zero-padding): with [rehydration] fuzzy = true
the proxy still restores them; with fuzzy = false they come back verbatim.

Usage:
    uv run python scripts/fake_upstream.py --port 9999 [--mangle]
"""

import argparse
import json
import re
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute

from llm_redact.eventstream import EventStreamMessage, serialize, string_header

MANGLE = False


def _mangle(token: str) -> str:
    # «EMAIL_001» -> «email-1»
    body = token.strip("«»").lower().replace("_", "-")
    body = re.sub(r"-0+(\d)", r"-\1", body)
    return f"«{body}»"


async def messages(request: Request) -> Response:
    body = await request.json()
    print("--- request body received by upstream ---")
    print(json.dumps(body, indent=2, ensure_ascii=False))

    flat = json.dumps(body.get("messages", []), ensure_ascii=False)
    tokens = re.findall("«[A-Z0-9_]+»", flat)
    if MANGLE:
        tokens = [_mangle(t) for t in tokens]
    echoed = " and ".join(tokens) if tokens else "no placeholders"
    reply = f"Upstream saw {echoed} in your message."

    if not body.get("stream"):
        return JSONResponse({"content": [{"type": "text", "text": reply}], "role": "assistant"})

    async def stream() -> Any:
        yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
        # Deliberately cut the reply into 7-char chunks so placeholders are
        # split across events; the proxy must reassemble them.
        for i in range(0, len(reply), 7):
            payload = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": reply[i : i + 7]},
            }
            yield (
                b"event: content_block_delta\ndata: "
                + json.dumps(payload, ensure_ascii=False).encode()
                + b"\n\n"
            )
        yield b'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'
        yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    return StreamingResponse(stream(), media_type="text/event-stream")


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
    print(f"--- request body received by upstream ({request.url.path}) ---")
    print(json.dumps(body, indent=2, ensure_ascii=False))

    flat = json.dumps(body.get("messages", []), ensure_ascii=False)
    tokens = re.findall("«[A-Z0-9_]+»", flat)
    if MANGLE:
        tokens = [_mangle(t) for t in tokens]
    echoed = " and ".join(tokens) if tokens else "no placeholders"
    reply = f"Upstream saw {echoed} in your message."

    if request.url.path.endswith("/converse"):
        return JSONResponse(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": reply}]}},
                "stopReason": "end_turn",
            }
        )

    async def stream() -> Any:
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


async def ollama_chat(request: Request) -> Response:
    body = await request.json()
    print(f"--- request body received by upstream ({request.url.path}) ---")
    print(json.dumps(body, indent=2, ensure_ascii=False))

    flat = json.dumps(body.get("messages", body.get("prompt", "")), ensure_ascii=False)
    tokens = re.findall("«[A-Z0-9_]+»", flat)
    if MANGLE:
        tokens = [_mangle(t) for t in tokens]
    echoed = " and ".join(tokens) if tokens else "no placeholders"
    reply = f"Upstream saw {echoed} in your message."
    generate = request.url.path.endswith("/generate")

    def chunk(text: str, done: bool) -> bytes:
        payload: dict[str, Any] = {"model": "fake", "done": done}
        if generate:
            payload["response"] = text
        else:
            payload["message"] = {"role": "assistant", "content": text}
        return json.dumps(payload, ensure_ascii=False).encode() + b"\n"

    if body.get("stream") is False:
        return Response(chunk(reply, True), media_type="application/json")

    async def stream() -> Any:
        # 7-char pieces split placeholders across NDJSON lines.
        for i in range(0, len(reply), 7):
            yield chunk(reply[i : i + 7], False)
        yield chunk("", True)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


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
        print(f"[fake realtime] <- {event.get('type')}: {json.dumps(event)[:200]}")
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
    tokens = re.findall("«[A-Z0-9_]+»", flat)
    _created_batches["msgbatch_fake"] = {"tokens": list(dict.fromkeys(tokens))}
    print(f"[fake] batch create carrying tokens: {tokens}")
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
                print(f"[fake] stored upload ({len(part.content)} bytes)")
    return JSONResponse({"id": "file_fake", "object": "file", "purpose": "batch"})


async def openai_file_content(request: Request) -> Response:
    return Response(
        content=_uploaded_files.get("file_fake", b""),
        media_type="application/octet-stream",
    )


async def openai_batch_create(request: Request) -> Response:
    return JSONResponse({"id": "batch_fake", "status": "validating"})


def main() -> None:
    global MANGLE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 in containers)")
    parser.add_argument(
        "--mangle", action="store_true", help="echo placeholders mangled (lowercase/hyphens)"
    )
    args = parser.parse_args()
    MANGLE = args.mangle
    app = Starlette(
        routes=[
            Route("/v1/messages", messages, methods=["POST"]),
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
            Route("/api/chat", ollama_chat, methods=["POST"]),
            Route("/api/generate", ollama_chat, methods=["POST"]),
            WebSocketRoute("/v1/realtime", realtime_ws),
        ]
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
