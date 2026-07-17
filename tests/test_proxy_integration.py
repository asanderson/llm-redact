"""In-process end-to-end tests: real proxy app, fake upstream via ASGITransport."""

import json
import re
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig
from llm_redact.detection.deny import DenyEntry
from llm_redact.detection.engine import DetectionConfig
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
PEM = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA7bq8V3xQ9c\n-----END RSA PRIVATE KEY-----"

received: dict[str, Any] = {}


def _fake_upstream() -> Starlette:
    """Anthropic- and OpenAI-shaped fake provider. Records request bodies and
    echoes back any placeholders it saw, split awkwardly across SSE chunks."""

    async def messages(request: Request) -> Response:
        body = await request.json()
        received["anthropic"] = body
        received["anthropic_headers"] = dict(request.headers)
        flat = json.dumps(body["messages"], ensure_ascii=False)
        tokens = re.findall("«[A-Z0-9_]+»", flat)
        echoed = " and ".join(tokens) if tokens else "no tokens"
        if body.get("stream"):
            reply = f"I see {echoed} here"
            mid = len(reply) // 2 + 1  # deliberately split inside a token

            async def stream() -> Any:
                yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
                for part in (reply[:mid], reply[mid:]):
                    payload = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": part},
                    }
                    yield (
                        b"event: content_block_delta\ndata: "
                        + json.dumps(payload).encode()
                        + b"\n\n"
                    )
                yield (
                    b"event: content_block_stop\n"
                    b'data: {"type": "content_block_stop", "index": 0}\n\n'
                )
                yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'

            return StreamingResponse(stream(), media_type="text/event-stream")
        return JSONResponse(
            {"content": [{"type": "text", "text": f"I see {echoed} here"}], "role": "assistant"}
        )

    async def count_tokens(request: Request) -> Response:
        received["count_tokens"] = await request.json()
        return JSONResponse({"input_tokens": 42})

    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        received["openai"] = body
        text = body["messages"][-1]["content"]
        tokens = [w for w in text.split() if w.startswith("«")]
        echoed = tokens[0] if tokens else "none"
        if body.get("stream"):
            half = len(echoed) // 2

            async def stream() -> Any:
                for part in (f"token {echoed[:half]}", f"{echoed[half:]} end"):
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": 0, "delta": {"content": part}, "finish_reason": None}
                        ],
                    }
                    yield b"data: " + json.dumps(chunk).encode() + b"\n\n"
                fin = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield b"data: " + json.dumps(fin).encode() + b"\n\n"
                yield b"data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")
        return JSONResponse(
            {"choices": [{"message": {"role": "assistant", "content": f"token {echoed}"}}]}
        )

    async def responses(request: Request) -> Response:
        body = await request.json()
        received["responses"] = body
        flat = json.dumps(body.get("input", ""), ensure_ascii=False)
        tokens = re.findall("\u00ab[A-Z0-9_]+\u00bb", flat)
        echoed = tokens[0] if tokens else "none"
        if body.get("stream"):
            half = max(1, len(echoed) // 2)

            async def stream() -> Any:
                yield (
                    b"event: response.created\ndata: "
                    + json.dumps(
                        {"type": "response.created", "response": {"id": "resp_1"}}
                    ).encode()
                    + b"\n\n"
                )
                for part in (f"saw {echoed[:half]}", f"{echoed[half:]} end"):
                    payload = {
                        "type": "response.output_text.delta",
                        "item_id": "item_1",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": part,
                    }
                    yield (
                        b"event: response.output_text.delta\ndata: "
                        + json.dumps(payload, ensure_ascii=False).encode()
                        + b"\n\n"
                    )
                completed = {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "output": [
                            {
                                "id": "item_1",
                                "type": "message",
                                "content": [{"type": "output_text", "text": f"saw {echoed} end"}],
                            }
                        ],
                    },
                }
                yield (
                    b"event: response.completed\ndata: "
                    + json.dumps(completed, ensure_ascii=False).encode()
                    + b"\n\n"
                )

            return StreamingResponse(stream(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": "resp_1",
                "output": [
                    {
                        "id": "item_1",
                        "type": "message",
                        "content": [{"type": "output_text", "text": f"saw {echoed}"}],
                    }
                ],
            }
        )

    async def gemini_generate(request: Request) -> Response:
        body = await request.json()
        received["gemini"] = body
        received["gemini_query"] = str(request.url.query)
        flat = json.dumps(body.get("contents", []), ensure_ascii=False)
        tokens = re.findall("«[A-Z0-9_]+»", flat)
        echoed = tokens[0] if tokens else "none"
        reply = f"saw {echoed} end"

        def chunks() -> list[dict[str, Any]]:
            # 7-char pieces: tokens split across chunk boundaries.
            pieces = [reply[i : i + 7] for i in range(0, len(reply), 7)]
            out = [
                {"candidates": [{"index": 0, "content": {"parts": [{"text": p}]}}]} for p in pieces
            ]
            out.append({"candidates": [{"index": 0, "finishReason": "STOP"}]})
            return out

        if request.url.path.endswith(":streamGenerateContent"):
            if request.query_params.get("alt") == "sse":

                async def stream() -> Any:
                    for chunk in chunks():
                        yield b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n"

                return StreamingResponse(stream(), media_type="text/event-stream")
            return JSONResponse(chunks())  # the no-alt JSON-array form
        return JSONResponse(
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {"parts": [{"text": reply}]},
                        "finishReason": "STOP",
                    }
                ]
            }
        )

    async def gemini_count(request: Request) -> Response:
        received["gemini_count"] = await request.json()
        return JSONResponse({"totalTokens": 42})

    async def conversations(request: Request) -> Response:
        if request.method == "DELETE":
            received["conversations_delete"] = request.url.path
            return JSONResponse({"id": "conv_1", "object": "conversation.deleted", "deleted": True})
        if request.method == "POST":
            body = await request.json()
            received["conversations_post"] = body
            received.setdefault("conversations_store", []).extend(body.get("items", []))
            return JSONResponse(
                {"id": "conv_1", "object": "conversation", "items": body.get("items", [])}
            )
        return JSONResponse({"object": "list", "data": received.get("conversations_store", [])})

    async def unknown(request: Request) -> Response:
        received["unknown_path"] = request.url.path
        received["unknown_host"] = request.url.hostname
        raw = await request.body()
        try:
            received["unknown_body"] = json.loads(raw) if raw else None
        except ValueError:
            received["unknown_body"] = None
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route("/v1/messages", messages, methods=["POST"]),
            Route("/v1/messages/count_tokens", count_tokens, methods=["POST"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            # Azure chat completions is wire-identical: reuse the handler.
            Route("/openai/deployments/d1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/responses", responses, methods=["POST"]),
            Route("/v1beta/models/gemini-test:generateContent", gemini_generate, methods=["POST"]),
            Route(
                "/v1beta/models/gemini-test:streamGenerateContent",
                gemini_generate,
                methods=["POST"],
            ),
            Route("/v1beta/models/gemini-test:countTokens", gemini_count, methods=["POST"]),
            Route("/v1/conversations", conversations, methods=["POST"]),
            Route("/v1/conversations/{cid}", conversations, methods=["GET", "DELETE"]),
            Route("/v1/conversations/{cid}/items", conversations, methods=["GET", "POST"]),
            Route("/{path:path}", unknown, methods=["GET", "POST"]),
        ]
    )


@pytest.fixture
def client() -> httpx.AsyncClient:
    received.clear()
    config = Config(
        providers={
            "anthropic": ProviderConfig(upstream_base_url="http://upstream"),
            "openai": ProviderConfig(upstream_base_url="http://upstream"),
            "gemini": ProviderConfig(upstream_base_url="http://upstream"),
            "azure": ProviderConfig(upstream_base_url=""),  # deliberately unconfigured
        }
    )
    proxy_app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app), base_url="http://127.0.0.1"
    )


def _anthropic_body(text: str, stream: bool = False) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "stream": stream,
        "messages": [{"role": "user", "content": text}],
    }


async def test_originals_never_reach_upstream(client: httpx.AsyncClient) -> None:
    body = _anthropic_body(f"my email {EMAIL} key {AWS_KEY} pem {PEM}")
    response = await client.post("/v1/messages", json=body)
    assert response.status_code == 200
    upstream_flat = json.dumps(received["anthropic"], ensure_ascii=False)
    assert EMAIL not in upstream_flat
    assert AWS_KEY not in upstream_flat
    assert "BEGIN RSA" not in upstream_flat
    assert "«EMAIL_001»" in upstream_flat
    assert "«AWS_KEY_001»" in upstream_flat
    assert "«PRIVATE_KEY_001»" in upstream_flat


async def test_system_note_injected_only_when_redacted(client: httpx.AsyncClient) -> None:
    await client.post("/v1/messages", json=_anthropic_body(f"mail {EMAIL}"))
    assert "privacy tokens" in received["anthropic"].get("system", "")
    await client.post("/v1/messages", json=_anthropic_body("nothing sensitive"))
    assert "system" not in received["anthropic"]


async def test_non_streaming_rehydration(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/messages", json=_anthropic_body(f"mail {EMAIL}"))
    text = response.json()["content"][0]["text"]
    assert EMAIL in text
    assert "«" not in text


async def test_streaming_rehydration_anthropic(client: httpx.AsyncClient) -> None:
    body = _anthropic_body(f"mail {EMAIL} and key {AWS_KEY}", stream=True)
    reassembled = ""
    async with client.stream("POST", "/v1/messages", json=body) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = b"".join([chunk async for chunk in response.aiter_bytes()])
    for block in raw.decode().split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: ") and "content_block_delta" in line:
                payload = json.loads(line[6:])
                reassembled += payload["delta"]["text"]
    assert EMAIL in reassembled
    assert AWS_KEY in reassembled
    assert "«" not in reassembled


async def test_streaming_rehydration_openai(client: httpx.AsyncClient) -> None:
    body = {
        "model": "gpt-4o",
        "stream": True,
        "messages": [{"role": "user", "content": f"mail {EMAIL}"}],
    }
    async with client.stream("POST", "/v1/chat/completions", json=body) as response:
        raw = b"".join([chunk async for chunk in response.aiter_bytes()])
    reassembled = ""
    for block in raw.decode().split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                for choice in json.loads(line[6:]).get("choices", []):
                    content = choice.get("delta", {}).get("content")
                    if content:
                        reassembled += content
    assert EMAIL in reassembled
    assert "«" not in reassembled
    assert raw.decode().rstrip().endswith("data: [DONE]")


async def test_count_tokens_redacted_but_not_rehydrated(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/messages/count_tokens", json=_anthropic_body(f"mail {EMAIL}"))
    assert response.status_code == 200
    assert EMAIL not in json.dumps(received["count_tokens"], ensure_ascii=False)
    assert response.json() == {"input_tokens": 42}


async def test_unknown_path_passes_through(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/models", content=b"not json {{", headers={})
    assert response.status_code == 200
    assert received["unknown_path"] == "/v1/models"


async def test_auth_headers_forwarded(client: httpx.AsyncClient) -> None:
    await client.post(
        "/v1/messages",
        json=_anthropic_body("hi"),
        headers={"x-api-key": "sk-ant-real-key-value-12345678", "anthropic-version": "2023-06-01"},
    )
    headers = received["anthropic_headers"]
    assert headers["x-api-key"] == "sk-ant-real-key-value-12345678"
    assert headers["anthropic-version"] == "2023-06-01"


async def test_json_error_reply_to_stream_request(client: httpx.AsyncClient) -> None:
    # Upstream answers a stream:true request with plain JSON (e.g. an error):
    # the content-type branch must handle it, not the stream flag.
    body = _anthropic_body("hi")
    body["stream"] = True
    del body["messages"][0]["content"]
    body["messages"][0]["content"] = "plain"

    response = await client.post("/v1/messages/count_tokens", json=body)
    assert response.status_code == 200
    assert response.json() == {"input_tokens": 42}


async def test_sqlite_vault_survives_restart(tmp_path) -> None:
    from llm_redact.config import VaultConfig

    db = tmp_path / "vault.db"
    config = Config(
        providers={
            "anthropic": ProviderConfig(upstream_base_url="http://upstream"),
            "openai": ProviderConfig(upstream_base_url="http://upstream"),
        },
        vault=VaultConfig(backend="sqlite", path=str(db), session="s1"),
    )

    def make_client() -> httpx.AsyncClient:
        app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        )

    received.clear()
    # "First run": redact an email, which persists the mapping.
    client1 = make_client()
    await client1.post("/v1/messages", json=_anthropic_body(f"mail {EMAIL}"))
    assert "«EMAIL_001»" in json.dumps(received["anthropic"], ensure_ascii=False)
    await client1.aclose()

    # "Restarted proxy": a fresh app on the same DB must rehydrate a token
    # issued before the restart (the fake upstream echoes it back).
    client2 = make_client()
    response = await client2.post(
        "/v1/messages", json=_anthropic_body("no secrets this time «EMAIL_001»?")
    )
    text = response.json()["content"][0]["text"]
    assert EMAIL in text
    await client2.aclose()


def _limited_client(limit: int) -> httpx.AsyncClient:
    received.clear()
    config = Config(
        max_body_bytes=limit,
        providers={
            "anthropic": ProviderConfig(upstream_base_url="http://upstream"),
            "openai": ProviderConfig(upstream_base_url="http://upstream"),
        },
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


async def test_oversized_body_fails_closed_with_provider_shape() -> None:
    client = _limited_client(limit=64)
    body = _anthropic_body("x" * 200)
    response = await client.post("/v1/messages", json=body)
    assert response.status_code == 413
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "request_too_large"
    # Fail closed: the upstream was never contacted.
    assert "anthropic" not in received

    openai_response = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "y" * 200}]},
    )
    assert openai_response.status_code == 413
    assert openai_response.json()["error"]["code"] == "request_too_large"
    assert "openai" not in received
    await client.aclose()


async def test_body_at_limit_passes() -> None:
    body = _anthropic_body("hello")
    exact = len(json.dumps(body, separators=(",", ":")).encode())
    client = _limited_client(limit=4096)
    # Comfortably under the limit; and an exactly-at-limit read must not 413.
    raw = json.dumps(body, separators=(",", ":")).encode().ljust(exact, b" ")
    response = await client.post(
        "/v1/messages", content=raw, headers={"content-type": "application/json"}
    )
    assert response.status_code == 200
    await client.aclose()


async def test_oversized_passthrough_still_forwards() -> None:
    client = _limited_client(limit=64)
    response = await client.post("/v1/models", content=b"z" * 500)
    assert response.status_code == 200
    assert received["unknown_path"] == "/v1/models"
    await client.aclose()


async def test_responses_streaming_rehydration(client: httpx.AsyncClient) -> None:
    body = {"model": "gpt-4o", "stream": True, "input": f"mail {EMAIL} please"}
    async with client.stream("POST", "/v1/responses", json=body) as response:
        raw = b"".join([chunk async for chunk in response.aiter_bytes()])
    # Request side: the upstream saw only placeholders.
    assert EMAIL not in json.dumps(received["responses"], ensure_ascii=False)
    reassembled = ""
    completed_text = None
    for block in raw.decode().split("\n\n"):
        for line in block.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if payload.get("type") == "response.output_text.delta":
                reassembled += payload["delta"]
            if payload.get("type") == "response.completed":
                completed_text = payload["response"]["output"][0]["content"][0]["text"]
    assert EMAIL in reassembled
    assert "\u00ab" not in reassembled
    assert completed_text is not None and EMAIL in completed_text


async def test_responses_non_streaming_rehydration(client: httpx.AsyncClient) -> None:
    body = {"model": "gpt-4o", "input": f"mail {EMAIL} please"}
    response = await client.post("/v1/responses", json=body)
    text = response.json()["output"][0]["content"][0]["text"]
    assert EMAIL in text
    assert "\u00ab" not in text


async def test_openai_conversations_round_trip(client: httpx.AsyncClient) -> None:
    # POST create: item content is redacted before upstream and the echoed
    # response is rehydrated for the client.
    body = {
        "items": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"email {EMAIL} for me"}],
            }
        ]
    }
    created = await client.post("/v1/conversations", json=body)
    assert created.status_code == 200
    upstream_seen = json.dumps(received["conversations_post"], ensure_ascii=False)
    assert EMAIL not in upstream_seen  # never leaked
    assert "«EMAIL_001»" in upstream_seen  # redacted
    assert EMAIL in json.dumps(created.json(), ensure_ascii=False)  # response rehydrated

    # GET items: the stored (placeholder) content is rehydrated on the way back.
    listed = await client.get("/v1/conversations/conv_1/items")
    assert EMAIL in json.dumps(listed.json(), ensure_ascii=False)
    assert "«" not in json.dumps(listed.json(), ensure_ascii=False)

    # DELETE carries ids only: it reaches the upstream unchanged (pass-through).
    deleted = await client.delete("/v1/conversations/conv_1")
    assert deleted.status_code == 200
    assert received["conversations_delete"].endswith("/v1/conversations/conv_1")


async def test_plaintext_document_redacted(client: httpx.AsyncClient) -> None:
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": f"Contact: {EMAIL}\nKey: {AWS_KEY}",
                        },
                    },
                    {"type": "text", "text": "summarize this document"},
                ],
            }
        ],
    }
    await client.post("/v1/messages", json=body)
    upstream_flat = json.dumps(received["anthropic"], ensure_ascii=False)
    assert EMAIL not in upstream_flat
    assert AWS_KEY not in upstream_flat
    assert "«EMAIL_001»" in upstream_flat


async def test_reserved_endpoints_never_forwarded(client: httpx.AsyncClient) -> None:
    status = await client.get("/__llm-redact/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["vault"]["backend"] == "memory"
    assert payload["audit"]["enabled"] is False
    # Open-core honesty (llm-redact-pro docs/licensing.md): the Free-only test env has no pro
    # package, so the licensed-features signal reports absent and no plugins.
    assert payload["license"]["package_installed"] is False
    assert payload["license"]["plugins"] == []

    audit = await client.get("/__llm-redact/audit")
    assert audit.status_code == 404  # disabled by default

    recent = await client.get("/__llm-redact/recent")
    assert recent.status_code == 200  # works without the audit DB
    assert recent.json() == {"entries": []}  # reserved paths are not recorded

    unknown = await client.get("/__llm-redact/anything")
    assert unknown.status_code == 404
    posted = await client.post("/__llm-redact/status", content=b"{}")
    assert posted.status_code == 405
    dashboard = await client.get("/__llm-redact/")
    assert dashboard.status_code == 200
    assert "data-llm-redact-dashboard" in dashboard.text
    # The non-forwarding guarantee: nothing above reached the fake upstream.
    assert received == {}


async def test_recent_buffer_without_audit(client: httpx.AsyncClient) -> None:
    # One non-streaming and one streaming request; audit stays disabled.
    await client.post("/v1/messages", json=_anthropic_body(f"mail {EMAIL}"))
    async with client.stream(
        "POST", "/v1/messages", json=_anthropic_body(f"key {AWS_KEY}", stream=True)
    ) as response:
        async for _ in response.aiter_bytes():
            pass

    assert (await client.get("/__llm-redact/audit")).status_code == 404
    entries = (await client.get("/__llm-redact/recent")).json()["entries"]
    assert len(entries) == 2
    # Newest first: the streamed request finalized last.
    assert entries[0]["streamed"] is True
    assert entries[0]["detections"] == {"AWS_KEY": 1}
    assert entries[0]["rehydrations"].get("AWS_KEY", 0) >= 1
    assert entries[1]["streamed"] is False
    assert entries[1]["detections"] == {"EMAIL": 1}
    assert all(e["provider"] == "anthropic" and e["status"] == 200 for e in entries)
    # Metadata only — never values.
    flat = json.dumps(entries)
    assert EMAIL not in flat and AWS_KEY not in flat

    limited = (await client.get("/__llm-redact/recent?limit=1")).json()["entries"]
    assert len(limited) == 1 and limited[0]["streamed"] is True
    # The /recent fetches themselves are reserved-path traffic: never rows.
    assert len((await client.get("/__llm-redact/recent")).json()["entries"]) == 2


async def test_metrics_endpoint(client: httpx.AsyncClient) -> None:
    # One adapter round trip, one oversized 413, one pass-through.
    await client.post("/v1/messages", json=_anthropic_body(f"mail {EMAIL}"))
    await client.post("/v1/models", content=b"probe")

    response = await client.get("/__llm-redact/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert 'llm_redact_requests_total{provider="anthropic",status="200"} 1' in text
    assert 'llm_redact_requests_total{provider="passthrough",status="200"} 1' in text
    assert 'llm_redact_detections_total{type="EMAIL"} 1' in text
    # Per-provider (and streamed) duration histogram.
    assert (
        'llm_redact_request_duration_seconds_bucket{provider="anthropic",streamed="false",'
        'le="+Inf"} 1' in text
    )
    assert (
        'llm_redact_request_duration_seconds_bucket{provider="passthrough",streamed="false",'
        'le="+Inf"} 1' in text
    )
    # GET-only, and never forwarded (upstream saw only the two real requests).
    assert (await client.post("/__llm-redact/metrics", content=b"x")).status_code == 405
    assert "metrics" not in json.dumps(received)


async def test_413_counted_in_metrics() -> None:
    client = _limited_client(limit=64)
    await client.post("/v1/messages", json=_anthropic_body("x" * 200))
    text = (await client.get("/__llm-redact/metrics")).text
    assert 'llm_redact_requests_total{provider="anthropic",status="413"} 1' in text
    await client.aclose()


def _gemini_body(text: str) -> dict[str, Any]:
    return {"contents": [{"role": "user", "parts": [{"text": text}]}]}


async def test_gemini_round_trip(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1beta/models/gemini-test:generateContent?key=TOPSECRETQUERYKEY",
        json=_gemini_body(f"mail {EMAIL}"),
    )
    assert response.status_code == 200
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    assert text == f"saw {EMAIL} end"
    upstream_flat = json.dumps(received["gemini"], ensure_ascii=False)
    assert EMAIL not in upstream_flat
    assert "«EMAIL_001»" in upstream_flat
    # The ?key= auth query is forwarded to the upstream untouched.
    assert "key=TOPSECRETQUERYKEY" in received["gemini_query"]


async def test_gemini_streaming_round_trip(client: httpx.AsyncClient) -> None:
    async with client.stream(
        "POST",
        "/v1beta/models/gemini-test:streamGenerateContent?alt=sse",
        json=_gemini_body(f"mail {EMAIL}"),
    ) as response:
        assert response.status_code == 200
        raw = (await response.aread()).decode()
    parts: list[str] = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            for candidate in json.loads(line[6:]).get("candidates", []):
                for part in (candidate.get("content") or {}).get("parts") or []:
                    if isinstance(part.get("text"), str):
                        parts.append(part["text"])
    assert "".join(parts) == f"saw {EMAIL} end"
    assert "«EMAIL_001»" not in raw


async def test_gemini_json_array_stream_round_trip(client: httpx.AsyncClient) -> None:
    # streamGenerateContent WITHOUT alt=sse: a buffered JSON array whose
    # elements split the token — the list-aware rehydrate_body must reassemble.
    response = await client.post(
        "/v1beta/models/gemini-test:streamGenerateContent",
        json=_gemini_body(f"mail {EMAIL}"),
    )
    assert response.status_code == 200
    chunks = response.json()
    assert isinstance(chunks, list)
    text = "".join(
        part["text"]
        for chunk in chunks
        for candidate in chunk.get("candidates", [])
        for part in (candidate.get("content") or {}).get("parts") or []
        if isinstance(part.get("text"), str)
    )
    assert text == f"saw {EMAIL} end"
    assert "«EMAIL_001»" not in response.text


async def test_gemini_count_tokens_redact_only(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1beta/models/gemini-test:countTokens", json=_gemini_body(f"mail {EMAIL}")
    )
    assert response.status_code == 200
    assert response.json() == {"totalTokens": 42}  # passed through untouched
    upstream_flat = json.dumps(received["gemini_count"], ensure_ascii=False)
    assert EMAIL not in upstream_flat
    assert "«EMAIL_001»" in upstream_flat


async def test_gemini_key_query_never_logged(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("DEBUG"):
        await client.post(
            "/v1beta/models/gemini-test:generateContent?key=TOPSECRETQUERYKEY",
            json=_gemini_body("hello"),
        )
    # The proxy's own log lines carry the path only, never the query. (The
    # httpx library logger DOES print full URLs — cli.py silences it and the
    # CHANGELOG documents the embedding caveat.)
    proxy_records = [r for r in caplog.records if r.name.startswith("llm_redact")]
    assert proxy_records  # the request log line exists
    assert all("TOPSECRETQUERYKEY" not in record.getMessage() for record in proxy_records)


async def test_azure_unconfigured_returns_502(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/openai/deployments/d1/chat/completions",
        json={"messages": [{"role": "user", "content": f"mail {EMAIL}"}]},
    )
    assert response.status_code == 502
    assert "providers.azure" in response.json()["error"]["message"]
    assert "openai" not in received  # never forwarded anywhere


async def test_openai_compatible_local_server_base() -> None:
    """Ollama/vLLM/LM Studio: point [providers.openai] at the local server."""
    received.clear()
    config = Config(
        providers={
            "anthropic": ProviderConfig(upstream_base_url="http://upstream"),
            "openai": ProviderConfig(upstream_base_url="http://localhost:11434"),
            "gemini": ProviderConfig(upstream_base_url="http://upstream"),
            "azure": ProviderConfig(upstream_base_url=""),
        }
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "llama3", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]},
    )
    assert response.status_code == 200
    assert EMAIL in response.json()["choices"][0]["message"]["content"]
    assert EMAIL not in json.dumps(received["openai"])


async def test_v1beta_passthrough_routes_to_gemini_upstream() -> None:
    received.clear()
    config = Config(
        providers={
            "anthropic": ProviderConfig(upstream_base_url="http://anthropic-upstream"),
            "openai": ProviderConfig(upstream_base_url="http://openai-upstream"),
            "gemini": ProviderConfig(upstream_base_url="http://gemini-upstream"),
            "azure": ProviderConfig(upstream_base_url=""),
        }
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")
    response = await client.get("/v1beta/models")
    assert response.status_code == 200
    assert received["unknown_path"] == "/v1beta/models"
    assert received["unknown_host"] == "gemini-upstream"


def _modes_client(modes: tuple[tuple[str, str], ...]) -> httpx.AsyncClient:
    received.clear()
    config = Config(
        providers={
            "anthropic": ProviderConfig(upstream_base_url="http://upstream"),
            "openai": ProviderConfig(upstream_base_url="http://upstream"),
            "gemini": ProviderConfig(upstream_base_url="http://upstream"),
            "azure": ProviderConfig(upstream_base_url=""),
        },
        detection=DetectionConfig(modes=modes),
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


async def test_block_mode_rejects_before_upstream() -> None:
    client = _modes_client((("us_ssn", "block"),))
    response = await client.post("/v1/messages", json=_anthropic_body("my ssn is 219-09-9999"))
    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"  # anthropic error shape: SDKs surface the message
    assert "SSN" in payload["error"]["message"]
    assert 'mode = "block"' in payload["error"]["message"]
    assert "219-09-9999" not in json.dumps(payload)  # never echo the value
    assert "anthropic" not in received  # nothing was forwarded

    status = (await client.get("/__llm-redact/status")).json()
    assert status["blocked_total"] == {"SSN": 1}
    metrics = (await client.get("/__llm-redact/metrics")).text
    assert 'llm_redact_blocked_total{type="SSN"} 1' in metrics
    assert 'llm_redact_requests_total{provider="anthropic",status="400"} 1' in metrics


async def test_warn_mode_forwards_original_and_counts() -> None:
    client = _modes_client((("phone_number", "warn"),))
    body = _anthropic_body(f"call +1 415 555 0100, mail {EMAIL}")
    response = await client.post("/v1/messages", json=body)
    assert response.status_code == 200
    upstream_flat = json.dumps(received["anthropic"], ensure_ascii=False)
    assert "+1 415 555 0100" in upstream_flat  # warn deliberately forwards the value
    assert EMAIL not in upstream_flat  # other rules still redact
    assert "«EMAIL_001»" in upstream_flat

    status = (await client.get("/__llm-redact/status")).json()
    assert status["warnings_total"] == {"PHONE": 1}
    assert status["detections_total"].get("PHONE") is None
    assert status["detection"]["modes"] == {"phone_number": "warn"}
    metrics = (await client.get("/__llm-redact/metrics")).text
    assert 'llm_redact_warnings_total{type="PHONE"} 1' in metrics

    # 3.3: the warn hit is attributed to THIS request in the recent buffer,
    # not just the process-lifetime aggregate — "did my key leak?" is
    # answerable per request.
    recent = (await client.get("/__llm-redact/recent?limit=1")).json()["entries"]
    assert recent[0]["warned"] == {"PHONE": 1}
    assert recent[0]["detections"].get("PHONE") is None


async def test_block_mode_covers_redact_only_routes() -> None:
    # count_tokens is REDACT_ONLY: block must fail closed there too.
    client = _modes_client((("email", "block"),))
    response = await client.post(
        "/v1/messages/count_tokens",
        json={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": EMAIL}]},
    )
    assert response.status_code == 400
    assert "count_tokens" not in received


async def test_block_error_type_is_not_request_too_large() -> None:
    # error_body() grew out of the 413 path; a blocked request must carry a
    # 400-appropriate error type or SDKs give misleading advice.
    client = _modes_client((("us_ssn", "block"),))
    response = await client.post("/v1/messages", json=_anthropic_body("ssn 219-09-9999"))
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_deny_string_round_trip() -> None:
    received.clear()
    config = Config(
        providers={
            "anthropic": ProviderConfig(upstream_base_url="http://upstream"),
            "openai": ProviderConfig(upstream_base_url="http://upstream"),
            "gemini": ProviderConfig(upstream_base_url="http://upstream"),
            "azure": ProviderConfig(upstream_base_url=""),
        },
        detection=DetectionConfig(deny_strings=(DenyEntry("Project Aurora"),)),
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")

    body = _anthropic_body("status of project aurora please")
    response = await client.post("/v1/messages", json=body)
    assert response.status_code == 200
    upstream_flat = json.dumps(received["anthropic"], ensure_ascii=False)
    assert "aurora" not in upstream_flat.lower()
    assert "«DENY_001»" in upstream_flat
    # The fake upstream echoes the token back; the tool sees the original
    # casing it sent.
    assert "project aurora" in response.json()["content"][0]["text"]

    status = (await client.get("/__llm-redact/status")).json()
    assert status["detections_total"] == {"DENY": 1}
    assert status["detection"]["deny_strings"] == 1


async def test_openai_embeddings_redact_only_no_note(client: httpx.AsyncClient) -> None:
    # Embeddings input is redacted, the response passes through verbatim,
    # and no system note is injected (the body has no field to carry one).
    response = await client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": [f"contact {EMAIL} today"]},
    )
    assert response.status_code == 200
    flat = json.dumps(received["unknown_body"], ensure_ascii=False)
    assert EMAIL not in flat
    assert "«EMAIL_001»" in flat
    assert "messages" not in received["unknown_body"]
    assert "system" not in received["unknown_body"]


async def test_gemini_embed_content_redact_only_no_note(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1beta/models/text-embedding-004:embedContent",
        json={"content": {"parts": [{"text": f"contact {EMAIL} today"}]}},
    )
    assert response.status_code == 200
    flat = json.dumps(received["unknown_body"], ensure_ascii=False)
    assert EMAIL not in flat
    assert "«EMAIL_001»" in flat
    assert "systemInstruction" not in received["unknown_body"]


async def test_anthropic_count_tokens_keeps_note(client: httpx.AsyncClient) -> None:
    # Pre-hook behavior preserved: the count includes the note the real
    # chat request will carry.
    response = await client.post(
        "/v1/messages/count_tokens",
        json={"model": "m", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]},
    )
    assert response.status_code == 200
    assert "privacy tokens" in json.dumps(received["count_tokens"])


async def test_gemini_count_tokens_keeps_note(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1beta/models/gemini-test:countTokens",
        json={"contents": [{"role": "user", "parts": [{"text": f"mail {EMAIL}"}]}]},
    )
    assert response.status_code == 200
    assert "privacy tokens" in json.dumps(received["gemini_count"])


def test_provider_inference_covers_pass_through_surfaces() -> None:
    """2.0 rider: paths the coverage doc calls 'deliberately forwarded'
    must infer their real provider — before these prefixes existed they
    were sent to the anthropic default and 404'd (the /v1/uploads bug
    class from 1.14.0)."""
    state = create_app(Config()).state.proxy
    expectations = {
        "/v1/moderations": "openai",
        "/v1/fine_tuning/jobs": "openai",
        "/v1/realtime/client_secrets": "openai",
        "/v1/vector_stores/vs_1/search": "openai",
        "/v1/assistants": "openai",
        "/v1/threads/th_1/messages": "openai",
        "/upload/v1beta/files": "gemini",
        "/guardrail/g1/version/1/apply": "bedrock",
        "/async-invoke": "bedrock",
    }
    for path, provider in expectations.items():
        assert state.provider_for(None, path) == provider, path
