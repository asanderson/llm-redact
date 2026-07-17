"""Ollama native adapter: routing, conservative note injection, NDJSON
stream rehydration (chat + generate), embeds redact-only."""

import json
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from llm_redact.config import Config
from llm_redact.providers import ALL_ADAPTERS, OllamaAdapter, RouteKind
from llm_redact.providers.base import SYSTEM_NOTE
from llm_redact.proxy import create_app
from llm_redact.rehydrate import RehydratorPool
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"

received: dict[str, Any] = {}


def test_routing_matrix() -> None:
    adapter = OllamaAdapter()
    assert adapter.matches("POST", "/api/chat") is RouteKind.CHAT
    assert adapter.matches("POST", "/api/generate") is RouteKind.CHAT
    assert adapter.matches("POST", "/api/embed") is RouteKind.REDACT_ONLY
    assert adapter.matches("POST", "/api/embeddings") is RouteKind.REDACT_ONLY
    assert adapter.matches("GET", "/api/chat") is RouteKind.NONE
    assert adapter.matches("POST", "/api/tags") is RouteKind.NONE
    assert adapter.matches("POST", "/v1/chat/completions") is RouteKind.NONE


def test_matchers_disjoint_with_all_adapters() -> None:
    adapters = [cls() for cls in ALL_ADAPTERS]
    for path in ("/api/chat", "/api/generate", "/api/embed", "/api/embeddings"):
        claims = [a.name for a in adapters if a.matches("POST", path) is not RouteKind.NONE]
        assert claims == ["ollama"], (path, claims)


class TestInjectSystemNote:
    def test_chat_appends_to_last_system_message(self) -> None:
        body = {
            "messages": [
                {"role": "system", "content": "one"},
                {"role": "system", "content": "two"},
                {"role": "user", "content": "hi"},
            ]
        }
        out = OllamaAdapter().inject_system_note(body)
        assert out["messages"][0]["content"] == "one"
        assert out["messages"][1]["content"] == f"two\n\n{SYSTEM_NOTE}"
        assert body["messages"][1]["content"] == "two"  # input not mutated

    def test_chat_without_system_untouched(self) -> None:
        # Creating a system message would override the Modelfile SYSTEM
        # template and change model behavior.
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert OllamaAdapter().inject_system_note(body) == body

    def test_generate_appends_to_existing_system(self) -> None:
        body = {"prompt": "hi", "system": "be brief"}
        out = OllamaAdapter().inject_system_note(body)
        assert out["system"] == f"be brief\n\n{SYSTEM_NOTE}"

    def test_generate_without_system_untouched(self) -> None:
        body = {"prompt": "hi"}
        assert OllamaAdapter().inject_system_note(body) == {"prompt": "hi"}


def _lines(payloads: list[dict[str, Any]]) -> list[bytes]:
    return [json.dumps(p, ensure_ascii=False).encode() for p in payloads]


def _run_lines(lines: list[bytes], vault: InMemoryVault) -> list[bytes]:
    adapter = OllamaAdapter()
    pool = RehydratorPool(vault)
    return [adapter.rehydrate_ndjson_line(line, pool) for line in lines]


class TestNdjsonRehydration:
    def test_generate_token_split_across_lines(self, vault: InMemoryVault) -> None:
        vault.placeholder_for("EMAIL", EMAIL)
        out = _run_lines(
            _lines(
                [
                    {"model": "m", "response": "mail «EMA", "done": False},
                    {"model": "m", "response": "IL_001» ok", "done": False},
                    {"model": "m", "response": "", "done": True, "eval_count": 9},
                ]
            ),
            vault,
        )
        text = "".join(json.loads(line)["response"] for line in out)
        assert text == f"mail {EMAIL} ok"

    def test_chat_leftover_folds_into_done_line(self, vault: InMemoryVault) -> None:
        out = _run_lines(
            _lines(
                [
                    {"message": {"role": "assistant", "content": "cut «EMAIL_"}, "done": False},
                    {"message": {"role": "assistant", "content": ""}, "done": True},
                ]
            ),
            vault,
        )
        # The unknown partial token is held back, then folded verbatim into
        # the done line — clients concatenate content from every chunk.
        assert json.loads(out[0])["message"]["content"] == "cut "
        assert json.loads(out[1])["message"]["content"] == "«EMAIL_"

    def test_chat_tool_call_arguments_restored(self, vault: InMemoryVault) -> None:
        vault.placeholder_for("EMAIL", EMAIL)
        line = _lines(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "send", "arguments": {"to": "«EMAIL_001»"}}}
                        ],
                    },
                    "done": False,
                }
            ]
        )
        (out,) = _run_lines(line, vault)
        assert json.loads(out)["message"]["tool_calls"][0]["function"]["arguments"] == {"to": EMAIL}

    def test_untouched_lines_are_byte_identical(self, vault: InMemoryVault) -> None:
        adapter = OllamaAdapter()
        pool = RehydratorPool(vault)
        for line in (
            b'{"model":"m",  "message": {"role":"assistant","content":"plain"},"done":false}',
            b'{"status": "pulling manifest"}',
            b"not json at all",
            b"",
            b"[1, 2, 3]",
        ):
            assert adapter.rehydrate_ndjson_line(line, pool) == line


def _fake_ollama() -> Starlette:
    async def chat(request: Request) -> Response:
        body = await request.json()
        received["chat"] = body
        flat = json.dumps(body, ensure_ascii=False)
        token = next((w for w in flat.split() if w.startswith("«")), "none")
        if body.get("stream") is False:
            return JSONResponse(
                {"model": "m", "message": {"role": "assistant", "content": f"saw {token}"}}
            )
        mid = len(token) // 2
        chunks = [
            {"model": "m", "message": {"role": "assistant", "content": f"saw {token[:mid]}"}},
            {"model": "m", "message": {"role": "assistant", "content": f"{token[mid:]} end"}},
            {"model": "m", "message": {"role": "assistant", "content": ""}, "done": True},
        ]
        ndjson = b"".join(json.dumps(c, ensure_ascii=False).encode() + b"\n" for c in chunks)

        async def stream() -> Any:
            yield ndjson

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    async def embed(request: Request) -> Response:
        received["embed"] = await request.json()
        return JSONResponse({"model": "m", "embeddings": [[0.1, 0.2]]})

    return Starlette(
        routes=[
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/embed", embed, methods=["POST"]),
        ]
    )


def _client() -> httpx.AsyncClient:
    received.clear()
    app = create_app(Config(), upstream_transport=httpx.ASGITransport(app=_fake_ollama()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


@pytest.mark.anyio
async def test_streaming_chat_round_trip() -> None:
    client = _client()
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": f"mail {EMAIL} now"}],
        "system": "ignored-for-chat",
    }
    response = await client.post("/api/chat", json=body)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    upstream_flat = json.dumps(received["chat"], ensure_ascii=False)
    assert EMAIL not in upstream_flat
    assert "«EMAIL_001»" in upstream_flat
    text = "".join(
        json.loads(line)["message"]["content"]
        for line in response.content.splitlines()
        if line.strip()
    )
    assert text == f"saw {EMAIL} end"


@pytest.mark.anyio
async def test_non_streaming_chat_round_trip() -> None:
    client = _client()
    body = {
        "model": "m",
        "stream": False,
        "messages": [{"role": "user", "content": f"mail {EMAIL} now"}],
    }
    response = await client.post("/api/chat", json=body)
    assert response.status_code == 200
    assert EMAIL not in json.dumps(received["chat"])
    assert response.json()["message"]["content"] == f"saw {EMAIL}"


@pytest.mark.anyio
async def test_embed_redact_only_no_note() -> None:
    client = _client()
    response = await client.post("/api/embed", json={"model": "m", "input": f"mail {EMAIL}"})
    assert response.status_code == 200
    assert received["embed"]["input"] == "mail «EMAIL_001»"
    # REDACT_ONLY: no note injected anywhere, vector response verbatim.
    assert SYSTEM_NOTE not in json.dumps(received["embed"])
    assert response.json()["embeddings"] == [[0.1, 0.2]]


class _ChunkedUpstream(httpx.AsyncBaseTransport):
    """Streams exact byte chunks — ASGITransport would coalesce them."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async def parts() -> Any:
            for chunk in self._chunks:
                yield chunk

        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content=parts(),
            request=request,
        )


@pytest.mark.anyio
async def test_lines_split_across_network_chunks() -> None:
    chunks = [
        json.dumps(
            {"model": "m", "message": {"role": "assistant", "content": c}}, ensure_ascii=False
        ).encode()
        + b"\n"
        for c in ("saw «EMA", "IL_001» end")
    ]
    chunks.append(b'{"model":"m","message":{"role":"assistant","content":""},"done":true}\n')
    stream_bytes = b"".join(chunks)
    # Split the byte stream at an awkward offset inside a token AND inside a
    # JSON line — the NDJSON parser must reassemble both.
    upstream = _ChunkedUpstream([stream_bytes[:31], stream_bytes[31:]])
    app = create_app(Config(), upstream_transport=upstream)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    body = {"model": "m", "messages": [{"role": "user", "content": f"mail {EMAIL} now"}]}
    response = await client.post("/api/chat", json=body)
    text = "".join(
        json.loads(line)["message"]["content"]
        for line in response.content.splitlines()
        if line.strip()
    )
    assert text == f"saw {EMAIL} end"
