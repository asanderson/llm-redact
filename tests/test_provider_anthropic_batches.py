"""Anthropic Message Batches: routing, per-entry note injection, JSONL results.

The results endpoint is line-framed (application/x-jsonl) but each line is a
COMPLETE result object, so restoration is whole-string per line — the sweep
here exercises the NDJSONParser reassembly at every chunk offset instead of
streaming-channel token splits.
"""

import json
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig
from llm_redact.ndjson import NDJSONParser
from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.base import SYSTEM_NOTE, RouteKind
from llm_redact.proxy import create_app
from llm_redact.rehydrate import RehydratorPool
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"


def test_batch_routing() -> None:
    adapter = AnthropicAdapter()
    assert adapter.matches("POST", "/v1/messages/batches") is RouteKind.REDACT_ONLY
    assert adapter.matches("GET", "/v1/messages/batches/msgbatch_01/results") is RouteKind.CHAT
    # Poll/list/cancel/delete carry processing metadata only: pass-through.
    assert adapter.matches("GET", "/v1/messages/batches") is RouteKind.NONE
    assert adapter.matches("GET", "/v1/messages/batches/msgbatch_01") is RouteKind.NONE
    assert adapter.matches("POST", "/v1/messages/batches/msgbatch_01/cancel") is RouteKind.NONE
    assert adapter.matches("DELETE", "/v1/messages/batches/msgbatch_01") is RouteKind.NONE
    # A nested path that only LOOKS like results must not match.
    assert adapter.matches("GET", "/v1/messages/batches/a/b/results") is RouteKind.NONE


def test_batch_note_injected_per_entry() -> None:
    adapter = AnthropicAdapter()
    body = {
        "requests": [
            {"custom_id": "a", "params": {"model": "m", "messages": []}},
            {"custom_id": "b", "params": {"system": "be brief", "messages": []}},
            {"custom_id": "weird", "params": "not-a-dict"},
            "not-a-dict-entry",
        ]
    }
    out = adapter.inject_system_note(body)
    assert out["requests"][0]["params"]["system"] == SYSTEM_NOTE
    assert out["requests"][1]["params"]["system"].startswith("be brief")
    assert SYSTEM_NOTE in out["requests"][1]["params"]["system"]
    # Unrecognized shapes forwarded untouched.
    assert out["requests"][2] == {"custom_id": "weird", "params": "not-a-dict"}
    assert out["requests"][3] == "not-a-dict-entry"
    # Non-batch bodies keep the plain Messages injection.
    assert adapter.inject_system_note({})["system"] == SYSTEM_NOTE


def _result_line(token: str, custom_id: str = "req_1") -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_01",
                "role": "assistant",
                "content": [{"type": "text", "text": f"sent it to {token}"}],
            },
        },
    }


def test_results_line_rehydrated() -> None:
    adapter = AnthropicAdapter()
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    pool = RehydratorPool(vault)
    line = json.dumps(_result_line(token)).encode()
    out = json.loads(adapter.rehydrate_ndjson_line(line, pool))
    assert out["result"]["message"]["content"][0]["text"] == f"sent it to {EMAIL}"
    assert out["custom_id"] == "req_1"
    assert pool.counts["EMAIL"] == 1


def test_results_line_untouched_shapes() -> None:
    adapter = AnthropicAdapter()
    pool = RehydratorPool(InMemoryVault())
    # An errored result has no tokens: the original bytes come back.
    errored = json.dumps(
        {"custom_id": "x", "result": {"type": "errored", "error": {"type": "invalid_request"}}}
    ).encode()
    assert adapter.rehydrate_ndjson_line(errored, pool) is errored
    # Non-JSON and non-object lines are forwarded byte-identically.
    assert adapter.rehydrate_ndjson_line(b"not json {", pool) == b"not json {"
    assert adapter.rehydrate_ndjson_line(b"[1, 2]", pool) == b"[1, 2]"
    assert adapter.rehydrate_ndjson_line(b"", pool) == b""


def test_results_stream_split_at_every_offset() -> None:
    """NDJSONParser reassembly: any chunking of the JSONL byte stream must
    produce the same restored lines as the unsplit stream."""
    adapter = AnthropicAdapter()
    vault = InMemoryVault()
    token_a = vault.placeholder_for("EMAIL", EMAIL)
    token_b = vault.placeholder_for("AWS_KEY", "AKIAIOSFODNN7EXAMPLE")
    doc = (
        json.dumps(_result_line(token_a, "req_1"))
        + "\n"
        + json.dumps(_result_line(token_b, "req_2"))
        + "\n"
    ).encode()

    def run(chunks: list[bytes]) -> bytes:
        parser = NDJSONParser()
        pool = RehydratorPool(vault)
        out = b""
        for chunk in chunks:
            for line in parser.feed(chunk):
                out += adapter.rehydrate_ndjson_line(line, pool) + b"\n"
        tail = parser.close()
        if tail:
            out += adapter.rehydrate_ndjson_line(tail, pool)
        return out

    expected = run([doc])
    assert EMAIL.encode() in expected and b"AKIAIOSFODNN7EXAMPLE" in expected
    for offset in range(1, len(doc)):
        assert run([doc[:offset], doc[offset:]]) == expected, f"split at {offset}"


# --- integration: real app, fake batch upstream -----------------------------

received: dict[str, Any] = {}


def _fake_batch_upstream() -> Starlette:
    async def create(request: Request) -> Response:
        received["create"] = await request.json()
        return JSONResponse({"id": "msgbatch_01", "processing_status": "in_progress"})

    async def poll(request: Request) -> Response:
        return JSONResponse(
            {"id": "msgbatch_01", "processing_status": "ended", "request_counts": {"succeeded": 1}}
        )

    async def results(request: Request) -> Response:
        # Echo back every placeholder the create request carried, one
        # result line per batch entry.
        flat = json.dumps(received["create"], ensure_ascii=False)
        import re as _re

        tokens = _re.findall("«[A-Z0-9_]+»", flat)
        body = b"".join(
            json.dumps(_result_line(token, f"req_{i}")).encode() + b"\n"
            for i, token in enumerate(dict.fromkeys(tokens))
        )
        return Response(content=body, media_type="application/x-jsonl")

    return Starlette(
        routes=[
            Route("/v1/messages/batches", create, methods=["POST"]),
            Route("/v1/messages/batches/msgbatch_01", poll, methods=["GET"]),
            Route("/v1/messages/batches/msgbatch_01/results", results, methods=["GET"]),
        ]
    )


@pytest.fixture
def client() -> httpx.AsyncClient:
    received.clear()
    config = Config(providers={"anthropic": ProviderConfig(upstream_base_url="http://upstream")})
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_batch_upstream()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


@pytest.mark.anyio
async def test_batch_round_trip(client: httpx.AsyncClient) -> None:
    create_body = {
        "requests": [
            {
                "custom_id": "req_1",
                "params": {
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": f"email {EMAIL} about the launch"}],
                },
            }
        ]
    }
    created = await client.post("/v1/messages/batches", json=create_body)
    assert created.status_code == 200

    # Upstream saw the placeholder, never the real value — and the note.
    upstream_params = received["create"]["requests"][0]["params"]
    upstream_text = upstream_params["messages"][0]["content"]
    assert EMAIL not in upstream_text and "«EMAIL_001»" in upstream_text
    assert SYSTEM_NOTE in upstream_params["system"]

    # Poll passes through untouched.
    polled = await client.get("/v1/messages/batches/msgbatch_01")
    assert polled.json()["processing_status"] == "ended"

    # Results come back restored, line-framed.
    results = await client.get("/v1/messages/batches/msgbatch_01/results")
    assert results.status_code == 200
    lines = [json.loads(line) for line in results.content.splitlines() if line]
    assert lines[0]["result"]["message"]["content"][0]["text"] == f"sent it to {EMAIL}"
