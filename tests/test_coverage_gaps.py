"""Coverage-gap closure: legacy completions, stored retrievals, Azure
files, and header-aware /v1/files routing."""

import json
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig
from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.azure_openai import AzureOpenAIAdapter
from llm_redact.providers.base import RouteKind
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.proxy import create_app
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"


def test_legacy_openai_stream_split_at_every_offset() -> None:
    """choices[].text deltas reassemble split tokens; [DONE] flushes the
    leftover as a text_completion-shaped synthetic chunk."""
    adapter = OpenAIAdapter()
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    full = f"sent to {token} ok"

    def run(split: int) -> str:
        pool = RehydratorPool(vault)
        out = ""
        for part in (full[:split], full[split:]):
            event = SSEEvent(
                data=json.dumps({"choices": [{"index": 0, "text": part, "finish_reason": None}]})
            )
            for rewritten in adapter.rehydrate_event(event, pool):
                payload = json.loads(rewritten.data)
                out += payload["choices"][0].get("text", "")
        done = SSEEvent(data="[DONE]")
        for rewritten in adapter.rehydrate_event(done, pool):
            if rewritten.data.strip() != "[DONE]":
                out += json.loads(rewritten.data)["choices"][0].get("text", "")
        return out

    expected = f"sent to {EMAIL} ok"
    for split in range(1, len(full)):
        assert run(split) == expected, f"split at {split}"


def test_legacy_anthropic_stream_split_at_every_offset() -> None:
    """completion-field deltas reassemble; the stop_reason event flushes."""
    adapter = AnthropicAdapter()
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    full = f"mail {token} done"

    def run(split: int) -> str:
        pool = RehydratorPool(vault)
        out = ""
        parts = [
            {"type": "completion", "completion": full[:split], "stop_reason": None},
            {"type": "completion", "completion": full[split:], "stop_reason": "stop_sequence"},
        ]
        for payload in parts:
            event = SSEEvent(event="completion", data=json.dumps(payload))
            for rewritten in adapter.rehydrate_event(event, pool):
                out += json.loads(rewritten.data)["completion"]
        return out

    expected = f"mail {EMAIL} done"
    for split in range(1, len(full)):
        assert run(split) == expected, f"split at {split}"


def test_legacy_routes_and_note_suppression() -> None:
    openai = OpenAIAdapter()
    anthropic = AnthropicAdapter()
    assert openai.matches("POST", "/v1/completions") is RouteKind.CHAT
    assert anthropic.matches("POST", "/v1/complete") is RouteKind.CHAT
    # Neither legacy body shape can carry the system note.
    assert not openai.wants_system_note(RouteKind.CHAT, "/v1/completions")
    assert not anthropic.wants_system_note(RouteKind.CHAT, "/v1/complete")


def test_azure_files_routes_and_raw_delegation() -> None:
    adapter = AzureOpenAIAdapter()
    assert adapter.matches("POST", "/openai/files") is RouteKind.REDACT_ONLY
    assert adapter.matches("GET", "/openai/files/file_a/content") is RouteKind.CHAT
    assert adapter.matches("GET", "/openai/files/file_a") is RouteKind.NONE
    assert adapter.matches("POST", "/openai/batches") is RouteKind.NONE
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    line = json.dumps({"response": {"body": {"content": f"echo {token}"}}}).encode()
    out = adapter.rehydrate_raw_body("/openai/files/file_a/content", line, Rehydrator(vault))
    assert out is not None and EMAIL.encode() in out


def test_openai_adapter_defers_anthropic_files_by_header() -> None:
    adapter = OpenAIAdapter()
    plain = adapter.matches_request("POST", "/v1/files", {"authorization": "Bearer x"})
    assert plain is RouteKind.REDACT_ONLY
    deferred = adapter.matches_request(
        "POST", "/v1/files", {"anthropic-version": "2023-06-01", "x-api-key": "k"}
    )
    assert deferred is RouteKind.NONE


# --- integration: anthropic-version files traffic reaches anthropic ----------

received: dict[str, Any] = {}


def _fake_upstreams() -> Starlette:
    async def anything(request: Request) -> Response:
        host = request.headers.get("host", "?")
        received.setdefault(host, []).append(request.url.path)
        return JSONResponse({"ok": True})

    return Starlette(routes=[Route("/{path:path}", anything, methods=["GET", "POST"])])


@pytest.mark.anyio
async def test_files_routing_by_header() -> None:
    received.clear()
    config = Config(
        providers={
            **Config().providers,
            "anthropic": ProviderConfig(upstream_base_url="http://up-anthropic"),
            "openai": ProviderConfig(upstream_base_url="http://up-openai"),
        }
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstreams()))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    # Anthropic-flavored files traffic passes through to the anthropic host.
    a = await client.get("/v1/files/f1", headers={"anthropic-version": "2023-06-01"})
    assert a.status_code == 200
    assert received["up-anthropic"] == ["/v1/files/f1"]
    # Plain files metadata still goes to the openai upstream.
    b = await client.get("/v1/files/f1")
    assert b.status_code == 200
    assert received["up-openai"] == ["/v1/files/f1"]
    await client.aclose()
