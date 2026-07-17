"""Named custom OpenAI-compatible upstreams under /custom/NAME/."""

import json
import tomllib
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from llm_redact.config import Config, ConfigError, ProviderConfig, parse_config
from llm_redact.config_write import emit_config_toml
from llm_redact.providers.base import RouteKind
from llm_redact.providers.custom import CustomOpenAIAdapter, CustomResponsesAdapter
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"


# --- config -----------------------------------------------------------------


def test_parse_nested_and_flat_forms() -> None:
    nested = parse_config(
        {"providers": {"custom": {"vllm": {"upstream_base_url": "http://127.0.0.1:8000"}}}},
        "<test>",
    )
    assert nested.providers["custom:vllm"].upstream_base_url == "http://127.0.0.1:8000"
    assert nested.providers["custom:vllm"].enabled
    flat = parse_config(
        {"providers": {"custom:vllm": {"upstream_base_url": "http://127.0.0.1:8000"}}},
        "<test>",
    )
    assert flat.providers["custom:vllm"] == nested.providers["custom:vllm"]


def test_parse_rejects_bad_custom_entries() -> None:
    with pytest.raises(ConfigError, match="must match"):
        parse_config(
            {"providers": {"custom": {"Bad Name": {"upstream_base_url": "http://x"}}}}, "<t>"
        )
    with pytest.raises(ConfigError, match="upstream_base_url is required"):
        parse_config({"providers": {"custom": {"vllm": {}}}}, "<t>")
    with pytest.raises(ConfigError, match="named subtables"):
        parse_config({"providers": {"custom": {"vllm": "not-a-table"}}}, "<t>")
    # Unknown built-in names keep their error.
    with pytest.raises(ConfigError, match="unknown provider"):
        parse_config({"providers": {"nonsense": {"upstream_base_url": "http://x"}}}, "<t>")


def test_emitter_round_trips_custom_providers() -> None:
    config = parse_config(
        {
            "providers": {
                "custom": {
                    "vllm": {"upstream_base_url": "http://127.0.0.1:8000"},
                    "lmstudio": {"upstream_base_url": "http://127.0.0.1:1234", "enabled": False},
                }
            }
        },
        "<test>",
    )
    text = emit_config_toml(config)
    assert "[providers.custom.vllm]" in text
    reparsed = parse_config(tomllib.loads(text), "<reparse>")
    assert reparsed.providers == config.providers


# --- adapter prefix stripping -------------------------------------------------


def test_adapter_matches_only_under_its_prefix() -> None:
    adapter = CustomOpenAIAdapter("vllm")
    assert adapter.name == "custom:vllm"
    assert adapter.matches("POST", "/custom/vllm/v1/chat/completions") is RouteKind.CHAT
    assert adapter.matches("POST", "/custom/vllm/v1/embeddings") is RouteKind.REDACT_ONLY
    assert adapter.matches("POST", "/custom/other/v1/chat/completions") is RouteKind.NONE
    assert adapter.matches("POST", "/v1/chat/completions") is RouteKind.NONE
    responses = CustomResponsesAdapter("vllm")
    assert responses.matches("POST", "/custom/vllm/v1/responses") is RouteKind.CHAT


def test_adapter_matches_non_v1_base_paths() -> None:
    # OpenAI-compatible upstreams serve varied base paths; without path
    # normalization these matched NOTHING and forwarded UNREDACTED.
    adapter = CustomOpenAIAdapter("groq")
    # Base path carries /openai/v1 (Groq), /api/v1 (OpenRouter), etc.
    assert adapter.matches("POST", "/custom/groq/openai/v1/chat/completions") is RouteKind.CHAT
    assert adapter.matches("POST", "/custom/groq/api/v1/embeddings") is RouteKind.REDACT_ONLY
    # Tool config put /v1 in upstream_base_url, so the inner path omits it.
    assert adapter.matches("POST", "/custom/groq/chat/completions") is RouteKind.CHAT
    # A genuinely unknown tail still falls through to pass-through.
    assert adapter.matches("POST", "/custom/groq/openai/v1/models") is RouteKind.NONE
    assert adapter.matches("POST", "/custom/groq/healthz") is RouteKind.NONE
    responses = CustomResponsesAdapter("fireworks")
    assert responses.matches("POST", "/custom/fireworks/inference/v1/responses") is RouteKind.CHAT


# --- integration ---------------------------------------------------------------

received: dict[str, Any] = {}


def _fake_upstreams() -> Starlette:
    """One app serving two 'hosts'; requests record which Host they hit."""

    async def chat(request: Request) -> Response:
        host = request.headers.get("host", "?")
        body = await request.json()
        received.setdefault(host, []).append((request.url.path, body))
        text = body["messages"][-1]["content"]
        token = next((w for w in text.split() if w.startswith("«")), "none")
        return JSONResponse(
            {"choices": [{"message": {"role": "assistant", "content": f"echo {token}"}}]}
        )

    async def anything(request: Request) -> Response:
        host = request.headers.get("host", "?")
        received.setdefault(host, []).append((request.url.path, None))
        return JSONResponse({"object": "list", "data": []})

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/{path:path}", anything, methods=["GET", "POST"]),
        ]
    )


@pytest.fixture
def client() -> httpx.AsyncClient:
    received.clear()
    config = Config(
        providers={
            **Config().providers,
            "custom:vllm": ProviderConfig(upstream_base_url="http://up-vllm"),
            "custom:lmstudio": ProviderConfig(upstream_base_url="http://up-lmstudio"),
            "custom:dead": ProviderConfig(upstream_base_url="http://up-dead", enabled=False),
        }
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstreams()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


@pytest.mark.anyio
async def test_two_custom_upstreams_side_by_side(client: httpx.AsyncClient) -> None:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": f"mail {EMAIL} now"}],
    }
    a = await client.post("/custom/vllm/v1/chat/completions", json=body)
    b = await client.post("/custom/lmstudio/v1/chat/completions", json=body)
    assert a.status_code == b.status_code == 200
    # Each upstream saw its own traffic, prefix-stripped and redacted.
    for host in ("up-vllm", "up-lmstudio"):
        path, seen = received[host][0]
        assert path == "/v1/chat/completions"
        flat = json.dumps(seen, ensure_ascii=False)
        assert EMAIL not in flat
        assert "«EMAIL_001»" in flat
    # And the client got the original value back from both.
    assert f"echo {EMAIL}" == a.json()["choices"][0]["message"]["content"]
    assert f"echo {EMAIL}" == b.json()["choices"][0]["message"]["content"]


@pytest.mark.anyio
async def test_passthrough_subpath_reaches_the_addressed_upstream(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/custom/vllm/v1/models")
    assert response.status_code == 200
    path, _body = received["up-vllm"][0]
    assert path == "/v1/models"  # prefix stripped on pass-through too


@pytest.mark.anyio
async def test_disabled_and_unknown_custom_fail_closed(client: httpx.AsyncClient) -> None:
    disabled = await client.post("/custom/dead/v1/chat/completions", json={"messages": []})
    assert disabled.status_code == 502
    assert "disabled" in disabled.text
    unknown = await client.post("/custom/nope/v1/chat/completions", json={"messages": []})
    assert unknown.status_code == 502
    assert "custom" in unknown.text
    assert "up-dead" not in received and "up-nope" not in received
