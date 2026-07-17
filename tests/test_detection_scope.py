"""Per-provider detection toggles ([providers.NAME] detection = false)
and per-MCP-server exemptions ([detection.mcp] exempt_servers)."""

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
from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.base import (
    restore_exempt_mcp_blocks,
    stash_exempt_mcp_blocks,
)
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.proxy import create_app
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault


def _redactor() -> Redactor:
    config = DetectionConfig()
    return Redactor(build_detectors(config), InMemoryVault(), build_allowlist(config))


EMAIL = "jane.doe@corp.example"
EMAIL_2 = "bob.roe@corp.example"
EMAIL_3 = "eve.poe@corp.example"


# --- config -----------------------------------------------------------------


def test_parse_provider_detection_flag() -> None:
    config = parse_config(
        {
            "providers": {
                "ollama": {"detection": False},
                "custom": {"vllm": {"upstream_base_url": "http://x", "detection": False}},
            }
        },
        "<test>",
    )
    assert config.providers["ollama"].detection is False
    assert config.providers["ollama"].enabled is True
    assert config.providers["custom:vllm"].detection is False
    assert config.providers["anthropic"].detection is True  # untouched default


def test_parse_mcp_exempt_servers() -> None:
    config = parse_config(
        {"detection": {"mcp": {"exempt_servers": ["zeta", "alpha", "alpha"]}}}, "<test>"
    )
    # Sorted + deduplicated: canonical equality, like modes.
    assert config.detection.mcp_exempt_servers == ("alpha", "zeta")
    assert parse_config({}, "<test>").detection.mcp_exempt_servers == ()


def test_parse_rejects_bad_mcp_sections() -> None:
    with pytest.raises(ConfigError, match="non-empty strings"):
        parse_config({"detection": {"mcp": {"exempt_servers": [""]}}}, "<t>")
    with pytest.raises(ConfigError, match="non-empty strings"):
        parse_config({"detection": {"mcp": {"exempt_servers": "internal"}}}, "<t>")
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config({"detection": {"mcp": {"servers": []}}}, "<t>")
    with pytest.raises(ConfigError, match="must be a table"):
        parse_config({"detection": {"mcp": []}}, "<t>")


def test_emitter_round_trips_both_settings() -> None:
    config = parse_config(
        {
            "providers": {"ollama": {"detection": False}},
            "detection": {"mcp": {"exempt_servers": ["internal"]}},
        },
        "<test>",
    )
    text = emit_config_toml(config)
    assert "detection = false" in text
    assert "[detection.mcp]" in text
    reparsed = parse_config(tomllib.loads(text), "<reparse>")
    assert reparsed.providers["ollama"].detection is False
    assert reparsed.detection.mcp_exempt_servers == ("internal",)


# --- stash/restore unit behavior ---------------------------------------------


def test_stash_is_identity_without_exempt_blocks() -> None:
    body = {"messages": [{"role": "user", "content": f"mail {EMAIL}"}]}
    assert stash_exempt_mcp_blocks(body, frozenset({"internal"})) == body


def test_restore_decides_on_original_not_sentinel_contents() -> None:
    # A body that HAPPENS to contain sentinel-shaped user data must come
    # through untouched — restore recomputes the predicate on the original.
    body = {"content": [{"type": "mcp_exempt_stash"}, {"type": "text", "text": "hi"}]}
    stashed = stash_exempt_mcp_blocks(body, frozenset({"internal"}))
    assert stashed == body
    assert restore_exempt_mcp_blocks(body, stashed, frozenset({"internal"})) == body


def _anthropic_mcp_body() -> dict[str, Any]:
    return {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "mcp_tool_use",
                        "id": "use_1",
                        "server_name": "internal",
                        "name": "lookup",
                        "input": {"email": EMAIL},
                    },
                    {
                        # Result blocks carry no server name: exempt only via
                        # the tool_use_id -> exempt mcp_tool_use correlation.
                        "type": "mcp_tool_result",
                        "tool_use_id": "use_1",
                        "content": [{"type": "text", "text": f"found {EMAIL_2}"}],
                    },
                    {"type": "text", "text": f"also mail {EMAIL_3}"},
                ],
            }
        ],
    }


def test_anthropic_exempt_server_blocks_bypass_detection() -> None:
    adapter = AnthropicAdapter()
    prepared = adapter.prepare_request(
        _anthropic_mcp_body(),
        _redactor(),
        inject_note=False,
        mcp_exempt=frozenset({"internal"}),
    )
    content = prepared["messages"][0]["content"]
    assert content[0]["input"] == {"email": EMAIL}  # exempt use: verbatim
    assert content[1]["content"][0]["text"] == f"found {EMAIL_2}"  # correlated result
    assert EMAIL_3 not in content[2]["text"]  # ordinary content still redacted
    assert "«EMAIL_001»" in content[2]["text"]
    flat = json.dumps(prepared, ensure_ascii=False)
    assert "mcp_exempt_stash" not in flat  # the sentinel never leaks


def test_anthropic_non_exempt_server_still_redacted() -> None:
    adapter = AnthropicAdapter()
    prepared = adapter.prepare_request(
        _anthropic_mcp_body(),
        _redactor(),
        inject_note=False,
        mcp_exempt=frozenset({"some-other-server"}),
    )
    flat = json.dumps(prepared, ensure_ascii=False)
    assert EMAIL not in flat and EMAIL_2 not in flat and EMAIL_3 not in flat


def test_uncorrelatable_result_block_fails_closed() -> None:
    adapter = AnthropicAdapter()
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "mcp_tool_result",
                        "tool_use_id": "use_unknown",
                        "content": [{"type": "text", "text": f"found {EMAIL}"}],
                    }
                ],
            }
        ],
    }
    prepared = adapter.prepare_request(
        body, _redactor(), inject_note=False, mcp_exempt=frozenset({"internal"})
    )
    assert EMAIL not in json.dumps(prepared, ensure_ascii=False)


def test_note_injection_composes_with_exemption() -> None:
    adapter = AnthropicAdapter()
    prepared = adapter.prepare_request(
        _anthropic_mcp_body(),
        _redactor(),
        inject_note=True,
        mcp_exempt=frozenset({"internal"}),
    )
    assert "system" in prepared  # EMAIL_3 was redacted, so the note landed
    assert prepared["messages"][0]["content"][0]["input"] == {"email": EMAIL}


def test_responses_mcp_call_exempt_by_server_label() -> None:
    adapter = OpenAIResponsesAdapter()
    arguments = json.dumps({"q": EMAIL})
    body = {
        "model": "m",
        "input": [
            {
                "type": "mcp_call",
                "id": "call_1",
                "server_label": "internal",
                "name": "lookup",
                "arguments": arguments,
                "output": f"found {EMAIL_2}",
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"mail {EMAIL_3}"}],
            },
        ],
    }
    prepared = adapter.prepare_request(
        body, _redactor(), inject_note=False, mcp_exempt=frozenset({"internal"})
    )
    assert prepared["input"][0]["arguments"] == arguments
    assert prepared["input"][0]["output"] == f"found {EMAIL_2}"
    assert EMAIL_3 not in json.dumps(prepared["input"][1], ensure_ascii=False)


# --- integration ---------------------------------------------------------------

received: dict[str, Any] = {}


def _fake_upstream() -> Starlette:
    async def messages(request: Request) -> Response:
        received["anthropic"] = await request.json()
        return JSONResponse({"content": [{"type": "text", "text": "ok"}]})

    async def chat(request: Request) -> Response:
        body = await request.json()
        received["openai"] = body
        text = body["messages"][-1]["content"]
        token = next((w for w in text.split() if w.startswith("«")), "«EMAIL_001»")
        return JSONResponse(
            {"choices": [{"message": {"role": "assistant", "content": f"echo {token}"}}]}
        )

    return Starlette(
        routes=[
            Route("/v1/messages", messages, methods=["POST"]),
            Route("/v1/chat/completions", chat, methods=["POST"]),
        ]
    )


@pytest.fixture
def client() -> httpx.AsyncClient:
    received.clear()
    config = Config(
        providers={
            **Config().providers,
            "anthropic": ProviderConfig(upstream_base_url="http://upstream"),
            "openai": ProviderConfig(upstream_base_url="http://upstream", detection=False),
        },
        detection=DetectionConfig(modes=(("private_key", "block"),)),
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    # A loopback base_url: the /config editor endpoint is host-check gated.
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


KEY_BLOCK = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAB3NzaC1yc2E\n-----END OPENSSH PRIVATE KEY-----"
)


@pytest.mark.anyio
async def test_detection_off_forwards_verbatim_but_rehydrates(client: httpx.AsyncClient) -> None:
    # Seed the vault through the detection-ON provider.
    seeded = await client.post(
        "/v1/messages",
        json={"model": "m", "messages": [{"role": "user", "content": f"mail {EMAIL} now"}]},
    )
    assert seeded.status_code == 200
    assert EMAIL not in json.dumps(received["anthropic"], ensure_ascii=False)

    # The detection-OFF provider forwards the same value untouched — and
    # without a note (the body must be byte-identical).
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "mail «EMAIL_001» now"}]},
    )
    assert response.status_code == 200
    seen = json.dumps(received["openai"], ensure_ascii=False)
    assert "privacy tokens" not in seen  # no system note injected
    # Rehydration stays active: the upstream echoed the placeholder and the
    # client got the original back.
    assert response.json()["choices"][0]["message"]["content"] == f"echo {EMAIL}"


@pytest.mark.anyio
async def test_detection_off_forwards_values_and_skips_block_mode(
    client: httpx.AsyncClient,
) -> None:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": f"key {KEY_BLOCK} mail {EMAIL}"}],
    }
    # detection = false: no block, no redaction — the value reaches the
    # upstream as-is (the deliberate, documented off-switch).
    response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    seen = received["openai"]["messages"][0]["content"]
    assert EMAIL in seen and KEY_BLOCK in seen

    # The same request through the detection-ON provider is blocked.
    blocked = await client.post("/v1/messages", json={"model": "m", "messages": body["messages"]})
    assert blocked.status_code == 400


@pytest.mark.anyio
async def test_status_reports_detection_off_providers(client: httpx.AsyncClient) -> None:
    status = (await client.get("/__llm-redact/status")).json()
    assert status["providers_detection_off"] == ["openai"]
    assert status["mcp_exempt_servers"] == 0


@pytest.mark.anyio
async def test_editor_view_carries_both_settings(client: httpx.AsyncClient) -> None:
    payload = (await client.get("/__llm-redact/config")).json()
    assert payload["editable"]["providers"]["openai"]["detection"] is False
    assert payload["editable"]["providers"]["anthropic"]["detection"] is True
    assert payload["editable"]["detection"]["mcp"] == {"exempt_servers": []}


@pytest.mark.anyio
async def test_mcp_exemption_end_to_end() -> None:
    received.clear()
    config = Config(
        providers={**Config().providers, "anthropic": ProviderConfig("http://upstream")},
        detection=DetectionConfig(mcp_exempt_servers=("internal",)),
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    response = await client.post("/v1/messages", json=_anthropic_mcp_body())
    assert response.status_code == 200
    seen = received["anthropic"]["messages"][0]["content"]
    assert seen[0]["input"] == {"email": EMAIL}
    assert seen[1]["content"][0]["text"] == f"found {EMAIL_2}"
    assert EMAIL_3 not in seen[2]["text"]
    await client.aclose()
