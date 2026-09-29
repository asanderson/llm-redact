"""Responses compaction and input-token counting are redacted.

``POST /v1/responses/compact`` and ``POST /v1/responses/input_tokens`` take
a responses.create body, and neither was matched: both passed through with
the whole ``input`` unredacted. Codex CLI compacts long sessions, and what it
sends is the ENTIRE conversation — in clear text, since the client holds the
restored values. Now, on every surface that serves the Responses API (OpenAI,
Azure's ``/openai/v1`` and api-version forms, custom providers, the Gemini
API's OpenAI surface):

- compact is a chat route: its input is redacted, and the note joins its
  ``instructions`` (the compaction must carry every token forward exactly,
  as the requests it condenses do); its answer (``response.compaction``) is
  restored — the retained messages and tool calls, arguments as JSON
  source — while the opaque, encrypted compaction item goes back as sent;
- input_tokens is redact-only: its body is redacted like the request it
  counts, the note included, so the count matches that request (Anthropic
  ``count_tokens``' stance); the count comes back as the provider sent it;
- neither creates a stored object (compaction is stateless), so nothing is
  reported to a session router as one.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig
from llm_redact.providers.azure_openai import AzureResponsesAdapter
from llm_redact.providers.base import SYSTEM_NOTE, RouteKind
from llm_redact.providers.custom import CustomResponsesAdapter, GeminiOpenAIResponsesAdapter
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"
ENCRYPTED = "gAAAAABo-opaque-compaction-state"

# (surface, path prefix, provider, headers)
SURFACES: list[tuple[str, str, str, dict[str, str]]] = [
    ("openai", "/v1", "openai", {"authorization": "Bearer sk-proj-FAKE"}),
    ("azure-v1", "/openai/v1", "azure", {"api-key": "az"}),
    ("azure-api-version", "/openai", "azure", {"api-key": "az"}),
    ("custom", "/custom/lm/v1", "custom:lm", {"authorization": "Bearer sk-proj-FAKE"}),
    ("custom-no-v1", "/custom/lm", "custom:lm", {"authorization": "Bearer sk-proj-FAKE"}),
    ("gemini", "/v1beta/openai", "gemini", {"authorization": "Bearer AIzaFAKE"}),
]
HOSTS = {
    "openai": "api.openai.com",
    "azure": "res.openai.azure.com",
    "custom:lm": "lm.local",
    "gemini": "generativelanguage.googleapis.com",
}


def _config() -> Config:
    providers = dict(Config().providers)
    providers["azure"] = ProviderConfig("https://res.openai.azure.com")
    providers["custom:lm"] = ProviderConfig("http://lm.local")
    return Config(providers=providers)


def test_both_routes_are_matched_on_every_responses_surface() -> None:
    adapters = {
        "/v1": OpenAIResponsesAdapter(),
        "/openai/v1": AzureResponsesAdapter(),
        "/openai": AzureResponsesAdapter(),
        "/custom/lm/v1": CustomResponsesAdapter("lm"),
        "/custom/lm": CustomResponsesAdapter("lm"),
        "/v1beta/openai": GeminiOpenAIResponsesAdapter(),
    }
    for prefix, adapter in adapters.items():
        compact, count = f"{prefix}/responses/compact", f"{prefix}/responses/input_tokens"
        assert adapter.matches("POST", compact) is RouteKind.CHAT, prefix
        assert adapter.matches("POST", count) is RouteKind.REDACT_ONLY, prefix
        # The note joins both: the compaction must keep tokens exactly, and
        # a count must match the request it counts (which carries the note).
        assert adapter.wants_system_note(RouteKind.CHAT, compact), prefix
        assert adapter.wants_system_note(RouteKind.REDACT_ONLY, count), prefix
        # Neither creates a stored object (compaction is stateless).
        for path in (compact, count):
            assert not adapter.tracks_object_ids("POST", path, {"input": "x"}), path
    # A Responses create keeps its own classification, and other methods on
    # these paths stay what they were (a read or delete of a response id).
    responses = OpenAIResponsesAdapter()
    assert responses.matches("POST", "/v1/responses") is RouteKind.CHAT
    assert responses.matches("GET", "/v1/responses/compact") is RouteKind.CHAT
    assert responses.matches("DELETE", "/v1/responses/input_tokens") is RouteKind.REDACT_ONLY
    assert responses.matches("POST", "/v1/responses/compact/x") is RouteKind.NONE
    assert not responses.wants_system_note(RouteKind.REDACT_ONLY, "/v1/responses/resp_1")


class _Upstream:
    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self.answer)


async def _post(app: Any, path: str, headers: dict[str, str], body: Any) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        return await client.post(path, json=body, headers=headers)


def _window() -> list[dict[str, Any]]:
    """A conversation window as the client holds it: restored values."""
    return [
        {"role": "user", "content": f"write to {EMAIL}"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": f"drafting for {EMAIL}"}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "send",
            "arguments": json.dumps({"to": EMAIL}),
        },
        {"type": "function_call_output", "call_id": "call_1", "output": f"sent to {EMAIL}"},
    ]


@pytest.mark.parametrize(("surface", "prefix", "provider", "headers"), SURFACES)
async def test_a_compaction_is_redacted_and_its_window_restored(
    surface: str, prefix: str, provider: str, headers: dict[str, str]
) -> None:
    compacted = {
        "id": "resp_cmp_1",
        "object": "response.compaction",
        "created_at": 1_700_000_000,
        "output": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"write to {TOKEN}"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "send",
                "arguments": json.dumps({"to": TOKEN}),
            },
            {"type": "compaction", "id": "cmp_1", "encrypted_content": ENCRYPTED},
        ],
        "usage": {"input_tokens": 90, "output_tokens": 12, "total_tokens": 102},
    }
    upstream = _Upstream(compacted)
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    body = {"model": "m", "input": _window(), "instructions": "be brief"}
    response = await _post(app, f"{prefix}/responses/compact", headers, body)
    assert response.status_code == 200, response.text
    (sent,) = upstream.requests
    assert sent.url.host == HOSTS[provider]
    forwarded = json.loads(sent.content)
    assert EMAIL not in sent.content.decode()
    assert TOKEN in forwarded["input"][0]["content"]
    assert json.loads(forwarded["input"][2]["arguments"]) == {"to": TOKEN}
    assert forwarded["instructions"].startswith("be brief")
    assert SYSTEM_NOTE in forwarded["instructions"]
    answer = response.json()
    assert answer["output"][0]["content"][0]["text"] == f"write to {EMAIL}"
    assert json.loads(answer["output"][1]["arguments"]) == {"to": EMAIL}
    assert answer["output"][2] == compacted["output"][2]  # opaque: as it came
    (row,) = app.state.proxy.recent
    assert row["provider"] == provider and row["detections"] == {"EMAIL": 4}


@pytest.mark.parametrize(("surface", "prefix", "provider", "headers"), SURFACES)
async def test_an_input_token_count_is_redacted_like_the_request_it_counts(
    surface: str, prefix: str, provider: str, headers: dict[str, str]
) -> None:
    count = {"object": "response.input_tokens", "input_tokens": 123}
    upstream = _Upstream(count)
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    body = {"model": "m", "input": _window(), "tools": []}
    response = await _post(app, f"{prefix}/responses/input_tokens", headers, body)
    assert response.status_code == 200, response.text
    (sent,) = upstream.requests
    assert sent.url.host == HOSTS[provider]
    assert EMAIL not in sent.content.decode()
    forwarded = json.loads(sent.content)
    assert forwarded["instructions"] == SYSTEM_NOTE  # as the counted request carries it
    assert response.json() == count
