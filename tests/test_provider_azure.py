"""Azure OpenAI adapter: routing differs; all event handling is inherited."""

import json
from typing import Any

import pytest

from llm_redact.providers.azure_openai import AzureOpenAIAdapter, AzureResponsesAdapter
from llm_redact.providers.base import RouteKind
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.rehydrate import RehydratorPool
from llm_redact.sse import SSEEvent
from llm_redact.vault import InMemoryVault


def test_routing_matrix() -> None:
    adapter = AzureOpenAIAdapter()
    assert adapter.matches("POST", "/openai/deployments/gpt4o/chat/completions") is RouteKind.CHAT
    assert adapter.matches("POST", "/openai/v1/chat/completions") is RouteKind.CHAT
    assert adapter.matches("GET", "/openai/deployments/gpt4o/chat/completions") is RouteKind.NONE
    assert adapter.matches("POST", "/v1/chat/completions") is RouteKind.NONE  # plain OpenAI
    # Redact-only since 0.6.0: previously passed through unredacted.
    assert adapter.matches("POST", "/openai/deployments/gpt4o/embeddings") is RouteKind.REDACT_ONLY
    assert adapter.matches("POST", "/openai/v1/embeddings") is RouteKind.REDACT_ONLY
    # No overlap in the other direction either.
    assert OpenAIAdapter().matches("POST", "/openai/v1/chat/completions") is RouteKind.NONE


def _chunk(*choices: dict[str, Any]) -> SSEEvent:
    return SSEEvent(data=json.dumps({"object": "chat.completion.chunk", "choices": list(choices)}))


@pytest.mark.parametrize("adapter_cls", [OpenAIAdapter, AzureOpenAIAdapter])
def test_streaming_behavior_is_inherited(adapter_cls: type[OpenAIAdapter]) -> None:
    """The canonical split-token stream behaves identically on both."""
    vault = InMemoryVault()
    vault.placeholder_for("EMAIL", "jane@corp.example")
    adapter = adapter_cls()
    pool = RehydratorPool(vault)
    events = [
        _chunk({"index": 0, "delta": {"content": "mail «EMA"}, "finish_reason": None}),
        _chunk({"index": 0, "delta": {"content": "IL_001» done"}, "finish_reason": None}),
        _chunk({"index": 0, "delta": {}, "finish_reason": "stop"}),
        SSEEvent(data="[DONE]"),
    ]
    out: list[SSEEvent] = []
    for event in events:
        out.extend(adapter.rehydrate_event(event, pool))
    text = "".join(
        choice.get("delta", {}).get("content") or ""
        for e in out
        if e.data and e.data != "[DONE]"
        for choice in json.loads(e.data).get("choices", [])
    )
    assert text == "mail jane@corp.example done"
    assert pool.flush_all() == {}


def test_azure_responses_routing() -> None:
    adapter = AzureResponsesAdapter()
    assert adapter.name == "azure"
    # POST on both the api-version form and the v1 preview.
    assert adapter.matches("POST", "/openai/responses") is RouteKind.CHAT
    assert adapter.matches("POST", "/openai/v1/responses") is RouteKind.CHAT
    # Stored response + input-item echoes rehydrated on GET.
    assert adapter.matches("GET", "/openai/responses/resp_abc") is RouteKind.CHAT
    assert adapter.matches("GET", "/openai/v1/responses/resp_abc") is RouteKind.CHAT
    assert adapter.matches("GET", "/openai/responses/resp_abc/input_items") is RouteKind.CHAT
    # DELETE and unrelated paths pass through.
    assert adapter.matches("DELETE", "/openai/responses/resp_abc") is RouteKind.NONE
    assert adapter.matches("POST", "/v1/responses") is RouteKind.NONE  # plain OpenAI Responses


def test_azure_responses_and_chat_matchers_disjoint() -> None:
    """The two azure adapters must never both claim a path — responses vs
    chat/completions|embeddings|files."""
    responses = AzureResponsesAdapter()
    chat = AzureOpenAIAdapter()
    paths = [
        ("POST", "/openai/responses"),
        ("POST", "/openai/v1/responses"),
        ("GET", "/openai/responses/resp_abc"),
        ("POST", "/openai/deployments/gpt4o/chat/completions"),
        ("POST", "/openai/v1/chat/completions"),
        ("POST", "/openai/v1/embeddings"),
        ("POST", "/openai/files"),
    ]
    for method, path in paths:
        claims = [a.matches(method, path) is not RouteKind.NONE for a in (responses, chat)]
        assert claims.count(True) <= 1, f"both azure adapters claim {method} {path}"
    # api.openai.com Responses is untouched by the azure responses matcher.
    assert OpenAIResponsesAdapter().matches("POST", "/openai/responses") is RouteKind.NONE


def test_azure_responses_streaming_inherited() -> None:
    """A split-token Responses stream rehydrates identically to OpenAI's."""
    vault = InMemoryVault()
    vault.placeholder_for("EMAIL", "jane@corp.example")
    adapter = AzureResponsesAdapter()
    pool = RehydratorPool(vault)
    events = [
        SSEEvent(
            event="response.output_text.delta",
            data=json.dumps(
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "mail «EMA",
                }
            ),
        ),
        SSEEvent(
            event="response.output_text.delta",
            data=json.dumps(
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "IL_001» done",
                }
            ),
        ),
    ]
    out: list[SSEEvent] = []
    for event in events:
        out.extend(adapter.rehydrate_event(event, pool))
    text = "".join(json.loads(e.data)["delta"] for e in out if e.data)
    assert text == "mail jane@corp.example done"
