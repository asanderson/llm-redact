"""Hand-authored request shapes for the NER prefetch suites: one or more per
provider adapter (every class in ``ALL_ADAPTERS``, plus a custom upstream),
each a body the adapter walks with names for a fake NER model to find,
values for the regex rules, and the parts of the walk a collecting pass must
reproduce exactly: system prompts and instructions, tool calls and tool
results, MCP blocks to an exempt server and provider-directed MCP config,
verbatim and label fields, a Bedrock count-tokens blob, repeated strings.

Shared by the adapter-level collecting-pass test (test_ner_prefetch.py) and
the end-to-end prefetch-on/off comparison through the real app.
"""

import base64
import json
from dataclasses import dataclass, field
from typing import Any

from llm_redact.config import Config, ProviderConfig

EMAIL = "jane.doe@corp.example"
NAME = "Jane Doe"
OTHER = "Bob Jones"
EXEMPT_SERVER = "exempt-srv"

OPENAI_HEADERS = {"authorization": "Bearer sk-proj-FAKEFAKEFAKE"}
ANTHROPIC_HEADERS = {"x-api-key": "sk-ant-api03-FAKEFAKE", "anthropic-version": "2023-06-01"}
GOOGLE_HEADERS = {"x-goog-api-key": "AIzaFAKEFAKEFAKE"}


@dataclass(frozen=True)
class Shape:
    """One request: the adapter class that must claim it (None: the custom
    upstream's), its method and path (query apart), the headers that
    attribute it to its provider, and its JSON body."""

    id: str
    adapter: str
    method: str
    path: str
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    query: str = ""


def _chat_messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": f"You help {NAME}."},
        {"role": "user", "content": f"Mail {NAME} at {EMAIL}, cc {OTHER}."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "send",
                        "arguments": json.dumps({"to": EMAIL, "who": NAME}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": f"sent to {NAME}"},
        {"role": "user", "content": f"Mail {NAME} at {EMAIL}, cc {OTHER}."},
    ]


def _anthropic_messages() -> dict[str, Any]:
    return {
        "model": "claude-x",
        "max_tokens": 64,
        "system": f"Assist {NAME}.",
        "mcp_servers": [
            {"type": "url", "url": "https://mcp.example", "authorization_token": EMAIL}
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Who is {OTHER}? Mail {EMAIL}."},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [{"type": "text", "text": f"{NAME} found"}],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "mcp_tool_use",
                        "id": "mcptu_1",
                        "name": "lookup",
                        "server_name": EXEMPT_SERVER,
                        "input": {"who": NAME},
                    },
                    {
                        "type": "mcp_tool_use",
                        "id": "mcptu_2",
                        "name": "lookup",
                        "server_name": "other-srv",
                        "input": {"who": OTHER},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "mcp_tool_result",
                        "tool_use_id": "mcptu_1",
                        "content": [{"type": "text", "text": f"{NAME} is exempt"}],
                    },
                    {
                        "type": "mcp_tool_result",
                        "tool_use_id": "mcptu_2",
                        "content": [{"type": "text", "text": f"{OTHER} is not"}],
                    },
                ],
            },
        ],
    }


def _responses_body() -> dict[str, Any]:
    return {
        "model": "m",
        "instructions": f"You assist {NAME}.",
        "tools": [
            {
                "type": "mcp",
                "server_label": "remote",
                "server_url": "https://mcp.example",
                "headers": {"Authorization": f"Bearer {EMAIL}"},
            },
            {"type": "function", "name": "send", "parameters": {"type": "object"}},
        ],
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": f"Mail {NAME} {EMAIL}"}]},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "send",
                "arguments": json.dumps({"to": OTHER}),
            },
            {"type": "function_call_output", "call_id": "c1", "output": f"done for {OTHER}"},
            {
                "type": "mcp_call",
                "server_label": EXEMPT_SERVER,
                "name": "lookup",
                "arguments": json.dumps({"who": NAME}),
                "output": f"{NAME} exempt",
            },
        ],
    }


def _gemini_body() -> dict[str, Any]:
    return {
        "systemInstruction": {"parts": [{"text": f"Help {NAME}."}]},
        "contents": [
            {"role": "user", "parts": [{"text": f"Mail {NAME} at {EMAIL}"}]},
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "send", "args": {"to": EMAIL, "who": OTHER}}}],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "send", "response": {"ok": f"sent {OTHER}"}}},
                    {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}},
                ],
            },
        ],
    }


def _bedrock_count_tokens() -> dict[str, Any]:
    inner = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "system": f"Help {NAME}",
        "messages": [{"role": "user", "content": f"Mail {OTHER} {EMAIL}"}],
    }
    blob = base64.b64encode(json.dumps(inner).encode()).decode()
    return {"input": {"invokeModel": {"body": blob}}}


VERTEX_MODELS = "/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-pro"
VERTEX_CLAUDE = "/v1/projects/p/locations/us-east5/publishers/anthropic/models/claude-x"

SHAPES: tuple[Shape, ...] = (
    Shape(
        "anthropic-messages",
        "AnthropicAdapter",
        "POST",
        "/v1/messages",
        _anthropic_messages(),
        ANTHROPIC_HEADERS,
    ),
    Shape(
        "anthropic-batch",
        "AnthropicAdapter",
        "POST",
        "/v1/messages/batches",
        {
            "requests": [
                {
                    "custom_id": "r1",
                    "params": {
                        "model": "claude-x",
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": f"{NAME} {EMAIL}"}],
                    },
                }
            ]
        },
        ANTHROPIC_HEADERS,
    ),
    Shape(
        "openai-chat",
        "OpenAIAdapter",
        "POST",
        "/v1/chat/completions",
        {"model": "m", "messages": _chat_messages()},
        OPENAI_HEADERS,
    ),
    Shape(
        "openai-embeddings",
        "OpenAIAdapter",
        "POST",
        "/v1/embeddings",
        {"model": "m", "input": [f"{NAME}", f"{EMAIL} and {OTHER}"]},
        OPENAI_HEADERS,
    ),
    Shape(
        "openai-fine-tuning",
        "OpenAIAdapter",
        "POST",
        "/v1/fine_tuning/jobs",
        {
            "model": "gpt-4o-mini",
            "training_file": "file-abc",
            "suffix": "my-model",
            "metadata": {"owner": NAME},
            "method": {
                "type": "reinforcement",
                "reinforcement": {
                    "grader": {
                        "type": "multi",
                        "name": f"grader for {OTHER}",
                        "graders": {"g1": {"type": "string_check", "name": f"check {NAME}"}},
                    }
                },
            },
        },
        OPENAI_HEADERS,
    ),
    Shape(
        "openai-vector-store",
        "OpenAIAdapter",
        "POST",
        "/v1/vector_stores",
        {"name": f"notes of {NAME}", "file_ids": ["file-1"], "metadata": {"by": OTHER}},
        OPENAI_HEADERS,
    ),
    Shape(
        "openai-responses",
        "OpenAIResponsesAdapter",
        "POST",
        "/v1/responses",
        _responses_body(),
        OPENAI_HEADERS,
    ),
    Shape(
        "gemini-generate",
        "GeminiAdapter",
        "POST",
        "/v1beta/models/gemini-2.5-pro:generateContent",
        _gemini_body(),
        GOOGLE_HEADERS,
    ),
    Shape(
        "gemini-cache",
        "GeminiAdapter",
        "POST",
        "/v1beta/cachedContents",
        {"model": "models/gemini-2.5-pro", "contents": [{"parts": [{"text": f"{NAME} {EMAIL}"}]}]},
        GOOGLE_HEADERS,
    ),
    Shape(
        "gemini-openai-chat",
        "GeminiOpenAIAdapter",
        "POST",
        "/v1beta/openai/chat/completions",
        {"model": "m", "messages": _chat_messages()},
        {"authorization": "Bearer AIzaFAKE"},
    ),
    Shape(
        "gemini-openai-responses",
        "GeminiOpenAIResponsesAdapter",
        "POST",
        "/v1beta/openai/responses",
        _responses_body(),
        {"authorization": "Bearer AIzaFAKE"},
    ),
    Shape(
        "vertex-generate",
        "VertexAdapter",
        "POST",
        VERTEX_MODELS + ":generateContent",
        _gemini_body(),
        {"authorization": "Bearer ya29.FAKE"},
    ),
    Shape(
        "vertex-claude",
        "ClaudeVertexAdapter",
        "POST",
        VERTEX_CLAUDE + ":rawPredict",
        {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 5,
            "system": f"Help {NAME}",
            "messages": [{"role": "user", "content": f"Mail {OTHER} at {EMAIL}"}],
        },
        {"authorization": "Bearer ya29.FAKE"},
    ),
    Shape(
        "azure-responses",
        "AzureResponsesAdapter",
        "POST",
        "/openai/v1/responses",
        _responses_body(),
        {"api-key": "az"},
    ),
    Shape(
        "azure-chat",
        "AzureOpenAIAdapter",
        "POST",
        "/openai/deployments/d/chat/completions",
        {"messages": _chat_messages()},
        {"api-key": "az"},
        query="api-version=2024-10-21",
    ),
    Shape(
        "bedrock-converse",
        "BedrockAdapter",
        "POST",
        "/model/anthropic.claude-x/converse",
        {
            "system": [{"text": f"Help {NAME}"}],
            "messages": [
                {"role": "user", "content": [{"text": f"Mail {OTHER} {EMAIL}"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "t1", "name": "f", "input": {"q": NAME}}}
                    ],
                },
            ],
        },
        {"authorization": "Bearer ABSK"},
    ),
    Shape(
        "bedrock-count-tokens",
        "BedrockAdapter",
        "POST",
        "/model/anthropic.claude-x/count-tokens",
        _bedrock_count_tokens(),
        {"authorization": "Bearer ABSK"},
    ),
    Shape(
        "cohere-v2",
        "CohereAdapter",
        "POST",
        "/v2/chat",
        {
            "model": "command-r",
            "messages": [
                {"role": "system", "content": f"Help {NAME}"},
                {"role": "user", "content": f"Mail {OTHER} at {EMAIL}"},
            ],
        },
        {"authorization": "Bearer co-key"},
    ),
    Shape(
        "cohere-v1",
        "CohereAdapter",
        "POST",
        "/v1/chat",
        {
            "message": f"Who is {NAME}?",
            "preamble": f"Assist {OTHER}",
            "chat_history": [{"role": "USER", "message": EMAIL}],
        },
        {"authorization": "Bearer co-key"},
    ),
    Shape(
        "ollama-chat",
        "OllamaAdapter",
        "POST",
        "/api/chat",
        {
            "model": "llama3",
            "messages": [
                {"role": "system", "content": f"Help {NAME}"},
                {"role": "user", "content": f"Mail {OTHER} at {EMAIL}"},
            ],
        },
    ),
    Shape(
        "ollama-generate",
        "OllamaAdapter",
        "POST",
        "/api/generate",
        {"model": "llama3", "system": f"Help {NAME}", "prompt": f"{OTHER} {EMAIL}"},
    ),
    Shape(
        "custom-chat",
        "CustomOpenAIAdapter",
        "POST",
        "/custom/lm/v1/chat/completions",
        {"model": "m", "messages": _chat_messages()},
        OPENAI_HEADERS,
    ),
)


def config(**overrides: Any) -> Config:
    """Every provider a shape addresses, configured (the cloud ones need an
    upstream), redacting with the system note on."""
    providers = dict(Config().providers)
    providers["azure"] = ProviderConfig("https://res.openai.azure.com")
    providers["bedrock"] = ProviderConfig("https://bedrock-runtime.us-east-1.amazonaws.com")
    providers["vertex"] = ProviderConfig("https://us-central1-aiplatform.googleapis.com")
    providers["custom:lm"] = ProviderConfig("http://lm.local")
    return Config(providers=providers, **overrides)
