"""``AuthorizationRequest.model``: the model the UPSTREAM runs the request
with, as far as the request itself says — reported by the matched adapter
(``ProviderAdapter.request_model``, ``WsAdapter.request_model``), never a
value the upstream ignores.

A body ``model`` is reported only on the routes whose upstream reads it; a
model named in the path (Azure deployments, Gemini/Vertex ``models/{m}``,
Claude on Vertex, Bedrock ``/model/{id}`` — percent-encoded ARNs decoded as
the matcher reads them) is the path's, whatever the body says; None where
the request does not name one or the upstream takes it from somewhere the
check cannot see (a Gemini Live setup frame, a multipart form, an OpenAI
realtime session set up by an ``intent``/``call_id``). A gate restricting
models then sees an unknown model, never a chosen one.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from starlette.datastructures import QueryParams

from llm_redact.config import Config, ProviderConfig
from llm_redact.providers import (
    AnthropicAdapter,
    AzureOpenAIAdapter,
    AzureResponsesAdapter,
    BedrockAdapter,
    ClaudeVertexAdapter,
    CohereAdapter,
    GeminiAdapter,
    GeminiOpenAIAdapter,
    GeminiOpenAIResponsesAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    OpenAIResponsesAdapter,
    ProviderAdapter,
    RouteKind,
    VertexAdapter,
)
from llm_redact.providers.base import body_string
from llm_redact.providers.custom import CustomOpenAIAdapter, CustomResponsesAdapter
from llm_redact.proxy import create_app
from llm_redact.realtime import (
    AzureRealtimeWs,
    GeminiLiveWs,
    OpenAIRealtimeWs,
    VertexLiveWs,
    WsAdapter,
)
from test_authorization_seam import AuthorizingGate, _client, _registry

M = {"model": "body-model"}
ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet"
VERTEX = "/v1/projects/p/locations/us-central1"

# (adapter, method, path, parsed body, the model the upstream runs)
HTTP_CASES: list[tuple[ProviderAdapter, str, str, Any, str | None]] = [
    # OpenAI: the body's model on the routes that read it...
    (OpenAIAdapter(), "POST", "/v1/chat/completions", M, "body-model"),
    (OpenAIAdapter(), "POST", "/v1/completions", M, "body-model"),
    (OpenAIAdapter(), "POST", "/v1/embeddings", M, "body-model"),
    (OpenAIAdapter(), "POST", "/v1/images/generations", M, "body-model"),
    (OpenAIAdapter(), "POST", "/v1/images/edits", M, "body-model"),
    (OpenAIAdapter(), "POST", "/v1/audio/speech", M, "body-model"),
    (OpenAIAdapter(), "POST", "/v1/videos", M, "body-model"),
    (OpenAIAdapter(), "POST", "/v1/fine_tuning/jobs", M, "body-model"),
    # ...never where it ignores one: a remix runs the video's own model, a
    # batch its input file's lines, the stores none at all.
    (OpenAIAdapter(), "POST", "/v1/videos/video_1/remix", M, None),
    (OpenAIAdapter(), "POST", "/v1/batches", M, None),
    (OpenAIAdapter(), "POST", "/v1/conversations", M, None),
    (OpenAIAdapter(), "POST", "/v1/vector_stores", M, None),
    (OpenAIAdapter(), "POST", "/v1/fine_tuning/jobs/ftjob-1/cancel", M, None),
    (OpenAIAdapter(), "GET", "/v1/chat/completions/chatcmpl-1", None, None),
    (OpenAIAdapter(), "GET", "/v1/models/gpt-4o", None, None),
    # Not a string, an empty one, an unparsed (multipart) body: none named.
    (OpenAIAdapter(), "POST", "/v1/chat/completions", {"model": 4}, None),
    (OpenAIAdapter(), "POST", "/v1/chat/completions", {"model": ""}, None),
    (OpenAIAdapter(), "POST", "/v1/chat/completions", None, None),
    (OpenAIAdapter(), "POST", "/v1/files", None, None),
    (OpenAIResponsesAdapter(), "POST", "/v1/responses", M, "body-model"),
    (OpenAIResponsesAdapter(), "POST", "/v1/responses/compact", M, "body-model"),
    (OpenAIResponsesAdapter(), "POST", "/v1/responses/input_tokens", M, "body-model"),
    (OpenAIResponsesAdapter(), "GET", "/v1/responses/resp_1", None, None),
    (OpenAIResponsesAdapter(), "DELETE", "/v1/responses/resp_1", M, None),
    # Anthropic.
    (AnthropicAdapter(), "POST", "/v1/messages", M, "body-model"),
    (AnthropicAdapter(), "POST", "/v1/messages/count_tokens", M, "body-model"),
    (AnthropicAdapter(), "POST", "/v1/complete", M, "body-model"),
    (AnthropicAdapter(), "POST", "/v1/messages/batches", M, None),
    (AnthropicAdapter(), "GET", "/v1/messages/batches/msgbatch_1", None, None),
    # Claude on Vertex: the path names it; a body model is ignored.
    (
        ClaudeVertexAdapter(),
        "POST",
        f"{VERTEX}/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict",
        M,
        "claude-sonnet-4-5@20250929",
    ),
    (
        ClaudeVertexAdapter(),
        "POST",
        "/v1/publishers/anthropic/models/claude-haiku-4-5:streamRawPredict",
        {},
        "claude-haiku-4-5",
    ),
    # Azure: a deployment runs whatever the body says; the v1 API and the
    # Responses API read the body (the deployment's name).
    (AzureOpenAIAdapter(), "POST", "/openai/deployments/big/chat/completions", M, "big"),
    (AzureOpenAIAdapter(), "POST", "/openai/deployments/big/completions", M, "big"),
    (AzureOpenAIAdapter(), "POST", "/openai/deployments/emb/embeddings", M, "emb"),
    (AzureOpenAIAdapter(), "POST", "/openai/deployments/img/images/generations", M, "img"),
    (AzureOpenAIAdapter(), "POST", "/openai/deployments/img/images/edits", None, "img"),
    (AzureOpenAIAdapter(), "POST", "/openai/deployments/tts/audio/speech", M, "tts"),
    (AzureOpenAIAdapter(), "POST", "/openai/v1/chat/completions", M, "body-model"),
    (AzureOpenAIAdapter(), "POST", "/openai/v1/embeddings", M, "body-model"),
    (AzureOpenAIAdapter(), "POST", "/openai/fine_tuning/jobs", M, "body-model"),
    (AzureOpenAIAdapter(), "POST", "/openai/v1/fine_tuning/jobs", M, "body-model"),
    (AzureOpenAIAdapter(), "POST", "/openai/v1/vector_stores", M, None),
    (AzureOpenAIAdapter(), "POST", "/openai/batches", M, None),
    (AzureOpenAIAdapter(), "POST", "/openai/v1/files", None, None),
    (AzureOpenAIAdapter(), "GET", "/openai/deployments/big", None, None),
    (AzureResponsesAdapter(), "POST", "/openai/responses", M, "body-model"),
    (AzureResponsesAdapter(), "POST", "/openai/v1/responses", M, "body-model"),
    (AzureResponsesAdapter(), "POST", "/openai/v1/responses/compact", M, "body-model"),
    (AzureResponsesAdapter(), "POST", "/openai/responses/input_tokens", M, "body-model"),
    (AzureResponsesAdapter(), "POST", "/openai/v1/responses/resp_1/cancel", M, None),
    (AzureResponsesAdapter(), "GET", "/openai/v1/responses/resp_1", None, None),
    # Gemini: the path's model (a tuned model as such); a context cache's
    # create reads its body's (a resource name, `models/` dropped alike).
    (GeminiAdapter(), "POST", "/v1beta/models/gemini-2.5-pro:generateContent", M, "gemini-2.5-pro"),
    (
        GeminiAdapter(),
        "POST",
        "/v1/models/gemini-2.5-pro:streamGenerateContent",
        M,
        "gemini-2.5-pro",
    ),
    (
        GeminiAdapter(),
        "POST",
        "/v1beta/models/text-embedding-004:embedContent",
        M,
        "text-embedding-004",
    ),
    (GeminiAdapter(), "POST", "/v1beta/models/g:batchGenerateContent", M, "g"),
    (GeminiAdapter(), "POST", "/v1beta/models/veo-3.0:predictLongRunning", M, "veo-3.0"),
    (
        GeminiAdapter(),
        "POST",
        "/v1beta/tunedModels/my-tune:generateContent",
        M,
        "tunedModels/my-tune",
    ),
    (
        GeminiAdapter(),
        "POST",
        "/v1beta/cachedContents",
        {"model": "models/gemini-2.0-flash-001"},
        "gemini-2.0-flash-001",
    ),
    (GeminiAdapter(), "POST", "/v1beta/cachedContents", {"contents": []}, None),
    (GeminiAdapter(), "POST", "/v1beta/files", M, None),
    (GeminiAdapter(), "GET", "/v1beta/models/gemini-2.5-pro", None, None),
    (GeminiAdapter(), "GET", "/v1beta/batches/b-1", None, None),
    # The Gemini API's OpenAI-compatible surface reads the body.
    (GeminiOpenAIAdapter(), "POST", "/v1beta/openai/chat/completions", M, "body-model"),
    (GeminiOpenAIAdapter(), "POST", "/v1beta/openai/embeddings", M, "body-model"),
    (GeminiOpenAIResponsesAdapter(), "POST", "/v1beta/openai/responses", M, "body-model"),
    # Vertex: a publisher model's id (with or without a project), an
    # endpoint as such; a context cache's create reads its body.
    (
        VertexAdapter(),
        "POST",
        f"{VERTEX}/publishers/google/models/gemini-2.5-pro:generateContent",
        M,
        "gemini-2.5-pro",
    ),
    (
        VertexAdapter(),
        "POST",
        "/v1beta1/publishers/google/models/gemini-2.5-flash:countTokens",
        M,
        "gemini-2.5-flash",
    ),
    (VertexAdapter(), "POST", f"{VERTEX}/endpoints/123:streamGenerateContent", M, "endpoints/123"),
    (
        VertexAdapter(),
        "POST",
        f"{VERTEX}/cachedContents",
        {"model": "projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash-001"},
        "gemini-2.0-flash-001",
    ),
    (VertexAdapter(), "GET", f"{VERTEX}/publishers/google/models/gemini-2.5-pro", None, None),
    (VertexAdapter(), "PATCH", f"{VERTEX}/cachedContents/c-1", M, None),
    # Bedrock: the path's model id — an ARN decoded as the matcher reads
    # it — and StartAsyncInvoke's body `modelId`.
    (BedrockAdapter(), "POST", "/model/amazon.nova-pro-v1:0/converse", M, "amazon.nova-pro-v1:0"),
    (BedrockAdapter(), "POST", f"/model/{ARN}/invoke", {"modelId": "x"}, ARN),
    (BedrockAdapter(), "POST", f"/model/{ARN}/invoke-with-response-stream", {}, ARN),
    (BedrockAdapter(), "POST", "/model/m/converse-stream", {}, "m"),
    (BedrockAdapter(), "POST", "/model/m/count-tokens", {}, "m"),
    (
        BedrockAdapter(),
        "POST",
        "/async-invoke",
        {"modelId": "amazon.nova-reel-v1:1"},
        "amazon.nova-reel-v1:1",
    ),
    (BedrockAdapter(), "POST", "/async-invoke", M, None),
    (BedrockAdapter(), "POST", "/guardrail/g-1/version/1/apply", M, None),
    (BedrockAdapter(), "GET", "/async-invoke", None, None),
    # Cohere: every route it recognizes reads the body.
    (CohereAdapter(), "POST", "/v2/chat", M, "body-model"),
    (CohereAdapter(), "POST", "/v2/embed", M, "body-model"),
    (CohereAdapter(), "POST", "/v2/rerank", M, "body-model"),
    (CohereAdapter(), "POST", "/v1/chat", M, "body-model"),
    (CohereAdapter(), "POST", "/v1/generate", M, "body-model"),
    # Ollama: the inference routes; /api/show is a metadata read.
    (OllamaAdapter(), "POST", "/api/chat", M, "body-model"),
    (OllamaAdapter(), "POST", "/api/generate", M, "body-model"),
    (OllamaAdapter(), "POST", "/api/embed", M, "body-model"),
    (OllamaAdapter(), "POST", "/api/embeddings", M, "body-model"),
    (OllamaAdapter(), "POST", "/api/show", M, None),
    (OllamaAdapter(), "GET", "/api/tags", None, None),
    # Custom providers: the OpenAI rules on the endpoint the tail names.
    (CustomOpenAIAdapter("vllm"), "POST", "/custom/vllm/v1/chat/completions", M, "body-model"),
    (
        CustomOpenAIAdapter("groq"),
        "POST",
        "/custom/groq/openai/v1/chat/completions",
        M,
        "body-model",
    ),
    (CustomOpenAIAdapter("vllm"), "POST", "/custom/vllm/v1/batches", M, None),
    (CustomResponsesAdapter("vllm"), "POST", "/custom/vllm/v1/responses", M, "body-model"),
]


@pytest.mark.parametrize(
    ("adapter", "method", "path", "parsed", "model"),
    HTTP_CASES,
    ids=[f"{type(case[0]).__name__}-{case[1]}-{case[2]}" for case in HTTP_CASES],
)
def test_the_model_an_adapter_reports(
    adapter: ProviderAdapter, method: str, path: str, parsed: Any, model: str | None
) -> None:
    # Every case is a route the adapter recognizes (only those are asked).
    assert adapter.matches(method, path) is not RouteKind.NONE
    assert adapter.request_model(method, path, parsed) == model


def test_every_adapter_has_its_routes_in_the_table() -> None:
    from llm_redact.providers import ALL_ADAPTERS

    covered = {type(case[0]) for case in HTTP_CASES}
    assert set(ALL_ADAPTERS) <= covered
    assert {CustomOpenAIAdapter, CustomResponsesAdapter} <= covered


def test_body_string() -> None:
    assert body_string({"model": "m"}, "model") == "m"
    assert body_string({"model": 4}, "model") is None
    assert body_string({"model": ""}, "model") is None
    assert body_string(["model"], "model") is None
    assert body_string(None, "model") is None


GEMINI_LIVE = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
VERTEX_LIVE = "/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

WS_CASES: list[tuple[WsAdapter, str, str, str | None]] = [
    (OpenAIRealtimeWs(), "/v1/realtime", "model=gpt-realtime", "gpt-realtime"),
    (OpenAIRealtimeWs(), "/v1/realtime", "", None),
    # Repeated: an upstream may read either occurrence.
    (OpenAIRealtimeWs(), "/v1/realtime", "model=a&model=b", None),
    # A session set up elsewhere (a transcription intent, a SIP call).
    (OpenAIRealtimeWs(), "/v1/realtime", "intent=transcription&model=a", None),
    (OpenAIRealtimeWs(), "/v1/realtime", "call_id=rtc_1&model=a", None),
    (OpenAIRealtimeWs(), "/v1/realtime/other", "model=a", None),
    # Azure GA names the deployment `model`; the preview, `deployment`.
    (AzureRealtimeWs(), "/openai/v1/realtime", "model=dep", "dep"),
    (AzureRealtimeWs(), "/openai/realtime", "api-version=v&deployment=dep&model=x", "dep"),
    (AzureRealtimeWs(), "/openai/realtime", "api-version=v&model=x", None),
    (AzureRealtimeWs(), "/openai/v1/realtime", "deployment=dep", None),
    (AzureRealtimeWs(), "/openai/v1/realtime", "model=dep&intent=transcription", None),
    # Gemini Live: the setup frame names it, after the check.
    (GeminiLiveWs(), GEMINI_LIVE, "model=models/gemini-2.5-flash&key=k", None),
    (VertexLiveWs(), VERTEX_LIVE, "model=x", None),
]


@pytest.mark.parametrize(
    ("adapter", "path", "query", "model"),
    WS_CASES,
    ids=[f"{type(case[0]).__name__}-{case[1]}?{case[2]}" for case in WS_CASES],
)
def test_the_model_a_realtime_adapter_reports(
    adapter: WsAdapter, path: str, query: str, model: str | None
) -> None:
    assert adapter.matches(path)
    assert adapter.request_model(path, QueryParams(query)) == model


# --- end to end: the fact the gate is handed ------------------------------------------


class Upstream:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})


def _app(upstream: Upstream) -> Any:
    providers = {
        **Config().providers,
        "azure": ProviderConfig("https://res.openai.azure.com"),
        "gemini": ProviderConfig("https://generativelanguage.googleapis.com"),
        "bedrock": ProviderConfig("https://bedrock-runtime.us-east-1.amazonaws.com"),
    }
    return create_app(Config(providers=providers), upstream_transport=httpx.MockTransport(upstream))


@pytest.mark.parametrize(
    ("path", "headers", "body", "model", "sent"),
    [
        (
            "/openai/deployments/gpt-5-big/chat/completions?api-version=2024-10-21",
            {"api-key": "k"},
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            "gpt-5-big",
            "/openai/deployments/gpt-5-big/chat/completions",
        ),
        (
            "/v1beta/models/gemini-2.5-pro:generateContent",
            {"x-goog-api-key": "k"},
            {"model": "models/gemini-2.0-flash", "contents": [{"parts": [{"text": "hi"}]}]},
            "gemini-2.5-pro",
            "/v1beta/models/gemini-2.5-pro:generateContent",
        ),
        (
            "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile"
            "%2Fus.anthropic.claude-3-5-sonnet/converse",
            {"authorization": "Bearer bedrock-key"},
            {"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
            ARN,
            "inference-profile%2Fus.anthropic.claude-3-5-sonnet/converse",
        ),
    ],
    ids=["azure-deployment", "gemini-path", "bedrock-arn"],
)
async def test_the_gate_is_handed_the_model_the_upstream_runs(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    headers: dict[str, str],
    body: Any,
    model: str,
    sent: str,
) -> None:
    gate = AuthorizingGate()
    _registry(monkeypatch, gate)
    upstream = Upstream()
    async with _client(_app(upstream)) as client:
        response = await client.post(path, json=body, headers=headers)
    assert response.status_code == 200
    (request,) = gate.requests
    assert request.model == model
    # ...exactly what was forwarded.
    (url,) = upstream.urls
    assert sent in url
