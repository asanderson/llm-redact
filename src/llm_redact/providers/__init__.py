from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.azure_openai import AzureOpenAIAdapter, AzureResponsesAdapter
from llm_redact.providers.base import ProviderAdapter, RouteKind
from llm_redact.providers.bedrock import BedrockAdapter
from llm_redact.providers.claude_vertex import ClaudeVertexAdapter
from llm_redact.providers.cohere import CohereAdapter
from llm_redact.providers.gemini import GeminiAdapter
from llm_redact.providers.ollama import OllamaAdapter
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.providers.vertex import VertexAdapter

ALL_ADAPTERS: tuple[type[ProviderAdapter], ...] = (
    AnthropicAdapter,
    OpenAIAdapter,
    OpenAIResponsesAdapter,
    GeminiAdapter,
    VertexAdapter,
    # After VertexAdapter: matches the same host's Claude publisher paths
    # (rawPredict), proven disjoint from Vertex's generateContent verbs.
    ClaudeVertexAdapter,
    # Both name "azure"; Responses first, matcher disjoint from the chat
    # adapter's (responses vs chat/completions|embeddings), proven by test.
    AzureResponsesAdapter,
    AzureOpenAIAdapter,
    BedrockAdapter,
    CohereAdapter,
    OllamaAdapter,
)

__all__ = [
    "ALL_ADAPTERS",
    "AnthropicAdapter",
    "AzureOpenAIAdapter",
    "AzureResponsesAdapter",
    "BedrockAdapter",
    "ClaudeVertexAdapter",
    "CohereAdapter",
    "GeminiAdapter",
    "OllamaAdapter",
    "OpenAIAdapter",
    "OpenAIResponsesAdapter",
    "ProviderAdapter",
    "RouteKind",
    "VertexAdapter",
]
