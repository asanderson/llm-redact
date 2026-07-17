"""Azure OpenAI adapter: OpenAI chat completions on Azure's path shapes.

The wire format (request body, response body, SSE event stream, tool-call
argument deltas, [DONE] sentinel) is identical to OpenAI chat completions —
everything is inherited. What differs is routing: Azure paths carry a
deployment segment (or the newer /openai/v1 preview form), the api-version
rides the query string (passed through untouched like all queries), and the
upstream is the customer's own resource URL, so [providers.azure] has no
default and the proxy answers 502 until it is configured.

A separate adapter (rather than loosening OpenAIAdapter's matcher) because
upstream selection is keyed on the adapter name: a widened openai matcher
would route Azure traffic to api.openai.com.
"""

import re

from llm_redact.providers.base import RouteKind
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.rehydrate import Rehydrator

_AZURE_PATH = re.compile(r"/openai/(?:deployments/[^/]+|v1)/(chat/completions|embeddings)")
_AZURE_FILE_CONTENT = re.compile(r"/openai/files/[^/]+/content")
# Azure Responses rides both the api-version form (/openai/responses) and the
# newer v1 preview (/openai/v1/responses); GET fetches a stored response and
# its input-item echoes, both of which must be rehydrated back to the client.
_AZURE_RESPONSES = re.compile(r"/openai/(?:v1/)?responses")
_AZURE_RESPONSE_ID = re.compile(r"/openai/(?:v1/)?responses/[^/]+")
_AZURE_RESPONSE_INPUT_ITEMS = re.compile(r"/openai/(?:v1/)?responses/[^/]+/input_items")


class AzureOpenAIAdapter(OpenAIAdapter):
    name = "azure"

    def matches(self, method: str, path: str) -> RouteKind:
        # Files + Batches ride Azure's own path shapes (api-version stays
        # in the query, untouched like every query); the multipart/JSONL
        # hooks are inherited from OpenAIAdapter verbatim.
        if method == "POST" and path == "/openai/files":
            return RouteKind.REDACT_ONLY
        if method == "GET" and _AZURE_FILE_CONTENT.fullmatch(path):
            return RouteKind.CHAT
        # /openai/batches + file metadata/list/delete: ids and processing
        # state only — pass-through, pinned by test.
        if method != "POST":
            return RouteKind.NONE
        match = _AZURE_PATH.fullmatch(path)
        if match is None:
            return RouteKind.NONE
        return RouteKind.REDACT_ONLY if match.group(1) == "embeddings" else RouteKind.CHAT

    def rehydrate_raw_body(self, path: str, raw: bytes, rehydrator: "Rehydrator") -> bytes | None:
        if _AZURE_FILE_CONTENT.fullmatch(path):
            # Delegate with an OpenAI-shaped path: the parent's line-by-line
            # JSONL restoration is path-gated on /v1/files/{id}/content.
            return super().rehydrate_raw_body("/v1/files/azure/content", raw, rehydrator)
        return super().rehydrate_raw_body(path, raw, rehydrator)


class AzureResponsesAdapter(OpenAIResponsesAdapter):
    """Azure OpenAI Responses API — the Codex/Responses wire format on Azure's
    path shapes. Event names, delta channels, stored-response rehydration, and
    note injection are all inherited from OpenAIResponsesAdapter verbatim; only
    routing differs (Azure carries the api-version in the query and the model
    in the body, so there is no deployment segment on the Responses path).

    Separate adapter, name "azure", so upstream selection reaches the
    customer's resource URL rather than api.openai.com — the same reason the
    chat adapter is split. Matchers proven disjoint from AzureOpenAIAdapter
    (responses vs chat/completions|embeddings) by test.
    """

    name = "azure"

    def matches(self, method: str, path: str) -> RouteKind:
        if method == "POST" and _AZURE_RESPONSES.fullmatch(path):
            return RouteKind.CHAT
        if method == "GET" and (
            _AZURE_RESPONSE_ID.fullmatch(path) or _AZURE_RESPONSE_INPUT_ITEMS.fullmatch(path)
        ):
            return RouteKind.CHAT
        return RouteKind.NONE
