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

Covered beyond chat: legacy completions, embeddings, image generation/edit
prompts and text-to-speech input on both path families; Files (multipart
JSONL upload, batch-output download) and Batches; the v1 Conversations item
store; and the model/deployment/file listings — recognized (REDACT_ONLY, a
no-op on a body-less GET) so that ``[providers.azure] auth = "identity"``,
which forwards only recognized routes, does not refuse them.
"""

import re

from llm_redact.providers.base import RouteKind
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.rehydrate import Rehydrator

# Inference routes on both path families: the api-version form carries a
# deployment segment, the v1 API (/openai/v1, the model in the body) does
# not. Verified against Microsoft's REST reference (2024-10-21 GA) and the
# OpenAI.v1 spec in Azure/azure-rest-api-specs (2026-09).
_AZURE_PATH = re.compile(
    r"/openai/(?:deployments/[^/]+|v1)/"
    r"(chat/completions|completions|embeddings|images/generations|images/edits|audio/speech)"
)
# Only chat bodies can carry the system note: legacy completions carry a bare
# prompt, and the media/embeddings bodies have no system field at all.
_CHAT_ROUTES = frozenset({"chat/completions", "completions"})
# Files + Batches ride Azure's own path shapes on both families (api-version
# stays in the query, untouched like every query).
_AZURE_FILES = re.compile(r"/openai/(?:v1/)?files")
_AZURE_FILE_ITEM = re.compile(r"/openai/(?:v1/)?files/[^/]+")
_AZURE_FILE_CONTENT = re.compile(r"/openai/(?:v1/)?files/[^/]+/content")
_AZURE_BATCH_COLLECTION = re.compile(r"/openai/(?:v1/)?batches")
_AZURE_BATCHES = re.compile(r"/openai/(?:v1/)?batches(?:/[^/]+)?")
_AZURE_BATCH_CANCEL = re.compile(r"/openai/(?:v1/)?batches/[^/]+/cancel")
# Model/deployment listings: metadata only.
_AZURE_METADATA = re.compile(r"/openai/(?:(?:v1/)?models|deployments)(?:/[^/]+)?")
# The Conversations item store (v1 API only), paired with Responses.
_AZURE_CONVERSATIONS = re.compile(r"/openai/v1/conversations(?:/[^/]+(?:/items(?:/[^/]+)?)?)?")
# Azure Responses rides both the api-version form (/openai/responses) and the
# newer v1 preview (/openai/v1/responses); GET fetches a stored response and
# its input-item echoes, both of which must be rehydrated back to the client.
_AZURE_RESPONSES = re.compile(r"/openai/(?:v1/)?responses")
_AZURE_RESPONSE_ID = re.compile(r"/openai/(?:v1/)?responses/[^/]+")
_AZURE_RESPONSE_INPUT_ITEMS = re.compile(r"/openai/(?:v1/)?responses/[^/]+/input_items")
_AZURE_RESPONSE_CANCEL = re.compile(r"/openai/(?:v1/)?responses/[^/]+/cancel")


class AzureOpenAIAdapter(OpenAIAdapter):
    name = "azure"

    def matches(self, method: str, path: str) -> RouteKind:
        # The content routes (chat, completions, embeddings, media, Files
        # upload/download, Conversations) are classified as the OpenAI
        # adapter classifies them. The rest does NOT mirror OpenAI, which
        # leaves them as pass-through: file list/metadata/delete, the batch
        # routes, model/deployment listings and conversation delete are
        # RECOGNIZED here, because a provider authorized with the proxy's own
        # identity forwards only recognized routes, and Azure is one of the
        # three that can be. On a body-less request REDACT_ONLY is a no-op,
        # so a key-authorized setup forwards the same bytes it did before.
        # Batch create/cancel and a single batch's GET are CHAT (the batch
        # object echoes the creator's own `metadata`); the batch COLLECTION
        # GET is REDACT_ONLY: a list spans batches other users created, and
        # restoring their placeholders in the READER's vault namespace
        # (llm-redact-pro named users) could hand one user a value bound in
        # another's — the list passes through unrestored instead, except for
        # items a session router attributes (`listing_item_session`), each
        # restored in the session that created it.
        if method == "POST":
            return self._match_post(path)
        if method == "GET":
            return self._match_get(path)
        if method == "DELETE" and (
            _AZURE_FILE_ITEM.fullmatch(path) or _AZURE_CONVERSATIONS.fullmatch(path)
        ):
            return RouteKind.REDACT_ONLY
        return RouteKind.NONE

    @staticmethod
    def _match_post(path: str) -> RouteKind:
        match = _AZURE_PATH.fullmatch(path)
        if match is not None:
            return RouteKind.CHAT if match.group(1) in _CHAT_ROUTES else RouteKind.REDACT_ONLY
        if _AZURE_FILES.fullmatch(path):
            # Multipart upload; the JSONL hooks are inherited verbatim.
            return RouteKind.REDACT_ONLY
        if _AZURE_BATCH_COLLECTION.fullmatch(path) or _AZURE_BATCH_CANCEL.fullmatch(path):
            # Batch create carries file ids and user `metadata`, which the
            # batch object echoes on every read: redacted out, restored back.
            return RouteKind.CHAT
        if _AZURE_CONVERSATIONS.fullmatch(path):
            return RouteKind.CHAT  # create / add items: redact + restore echo
        return RouteKind.NONE

    @staticmethod
    def _match_get(path: str) -> RouteKind:
        if _AZURE_BATCH_COLLECTION.fullmatch(path):
            return RouteKind.REDACT_ONLY  # the list: only attributed items restored
        if (
            _AZURE_FILE_CONTENT.fullmatch(path)
            or _AZURE_BATCHES.fullmatch(path)
            or _AZURE_CONVERSATIONS.fullmatch(path)
        ):
            # Batch output JSONL, batch objects (echoed metadata), stored
            # conversation items: content restored.
            return RouteKind.CHAT
        if (
            _AZURE_FILES.fullmatch(path)
            or _AZURE_FILE_ITEM.fullmatch(path)
            or _AZURE_METADATA.fullmatch(path)
        ):
            return RouteKind.REDACT_ONLY
        return RouteKind.NONE

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        if _AZURE_FILES.fullmatch(path):
            # Uploads inject per JSONL line (chat-shaped lines only), as on
            # OpenAI's /v1/files.
            return True
        match = _AZURE_PATH.fullmatch(path)
        # Only chat completions carry `messages`: the legacy completions,
        # batch and conversation bodies would be corrupted by one.
        return kind is RouteKind.CHAT and match is not None and match.group(1) == "chat/completions"

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
        if method == "POST" and (
            _AZURE_RESPONSES.fullmatch(path) or _AZURE_RESPONSE_CANCEL.fullmatch(path)
        ):
            # Cancel answers with the Response object (partial output
            # included), so it is restored like a stored-response GET.
            return RouteKind.CHAT
        if method == "GET" and (
            _AZURE_RESPONSE_ID.fullmatch(path) or _AZURE_RESPONSE_INPUT_ITEMS.fullmatch(path)
        ):
            return RouteKind.CHAT
        if method == "DELETE" and _AZURE_RESPONSE_ID.fullmatch(path):
            # Ids only; recognized so identity auth can delete what it stored.
            return RouteKind.REDACT_ONLY
        return RouteKind.NONE
