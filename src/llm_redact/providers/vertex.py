"""Google Vertex AI adapter: Gemini bodies behind Google Cloud paths.

Vertex serves the same generateContent/streamGenerateContent/countTokens
wire format as the Gemini API, so everything — SSE channels, the buffered
JSON-array stream form, system-note injection, drift sets, error shape —
is inherited from GeminiAdapter; only route matching differs. Auth is a
Bearer token (no request signing), so body rewriting is safe.

Paths (v1 and v1beta1) on {region}-aiplatform.googleapis.com:
  /v1/projects/{p}/locations/{l}/publishers/{pub}/models/{m}:generateContent
  /v1/projects/{p}/locations/{l}/endpoints/{id}:streamGenerateContent
and express mode (global host, no project prefix):
  /v1/publishers/google/models/{m}:generateContent

Beyond inference (verified against the v1 REST reference, 2026-09):

- Context caching (``projects.locations.cachedContents``): the create
  carries ``contents``/``systemInstruction``/``tools`` — content, so it is
  REDACT_ONLY exactly like the Gemini API's cache create, on the STATIC
  vault session (the batch stance: a cache is reused later with no
  first-message anchor). get/list/patch/delete are recognized too
  (REDACT_ONLY: a body-less or ttl-only request, and ``contents`` is
  INPUT-ONLY — never returned), so a provider authorized with the proxy's
  own identity can manage its caches.
- ``:computeTokens`` and ``:embedContent`` carry content; their responses
  (token ids/bytes, vectors) have nothing to restore → REDACT_ONLY.
- ``:fetchPredictOperation`` polls a ``predictLongRunning`` (Veo) job; its
  body is an operation name → REDACT_ONLY (recognized for identity auth).
- Model metadata GETs (``publishers/{pub}/models[/{m}]`` — the google-genai
  SDK's ``models.list``/``models.get`` — and the project Model Registry
  ``projects/{p}/locations/{l}/models[/{m}]``) → REDACT_ONLY: no body, no
  content either way; recognizing them keeps identity auth from refusing
  the SDK's routine model lookups.

There is no default upstream — the host embeds the customer's region — so
vertex routes answer 502 until [providers.vertex] is configured, exactly
like azure.
"""

import re
from typing import Any

from llm_redact.providers.base import RouteKind
from llm_redact.providers.gemini import GeminiAdapter, cache_object_ids

_VERSION = r"/(?:v1|v1beta1)/"
_PROJECT = r"projects/[^/]+/locations/[^/]+/"
_VERTEX_PATH = re.compile(
    _VERSION + r"(?:" + _PROJECT + r")?"
    r"(?:publishers/[^/]+/models/[^/:]+|endpoints/[^/:]+)"
    # predict (Imagen) / predictLongRunning (Veo) stay DISJOINT from
    # Claude-on-Vertex's rawPredict/streamRawPredict: the colon anchors the
    # verb, and "rawPredict" is not in this alternation (proven by test).
    r":(generateContent|streamGenerateContent|countTokens|predict|predictLongRunning"
    r"|computeTokens|embedContent|fetchPredictOperation)"
)
_VERTEX_REDACT_ONLY_VERBS = frozenset(
    {
        "countTokens",
        "predict",
        "predictLongRunning",
        "computeTokens",
        "embedContent",
        "fetchPredictOperation",
    }
)
# Context caches live under a project/location only (no express-mode form).
_CACHED_COLLECTION = re.compile(_VERSION + _PROJECT + r"cachedContents")
_CACHED_ITEM = re.compile(_VERSION + _PROJECT + r"cachedContents/[^/:]+")
# Model metadata: publisher models (with or without the project prefix) and
# the project's Model Registry. The colon exclusion keeps every :verb out.
_MODEL_METADATA = re.compile(
    _VERSION + r"(?:(?:" + _PROJECT + r")?publishers/[^/]+|" + _PROJECT[:-1] + r")"
    r"/models(?:/[^/:]+)?"
)


class VertexAdapter(GeminiAdapter):
    name = "vertex"

    def matches(self, method: str, path: str) -> RouteKind:
        if method == "GET":
            if (
                _MODEL_METADATA.fullmatch(path)
                or _CACHED_COLLECTION.fullmatch(path)
                or _CACHED_ITEM.fullmatch(path)
            ):
                return RouteKind.REDACT_ONLY
            return RouteKind.NONE
        if method in ("PATCH", "DELETE"):
            return RouteKind.REDACT_ONLY if _CACHED_ITEM.fullmatch(path) else RouteKind.NONE
        if method != "POST":
            return RouteKind.NONE
        if _CACHED_COLLECTION.fullmatch(path):
            # Cache create: contents + systemInstruction to redact; the
            # response is metadata (contents are input-only).
            return RouteKind.REDACT_ONLY
        match = _VERTEX_PATH.fullmatch(path)
        if match is None:
            return RouteKind.NONE
        if match.group(1) in _VERTEX_REDACT_ONLY_VERBS:
            return RouteKind.REDACT_ONLY
        return RouteKind.CHAT

    def tracks_object_ids(self, method: str, path: str) -> bool:
        # A Vertex context cache, cited later as `cachedContent` by
        # generateContent bodies — the Gemini API stance on Vertex paths.
        return method == "POST" and _CACHED_COLLECTION.fullmatch(path) is not None

    def object_ids_from_body(self, method: str, path: str, body: Any) -> tuple[str, ...]:
        return cache_object_ids(body)

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        return kind is RouteKind.CHAT or path.endswith(":countTokens")
