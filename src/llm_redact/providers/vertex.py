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

There is no default upstream — the host embeds the customer's region — so
vertex routes answer 502 until [providers.vertex] is configured, exactly
like azure.
"""

import re

from llm_redact.providers.base import RouteKind
from llm_redact.providers.gemini import GeminiAdapter

_VERTEX_PATH = re.compile(
    r"/(?:v1|v1beta1)/(?:projects/[^/]+/locations/[^/]+/)?"
    r"(?:publishers/[^/]+/models/[^/:]+|endpoints/[^/:]+)"
    # predict (Imagen) / predictLongRunning (Veo) stay DISJOINT from
    # Claude-on-Vertex's rawPredict/streamRawPredict: the colon anchors the
    # verb, and "rawPredict" is not in this alternation (proven by test).
    r":(generateContent|streamGenerateContent|countTokens|predict|predictLongRunning)"
)
_VERTEX_REDACT_ONLY_VERBS = frozenset({"countTokens", "predict", "predictLongRunning"})


class VertexAdapter(GeminiAdapter):
    name = "vertex"

    def matches(self, method: str, path: str) -> RouteKind:
        if method != "POST":
            return RouteKind.NONE
        match = _VERTEX_PATH.fullmatch(path)
        if match is None:
            return RouteKind.NONE
        if match.group(1) in _VERTEX_REDACT_ONLY_VERBS:
            return RouteKind.REDACT_ONLY
        return RouteKind.CHAT

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        return kind is RouteKind.CHAT or path.endswith(":countTokens")
