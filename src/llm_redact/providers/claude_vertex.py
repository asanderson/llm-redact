"""Claude-on-Vertex adapter: Anthropic Messages bodies behind Vertex paths.

Claude models served through Vertex AI speak the Anthropic Messages
request/response format (the body carries ``anthropic_version:
"vertex-2023-10-16"`` and omits ``model`` — the model is in the path), not
Gemini's generateContent shape. So everything — outbound redaction, MCP
handling, note injection, and Anthropic-SSE rehydration — is inherited from
AnthropicAdapter; only route matching and the upstream name differ.

Paths (v1/v1beta1) on {region}-aiplatform.googleapis.com, with or without
the projects/locations prefix:
  /v1/projects/{p}/locations/{l}/publishers/anthropic/models/{m}:rawPredict
  /v1/publishers/anthropic/models/{m}:streamRawPredict

Anchored to ``publishers/anthropic`` specifically: other publishers (Llama
etc.) also use ``:rawPredict`` but with entirely different body formats, so
matching them here would mis-rehydrate their traffic. Shares the
[providers.vertex] upstream (the host embeds the region) — 502 until
configured, exactly like the Gemini VertexAdapter; its matcher is disjoint
from that adapter's (rawPredict vs generateContent verbs).
"""

import re

from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.base import RouteKind

_CLAUDE_VERTEX_PATH = re.compile(
    r"/(?:v1|v1beta1)/(?:projects/[^/]+/locations/[^/]+/)?"
    r"publishers/anthropic/models/[^/:]+"
    r":(rawPredict|streamRawPredict)"
)


class ClaudeVertexAdapter(AnthropicAdapter):
    # Same upstream as the Gemini Vertex adapter (region-embedded host);
    # both resolve to [providers.vertex]. Matchers are proven disjoint.
    name = "vertex"

    def matches(self, method: str, path: str) -> RouteKind:
        # rawPredict returns a JSON Messages object; streamRawPredict returns
        # Anthropic SSE. Both are CHAT (the proxy branches on the response
        # content-type, so one classification covers both).
        if method != "POST":
            return RouteKind.NONE
        return RouteKind.CHAT if _CLAUDE_VERTEX_PATH.fullmatch(path) else RouteKind.NONE

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        # The rawPredict body is a Messages body with a `system` field, so
        # the note injects cleanly (unlike the legacy /v1/complete path the
        # base class special-cases, which does not exist on Vertex).
        return kind is RouteKind.CHAT
