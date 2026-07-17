"""Cohere native API adapter (api.cohere.com).

Covers the v2 surface and the legacy v1 chat/generate endpoints:

- ``POST /v2/chat`` — CHAT. Request ``messages[]`` (string or content-block
  content) are redacted; the non-streaming response
  (``message.content[].text``, ``message.tool_plan``, tool-call
  ``arguments``) is rehydrated by the generic jsonwalk with an ``arguments``
  JSON-source override, and the streaming SSE form is rehydrated per channel.
- ``POST /v2/embed`` and ``POST /v2/rerank`` — REDACT_ONLY. The embedding
  input and rerank query/documents are redactable content; the responses are
  vectors / ranked indices with nothing to restore.
- ``POST /v1/chat`` and ``POST /v1/generate`` — CHAT (deprecated by Cohere).
  Non-streaming responses ride the generic rehydrate; the v1 streaming form
  (``event_type: text-generation``) has its own text channel.

Streaming event shapes follow Cohere's documented v2 vocabulary; like every
other streamed adapter they are pinned by hand-authored fixtures and a live
drift test (observed types ⊆ ``KNOWN_COHERE_EVENT_TYPES``). An event this
adapter does not recognize is forwarded byte-identically — the worst case is
a placeholder reaching the user unrestored, never a wrong or leaked value.
"""

import json
from collections.abc import Hashable
from typing import Any

from llm_redact.jsonwalk import transform_strings
from llm_redact.providers.base import SYSTEM_NOTE, ProviderAdapter, RouteKind
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent

_CHAT_PATHS = frozenset({"/v2/chat", "/v1/chat", "/v1/generate"})
_REDACT_ONLY_PATHS = frozenset({"/v2/embed", "/v2/rerank"})

# type → (channel-kind, json_source) for v2 streaming deltas.
_DELTA_TYPES = {
    "content-delta": ("text", False),
    "tool-plan-delta": ("tool_plan", False),
    "tool-call-delta": ("args", True),
}
# The *-end events that flush a per-index channel before passing through.
_END_TYPES = {"content-end": "text", "tool-call-end": "args"}

KNOWN_COHERE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        # v2
        "message-start",
        "content-start",
        "content-delta",
        "content-end",
        "tool-plan-delta",
        "tool-call-start",
        "tool-call-delta",
        "tool-call-end",
        "citation-start",
        "citation-end",
        "message-end",
        "debug",
        # v1 (event_type field)
        "stream-start",
        "text-generation",
        "tool-calls-generation",
        "tool-calls-chunk",
        "citation-generation",
        "search-queries-generation",
        "search-results",
        "stream-end",
    }
)


class CohereAdapter(ProviderAdapter):
    name = "cohere"

    def matches(self, method: str, path: str) -> RouteKind:
        if method != "POST":
            return RouteKind.NONE
        if path in _CHAT_PATHS:
            return RouteKind.CHAT
        if path in _REDACT_ONLY_PATHS:
            return RouteKind.REDACT_ONLY
        return RouteKind.NONE

    def error_body(self, message: str, *, status: int = 413) -> dict[str, Any]:
        # Cohere errors are a flat {"message": ...} object.
        return {"message": message}

    def inject_system_note(self, body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        if isinstance(body.get("messages"), list):
            # v2 chat (and v1 chat when it uses messages): a system message is
            # the token-preservation note's home.
            body["messages"] = [{"role": "system", "content": SYSTEM_NOTE}, *body["messages"]]
        elif "message" in body:
            # v1 chat: `preamble` is the system slot; append idempotently.
            preamble = body.get("preamble")
            if not (isinstance(preamble, str) and SYSTEM_NOTE in preamble):
                body["preamble"] = (
                    f"{preamble}\n\n{SYSTEM_NOTE}"
                    if isinstance(preamble, str) and preamble
                    else SYSTEM_NOTE
                )
        # v1 generate (bare `prompt`) has no system field; a note would have to
        # be prepended to the prompt and change the generation — left untouched.
        return body

    def rehydrate_body(self, body: Any, rehydrator: Rehydrator) -> Any:
        # tool_calls[].function.arguments is raw JSON source, like OpenAI's.
        return transform_strings(
            body,
            rehydrator.rehydrate_text,
            key_overrides={"arguments": rehydrator.rehydrate_json_source_text},
        )

    # --- streaming -----------------------------------------------------------

    @staticmethod
    def _index(payload: dict[str, Any]) -> int:
        idx = payload.get("index")
        return idx if isinstance(idx, int) else 0

    @staticmethod
    def _synthetic(key: tuple[Any, ...], leftover: str) -> SSEEvent:
        kind = key[0]
        if kind == "text":
            payload: dict[str, Any] = {
                "type": "content-delta",
                "index": key[1],
                "delta": {"message": {"content": {"text": leftover}}},
            }
        elif kind == "args":
            payload = {
                "type": "tool-call-delta",
                "index": key[1],
                "delta": {"message": {"tool_calls": {"function": {"arguments": leftover}}}},
            }
        elif kind == "tool_plan":
            payload = {"type": "tool-plan-delta", "delta": {"message": {"tool_plan": leftover}}}
        else:  # v1 text-generation
            payload = {"event_type": "text-generation", "text": leftover}
        return SSEEvent(data=json.dumps(payload, ensure_ascii=False))

    def _flush(self, leftovers: dict[Hashable, str]) -> list[SSEEvent]:
        return [
            self._synthetic(key, text)
            for key, text in leftovers.items()
            if isinstance(key, tuple) and text
        ]

    def rehydrate_event(self, event: SSEEvent, pool: RehydratorPool) -> list[SSEEvent]:
        if not event.data:
            return [event]
        try:
            payload = json.loads(event.data)
        except ValueError:
            return [event]
        if not isinstance(payload, dict):
            return [event]

        event_type = payload.get("type")

        # v2 content / tool-plan / tool-call deltas.
        if event_type in _DELTA_TYPES:
            kind, json_source = _DELTA_TYPES[event_type]
            message = (payload.get("delta") or {}).get("message") or {}
            if kind == "text":
                node = message.get("content")
                field = "text"
            elif kind == "tool_plan":
                node, field = message, "tool_plan"
            else:
                node = ((message.get("tool_calls") or {}).get("function")) or {}
                field = "arguments"
            if isinstance(node, dict) and isinstance(node.get(field), str):
                key = ("tool_plan",) if kind == "tool_plan" else (kind, self._index(payload))
                node[field] = pool.get(key, json_source=json_source).feed(node[field])
                event.data = json.dumps(payload, ensure_ascii=False)
            return [event]

        # v2 *-end: flush that index's channel first (leftover → synthetic
        # delta), then emit the end event.
        if event_type in _END_TYPES:
            key = (_END_TYPES[event_type], self._index(payload))
            leftover = pool.flush(key)
            synthetic = self._flush({key: leftover}) if leftover else []
            return [*synthetic, event]

        # v2 message-end: flush every remaining channel, then the end event.
        if event_type == "message-end":
            return [*self._flush(pool.flush_all()), event]

        # v1 streaming: text-generation deltas on a single channel, flushed at
        # stream-end.
        v1_type = payload.get("event_type")
        if v1_type == "text-generation" and isinstance(payload.get("text"), str):
            payload["text"] = pool.get(("v1text",)).feed(payload["text"])
            event.data = json.dumps(payload, ensure_ascii=False)
            return [event]
        if v1_type == "stream-end":
            return [*self._flush(pool.flush_all()), event]

        # message-start / content-start / citations / unknown: pass through.
        return [event]
