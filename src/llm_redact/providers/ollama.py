"""Ollama native API adapter (/api/chat, /api/generate; NDJSON streams).

Ollama's OpenAI-compatible endpoints (/v1/...) already flow through the
OpenAI adapter by pointing [providers.openai] at the daemon; this adapter
covers tools that speak the NATIVE API. Verified against docs/api.md and
server/routes.go (2026-07): streaming responses are
``application/x-ndjson`` — one JSON object per line — while
``stream: false`` answers plain ``application/json``, so the proxy's
branch-on-response-content-type rule stays unambiguous. Chat tool-call
``arguments`` are parsed objects arriving complete in one line (plain
walk with whole-string restoration, like Gemini's functionCall.args).

Rehydration channels: one per stream — chat rewrites
``message.content``, generate rewrites ``response``. The ``done: true``
line is the flush point: its content is empty upstream, so any held
partial-token leftover is folded into it (clients concatenate content
from every chunk, the done line included).

The system note is appended ONLY to an existing system prompt (the last
system-role message in /api/chat, the ``system`` string in
/api/generate). Creating one where the client sent none would override
the model's Modelfile SYSTEM template and change behavior; a missing
note only weakens token preservation.
"""

import json
from typing import Any

from llm_redact.jsonwalk import transform_strings
from llm_redact.providers.base import SYSTEM_NOTE, ProviderAdapter, RouteKind
from llm_redact.rehydrate import RehydratorPool
from llm_redact.sse import SSEEvent

_CHAT_CHANNEL = ("ollama", "chat")
_GENERATE_CHANNEL = ("ollama", "generate")


class OllamaAdapter(ProviderAdapter):
    name = "ollama"
    handles_ndjson = True

    def matches(self, method: str, path: str) -> RouteKind:
        if method != "POST":
            return RouteKind.NONE
        if path in ("/api/chat", "/api/generate"):
            return RouteKind.CHAT
        # /api/embeddings is the deprecated predecessor of /api/embed; both
        # carry user text in, vectors out.
        if path in ("/api/embed", "/api/embeddings"):
            return RouteKind.REDACT_ONLY
        return RouteKind.NONE

    def error_body(self, message: str, *, status: int = 413) -> dict[str, Any]:
        return {"error": message}  # Ollama's wire shape for errors

    def inject_system_note(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if isinstance(messages, list):
            # /api/chat: append to the LAST system message. A client that
            # sent any system message has already overridden the Modelfile
            # SYSTEM, so appending is safe; creating one would do that
            # override ourselves.
            for index in range(len(messages) - 1, -1, -1):
                message = messages[index]
                if (
                    isinstance(message, dict)
                    and message.get("role") == "system"
                    and isinstance(message.get("content"), str)
                ):
                    new_messages = list(messages)
                    new_messages[index] = {
                        **message,
                        "content": f"{message['content']}\n\n{SYSTEM_NOTE}",
                    }
                    return {**body, "messages": new_messages}
            return body
        if isinstance(body.get("system"), str):
            # /api/generate: the field's presence means the client already
            # overrides the Modelfile SYSTEM; appending is safe.
            return {**body, "system": f"{body['system']}\n\n{SYSTEM_NOTE}"}
        return body

    def rehydrate_event(self, event: SSEEvent, pool: RehydratorPool) -> list[SSEEvent]:
        return [event]  # Ollama never streams SSE; unreachable in practice

    def rehydrate_ndjson_line(self, line: bytes, pool: RehydratorPool) -> bytes:
        try:
            payload = json.loads(line)
        except ValueError:
            return line  # non-JSON lines (or blanks) pass through verbatim
        if not isinstance(payload, dict):
            return line
        done = payload.get("done") is True

        if isinstance(payload.get("response"), str):  # /api/generate chunk
            new_text = pool.get(_GENERATE_CHANNEL).feed(payload["response"])
            if done:
                new_text += pool.flush(_GENERATE_CHANNEL)
            if new_text == payload["response"]:
                return line
            return json.dumps({**payload, "response": new_text}, ensure_ascii=False).encode()

        message = payload.get("message")
        if not isinstance(message, dict):
            return line
        new_message = dict(message)
        changed = False
        if isinstance(message.get("content"), str):
            new_content = pool.get(_CHAT_CHANNEL).feed(message["content"])
            if done:
                new_content += pool.flush(_CHAT_CHANNEL)
            if new_content != message["content"]:
                new_message["content"] = new_content
                changed = True
        elif done:
            leftover = pool.flush(_CHAT_CHANNEL)
            if leftover:
                new_message["content"] = leftover
                changed = True
        if "tool_calls" in message:
            # Parsed objects, complete in one line: whole-string restoration
            # (restores through the pool so counts flow into audit/status).
            new_calls = transform_strings(message["tool_calls"], pool.rehydrate_whole)
            if new_calls != message["tool_calls"]:
                new_message["tool_calls"] = new_calls
                changed = True
        if not changed:
            return line
        return json.dumps({**payload, "message": new_message}, ensure_ascii=False).encode()
