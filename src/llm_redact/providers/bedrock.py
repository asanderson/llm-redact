"""AWS Bedrock runtime adapter (bearer-token auth; SigV4 is a non-goal).

Covers the four runtime inference routes:

    POST /model/{modelId}/invoke
    POST /model/{modelId}/invoke-with-response-stream
    POST /model/{modelId}/converse
    POST /model/{modelId}/converse-stream

``modelId`` may be a percent-encoded ARN whose decoded form contains
slashes and colons: matching runs on the decoded path with a greedy id
(embedded slashes stay inside the id), and the proxy forwards the *raw*
path so the upstream sees exactly the encoding the client sent.

converse/converse-stream use AWS's model-agnostic schema; invoke bodies
are model-native (Anthropic Messages for Claude models). The system note
is injected only into positively recognized shapes — Claude native
(``anthropic_version`` present) or Converse-style messages — and any
other native body is left untouched: a wrong-schema field corrupts the
request, while a missing note only weakens token preservation. Redaction
itself is shape-independent (jsonwalk) and covers every route.

Streaming responses are application/vnd.amazon.eventstream (binary
frames — see eventstream.py), not SSE, so ``rehydrate_event`` is
unreachable in practice and passes through if it ever fires; the proxy
routes these streams through ``rehydrate_eventstream_message`` instead.
converse-stream events are rewritten natively; invoke-with-response-stream
``chunk`` frames base64-wrap the model's native events, rewritten for
Claude (the shared Messages-API logic) and forwarded verbatim for every
other model — an unrecognized shape must never be corrupted.
"""

import base64
import json
import re
from collections.abc import Callable
from typing import Any

from llm_redact.eventstream import EventStreamMessage, string_header
from llm_redact.providers.anthropic import (
    inject_anthropic_system_note,
    rehydrate_messages_payload,
)
from llm_redact.providers.base import SYSTEM_NOTE, ProviderAdapter, RouteKind
from llm_redact.rehydrate import RehydratorPool
from llm_redact.sse import SSEEvent

# Greedy id + $-anchored action: backtracking keeps "/invoke" etc. out of
# the id unless a later action segment also matches (then the longest id
# wins, which is what an ARN with encoded slashes needs).
_ROUTE = re.compile(r"^/model/.+/(?:invoke|invoke-with-response-stream|converse|converse-stream)$")

# Every :event-type a ConverseStream response is known to carry. The live
# drift test (tests/test_live.py) asserts observed types ⊆ this set: the
# adapter forwards unknown frames verbatim, so a new event type is not a
# correctness bug — but it IS a signal that the schema moved and text might
# be flowing through a frame we don't rewrite.
KNOWN_CONVERSE_EVENT_TYPES = frozenset(
    {
        "messageStart",
        "contentBlockStart",
        "contentBlockDelta",
        "contentBlockStop",
        "messageStop",
        "metadata",
    }
)

_CONVERSE_BLOCK_KEYS = frozenset(
    {
        "text",
        "image",
        "document",
        "video",
        "toolUse",
        "toolResult",
        "guardContent",
        "reasoningContent",
        "cachePoint",
        "citationsContent",
    }
)


def _looks_like_converse(body: dict[str, Any]) -> bool:
    """Positively recognize the Converse request schema.

    Converse content blocks are keyed unions without a "type" discriminator
    ({"text": ...}, {"toolUse": ...}); Anthropic and OpenAI-style blocks
    always carry "type", and OpenAI-style messages usually carry plain
    string content. Amazon Nova's native invoke schema mirrors Converse, so
    a Nova invoke body passing this check gets a *correct* injection.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        content = message.get("content")
        if not isinstance(content, list) or not content:
            return False
        for block in content:
            if not isinstance(block, dict) or not block:
                return False
            if not _CONVERSE_BLOCK_KEYS.issuperset(block):
                return False
    system = body.get("system")
    return system is None or isinstance(system, list)


def _event_headers(event_type: str) -> list[tuple[str, int, object]]:
    return [
        string_header(":message-type", "event"),
        string_header(":event-type", event_type),
        string_header(":content-type", "application/json"),
    ]


class BedrockAdapter(ProviderAdapter):
    name = "bedrock"
    handles_eventstream = True

    def matches(self, method: str, path: str) -> RouteKind:
        if method == "POST" and _ROUTE.match(path):
            return RouteKind.CHAT
        return RouteKind.NONE

    def error_body(self, message: str, *, status: int = 413) -> dict[str, Any]:
        # Bedrock error bodies carry only a message; the exception type
        # travels in the x-amzn-errortype header, which proxy-generated
        # responses do not fake.
        return {"message": message}

    def inject_system_note(self, body: dict[str, Any]) -> dict[str, Any]:
        if "anthropic_version" in body:
            # Claude invoke bodies are Messages-API-shaped (the version
            # field is required on Bedrock, so it is a reliable marker).
            return inject_anthropic_system_note(body)
        if _looks_like_converse(body):
            body = dict(body)
            system = body.get("system")
            note = {"text": SYSTEM_NOTE}
            body["system"] = [*system, note] if isinstance(system, list) else [note]
        return body

    def rehydrate_event(self, event: SSEEvent, pool: RehydratorPool) -> list[SSEEvent]:
        return [event]

    def rehydrate_eventstream_message(
        self, message: EventStreamMessage, pool: RehydratorPool
    ) -> list[EventStreamMessage]:
        if message.message_type != "event":
            return [message]  # exceptions and errors pass through untouched
        try:
            payload = json.loads(message.payload)
        except ValueError:
            return [message]
        if not isinstance(payload, dict):
            return [message]
        event_type = message.event_type
        if event_type == "chunk":
            return self._rehydrate_invoke_chunk(message, payload, pool)
        if event_type == "contentBlockDelta":
            return self._rehydrate_converse_delta(message, payload, pool)
        if event_type == "contentBlockStop":
            return self._rehydrate_converse_stop(message, payload, pool)
        return [message]

    def _rehydrate_invoke_chunk(
        self, message: EventStreamMessage, payload: dict[str, Any], pool: RehydratorPool
    ) -> list[EventStreamMessage]:
        """invoke-with-response-stream: {"bytes": b64(model-native event)}.

        Claude events are the Messages-API stream events, rewritten by the
        logic shared with AnthropicAdapter; any other inner shape (and any
        undecodable payload) is forwarded verbatim.
        """
        encoded = payload.get("bytes")
        if not isinstance(encoded, str):
            return [message]
        try:
            inner = json.loads(base64.b64decode(encoded, validate=True))
        except ValueError:  # covers binascii.Error (its subclass) too
            return [message]
        if not isinstance(inner, dict):
            return [message]
        rewritten = rehydrate_messages_payload(inner, pool)
        if rewritten is None:
            return [message]
        out: list[EventStreamMessage] = []
        for item in rewritten:
            if item is inner:
                out.append(message)  # unchanged: byte-identical re-emit
                continue
            raw = json.dumps(item, ensure_ascii=False).encode("utf-8")
            body = json.dumps({**payload, "bytes": base64.b64encode(raw).decode("ascii")}).encode(
                "utf-8"
            )
            if item.get("type") == inner.get("type"):
                message.payload = body  # the rewritten form of this event
                out.append(message)
            else:
                # Synthetic flush delta injected before a stop; same headers.
                out.append(EventStreamMessage(headers=list(message.headers), payload=body))
        return out

    def _rehydrate_converse_delta(
        self, message: EventStreamMessage, payload: dict[str, Any], pool: RehydratorPool
    ) -> list[EventStreamMessage]:
        index = payload.get("contentBlockIndex", 0)
        delta = payload.get("delta")
        if not isinstance(delta, dict):
            return [message]
        new_delta: dict[str, Any]
        if isinstance(delta.get("text"), str):
            new_delta = {**delta, "text": pool.get(("text", index)).feed(delta["text"])}
        elif isinstance(delta.get("reasoningContent"), dict) and isinstance(
            delta["reasoningContent"].get("text"), str
        ):
            reasoning = delta["reasoningContent"]
            new_delta = {
                **delta,
                "reasoningContent": {
                    **reasoning,
                    "text": pool.get(("reasoning", index)).feed(reasoning["text"]),
                },
            }
        elif isinstance(delta.get("toolUse"), dict) and isinstance(
            delta["toolUse"].get("input"), str
        ):
            # toolUse.input streams partial JSON *source*, like Anthropic's
            # input_json_delta — escapes must stay escapes.
            tool = delta["toolUse"]
            new_delta = {
                **delta,
                "toolUse": {
                    **tool,
                    "input": pool.get(("tool", index), json_source=True).feed(tool["input"]),
                },
            }
        else:
            return [message]
        message.payload = json.dumps({**payload, "delta": new_delta}, ensure_ascii=False).encode(
            "utf-8"
        )
        return [message]

    def _rehydrate_converse_stop(
        self, message: EventStreamMessage, payload: dict[str, Any], pool: RehydratorPool
    ) -> list[EventStreamMessage]:
        index = payload.get("contentBlockIndex", 0)
        synthetic: list[EventStreamMessage] = []
        channel_shapes: list[tuple[tuple[str, Any], Callable[[str], dict[str, Any]]]] = [
            (("text", index), lambda text: {"text": text}),
            (("reasoning", index), lambda text: {"reasoningContent": {"text": text}}),
            (("tool", index), lambda text: {"toolUse": {"input": text}}),
        ]
        for channel, delta_shape in channel_shapes:
            leftover = pool.flush(channel)
            if leftover:
                body = json.dumps(
                    {"contentBlockIndex": index, "delta": delta_shape(leftover)},
                    ensure_ascii=False,
                ).encode("utf-8")
                synthetic.append(
                    EventStreamMessage(headers=_event_headers("contentBlockDelta"), payload=body)
                )
        return [*synthetic, message]
