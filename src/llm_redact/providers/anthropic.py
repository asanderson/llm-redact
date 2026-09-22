"""Anthropic Messages API adapter (/v1/messages + Message Batches, native SSE).

Message Batches (verified against docs.anthropic.com, 2026-07): creation is
plain JSON whose ``requests[].params`` entries are full Messages bodies —
the generic walk redacts them and the system note is injected per entry.
The results endpoint returns ``application/x-jsonl``, one complete result
object per line ({custom_id, result:{type, message?}}) — rehydrated line by
line with whole-string restoration (lines are complete, no streaming
channels needed). Poll/list/cancel/delete carry processing metadata only in
both directions and deliberately pass through.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from llm_redact.jsonwalk import transform_strings
from llm_redact.providers.base import SYSTEM_NOTE, ProviderAdapter, RouteKind
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import RehydratorPool
from llm_redact.sse import SSEEvent

_PASSTHROUGH_EVENTS = frozenset(
    {"message_start", "message_delta", "message_stop", "ping", "error", "content_block_start"}
)

_BATCH_RESULTS_RE = re.compile(r"/v1/messages/batches/[^/]+/results")


def inject_anthropic_system_note(body: dict[str, Any]) -> dict[str, Any]:
    """Messages-API note injection, shared with the Bedrock adapter
    (Claude invoke bodies carry the same `system` field shapes)."""
    body = dict(body)
    system = body.get("system")
    if system is None:
        body["system"] = SYSTEM_NOTE
    elif isinstance(system, str):
        body["system"] = f"{system}\n\n{SYSTEM_NOTE}"
    elif isinstance(system, list):
        body["system"] = [*system, {"type": "text", "text": SYSTEM_NOTE}]
    return body


def rehydrate_messages_payload(
    payload: dict[str, Any], pool: RehydratorPool
) -> list[dict[str, Any]] | None:
    """Rewrite one parsed Messages-API stream event payload.

    Shared by the SSE path and Bedrock invoke-with-response-stream, whose
    chunk frames wrap the very same events in base64. Returns None for
    events this logic does not rewrite — the caller forwards the original
    bytes verbatim. Never mutates its argument: an element of the result
    that ``is payload`` is unchanged (the caller may reuse original bytes);
    synthetic flush deltas precede a content_block_stop.
    """
    event_type = payload.get("type")

    if event_type == "content_block_delta":
        index = payload.get("index", 0)
        delta = payload.get("delta", {})
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            new_delta = {**delta, "text": pool.get(("text", index)).feed(delta.get("text", ""))}
        elif delta_type == "thinking_delta":
            new_delta = {
                **delta,
                "thinking": pool.get(("thinking", index)).feed(delta.get("thinking", "")),
            }
        elif delta_type == "input_json_delta":
            new_delta = {
                **delta,
                "partial_json": pool.get(("tool", index), json_source=True).feed(
                    delta.get("partial_json", "")
                ),
            }
        else:
            return None
        return [{**payload, "delta": new_delta}]

    if event_type == "content_block_stop":
        index = payload.get("index", 0)
        synthetic: list[dict[str, Any]] = []
        channel_shapes: list[tuple[tuple[str, Any], Callable[[str], dict[str, Any]]]] = [
            (("text", index), lambda text: {"type": "text_delta", "text": text}),
            (("thinking", index), lambda text: {"type": "thinking_delta", "thinking": text}),
            (("tool", index), lambda text: {"type": "input_json_delta", "partial_json": text}),
        ]
        for channel, delta_payload in channel_shapes:
            leftover = pool.flush(channel)
            if leftover:
                synthetic.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": delta_payload(leftover),
                    }
                )
        return [*synthetic, payload]

    return None


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"
    handles_ndjson = True  # batch results stream application/x-jsonl

    def matches(self, method: str, path: str) -> RouteKind:
        if method == "POST":
            if path == "/v1/messages":
                return RouteKind.CHAT
            # Full message content flows through token counting too; redact
            # it, but the response (a count) contains nothing to rehydrate.
            if path == "/v1/messages/count_tokens":
                return RouteKind.REDACT_ONLY
            # Batch creation: requests[].params are full Messages bodies.
            # The response is processing metadata — nothing to rehydrate.
            if path == "/v1/messages/batches":
                return RouteKind.REDACT_ONLY
            # Legacy Text Completions: deprecated since 2023 but still a
            # content leak for old tools. prompt redacted, completion
            # rehydrated (streaming included); NO system note — the body
            # has no system field in this API.
            if path == "/v1/complete":
                return RouteKind.CHAT
        elif method == "GET" and _BATCH_RESULTS_RE.fullmatch(path):
            # Results: no request body to redact (redaction no-ops without
            # one); the JSONL response rehydrates via rehydrate_ndjson_line.
            return RouteKind.CHAT
        # Batch poll/list/cancel/delete carry processing metadata only in
        # both directions — deliberate pass-through, pinned by test.
        return RouteKind.NONE

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        # count_tokens accepts the same `system` field as /v1/messages, and
        # the note is part of what the real request will carry — keep the
        # count honest (this preserves pre-hook behavior exactly). The
        # legacy /v1/complete body has no system field: a note would 400.
        return path != "/v1/complete"

    def error_body(self, message: str, *, status: int = 413) -> dict[str, Any]:
        error_type = {413: "request_too_large", 502: "api_error"}.get(
            status, "invalid_request_error"
        )
        return {"type": "error", "error": {"type": error_type, "message": message}}

    def prepare_request(
        self,
        body: dict[str, Any],
        redactor: Redactor,
        *,
        inject_note: bool,
        mcp_exempt: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        # mcp_servers blocks are provider-directed configuration: the
        # provider must hold the REAL authorization_token to call the MCP
        # server on the model's behalf — redacting it breaks the connector
        # (the same stance as tools[].type == "mcp" on the OpenAI side).
        # Stripped before redaction so its values are never counted.
        mcp_servers = body.get("mcp_servers")
        if isinstance(mcp_servers, list):
            stripped = {key: value for key, value in body.items() if key != "mcp_servers"}
            prepared = super().prepare_request(
                stripped, redactor, inject_note=inject_note, mcp_exempt=mcp_exempt
            )
            return {**prepared, "mcp_servers": mcp_servers}
        return super().prepare_request(
            body, redactor, inject_note=inject_note, mcp_exempt=mcp_exempt
        )

    def inject_system_note(self, body: dict[str, Any]) -> dict[str, Any]:
        requests = body.get("requests")
        if isinstance(requests, list):
            # Batch create: each entry's params is a Messages body by the
            # endpoint's contract — inject per entry; anything not
            # positively dict-shaped is forwarded untouched.
            return {
                **body,
                "requests": [
                    {**item, "params": inject_anthropic_system_note(item["params"])}
                    if isinstance(item, dict) and isinstance(item.get("params"), dict)
                    else item
                    for item in requests
                ],
            }
        return inject_anthropic_system_note(body)

    def rehydrate_ndjson_line(self, line: bytes, pool: RehydratorPool) -> bytes:
        # Batch results: one complete result object per line — whole-string
        # restoration through the pool (counts flow into audit/status).
        # Anything unparseable is forwarded byte-identically.
        try:
            payload = json.loads(line)
        except ValueError:
            return line
        if not isinstance(payload, dict):
            return line
        rehydrated = transform_strings(payload, pool.rehydrate_whole)
        if rehydrated == payload:
            return line
        return json.dumps(rehydrated, ensure_ascii=False).encode("utf-8")

    def rehydrate_event(self, event: SSEEvent, pool: RehydratorPool) -> list[SSEEvent]:
        if event.event in _PASSTHROUGH_EVENTS or not event.data:
            return [event]
        try:
            payload = json.loads(event.data)
        except ValueError:
            return [event]
        if not isinstance(payload, dict):
            return [event]
        if payload.get("type") == "completion" and isinstance(payload.get("completion"), str):
            # Legacy /v1/complete stream: incremental text in `completion`;
            # the stop_reason-bearing event is the flush point.
            channel = ("legacy_completion",)
            new_text = pool.get(channel).feed(payload["completion"])
            if payload.get("stop_reason"):
                new_text += pool.flush(channel)
            if new_text != payload["completion"]:
                event.data = json.dumps({**payload, "completion": new_text}, ensure_ascii=False)
            return [event]
        payloads = rehydrate_messages_payload(payload, pool)
        if payloads is None:
            return [event]
        events: list[SSEEvent] = []
        for item in payloads:
            if item is payload:
                events.append(event)  # unchanged: original bytes and fields
            elif item.get("type") == payload.get("type"):
                # The rewritten form of this event: keep its envelope.
                event.data = json.dumps(item, ensure_ascii=False)
                events.append(event)
            else:
                # Synthetic flush delta injected before a stop.
                events.append(
                    SSEEvent(event="content_block_delta", data=json.dumps(item, ensure_ascii=False))
                )
        return events
