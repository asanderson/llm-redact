"""Anthropic Messages API adapter (/v1/messages + Message Batches, native SSE).

Message Batches (verified against docs.anthropic.com, 2026-07): creation is
plain JSON whose ``requests[].params`` entries are full Messages bodies —
the generic walk redacts them and the system note is injected per entry.
The results endpoint returns ``application/x-jsonl``, one complete result
object per line ({custom_id, result:{type, message?}}) — rehydrated line by
line with whole-string restoration (lines are complete, no streaming
channels needed). Poll/list/cancel/delete carry processing metadata only in
both directions: recognized as redact-only (a no-op on a body-less request),
like the model listing — so a routed request spending a key the proxy holds
may still reach them, while an unrecognized route never can.
"""

import re
from collections.abc import Callable, Mapping
from typing import Any

from llm_redact.jsonwalk import json_bytes, json_text, loads_bounded, transform_strings
from llm_redact.providers.attribution import provider_markers
from llm_redact.providers.base import SYSTEM_NOTE, ProviderAdapter, RouteKind
from llm_redact.providers.documents import redact_files_upload, rehydrate_download
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent

_PASSTHROUGH_EVENTS = frozenset(
    {"message_start", "message_delta", "message_stop", "ping", "error", "content_block_start"}
)

_BATCH_RESULTS_RE = re.compile(r"/v1/messages/batches/[^/]+/results")
# Batch list/poll/delete (GET, DELETE) and cancel (POST): processing
# metadata only in both directions.
_BATCH_ITEM_RE = re.compile(r"/v1/messages/batches/[^/]+")
_BATCH_CANCEL_RE = re.compile(r"/v1/messages/batches/[^/]+/cancel")
# The model listing and one model, shared with OpenAI (and the Gemini API's
# v1 surface): Anthropic's only when the request carries its marker alone.
_MODELS_RE = re.compile(r"/v1/models(?:/[^/]+)?")
# The Files API (beta), whose paths are OpenAI's too: Anthropic's only when
# the request carries its marker alone (like the model listing). The upload
# is a multipart/form-data document (a PDF, a text file, an image) — a TEXT
# file redacted as text, a binary one forwarded as sent only with the
# client's own key (``providers.documents``, the OpenAI Files upload's
# policy) — and every file object echoes the uploaded ``filename``: the
# upload's answer, the list and a file's metadata restore it. A file's
# content (downloadable only for files a tool created) is restored when it
# is text; delete carries the id only.
_FILES = "/v1/files"
_FILE_RE = re.compile(r"/v1/files/[^/]+")
_FILE_CONTENT_RE = re.compile(r"/v1/files/[^/]+/content")
# The routes whose bodies carry the Messages `system` field the note joins.
_NOTE_PATHS = frozenset({"/v1/messages", "/v1/messages/count_tokens", "/v1/messages/batches"})

# The files a server tool WROTE, as a Messages answer names them: a code
# execution tool result (``code_execution_tool_result``, and the
# ``bash_code_execution_tool_result`` / ``text_editor_…`` forms of the
# current tool) lists each file its run created as an ``…_output`` entry of
# its ``content.content`` with a ``file_id`` — downloadable through the
# Files API (``GET /v1/files/{id}/content``) and citable by later messages.
# Only these are the provider's creations: a ``container_upload`` or a
# document ``source.file_id`` names a file the REQUEST supplied.
_CODE_RESULT_SUFFIX = "code_execution_tool_result"
_OUTPUT_SUFFIX = "_output"
# A key every event naming a generated file carries (never inside a JSON
# string: a quote there is escaped), so an event without it is never parsed.
_FILE_ID_KEY = '"file_id"'
# The code execution CONTAINER a message ran in: the answer names it
# (``container: {"id", "expires_at"}``; streamed in ``message_start``'s
# message or ``message_delta``'s delta), and a later request reuses it by
# that id (its top-level ``container``) — files earlier runs wrote live
# there. Reported like a generated file; a container the request itself
# named is dropped by the proxy (``_uncited``), so only a new one is the
# requester's. The key gates event parsing like ``_FILE_ID_KEY``.
_CONTAINER_KEY = '"container"'


def generated_file_ids(blocks: Any) -> tuple[str, ...]:
    """The files the code execution results among ``blocks`` (Messages
    content blocks) name as their run's outputs, each once, in order."""
    found: list[str] = []
    for block in blocks if isinstance(blocks, list) else ():
        kind = block.get("type") if isinstance(block, dict) else None
        if not (isinstance(kind, str) and kind.endswith(_CODE_RESULT_SUFFIX)):
            continue
        result = block.get("content")
        outputs = result.get("content") if isinstance(result, dict) else None
        for output in outputs if isinstance(outputs, list) else ():
            if not isinstance(output, dict):
                continue
            output_kind, file_id = output.get("type"), output.get("file_id")
            if (
                isinstance(output_kind, str)
                and output_kind.endswith(_OUTPUT_SUFFIX)
                and isinstance(file_id, str)
                and file_id
            ):
                found.append(file_id)
    return tuple(dict.fromkeys(found))


def container_ids(container: Any) -> tuple[str, ...]:
    """The code execution container a Messages answer names (its
    ``container`` object's ``id``), if any."""
    container_id = container.get("id") if isinstance(container, dict) else None
    return (container_id,) if isinstance(container_id, str) and container_id else ()


def _files_route(method: str, path: str) -> RouteKind:
    """How the Files API route ``method`` ``path`` is handled (for a request
    carrying the Anthropic marker alone): the upload, the list, a file's
    metadata and its content are CHAT, delete redact-only."""
    if (method in ("POST", "GET") and path == _FILES) or (
        method == "GET" and (_FILE_RE.fullmatch(path) or _FILE_CONTENT_RE.fullmatch(path))
    ):
        return RouteKind.CHAT
    if method == "DELETE" and _FILE_RE.fullmatch(path):
        return RouteKind.REDACT_ONLY
    return RouteKind.NONE


def _creates_message(path: str) -> bool:
    return path.rstrip("/") == "/v1/messages"


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
        # both directions: recognized, redact-only (a body-less no-op).
        if (
            (method == "GET" and (path == "/v1/messages/batches" or _BATCH_ITEM_RE.fullmatch(path)))
            or (method == "POST" and _BATCH_CANCEL_RE.fullmatch(path))
            or (method == "DELETE" and _BATCH_ITEM_RE.fullmatch(path))
        ):
            return RouteKind.REDACT_ONLY
        return RouteKind.NONE

    def matches_request(
        self,
        method: str,
        path: str,
        headers: "Mapping[str, str] | None" = None,
        query: str = "",
    ) -> RouteKind:
        # GET /v1/models is shared with OpenAI: Anthropic's when the request
        # carries anthropic-version (every Anthropic SDK request does) and no
        # other provider's marker. Metadata only — recognized, redact-only.
        if provider_markers(headers, query) == {"anthropic"}:
            if method == "GET" and _MODELS_RE.fullmatch(path):
                return RouteKind.REDACT_ONLY
            files = _files_route(method, path)
            if files is not RouteKind.NONE:
                return files
        return self.matches(method, path)

    def redacts_multipart(self, path: str) -> bool:
        return path == _FILES

    def redact_multipart(
        self,
        path: str,
        body: bytes,
        boundary: bytes,
        redactor: Redactor,
        *,
        inject_note: bool,
        require_scanned: bool = False,
        forward_binary: Callable[[int], None] | None = None,
    ) -> bytes | None:
        # The Files upload: the document part (and any form field) to the
        # shared document policy, every part's file name redacted.
        return redact_files_upload(
            body,
            boundary,
            redactor,
            require_scanned=require_scanned,
            forward_binary=forward_binary,
        )

    def rehydrate_raw_body(self, path: str, raw: bytes, rehydrator: Rehydrator) -> bytes | None:
        if _FILE_CONTENT_RE.fullmatch(path) is None:
            return None
        return rehydrate_download(raw, rehydrator)

    def lists_objects(self, method: str, path: str) -> bool:
        return method == "GET" and path == _FILES

    def listing_items(self, body: Any) -> list[Any] | None:
        # ``{"data": [...], "has_more", "first_id", "last_id"}``.
        items = body.get("data") if isinstance(body, dict) else None
        return items if isinstance(items, list) else None

    def tracks_object_ids(self, method: str, path: str, body: Any = None) -> bool:
        # A message batch (its later results are read by id), a Files API
        # upload (POST /v1/files with anthropic-version, read back by id and
        # cited by later messages as a document or container_upload
        # `file_id`), and a message: the
        # files its code execution runs wrote (generated_file_ids) and the
        # container they ran in (container_ids).
        tail = path.rstrip("/")
        return method == "POST" and (
            tail.endswith("/messages/batches") or tail in ("/v1/files", "/v1/messages")
        )

    def object_ids_from_body(self, method: str, path: str, body: Any) -> tuple[str, ...]:
        if _creates_message(path):
            # The message id names no stored object; its generated files and
            # the container its code ran in do.
            if not isinstance(body, dict):
                return ()
            found = generated_file_ids(body.get("content")) + container_ids(body.get("container"))
            return tuple(dict.fromkeys(found))
        if isinstance(body, dict) and isinstance(body.get("id"), str) and body["id"]:
            return (str(body["id"]),)
        return ()

    def object_ids_from_event(self, method: str, path: str, event: SSEEvent) -> tuple[str, ...]:
        # A streamed message: a code execution result arrives WHOLE in its
        # content_block_start (server tool results are never streamed as
        # deltas); message_start's content is read too, for completeness.
        # The container: message_start's message, or message_delta's delta.
        if not _creates_message(path):
            return super().object_ids_from_event(method, path, event)
        if _FILE_ID_KEY not in event.data and _CONTAINER_KEY not in event.data:
            return ()
        try:
            payload = loads_bounded(event.data)
        except ValueError:
            return ()
        if not isinstance(payload, dict):
            return ()
        kind = payload.get("type")
        if kind == "content_block_start":
            return generated_file_ids([payload.get("content_block")])
        if kind == "message_delta":
            delta = payload.get("delta")
            return container_ids(delta.get("container") if isinstance(delta, dict) else None)
        message = payload.get("message") if kind == "message_start" else None
        if not isinstance(message, dict):
            return ()
        found = generated_file_ids(message.get("content")) + container_ids(message.get("container"))
        return tuple(dict.fromkeys(found))

    def reports_object_ids_once(self, method: str, path: str) -> bool:
        # A message's files are named block by block: every event is read.
        return not _creates_message(path)

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        # count_tokens accepts the same `system` field as /v1/messages, and
        # the note is part of what the real request will carry — keep the
        # count honest (this preserves pre-hook behavior exactly); a batch
        # create injects per entry. The legacy /v1/complete body has no
        # system field (a note would 400), and the metadata routes carry no
        # Messages body at all.
        return path in _NOTE_PATHS

    def error_body(self, message: str, *, status: int = 413) -> dict[str, Any]:
        # Routing (the llm-redact-pro routing layer) adds the budget 402 and
        # the local count_tokens 404; each maps to the SDK's matching error
        # class so the tool classifies the refusal the way the API would. The
        # 429 entry is the API's own class for completeness only: no
        # proxy-generated 429 exists — every 429 a client sees is the
        # upstream's own body.
        error_type = {
            402: "billing_error",
            404: "not_found_error",
            413: "request_too_large",
            429: "rate_limit_error",
            502: "api_error",
        }.get(status, "invalid_request_error")
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
            payload = loads_bounded(line)
        except ValueError:
            return line
        if not isinstance(payload, dict):
            return line
        rehydrated = transform_strings(payload, pool.rehydrate_whole)
        if rehydrated == payload:
            return line
        return json_bytes(rehydrated)

    def rehydrate_event(self, event: SSEEvent, pool: RehydratorPool) -> list[SSEEvent]:
        if event.event in _PASSTHROUGH_EVENTS or not event.data:
            return [event]
        try:
            payload = loads_bounded(event.data)
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
                event.data = json_text({**payload, "completion": new_text})
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
                event.data = json_text(item)
                events.append(event)
            else:
                # Synthetic flush delta injected before a stop.
                events.append(SSEEvent(event="content_block_delta", data=json_text(item)))
        return events
