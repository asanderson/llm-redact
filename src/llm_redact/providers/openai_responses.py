"""OpenAI Responses API adapter (/v1/responses — Codex CLI traffic).

Event names and payload shapes follow the public API as of this adapter's
writing; they are deliberately isolated here (see the fixture-driven tests)
because event-shape drift is the top risk for this endpoint.

Synthetic flush events omit ``sequence_number``: renumbering would require
rewriting every subsequent event. Documented deviation.
"""

import re
from collections.abc import Hashable
from typing import Any

from llm_redact.jsonwalk import json_text, loads_bounded, transform_strings
from llm_redact.providers.base import (
    SYSTEM_NOTE,
    ProviderAdapter,
    RouteKind,
    restore_mcp_tools,
    strip_mcp_tools,
)
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent

_RESPONSE_ID_PATH = re.compile(r"/v1/responses/[^/]+")
_INPUT_ITEMS_PATH = re.compile(r"/v1/responses/[^/]+/input_items")
# Two POSTs that take a responses.create body and create nothing stored:
# compaction (Codex CLI compacts long sessions — the whole conversation) and
# the input-token count. Suffixes, so every prefix that wraps this adapter
# (Azure's /openai and /openai/v1, custom providers) reads them the same.
_COMPACT = "/responses/compact"
_INPUT_TOKENS = "/responses/input_tokens"

# type → (channel kind, delta field, json_source)
_DELTA_EVENTS = {
    "response.output_text.delta": ("text", "delta", False),
    "response.refusal.delta": ("refusal", "delta", False),
    "response.function_call_arguments.delta": ("args", "delta", True),
    # MCP connector calls stream their arguments exactly like function
    # calls: raw JSON source, keyed per item.
    "response.mcp_call_arguments.delta": ("args", "delta", True),
}

# type → field carrying the repeated full value on the done event
_DONE_EVENTS = {
    "response.output_text.done": ("text", "text", False),
    "response.refusal.done": ("refusal", "refusal", False),
    "response.function_call_arguments.done": ("args", "arguments", True),
    "response.mcp_call_arguments.done": ("args", "arguments", True),
}

_TERMINAL_EVENTS = frozenset({"response.completed", "response.failed", "response.incomplete"})

# The files the code interpreter WROTE into its container, as an answer
# names them: an output_text part's ``container_file_citation`` annotation
# (``file_id`` + ``container_id``), and a code interpreter call's output
# entries of type ``files`` (``files[].file_id``; the ``results`` form of
# earlier API versions too). A ``file_citation`` names a file the request's
# file_search read — the user's own upload, never the provider's creation.
# A code interpreter call — and a citation of a file it wrote — also names
# its CONTAINER (``container_id``): one the call created (``{"type":
# "auto"}``), reported as the requester's — a container the request itself
# names is an existing one, never reported (the proxy drops every id the
# request body carries).
_CONTAINER_CITATION = "container_file_citation"
_CODE_INTERPRETER_CALL = "code_interpreter_call"
_CALL_OUTPUT_KEYS = ("outputs", "results")
# The keys every event naming such a file or container carries (never
# inside a JSON string: a quote there is escaped), so an event without one
# is never parsed.
_FILE_ID_KEY = '"file_id"'
_CONTAINER_ID_KEY = '"container_id"'


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _annotation_file(annotation: Any, found: list[str]) -> None:
    if isinstance(annotation, dict) and annotation.get("type") == _CONTAINER_CITATION:
        for key in ("file_id", "container_id"):
            object_id = annotation.get(key)
            if isinstance(object_id, str) and object_id:
                found.append(object_id)


def _part_files(part: Any, found: list[str]) -> None:
    annotations = part.get("annotations") if isinstance(part, dict) else None
    for annotation in annotations if isinstance(annotations, list) else ():
        _annotation_file(annotation, found)


def _item_files(item: Any, found: list[str]) -> None:
    if not isinstance(item, dict):
        return
    if item.get("type") == _CODE_INTERPRETER_CALL:
        container_id = item.get("container_id")
        if isinstance(container_id, str) and container_id:
            found.append(container_id)
        for key in _CALL_OUTPUT_KEYS:
            for output in _dicts(item.get(key)):
                for file in _dicts(output.get("files")):
                    file_id = file.get("file_id")
                    if isinstance(file_id, str) and file_id:
                        found.append(file_id)
        return
    content = item.get("content")
    for part in content if isinstance(content, list) else ():
        _part_files(part, found)


def container_file_ids(response: Any) -> tuple[str, ...]:
    """The container files a Response's ``output`` names — and the
    containers its code interpreter calls ran in — each once."""
    found: list[str] = []
    output = response.get("output") if isinstance(response, dict) else None
    for item in output if isinstance(output, list) else ():
        _item_files(item, found)
    return tuple(dict.fromkeys(found))


def _event_container_files(payload: dict[str, Any]) -> tuple[str, ...]:
    """The container files one stream event names: an added annotation,
    a finished content part or output item, a terminal event's Response."""
    found: list[str] = []
    _annotation_file(payload.get("annotation"), found)
    _part_files(payload.get("part"), found)
    _item_files(payload.get("item"), found)
    found.extend(container_file_ids(payload.get("response")))
    return tuple(dict.fromkeys(found))


def _creates_response(path: str) -> bool:
    """POST to the Responses collection itself (under any prefix: OpenAI's
    /v1, Azure's /openai and /openai/v1, a custom provider's)."""
    return path.rstrip("/").endswith("/responses")


# Every event type this adapter knows about — handled or deliberately passed
# through. The live-API drift detector asserts observed ⊆ KNOWN_EVENT_TYPES,
# so a new event name introduced by the API fails loudly instead of being
# silently forwarded without scrutiny.
KNOWN_EVENT_TYPES: frozenset[str] = (
    frozenset(_DELTA_EVENTS)
    | frozenset(_DONE_EVENTS)
    | _TERMINAL_EVENTS
    | frozenset(
        {
            "response.created",
            "response.in_progress",
            "response.queued",
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.annotation.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.mcp_call.in_progress",
            "response.mcp_call.completed",
            "response.mcp_call.failed",
            "response.mcp_list_tools.in_progress",
            "response.mcp_list_tools.completed",
            "response.mcp_list_tools.failed",
            "error",
        }
    )
)


class OpenAIResponsesAdapter(ProviderAdapter):
    # Shares [providers.openai] upstream config with the chat adapter on
    # purpose: both endpoints live on the same API host.
    name = "openai"

    def prepare_request(
        self,
        body: dict[str, Any],
        redactor: Redactor,
        *,
        inject_note: bool,
        mcp_exempt: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        # tools[].type == "mcp" entries carry provider-directed MCP server
        # config (server_url, headers/authorization): the provider must
        # receive the real credential, so these blocks bypass redaction —
        # stripped first so nothing in them is counted as a detection.
        stripped = strip_mcp_tools(body)
        prepared = super().prepare_request(
            stripped, redactor, inject_note=inject_note, mcp_exempt=mcp_exempt
        )
        return restore_mcp_tools(body, prepared)  # type: ignore[no-any-return]

    def matches(self, method: str, path: str) -> RouteKind:
        if method == "POST" and path == "/v1/responses":
            return RouteKind.CHAT
        if method == "POST" and path == "/v1" + _COMPACT:
            # The window to compact goes up redacted; the compacted window
            # (response.compaction: the retained messages and tool calls,
            # then one opaque, encrypted compaction item) comes back
            # restored. It never streams.
            return RouteKind.CHAT
        if method == "POST" and path == "/v1" + _INPUT_TOKENS:
            # A create body in, a number out: nothing to restore.
            return RouteKind.REDACT_ONLY
        # A stored response fetched later must be rehydrated too, or the
        # client sees the placeholders that were sent upstream.
        if method == "GET" and (
            _RESPONSE_ID_PATH.fullmatch(path) or _INPUT_ITEMS_PATH.fullmatch(path)
        ):
            # Stored responses AND their input-item echoes both repeat
            # content that carried placeholders upstream.
            return RouteKind.CHAT
        if method == "DELETE" and _RESPONSE_ID_PATH.fullmatch(path):
            # A stored response's delete carries its id only: recognized
            # (a no-op redaction), like the chat adapter's deletes.
            return RouteKind.REDACT_ONLY
        return RouteKind.NONE

    def error_body(self, message: str, *, status: int = 413) -> dict[str, Any]:
        return {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "request_too_large" if status == 413 else None,
            }
        }

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        # A compaction is a chat route: the note joins its instructions, so
        # the compacted state carries every token forward exactly, as the
        # requests it condenses did. The input-token count takes the create
        # body it counts, and that request carries the note: counted too
        # (Anthropic count_tokens' stance).
        return kind is RouteKind.CHAT or path.endswith(_INPUT_TOKENS)

    def inject_system_note(self, body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        instructions = body.get("instructions")
        if isinstance(instructions, str) and instructions:
            body["instructions"] = f"{instructions}\n\n{SYSTEM_NOTE}"
        else:
            body["instructions"] = SYSTEM_NOTE
        return body

    def response_id_from_body(self, body: Any) -> str | None:
        if isinstance(body, dict) and isinstance(body.get("id"), str):
            return str(body["id"])
        return None

    def tracks_object_ids(self, method: str, path: str, body: Any = None) -> bool:
        # A Response's answer names the files its code interpreter wrote
        # (container_file_ids), reported as the creator's. A read by id, a
        # cancel or the input items create nothing and are never tracked.
        return method == "POST" and _creates_response(path)

    def object_ids_from_body(self, method: str, path: str, body: Any) -> tuple[str, ...]:
        return container_file_ids(body)

    def object_ids_from_event(self, method: str, path: str, event: SSEEvent) -> tuple[str, ...]:
        if _FILE_ID_KEY not in event.data and _CONTAINER_ID_KEY not in event.data:
            return ()
        try:
            payload = loads_bounded(event.data)
        except ValueError:
            return ()
        return _event_container_files(payload) if isinstance(payload, dict) else ()

    def reports_object_ids_once(self, method: str, path: str) -> bool:
        # Files are named as their annotations and items arrive.
        return False

    def response_id_from_event(self, event: SSEEvent) -> str | None:
        if event.event != "response.created" or not event.data:
            return None
        try:
            payload = loads_bounded(event.data)
        except ValueError:
            return None
        response = payload.get("response")
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            return str(response["id"])
        return None

    def rehydrate_body(self, body: Any, rehydrator: Rehydrator) -> Any:
        # Output items carry text in content[].text and raw JSON source in
        # function-call `arguments` — same split as the chat adapter.
        return transform_strings(
            body,
            rehydrator.rehydrate_text,
            key_overrides={"arguments": rehydrator.rehydrate_json_source_text},
        )

    def _rehydrate_embedded(self, obj: Any, pool: RehydratorPool) -> Any:
        return transform_strings(
            obj,
            pool.rehydrate_whole,
            key_overrides={
                "arguments": lambda s: pool.rehydrate_whole(s, json_source=True),
            },
        )

    @staticmethod
    def _channel_key(kind: str, payload: dict[str, Any]) -> tuple[Any, ...]:
        item_id = payload.get("item_id")
        output_index = payload.get("output_index", 0)
        if kind == "args":
            return (item_id, "args", output_index)
        return (item_id, kind, output_index, payload.get("content_index", 0))

    @staticmethod
    def _synthetic_delta(key: tuple[Any, ...], leftover: str) -> SSEEvent:
        if key[1] == "args":
            payload: dict[str, Any] = {
                "type": "response.function_call_arguments.delta",
                "item_id": key[0],
                "output_index": key[2],
                "delta": leftover,
            }
        else:
            event_type = (
                "response.output_text.delta" if key[1] == "text" else "response.refusal.delta"
            )
            payload = {
                "type": event_type,
                "item_id": key[0],
                "output_index": key[2],
                "content_index": key[3],
                "delta": leftover,
            }
        return SSEEvent(event=payload["type"], data=json_text(payload))

    def _flush_to_events(self, leftovers: dict[Hashable, str]) -> list[SSEEvent]:
        return [
            self._synthetic_delta(key, text)
            for key, text in leftovers.items()
            if isinstance(key, tuple) and text
        ]

    def rehydrate_event(self, event: SSEEvent, pool: RehydratorPool) -> list[SSEEvent]:
        if not event.data or event.data.strip() == "[DONE]":
            # The Responses API ends on response.completed; [DONE] is handled
            # defensively for compatibility.
            if event.data.strip() == "[DONE]":
                return [*self._flush_to_events(pool.flush_all()), event]
            return [event]
        try:
            payload = loads_bounded(event.data)
        except ValueError:
            return [event]
        event_type = payload.get("type")

        if event_type in _DELTA_EVENTS:
            kind, field, json_source = _DELTA_EVENTS[event_type]
            key = self._channel_key(kind, payload)
            if isinstance(payload.get(field), str):
                payload[field] = pool.get(key, json_source=json_source).feed(payload[field])
                event.data = json_text(payload)
            return [event]

        if event_type in _DONE_EVENTS:
            kind, field, json_source = _DONE_EVENTS[event_type]
            key = self._channel_key(kind, payload)
            leftover = pool.flush(key)
            synthetic = [self._synthetic_delta(key, leftover)] if leftover else []
            if isinstance(payload.get(field), str):
                payload[field] = pool.rehydrate_whole(payload[field], json_source=json_source)
                event.data = json_text(payload)
            return [*synthetic, event]

        if event_type == "response.content_part.done":
            part = payload.get("part")
            if isinstance(part, dict):
                payload["part"] = self._rehydrate_embedded(part, pool)
                event.data = json_text(payload)
            return [event]

        if event_type == "response.output_item.done":
            item = payload.get("item")
            synthetic = []
            if isinstance(item, dict):
                item_id = item.get("id")
                synthetic = self._flush_to_events(
                    pool.flush_matching(lambda k: isinstance(k, tuple) and k[0] == item_id)
                )
                payload["item"] = self._rehydrate_embedded(item, pool)
                event.data = json_text(payload)
            return [*synthetic, event]

        if event_type in _TERMINAL_EVENTS:
            synthetic = self._flush_to_events(pool.flush_all())
            response = payload.get("response")
            if isinstance(response, dict):
                payload["response"] = self._rehydrate_embedded(response, pool)
                event.data = json_text(payload)
            return [*synthetic, event]

        # created / in_progress / output_item.added / content_part.added /
        # annotation and reasoning events / error: pass through untouched.
        return [event]
