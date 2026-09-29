"""WebSocket relay for realtime provider APIs (the ``realtime`` extra).

OpenAI Realtime and Gemini Live speak WebSocket, so their traffic never
touches the HTTP proxy path. This module relays a client WS connection to
the provider's wss endpoint (derived from the SAME ``[providers.*]``
upstream the HTTP adapter uses, scheme-swapped), redacting outbound JSON
events and rehydrating inbound ones.

Design rules, matching the HTTP side:
- A frame the adapter cannot positively parse — non-JSON text, binary the
  adapter doesn't claim, unknown event shapes — is forwarded
  byte-identically. Never break the tool.
- Upgrade headers and query strings (Gemini carries ``?key=``; browser
  OpenAI clients carry the key in a subprotocol) pass through untouched
  and are NEVER logged. Log lines carry path and counts only. The one
  exception is a provider authorized with the proxy's OWN cloud identity
  (``auth = "identity"``: Azure OpenAI Realtime, the Vertex AI Live API):
  every client credential channel — headers, query, subprotocols — is
  stripped and the upgrade is authorized by the registered
  ``UpstreamAuth``, on the exact documented paths only.
- Without the ``websockets`` package, uvicorn itself refuses upgrades
  before this module runs (its auto WS protocol is None); the handler
  also guards the import so other servers and test transports degrade to
  a clean close instead of a traceback.
- One ``record_request`` per connection at close (streamed, duration,
  detection/rehydration counts) — metrics, /recent, /events, audit, and
  otel all inherit from that single call.

Sessions: realtime connections use the STATIC vault session. The
per-conversation router derives namespaces from a first-user-message
anchor that does not exist at upgrade time; rather than guess (and risk
cross-conversation restores), the fallback session owns WS traffic. The
session comes from ``state.context_for`` like HTTP, so a session router
that scopes by user (llm-redact-pro's named users) hands each user's
connection that user's own copy of the static session. docs/providers.md
documents this.

Token floors: the provider holds a realtime conversation, so a token any
earlier client frame carried is still in it. A connection keeps a RUNNING
floor (``frame_floors`` of every client frame, raised before the frame is
redacted): a new value is never numbered onto a token the conversation
already holds — the per-request floor of the HTTP path, per connection.
"""

import asyncio
import contextlib
import json
import logging
import time
import urllib.parse
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from llm_redact.audit import AuditWriteError
from llm_redact.jsonwalk import STRUCTURAL_KEYS, transform_strings
from llm_redact.placeholders import json_floors, may_carry_tokens, token_floors
from llm_redact.plugin_api import UpstreamAuthError
from llm_redact.providers.base import SYSTEM_NOTE, restore_mcp_tools, strip_mcp_tools
from llm_redact.redactor import BlockedRequest, Redactor, UnredactableRequest
from llm_redact.rehydrate import RehydratorPool

if TYPE_CHECKING:
    from llm_redact.plugin_api import UpstreamAuth
    from llm_redact.proxy import ProxyState, RequestContext

logger = logging.getLogger("llm_redact")


class _TeeCounter(Counter[str]):
    """Counts this connection's detections while forwarding increments to
    the process total. The HTTP path diffs the shared counter around its
    synchronous redact call; a WS connection redacts incrementally across
    awaits, so concurrent connections would corrupt each other's diffs —
    per-connection counting is the only exact answer here."""

    def __init__(self, total: "Counter[str]") -> None:
        super().__init__()
        self._total = total

    def __setitem__(self, key: str, value: int) -> None:
        self._total[key] += value - self[key]
        super().__setitem__(key, value)


# Generous frame cap both directions: realtime events embed base64 audio
# chunks; the default 1 MiB in `websockets` is too small for those.
MAX_FRAME_BYTES = 16 * 1024 * 1024

# Upgrade-mechanics headers the upstream client library must generate
# itself; everything else (authorization, x-goog-api-key, openai-beta,
# user-agent, ...) passes through verbatim.
_HOP_HEADERS = frozenset(
    {
        "host",
        "connection",
        "upgrade",
        "sec-websocket-key",
        "sec-websocket-version",
        "sec-websocket-extensions",
        "sec-websocket-protocol",  # negotiated separately, see subprotocols
        "content-length",
    }
)


def websockets_available() -> bool:
    """Import guard; monkeypatched by tests to pin the no-extra behavior."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


class WsAdapter:
    """Base realtime adapter: path matching plus per-frame rewriting.

    The base implementation is a byte-identical pass-through; provider
    subclasses override the rewrite hooks. ``rehydrate_message`` receives
    the connection's RehydratorPool so token fragments split across
    frames reassemble exactly like the SSE/NDJSON paths, and returns a
    LIST of frames — a flushed leftover becomes its own synthetic delta
    frame ahead of the event that triggered the flush.
    """

    name = "ws"
    provider = ""
    # The exact paths this adapter lets the proxy authorize with its OWN
    # cloud identity (``[providers.NAME] auth = "identity"``): only the
    # documented realtime endpoints, never a subpath or look-alike — the
    # identity is lent to the routes llm-redact recognizes and redacts.
    identity_paths: frozenset[str] = frozenset()

    def matches(self, path: str) -> bool:
        raise NotImplementedError

    def authorizable(self, path: str) -> bool:
        return path in self.identity_paths

    def redact_message(
        self,
        data: str | bytes,
        ctx: "RequestContext",
        *,
        inject_note: bool = False,
        require_json: bool = False,
    ) -> str | bytes:
        """Rewrite one client frame. ``require_json`` (identity auth) turns
        the verbatim forward of a frame the adapter cannot walk into
        ``UnredactableRequest``: the proxy's own identity never carries an
        unscanned frame."""
        return _unparsed_frame(data, require_json)

    def rehydrate_message(self, data: str | bytes, pool: RehydratorPool) -> list[str | bytes]:
        return [data]

    def _inject_note(self, payload: Any) -> None:
        """Append SYSTEM_NOTE to positively-recognized instruction fields,
        in place. NEVER creates the field: realtime session updates replace
        instructions wholesale, so a created field would overwrite the
        provider's server-side default (the Ollama Modelfile stance)."""


# Realtime events add enum/identifier fields the HTTP APIs don't have, plus
# base64 audio under `audio` (never `data`). Everything else follows the
# HTTP rule: walk every string value so unknown future event shapes stay
# covered (a missed redaction is a leak; the skip set guards the enums).
# Like every skip set, it skips SCALARS only (jsonwalk): the GA `audio`
# session object (its transcription prompt is user text), a `format` or
# `tool_choice` object, and any user JSON reusing these names are walked;
# the modalities arrays are skipped at their schema positions
# (jsonwalk.ENUM_LIST_POSITIONS).
_REALTIME_STRUCTURAL_KEYS = STRUCTURAL_KEYS | frozenset(
    {
        "audio",
        "event_id",
        "item_id",
        "previous_item_id",
        "response_id",
        "session_id",
        "object",
        "status",
        "voice",
        "modalities",
        "output_modalities",
        "input_audio_format",
        "output_audio_format",
        "format",
        "tool_choice",
        "turn_detection",
        "eagerness",
    }
)

# Server event type → (channel kind, delta field). The GA API renamed the
# beta delta events; both spellings are handled so either vintage works.
_RT_DELTA_EVENTS = {
    "response.text.delta": ("text", "delta", False),
    "response.output_text.delta": ("text", "delta", False),
    "response.audio_transcript.delta": ("transcript", "delta", False),
    "response.output_audio_transcript.delta": ("transcript", "delta", False),
    "response.function_call_arguments.delta": ("args", "delta", True),
    "response.mcp_call_arguments.delta": ("args", "delta", True),
}

# Server event type → field carrying the repeated full value.
_RT_DONE_EVENTS = {
    "response.text.done": ("text", "text", False),
    "response.output_text.done": ("text", "text", False),
    "response.audio_transcript.done": ("transcript", "transcript", False),
    "response.output_audio_transcript.done": ("transcript", "transcript", False),
    "response.function_call_arguments.done": ("args", "arguments", True),
    "response.mcp_call_arguments.done": ("args", "arguments", True),
}

# Events whose embedded object is rehydrated whole: items echo the redacted
# input back to the client (restoring is correct — the client owns the
# originals), sessions echo redacted instructions, response.done embeds the
# final output items.
_RT_EMBEDDED_EVENTS = {
    "conversation.item.created": "item",
    "conversation.item.added": "item",
    "conversation.item.done": "item",
    "conversation.item.retrieved": "item",
    "session.created": "session",
    "session.updated": "session",
    "response.content_part.done": "part",
}

# Every server event type this adapter knows about — handled or deliberately
# passed through. The live drift test asserts observed ⊆ this set, so a new
# event name introduced by the API fails loudly (the adapter forwards
# unknown frames verbatim, so drift is a schema signal, not a crash).
KNOWN_REALTIME_EVENT_TYPES: frozenset[str] = (
    frozenset(_RT_DELTA_EVENTS)
    | frozenset(_RT_DONE_EVENTS)
    | frozenset(_RT_EMBEDDED_EVENTS)
    | frozenset(
        {
            "error",
            "conversation.created",
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.failed",
            "conversation.item.input_audio_transcription.segment",
            "conversation.item.truncated",
            "conversation.item.deleted",
            "input_audio_buffer.committed",
            "input_audio_buffer.cleared",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.timeout_triggered",
            "response.created",
            "response.done",
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.audio.delta",
            "response.audio.done",
            "response.output_audio.delta",
            "response.output_audio.done",
            "rate_limits.updated",
            "output_audio_buffer.started",
            "output_audio_buffer.stopped",
            "output_audio_buffer.cleared",
            "mcp_list_tools.in_progress",
            "mcp_list_tools.completed",
            "mcp_list_tools.failed",
            "response.mcp_call.in_progress",
            "response.mcp_call.completed",
            "response.mcp_call.failed",
        }
    )
)


class OpenAIRealtimeWs(WsAdapter):
    """/v1/realtime (beta and GA vocabularies).

    Outbound: every string value in every client event is redacted through
    the realtime skip set — conversation.item.create content, response
    and session instructions, and function_call_output payloads included;
    base64 ``audio`` never touches the detectors (the media non-goal).

    Inbound: delta events feed StreamingRehydrator channels keyed
    (item_id, kind, output_index, content_index) so tokens split across
    frames reassemble; ``*.done`` events flush their channel (leftover →
    synthetic delta frame first) and rehydrate the repeated full value;
    ``response.done`` flushes everything and rehydrates the embedded
    response. Frames that fail to parse are forwarded byte-identically.
    """

    name = "openai-realtime"
    provider = "openai"

    def matches(self, path: str) -> bool:
        return path == "/v1/realtime" or path.startswith("/v1/realtime/")

    def redact_message(
        self,
        data: str | bytes,
        ctx: "RequestContext",
        *,
        inject_note: bool = False,
        require_json: bool = False,
    ) -> str | bytes:
        parsed = parse_json_text(data)
        if parsed is None:
            return _unparsed_frame(data, require_json)
        payload, was_binary = parsed
        # MCP connector tool entries (session/response tools with
        # type == "mcp") are provider-directed config whose credentials
        # the provider must receive unredacted — stripped before the walk,
        # restored after (nothing in them is counted).
        redacted = transform_strings(
            strip_mcp_tools(payload), ctx.redactor.redact_text, skip_keys=_REALTIME_STRUCTURAL_KEYS
        )
        redacted = restore_mcp_tools(payload, redacted)
        if inject_note:
            self._inject_note(redacted)
        return _dump_frame(redacted, was_binary)

    def _inject_note(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        target_field = {"session.update": "session", "response.create": "response"}.get(
            str(payload.get("type"))
        )
        if target_field is None:
            return
        target = payload.get(target_field)
        if not isinstance(target, dict):
            return
        instructions = target.get("instructions")
        # Only when present and non-empty (updates replace instructions
        # wholesale — creating them would clobber the server default), and
        # only once (clients may resend the session they were handed).
        if isinstance(instructions, str) and instructions and SYSTEM_NOTE not in instructions:
            target["instructions"] = f"{instructions}\n\n{SYSTEM_NOTE}"

    @staticmethod
    def _channel_key(kind: str, payload: dict[str, Any]) -> tuple[Any, ...]:
        return (
            payload.get("item_id"),
            kind,
            payload.get("output_index", 0),
            payload.get("content_index", 0),
        )

    def _synthetic_delta(self, key: tuple[Any, ...], leftover: str) -> dict[str, Any]:
        event_type = {
            "text": "response.output_text.delta",
            "transcript": "response.output_audio_transcript.delta",
            "args": "response.function_call_arguments.delta",
        }[key[1]]
        return {
            "type": event_type,
            "item_id": key[0],
            "output_index": key[2],
            "content_index": key[3],
            "delta": leftover,
        }

    def _flush_frames(self, leftovers: dict[Any, str], was_binary: bool) -> list[str | bytes]:
        return [
            _dump_frame(self._synthetic_delta(key, text), was_binary)
            for key, text in leftovers.items()
            if isinstance(key, tuple) and text
        ]

    def _rehydrate_embedded(self, obj: Any, pool: RehydratorPool) -> Any:
        return transform_strings(
            obj,
            pool.rehydrate_whole,
            key_overrides={"arguments": lambda s: pool.rehydrate_whole(s, json_source=True)},
        )

    def rehydrate_message(self, data: str | bytes, pool: RehydratorPool) -> list[str | bytes]:
        parsed = parse_json_text(data)
        if parsed is None:
            return [data]
        payload, was_binary = parsed
        if not isinstance(payload, dict):
            return [data]
        event_type = payload.get("type")

        if event_type in _RT_DELTA_EVENTS:
            kind, field, json_source = _RT_DELTA_EVENTS[event_type]
            key = self._channel_key(kind, payload)
            if isinstance(payload.get(field), str):
                payload[field] = pool.get(key, json_source=json_source).feed(payload[field])
                return [_dump_frame(payload, was_binary)]
            return [data]

        if event_type in _RT_DONE_EVENTS:
            kind, field, json_source = _RT_DONE_EVENTS[event_type]
            key = self._channel_key(kind, payload)
            leftover = pool.flush(key)
            frames: list[str | bytes] = (
                [_dump_frame(self._synthetic_delta(key, leftover), was_binary)] if leftover else []
            )
            if isinstance(payload.get(field), str):
                payload[field] = pool.rehydrate_whole(payload[field], json_source=json_source)
            frames.append(_dump_frame(payload, was_binary))
            return frames

        if event_type == "response.output_item.done":
            item = payload.get("item")
            frames = []
            if isinstance(item, dict):
                item_id = item.get("id")
                frames = self._flush_frames(
                    pool.flush_matching(lambda k: isinstance(k, tuple) and k[0] == item_id),
                    was_binary,
                )
                payload["item"] = self._rehydrate_embedded(item, pool)
            frames.append(_dump_frame(payload, was_binary))
            return frames

        if event_type == "response.done":
            frames = self._flush_frames(pool.flush_all(), was_binary)
            response = payload.get("response")
            if isinstance(response, dict):
                payload["response"] = self._rehydrate_embedded(response, pool)
            frames.append(_dump_frame(payload, was_binary))
            return frames

        embedded_field = _RT_EMBEDDED_EVENTS.get(str(event_type))
        if embedded_field is not None:
            embedded = payload.get(embedded_field)
            if isinstance(embedded, dict):
                payload[embedded_field] = self._rehydrate_embedded(embedded, pool)
                return [_dump_frame(payload, was_binary)]
            return [data]

        # session/audio/buffer bookkeeping, rate limits, errors, user-audio
        # transcription (audio we never redacted): pass through untouched.
        return [data]


# Azure OpenAI Realtime: the preview form (``?api-version=…&deployment=…``)
# and the GA v1 form (``/openai/v1/realtime?model=<deployment>``, no
# api-version — Microsoft's preview-to-GA migration guide; openai-node's
# Azure client builds exactly this path).
_AZURE_REALTIME_PATHS = frozenset({"/openai/realtime", "/openai/v1/realtime"})


class AzureRealtimeWs(OpenAIRealtimeWs):
    """Azure OpenAI Realtime — the OpenAI Realtime event vocabulary on Azure's
    paths: ``/openai/realtime`` (preview; api-version and deployment in the
    query) and ``/openai/v1/realtime`` (GA; ``model=<deployment>``).

    Everything (the outbound walk, inbound channels, *.done flush, note
    injection into session/response instructions) is inherited from
    OpenAIRealtimeWs; only the path and the upstream provider differ, so the
    connection reaches the customer's own resource wss URL derived from
    ``[providers.azure]``. Matcher proven disjoint from OpenAIRealtimeWs
    (/openai/realtime vs /v1/realtime) and Gemini Live by test.
    """

    name = "azure-realtime"
    provider = "azure"
    identity_paths = _AZURE_REALTIME_PATHS

    def matches(self, path: str) -> bool:
        return path in _AZURE_REALTIME_PATHS or path.startswith(
            ("/openai/realtime/", "/openai/v1/realtime/")
        )


# Gemini Live adds mime/voice/config enums; base64 audio rides in `data`
# (already structural) inside realtimeInput mediaChunks. Scalars only, like
# every skip set: toolResponse.functionResponses[].response is an opaque
# position walked in full (jsonwalk.OPAQUE_POSITIONS). The proto JSON
# mapping accepts the original snake_case field names too (Google's own
# Vertex Live notebook sends `mime_type`, `voice_name`, …), so both
# spellings are skipped.
_GEMINI_LIVE_STRUCTURAL_KEYS = _REALTIME_STRUCTURAL_KEYS | frozenset(
    {
        "mimeType",
        "mime_type",
        "voiceName",
        "voice_name",
        "languageCode",
        "language_code",
        "responseModalities",
        "response_modalities",
        "handle",
    }
)

# Top-level message keys, for the live drift detector (messages are
# unnamed — key-set drift is the analogue of unknown event types).
KNOWN_LIVE_CLIENT_KEYS = frozenset({"setup", "clientContent", "realtimeInput", "toolResponse"})
KNOWN_LIVE_SERVER_KEYS = frozenset(
    {
        "setupComplete",
        "serverContent",
        "toolCall",
        "toolCallCancellation",
        "usageMetadata",
        "goAway",
        "sessionResumptionUpdate",
        "error",
    }
)


class GeminiLiveWs(WsAdapter):
    """BidiGenerateContent (v1alpha/v1beta), JSON over text OR binary frames.

    Outbound: setup.systemInstruction, clientContent turns, realtimeInput
    text, and toolResponse payloads are walked; base64 media (`data`) and
    mime/voice enums never touch the detectors.

    Inbound: serverContent modelTurn parts stream text across messages —
    one channel per (text|thought) kind, mirroring the HTTP Gemini
    adapter; outputTranscription streams on its own channel.
    turnComplete/generationComplete are the flush points: a leftover
    appends to the message's last matching part when it has a modelTurn,
    else it becomes a synthetic serverContent frame ahead of the flush
    message. toolCall functionCalls[].args is a parsed object (plain
    walk). Unparseable frames forward byte-identically.
    """

    name = "gemini-live"
    provider = "gemini"

    def matches(self, path: str) -> bool:
        return path.startswith("/ws/google.ai.generativelanguage.") and path.endswith(
            ".GenerativeService.BidiGenerateContent"
        )

    def redact_message(
        self,
        data: str | bytes,
        ctx: "RequestContext",
        *,
        inject_note: bool = False,
        require_json: bool = False,
    ) -> str | bytes:
        parsed = parse_json_text(data)
        if parsed is None:
            return _unparsed_frame(data, require_json)
        payload, was_binary = parsed
        redacted = transform_strings(
            payload, ctx.redactor.redact_text, skip_keys=_GEMINI_LIVE_STRUCTURAL_KEYS
        )
        if inject_note:
            self._inject_note(redacted)
        return _dump_frame(redacted, was_binary)

    def _inject_note(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        setup = payload.get("setup")
        if not isinstance(setup, dict):
            return
        # camelCase or the proto's snake_case spelling (both accepted).
        instruction = setup.get("systemInstruction", setup.get("system_instruction"))
        if not isinstance(instruction, dict) or not isinstance(instruction.get("parts"), list):
            # Absent systemInstruction stays absent: creating one would
            # change model behavior beyond token preservation.
            return
        parts = instruction["parts"]
        for part in parts:
            if isinstance(part, dict) and part.get("text") == SYSTEM_NOTE:
                return
        parts.append({"text": SYSTEM_NOTE})

    @staticmethod
    def _part_kind(part: dict[str, Any]) -> str:
        return "thought" if part.get("thought") else "text"

    def _rehydrate_parts(self, model_turn: dict[str, Any], pool: RehydratorPool) -> None:
        parts = model_turn.get("parts")
        if not isinstance(parts, list):
            return
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                channel = pool.get(("modelTurn", self._part_kind(part)))
                part["text"] = channel.feed(part["text"])

    def _flush_into(self, payload: dict[str, Any], pool: RehydratorPool) -> list[dict[str, Any]]:
        """Drain every channel; return extra frames to emit first."""
        leftovers = pool.flush_all()
        if not any(leftovers.values()):
            return []
        server_content = payload.get("serverContent")
        model_turn = server_content.get("modelTurn") if isinstance(server_content, dict) else None
        extra: list[dict[str, Any]] = []
        synthetic_parts: list[dict[str, Any]] = []
        for key, text in leftovers.items():
            if not text:
                continue
            if isinstance(key, tuple) and key[0] == "modelTurn":
                part: dict[str, Any] = {"text": text}
                if key[1] == "thought":
                    part["thought"] = True
                appended = False
                if isinstance(model_turn, dict) and isinstance(model_turn.get("parts"), list):
                    for existing in reversed(model_turn["parts"]):
                        if (
                            isinstance(existing, dict)
                            and isinstance(existing.get("text"), str)
                            and self._part_kind(existing) == key[1]
                        ):
                            existing["text"] += text
                            appended = True
                            break
                if not appended:
                    synthetic_parts.append(part)
            elif isinstance(key, tuple) and key[0] == "outputTranscription":
                extra.append({"serverContent": {"outputTranscription": {"text": text}}})
        if synthetic_parts:
            extra.insert(0, {"serverContent": {"modelTurn": {"parts": synthetic_parts}}})
        return extra

    def rehydrate_message(self, data: str | bytes, pool: RehydratorPool) -> list[str | bytes]:
        parsed = parse_json_text(data)
        if parsed is None:
            return [data]
        payload, was_binary = parsed
        if not isinstance(payload, dict):
            return [data]

        server_content = payload.get("serverContent")
        if isinstance(server_content, dict):
            model_turn = server_content.get("modelTurn")
            if isinstance(model_turn, dict):
                self._rehydrate_parts(model_turn, pool)
            transcription = server_content.get("outputTranscription")
            if isinstance(transcription, dict) and isinstance(transcription.get("text"), str):
                channel = pool.get(("outputTranscription",))
                transcription["text"] = channel.feed(transcription["text"])
            frames: list[str | bytes] = []
            if server_content.get("turnComplete") or server_content.get("generationComplete"):
                frames = [
                    _dump_frame(extra, was_binary) for extra in self._flush_into(payload, pool)
                ]
            frames.append(_dump_frame(payload, was_binary))
            return frames

        tool_call = payload.get("toolCall")
        if isinstance(tool_call, dict):
            calls = tool_call.get("functionCalls")
            if isinstance(calls, list):
                tool_call["functionCalls"] = [
                    transform_strings(call, pool.rehydrate_whole) for call in calls
                ]
            return [_dump_frame(payload, was_binary)]

        # setupComplete / usageMetadata / goAway / sessionResumptionUpdate /
        # inputTranscription-only and unknown shapes: pass through.
        return [data]


# Vertex AI Live API: the Gemini Live protocol behind Vertex's regional
# host, ``wss://{region}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.
# {v1|v1beta1}.LlmBidiService/BidiGenerateContent`` (Google's Vertex Live
# notebook uses v1; the google-genai SDK builds the path from its Vertex
# api_version, v1beta1 by default). Bearer-token auth, so the proxy can
# authorize it with its own identity.
_VERTEX_LIVE_PATHS = frozenset(
    f"/ws/google.cloud.aiplatform.{version}.LlmBidiService/BidiGenerateContent"
    for version in ("v1", "v1beta1")
)


class VertexLiveWs(GeminiLiveWs):
    """Gemini Live on Vertex AI (``LlmBidiService``). Message handling —
    setup/clientContent/realtimeInput/toolResponse outbound, serverContent
    and toolCall inbound, the turnComplete flush — is inherited from
    GeminiLiveWs (Vertex speaks the same BidiGenerateContent messages); only
    the path and the upstream (``[providers.vertex]``) differ. Matcher
    disjoint from GeminiLiveWs (google.ai.generativelanguage vs
    google.cloud.aiplatform), proven by test."""

    name = "vertex-live"
    provider = "vertex"
    identity_paths = _VERTEX_LIVE_PATHS

    def matches(self, path: str) -> bool:
        return path in _VERTEX_LIVE_PATHS


ALL_WS_ADAPTERS: tuple[type[WsAdapter], ...] = (
    OpenAIRealtimeWs,
    # Azure Realtime shares the OpenAI vocabulary on /openai/realtime and
    # /openai/v1/realtime; matcher disjoint from OpenAIRealtimeWs
    # (/v1/realtime) and both Gemini Live adapters.
    AzureRealtimeWs,
    GeminiLiveWs,
    VertexLiveWs,
)


def ws_adapter_for(path: str, adapters: list[WsAdapter]) -> WsAdapter | None:
    for adapter in adapters:
        if adapter.matches(path):
            return adapter
    return None


def _request_path(scope: Mapping[str, Any], path: str) -> str:
    """The path to forward, as the client sent it (the HTTP rule in
    proxy._upstream_path): the raw path when it is ASCII, else ``path``."""
    raw_path = scope.get("raw_path")
    try:
        return raw_path.split(b"?", 1)[0].decode("ascii") if raw_path else path
    except UnicodeDecodeError:
        return path


def _upstream_http_url(base_url: str, path: str, query_string: bytes) -> str:
    """The connection's upstream URL in its HTTP form: the configured
    upstream's scheme, host AND base path (an API-management base such as
    ``https://gw.example/my-api`` is a different API from its host root),
    then ``path``, with the RAW query preserved — it may carry credentials,
    so it is forwarded exactly and never re-encoded. This is the form an
    identity authorizer sees (a WebSocket upgrade is an HTTP GET to it)."""
    parsed = urllib.parse.urlsplit(base_url)
    url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}{path}"
    if query_string:
        url += "?" + query_string.decode("latin-1")
    return url


def _ws_form(http_url: str) -> str:
    """The provider's wss URL for this connection: the HTTP form,
    scheme-swapped (https→wss, anything else — local/test upstreams — →ws)."""
    scheme, rest = http_url.split("://", 1)
    return ("wss" if scheme == "https" else "ws") + "://" + rest


# Credential-bearing WebSocket subprotocols. Browsers cannot set upgrade
# headers, so clients smuggle keys in the offered subprotocol list — OpenAI's
# SDK offers `openai-insecure-api-key.<key>`, other clients a bare marker
# entry followed by the secret as the NEXT entry (`["bearer", "<jwt>"]`).
# Under the proxy's own identity only an ALLOWLIST of known non-credential
# offers is forwarded (identity_subprotocols); these markers additionally
# drop the entry after them.
_CREDENTIAL_SUBPROTOCOL_MARKERS = (
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "token",
    "secret",
    "password",
)


def is_credential_subprotocol(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _CREDENTIAL_SUBPROTOCOL_MARKERS)


# The subprotocols a realtime client legitimately offers: OpenAI's SDKs
# (and Azure's, which reuse them) offer `realtime` plus
# `openai-beta.realtime-v1`; google-genai's Live clients (Gemini API and
# Vertex) offer none. Organization/project ids (`openai-organization.*`,
# `openai-project.*`) select an OpenAI billing scope and mean nothing on the
# identity providers, so they are not forwarded either.
_IDENTITY_SUBPROTOCOLS = frozenset({"realtime"})
_IDENTITY_SUBPROTOCOL_PREFIXES = ("openai-beta.",)


def identity_subprotocols(offered: Sequence[str]) -> list[str]:
    """The offered subprotocols forwarded under the proxy's own identity:
    allowlisted, known non-credential values only — anything else may be
    a credential (marker matching can never enumerate every spelling) —
    and never the entry after a credential marker (its value)."""
    kept: list[str] = []
    after_marker = False
    for value in offered:
        if after_marker:
            after_marker = False
            continue
        if is_credential_subprotocol(value):
            # A BARE marker (`bearer`, `authorization`, `openai-insecure-
            # api-key.`) carries its secret in the next entry.
            after_marker = value.lower().rstrip(".-_=: ").endswith(_CREDENTIAL_SUBPROTOCOL_MARKERS)
            continue
        if value in _IDENTITY_SUBPROTOCOLS or value.startswith(_IDENTITY_SUBPROTOCOL_PREFIXES):
            kept.append(value)
    return kept


def _filtered_headers(websocket: WebSocket) -> list[tuple[str, str]]:
    # The proxy's own x-llm-redact-* namespace never reaches the upstream,
    # whether or not an access gate consumed it (same rule as HTTP).
    return [
        (name, value)
        for name, value in websocket.headers.items()
        if name.lower() not in _HOP_HEADERS and not name.lower().startswith("x-llm-redact-")
    ]


def _sendable_close_code(code: int) -> int:
    # 1005 (no status) and 1006 (abnormal) are reserved: they describe how a
    # connection ended and must not appear in a close frame we send.
    return 1000 if code in (1005, 1006) else code


# RFC 6455 §5.5: a control frame payload is at most 125 bytes, two of
# which are the close code.
_MAX_CLOSE_REASON_BYTES = 123


def _close_reason(reason: str) -> str:
    """``reason`` cut to fit a close frame (UTF-8, never mid-character) — an
    oversized reason would fail the close and lose the message entirely."""
    return reason.encode("utf-8")[:_MAX_CLOSE_REASON_BYTES].decode("utf-8", errors="ignore")


async def _reject(websocket: WebSocket, reason: str) -> None:
    """Accept-then-close: unlike a handshake 403, the close reason reaches
    the client library where a user can read it."""
    with contextlib.suppress(Exception):
        await websocket.accept()
        await websocket.close(code=1011, reason=_close_reason(reason))


def _record_ws_refusal(
    state: "ProxyState", adapter: WsAdapter, path: str, status: int, started: float
) -> None:
    """The recorded row for a connection refused before any upstream
    contact by a proxy rule (HTTP records its 403/400 the same way)."""
    state.record_request(
        session=state.config.vault.session,
        provider=adapter.provider,
        method="WS",
        path=path,
        status=status,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
    )


def _record_refused(
    state: "ProxyState",
    ctx: "RequestContext",
    adapter: WsAdapter,
    path: str,
    started: float,
    *,
    audit_token: object | None = None,
) -> None:
    """The recorded 502 for a connection the upstream never got (no
    credential, or the dial failed): one metadata-only row, like HTTP."""
    state.record_request(
        session=ctx.session_id,
        provider=adapter.provider,
        method="WS",
        path=path,
        status=502,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        audit_token=audit_token,
    )


async def _authorize_upgrade(
    state: "ProxyState",
    websocket: WebSocket,
    adapter: WsAdapter,
    upstream_auth: "UpstreamAuth",
    path: str,
    upstream_path: str,
    http_url: str,
    headers: list[tuple[str, str]],
    subprotocols: list[str],
    ctx: "RequestContext",
    started: float,
) -> tuple[str, list[tuple[str, str]], list[str]] | None:
    """The proxy's own cloud identity on a WebSocket upgrade (the HTTP rule:
    strip every client credential, authorize the request as it will be
    sent, send exactly what was authorized).

    Client credentials go from all three channels a WebSocket client can
    use — headers, the query, and the offered subprotocols — then the
    authorizer sees the upgrade as the HTTP GET it is (the https form of
    the upstream URL, an empty body). Returns the URL, headers and
    subprotocols to dial with, or None after refusing the connection (the
    reason names the credential SOURCE only; counted as an upstream error
    and recorded, never forwarded). Headers, URLs and subprotocols are
    never logged."""
    from llm_redact.proxy import _same_upstream, strip_client_credentials, upstream_base_path

    provider = adapter.provider
    http_url, headers = strip_client_credentials(http_url, headers)
    subprotocols = identity_subprotocols(subprotocols)
    base_url = state.config.providers[provider].upstream_base_url
    if not _same_upstream(
        http_url, base_url, exact_path=upstream_base_path(base_url) + upstream_path
    ):
        # Belt and braces behind origin_form_target and the exact
        # identity_paths: the proxy's identity goes to exactly the
        # configured upstream path it was matched on.
        _record_ws_refusal(state, adapter, path, 400, started)
        await _reject(websocket, "the request target must be a path")
        return None
    try:
        headers = await upstream_auth.authorize("GET", http_url, headers, b"")
    except Exception as exc:
        source = str(exc) if isinstance(exc, UpstreamAuthError) else type(exc).__name__
        state.upstream_errors[provider] += 1
        logger.warning(
            "WS %s -> upstream credentials unavailable for %s (%s)", path, provider, source
        )
        _record_refused(state, ctx, adapter, path, started)
        await _reject(
            websocket,
            f"the proxy could not obtain its own {provider} cloud credentials ({source});"
            " nothing was forwarded",
        )
        return None
    return http_url, headers, subprotocols


async def ws_handle(websocket: WebSocket) -> None:
    state: ProxyState = websocket.app.state.proxy
    from llm_redact.proxy import has_dot_segment, origin_form_target

    if not origin_form_target(websocket.scope):
        await _reject(websocket, "the request target must be a path")
        return
    if has_dot_segment(websocket.scope):
        # The HTTP rule: never forwarded, recorded, or logged with its path.
        await _reject(websocket, "the request path must not contain '.' or '..' segments")
        return
    path = websocket.url.path

    if path.startswith("/__llm-redact"):
        await _reject(websocket, "reserved path")
        return
    # Client admission (llm-redact-pro's access gate): same rule as HTTP.
    # The gate scrubs its credentials from the scope; its refusal is
    # accept-then-close so the reason reaches the client library.
    admission = await state.admit(websocket, "websocket")
    path = websocket.scope["path"]
    from llm_redact.proxy import IDENTITY_PATH_PREFIX

    if path.startswith(IDENTITY_PATH_PREFIX):
        # An identity prefix still present after admission: its next segment
        # is a key, so the path is never logged (the HTTP rule), and no
        # realtime route exists under it anyway.
        await _reject(websocket, admission.refusal or "no realtime route for this path")
        return
    if admission.refusal is not None:
        logger.info("WS %s -> refused by the access gate", path)
        await _reject(websocket, admission.refusal)
        return
    if path.startswith("/__llm-redact"):
        await _reject(websocket, "reserved path")
        return
    # Attribute the connection's single record_request row (at close; same
    # task, so the proxy's contextvar carries it).
    from llm_redact.proxy import _REQUEST_USER

    _REQUEST_USER.set(admission.subject)
    adapter = ws_adapter_for(path, state.ws_adapters)
    if adapter is None:
        # Unlike unmatched HTTP traffic there is no default upstream to
        # forward an unknown WS path to; refusing is the only safe answer.
        await _reject(websocket, "no realtime route for this path")
        return
    provider_config = state.config.providers.get(adapter.provider)
    if provider_config is None or not provider_config.upstream_base_url:
        await _reject(websocket, f"[providers.{adapter.provider}] upstream not configured")
        return
    if not provider_config.enabled:
        # Same fail-closed stance as HTTP: a disabled provider must never
        # fall through to any forwarding path.
        logger.info("WS %s -> refused (provider %s disabled)", path, adapter.provider)
        await _reject(websocket, f"provider {adapter.provider} disabled in llm-redact config")
        return
    upstream_auth = state.upstream_auth.get(adapter.provider)
    if provider_config.auth != "passthrough" and (
        upstream_auth is None or not adapter.authorizable(path)
    ):
        # auth = "identity": the proxy lends its own cloud identity only to
        # the realtime endpoints it recognizes (exact documented paths).
        # Anything else — a subpath, or a provider with no authorizer —
        # is refused: forwarding the client's credential (or none) would
        # silently break the configured contract.
        logger.info("WS %s -> refused (provider %s uses identity auth)", path, adapter.provider)
        _record_ws_refusal(state, adapter, path, 403, time.perf_counter())
        await _reject(
            websocket,
            f'[providers.{adapter.provider}] auth = "identity": only the realtime routes'
            " llm-redact recognizes are authorized",
        )
        return
    if not websockets_available():
        await _reject(
            websocket,
            "realtime support requires the websockets package;"
            " install it: uv sync --extra realtime",
        )
        return

    import websockets

    from llm_redact.proxy import RequestContext  # runtime: avoids the import cycle

    started = time.perf_counter()
    static_ctx = state.context_for(None, "GET", path, None)
    if static_ctx.sealed:
        # A session the router says must stay empty cannot carry a
        # conversation whose every message is redacted into it.
        logger.info("WS %s -> refused (sealed session)", path)
        _record_ws_refusal(state, adapter, path, 403, started)
        await _reject(websocket, "the session router sealed this connection's vault session")
        return
    # Thin per-connection wrapper (the context_for pattern: object
    # construction only): a tee counter gives exact per-connection
    # detection counts that still land in the process totals. Its redactor
    # is rebound as the connection's token floor rises (frame_floors).
    connection_counts = _TeeCounter(state.detection_counts)
    ctx = RequestContext(
        static_ctx.session_id,
        static_ctx.vault,
        Redactor(
            state.detectors,
            static_ctx.vault,
            state.allowlist,
            counts=connection_counts,
            modes=state.modes,
            warn_counts=state.warn_counts,
        ),
        static_ctx.rehydrator,
    )
    pool = RehydratorPool(ctx.vault, fuzzy=state.config.rehydration.fuzzy)

    upstream_path = _request_path(websocket.scope, path)
    http_url = _upstream_http_url(
        provider_config.upstream_base_url, upstream_path, websocket.scope.get("query_string", b"")
    )
    from llm_redact.proxy import _same_upstream

    if not _same_upstream(http_url, provider_config.upstream_base_url):
        # The HTTP rule: whatever the path holds, the connection goes to the
        # configured upstream (host AND base path) or nowhere.
        _record_ws_refusal(state, adapter, path, 400, started)
        await _reject(websocket, "the request target must be a path")
        return
    headers = _filtered_headers(websocket)
    subprotocols = list(websocket.scope.get("subprotocols") or [])
    if upstream_auth is not None:
        authorized = await _authorize_upgrade(
            state,
            websocket,
            adapter,
            upstream_auth,
            path,
            upstream_path,
            http_url,
            headers,
            subprotocols,
            ctx,
            started,
        )
        if authorized is None:
            return
        http_url, headers, subprotocols = authorized
    url = _ws_form(http_url)

    # [audit] required: same rule as HTTP — no durably committed audit row,
    # no upstream contact. The START row commits before the upstream dial;
    # frame counts land in the END row at close. Refusal is the standard
    # accept-then-close so the reason reaches the client.
    try:
        audit_token = state.begin_audit(
            session=ctx.session_id,
            provider=adapter.provider,
            method="WS",
            path=path,
            detections={},
        )
    except AuditWriteError as problem:
        logger.critical(
            "WS %s -> refused; audit write failed with [audit] required (%s)",
            path,
            type(problem).__name__,
        )
        await _reject(websocket, "audit log unavailable and [audit] required is enabled")
        return

    try:
        upstream = await websockets.connect(
            url,
            additional_headers=headers,
            subprotocols=[websockets.Subprotocol(s) for s in subprotocols] or None,
            max_size=MAX_FRAME_BYTES,
            open_timeout=30,
        )
    except Exception as problem:  # DNS, TLS, refusals, handshake rejections
        # The exception may embed the URL (query auth!) — log the class only.
        # Counted and recorded like an HTTP upstream fault (the audit START
        # row, if any, gets its END row here).
        logger.warning("WS %s -> upstream connect failed (%s)", path, type(problem).__name__)
        state.upstream_errors[adapter.provider] += 1
        _record_refused(state, ctx, adapter, path, started, audit_token=audit_token)
        await _reject(websocket, "upstream websocket connect failed")
        return

    await websocket.accept(subprotocol=upstream.subprotocol)
    logger.info("WS %s -> connected (provider %s)", path, adapter.provider)
    if not provider_config.detection:
        logger.info(
            "WS %s forwarded unredacted ([providers.%s] detection = false)",
            path,
            adapter.provider,
        )
    status: int | None = 101
    # Under the proxy's own identity a frame the adapter cannot walk is
    # refused, never relayed verbatim (the HTTP body rule; detection = false
    # stays the explicit unredacted opt-out).
    require_json = upstream_auth is not None

    async def close_on_policy(reason: str) -> None:
        # The client FIRST: closing the upstream first lets upstream_to_client
        # mirror the upstream's 1000 to the client ahead of the 1008.
        with contextlib.suppress(RuntimeError):
            await websocket.close(code=1008, reason=_close_reason(reason))
        await upstream.close(code=1000)

    async def client_to_upstream() -> None:
        nonlocal status
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                code = _sendable_close_code(int(message.get("code") or 1000))
                await upstream.close(code=code)
                return
            data: str | bytes
            if message.get("text") is not None:
                data = message["text"]
            else:
                data = message.get("bytes") or b""
            # The connection's running token floor, raised BEFORE this
            # frame's values are numbered: the provider holds the whole
            # conversation, so a token any earlier client frame carried is
            # still in it (the client never resends history, as on HTTP).
            # Upstream frames are not read: model output is provider-side
            # history, as on HTTP (and a session echo carries the proxy's
            # own note, whose example token must not raise the floor).
            ctx.redactor = ctx.redactor.with_floors(frame_floors(data))
            try:
                # [providers.NAME] detection = false applies to realtime
                # frames too: forwarded untouched (rehydration inbound
                # stays active), the same off-switch as the HTTP path.
                outbound = (
                    data
                    if not provider_config.detection
                    else adapter.redact_message(
                        data,
                        ctx,
                        inject_note=state.config.inject_system_note,
                        require_json=require_json,
                    )
                )
                await upstream.send(outbound)
            except UnredactableRequest as refused:
                # Closed like a block: the frame never reaches the upstream
                # and the session cannot continue without it. The row
                # records 400 (the HTTP refusal status).
                logger.info("WS %s -> refused (%s)", path, refused)
                status = 400
                await close_on_policy(f"refused by llm-redact ({refused})")
                return
            except BlockedRequest as blocked:
                # Block mode on a realtime stream: the event must never
                # reach the upstream, and the connection cannot continue
                # coherently without it — close both sides (1008 = policy
                # violation; detector type only, never the value).
                logger.info("WS %s -> blocked (%s)", path, blocked)
                await close_on_policy(f"blocked by llm-redact policy ({blocked})")
                return

    async def upstream_to_client() -> None:
        try:
            async for frame in upstream:
                for out in adapter.rehydrate_message(frame, pool):
                    if isinstance(out, str):
                        await websocket.send_text(out)
                    else:
                        await websocket.send_bytes(out)
        except websockets.exceptions.ConnectionClosed:
            # Iteration ends cleanly only for OK closes (1000/1001); any
            # other close code arrives as this exception. Either way the
            # code/reason are on the connection now — mirror them below.
            pass
        code = _sendable_close_code(upstream.close_code or 1000)
        with contextlib.suppress(RuntimeError):
            await websocket.close(code=code, reason=upstream.close_reason or "")

    try:
        tasks = {
            asyncio.create_task(client_to_upstream()),
            asyncio.create_task(upstream_to_client()),
        }
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(
                exc, (WebSocketDisconnect, websockets.exceptions.ConnectionClosed)
            ):
                raise exc
    except Exception as problem:
        status = 500
        logger.warning("WS %s -> relay error (%s)", path, type(problem).__name__)
    finally:
        with contextlib.suppress(Exception):
            await upstream.close()
        with contextlib.suppress(Exception):
            await websocket.close()
        state.rehydration_counts.update(pool.counts)
        state.record_request(
            session=ctx.session_id,
            provider=adapter.provider,
            method="WS",
            path=path,
            status=status,
            started=started,
            streamed=True,
            detections=dict(connection_counts),
            rehydrations=dict(pool.counts),
            audit_token=audit_token,
        )


def _unparsed_frame(data: str | bytes, require_json: bool) -> str | bytes:
    """A frame that is not JSON: forwarded byte-identically, or — under the
    proxy's own identity — refused (the frame KIND only, never its bytes)."""
    if require_json:
        raise UnredactableRequest("realtime frame is not JSON llm-redact can redact")
    return data


def parse_json_text(data: str | bytes) -> tuple[Any, bool] | None:
    """(parsed, was_binary) when ``data`` is a JSON text/binary frame, else
    None — the caller must then forward the frame byte-identically. A
    parsed client frame is ALWAYS re-serialized (``_dump_frame``), never
    forwarded as its original bytes, so a repeated key's earlier
    occurrence (dropped by the parse, never walked) cannot leave."""
    try:
        if isinstance(data, bytes):
            return json.loads(data.decode("utf-8")), True
        return json.loads(data), False
    except (ValueError, UnicodeDecodeError):
        return None


def frame_floors(data: str | bytes) -> dict[str, int]:
    """The token floors of one client frame: its JSON decoded when it
    parses, else its text. A connection accumulates them (its running
    floor): a realtime conversation lives on the provider, so a token any
    earlier frame carried is still in it."""
    if not may_carry_tokens(data):
        return {}
    parsed = parse_json_text(data)
    if parsed is not None:
        return json_floors(parsed[0])
    return token_floors(data if isinstance(data, str) else data.decode("utf-8", "replace"))


def _dump_frame(payload: Any, was_binary: bool) -> str | bytes:
    """Re-serialize a rewritten event in the SAME frame type it arrived in
    (Gemini Live sends JSON in binary frames; OpenAI uses text)."""
    text = json.dumps(payload, ensure_ascii=False)
    return text.encode("utf-8") if was_binary else text
