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
  and are NEVER logged. Log lines carry path and counts only.
- Without the ``websockets`` package, uvicorn itself refuses upgrades
  before this module runs (its auto WS protocol is None); the handler
  also guards the import so other servers and test transports degrade to
  a clean close instead of a traceback.
- One ``record_request`` per connection at close (streamed, duration,
  detection/rehydration counts) — metrics, /recent, /events, audit, and
  otel all inherit from that single call.

Sessions: realtime connections always use the STATIC vault session. The
per-conversation router derives namespaces from a first-user-message
anchor that does not exist at upgrade time; rather than guess (and risk
cross-conversation restores), the fallback session owns WS traffic. The
README documents this.
"""

import asyncio
import contextlib
import json
import logging
import time
import urllib.parse
from collections import Counter
from typing import TYPE_CHECKING, Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from llm_redact.audit import AuditWriteError
from llm_redact.jsonwalk import STRUCTURAL_KEYS, transform_strings
from llm_redact.providers.base import SYSTEM_NOTE, restore_mcp_tools, strip_mcp_tools
from llm_redact.redactor import BlockedRequest, Redactor
from llm_redact.rehydrate import RehydratorPool

if TYPE_CHECKING:
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
        "x-llm-redact-user",  # OUR credential: identity only, never forwarded
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

    def matches(self, path: str) -> bool:
        raise NotImplementedError

    def redact_message(
        self, data: str | bytes, ctx: "RequestContext", *, inject_note: bool = False
    ) -> str | bytes:
        return data

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
        self, data: str | bytes, ctx: "RequestContext", *, inject_note: bool = False
    ) -> str | bytes:
        parsed = parse_json_text(data)
        if parsed is None:
            return data
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


class AzureRealtimeWs(OpenAIRealtimeWs):
    """Azure OpenAI Realtime — the OpenAI Realtime event vocabulary on Azure's
    path (``/openai/realtime``, with api-version and deployment in the query).

    Everything (the outbound walk, inbound channels, *.done flush, note
    injection into session/response instructions) is inherited from
    OpenAIRealtimeWs; only the path and the upstream provider differ, so the
    connection reaches the customer's own resource wss URL derived from
    ``[providers.azure]``. Matcher proven disjoint from OpenAIRealtimeWs
    (/openai/realtime vs /v1/realtime) and Gemini Live by test.
    """

    name = "azure-realtime"
    provider = "azure"

    def matches(self, path: str) -> bool:
        return path == "/openai/realtime" or path.startswith("/openai/realtime/")


# Gemini Live adds mime/voice/config enums; base64 audio rides in `data`
# (already structural) inside realtimeInput mediaChunks.
_GEMINI_LIVE_STRUCTURAL_KEYS = _REALTIME_STRUCTURAL_KEYS | frozenset(
    {"mimeType", "voiceName", "languageCode", "responseModalities", "handle"}
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
        self, data: str | bytes, ctx: "RequestContext", *, inject_note: bool = False
    ) -> str | bytes:
        parsed = parse_json_text(data)
        if parsed is None:
            return data
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
        instruction = setup.get("systemInstruction")
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


ALL_WS_ADAPTERS: tuple[type[WsAdapter], ...] = (
    OpenAIRealtimeWs,
    # Azure Realtime shares the OpenAI vocabulary on /openai/realtime; matcher
    # disjoint from OpenAIRealtimeWs (/v1/realtime) and Gemini Live.
    AzureRealtimeWs,
    GeminiLiveWs,
)


def ws_adapter_for(path: str, adapters: list[WsAdapter]) -> WsAdapter | None:
    for adapter in adapters:
        if adapter.matches(path):
            return adapter
    return None


def _upstream_ws_url(base_url: str, path: str, query_string: bytes) -> str:
    """The provider's wss URL for this connection.

    Scheme-swapped from the configured HTTP upstream (https→wss, http→ws
    for local/test upstreams) with the RAW query preserved — it may carry
    credentials, so it is forwarded exactly and never re-encoded.
    """
    parsed = urllib.parse.urlsplit(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    url = f"{scheme}://{parsed.netloc}{path}"
    if query_string:
        url += "?" + query_string.decode("latin-1")
    return url


def _filtered_headers(websocket: WebSocket) -> list[tuple[str, str]]:
    return [
        (name, value)
        for name, value in websocket.headers.items()
        if name.lower() not in _HOP_HEADERS
    ]


def _sendable_close_code(code: int) -> int:
    # 1005 (no status) and 1006 (abnormal) are reserved: they describe how a
    # connection ended and must not appear in a close frame we send.
    return 1000 if code in (1005, 1006) else code


async def _reject(websocket: WebSocket, reason: str) -> None:
    """Accept-then-close: unlike a handshake 403, the close reason reaches
    the client library where a user can read it."""
    with contextlib.suppress(Exception):
        await websocket.accept()
        await websocket.close(code=1011, reason=reason)


async def ws_handle(websocket: WebSocket) -> None:
    state: ProxyState = websocket.app.state.proxy
    path = websocket.url.path

    if path.startswith("/__llm-redact"):
        await _reject(websocket, "reserved path")
        return
    # Named-user enforcement (2.0 licensing): same rule as HTTP. The header
    # is the WS identity channel (it is in _HOP_HEADERS, so it can never
    # reach the upstream); refusal is accept-then-close so the reason
    # reaches the client library.
    user_name = state.resolve_user(websocket.headers.get("x-llm-redact-user"))
    if state.user_enforcement_required() and user_name is None:
        logger.info("WS %s -> refused (named-user key required)", path)
        await _reject(websocket, "a named-user key is required (x-llm-redact-user header)")
        return
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
    if not websockets_available():
        await _reject(
            websocket,
            "realtime support requires the websockets package;"
            " install it: uv sync --extra realtime",
        )
        return

    import websockets

    url = _upstream_ws_url(
        provider_config.upstream_base_url, path, websocket.scope.get("query_string", b"")
    )
    subprotocols = list(websocket.scope.get("subprotocols") or [])

    from llm_redact.proxy import RequestContext  # runtime: avoids the import cycle

    started = time.perf_counter()
    static_ctx = state.context_for(None, "GET", path, None)
    # Thin per-connection wrapper (the context_for pattern: object
    # construction only): a tee counter gives exact per-connection
    # detection counts that still land in the process totals.
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
            additional_headers=_filtered_headers(websocket),
            subprotocols=[websockets.Subprotocol(s) for s in subprotocols] or None,
            max_size=MAX_FRAME_BYTES,
            open_timeout=30,
        )
    except Exception as problem:  # DNS, TLS, refusals, handshake rejections
        # The exception may embed the URL (query auth!) — log the class only.
        logger.warning("WS %s -> upstream connect failed (%s)", path, type(problem).__name__)
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

    async def client_to_upstream() -> None:
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
            try:
                # [providers.NAME] detection = false applies to realtime
                # frames too: forwarded untouched (rehydration inbound
                # stays active), the same off-switch as the HTTP path.
                outbound = (
                    data
                    if not provider_config.detection
                    else adapter.redact_message(
                        data, ctx, inject_note=state.config.inject_system_note
                    )
                )
                await upstream.send(outbound)
            except BlockedRequest as blocked:
                # Block mode on a realtime stream: the event must never
                # reach the upstream, and the connection cannot continue
                # coherently without it — close both sides (1008 = policy
                # violation; detector type only, never the value).
                logger.info("WS %s -> blocked (%s)", path, blocked)
                await upstream.close(code=1000)
                with contextlib.suppress(RuntimeError):
                    await websocket.close(
                        code=1008, reason=f"blocked by llm-redact policy ({blocked})"
                    )
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


def parse_json_text(data: str | bytes) -> tuple[Any, bool] | None:
    """(parsed, was_binary) when ``data`` is a JSON text/binary frame, else
    None — the caller must then forward the frame byte-identically."""
    try:
        if isinstance(data, bytes):
            return json.loads(data.decode("utf-8")), True
        return json.loads(data), False
    except (ValueError, UnicodeDecodeError):
        return None


def _dump_frame(payload: Any, was_binary: bool) -> str | bytes:
    """Re-serialize a rewritten event in the SAME frame type it arrived in
    (Gemini Live sends JSON in binary frames; OpenAI uses text)."""
    text = json.dumps(payload, ensure_ascii=False)
    return text.encode("utf-8") if was_binary else text
