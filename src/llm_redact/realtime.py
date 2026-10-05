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

Frame checks: a session router with the optional ``realtime_frame_refusal``
(llm-redact-pro: another user's stored object, a cloud storage location) is
asked for every client frame that parses as JSON, before it is redacted or
sent — the realtime twin of the HTTP ``object_access_refusal``. A refusal (or
a failed check) closes the connection 1008, nothing of the frame sent. A
router with the optional ``realtime_server_frame`` is handed every UPSTREAM
frame that parses as JSON (its own parse, before the frame is restored or
sent; read-only, a failure contained): llm-redact-pro records whose Live
session a resumption handle belongs to.

Setup-frame models: a Gemini or Vertex Live connection's model is named by
its setup frame, not the upgrade (``WsAdapter.model_in_frame``). With an
access gate that authorizes requests (its optional ``authorize_request``),
the gate is asked again with that model for the connection's first client
frame — which must be a setup naming one — and for every later setup, FIRST:
before the session router's frame check, the frame's floors, redaction and
send. On OpenAI and Azure realtime connections, whose upgrade names the
model, a ``session.update`` whose ``session`` carries ``model`` is checked
the same way (``WsAdapter.model_in_update``); other frames go as before. A
refusal closes the connection 1008, nothing of the frame sent; a connection
revoked while an awaited answer runs forwards nothing more.

Reloads: a connection is served under the admission it was opened with
(``RealtimeRelay``). A reload that changes it — the provider's settings, the
authorizer that opened its upstream session, the detection policy — revokes
the relay in the same synchronous step that swaps the configuration in: it
closes both sides with 1012 (reconnect) and never forwards a client frame
it reads after the swap under the configuration it was opened with.
"""

import asyncio
import contextlib
import dataclasses
import functools
import inspect
import json
import logging
import threading
import time
import urllib.parse
from collections import Counter
from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from starlette.datastructures import QueryParams
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from llm_redact.audit import AuditWriteError
from llm_redact.authorization import content_facts
from llm_redact.connections import ACCESS_CLOSE_CODE
from llm_redact.detection.base import Detection
from llm_redact.detection.engine import DetectorPlan
from llm_redact.jsonwalk import (
    MAX_JSON_DEPTH,
    STRUCTURAL_KEYS,
    JsonTooDeep,
    json_bytes,
    json_text,
    loads_bounded,
    transform_strings,
)
from llm_redact.metrics import LocalRefusal
from llm_redact.ner_prefetch import collect, prefetch
from llm_redact.placeholders import json_floors, may_carry_tokens, token_floors
from llm_redact.plugin_api import ContentFacts, UpstreamAuthError
from llm_redact.providers.base import (
    SYSTEM_NOTE,
    body_string,
    restore_mcp_tools,
    strip_mcp_tools,
)
from llm_redact.providers.gemini import StreamedText, gemini_model
from llm_redact.providers.vertex import vertex_model
from llm_redact.redactor import (
    BlockedRequest,
    PlaceholderLimitReached,
    Redactor,
    TooManyStrings,
    UnredactableRequest,
)
from llm_redact.rehydrate import RehydratorPool
from llm_redact.vault import run_batched

if TYPE_CHECKING:
    from llm_redact.authorization import OverlayBuild
    from llm_redact.config import ProviderConfig
    from llm_redact.detection.base import Detector
    from llm_redact.detection.engine import Allowlist
    from llm_redact.overrides import OverrideScope
    from llm_redact.plugin_api import AuthorizationRequest, ConnectionRecheck, UpstreamAuth
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


@functools.cache
def _connect_without_redirects() -> Any:
    """``websockets``' asyncio ``connect``, refusing every handshake redirect.

    The stock client follows 3xx upgrade responses (cross-origin included)
    and resends its headers on each hop — under identity auth the proxy's
    OWN cloud credential, to a URL nobody authorized. The relay treats a
    redirect as a failed dial instead (the HTTP side never follows one
    either): the handshake's InvalidStatus is raised as is."""
    from websockets.asyncio.client import connect

    class NoRedirectConnect(connect):
        def process_redirect(self, exc: Exception) -> Exception | str:
            return exc

    return NoRedirectConnect


def websockets_available() -> bool:
    """Import guard; monkeypatched by tests to pin the no-extra behavior."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


# The close reason of a later Live setup naming no model (``_FrameModels``).
SETUP_MODEL = "llm-redact: a setup frame must name its model; the frame was not forwarded"


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

    # True when the upstream takes the connection's model from a client
    # FRAME rather than the upgrade (Gemini/Vertex Live: the setup message):
    # the upgrade's ``AuthorizationRequest`` says ``model_in_frame`` and, with
    # an access gate that authorizes requests, the relay asks again with the
    # model the frame names (``frame_model``) before forwarding anything.
    model_in_frame = False
    # True when a client frame MAY change the model the upgrade named (an
    # OpenAI-vocabulary ``session.update`` carrying ``session.model``): with
    # an access gate that authorizes requests, such a frame is asked about
    # like a Live setup; every other frame goes as before.
    model_in_update = False
    # The close reason of a frame that sets the model to something other
    # than a non-empty string (``sets_model`` true, ``frame_model`` None).
    no_model_reason = SETUP_MODEL

    def request_model(self, path: str, query: QueryParams) -> str | None:
        """The model the upstream runs this connection with, as far as the
        upgrade names it — the access gate's ``AuthorizationRequest.model``;
        never a value the upstream ignores. None — unknown — by default (a
        Gemini Live setup frame names its model after the upgrade:
        ``model_in_frame``)."""
        return None

    def sets_model(self, payload: Any) -> bool:
        """Whether this parsed client frame can choose the connection's model
        (a ``model_in_frame`` adapter's setup message, a ``model_in_update``
        adapter's model update — whatever it holds)."""
        return False

    def frame_model(self, payload: Any) -> str | None:
        """The model this parsed client frame names, read as the HTTP adapter
        reports a model; None when it names none (or not as a non-empty
        string)."""
        return None

    def redact_message(
        self,
        data: str | bytes,
        ctx: "RequestContext",
        *,
        inject_note: bool = False,
        require_json: bool = False,
        parsed: tuple[Any, bool] | None = None,
    ) -> str | bytes:
        """Rewrite one client frame. ``require_json`` (identity auth) turns
        the verbatim forward of a frame the adapter cannot walk into
        ``UnredactableRequest``: the proxy's own identity never carries an
        unscanned frame. ``parsed``: the frame's ``parse_client_frame``
        result when the relay already has it (the session router's frame
        check read it), so it is walked without a second parse."""
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


def _session_model(endpoint: bool, query: QueryParams, name: str) -> str | None:
    """An OpenAI-vocabulary realtime session's model: the ``name`` query
    parameter's one value on the realtime ``endpoint`` itself — None when it
    is absent or repeated (an upstream may read either occurrence), on any
    other path, and when the query sets the session up elsewhere (an
    ``intent`` — its frames choose the model — or a SIP ``call_id``, whose
    model was set when the call was accepted)."""
    if not endpoint or "intent" in query or "call_id" in query:
        return None
    values = query.getlist(name)
    return values[0] if len(values) == 1 and values[0] else None


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

    # A `session.update` naming `session.model` is put to the access gate
    # like a Live setup (conservative: whatever the API makes of it).
    model_in_update = True
    no_model_reason = (
        "llm-redact: session.update must name session.model as a non-empty string;"
        " the frame was not forwarded"
    )

    def request_model(self, path: str, query: QueryParams) -> str | None:
        # The `model` query parameter of the realtime endpoint — unless the
        # session's model is set elsewhere: by its frames (a transcription
        # `intent`) or when the call was accepted (a SIP `call_id`).
        return _session_model(path == "/v1/realtime", query, "model")

    def sets_model(self, payload: Any) -> bool:
        if not isinstance(payload, dict) or payload.get("type") != "session.update":
            return False
        session = payload.get("session")
        return isinstance(session, dict) and "model" in session

    def frame_model(self, payload: Any) -> str | None:
        if not self.sets_model(payload):
            return None
        return body_string(payload["session"], "model")

    def redact_message(
        self,
        data: str | bytes,
        ctx: "RequestContext",
        *,
        inject_note: bool = False,
        require_json: bool = False,
        parsed: tuple[Any, bool] | None = None,
    ) -> str | bytes:
        if parsed is None:
            parsed = parse_client_frame(data)
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

    def request_model(self, path: str, query: QueryParams) -> str | None:
        # The deployment the query names: `model` on the GA path,
        # `deployment` on the preview one (its `model` is not what runs).
        if path == "/openai/realtime":
            return _session_model(True, query, "deployment")
        return _session_model(path == "/openai/v1/realtime", query, "model")


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
    message. Everything else in a model message is restored whole by a
    walk from the message's root (toolCall functionCalls[].args at its
    opaque position, code and grounding parts). Bookkeeping, unknown and
    unparseable frames forward byte-identically.
    """

    name = "gemini-live"
    provider = "gemini"
    # The setup message (the connection's first) names the model.
    model_in_frame = True

    def matches(self, path: str) -> bool:
        return path.startswith("/ws/google.ai.generativelanguage.") and path.endswith(
            ".GenerativeService.BidiGenerateContent"
        )

    def sets_model(self, payload: Any) -> bool:
        # `setup` is its own JSON name in both the camelCase and the proto
        # field-name spelling.
        return isinstance(payload, dict) and "setup" in payload

    def frame_model(self, payload: Any) -> str | None:
        setup = payload.get("setup") if isinstance(payload, dict) else None
        model = body_string(setup, "model")
        return self._model_name(model) if model is not None else None

    @staticmethod
    def _model_name(name: str) -> str:
        # The Gemini API's reading (`models/{m}` → `m`), as on HTTP.
        return gemini_model(name)

    def redact_message(
        self,
        data: str | bytes,
        ctx: "RequestContext",
        *,
        inject_note: bool = False,
        require_json: bool = False,
        parsed: tuple[Any, bool] | None = None,
    ) -> str | bytes:
        if parsed is None:
            parsed = parse_client_frame(data)
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

    @staticmethod
    def _hold_streamed(payload: dict[str, Any]) -> dict[str, Any]:
        """``payload`` with the values that stream through channels — the
        modelTurn text parts and outputTranscription.text — held as
        ``StreamedText``, copied along those paths only."""
        server_content = dict(payload["serverContent"])
        model_turn = server_content.get("modelTurn")
        if isinstance(model_turn, dict) and isinstance(model_turn.get("parts"), list):
            server_content["modelTurn"] = {
                **model_turn,
                "parts": [
                    {**part, "text": StreamedText(part["text"])}
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                    else part
                    for part in model_turn["parts"]
                ],
            }
        transcription = server_content.get("outputTranscription")
        if isinstance(transcription, dict) and isinstance(transcription.get("text"), str):
            server_content["outputTranscription"] = {
                **transcription,
                "text": StreamedText(transcription["text"]),
            }
        return {**payload, "serverContent": server_content}

    def _rehydrate_parts(self, model_turn: dict[str, Any], pool: RehydratorPool) -> None:
        parts = model_turn.get("parts")
        if not isinstance(parts, list):
            return
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), StreamedText):
                channel = pool.get(("modelTurn", self._part_kind(part)))
                part["text"] = channel.feed(part["text"].text)

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

        # A model message is walked from its ROOT, like the HTTP adapter's
        # chunks, so each value keeps its position's context: a toolCall's
        # functionCalls[].args is an opaque position (a tool parameter named
        # `id` or `name` is restored), and code/grounding in serverContent
        # are restored whole. Only the streamed values are held out of the
        # walk and fed to their channels.
        if not isinstance(payload.get("serverContent"), dict):
            if isinstance(payload.get("toolCall"), dict):
                return [_dump_frame(transform_strings(payload, pool.rehydrate_whole), was_binary)]
            # setupComplete / usageMetadata / goAway / sessionResumptionUpdate
            # and unknown shapes: pass through.
            return [data]

        walked = transform_strings(self._hold_streamed(payload), pool.rehydrate_whole)
        server_content = walked["serverContent"]
        model_turn = server_content.get("modelTurn")
        if isinstance(model_turn, dict):
            self._rehydrate_parts(model_turn, pool)
        transcription = server_content.get("outputTranscription")
        if isinstance(transcription, dict) and isinstance(transcription.get("text"), StreamedText):
            channel = pool.get(("outputTranscription",))
            transcription["text"] = channel.feed(transcription["text"].text)
        frames: list[str | bytes] = []
        if server_content.get("turnComplete") or server_content.get("generationComplete"):
            frames = [_dump_frame(extra, was_binary) for extra in self._flush_into(walked, pool)]
        frames.append(_dump_frame(walked, was_binary))
        return frames


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

    @staticmethod
    def _model_name(name: str) -> str:
        # Vertex's reading, as on HTTP: a publisher model's id
        # (`projects/{p}/locations/{l}/publishers/google/models/{m}` → `m`),
        # an endpoint as `endpoints/{e}`.
        return vertex_model(name)


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
    # whether or not an access gate consumed it (same rule as HTTP), and
    # neither does a method override (every relay is a matched route, read
    # as the GET upgrade it is; proxy.METHOD_OVERRIDE_HEADERS).
    from llm_redact.proxy import METHOD_OVERRIDE_HEADERS

    return [
        (name, value)
        for name, value in websocket.headers.items()
        if name.lower() not in _HOP_HEADERS
        and not name.lower().startswith("x-llm-redact-")
        and name.lower() not in METHOD_OVERRIDE_HEADERS
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


async def _reject(websocket: WebSocket, reason: str, *, code: int = 1011) -> None:
    """Accept-then-close: unlike a handshake 403, the close reason reaches
    the client library where a user can read it. 1011 for a connection the
    proxy cannot serve; 1008 (policy violation) for one it refuses to."""
    with contextlib.suppress(Exception):
        await websocket.accept()
        await websocket.close(code=code, reason=_close_reason(reason))


def _record_ws_refusal(
    state: "ProxyState",
    adapter: WsAdapter | None,
    path: str,
    status: int,
    started: float,
    *,
    kind: LocalRefusal,
    session: str | None = None,
) -> None:
    """The recorded row for a connection refused before any upstream
    contact by a proxy rule — the access gate (403), an unusable provider
    (502), an audit START that could not commit (503), a sealed session or
    a malformed target — as HTTP records its refusals."""
    state.record_request(
        session=session if session is not None else state.config.vault.session,
        provider=adapter.provider if adapter is not None else None,
        method="WS",
        path=path,
        status=status,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal=kind,
    )


def _record_refused(
    state: "ProxyState",
    ctx: "RequestContext",
    adapter: WsAdapter,
    path: str,
    started: float,
    *,
    kind: LocalRefusal,
    audit_token: object | None = None,
    status: int = 502,
) -> None:
    """The recorded row for a connection whose client the upstream never
    served — no credential, the dial failed (502), or it was revoked while
    dialled (``status``): one metadata-only row, like HTTP, ending the audit
    START row if there is one."""
    state.record_request(
        session=ctx.session_id,
        provider=adapter.provider,
        method="WS",
        path=path,
        status=status,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        audit_token=audit_token,
        refusal=kind,
    )


def _revoked_kind(relay: "RealtimeRelay") -> LocalRefusal:
    """A connection revoked before it was relayed: by the access gate, or
    by a reload that changed its admission."""
    return "access_gate" if relay.revoked == ACCESS_REVOKED else "reload"


# The close code a reload closes a relay with: 1012 Service Restart (the IANA
# WebSocket close code registry) — the service restarted and the client may
# reconnect, which is exactly how the new configuration reaches it. uvicorn
# closes its connections with the same code when it shuts down. Not 1008:
# the client violated nothing; not 1011: nothing failed; not 1001: the proxy
# is not going away.
RELOAD_CLOSE_CODE = 1012


def _reload_reason(changed: str) -> str:
    """The client-facing close reason: what changed, never a value."""
    return f"llm-redact config reload changed this connection's {changed}; reconnect"


# ``RealtimeRelay.revoked`` of a relay closed because its admission ended (the
# access gate revoked its user or credential, or its re-check refused it) —
# not a configuration phrase, so never mistaken for what a reload changed.
ACCESS_REVOKED = "access"


class RealtimeRelay:
    """One open realtime connection's admission, as a reload sees it.

    A relay serves its whole connection under what it was admitted with: its
    provider's settings (``[providers.NAME]`` upstream, enabled, detection,
    auth, region), the upstream authorizer that opens its upstream session
    under ``auth = "identity"``, and the detectors, allowlist and modes its
    client frames are redacted with — read in the one synchronous stretch
    after admission, so they are one consistent snapshot. ProxyState holds
    every open relay (``realtime_relays``), and ``apply_config`` REVOKES
    each relay whose admission its swap changed (``stale``) in the same
    synchronous step as the swap.

    The access gate's admission rides along (``subject``, ``grant``,
    ``recheck``; ``plugin_api.Admission``): ProxyState's ``connections``
    (``connections.LiveConnections``) closes the relay through
    ``close_for_access`` when the gate revokes it or its re-check refuses
    it — the same revocation, with the client closed 1008 instead of 1012.

    A revoked relay never dials an upstream it has not dialled yet, never
    redacts or forwards a client frame it reads after the swap, and closes
    both sides (the client with 1012: reconnect). The relay checks
    ``revoked`` between reading a frame and redacting it with no await in
    between, and the swap runs on the same event loop (SIGHUP's handler,
    the config editor's request), so a frame read after the swap is always
    seen as revoked: the only frame that can still leave under the old
    admission is one the relay had already read, redacted and handed to
    the upstream socket when the swap ran. An access revocation may come
    from another thread: a frame the relay read just before it is still
    sent."""

    kind = "realtime"
    __slots__ = (
        "_lock",
        "_loop",
        "_revoked_event",
        "allowlist",
        "close_code",
        "close_reason",
        "detectors",
        "grant",
        "modes",
        "provider",
        "provider_config",
        "recheck",
        "revoked",
        "subject",
        "upstream_auth",
    )

    def __init__(
        self,
        provider: str,
        provider_config: "ProviderConfig",
        upstream_auth: "UpstreamAuth | None",
        detectors: "Sequence[Detector]",
        allowlist: "Allowlist",
        modes: Mapping[str, str],
        *,
        subject: str | None = None,
        grant: str | None = None,
        recheck: "ConnectionRecheck | None" = None,
    ) -> None:
        self.provider = provider
        self.provider_config = provider_config
        self.upstream_auth = upstream_auth
        self.detectors = detectors
        self.allowlist = allowlist
        self.modes = modes
        self.subject = subject
        self.grant = grant
        self.recheck = recheck
        # Once revoked: what the reload changed (a fixed phrase, value-free)
        # or ACCESS_REVOKED; the client's close code and reason are set
        # before it.
        self.revoked: str | None = None
        self.close_code = RELOAD_CLOSE_CODE
        self.close_reason = ""
        self._lock = threading.Lock()
        self._loop = asyncio.get_running_loop()
        self._revoked_event = asyncio.Event()

    def stale(self, state: "ProxyState") -> str | None:
        """What of this relay's admission ``state`` no longer grants, or
        None. Any field of its provider's settings; its authorizer
        (``apply_config`` rebuilds every authorizer when any provider's
        identity settings change, and closes the ones it displaces); the
        detection objects, which ``apply_config`` rebuilds exactly when
        ``[detection]`` changed. The rest of what a relay uses is read per
        frame, live (note injection, ``max_body_strings``), or cannot change
        without a restart (the vault session, the access gate)."""
        if state.config.providers.get(self.provider) != self.provider_config:
            return f"[providers.{self.provider}] settings"
        if state.upstream_auth.get(self.provider) is not self.upstream_auth:
            return "upstream authorizer"
        if (
            state.detectors is not self.detectors
            or state.allowlist is not self.allowlist
            or state.modes is not self.modes
        ):
            return "[detection] policy"
        return None

    @property
    def closing(self) -> bool:
        """Revoked (a reload, or its access): closing, whenever the relay's
        handler ends (``connections.TrackedConnection``)."""
        return self.revoked is not None

    def revoke(self, changed: str) -> None:
        """Mark the relay revoked — at once, for its per-frame check — and
        wake it to close both sides. Never raises: a reload revokes every
        relay it changed."""
        self._revoke(changed, RELOAD_CLOSE_CODE, _reload_reason(changed))

    def close_for_access(self, reason: str) -> bool:
        """The access gate ended this connection's admission (``connections.
        TrackedConnection``): revoke it, the client to be closed 1008 with
        ``reason``. Thread-safe; True when this call revoked it."""
        return self._revoke(ACCESS_REVOKED, ACCESS_CLOSE_CODE, reason)

    def _revoke(self, changed: str, code: int, reason: str) -> bool:
        with self._lock:  # first wins, whichever thread revokes
            if self.revoked is not None:
                return False
            self.close_code = code
            self.close_reason = reason
            self.revoked = changed
        # Thread-safe, and fine from the loop's own thread (where reloads
        # run). A closed loop has no connection left to close.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._revoked_event.set)
        return True

    def revocation_log(self) -> str:
        """Why the relay was revoked, for the proxy's log: a fixed phrase
        (never the gate's reason, which only the client is told)."""
        if self.revoked == ACCESS_REVOKED:
            return "its access was revoked"
        return f"a config reload changed its {self.revoked}"

    async def wait_revoked(self) -> None:
        await self._revoked_event.wait()


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
        _record_ws_refusal(state, adapter, path, 400, started, kind="request_target")
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
        _record_refused(state, ctx, adapter, path, started, kind="upstream_auth")
        await _reject(
            websocket,
            f"the proxy could not obtain its own {provider} cloud credentials ({source});"
            " nothing was forwarded",
        )
        return None
    return http_url, headers, subprotocols


def _upgrade_request(
    adapter: WsAdapter, path: str, query: QueryParams, identity: bool
) -> "AuthorizationRequest":
    """The facts of a realtime upgrade, as the access gate's
    ``authorize_request`` is asked about it: the model the upgrade names, or
    — for an adapter whose setup frame names it — None and
    ``model_in_frame`` (the relay asks again with the frame's model)."""
    from llm_redact.plugin_api import AuthorizationRequest

    return AuthorizationRequest(
        surface="websocket",
        provider=adapter.provider,
        adapter=adapter.name,
        kind="chat",
        method="GET",
        path=path,
        model=None if adapter.model_in_frame else adapter.request_model(path, query),
        identity=identity,
        model_in_frame=adapter.model_in_frame,
    )


async def _authorize_connection(
    state: "ProxyState",
    request: "AuthorizationRequest | None",
    relay: RealtimeRelay,
    path: str,
) -> tuple["OverlayBuild | None", tuple[int, LocalRefusal, str, int] | None]:
    """The access gate's authorization of this upgrade (``request``: its
    facts, None without the gate's optional ``authorize_request``) and the
    requester's detection overlay (its optional ``detection_overlay``),
    before the session is opened, the audit START row and any dial. Returns
    the overlay build and None, or None and the refusal: (row status, kind,
    close reason, close code). ``relay`` is already held (revocable) while
    the check runs: one a reload or the access gate revoked meanwhile is
    refused like a revocation before the dial — 1012 and row 503 for a
    reload, 1008 with the gate's reason and row 403 for its access."""
    authorization = state.authorization
    if request is not None:
        verdict = authorization.refusal(request, f"WS {path}")
        if inspect.isawaitable(verdict):
            verdict = await verdict
        if verdict is not None:
            logger.info("WS %s -> refused by the access gate (authorization)", path)
            return None, (403, "authorization", verdict, ACCESS_CLOSE_CODE)
        if relay.revoked is not None:
            logger.info("WS %s -> refused (%s)", path, relay.revocation_log())
            refused = 403 if relay.revoked == ACCESS_REVOKED else 503
            return None, (refused, _revoked_kind(relay), relay.close_reason, relay.close_code)
    if not authorization.overlays:
        return None, None
    overlay, refusal = authorization.overlay(state.overlay_builds, f"WS {path}")
    if refusal is not None:
        logger.info("WS %s -> refused by the access gate (detection overlay)", path)
        return None, (403, "authorization", refusal, ACCESS_CLOSE_CODE)
    return overlay, None


async def ws_handle(websocket: WebSocket) -> None:
    state: ProxyState = websocket.app.state.proxy
    from llm_redact.proxy import has_dot_segment, origin_form_target

    if not origin_form_target(websocket.scope):
        state.count_local_refusal("request_target", None)
        await _reject(websocket, "the request target must be a path")
        return
    if has_dot_segment(websocket.scope):
        # The HTTP rule: never forwarded, recorded, or logged with its path.
        state.count_local_refusal("request_target", None)
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
        state.count_local_refusal("identity_path", None)
        await _reject(websocket, admission.refusal or "no realtime route for this path")
        return
    if path.startswith("/__llm-redact"):
        # Reached through a stripped prefix: never served here, never
        # recorded (the HTTP rule — before the gate's refusal is applied).
        state.count_local_refusal("identity_path", None)
        await _reject(websocket, "reserved path")
        return
    # Attribute the connection's record_request row — a refusal's below,
    # or the connection's at close (same task, so the proxy's contextvar
    # carries it). From here on every refusal is recorded like HTTP's.
    from llm_redact.proxy import _REQUEST_USER

    _REQUEST_USER.set(admission.subject)
    started = time.perf_counter()
    adapter = ws_adapter_for(path, state.ws_adapters)
    provider_config = state.config.providers.get(adapter.provider) if adapter is not None else None
    # Cross-site WebSocket hijacking (the HTTP rule, proxy.request_origin_
    # refusal): browsers apply no CORS to a handshake and always send Origin,
    # so a web page's connection is refused before anything is dialled —
    # first, so a page learns nothing about the routes behind it. A
    # connection that would spend the proxy's own identity also needs a host
    # name the proxy answers to.
    from llm_redact.proxy import REQUEST_ORIGIN_REFUSALS, request_origin_refusal

    origin_refusal = request_origin_refusal(
        websocket,
        state,
        lends_credential=provider_config is not None and provider_config.auth != "passthrough",
    )
    if origin_refusal is not None:
        state.request_origin_refusals[origin_refusal] += 1
        logger.info("WS %s -> refused (request origin: %s)", path, origin_refusal)
        _record_ws_refusal(state, adapter, path, 403, started, kind="request_origin")
        await _reject(websocket, REQUEST_ORIGIN_REFUSALS[origin_refusal], code=1008)
        return
    if admission.refusal is not None:
        logger.info("WS %s -> refused by the access gate", path)
        _record_ws_refusal(state, adapter, path, 403, started, kind="access_gate")
        await _reject(websocket, admission.refusal)
        return
    if adapter is None:
        # Unlike unmatched HTTP traffic there is no default upstream to
        # forward an unknown WS path to; refusing is the only safe answer.
        state.count_local_refusal("unattributed", None)
        await _reject(websocket, "no realtime route for this path")
        return
    if provider_config is None or not provider_config.upstream_base_url:
        _record_ws_refusal(state, adapter, path, 502, started, kind="no_upstream")
        await _reject(websocket, f"[providers.{adapter.provider}] upstream not configured")
        return
    if not provider_config.enabled:
        # Same fail-closed stance as HTTP: a disabled provider must never
        # fall through to any forwarding path.
        logger.info("WS %s -> refused (provider %s disabled)", path, adapter.provider)
        _record_ws_refusal(state, adapter, path, 502, started, kind="disabled_provider")
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
        _record_ws_refusal(state, adapter, path, 403, started, kind="identity_route")
        await _reject(
            websocket,
            f'[providers.{adapter.provider}] auth = "identity": only the realtime routes'
            " llm-redact recognizes are authorized",
        )
        return
    if not websockets_available():
        state.count_local_refusal("realtime_unavailable", adapter.provider)
        await _reject(
            websocket,
            "realtime support requires the websockets package;"
            " install it: uv sync --extra realtime",
        )
        return

    # The connection's admission: its provider's settings and authorizer
    # (read above) and the detection policy, all read since admission with
    # no await in between, and the access gate's verdict. Held by ProxyState
    # from here on — the awaited authorization check included — so a reload
    # that changes it, or the gate ending it, revokes the relay at any point
    # (RealtimeRelay); every way out releases it.
    relay = RealtimeRelay(
        adapter.provider,
        provider_config,
        upstream_auth,
        state.detectors,
        state.allowlist,
        state.modes,
        subject=admission.subject,
        grant=getattr(admission, "grant", None),
        recheck=getattr(admission, "recheck", None),
    )
    state.realtime_relays.add(relay)
    state.connections.track(relay)
    try:
        await _admitted(state, websocket, adapter, relay, path, started)
    finally:
        state.connections.untrack(relay)
        state.realtime_relays.discard(relay)


async def _admitted(
    state: "ProxyState",
    websocket: WebSocket,
    adapter: WsAdapter,
    relay: RealtimeRelay,
    path: str,
    started: float,
) -> None:
    """One admitted connection, its relay held: the access gate's
    authorization, the connection's session, then the relay itself."""
    authorization = state.authorization
    # The upgrade's facts, built when the gate asks about requests or about
    # their content (authorize_request / authorize_content).
    upgrade = (
        _upgrade_request(adapter, path, websocket.query_params, relay.upstream_auth is not None)
        if authorization.authorizes or authorization.checks_content
        else None
    )
    request = upgrade if authorization.authorizes else None
    overlay, refusal = await _authorize_connection(state, request, relay, path)
    if refusal is not None:
        _record_ws_refusal(state, adapter, path, refusal[0], started, kind=refusal[1])
        await _reject(websocket, refusal[2], code=refusal[3])
        return

    try:
        static_ctx = state.context_for(None, "GET", path, None)
    except state.vault_faults as fault:
        # Opening the connection's session reads the vault (a new session's
        # view loads its rows): a fault refuses the connection before any
        # dial, recorded as the HTTP path's 503, counted as the "vault"
        # bookkeeping stage, logged by exception TYPE only.
        state.bookkeeping_errors["vault"] += 1
        logger.error(
            "WS %s -> closed 1011 (vault read failed opening the session: %s)",
            path,
            type(fault).__name__,
        )
        _record_ws_refusal(state, adapter, path, 503, started, kind="vault_fault")
        await _reject(websocket, "llm-redact could not open this connection's vault session")
        return
    if static_ctx.sealed:
        # A session the router says must stay empty cannot carry a
        # conversation whose every message is redacted into it.
        logger.info("WS %s -> refused (sealed session)", path)
        _record_ws_refusal(state, adapter, path, 403, started, kind="sealed_session")
        await _reject(websocket, "the session router sealed this connection's vault session")
        return
    # A connection whose model a setup frame names, or a frame may change
    # (session.update): the gate is asked again with that model before the
    # frame is forwarded.
    frame_request = (
        request
        if request is not None and (adapter.model_in_frame or adapter.model_in_update)
        else None
    )
    await _relay(
        state,
        websocket,
        adapter,
        relay,
        static_ctx,
        path,
        started,
        overlay,
        frame_request,
        upgrade if authorization.checks_content else None,
    )


async def _relay(
    state: "ProxyState",
    websocket: WebSocket,
    adapter: WsAdapter,
    relay: RealtimeRelay,
    static_ctx: "RequestContext",
    path: str,
    started: float,
    overlay: "OverlayBuild | None" = None,
    frame_request: "AuthorizationRequest | None" = None,
    content_request: "AuthorizationRequest | None" = None,
) -> None:
    """Dial, relay and record one admitted connection, under ``relay``'s
    admission until a reload revokes it — with the requester's detection
    overlay (built against the relay's own detection objects), if any;
    when ``frame_request`` is given (the upgrade's facts), the access gate
    asked again with the model each setup frame or model update names
    (``_FrameModels``); and when ``content_request`` is given (the upgrade's
    facts again), the gate's ``authorize_content`` asked about each client
    frame's redaction before it is sent — with the latest model check's
    request when there is one."""
    import websockets

    from llm_redact.proxy import RequestContext  # runtime: avoids the import cycle

    provider_config = relay.provider_config
    upstream_auth = relay.upstream_auth
    # Thin per-connection wrapper (the context_for pattern: object
    # construction only): a tee counter gives exact per-connection
    # detection counts that still land in the process totals. Its redactor
    # is rebound as the connection's token floor rises (frame_floors).
    connection_counts = _TeeCounter(state.detection_counts)
    ctx = RequestContext(
        static_ctx.session_id,
        static_ctx.vault,
        Redactor(
            relay.detectors,
            static_ctx.vault,
            relay.allowlist,
            counts=connection_counts,
            modes=overlay.modes if overlay is not None else relay.modes,
            warn_counts=state.warn_counts,
            added_deny=overlay.deny if overlay is not None else None,
            final_blocks=overlay.final_blocks if overlay is not None else frozenset(),
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
        _record_ws_refusal(state, adapter, path, 400, started, kind="request_target")
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
        if relay.revoked is not None:
            # A reload changed the connection's admission (or the access gate
            # revoked it) while the proxy was obtaining its credential: the
            # upstream is never dialled under it (the only await between
            # admission and the dial). Recorded as the reload's 503 or the
            # gate's 403.
            logger.info("WS %s -> refused (%s)", path, relay.revocation_log())
            refused = 403 if relay.revoked == ACCESS_REVOKED else 503
            _record_ws_refusal(state, adapter, path, refused, started, kind=_revoked_kind(relay))
            await _reject(websocket, relay.close_reason, code=relay.close_code)
            return
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
        # The HTTP 503 twin: no upstream contact, recorded to metrics and
        # /recent (its own audit write fails too — logged loudly there).
        logger.critical(
            "WS %s -> refused; audit write failed with [audit] required (%s)",
            path,
            type(problem).__name__,
        )
        _record_ws_refusal(
            state, adapter, path, 503, started, kind="audit_unavailable", session=ctx.session_id
        )
        await _reject(websocket, "audit log unavailable and [audit] required is enabled")
        return

    try:
        upstream = await _connect_without_redirects()(
            url,
            additional_headers=headers,
            subprotocols=[websockets.Subprotocol(s) for s in subprotocols] or None,
            max_size=MAX_FRAME_BYTES,
            open_timeout=30,
        )
    except Exception as problem:  # DNS, TLS, refusals, handshake rejections, redirects
        # The exception may embed the URL (query auth!) — log the class only.
        # Counted and recorded like an HTTP upstream fault (the audit START
        # row, if any, gets its END row here).
        logger.warning("WS %s -> upstream connect failed (%s)", path, type(problem).__name__)
        state.upstream_errors[adapter.provider] += 1
        _record_refused(
            state, ctx, adapter, path, started, kind="upstream_fault", audit_token=audit_token
        )
        await _reject(websocket, "upstream websocket connect failed")
        return

    if relay.revoked is not None:
        # Revoked while the upstream was dialled (a reload, or the access
        # gate): the dial completes and closes; the client is never accepted
        # onto it, and no upstream frame (its greeting) reaches it.
        logger.info("WS %s -> refused after dialling (%s)", path, relay.revocation_log())
        with contextlib.suppress(Exception):
            await upstream.close(code=1000)
        refused = 403 if relay.revoked == ACCESS_REVOKED else 503
        _record_refused(
            state,
            ctx,
            adapter,
            path,
            started,
            kind=_revoked_kind(relay),
            audit_token=audit_token,
            status=refused,
        )
        await _reject(websocket, relay.close_reason, code=relay.close_code)
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
    # The local-refusal kind of a frame the proxy refused (the connection
    # then closes), counted once in the connection's row.
    refusal: LocalRefusal | None = None
    # Under the proxy's own identity a frame the adapter cannot walk is
    # refused, never relayed verbatim (the HTTP body rule). detection = false
    # relays frames untouched — unless the session router checks frames
    # (below): then what its check read is what is sent.
    require_json = upstream_auth is not None
    # The session router's per-frame check (the optional
    # realtime_frame_refusal, llm-redact-pro's stored-object and storage
    # policy): asked for every client frame, read once per connection.
    check_frames = state.checks_realtime_frames
    # The session router's observation of upstream frames (the optional
    # realtime_server_frame, llm-redact-pro's Live resumption-handle
    # records): read once per connection; without it no upstream frame is
    # parsed for the router.
    observe_frames = state.observes_realtime_server_frames
    # NER off the event loop (ner_prefetch): with a model-backed detector
    # in the connection's plan, each client frame's strings are detected
    # by the models on the worker thread before the frame is checked and
    # redacted.
    prefetching = provider_config.detection and bool(ctx.redactor.plan.heavy_indices)

    # The access gate asked again with the model each setup frame or model
    # update names (its optional authorize_request; Gemini/Vertex Live
    # setups, OpenAI/Azure session.update): only with that member, so without
    # it no frame is parsed for it.
    frame_models = (
        _FrameModels(state, adapter, frame_request, f"WS {path}")
        if frame_request is not None
        else None
    )

    # "once" / "always" once a frame passed a refusal on its requester's
    # approved override (overrides.py): the connection's row says so.
    override_marker: str | None = None

    async def close_on_policy(reason: str, code: int = 1008) -> None:
        # The client FIRST: closing the upstream first lets upstream_to_client
        # mirror the upstream's 1000 to the client ahead of the 1008.
        with contextlib.suppress(RuntimeError):
            await websocket.close(code=code, reason=_close_reason(reason))
        await upstream.close(code=1000)

    async def close_on_revoke() -> None:
        # Only a connection still open on both sides: one that a policy
        # close, the client or the upstream ended first keeps that close.
        if (
            websocket.application_state is not WebSocketState.CONNECTED
            or websocket.client_state is not WebSocketState.CONNECTED
        ):
            return
        logger.info("WS %s -> closed %d (%s)", path, relay.close_code, relay.revocation_log())
        with contextlib.suppress(Exception):  # the client may vanish meanwhile
            await close_on_policy(relay.close_reason, code=relay.close_code)

    async def client_to_upstream() -> None:
        nonlocal status, override_marker, refusal
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                code = _sendable_close_code(int(message.get("code") or 1000))
                await upstream.close(code=code)
                return
            if relay.revoked is not None:
                # A reload changed this connection's admission (or the access
                # gate ended it): the frame is neither redacted under the
                # policy it was opened with nor sent on its upstream session.
                # The relay closes (1012, or 1008).
                return
            data: str | bytes
            if message.get("text") is not None:
                data = message["text"]
            else:
                data = message.get("bytes") or b""
            try:
                frame_table: tuple[DetectorPlan, dict[str, dict[int, list[Detection]]]] | None
                frame_table = None
                if prefetching:
                    # Before every check, so that nothing below waits: a
                    # revocation landing meanwhile is seen right after.
                    frame_table = await _prefetch_frame(
                        adapter,
                        data,
                        ctx,
                        require_json=require_json,
                        limit=state.config.max_body_strings,
                    )
                    if relay.revoked is not None:
                        # The frame is neither redacted nor sent under the
                        # admission it was read with (1012 / 1008).
                        return
                parsed: tuple[Any, bool] | None = None
                # Whether the gate checked a model this frame names: then it
                # is sent as the check read it, even with detection = false.
                model_checked = False
                if frame_models is not None:
                    # The access gate's check of the model a frame names,
                    # FIRST: on Live the first frame must be a setup naming
                    # one, a later setup is checked alike, and a frame that
                    # is not JSON (its model unknowable) is refused; on
                    # OpenAI/Azure a session.update carrying session.model is
                    # checked. Its parse is handed on, never parsed twice.
                    parsed = parse_client_frame(data)
                    verdict = frame_models.verdict(parsed)
                    model_checked = frame_models.checked
                    if inspect.isawaitable(verdict):
                        verdict = await verdict
                        if relay.revoked is not None:
                            # Revoked while the gate decided: the frame is
                            # neither checked further, redacted nor sent
                            # under the admission it was read with.
                            return
                    if verdict is not None:
                        raise _ModelRefused(verdict)
                # The session router's frame check, then synchronously (no
                # await since the revoked check above, or since the one
                # after the gate's awaited answer): the frame as parsed —
                # handed on to redaction, never parsed twice.
                if check_frames:
                    parsed = _checked_frame(
                        state,
                        adapter,
                        path,
                        data,
                        identity=require_json,
                        session_id=ctx.session_id,
                        parsed=parsed,
                    )
                # The connection's running token floor, raised BEFORE this
                # frame's values are numbered: the provider holds the whole
                # conversation, so a token any earlier client frame carried
                # is still in it (the client never resends history, as on
                # HTTP). Upstream frames are not read: model output is
                # provider-side history, as on HTTP (and a session echo
                # carries the proxy's own note, whose «EMAIL_000» example is
                # never issued anyway).
                ctx.redactor = ctx.redactor.with_floors(frame_floors(data))
                # Each client frame is a body of its own: redacted through a
                # copy that counts its strings against max_body_strings (the
                # frame cap, MAX_FRAME_BYTES, bounds bytes only).
                # The requester's approved overrides, asked only where a
                # block-mode value would close the connection (overrides.py);
                # never on a connection authorized with the proxy's own
                # identity (every refusal under it is final, no code).
                frame_scope = None if require_json else state.override_scope()
                frame_redactor = ctx.redactor.with_budget(state.config.max_body_strings)
                if frame_table is not None:
                    frame_redactor = frame_redactor.with_precomputed(frame_table[1], frame_table[0])
                if frame_scope is not None:
                    frame_redactor = frame_redactor.with_overrides(frame_scope)
                frame_ctx = RequestContext(
                    ctx.session_id, ctx.vault, frame_redactor, ctx.rehydrator
                )
                # The access gate's authorize_content (below): whether this
                # frame's content is redacted (a JSON frame, detection on —
                # parsed here once, handed on, when nothing above read it)
                # and the counts as its redaction starts: the frame's own
                # share, taken with no await before the redaction.
                frame_scanned = False
                if content_request is not None:
                    if (
                        provider_config.detection
                        and parsed is None
                        and frame_models is None
                        and not check_frames
                    ):
                        parsed = parse_client_frame(data)
                    frame_scanned = provider_config.detection and parsed is not None
                    detected_before = dict(connection_counts)
                    warned_before = dict(state.warn_counts)
                frame_marker: str | None = None
                outbound: str | bytes
                if not provider_config.detection:
                    # [providers.NAME] detection = false applies to realtime
                    # frames too: forwarded unredacted (rehydration inbound
                    # stays active), the same off-switch as the HTTP path —
                    # as the frame check read it when it read it (a repeated
                    # key's earlier occurrence never leaves), else untouched.
                    outbound = (
                        data
                        if parsed is None or not (check_frames or model_checked)
                        else _dump_frame(*parsed)
                    )
                else:
                    # One vault transaction per frame, committed before the
                    # frame is sent (run_batched).
                    outbound = run_batched(
                        ctx.vault,
                        functools.partial(
                            adapter.redact_message,
                            data,
                            frame_ctx,
                            inject_note=state.config.inject_system_note,
                            require_json=require_json,
                            parsed=parsed,
                        ),
                    )
                    if frame_scope is not None:
                        # Consumed as the frame passes (no await since the
                        # redaction): a one-time grant another request used
                        # first refuses the frame.
                        ok, frame_marker = frame_scope.commit()
                        if not ok:
                            raise _OverrideRaced(frame_scope.fault)
                if content_request is not None:
                    # What this frame's redaction found, put to the access
                    # gate before it is sent; a frame it refuses (or one read
                    # under an admission revoked while it decided) hands a
                    # one-time override it used back and is never sent.
                    content = content_facts(
                        scanned=frame_scanned,
                        detected=_grown(connection_counts, detected_before)
                        if frame_scanned
                        else None,
                        warned=_grown(state.warn_counts, warned_before) if frame_scanned else None,
                        overridden=frame_marker is not None,
                        overridden_types=frame_scope.allowed_types()
                        if frame_scope is not None
                        else None,
                    )
                    if not await _content_allowed(
                        state,
                        frame_models.latest if frame_models is not None else content_request,
                        content,
                        relay,
                        frame_scope,
                        f"WS {path}",
                    ):
                        return
                if frame_scope is not None:
                    override_marker = frame_marker or override_marker
                    # Handed to the upstream next, with no check between.
                    frame_scope.settle(sent=True)
                await upstream.send(outbound)
            except _ContentRefused as refused:
                # The access gate refused what this frame's redaction found
                # (or its check failed): never sent — closed 1008 with the
                # gate's reason or the core's fixed text (never logged); the
                # row records the HTTP 403.
                logger.info("WS %s -> refused by the access gate (a frame's content)", path)
                status, refusal = 403, "authorization"
                await close_on_policy(refused.reason)
                return
            except _ModelRefused as refused:
                # The access gate refused the model a setup frame names (or
                # its check failed), or the frame named none: never redacted
                # or sent — closed 1008 with the gate's reason or the core's
                # fixed text (never logged); the row records the HTTP 403.
                logger.info("WS %s -> refused by the access gate (a setup frame's model)", path)
                status, refusal = 403, "authorization"
                await close_on_policy(refused.reason)
                return
            except _FrameRefused as refused:
                # The session router refused the frame (another user's stored
                # object, a storage location, …) or its check failed: never
                # redacted or sent, and the conversation cannot continue
                # without it — closed 1008 with the router's fixed reason
                # (never logged here); the row records the HTTP 403.
                logger.info("WS %s -> refused (the session router's frame check)", path)
                status, refusal = 403, "object_access"
                await close_on_policy(refused.reason)
                return
            except TooManyStrings as refused:
                # Too many strings to redact in one frame: never relayed
                # (1009, message too big); the row records the HTTP 413.
                logger.info("WS %s -> refused (%s)", path, refused)
                status, refusal = 413, "too_many_strings"
                await close_on_policy(f"refused by llm-redact ({refused})", code=1009)
                return
            except UnredactableRequest as refused:
                # Closed like a block: the frame never reaches the upstream
                # and the session cannot continue without it. The row
                # records 400 (the HTTP refusal status).
                logger.info("WS %s -> refused (%s)", path, refused)
                status = 400
                refusal = (
                    "placeholder_limit"
                    if isinstance(refused, PlaceholderLimitReached)
                    else "unredactable"
                )
                await close_on_policy(f"refused by llm-redact ({refused})")
                return
            except BlockedRequest as blocked:
                # Block mode on a realtime stream: the event must never
                # reach the upstream, and the connection cannot continue
                # coherently without it — close both sides (1008 = policy
                # violation; detector type only, never the value). The
                # reason carries the refusal's code when it can: approved,
                # it lets the value through on the next connection.
                logger.info("WS %s -> blocked (%s)", path, blocked)
                refusal = "blocked_value"
                allow_code = (
                    frame_scope.refusal_code("block", adapter.provider, "WS", path)
                    if frame_scope is not None
                    else None
                )
                await close_on_policy(
                    blocked_reason(
                        blocked.detector_type,
                        allow_code,
                        named=frame_scope is not None and bool(frame_scope.subject),
                        config_arg=frame_scope.config_arg if frame_scope is not None else None,
                    )
                )
                return
            except _OverrideRaced as raced:
                logger.info("WS %s -> refused (a one-time override could not be used)", path)
                status = 400
                refusal = "override_fault" if raced.fault else "override_raced"
                await close_on_policy(
                    OVERRIDE_FAULT_REASON if raced.fault else OVERRIDE_RACED_REASON
                )
                return
            except state.vault_faults as fault:
                # The vault could not record this frame's placeholders (a
                # write or its COMMIT failed; the frame's batch rolled
                # back): never sent, and the session cannot continue without
                # it — closed 1011 (internal error), recorded as the HTTP
                # path's 503, counted as the "vault" bookkeeping stage;
                # exception TYPE only.
                state.bookkeeping_errors["vault"] += 1
                logger.error(
                    "WS %s -> closed 1011 (vault write failed: %s); frame not sent",
                    path,
                    type(fault).__name__,
                )
                status, refusal = 503, "vault_fault"
                await close_on_policy(
                    "llm-redact could not record this frame's placeholders", code=1011
                )
                return

    async def upstream_to_client() -> None:
        try:
            async for frame in upstream:
                if relay.revoked is not None:
                    # Revoked (a reload, or the access gate — perhaps from
                    # another thread): no upstream frame is restored or sent
                    # to the client after it; the relay closes.
                    return
                if observe_frames:
                    # BEFORE the frame is restored or sent (synchronously, no
                    # await since the revoked check): what the router
                    # records from it is on record before the client can
                    # present it back. Its own parse — it never changes the
                    # frame — and a failure is contained (the frame is sent).
                    with state.map_write_barrier() as map_writes:
                        _observe_server_frame(
                            state,
                            adapter,
                            path,
                            frame,
                            identity=require_json,
                            session_id=ctx.session_id,
                        )
                    if map_writes:
                        # [vault] map_writes = "before_answer": a frame whose
                        # observation queued a durable map write (a Live
                        # resumption handle's owner) is held until it landed,
                        # so the client can resume on any replica. The
                        # revoked check before each send below follows this
                        # await: nothing is sent once the relay is revoked.
                        await state.await_map_writes(map_writes, f"WS {path}")
                for out in adapter.rehydrate_message(frame, pool):
                    if relay.revoked is not None:
                        return
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
            # A reload or the access gate revoking the relay wakes it here,
            # whichever way frames are (or are not) flowing.
            asyncio.create_task(relay.wait_revoked()),
        }
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if relay.revoked is not None:
            await close_on_revoke()
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(
                exc, (WebSocketDisconnect, websockets.exceptions.ConnectionClosed)
            ):
                raise exc
    except Exception as problem:
        status, refusal = 500, "delivery_fault"
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
            override=override_marker,
            refusal=refusal,
        )


class _OverrideRaced(Exception):
    """A frame relied on a one-time override another request used first
    (``fault``: the store could not record the use)."""

    def __init__(self, fault: bool) -> None:
        super().__init__()
        self.fault = fault


OVERRIDE_RACED_REASON = "llm-redact: the one-time override was already used; frame not forwarded"
OVERRIDE_FAULT_REASON = (
    "llm-redact: the one-time override could not be recorded; frame not forwarded"
)


def blocked_reason(
    detector_type: str, code: str | None, *, named: bool = False, config_arg: str | None = None
) -> str:
    """The 1008 close reason for a block-mode value: the type, and how to
    allow it when there is a code — the CLI with the code for the local
    operator, the dashboard for a ``named`` user — the longest wording that
    fits a close frame's 123 bytes, so the hint is never cut off. The CLI
    hint names ``config_arg`` (``--config``, the file the proxy was started
    with) when it fits, shortening the wording and dropping the type before
    the path (about 49 characters of path fit); a path too long for any
    wording falls back to the plain hint (the HTTP refusal of the same value
    carries it whole)."""
    if code is None:
        return f"blocked by llm-redact policy ({detector_type})"
    if named:
        hints = ["Refusal overrides in the llm-redact dashboard"]
    else:
        hints = [f"llm-redact override {code} --once|--always"]
        if config_arg is not None:
            hints.insert(0, f"llm-redact override --config {config_arg} {code} --once|--always")
    reasons = [
        reason
        for hint in hints
        for reason in (
            f"blocked by llm-redact policy ({detector_type}); to allow: {hint}",
            f"blocked ({detector_type}); allow: {hint}",
            f"blocked by llm-redact policy; to allow: {hint}",
            f"blocked; allow: {hint}",
        )
    ]
    # The last one (the plain hint, no type) always fits (pinned by test).
    fits = (r for r in reasons if len(r.encode("utf-8")) <= _MAX_CLOSE_REASON_BYTES)
    return next(fits, reasons[-1])


class _FrameRefused(Exception):
    """The session router's frame check refused a client frame (or failed):
    ``reason`` is the close reason — the router's fixed text, or the core's
    fault text — and is never logged."""

    def __init__(self, reason: str) -> None:
        super().__init__("refused by the session router's frame check")
        self.reason = reason


# The close reasons of a frame the core cannot put to the access gate's model
# check (``_FrameModels``); each fits a close frame's 123 bytes.
SETUP_FIRST = "llm-redact: the first frame must be a setup naming its model; nothing was forwarded"
FRAME_NOT_JSON = "llm-redact: a frame that is not JSON cannot be authorized; it was not forwarded"


class _ContentRefused(Exception):
    """The access gate's ``authorize_content`` refused a client frame (or
    its check failed): ``reason`` is the close reason — the gate's fixed
    text or the core's — and is never logged."""

    def __init__(self, reason: str) -> None:
        super().__init__("refused by the access gate's content check")
        self.reason = reason


def _grown(after: Mapping[str, int], before: Mapping[str, int]) -> dict[str, int]:
    """The counts that grew from ``before`` to ``after`` (one frame's own
    share), with the growth."""
    return {
        name: count - before.get(name, 0)
        for name, count in after.items()
        if count > before.get(name, 0)
    }


async def _prefetch_frame(
    adapter: WsAdapter,
    data: str | bytes,
    ctx: "RequestContext",
    *,
    require_json: bool,
    limit: int,
) -> "tuple[DetectorPlan, dict[str, dict[int, list[Detection]]]] | None":
    """The connection redactor's plan and its heavy detectors' results for
    every string this client frame's redaction will scan (the adapter's own
    ``redact_message`` run with a collector, which parses the frame itself)
    — None when the collecting pass failed (the frame is then detected
    inline, and refused by its real pass if it must be)."""
    from llm_redact.proxy import RequestContext  # runtime: avoids the import cycle

    plan = ctx.redactor.plan
    strings = collect(
        lambda collector: adapter.redact_message(
            data,
            RequestContext(ctx.session_id, ctx.vault, collector, ctx.rehydrator),
            inject_note=False,
            require_json=require_json,
        ),
        limit=limit,
    )
    if strings is None:
        return None
    return plan, await prefetch(plan, strings)


async def _content_allowed(
    state: "ProxyState",
    request: "AuthorizationRequest",
    content: ContentFacts,
    relay: RealtimeRelay,
    scope: "OverrideScope | None",
    where: str,
) -> bool:
    """The access gate's ``authorize_content`` on one client frame: True to
    send it; False when the relay was revoked while an awaited answer ran
    (the frame is then dropped and the relay closes as revoked); raises
    ``_ContentRefused`` for a refusal. On every way but True the one-time
    override the frame used is handed back (``scope.settle``)."""
    allowed = False
    try:
        verdict = state.authorization.content_refusal(request, content, where)
        if inspect.isawaitable(verdict):
            verdict = await verdict
            if relay.revoked is not None:
                # Revoked while the gate decided: the frame is not sent
                # under the admission it was read with.
                return False
        if verdict is not None:
            raise _ContentRefused(verdict)
        allowed = True
        return True
    finally:
        if not allowed and scope is not None:
            scope.settle(sent=False)


class _ModelRefused(Exception):
    """The access gate refused the model a setup frame names, its check
    failed, or the frame named none: ``reason`` is the close reason — the
    gate's fixed text or the core's — and is never logged."""

    def __init__(self, reason: str) -> None:
        super().__init__("refused by the access gate's model check")
        self.reason = reason


class _FrameModels:
    """The access gate's ``authorize_request`` asked about the model a
    realtime connection's frames name, with the upgrade's facts and
    ``model`` = that model, ``model_in_frame`` False.

    An adapter with ``model_in_frame`` (Gemini and Vertex Live): the
    connection's first client frame must be a setup naming a model; every
    later frame holding a setup must name one too and is checked alike; a
    frame that is not JSON is refused (it could name a model the check
    cannot read). An adapter with ``model_in_update`` (OpenAI, Azure): a
    frame that sets the model (``session.update`` carrying
    ``session.model``) must name it as a non-empty string and is checked;
    every other frame, a frame that is not JSON included, goes as before.
    ``checked``: whether the last frame's model was put to the gate."""

    def __init__(
        self,
        state: "ProxyState",
        adapter: WsAdapter,
        request: "AuthorizationRequest",
        where: str,
    ) -> None:
        self._authorization = state.authorization
        self._adapter = adapter
        self._request = request
        self._where = where
        self._first = adapter.model_in_frame
        self.checked = False
        # The request the gate was last asked with (the upgrade's until a
        # frame names a model): what its authorize_content is asked with.
        self.latest = request

    def verdict(self, parsed: tuple[Any, bool] | None) -> "str | None | Awaitable[str | None]":
        """None (forward the frame), a refusal's close reason, or an
        awaitable of either (the gate's bounded answer)."""
        first, self._first = self._first, False
        self.checked = False
        setup_first = self._adapter.model_in_frame
        if parsed is None:
            if first:
                return SETUP_FIRST
            return FRAME_NOT_JSON if setup_first else None
        payload = parsed[0]
        if not self._adapter.sets_model(payload):
            return SETUP_FIRST if first else None
        model = self._adapter.frame_model(payload)
        if model is None:
            return SETUP_FIRST if first else self._adapter.no_model_reason
        self.checked = True
        request = dataclasses.replace(self._request, model=model, model_in_frame=False)
        self.latest = request
        return self._authorization.refusal(request, self._where)


def _checked_frame(
    state: "ProxyState",
    adapter: WsAdapter,
    path: str,
    data: str | bytes,
    *,
    identity: bool,
    session_id: str,
    parsed: tuple[Any, bool] | None = None,
) -> tuple[Any, bool] | None:
    """One client frame parsed and put to the session router's frame check
    (``ProxyState.realtime_frame_refusal``): the parse (payload, was_binary)
    it passed, or None for a frame that is not JSON — which the check cannot
    read, so it is refused under the proxy's own identity
    (``UnredactableRequest``, like a frame no adapter can walk) and relayed
    as it came under the client's own key. A frame nesting too deep is
    refused whatever the credential; a refusal raises ``_FrameRefused``.
    ``parsed``: the frame's parse when the relay already has it."""
    if parsed is None:
        parsed = parse_client_frame(data)
    if parsed is None:
        _unparsed_frame(data, identity)
        return None
    refusal = state.realtime_frame_refusal(
        adapter.name, path, parsed[0], identity=identity, session_id=session_id
    )
    if refusal is not None:
        raise _FrameRefused(refusal)
    return parsed


def _observe_server_frame(
    state: "ProxyState",
    adapter: WsAdapter,
    path: str,
    data: str | bytes,
    *,
    identity: bool,
    session_id: str,
) -> None:
    """Hand one upstream frame to the session router's observer
    (``ProxyState.realtime_server_frame``) as a fresh parse of the
    provider's bytes — placeholders, never a restored value. A frame that is
    not JSON, or nests deeper than the proxy's JSON bound, is not observed
    (it is forwarded as it came, like any frame the adapter cannot walk)."""
    parsed = parse_json_text(data)
    if parsed is not None:
        state.realtime_server_frame(
            adapter.name, path, parsed[0], identity=identity, session_id=session_id
        )


def _unparsed_frame(data: str | bytes, require_json: bool) -> str | bytes:
    """A frame that is not JSON: forwarded byte-identically, or — under the
    proxy's own identity — refused (the frame KIND only, never its bytes)."""
    if require_json:
        raise UnredactableRequest("realtime frame is not JSON llm-redact can redact")
    return data


class _UnreadableJson(Exception):
    """A frame the JSON parser refuses for a reason other than its syntax
    (an integer past Python's int-digit limit): it may well be JSON, so it
    is never treated as "not JSON" (which a client frame under the client's
    own key is forwarded as, unread)."""


def _parse_frame(data: str | bytes) -> tuple[Any, bool] | None:
    """(parsed, was_binary) when ``data`` is a JSON text/binary frame, None
    when it is not JSON (a syntax error, or a binary frame that is not
    UTF-8); JsonTooDeep when it nests deeper than MAX_JSON_DEPTH and
    _UnreadableJson when the parser refuses it otherwise (no walk could
    read either)."""
    try:
        if isinstance(data, bytes):
            return loads_bounded(data.decode("utf-8")), True
        return loads_bounded(data), False
    except JsonTooDeep:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    except ValueError:
        raise _UnreadableJson from None


def parse_json_text(data: str | bytes) -> tuple[Any, bool] | None:
    """(parsed, was_binary) when ``data`` is a JSON text/binary frame the
    proxy reads, else None — the caller must then forward the frame
    byte-identically: an UPSTREAM frame nesting too deep to walk goes to
    the client as it came (its placeholders left in place). A client frame
    goes through ``parse_client_frame``."""
    try:
        return _parse_frame(data)
    except (JsonTooDeep, _UnreadableJson):
        return None


def parse_client_frame(data: str | bytes) -> tuple[Any, bool] | None:
    """``parse_json_text`` for a CLIENT frame: one nesting JSON too deep to
    walk, or one the parser refuses otherwise (an integer past the
    int-digit limit), is refused (UnredactableRequest: closed 1008,
    recorded 400 — the HTTP twin's answer), never forwarded unredacted as
    if it were not JSON. A parsed client frame is ALWAYS re-serialized
    (``_dump_frame``), never forwarded as its original bytes, so a repeated
    key's earlier occurrence (dropped by the parse, never walked) cannot
    leave."""
    try:
        return _parse_frame(data)
    except JsonTooDeep:
        raise UnredactableRequest(
            f"realtime frame nests JSON deeper than {MAX_JSON_DEPTH} levels"
        ) from None
    except _UnreadableJson:
        raise UnredactableRequest("realtime frame holds JSON llm-redact cannot read") from None


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
    return json_bytes(payload) if was_binary else json_text(payload)
