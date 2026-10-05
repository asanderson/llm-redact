"""Starlette app: the transparent proxy itself.

Design rules enforced here:
- Non-JSON or unrecognized traffic is forwarded verbatim — never break the
  agentic tool — but only to the provider it is POSITIVELY attributed to
  (a path family or one provider's markers), never a guessed one, and never
  with a credential the proxy holds; an unattributable request, or another
  spelling of a recognized route, is refused locally instead.
- An upstream redirect is never relayed where the client's repeat of its
  original request would carry what the proxy protects.
- Streaming vs JSON handling branches on the upstream *response*
  content-type, never the request's ``stream`` flag: an error reply to a
  streaming request arrives as plain JSON.
- Auth headers pass through untouched and are never logged; the log line
  contains only path, status, and detection counts.
"""

import asyncio
import dataclasses
import functools
import importlib.resources
import importlib.util
import inspect
import itertools
import json
import logging
import os
import re
import secrets
import signal
import sqlite3
import time
import unicodedata
import urllib.parse
from collections import Counter, deque
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from concurrent.futures import Future
from contextlib import AbstractContextManager, asynccontextmanager, nullcontext, suppress
from contextvars import ContextVar
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import httpx
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.requests import HTTPConnection, Request
from starlette.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute

from llm_redact import __version__
from llm_redact.audit import (
    AuditLog,
    AuditRecord,
    AuditWriteError,
    WriteAheadAudit,
)
from llm_redact.audit_s3 import AzureAuditSink, S3AuditSink
from llm_redact.authorization import (
    GateAuthorization,
    OverlayBuild,
    OverlayBuilds,
    content_facts,
)
from llm_redact.config import (
    DEFAULT_MAP_WRITE_WAIT_SECONDS,
    RDBMS_BACKENDS,
    RESTART_ONLY_KEYS,
    Config,
    ConfigError,
    ProviderConfig,
    _is_loopback_host,
    apply_env_overrides,
    default_config_path,
    identity_upstream_problem,
    load_config,
    map_writes_mode,
    normalize_origin,
    resolve_config_path,
    resolve_credentials,
    unsupported_plugin_capabilities,
)
from llm_redact.connections import CAUSES as CONNECTION_CLOSE_CAUSES
from llm_redact.connections import EventStream, LiveConnections, recheck_interval
from llm_redact.detection.engine import (
    active_rule_names,
    build_allowlist,
    build_detectors,
    build_modes,
)
from llm_redact.eventstream import EventStreamError, EventStreamParser
from llm_redact.eventstream import serialize as serialize_eventstream
from llm_redact.jsonwalk import (
    MAX_JSON_DEPTH,
    JsonTooDeep,
    json_bytes,
    loads_bounded,
    loads_request,
)
from llm_redact.licensing import ResolvedLicense, resolve_license
from llm_redact.metrics import (
    MAX_PLUGIN_SAMPLES,
    LocalRefusal,
    Metrics,
    plugin_metric_lines,
)
from llm_redact.multipart import parse as parse_multipart
from llm_redact.multipart import parse_boundary as parse_multipart_boundary
from llm_redact.ndjson import NDJSONParser
from llm_redact.overrides import (
    DISABLED_REASON,
    PROXY_BUSY_TIMEOUT_MS,
    TO_ENABLE,
    OverrideError,
    OverrideScope,
    OverrideStore,
    default_overrides_path,
    hint_config,
    printable,
    raced_message,
)
from llm_redact.placeholders import PLACEHOLDER_RE, json_floors, may_carry_tokens
from llm_redact.plugin_api import (
    AccessGate,
    Admission,
    AuthorizationRequest,
    ContentFacts,
    Dashboard,
    HopRequest,
    HopResult,
    LocalAnswer,
    ResponseContext,
    ResponseObserver,
    RouteDelivery,
    RouteInbound,
    RoutePlan,
    Router,
    RouteRefusal,
    Telemetry,
    UploadInspector,
    UpstreamAuth,
    UpstreamAuthError,
)
from llm_redact.providers import ALL_ADAPTERS, ProviderAdapter, RouteKind
from llm_redact.providers.attribution import (
    CUSTOM_ROUTE_PREFIX,
    OPENAI_PREFIXES,
    attribute,
    path_family,
    unattributed_reason,
    under,
)
from llm_redact.providers.base import (
    InspectedUpload,
    UnscannedBinaryFile,
    VerbatimFieldRedacted,
    prepare_route_request,
)
from llm_redact.providers.custom import build_custom_adapters, custom_prefix
from llm_redact.providers.openai import BINARY_FILE, remember_raw_texts
from llm_redact.realtime import (
    ALL_WS_ADAPTERS,
    RealtimeRelay,
    WsAdapter,
    websockets_available,
    ws_handle,
)
from llm_redact.redactor import (
    BlockedRequest,
    PlaceholderLimitReached,
    Redactor,
    StringBudget,
    TooManyStrings,
    UnredactableRequest,
)
from llm_redact.registry import get_registry, loaded_plugins, pro_package_installed
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent, SSEParser, serialize
from llm_redact.upload_content import classify_file
from llm_redact.upload_inspection import (
    BinaryValuesDetected,
    Limits,
    inspect_parts,
    inspector_limits,
    judge,
)
from llm_redact.upload_view import read_upload, read_upload_metadata
from llm_redact.vault import Vault, VaultManager, run_batched
from llm_redact.vault_crypto import key_source as vault_key_source
from llm_redact.vault_crypto import require_vault_key_source
from llm_redact.vault_writer import SHUTDOWN_DRAIN_SECONDS, awaited_writes

# Local endpoints under this prefix are answered by the proxy itself and are
# never forwarded upstream (see the first statement of handle()).
RESERVED_PREFIX = "/__llm-redact"
# The browser dashboard's paths (the bare prefix and trailing-slash form
# serve the page; /config is its editor, /preview its dry run). Answered by
# the registered llm-redact-pro Dashboard, else a 404 naming the package.
DASHBOARD_PATHS = frozenset(
    {
        RESERVED_PREFIX,
        f"{RESERVED_PREFIX}/",
        f"{RESERVED_PREFIX}/config",
        f"{RESERVED_PREFIX}/preview",
    }
)

# [vault] map_writes = "before_answer": how long an answer waits for the
# durable map writes it caused to land (``ProxyState.await_map_writes``).
# Past it the answer is sent anyway — another replica may then read the
# record as unknown a moment longer (refused or sealed, never a wrong
# value) — and the lag is counted (bookkeeping stage ``map_write_wait``).
# The default of ``[vault] map_write_wait_seconds``, which sets it.
MAP_WRITE_WAIT_SECONDS = DEFAULT_MAP_WRITE_WAIT_SECONDS
# The bookkeeping stage an answer sent before its map writes landed counts in.
MAP_WRITE_WAIT_STAGE = "map_write_wait"
# The bookkeeping stage a plugin's metrics samples count in when they cannot
# be read (a fault, a timeout, a call still running) or a sample is invalid.
PLUGIN_METRICS_STAGE = "plugin_metrics"
# How long a /metrics scrape waits for the access gate's samples (read in a
# worker thread): past it the core's own metrics are rendered without them.
PLUGIN_METRICS_TIMEOUT_SECONDS = 2.0


def _retrieve(future: "asyncio.Future[Any]") -> None:
    if not future.cancelled():
        future.exception()


def _listed_samples(samples: Callable[[], object]) -> list[object]:
    """``samples()`` read in full (in a worker thread): a lazy iterable is
    consumed there too, never on the event loop. At most one sample past
    ``MAX_PLUGIN_SAMPLES`` is taken, so an endless iterable is bounded
    (``plugin_metric_lines`` drops the extra one, counted). TypeError when
    the answer is not iterable."""
    answer = samples()
    if inspect.isawaitable(answer):
        if inspect.iscoroutine(answer):
            answer.close()
        raise TypeError("metrics_samples answered an awaitable")
    return list(itertools.islice(answer, MAX_PLUGIN_SAMPLES + 1))  # type: ignore[call-overload]


# How often the [vault] session_ttl_days background task sweeps for idle
# sessions. Retention is a slow signal; hourly is ample and keeps the sqlite
# work negligible.
_TTL_PRUNE_INTERVAL_SECONDS = 3600.0
_LICENSE_REFRESH_INTERVAL_SECONDS = 86400.0

# The llm-redact-pro sinks' own bound on one upload (their HTTP client's
# timeout): the shutdown deadline below must never be the one that cuts a
# slow but working upload short — the final flush uploads the sink's
# in-memory START/AMEND rows first, and rows it could not ship then are lost
# from the off-machine copy (spooled database rows are not: they wait).
_SINK_UPLOAD_TIMEOUT_SECONDS = 30.0
# Shutdown bound on the off-machine audit sinks' final flush (their aclose(),
# both sinks concurrently, from the still-open audit database): past it the
# flushes are cancelled — unshipped spooled rows stay in the database and ship
# at the next start — so a hanging store never keeps the audit database and
# the vault from closing. Above one upload's own timeout (plus slack for the
# client close), so only a stuck sink or a long backlog drain reaches it.
_SINK_CLOSE_TIMEOUT_SECONDS = _SINK_UPLOAD_TIMEOUT_SECONDS + 15.0
# How long a cancelled flush gets to unwind (close its HTTP client) before
# shutdown abandons it and closes the audit database anyway.
_SINK_CANCEL_GRACE_SECONDS = 1.0

# The inbound request's W3C traceparent, captured at the top of handle() and
# read at finalization time so the OTel span (built then) can parent into the
# caller's trace even across the streaming boundary — same task, same context.
_INBOUND_TRACEPARENT: ContextVar[str | None] = ContextVar("llm_redact_traceparent", default=None)
# The subject the access gate (llm-redact-pro) admitted the current request
# as: set once in handle()/ws_handle() after admission, read at finalization
# by record_request — the same task-context trick as the traceparent, so the
# streaming finalizers attribute without threading a parameter through.
_REQUEST_USER: ContextVar[str | None] = ContextVar("llm_redact_user", default=None)


class _RequestTiming:
    """How long one HTTP request waited on something other than the proxy
    itself: the upstream (sending it, its answer's headers and body, each
    streamed chunk, a routed retry's delay) and the client (its request
    body, a stream's consumer). ``record_request`` subtracts it from the
    request's duration: ``llm_redact_proxy_overhead_seconds``. Set fresh by
    handle() for every HTTP request (the task-context trick again: the
    streaming finalizers read the same object)."""

    __slots__ = ("waited",)

    def __init__(self) -> None:
        self.waited = 0.0


_REQUEST_TIMING: ContextVar[_RequestTiming | None] = ContextVar(
    "llm_redact_request_timing", default=None
)

_T = TypeVar("_T")


async def _waited(awaitable: Awaitable[_T]) -> _T:
    """Await ``awaitable`` — a wait on the upstream or the client — and
    charge its time to the current request's ``_RequestTiming``."""
    timing = _REQUEST_TIMING.get()
    started = time.perf_counter()
    try:
        return await awaitable
    finally:
        if timing is not None:
            timing.waited += time.perf_counter() - started


class _UpstreamPaced:
    """An upstream body's chunks, the wait for each charged to the request
    (``_RequestTiming``). An iterator object, not a generator: nothing of
    its own to finalize."""

    def __init__(self, chunks: AsyncIterable[bytes]) -> None:
        self._chunks = aiter(chunks)
        self._timing = _REQUEST_TIMING.get()

    def __aiter__(self) -> "_UpstreamPaced":
        return self

    async def __anext__(self) -> bytes:
        started = time.perf_counter()
        try:
            return await anext(self._chunks)
        finally:
            if self._timing is not None:
                self._timing.waited += time.perf_counter() - started


class _ClientPaced:
    """A streamed answer handed to the client: the time between yielding a
    chunk and being asked for the next — the client (and the server
    writing to it) taking it — charged to the request. The stream's own
    finalizer (``record_request``) runs inside the last ``__anext__``,
    after every such wait was charged."""

    def __init__(self, stream: AsyncIterator[bytes]) -> None:
        self._stream = stream
        self._timing = _REQUEST_TIMING.get()
        self._handed: float | None = None

    def __aiter__(self) -> "_ClientPaced":
        return self

    async def __anext__(self) -> bytes:
        if self._handed is not None and self._timing is not None:
            self._timing.waited += time.perf_counter() - self._handed
        chunk = await anext(self._stream)
        self._handed = time.perf_counter()
        return chunk


def _synchronous_audit_answer(answer: object, member: str) -> None:
    """Refuse an awaitable answer from a write-ahead audit member.

    ``begin``/``finalize``/``amend`` are synchronous: each must have made
    its row durable before it returns. An ``async def`` member returns a
    coroutine instead — nothing committed yet — and the core never awaits
    one on the request path, so the answer is closed unrun (no "never
    awaited" warning, the member's body never runs) and treated as the
    write fault it is: :class:`AuditWriteError`, never a durable row (a
    coroutine returned from ``begin`` once counted as a valid token, so no
    START row was ever written). A pending asyncio Task or Future is
    cancelled, so a write it schedules never lands after the refusal (an
    orphan START row for a request that never left, an END row after the
    CRITICAL line said it failed); any other awaitable is refused unawaited."""
    if inspect.isawaitable(answer):
        if inspect.iscoroutine(answer):
            answer.close()
        elif isinstance(answer, asyncio.Future):
            answer.cancel()
        raise AuditWriteError(f"write-ahead audit {member}() returned an awaitable")


def _require_synchronous_audit(log: WriteAheadAudit, amend: object) -> None:
    """Refuse at startup a write-ahead log whose ``begin``/``finalize``/
    ``amend`` is declared ``async def`` (a coroutine or async-generator
    function): its every answer would be refused per request
    (``_synchronous_audit_answer``) — a 503 for every request, or a CRITICAL
    END-row fault for every answer — so the deploy gate (``serve --check``)
    must report it rather than call it OK. A synchronous member that answers
    an awaitable is still refused per request."""
    members = (("begin", log.begin), ("finalize", log.finalize), ("amend", amend))
    asynchronous = [
        f"{name}()"
        for name, member in members
        if inspect.iscoroutinefunction(member) or inspect.isasyncgenfunction(member)
    ]
    if asynchronous:
        raise ConfigError(
            "[audit] required = true needs synchronous write-ahead audit members;"
            f" async def: {', '.join(asynchronous)}"
        )


def _finalize_audit(log: WriteAheadAudit, token: object, entry: AuditRecord) -> None:
    """Commit the END row ``token`` names; raises :class:`AuditWriteError`
    on a write fault, an awaitable answer included (closed unrun)."""
    finalize: Callable[[object, AuditRecord], object] = log.finalize
    _synchronous_audit_answer(finalize(token, entry), "finalize")


class _EarlyAudit:
    """The ``[audit] required`` START row an upload wrote BEFORE its binary
    parts were handed to the upload inspector (``_handle``'s
    ``before_inspection``), held for the request until its END row: the
    request's own ``record_request`` finalizes it — every refusal after the
    inspection included, since they record without a token — and
    ``handle()`` closes one still held on a way out that recorded nothing,
    so every START row gets exactly one END row. ``started``: the local
    refusals and the START row were already applied (the later sites skip
    them); the token is None when required mode is off. When the
    redaction then finds values (or the request passes on an override),
    the row is amended with them before the send (``amend``, a log with
    the optional member) or superseded by a second START row carrying
    them (``supersede``, a log without it): ``_start_after_early``."""

    def __init__(self) -> None:
        self.started = False
        self._token: object | None = None
        self._row: dict[str, Any] = {}

    def hold(self, token: object | None, **row: Any) -> None:
        self.started = True
        self._token = token
        self._row = row

    @property
    def holds_token(self) -> bool:
        return self._token is not None

    def take(self) -> object | None:
        token, self._token = self._token, None
        return token

    def amend(
        self, state: "ProxyState", *, detections: dict[str, int], warned: dict[str, int]
    ) -> None:
        """Amend the held START row with the request's counts, before
        upstream contact (``ProxyState.amend_audit``): it stays the
        request's one START row and the token stays held, so the request's
        own row — the send's END, or the 503 refusal's when the amendment
        raises ``AuditWriteError`` — finalizes it."""
        # Called only while a token is held, which only the write-ahead log mints.
        assert self._token is not None
        state.amend_audit(
            self._token,
            session=self._row["session"],
            provider=self._row["provider"],
            method=self._row["method"],
            path=self._row["path"],
            detections=detections,
            warned=warned,
            duration_ms=(time.perf_counter() - self._row["started"]) * 1000.0,
        )

    def supersede(self, state: "ProxyState") -> None:
        """End the held START row now, before upstream contact: a START row
        written after redaction (carrying the request's detections and
        warned counts) replaces it as the request's write-ahead record.
        Its END row goes straight to the write-ahead log (``status`` None,
        no detections) — never through ``record_request``, so metrics, the
        recent buffer and the sinks count the request once. A fault is
        logged CRITICAL by type only: the counted START row is committed,
        and the START row left open is adopted as interrupted later."""
        token, log = self.take(), state.write_ahead_audit
        # Called only while a token is held, which only that log mints.
        assert token is not None and log is not None
        entry = state._audit_entry(
            session=self._row["session"],
            provider=self._row["provider"],
            method=self._row["method"],
            path=self._row["path"],
            detections={},
            warned=None,
            duration_ms=(time.perf_counter() - self._row["started"]) * 1000.0,
        )
        try:
            _finalize_audit(log, token, entry)
        except AuditWriteError as exc:
            logger.critical(
                "audit write failed ending a superseded START row (%s %s): %s",
                self._row["method"],
                self._row["path"],
                type(exc).__name__,
            )

    def close(self, state: "ProxyState", status: int | None) -> None:
        token = self.take()
        if token is not None:
            state.record_request(
                **self._row,
                status=status,
                streamed=False,
                detections={},
                rehydrations={},
                audit_token=token,
                refusal=None,
            )


# The current request's early START row (``_EarlyAudit``), set by handle()
# and read by record_request, which finalizes it with the request's own row
# — the task-context trick again, so no refusal site threads a token.
_EARLY_AUDIT: ContextVar[_EarlyAudit | None] = ContextVar("llm_redact_early_audit", default=None)

# Whether the current request passed a refusal on an approved override
# ("once" / "always", overrides.py), read by record_request like the user —
# the row says so (the kind of use only). Set only once the request is
# handed to the upstream (``_UploadFate`` settled sent): a request refused
# before that hands its one-time grant back, and its row says no override.
_REQUEST_OVERRIDE: ContextVar[str | None] = ContextVar("llm_redact_override", default=None)
# The override the request committed as it passed the refusal, its fate not
# yet known: what the write-ahead START row (written right before the send)
# says is about to leave.
_COMMITTED_OVERRIDE: ContextVar[str | None] = ContextVar(
    "llm_redact_committed_override", default=None
)
# The 403 text when the session router's ownership check fails or answers
# something other than a reason (never an id or a user name).
_OBJECT_ACCESS_FAULT = (
    "llm-redact: the stored-object ownership check failed; the request was not forwarded"
)
# The realtime twin: the close reason when the session router's per-frame
# check fails or answers something other than a reason (fits a close frame's
# 123 bytes), and the bookkeeping stage that counts it.
REALTIME_FRAME_FAULT = "llm-redact: the realtime frame check failed; the frame was not forwarded"
REALTIME_FRAME_STAGE = "realtime_frame"
# The bookkeeping stage a failed observation of a realtime upstream frame
# (the optional SessionRouter.realtime_server_frame) counts in.
REALTIME_SERVER_FRAME_STAGE = "realtime_server_frame"
# A listed item the session router failed to answer for (its call raised):
# delivered exactly as the provider sent it, never restored in a session.
_UNANSWERED = object()

# Response headers stamped on every reserved-endpoint reply (dashboard, status,
# metrics, config editor, everything under RESERVED_PREFIX). The dashboard is
# fully self-contained — inline <style>/<script>, same-origin fetch/EventSource,
# no external resources — so a strict CSP holds it intact while blocking any
# injected content from loading remote code, framing the page, or leaking a
# Referer. Applied in one place (handle()) so it cannot drift per handler.
_SECURITY_HEADERS = {
    "content-security-policy": (
        "default-src 'none'; "
        "script-src 'unsafe-inline'; "
        "style-src 'unsafe-inline'; "
        "connect-src 'self'; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "frame-ancestors 'none'"
    ),
    "x-frame-options": "DENY",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
}

logger = logging.getLogger("llm_redact")


class RequestContext:
    """The session-scoped objects one request redacts and rehydrates with.

    ``sealed``: None, or — when the session router resolved this request to
    a session that must stay EMPTY (``SessionRouter.sealed``) — the fixed
    reason a redaction is refused with: the redactor refuses to write
    (``SealedSessionError``), so a request with anything to redact is
    refused and the session is never populated."""

    __slots__ = ("session_id", "vault", "redactor", "rehydrator", "sealed")

    def __init__(
        self,
        session_id: str,
        vault: Vault,
        redactor: Redactor,
        rehydrator: Rehydrator,
        sealed: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.vault = vault
        self.redactor = redactor
        self.rehydrator = rehydrator
        self.sealed = sealed


class SealedSessionError(Exception):
    """A request resolved to a sealed session had something to redact:
    writing it would populate a session that must stay empty."""


class _SealedVault:
    """A read-through view of a sealed session: lookups pass through, and
    any new placeholder raises ``SealedSessionError`` instead of writing."""

    def __init__(self, vault: Vault) -> None:
        self._vault = vault

    def placeholder_for(self, detector_type: str, original: str, *, floor: int = 0) -> str:
        # Whatever the request's token floor: nothing is ever written here.
        raise SealedSessionError(detector_type)

    def original_for(self, placeholder: str) -> str | None:
        return self._vault.original_for(placeholder)

    def close(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self._vault)


# The 403 text when a request resolved to a sealed session would redact a
# value (never the value, never an id), unless the router named its own.
_SEALED_REFUSAL = (
    "llm-redact: this request is served in a vault session that must stay empty (it"
    " reaches content llm-redact cannot attribute to you), and it carries values that"
    " would need redacting there; it was not forwarded"
)


# Hop-by-hop / recomputed headers dropped when forwarding either direction.
_SKIP_REQUEST_HEADERS = frozenset({"host", "content-length", "connection", "accept-encoding"})
# Every request header in the proxy's own namespace is for the proxy, never
# the provider: dropped before forwarding whether or not a plugin consumed
# it (a named-user key sent to a proxy without an access gate must still
# never leave the machine).
OWN_HEADER_PREFIX = "x-llm-redact-"
# METHOD OVERRIDES: a header or query parameter asking the upstream to run a
# method other than the request line's (Google's front end honors
# X-HTTP-Method-Override, OData services X-HTTP-Method, web frameworks a
# `_method` parameter). A matched route is redacted, restored, tracked and
# ownership-checked by the request line's method — a create overridden into
# a listing would be restored in the caller's session and its items claimed —
# so a matched route carrying one is refused (``method_override``) and the
# headers never leave on a matched route. Pass-through traffic (the
# client's own key, nothing read) forwards them as sent.
METHOD_OVERRIDE_HEADERS = frozenset(
    {"x-http-method-override", "x-http-method", "x-method-override"}
)
_METHOD_OVERRIDE_PARAMS = frozenset(
    {
        "_method",
        "$method",
        "httpmethod",
        "$httpmethod",
        *METHOD_OVERRIDE_HEADERS,
        *(f"${name}" for name in METHOD_OVERRIDE_HEADERS),
    }
)


def method_override(headers: Mapping[str, str], query: str) -> str | None:
    """Which KIND of method override the request carries ("header" or
    "query parameter"), or None. Names compare case-insensitively; query
    names are read decoded (``+`` and percent escapes)."""
    if any(name.lower() in METHOD_OVERRIDE_HEADERS for name in headers):
        return "header"
    names = (name for name, _ in urllib.parse.parse_qsl(query, keep_blank_values=True))
    if any(name.lower() in _METHOD_OVERRIDE_PARAMS for name in names):
        return "query parameter"
    return None


# The named-user base-path prefix. Only an access gate (llm-redact-pro)
# can accept it; a path still carrying it after admission is answered
# locally and never forwarded or recorded (its next segment is a key).
IDENTITY_PATH_PREFIX = "/u/"
# The access gate's admin endpoints (llm-redact-pro): dispatched only to a
# registered gate, like DASHBOARD_PATHS to the dashboard.
ACCESS_PATHS = frozenset(
    {
        f"{RESERVED_PREFIX}/users",
        f"{RESERVED_PREFIX}/users/invite",
        f"{RESERVED_PREFIX}/users/revoke",
    }
)
# The gate's browser sign-in endpoints (llm-redact-pro): reachable WITHOUT
# dashboard admission — they are how a browser obtains it. Everything under
# AUTH_PREFIX goes to the gate (sign-in methods add their own pages and JSON
# endpoints there, e.g. passkeys); AUTH_PATHS names the original three.
AUTH_PREFIX = f"{RESERVED_PREFIX}/auth"
AUTH_PATHS = frozenset(
    {
        f"{AUTH_PREFIX}/login",
        f"{AUTH_PREFIX}/callback",
        f"{AUTH_PREFIX}/logout",
    }
)


def _is_auth_path(path: str) -> bool:
    """A gate sign-in path: anything BELOW AUTH_PREFIX. The bare prefix is
    not one — it would skip dashboard admission for a path the gate never
    claimed (it fell through to the gate's user-admin handler)."""
    return path.startswith(AUTH_PREFIX + "/")


def origin_form_target(scope: Mapping[str, Any]) -> bool:
    """Whether the request target is origin-form (``/path``). The h11
    parser accepts other forms — ``%2Fv1…@evil.example/x`` or an absolute
    URL — and the raw path is forwarded verbatim after the upstream base,
    so anything else could move the request to a host the client chose."""
    raw_path = scope.get("raw_path")
    if isinstance(raw_path, bytes | bytearray):
        return raw_path.startswith(b"/")
    path = scope.get("path")
    return isinstance(path, str) and path.startswith("/")


# A `.`/`..` path segment, in any spelling an upstream might resolve:
# percent-encoded dots (%2E), and backslash separators (%5C, `\`) that
# some front ends (IIS/APIM-style gateways) treat as `/`.
_DOT_SEGMENT = re.compile(r"(?:^|/)\.{1,2}(?=/|$)")


def _dot_normalized(path: str) -> str:
    return path.lower().replace("%2e", ".").replace("%5c", "/").replace("\\", "/")


def has_dot_segment(scope: Mapping[str, Any]) -> bool:
    """Whether the request path has a ``.`` or ``..`` segment in its raw
    or decoded form. Matching runs on the decoded path while the raw path is
    forwarded after the upstream base, and httpx (like most servers) resolves
    dot segments: ``/async-invoke/../../any/op`` would match a recognized
    route here yet be sent — signed, under identity auth — to a different
    path upstream. No API llm-redact serves uses such a segment."""
    raw_path = scope.get("raw_path")
    if isinstance(raw_path, bytes | bytearray):
        raw = raw_path.split(b"?", 1)[0].decode("latin-1")
        if _DOT_SEGMENT.search(_dot_normalized(raw)):
            return True
    path = scope.get("path")
    return isinstance(path, str) and _DOT_SEGMENT.search(_dot_normalized(path)) is not None


def has_empty_segment(scope: Mapping[str, Any]) -> bool:
    """Whether the request path has an empty segment (``//``) in its raw or
    decoded form, backslash separators counted as ``/`` (the dot-segment
    spellings). Matching compares exact paths, so ``/v1//chat/completions``
    would miss the chat route and be forwarded unredacted to an upstream
    that merges slashes, and a leading ``//api/…`` would miss its path
    family altogether. No API llm-redact serves uses an empty segment (a
    Bedrock ARN encodes its slashes as ``%2F``), and a base URL ending in
    ``/`` joined naively onto an endpoint path is the usual cause."""
    raw_path = scope.get("raw_path")
    if isinstance(raw_path, bytes | bytearray):
        raw = raw_path.split(b"?", 1)[0].decode("latin-1")
        if "//" in _dot_normalized(raw):
            return True
    path = scope.get("path")
    return isinstance(path, str) and "//" in _dot_normalized(path)


# SCIM 2.0 provisioning (llm-redact-pro): everything under this prefix goes
# to the gate, which authenticates the identity provider's bearer token
# itself (no browser: no Origin check, no dashboard admission).
SCIM_PREFIX = f"{RESERVED_PREFIX}/scim/v2"
# Reserved endpoints a monitoring system polls: never behind dashboard
# admission (metadata-free probes and Prometheus counters).
PROBE_PATHS = frozenset(
    {
        f"{RESERVED_PREFIX}/healthz",
        f"{RESERVED_PREFIX}/readyz",
        f"{RESERVED_PREFIX}/metrics",
    }
)
_SKIP_RESPONSE_HEADERS = frozenset({"content-length", "content-encoding", "transfer-encoding"})

# Line-framed JSON response types, all served by the same NDJSON rehydration
# path: Ollama streams application/x-ndjson; Anthropic batch results are
# application/x-jsonl. (The upstream's own content-type header passes through
# to the client — the branch only picks the processing path.)
_JSONL_CONTENT_TYPES = ("application/x-ndjson", "application/x-jsonl", "application/jsonl")


def _resolve_license_info(config: Config) -> ResolvedLicense:
    """Resolve the configured license key for informational surfacing.

    The AGPL core enforces nothing: no tier gates, no seat caps, no cloud
    entitlements — every subsystem in this repository works keyless. The
    resolved tier is surfaced (/status, dashboard, doctor) and available to
    the llm-redact-pro plugin, whose own factories decide what its paid
    subsystems honor. Warnings (a key configured without the pro package,
    expiry grace) are logged loudly here — never silently. Shared by
    startup and apply_config (SIGHUP + editor) so the two can never
    disagree."""
    resolved = resolve_license(
        env=dict(os.environ),
        config_key=config.license.key,
        config_key_file=config.license.key_file,
    )
    for warning in resolved.warnings:
        logger.warning("license: %s", warning)
    return resolved


class ProxyState:
    def __init__(
        self,
        config: Config,
        upstream_transport: httpx.AsyncBaseTransport | None,
        config_path: Path | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.started_at = time.time()
        # License resolution is informational only (the FOSS core has no
        # tier gates): the tier is surfaced and handed to the pro plugin's
        # factories. What fails closed is a config that requests a
        # subsystem only llm-redact-pro implements when that package is
        # absent — those errors come from the registry factories below.
        self.license: ResolvedLicense = _resolve_license_info(config)
        # Swappable subsystems (vault/sessions/telemetry/…) are built through
        # the plugin registry: the Free defaults live in-tree; the paid
        # llm-redact-pro package overrides them via an entry-point hook
        # (llm-redact-pro docs/licensing.md). Wiring only — the license gate above
        # is the sole tier chokepoint.
        registry = get_registry()
        # [vault.kms] fails closed BEFORE any vault is built: a local key
        # variable alongside it, or no plugin able to unwrap it (an older
        # llm-redact-pro would silently build its cipher from the env key).
        require_vault_key_source(config.vault, registry)
        self.vault_manager: VaultManager = registry.build_vault_manager(config.vault)
        self.vault: Vault = self.vault_manager.get(config.vault.session)
        # What a vault fault raises while a request's placeholders are issued
        # (a write, its batch's COMMIT): refused 503, never a bare 500.
        self.vault_faults = vault_fault_types(self.vault_manager)
        self.detectors = build_detectors(config.detection)
        self.allowlist = build_allowlist(config.detection)
        self.modes = build_modes(config.detection)
        # The access gate's detection overlays (authorization.py), built per
        # distinct overlay against exactly these detection objects.
        self.overlay_builds = OverlayBuilds(config.detection, self.modes)
        # Process-lifetime totals by type (for /status), shared across all
        # per-session redactors/rehydrators.
        self.detection_counts: Counter[str] = Counter()
        self.rehydration_counts: Counter[str] = Counter()
        self.warn_counts: Counter[str] = Counter()
        self.blocked_counts: Counter[str] = Counter()
        # Upstream transport faults (connect/read/timeout/mid-body drop) by
        # provider — a resilience health signal, metadata only. The proxy
        # fails these closed with a 502; this counts how often.
        self.upstream_errors: Counter[str] = Counter()
        # Faults in the proxy's own bookkeeping, by stage: after the provider
        # answered, "response_id" / "object_ids" / "listing" are session
        # bookkeeping and "response_observer" the router's observation of the
        # answer (contained — the answer is still delivered, never a
        # wrong value) and "delivery" is restoring the answer itself (a
        # buffered one fails closed with a recorded 502; a stream is cut);
        # before any upstream contact, "vault" is issuing a request's
        # placeholders (a vault write or its COMMIT failed: rolled back, a
        # recorded 503 — a realtime frame closes 1011); "vault_check" is a
        # vault view's staleness check that could not read its database
        # (contained: the view keeps serving its cache, never a wrong value);
        # "recheck" is an open connection's access re-check that failed or
        # timed out (the connection is closed: connections.LiveConnections);
        # "realtime_frame" a realtime client frame's check that failed (the
        # connection is closed) and "realtime_server_frame" the router's
        # observation of an upstream frame (contained: the frame is sent).
        self.bookkeeping_errors: Counter[str] = Counter()
        bind_fault_counter = getattr(self.vault_manager, "bind_fault_counter", None)
        if callable(bind_fault_counter):
            bind_fault_counter(self.bookkeeping_errors)
        # The durable maps (Responses chains, owner records, Live handles)
        # are written after the provider answered: a background writer
        # (llm_redact.vault_writer) keeps a slow disk or a remote database
        # round trip off the event loop, its overlay answering lookups until
        # each write lands. Optional (getattr): a manager without it writes
        # synchronously, as before.
        background = getattr(self.vault_manager, "write_maps_in_background", None)
        if callable(background):
            background()
        # [vault] map_writes (restart-only): "before_answer" holds an answer
        # until the map writes it caused landed, so another replica sharing
        # the vault finds the record as soon as the client can cite it
        # (``map_write_barrier`` / ``await_map_writes``). Nothing is awaited
        # for a manager that writes synchronously (no background writer):
        # its writes have landed when the call returns — reported as
        # "synchronous", the mode it actually runs (never a claimed wait).
        self.map_writes = map_writes_mode(config.vault) if callable(background) else "synchronous"
        self.awaits_map_writes = self.map_writes == "before_answer"
        self.map_write_wait_seconds = config.vault.map_write_wait_seconds
        # Answers sent before their map writes landed, by cause: "bound"
        # (the wait reached map_write_wait_seconds) or "stuck" (not waited
        # for: the writer is stuck behind an earlier write a wait gave up
        # on). Also counted under the bookkeeping stage MAP_WRITE_WAIT_STAGE.
        self.map_write_wait_timeouts: Counter[str] = Counter()
        # Whether the last wait ran out of time (logged once per episode).
        self._map_writes_lagging = False
        # A write the last timed-out wait gave up on, while it has not left
        # the writer: one writer applies writes in submission order, so
        # every later write is queued behind it and waiting again is futile
        # (a hung database would otherwise cost EVERY answer the bound).
        self._map_write_stuck: Future[None] | None = None
        # Requests refused as a web page's (request_origin_refusal), by kind:
        # "host" (a name the proxy does not answer to — DNS rebinding, or an
        # alias not in allowed_hosts), "origin", "fetch_site". Kinds only.
        self.request_origin_refusals: Counter[str] = Counter()
        # Binary file parts of uploads forwarded UNSCANNED ([detection]
        # binary_uploads = "forward", the client's own credential), by
        # provider: an honesty counter — those bytes left unread.
        self.unscanned_uploads: Counter[str] = Counter()
        self.redactor = Redactor(
            self.detectors,
            self.vault,
            self.allowlist,
            counts=self.detection_counts,
            modes=self.modes,
            warn_counts=self.warn_counts,
        )
        self.rehydrator = Rehydrator(
            self.vault, fuzzy=config.rehydration.fuzzy, counts=self.rehydration_counts
        )
        # A vault manager without a durable response map (the in-memory one)
        # passes no lookup: a lookup that always answers "unknown" would be
        # read as "that session was pruned" and orphan every Responses chain.
        self.session_router = registry.build_session_router(
            config.vault,
            durable_lookup=(
                self.vault_manager.lookup_response_session
                if getattr(self.vault_manager, "durable_response_map", True)
                else None
            ),
        )
        # Optional ownership members (plugin_api.SessionRouter), read ONCE:
        # the router is restart-only, and a router without them costs the
        # hot path one `is None` test each.
        self._object_access_refusal = getattr(self.session_router, "object_access_refusal", None)
        self._listing_item_session = getattr(self.session_router, "listing_item_session", None)
        self._listing_item_sessions = getattr(self.session_router, "listing_item_sessions", None)
        self._record_object_id = getattr(self.session_router, "record_object_id", None)
        self._sealed = getattr(self.session_router, "sealed", None)
        # Optional response observation (plugin_api.SessionRouter), read ONCE
        # like the members above: without it, delivering an answer costs one
        # test of this flag and no observation code runs.
        self._response_observer = getattr(self.session_router, "response_observer", None)
        self.observes_responses = self._response_observer is not None
        # Optional per-frame realtime check (plugin_api.SessionRouter), read
        # ONCE: a relay reads `checks_realtime_frames` once per connection.
        self._realtime_frame_refusal = getattr(self.session_router, "realtime_frame_refusal", None)
        # Optional observation of realtime SERVER frames, read ONCE likewise:
        # without it no upstream frame is parsed for the router.
        self._realtime_server_frame = getattr(self.session_router, "realtime_server_frame", None)
        self._static_context = RequestContext(
            config.vault.session, self.vault, self.redactor, self.rehydrator
        )
        self._known_sessions: set[str] = {config.vault.session}
        # Per-conversation sessions born with placeholders already in their
        # first message: the history-compaction signature (the anchor was
        # rewritten, so the conversation forked into a fresh namespace —
        # deliberately fail-safe; see docs/compaction-relink.md).
        self.compaction_forks = 0
        self.adapters: list[ProviderAdapter] = [
            cls() for cls in ALL_ADAPTERS
        ] + build_custom_adapters(config.providers)
        self.ws_adapters: list[WsAdapter] = [cls() for cls in ALL_WS_ADAPTERS]
        # Every open realtime relay's admission (realtime.RealtimeRelay):
        # apply_config revokes each one its swap changed, so a relay never
        # outlives the configuration it was admitted under.
        self.realtime_relays: set[RealtimeRelay] = set()
        self.client = httpx.AsyncClient(
            transport=upstream_transport, timeout=httpx.Timeout(600.0, connect=10.0)
        )
        self.metrics = Metrics(__version__)
        self.metrics.map_writes_mode = self.map_writes
        # The access gate's own gauges (its optional metrics_samples, read
        # off the event loop by ``plugin_metric_text``): the call in flight,
        # which a scrape never starts a second of (a concurrent scrape waits
        # for it until its deadline, the loop time its bound ends), and
        # whether the last one failed (logged once per episode).
        self._plugin_metrics_call: asyncio.Future[list[object]] | None = None
        self._plugin_metrics_deadline = 0.0
        self._plugin_metrics_failing = False
        # Last-N request summaries for the dashboard's recent table: memory
        # only, metadata only (types and counts — never values), available
        # whether or not the audit DB is enabled.
        self.recent: deque[dict[str, Any]] = deque(maxlen=200)
        # Live subscribers to the /events SSE feed: each gets its own
        # bounded queue; a slow consumer drops events (it still has the
        # dashboard's poll fallback) rather than backpressuring the proxy.
        self.event_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        # Per-process CSRF token for the guarded local POSTs (session prune,
        # user invite/revoke, the pro dashboard's editor and preview): handed
        # out only by the dashboard's same-origin GET (the proxy never sends
        # CORS headers), required as a custom header on every such POST.
        self.csrf_token = secrets.token_urlsafe(32)
        # The packaged user guide, served at /__llm-redact/guide — same
        # self-contained, load-once treatment as the dashboard.
        self.guide_html = (
            importlib.resources.files("llm_redact").joinpath("user_guide.html").read_text("utf-8")
        )
        if loaded_plugins():
            # Version skew: a plugin predating a configured capability
            # (sink/email auth modes) must refuse, never silently degrade.
            unsupported = unsupported_plugin_capabilities(config, registry.config_capabilities)
            if unsupported is not None:
                raise ConfigError(unsupported)
        # Audit log + off-machine sinks: registry-built (paid), each fail-closed
        # inside its factory (tamper chain without a key / fernet without a key
        # raise ConfigError there). The flush loops start in the lifespan.
        self.audit: AuditLog | None = registry.build_audit(config.audit)
        # [audit] required needs the write-ahead pair, resolved ONCE here
        # (audit is restart-only — apply_config pins it, so this cannot go
        # stale): a pro package predating the pair, or a required config
        # that somehow built no log at all, must refuse at startup — never
        # silently run fail-open under a config that promises zero loss.
        self.write_ahead_audit: WriteAheadAudit | None = None
        # The log's OPTIONAL amend member (WriteAheadAudit's docstring), read
        # once: an upload's START row written before its inspection is then
        # amended with the request's counts instead of followed by a second
        # START row. None: a log predating it (two START rows, as before).
        self.write_ahead_amend: Callable[[object, AuditRecord], None] | None = None
        if config.audit.required:
            if not isinstance(self.audit, WriteAheadAudit):
                raise ConfigError(
                    "[audit] required = true needs a llm-redact-pro version with"
                    " write-ahead audit support (AuditLog.begin/finalize)"
                )
            self.write_ahead_audit = self.audit
            amend = getattr(self.audit, "amend", None)
            self.write_ahead_amend = amend if callable(amend) else None
            _require_synchronous_audit(self.audit, self.write_ahead_amend)
        self.audit_s3: S3AuditSink | None
        self.audit_azure: AzureAuditSink | None
        self.audit_s3, self.audit_azure = registry.build_audit_sinks(config.audit)
        # None unless [otel] enabled = true (then the extra must be
        # installed — build_telemetry fails loudly with the install hint).
        self.telemetry: Telemetry | None = registry.build_telemetry(config.otel)
        # Client admission (llm-redact-pro): None serves the implicit single
        # local user; the Free default fails closed when a paid tier or
        # [users] config expects access control no gate provides.
        self.access_gate: AccessGate | None = registry.build_access_gate(config, self.license)
        # Optional gate members (plugin_api.AccessGate): dashboard admission,
        # and the public origin it makes reachable. The origin is honored
        # only together with admission, so a wider Host is never accepted
        # without the gate authenticating the reserved endpoints.
        self.guards_dashboard = bool(getattr(self.access_gate, "guards_dashboard", False))
        self.public_origin: tuple[str, str, str] | None = (
            _parse_public_origin(self.access_gate) if self.guards_dashboard else None
        )
        # Optional authorization members (authorize_request,
        # detection_overlay), read once: without them a request pays one
        # attribute test each.
        self.authorization = GateAuthorization(self.access_gate, self.bookkeeping_errors)
        # Optional: the gate may drop a deleted user's sessions through the
        # live vault manager (plugin_api.SessionStore).
        bind_sessions = getattr(self.access_gate, "bind_sessions", None)
        if callable(bind_sessions):
            bind_sessions(_LiveSessions(self))
        # Open long-lived connections (realtime relays, live-events streams)
        # and the admission each opened under (plugin_api.ConnectionControl):
        # the gate may close them the moment it revokes a user or a
        # credential, and the lifespan's backstop re-checks them every
        # recheck_interval (an optional gate member, read once; the backstop
        # starts at once when the gate declares one, else with the first
        # connection whose admission carries a recheck).
        self.connections = LiveConnections(
            self.bookkeeping_errors,
            interval=recheck_interval(self.access_gate),
            eager=getattr(self.access_gate, "recheck_interval", None) is not None,
        )
        bind_connections = getattr(self.access_gate, "bind_connections", None)
        if callable(bind_connections):
            bind_connections(self.connections)
        for warning in config.routing.warnings:  # parser-owned (inert upstreams, metered
            logger.warning("routing: %s", warning)  # defaults, dead chains) — always logged
        self.router: Router | None = registry.build_router(config, self.license.tier)
        # The browser dashboard (llm-redact-pro): None without the package
        # or when it declines the tier; the dashboard paths then 404 with
        # the reason. Rebuilt by apply_config when a reload changes the tier.
        self.dashboard: Dashboard | None = registry.build_dashboard(self.license.tier)
        # [providers.NAME] auth = "identity" (llm-redact-pro): provider name ->
        # the authorizer that signs its requests with the proxy's own cloud
        # identity. Empty unless configured; the Free default fails closed.
        self.upstream_auth: dict[str, UpstreamAuth] = _build_upstream_auths(config.providers)
        # Binary upload parts (PDFs, Office documents) read as text for the
        # core to scan (plugin_api.UploadInspector: the core's [extraction]
        # extractors unless a plugin replaces the factory): None keeps the
        # unscanned-binary rules. Restart-only: built once with the resolved
        # tier, its declared bounds read once (inspector_limits).
        self.upload_inspector: UploadInspector | None = registry.build_upload_inspector(
            config, self.license.tier
        )
        if config.extraction.enabled and self.upload_inspector is None:
            # A plugin's factory that predates the core's [extraction] (it
            # read the section from its own config) would silently drop it.
            raise ConfigError(
                "[extraction] enabled = true, but the registered upload inspector factory"
                " (a plugin's) built none: upgrade the plugin (llm-redact-pro) to a version"
                " that leaves [extraction] to the core"
            )
        self.inspection_limits: Limits | None = (
            inspector_limits(self.upload_inspector) if self.upload_inspector is not None else None
        )
        # Inspected binary upload parts by (provider, outcome) — an honesty
        # counter like unscanned_uploads (upload_inspection.OUTCOMES).
        self.inspected_uploads: Counter[tuple[str, str]] = Counter()
        # Refusal overrides (overrides.py): the requester's approved
        # one-time grants and every-time rules, read from their store only
        # where a detection refusal is being decided. None when
        # [overrides] enabled = false: every refusal is final, no code.
        self.overrides: OverrideStore | None = (
            OverrideStore(
                Path(config.overrides.path).expanduser()
                if config.overrides.path
                else default_overrides_path(),
                ttl_seconds=config.overrides.ttl_minutes * 60,
                busy_timeout_ms=PROXY_BUSY_TIMEOUT_MS,
            )
            if config.overrides.enabled
            else None
        )
        # The config file the CLI hint names (``--config``) when the CLI's
        # default search would not find the one this proxy was started with.
        self.override_config_arg = (
            hint_config(config_path, os.environ) if self.overrides is not None else None
        )

    def override_scope(self) -> OverrideScope | None:
        """This request's view of its requester's overrides — the subject
        the access gate admitted, else the local operator — or None when
        overrides are off."""
        if self.overrides is None:
            return None
        subject = _REQUEST_USER.get() or ""
        # The access gate is asked nothing here: its answers (can this
        # requester approve, under which stable id are its records kept) are
        # read only once a refusal is being decided — never per request or
        # per realtime frame.
        return OverrideScope(
            self.overrides,
            subject,
            approvable=functools.partial(self._approves_overrides, subject),
            owner=functools.partial(self.override_owner, subject),
            config_arg=self.override_config_arg,
        )

    def override_owner(self, subject: str) -> str | None:
        """The key a requester's override records are stored under: "" for
        the local operator; for a named user the access gate's OPTIONAL
        ``override_subject(subject)`` — a stable id that survives a rename
        and is never reused for another user (llm-redact-pro: the user's
        namespace) — prefixed ``id:``, else (no such member) the subject
        itself. None when the member fails or answers anything but a
        non-empty string: then no override applies and no code is minted
        (never the subject instead: a reused name would inherit records)."""
        if not subject:
            return ""
        stable = getattr(self.access_gate, "override_subject", None)
        if stable is None:
            return subject
        try:
            key = stable(subject)
        except Exception as exc:  # noqa: BLE001 — no override, never another's records
            logger.warning("overrides: override_subject failed (%s)", type(exc).__name__)
            return None
        if not isinstance(key, str) or not key:
            logger.warning("overrides: override_subject answered no id")
            return None
        return f"id:{key}"

    def _approves_overrides(self, subject: str) -> bool:
        """Whether this requester can approve its own refusal: the local
        operator always (the CLI); a named user only when the access gate's
        OPTIONAL ``approves_overrides(subject)`` says True (llm-redact-pro:
        a user who can sign in to the dashboard) — absent, False or an
        exception: no code and no hint it could not act on."""
        if not subject:
            return True
        approves = getattr(self.access_gate, "approves_overrides", None)
        if approves is None:
            return False
        try:
            return approves(subject) is True
        except Exception as exc:  # noqa: BLE001 — a hint, never a failed request
            logger.warning("overrides: approves_overrides failed (%s)", type(exc).__name__)
            return False

    async def browser_signed_in(self, conn: HTTPConnection, subject: str) -> bool:
        """Whether a person signed in to the dashboard as ``subject`` in a
        browser — the access gate's OPTIONAL ``browser_signed_in(conn,
        subject)`` says True (a session its browser sign-in established),
        never a credential an agent holds (an API key, a per-user key, a
        bearer token, a certificate). Absent, False, anything but True or an
        exception: no (the refusal-override POSTs need one)."""
        signed_in = getattr(self.access_gate, "browser_signed_in", None)
        if signed_in is None or not subject:
            return False
        try:
            verdict = signed_in(conn, subject)
            if inspect.isawaitable(verdict):
                verdict = await verdict
        except Exception as exc:  # noqa: BLE001 — no proof of a person
            logger.warning("overrides: browser_signed_in failed (%s)", type(exc).__name__)
            return False
        return verdict is True

    async def admit(self, conn: HTTPConnection, surface: str) -> Admission:
        """The access gate's verdict (it scrubs its own credentials from the
        scope first); without a gate every client is the implicit single
        local user. A gate may answer directly or with an awaitable."""
        if self.access_gate is None:
            return Admission()
        verdict = self.access_gate.admit(conn, surface)
        if inspect.isawaitable(verdict):
            return await verdict
        return verdict

    def _is_compaction_fork(self, session_id: str, vault: Vault, flat_body: str) -> bool:
        """A session first seen by this process is a compaction fork only when
        its history carries placeholders it cannot own: the session must be
        EMPTY (a persisted session resumed after a restart owns its tokens)
        and conversation-derived (a session the router marks durable, e.g.
        llm-redact-pro's per-user copy of the static session, is not anchored
        on a first message, so compaction cannot fork it)."""
        if not PLACEHOLDER_RE.search(flat_body) or len(vault) > 0:
            return False
        is_durable = getattr(self.session_router, "is_durable", None)
        if is_durable is None:
            return True
        try:
            # The prune's contract: only an explicit False says "not durable";
            # a failing or sloppy router never fails the request over a metric.
            return is_durable(session_id) is False
        except Exception:  # noqa: BLE001
            return False

    def context_for(
        self,
        adapter: ProviderAdapter | None,
        method: str,
        path: str,
        parsed_body: Any,
        overlay: OverlayBuild | None = None,
    ) -> RequestContext:
        """The request's session objects. ``overlay``: the requester's
        detection overlay (``authorization.OverlayBuilds.build``, against
        this generation's detection objects — read in the same synchronous
        stretch): the request's redactor runs its modes instead of the
        configured ones, and its deny strings on top of the configured
        detectors; None keeps the shared static context (the fast path)."""
        if self.session_router.mode == "static":
            return self._overlaid(self._static_context, overlay)
        session_id = self.session_router.resolve(
            adapter.name if adapter is not None else None, method, path, parsed_body
        )
        sealed = self._session_sealed(session_id)
        if session_id == self._static_context.session_id and sealed is None:
            return self._overlaid(self._static_context, overlay)
        vault = self.vault_manager.get(session_id)
        if session_id not in self._known_sessions:
            self._known_sessions.add(session_id)
            # Searched for placeholders only, never encoded or sent: the
            # plain dump is fine even with a lone surrogate in it.
            flat = json.dumps(parsed_body, ensure_ascii=False) if parsed_body is not None else ""
            if self._is_compaction_fork(session_id, vault, flat):
                # A brand-new conversation whose history already contains
                # placeholder tokens: the compaction signature. Tokens owned
                # by the original session pass through verbatim from here on
                # (never wrong-value) — surfaced so users can see why.
                self.compaction_forks += 1
                logger.info(
                    "new conversation session %s carries existing placeholders —"
                    " likely a compacted history; forked fail-safe (fork #%d)",
                    session_id,
                    self.compaction_forks,
                )
            else:
                logger.info(
                    "new conversation session %s (sessions this run: %d)",
                    session_id,
                    len(self._known_sessions),
                )
        # Thin per-request wrappers over the shared detectors, allowlist and
        # counters: object construction only — no regex compilation, no DB open.
        redactor = Redactor(
            self.detectors,
            _SealedVault(vault) if sealed is not None else vault,
            self.allowlist,
            counts=self.detection_counts,
            modes=overlay.modes if overlay is not None else self.modes,
            warn_counts=self.warn_counts,
            added_deny=overlay.deny if overlay is not None else None,
            final_blocks=overlay.final_blocks if overlay is not None else frozenset(),
        )
        rehydrator = Rehydrator(
            vault, fuzzy=self.config.rehydration.fuzzy, counts=self.rehydration_counts
        )
        return RequestContext(session_id, vault, redactor, rehydrator, sealed=sealed)

    def _overlaid(self, ctx: RequestContext, overlay: OverlayBuild | None) -> RequestContext:
        """``ctx`` (an unsealed shared context) itself, or — with an overlay
        — a thin copy whose redactor runs the overlay's modes and added deny
        strings over the same detectors, vault, allowlist, counters and
        rehydrator."""
        if overlay is None:
            return ctx
        redactor = Redactor(
            self.detectors,
            ctx.vault,
            self.allowlist,
            counts=self.detection_counts,
            modes=overlay.modes,
            warn_counts=self.warn_counts,
            added_deny=overlay.deny,
            final_blocks=overlay.final_blocks,
        )
        return RequestContext(ctx.session_id, ctx.vault, redactor, ctx.rehydrator)

    def _session_sealed(self, session_id: str) -> str | None:
        """The optional ``SessionRouter.sealed`` for the session this
        request was just resolved to, as the reason a redaction into it is
        refused with (None: not sealed). A non-empty string is the router's
        own fixed reason; any other truthy answer — and a router that
        raises — seals with the core's: the cost of being wrong is a
        refusal, never a populated session."""
        check = self._sealed
        if check is None:
            return None
        try:
            verdict = check(session_id)
        except Exception as exc:  # noqa: BLE001 — fail closed
            logger.warning("session router sealed failed (%s); sealing", type(exc).__name__)
            return _SEALED_REFUSAL
        if isinstance(verdict, str) and verdict:
            return verdict
        return _SEALED_REFUSAL if verdict else None

    def map_writes_pending(self) -> int:
        """The durable map writes queued or in flight on the vault's
        background writer (its OPTIONAL ``map_writes_pending``); 0 without
        one, or when the manager's answer is not a count."""
        pending = getattr(self.vault_manager, "map_writes_pending", None)
        if not callable(pending):
            return 0
        try:
            count = pending()
        except Exception as exc:  # noqa: BLE001 — a gauge never fails the scrape
            logger.warning("vault map writer depth unreadable (%s)", type(exc).__name__)
            return 0
        return count if isinstance(count, int) and not isinstance(count, bool) else 0

    def map_write_barrier(self) -> AbstractContextManager[list[Future[None]] | None]:
        """Around the synchronous bookkeeping of an answer (or of one
        streamed event, or realtime frame): with ``map_writes =
        "before_answer"``, collects a future for each durable map write the
        block queues (``vault_writer.awaited_writes``) — the caller awaits
        them (``await_map_writes``) before it sends what carries the
        recorded id. Otherwise None: nothing to wait for."""
        return awaited_writes() if self.awaits_map_writes else nullcontext(None)

    async def await_map_writes(self, pending: Sequence[Future[None]], where: str) -> None:
        """Wait — on the event loop, never blocking it: each write runs on
        the vault's writer thread — until the ``pending`` map writes left the
        writer (landed, or failed: a failed write reads as unknown, as it
        did written synchronously), at most ``map_write_wait_seconds``. Past
        that the answer goes on anyway (only lag: another replica reads the
        record as unknown a little longer — refused or sealed, never a wrong
        value), counted under the ``map_write_wait`` bookkeeping stage and
        logged once per episode; ``where`` (method and path, or "WS path")
        names the request, never an id. While a write a timed-out wait gave
        up on has still not left the writer, the answer is sent at once
        (counted the same): the writer is stuck behind it, so one answer per
        episode pays the bound, not every answer."""
        stuck = self._map_write_stuck
        if stuck is not None and not stuck.done():
            self.bookkeeping_errors[MAP_WRITE_WAIT_STAGE] += 1
            self.map_write_wait_timeouts["stuck"] += 1
            return
        self._map_write_stuck = None
        # The writes are not cancelled when the wait gives up (a cancelled
        # future would read as left the writer): their wrappers simply
        # complete unobserved once they do.
        waits = [asyncio.wrap_future(future) for future in pending]
        _done, late = await asyncio.wait(waits, timeout=self.map_write_wait_seconds)
        if late:
            self._map_write_stuck = next((f for f in pending if not f.done()), None)
            self.bookkeeping_errors[MAP_WRITE_WAIT_STAGE] += 1
            self.map_write_wait_timeouts["bound"] += 1
            if not self._map_writes_lagging:
                self._map_writes_lagging = True
                logger.warning(
                    "%s -> sent before its vault map writes landed (waited %ss); another"
                    " replica may read the record as unknown until they do",
                    where,
                    self.map_write_wait_seconds,
                )
            return
        if self._map_writes_lagging:
            self._map_writes_lagging = False
            logger.info("vault map writes land in time again; answers wait for them")

    def record_response_id(self, response_id: str, session_id: str) -> None:
        if self.session_router.mode == "static":
            return
        # The router may veto the durable mirror (it refused to move a
        # response into another namespace, or never reads the map): writing
        # it anyway would let the durable map re-home the owner's chain.
        if self.session_router.record_response_id(response_id, session_id) is False:
            return
        self.vault_manager.record_response_session(response_id, session_id)

    def object_tracker(
        self,
        adapter: ProviderAdapter | None,
        method: str,
        path: str,
        headers: "Mapping[str, str] | None" = None,
        body: Any = None,
        query: str = "",
    ) -> ProviderAdapter | None:
        """The adapter whose ``tracks_object_ids`` claims this request — the
        routed one, else (pass-through) the addressed provider's — or None,
        also whenever the router tracks no ownership (no optional
        ``record_object_id``). Asked in EVERY mode: a router in static mode
        may still separate namespaces (llm-redact-pro serves unattributed
        traffic on the static path next to named users, and must know which
        objects that shared session created). ``body`` is the parsed request
        body (a stored chat completion is flagged there)."""
        if self._record_object_id is None:
            return None
        if adapter is not None:
            return adapter if adapter.tracks_object_ids(method, path, body=body) else None
        name = self.provider_for(None, path, headers, query)
        for candidate in self.adapters:
            if candidate.name == name and candidate.tracks_object_ids(method, path, body=body):
                return candidate
        return None

    def record_object_ids(self, object_ids: Sequence[str], session_id: str) -> None:
        """Ids of provider-stored objects (files, batches, conversations)
        created or first seen in ``session_id`` — reported to a router that
        tracks ownership (the optional ``record_object_id``, in any mode),
        mirrored in the durable map unless the router vetoes it: as an owner
        record (``record_object_session``, bounded apart from the Responses
        rows) where the vault manager keeps one."""
        record = self._record_object_id
        if record is None:
            return
        manager = self.vault_manager
        mirror = getattr(manager, "record_object_session", manager.record_response_session)
        for object_id in object_ids:
            if record(object_id, session_id) is not False:
                mirror(object_id, session_id)

    def response_observer(self, context: ResponseContext) -> ResponseObserver | None:
        """The session router's observer for one upstream answer (optional
        ``response_observer``), or None: the router declined, has no such
        member, answered something that cannot be called, or failed — a
        failure contained like the bookkeeping after an answer (counted,
        logged by exception type only)."""
        factory = self._response_observer
        if factory is None:
            return None
        try:
            observer = factory(context)
        except Exception as exc:  # noqa: BLE001 — contained by design
            _bookkeeping_fault(self, _OBSERVER_STAGE, context.method, context.path, exc)
            return None
        return observer if callable(observer) else None

    @property
    def checks_object_access(self) -> bool:
        """Whether the session router checks stored-object access at all
        (the optional ``object_access_refusal``)."""
        return self._object_access_refusal is not None

    def object_access_refusal(
        self, adapter_name: str | None, method: str, path: str, body: Any, *, identity: bool
    ) -> str | None:
        """The session router's refusal of a request that reaches another
        namespace's stored object (optional ``object_access_refusal``), or
        None. A router that raises refuses: an ownership check that cannot
        answer must not wave the request through to the upstream."""
        check = self._object_access_refusal
        if check is None:
            return None
        try:
            verdict = check(adapter_name, method, path, body, identity=identity)
        except Exception as exc:  # noqa: BLE001 — fail closed
            logger.warning(
                "session router object_access_refusal failed (%s); refusing",
                type(exc).__name__,
            )
            return _OBJECT_ACCESS_FAULT
        if verdict is None:
            return None
        # Any other non-None answer refuses; only a non-empty string is
        # the router's own (fixed) reason.
        return verdict if isinstance(verdict, str) and verdict else _OBJECT_ACCESS_FAULT

    @property
    def checks_realtime_frames(self) -> bool:
        """Whether the session router checks realtime client frames at all
        (the optional ``realtime_frame_refusal``)."""
        return self._realtime_frame_refusal is not None

    def realtime_frame_refusal(
        self, adapter_name: str, path: str, frame: Any, *, identity: bool, session_id: str
    ) -> str | None:
        """The session router's refusal of one parsed realtime client frame
        (optional ``realtime_frame_refusal``), or None. A router that raises
        or answers anything but None or a non-empty string refuses — the
        frame check's fault is counted (bookkeeping stage
        ``realtime_frame``) and logged by exception or answer TYPE only: a
        check that cannot answer must not wave the frame through."""
        check = self._realtime_frame_refusal
        if check is None:
            return None
        try:
            verdict = check(adapter_name, path, frame, identity=identity, session_id=session_id)
        except Exception as exc:  # noqa: BLE001 — fail closed
            fault = type(exc).__name__
        else:
            if verdict is None or (isinstance(verdict, str) and verdict):
                return verdict
            fault = f"answered {type(verdict).__name__}"
        self.bookkeeping_errors[REALTIME_FRAME_STAGE] += 1
        logger.warning(
            "WS %s -> session router realtime_frame_refusal failed (%s); closing", path, fault
        )
        return REALTIME_FRAME_FAULT

    @property
    def observes_realtime_server_frames(self) -> bool:
        """Whether the session router observes realtime upstream frames at
        all (the optional ``realtime_server_frame``)."""
        return self._realtime_server_frame is not None

    def realtime_server_frame(
        self, adapter_name: str, path: str, frame: Any, *, identity: bool, session_id: str
    ) -> None:
        """Hand the session router one parsed realtime upstream frame
        (optional ``realtime_server_frame``): read-only, before the frame is
        restored or sent. A router that raises is contained — counted as the
        ``realtime_server_frame`` bookkeeping stage and logged by exception
        TYPE only — and the frame is delivered as usual: what a router fails
        to record is its unknown case (llm-redact-pro then refuses a Live
        session resumption it cannot attribute), never a wrong value."""
        observe = self._realtime_server_frame
        if observe is None:
            return
        try:
            observe(adapter_name, path, frame, identity=identity, session_id=session_id)
        except Exception as exc:  # noqa: BLE001 — contained by design
            self.bookkeeping_errors[REALTIME_SERVER_FRAME_STAGE] += 1
            logger.warning(
                "WS %s -> session router realtime_server_frame failed (%s); frame delivered",
                path,
                type(exc).__name__,
            )

    def object_lister(
        self,
        adapter: ProviderAdapter | None,
        method: str,
        path: str,
        headers: "Mapping[str, str] | None" = None,
        query: str = "",
    ) -> ProviderAdapter | None:
        """The adapter whose ``lists_objects`` claims this request (the
        routed one, else the addressed provider's), or None — always None
        when the session router attributes no listed items."""
        if self._listing_item_session is None and self._listing_item_sessions is None:
            return None
        if adapter is not None:
            return adapter if adapter.lists_objects(method, path) else None
        name = self.provider_for(None, path, headers, query)
        for candidate in self.adapters:
            if candidate.name == name and candidate.lists_objects(method, path):
                return candidate
        return None

    def listing_restorers(self, object_ids: Sequence[str]) -> dict[str, Rehydrator | None]:
        """How each listed object is delivered, for one listing: an id ABSENT
        from the answer stays as this request's own session delivers it (the
        router named no session for it); ``None`` — the item is delivered
        exactly as the provider sent it, because the router's answer failed
        (for the whole listing, or for that item: a router that cannot answer
        vouches for nothing, and the listing's own session may hold other
        values under the same token names), or the named session does not
        exist or holds nothing (it is never created here), or is not the
        session the durable map records the object in (a session pruned
        since the object was created and recreated with new values: its
        placeholders are not the object's — or the check itself failed); a
        rehydrator — restored in the named session."""
        named, unanswered = self._listing_sessions(object_ids)
        restorers: dict[str, Rehydrator | None] = dict.fromkeys(unanswered)
        if not named:
            return restorers
        recorded = self._recorded_sessions(named)
        rehydrators: dict[str, Rehydrator | None] = {}
        for object_id, session_id in named.items():
            if recorded is not None and recorded.get(object_id) != session_id:
                restorers[object_id] = None
                continue
            if session_id not in rehydrators:
                try:
                    rehydrators[session_id] = self._existing_session_rehydrator(session_id)
                except Exception as exc:  # noqa: BLE001 — a vault read that failed
                    # The named owner's session cannot be read: its items go
                    # out as the provider sent them, never what this
                    # request's session would make of them (another value).
                    _listing_fault(self, exc)
                    rehydrators[session_id] = None
            restorers[object_id] = rehydrators[session_id]
        return restorers

    def _listing_sessions(self, object_ids: Sequence[str]) -> tuple[dict[str, str], list[str]]:
        """The session the router names for each listed object — asked once
        for the whole listing (the optional ``listing_item_sessions``), else
        per item (``listing_item_session``) — and the ids it failed to
        answer for: every id when the batched answer raises or miscounts,
        else each id whose own answer raised (each failure counted as a
        ``listing`` bookkeeping fault). Ids it names no session for are in
        neither."""
        batched = self._listing_item_sessions
        if batched is not None:
            try:
                answer = list(batched(list(object_ids)))
            except Exception as exc:  # noqa: BLE001 — unsure means placeholders
                self.bookkeeping_errors["listing"] += 1
                logger.warning(
                    "session router listing_item_sessions failed (%s);"
                    " items delivered as the provider sent them",
                    type(exc).__name__,
                )
                return {}, list(object_ids)
            if len(answer) != len(object_ids):
                self.bookkeeping_errors["listing"] += 1
                logger.warning(
                    "session router listing_item_sessions miscounted;"
                    " items delivered as the provider sent them"
                )
                return {}, list(object_ids)
            pairs = list(zip(object_ids, answer, strict=True))
        else:
            pairs = [
                (object_id, self._listing_item_session_of(object_id)) for object_id in object_ids
            ]
        named = {object_id: session for object_id, session in pairs if isinstance(session, str)}
        return named, [object_id for object_id, session in pairs if session is _UNANSWERED]

    def _listing_item_session_of(self, object_id: str) -> Any:
        """The router's per-item answer (only a string names a session), or
        ``_UNANSWERED`` when asking raised."""
        name_session = self._listing_item_session
        if name_session is None:
            return None
        try:
            return name_session(object_id)
        except Exception as exc:  # noqa: BLE001 — unsure means placeholders
            self.bookkeeping_errors["listing"] += 1
            logger.warning(
                "session router listing_item_session failed (%s);"
                " item delivered as the provider sent it",
                type(exc).__name__,
            )
            return _UNANSWERED

    def _recorded_sessions(self, named: Mapping[str, str]) -> dict[str, str] | None:
        """The durable map's record of each named object (one batched query
        where the vault manager offers one), or None without a durable map
        (the in-memory manager never prunes, so there the router's record is
        the truth). A lookup that fails answers "nothing recorded": every
        named item then keeps the provider's placeholders."""
        manager = self.vault_manager
        if not getattr(manager, "durable_response_map", True):
            return None
        try:
            many = getattr(manager, "lookup_response_sessions", None)
            if many is not None:
                return dict(many(named))
            found = ((object_id, manager.lookup_response_session(object_id)) for object_id in named)
            return {object_id: session for object_id, session in found if session is not None}
        except Exception as exc:  # noqa: BLE001 — unsure means placeholders
            logger.warning(
                "listing ownership check failed (%s); items left as is", type(exc).__name__
            )
            return {}

    def _existing_session_rehydrator(self, session_id: str) -> Rehydrator | None:
        """A rehydrator over ``session_id`` when it exists and holds
        mappings, else None — never creating it."""
        has_session = getattr(self.vault_manager, "has_session", None)
        if has_session is not None and not has_session(session_id):
            return None
        vault = self.vault_manager.get(session_id)
        if len(vault) == 0:
            return None
        return Rehydrator(
            vault, fuzzy=self.config.rehydration.fuzzy, counts=self.rehydration_counts
        )

    def reload(self) -> None:
        """Rebuild hot-swappable config on SIGHUP; never crash a running proxy.

        Hot: detection rules/allowlists/custom rules/NER, rehydration.fuzzy,
        inject_system_note, max_body_bytes, provider upstreams. Requires
        restart (kept with a warning): vault, audit, host, port. An open
        realtime connection whose provider settings, upstream authorizer or
        detection policy changed is closed 1012 (reconnect). The access
        gate's optional ``reload`` then re-reads its own policy.
        """
        try:
            fresh = apply_env_overrides(load_config(self.config_path))
            # apply_config builds before it swaps, so a failure here (unknown
            # rule names, bad custom regex, missing NER extra — all deferred
            # past parse_config) leaves the running state untouched.
            self.apply_config(fresh)
        # ConfigError and tomllib.TOMLDecodeError are both ValueErrors.
        except (ValueError, OSError, re.error, ImportError) as exc:
            logger.error("config reload failed; keeping current config: %s", exc)

    def apply_config(self, fresh: Config) -> list[str]:
        """Swap in the hot-swappable parts of ``fresh``; shared by SIGHUP
        reload and the config editor endpoint.

        Restart-only fields (vault, audit, host, port, log, tls, otel) are
        pinned to their running values; the names of any that differed are
        returned (and warned) so callers can surface "restart required".
        """
        restart_required = [
            field_name
            for field_name in RESTART_ONLY_KEYS
            if getattr(fresh, field_name) != getattr(self.config, field_name)
        ]
        # Plugin-owned sections (plugin_api.ConfigSection) are restart-only
        # too, reported by their own section names.
        restart_required += sorted(
            name
            for name in set(fresh.extensions) | set(self.config.extensions)
            if fresh.extensions.get(name) != self.config.extensions.get(name)
        )
        for field_name in restart_required:
            logger.warning(
                "config reload: [%s] changes require restart; keeping current", field_name
            )
        effective = dataclasses.replace(
            fresh,
            extensions=self.config.extensions,
            **{name: getattr(self.config, name) for name in RESTART_ONLY_KEYS},
        )

        # Re-resolve the license BEFORE anything is built or swapped:
        # [license] itself is hot, so renewals apply without a restart.
        license_resolved = _resolve_license_info(effective)

        # Build everything first, then swap in one block: in-flight requests
        # keep their old object references.
        if effective.detection == self.config.detection:
            detectors = self.detectors
            allowlist = self.allowlist
            modes = self.modes
            overlay_builds = self.overlay_builds
        else:
            detectors = build_detectors(effective.detection)
            allowlist = build_allowlist(effective.detection)
            modes = build_modes(effective.detection)
            # Every cached overlay build dies with the objects it extended.
            overlay_builds = OverlayBuilds(effective.detection, modes)
        redactor = Redactor(
            detectors,
            self.vault,
            allowlist,
            counts=self.detection_counts,
            modes=modes,
            warn_counts=self.warn_counts,
        )
        rehydrator = Rehydrator(
            self.vault, fuzzy=effective.rehydration.fuzzy, counts=self.rehydration_counts
        )

        adapters = self.adapters
        if set(effective.providers) != set(self.config.providers):
            # Custom upstreams appeared/vanished: rebuild the adapter list
            # (in-flight requests keep their old adapter references). Built
            # into a local and swapped with everything else: the router
            # build below can raise, and a half-applied reload (new adapters
            # under the old config) would leave a still-configured custom
            # upstream with no adapter, i.e. forwarded unredacted.
            adapters = [cls() for cls in ALL_ADAPTERS] + build_custom_adapters(effective.providers)
        # Upstream authorizers are hot: rebuilt only when a provider's auth
        # settings changed (a rebuild drops cached cloud credentials), BEFORE
        # the router reconfigures itself in place — a refusal here (package
        # absent, no region) closes what it built and changes nothing. The
        # displaced ones close at swap time; in-flight requests keep the
        # authorizer they already hold.
        upstream_auth = self.upstream_auth
        if _auth_settings(effective.providers) != _auth_settings(self.config.providers):
            upstream_auth = _build_upstream_auths(effective.providers)
        # Routing is hot (decision 16 / R-34). The router validates-then-swaps
        # its own state; it raises only ConfigError and changes nothing when
        # it does, so this sits after every other build and before the swap.
        # A reload that turns routing ON builds through the registry (where
        # the tier is checked — like the access gate, a running router is never
        # re-gated); one that turns it OFF drops it (closed at swap time).
        router = self.router
        try:
            if effective.routing.enabled:
                if router is None:
                    router = get_registry().build_router(effective, license_resolved.tier)
                else:
                    router.reconfigure(effective)
            else:
                router = None
            dashboard = self.dashboard
            if license_resolved.tier != self.license.tier:
                dashboard = get_registry().build_dashboard(license_resolved.tier)
        except BaseException:
            if upstream_auth is not self.upstream_auth:
                _close_upstream_auths(upstream_auth)
            raise
        self.config = effective
        self.license = license_resolved
        self.adapters = adapters
        self.detectors = detectors
        self.allowlist = allowlist
        self.modes = modes
        self.overlay_builds = overlay_builds
        self.redactor = redactor
        self.rehydrator = rehydrator
        self._static_context = RequestContext(
            effective.vault.session, self.vault, redactor, rehydrator
        )
        displaced_router = self.router if router is not self.router else None
        self.router = router
        self.dashboard = dashboard
        displaced_auths = self.upstream_auth if upstream_auth is not self.upstream_auth else {}
        self.upstream_auth = upstream_auth
        # Open realtime relays: revoked in this same synchronous step, before
        # anything displaced is closed — a relay whose admission the swap
        # changed forwards no frame it reads from here on, and closes 1012.
        self._revoke_stale_relays()
        # The access gate's own policy (its optional reload: the files its
        # restart-only section names), re-read against the configuration
        # now live — on every apply, even one that changed nothing. Never
        # raises: a gate that keeps its previous policy, or fails, is
        # logged and counted (bookkeeping stage gate_reload).
        self.authorization.reload_policy(effective)
        if displaced_router is not None:
            displaced_router.close()
        _close_upstream_auths(displaced_auths)
        for warning in effective.routing.warnings:
            logger.warning("routing: %s", warning)
        logger.info("config reloaded (%d detection rules)", len(detectors))
        return restart_required

    def _revoke_stale_relays(self) -> None:
        """Revoke every open realtime relay whose admission — its provider's
        settings, its upstream authorizer, the detection policy — the live
        configuration no longer grants (``RealtimeRelay.stale``). A relay the
        reload did not touch keeps running."""
        for relay in list(self.realtime_relays):
            changed = relay.stale(self)
            if changed is not None:
                relay.revoke(changed)

    # --- DashboardHost (plugin_api): what the pro dashboard may use --------

    def config_file_path(self) -> Path:
        """The file a dashboard config edit is written to."""
        if self.config_path is not None:
            return self.config_path
        return resolve_config_path() or default_config_path()

    def host_allowed(self, request: Request) -> bool:
        return _host_allowed(request, self)

    def origin_allowed(self, request: Request) -> bool:
        return _origin_allowed(request, self)

    async def guarded_post_json(self, request: Request) -> tuple[Any, None] | tuple[None, Response]:
        return await _guarded_post_json(request, self)

    def validate_config(self, candidate: Config) -> None:
        """Dry-run everything apply_config would build for ``candidate``
        (the file-level config, env overrides NOT yet applied), changing
        nothing: detectors/allowlists/modes (bad regexes, unknown rules,
        missing NER extras), license resolution (a bad [license] value),
        the routing credentials (an ``env:VAR`` that vanished), and the
        router's own checks — or, when the file enabled routing since
        startup, a build-and-close probe so a refusal (package absent,
        Free tier, bad price file) surfaces here rather than after a write.
        """
        build_detectors(candidate.detection)
        build_allowlist(candidate.detection)
        build_modes(candidate.detection)
        effective = apply_env_overrides(candidate)
        resolved = _resolve_license_info(effective)
        resolve_credentials(candidate.routing, os.environ)
        if self.router is not None:
            self.router.validate(candidate)
        elif candidate.routing.enabled:
            probe = get_registry().build_router(effective, resolved.tier)
            if probe is not None:
                probe.close()
        # Upstream authorizers: a build-and-close probe (construction does no
        # network I/O), so a missing package or an unresolvable region is a
        # 400 in the editor rather than a failed apply after the write.
        _close_upstream_auths(_build_upstream_auths(effective.providers))
        # The access gate's own dry run of its policy (its optional
        # validate_reload): a refusal is a ConfigError like any above.
        self.authorization.validate_policy(candidate)

    def preview(self, text: str) -> dict[str, Any]:
        """Run the LIVE detectors/allowlist/modes over ``text`` on a
        throwaway vault with fresh counters: no upstream request, and the
        live vault, metrics and audit are never touched. Warn-mode values
        stay in ``redacted`` — a real request forwards them (honest); a
        block-mode match reports its type only, never the value."""
        from llm_redact.vault import InMemoryVault

        redactor = Redactor(self.detectors, InMemoryVault(), self.allowlist, modes=self.modes)
        blocked: dict[str, str] | None = None
        redacted: str | None = None
        try:
            redacted = redactor.redact_text(text)
        except BlockedRequest as exc:
            blocked = {"type": exc.detector_type}
        return {
            "redacted": redacted,
            "detections": dict(redactor.counts),
            "warnings": dict(redactor.warn_counts),
            "blocked": blocked,
        }

    def refresh_license_warnings(self, *, today: date | None = None) -> None:
        """Re-resolve the license for WARNING purposes only.

        A long-running proxy never re-reads its license, so the T-30
        pre-expiry warning (and grace entry) would stay invisible until a
        restart. This daily refresh updates ``self.license.warnings`` and
        ``in_grace`` — it NEVER changes the enforced tier: a renewal that
        lands late must not take a running proxy down, and a lapse must not
        silently disable configured features mid-flight (restart enforces).
        A renewed key_file dropped on disk clears the warning on the next
        tick the same way.
        """
        fresh = resolve_license(
            env=dict(os.environ),
            config_key=self.config.license.key,
            config_key_file=self.config.license.key_file,
            today=today,
        )
        current = self.license
        if fresh.tier == current.tier:
            updated = dataclasses.replace(current, warnings=fresh.warnings, in_grace=fresh.in_grace)
        else:
            detail = "; ".join(fresh.warnings) or "renew or update the license key"
            updated = dataclasses.replace(
                current,
                warnings=(
                    f"license now resolves to the {fresh.tier} tier but the"
                    f" {current.tier} tier stays enforced until restart — {detail}",
                ),
            )
        for warning in set(updated.warnings) - set(current.warnings):
            logger.warning("license: %s", warning)
        self.license = updated

    @staticmethod
    def _audit_entry(
        *,
        session: str,
        provider: str | None,
        method: str,
        path: str,
        detections: dict[str, int],
        warned: dict[str, int] | None,
        status: int | None = None,
        duration_ms: float = 0.0,
        streamed: bool = False,
        rehydrations: dict[str, int] | None = None,
        override: str | None = None,
    ) -> AuditRecord:
        """The one construction site for audit rows: START rows take the
        defaults, END/classic rows override them — a future AuditRecord
        field is threaded here once, never per call site."""
        return AuditRecord(
            ts=datetime.now(tz=UTC).isoformat(timespec="seconds"),
            session=session,
            provider=provider,
            method=method,
            path=path,
            status=status,
            duration_ms=duration_ms,
            streamed=streamed,
            detections=detections,
            rehydrations=rehydrations or {},
            user=_REQUEST_USER.get(),
            warned=warned or None,
            override=override,
        )

    def begin_audit(
        self,
        *,
        session: str,
        provider: str | None,
        method: str,
        path: str,
        detections: dict[str, int],
        warned: dict[str, int] | None = None,
    ) -> object | None:
        """Write-ahead audit intent ([audit] required): durably commit a
        START row BEFORE any upstream contact.

        Returns the token ``record_request`` finalizes with, or None when
        required mode is off (the fail-open hot path is unchanged). Raises
        :class:`AuditWriteError` when the row cannot be committed — the
        caller answers a provider-shaped 503 without touching the upstream.
        """
        if self.write_ahead_audit is None:
            return None
        token = self.write_ahead_audit.begin(
            self._audit_entry(
                session=session,
                provider=provider,
                method=method,
                path=path,
                detections=detections,
                warned=warned,
                override=_COMMITTED_OVERRIDE.get(),
            )
        )
        # An awaitable token (an ``async def begin``) is a START row not yet
        # committed — the core never awaits it on the request path.
        _synchronous_audit_answer(token, "begin")
        if token is None:
            # None is the "required mode off" sentinel record_request
            # dispatches on; a write-ahead log must never mint it. Fail
            # closed like any other begin-side fault rather than silently
            # orphaning the START row behind a record()-routed END.
            raise AuditWriteError("write-ahead audit begin() returned no token")
        return token

    def amend_audit(
        self,
        token: object,
        *,
        session: str,
        provider: str | None,
        method: str,
        path: str,
        detections: dict[str, int],
        warned: dict[str, int] | None,
        duration_ms: float,
    ) -> None:
        """Durably amend the write-ahead START row ``token`` names with the
        request's counts (and the override it is about to use), BEFORE any
        upstream contact, through the log's optional ``amend`` member: the
        request keeps one START row. Asked only when the log has the member
        (``write_ahead_amend``); raises :class:`AuditWriteError` when the
        amendment cannot be committed — the caller refuses 503 like a START
        row that cannot be committed. The member is synchronous: one that
        answers an awaitable (an ``async def``) has made nothing durable yet
        and the core never awaits it on the request path, so that answer is
        closed unrun and treated as an amendment that cannot be committed —
        never as a durable one."""
        amend = self.write_ahead_amend
        assert amend is not None  # asked only when the log has the member
        answer: object = amend(
            token,
            self._audit_entry(
                session=session,
                provider=provider,
                method=method,
                path=path,
                detections=detections,
                warned=warned,
                duration_ms=duration_ms,
                override=_COMMITTED_OVERRIDE.get(),
            ),
        )
        _synchronous_audit_answer(answer, "amend")

    def record_request(
        self,
        *,
        session: str,
        provider: str | None,
        method: str,
        path: str,
        status: int | None,
        started: float,
        streamed: bool,
        detections: dict[str, int],
        rehydrations: dict[str, int],
        warned: dict[str, int] | None = None,
        audit_token: object | None = None,
        route: dict[str, Any] | None = None,
        override: str | None = None,
        refusal: LocalRefusal | None = None,
    ) -> None:
        """Always update in-memory metrics and the recent buffer; write an
        audit row when enabled (finalizing the write-ahead START row when
        ``begin_audit`` issued a token for this request — or an upload wrote
        its START row before its inspection: ``_EarlyAudit``). ``override`` (else
        the request's own, _REQUEST_OVERRIDE) marks a request that passed a
        refusal on an approved override: "once" or "always". ``refusal``: the
        response is the proxy's own instead of the upstream's — counted once
        under that kind (``llm_redact_local_refusals_total``); every call
        site states it, None included (pinned by test)."""
        override = override or _REQUEST_OVERRIDE.get()
        if audit_token is None:
            early = _EARLY_AUDIT.get()
            if early is not None:
                audit_token = early.take()
        duration_seconds = time.perf_counter() - started
        self.metrics.observe_request(provider, status, duration_seconds, streamed)
        if refusal is not None:
            self.metrics.count_local_refusal(refusal, provider)
        timing = _REQUEST_TIMING.get()
        if timing is not None and method != "WS":
            self.metrics.observe_overhead(provider, duration_seconds - timing.waited)
        row = {
            "ts": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "session": session,
            "provider": provider,
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": duration_seconds * 1000.0,
            "streamed": streamed,
            "detections": detections,
            "rehydrations": rehydrations,
            # Warn-mode hits attributed to THIS request (types+counts): these
            # values were FORWARDED upstream — the one number an operator
            # auditing "did my key leak?" needs per-request, not aggregated.
            "warned": warned or {},
            # Attribution is the user NAME only — the key never leaves
            # identity extraction. None on single-user deployments.
            "user": _REQUEST_USER.get(),
            # Routing decision (the llm-redact-pro routing layer): None when
            # the request took the legacy path, else the router's row —
            # rule/upstream/hops/auth/class/reissue. Audit rows are unchanged
            # (AuditRecord is shared with pro).
            "route": route,
            # A refusal this request passed on its requester's approved
            # override (overrides.py): "once" | "always" | None. What passed
            # was FORWARDED as sent (a value, or a body/part unscanned).
            "override": override,
        }
        self.recent.append(row)
        for queue in list(self.event_subscribers):
            # A full queue means a slow consumer: drop the event for that
            # subscriber (its poll fallback self-heals) rather than block.
            with suppress(asyncio.QueueFull):
                queue.put_nowait(row)
        if self.telemetry is not None:
            self.telemetry.record(row, duration_seconds, traceparent=_INBOUND_TRACEPARENT.get())
        if self.audit_s3 is not None:
            # Same row, same metadata-only contract; buffered and shipped
            # in batches by the sink's flush loop.
            self.audit_s3.add(row)
        if self.audit_azure is not None:
            self.audit_azure.add(row)
        if self.audit is None:
            return
        entry = self._audit_entry(
            session=session,
            provider=provider,
            method=method,
            path=path,
            detections=detections,
            warned=warned,
            status=status,
            duration_ms=duration_seconds * 1000.0,
            streamed=streamed,
            rehydrations=rehydrations,
            override=override,
        )
        try:
            # A token exists only when begin_audit resolved the write-ahead
            # log (and it never mints None) — the conjunct narrows the type.
            if audit_token is not None and self.write_ahead_audit is not None:
                _finalize_audit(self.write_ahead_audit, audit_token, entry)
            else:
                self.audit.record(entry)
        except AuditWriteError as exc:
            # Required mode, END-side write fault: the response is already
            # committed to the client, so refusal is impossible — the START
            # row (or, for proxy-local replies, nothing) is what survives.
            # Loud by design; type only, never row contents.
            logger.critical(
                "audit write failed AFTER response (%s %s): %s", method, path, type(exc).__name__
            )

    def count_local_refusal(self, kind: LocalRefusal, provider: str | None) -> None:
        """A refusal the proxy answers WITHOUT recording a row (its path may
        hold a key, or there is no request to attribute): counted alone,
        exactly once (``record_request(refusal=)`` counts every other)."""
        self.metrics.count_local_refusal(kind, provider)

    async def plugin_metric_text(self) -> str:
        """The access gate's own gauges (its OPTIONAL ``metrics_samples``),
        as exposition lines under the core's rules (``plugin_metric_lines``):
        asked in a worker thread, at most ``PLUGIN_METRICS_TIMEOUT_SECONDS``
        — a slow or hung plugin never blocks the event loop or the scrape,
        and a scrape never starts a second call while one is still running:
        a concurrent scrape shares the call in flight, waiting for it only
        until that call's own deadline (two scrapes at once — an HA pair of
        Prometheus servers — both get the gauges and count nothing); a call
        still running past its deadline (an earlier scrape gave up on it)
        is not waited for again. A fault, a timeout, a call still running
        past its bound, an answer that is not iterable and every dropped
        sample count under the bookkeeping stage ``plugin_metrics`` — once
        per scrape rendered without them (logged once per episode, by
        exception TYPE only); the core's own metrics are rendered either
        way."""
        samples = getattr(self.access_gate, "metrics_samples", None)
        if not callable(samples):
            return ""
        now = asyncio.get_running_loop().time()
        call = self._plugin_metrics_call
        if call is not None and not call.done():
            if now >= self._plugin_metrics_deadline:
                self._plugin_metrics_fault("still running")
                return ""
        else:
            call = asyncio.ensure_future(asyncio.to_thread(_listed_samples, samples))
            # A call the scrape stopped waiting for still ends: its outcome is
            # retrieved then (never an unretrieved exception), and discarded.
            call.add_done_callback(_retrieve)
            self._plugin_metrics_call = call
            self._plugin_metrics_deadline = now + PLUGIN_METRICS_TIMEOUT_SECONDS
        try:
            answer = await asyncio.wait_for(
                asyncio.shield(call), self._plugin_metrics_deadline - now
            )
            lines, dropped = plugin_metric_lines(answer)
        except TimeoutError:
            self._plugin_metrics_fault("TimeoutError")
            return ""
        except Exception as exc:  # noqa: BLE001 — a plugin's fault never fails the scrape
            self._plugin_metrics_fault(type(exc).__name__)
            return ""
        if dropped:
            self.bookkeeping_errors[PLUGIN_METRICS_STAGE] += dropped
        if self._plugin_metrics_failing:
            self._plugin_metrics_failing = False
            logger.info("the access gate's metrics samples are read again")
        return "".join(line + "\n" for line in lines)

    def _plugin_metrics_fault(self, kind: str) -> None:
        self.bookkeeping_errors[PLUGIN_METRICS_STAGE] += 1
        if not self._plugin_metrics_failing:
            self._plugin_metrics_failing = True
            logger.warning("the access gate's metrics samples were not read (%s)", kind)

    def finish_route(self, delivery: RouteDelivery, status: int | None) -> dict[str, Any]:
        """Close the books on a routed request: the routed-requests metric
        (every routed outcome, 502s included — decision 19) and the router's
        own finish (spend), returning the `route` row record_request stores."""
        self.metrics.routed[(delivery.upstream, delivery.rule or "-")] += 1
        return delivery.finish(status)

    def route(
        self,
        method: str,
        path: str,
        headers: "Mapping[str, str] | None" = None,
        query: str = "",
    ) -> tuple[ProviderAdapter | None, RouteKind]:
        for adapter in self.adapters:
            kind = adapter.matches_request(method, path, headers, query)
            if kind is not RouteKind.NONE:
                return adapter, kind
        return None, RouteKind.NONE

    def provider_for(
        self,
        adapter: ProviderAdapter | None,
        path: str,
        headers: "Mapping[str, str] | None" = None,
        query: str = "",
    ) -> str | None:
        """The provider a request is addressed to: its matched adapter's, or
        — pass-through — the provider it is POSITIVELY attributed to by its
        path family or one provider's markers
        (``providers.attribution.attribute``). None when it cannot be
        attributed: such a request is answered locally, never forwarded to
        a guessed provider (the old anthropic default sent an OpenAI key
        and prompt to api.anthropic.com)."""
        if adapter is not None and adapter.name in self.config.providers:
            return adapter.name
        return attribute(path, headers, query)


def _request_headers(request: Request, *, matched: bool) -> list[tuple[str, str]]:
    """The client's headers as forwarded: hop-by-hop ones and the proxy's own
    namespace dropped — and, on a ``matched`` route, every method override
    (refused before this; dropped again here as a second layer)."""
    headers = [
        (name, value)
        for name, value in request.headers.items()
        if name.lower() not in _SKIP_REQUEST_HEADERS
        and not name.lower().startswith(OWN_HEADER_PREFIX)
        and not (matched and name.lower() in METHOD_OVERRIDE_HEADERS)
    ]
    # Compressed upstream bodies would force re-encoding bookkeeping on the
    # streaming path; identity keeps the byte stream directly rewritable.
    headers.append(("accept-encoding", "identity"))
    return headers


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    return {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() not in _SKIP_RESPONSE_HEADERS
    }


def _auth_settings(
    providers: Mapping[str, ProviderConfig],
) -> dict[str, tuple[str, str | None, str]]:
    """What an upstream authorizer is built from, per provider: a reload
    rebuilds the authorizers only when this changes (a rebuild drops the
    cached cloud credentials)."""
    return {
        name: (provider.auth, provider.region, provider.upstream_base_url)
        for name, provider in providers.items()
        if provider.auth != "passthrough"
    }


def _build_upstream_auths(providers: Mapping[str, ProviderConfig]) -> dict[str, UpstreamAuth]:
    """One registry-built authorizer per provider that wants one. A failure
    part-way closes what was already built and re-raises (ConfigError)."""
    registry = get_registry()
    built: dict[str, UpstreamAuth] = {}
    try:
        for name, provider in sorted(providers.items()):
            problem = (
                identity_upstream_problem(provider.upstream_base_url)
                if provider.auth != "passthrough"
                else None
            )
            if problem is not None:
                # Also enforced at parse time; this covers configs built in
                # code (and anything that bypassed the parser).
                raise ConfigError(f"[providers.{name}] {problem}")
            auth = registry.build_upstream_auth(name, provider)
            if auth is not None:
                built[name] = auth
            elif provider.auth != "passthrough":
                # A plugin that answers None for an identity provider would
                # otherwise forward the client's own credential in its place.
                raise ConfigError(
                    f'[providers.{name}] auth = "{provider.auth}" but the installed'
                    " llm-redact-pro built no authorizer for it; upgrade llm-redact-pro"
                )
    except BaseException:
        _close_upstream_auths(built)
        raise
    return built


def identity_exposure_warning(state: ProxyState, host: str) -> str | None:
    """The startup warning for a proxy that lends its own cloud identity to
    whoever can reach it: identity-authorized providers on a non-loopback
    bind with no access gate to say who the clients are. (Transport mTLS
    still admits every certificate its CA signed.) None when not exposed."""
    if not state.upstream_auth or state.access_gate is not None or _is_loopback_host(host):
        return None
    names = ", ".join(sorted(state.upstream_auth))
    return (
        f"providers {names} use the proxy's own cloud identity and this proxy binds"
        f" {host} with no access gate: every client that can connect spends that"
        " identity; install llm-redact-pro access control or bind 127.0.0.1"
    )


def _sink_close_outcome(task: asyncio.Future[None]) -> None:
    """Retrieve one audit sink's final-flush outcome: a fault is logged by
    exception TYPE only (its message may quote a URL or a SAS), never
    raised — the rest of the shutdown must run."""
    if task.cancelled():
        return
    problem = task.exception()
    if problem is not None:
        logger.warning("an audit sink's final flush failed (%s)", type(problem).__name__)


async def _close_audit_sinks(sinks: Sequence[S3AuditSink | AzureAuditSink], timeout: float) -> None:
    """The sinks' final flush at shutdown, bounded (``_SINK_CLOSE_TIMEOUT_SECONDS``).

    Runs every sink's ``aclose()`` concurrently while the audit database is
    still open, so the END rows the last requests committed reach the
    off-machine copy. A flush still running at the deadline is cancelled
    (its spooled rows stay unshipped in the database for the next start); one
    that ignores the cancellation for ``_SINK_CANCEL_GRACE_SECONDS`` is
    abandoned — its outcome retrieved whenever it ends — and the shutdown
    goes on to close the audit database. Abandoning bounds those closes,
    not the process exit: the event loop's teardown still awaits a task
    that keeps ignoring its cancellation (the supervisor's kill ends it)."""
    if not sinks:
        return
    tasks = {asyncio.ensure_future(sink.aclose()) for sink in sinks}
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    for task in done:
        _sink_close_outcome(task)
    if not pending:
        return
    logger.warning(
        "%d audit sink(s) did not finish the final flush within %g s; cancelled"
        " (unshipped spooled rows upload at the next start)",
        len(pending),
        timeout,
    )
    for task in pending:
        task.cancel()
    unwound, stuck = await asyncio.wait(pending, timeout=_SINK_CANCEL_GRACE_SECONDS)
    for task in unwound:
        _sink_close_outcome(task)
    for task in stuck:
        task.add_done_callback(_sink_close_outcome)
    if stuck:
        logger.warning("%d audit sink(s) ignored the cancellation; abandoned", len(stuck))


def _close_contained(what: str, close: Callable[[], object]) -> None:
    """Close one plugin-supplied object at shutdown: a fault is logged by
    exception TYPE only (a plugin's message may quote a URL or a credential)
    and never stops the rest of the shutdown — the sinks' final flush and
    the audit database and vault closes come after it."""
    try:
        close()
    except Exception as problem:
        logger.warning("closing the %s failed (%s)", what, type(problem).__name__)


def _close_upstream_auths(auths: Mapping[str, UpstreamAuth]) -> None:
    for name, auth in auths.items():
        try:
            auth.close()
        except Exception as exc:  # a close fault must not break a reload/shutdown
            logger.warning("upstream auth for %s failed to close (%s)", name, type(exc).__name__)


# Client credential channels removed before the proxy authorizes a request
# with its own cloud identity: the provider must see only the proxy's
# credential, and a client's key or signature must never ride along (a
# client SigV4 signature would also be wrong: it covers the unredacted body).
# At least the access gate's brokered set (authorization, x-api-key,
# x-goog-api-key, api-key, ?key=), widened to every header naming an API key
# or an authorization, AWS signing headers, cookies, and an API-management
# subscription key. Google's quota/billing-project and legacy IAM selector
# headers go too: under the proxy's identity they would let a client bill
# (or act in) a project the operator never chose.
_CREDENTIAL_HEADERS = frozenset(
    {
        "cookie",
        "password",
        "passwd",
        "ocp-apim-subscription-key",
        "x-goog-user-project",
        "x-goog-quota-user",
        "x-goog-iam-authority-selector",
        "x-goog-iam-authorization-token",
    }
)
# Query twins, compared case-insensitively with any leading `$` dropped
# (Google's system parameters accept `$key`, `$userProject`), plus APIM's
# `subscription-key`, OAuth 1/legacy `oauth_token`, and password params.
_CREDENTIAL_QUERY_PARAMS = frozenset(
    {
        "key",
        "api-key",
        "api_key",
        "access_token",
        "oauth_token",
        "userproject",
        "quotauser",
        "subscription-key",
        "password",
        "passwd",
    }
)


def _is_credential_header(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in _CREDENTIAL_HEADERS
        or "authorization" in lowered
        or lowered.endswith("api-key")
        or lowered.startswith("x-amz-")
    )


def _strip_credential_query(query: str) -> str:
    """The raw query string minus every credential parameter (``key=`` and
    ``$key=``, ``api-key=``, ``access_token=``, ``userProject=``,
    ``quotaUser=``, ``subscription-key=``, …, presigned-URL ``X-Amz-*``, and
    any ``*authorization*`` parameter — a WebSocket client that cannot set
    headers may carry ``Authorization=Bearer …`` in the query, a channel
    openai-node's Azure Realtime client recognizes); the rest is kept
    byte-for-byte in order."""
    kept = []
    for part in query.split("&"):
        name = urllib.parse.unquote_plus(part.split("=", 1)[0]).lower().lstrip("$")
        if name in _CREDENTIAL_QUERY_PARAMS or "authorization" in name or name.startswith("x-amz-"):
            continue
        kept.append(part)
    return "&".join(kept)


def strip_client_credentials(
    url: str, headers: Sequence[tuple[str, str]]
) -> tuple[str, list[tuple[str, str]]]:
    """``url`` in its on-the-wire form without credential query parameters,
    and ``headers`` without any client credential channel."""
    base, _, query = url.partition("?")
    stripped = _strip_credential_query(query) if query else ""
    final = str(httpx.URL(base + ("?" + stripped if stripped else "")))
    return final, [(name, value) for name, value in headers if not _is_credential_header(name)]


class RequestMeta(NamedTuple):
    """Per-request context handle() threads into the streaming finalizers,
    which outlive the HTTP handler and call record_request at stream end."""

    method: str
    path: str
    started: float
    detections: dict[str, int]
    warned: dict[str, int]
    audit_token: object | None = None


def _route_log_suffix(row: Mapping[str, Any]) -> str:
    """The R-29 log fields for a routed request (unrouted lines never carry
    them). Fixed keys from the router's row: names, ids, modes and classes
    only — the core prints nothing else the router hands it."""
    return (
        f" rule={row.get('rule') or '-'} upstream={row.get('upstream') or '-'}"
        f" hops={row.get('hops', 0)} auth={row.get('auth', '-')}"
        f" class={row.get('class', '-')} reissue={row.get('reissue', '-')}"
    )


async def _stream_rehydrated(
    upstream: httpx.Response,
    adapter: ProviderAdapter,
    state: ProxyState,
    ctx: RequestContext,
    *,
    request_meta: RequestMeta,
    route: RouteDelivery | None = None,
    object_tracker: ProviderAdapter | None = None,
    request_body: Any = None,
    observe: ResponseObserver | None = None,
) -> AsyncIterator[bytes]:
    method, path, started, detections, warned, audit_token = request_meta
    parser = SSEParser()
    pool = RehydratorPool(ctx.vault, fuzzy=state.config.rehydration.fuzzy)
    response_id_seen = False
    objects = (
        _StreamedObjects(object_tracker, method, path, request_body)
        if object_tracker is not None
        else None
    )
    status = upstream.status_code  # 502 when the proxy itself cut the stream
    cut = False  # whether it did (a delivery_fault local refusal)
    try:
        async for chunk in _UpstreamPaced(upstream.aiter_bytes()):
            for event in parser.feed(chunk):
                with state.map_write_barrier() as map_writes:
                    if not response_id_seen:
                        response_id = adapter.response_id_from_event(event)
                        if response_id is not None:
                            # Session bookkeeping never cuts the stream
                            # (contained, counted); the first id suffices
                            # either way.
                            _contained(
                                state,
                                "response_id",
                                method,
                                path,
                                state.record_response_id,
                                response_id,
                                ctx.session_id,
                            )
                            response_id_seen = True
                    if objects is not None and objects.report(state, ctx, event):
                        objects = None  # nothing more to read from this stream
                    if observe is not None:
                        observe = _observe(state, observe, method, path, event.data)
                if map_writes:
                    # map_writes = "before_answer": the event naming the id
                    # waits until every replica can read its record.
                    await state.await_map_writes(map_writes, f"{method} {path}")
                for out in adapter.rehydrate_event(event, pool):
                    if route is not None:
                        out = route.observe_event(out)
                    yield serialize(out)
        for event in parser.close():
            with state.map_write_barrier() as map_writes:
                if objects is not None:
                    objects.report(state, ctx, event)
                if observe is not None:
                    observe = _observe(state, observe, method, path, event.data)
            if map_writes:
                await state.await_map_writes(map_writes, f"{method} {path}")
            for out in adapter.rehydrate_event(event, pool):
                if route is not None:
                    out = route.observe_event(out)
                yield serialize(out)
        # Anything still held back at stream end is emitted as raw text of a
        # final comment-free flush; adapters normally leave nothing here.
        for _key, text in pool.flush_all().items():
            logger.warning("unflushed stream leftover discarded (%d chars)", len(text))
    except httpx.TransportError as exc:
        if route is not None:
            # A fault AFTER the first byte reached the client: propagated
            # as-is (never re-issued — R-17), classified stream_error. Type only.
            route.mark_failed("stream_error")
            logger.warning(
                "%s %s stream failed after first byte (%s)%s",
                method,
                path,
                type(exc).__name__,
                _route_log_suffix(route.row()),
            )
        raise
    except Exception as exc:
        status = _stream_delivery_fault(state, method, path, exc)
        cut = True
        if route is not None:
            route.mark_failed("stream_error")
        raise
    finally:
        with suppress(Exception):
            # A stream that errored mid-body can make aclose() itself raise;
            # that must never skip the finalization below.
            await upstream.aclose()
        # Streamed requests finalize their totals and audit row here — the
        # HTTP handler returned long before the stream ended.
        state.rehydration_counts.update(pool.counts)
        state.record_request(
            session=ctx.session_id,
            provider=adapter.name,
            method=method,
            path=path,
            status=status,
            started=started,
            streamed=True,
            detections=detections,
            rehydrations=dict(pool.counts),
            warned=warned,
            audit_token=audit_token,
            refusal="delivery_fault" if cut else None,
            route=(state.finish_route(route, status) if route is not None else None),
        )


class _StreamedObjects:
    """The stored-object ids a tracked stream names, reported to the session
    router as they appear — each id once, and never one the request itself
    cites (``_uncited``). A stream that names all its objects on one event
    (``reports_object_ids_once``: a stored chat completion's id rides every
    chunk) is read until the first event naming some; any other (the files
    a tool run wrote, named on the events that carry them) to its end."""

    def __init__(self, tracker: ProviderAdapter, method: str, path: str, request_body: Any) -> None:
        self._tracker = tracker
        self._method = method
        self._path = path
        self._request_body = request_body
        self._once = tracker.reports_object_ids_once(method, path)
        self._seen: set[str] = set()

    def report(self, state: ProxyState, ctx: RequestContext, event: SSEEvent) -> bool:
        """Report what ``event`` newly names — contained, like the buffered
        report: the stream goes on either way. True when the stream needs no
        further reading (its objects are named, or reading them failed)."""
        named = False

        def record() -> None:
            nonlocal named
            found = self._tracker.object_ids_from_event(self._method, self._path, event)
            fresh = [object_id for object_id in found if object_id not in self._seen]
            if not fresh:
                return
            named = True
            self._seen.update(fresh)
            created = _uncited(fresh, self._request_body)
            if created:
                state.record_object_ids(created, ctx.session_id)

        recorded = _contained(state, "object_ids", self._method, self._path, record)
        return not recorded or (named and self._once)


def _uncited(object_ids: Sequence[str], request_body: Any) -> list[str]:
    """``object_ids`` without any the request body carries as a string: an
    id the request itself cites (a file it attached or uploaded into a
    container, an earlier answer's citation it resends) names an EXISTING
    object the answer merely echoes — never one this request created, so it
    is never reported as the requester's. Read only when an answer names
    objects; the walk stops once every id is found."""
    wanted = set(object_ids)
    cited: set[str] = set()
    stack: list[Any] = [request_body]
    while stack and len(cited) < len(wanted):
        node = stack.pop()
        if isinstance(node, str):
            if node in wanted:
                cited.add(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return [object_id for object_id in object_ids if object_id not in cited]


def _stream_delivery_fault(state: ProxyState, method: str, path: str, exc: Exception) -> int:
    """A fault restoring a stream after its status went out (a vault read, a
    router hook): the stream is cut — the honest signal, as for an upstream
    drop — and the fault counted and logged by exception TYPE. Returns the
    status the finally block records (and the audit END carries): 502, the
    proxy's failure — a stream the proxy itself cut is never booked as the
    upstream's success."""
    state.bookkeeping_errors["delivery"] += 1
    logger.warning(
        "%s %s stream cut: restoring the upstream answer failed (%s)",
        method,
        path,
        type(exc).__name__,
    )
    return 502


async def _stream_rehydrated_eventstream(
    upstream: httpx.Response,
    adapter: ProviderAdapter,
    state: ProxyState,
    ctx: RequestContext,
    *,
    request_meta: RequestMeta,
    observe: ResponseObserver | None = None,
) -> AsyncIterator[bytes]:
    """The binary-framing twin of _stream_rehydrated (Bedrock streams).

    Any framing violation degrades to verbatim pass-through of every byte
    not yet returned as a parsed frame, then of the rest of the stream:
    forwarding unrestored placeholders is safe; guessing at corrupted
    frames is not. Error messages carry lengths and counts, never values.
    """
    method, path, started, detections, warned, audit_token = request_meta
    parser = EventStreamParser(max_frame_bytes=state.config.max_body_bytes)
    pool = RehydratorPool(ctx.vault, fuzzy=state.config.rehydration.fuzzy)
    degraded = False
    status = upstream.status_code  # 502 when the proxy itself cut the stream
    cut = False  # whether it did (a delivery_fault local refusal)
    try:
        async for chunk in _UpstreamPaced(upstream.aiter_bytes()):
            if degraded:
                yield chunk
                continue
            try:
                frames = parser.feed(chunk)
            except EventStreamError as exc:
                logger.warning(
                    "event stream framing error on %s (%s); passing through verbatim", path, exc
                )
                degraded = True
                yield parser.residual
                continue
            for frame in frames:
                if observe is not None:
                    observe = await _observed(state, observe, method, path, frame.payload)
                for out in adapter.rehydrate_eventstream_message(frame, pool):
                    yield serialize_eventstream(out)
        if not degraded:
            try:
                parser.close()
            except EventStreamError as exc:
                logger.warning("event stream truncated on %s (%s); forwarding tail", path, exc)
                yield parser.residual
            for _key, text in pool.flush_all().items():
                logger.warning("unflushed stream leftover discarded (%d chars)", len(text))
    except httpx.TransportError:
        raise
    except Exception as exc:
        status = _stream_delivery_fault(state, method, path, exc)
        cut = True
        raise
    finally:
        with suppress(Exception):
            # A stream that errored mid-body can make aclose() itself raise;
            # that must never skip the finalization below.
            await upstream.aclose()
        # Streamed requests finalize their totals and audit row here — the
        # HTTP handler returned long before the stream ended.
        state.rehydration_counts.update(pool.counts)
        state.record_request(
            session=ctx.session_id,
            provider=adapter.name,
            method=method,
            path=path,
            status=status,
            started=started,
            streamed=True,
            detections=detections,
            rehydrations=dict(pool.counts),
            warned=warned,
            audit_token=audit_token,
            refusal="delivery_fault" if cut else None,
        )


def _refusals_by_kind(refusals: Counter[tuple[str, str]]) -> dict[str, int]:
    by_kind: Counter[str] = Counter()
    for (kind, _provider), count in refusals.items():
        by_kind[kind] += count
    return dict(sorted(by_kind.items()))


def _sink_counts(state: ProxyState, attribute: str) -> Counter[str]:
    """One counter of the enabled off-machine audit sinks, by sink ("s3",
    "azure"): the sink seam's ``batches_uploaded`` / ``rows_dropped``
    (audit_s3's Protocols), read defensively — a sink without the attribute
    (or with a non-integer one) is left out, never a failed scrape."""
    counts: Counter[str] = Counter()
    for name, sink in (("s3", state.audit_s3), ("azure", state.audit_azure)):
        value = getattr(sink, attribute, None) if sink is not None else None
        if isinstance(value, int) and not isinstance(value, bool):
            counts[name] = value
    return counts


def _dashboard_unavailable(state: ProxyState) -> str:
    """Why the dashboard paths 404 — never a guess: the package is absent,
    present but its plugin did not register, or present but it declined
    the dashboard (the pro builder honors the license tier)."""
    free_apis = (
        "the free core serves JSON status at /__llm-redact/status and Prometheus"
        " metrics at /__llm-redact/metrics (see `llm-redact status` and"
        " `llm-redact preview`)"
    )
    if not pro_package_installed():
        return (
            "the web dashboard (status view, config editor, redaction preview) is"
            f" provided by the llm-redact-pro package, which is not installed; {free_apis}"
        )
    if not loaded_plugins():
        return (
            "llm-redact-pro is installed but its plugin did not register (see the"
            f" startup log), so the web dashboard is unavailable; {free_apis}"
        )
    return (
        "llm-redact-pro did not enable the web dashboard: it needs a Pro license key"
        f" (current tier: {state.license.tier}) and llm-redact-pro 0.4 or newer; {free_apis}"
    )


# The access gate's optional posture lines in its /status ``users`` block
# (``AccessGate.status``): at most this many, each at most this long.
GATE_POSTURE_LINES = 8
GATE_POSTURE_CHARS = 200


def gate_posture(value: object) -> list[str] | None:
    """The ``posture`` an access gate's ``status()`` reported, as /status
    serves it: its first ``GATE_POSTURE_LINES`` non-empty strings, each with
    every non-printable character escaped (``overrides.printable``) and cut
    to ``GATE_POSTURE_CHARS``; any other entry dropped. None — the key is
    dropped — for a value that is not a list (never a fault)."""
    if not isinstance(value, list):
        return None
    lines = [entry for entry in value if isinstance(entry, str) and entry]
    return [_posture_line(entry) for entry in lines[:GATE_POSTURE_LINES]]


def _posture_line(text: str) -> str:
    escaped = printable(text)
    if len(escaped) > GATE_POSTURE_CHARS:
        return escaped[: GATE_POSTURE_CHARS - 1] + "…"
    return escaped


def users_block(gate: AccessGate | None) -> dict[str, Any]:
    """The /status ``users`` block: the access gate's ``status()`` (a copy),
    its optional ``posture`` sanitized (``gate_posture``); without a gate, no
    registry and nothing enforced."""
    if gate is None:
        return {"registry": False, "enforcement": False}
    block = gate.status()
    if not isinstance(block, dict) or "posture" not in block:
        return block
    block = dict(block)
    posture = gate_posture(block.pop("posture"))
    if posture is not None:
        block["posture"] = posture
    return block


async def _handle_local(
    request: Request, state: ProxyState, admission: Admission | None = None
) -> Response:
    """Answer reserved /__llm-redact endpoints locally. Metadata only —
    never values; allowlists reported as counts. The dashboard paths are
    delegated to the llm-redact-pro Dashboard (its config editor is the one
    exception on both fronts: it accepts POST behind the guard chain and
    returns allowlist values). ``admission`` is the dashboard admission
    (``_admit_reserved``), recorded with a live-events stream."""
    admission = admission or Admission()
    path = request.url.path

    # The browser dashboard (page, config editor, redaction preview) is the
    # llm-redact-pro surface: only these fixed paths are ever dispatched to
    # it, so a plugin can never shadow a core endpoint below.
    if path in DASHBOARD_PATHS:
        if state.dashboard is not None:
            return await state.dashboard.handle(request, state)
        return JSONResponse({"error": _dashboard_unavailable(state)}, status_code=404)
    if path in (f"{RESERVED_PREFIX}/sessions", f"{RESERVED_PREFIX}/sessions/prune"):
        return await _handle_sessions(request, state)
    if path in OVERRIDE_PATHS:
        return await _handle_overrides(request, state, admission)
    scim = path == SCIM_PREFIX or path.startswith(SCIM_PREFIX + "/")
    if path in ACCESS_PATHS or _is_auth_path(path) or scim:
        # Host/Origin checked here too (defense in depth — the gate runs the
        # full guard chain itself) so a rebinding page learns nothing. SCIM
        # clients are identity providers, not browsers: Host only. The
        # sign-in paths keep the Origin check for every method, POST included.
        if not _host_allowed(request, state):
            return JSONResponse({"error": "host not allowed"}, status_code=403)
        if not scim and not _origin_allowed(request, state):
            return JSONResponse({"error": "origin not allowed"}, status_code=403)
        if state.access_gate is not None:
            return await state.access_gate.handle(request, state)
        return JSONResponse(
            {"error": "named users and access control require the llm-redact-pro package"},
            status_code=404,
        )
    if request.method != "GET":
        return JSONResponse({"error": "method not allowed"}, status_code=405)

    # Liveness / readiness probes: intentionally DB-free and un-gated (unlike
    # /status, which queries the vault on every call). A container HEALTHCHECK
    # or k8s probe hits these; they reveal nothing sensitive.
    if path == f"{RESERVED_PREFIX}/healthz":
        return JSONResponse({"status": "ok"})
    if path == f"{RESERVED_PREFIX}/readyz":
        return JSONResponse(
            {"status": "ready", "version": __version__, "realtime": websockets_available()}
        )

    if path == f"{RESERVED_PREFIX}/guide":
        return Response(
            content=state.guide_html,
            media_type="text/html; charset=utf-8",
            headers={"cache-control": "no-store"},
        )

    if path == f"{RESERVED_PREFIX}/status":
        config = state.config
        vault_block: dict[str, Any] = {
            "backend": config.vault.backend,
            "entries": state.vault_manager.total_entries(),
            "sessions": state.vault_manager.session_count(),
            # "kms:<provider>" | "local" (env / key command / keychain) |
            # None (unencrypted). Posture only: never the key or key id.
            "key_source": vault_key_source(config.vault),
            # The effective [vault] map_writes: "before_answer" (an answer
            # waits for its durable map writes: every replica reads them) or
            # "background" (this process answers from the writer's overlay
            # at once; another replica once the write landed) — or
            # "synchronous" (no background writer: each write landed before
            # its call returned, e.g. the in-memory vault).
            "map_writes": state.map_writes,
            # How long an answer waits for them ([vault]
            # map_write_wait_seconds), the writes queued or in flight now,
            # and the answers sent before theirs landed, by cause ("bound":
            # the wait ran out; "stuck": not waited for, the writer is stuck
            # behind an earlier write). Counts only.
            "map_write_wait_seconds": state.map_write_wait_seconds,
            "map_writes_pending": state.map_writes_pending(),
            "map_write_wait_timeouts_total": dict(state.map_write_wait_timeouts),
        }
        if config.vault.backend in RDBMS_BACKENDS:
            from llm_redact.vault_rdbms import (
                ENV_REMOTE_PLAINTEXT,
                identity_tls_unverified,
                managed_dbms_cloud,
            )

            # Honesty fields: a recognized managed-DBMS host and the
            # remote-plaintext hatch are opt-in postures — never silent.
            vault_block["managed_cloud"] = managed_dbms_cloud(config.vault)
            # "password" (static) or "identity" (a per-connect token minted
            # from the proxy's cloud identity) — never the credential itself.
            vault_block["auth"] = config.vault.rdbms.auth
            vault_block["remote_plaintext"] = (
                config.vault.encryption != "fernet" and os.environ.get(ENV_REMOTE_PLAINTEXT) == "1"
            )
            # The identity token's TLS link is unverified (the hatch is set).
            vault_block["tls_unverified"] = identity_tls_unverified(config.vault)
            # The database user could not ALTER the response map to add its
            # kind column: stored objects' owner records share the Responses
            # rows' bound (a manager without the member reports False).
            vault_block["owner_bound_shared"] = (
                getattr(state.vault_manager, "owner_bound_shared", False) is True
            )
        return JSONResponse(
            {
                "version": __version__,
                "uptime_seconds": round(time.time() - state.started_at, 1),
                "started_at": datetime.fromtimestamp(state.started_at, tz=UTC).isoformat(
                    timespec="seconds"
                ),
                "session": config.vault.session,
                "session_mode": config.vault.session_mode,
                "compaction_forks": state.compaction_forks,
                "vault": vault_block,
                "detections_total": dict(state.redactor.counts),
                "rehydrations_total": dict(state.rehydration_counts),
                "warnings_total": dict(state.warn_counts),
                "blocked_total": dict(state.blocked_counts),
                # Refusal overrides (overrides.py): live pending codes,
                # one-time grants and every-time rules, and the refusals
                # passed on them by this process ("once" / "always").
                "overrides": _overrides_status(state),
                "upstream_errors_total": dict(state.upstream_errors),
                "bookkeeping_errors_total": dict(state.bookkeeping_errors),
                # Responses the proxy generated itself instead of forwarding
                # (metrics.LOCAL_REFUSAL_KINDS), by kind — summed over
                # providers (/metrics carries the provider label).
                "local_refusals_total": _refusals_by_kind(state.metrics.local_refusals),
                # Open long-lived connections (realtime relays, live-events
                # streams) by kind, and those closed because their admission
                # ended, by cause: the access gate revoked them ("revoked"),
                # their re-check refused them ("recheck") or failed
                # ("recheck_error"). Counts only — never a user or a grant.
                "connections": {
                    "open": state.connections.open_counts(),
                    "closed_total": {
                        cause: state.connections.closed[cause] for cause in CONNECTION_CLOSE_CAUSES
                    },
                    "recheck_interval_seconds": state.connections.interval,
                },
                # Requests refused before any upstream contact as a web page's
                # (CSRF, DNS rebinding, cross-site WebSocket) or, when they
                # would spend a credential the proxy holds, as addressed to a
                # host name it does not answer to — by kind, never a value.
                "request_origin_refusals_total": dict(state.request_origin_refusals),
                # Binary upload file parts forwarded UNSCANNED with the
                # client's own key ([detection] binary_uploads = "forward"),
                # by provider — never a name or a byte of them.
                "unscanned_uploads_total": dict(state.unscanned_uploads),
                # Binary upload file parts an upload inspector read as text,
                # by provider and outcome (upload_inspection.OUTCOMES):
                # "clean" ones went out after a clean scan of their
                # EXTRACTED text only ("clean_refused": scanned clean, but
                # the upload was refused; "overridden": clean only because
                # an approved override let its values through). Counts only.
                "inspected_uploads_total": _inspected_by_provider(state.inspected_uploads),
                "upload_inspector": _inspector_status(state),
                # How many browser origins the operator listed in
                # allowed_origins (the count, not the list): pages there can
                # read restored values back through the proxy — opt-in.
                "allowed_origins": len(config.allowed_origins),
                "detection": {
                    "enabled_rules": list(config.detection.enabled),
                    # None = all languages; otherwise the active scope and
                    # the enabled rules it leaves unbuilt.
                    "languages": (
                        list(config.detection.languages)
                        if config.detection.languages is not None
                        else None
                    ),
                    "language_inactive_rules": sorted(
                        set(config.detection.enabled) - set(active_rule_names(config.detection))
                    ),
                    "custom_rules": len(config.detection.custom_rules),
                    "allowlist_entries": len(config.detection.allowlist),
                    "allowlist_patterns": len(config.detection.allowlist_patterns),
                    "allowlist_by_type_entries": sum(
                        len(values) for _type, values in config.detection.allowlist_by_type
                    ),
                    "ner_enabled": config.detection.ner.enabled,
                    "modes": {name: mode for name, mode in config.detection.modes},
                    # Count only: deny values are themselves secrets. The
                    # config editor GET returns them — the same documented
                    # exception as allowlist values, behind the same checks.
                    "deny_strings": len(config.detection.deny_strings),
                },
                "rehydration": {"fuzzy": config.rehydration.fuzzy},
                "max_body_bytes": config.max_body_bytes,
                "max_body_strings": config.max_body_strings,
                "inject_system_note": config.inject_system_note,
                "providers": {
                    name: provider.upstream_base_url for name, provider in config.providers.items()
                },
                "providers_disabled": sorted(
                    name for name, provider in config.providers.items() if not provider.enabled
                ),
                # Loud honesty for the per-provider off-switch: requests to
                # these providers are forwarded WITHOUT redaction.
                "providers_detection_off": sorted(
                    name for name, provider in config.providers.items() if not provider.detection
                ),
                # How the proxy authenticates to each upstream: "passthrough"
                # (the client's credential) or "identity" (the proxy's OWN
                # cloud identity — any client that reaches it spends that).
                # Modes only; never a credential, region or source detail.
                "providers_auth": {
                    name: provider.auth for name, provider in config.providers.items()
                },
                "mcp_exempt_servers": len(config.detection.mcp_exempt_servers),
                "audit": {
                    "enabled": state.audit is not None,
                    "rows": state.audit.count() if state.audit is not None else 0,
                    "tamper_evident": config.audit.tamper_evident,
                    "required": config.audit.required,
                    "s3": {
                        "enabled": state.audit_s3 is not None,
                        "encryption": config.audit.s3.encryption == "fernet",
                        "auth": config.audit.s3.auth,
                        "batches_uploaded": (
                            state.audit_s3.batches_uploaded if state.audit_s3 is not None else 0
                        ),
                        "rows_dropped": (
                            state.audit_s3.rows_dropped if state.audit_s3 is not None else 0
                        ),
                    },
                    "azure": {
                        "enabled": state.audit_azure is not None,
                        "encryption": config.audit.azure.encryption == "fernet",
                        "auth": config.audit.azure.auth,
                        "batches_uploaded": (
                            state.audit_azure.batches_uploaded
                            if state.audit_azure is not None
                            else 0
                        ),
                        "rows_dropped": (
                            state.audit_azure.rows_dropped if state.audit_azure is not None else 0
                        ),
                    },
                },
                "otel_enabled": state.telemetry is not None,
                # WS relay readiness: without the websockets package uvicorn
                # refuses upgrades, so realtime APIs bypass nothing — they
                # simply cannot connect.
                "realtime_available": websockets_available(),
                # Effective license state (llm-redact-pro docs/licensing.md): tier, user
                # cap, cloud entitlements, expiry — metadata only, never the
                # key itself. Warnings surface invalid-key-fell-to-Free and
                # the expiry grace window (never silent).
                # The access gate's optional authorization seams: whether it
                # authorizes requests, hands out detection overlays,
                # authorizes what the redaction found, and reloads its own
                # policy with the configuration.
                "access": {
                    "authorizes_requests": state.authorization.authorizes,
                    "detection_overlays": state.authorization.overlays,
                    "authorizes_content": state.authorization.checks_content,
                    "reloads_policy": state.authorization.reloads,
                },
                # The access gate's own block (llm-redact-pro); without one,
                # no registry and nothing enforced.
                # Its optional `posture` lines sanitized (gate_posture).
                "users": users_block(state.access_gate),
                "license": {
                    "tier": state.license.tier,
                    "source": state.license.source,
                    "in_grace": state.license.in_grace,
                    "warnings": list(state.license.warnings),
                    "max_users": state.license.max_users,
                    "org": (
                        state.license.license.org if state.license.license is not None else None
                    ),
                    "expires": (
                        state.license.license.expires.isoformat()
                        if state.license.license is not None
                        else None
                    ),
                    # Open-core honesty (llm-redact-pro docs/licensing.md): is the paid
                    # llm-redact-pro package present, and did its plugin
                    # register? `plugins` empty while `package_installed` is
                    # true means paid features are silently OFF — surfaced,
                    # never assumed. Both independent of the license tier.
                    "package_installed": pro_package_installed(),
                    "plugins": sorted(loaded_plugins()),
                },
                # The routing block (R-30): produced by the router (metadata
                # only — credential MODES, never variable names or values);
                # `{"enabled": false}` whenever no router is held.
                "routing": state.router.status()
                if state.router is not None
                else {"enabled": False},
            }
        )

    if path == f"{RESERVED_PREFIX}/metrics":
        # The access gate's own gauges first (awaited off the loop, bounded):
        # a sample it drops counts under the bookkeeping stage the core's
        # render then shows.
        plugin_text = await state.plugin_metric_text()
        return Response(
            content=state.metrics.render(
                detections=state.detection_counts,
                rehydrations=state.rehydration_counts,
                warnings=state.warn_counts,
                blocked=state.blocked_counts,
                vault_entries=state.vault_manager.total_entries(),
                vault_sessions=state.vault_manager.session_count(),
                compaction_forks=state.compaction_forks,
                upstream_errors=state.upstream_errors,
                bookkeeping_errors=state.bookkeeping_errors,
                connections_closed=state.connections.closed,
                unscanned_uploads=state.unscanned_uploads,
                inspected_uploads=state.inspected_uploads,
                overrides_used=state.overrides.used if state.overrides is not None else None,
                audit_sink_batches=_sink_counts(state, "batches_uploaded"),
                audit_sink_rows_dropped=_sink_counts(state, "rows_dropped"),
                map_write_queue_depth=state.map_writes_pending(),
                map_write_wait_timeouts=state.map_write_wait_timeouts,
            )
            + plugin_text,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    if path == f"{RESERVED_PREFIX}/events":
        # Live SSE feed of recent-request rows (same metadata-only shape as
        # /recent). Host-check gated like /sessions: DNS rebinding protects
        # readable endpoints too. The dashboard falls back to polling when
        # EventSource fails, so dropping a slow consumer's events is safe.
        if not _host_allowed(request, state):
            return JSONResponse({"error": "host not allowed"}, status_code=403)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        # A long-lived connection: tracked with the dashboard admission it
        # opened under, so the access gate (or its re-check) can end it.
        stream = EventStream(
            queue,
            subject=admission.subject,
            grant=getattr(admission, "grant", None),
            recheck=getattr(admission, "recheck", None),
        )
        state.event_subscribers.add(queue)
        state.connections.track(stream)

        async def event_stream() -> AsyncIterator[bytes]:
            try:
                yield b": connected\n\n"
                while not stream.closed:
                    try:
                        row = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except TimeoutError:
                        yield b": keepalive\n\n"
                        continue
                    if stream.closed:
                        # Its admission ended: the stream ends here (the
                        # dashboard reconnects, and the gate decides again)
                        # — or the server is shutting down.
                        if not stream.at_shutdown:
                            logger.info("events stream closed (its access was revoked)")
                        return
                    yield b"data: " + json_bytes(row) + b"\n\n"
            finally:
                state.connections.untrack(stream)
                state.event_subscribers.discard(queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-store"},
        )

    if path == f"{RESERVED_PREFIX}/recent":
        # The audit table's in-memory sibling: same row shape, newest first,
        # capped at the ring buffer size, available without the audit DB.
        # Host-gated like /events (its SSE twin): /recent exposes the exact
        # same rows, so leaving it open would defeat the rebinding defense on
        # /events by letting a rebound page poll here instead.
        if not _host_allowed(request, state):
            return JSONResponse({"error": "host not allowed"}, status_code=403)
        try:
            limit = int(request.query_params.get("limit", "50"))
        except ValueError:
            limit = 50
        limit = max(1, min(limit, 200))
        entries = list(state.recent)[-limit:]
        entries.reverse()
        return JSONResponse({"entries": entries})

    if path == f"{RESERVED_PREFIX}/audit":
        # Host-gated: audit rows carry the same request metadata (paths,
        # providers, counts, user names) as /recent and /events.
        if not _host_allowed(request, state):
            return JSONResponse({"error": "host not allowed"}, status_code=403)
        if state.audit is None:
            return JSONResponse(
                {"error": "audit log is disabled; set [audit] enabled = true"}, status_code=404
            )
        try:
            limit = int(request.query_params.get("limit", "50"))
        except ValueError:
            limit = 50
        limit = max(1, min(limit, 1000))
        return JSONResponse({"entries": state.audit.recent(limit)})

    return JSONResponse({"error": "unknown llm-redact endpoint"}, status_code=404)


# Body cap for the guarded local POSTs (session prune, user invite/revoke,
# and the pro dashboard's config editor and preview).
_CONFIG_BODY_LIMIT = 1024 * 1024
CSRF_HEADER = "x-llm-redact-csrf"


def _inspected_by_provider(counts: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """``inspected_uploads`` as /status reports it: provider -> outcome -> count."""
    nested: dict[str, dict[str, int]] = {}
    for (provider, outcome), count in sorted(counts.items()):
        nested.setdefault(provider, {})[outcome] = count
    return nested


def _inspector_status(state: ProxyState) -> dict[str, Any]:
    """The /status ``upload_inspector`` block: off, or the core's bounds
    and the inspector's own metadata (a fault there is reported by type,
    never raised into /status)."""
    inspector, limits = state.upload_inspector, state.inspection_limits
    if inspector is None or limits is None:
        return {"enabled": False}
    block: dict[str, Any] = {
        "enabled": True,
        "timeout_seconds": limits.timeout,
        "max_part_bytes": limits.max_bytes,
    }
    try:
        block["inspector"] = inspector.status()
    except Exception as exc:  # noqa: BLE001 — metadata only, never fatal
        block["inspector"] = {"status_error": type(exc).__name__}
    return block


def _allowed_hostnames(state: ProxyState) -> set[str]:
    """The host names the proxy answers to: its loopback names, its bind
    host, the operator's ``allowed_hosts``, and the access gate's public
    origin (only for a gate that guards the reserved endpoints)."""
    names = {"127.0.0.1", "localhost", "::1", state.config.host.lower()}
    names.update(state.config.allowed_hosts)
    if state.public_origin is not None:
        names.add(state.public_origin[1])
    return names


def _parse_public_origin(gate: object) -> tuple[str, str, str] | None:
    """The gate's ``public_origin()`` as (scheme, lowercase host, origin
    string), or None. Anything but a plain http(s) origin (no path, query,
    fragment or credentials) is a startup ConfigError — never a silently
    wider Host check."""
    getter = getattr(gate, "public_origin", None)
    raw = getter() if callable(getter) else None
    if raw is None:
        return None
    parsed = urllib.parse.urlsplit(str(raw))
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ConfigError(
            "the access gate's public origin must be a plain http(s)://host[:port] URL"
        )
    default_port = 443 if parsed.scheme == "https" else 80
    port = f":{parsed.port}" if parsed.port not in (None, default_port) else ""
    host = parsed.hostname.lower()
    display_host = f"[{host}]" if ":" in host else host
    return parsed.scheme, host, f"{parsed.scheme}://{display_host}{port}"


async def _admit_reserved(
    request: Request, state: ProxyState
) -> tuple[Response | None, Admission | None]:
    """Dashboard admission for the reserved endpoints (only when the gate
    opts in with ``guards_dashboard``): every reserved path except the
    monitoring probes and the gate's own paths (sign-in and SCIM
    authenticate themselves). A None response admits (with the admission,
    when the gate was asked); otherwise the refusal — a 303 to the gate's
    same-proxy sign-in page for a browser GET, else a 403. Nothing here is
    forwarded."""
    if not state.guards_dashboard:
        return None, None
    path = request.url.path
    if (
        path in PROBE_PATHS
        or _is_auth_path(path)
        or path == SCIM_PREFIX
        or path.startswith(SCIM_PREFIX + "/")
    ):
        return None, None
    admission = await state.admit(request, "dashboard")
    if admission.refusal is None:
        return None, admission
    redirect = admission.redirect
    if (
        request.method == "GET"
        and redirect is not None
        and redirect.startswith(RESERVED_PREFIX + "/")
        and "\\" not in redirect
        and "//" not in redirect
    ):
        return RedirectResponse(redirect, status_code=303), admission
    return JSONResponse({"error": admission.refusal}, status_code=403), admission


def _host_allowed(request: HTTPConnection, state: ProxyState) -> bool:
    """DNS-rebinding defense: a rebinding page's requests carry the
    attacker's domain in Host, while local browsers and tools send the
    loopback name they connected to (or a name listed in allowed_hosts)."""
    hostname = request.url.hostname
    return hostname is not None and hostname.lower() in _allowed_hostnames(state)


def _origin_allowed(request: Request, state: ProxyState) -> bool:
    """Absent Origin (curl, same-origin GET) is fine — the CSRF token still
    gates POST. A present Origin must be a local origin ('null', a
    malformed value and everything else is rejected); https origins exist
    only when the proxy itself serves TLS."""
    origin = request.headers.get("origin")
    if origin is None:
        return True
    if state.public_origin is not None and origin.lower() == state.public_origin[2]:
        return True
    try:
        parsed = urllib.parse.urlsplit(origin)
    except ValueError:  # e.g. an unterminated IPv6 literal
        return False
    schemes = ("http", "https") if state.config.tls.enabled else ("http",)
    return parsed.scheme in schemes and (parsed.hostname or "").lower() in _allowed_hostnames(state)


# --- requests from web pages (CSRF, DNS rebinding) ------------------------------
#
# Every forwarded request borrows something the proxy holds: its vault (a
# rehydrating route restores the operator's values into whatever the upstream
# echoes, so a page with its OWN provider key could read the vault back token
# by token), keyless upstreams it can reach (a local Ollama or vLLM), and —
# under identity auth or a routed operator key — a credential. A web page in
# the operator's browser can send requests to 127.0.0.1 (a "simple" POST needs
# no preflight; WebSocket handshakes get no CORS at all) or, after DNS
# rebinding, be same-origin with the proxy and read the answers.

# Fetch Metadata values of a request the proxy's own page made, or that the
# user typed or bookmarked; "same-site" and "cross-site" name another origin.
_OWN_FETCH_SITES = frozenset({"same-origin", "none"})
# A request URL's scheme as a web origin's (a WebSocket handshake comes from a
# page on the http(s) origin of the same host and port).
_ORIGIN_SCHEMES = {"http": "http", "https": "https", "ws": "http", "wss": "https"}
# What a refused request is told, by refusal kind — the kind only, never the
# Host or Origin it carried; short enough for a WebSocket close frame.
REQUEST_ORIGIN_REFUSALS = {
    "host": (
        "this proxy does not answer to that host name (a DNS-rebinding defense);"
        " list it in allowed_hosts if a client uses it"
    ),
    "origin": (
        "a web page on another origin sent this request; llm-redact serves only the origins"
        " in allowed_origins"
    ),
    "fetch_site": (
        "a page on another site or origin sent this request without an Origin to check"
        " against allowed_origins"
    ),
}


def _browser_request(conn: HTTPConnection) -> bool:
    """Whether a browser sent this request: it carries Origin or a Fetch
    Metadata (``Sec-Fetch-*``) header. Page script can neither set nor
    remove either (forbidden header names); CLI tools and SDKs send
    neither."""
    return any(
        name == b"origin" or name.startswith(b"sec-fetch-") for name, _ in conn.scope["headers"]
    )


def _own_origin(conn: HTTPConnection, state: ProxyState, origin: str) -> bool:
    """Whether ``origin`` (an Origin header value) is this request's own:
    the access gate's public origin, or exactly the scheme, host and port
    the request was sent to (its Host, which the caller has checked).
    Port-exact, unlike ``_origin_allowed``: an API route has no CSRF token
    to stop a page served on another port of the same host name."""
    value = origin.strip().lower()
    if state.public_origin is not None and value == state.public_origin[2]:
        return True
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError:  # a malformed port or IPv6 literal
        return False
    own = conn.url
    scheme = _ORIGIN_SCHEMES.get(own.scheme)
    if (
        scheme is None
        or parsed.scheme != scheme
        or own.hostname is None
        or "@" in parsed.netloc  # userinfo: not a serialized origin
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return False
    default = 443 if scheme == "https" else 80
    return (parsed.hostname, port or default) == (own.hostname.lower(), own.port or default)


def request_origin_refusal(
    conn: HTTPConnection, state: ProxyState, *, lends_credential: bool
) -> str | None:
    """Why a request bound for an upstream must be refused as a web page's,
    or None: a key of ``REQUEST_ORIGIN_REFUSALS``.

    A request with browser markers (``_browser_request``) must be addressed
    to a host name the proxy answers to (DNS rebinding), carry only its own
    origin (CSRF, cross-site WebSocket hijacking) and a ``Sec-Fetch-Site``
    of ``same-origin`` or ``none``. The one exception is the operator's
    opt-in: an Origin listed in ``allowed_origins`` (compared in its
    serialized form, ``normalize_origin``) is served although it is
    cross-site by definition — its Origin, which page script cannot forge,
    vouches for it; the Host rule still holds. One that would spend a
    credential the proxy holds (``lends_credential``) must name such a host
    even without markers — a browser lacking Fetch Metadata sends none on a
    same-origin GET — unless it arrived over TLS: a browser verifies the
    proxy's certificate against the name it resolved, so a rebound page
    never reaches a TLS listener, and a team's clients may use any name the
    certificate covers. CLI tools and SDKs send no browser markers, so an
    alias host (a compose service) keeps working on the client's own
    credential."""
    if not _browser_request(conn):
        # No Origin, no Sec-Fetch-*: only the lent credential's host rule.
        if lends_credential and conn.url.scheme not in ("https", "wss"):
            return None if _host_allowed(conn, state) else "host"
        return None
    if not _host_allowed(conn, state):
        return "host"
    headers = conn.headers
    listed = False
    for origin in headers.getlist("origin"):
        if _own_origin(conn, state, origin):
            continue
        if normalize_origin(origin) not in state.config.allowed_origins:
            return "origin"
        listed = True
    if listed:
        return None  # a listed page's fetch is cross-site by definition
    for site in headers.getlist("sec-fetch-site"):
        if site.strip().lower() not in _OWN_FETCH_SITES:
            return "fetch_site"
    return None


def _request_origin_refused(
    state: ProxyState,
    adapter: ProviderAdapter | None,
    kind: str,
    *,
    provider_name: str | None,
    request: Request,
    path: str,
    started: float,
) -> JSONResponse:
    """A request ``request_origin_refusal`` refused: a recorded,
    provider-shaped 403 before any credential fetch or upstream contact,
    counted by kind. The answer and the log line name the kind only —
    never the Host or Origin the request carried."""
    state.request_origin_refusals[kind] += 1
    message = f"llm-redact: {REQUEST_ORIGIN_REFUSALS[kind]}; the request was not forwarded"
    error = adapter.error_body(message, status=403) if adapter is not None else {"error": message}
    state.record_request(
        session=state.config.vault.session,
        provider=provider_name,
        method=request.method,
        path=path,
        status=403,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal="request_origin",
    )
    logger.info("%s %s -> 403 refused (request origin: %s)", request.method, path, kind)
    return JSONResponse(error, status_code=403)


async def _stream_rehydrated_ndjson(
    upstream: httpx.Response,
    adapter: ProviderAdapter,
    state: ProxyState,
    ctx: RequestContext,
    *,
    request_meta: RequestMeta,
    route: RouteDelivery | None = None,
    observe: ResponseObserver | None = None,
) -> AsyncIterator[bytes]:
    """The NDJSON twin of _stream_rehydrated (Ollama streams).

    One JSON object per line; a line that the adapter cannot parse is
    forwarded byte-identically (an unrestored placeholder is safe,
    corrupted output is not). The done:true line is the adapter's flush
    point, so leftovers normally never reach stream close."""
    method, path, started, detections, warned, audit_token = request_meta
    parser = NDJSONParser()
    pool = RehydratorPool(ctx.vault, fuzzy=state.config.rehydration.fuzzy)
    status = upstream.status_code  # 502 when the proxy itself cut the stream
    cut = False  # whether it did (a delivery_fault local refusal)
    try:
        async for chunk in _UpstreamPaced(upstream.aiter_bytes()):
            for line in parser.feed(chunk):
                if observe is not None:
                    observe = await _observed(state, observe, method, path, line)
                out = adapter.rehydrate_ndjson_line(line, pool)
                if route is not None:
                    out = route.observe_line(out)
                yield out + b"\n"
        tail = parser.close()
        if tail:
            # A stream that ended without a final newline: the tail may
            # still be one complete JSON object.
            if observe is not None:
                observe = await _observed(state, observe, method, path, tail)
            out = adapter.rehydrate_ndjson_line(tail, pool)
            if route is not None:
                out = route.observe_line(out)
            yield out
        for _key, text in pool.flush_all().items():
            logger.warning("unflushed stream leftover discarded (%d chars)", len(text))
    except httpx.TransportError as exc:
        if route is not None:
            # A fault AFTER the first byte reached the client: propagated
            # as-is (never re-issued — R-17), classified stream_error. Type only.
            route.mark_failed("stream_error")
            logger.warning(
                "%s %s stream failed after first byte (%s)%s",
                method,
                path,
                type(exc).__name__,
                _route_log_suffix(route.row()),
            )
        raise
    except Exception as exc:
        status = _stream_delivery_fault(state, method, path, exc)
        cut = True
        if route is not None:
            route.mark_failed("stream_error")
        raise
    finally:
        with suppress(Exception):
            # A stream that errored mid-body can make aclose() itself raise;
            # that must never skip the finalization below.
            await upstream.aclose()
        # Streamed requests finalize their totals and audit row here — the
        # HTTP handler returned long before the stream ended.
        state.rehydration_counts.update(pool.counts)
        state.record_request(
            session=ctx.session_id,
            provider=adapter.name,
            method=method,
            path=path,
            status=status,
            started=started,
            streamed=True,
            detections=detections,
            rehydrations=dict(pool.counts),
            warned=warned,
            audit_token=audit_token,
            refusal="delivery_fault" if cut else None,
            route=(state.finish_route(route, status) if route is not None else None),
        )


async def _handle_sessions(request: Request, state: ProxyState) -> Response:
    """Vault session browser (GET) and prune (POST /prune) — the prune
    endpoint sits behind the exact guard stack as the config editor.
    Metadata only: session ids, entry counts, timestamps — never values."""
    if not _host_allowed(request, state):
        return JSONResponse({"error": "host not allowed"}, status_code=403)
    if not _origin_allowed(request, state):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)

    if request.url.path == f"{RESERVED_PREFIX}/sessions":
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        return JSONResponse(
            {
                "backend": state.config.vault.backend,
                "session_mode": state.config.vault.session_mode,
                "active_session": state.config.vault.session,
                "sessions": state.vault_manager.sessions_summary(),
            },
            headers={"cache-control": "no-store"},
        )

    if request.method != "POST":
        return JSONResponse({"error": "method not allowed"}, status_code=405)
    payload, guard_error = await _guarded_post_json(request, state)
    if guard_error is not None:
        return guard_error
    days = payload.get("older_than_days") if isinstance(payload, dict) else None
    if not isinstance(days, int) or isinstance(days, bool) or days < 0:
        return JSONResponse(
            {"error": 'body must be {"older_than_days": N} with N a whole number of days'},
            status_code=400,
        )
    if state.config.vault.backend != "sqlite":
        return JSONResponse(
            {"error": 'pruning requires [vault] backend = "sqlite" (memory dies with the process)'},
            status_code=400,
        )
    # The static session is the always-live fallback namespace: never
    # pruned from the live process (the CLI can, with the proxy stopped);
    # the session router may name more such sessions (a per-user copy).
    pruned = state.vault_manager.prune_sessions(days, exclude=_prune_exclusions(state))
    logger.info("pruned %d idle vault session(s) via /sessions/prune", pruned)
    return JSONResponse({"pruned": pruned})


# The refusal-override endpoints (overrides.py): the requester's own pending
# codes and rules (GET), and the dashboard's Allow once / Always allow /
# Revoke buttons (guarded POSTs).
OVERRIDE_PATHS = frozenset(
    {
        f"{RESERVED_PREFIX}/overrides",
        f"{RESERVED_PREFIX}/overrides/approve",
        f"{RESERVED_PREFIX}/overrides/revoke",
    }
)


async def _handle_overrides(request: Request, state: ProxyState, admission: Admission) -> Response:
    """The requester's refusal overrides, value-free: list them (GET), and
    approve a pending refusal once or always, or revoke one (POST behind the
    guard chain — Host, Origin, the CSRF token only the llm-redact-pro
    dashboard hands out). The requester is the subject the access gate
    admitted to the dashboard, else the local operator: only its own records
    are listed, approved or revoked. The POSTs are served only to a subject
    an access gate that guards the dashboard signed in IN A BROWSER
    (``can_approve``, ``ProxyState.browser_signed_in``): without one the
    CSRF token is no proof of a person — any local client can fetch it, and
    an agent holding its user's API key is admitted to the dashboard too —
    so the local operator approves with the CLI."""
    if not _host_allowed(request, state):
        return JSONResponse({"error": "host not allowed"}, status_code=403)
    if not _origin_allowed(request, state):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    store = state.overrides
    if store is None:
        # Off (the default): a local 404 naming the setting — nothing is
        # listed (the store, if any, holds only inert records), approved or
        # revoked, and the file is never opened.
        return JSONResponse({"error": f"{DISABLED_REASON}; {TO_ENABLE}"}, status_code=404)
    subject = admission.subject or ""
    # The key its records are kept under (the gate's stable id, when it
    # supplies one): none read — the gate could not name it — is no answer.
    owner = state.override_owner(subject)
    if owner is None:
        return JSONResponse(
            {"error": "the access gate could not name this requester's overrides"},
            status_code=503,
        )
    # Only a requester a PERSON signed in to the dashboard in a browser
    # approves or revokes here: without one, any local client (an agent with
    # curl) reads the CSRF token from the dashboard and could approve its own
    # refusal — and so could an agent presenting its user's API key, which
    # a gate also admits to the dashboard (``browser_signed_in``).
    can_approve = (
        state.guards_dashboard and bool(subject) and await state.browser_signed_in(request, subject)
    )
    if request.url.path == f"{RESERVED_PREFIX}/overrides":
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        return JSONResponse(
            {
                "subject": subject or None,
                "can_approve": can_approve,
                "entries": [entry.as_dict() for entry in store.entries(owner)],
            },
            headers={"cache-control": "no-store"},
        )
    if request.method != "POST":
        return JSONResponse({"error": "method not allowed"}, status_code=405)
    if not can_approve:
        return JSONResponse({"error": _OVERRIDE_SIGN_IN}, status_code=403)
    payload, guard_error = await _guarded_post_json(request, state)
    if guard_error is not None:
        return guard_error
    entry_id = payload.get("id") if isinstance(payload, dict) else None
    try:
        if request.url.path.endswith("/approve"):
            scope = payload.get("scope") if isinstance(payload, dict) else None
            approved = store.approve(
                str(scope), approver=owner or None, pending_id=_entry_id(entry_id)
            )
            logger.info("override %s approved (%s) from the dashboard", approved.id, scope)
            return JSONResponse({"approved": approved.as_dict()})
        store.revoke(_entry_id(entry_id), subject=owner)
    except OverrideError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    logger.info("override %s revoked from the dashboard", entry_id)
    return JSONResponse({"revoked": entry_id})


_OVERRIDE_SIGN_IN = (
    "approving or revoking a refusal override here needs a person signed in to the"
    " dashboard in a browser (llm-redact-pro [auth.dashboard]; an API key or a user key"
    " is not one); on the proxy's machine run"
    " `llm-redact override CODE --once|--always` (or `override revoke ID`) instead"
)


def _entry_id(value: object) -> str:
    if not isinstance(value, str):
        raise OverrideError('body must be {"id": "p12"} (or "r12"; with "scope" to approve)')
    return value


def _overrides_status(state: ProxyState) -> dict[str, Any]:
    """The /status ``overrides`` block (counts only — never a value, a
    digest, a code or a subject)."""
    store = state.overrides
    if store is None:
        return {"enabled": False}
    block: dict[str, Any] = {
        "enabled": True,
        "ttl_minutes": state.config.overrides.ttl_minutes,
        "used_total": dict(store.used),
    }
    try:
        block.update(store.counts())
    except Exception as exc:  # noqa: BLE001 — a status read never fails the page
        logger.warning("overrides: the store could not be read (%s)", type(exc).__name__)
    return block


class _LiveSessions:
    """``plugin_api.SessionStore`` over the running proxy's vault manager
    (handed to an access gate's optional ``bind_sessions``)."""

    def __init__(self, state: ProxyState) -> None:
        self._state = state

    def session_ids(self) -> list[str]:
        return [str(row["session"]) for row in self._state.vault_manager.sessions_summary()]

    def forget(self, session_ids: Iterable[str]) -> int:
        state = self._state
        static = state.config.vault.session
        doomed = [session_id for session_id in set(session_ids) if session_id != static]
        if not doomed:
            return 0
        forgotten = state.vault_manager.forget_sessions(doomed)
        state._known_sessions.difference_update(doomed)
        logger.info("forgot %d vault session(s) on the access gate's request", forgotten)
        return forgotten


def _prune_exclusions(state: ProxyState) -> frozenset[str]:
    """The sessions a live prune must keep: the configured static session,
    plus every current session the router marks durable (its optional
    ``is_durable``, see plugin_api.SessionRouter) — provider-side state
    (Responses chains, batches) still points into them, and a recreated
    session would hand out the same placeholder numbers for new values."""
    keep = {state.config.vault.session}
    is_durable = getattr(state.session_router, "is_durable", None)
    if is_durable is None:
        return frozenset(keep)
    for row in state.vault_manager.sessions_summary():
        session_id = str(row["session"])
        try:
            verdict = is_durable(session_id)
        except Exception as exc:  # noqa: BLE001 — a failing router can only keep more
            logger.warning(
                "session router is_durable failed (%s); keeping the session", type(exc).__name__
            )
            keep.add(session_id)
            continue
        if verdict is not False:
            # Only an explicit False releases a session; True keeps it, and
            # so does any other answer (None, a count) from a buggy router.
            keep.add(session_id)
    return frozenset(keep)


async def _guarded_post_json(
    request: Request, state: ProxyState
) -> tuple[Any, None] | tuple[None, Response]:
    """The POST guard chain shared by every local mutating endpoint: CSRF
    header (readable only via a same-origin GET), content type, 1 MiB body
    cap, JSON parse. Host and Origin were already checked by the caller."""
    token = request.headers.get(CSRF_HEADER, "")
    if not secrets.compare_digest(token, state.csrf_token):
        return None, JSONResponse({"error": "missing or invalid CSRF token"}, status_code=403)
    content_type = request.headers.get("content-type", "")
    if content_type.split(";")[0].strip().lower() != "application/json":
        return None, JSONResponse(
            {"error": "content-type must be application/json"}, status_code=415
        )
    raw_body = await _read_capped(request, _CONFIG_BODY_LIMIT)
    if raw_body is None:
        return None, JSONResponse({"error": "request body over 1 MiB"}, status_code=413)
    try:
        return loads_bounded(raw_body), None
    except ValueError as exc:  # JSONDecodeError and JsonTooDeep alike
        return None, JSONResponse({"error": f"invalid JSON: {exc}"}, status_code=400)


def _body_too_large(
    state: "ProxyState",
    request: Request,
    adapter: ProviderAdapter,
    *,
    session: str,
    path: str,
    started: float,
    cap: str,
    limit: int,
) -> Response:
    """The recorded, provider-shaped 413 for a redactable body over one of
    its caps — ``max_body_bytes`` (too big to buffer) or ``max_body_strings``
    (too many strings or multipart parts to redact on the event loop) —
    answered before any upstream contact: forwarding it unredacted is never
    an option. The message names the cap, never the content."""
    logger.info("%s %s -> 413 body over %s (%d)", request.method, path, cap, limit)
    state.record_request(
        session=session,
        provider=adapter.name,
        method=request.method,
        path=path,
        status=413,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal="too_large" if cap == "max_body_bytes" else "too_many_strings",
    )
    return Response(
        content=json.dumps(
            adapter.error_body(f"request body exceeds llm-redact {cap} ({limit})")
        ).encode("utf-8"),
        status_code=413,
        media_type="application/json",
    )


def _multipart_parts_over(
    headers: Headers,
    body: bytes,
    limit: int,
    adapter: ProviderAdapter | None = None,
    path: str = "",
) -> bool:
    """Whether a multipart body may hold more than ``limit`` parts, decided
    without parsing it: every part multipart.parse finds ends at a
    ``CRLF--boundary`` delimiter of its own, so their count bounds the
    parts (bytes.count: no allocation per part). A body that is not
    multipart — multipart/form-data, or the multipart type ``adapter``
    reads on ``path`` (``multipart_boundary``) — has no parts."""
    content_type = headers.get("content-type", "")
    boundary = (
        adapter.multipart_boundary(path, content_type)
        if adapter is not None
        else parse_multipart_boundary(content_type)
    )
    return boundary is not None and body.count(b"\r\n--" + boundary) > limit


async def _read_capped(request: Request, limit: int) -> bytes | None:
    """Read the body, aborting as soon as it exceeds ``limit`` bytes.

    Reading incrementally (rather than trusting Content-Length) bounds proxy
    memory even against a lying header. Returns None when over the limit.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _lends_credential(plan: RoutePlan) -> bool:
    """Whether a routed request may reach its provider with a credential the
    PROXY holds (``RoutePlan.proxy_credential``, optional): only an explicit
    False — every upstream the plan may use forwards the client's own
    credential — says no; a plan without the member counts as the proxy's
    (fail closed: the stored-object check is then only stricter)."""
    return getattr(plan, "proxy_credential", True) is not False


class _Unreadable(NamedTuple):
    """Why a request body cannot be forwarded as it is: the proxy cannot
    read it — for the stored-object check, or to redact it (status,
    message — the body's KIND only, never its content)."""

    status: int
    message: str
    # False: refused whatever credential the request spends (the message
    # then names no credential).
    credential_bound: bool = True
    # The local-refusal kind when the status does not name it
    # (``_unreadable_kind``): a 413 says which cap.
    kind: LocalRefusal | None = None


def _unreadable_kind(unreadable: "_Unreadable", default: LocalRefusal) -> LocalRefusal:
    """The local-refusal kind of an unreadable body: its own when it names
    one, a content coding's 415 ``unsupported_encoding``, else ``default``."""
    if unreadable.kind is not None:
        return unreadable.kind
    return "unsupported_encoding" if unreadable.status == 415 else default


def _raced_kind(scope: "OverrideScope | None") -> LocalRefusal:
    """A one-time override that could not be used: lost to another request
    (``override_raced``) or not recorded by the store (``override_fault``)."""
    return "override_fault" if scope is not None and scope.fault else "override_raced"


_NOT_JSON_OBJECT = "the request body is not a JSON object llm-redact can redact"
_CONTENT_ENCODED = (
    "the request body is content-encoded (llm-redact does not decode request bodies;"
    " send it uncompressed)"
)
# RFC 9110 §15.5.16: a 415 over a content coding names the codings accepted.
_IDENTITY_ONLY = {"accept-encoding": "identity"}


def _unscanned_body(
    adapter: ProviderAdapter,
    path: str,
    headers: Headers,
    body: bytes,
    parsed: Any,
    *,
    too_deep: bool = False,
) -> _Unreadable | None:
    """Why a non-empty request body on a matched route must NOT be
    forwarded, or None when the proxy reads — and so redacts — all of it.
    The scanned-body rule: it holds where redaction applies (detection on,
    the client's own key included) and wherever the request spends a
    credential the proxy holds, whatever ``detection`` says.

    Forwardable: a JSON object (what ``prepare_request`` walks; JSON is read
    from the bytes whatever the content-type, a UTF-8 BOM or UTF-16/32
    encoding included), or canonical multipart on a route whose
    ``redact_multipart`` scans it (each part is then scanned, or the
    request refused). Everything else would be forwarded verbatim, and a
    lenient upstream decodes what the proxy never read: non-JSON bytes,
    invalid UTF-8, bytes after the JSON value, a top-level array or scalar
    (``null`` included — never walked), JSON nesting deeper than
    ``MAX_JSON_DEPTH`` (``too_deep``: no walk could read it), whitespace
    only, multipart on any
    other route, and any content-encoded body (415: the proxy never decodes
    one) — every coding of every Content-Encoding header counts, as the
    upstream reads them all. A repeated Content-Type is refused too: it is a
    singleton field, and a second one could name a multipart boundary the
    proxy never parsed with. The route is checked before the multipart
    parse, so a body refused on its route is never parsed. The message
    names the body's KIND only — never its content."""
    if _content_encoded(headers):
        return _Unreadable(415, _CONTENT_ENCODED)
    content_types = headers.getlist("content-type")
    if len(content_types) > 1:
        return _Unreadable(400, "the request carries more than one Content-Type header")
    if too_deep:
        return _Unreadable(400, f"the request body nests JSON deeper than {MAX_JSON_DEPTH} levels")
    if isinstance(parsed, dict):
        return None
    boundary = (
        adapter.multipart_boundary(path, content_types[0])
        if parsed is None and content_types
        else None
    )
    if boundary is None:
        return _Unreadable(400, _NOT_JSON_OBJECT)
    if not adapter.redacts_multipart(path):
        return _Unreadable(
            400, "the multipart body is on a route llm-redact does not redact multipart for"
        )
    if parse_multipart(body, boundary) is None:
        return _Unreadable(
            400, "the multipart body is outside the canonical form llm-redact can redact"
        )
    return None


def _body_overridable(
    state: ProxyState,
    unscanned: _Unreadable,
    proxy_credential: bool,
    body: bytes,
    headers: Headers,
) -> bool:
    """Whether a body the scanned-body rule refuses is a CONTENT refusal its
    requester may override (overrides.py): only a body that is simply not a
    JSON object and cannot be read as JSON by any reader
    (``_plain_text_body``) — never a content coding, a repeated
    Content-Type, JSON nested too deep or a multipart body (framing: two
    readers could disagree) — sent with the client's OWN credential, on a
    proxy that checks no stored-object access (that check reads the body;
    an unread one could cite another user's object)."""
    return (
        state.overrides is not None
        and unscanned.message == _NOT_JSON_OBJECT
        and not proxy_credential
        and not state.checks_object_access
        and _plain_text_body(body, headers)
    )


# Leading characters a lenient JSON reader may skip before a value: every
# Unicode space, byte-order marks and control characters (a NUL-interleaved
# UTF-16/32 body included).
_LENIENT_SKIP = "".join(chr(c) for c in range(0x20) if not chr(c).isspace()) + "\x7f\ufeff\ufffe"
_STRUCTURED_MEDIA = ("multipart", "x-www-form-urlencoded")


def _plain_text_body(body: bytes, headers: Headers) -> bool:
    """Whether a refused body is plain text NO reader takes for a request
    object, so an approval to forward it unscanned can never carry a JSON
    request past redaction: valid UTF-8 whose first character past every
    space, byte-order mark and control character opens neither an object
    nor an array (JSON with one trailing byte or one invalid UTF-8 byte —
    a body a first-value or lenient reader still reads as a chat request —
    is never overridable), sent under no multipart or form media type."""
    media = (headers.get("content-type") or "").lower()
    if any(kind in media for kind in _STRUCTURED_MEDIA):
        return False
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return False
    lead = text
    while True:
        stripped = lead.lstrip().lstrip(_LENIENT_SKIP)
        if stripped == lead:
            break
        lead = stripped
    return lead[:1] not in ("", "{", "[")


def _forward_on_rule(
    scope: OverrideScope,
    rules: list[tuple[str, int]],
    forwarded: list[int],
    route: tuple[str, str, str],
    count: int,
) -> None:
    """``redact_multipart``'s ``forward_binary`` under ``binary_uploads =
    "refuse"`` with the client's own key: told how many binary file parts
    the upload holds once it was read in full, it forwards them only on the
    requester's approved ``binary_upload`` rule for this route (kept in
    ``rules``, used once the request passes), else refuses the upload as a
    binary part without one is refused (``UnscannedBinaryFile``)."""
    rule = scope.route_rule("binary_upload", *route)
    if rule is None:
        raise UnscannedBinaryFile(BINARY_FILE)
    rules.append(rule)
    forwarded.append(count)


def _route_override(
    scope: OverrideScope,
    upload: "_UploadFate",
    kind: str,
    provider: str,
    method: str,
    path: str,
) -> tuple[bool, str | None]:
    """A route refusal put to the requester's overrides: (True, None) when
    an approved one passes it (used now, the request marked), else (False,
    the refusal's hint — None when no code could be minted)."""
    rule = scope.route_rule(kind, provider, method, path)
    if rule is not None:
        scope.use_route(rule)
        if _commit_overrides(scope, upload):
            return True, None
    return False, scope.refusal_hint(kind, provider, method, path)


def _commit_overrides(scope: OverrideScope | None, upload: "_UploadFate") -> bool:
    """Consume what the request used of its requester's overrides, as it
    passes the refusal (``OverrideScope.commit``), and mark the request's
    row; the use is settled with the request's fate (``upload``: a request
    refused before it reaches the upstream hands a one-time grant back).
    False when a one-time grant it relied on was used by another request
    first (or could not be recorded): it must be refused."""
    if scope is None:
        return True
    ok, marker = scope.commit()
    if marker is not None:
        _COMMITTED_OVERRIDE.set(marker)
        upload.hold(scope.settle)
        upload.hold(functools.partial(_mark_override, marker))
    return ok


def _mark_override(marker: str, sent: bool) -> None:
    """The request's rows say it passed on an override only once it is
    handed to the upstream (``_UploadFate`` settled sent)."""
    if sent:
        _REQUEST_OVERRIDE.set(marker)


def _scanned_body_clause(*, identity: bool, proxy_credential: bool) -> str:
    """Why the scanned-body rule holds for this request: whose credential
    it would be sent with."""
    if identity:
        return "this provider is authorized with the proxy's own identity"
    if proxy_credential:
        return "this request would be sent with the proxy's own provider credential"
    return "on this route llm-redact forwards only bodies it has redacted"


def _content_encoded(headers: Headers) -> bool:
    """Whether any coding of any Content-Encoding header is not identity
    (the upstream reads them all; the proxy decodes none)."""
    return any(
        coding.strip().lower() not in ("", "identity")
        for value in headers.getlist("content-encoding")
        for coding in value.split(",")
    )


def _ownership_body(
    adapter: ProviderAdapter,
    path: str,
    headers: Headers,
    body: bytes,
    parsed: Any,
    *,
    proxy_credential: bool,
    scanned: bool,
    max_body_bytes: int,
    max_parts: int,
) -> tuple[Any, bytes | None, _Unreadable | None]:
    """What the stored-object check reads of a non-empty request body on a
    MATCHED route, the body to forward in its place (a checked upload whose
    repeated-key lines were re-serialized), and why a body the proxy would
    send with its OWN credential cannot be checked.

    Under the proxy's credential a body the check cannot read is never
    sent: a content-encoded one, one with more than one Content-Type (a
    second could name a multipart boundary the check never parsed with),
    and what ``upload_view.read_upload`` cannot read. The route's JSON is
    ``parsed``. A multipart/form-data upload is read whatever the
    credential — the lines of an uploaded batch file are requests the
    provider runs later, with the credential the upload is sent with, and a
    form field can name a file; whether the route redacts (``detection``)
    changes nothing here. A single-request upload whose first part is the
    created file's JSON metadata (``adapter.upload_metadata_boundary``: the
    Gemini API's multipart/related upload) is read as the metadata object
    — the create's body, which can choose the file's name — and, like a
    JSON body, one it cannot read is refused wherever the scanned-body rule
    holds (``scanned``: redaction applies, or the proxy's credential is
    spent), not only under the proxy's credential; with the client's own
    key and ``detection = false`` it goes out unchecked, as such a JSON
    body does. (A pass-through route is never sent with the proxy's
    credential — refused before the body is read — so its body is never
    read for the check.) Wherever such a body reaches this check the
    scanned-body rule (``_unscanned_body``) has already refused a content
    coding, a repeated Content-Type and an upload over the parts cap under
    the proxy's credential; the check keeps those refusals of its own, so
    it fails closed whatever runs before it."""
    if proxy_credential:
        if _content_encoded(headers):
            return None, None, _Unreadable(415, _CONTENT_ENCODED)
        if len(headers.getlist("content-type")) > 1:
            return None, None, _Unreadable(400, "the request carries more than one Content-Type")
    if parsed is not None:
        return parsed, None, None
    content_type = headers.get("content-type", "")
    metadata_boundary = adapter.upload_metadata_boundary(path, content_type)
    boundary = metadata_boundary or parse_multipart_boundary(content_type)
    if boundary is None:
        return None, None, None
    if body.count(b"\r\n--" + boundary) > max_parts:
        # More parts than max_body_strings allows: never parsed, here or by
        # redaction (which answers it 413) — a body of many empty parts costs
        # the event loop per part. Under the proxy's credential it cannot be
        # checked, so it is refused.
        too_many = _Unreadable(
            413,
            f"the upload has more parts than llm-redact max_body_strings ({max_parts})",
            kind="too_many_strings",
        )
        return None, None, too_many if proxy_credential else None
    view = (
        read_upload_metadata(body, boundary)
        if metadata_boundary is not None
        else read_upload(body, boundary, max_json_bytes=max_body_bytes, max_lines=max_parts)
    )
    unreadable = None
    if view.oversized:
        unreadable = _Unreadable(
            413,
            f"the upload carries more than llm-redact max_body_bytes ({max_body_bytes})",
            kind="too_large",
        )
    elif view.too_many_lines:
        # Counted before each is parsed: the check never parses more lines
        # than max_body_strings allows (redaction's own line charge).
        unreadable = _Unreadable(
            413,
            f"the upload has more JSON lines than llm-redact max_body_strings ({max_parts})",
            kind="too_many_strings",
        )
    elif view.problem is not None:
        unreadable = _Unreadable(400, view.problem)
    if unreadable is not None:
        if proxy_credential:
            return None, None, unreadable
        if metadata_boundary is not None and scanned:
            # The client's own key where redaction applies: a file create's
            # metadata is refused unread exactly as a JSON body the proxy
            # cannot read is (the scanned-body rule) — a provider reading it
            # leniently could otherwise choose a file name nobody checked.
            return (
                None,
                None,
                unreadable._replace(
                    message=f"{unreadable.message}, and the stored objects a file create"
                    " names must be checked",
                    credential_bound=False,
                ),
            )
        # With the client's own credential the upload goes out as the route
        # sends it, unread (the provider authorizes the client).
        return None, None, None
    if view.normalized is not None and _reclassified(body, view.normalized, boundary):
        # The re-serialized body is what redaction then reads: a part it
        # would read as something else (a text file turned "binary" goes out
        # unscanned) is refused whatever the credential. upload_view never
        # rewrites such a part; this holds even if it did.
        return None, None, _Unreadable(400, _RECLASSIFIED, credential_bound=False)
    return view.cited, view.normalized, None


_RECLASSIFIED = "re-reading the upload for the stored-object check changed what a file part is"


def _reclassified(body: bytes, normalized: bytes, boundary: bytes) -> bool:
    """Whether a part the re-serialized upload ``normalized`` rewrote reads
    as another kind of file than it was (``classify_file``). Only rewritten
    parts are classified — their object lines were each counted by the
    check, and the JSONL test stops at a part's first line that is not one
    — so this costs no more than the check's own bounded reading. A
    different part count, or a body outside the grammar, counts as
    reclassified."""
    before = parse_multipart(body, boundary)
    after = parse_multipart(normalized, boundary)
    if before is None or after is None or len(before.parts) != len(after.parts):
        return True
    return any(
        classify_file(old.content).kind != classify_file(new.content).kind
        for old, new in zip(before.parts, after.parts, strict=True)
        if old.content != new.content
    )


class _Misaddressed(NamedTuple):
    """An unrecognized path that names a route llm-redact recognizes: the
    status to answer, the provider it names and the adapter whose error
    shape answers it (None: a generic shape), the message and the log reason
    (kinds only)."""

    status: int
    provider: str
    adapter: ProviderAdapter | None
    message: str
    why: str


_SPELLING = (
    "llm-redact: the request path must be spelled exactly as the provider's API defines it"
    " (its exact case; no trailing '/', no '\\', no ';' parameters, no trailing spaces, tabs"
    " or dots, no double encoding); this spelling of an API route llm-redact redacts was not"
    " forwarded"
)
_MISSING_VERSION = (
    "llm-redact: this path lacks the API's /v1 segment, so it was not forwarded: an"
    " OpenAI-compatible tool's base URL must end in /v1 (for example"
    " OPENAI_BASE_URL=http://127.0.0.1:8787/v1); an upstream that serves its API without"
    " /v1 is a [providers.custom.NAME] upstream"
)
_EXTRA_PREFIX = (
    "llm-redact: this path carries an extra prefix before an API route llm-redact redacts"
    " (a base URL that repeats the API version, such as .../v1/v1/...), so it was not"
    " forwarded; check the tool's base URL"
)
# The Gemini API's media families: its upload and download endpoints are
# the API's own routes under a leading /upload or /download by design (a
# resumable upload's data chunk, unrecognized, goes to /upload/v1beta/files
# like the recognized upload), never a sign of a misplaced base URL.
_MEDIA_FAMILIES = ("/upload/v1beta/", "/download/v1beta/")
# The longest extra prefix looked for, in segments. A base URL mistake adds
# one or two (/v1/v1/…, /api/v1/…); each candidate tail costs a match of up
# to the whole path, so trying every tail was quadratic in its length.
_MAX_EXTRA_PREFIX = 8
# IIS's non-standard %uXXXX escape.
_PERCENT_U = re.compile(r"%[uU]([0-9A-Fa-f]{4})")
# What IIS trims from the end of a path segment (Windows file-name rules:
# spaces and dots), and the other whitespace a decoded %09/%0B/%0C carries.
_SEGMENT_TRAILER = " \t\r\n\x0b\x0c."


def _normalized_path(path: str) -> str:
    """``path`` (decoded) as a front end that normalizes paths serves it:
    Unicode compatibility forms folded (NFKC: a full-width letter is its
    ASCII one), ``\\`` read as ``/`` (IIS, Azure API Management, Envoy's
    path normalization), and in every segment its ``;params`` dropped
    (Tomcat, Jetty, Spring) and trailing whitespace and dots trimmed (IIS);
    the empty segments that leaves (and a trailing ``/``) merged."""
    text = unicodedata.normalize("NFKC", path).replace("\\", "/")
    segments = (part.split(";", 1)[0].rstrip(_SEGMENT_TRAILER) for part in text.split("/"))
    return "/" + "/".join(segment for segment in segments if segment)


def _decoded_again(path: str) -> str:
    """``path`` percent-decoded once more — IIS's ``%uXXXX`` escapes and a
    ``+`` for a space included — as a gateway that decodes before an app
    server decodes again would read it."""
    unescaped = _PERCENT_U.sub(lambda match: chr(int(match.group(1), 16)), path)
    return urllib.parse.unquote_plus(unescaped)


def _spellings(path: str) -> tuple[str, ...]:
    """The spellings of ``path`` an upstream (or a front end before it) may
    serve as the same route: without a trailing ``/``; normalized
    (``_normalized_path``), as sent and decoded once more; each in its own
    case, lower case and folded the way .NET's OrdinalIgnoreCase compares
    (upper-cased first: a dotless ``ı`` is ``I``). A bounded few, each
    linear in the path's length."""
    stripped = path[:-1] if len(path) > 1 and path.endswith("/") else path
    forms = (stripped, _normalized_path(path), _normalized_path(_decoded_again(path)))
    return tuple(
        dict.fromkeys(
            spelling for form in forms for spelling in (form, form.lower(), form.upper().lower())
        )
    )


def _misaddressed(
    state: ProxyState,
    method: str,
    path: str,
    headers: "Mapping[str, str]",
    query: str,
) -> _Misaddressed | None:
    """Why an UNRECOGNIZED path must not be forwarded although it names a
    route llm-redact recognizes — or None. Forwarded, each would carry its
    body unredacted to an upstream that may serve it as that very route
    (routers that ignore a trailing ``/`` or case: Express, fiber, ASP.NET;
    front ends that normalize paths: IIS, API Management, Tomcat, Envoy):

    - another spelling of the route (``_spellings``) — a trailing ``/``,
      another case, ``\\`` for ``/``, ``;params``, trailing whitespace or
      dots, a second encoding (400; the matched path must be the forwarded
      path, as for dot segments);
    - the route without its ``/v1`` segment, or an OpenAI resource without
      it (404: an OpenAI-compatible base URL that lacks ``/v1``);
    - the route under an extra leading prefix of up to ``_MAX_EXTRA_PREFIX``
      segments (404: a base URL repeating the version). Azure's ``/openai/…``
      and custom ``/custom/NAME/…`` paths embed OpenAI routes by design, and
      their adapters read their own tails; an Azure tail is never taken as a
      sign of an extra prefix. Nor are the Gemini API's ``/upload/v1beta/…``
      and ``/download/v1beta/…`` media families (``_MEDIA_FAMILIES``).

    Only a path one of whose spellings matches a route is refused: a Gemini
    ``:method`` or a Bedrock ARN is matched as sent, and a spelling of a
    pass-through route stays pass-through. It costs a bounded number of
    route matches, each linear in the path's length: it runs before most
    refusals, for any client.
    """
    spellings = _spellings(path)
    for candidate in spellings:
        if candidate == path:
            continue
        adapter, _ = state.route(method, candidate, headers, query)
        if adapter is not None:
            return _Misaddressed(
                400, adapter.name, adapter, _SPELLING, "spelling of a recognized route"
            )
    if path_family(path) is None and not under(path.lower(), "/v1"):
        for candidate in spellings:
            versioned = "/v1" + candidate
            adapter, _ = state.route(method, versioned, headers, query)
            if adapter is not None:
                return _Misaddressed(
                    404, adapter.name, adapter, _MISSING_VERSION, "path without /v1"
                )
            if any(under(versioned, prefix) for prefix in OPENAI_PREFIXES):
                return _Misaddressed(404, "openai", None, _MISSING_VERSION, "path without /v1")
    if path.startswith((CUSTOM_ROUTE_PREFIX, "/openai/", *_MEDIA_FAMILIES)):
        return None
    for candidate in spellings:
        segments = candidate.split("/")
        for index in range(2, min(len(segments), _MAX_EXTRA_PREFIX + 2)):
            tail = "/" + "/".join(segments[index:])
            if tail.startswith("/openai/"):
                # Azure's family name is a segment of other providers'
                # OpenAI-compatible base paths too (Gemini /v1beta/openai/,
                # Groq /openai/v1): never a sign of a misplaced base URL.
                continue
            adapter, _ = state.route(method, tail, headers, query)
            if adapter is not None:
                return _Misaddressed(404, adapter.name, adapter, _EXTRA_PREFIX, "extra path prefix")
    return None


def _misaddressed_refused(
    state: ProxyState,
    misaddressed: _Misaddressed,
    *,
    request: Request,
    path: str,
    started: float,
) -> JSONResponse:
    """A recorded, provider-shaped refusal of a misaddressed path (never
    forwarded; no upstream contact)."""
    state.record_request(
        session=state.config.vault.session,
        provider=misaddressed.provider,
        method=request.method,
        path=path,
        status=misaddressed.status,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal="misaddressed",
    )
    logger.info(
        "%s %s -> %d refused (%s)", request.method, path, misaddressed.status, misaddressed.why
    )
    body = (
        misaddressed.adapter.error_body(misaddressed.message, status=misaddressed.status)
        if misaddressed.adapter is not None
        else {
            "type": "error",
            "error": {"type": "not_found_error", "message": misaddressed.message},
        }
    )
    return JSONResponse(body, status_code=misaddressed.status)


def _method_override_refused(
    state: ProxyState,
    adapter: ProviderAdapter | None,
    kind: str,
    *,
    provider_name: str,
    request: Request,
    path: str,
    started: float,
) -> JSONResponse:
    """A matched route carrying a method override (``method_override``) —
    or, when the access gate authorizes requests, any route (pass-through:
    adapter None): a recorded, provider-shaped 400 — never forwarded, no
    upstream contact."""
    state.record_request(
        session=state.config.vault.session,
        provider=provider_name,
        method=request.method,
        path=path,
        status=400,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal="method_override",
    )
    logger.info("%s %s -> 400 refused (method override %s)", request.method, path, kind)
    if adapter is None:
        message = (
            f"llm-redact: the request carries an HTTP method override {kind}; this proxy"
            " authorizes each request by its own method, so it refuses one asking the"
            " upstream to run another. Send the request with the method it means; the"
            " request was not forwarded"
        )
        return JSONResponse({"error": message}, status_code=400)
    message = (
        f"llm-redact: the request carries an HTTP method override {kind}; llm-redact reads"
        f" a {adapter.name} API request by its own method, so it refuses one asking the"
        " provider to run another. Send the request with the method it means; the request"
        " was not forwarded"
    )
    return JSONResponse(adapter.error_body(message, status=400), status_code=400)


def _unattributed_refused(
    state: ProxyState, request: Request, *, path: str, started: float
) -> JSONResponse:
    """No provider can be attributed (``providers.attribution.attribute``):
    a recorded local 404 — forwarding to a guessed provider would hand it
    another provider's credential and the client's unredacted content. The
    message names the path (never the query) and the marker KINDS."""
    reason = unattributed_reason(request.headers, request.url.query)
    state.record_request(
        session=state.config.vault.session,
        provider=None,
        method=request.method,
        path=path,
        status=404,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal="unattributed",
    )
    logger.info("%s %s -> 404 no provider attributable (%s)", request.method, path, reason)
    message = (
        f"llm-redact: {request.method} {path} is not an API path llm-redact can attribute to a"
        f" provider ({reason}), so it was not forwarded; see docs/providers.md for each"
        " tool's base URL"
    )
    return JSONResponse(
        {"type": "error", "error": {"type": "not_found_error", "message": message}},
        status_code=404,
    )


def _unrecognized_route_refused(
    state: ProxyState,
    request: Request,
    provider_name: str,
    holder: str,
    *,
    path: str,
    started: float,
) -> JSONResponse:
    """A credential the PROXY holds — its cloud identity, or a routed
    upstream's own key (or none) — is lent only to the API routes
    llm-redact recognizes (and redacts): a recorded 403 before the body is
    read, any audit row, credential fetch or upstream contact."""
    message = (
        f"llm-redact: {holder}; only the API routes llm-redact recognizes are forwarded"
        " with a credential the proxy holds"
    )
    state.record_request(
        session=state.config.vault.session,
        provider=provider_name,
        method=request.method,
        path=path,
        status=403,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal="identity_route",
    )
    logger.info(
        "%s %s -> 403 unrecognized route for a credential the proxy holds", request.method, path
    )
    return JSONResponse({"error": message}, status_code=403)


def _credential_protocol_refused(
    state: ProxyState,
    adapter: ProviderAdapter,
    reason: str,
    *,
    provider_name: str,
    request: Request,
    path: str,
    started: float,
) -> JSONResponse:
    """A recognized route the adapter will not serve with a credential the
    PROXY holds (``ProviderAdapter.proxy_credential_refusal``): a recorded,
    provider-shaped 403 before redaction, the audit START row, a plan's
    begin() and any upstream contact. ``reason`` names the protocol only."""
    message = (
        f"llm-redact: {reason}; this request would be sent with a credential the proxy holds,"
        " so it was not forwarded"
    )
    state.record_request(
        session=state.config.vault.session,
        provider=provider_name,
        method=request.method,
        path=path,
        status=403,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal="credential_protocol",
    )
    logger.info(
        "%s %s -> 403 protocol not served with a credential the proxy holds", request.method, path
    )
    return JSONResponse(adapter.error_body(message, status=403), status_code=403)


class _CountWindow:
    """One request's share of the process-wide detection and warn-mode
    counts: the totals as its redaction starts (``detections``,
    ``warned``), diffed when it ends (``_count_delta``). Redaction is
    synchronous, so nothing else counts inside the window — except across
    an await: an upload's inspection may take minutes while other requests
    redact, so it ``restart``s the window once it returns (the diff trick
    corrupts across awaits; realtime counts per connection for that
    reason)."""

    def __init__(self, state: ProxyState) -> None:
        self._state = state
        self.restart()

    def restart(self) -> None:
        self.detections = dict(self._state.detection_counts)
        self.warned = dict(self._state.warn_counts)


async def _inspect_upload(
    state: ProxyState,
    request: Request,
    adapter: ProviderAdapter,
    path: str,
    body: bytes,
    boundary: bytes,
    redactor: Redactor,
    *,
    provider_name: str,
    identity: bool,
    max_body_bytes: int,
    max_body_strings: int,
    outcomes: Counter[str],
    window: _CountWindow,
    before_inspection: Callable[[], None],
) -> tuple[InspectedUpload | None, dict[str, int]]:
    """An upload's binary file parts read as text by the upload inspector
    and judged (``upload_inspection``): the adapter's reading of ``body``
    (parsed and classified once, every part's headers checked) with the
    parts cleared to go out byte-identical — None when the route reads no
    file parts — and the token floors of the extracted texts (a placeholder
    a file carries inside a compressed stream bounds this request's new
    numbers). With a rule in block mode, every string the redaction will
    scan is checked for one first (``_block_check``): an upload it refuses
    is never handed to the inspector — nor one ``before_inspection`` refuses
    (it raises ``_RefusedBeforeInspection``: the local refusals that do not
    need the redacted bytes and the ``[audit] required`` START row, applied
    only for an upload with a binary part to inspect). Each part's outcome is added to
    ``outcomes`` — counted by the caller once it knows whether the upload
    went out (``_count_inspections``). The inspection is the one await inside the
    request's count ``window``: it restarts once the inspector returns,
    before the extracted texts are scanned (their warn-mode values are this
    request's). Raises BlockedRequest or BinaryValuesDetected
    for a value found in an extracted text, and what reading the upload
    raises (UnredactableRequest, TooManyStrings) — all before anything is
    written or sent."""
    inspector = state.upload_inspector
    limits = state.inspection_limits
    assert inspector is not None and limits is not None
    reading = adapter.read_multipart(path, body, boundary, redactor.charge)
    if reading is None:
        return None, {}
    parts = reading.binary_parts()
    if not parts:
        return InspectedUpload(reading), {}
    if redactor.blocks:
        # The inspector may send a file to a service: a block-mode value in
        # another part or a file name refuses the request before that.
        reading.require_unblocked(_block_check(redactor, max_body_strings))
    # A request refused anyway never sends a file to the inspector (which may
    # send it off the machine): no upstream configured, a routed local
    # refusal, a START row [audit] required cannot commit.
    before_inspection()
    results = await inspect_parts(
        inspector, parts, provider=provider_name, identity=identity, limits=limits
    )
    # Other requests redacted while this one waited: their counts are theirs.
    window.restart()
    verdict = judge(
        results,
        redactor,
        identity=identity,
        text_budget=max_body_bytes,
        convertible=adapter.converts_upload(path, reading),
    )
    outcomes.update(verdict.outcomes)
    # Counts and outcomes only — never a file name, a type found or content.
    logger.info(
        "%s %s inspected %d binary upload file part(s): %s",
        request.method,
        path,
        len(parts),
        " ".join(f"{outcome}={count}" for outcome, count in sorted(verdict.outcomes.items())),
    )
    if verdict.over_budget is not None:
        raise verdict.over_budget  # every part's outcome counted first
    if verdict.blocked is not None:
        raise BlockedRequest(verdict.blocked)
    if verdict.detected:
        raise BinaryValuesDetected(verdict.detected)
    return InspectedUpload(reading, verdict.cleared, verdict.converted), verdict.floors


class _RefusedBeforeInspection(Exception):
    """``before_inspection`` refused the upload: ``response`` (already
    recorded) is the answer, and no part was handed to the inspector."""

    def __init__(self, response: Response) -> None:
        super().__init__("refused before inspection")
        self.response = response


def _block_check(redactor: Redactor, limit: int) -> Callable[[str], str]:
    """``redactor``'s block-mode rules as a read-only check of one string
    (``UploadReading.require_unblocked``): BlockedRequest (the type only)
    for a string ``redact_text`` would refuse, nothing issued or counted.
    Counted against a budget of its own of ``limit`` strings
    (TooManyStrings, as the redaction — which charges the same strings
    again — would refuse), so it never walks more than the redaction may."""
    budget = StringBudget(limit)

    def check(text: str) -> str:
        budget.charge(1)
        blocked = redactor.blocked_type(text)
        if blocked is not None:
            raise BlockedRequest(blocked)
        return text

    return check


def _count_inspections(
    state: ProxyState, provider_name: str, outcomes: Counter[str], *, sent: bool
) -> None:
    """One upload's inspected binary parts into ``inspected_uploads``, once
    its fate is known: ``clean`` only when the upload was handed to the
    upstream (``sent``) — a part that scanned clean in an upload the proxy
    refused (a value in another part, a block, a header rule, a credential
    the proxy holds that the inspection did not allow, or any refusal after
    redaction: the upstream authorizer, a routed budget) is
    ``clean_refused``, never reported as forwarded — and a part converted to
    its redacted text (convert mode) ``converted`` or ``converted_refused``
    alike, as is one clean only because an override let its values through
    (``overridden`` / ``overridden_refused``). (No upstream configured and a
    failed ``[audit] required`` START row refuse before the inspection:
    nothing is inspected, so nothing is counted.)"""
    for outcome, count in outcomes.items():
        if outcome in ("clean", "converted", "overridden") and not sent:
            outcome = f"{outcome}_refused"
        state.inspected_uploads[(provider_name, outcome)] += count


class _UploadFate:
    """One request's upload honesty counts — its inspected binary parts'
    outcomes and the binary parts it forwards unscanned — and the text
    parts its downloads must restore raw (``remember_raw_texts``), held
    until its fate is known and settled ONCE: ``sent`` when the request is handed to
    the upstream (a send that then fails in transit included: bytes may
    have left), refused on every other way out of ``handle()`` (its
    ``finally``), so a refusal after redaction never counts a part as
    forwarded."""

    def __init__(self) -> None:
        self._pending: list[Callable[[bool], None]] = []

    def hold(self, settle: Callable[[bool], None]) -> None:
        # The upload's counts, and the one-time overrides the request used
        # (OverrideScope.settle: handed back when it is refused instead).
        self._pending.append(settle)

    def settle(self, *, sent: bool) -> None:
        pending, self._pending = self._pending, []
        for settle in pending:
            settle(sent)


def _settle_upload(
    state: ProxyState,
    method: str,
    path: str,
    provider_name: str,
    outcomes: Counter[str],
    unscanned: list[int],
    raw_texts: list[bytes],
    sent: bool,
) -> None:
    """An upload's counts once its fate is known (``_UploadFate``): the
    inspected parts (``_count_inspections``) and, only when it went out,
    the text parts redacted as one text (remembered for their download:
    ``remember_raw_texts`` — a refused request never evicts what sent ones
    remembered) and the binary parts forwarded unscanned — a count and the
    path only, never a file name or content."""
    _count_inspections(state, provider_name, outcomes, sent=sent)
    if sent:
        remember_raw_texts(raw_texts)
    if sent and unscanned:
        state.unscanned_uploads[provider_name] += unscanned[0]
        logger.info(
            "%s %s forwarded %d binary upload file part(s) unscanned"
            ' ([detection] binary_uploads = "forward")',
            method,
            path,
            unscanned[0],
        )


def _plan_request(
    router: Router,
    request: Request,
    adapter: ProviderAdapter | None,
    provider_name: str,
    path: str,
    model: str | None,
) -> RoutePlan | RouteRefusal | None:
    """Ask the routing layer to plan this request (side-effect free until
    ``begin``)."""
    return router.plan(
        RouteInbound(
            adapter_name=adapter.name if adapter is not None else None,
            provider_name=provider_name,
            method=request.method,
            path=path,
            raw_path=_upstream_path(request, path),
            query=request.url.query,
            headers=request.headers,
            model=model,
        )
    )


async def handle(request: Request) -> Response:
    """The catch-all route (``_handle``). An upload's honesty counts are
    settled once its fate is known: sent when handed to the upstream, else
    refused — here, on every other way out (``_UploadFate``). A START row an
    upload wrote before its inspection (``_EarlyAudit``) that no recorded
    row finalized is closed here with the answer's status (None: an
    exception), so it never stays without its END row."""
    upload = _UploadFate()
    early_audit = _EarlyAudit()
    reset = _EARLY_AUDIT.set(early_audit)
    # Never reset: a stream's finalizer, after handle() returned, reads it.
    _REQUEST_TIMING.set(_RequestTiming())
    status: int | None = None
    try:
        response = await _handle(request, upload)
        status = response.status_code
        return response
    finally:
        _EARLY_AUDIT.reset(reset)
        upload.settle(sent=False)
        early_audit.close(request.app.state.proxy, status)


async def _handle(request: Request, upload: _UploadFate) -> Response:
    state: ProxyState = request.app.state.proxy
    if not origin_form_target(request.scope):
        # Never routed, forwarded, recorded or logged with its target.
        state.count_local_refusal("request_target", None)
        return JSONResponse({"error": "the request target must be a path"}, status_code=400)
    if has_dot_segment(request.scope):
        # Refused before routing, admission or any upstream contact (and,
        # like the target check above, never recorded or logged: the path
        # may still hold an identity-prefix key).
        state.count_local_refusal("request_target", None)
        return JSONResponse(
            {"error": "the request path must not contain '.' or '..' segments"}, status_code=400
        )
    if has_empty_segment(request.scope):
        # Refused like a dot segment, and for the same reason: matching and
        # forwarding must address the same resource (never recorded or
        # logged with its path — it may still hold an identity-prefix key).
        logger.info("%s -> 400 empty path segment (path withheld)", request.method)
        state.count_local_refusal("request_target", None)
        return JSONResponse(
            {
                "error": "the request path must not contain an empty segment ('//');"
                " check the tool's base URL for a trailing '/'"
            },
            status_code=400,
        )
    path = request.url.path

    # Reserved local endpoints are answered here, before any routing or
    # upstream code runs — this early return is the non-forwarding guarantee.
    if path.startswith(RESERVED_PREFIX):
        response, admission = await _admit_reserved(request, state)
        if response is None:
            response = await _handle_local(request, state, admission)
        # Stamp browser-hardening headers on every reserved reply in one place
        # (setdefault so a handler that set its own header still wins).
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    # Whether the client presented a credential for the PROXY itself (an
    # identity path prefix, an x-llm-redact-* header). Read from the raw
    # scope before admission, which scrubs it: the client still holds it,
    # and would re-send it wherever a relayed upstream redirect pointed.
    presented_credential = path.startswith(IDENTITY_PATH_PREFIX) or any(
        name.lower().startswith(OWN_HEADER_PREFIX.encode())
        for name, _ in request.scope.get("headers") or ()
    )

    # Client admission (llm-redact-pro's access gate): it removes every
    # credential it recognizes from the scope before anything else reads
    # the path or headers. Its refusal, if any, is applied further down.
    admission = await state.admit(request, "http")
    presented_credential = presented_credential or admission.subject is not None
    path = request.scope["path"]
    if path.startswith(RESERVED_PREFIX):
        # Only reachable through a stripped prefix (/u/<key>/__llm-redact/…):
        # reserved endpoints are served at their own path, never forwarded.
        state.count_local_refusal("identity_path", None)
        return JSONResponse({"error": "not found"}, status_code=404)
    if path.startswith(IDENTITY_PATH_PREFIX):
        # An identity prefix no gate claimed: its next segment is a key, so
        # the request is neither forwarded nor recorded (both would carry it).
        logger.info("%s /u/... -> 404 unclaimed identity path prefix", request.method)
        state.count_local_refusal("identity_path", None)
        return JSONResponse(
            {"error": "this proxy does not accept /u/<key>/ identity paths (no access gate)"},
            status_code=404,
        )
    _REQUEST_USER.set(admission.subject)
    _REQUEST_OVERRIDE.set(None)
    _COMMITTED_OVERRIDE.set(None)

    # Captured once per request; read at finalization (incl. the streaming
    # finalizer, same task context) so an OTel span can parent into the
    # caller's trace. Trivial when telemetry is off; traceparent isn't secret.
    _INBOUND_TRACEPARENT.set(request.headers.get("traceparent"))
    started = time.perf_counter()
    query = request.url.query
    adapter, kind = state.route(request.method, path, request.headers, query)

    # A disabled provider fails closed before anything is read or forwarded:
    # matched routes AND pass-through traffic attributed to it are answered
    # here (forwarding pass-through would send unredacted bodies to it). A
    # request no provider can be attributed to is answered locally too.
    provider_name = state.provider_for(adapter, path, request.headers, query)
    provider_conf = state.config.providers.get(provider_name) if provider_name else None
    # Read ONCE, with the provider's config and before the first await (the
    # body read): a reload applied while the body is still arriving must not
    # change what this request was admitted as. Every identity decision
    # below — the unrecognized-route 403, the ownership check, the body
    # rule, stripping and signing — and the upstream it goes to use these
    # two snapshots; the reload's displaced authorizer stays with the
    # requests that already hold it.
    upstream_auth = state.upstream_auth.get(provider_name) if provider_name else None
    # A web page's request (CSRF, DNS rebinding) never reaches an upstream:
    # refused before anything below answers, reads the body or signs. A
    # routed plan that spends a key the proxy holds is checked again once
    # planned (below).
    origin_refusal = request_origin_refusal(
        request, state, lends_credential=upstream_auth is not None
    )
    if origin_refusal is not None:
        return _request_origin_refused(
            state,
            adapter,
            origin_refusal,
            provider_name=provider_name,
            request=request,
            path=path,
            started=started,
        )
    if path == "/" and request.method in ("GET", "HEAD"):
        # The proxy's base URL itself is no provider's API: answered here (a
        # client's liveness probe — the ollama CLI's HEAD / heartbeat),
        # never forwarded or recorded. After the request-origin check, like
        # every answer to a path outside the reserved prefix.
        return Response(
            b"llm-redact is running\n", media_type="text/plain", headers=dict(_SECURITY_HEADERS)
        )
    if adapter is None and admission.refusal is None:
        # An unrecognized path that is another spelling of a recognized
        # route, or that route without its /v1 or under an extra prefix:
        # refused, never forwarded unredacted to an upstream that may serve
        # it as that route. Looked for only once the request-origin rule and
        # the access gate admit the request: a refused one (a web page's, an
        # unknown client's) gets its refusal without this routing work.
        misaddressed = _misaddressed(state, request.method, path, request.headers, query)
        if misaddressed is not None:
            return _misaddressed_refused(
                state, misaddressed, request=request, path=path, started=started
            )
    if provider_name is None:
        return _unattributed_refused(state, request, path=path, started=started)
    if provider_conf is None:
        # /custom/<name>/ with no [providers.custom.<name>] entry: there is
        # nowhere sane to forward, and guessing would leak.
        state.record_request(
            session=state.config.vault.session,
            provider=provider_name,
            method=request.method,
            path=path,
            status=502,
            started=started,
            streamed=False,
            detections={},
            rehydrations={},
            refusal="no_upstream",
        )
        logger.info("%s %s -> 502 unknown custom provider", request.method, path)
        return JSONResponse(
            {
                "error": f"no [providers.custom.{provider_name.removeprefix('custom:')}]"
                " upstream is configured"
            },
            status_code=502,
        )
    if not provider_conf.enabled:
        message = (
            f"the {provider_name} provider is disabled in llm-redact config"
            f" ([providers.{provider_name}] enabled = false)"
        )
        error = (
            adapter.error_body(message, status=502) if adapter is not None else {"error": message}
        )
        state.record_request(
            session=state.config.vault.session,
            provider=provider_name,
            method=request.method,
            path=path,
            status=502,
            started=started,
            streamed=False,
            detections={},
            rehydrations={},
            refusal="disabled_provider",
        )
        logger.info("%s %s -> 502 provider %s disabled", request.method, path, provider_name)
        return JSONResponse(error, status_code=502)

    # The admitted config's upstream, not a reload's (the legacy path's).
    upstream_base = provider_conf.upstream_base_url

    if admission.refusal is not None:
        # The access gate's refusal (llm-redact-pro), applied after the
        # disabled-provider 502 and recorded like every other proxy-generated
        # response. The gate's message never echoes a presented credential.
        message = admission.refusal
        error = (
            adapter.error_body(message, status=403) if adapter is not None else {"error": message}
        )
        state.record_request(
            session=state.config.vault.session,
            provider=provider_name,
            method=request.method,
            path=path,
            status=403,
            started=started,
            streamed=False,
            detections={},
            rehydrations={},
            refusal="access_gate",
        )
        logger.info("%s %s -> 403 refused by the access gate", request.method, path)
        return JSONResponse(error, status_code=403)

    # A matched route is read by the request line's method, and so is every
    # route the access gate authorizes (its `method` fact): an upstream that
    # honors an override would run another (a create turned into a
    # listing). Without the gate's authorize_request, pass-through forwards
    # the override as sent.
    override = (
        method_override(request.headers, query)
        if adapter is not None or state.authorization.authorizes
        else None
    )
    if override is not None:
        # Refused before the body is read, any credential or upstream
        # contact; the message names the KIND only.
        return _method_override_refused(
            state,
            adapter,
            override,
            provider_name=provider_name,
            request=request,
            path=path,
            started=started,
        )

    if adapter is None and upstream_auth is not None:
        # The proxy lends its own cloud identity only to the API routes it
        # recognizes (and redacts). Signing pass-through traffic would hand
        # any client the whole cloud API as the proxy's principal, bodies
        # unredacted — so an unrecognized path is refused here, never sent.
        return _unrecognized_route_refused(
            state,
            request,
            provider_name,
            f"{provider_name} is authorized with the proxy's own identity"
            f' ([providers.{provider_name}] auth = "identity")',
            path=path,
            started=started,
        )

    if state.router is not None:
        answer = state.router.local_answer(request.method, path, request.headers)
        if answer is not None:
            # R-15 model discovery: a local answer (never forwarded), placed
            # where every other proxy-generated reply for a real API path sits —
            # AFTER the disabled-provider 502 and the named-user 403, so an
            # unauthenticated client on a team deployment learns nothing the
            # gates would refuse. It needs nothing from the body — and it is
            # given only once the access gate's request authorization allows
            # it, like any request (nothing read, nothing forwarded).
            if state.authorization.authorizes:
                local_asked = _authorization_request(
                    request,
                    adapter,
                    kind,
                    provider_name=provider_name,
                    path=path,
                    model=adapter.request_model(request.method, path, None)
                    if adapter is not None
                    else None,
                    identity=upstream_auth is not None,
                )
                refused = await _authorization_check(
                    state,
                    request,
                    local_asked,
                    adapter,
                    provider_name=provider_name,
                    path=path,
                    started=started,
                )
                if refused is not None:
                    return refused
            return _answer_locally(state, request, answer, path=path, started=started)

    # Routing (the llm-redact-pro routing layer) plans an UNRECOGNIZED route
    # here, before its body is read — there is no model to read from it — so
    # a plan that would spend a credential the proxy holds refuses it before
    # the unbounded body read, the audit START row, begin() and any hop. A
    # recognized route is planned once its body is parsed (below).
    plan: RoutePlan | None = None
    if adapter is None and state.router is not None and upstream_auth is None:
        planned = _plan_request(state.router, request, None, provider_name, path, None)
        if isinstance(planned, RouteRefusal):
            # No rule and no default for this protocol: proxy-generated
            # 502, never forwarded by guesswork (decision 2).
            return _route_refusal(
                state,
                state.config.vault.session,
                adapter,
                planned,
                request=request,
                path=path,
                started=started,
                kind="no_route",
            )
        plan = planned
        if plan is not None and _lends_credential(plan):
            # An operator key (or no key at all) is the identity case again:
            # lent to an unrecognized route it would carry any client's
            # unredacted body anywhere on the provider's API as the operator
            # (the [auth] broker shape) — refused, never forwarded.
            return _unrecognized_route_refused(
                state,
                request,
                provider_name,
                f"this request would reach {provider_name} with a credential the proxy holds"
                " (a routed upstream's own key, or none)",
                path=path,
                started=started,
            )

    # This request's facts as the access gate is asked about it — its
    # authorize_request and, after redaction, its authorize_content get the
    # SAME object. Built only when the gate has either member.
    asked: AuthorizationRequest | None = None
    # The body caps this request is held to, read once like its provider
    # config: a reload while the body arrives changes neither.
    max_body_bytes = state.config.max_body_bytes
    max_body_strings = state.config.max_body_strings
    if adapter is not None:
        # Redactable routes fail closed on oversized bodies: the proxy must
        # buffer the whole body to redact it, and forwarding unredacted is
        # never acceptable. Pass-through routes below are unaffected.
        capped = await _waited(_read_capped(request, max_body_bytes))
        if capped is None:
            return _body_too_large(
                state,
                request,
                adapter,
                session=state.config.vault.session,
                path=path,
                started=started,
                cap="max_body_bytes",
                limit=max_body_bytes,
            )
        body_bytes = capped
    else:
        if state.authorization.authorizes or state.authorization.checks_content:
            asked = _authorization_request(
                request,
                None,
                kind,
                provider_name=provider_name,
                path=path,
                model=None,
                identity=False,
            )
        if asked is not None and state.authorization.authorizes:
            # Pass-through: its facts are all known before the body (none of
            # it is read for them, no model), so the access gate decides
            # before the unbounded read — and with the client's own credential
            # only (one the proxy holds was refused above, the identity and
            # lending-plan 403s).
            refused = await _authorization_check(
                state,
                request,
                asked,
                None,
                provider_name=provider_name,
                path=path,
                started=started,
            )
            if refused is not None:
                return refused
        body_bytes = await _waited(request.body())

    parsed: Any = None
    # A repeated JSON key: the parse keeps the last occurrence, so the walk
    # never sees the earlier ones — such a body is always re-serialized.
    duplicate_keys = False
    # JSON nesting deeper than any walk may recurse: unreadable, like any
    # other body the proxy cannot read (the scanned-body rule below).
    too_deep = False
    if adapter is not None and body_bytes:
        try:
            parsed, duplicate_keys = loads_request(body_bytes)
        except JsonTooDeep:
            too_deep = True
        except ValueError:
            parsed = None

    # Routing (the llm-redact-pro routing layer): the router plans BEFORE the
    # stored-object check below — the check must know whose credential the
    # request will spend (a routed upstream may send the operator's own key)
    # — and before redaction, because the FIRST upstream's
    # inject_system_note governs the prepared body (decision 4; the redacted
    # body is reused on later hops). A plan is side-effect free until
    # begin(): one abandoned by a refusal below is simply dropped. An
    # unrouted request — no router held, or a provider outside the router's
    # protocols (plan() returns None) — never constructs a routing object and
    # takes the legacy path below byte-for-byte.
    # A provider the proxy authorizes with its own cloud identity is never
    # routed (the routing protocols are anthropic/openai/gemini/ollama, and
    # a routed hop carries the router's own credentials): the router is not
    # even asked, so the request below is always signed by its authorizer.
    if adapter is not None and state.router is not None and upstream_auth is None:
        model = parsed.get("model") if isinstance(parsed, dict) else None
        planned = _plan_request(
            state.router,
            request,
            adapter,
            provider_name,
            path,
            model if isinstance(model, str) else None,
        )
        if isinstance(planned, RouteRefusal):
            # No rule and no default for this protocol: proxy-generated
            # 502, never forwarded by guesswork (decision 2) — before
            # redaction, like the router decided it.
            return _route_refusal(
                state,
                state.config.vault.session,
                adapter,
                planned,
                request=request,
                path=path,
                started=started,
                kind="no_route",
            )
        plan = planned

    # Whose credential the provider will see. The proxy's OWN — its cloud
    # identity, or a routed plan's operator key (RoutePlan.proxy_credential;
    # a plan without the member counts as the proxy's: fail closed) — makes
    # every client the same principal upstream, so the stored-object check
    # applies its identity policy.
    proxy_credential = upstream_auth is not None or (plan is not None and _lends_credential(plan))
    if proxy_credential and upstream_auth is None:
        # A routed plan spends a key the proxy holds (or none at all): the
        # host-name rule, applied above to browser requests only, now holds
        # for every client — before the plan begins, redaction, or any hop.
        origin_refusal = request_origin_refusal(request, state, lends_credential=True)
        if origin_refusal is not None:
            return _request_origin_refused(
                state,
                adapter,
                origin_refusal,
                provider_name=provider_name,
                request=request,
                path=path,
                started=started,
            )
    if proxy_credential and adapter is not None:
        credential_refusal = adapter.proxy_credential_refusal(
            request.method, path, request.headers, query
        )
        if credential_refusal is not None:
            # A recognized route whose protocol cannot be served with the
            # proxy's credential (its answer would hand the client a
            # capability minted under it): refused before redaction, the
            # plan's begin() and any upstream contact.
            return _credential_protocol_refused(
                state,
                adapter,
                credential_refusal,
                provider_name=provider_name,
                request=request,
                path=path,
                started=started,
            )
    detection_off = provider_conf is not None and not provider_conf.detection
    if adapter is not None and body_bytes and (proxy_credential or not detection_off):
        # The scanned-body rule: a matched route forwards only a body the
        # proxy read — where redaction applies (detection on, the client's
        # own key included) and wherever the proxy's own credential is spent
        # (detection = false turns redaction off, never this rule: the
        # ownership check and object tracking read the parsed body too). A
        # body it cannot read would go out verbatim, and a lenient upstream
        # decodes it anyway (inflates gzip, takes the first JSON value,
        # substitutes bad bytes). Refused here, before the stored-object
        # check, the session, redaction, any credential, the plan's begin()
        # and any upstream contact. detection = false with the client's own
        # key, and unmatched pass-through (only ever sent with the client's
        # own credential: refused above under one the proxy holds), still
        # forward such bodies verbatim.
        if parsed is None and _multipart_parts_over(
            request.headers, body_bytes, max_body_strings, adapter, path
        ):
            # More parts than max_body_strings allows: refused before any
            # parse — a body of many empty parts costs the event loop per
            # part, not per byte. Like max_body_bytes, the cap holds on every
            # route this rule covers.
            return _body_too_large(
                state,
                request,
                adapter,
                session=state.config.vault.session,
                path=path,
                started=started,
                cap="max_body_strings",
                limit=max_body_strings,
            )
        unscanned = _unscanned_body(
            adapter, path, request.headers, body_bytes, parsed, too_deep=too_deep
        )
        hint: str | None = None
        if unscanned is not None and _body_overridable(
            state, unscanned, proxy_credential, body_bytes, request.headers
        ):
            # A body that is simply not a JSON object, with the client's own
            # key and no stored-object check: its requester may have
            # approved forwarding it unscanned on this route (overrides.py).
            body_scope = state.override_scope()
            assert body_scope is not None  # _body_overridable: overrides are on
            passed, hint = _route_override(
                body_scope, upload, "unscanned_body", provider_name, request.method, path
            )
            if passed:
                logger.info(
                    "%s %s forwarded unscanned (a body that is not a JSON object; override)",
                    request.method,
                    path,
                )
                unscanned = None
                # Forwarded exactly as sent: nothing below reads it as JSON.
                parsed = None
        if unscanned is not None:
            return _unscanned_body_refused(
                state,
                adapter,
                unscanned,
                clause=_scanned_body_clause(
                    identity=upstream_auth is not None, proxy_credential=proxy_credential
                ),
                provider_name=provider_name,
                request=request,
                path=path,
                started=started,
                hint=hint,
            )
    check_body: Any = parsed
    # An upload whose checked lines repeat a key, re-serialized: what a
    # route that forwards the body unredacted (detection = false) sends.
    checked_upload: bytes | None = None
    if adapter is not None and body_bytes and state.checks_object_access:
        # What the check reads: a matched route's JSON, an upload's lines
        # and form fields (redaction or not — ownership is access control).
        # A body the check cannot read is never sent with the proxy's
        # credential. (A pass-through body is never read: under the proxy's
        # credential an unrecognized route was refused above.)
        check_body, checked_upload, unreadable = _ownership_body(
            adapter,
            path,
            request.headers,
            body_bytes,
            parsed,
            proxy_credential=proxy_credential,
            scanned=proxy_credential or not detection_off,
            max_body_bytes=max_body_bytes,
            max_parts=max_body_strings,
        )
        if unreadable is not None:
            return _unchecked_body_refused(
                state,
                adapter,
                unreadable,
                provider_name=provider_name,
                request=request,
                path=path,
                started=started,
                identity=upstream_auth is not None,
            )
    refusal = state.object_access_refusal(
        adapter.name if adapter is not None else None,
        request.method,
        path,
        check_body,
        identity=proxy_credential,
    )
    if refusal is not None:
        # The session router (llm-redact-pro named users) refused a request
        # that reaches another namespace's stored object: answered here,
        # before the audit START row, redaction, any upstream credential
        # and any upstream contact — routed or not. The reason is the
        # router's fixed text, never an id.
        return _object_access_refused(
            state,
            adapter,
            refusal,
            provider_name=provider_name,
            request=request,
            path=path,
            started=started,
        )

    # The access gate's authorization (llm-redact-pro roles) of a MATCHED
    # route: asked with the request's facts once everything above that may
    # refuse it has (its body parsed, its plan made) — before the session,
    # redaction, the audit START row, the upstream authorizer and any
    # upstream contact. (A pass-through route was asked before its body was
    # read, a local answer before it was given.) Without the member: one
    # attribute test, no await.
    authorization = state.authorization
    if adapter is not None and (authorization.authorizes or authorization.checks_content):
        asked = _authorization_request(
            request,
            adapter,
            kind,
            provider_name=provider_name,
            path=path,
            model=adapter.request_model(request.method, path, parsed),
            identity=proxy_credential,
        )
    if authorization.authorizes and adapter is not None:
        assert asked is not None  # built above
        refused = await _authorization_check(
            state,
            request,
            asked,
            adapter,
            provider_name=provider_name,
            path=path,
            started=started,
        )
        if refused is not None:
            return refused
    # The requester's detection overlay (tighten-only), read in the same
    # synchronous stretch as the session below, so it extends exactly the
    # detection objects the request redacts with.
    overlay: OverlayBuild | None = None
    if authorization.overlays:
        overlay, overlay_refusal = authorization.overlay(
            state.overlay_builds, f"{request.method} {path}"
        )
        if overlay_refusal is not None:
            return _authorization_refused(
                state,
                adapter,
                overlay_refusal,
                provider_name=provider_name,
                request=request,
                path=path,
                started=started,
            )

    # Session resolution hashes the raw (pre-redaction) conversation anchor,
    # so it must happen before prepare_request. Opening a session this
    # process has not seen reads the vault (its view loads the rows; a
    # router may read the durable response map): a fault there is refused
    # like every vault fault — the request has no session yet, so the row
    # carries the configured one.
    try:
        ctx = state.context_for(adapter, request.method, path, parsed, overlay)
    except state.vault_faults as exc:
        return _vault_fault_refused(
            state,
            state.config.vault.session,
            adapter,
            exc,
            request=request,
            path=path,
            started=started,
            provider_name=provider_name,
            opening=True,
        )

    # This request's share of the process-wide counts (the diff trick).
    window = _CountWindow(state)
    # The requester's approved overrides, asked only where detection refuses
    # (overrides.py); None when [overrides] enabled = false — and under a
    # credential the proxy holds (its identity, a routed plan's operator
    # key): every refusal under one is final, so it carries no code and no
    # approval passes it.
    scope = None if proxy_credential else state.override_scope()

    def allow_suffix(kind: str) -> str:
        # The refusal's single-use code, when it can carry one: minted now
        # (what it matches: the values this request had refused, or the
        # route), bound to the requester.
        hint = (
            scope.refusal_hint(kind, provider_name or "", request.method, path)
            if scope is not None
            else None
        )
        return f"; {hint}" if hint is not None else ""

    def blocked_response(exc: BlockedRequest, blocked_adapter: ProviderAdapter) -> JSONResponse:
        # A block-mode rule matched: fail closed before any upstream
        # contact. 400 (not 403): SDKs surface it as a non-retryable
        # BadRequestError whose message the tool displays, while 403
        # triggers misleading "check your API key" advice.
        state.blocked_counts[exc.detector_type] += 1
        logger.info("%s %s -> 400 blocked (rule type %s)", request.method, path, exc.detector_type)
        state.record_request(
            session=ctx.session_id,
            provider=blocked_adapter.name,
            method=request.method,
            path=path,
            status=400,
            started=started,
            streamed=False,
            detections={exc.detector_type: 1},
            rehydrations={},
            refusal="blocked_value",
        )
        return JSONResponse(
            blocked_adapter.error_body(
                f"llm-redact: request blocked; a {exc.detector_type} value was"
                ' detected and this rule is configured with mode = "block"' + allow_suffix("block"),
                status=400,
            ),
            status_code=400,
        )

    def refused_response(
        message: str, refused_adapter: ProviderAdapter, why: str, kind: LocalRefusal
    ) -> JSONResponse:
        # A body the proxy cannot redact (or, under identity auth, cannot
        # vouch for) fails closed before any upstream contact: 400,
        # recorded, the message naming the field or format only.
        logger.info("%s %s -> 400 refused (%s)", request.method, path, why)
        state.record_request(
            session=ctx.session_id,
            provider=refused_adapter.name,
            method=request.method,
            path=path,
            status=400,
            started=started,
            streamed=False,
            detections={},
            rehydrations={},
            refusal=kind,
        )
        return JSONResponse(refused_adapter.error_body(message, status=400), status_code=400)

    def sealed_response(sealed_adapter: ProviderAdapter) -> JSONResponse:
        # The session router sealed this request's session (it must stay
        # empty) and redaction would have written to it: refused before
        # any upstream contact, the session untouched.
        logger.info("%s %s -> 403 refused (sealed session)", request.method, path)
        state.record_request(
            session=ctx.session_id,
            provider=sealed_adapter.name,
            method=request.method,
            path=path,
            status=403,
            started=started,
            streamed=False,
            detections={},
            rehydrations={},
            refusal="sealed_session",
        )
        reason = ctx.sealed or _SEALED_REFUSAL
        return JSONResponse(sealed_adapter.error_body(reason, status=403), status_code=403)

    def too_many_strings(capped_adapter: ProviderAdapter) -> Response:
        # More strings than max_body_strings: counted during redaction, so
        # refused before any upstream contact (placeholders issued for the
        # strings before the limit are harmless — the vault is
        # deterministic and nothing is forwarded).
        return _body_too_large(
            state,
            request,
            capped_adapter,
            session=ctx.session_id,
            path=path,
            started=started,
            cap="max_body_strings",
            limit=max_body_strings,
        )

    note_wanted = plan.inject_system_note if plan is not None else state.config.inject_system_note

    outbound = body_bytes
    # The decoded form of `outbound` (None for pass-through / non-JSON
    # bodies): the routed path applies per-hop body rewrites to it.
    outbound_obj: dict[str, Any] | None = parsed if isinstance(parsed, dict) else None
    # Whether the redaction below read the request's content (the access
    # gate's ContentFacts.scanned), and the binary upload parts it forwards
    # unscanned (counted once the upload was read in full).
    scanned = False
    binary_forwarded: list[int] = []
    # The MCP blocks to an exempt server the redaction held out (sent
    # unscanned): ContentFacts.exempt_blocks.
    exempt_blocks = [0]
    if adapter is not None and detection_off:
        # [providers.NAME] detection = false: the deliberate off-switch.
        # The request is forwarded byte-identical — no detection, no deny
        # strings, no block modes, no note injection. Rehydration below
        # stays active so placeholders from history still restore. Loud
        # honesty: one log line per request; /status lists the provider.
        logger.info(
            "%s %s forwarded unredacted ([providers.%s] detection = false)",
            request.method,
            path,
            provider_name,
        )
        if duplicate_keys:
            # Except a repeated JSON key: the ownership check and object
            # tracking read its LAST occurrence, so exactly that is sent — a
            # first-wins upstream would otherwise act on a value nobody
            # checked (another user's previous_response_id, a `store`).
            outbound = json_bytes(parsed)
        elif checked_upload is not None:
            # Likewise an uploaded line repeating a key (a batch input
            # file's request, run later with this upload's credential).
            outbound = checked_upload
    elif adapter is not None and isinstance(parsed, dict):
        # Token floors: a new value is never numbered onto a token this body
        # already carries (one its session never issued — a compacted
        # history, a pasted answer — would otherwise gain a second meaning).
        # The whole decoded body counts, fields the walk skips included; a
        # body with no guillemet in any encoding pays only the byte gate.
        # The copy is this body's own: it counts the strings it redacts
        # against max_body_strings.
        budgeted = ctx.redactor.with_budget(max_body_strings)
        if scope is not None:
            budgeted = budgeted.with_overrides(scope)
        redactor = (
            budgeted.with_floors(json_floors(parsed)) if may_carry_tokens(body_bytes) else budgeted
        )
        try:
            # Every new value of the body in ONE vault transaction (one fsync
            # per request, not per value), committed before anything is
            # forwarded; any refusal below rolls it back whole.
            prepared = run_batched(
                ctx.vault,
                functools.partial(
                    prepare_route_request,
                    adapter,
                    request.method,
                    path,
                    parsed,
                    redactor,
                    inject_note=note_wanted and adapter.wants_system_note(kind, path),
                    mcp_exempt=frozenset(state.config.detection.mcp_exempt_servers),
                    exempt_blocks=exempt_blocks,
                ),
            )
        except BlockedRequest as exc:
            return blocked_response(exc, adapter)
        except TooManyStrings:
            return too_many_strings(adapter)
        except PlaceholderLimitReached as exc:
            return refused_response(
                str(exc), adapter, "no placeholder number left", "placeholder_limit"
            )
        except VerbatimFieldRedacted as exc:
            return refused_response(
                str(exc) + allow_suffix("verbatim_field"),
                adapter,
                "verbatim field",
                "verbatim_field",
            )
        except UnredactableRequest as exc:
            return refused_response(str(exc), adapter, "undecodable field", "unredactable")
        except SealedSessionError:
            return sealed_response(adapter)
        except state.vault_faults as exc:
            return _vault_fault_refused(
                state, ctx.session_id, adapter, exc, request=request, path=path, started=started
            )
        if not _commit_overrides(scope, upload):
            assert scope is not None  # nothing to commit without one
            return refused_response(
                raced_message(scope), adapter, "one-time override", _raced_kind(scope)
            )
        outbound_obj = prepared
        scanned = True
        # No-op short-circuit: redaction increments detection_counts, and note
        # injection is gated on a redaction actually happening (base
        # prepare_request), so an unchanged count means the prepared body is
        # byte-for-byte the original. Forward the raw bytes and skip the
        # parse→dump round-trip — the common nothing-to-redact large-body case.
        # Never with a repeated key: the raw bytes still hold the earlier
        # occurrences the walk never saw (an upstream may keep the first).
        if duplicate_keys or sum(state.detection_counts.values()) != sum(
            window.detections.values()
        ):
            outbound = json_bytes(prepared)
    elif adapter is not None and parsed is None and body_bytes:
        # Matched routes with non-JSON bodies: only canonical multipart on a
        # route that scans it gets here (the scanned-body rule, above) —
        # uploads (OpenAI /v1/files: JSONL file parts, file names, form
        # fields) and the prompt-field media routes. Every part must be
        # scanned: one the adapter would forward unscanned refuses the whole
        # request (require_scanned). What the upload cites (its lines and
        # form fields) was checked above, before anything was redacted
        # (_ownership_body).
        boundary = adapter.multipart_boundary(path, request.headers.get("content-type", ""))
        if boundary is not None:
            # A binary file part (a PDF, an image) cannot be redacted: with
            # the client's own credential it is forwarded unscanned unless
            # [detection] binary_uploads = "refuse"; under a credential the
            # proxy holds it is never sent (the proxy vouches only for what
            # it read) — unless an upload inspector read it as text that
            # scanned clean (below). Counted once the upload was read in
            # full.
            forward_binary = (
                binary_forwarded.append
                if not proxy_credential and state.config.detection.binary_uploads == "forward"
                else None
            )
            # binary_uploads = "refuse" with the client's own key: the
            # requester may have approved forwarding this route's binary
            # parts unscanned (overrides.py). Looked up only once the upload
            # was read and holds a binary part it would refuse — never for
            # an upload of text (the lookup may ask the access gate) — and
            # used only then; without a rule the part is refused as before.
            binary_rules: list[tuple[str, int]] = []
            if forward_binary is None and not proxy_credential and scope is not None:
                forward_binary = functools.partial(
                    _forward_on_rule,
                    scope,
                    binary_rules,
                    binary_forwarded,
                    (provider_name, request.method, path),
                )
            # The body the stored-object check read, when it re-serialized a
            # line repeating a key: every part — a text or binary file's too
            # — then goes out as the check read it.
            upload_body = checked_upload if checked_upload is not None else body_bytes
            # This body's own copy, counting its strings (form fields, file
            # names, JSONL lines, extracted texts) against max_body_strings.
            upload_redactor = ctx.redactor.with_budget(max_body_strings)
            if scope is not None:
                upload_redactor = upload_redactor.with_overrides(scope)
            # The inspected binary parts' outcomes and the binary parts
            # forwarded unscanned, counted once the upload is handed to the
            # upstream or refused (_UploadFate: a refusal after redaction —
            # no upstream, the audit START, the authorizer — counts nothing
            # as forwarded).
            inspection_outcomes: Counter[str] = Counter()
            # Its text parts redacted as one text, remembered for their
            # download once it is handed to the upstream (never on a refusal).
            raw_texts: list[bytes] = []
            upload.hold(
                functools.partial(
                    _settle_upload,
                    state,
                    request.method,
                    path,
                    provider_name,
                    inspection_outcomes,
                    binary_forwarded,
                    raw_texts,
                )
            )

            def before_inspection() -> None:
                # What refuses this request whatever the redaction finds,
                # applied BEFORE its binary parts go to the inspector (which
                # may send them off the machine) instead of after: the legacy
                # path's missing upstream and target, a routed plan's local
                # refusal, then the [audit] required START row — held for
                # the request (_EarlyAudit): every later refusal's row, the
                # authorizer's and a routed budget's included, is its END
                # row. The authorizer itself needs the final bytes: after.
                if plan is None:
                    if not upstream_base:
                        raise _RefusedBeforeInspection(
                            _upstream_unconfigured(
                                state, ctx, adapter, request=request, path=path, started=started
                            )
                        )
                    if (
                        _legacy_target(
                            request,
                            path,
                            provider_name,
                            upstream_base,
                            upstream_auth,
                            matched=adapter is not None,
                        )
                        is None
                    ):
                        raise _RefusedBeforeInspection(_target_refused(state, provider_name))
                else:
                    local = plan.local_refusal()
                    if local is not None:
                        raise _RefusedBeforeInspection(
                            _route_refusal(
                                state,
                                ctx.session_id,
                                adapter,
                                local,
                                request=request,
                                path=path,
                                started=started,
                                kind="route_unsupported",
                            )
                        )
                token, audit_refusal = _begin_audit_guarded(
                    state,
                    ctx,
                    adapter,
                    request=request,
                    path=path,
                    started=started,
                    new_counts={},
                    new_warned={},
                )
                if audit_refusal is not None:
                    raise _RefusedBeforeInspection(audit_refusal)
                early = _EARLY_AUDIT.get()
                assert early is not None  # set by handle() for every request
                early.hold(
                    token,
                    session=ctx.session_id,
                    provider=adapter.name,
                    method=request.method,
                    path=path,
                    started=started,
                )

            try:
                inspected: InspectedUpload | None = None
                if state.upload_inspector is not None:
                    # Before redaction, and awaited HERE — never inside the
                    # vault batch below: the upload read once, its binary
                    # parts read as text by the plugin and scanned (no
                    # placeholder issued), the reading handed back.
                    inspected, floors = await _inspect_upload(
                        state,
                        request,
                        adapter,
                        path,
                        upload_body,
                        boundary,
                        upload_redactor,
                        provider_name=provider_name,
                        identity=proxy_credential,
                        max_body_bytes=max_body_bytes,
                        max_body_strings=max_body_strings,
                        outcomes=inspection_outcomes,
                        window=window,
                        before_inspection=before_inspection,
                    )
                    upload_redactor = upload_redactor.with_floors(floors)
                # One vault transaction for the whole upload (run_batched).
                rewritten = run_batched(
                    ctx.vault,
                    functools.partial(
                        adapter.redact_multipart,
                        path,
                        upload_body,
                        boundary,
                        upload_redactor,
                        inject_note=note_wanted and adapter.wants_system_note(kind, path),
                        # The scanned-body rule, part by part: an unscanned
                        # piece refuses the whole request — a binary file
                        # part only when forward_binary is None and the
                        # inspection did not clear it.
                        require_scanned=True,
                        forward_binary=forward_binary,
                        inspected=inspected,
                        remember_text=raw_texts.append,
                    ),
                )
            except _RefusedBeforeInspection as exc:
                return exc.response
            except BinaryValuesDetected as exc:
                # Values the proxy would redact, inside a file it cannot
                # rewrite: refused, naming their types only.
                return refused_response(
                    str(exc) + allow_suffix("binary_values"),
                    adapter,
                    "values in a binary upload",
                    "binary_values",
                )
            except BlockedRequest as exc:
                # One leaking line in an uploaded file is a leak: the
                # whole request is rejected.
                return blocked_response(exc, adapter)
            except TooManyStrings:
                return too_many_strings(adapter)
            except PlaceholderLimitReached as exc:
                return refused_response(
                    str(exc), adapter, "no placeholder number left", "placeholder_limit"
                )
            except UnredactableRequest as exc:
                clause = _scanned_body_clause(
                    identity=upstream_auth is not None, proxy_credential=proxy_credential
                )
                # An unscannable binary part with the client's own key is a
                # CONTENT refusal its requester may override (binary_upload);
                # every other piece (a framing or header rule) and anything
                # under a credential the proxy holds is final.
                overridable = isinstance(exc, UnscannedBinaryFile) and not proxy_credential
                return refused_response(
                    f"llm-redact: {exc}, and {clause}; the request was not forwarded"
                    + (allow_suffix("binary_upload") if overridable else ""),
                    adapter,
                    "unscanned multipart content",
                    "unscanned_upload",
                )
            except SealedSessionError:
                return sealed_response(adapter)
            except state.vault_faults as exc:
                return _vault_fault_refused(
                    state,
                    ctx.session_id,
                    adapter,
                    exc,
                    request=request,
                    path=path,
                    started=started,
                )
            if binary_rules and binary_forwarded:
                scope.use_route(binary_rules[0])  # type: ignore[union-attr]
            if not _commit_overrides(scope, upload):
                assert scope is not None  # nothing to commit without one
                return refused_response(
                    raced_message(scope), adapter, "one-time override", _raced_kind(scope)
                )
            scanned = True
            if rewritten is not None:
                outbound = rewritten
            elif checked_upload is not None:
                outbound = checked_upload

    new_counts = _count_delta(state.detection_counts, window.detections)
    # Same diff trick for warn-mode hits: attribute forwarded-unredacted
    # values to THIS request, not just the process-lifetime aggregate.
    new_warned = _count_delta(state.warn_counts, window.warned)

    if authorization.checks_content:
        # The access gate's optional authorize_content on what the redaction
        # found: its facts are THIS request's, taken above with no await
        # since the redaction (another request's counts never leak in), and
        # asked before the audit START row, the upstream authorizer, a
        # routed plan's begin() and any upstream contact. A refusal hands a
        # one-time override it used back (handle()'s refused settlement).
        assert asked is not None  # built for every forwarded request
        refused = await _content_check(
            state,
            request,
            asked,
            content_facts(
                scanned=scanned,
                detected=new_counts if scanned else None,
                warned=new_warned if scanned else None,
                unscanned_parts=binary_forwarded[0] if binary_forwarded else 0,
                overridden=_COMMITTED_OVERRIDE.get() is not None,
                overridden_types=scope.allowed_types() if scope is not None else None,
                exempt_blocks=exempt_blocks[0],
            ),
            ctx,
            adapter,
            provider_name=provider_name,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
        )
        if refused is not None:
            return refused

    if plan is not None:
        return await _handle_routed(
            request,
            state,
            ctx,
            plan,
            adapter=adapter,
            kind=kind,
            outbound=outbound,
            outbound_obj=outbound_obj,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
            identity=proxy_credential,
            upload=upload,
        )

    if not upstream_base:
        # Providers without a default upstream (azure) answer 502 until
        # configured — proxy-generated, never forwarded. (An upload with a
        # binary part to inspect was answered this before its inspection.)
        return _upstream_unconfigured(
            state, ctx, adapter, request=request, path=path, started=started
        )
    target = _legacy_target(
        request, path, provider_name, upstream_base, upstream_auth, matched=adapter is not None
    )
    if target is None:
        return _target_refused(state, provider_name)
    url, headers = target
    if upstream_auth is not None:
        # The proxy's own cloud identity (the client's credentials stripped
        # by _legacy_target): authorize the FINAL request — the on-the-wire
        # URL and the bytes below, after redaction and note injection — and
        # send exactly that.
        try:
            headers = await upstream_auth.authorize(request.method, url, headers, outbound)
        except Exception as exc:
            return _upstream_auth_failure(
                state,
                ctx,
                adapter,
                exc,
                provider_name=provider_name,
                request=request,
                path=path,
                started=started,
                new_counts=new_counts,
                new_warned=new_warned,
            )

    upstream_request = state.client.build_request(
        request.method, url, headers=headers, content=outbound
    )

    early = _EARLY_AUDIT.get()
    if early is not None and early.started:
        # An upload whose START row was written before its inspection.
        audit_token, audit_refusal = _start_after_early(
            state,
            early,
            ctx,
            adapter,
            request=request,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
        )
    else:
        audit_token, audit_refusal = _begin_audit_guarded(
            state,
            ctx,
            adapter,
            request=request,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
        )
    if audit_refusal is not None:
        return audit_refusal

    # Handed to the upstream: an upload's parts now count as forwarded.
    upload.settle(sent=True)
    try:
        upstream = await _waited(state.client.send(upstream_request, stream=True))
    except httpx.TransportError as exc:
        # Connect/handshake/header fault: no response body was produced, so
        # there is nothing to close and the streaming generators (which own
        # their own read-fault finalization) never start.
        return _fault_response(
            state,
            ctx,
            adapter,
            exc,
            request=request,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
            audit_token=audit_token,
            route=None,
        )

    logger.info(
        "%s %s -> %d%s",
        request.method,
        path,
        upstream.status_code,
        _redacted_summary(new_counts),
    )
    return await _deliver(
        request,
        state,
        ctx,
        adapter,
        kind,
        upstream,
        path=path,
        started=started,
        new_counts=new_counts,
        new_warned=new_warned,
        audit_token=audit_token,
        route=None,
        request_body=parsed,
        sent_body=outbound_obj,
        identity=proxy_credential,
        # A following client repeats its ORIGINAL request at the Location:
        # safe to relay only when that carries nothing the proxy protects —
        # no body the proxy redacted (a pass-through body went out verbatim
        # anyway), no credential for the proxy itself, not the proxy's own
        # identity, and not a custom upstream (a relative Location resolves
        # against the proxy without the /custom/NAME prefix).
        relay_redirects=upstream_auth is None
        and not presented_credential
        and not provider_name.startswith("custom:")
        and (adapter is None or not body_bytes),
    )


def vault_fault_types(manager: object) -> tuple[type[BaseException], ...]:
    """What a vault fault raises while a request's placeholders are issued
    (``vault.run_batched``): sqlite's errors — the sqlite vaults, whose batch
    rolls back and re-raises a failed write or COMMIT — and the manager's
    own (optional ``fault_types``: an RDBMS vault's DB-API driver errors and
    its ``RdbmsAllocationError``). Anything else is not the vault's and
    propagates as before."""
    declared = getattr(manager, "fault_types", ())
    return (sqlite3.Error, *tuple(declared))


def _vault_fault_refused(
    state: ProxyState,
    session: str,
    adapter: ProviderAdapter | None,
    exc: BaseException,
    *,
    request: Request,
    path: str,
    started: float,
    provider_name: str | None = None,
    opening: bool = False,
) -> JSONResponse:
    """The vault could not issue this request's placeholders — a write or
    its batch's COMMIT failed, and the batch rolled back whole — or, with
    ``opening``, could not open the session the request resolved to (a new
    session's view reads its rows; the session router may read the durable
    response map): a recorded, provider-shaped 503 before any upstream
    contact (the audit refusal's twin; nothing was forwarded or signed),
    counted as the "vault" bookkeeping stage and logged by exception TYPE
    only. A pass-through request (no adapter) gets the generic shape."""
    state.bookkeeping_errors["vault"] += 1
    logger.error(
        "%s %s -> 503 %s (%s); nothing forwarded",
        request.method,
        path,
        "vault read failed opening the session" if opening else "vault write failed",
        type(exc).__name__,
    )
    state.record_request(
        session=session,
        provider=adapter.name if adapter is not None else provider_name,
        method=request.method,
        path=path,
        status=503,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal="vault_fault",
    )
    message = (
        "llm-redact: the vault could not open this request's session; the request was not forwarded"
        if opening
        else "llm-redact: the vault could not record this request's placeholders; the request"
        " was not forwarded"
    )
    body = adapter.error_body(message, status=503) if adapter is not None else {"error": message}
    return JSONResponse(body, status_code=503)


def _count_delta(after: Counter[str], before: dict[str, int]) -> dict[str, int]:
    """Per-request attribution of a process-lifetime counter: the types
    whose count grew during this request, with the growth."""
    return {k: v - before.get(k, 0) for k, v in after.items() if v - before.get(k, 0) > 0}


def _redacted_summary(new_counts: dict[str, int]) -> str:
    return (
        " redacted: " + " ".join(f"{k}×{v}" for k, v in sorted(new_counts.items()))
        if new_counts
        else ""
    )


def _same_upstream(url: str, upstream_base: str, *, exact_path: str | None = None) -> bool:
    """Whether ``url`` still addresses the configured upstream: scheme and
    netloc (userinfo included — a smuggled ``user@`` is a different URL),
    and a path inside the base URL's path (an API-management or gateway
    base such as ``https://gw.example/my-api`` must not reach a sibling
    API). With ``exact_path`` — the base path plus the request's raw path,
    for requests the proxy authorizes with its own identity — the path
    httpx will send must be exactly that: nothing normalized away."""
    try:
        built, base = httpx.URL(url), httpx.URL(upstream_base)
    except httpx.InvalidURL:
        return False
    if (built.scheme, built.userinfo, built.host, built.port) != (
        base.scheme,
        base.userinfo,
        base.host,
        base.port,
    ):
        return False
    built_path = built.raw_path.split(b"?", 1)[0]
    base_path = base.raw_path.split(b"?", 1)[0].rstrip(b"/")
    if base_path and built_path != base_path and not built_path.startswith(base_path + b"/"):
        return False
    return exact_path is None or built_path == exact_path.encode("utf-8")


def upstream_base_path(upstream_base: str) -> str:
    """The configured upstream base URL's path, no trailing slash."""
    return urllib.parse.urlsplit(upstream_base).path.rstrip("/")


def _upstream_path(request: Request, path: str) -> str:
    """The path to forward, exactly as the client sent it: Bedrock model
    ids are often percent-encoded ARNs whose %2F/%3A must reach the
    upstream unchanged — the decoded `path` would hand it a different path
    structure (httpx preserves existing %XX escapes). raw_path excludes
    the query per the ASGI spec; the split defends non-compliant servers."""
    raw_path: bytes | None = request.scope.get("raw_path")
    try:
        return raw_path.split(b"?", 1)[0].decode("ascii") if raw_path else path
    except UnicodeDecodeError:
        return path


def _upstream_unconfigured(
    state: ProxyState,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    *,
    request: Request,
    path: str,
    started: float,
) -> JSONResponse:
    """A provider without an upstream (azure has no default): the
    proxy-generated, recorded 502 — never forwarded."""
    provider_name = adapter.name if adapter is not None else "unknown"
    error = (
        adapter.error_body(f"configure [providers.{provider_name}] upstream_base_url", status=502)
        if adapter is not None
        else {"error": f"no upstream configured for {path}"}
    )
    state.record_request(
        session=ctx.session_id,
        provider=provider_name,
        method=request.method,
        path=path,
        status=502,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal="no_upstream",
    )
    logger.info("%s %s -> 502 upstream not configured", request.method, path)
    return JSONResponse(error, status_code=502)


def _target_refused(state: ProxyState, provider_name: str | None) -> JSONResponse:
    """The built upstream URL would not address the configured upstream: a
    400, never recorded or logged with its path — counted alone."""
    state.count_local_refusal("request_target", provider_name)
    return JSONResponse({"error": "the request target must be a path"}, status_code=400)


def _legacy_target(
    request: Request,
    path: str,
    provider_name: str,
    upstream_base: str,
    upstream_auth: UpstreamAuth | None,
    *,
    matched: bool,
) -> tuple[str, list[tuple[str, str]]] | None:
    """The legacy path's upstream URL and forwarded headers — under the
    proxy's own identity (``upstream_auth``) with every client credential
    stripped — or None when the URL would not address the configured
    upstream (the caller's 400). Read from the request line and headers
    only, never the body: checked again before an upload's inspection."""
    upstream_path = _upstream_path(request, path)
    if provider_name.startswith("custom:"):
        # The /custom/NAME prefix is proxy-local routing, not part of the
        # upstream's namespace (names are plain [a-z0-9-], so the byte
        # prefix is unambiguous even in a percent-encoded raw path).
        route_prefix = custom_prefix(provider_name)
        if upstream_path.startswith(route_prefix):
            upstream_path = upstream_path[len(route_prefix) :] or "/"
    url = upstream_base + upstream_path
    if request.url.query:
        url += "?" + request.url.query
    if not _same_upstream(url, upstream_base):
        # Belt and braces behind origin_form_target: whatever the path
        # holds, the request goes to the configured upstream or nowhere.
        return None
    headers = _request_headers(request, matched=matched)
    if upstream_auth is None:
        return url, headers
    url, headers = strip_client_credentials(url, headers)
    if not _same_upstream(
        url, upstream_base, exact_path=upstream_base_path(upstream_base) + upstream_path
    ):
        # The URL the authorizer would sign must address exactly the path
        # the route was matched on (httpx normalizes before send).
        return None
    return url, headers


def _start_after_early(
    state: ProxyState,
    early: _EarlyAudit,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    *,
    request: Request,
    path: str,
    started: float,
    new_counts: dict[str, int],
    new_warned: dict[str, int],
) -> tuple[object | None, JSONResponse | None]:
    """The write-ahead token for an upload whose START row was written
    before its inspection (``_EarlyAudit``), at the send: that row carries
    no detections (the redaction had not run), so when the redaction found
    values — warn-mode ones are FORWARDED — the record durable before
    upstream contact must still say what leaves. Likewise when the request
    passed a refusal on an approved override (its value, or a binary part,
    goes out as sent): the marker must be durable too. A log with the
    optional ``amend`` member amends the early row with them here
    (``_EarlyAudit.amend``): the request keeps ONE START row, which its END
    row finalizes. A log without it gets a second START row carrying them,
    and the early row is ended (``_EarlyAudit.supersede``). Either write
    that cannot commit is the 503 refusal, whose row ends the early one.
    Nothing found and no override: the early row is the request's."""
    if early.holds_token and (new_counts or new_warned or _COMMITTED_OVERRIDE.get()):
        if state.write_ahead_amend is not None:
            try:
                early.amend(state, detections=new_counts, warned=new_warned)
            except AuditWriteError as exc:
                return None, _audit_unavailable(
                    state,
                    ctx,
                    adapter,
                    exc,
                    request=request,
                    path=path,
                    started=started,
                    new_counts=new_counts,
                    new_warned=new_warned,
                )
            return early.take(), None
        token, refusal = _begin_audit_guarded(
            state,
            ctx,
            adapter,
            request=request,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
        )
        if refusal is None:
            early.supersede(state)
        return token, refusal
    return early.take(), None


def _begin_audit_guarded(
    state: ProxyState,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    *,
    request: Request,
    path: str,
    started: float,
    new_counts: dict[str, int],
    new_warned: dict[str, int],
) -> tuple[object | None, JSONResponse | None]:
    """[audit] required: no durably committed audit row, no upstream contact.
    The write-ahead START row commits HERE — after redaction (detections
    known), before any byte leaves for the provider; an upload with a binary
    part to inspect also commits one BEFORE its inspection (no detections
    yet: ``before_inspection``), amended — or, by a log without ``amend``,
    superseded — at the send when the redaction found values
    (``_start_after_early``). A None token means
    required mode is off and nothing downstream changes; a refusal is the
    provider-shaped 503 the caller returns instead of contacting anyone."""
    try:
        token = state.begin_audit(
            session=ctx.session_id,
            provider=adapter.name if adapter is not None else None,
            method=request.method,
            path=path,
            detections=new_counts,
            warned=new_warned,
        )
    except AuditWriteError as exc:
        return None, _audit_unavailable(
            state,
            ctx,
            adapter,
            exc,
            request=request,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
        )
    return token, None


def _audit_unavailable(
    state: ProxyState,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    exc: AuditWriteError,
    *,
    request: Request,
    path: str,
    started: float,
    new_counts: dict[str, int],
    new_warned: dict[str, int],
) -> JSONResponse:
    """The write-ahead record (a START row, or the amendment of an upload's
    early one) could not be committed: the audit-storage twin of the
    upstream-fault 502 — a provider-shaped 503 instead of any upstream
    contact, recorded to metrics/recent (the audit write for this row will
    itself fail — record_request logs that loudly; an upload's early START
    row is ended by it). Logged by type only."""
    logger.critical(
        "%s %s -> 503 audit write failed with [audit] required (%s)",
        request.method,
        path,
        type(exc).__name__,
    )
    state.record_request(
        session=ctx.session_id,
        provider=adapter.name if adapter is not None else None,
        method=request.method,
        path=path,
        status=503,
        started=started,
        streamed=False,
        detections=new_counts,
        rehydrations={},
        warned=new_warned,
        refusal="audit_unavailable",
    )
    message = "llm-redact: audit log unavailable and [audit] required is enabled"
    body = adapter.error_body(message, status=503) if adapter is not None else {"error": message}
    return JSONResponse(body, status_code=503)


def _upstream_auth_failure(
    state: ProxyState,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    exc: Exception,
    *,
    provider_name: str,
    request: Request,
    path: str,
    started: float,
    new_counts: dict[str, int],
    new_warned: dict[str, int],
) -> JSONResponse:
    """The proxy's cloud identity produced no credential: nothing is
    forwarded (never the client's credential in its place, never an
    unauthenticated request). A recorded, provider-shaped 502 — the
    credential twin of the upstream-fault 502 — counted as an upstream
    error. It names the credential SOURCE only: an ``UpstreamAuthError``
    message by contract, any other exception by its TYPE."""
    source = str(exc) if isinstance(exc, UpstreamAuthError) else type(exc).__name__
    state.upstream_errors[provider_name] += 1
    logger.warning(
        "%s %s -> 502 upstream credentials unavailable for %s (%s)",
        request.method,
        path,
        provider_name,
        source,
    )
    state.record_request(
        session=ctx.session_id,
        provider=provider_name,
        method=request.method,
        path=path,
        status=502,
        started=started,
        streamed=False,
        detections=new_counts,
        rehydrations={},
        warned=new_warned,
        refusal="upstream_auth",
    )
    message = (
        f"llm-redact: the proxy could not obtain its own {provider_name} cloud"
        f" credentials ({source}); nothing was forwarded"
    )
    body = adapter.error_body(message, status=502) if adapter is not None else {"error": message}
    return JSONResponse(body, status_code=502)


def _fault_response(
    state: ProxyState,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    exc: httpx.TransportError,
    *,
    request: Request,
    path: str,
    started: float,
    new_counts: dict[str, int],
    new_warned: dict[str, int],
    audit_token: object | None,
    route: RouteDelivery | None,
) -> JSONResponse:
    """The upstream connection failed — refused, reset, timed out, or
    dropped mid-body. Fail closed with a provider-shaped 502: the tool sees
    a clean gateway error, never a partial or wrong body. Record the fault
    so metrics/audit see it — the buffered twin of the streaming branches'
    finally-block finalization. By exception TYPE only: an httpx message
    can embed the upstream URL (query auth). A routed request counts the
    fault against its upstream NAME and closes its route as `transport`."""
    row: dict[str, Any] | None = None
    if route is not None:
        route.mark_failed("transport")
        row = state.finish_route(route, 502)
        state.upstream_errors[route.upstream] += 1
    else:
        state.upstream_errors[adapter.name if adapter is not None else "passthrough"] += 1
    logger.warning(
        "%s %s -> 502 upstream fault (%s)%s",
        request.method,
        path,
        type(exc).__name__,
        _route_log_suffix(row) if row is not None else "",
    )
    state.record_request(
        session=ctx.session_id,
        provider=adapter.name if adapter is not None else None,
        method=request.method,
        path=path,
        status=502,
        started=started,
        streamed=False,
        detections=new_counts,
        rehydrations={},
        warned=new_warned,
        audit_token=audit_token,
        route=row,
        refusal="upstream_fault",
    )
    body = (
        adapter.error_body("llm-redact: upstream request failed", status=502)
        if adapter is not None
        else {"error": "llm-redact: upstream request failed"}
    )
    return JSONResponse(
        body, status_code=502, headers=dict(route.headers) if route is not None else None
    )


async def _deliver(
    request: Request,
    state: ProxyState,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    kind: RouteKind,
    upstream: httpx.Response,
    *,
    path: str,
    started: float,
    new_counts: dict[str, int],
    new_warned: dict[str, int],
    audit_token: object | None,
    route: RouteDelivery | None,
    request_body: Any = None,
    relay_redirects: bool = False,
    sent_body: Any = None,
    identity: bool = False,
) -> Response:
    """Hand an upstream response to the client: the streaming branches
    (chosen by the upstream RESPONSE content-type, never the request's
    stream flag) rehydrate as they go and finalize at stream end; anything
    else is buffered. With `route` (a routed request) the same branches
    also run the router's delivery hooks (a rewritten model id restored,
    usage tracked for its budget ledger) and stamp the x-llm-redact-*
    headers. An upstream redirect is relayed only on the unrouted requests
    ``relay_redirects`` marks (see ``_redirect_refused``). A session router
    with the optional ``response_observer`` observes the answer as the
    provider sent it (``sent_body``: the request body as sent upstream;
    ``identity``: a credential the proxy holds was spent)."""
    if _is_redirect(upstream) and not (relay_redirects and route is None):
        await upstream.aclose()
        return _redirect_refused(
            state,
            ctx,
            adapter,
            upstream.status_code,
            request=request,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
            audit_token=audit_token,
            route=route,
        )
    content_type = upstream.headers.get("content-type", "")
    headers = _response_headers(upstream)
    if identity and adapter is not None and adapter.capability_response_headers:
        # A capability the provider minted for the PROXY's credential (an
        # upload session URL) never reaches a client: whatever it grants
        # would be spent unread, as the proxy's principal.
        headers = {
            name: value
            for name, value in headers.items()
            if name.lower() not in adapter.capability_response_headers
        }
    if route is not None:
        headers.update(route.headers)
    request_meta = RequestMeta(request.method, path, started, new_counts, new_warned, audit_token)
    # A downloaded FILE is restored buffered, per file, whatever its
    # Content-Type names (JSON, JSON Lines, an event stream): never by a
    # streaming or whole-body JSON reading (``restores_file_download``).
    file_download = (
        kind is RouteKind.CHAT
        and adapter is not None
        and adapter.restores_file_download(request.method, path)
    )
    streamed = kind is RouteKind.CHAT and adapter is not None and not file_download
    observe = (
        state.response_observer(
            ResponseContext(
                adapter.name if adapter is not None else None,
                request.method,
                path,
                upstream.status_code,
                content_type,
                sent_body,
                ctx.session_id,
                identity,
            )
        )
        if state.observes_responses
        else None
    )

    if streamed and adapter is not None and "text/event-stream" in content_type:
        return StreamingResponse(
            _ClientPaced(
                _stream_rehydrated(
                    upstream,
                    adapter,
                    state,
                    ctx,
                    request_meta=request_meta,
                    route=route,
                    object_tracker=(
                        state.object_tracker(
                            adapter,
                            request.method,
                            path,
                            request.headers,
                            body=request_body,
                            query=request.url.query,
                        )
                        if 200 <= upstream.status_code < 300
                        else None
                    ),
                    request_body=request_body,
                    observe=observe,
                )
            ),
            status_code=upstream.status_code,
            headers=headers,
            media_type="text/event-stream",
        )

    if (
        streamed
        and adapter is not None
        and adapter.handles_eventstream
        and "application/vnd.amazon.eventstream" in content_type
    ):
        # Bedrock only — never a routed protocol, so no route wrapper.
        return StreamingResponse(
            _ClientPaced(
                _stream_rehydrated_eventstream(
                    upstream, adapter, state, ctx, request_meta=request_meta, observe=observe
                )
            ),
            status_code=upstream.status_code,
            headers=headers,
            media_type="application/vnd.amazon.eventstream",
        )

    if (
        streamed
        and adapter is not None
        and adapter.handles_ndjson
        and any(t in content_type for t in _JSONL_CONTENT_TYPES)
    ):
        return StreamingResponse(
            _ClientPaced(
                _stream_rehydrated_ndjson(
                    upstream,
                    adapter,
                    state,
                    ctx,
                    request_meta=request_meta,
                    route=route,
                    observe=observe,
                )
            ),
            status_code=upstream.status_code,
            headers=headers,
            media_type="application/x-ndjson",
        )

    try:
        raw = await _waited(upstream.aread())
    except httpx.TransportError as exc:
        # Upstream dropped mid-body on a buffered response: close the
        # connection we opened (else it leaks) and fail closed with a 502.
        await upstream.aclose()
        return _fault_response(
            state,
            ctx,
            adapter,
            exc,
            request=request,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
            audit_token=audit_token,
            route=route,
        )
    await upstream.aclose()

    received = raw  # the provider's own bytes: what an observer reads
    rehydration_counts_before = dict(state.rehydration_counts)
    with state.map_write_barrier() as map_writes:
        try:
            raw = _restore_buffered(
                request,
                state,
                ctx,
                adapter,
                kind,
                route,
                raw,
                path=path,
                content_type=content_type,
                status=upstream.status_code,
                request_body=request_body,
            )
        except Exception as exc:  # noqa: BLE001 — never a bare 500 once the provider answered
            return _delivery_fault_response(
                state,
                ctx,
                adapter,
                exc,
                request=request,
                path=path,
                started=started,
                new_counts=new_counts,
                new_warned=new_warned,
                audit_token=audit_token,
                route=route,
            )
        if observe is not None and received and "application/json" in content_type:
            _observe(state, observe, request.method, path, received)
    # This answer's share of the process-wide counts, taken before any await.
    rehydrations = _count_delta(state.rehydration_counts, rehydration_counts_before)
    if map_writes:
        # [vault] map_writes = "before_answer": the ids this answer recorded
        # reach the client only once every replica can read their records.
        await state.await_map_writes(map_writes, f"{request.method} {path}")

    # Every request is recorded — pass-through included (provider=None maps
    # to the "passthrough" metrics label); audit rows likewise when enabled.
    state.record_request(
        session=ctx.session_id,
        provider=adapter.name if adapter is not None else None,
        method=request.method,
        path=path,
        status=upstream.status_code,
        started=started,
        streamed=False,
        detections=new_counts,
        rehydrations=rehydrations,
        warned=new_warned,
        audit_token=audit_token,
        route=(state.finish_route(route, upstream.status_code) if route is not None else None),
        refusal=None,
    )

    return Response(content=raw, status_code=upstream.status_code, headers=headers)


def _is_redirect(upstream: httpx.Response) -> bool:
    """A redirect a client would follow: 3xx with a Location (304 Not
    Modified is a cache answer, not a redirect)."""
    status = upstream.status_code
    return 300 <= status < 400 and status != 304 and "location" in upstream.headers


def _redirect_refused(
    state: ProxyState,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    status: int,
    *,
    request: Request,
    path: str,
    started: float,
    new_counts: dict[str, int],
    new_warned: dict[str, int],
    audit_token: object | None,
    route: RouteDelivery | None,
) -> JSONResponse:
    """The upstream answered a redirect where relaying it would leak: a
    following client (httpx, fetch, reqwest) repeats its ORIGINAL request at
    the Location — the unredacted body the proxy redacted, and the credential
    headers those clients keep across hosts (``x-api-key``, ``api-key``,
    ``x-llm-redact-user``) — or, for a relative Location, at a proxy path that
    may belong to another provider; a routed or identity-signed request is
    never handed back to the client to finish elsewhere. A recorded,
    provider-shaped 502 counted as an upstream error, naming the status
    only: the Location is never relayed or logged (it can carry a presigned
    credential)."""
    if route is not None:
        state.upstream_errors[route.upstream] += 1
    else:
        state.upstream_errors[adapter.name if adapter is not None else "passthrough"] += 1
    row = state.finish_route(route, 502) if route is not None else None
    logger.warning(
        "%s %s -> 502 the upstream answered a redirect (%d), not relayed%s",
        request.method,
        path,
        status,
        _route_log_suffix(row) if row is not None else "",
    )
    state.record_request(
        session=ctx.session_id,
        provider=adapter.name if adapter is not None else None,
        method=request.method,
        path=path,
        status=502,
        started=started,
        streamed=False,
        detections=new_counts,
        rehydrations={},
        warned=new_warned,
        audit_token=audit_token,
        route=row,
        refusal="redirect_refused",
    )
    message = (
        f"llm-redact: the upstream answered a redirect ({status}), which llm-redact does not"
        " relay: a client following it would re-send the original request, unredacted and"
        " with its credentials, to wherever it points; set the provider's upstream_base_url"
        " to the API's final https URL"
    )
    body = adapter.error_body(message, status=502) if adapter is not None else {"error": message}
    return JSONResponse(
        body, status_code=502, headers=dict(route.headers) if route is not None else None
    )


def _restore_buffered(
    request: Request,
    state: ProxyState,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    kind: RouteKind,
    route: RouteDelivery | None,
    raw: bytes,
    *,
    path: str,
    content_type: str,
    status: int,
    request_body: Any,
) -> bytes:
    """A buffered upstream answer as the client receives it: rehydrated in
    the request's session (CHAT), observed by the router, listed objects
    restored in their owners' sessions, stored objects and response ids
    reported. The session bookkeeping is contained (``_contained``: the
    answer is still delivered); a fault restoring the answer itself raises
    to ``_deliver``'s backstop."""
    received = raw  # the provider's own bytes (a listing restores items from these)
    rehydration_counts_before = dict(state.rehydration_counts)
    payload: Any = None
    if (
        kind is RouteKind.CHAT
        and adapter is not None
        and adapter.restores_file_download(request.method, path)
    ):
        # A downloaded file, whatever its Content-Type (a JSON file served
        # as application/json included): restored per file, as it was
        # redacted on upload — never walked as one JSON body, which would
        # skip its keys and structural names and re-serialize the file.
        raw_rehydrated = adapter.rehydrate_raw_body(path, raw, ctx.rehydrator) if raw else None
        if raw_rehydrated is not None:
            raw = raw_rehydrated
    elif kind is RouteKind.CHAT and adapter is not None and "application/json" in content_type:
        try:
            payload = loads_bounded(raw)
        except ValueError:
            payload = None  # not one JSON value: forwarded as sent
        if payload is not None:
            response_id = adapter.response_id_from_body(payload)
            if response_id is not None:
                _contained(
                    state,
                    "response_id",
                    request.method,
                    path,
                    state.record_response_id,
                    response_id,
                    ctx.session_id,
                )
            rehydrated = adapter.rehydrate_body(payload, ctx.rehydrator)
            # No-op short-circuit: every restore increments rehydration_counts
            # (a miss passes through verbatim), so an unchanged count means the
            # response had no tokens to restore — forward the original bytes
            # instead of re-serializing.
            changed = sum(state.rehydration_counts.values()) != sum(
                rehydration_counts_before.values()
            )
            if route is not None and route.observe_payload(rehydrated, kind):
                changed = True
            if changed:
                raw = json_bytes(rehydrated)
    elif (
        route is not None
        and raw
        and "application/json" in content_type
        and route.wants_payload(kind)
    ):
        # A routed JSON response outside the CHAT branches (REDACT_ONLY —
        # embeddings — or pass-through): nothing to rehydrate, but the router
        # may still want the body (a rewritten model id restored in every
        # top-level `model` field — decision 12 — and a REDACT_ONLY body's
        # own usage block, the billed call's, R-24). A body the router does
        # not want is not parsed at all (a large file listing forwards
        # untouched).
        try:
            payload = loads_bounded(raw)
        except ValueError:
            payload = None
        if payload is not None and route.observe_payload(payload, kind):
            raw = json_bytes(payload)

    lister = (
        state.object_lister(adapter, request.method, path, request.headers, request.url.query)
        if raw and 200 <= status < 300 and "application/json" in content_type
        else None
    )
    if lister is not None:
        # A listing of stored objects: the session router may name the
        # session each listed object was created in (its owner's own
        # listing), so those items are restored there; the rest keep what
        # this request's session made of them. Never recorded as ownership.
        restored = _restore_listing(state, lister, received, raw)
        if restored is not None:
            raw = restored

    tracker = (
        state.object_tracker(
            adapter,
            request.method,
            path,
            request.headers,
            body=request_body,
            query=request.url.query,
        )
        if raw and 200 <= status < 300 and "application/json" in content_type
        else None
    )
    if tracker is not None:
        # Objects the provider stores for later reads (uploaded files,
        # batches, stored conversations, the files a tool run wrote): their
        # ids go to the session router with the session that created them —
        # never an id the request itself cites. A body no adapter tracks is
        # never parsed here (pass-through routes carry no adapter, so the
        # provider's is looked up by name). Contained, reading included: a
        # lost record is the router's unknown-object case, never a lost answer.
        try:
            stored = payload if payload is not None else loads_bounded(raw)
        except ValueError:
            stored = None
        _contained(
            state,
            "object_ids",
            request.method,
            path,
            _report_object_ids,
            state,
            tracker,
            request.method,
            path,
            stored,
            request_body,
            ctx.session_id,
        )
    return raw


def _report_object_ids(
    state: ProxyState,
    tracker: ProviderAdapter,
    method: str,
    path: str,
    answer: Any,
    request_body: Any,
    session_id: str,
) -> None:
    """Report the stored objects a buffered answer names that the request
    itself does not cite (``_uncited``)."""
    object_ids = tracker.object_ids_from_body(method, path, answer)
    created = _uncited(object_ids, request_body) if object_ids else []
    if created:
        state.record_object_ids(created, session_id)


def _contained(
    state: ProxyState,
    stage: str,
    method: str,
    path: str,
    bookkeeping: Callable[..., object],
    *args: Any,
) -> bool:
    """Run session bookkeeping that follows the provider's answer (a router
    call, a durable-map write): a fault is counted and logged by stage and
    exception TYPE (a message can carry an id or a DSN) and never reaches
    the response — the answer the provider already produced (and billed) is
    delivered. Losing a record is fail-safe by construction: an unrecorded
    response id or stored object is the router's unknown-object case (an
    empty session: placeholders pass through), never a wrong value. False
    when it failed."""
    try:
        bookkeeping(*args)
    except Exception as exc:  # noqa: BLE001 — contained by design, see above
        _bookkeeping_fault(state, stage, method, path, exc)
        return False
    return True


def _bookkeeping_fault(
    state: ProxyState, stage: str, method: str, path: str, exc: Exception
) -> None:
    """Count and log one contained bookkeeping fault: the stage and the
    exception TYPE only."""
    state.bookkeeping_errors[stage] += 1
    logger.warning(
        "%s %s -> %s bookkeeping failed (%s); the answer is delivered",
        method,
        path,
        stage,
        type(exc).__name__,
    )


# The bookkeeping stage a session router's response observer faults count in.
_OBSERVER_STAGE = "response_observer"


def _observe(
    state: ProxyState, observe: ResponseObserver, method: str, path: str, data: bytes | str
) -> ResponseObserver | None:
    """Hand ``observe`` its OWN parse of one JSON value of the provider's
    answer (``data``: a buffered body, an SSE event's data, an NDJSON line,
    an eventstream frame's payload) — before rehydration, and never the
    object the client's answer is built from, so the observer can change
    nothing the client receives. What is not JSON (or nests deeper than
    MAX_JSON_DEPTH) is skipped. Returns ``observe``, or None once it failed:
    the fault is contained (``_contained``) and that answer is observed no
    further."""
    try:
        payload = loads_bounded(data)
    except ValueError:
        return observe
    return observe if _contained(state, _OBSERVER_STAGE, method, path, observe, payload) else None


async def _observed(
    state: ProxyState, observe: ResponseObserver, method: str, path: str, data: bytes | str
) -> ResponseObserver | None:
    """``_observe`` one streamed value — then, with ``[vault] map_writes =
    "before_answer"``, wait for the map writes the observer queued (a
    record it made of this value) before the value is sent on."""
    with state.map_write_barrier() as map_writes:
        observe_next = _observe(state, observe, method, path, data)
    if map_writes:
        await state.await_map_writes(map_writes, f"{method} {path}")
    return observe_next


def _delivery_fault_response(
    state: ProxyState,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    exc: Exception,
    *,
    request: Request,
    path: str,
    started: float,
    new_counts: dict[str, int],
    new_warned: dict[str, int],
    audit_token: object | None,
    route: RouteDelivery | None,
) -> JSONResponse:
    """The provider answered, but restoring that answer failed (a vault read
    that cannot complete, a router hook): fail closed with a provider-shaped
    502 — never a bare 500, never a partial or unrestored body — recorded
    (the ``[audit] required`` END row included) and counted. The upstream
    itself did not fail: not an upstream error, and the route is closed as
    a 502 without a fault class. By exception TYPE only."""
    state.bookkeeping_errors["delivery"] += 1
    row = state.finish_route(route, 502) if route is not None else None
    logger.warning(
        "%s %s -> 502 restoring the upstream answer failed (%s)%s",
        request.method,
        path,
        type(exc).__name__,
        _route_log_suffix(row) if row is not None else "",
    )
    state.record_request(
        session=ctx.session_id,
        provider=adapter.name if adapter is not None else None,
        method=request.method,
        path=path,
        status=502,
        started=started,
        streamed=False,
        detections=new_counts,
        rehydrations={},
        warned=new_warned,
        audit_token=audit_token,
        route=row,
        refusal="delivery_fault",
    )
    message = "llm-redact: the upstream answer could not be restored; nothing was delivered"
    body = adapter.error_body(message, status=502) if adapter is not None else {"error": message}
    return JSONResponse(
        body, status_code=502, headers=dict(route.headers) if route is not None else None
    )


def _object_access_refused(
    state: ProxyState,
    adapter: ProviderAdapter | None,
    message: str,
    *,
    provider_name: str,
    request: Request,
    path: str,
    started: float,
) -> JSONResponse:
    """The session router's stored-object refusal: a recorded,
    provider-shaped 403 (the access gate's conventions), sent before any
    upstream contact."""
    error = adapter.error_body(message, status=403) if adapter is not None else {"error": message}
    state.record_request(
        session=state.config.vault.session,
        provider=provider_name,
        method=request.method,
        path=path,
        status=403,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal="object_access",
    )
    logger.info("%s %s -> 403 refused by the session router (stored object)", request.method, path)
    return JSONResponse(error, status_code=403)


def _authorization_request(
    request: Request,
    adapter: ProviderAdapter | None,
    kind: RouteKind,
    *,
    provider_name: str,
    path: str,
    model: str | None,
    identity: bool,
) -> AuthorizationRequest:
    """This request's FACTS (``plugin_api.AuthorizationRequest``), as the
    access gate's ``authorize_request`` and ``authorize_content`` get them."""
    return AuthorizationRequest(
        surface="http",
        provider=provider_name,
        adapter=adapter.name if adapter is not None else None,
        kind=kind.value,
        method=request.method,
        path=path,
        model=model,
        identity=identity,
    )


async def _authorization_check(
    state: ProxyState,
    request: Request,
    asked: AuthorizationRequest,
    adapter: ProviderAdapter | None,
    *,
    provider_name: str,
    path: str,
    started: float,
) -> JSONResponse | None:
    """The access gate's optional ``authorize_request`` on this request's
    FACTS (``asked``; asked only when the gate has the member): None when
    it allows the request, else the recorded 403 refusing it. A synchronous
    answer adds no event-loop turn."""
    verdict = state.authorization.refusal(asked, f"{request.method} {path}")
    if inspect.isawaitable(verdict):
        verdict = await verdict
    if verdict is None:
        return None
    return _authorization_refused(
        state,
        adapter,
        verdict,
        provider_name=provider_name,
        request=request,
        path=path,
        started=started,
    )


async def _content_check(
    state: ProxyState,
    request: Request,
    asked: AuthorizationRequest,
    content: ContentFacts,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    *,
    provider_name: str,
    path: str,
    started: float,
    new_counts: dict[str, int],
    new_warned: dict[str, int],
) -> JSONResponse | None:
    """The access gate's optional ``authorize_content`` on what this
    request's redaction found (``content``, the request's own counts, taken
    before this await): None when it allows the request, else the recorded
    403 refusing it — before the audit START row, the upstream authorizer, a
    routed plan's ``begin()`` and any upstream contact. The placeholders the
    redaction issued stay in the vault (harmless: nothing was forwarded); a
    one-time override it used is handed back by ``handle()``'s refused
    settlement (``_UploadFate``)."""
    verdict = state.authorization.content_refusal(asked, content, f"{request.method} {path}")
    if inspect.isawaitable(verdict):
        verdict = await verdict
    if verdict is None:
        return None
    return _authorization_refused(
        state,
        adapter,
        verdict,
        provider_name=provider_name,
        request=request,
        path=path,
        started=started,
        session=ctx.session_id,
        detections=new_counts,
        warned=new_warned,
    )


def _authorization_refused(
    state: ProxyState,
    adapter: ProviderAdapter | None,
    message: str,
    *,
    provider_name: str,
    request: Request,
    path: str,
    started: float,
    session: str | None = None,
    detections: dict[str, int] | None = None,
    warned: dict[str, int] | None = None,
) -> JSONResponse:
    """The access gate's authorization refusal (or the core's fixed text
    when its check or the requester's detection overlay failed): a recorded,
    provider-shaped 403, sent before any upstream contact — before the
    session and redaction, except a refusal of what the redaction found
    (``authorize_content``: its row carries the request's session and
    counts). The reason reaches the client only, never the log."""
    error = adapter.error_body(message, status=403) if adapter is not None else {"error": message}
    state.record_request(
        session=session if session is not None else state.config.vault.session,
        provider=provider_name,
        method=request.method,
        path=path,
        status=403,
        started=started,
        streamed=False,
        detections=detections or {},
        rehydrations={},
        warned=warned or None,
        refusal="authorization",
    )
    logger.info("%s %s -> 403 refused by the access gate (authorization)", request.method, path)
    return JSONResponse(error, status_code=403)


def _unchecked_body_refused(
    state: ProxyState,
    adapter: ProviderAdapter | None,
    unreadable: _Unreadable,
    *,
    provider_name: str,
    request: Request,
    path: str,
    started: float,
    identity: bool,
) -> JSONResponse:
    """A request the proxy would send with its own credential — its cloud
    identity (``identity``), or a routed plan's — whose body the
    stored-object check cannot read (or, not ``credential_bound``, any
    request whose upload the check's re-reading would change): refused,
    recorded, before any upstream contact (provider-shaped on a matched
    route, the pass-through shape otherwise)."""
    credential = (
        "this provider is authorized with the proxy's own identity"
        if identity
        else "this request would be sent with the proxy's own provider credential"
    )
    message = (
        f"llm-redact: {unreadable.message}, and {credential}, so the stored objects it"
        " cites must be checked; the request was not forwarded"
        if unreadable.credential_bound
        else f"llm-redact: {unreadable.message}; the request was not forwarded"
    )
    state.record_request(
        session=state.config.vault.session,
        provider=provider_name,
        method=request.method,
        path=path,
        status=unreadable.status,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal=_unreadable_kind(unreadable, "unchecked_body"),
    )
    logger.info(
        "%s %s -> %d refused (%s)",
        request.method,
        path,
        unreadable.status,
        "unchecked body under the proxy's credential"
        if unreadable.credential_bound
        else "an upload the stored-object check cannot read as sent",
    )
    error = (
        adapter.error_body(message, status=unreadable.status)
        if adapter is not None
        else {"error": message}
    )
    return JSONResponse(
        error,
        status_code=unreadable.status,
        headers=_IDENTITY_ONLY if unreadable.status == 415 else None,
    )


def _unscanned_body_refused(
    state: ProxyState,
    adapter: ProviderAdapter,
    unreadable: _Unreadable,
    *,
    clause: str,
    provider_name: str,
    request: Request,
    path: str,
    started: float,
    hint: str | None = None,
) -> JSONResponse:
    """A body the scanned-body rule refuses (``_unscanned_body``): recorded,
    provider-shaped, before the stored-object check, the session, redaction,
    any credential, the plan and any upstream contact — so no audit START
    row either. The message names the body's kind and why the rule holds
    (``clause``), never the content."""
    message = f"llm-redact: {unreadable.message}, and {clause}; the request was not forwarded"
    if hint is not None:
        message += f"; {hint}"
    state.record_request(
        session=state.config.vault.session,
        provider=provider_name,
        method=request.method,
        path=path,
        status=unreadable.status,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal=_unreadable_kind(unreadable, "scanned_body"),
    )
    logger.info(
        "%s %s -> %d refused (a request body llm-redact cannot redact)",
        request.method,
        path,
        unreadable.status,
    )
    return JSONResponse(
        adapter.error_body(message, status=unreadable.status),
        status_code=unreadable.status,
        headers=_IDENTITY_ONLY if unreadable.status == 415 else None,
    )


def _restore_listing(
    state: ProxyState, lister: ProviderAdapter, upstream_raw: bytes, raw: bytes
) -> bytes | None:
    """Restore each listed stored object in the session the router names
    for it (``listing_item_session``); an item the router named no session
    for, and everything outside the item array, stays exactly as ``raw``
    (the bytes about to be delivered) has it. A named item is rebuilt from
    the provider's own bytes (``upstream_raw``) — rehydrated as a whole
    object with the adapter's non-streaming transform, never a second pass
    over an already-restored item; a named session that does not exist (or
    holds nothing) restores nothing, and neither does a router that failed
    to answer, so such an item is delivered exactly as the provider sent
    it. None when nothing changed (the bytes are then forwarded
    untouched)."""
    try:
        original = loads_bounded(upstream_raw)
    except ValueError:
        return None
    items = lister.listing_items(original)
    if not items:
        return None
    listed = {
        index: object_id
        for index, object_id in enumerate(map(lister.listing_item_id, items))
        if object_id is not None
    }
    by_id = state.listing_restorers(list(dict.fromkeys(listed.values())))
    restorers = {
        index: by_id[object_id] for index, object_id in listed.items() if object_id in by_id
    }
    if not restorers:
        return None
    # A fresh tree to edit (raw is JSON: upstream_raw or a dump of it).
    delivered = loads_bounded(raw)
    delivered_items = lister.listing_items(delivered)
    if delivered_items is None or len(delivered_items) != len(items):
        return None  # a routed rewrite changed the shape: leave it alone
    changed = False
    for index, rehydrator in restorers.items():
        restored = items[index]
        if rehydrator is not None:
            try:
                restored = lister.rehydrate_body(items[index], rehydrator)
            except Exception as exc:  # noqa: BLE001 — as above: the provider's item
                _listing_fault(state, exc)
        if restored != delivered_items[index]:
            delivered_items[index] = restored
            changed = True
    return json_bytes(delivered) if changed else None


def _listing_fault(state: ProxyState, exc: Exception) -> None:
    """A listed item whose owner's session could not be read: counted and
    logged by exception TYPE; the item goes out as the provider sent it."""
    state.bookkeeping_errors["listing"] += 1
    logger.warning(
        "listing restore failed for an item (%s); delivered as the provider sent it",
        type(exc).__name__,
    )


# ---------------------------------------------------------------------------
# Routing: the policy-free driver of a routed request. Every DECISION (rule
# selection, credentials, chains, cooldowns, plan-limit classification,
# budgets, prices, model rewrite/restore) is the registered Router's — the
# llm-redact-pro routing layer behind the plugin_api contract; the core only
# issues hops, closes what it does not deliver, delivers through the shared
# branches above, and finalizes.
# ---------------------------------------------------------------------------


def _route_refusal(
    state: ProxyState,
    session: str,
    adapter: ProviderAdapter | None,
    refusal: RouteRefusal,
    *,
    kind: LocalRefusal,
    request: Request,
    path: str,
    started: float,
    new_counts: dict[str, int] | None = None,
    new_warned: dict[str, int] | None = None,
    audit_token: object | None = None,
) -> JSONResponse:
    """A proxy-generated routing refusal (the router's 502 no_route, 404
    count_tokens, 402 budget — ``kind``: which of the three the router
    answered, by where it answered it): provider-shaped, recorded with its
    route row, no upstream contact."""
    row = dict(refusal.row)
    logger.info("%s %s -> %d%s", request.method, path, refusal.status, _route_log_suffix(row))
    state.record_request(
        session=session,
        provider=adapter.name if adapter is not None else None,
        method=request.method,
        path=path,
        status=refusal.status,
        started=started,
        streamed=False,
        detections=new_counts or {},
        rehydrations={},
        warned=new_warned,
        audit_token=audit_token,
        route=row,
        refusal=kind,
    )
    body = (
        adapter.error_body(refusal.message, status=refusal.status)
        if adapter is not None
        else {"error": refusal.message}
    )
    return JSONResponse(
        body,
        status_code=refusal.status,
        headers=dict(refusal.headers) if refusal.headers else None,
    )


def _answer_locally(
    state: ProxyState, request: Request, answer: LocalAnswer, *, path: str, started: float
) -> Response:
    """A request the router answers before the body is read (R-15): recorded
    under the router's provider label, logged with its reason token."""
    state.record_request(
        session=state.config.vault.session,
        provider=answer.provider,
        method=request.method,
        path=path,
        status=answer.status,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
        refusal=None,
    )
    logger.info(
        "%s %s -> %d answered locally (%s)", request.method, path, answer.status, answer.reason
    )
    return JSONResponse(dict(answer.body), status_code=answer.status)


def _hop_timeout(remaining: float) -> httpx.Timeout:
    """The httpx timeout for one hop, derived from the deadline's remaining
    budget (R-18: `request_deadline_seconds` bounds the TOTAL wall time, so
    a hop that accepts the connection and then hangs must fail inside it —
    not after the shared client's 600 s). Each phase is capped by what is
    left; the caps never exceed the client's own (600 s read/write/pool,
    10 s connect), so the default 600 s deadline changes nothing. The
    floor keeps a deadline that expires mid-hop from turning into a zero
    timeout (httpx treats 0 as "no timeout"). A DELIVERED stream keeps this
    per-read timeout for its lifetime: under a short deadline a stream that
    falls silent for longer than the budget is cut off (finalized as a
    stream error) rather than allowed to outlive it."""
    budget = max(remaining, 0.001)
    return httpx.Timeout(min(budget, 600.0), connect=min(budget, 10.0))


async def _discard(response: httpx.Response | None) -> None:
    """Close an upstream response that will not be delivered (a failed hop
    or a throttled attempt) before the next one is sent."""
    if response is not None:
        with suppress(Exception):
            await response.aclose()


def _streams_to_client(content_type: str) -> bool:
    """Whether `_deliver` streams a response of this content type (SSE or
    NDJSON) rather than buffering it — the R-17 line between a body that can
    still be re-issued and one whose first byte is already on its way."""
    return "text/event-stream" in content_type or any(
        t in content_type for t in _JSONL_CONTENT_TYPES
    )


async def _read_buffered(response: httpx.Response) -> str | None:
    """Read the body of a response `_deliver` would buffer, so a mid-body
    drop surfaces while no byte has reached the client (decision 8: the
    fault is status key "502" for the chain lookup). Returns the fault's
    exception TYPE name (an httpx message can embed the URL), or None when
    the body is in hand — `aread()` is idempotent, so the delivery branch
    re-reads it for free — or will be streamed."""
    if _streams_to_client(response.headers.get("content-type", "")):
        return None
    try:
        await _waited(response.aread())
    except httpx.TransportError as exc:
        return type(exc).__name__
    return None


async def _issue_hop(
    state: ProxyState, hop: HopRequest, deadline: float, *, method: str, path: str
) -> tuple[HopResult, httpx.Response | None]:
    """Send one attempt and classify what came back — mechanics only. A hop
    the router marked unavailable is a fault without a send (the text names
    the variable only). Faults log by exception TYPE (an httpx message can
    embed the URL) and count against the upstream NAME (decision 8). A body
    the delivery branch would BUFFER is read now (R-17: no byte has reached
    the client, so a mid-body drop is still re-issuable); a streamed body is
    left untouched."""
    if hop.unavailable is not None:
        logger.warning(
            "%s %s upstream %s unavailable: %s", method, path, hop.upstream, hop.unavailable
        )
        state.upstream_errors[hop.upstream] += 1
        return HopResult(hop.upstream, None, {}, "MissingCredential"), None
    upstream_request = state.client.build_request(
        method,
        hop.url,
        headers=list(hop.headers),
        content=hop.body,
        timeout=_hop_timeout(deadline - time.monotonic()),
    )
    try:
        response = await _waited(state.client.send(upstream_request, stream=True))
    except httpx.TransportError as exc:
        logger.warning(
            "%s %s upstream %s fault (%s)", method, path, hop.upstream, type(exc).__name__
        )
        state.upstream_errors[hop.upstream] += 1
        return HopResult(hop.upstream, None, {}, type(exc).__name__), None
    if _is_redirect(response):
        # Never relayed (a following client would re-send the unredacted
        # original, credentials and all — _redirect_refused): a fault of
        # this hop, so the router may fail over; the Location is never
        # logged.
        logger.warning(
            "%s %s upstream %s answered a redirect (%d), not relayed",
            method,
            path,
            hop.upstream,
            response.status_code,
        )
        await _discard(response)
        state.upstream_errors[hop.upstream] += 1
        return HopResult(hop.upstream, None, {}, "UpstreamRedirect"), None
    fault = await _read_buffered(response)
    if fault is not None:
        logger.warning(
            "%s %s upstream %s fault while reading the body (%s)", method, path, hop.upstream, fault
        )
        await _discard(response)
        state.upstream_errors[hop.upstream] += 1
        return HopResult(hop.upstream, None, {}, fault), None
    return HopResult(hop.upstream, response.status_code, response.headers, None), response


async def _handle_routed(
    request: Request,
    state: ProxyState,
    ctx: RequestContext,
    plan: RoutePlan,
    *,
    adapter: ProviderAdapter | None,
    kind: RouteKind,
    outbound: bytes,
    outbound_obj: dict[str, Any] | None,
    path: str,
    started: float,
    new_counts: dict[str, int],
    new_warned: dict[str, int],
    identity: bool = False,
    upload: _UploadFate | None = None,
) -> Response:
    """The routed request path: the router's pre-audit refusal, the
    write-ahead audit START row, the first hop, then issue/decide until the
    router stops; delivery rides the shared branches with the router's
    delivery hooks. Every decision is the router's; every byte moved is the
    core's. An upload's counts settle as sent at the first hop actually
    issued (``upload``; a refusal before one leaves them to the caller)."""
    method = request.method
    early = _EARLY_AUDIT.get()
    if early is not None and early.started:
        # An upload with a binary part to inspect: its local refusal was
        # asked and its START row written before the inspection.
        audit_token, audit_refusal = _start_after_early(
            state,
            early,
            ctx,
            adapter,
            request=request,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
        )
        if audit_refusal is not None:
            return audit_refusal
    else:
        # The count_tokens 404 (decision 7): a refusal that never counts as
        # an attempt, so it precedes the write-ahead START row.
        refused = plan.local_refusal()
        if refused is not None:
            return _route_refusal(
                state,
                ctx.session_id,
                adapter,
                refused,
                request=request,
                path=path,
                started=started,
                kind="route_unsupported",
                new_counts=new_counts,
                new_warned=new_warned,
            )
        audit_token, audit_refusal = _begin_audit_guarded(
            state,
            ctx,
            adapter,
            request=request,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
        )
        if audit_refusal is not None:
            return audit_refusal
    # The budget 402 (an attempt that was refused locally, so it carries the
    # audit token) or the first hop (hop 1, or hop 2 of the budget chain).
    first = plan.begin(
        outbound, outbound_obj, _request_headers(request, matched=adapter is not None)
    )
    if isinstance(first, RouteRefusal):
        return _route_refusal(
            state,
            ctx.session_id,
            adapter,
            first,
            request=request,
            path=path,
            started=started,
            kind="budget",
            new_counts=new_counts,
            new_warned=new_warned,
            audit_token=audit_token,
        )
    hop: HopRequest = first
    response: httpx.Response | None = None
    while True:
        if hop.reissued_from is not None:
            state.metrics.reissues[(hop.reissued_from, hop.upstream)] += 1
        if upload is not None and hop.unavailable is None:
            upload.settle(sent=True)  # this hop is sent
        result, response = await _issue_hop(state, hop, plan.deadline, method=method, path=path)
        decision = plan.decide(result)
        if decision.next is None:
            break
        await _discard(response)
        response = None
        if decision.wait_seconds > 0:
            await _waited(plan.wait(decision.wait_seconds))
        hop = decision.next
    route = plan.delivery()
    if response is None:
        # The last hop faulted (or its credential vanished) and nothing took
        # over: the provider-shaped 502, recorded and closed by the router.
        row = state.finish_route(route, 502)
        logger.info("%s %s -> 502%s", method, path, _route_log_suffix(row))
        state.record_request(
            session=ctx.session_id,
            provider=adapter.name if adapter is not None else None,
            method=method,
            path=path,
            status=502,
            started=started,
            streamed=False,
            detections=new_counts,
            rehydrations={},
            warned=new_warned,
            audit_token=audit_token,
            route=row,
            refusal="upstream_fault",
        )
        message = "llm-redact: upstream request failed"
        error = (
            adapter.error_body(message, status=502) if adapter is not None else {"error": message}
        )
        return JSONResponse(error, status_code=502, headers=dict(route.headers))
    logger.info(
        "%s %s -> %d%s%s",
        method,
        path,
        response.status_code,
        _route_log_suffix(route.row()),
        _redacted_summary(new_counts),
    )
    return await _deliver(
        request,
        state,
        ctx,
        adapter,
        kind,
        response,
        path=path,
        started=started,
        new_counts=new_counts,
        new_warned=new_warned,
        audit_token=audit_token,
        route=route,
        request_body=outbound_obj,
        sent_body=outbound_obj,
        identity=identity,
    )


async def _drain_map_writes(state: ProxyState) -> None:
    """At shutdown, before the vault closes: wait (off the loop, at most
    ``SHUTDOWN_DRAIN_SECONDS``) for the durable map writes still queued to
    land; ``close`` then counts and drops what has not, logging their
    number only."""
    drain = getattr(state.vault_manager, "drain_map_writes", None)
    if callable(drain):
        await asyncio.to_thread(drain, SHUTDOWN_DRAIN_SECONDS)


def create_app(
    config: Config,
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    config_path: Path | None = None,
) -> Starlette:
    state = ProxyState(config, upstream_transport, config_path=config_path)

    async def ttl_prune_loop(interval: float) -> None:
        """Retention: prune whole sessions idle longer than session_ttl_days.

        Reuses the live-safe manager prune (whole sessions only, evicts cached
        views) and never touches the active static session or a session the
        router marks durable — the same never-wrong-value discipline as the
        CLI prune. Sleep-first so startup is untouched; failures are logged,
        never fatal."""
        ttl = state.config.vault.session_ttl_days
        while True:
            await asyncio.sleep(interval)
            try:
                # Recomputed per pass: the router's durable sessions (a named
                # user's namespace) appear as users show up.
                pruned = state.vault_manager.prune_sessions(ttl, exclude=_prune_exclusions(state))
            except Exception:
                logger.exception("session-ttl prune failed")
                continue
            if pruned:
                logger.info("session-ttl: pruned %d session(s) idle > %d days", pruned, ttl)

    async def license_refresh_loop(interval: float) -> None:
        """Daily re-resolution of the license for warning purposes only —
        the T-30 expiry warning must reach a proxy that never restarts.
        Sleep-first; failures are logged, never fatal; the enforced tier
        never changes here (refresh_license_warnings documents why)."""
        while True:
            await asyncio.sleep(interval)
            try:
                state.refresh_license_warnings()
            except Exception:
                logger.exception("license warning refresh failed")

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        sighup_registered = False
        if hasattr(signal, "SIGHUP"):
            try:
                loop.add_signal_handler(signal.SIGHUP, state.reload)
                sighup_registered = True
            except (NotImplementedError, RuntimeError, ValueError):
                # Windows event loops / non-main threads (uvloop raises
                # ValueError there, asyncio RuntimeError): reload unavailable.
                logger.debug("SIGHUP reload unavailable on this platform")
        # Each active off-machine audit sink gets a flush-loop task; at
        # shutdown the loops are cancelled and each sink gets a bounded final
        # flush BEFORE the audit database closes, so no tail is lost.
        # The optional session-ttl prune loop rides in the same task list.
        audit_sinks = [s for s in (state.audit_s3, state.audit_azure) if s is not None]
        background_tasks = [asyncio.create_task(sink.run()) for sink in audit_sinks]
        if state.config.vault.session_ttl_days > 0:
            background_tasks.append(
                asyncio.create_task(ttl_prune_loop(_TTL_PRUNE_INTERVAL_SECONDS))
            )
        # Always on: [license] is hot-editable, so a key can appear after
        # startup; a keyless daily re-resolution is a no-op.
        background_tasks.append(
            asyncio.create_task(license_refresh_loop(_LICENSE_REFRESH_INTERVAL_SECONDS))
        )
        # The open-connection re-check backstop (connections.LiveConnections):
        # started now, or with the first connection whose admission carries
        # a recheck; stopped before the gate closes.
        state.connections.start()
        try:
            yield
        finally:
            # Drain order (docs/resilience.md, "Shutdown order"): the server
            # has stopped accepting and waited for in-flight requests, so
            # every request finalizer has written its END row. Then: stop
            # the background work, close what serves requests, give the
            # sinks their final flush from the STILL-OPEN audit database,
            # close the audit database, and the vault last (its background
            # map writes drained before the sinks' flush).
            await state.connections.stop()
            if sighup_registered:
                loop.remove_signal_handler(signal.SIGHUP)
            for task in background_tasks:
                task.cancel()
            # A task that had already died (a sink's run() raising) must not
            # cut the shutdown short before the databases close.
            for outcome in await asyncio.gather(*background_tasks, return_exceptions=True):
                if isinstance(outcome, Exception):
                    logger.warning("a background task had failed (%s)", type(outcome).__name__)
            await state.client.aclose()
            # The vault's background map writes land now (bounded): nothing
            # submits another once the requests have finished.
            await _drain_map_writes(state)
            if state.router is not None:
                _close_contained("router", state.router.close)
            _close_upstream_auths(state.upstream_auth)
            if state.access_gate is not None:
                _close_contained("access gate", state.access_gate.close)
            if state.upload_inspector is not None:
                # Its worker processes and HTTP clients; a fault closing them
                # never stops the rest of the shutdown.
                try:
                    await state.upload_inspector.aclose()
                except Exception:
                    logger.exception("closing the upload inspector failed")
            # Final flush so shutdown never silently drops the tail; bounded,
            # so a hanging store never keeps the databases open.
            await _close_audit_sinks(audit_sinks, _SINK_CLOSE_TIMEOUT_SECONDS)
            if state.audit is not None:
                state.audit.close()
            if state.overrides is not None:
                state.overrides.close()
            state.vault_manager.close()
            if state.telemetry is not None:
                # Flush the batched exporters; telemetry buffered at shutdown
                # would otherwise be dropped.
                state.telemetry.shutdown()

    app = Starlette(
        routes=[
            Route(
                "/{path:path}",
                handle,
                methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
            ),
            # Realtime provider APIs upgrade to WebSocket; the relay in
            # realtime.py refuses unknown paths (there is no default WS
            # upstream to fall through to).
            WebSocketRoute("/{path:path}", ws_handle),
        ],
        lifespan=lifespan,
    )
    app.state.proxy = state
    return app
