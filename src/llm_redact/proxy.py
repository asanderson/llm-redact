"""Starlette app: the transparent proxy itself.

Design rules enforced here:
- Non-JSON or unrecognized traffic is forwarded verbatim — never break the
  agentic tool.
- Streaming vs JSON handling branches on the upstream *response*
  content-type, never the request's ``stream`` flag: an error reply to a
  streaming request arrives as plain JSON.
- Auth headers pass through untouched and are never logged; the log line
  contains only path, status, and detection counts.
"""

import asyncio
import dataclasses
import importlib.resources
import importlib.util
import inspect
import json
import logging
import os
import re
import secrets
import signal
import time
import urllib.parse
from collections import Counter, deque
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NamedTuple

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
from llm_redact.config import (
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
    resolve_config_path,
    resolve_credentials,
    unsupported_plugin_capabilities,
)
from llm_redact.detection.engine import (
    active_rule_names,
    build_allowlist,
    build_detectors,
    build_modes,
)
from llm_redact.eventstream import EventStreamError, EventStreamParser
from llm_redact.eventstream import serialize as serialize_eventstream
from llm_redact.jsonwalk import loads_request
from llm_redact.licensing import ResolvedLicense, resolve_license
from llm_redact.metrics import Metrics
from llm_redact.multipart import parse as parse_multipart
from llm_redact.multipart import parse_boundary as parse_multipart_boundary
from llm_redact.ndjson import NDJSONParser
from llm_redact.placeholders import PLACEHOLDER_RE, json_floors, may_carry_tokens
from llm_redact.plugin_api import (
    AccessGate,
    Admission,
    Dashboard,
    HopRequest,
    HopResult,
    LocalAnswer,
    RouteDelivery,
    RouteInbound,
    RoutePlan,
    Router,
    RouteRefusal,
    Telemetry,
    UpstreamAuth,
    UpstreamAuthError,
)
from llm_redact.providers import ALL_ADAPTERS, ProviderAdapter, RouteKind
from llm_redact.providers.custom import CUSTOM_ROUTE_PREFIX, build_custom_adapters, custom_prefix
from llm_redact.realtime import ALL_WS_ADAPTERS, WsAdapter, websockets_available, ws_handle
from llm_redact.redactor import (
    BlockedRequest,
    PlaceholderLimitReached,
    Redactor,
    UnredactableRequest,
)
from llm_redact.registry import get_registry, loaded_plugins, pro_package_installed
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent, SSEParser, serialize
from llm_redact.vault import Vault, VaultManager
from llm_redact.vault_crypto import key_source as vault_key_source
from llm_redact.vault_crypto import require_vault_key_source

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

# How often the [vault] session_ttl_days background task sweeps for idle
# sessions. Retention is a slow signal; hourly is ample and keeps the sqlite
# work negligible.
_TTL_PRUNE_INTERVAL_SECONDS = 3600.0
_LICENSE_REFRESH_INTERVAL_SECONDS = 86400.0

# The inbound request's W3C traceparent, captured at the top of handle() and
# read at finalization time so the OTel span (built then) can parent into the
# caller's trace even across the streaming boundary — same task, same context.
_INBOUND_TRACEPARENT: ContextVar[str | None] = ContextVar("llm_redact_traceparent", default=None)
# The subject the access gate (llm-redact-pro) admitted the current request
# as: set once in handle()/ws_handle() after admission, read at finalization
# by record_request — the same task-context trick as the traceparent, so the
# streaming finalizers attribute without threading a parameter through.
_REQUEST_USER: ContextVar[str | None] = ContextVar("llm_redact_user", default=None)
# The 403 text when the session router's ownership check fails or answers
# something other than a reason (never an id or a user name).
_OBJECT_ACCESS_FAULT = (
    "llm-redact: the stored-object ownership check failed; the request was not forwarded"
)

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
        self.detectors = build_detectors(config.detection)
        self.allowlist = build_allowlist(config.detection)
        self.modes = build_modes(config.detection)
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
        # Faults in what runs AFTER the provider answered, by stage:
        # "response_id" / "object_ids" / "listing" are session bookkeeping
        # (contained — the answer is still delivered, never a wrong value);
        # "delivery" is restoring the answer itself (a buffered one fails
        # closed with a recorded 502; a stream is cut).
        self.bookkeeping_errors: Counter[str] = Counter()
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
        self.client = httpx.AsyncClient(
            transport=upstream_transport, timeout=httpx.Timeout(600.0, connect=10.0)
        )
        self.metrics = Metrics(__version__)
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
        if config.audit.required:
            if not isinstance(self.audit, WriteAheadAudit):
                raise ConfigError(
                    "[audit] required = true needs a llm-redact-pro version with"
                    " write-ahead audit support (AuditLog.begin/finalize)"
                )
            self.write_ahead_audit = self.audit
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
        # Optional: the gate may drop a deleted user's sessions through the
        # live vault manager (plugin_api.SessionStore).
        bind_sessions = getattr(self.access_gate, "bind_sessions", None)
        if callable(bind_sessions):
            bind_sessions(_LiveSessions(self))
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
        self, adapter: ProviderAdapter | None, method: str, path: str, parsed_body: Any
    ) -> RequestContext:
        if self.session_router.mode == "static":
            return self._static_context
        session_id = self.session_router.resolve(
            adapter.name if adapter is not None else None, method, path, parsed_body
        )
        sealed = self._session_sealed(session_id)
        if session_id == self._static_context.session_id and sealed is None:
            return self._static_context
        vault = self.vault_manager.get(session_id)
        if session_id not in self._known_sessions:
            self._known_sessions.add(session_id)
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
            modes=self.modes,
            warn_counts=self.warn_counts,
        )
        rehydrator = Rehydrator(
            vault, fuzzy=self.config.rehydration.fuzzy, counts=self.rehydration_counts
        )
        return RequestContext(session_id, vault, redactor, rehydrator, sealed=sealed)

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
        name = self.provider_for(None, path, headers)
        for candidate in self.adapters:
            if candidate.name == name and candidate.tracks_object_ids(method, path, body=body):
                return candidate
        return None

    def record_object_ids(self, object_ids: Sequence[str], session_id: str) -> None:
        """Ids of provider-stored objects (files, batches, conversations)
        created or first seen in ``session_id`` — reported to a router that
        tracks ownership (the optional ``record_object_id``, in any mode),
        mirrored in the durable map unless the router vetoes it."""
        record = self._record_object_id
        if record is None:
            return
        for object_id in object_ids:
            if record(object_id, session_id) is not False:
                self.vault_manager.record_response_session(object_id, session_id)

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

    def object_lister(
        self,
        adapter: ProviderAdapter | None,
        method: str,
        path: str,
        headers: "Mapping[str, str] | None" = None,
    ) -> ProviderAdapter | None:
        """The adapter whose ``lists_objects`` claims this request (the
        routed one, else the addressed provider's), or None — always None
        when the session router attributes no listed items."""
        if self._listing_item_session is None and self._listing_item_sessions is None:
            return None
        if adapter is not None:
            return adapter if adapter.lists_objects(method, path) else None
        name = self.provider_for(None, path, headers)
        for candidate in self.adapters:
            if candidate.name == name and candidate.lists_objects(method, path):
                return candidate
        return None

    def listing_restorers(self, object_ids: Sequence[str]) -> dict[str, Rehydrator | None]:
        """How each listed object is delivered, for one listing: an id ABSENT
        from the answer stays as this request's own session delivers it (no
        session named, or the router failed); ``None`` — the item is
        delivered exactly as the provider sent it, because the named session
        does not exist or holds nothing (it is never created here), or is
        not the session the durable map records the object in (a session
        pruned since the object was created and recreated with new values:
        its placeholders are not the object's — or the check itself failed);
        a rehydrator — restored in the named session."""
        named = self._listing_sessions(object_ids)
        if not named:
            return {}
        recorded = self._recorded_sessions(named)
        rehydrators: dict[str, Rehydrator | None] = {}
        restorers: dict[str, Rehydrator | None] = {}
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

    def _listing_sessions(self, object_ids: Sequence[str]) -> dict[str, str]:
        """The session the router names for each listed object — asked once
        for the whole listing (the optional ``listing_item_sessions``), else
        per item (``listing_item_session``); ids it names none for (or that
        fail) are left out."""
        batched = self._listing_item_sessions
        if batched is not None:
            try:
                answer = list(batched(list(object_ids)))
            except Exception as exc:  # noqa: BLE001 — unsure means placeholders
                logger.warning(
                    "session router listing_item_sessions failed (%s); items left as is",
                    type(exc).__name__,
                )
                return {}
            if len(answer) != len(object_ids):
                logger.warning("session router listing_item_sessions miscounted; items left as is")
                return {}
            pairs = list(zip(object_ids, answer, strict=True))
        else:
            pairs = [
                (object_id, self._listing_item_session_of(object_id)) for object_id in object_ids
            ]
        return {object_id: session for object_id, session in pairs if isinstance(session, str)}

    def _listing_item_session_of(self, object_id: str) -> Any:
        """The router's per-item answer (only a string names a session)."""
        name_session = self._listing_item_session
        if name_session is None:
            return None
        try:
            return name_session(object_id)
        except Exception as exc:  # noqa: BLE001 — unsure means placeholders
            logger.warning(
                "session router listing_item_session failed (%s); item left as is",
                type(exc).__name__,
            )
            return None

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
        restart (kept with a warning): vault, audit, host, port.
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
        else:
            detectors = build_detectors(effective.detection)
            allowlist = build_allowlist(effective.detection)
            modes = build_modes(effective.detection)
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
        self.redactor = redactor
        self.rehydrator = rehydrator
        self._static_context = RequestContext(
            effective.vault.session, self.vault, redactor, rehydrator
        )
        if self.router is not None and router is not self.router:
            self.router.close()
        self.router = router
        self.dashboard = dashboard
        if upstream_auth is not self.upstream_auth:
            displaced = self.upstream_auth
            self.upstream_auth = upstream_auth
            _close_upstream_auths(displaced)
        for warning in effective.routing.warnings:
            logger.warning("routing: %s", warning)
        logger.info("config reloaded (%d detection rules)", len(detectors))
        return restart_required

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
            )
        )
        if token is None:
            # None is the "required mode off" sentinel record_request
            # dispatches on; a write-ahead log must never mint it. Fail
            # closed like any other begin-side fault rather than silently
            # orphaning the START row behind a record()-routed END.
            raise AuditWriteError("write-ahead audit begin() returned no token")
        return token

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
    ) -> None:
        """Always update in-memory metrics and the recent buffer; write an
        audit row when enabled (finalizing the write-ahead START row when
        ``begin_audit`` issued a token for this request)."""
        duration_seconds = time.perf_counter() - started
        self.metrics.observe_request(provider, status, duration_seconds, streamed)
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
        )
        try:
            # A token exists only when begin_audit resolved the write-ahead
            # log (and it never mints None) — the conjunct narrows the type.
            if audit_token is not None and self.write_ahead_audit is not None:
                self.write_ahead_audit.finalize(audit_token, entry)
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

    def finish_route(self, delivery: RouteDelivery, status: int | None) -> dict[str, Any]:
        """Close the books on a routed request: the routed-requests metric
        (every routed outcome, 502s included — decision 19) and the router's
        own finish (spend), returning the `route` row record_request stores."""
        self.metrics.routed[(delivery.upstream, delivery.rule or "-")] += 1
        return delivery.finish(status)

    def route(
        self, method: str, path: str, headers: "Mapping[str, str] | None" = None
    ) -> tuple[ProviderAdapter | None, RouteKind]:
        for adapter in self.adapters:
            kind = adapter.matches_request(method, path, headers)
            if kind is not RouteKind.NONE:
                return adapter, kind
        return None, RouteKind.NONE

    def provider_for(
        self,
        adapter: ProviderAdapter | None,
        path: str,
        headers: "Mapping[str, str] | None" = None,
        query: str = "",
    ) -> str:
        if adapter is not None and adapter.name in self.config.providers:
            return adapter.name
        # Pass-through traffic: infer the provider from well-known paths;
        # Anthropic is the default because that is the primary target tool.
        # An explicit path family wins over every header or key heuristic.
        if path.startswith("/v1beta/"):
            return "gemini"
        if path.startswith(("/v1/projects/", "/v1/publishers/", "/v1beta1/")):
            # Vertex AI (express mode and service-account API keys send a
            # Google API key too — still Vertex's traffic, never the
            # Gemini API's host).
            return "vertex"
        if path.startswith("/v1/") and _google_authenticated(headers, query):
            # Gemini's v1 surface (GET /v1/models[/{m}], …) shares OpenAI's
            # /v1 prefix; a Google API key (x-goog-api-key or ?key=) marks
            # whose traffic this is — inferring openai here would send the
            # Google key to api.openai.com.
            return "gemini"
        if (
            headers is not None
            and "anthropic-version" in headers
            and path.startswith(("/v1/files", "/v1/batches"))
        ):
            # Anthropic's beta Files API shares OpenAI's paths; the header
            # marks whose traffic this is so pass-through reaches the
            # right upstream (their uploads are documents — media
            # non-goal — but misrouting them would break the tool).
            return "anthropic"
        if path.startswith(CUSTOM_ROUTE_PREFIX):
            # Pass-through under a custom prefix is still addressed to that
            # upstream; an unknown name yields a key with no config entry,
            # which handle() answers 502 (never forwarded by guesswork).
            name = path[len(CUSTOM_ROUTE_PREFIX) :].split("/", 1)[0]
            return f"custom:{name}"
        if path.startswith("/openai/"):
            return "azure"
        if path.startswith(("/model/", "/guardrail/", "/async-invoke")):
            # /guardrail (ApplyGuardrail) and /async-invoke (StartAsyncInvoke)
            # are bedrock-runtime paths outside the /model/ regex; without
            # these prefixes they misrouted to the anthropic default.
            return "bedrock"
        if path.startswith("/upload/v1beta/"):
            # Gemini's resumable/multipart Files upload starts with /upload/,
            # not /v1beta/ — any Gemini tool uploading a file through the
            # proxy hit the wrong host before this prefix existed.
            return "gemini"
        if path.startswith("/api/"):
            return "ollama"
        if path.startswith(
            (
                "/v1/chat",
                "/v1/completions",
                "/v1/embeddings",
                "/v1/models",
                "/v1/responses",
                # /v1/uploads is the multipart Uploads API sibling of files/
                # batches: its CONTENT is a documented non-goal (cross-request
                # part protocol), but the traffic must still reach OpenAI —
                # omitting it here sent Uploads to the anthropic default.
                "/v1/files",
                "/v1/uploads",
                "/v1/batches",
                # Media endpoints: matched routes cover the text-bearing
                # POSTs; the rest (variations, job GETs) must still reach
                # the OpenAI upstream rather than the anthropic default.
                "/v1/images",
                "/v1/audio",
                "/v1/videos",
                # Deliberately-forwarded OpenAI surfaces that still must
                # reach the OpenAI upstream (each 404'd against the
                # anthropic default before these prefixes existed):
                # moderations/fine-tuning are documented pass-throughs,
                # /v1/realtime covers the HTTP side (client_secrets etc.),
                # vector stores back the Responses file_search tool, and
                # assistants/threads remain callable until their sunset.
                "/v1/moderations",
                "/v1/fine_tuning",
                "/v1/realtime",
                "/v1/vector_stores",
                "/v1/assistants",
                "/v1/threads",
            )
        ):
            return "openai"
        return "anthropic"


def _google_authenticated(headers: "Mapping[str, str] | None", query: str) -> bool:
    """Whether a request carries a Google API key: the x-goog-api-key
    header or a ``key``/``$key`` query parameter (no other provider the
    proxy infers uses either)."""
    if headers is not None and "x-goog-api-key" in headers:
        return True
    return any(
        urllib.parse.unquote_plus(part.split("=", 1)[0]).lower().lstrip("$") == "key"
        for part in query.split("&")
        if part
    )


def _request_headers(request: Request) -> list[tuple[str, str]]:
    headers = [
        (name, value)
        for name, value in request.headers.items()
        if name.lower() not in _SKIP_REQUEST_HEADERS
        and not name.lower().startswith(OWN_HEADER_PREFIX)
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
) -> AsyncIterator[bytes]:
    method, path, started, detections, warned, audit_token = request_meta
    parser = SSEParser()
    pool = RehydratorPool(ctx.vault, fuzzy=state.config.rehydration.fuzzy)
    response_id_seen = False
    status = upstream.status_code  # 502 when the proxy itself cut the stream
    try:
        async for chunk in upstream.aiter_bytes():
            for event in parser.feed(chunk):
                if not response_id_seen:
                    response_id = adapter.response_id_from_event(event)
                    if response_id is not None:
                        # Session bookkeeping never cuts the stream (contained,
                        # counted); the first id suffices either way.
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
                if object_tracker is not None and _record_streamed_object_ids(
                    state, ctx, object_tracker, method, path, event
                ):
                    object_tracker = None  # the first event naming it suffices
                for out in adapter.rehydrate_event(event, pool):
                    if route is not None:
                        out = route.observe_event(out)
                    yield serialize(out)
        for event in parser.close():
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
            route=(state.finish_route(route, status) if route is not None else None),
        )


def _record_streamed_object_ids(
    state: ProxyState,
    ctx: RequestContext,
    tracker: ProviderAdapter,
    method: str,
    path: str,
    event: SSEEvent,
) -> bool:
    """Report the stored-object ids a streamed event names (a stored chat
    completion's chunks carry its id); True once some were reported."""
    try:
        payload = json.loads(event.data)
    except ValueError:
        return False  # [DONE], keep-alives, anything not JSON
    object_ids = tracker.object_ids_from_body(method, path, payload)
    if object_ids:
        # Contained like the buffered report: the stream goes on either way.
        _contained(
            state, "object_ids", method, path, state.record_object_ids, object_ids, ctx.session_id
        )
    return bool(object_ids)


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
    try:
        async for chunk in upstream.aiter_bytes():
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
        )


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


async def _handle_local(request: Request, state: ProxyState) -> Response:
    """Answer reserved /__llm-redact endpoints locally. Metadata only —
    never values; allowlists reported as counts. The dashboard paths are
    delegated to the llm-redact-pro Dashboard (its config editor is the one
    exception on both fronts: it accepts POST behind the guard chain and
    returns allowlist values)."""
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
                "upstream_errors_total": dict(state.upstream_errors),
                "bookkeeping_errors_total": dict(state.bookkeeping_errors),
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
                # The access gate's own block (llm-redact-pro); without one,
                # no registry and nothing enforced.
                "users": (
                    state.access_gate.status()
                    if state.access_gate is not None
                    else {"registry": False, "enforcement": False}
                ),
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
            ),
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
        state.event_subscribers.add(queue)

        async def event_stream() -> AsyncIterator[bytes]:
            try:
                yield b": connected\n\n"
                while True:
                    try:
                        row = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except TimeoutError:
                        yield b": keepalive\n\n"
                        continue
                    yield b"data: " + json.dumps(row, ensure_ascii=False).encode() + b"\n\n"
            finally:
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


def _allowed_hostnames(state: ProxyState) -> set[str]:
    names = {"127.0.0.1", "localhost", "::1", state.config.host.lower()}
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


async def _admit_reserved(request: Request, state: ProxyState) -> Response | None:
    """Dashboard admission for the reserved endpoints (only when the gate
    opts in with ``guards_dashboard``): every reserved path except the
    monitoring probes and the gate's own paths (sign-in and SCIM
    authenticate themselves). None admits; otherwise the refusal —
    a 303 to the gate's same-proxy sign-in page for a browser GET, else a
    403. Nothing here is forwarded."""
    if not state.guards_dashboard:
        return None
    path = request.url.path
    if (
        path in PROBE_PATHS
        or _is_auth_path(path)
        or path == SCIM_PREFIX
        or path.startswith(SCIM_PREFIX + "/")
    ):
        return None
    admission = await state.admit(request, "dashboard")
    if admission.refusal is None:
        return None
    redirect = admission.redirect
    if (
        request.method == "GET"
        and redirect is not None
        and redirect.startswith(RESERVED_PREFIX + "/")
        and "\\" not in redirect
        and "//" not in redirect
    ):
        return RedirectResponse(redirect, status_code=303)
    return JSONResponse({"error": admission.refusal}, status_code=403)


def _host_allowed(request: Request, state: ProxyState) -> bool:
    """DNS-rebinding defense: a rebinding page's requests carry the
    attacker's domain in Host, while local browsers and tools send the
    loopback name they connected to."""
    hostname = request.url.hostname
    return hostname is not None and hostname.lower() in _allowed_hostnames(state)


def _origin_allowed(request: Request, state: ProxyState) -> bool:
    """Absent Origin (curl, same-origin GET) is fine — the CSRF token still
    gates POST. A present Origin must be a local origin ('null' and
    everything else is rejected); https origins exist only when the proxy
    itself serves TLS."""
    origin = request.headers.get("origin")
    if origin is None:
        return True
    if state.public_origin is not None and origin.lower() == state.public_origin[2]:
        return True
    parsed = urllib.parse.urlsplit(origin)
    schemes = ("http", "https") if state.config.tls.enabled else ("http",)
    return parsed.scheme in schemes and (parsed.hostname or "").lower() in _allowed_hostnames(state)


async def _stream_rehydrated_ndjson(
    upstream: httpx.Response,
    adapter: ProviderAdapter,
    state: ProxyState,
    ctx: RequestContext,
    *,
    request_meta: RequestMeta,
    route: RouteDelivery | None = None,
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
    try:
        async for chunk in upstream.aiter_bytes():
            for line in parser.feed(chunk):
                out = adapter.rehydrate_ndjson_line(line, pool)
                if route is not None:
                    out = route.observe_line(out)
                yield out + b"\n"
        tail = parser.close()
        if tail:
            # A stream that ended without a final newline: the tail may
            # still be one complete JSON object.
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
        return json.loads(raw_body), None
    except json.JSONDecodeError as exc:
        return None, JSONResponse({"error": f"invalid JSON: {exc}"}, status_code=400)


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


def _identity_body_problem(
    adapter: ProviderAdapter, path: str, headers: Headers, body: bytes, parsed: Any
) -> str | None:
    """Why a non-empty request body on a matched route must NOT be signed
    with the proxy's own identity, or None when the proxy redacts it.

    Signable: a JSON object (what ``prepare_request`` walks; JSON is read
    from the bytes whatever the content-type, a UTF-8 BOM or UTF-16/32
    encoding included), or canonical multipart on a route whose
    ``redact_multipart`` scans it. Everything else would be forwarded
    verbatim: non-JSON bytes, invalid UTF-8, a top-level array or scalar
    (``null`` included — never walked), whitespace only, multipart on any
    other route, and any content-encoded body (the proxy never decodes
    one, so it cannot see what the upstream would) — every coding of every
    Content-Encoding header counts, as the upstream reads them all. A
    repeated Content-Type is refused too: it is a singleton field, and a
    second one could name a multipart boundary the proxy never parsed
    with. The route is checked before the multipart parse, so a body
    refused on its route is never parsed. The result names the body's
    KIND only — never its content."""
    codings = (
        coding.strip().lower()
        for value in headers.getlist("content-encoding")
        for coding in value.split(",")
    )
    if any(coding not in ("", "identity") for coding in codings):
        return "the request body is content-encoded (llm-redact does not decode request bodies)"
    content_types = headers.getlist("content-type")
    if len(content_types) > 1:
        return "the request carries more than one Content-Type header"
    if isinstance(parsed, dict):
        return None
    boundary = (
        parse_multipart_boundary(content_types[0]) if parsed is None and content_types else None
    )
    if boundary is None:
        return "the request body is not a JSON object llm-redact can redact"
    if not adapter.redacts_multipart(path):
        return "the multipart body is on a route llm-redact does not redact multipart for"
    if parse_multipart(body, boundary) is None:
        return "the multipart body is outside the canonical form llm-redact can redact"
    return None


def _lends_credential(plan: RoutePlan) -> bool:
    """Whether a routed request may reach its provider with a credential the
    PROXY holds (``RoutePlan.proxy_credential``, optional): only an explicit
    False — every upstream the plan may use forwards the client's own
    credential — says no; a plan without the member counts as the proxy's
    (fail closed: the stored-object check is then only stricter)."""
    return getattr(plan, "proxy_credential", True) is not False


class _Unreadable(NamedTuple):
    """Why a pass-through body cannot be checked (status, message — the
    body's KIND only, never its content)."""

    status: int
    message: str


# The bytes a JSON document may start with before its first value: ASCII
# whitespace, the UTF-8/16/32 byte-order marks, and the NULs of UTF-16/32.
_JSON_LEAD = re.compile(rb"[^\x00\t\n\r \xef\xbb\xbf\xfe\xff]")


def _pass_through_check_body(
    headers: Headers, body: bytes, max_body_bytes: int
) -> tuple[Any, _Unreadable | None]:
    """A routed pass-through body, parsed for the stored-object check only
    (the bytes are forwarded as they are): ``(parsed JSON or None, None)``,
    or ``(None, why)`` when the body could cite an object the check cannot
    see — content-encoded (the proxy never decodes one; every
    Content-Encoding header counts), a JSON document over
    ``max_body_bytes``, or one that repeats a key (the check would see the
    last occurrence, an upstream may keep the first). A body that is not
    JSON at all (a multipart upload, audio) has nothing the check reads:
    None."""
    for value in headers.getlist("content-encoding"):
        if any(c.strip().lower() not in ("", "identity") for c in value.split(",")):
            return None, _Unreadable(
                400,
                "the request body is content-encoded (llm-redact does not decode request bodies)",
            )
    if len(body) > max_body_bytes:
        lead = _JSON_LEAD.search(body)
        if lead is not None and body[lead.start()] in b"{[":
            return None, _Unreadable(
                413, f"the request body exceeds llm-redact max_body_bytes ({max_body_bytes})"
            )
        return None, None
    try:
        parsed, duplicate_keys = loads_request(body)
    except ValueError:
        return None, None
    if duplicate_keys:
        return None, _Unreadable(400, "the request body repeats a JSON key")
    return parsed, None


async def handle(request: Request) -> Response:
    state: ProxyState = request.app.state.proxy
    if not origin_form_target(request.scope):
        # Never routed, forwarded, recorded or logged with its target.
        return JSONResponse({"error": "the request target must be a path"}, status_code=400)
    if has_dot_segment(request.scope):
        # Refused before routing, admission or any upstream contact (and,
        # like the target check above, never recorded or logged: the path
        # may still hold an identity-prefix key).
        return JSONResponse(
            {"error": "the request path must not contain '.' or '..' segments"}, status_code=400
        )
    path = request.url.path

    # Reserved local endpoints are answered here, before any routing or
    # upstream code runs — this early return is the non-forwarding guarantee.
    if path.startswith(RESERVED_PREFIX):
        response = await _admit_reserved(request, state)
        if response is None:
            response = await _handle_local(request, state)
        # Stamp browser-hardening headers on every reserved reply in one place
        # (setdefault so a handler that set its own header still wins).
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    # Client admission (llm-redact-pro's access gate): it removes every
    # credential it recognizes from the scope before anything else reads
    # the path or headers. Its refusal, if any, is applied further down.
    admission = await state.admit(request, "http")
    path = request.scope["path"]
    if path.startswith(RESERVED_PREFIX):
        # Only reachable through a stripped prefix (/u/<key>/__llm-redact/…):
        # reserved endpoints are served at their own path, never forwarded.
        return JSONResponse({"error": "not found"}, status_code=404)
    if path.startswith(IDENTITY_PATH_PREFIX):
        # An identity prefix no gate claimed: its next segment is a key, so
        # the request is neither forwarded nor recorded (both would carry it).
        logger.info("%s /u/... -> 404 unclaimed identity path prefix", request.method)
        return JSONResponse(
            {"error": "this proxy does not accept /u/<key>/ identity paths (no access gate)"},
            status_code=404,
        )
    _REQUEST_USER.set(admission.subject)

    # Captured once per request; read at finalization (incl. the streaming
    # finalizer, same task context) so an OTel span can parent into the
    # caller's trace. Trivial when telemetry is off; traceparent isn't secret.
    _INBOUND_TRACEPARENT.set(request.headers.get("traceparent"))
    started = time.perf_counter()
    adapter, kind = state.route(request.method, path, request.headers)

    # A disabled provider fails closed before anything is read or forwarded:
    # matched routes AND pass-through traffic inferred to it are answered
    # here (forwarding pass-through would send unredacted bodies to it).
    provider_name = state.provider_for(adapter, path, request.headers, request.url.query)
    provider_conf = state.config.providers.get(provider_name)
    # Read ONCE, with the provider's config and before the first await (the
    # body read): a reload applied while the body is still arriving must not
    # change what this request was admitted as. Every identity decision
    # below — the unrecognized-route 403, the ownership check, the body
    # rule, stripping and signing — and the upstream it goes to use these
    # two snapshots; the reload's displaced authorizer stays with the
    # requests that already hold it.
    upstream_auth = state.upstream_auth.get(provider_name)
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
        )
        logger.info("%s %s -> 502 provider %s disabled", request.method, path, provider_name)
        return JSONResponse(error, status_code=502)

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
        )
        logger.info("%s %s -> 403 refused by the access gate", request.method, path)
        return JSONResponse(error, status_code=403)

    if adapter is None and upstream_auth is not None:
        # The proxy lends its own cloud identity only to the API routes it
        # recognizes (and redacts). Signing pass-through traffic would hand
        # any client the whole cloud API as the proxy's principal, bodies
        # unredacted — so an unrecognized path is refused here, never sent.
        message = (
            f"llm-redact: {provider_name} is authorized with the proxy's own identity"
            f' ([providers.{provider_name}] auth = "identity"); only the API routes'
            " llm-redact recognizes are forwarded"
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
        )
        logger.info("%s %s -> 403 unrecognized route for identity auth", request.method, path)
        return JSONResponse({"error": message}, status_code=403)

    if state.router is not None:
        answer = state.router.local_answer(request.method, path, request.headers)
        if answer is not None:
            # R-15 model discovery: a local answer (never forwarded), placed
            # where every other proxy-generated reply for a real API path sits —
            # AFTER the disabled-provider 502 and the named-user 403, so an
            # unauthenticated client on a team deployment learns nothing the
            # gates would refuse. It needs nothing from the body.
            return _answer_locally(state, request, answer, path=path, started=started)

    if adapter is not None:
        # Redactable routes fail closed on oversized bodies: the proxy must
        # buffer the whole body to redact it, and forwarding unredacted is
        # never acceptable. Pass-through routes below are unaffected.
        capped = await _read_capped(request, state.config.max_body_bytes)
        if capped is None:
            logger.info(
                "%s %s -> 413 body over %d bytes", request.method, path, state.config.max_body_bytes
            )
            state.record_request(
                session=state.config.vault.session,
                provider=adapter.name,
                method=request.method,
                path=path,
                status=413,
                started=started,
                streamed=False,
                detections={},
                rehydrations={},
            )
            return Response(
                content=json.dumps(
                    adapter.error_body(
                        f"request body exceeds llm-redact max_body_bytes"
                        f" ({state.config.max_body_bytes})"
                    )
                ).encode("utf-8"),
                status_code=413,
                media_type="application/json",
            )
        body_bytes = capped
    else:
        body_bytes = await request.body()

    parsed: Any = None
    # A repeated JSON key: the parse keeps the last occurrence, so the walk
    # never sees the earlier ones — such a body is always re-serialized.
    duplicate_keys = False
    if adapter is not None and body_bytes:
        try:
            parsed, duplicate_keys = loads_request(body_bytes)
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
    plan: RoutePlan | None = None
    # A provider the proxy authorizes with its own cloud identity is never
    # routed (the routing protocols are anthropic/openai/gemini/ollama, and
    # a routed hop carries the router's own credentials): the router is not
    # even asked, so the request below is always signed by its authorizer.
    if state.router is not None and upstream_auth is None:
        model = parsed.get("model") if isinstance(parsed, dict) else None
        planned = state.router.plan(
            RouteInbound(
                adapter_name=adapter.name if adapter is not None else None,
                provider_name=provider_name,
                method=request.method,
                path=path,
                raw_path=_upstream_path(request, path),
                query=request.url.query,
                headers=request.headers,
                model=model if isinstance(model, str) else None,
            )
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
            )
        plan = planned

    # Whose credential the provider will see. The proxy's OWN — its cloud
    # identity, or a routed plan's operator key (RoutePlan.proxy_credential;
    # a plan without the member counts as the proxy's: fail closed) — makes
    # every client the same principal upstream, so the stored-object check
    # applies its identity policy.
    proxy_credential = upstream_auth is not None or (plan is not None and _lends_credential(plan))
    check_body: Any = parsed
    if adapter is None and proxy_credential and body_bytes and state.checks_object_access:
        # A routed pass-through request spent with the proxy's credential:
        # its body may cite a stored object (a vector store's file_ids, a
        # fine-tuning job's training_file), so it is parsed for the check
        # alone — the bytes are still forwarded verbatim. A body the check
        # cannot read is never sent with the proxy's credential.
        check_body, unreadable = _pass_through_check_body(
            request.headers, body_bytes, state.config.max_body_bytes
        )
        if unreadable is not None:
            return _unchecked_body_refused(
                state,
                unreadable,
                provider_name=provider_name,
                request=request,
                path=path,
                started=started,
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

    # Session resolution hashes the raw (pre-redaction) conversation anchor,
    # so it must happen before prepare_request.
    ctx = state.context_for(adapter, request.method, path, parsed)

    detection_counts_before = dict(state.detection_counts)
    warn_counts_before = dict(state.warn_counts)

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
        )
        return JSONResponse(
            blocked_adapter.error_body(
                f"llm-redact: request blocked; a {exc.detector_type} value was"
                ' detected and this rule is configured with mode = "block"',
                status=400,
            ),
            status_code=400,
        )

    def refused_response(message: str, refused_adapter: ProviderAdapter, why: str) -> JSONResponse:
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
        )
        reason = ctx.sealed or _SEALED_REFUSAL
        return JSONResponse(sealed_adapter.error_body(reason, status=403), status_code=403)

    note_wanted = plan.inject_system_note if plan is not None else state.config.inject_system_note

    outbound = body_bytes
    # The decoded form of `outbound` (None for pass-through / non-JSON
    # bodies): the routed path applies per-hop body rewrites to it.
    outbound_obj: dict[str, Any] | None = parsed if isinstance(parsed, dict) else None
    detection_off = provider_conf is not None and not provider_conf.detection
    if upstream_auth is not None and adapter is not None and body_bytes:
        # The proxy's own identity signs only a body the proxy could read:
        # one it cannot walk would otherwise be forwarded verbatim —
        # refused here, before redaction, any credential fetch or upstream
        # contact. detection = false turns REDACTION off, not this rule:
        # the ownership check (object_access_refusal, above) and object
        # tracking read the parsed body too, and a body they could not
        # read (gzip, non-JSON bytes) must not carry the proxy's identity.
        problem = _identity_body_problem(adapter, path, request.headers, body_bytes, parsed)
        if problem is not None:
            return refused_response(
                f"llm-redact: {problem}, and this provider is authorized with the proxy's"
                " own identity; the request was not forwarded",
                adapter,
                f"{problem} under identity auth",
            )
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
            outbound = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
    elif adapter is not None and isinstance(parsed, dict):
        # Token floors: a new value is never numbered onto a token this body
        # already carries (one its session never issued — a compacted
        # history, a pasted answer — would otherwise gain a second meaning).
        # The whole decoded body counts, fields the walk skips included; a
        # body with no guillemet in any encoding pays only the byte gate.
        redactor = (
            ctx.redactor.with_floors(json_floors(parsed))
            if may_carry_tokens(body_bytes)
            else ctx.redactor
        )
        try:
            prepared = adapter.prepare_request(
                parsed,
                redactor,
                inject_note=note_wanted and adapter.wants_system_note(kind, path),
                mcp_exempt=frozenset(state.config.detection.mcp_exempt_servers),
            )
        except BlockedRequest as exc:
            return blocked_response(exc, adapter)
        except PlaceholderLimitReached as exc:
            return refused_response(str(exc), adapter, "no placeholder number left")
        except UnredactableRequest as exc:
            return refused_response(str(exc), adapter, "undecodable field")
        except SealedSessionError:
            return sealed_response(adapter)
        outbound_obj = prepared
        # No-op short-circuit: redaction increments detection_counts, and note
        # injection is gated on a redaction actually happening (base
        # prepare_request), so an unchanged count means the prepared body is
        # byte-for-byte the original. Forward the raw bytes and skip the
        # parse→dump round-trip — the common nothing-to-redact large-body case.
        # Never with a repeated key: the raw bytes still hold the earlier
        # occurrences the walk never saw (an upstream may keep the first).
        if duplicate_keys or sum(state.detection_counts.values()) != sum(
            detection_counts_before.values()
        ):
            outbound = json.dumps(prepared, ensure_ascii=False).encode("utf-8")
    elif adapter is not None and parsed is None and body_bytes:
        # Matched routes with non-JSON bodies: multipart uploads (OpenAI
        # /v1/files) get their JSONL file parts redacted; anything the
        # adapter declines to rewrite forwards verbatim (the non-JSON-body
        # default that keeps unknown formats working) — except under
        # identity auth: refused above (_identity_body_problem), and any
        # part the adapter would forward unscanned refused here.
        boundary = parse_multipart_boundary(request.headers.get("content-type", ""))
        if boundary is not None:
            # The requests an uploaded file's lines carry (a batch input
            # file) are run by the provider later, with the credential this
            # upload is sent with: collected for the stored-object check.
            lines: list[Any] | None = [] if state.checks_object_access else None
            if (
                lines is not None
                and proxy_credential
                and adapter.redacts_multipart(path)
                and parse_multipart(body_bytes, boundary) is None
            ):
                # An upload the adapter would forward unread (outside the
                # canonical grammar) could carry lines citing any stored
                # object: never sent with the proxy's own credential.
                return refused_response(
                    "llm-redact: the multipart body is outside the canonical form llm-redact"
                    " can check, and this request would be sent with the proxy's own provider"
                    " credential; the request was not forwarded",
                    adapter,
                    "unchecked multipart under the proxy's credential",
                )
            try:
                rewritten = adapter.redact_multipart(
                    path,
                    body_bytes,
                    boundary,
                    ctx.redactor,
                    inject_note=note_wanted and adapter.wants_system_note(kind, path),
                    # Under the proxy's own identity every part must be
                    # scanned: an unscanned piece refuses the whole request.
                    require_scanned=upstream_auth is not None,
                    cited=lines,
                )
            except BlockedRequest as exc:
                # One leaking line in an uploaded file is a leak: the
                # whole request is rejected.
                return blocked_response(exc, adapter)
            except PlaceholderLimitReached as exc:
                return refused_response(str(exc), adapter, "no placeholder number left")
            except UnredactableRequest as exc:
                return refused_response(
                    f"llm-redact: {exc}, and this provider is authorized with the proxy's"
                    " own identity; the request was not forwarded",
                    adapter,
                    "unscanned multipart content under identity auth",
                )
            except SealedSessionError:
                return sealed_response(adapter)
            refusal = (
                state.object_access_refusal(
                    adapter.name, request.method, path, lines, identity=proxy_credential
                )
                if lines
                else None
            )
            if refusal is not None:
                # A line citing another namespace's stored object: refused
                # like the same citation in a JSON body — still before any
                # upstream contact (what was redacted stays in this
                # request's own session).
                return _object_access_refused(
                    state,
                    adapter,
                    refusal,
                    provider_name=provider_name,
                    request=request,
                    path=path,
                    started=started,
                )
            if rewritten is not None:
                outbound = rewritten

    new_counts = _count_delta(state.detection_counts, detection_counts_before)
    # Same diff trick for warn-mode hits: attribute forwarded-unredacted
    # values to THIS request, not just the process-lifetime aggregate.
    new_warned = _count_delta(state.warn_counts, warn_counts_before)

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
        )

    upstream_base = provider_conf.upstream_base_url  # the admitted config's, not a reload's
    if not upstream_base:
        # Providers without a default upstream (azure) answer 502 until
        # configured — proxy-generated, never forwarded.
        provider_name = adapter.name if adapter is not None else "unknown"
        error = (
            adapter.error_body(
                f"configure [providers.{provider_name}] upstream_base_url", status=502
            )
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
        )
        logger.info("%s %s -> 502 upstream not configured", request.method, path)
        return JSONResponse(error, status_code=502)
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
        return JSONResponse({"error": "the request target must be a path"}, status_code=400)
    headers = _request_headers(request)
    if upstream_auth is not None:
        # The proxy's own cloud identity: strip every client credential, then
        # authorize the FINAL request — the on-the-wire URL and the bytes
        # below, after redaction and note injection — and send exactly that.
        url, headers = strip_client_credentials(url, headers)
        if not _same_upstream(
            url, upstream_base, exact_path=upstream_base_path(upstream_base) + upstream_path
        ):
            # The URL the authorizer would sign must address exactly the
            # path the route was matched on (httpx normalizes before send).
            return JSONResponse({"error": "the request target must be a path"}, status_code=400)
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

    try:
        upstream = await state.client.send(upstream_request, stream=True)
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
    )


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
    known), before any byte leaves for the provider. A None token means
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
        # The audit-storage twin of the upstream-fault 502: provider-shaped
        # 503, recorded to metrics/recent (the audit write for this row will
        # itself fail — record_request logs that loudly). Type only.
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
        )
        message = "llm-redact: audit log unavailable and [audit] required is enabled"
        body = (
            adapter.error_body(message, status=503) if adapter is not None else {"error": message}
        )
        return None, JSONResponse(body, status_code=503)
    return token, None


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
) -> Response:
    """Hand an upstream response to the client: the streaming branches
    (chosen by the upstream RESPONSE content-type, never the request's
    stream flag) rehydrate as they go and finalize at stream end; anything
    else is buffered. With `route` (a routed request) the same branches
    also run the router's delivery hooks (a rewritten model id restored,
    usage tracked for its budget ledger) and stamp the x-llm-redact-*
    headers."""
    content_type = upstream.headers.get("content-type", "")
    headers = _response_headers(upstream)
    if route is not None:
        headers.update(route.headers)
    request_meta = RequestMeta(request.method, path, started, new_counts, new_warned, audit_token)

    if kind is RouteKind.CHAT and adapter is not None and "text/event-stream" in content_type:
        return StreamingResponse(
            _stream_rehydrated(
                upstream,
                adapter,
                state,
                ctx,
                request_meta=request_meta,
                route=route,
                object_tracker=(
                    state.object_tracker(
                        adapter, request.method, path, request.headers, body=request_body
                    )
                    if 200 <= upstream.status_code < 300
                    else None
                ),
            ),
            status_code=upstream.status_code,
            headers=headers,
            media_type="text/event-stream",
        )

    if (
        kind is RouteKind.CHAT
        and adapter is not None
        and adapter.handles_eventstream
        and "application/vnd.amazon.eventstream" in content_type
    ):
        # Bedrock only — never a routed protocol, so no route wrapper.
        return StreamingResponse(
            _stream_rehydrated_eventstream(
                upstream, adapter, state, ctx, request_meta=request_meta
            ),
            status_code=upstream.status_code,
            headers=headers,
            media_type="application/vnd.amazon.eventstream",
        )

    if (
        kind is RouteKind.CHAT
        and adapter is not None
        and adapter.handles_ndjson
        and any(t in content_type for t in _JSONL_CONTENT_TYPES)
    ):
        return StreamingResponse(
            _stream_rehydrated_ndjson(
                upstream, adapter, state, ctx, request_meta=request_meta, route=route
            ),
            status_code=upstream.status_code,
            headers=headers,
            media_type="application/x-ndjson",
        )

    try:
        raw = await upstream.aread()
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

    rehydration_counts_before = dict(state.rehydration_counts)
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
        rehydrations=_count_delta(state.rehydration_counts, rehydration_counts_before),
        warned=new_warned,
        audit_token=audit_token,
        route=(state.finish_route(route, upstream.status_code) if route is not None else None),
    )

    return Response(content=raw, status_code=upstream.status_code, headers=headers)


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
    if kind is RouteKind.CHAT and adapter is not None and "application/json" in content_type:
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
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
                raw = json.dumps(rehydrated, ensure_ascii=False).encode("utf-8")
    elif kind is RouteKind.CHAT and adapter is not None and raw:
        # Non-JSON buffered CHAT responses: file downloads whose contents
        # can carry placeholders (OpenAI batch output JSONL). The adapter
        # decides; None leaves the bytes untouched.
        raw_rehydrated = adapter.rehydrate_raw_body(path, raw, ctx.rehydrator)
        if raw_rehydrated is not None:
            raw = raw_rehydrated
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
            payload = json.loads(raw)
        except ValueError:
            payload = None
        if payload is not None and route.observe_payload(payload, kind):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    lister = (
        state.object_lister(adapter, request.method, path, request.headers)
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
        state.object_tracker(adapter, request.method, path, request.headers, body=request_body)
        if raw and 200 <= status < 300 and "application/json" in content_type
        else None
    )
    if tracker is not None:
        # Objects the provider stores for later reads (uploaded files,
        # batches, stored conversations): their ids go to the session router
        # with the session that created them. A body no adapter tracks is
        # never parsed here (pass-through routes carry no adapter, so the
        # provider's is looked up by name).
        try:
            stored = payload if payload is not None else json.loads(raw)
        except ValueError:
            stored = None
        object_ids = tracker.object_ids_from_body(request.method, path, stored)
        if object_ids:
            _contained(
                state,
                "object_ids",
                request.method,
                path,
                state.record_object_ids,
                object_ids,
                ctx.session_id,
            )
    return raw


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
        state.bookkeeping_errors[stage] += 1
        logger.warning(
            "%s %s -> %s bookkeeping failed (%s); the answer is delivered",
            method,
            path,
            stage,
            type(exc).__name__,
        )
        return False
    return True


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
    )
    logger.info("%s %s -> 403 refused by the session router (stored object)", request.method, path)
    return JSONResponse(error, status_code=403)


def _unchecked_body_refused(
    state: ProxyState,
    unreadable: _Unreadable,
    *,
    provider_name: str,
    request: Request,
    path: str,
    started: float,
) -> JSONResponse:
    """A routed pass-through request the proxy would send with its own
    credential, whose body the stored-object check cannot read: refused,
    recorded, before any upstream contact (the pass-through error shape —
    no adapter matched)."""
    message = (
        f"llm-redact: {unreadable.message}, and this request would be sent with the"
        " proxy's own provider credential, so the stored objects it cites must be"
        " checked; the request was not forwarded"
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
    )
    logger.info(
        "%s %s -> %d refused (unchecked body under the proxy's credential)",
        request.method,
        path,
        unreadable.status,
    )
    return JSONResponse({"error": message}, status_code=unreadable.status)


def _restore_listing(
    state: ProxyState, lister: ProviderAdapter, upstream_raw: bytes, raw: bytes
) -> bytes | None:
    """Restore each listed stored object in the session the router names
    for it (``listing_item_session``); every other item, and everything
    outside the item array, stays exactly as ``raw`` (the bytes about to be
    delivered) has it. A named item is rebuilt from the provider's own
    bytes (``upstream_raw``) — rehydrated as a whole object with the
    adapter's non-streaming transform, never a second pass over an
    already-restored item; a named session that does not exist (or holds
    nothing) restores nothing, so that item is delivered exactly as the
    provider sent it. None when nothing changed (the bytes are then
    forwarded untouched)."""
    try:
        original = json.loads(upstream_raw)
    except ValueError:
        return None
    items = lister.listing_items(original)
    if not items:
        return None
    listed = {
        index: item["id"]
        for index, item in enumerate(items)
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    by_id = state.listing_restorers(list(dict.fromkeys(listed.values())))
    restorers = {
        index: by_id[object_id] for index, object_id in listed.items() if object_id in by_id
    }
    if not restorers:
        return None
    delivered = json.loads(raw)  # a fresh tree to edit (raw is JSON: upstream_raw or a dump)
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
    return json.dumps(delivered, ensure_ascii=False).encode("utf-8") if changed else None


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
    request: Request,
    path: str,
    started: float,
    new_counts: dict[str, int] | None = None,
    new_warned: dict[str, int] | None = None,
    audit_token: object | None = None,
) -> JSONResponse:
    """A proxy-generated routing refusal (the router's 502 no_route, 404
    count_tokens, 402 budget): provider-shaped, recorded with its route
    row, no upstream contact."""
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
        await response.aread()
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
        response = await state.client.send(upstream_request, stream=True)
    except httpx.TransportError as exc:
        logger.warning(
            "%s %s upstream %s fault (%s)", method, path, hop.upstream, type(exc).__name__
        )
        state.upstream_errors[hop.upstream] += 1
        return HopResult(hop.upstream, None, {}, type(exc).__name__), None
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
) -> Response:
    """The routed request path: the router's pre-audit refusal, the
    write-ahead audit START row, the first hop, then issue/decide until the
    router stops; delivery rides the shared branches with the router's
    delivery hooks. Every decision is the router's; every byte moved is the
    core's."""
    method = request.method
    # The count_tokens 404 (decision 7): a refusal that never counts as an
    # attempt, so it precedes the write-ahead START row.
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
    first = plan.begin(outbound, outbound_obj, _request_headers(request))
    if isinstance(first, RouteRefusal):
        return _route_refusal(
            state,
            ctx.session_id,
            adapter,
            first,
            request=request,
            path=path,
            started=started,
            new_counts=new_counts,
            new_warned=new_warned,
            audit_token=audit_token,
        )
    hop: HopRequest = first
    response: httpx.Response | None = None
    while True:
        if hop.reissued_from is not None:
            state.metrics.reissues[(hop.reissued_from, hop.upstream)] += 1
        result, response = await _issue_hop(state, hop, plan.deadline, method=method, path=path)
        decision = plan.decide(result)
        if decision.next is None:
            break
        await _discard(response)
        response = None
        if decision.wait_seconds > 0:
            await plan.wait(decision.wait_seconds)
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
    )


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
            except (NotImplementedError, RuntimeError):
                # Windows event loops / non-main threads: reload unavailable.
                logger.debug("SIGHUP reload unavailable on this platform")
        # Each active off-machine audit sink gets a flush-loop task; both are
        # cancelled and given a final flush at shutdown so no tail is lost.
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
        try:
            yield
        finally:
            if sighup_registered:
                loop.remove_signal_handler(signal.SIGHUP)
            await state.client.aclose()
            state.vault_manager.close()
            if state.router is not None:
                state.router.close()
            _close_upstream_auths(state.upstream_auth)
            if state.access_gate is not None:
                state.access_gate.close()
            if state.audit is not None:
                state.audit.close()
            for task in background_tasks:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            for sink in audit_sinks:
                # Final flush so shutdown never silently drops the tail.
                await sink.aclose()
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
