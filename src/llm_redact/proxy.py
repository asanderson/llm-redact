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
import json
import logging
import os
import re
import secrets
import signal
import time
import urllib.parse
from collections import Counter, deque
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NamedTuple

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
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
    apply_env_overrides,
    default_config_path,
    load_config,
    resolve_config_path,
    resolve_credentials,
)
from llm_redact.detection.engine import (
    active_rule_names,
    build_allowlist,
    build_detectors,
    build_modes,
)
from llm_redact.eventstream import EventStreamError, EventStreamParser
from llm_redact.eventstream import serialize as serialize_eventstream
from llm_redact.licensing import ResolvedLicense, resolve_license
from llm_redact.metrics import Metrics
from llm_redact.multipart import parse_boundary as parse_multipart_boundary
from llm_redact.ndjson import NDJSONParser
from llm_redact.placeholders import PLACEHOLDER_RE
from llm_redact.plugin_api import (
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
)
from llm_redact.providers import ALL_ADAPTERS, ProviderAdapter, RouteKind
from llm_redact.providers.custom import CUSTOM_ROUTE_PREFIX, build_custom_adapters, custom_prefix
from llm_redact.realtime import ALL_WS_ADAPTERS, WsAdapter, websockets_available, ws_handle
from llm_redact.redactor import BlockedRequest, Redactor
from llm_redact.registry import get_registry, loaded_plugins, pro_package_installed
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEParser, serialize
from llm_redact.users import UsersError, UsersStore, send_verification_email
from llm_redact.vault import Vault, VaultManager

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
# The resolved named-user for the current request (2.0 licensing): set once
# in handle() after identity extraction, read at finalization by
# record_request — the same task-context trick as the traceparent, so the
# streaming finalizers attribute without threading a parameter through.
_REQUEST_USER: ContextVar[str | None] = ContextVar("llm_redact_user", default=None)

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
    """The session-scoped objects one request redacts and rehydrates with."""

    __slots__ = ("session_id", "vault", "redactor", "rehydrator")

    def __init__(
        self, session_id: str, vault: Vault, redactor: Redactor, rehydrator: Rehydrator
    ) -> None:
        self.session_id = session_id
        self.vault = vault
        self.redactor = redactor
        self.rehydrator = rehydrator


# Hop-by-hop / recomputed headers dropped when forwarding either direction.
_SKIP_REQUEST_HEADERS = frozenset(
    {"host", "content-length", "connection", "accept-encoding", "x-llm-redact-user"}
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
        self.session_router = registry.build_session_router(
            config.vault,
            durable_lookup=self.vault_manager.lookup_response_session,
        )
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
        # Named-user registry (2.0 licensing): opened on Pro+ tiers only.
        # The Free tier is the implicit single local user — no registry, no
        # enforcement, no new file on disk.
        self.users_store: UsersStore | None = registry.build_users_store(
            config.users, self.license.tier
        )
        for warning in config.routing.warnings:  # parser-owned (inert upstreams, metered
            logger.warning("routing: %s", warning)  # defaults, dead chains) — always logged
        self.router: Router | None = registry.build_router(config, self.license.tier)
        # The browser dashboard (llm-redact-pro): None without the package
        # or when it declines the tier; the dashboard paths then 404 with
        # the reason. Rebuilt by apply_config when a reload changes the tier.
        self.dashboard: Dashboard | None = registry.build_dashboard(self.license.tier)

    def resolve_user(self, presented_key: str | None) -> str | None:
        if presented_key is None or self.users_store is None:
            return None
        return self.users_store.lookup_key(presented_key)

    def user_enforcement_required(self) -> bool:
        """Named-user keys become mandatory once there are two or more
        verified users to tell apart (the llm-redact-pro users registry —
        without it there is nothing to enforce). This is access control
        for a multi-user deployment, not a license restriction: a solo
        user stays implicit with zero setup friction, and binding beyond
        loopback is purely a TLS question (validate_bind_security), never
        a seat question. Reads the live registry so a CLI invite/revoke
        applies immediately."""
        if self.users_store is None:
            return False
        return self.users_store.verified_count() >= 2

    def context_for(
        self, adapter: ProviderAdapter | None, method: str, path: str, parsed_body: Any
    ) -> RequestContext:
        if self.session_router.mode == "static":
            return self._static_context
        session_id = self.session_router.resolve(
            adapter.name if adapter is not None else None, method, path, parsed_body
        )
        if session_id == self._static_context.session_id:
            return self._static_context
        if session_id not in self._known_sessions:
            self._known_sessions.add(session_id)
            flat = json.dumps(parsed_body, ensure_ascii=False) if parsed_body is not None else ""
            if PLACEHOLDER_RE.search(flat):
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
        vault = self.vault_manager.get(session_id)
        # Thin per-request wrappers over the shared detectors, allowlist and
        # counters: object construction only — no regex compilation, no DB open.
        redactor = Redactor(
            self.detectors,
            vault,
            self.allowlist,
            counts=self.detection_counts,
            modes=self.modes,
            warn_counts=self.warn_counts,
        )
        rehydrator = Rehydrator(
            vault, fuzzy=self.config.rehydration.fuzzy, counts=self.rehydration_counts
        )
        return RequestContext(session_id, vault, redactor, rehydrator)

    def record_response_id(self, response_id: str, session_id: str) -> None:
        if self.session_router.mode == "static":
            return
        self.session_router.record_response_id(response_id, session_id)
        self.vault_manager.record_response_session(response_id, session_id)

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
        for field_name in restart_required:
            logger.warning(
                "config reload: [%s] changes require restart; keeping current", field_name
            )
        effective = dataclasses.replace(
            fresh, **{name: getattr(self.config, name) for name in RESTART_ONLY_KEYS}
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
        # Routing is hot (decision 16 / R-34). The router validates-then-swaps
        # its own state; it raises only ConfigError and changes nothing when
        # it does, so this sits after every other build and before the swap.
        # A reload that turns routing ON builds through the registry (where
        # the tier is checked — like users_store, a running router is never
        # re-gated); one that turns it OFF drops it (closed at swap time).
        router = self.router
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
    ) -> str:
        if adapter is not None and adapter.name in self.config.providers:
            return adapter.name
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
        # Pass-through traffic: infer the provider from well-known paths;
        # Anthropic is the default because that is the primary target tool.
        if path.startswith("/v1beta/"):
            return "gemini"
        if path.startswith(("/v1/projects/", "/v1beta1/")):
            return "vertex"
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

    def upstream_for(
        self,
        adapter: ProviderAdapter | None,
        path: str,
        headers: "Mapping[str, str] | None" = None,
    ) -> str:
        return self.config.providers[self.provider_for(adapter, path, headers)].upstream_base_url


def _request_headers(request: Request) -> list[tuple[str, str]]:
    headers = [
        (name, value)
        for name, value in request.headers.items()
        if name.lower() not in _SKIP_REQUEST_HEADERS
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
) -> AsyncIterator[bytes]:
    method, path, started, detections, warned, audit_token = request_meta
    parser = SSEParser()
    pool = RehydratorPool(ctx.vault, fuzzy=state.config.rehydration.fuzzy)
    response_id_seen = False
    try:
        async for chunk in upstream.aiter_bytes():
            for event in parser.feed(chunk):
                if not response_id_seen:
                    response_id = adapter.response_id_from_event(event)
                    if response_id is not None:
                        state.record_response_id(response_id, ctx.session_id)
                        response_id_seen = True
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
            status=upstream.status_code,
            started=started,
            streamed=True,
            detections=detections,
            rehydrations=dict(pool.counts),
            warned=warned,
            audit_token=audit_token,
            route=(state.finish_route(route, upstream.status_code) if route is not None else None),
        )


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
            status=upstream.status_code,
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
    if path in (
        f"{RESERVED_PREFIX}/users",
        f"{RESERVED_PREFIX}/users/invite",
        f"{RESERVED_PREFIX}/users/revoke",
    ):
        return await _handle_users(request, state)
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
        }
        if config.vault.backend in RDBMS_BACKENDS:
            from llm_redact.vault_rdbms import ENV_REMOTE_PLAINTEXT, managed_dbms_cloud

            # Honesty fields: a recognized managed-DBMS host and the
            # remote-plaintext hatch are opt-in postures — never silent.
            vault_block["managed_cloud"] = managed_dbms_cloud(config.vault)
            vault_block["remote_plaintext"] = (
                config.vault.encryption != "fernet" and os.environ.get(ENV_REMOTE_PLAINTEXT) == "1"
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
                "upstream_errors_total": dict(state.upstream_errors),
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
                "mcp_exempt_servers": len(config.detection.mcp_exempt_servers),
                "audit": {
                    "enabled": state.audit is not None,
                    "rows": state.audit.count() if state.audit is not None else 0,
                    "tamper_evident": config.audit.tamper_evident,
                    "required": config.audit.required,
                    "s3": {
                        "enabled": state.audit_s3 is not None,
                        "encryption": config.audit.s3.encryption == "fernet",
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
                "users": {
                    "registry": state.users_store is not None,
                    "verified": (
                        state.users_store.verified_count() if state.users_store is not None else 0
                    ),
                    "active": (
                        state.users_store.active_count() if state.users_store is not None else 0
                    ),
                    "enforcement": state.user_enforcement_required(),
                },
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
    return {"127.0.0.1", "localhost", "::1", state.config.host.lower()}


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
            status=upstream.status_code,
            started=started,
            streamed=True,
            detections=detections,
            rehydrations=dict(pool.counts),
            warned=warned,
            audit_token=audit_token,
            route=(state.finish_route(route, upstream.status_code) if route is not None else None),
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
    # pruned from the live process (the CLI can, with the proxy stopped).
    pruned = state.vault_manager.prune_sessions(
        days, exclude=frozenset({state.config.vault.session})
    )
    logger.info("pruned %d idle vault session(s) via /sessions/prune", pruned)
    return JSONResponse({"pruned": pruned})


async def _handle_users(request: Request, state: ProxyState) -> Response:
    """Named-user browser (GET /users) and invite/revoke (POST, behind the
    full config-editor guard stack). Metadata only: names, emails, statuses
    — never verification codes, never key hashes. Invite returns the code
    to the dashboard for manual delivery (or sends email when [email] is
    configured), mirroring the CLI."""
    if not _host_allowed(request, state):
        return JSONResponse({"error": "host not allowed"}, status_code=403)
    if not _origin_allowed(request, state):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    if state.users_store is None:
        return JSONResponse(
            {"error": "user management requires the llm-redact-pro package (see docs/editions.md)"},
            status_code=403,
        )

    if request.url.path == f"{RESERVED_PREFIX}/users":
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        return JSONResponse(
            {
                "max_users": state.license.max_users,
                "active": state.users_store.active_count(),
                "verified": state.users_store.verified_count(),
                "enforcement": state.user_enforcement_required(),
                "users": [
                    {
                        "name": row.name,
                        "email": row.email,
                        "status": row.status,
                        "invited_at": row.invited_at,
                        "verified_at": row.verified_at,
                    }
                    for row in state.users_store.list_users()
                ],
            },
            headers={"cache-control": "no-store"},
        )

    if request.method != "POST":
        return JSONResponse({"error": "method not allowed"}, status_code=405)
    payload, guard_error = await _guarded_post_json(request, state)
    if guard_error is not None:
        return guard_error
    if not isinstance(payload, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

    if request.url.path == f"{RESERVED_PREFIX}/users/invite":
        name = payload.get("name")
        email_addr = payload.get("email")
        if not isinstance(name, str) or not isinstance(email_addr, str):
            return JSONResponse(
                {"error": 'body must be {"name": "...", "email": "..."}'}, status_code=400
            )
        try:
            code = state.users_store.invite(name, email_addr, max_users=state.license.max_users)
        except UsersError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        sent = False
        if state.config.email.configured:
            assert state.config.email.smtp_host is not None
            assert state.config.email.from_address is not None
            try:
                send_verification_email(
                    smtp_host=state.config.email.smtp_host,
                    smtp_port=state.config.email.smtp_port,
                    starttls=state.config.email.starttls,
                    username=state.config.email.username,
                    password_env=state.config.email.password_env,
                    from_address=state.config.email.from_address,
                    to_address=email_addr,
                    display_name=name,
                    code=code,
                )
                sent = True
            except (OSError, UsersError) as exc:
                logger.warning("verification email failed: %s", exc)
        # The code goes back to the ADMIN's same-origin dashboard only when
        # it was not emailed — manual delivery mirrors the CLI --print-code.
        return JSONResponse({"invited": email_addr, "sent": sent, "code": None if sent else code})

    email_addr = payload.get("email")
    if not isinstance(email_addr, str):
        return JSONResponse({"error": 'body must be {"email": "..."}'}, status_code=400)
    try:
        state.users_store.revoke(email_addr, purge=bool(payload.get("purge", False)))
    except UsersError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"revoked": email_addr})


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


USER_KEY_HEADER = "x-llm-redact-user"
_USER_PATH_PREFIX = "/u/"


def _extract_user_key(request: Request) -> str | None:
    """Pull the named-user key off the request and SCRUB it in place.

    Two channels (llm-redact-pro docs/licensing.md): the universal ``/u/<key>/`` base-path
    prefix (the one knob every tool has is its base URL) and the
    ``x-llm-redact-user`` header. Both are removed here — from
    scope["path"], scope["raw_path"], and scope["headers"] — before any
    routing, logging, recording, or forwarding code can see them: the key
    is OUR credential and must never reach a provider or a log line.
    """
    scope = request.scope
    key: str | None = None
    path: str = scope["path"]
    if path.startswith(_USER_PATH_PREFIX):
        candidate, _, remainder = path[len(_USER_PATH_PREFIX) :].partition("/")
        if candidate:
            key = candidate
            scope["path"] = "/" + remainder
            # Scrub raw_path too (forwarding builds the upstream URL from it).
            # A byte-prefix match on the DECODED candidate silently fails when
            # the key segment is percent-encoded (`/u/lrk_%41BC/...`), which
            # would leave our identity credential in the forwarded URL. Strip
            # the first RAW segment after /u/ instead — encoding-agnostic — and
            # if raw_path doesn't start with /u/ at all (an encoded prefix),
            # fail closed by rebuilding it from the already-scrubbed path.
            raw: bytes | None = scope.get("raw_path")
            prefix_bytes = _USER_PATH_PREFIX.encode("latin-1")
            if raw is not None:
                if raw.startswith(prefix_bytes):
                    _, _, remainder_raw = raw[len(prefix_bytes) :].partition(b"/")
                    scope["raw_path"] = (b"/" + remainder_raw) if remainder_raw else b"/"
                else:
                    scope["raw_path"] = scope["path"].encode("latin-1")
    remaining: list[tuple[bytes, bytes]] = []
    header_key: str | None = None
    for name, value in scope["headers"]:
        if name.lower() == USER_KEY_HEADER.encode("ascii"):
            header_key = value.decode("latin-1").strip()
        else:
            remaining.append((name, value))
    if header_key is not None:
        scope["headers"] = remaining
        # Starlette caches Headers on first access; drop any cache so the
        # scrubbed list is what every later reader (incl. forwarding) sees.
        if hasattr(request, "_headers"):
            del request._headers
    return key if key is not None else (header_key or None)


async def handle(request: Request) -> Response:
    state: ProxyState = request.app.state.proxy
    path = request.url.path

    # Reserved local endpoints are answered here, before any routing or
    # upstream code runs — this early return is the non-forwarding guarantee.
    if path.startswith(RESERVED_PREFIX):
        response = await _handle_local(request, state)
        # Stamp browser-hardening headers on every reserved reply in one place
        # (setdefault so a handler that set its own header still wins).
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    # Named-user identity (2.0 licensing): extract and SCRUB the key before
    # anything else reads the path or headers, then resolve it to a name.
    user_key = _extract_user_key(request)
    path = request.scope["path"]  # the /u/<key> prefix is gone from here on
    user_name = state.resolve_user(user_key)
    _REQUEST_USER.set(user_name)

    # Captured once per request; read at finalization (incl. the streaming
    # finalizer, same task context) so an OTel span can parent into the
    # caller's trace. Trivial when telemetry is off; traceparent isn't secret.
    _INBOUND_TRACEPARENT.set(request.headers.get("traceparent"))
    started = time.perf_counter()
    adapter, kind = state.route(request.method, path, request.headers)

    # A disabled provider fails closed before anything is read or forwarded:
    # matched routes AND pass-through traffic inferred to it are answered
    # here (forwarding pass-through would send unredacted bodies to it).
    provider_name = state.provider_for(adapter, path, request.headers)
    provider_conf = state.config.providers.get(provider_name)
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

    if state.user_enforcement_required() and user_name is None:
        # Named-user enforcement (llm-redact-pro docs/licensing.md): required on team
        # deployments (2+ verified users, or any non-loopback bind). The
        # refusal carries instructions, never echoes a presented key, and
        # is recorded like every other proxy-generated response.
        message = (
            "this llm-redact proxy requires a named-user key: pass it via the"
            f" /u/<key>/ URL path prefix or the {USER_KEY_HEADER} header"
            " (llm-redact users verify issues keys)"
        )
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
        logger.info("%s %s -> 403 named-user key required", request.method, path)
        return JSONResponse(error, status_code=403)

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
    if adapter is not None and body_bytes:
        try:
            parsed = json.loads(body_bytes)
        except ValueError:
            parsed = None

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

    # Routing (the llm-redact-pro routing layer): the router plans BEFORE
    # redaction because the FIRST upstream's inject_system_note governs the
    # prepared body (decision 4; the redacted body is reused on later hops).
    # An unrouted request — no router held, or a provider outside the
    # router's protocols (plan() returns None) — never constructs a routing
    # object and takes the legacy path below byte-for-byte.
    plan: RoutePlan | None = None
    if state.router is not None:
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
                state, ctx, adapter, planned, request=request, path=path, started=started
            )
        plan = planned
    note_wanted = plan.inject_system_note if plan is not None else state.config.inject_system_note

    outbound = body_bytes
    # The decoded form of `outbound` (None for pass-through / non-JSON
    # bodies): the routed path applies per-hop body rewrites to it.
    outbound_obj: dict[str, Any] | None = parsed if isinstance(parsed, dict) else None
    detection_off = provider_conf is not None and not provider_conf.detection
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
    elif adapter is not None and isinstance(parsed, dict):
        try:
            prepared = adapter.prepare_request(
                parsed,
                ctx.redactor,
                inject_note=note_wanted and adapter.wants_system_note(kind, path),
                mcp_exempt=frozenset(state.config.detection.mcp_exempt_servers),
            )
        except BlockedRequest as exc:
            return blocked_response(exc, adapter)
        outbound_obj = prepared
        # No-op short-circuit: redaction increments detection_counts, and note
        # injection is gated on a redaction actually happening (base
        # prepare_request), so an unchanged count means the prepared body is
        # byte-for-byte the original. Forward the raw bytes and skip the
        # parse→dump round-trip — the common nothing-to-redact large-body case.
        if sum(state.detection_counts.values()) != sum(detection_counts_before.values()):
            outbound = json.dumps(prepared, ensure_ascii=False).encode("utf-8")
    elif adapter is not None and parsed is None and body_bytes:
        # Matched routes with non-JSON bodies: multipart uploads (OpenAI
        # /v1/files) get their JSONL file parts redacted; anything the
        # adapter declines to rewrite forwards verbatim (the non-JSON-body
        # default that keeps unknown formats working).
        boundary = parse_multipart_boundary(request.headers.get("content-type", ""))
        if boundary is not None:
            try:
                rewritten = adapter.redact_multipart(
                    path,
                    body_bytes,
                    boundary,
                    ctx.redactor,
                    inject_note=note_wanted and adapter.wants_system_note(kind, path),
                )
            except BlockedRequest as exc:
                # One leaking line in an uploaded file is a leak: the
                # whole request is rejected.
                return blocked_response(exc, adapter)
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

    upstream_base = state.upstream_for(adapter, path, request.headers)
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

    upstream_request = state.client.build_request(
        request.method, url, headers=_request_headers(request), content=outbound
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
                upstream, adapter, state, ctx, request_meta=request_meta, route=route
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
    payload: Any = None
    if kind is RouteKind.CHAT and adapter is not None and "application/json" in content_type:
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        if payload is not None:
            response_id = adapter.response_id_from_body(payload)
            if response_id is not None:
                state.record_response_id(response_id, ctx.session_id)
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
    ctx: RequestContext,
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
        session=ctx.session_id,
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
            ctx,
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
            ctx,
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
        views) and never touches the active static session — the same
        never-wrong-value discipline as the CLI prune. Sleep-first so startup
        is untouched; failures are logged, never fatal."""
        ttl = state.config.vault.session_ttl_days
        exclude = frozenset({state.config.vault.session})
        while True:
            await asyncio.sleep(interval)
            try:
                pruned = state.vault_manager.prune_sessions(ttl, exclude=exclude)
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
