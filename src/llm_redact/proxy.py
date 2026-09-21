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
import hashlib
import importlib.resources
import importlib.util
import json
import logging
import os
import re
import secrets
import signal
import time
import tomllib
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
    Config,
    ConfigError,
    VaultConfig,
    apply_env_overrides,
    default_config_path,
    load_config,
    parse_config,
    resolve_config_path,
    resolve_credentials,
)
from llm_redact.config_write import emit_config_toml, write_config_atomic
from llm_redact.detection.engine import (
    active_rule_names,
    build_allowlist,
    build_detectors,
    build_modes,
)
from llm_redact.detection.regex_rules import BUILTIN_RULES
from llm_redact.eventstream import EventStreamError, EventStreamParser
from llm_redact.eventstream import serialize as serialize_eventstream
from llm_redact.licensing import ResolvedLicense, resolve_license
from llm_redact.metrics import Metrics
from llm_redact.multipart import parse_boundary as parse_multipart_boundary
from llm_redact.ndjson import NDJSONParser
from llm_redact.placeholders import PLACEHOLDER_RE
from llm_redact.plugin_api import Telemetry
from llm_redact.pricing import PriceTable, StreamUsageTracker, Usage, parse_usage
from llm_redact.providers import ALL_ADAPTERS, ProviderAdapter, RouteKind
from llm_redact.providers.custom import CUSTOM_ROUTE_PREFIX, build_custom_adapters, custom_prefix
from llm_redact.realtime import ALL_WS_ADAPTERS, WsAdapter, websockets_available, ws_handle
from llm_redact.redactor import BlockedRequest, Redactor
from llm_redact.registry import get_registry, loaded_plugins, pro_package_installed
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.routing import (
    HOPS_HEADER,
    REISSUE_HEADER,
    RETRY_SAME,
    SPECIAL_STATUS_KEYS,
    UPSTREAM_HEADER,
    MissingCredential,
    PricesConfig,
    RouteRule,
    RoutingConfig,
    RoutingState,
    RuleMatch,
    UpstreamConfig,
    apply_body_rewrites,
    classify_auth,
    is_stateful_request,
    literal_models,
    outbound_headers,
    parse_retry_after,
    request_protocol,
    restore_model,
    select_rule,
    status_key,
    upstream_url,
)
from llm_redact.spend import Budget, BudgetLedger, InMemorySpendStore, SpendStore, SqliteSpendStore
from llm_redact.sse import SSEEvent, SSEParser, serialize
from llm_redact.users import UsersError, UsersStore, send_verification_email
from llm_redact.vault import Vault, VaultManager, default_vault_path

# Local endpoints under this prefix are answered by the proxy itself and are
# never forwarded upstream (see the first statement of handle()).
RESERVED_PREFIX = "/__llm-redact"

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


def _build_price_table(prices: PricesConfig) -> PriceTable:
    """The effective price table: builtin or `[prices] table = PATH`, with
    `[prices.override."id"]` entries winning. A bad file is a ConfigError
    (startup / serve --check / reload all report it the same way)."""
    base = (
        PriceTable.builtin()
        if prices.table == "builtin"
        else PriceTable.from_file(Path(prices.table).expanduser())
    )
    return base.with_overrides(dict(prices.overrides))


def _build_spend_store(vault: VaultConfig) -> SpendStore:
    """Where spend rows live (R-25): a `spend` table in the sqlite vault DB
    file (own connection), in-process memory for the memory backend AND
    for RDBMS vaults (the server-side schema is the pro package's; spend is
    a local operator ledger, documented as in-process there)."""
    if vault.backend == "sqlite":
        path = Path(vault.path).expanduser() if vault.path else default_vault_path()
        return SqliteSpendStore(path)
    return InMemorySpendStore()


def _budgets_for(routing: RoutingConfig) -> dict[str, Budget]:
    # Every upstream gets an entry (budget-less ones included) so the ledger
    # snapshot — and /status — always lists every configured upstream.
    return {
        upstream.name: Budget(
            usd=upstream.monthly_budget_usd,
            tokens=upstream.monthly_budget_tokens,
            zero_cost=upstream.zero_cost,
        )
        for upstream in routing.upstreams
    }


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
        # Routing credentials (R-3) resolve HERE, before anything is opened:
        # an `env:VAR` upstream whose variable is unset fails startup (and
        # serve --check) with a ConfigError naming the VAR — never its value.
        resolve_credentials(config.routing, os.environ)
        for warning in config.routing.warnings:
            logger.warning("routing: %s", warning)
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
        # Per-process CSRF token for the config editor: readable only via a
        # same-origin GET (the proxy never sends CORS headers), required as a
        # custom header on POST /__llm-redact/config.
        self.csrf_token = secrets.token_urlsafe(32)
        # Package data, read once: the dashboard is a single self-contained
        # HTML file (inline CSS/JS, no CDNs) polling the local endpoints.
        self.dashboard_html = (
            importlib.resources.files("llm_redact").joinpath("dashboard.html").read_text("utf-8")
        )
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
        # Routing layer (docs/routing.md). The unrouted path never touches
        # these; the routed path does one select_rule plus dict lookups.
        # Cooldown/counter state is in-process and survives apply_config
        # (prune_to drops names a reload removed).
        self.routing_state = RoutingState()
        self.routing_upstreams: dict[str, UpstreamConfig] = {
            upstream.name: upstream for upstream in config.routing.upstreams
        }
        self.price_table: PriceTable = _build_price_table(config.prices)
        # The durable spend table is opened in the vault file only when
        # routing is live (a legacy sqlite vault is never touched); a reload
        # that enables routing upgrades the store then (apply_config).
        self.spend_store: SpendStore = (
            _build_spend_store(config.vault) if config.routing.enabled else InMemorySpendStore()
        )
        self.budget_ledger = BudgetLedger(
            self.spend_store,
            _budgets_for(config.routing),
            reset_day=config.routing.budget_reset_day,
        )

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
            for field_name in (
                "vault",
                "audit",
                "host",
                "port",
                "log",
                "tls",
                "otel",
                "users",
                "email",
            )
            if getattr(fresh, field_name) != getattr(self.config, field_name)
        ]
        for field_name in restart_required:
            logger.warning(
                "config reload: [%s] changes require restart; keeping current", field_name
            )
        effective = dataclasses.replace(
            fresh,
            vault=self.config.vault,
            audit=self.config.audit,
            host=self.config.host,
            port=self.config.port,
            log=self.config.log,
            tls=self.config.tls,
            otel=self.config.otel,
            users=self.config.users,
            email=self.config.email,
        )

        # Re-resolve the license BEFORE anything is built or swapped:
        # [license] itself is hot, so renewals apply without a restart.
        license_resolved = _resolve_license_info(effective)
        # Routing credentials likewise: a reload naming an `env:VAR` that is
        # missing raises here (ConfigError, VAR name only) and the caller
        # keeps the running config — nothing below has been swapped yet.
        resolve_credentials(effective.routing, os.environ)
        price_table = (
            self.price_table
            if effective.prices == self.config.prices
            else _build_price_table(effective.prices)
        )
        # Spend storage follows the (restart-only) vault: enabling routing on
        # a sqlite vault opens the durable table now rather than at restart.
        spend_store = self.spend_store
        if (
            effective.routing.enabled
            and effective.vault.backend == "sqlite"
            and isinstance(spend_store, InMemorySpendStore)
        ):
            spend_store = _build_spend_store(effective.vault)
        budgets = _budgets_for(effective.routing)
        if (
            spend_store is not self.spend_store
            or effective.routing.budget_reset_day != self.config.routing.budget_reset_day
        ):
            budget_ledger = BudgetLedger(
                spend_store, budgets, reset_day=effective.routing.budget_reset_day
            )
        else:
            budget_ledger = self.budget_ledger
            budget_ledger.rebudget(budgets)

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

        if set(effective.providers) != set(self.config.providers):
            # Custom upstreams appeared/vanished: rebuild the adapter list
            # (in-flight requests keep their old adapter references).
            self.adapters = [cls() for cls in ALL_ADAPTERS] + build_custom_adapters(
                effective.providers
            )
        self.config = effective
        self.license = license_resolved
        self.detectors = detectors
        self.allowlist = allowlist
        self.modes = modes
        self.redactor = redactor
        self.rehydrator = rehydrator
        self._static_context = RequestContext(
            effective.vault.session, self.vault, redactor, rehydrator
        )
        self.routing_upstreams = {u.name: u for u in effective.routing.upstreams}
        self.price_table = price_table
        self.spend_store = spend_store
        self.budget_ledger = budget_ledger
        self.routing_state.prune_to(effective.routing.upstream_names())
        for warning in effective.routing.warnings:
            logger.warning("routing: %s", warning)
        logger.info("config reloaded (%d detection rules)", len(detectors))
        return restart_required

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
            # Routing decision (docs/routing.md): None when the request took
            # the legacy path, else rule/upstream/hops/auth/class/reissue.
            # Audit rows are unchanged (AuditRecord is shared with pro).
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

    def finish_route(
        self, delivery: "RouteDelivery", status: int | None, usage: Usage | None
    ) -> dict[str, Any]:
        """Close the books on a routed request: the routed-requests metric,
        spend attributed to the delivering upstream with its hop number
        (2xx with a parsable usage block only — passthrough included, in
        tokens, so `spend` shows subscription usage), and the `route` row
        record_request stores."""
        self.metrics.routed[(delivery.upstream.name, delivery.rule_id or "-")] += 1
        if usage is not None and status is not None and 200 <= status < 300:
            self.budget_ledger.record(
                upstream=delivery.upstream.name,
                model=delivery.sent_model or "unknown",
                hop=delivery.hops,
                usage=usage,
                price_table=self.price_table,
            )
        return delivery.as_row()

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


# The Gemini model id lives in the request PATH (decision 15b):
# /v1beta/models/{model}:generateContent. Group 2 is the id, matched on the
# raw (still percent-encoded) path so a rewrite leaves the rest byte-exact.
_GEMINI_MODEL_SEGMENT = re.compile(
    r"^(/(?:v1|v1beta)/(?:models|tunedModels)/)([^/:]+)(:[A-Za-z]+)$"
)


def gemini_path_model(path: str) -> str | None:
    """The model id between `/models/` and the `:verb` of a Gemini path, or
    None when the path has no such segment."""
    match = _GEMINI_MODEL_SEGMENT.match(path)
    return match.group(2) if match is not None else None


def _rewrite_gemini_path(raw_path: str, model: str) -> str:
    match = _GEMINI_MODEL_SEGMENT.match(raw_path)
    if match is None:
        return raw_path
    return match.group(1) + urllib.parse.quote(model, safe="") + match.group(3)


def _restore_model_payload(payload: Any, original_model: str, protocol: str) -> bool:
    """R-9 on a decoded response: the routing helper's `model` /
    `message.model` / `response.model` fields, plus Gemini's `modelVersion`,
    on a dict or on each element of Gemini's buffered array form. In place;
    returns whether anything changed."""
    if isinstance(payload, list):
        changed = False
        for item in payload:
            changed = _restore_model_payload(item, original_model, protocol) or changed
        return changed
    changed = restore_model(payload, original_model)
    if (
        protocol == "gemini"
        and isinstance(payload, dict)
        and isinstance(payload.get("modelVersion"), str)
        and payload["modelVersion"] != original_model
    ):
        payload["modelVersion"] = original_model
        changed = True
    return changed


def _restore_model_text(text: str, original_model: str, protocol: str) -> str:
    """`_restore_model_payload` over one serialized SSE data payload or NDJSON
    line; unparsable or unchanged text comes back byte-identical."""
    try:
        payload = json.loads(text)
    except ValueError:
        return text
    if not _restore_model_payload(payload, original_model, protocol):
        return text
    return json.dumps(payload, ensure_ascii=False)


class RouteDelivery:
    """The routing outcome of one request, threaded into the delivery branch
    and its finalizer: which rule/upstream/hop produced the response, the
    debug/reissue headers to add, the model id to restore, and the usage
    tracker the budget ledger reads at the end (docs/routing.md)."""

    __slots__ = (
        "rule_id",
        "upstream",
        "hops",
        "auth",
        "status_class",
        "reissue",
        "protocol",
        "original_model",
        "sent_model",
        "headers",
        "tracker",
        "method",
        "path",
    )

    def __init__(
        self,
        *,
        rule_id: str | None,
        upstream: UpstreamConfig,
        hops: int,
        auth: str,
        status_class: str,
        reissue: str,
        protocol: str,
        original_model: str | None,
        sent_model: str | None,
        headers: dict[str, str],
        method: str,
        path: str,
    ) -> None:
        self.rule_id = rule_id
        self.upstream = upstream
        self.hops = hops
        self.auth = auth
        self.status_class = status_class
        self.reissue = reissue
        self.protocol = protocol
        # Set only when the rule rewrote the model: the id the client sent
        # (restored on the way back) vs the id the upstream saw (priced).
        self.original_model = original_model
        self.sent_model = sent_model
        self.headers = headers
        self.tracker = StreamUsageTracker(protocol)
        self.method = method
        self.path = path

    def as_row(self) -> dict[str, Any]:
        return {
            "rule": self.rule_id,
            "upstream": self.upstream.name,
            "hops": self.hops,
            "auth": self.auth,
            "class": self.status_class,
            "reissue": self.reissue,
        }

    def observe_event(self, event: SSEEvent) -> SSEEvent:
        """Per delivered SSE event, after the adapter's rehydration: feed the
        usage tracker and restore the original model id (R-9)."""
        if event.data:
            self.tracker.feed_sse_data(event.data)
            if self.original_model is not None:
                event.data = _restore_model_text(event.data, self.original_model, self.protocol)
        return event

    def observe_line(self, line: bytes) -> bytes:
        """The NDJSON twin of observe_event (one line, without its newline)."""
        self.tracker.feed_ndjson_line(line)
        if self.original_model is None:
            return line
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError:
            return line
        restored = _restore_model_text(text, self.original_model, self.protocol)
        return line if restored is text else restored.encode("utf-8")

    def stream_failed(self, exc: httpx.TransportError) -> None:
        """A fault AFTER the first byte reached the client: propagated as-is
        (never re-issued — R-17), classified stream_error. Type only."""
        self.status_class = "stream_error"
        logger.warning(
            "%s %s stream failed after first byte (%s)%s",
            self.method,
            self.path,
            type(exc).__name__,
            _route_log_suffix(self.as_row()),
        )


def _route_log_suffix(row: dict[str, Any]) -> str:
    """The R-29 log fields for a routed request (unrouted lines never carry them)."""
    return (
        f" rule={row['rule'] or '-'} upstream={row['upstream'] or '-'} hops={row['hops']}"
        f" auth={row['auth']} class={row['class']} reissue={row['reissue']}"
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
            route.stream_failed(exc)
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
            route=(
                state.finish_route(route, upstream.status_code, route.tracker.result())
                if route is not None
                else None
            ),
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


_SPEND_STATUS_KEYS = (
    "period",
    "in_tokens",
    "out_tokens",
    "cache_read",
    "cache_write",
    "usd",
    "budget_usd",
    "budget_tokens",
    "remaining_usd",
    "remaining_tokens",
    "unpriced_rows",
    "reissue_usd",
    "reissue_tokens",
)


def _routing_status(state: ProxyState) -> dict[str, Any]:
    """The /status `routing` block (R-30): per-upstream health, cooldown,
    counters and spend. Credential MODE only — variable names and values
    never appear here. `{"enabled": false}` when routing is off."""
    routing = state.config.routing
    if not routing.enabled:
        return {"enabled": False}
    spend = state.budget_ledger.snapshot()
    upstreams: dict[str, Any] = {}
    for upstream in routing.upstreams:
        snapshot = state.routing_state.snapshot(upstream.name)
        entry = spend.get(upstream.name, {})
        exhausted = upstream.has_budget and state.budget_ledger.exhausted(upstream.name)
        upstreams[upstream.name] = {
            "protocol": upstream.protocol,
            "credential": upstream.credential_mode,
            "cost": upstream.cost,
            "legacy": upstream.legacy,
            "state": "budget_exhausted" if exhausted else snapshot["state"],
            "cooldown_remaining_seconds": snapshot["cooldown_remaining_seconds"],
            "requests": snapshot["requests"],
            "reissues_last_hour": snapshot["reissues_last_hour"],
            "last_error_class": snapshot["last_error_class"],
            "last_error_at": snapshot["last_error_at"],
            "spend": {key: entry.get(key) for key in _SPEND_STATUS_KEYS},
        }
    return {
        "enabled": True,
        "default_upstreams": dict(routing.default_upstreams),
        "rules": len(routing.rules),
        "reissues_last_hour": state.routing_state.reissues_last_hour(),
        "plan_limit_detection": routing.plan_limit_detection,
        "expose_models": routing.expose_models,
        "upstreams": upstreams,
        "unpriced_models": sorted(state.price_table.unknown_models),
        "warnings": list(routing.warnings),
    }


async def _handle_local(request: Request, state: ProxyState) -> Response:
    """Answer reserved /__llm-redact endpoints locally. Metadata only —
    never values; allowlists reported as counts. The /config editor endpoint
    is the one exception on both fronts: it accepts POST (behind the layered
    checks in _handle_config) and returns allowlist values."""
    path = request.url.path

    if path == f"{RESERVED_PREFIX}/config":
        return await _handle_config(request, state)
    if path == f"{RESERVED_PREFIX}/preview":
        return await _handle_preview(request, state)
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

    # The dashboard fetches absolute /__llm-redact/* paths, so both the bare
    # prefix and the trailing-slash form serve it (no redirect round-trip).
    if path in (RESERVED_PREFIX, f"{RESERVED_PREFIX}/"):
        return Response(
            content=state.dashboard_html,
            media_type="text/html; charset=utf-8",
            headers={"cache-control": "no-store"},
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
                    "clouds": list(state.license.clouds),
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
                "routing": _routing_status(state),
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


# Keys the editor may change; host/port/vault/audit are restart-only and are
# always taken from the on-disk file, never from the request.
_EDITABLE_KEYS = frozenset(
    {"inject_system_note", "max_body_bytes", "rehydration", "detection", "providers", "license"}
)
_READONLY_KEYS = frozenset(
    {"host", "port", "vault", "audit", "log", "tls", "otel", "users", "email"}
)
# Hot-reloadable (SIGHUP / apply_config) but NOT editable in the dashboard:
# the routing sections are preserved from FILE truth by the editor's merge,
# and a POST naming one of them is refused (decision 16, docs/routing.md).
_FILE_PRESERVED_KEYS = frozenset({"upstreams", "routing", "prices"})
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


def _config_target_path(state: ProxyState) -> Path:
    if state.config_path is not None:
        return state.config_path
    return resolve_config_path() or default_config_path()


def _config_fingerprint(path: Path) -> str | None:
    """Content hash of the config file, used by the editor's stale-form
    guard: a Save against a file that changed since the form loaded (a CLI
    edit, a SIGHUP'd rewrite, another browser tab) must not silently
    last-writer-wins over it."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _editable_view(config: Config) -> dict[str, Any]:
    ner = config.detection.ner
    return {
        "inject_system_note": config.inject_system_note,
        "max_body_bytes": config.max_body_bytes,
        "rehydration": {"fuzzy": config.rehydration.fuzzy},
        "detection": {
            "enabled": list(config.detection.enabled),
            "languages": (
                list(config.detection.languages) if config.detection.languages is not None else None
            ),
            "allowlist": list(config.detection.allowlist),
            "allowlist_patterns": list(config.detection.allowlist_patterns),
            "allowlist_by_type": {
                detector_type: list(values)
                for detector_type, values in config.detection.allowlist_by_type
            },
            "modes": {name: mode for name, mode in config.detection.modes},
            "mcp": {"exempt_servers": list(config.detection.mcp_exempt_servers)},
            "deny_strings": [
                {
                    "value": entry.value,
                    "case_sensitive": entry.case_sensitive,
                    "type": entry.detector_type,
                }
                for entry in config.detection.deny_strings
            ],
            "custom_rules": [
                {
                    "name": rule.name,
                    "type": rule.detector_type,
                    "pattern": rule.pattern,
                    "priority": rule.priority,
                    # Only surface the optional gate/prefilter fields when set,
                    # so a rule without them round-trips to the same TOML.
                    **({"validator": rule.validator} if rule.validator is not None else {}),
                    **({"required": list(rule.required)} if rule.required else {}),
                    **({"anchors": list(rule.anchors)} if rule.anchors else {}),
                }
                for rule in config.detection.custom_rules
            ],
            "ner": {
                "enabled": ner.enabled,
                "backend": ner.backend,
                "backends": list(ner.active_backends()),
                "entities": list(ner.entities),
                "max_chars": ner.max_chars,
                "score_threshold": ner.score_threshold,
                "language": ner.language,
                "model": ner.model,
                "models": dict(ner.models),
            },
        },
        "providers": {
            name: {
                "upstream_base_url": provider.upstream_base_url,
                "enabled": provider.enabled,
                "detection": provider.detection,
            }
            for name, provider in config.providers.items()
        },
    }


async def _handle_config(request: Request, state: ProxyState) -> Response:
    """The config editor endpoint. Layered checks, in order: Host (DNS
    rebinding), Origin, then for POST the CSRF header, content type, and a
    1 MiB body cap. OPTIONS gets 405 with no CORS headers, so cross-origin
    fetches carrying the custom header die at preflight."""
    if not _host_allowed(request, state):
        return JSONResponse({"error": "host not allowed"}, status_code=403)
    if not _origin_allowed(request, state):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)

    if request.method == "GET":
        target = _config_target_path(state)
        return JSONResponse(
            {
                "csrf_token": state.csrf_token,
                "config_path": str(target),
                "config_file_exists": target.exists(),
                "config_fingerprint": _config_fingerprint(target),
                "editable": _editable_view(state.config),
                "readonly": {
                    "host": state.config.host,
                    "port": state.config.port,
                    "vault": {
                        "backend": state.config.vault.backend,
                        "path": state.config.vault.path,
                        "session": state.config.vault.session,
                        "session_mode": state.config.vault.session_mode,
                        "encryption": state.config.vault.encryption,
                    },
                    "audit": {
                        "enabled": state.config.audit.enabled,
                        "path": state.config.audit.path,
                        "max_rows": state.config.audit.max_rows,
                        "tamper_evident": state.config.audit.tamper_evident,
                        "required": state.config.audit.required,
                        "s3": {
                            "enabled": state.config.audit.s3.enabled,
                            "provider": state.config.audit.s3.provider,
                            "bucket": state.config.audit.s3.bucket,
                        },
                        "azure": {
                            "enabled": state.config.audit.azure.enabled,
                            "account": state.config.audit.azure.account,
                            "container": state.config.audit.azure.container,
                        },
                    },
                    "log": {"format": state.config.log.format},
                    "tls": {
                        "enabled": state.config.tls.enabled,
                        "mutual": state.config.tls.mutual,
                    },
                    "otel": {
                        "enabled": state.config.otel.enabled,
                        "endpoint": state.config.otel.endpoint,
                        "service_name": state.config.otel.service_name,
                    },
                },
                "builtin_rules": sorted(rule.name for rule in BUILTIN_RULES),
                # Language tags per rule (untagged rules are universal) plus
                # the enabled-but-scoped-out list, so the editor's effective-
                # rule display can never disagree with what actually runs.
                "builtin_rule_languages": {
                    rule.name: list(rule.languages)
                    for rule in BUILTIN_RULES
                    if rule.languages is not None
                },
                "language_inactive_rules": sorted(
                    set(state.config.detection.enabled)
                    - set(active_rule_names(state.config.detection))
                ),
                "warnings": [
                    "saving rewrites the config file; comments are not preserved "
                    "(one .bak of the previous file is kept)"
                ],
            },
            headers={"cache-control": "no-store"},
        )
    if request.method != "POST":
        return JSONResponse({"error": "method not allowed"}, status_code=405)
    return await _handle_config_post(request, state)


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
            route.stream_failed(exc)
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
            route=(
                state.finish_route(route, upstream.status_code, route.tracker.result())
                if route is not None
                else None
            ),
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


async def _handle_preview(request: Request, state: ProxyState) -> Response:
    """Config dry-run: run the LIVE detection pipeline over caller-supplied
    text and report what WOULD be redacted / warned / blocked — no upstream
    request, no vault write, no metrics, no audit. The caller's text comes
    back masked, so no new value ever leaves the box; behind the same
    Host/Origin/CSRF guard as the config editor."""
    if not _host_allowed(request, state):
        return JSONResponse({"error": "host not allowed"}, status_code=403)
    if not _origin_allowed(request, state):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    if request.method != "POST":
        return JSONResponse({"error": "method not allowed"}, status_code=405)
    payload, guard_error = await _guarded_post_json(request, state)
    if guard_error is not None:
        return guard_error
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        return JSONResponse({"error": 'body must be {"text": "..."}'}, status_code=400)

    # A throwaway vault + fresh counters: the live vault, metrics, and audit
    # are never touched. Reuses the live detectors/allowlist/modes so the
    # preview matches exactly what a real request would do.
    from llm_redact.vault import InMemoryVault

    redactor = Redactor(state.detectors, InMemoryVault(), state.allowlist, modes=state.modes)
    blocked: dict[str, str] | None = None
    redacted: str | None = None
    try:
        redacted = redactor.redact_text(text)
    except BlockedRequest as exc:
        # A block-mode rule matched: the real request would be a 400 before
        # any upstream contact. Report the type (never the value).
        blocked = {"type": exc.detector_type}
    return JSONResponse(
        {
            "redacted": redacted,
            "detections": dict(redactor.counts),
            # Warn-mode values are LEFT IN the redacted text and forwarded on
            # a real request — the preview shows exactly that (honest).
            "warnings": dict(redactor.warn_counts),
            "blocked": blocked,
        },
        headers={"cache-control": "no-store"},
    )


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


async def _handle_config_post(request: Request, state: ProxyState) -> Response:
    payload, guard_error = await _guarded_post_json(request, state)
    if guard_error is not None:
        return guard_error
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        return JSONResponse({"error": 'body must be {"config": {...}}'}, status_code=400)
    edits: dict[str, Any] = payload["config"]

    readonly_hit = sorted(set(edits) & _READONLY_KEYS)
    if readonly_hit:
        return JSONResponse(
            {"error": f"key(s) {readonly_hit} require a restart and cannot be edited here"},
            status_code=400,
        )
    preserved_hit = sorted(set(edits) & _FILE_PRESERVED_KEYS)
    if preserved_hit:
        return JSONResponse(
            {
                "error": f"key(s) {preserved_hit} are not editable here: edit the file and"
                " reload (SIGHUP, after `llm-redact serve --check`)"
            },
            status_code=400,
        )
    unknown = sorted(set(edits) - _EDITABLE_KEYS)
    if unknown:
        return JSONResponse({"error": f"unknown key(s) {unknown}"}, status_code=400)

    path = _config_target_path(state)
    fingerprint = payload.get("fingerprint")
    if isinstance(fingerprint, str) and fingerprint:
        current = _config_fingerprint(path)
        if current is not None and current != fingerprint:
            return JSONResponse(
                {
                    "error": "the config file changed since this editor loaded"
                    " (another edit or a reload) — reload the page and re-apply"
                    " your changes"
                },
                status_code=409,
            )

    # Merge over FILE truth: readonly sections come from the file verbatim
    # (so env-var host/port overrides are never baked in), and editable keys
    # not present in the request keep their file values.
    file_raw: dict[str, Any] = {}
    if path.exists():
        try:
            file_raw = tomllib.loads(path.read_text())
        except (tomllib.TOMLDecodeError, OSError):
            return JSONResponse(
                {"error": f"the config file at {path} is not valid TOML; fix it manually"},
                status_code=409,
            )
    merged = {
        key: value
        for key, value in file_raw.items()
        if key in _READONLY_KEYS or key in _FILE_PRESERVED_KEYS
    }
    for key in _EDITABLE_KEYS:
        if key in edits:
            merged[key] = edits[key]
        elif key in file_raw:
            merged[key] = file_raw[key]

    # Validation runs the exact production paths: parse_config, then a
    # dry-run build of detectors/allowlists (bad regexes, unknown rules,
    # missing NER extras).
    try:
        candidate = parse_config(merged, "<config editor>")
        build_detectors(candidate.detection)
        build_allowlist(candidate.detection)
        build_modes(candidate.detection)
        # License resolution runs at VALIDATION time (informational only —
        # the FOSS core has no tier gates) so a bad [license] value 400s
        # here, before the file write.
        _resolve_license_info(apply_env_overrides(candidate))
        # The preserved routing sections re-validate with the rest: an
        # `env:VAR` credential that vanished since startup 400s here too.
        resolve_credentials(candidate.routing, os.environ)
    except (ValueError, TypeError, re.error, ImportError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    text = emit_config_toml(candidate)
    if parse_config(tomllib.loads(text), "<emitter check>") != candidate:
        logger.error("config editor: emitter round-trip mismatch; nothing written")
        return JSONResponse({"error": "internal emitter round-trip mismatch"}, status_code=500)
    try:
        backup = write_config_atomic(path, text)
    except OSError as exc:
        return JSONResponse({"error": f"could not write {path}: {exc.strerror}"}, status_code=500)
    # No await between validation and swap: SIGHUP reload cannot interleave.
    restart_required = state.apply_config(apply_env_overrides(candidate))
    logger.info("config editor: applied and wrote %s", path)
    return JSONResponse(
        {
            "applied": True,
            "path": str(path),
            "backup": str(backup) if backup is not None else None,
            "restart_required": restart_required,
        }
    )


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

    routing = state.config.routing
    if (
        routing.enabled
        and routing.expose_models
        and request.method == "GET"
        and path == "/v1/models"
    ):
        # R-15 model discovery: answered locally, BEFORE adapter routing.
        return _models_response(state, request, started)

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

    # Routing (docs/routing.md): the rule is selected BEFORE redaction
    # because the FIRST upstream's inject_system_note governs the prepared
    # body (decision 4; the redacted body is reused on later hops). An
    # unrouted request — routing off, or a provider outside the four
    # protocols — never constructs a routing object and takes the legacy
    # path below byte-for-byte.
    plan: _RoutePlan | None = None
    if state.config.routing.enabled:
        protocol = request_protocol(adapter.name if adapter is not None else None, provider_name)
        if protocol is not None:
            plan = _plan_route(state, request, protocol, parsed, path)
            if plan.primary is None:
                # No rule and no default for this protocol: proxy-generated
                # 502, never forwarded by guesswork (decision 2).
                return _route_refusal(
                    state,
                    ctx,
                    adapter,
                    request=request,
                    path=path,
                    started=started,
                    status=502,
                    message=(
                        "llm-redact routing: no rule matched and no default_upstream"
                        f" for protocol {protocol}"
                    ),
                    row={
                        "rule": plan.rule.id if plan.rule is not None else None,
                        "upstream": None,
                        "hops": 0,
                        "auth": plan.auth,
                        "class": "no_route",
                        "reissue": "no",
                    },
                )
    note_wanted = (
        plan.primary.inject_system_note
        if plan is not None and plan.primary is not None
        else state.config.inject_system_note
    )

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
    route: "RouteDelivery | None",
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
        route.status_class = "transport"
        row = state.finish_route(route, 502, None)
        state.upstream_errors[route.upstream.name] += 1
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
    return JSONResponse(body, status_code=502, headers=route.headers if route is not None else None)


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
    route: "RouteDelivery | None",
) -> Response:
    """Hand an upstream response to the client: the streaming branches
    (chosen by the upstream RESPONSE content-type, never the request's
    stream flag) rehydrate as they go and finalize at stream end; anything
    else is buffered. With `route` (a routed request) the same branches
    also restore a rewritten model id, feed the usage tracker for the
    budget ledger, and stamp the x-llm-redact-* headers."""
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
    usage: Usage | None = None
    if kind is RouteKind.CHAT and adapter is not None and "application/json" in content_type:
        try:
            payload: Any = json.loads(raw)
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
            if route is not None:
                usage = parse_usage(route.protocol, payload)
                if route.original_model is not None and _restore_model_payload(
                    rehydrated, route.original_model, route.protocol
                ):
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
        route=(
            state.finish_route(route, upstream.status_code, usage) if route is not None else None
        ),
    )

    return Response(content=raw, status_code=upstream.status_code, headers=headers)


# ---------------------------------------------------------------------------
# Routing (docs/routing.md): rule selection, the hop loop, local refusals.
# ---------------------------------------------------------------------------


class _RoutePlan(NamedTuple):
    protocol: str
    rule: RouteRule | None
    primary: UpstreamConfig | None  # None = no rule and no default (502 no_route)
    auth: str
    model: str | None


# Stands in for "no rule matched" in apply_body_rewrites: no model rewrite,
# while the default upstream's body_defaults / stream_options still apply.
_DEFAULT_RULE = RouteRule(id="", match=RuleMatch(protocol=""), upstream="")

# The throttle wait (R-7 retry-same); a module attribute so tests can stub
# the sleep without touching the event loop's own.
_RETRY_SLEEP = asyncio.sleep


def _plan_route(
    state: ProxyState, request: Request, protocol: str, parsed: Any, path: str
) -> _RoutePlan:
    """One select_rule per request: auth from the inbound headers, model from
    the body (or, for Gemini, from the `/models/{m}:verb` path segment),
    then the first matching rule, else the protocol's default upstream."""
    routing = state.config.routing
    auth = classify_auth(request.headers, routing.oauth_beta_marker)
    model = parsed.get("model") if isinstance(parsed, dict) else None
    if not isinstance(model, str):
        model = gemini_path_model(path) if protocol == "gemini" else None
    rule = select_rule(
        routing, protocol=protocol, model=model, headers=request.headers, path=path, auth=auth
    )
    name = rule.upstream if rule is not None else routing.default_for(protocol)
    primary = state.routing_upstreams.get(name) if name is not None else None
    return _RoutePlan(protocol, rule, primary, auth, model)


def _route_refusal(
    state: ProxyState,
    ctx: RequestContext,
    adapter: ProviderAdapter | None,
    *,
    request: Request,
    path: str,
    started: float,
    status: int,
    message: str,
    row: dict[str, Any],
    new_counts: dict[str, int] | None = None,
    new_warned: dict[str, int] | None = None,
    audit_token: object | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """A proxy-generated routing refusal (502 no_route, 404 count_tokens,
    402 budget): provider-shaped, recorded with its route row, no upstream
    contact. The message names upstreams/protocols, never values."""
    logger.info("%s %s -> %d%s", request.method, path, status, _route_log_suffix(row))
    state.record_request(
        session=ctx.session_id,
        provider=adapter.name if adapter is not None else None,
        method=request.method,
        path=path,
        status=status,
        started=started,
        streamed=False,
        detections=new_counts or {},
        rehydrations={},
        warned=new_warned,
        audit_token=audit_token,
        route=row,
    )
    body = adapter.error_body(message, status=status) if adapter is not None else {"error": message}
    return JSONResponse(body, status_code=status, headers=headers)


def _resolved_status_key(rule: RouteRule, key: str) -> str | None:
    """The on_status key `rule.chain_for(key)` resolves through (the class
    the log line reports — same precedence: special > exact > class), or
    None when the rule has no chain for this status."""
    if key in SPECIAL_STATUS_KEYS:
        candidates: tuple[str, ...] = (key, "429", "4xx")
    elif key.isdigit():
        candidates = (key, f"{key[0]}xx")
    else:
        candidates = (key,)
    table = dict(rule.on_status)
    for candidate in candidates:
        if candidate in table:
            return candidate
    return None


def _next_candidate(
    state: ProxyState, chain: tuple[str, ...], position: int, *, count_tokens_path: bool
) -> tuple[UpstreamConfig | None, int]:
    """The next eligible chain member from `position`: known, not a
    passthrough upstream (decision 3, asserted at runtime as well as at
    parse), healthy (not in cooldown), not budget-exhausted, and able to
    serve count_tokens when that is the path. Returns the member and the
    position after it (None = the chain is exhausted)."""
    while position < len(chain):
        name = chain[position]
        position += 1
        candidate = state.routing_upstreams.get(name)
        if candidate is None:
            continue
        if candidate.is_passthrough:
            logger.warning("routing: chain member %s is a passthrough upstream; skipped", name)
            continue
        if not state.routing_state.healthy(name):
            continue
        if candidate.has_budget and state.budget_ledger.exhausted(name):
            continue
        if count_tokens_path and not candidate.count_tokens:
            continue
        return candidate, position
    return None, position


async def _discard(response: httpx.Response | None) -> None:
    """Close an upstream response that will not be delivered (a failed hop
    or a throttled attempt) before the next one is sent."""
    if response is not None:
        with suppress(Exception):
            await response.aclose()


def _models_response(state: ProxyState, request: Request, started: float) -> Response:
    """R-15: `GET /v1/models` answered locally from the configured model
    names — Anthropic shape when the request carries `anthropic-version`,
    OpenAI shape otherwise. Recorded with provider `routing`."""
    models = literal_models(state.config.routing)
    body: dict[str, Any]
    if "anthropic-version" in request.headers:
        body = {
            "data": [
                {
                    "type": "model",
                    "id": model,
                    "display_name": model,
                    "created_at": "2026-01-01T00:00:00Z",
                }
                for model in models
            ],
            "has_more": False,
            "first_id": models[0] if models else None,
            "last_id": models[-1] if models else None,
        }
    else:
        body = {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "created": 0, "owned_by": "llm-redact"}
                for model in models
            ],
        }
    state.record_request(
        session=state.config.vault.session,
        provider="routing",
        method="GET",
        path="/v1/models",
        status=200,
        started=started,
        streamed=False,
        detections={},
        rehydrations={},
    )
    logger.info("GET /v1/models -> 200 answered locally (routing.expose_models)")
    return JSONResponse(body)


async def _handle_routed(
    request: Request,
    state: ProxyState,
    ctx: RequestContext,
    plan: _RoutePlan,
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
    """The routed request path (docs/routing.md): local gates (count_tokens,
    budget), then the hop loop — per hop the outbound headers, body rewrites
    and URL for the target upstream; the response status classified into
    a status key; the rule's chain deciding deliver / retry-same / re-issue.
    Every decision happens on the upstream response HEADERS, before any
    body byte reaches the client (R-17), and a response not delivered is
    closed before the next hop. Delivery then rides the shared branches
    with the route wrappers."""
    routing = state.config.routing
    rstate = state.routing_state
    rule = plan.rule
    primary = plan.primary
    assert primary is not None  # handle() answered no_route before calling
    method = request.method
    rule_id = rule.id if rule is not None else None
    count_tokens_path = plan.protocol == "anthropic" and path == "/v1/messages/count_tokens"

    def refusal(
        status: int,
        message: str,
        status_class: str,
        *,
        audit_token: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> JSONResponse:
        return _route_refusal(
            state,
            ctx,
            adapter,
            request=request,
            path=path,
            started=started,
            status=status,
            message=message,
            row={
                "rule": rule_id,
                "upstream": primary.name,
                "hops": 0,
                "auth": plan.auth,
                "class": status_class,
                "reissue": "no",
            },
            new_counts=new_counts,
            new_warned=new_warned,
            audit_token=audit_token,
            headers=headers,
        )

    # count_tokens gate (decision 7): no upstream contact, no fallback.
    if count_tokens_path and not primary.count_tokens:
        return refusal(
            404,
            f"llm-redact routing: upstream {primary.name} does not implement count_tokens",
            "404",
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

    # Model rewrite (R-9): the body's `model` (or, for Gemini, the path
    # segment) is replaced before forwarding and the ORIGINAL id restored
    # on the way back; spend is priced at the id the upstream saw.
    upstream_path = _upstream_path(request, path)
    rewrite_rule = rule if rule is not None else _DEFAULT_RULE
    original_model: str | None = None
    sent_model = plan.model
    if rule is not None and rule.model_rewrite is not None:
        sent_model = rule.model_rewrite
        if plan.model != rule.model_rewrite:
            original_model = plan.model
        if plan.protocol == "gemini":
            upstream_path = _rewrite_gemini_path(upstream_path, rule.model_rewrite)
            # The body never carries `model` on this protocol — Google
            # rejects one — so the routing helper must not add it.
            rewrite_rule = dataclasses.replace(rule, model_rewrite=None)

    inbound = _request_headers(request)
    deadline = time.monotonic() + routing.request_deadline_seconds
    current = primary
    hops = 0
    reissued = False
    retried_same = False
    stateful: bool | None = None  # R-22, computed once and only when needed
    chain: tuple[str, ...] | None = None
    position = 0
    skip_reason: str | None = None
    status_class = "ok"
    response: httpx.Response | None = None

    # Budget gate on the primary (decision 11): 402, unless the rule's
    # on_budget_exhausted chain continues — the primary then counts as the
    # attempt that was refused locally, so the first member is hop 2.
    if primary.has_budget and state.budget_ledger.exhausted(primary.name):
        candidate: UpstreamConfig | None = None
        if rule is not None and rule.on_budget_exhausted:
            chain = rule.on_budget_exhausted
            hops = 1
            candidate, position = _next_candidate(
                state, chain, 0, count_tokens_path=count_tokens_path
            )
        if candidate is None:
            return refusal(
                402,
                f"llm-redact routing: upstream {primary.name} budget exhausted for this period",
                "budget_exhausted",
                audit_token=audit_token,
                headers=(
                    {REISSUE_HEADER: "skipped; reason=no-candidate"} if chain is not None else None
                ),
            )
        rstate.record_reissue(primary.name, candidate.name)
        state.metrics.reissues[(primary.name, candidate.name)] += 1
        reissued = True
        current = candidate

    while True:
        hops += 1
        rstate.record_request(current.name)
        response = None
        failed = False
        retry_after: float | None = None
        try:
            headers_out = outbound_headers(
                inbound, current, environ=os.environ, oauth_marker=routing.oauth_beta_marker
            )
        except MissingCredential as exc:
            # The VAR vanished since startup: names the variable only.
            logger.warning("%s %s upstream %s unavailable: %s", method, path, current.name, exc)
            failed = True
        else:
            body = outbound
            if outbound_obj is not None:
                rewritten = apply_body_rewrites(outbound_obj, upstream=current, rule=rewrite_rule)
                if rewritten is not None:
                    body = json.dumps(rewritten, ensure_ascii=False).encode("utf-8")
            url = upstream_url(current, upstream_path, request.url.query)
            upstream_request = state.client.build_request(
                method, url, headers=headers_out, content=body
            )
            try:
                response = await state.client.send(upstream_request, stream=True)
            except httpx.TransportError as exc:
                # Type only: an httpx message can embed the URL (query auth).
                logger.warning(
                    "%s %s upstream %s fault (%s)", method, path, current.name, type(exc).__name__
                )
                failed = True
        if failed:
            # Decision 8: a transport fault is status key "502" for the
            # chain lookup and counts against the upstream by NAME.
            state.upstream_errors[current.name] += 1
            key = "502"
            status_class = "transport"
        else:
            assert response is not None
            if 200 <= response.status_code < 300:
                status_class = "ok"
                break
            key = status_key(
                response.status_code,
                protocol=plan.protocol,
                response_headers=response.headers,
                config=routing,
            )
            status_class = key
            retry_after = parse_retry_after(response.headers.get("retry-after"))

        resolved = _resolved_status_key(rule, key) if rule is not None else None
        action = rule.chain_for(key) if rule is not None and resolved is not None else None
        if chain is None:
            if action is None:
                break  # no chain for this status: deliver the response as-is
            if action == RETRY_SAME:
                # R-7 / decision 9: one retry of the SAME upstream after
                # max(retry-after, 2 s); a longer wait than the throttle cap
                # (or the deadline) hands the 429 back — the client owns
                # long retries. Never a cooldown, never another upstream.
                wait = max(retry_after or 0.0, 2.0)
                if (
                    retried_same
                    or hops >= routing.max_hops
                    or wait > routing.throttle_retry_max_seconds
                    or time.monotonic() + wait > deadline
                ):
                    break
                await _discard(response)
                retried_same = True
                await _RETRY_SLEEP(wait)
                continue
            assert isinstance(resolved, str) and isinstance(action, tuple)
            status_class = resolved
            # R-19 cooldown for the FAILED upstream: honour retry-after when
            # it is longer (RoutingState re-applies the 3600 s cap).
            rstate.mark_unhealthy(
                current.name, max(current.cooldown_seconds, retry_after or 0.0), key
            )
            if rule is not None and rule.reissue_policy == "never":
                break
            if rule is not None and rule.reissue_policy == "stateless-only":
                if stateful is None:
                    stateful = is_stateful_request(outbound_obj)
                if stateful:
                    skip_reason = "stateful"
                    break
            chain = action
            position = 0
        elif isinstance(action, tuple):
            # A chain member failed with a status the rule lists: it enters
            # cooldown too; the chain continues with the next member either
            # way (walkthrough 6 — an unlisted 4xx from a member still moves
            # on to the next one).
            assert isinstance(resolved, str)
            status_class = resolved
            rstate.mark_unhealthy(
                current.name, max(current.cooldown_seconds, retry_after or 0.0), key
            )
        if hops >= routing.max_hops or time.monotonic() >= deadline:
            break
        candidate, position = _next_candidate(
            state, chain, position, count_tokens_path=count_tokens_path
        )
        if candidate is None:
            skip_reason = "no-candidate"
            break
        await _discard(response)
        rstate.record_reissue(current.name, candidate.name)
        state.metrics.reissues[(current.name, candidate.name)] += 1
        reissued = True
        current = candidate

    reissue = f"skipped:{skip_reason}" if skip_reason is not None else ("yes" if reissued else "no")
    extra: dict[str, str] = {}
    if hops >= 2 or routing.debug_headers:
        extra[UPSTREAM_HEADER] = current.name
        extra[HOPS_HEADER] = str(hops)
    if skip_reason is not None:
        extra[REISSUE_HEADER] = f"skipped; reason={skip_reason}"
    route = RouteDelivery(
        rule_id=rule_id,
        upstream=current,
        hops=hops,
        auth=plan.auth,
        status_class=status_class,
        reissue=reissue,
        protocol=plan.protocol,
        original_model=original_model,
        sent_model=sent_model,
        headers=extra,
        method=method,
        path=path,
    )

    if response is None:
        # The last hop was a transport fault (or a vanished credential) and
        # nothing took over: the provider-shaped 502, recorded and closed
        # as class=transport (the fault was already warned and counted).
        row = state.finish_route(route, 502, None)
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
        return JSONResponse(error, status_code=502, headers=extra)

    logger.info(
        "%s %s -> %d%s%s",
        method,
        path,
        response.status_code,
        _route_log_suffix(route.as_row()),
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
            state.spend_store.close()
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
