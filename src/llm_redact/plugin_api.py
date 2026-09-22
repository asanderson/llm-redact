"""Stable contract for out-of-tree plugins (the paid ``llm-redact-pro``
package registers against these names).

The open-core split (llm-redact-pro docs/licensing.md) lets a separately-distributed
package supply the paid vault/audit/session/telemetry/users implementations.
Those implementations must bind to a *supported* surface, not to private
internals that move between releases. This module is that surface: re-exports
promoted from the Free core with public names. Everything here is part of the
plugin API contract — change it deliberately, never incidentally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .providers.base import RouteKind
from .sse import SSEEvent
from .vault import (
    _MAX_RESPONSE_ROWS as MAX_RESPONSE_ROWS,
)
from .vault import (
    _RESPONSE_PRUNE_EVERY as RESPONSE_PRUNE_EVERY,
)
from .vault import (
    Vault,
    VaultKeyError,
    VaultManager,
)

if TYPE_CHECKING:
    from .config import Config


class Telemetry(Protocol):
    """The telemetry recorder surface the proxy depends on.

    The concrete implementation (OpenTelemetry export) lives in the paid
    ``llm-redact-pro`` package; the Free core holds only this structural
    contract so ``proxy.py`` stays type-checked without importing pro.
    """

    def record(
        self, row: dict[str, Any], duration_seconds: float, *, traceparent: str | None = None
    ) -> None: ...

    def shutdown(self) -> None: ...


class VaultCipher(Protocol):
    """The at-rest vault cipher surface the Free vault code branches on.

    The concrete implementation (Fernet + HKDF, env/keyring key resolution)
    lives in the paid ``llm-redact-pro`` package; the Free vault classes accept
    a ``VaultCipher | None`` and are inert (unencrypted) when it is None, which
    is always the case unless the pro package supplies one.
    """

    def mac(self, session_id: str, detector_type: str, original: str) -> str: ...

    def encrypt(self, original: str) -> bytes: ...

    def decrypt(self, token: bytes) -> str: ...

    def key_check(self) -> str: ...


class SessionRouter(Protocol):
    """The session-routing surface the proxy depends on.

    Static-mode routing — one shared placeholder namespace, ``mode ==
    "static"`` — lives in the Free core (``StaticSessionRouter``); the
    per-conversation router (which derives a per-conversation session id from
    the conversation anchor) lives in the paid ``llm-redact-pro`` package. The
    proxy holds only this structural contract and reads ``mode`` to keep the
    static hot path (a prebuilt ``RequestContext``, no per-request resolution).

    ``resolve`` and ``record_response_id`` are only invoked when ``mode`` is not
    ``"static"``, so the Free static router implements them as inert stubs.
    """

    mode: str

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str: ...

    def record_response_id(self, response_id: str, session_id: str) -> None: ...


# --- upstream routing seam ----------------------------------------------------
# The routing LAYER (rule selection, credential modes, fallback chains,
# cooldowns, plan-limit detection, budgets, prices, spend, model discovery)
# is a paid subsystem implemented in llm-redact-pro. The core keeps only the
# MECHANICS of a routed request — issue a hop, close what is not delivered,
# deliver with rehydration through the shared branches, finalize and record —
# and asks the Router for every decision through the objects below.
# Metadata discipline binds every field the core may log or surface:
# upstream NAMES, rule ids, credential MODES, status classes; never header
# values, key material, environment variable VALUES, or body text (an
# unavailable hop names the unset variable, never its value).


@dataclass(frozen=True)
class RouteInbound:
    """What the core knows about one request when it asks for a plan.

    Built once per request BEFORE redaction (decision 4: the FIRST upstream's
    note setting governs the prepared body). ``headers`` is the inbound
    header mapping (case-insensitive). ``path`` is the decoded path rules
    match; ``raw_path`` is the percent-encoding-preserving path the core
    forwards; ``query`` the raw query string. ``model`` is the body's
    top-level ``model`` when it is a string, else None (a Gemini request
    carries its model in the path — the router derives it). The core never
    hands the router a request body: the redacted body reaches the plan at
    ``begin`` time only.
    """

    adapter_name: str | None
    provider_name: str
    method: str
    path: str
    raw_path: str
    query: str
    headers: Mapping[str, str]
    model: str | None


@dataclass(frozen=True)
class RouteRefusal:
    """A proxy-generated, provider-shaped refusal answered WITHOUT contacting
    any upstream. ``message`` goes through ``adapter.error_body(message,
    status=status)``; ``row`` is stored as the recent/events row's ``route``
    field and formatted into the log line — its keys are exactly
    ``rule``, ``upstream``, ``hops``, ``auth``, ``class``, ``reissue``;
    ``headers`` are stamped on the reply. Names upstreams/protocols only.
    """

    status: int
    message: str
    row: Mapping[str, Any]
    headers: Mapping[str, str] | None = None


@dataclass(frozen=True)
class HopRequest:
    """One attempt the core sends; everything about it was decided by the
    router. ``upstream`` is the upstream NAME (the log, metric and
    fault-counter label). ``headers`` are the complete outbound headers
    (credentials already applied — the core adds nothing); ``body`` the
    bytes to send. ``reissued_from`` names the upstream this hop replaces
    (the core counts ``llm_redact_reissues_total{from_upstream,to_upstream}``);
    a retry of the SAME upstream leaves it None. ``unavailable`` set means
    the router could not mint the hop (a credential variable vanished): the
    text names the VARIABLE only, the core sends nothing, logs it, counts a
    fault against ``upstream`` and reports ``HopResult.fault =
    "MissingCredential"`` so chain logic sees it like a transport fault.
    """

    upstream: str
    url: str
    headers: Sequence[tuple[str, str]]
    body: bytes
    reissued_from: str | None = None
    unavailable: str | None = None


@dataclass(frozen=True)
class HopResult:
    """What one attempt produced: a status with the upstream's response
    headers (body untouched — a body the core will stream has not been read;
    one it will buffer has, so a mid-body drop surfaces here as ``fault``),
    or a fault (``status`` None, ``fault`` = the exception TYPE name, never
    its message — an httpx message can embed the URL)."""

    upstream: str
    status: int | None
    headers: Mapping[str, str]
    fault: str | None


@dataclass(frozen=True)
class HopDecision:
    """The router's answer to a HopResult. ``next`` None = stop: deliver the
    response in hand, or a 502 when there is none. Otherwise the core closes
    the undelivered response, awaits ``RoutePlan.wait(wait_seconds)`` when
    positive, and sends ``next``. Retry-same vs re-issue is the router's
    own bookkeeping (``HopRequest.reissued_from``); the core does not
    distinguish them."""

    next: HopRequest | None
    wait_seconds: float = 0.0


@dataclass(frozen=True)
class LocalAnswer:
    """A request the router answers locally before the body is read (R-15
    model discovery). ``provider`` labels the recorded row; ``reason`` is the
    short token the core's log line quotes: ``METHOD PATH -> STATUS answered
    locally (reason)``."""

    status: int
    body: Mapping[str, Any]
    provider: str
    reason: str


class RouteDelivery(Protocol):
    """The delivering hop's outcome, threaded into the core's delivery
    branches and their finalizers.

    ``upstream`` / ``rule`` label ``llm_redact_routed_requests_total``
    (``rule`` None renders as ``-``). ``headers`` are merged into the client
    reply (x-llm-redact-*). ``observe_event`` / ``observe_line`` run on every
    delivered SSE event / NDJSON line AFTER the adapter's rehydration and
    BEFORE serialization (usage tracking, model-id restoration);
    ``observe_line`` returns the SAME object when unchanged. ``wants_payload``
    says whether a non-CHAT JSON body must be parsed at all (a large
    pass-through listing forwards untouched when False). ``observe_payload``
    runs on a buffered JSON body (CHAT after rehydration; REDACT_ONLY or
    pass-through when wanted) and returns whether it changed it in place.
    ``mark_failed`` records a fault class: ``"transport"`` (a buffered read
    failed before any byte reached the client) or ``"stream_error"`` (a fault
    after the first byte — never re-issued, R-17). ``row`` is the current
    route row (keys rule/upstream/hops/auth/class/reissue); ``finish`` closes
    the books (spend) exactly once and returns the final row.
    """

    upstream: str
    rule: str | None
    headers: Mapping[str, str]

    def observe_event(self, event: SSEEvent) -> SSEEvent: ...

    def observe_line(self, line: bytes) -> bytes: ...

    def wants_payload(self, kind: RouteKind) -> bool: ...

    def observe_payload(self, payload: Any, kind: RouteKind) -> bool: ...

    def mark_failed(self, status_class: str) -> None: ...

    def row(self) -> dict[str, Any]: ...

    def finish(self, status: int | None) -> dict[str, Any]: ...


class RoutePlan(Protocol):
    """One routed request's decision sequence. The core calls, in order:
    ``inject_system_note`` (read before redaction); ``local_refusal`` (after
    redaction, BEFORE the write-ahead audit START row — a refusal that never
    counts as an attempt: the count_tokens 404); ``begin`` (after the audit
    token, with the redacted body, its decoded form or None, and the
    forwardable inbound headers — hop-by-hop headers dropped,
    ``accept-encoding: identity`` added; returns the first hop, or a refusal
    that IS an attempt: the budget 402); then ``decide`` once per HopResult
    until ``next`` is None; then ``delivery`` once. ``deadline`` is a
    ``time.monotonic()`` value valid after ``begin``; the core derives each
    hop's httpx timeout from it. ``wait`` is the sleep the core awaits for a
    positive ``wait_seconds`` (injectable by the router's tests). Every
    decision is made on response HEADERS; the core never sends a byte to the
    client before ``decide`` returned ``next = None``.
    """

    inject_system_note: bool
    deadline: float

    def local_refusal(self) -> RouteRefusal | None: ...

    def begin(
        self,
        outbound: bytes,
        outbound_obj: Mapping[str, Any] | None,
        forward_headers: Sequence[tuple[str, str]],
    ) -> HopRequest | RouteRefusal: ...

    def decide(self, result: HopResult) -> HopDecision: ...

    async def wait(self, seconds: float) -> None: ...

    def delivery(self) -> RouteDelivery: ...


class Router(Protocol):
    """The routing layer the proxy holds when ``[routing] enabled = true``
    (``ProxyState.router`` is None otherwise — the unrouted path never
    consults it).

    ``local_answer`` is consulted after the disabled-provider / named-user
    gates and before the body is read. ``plan`` returns None for a request
    the layer does not route (a provider outside its protocols — the request
    then takes the unrouted path byte-for-byte), a RouteRefusal (no_route),
    or a RoutePlan. ``status`` is the ``/status`` ``routing`` block (metadata
    only). ``validate`` is the config editor's dry-run: the checks
    ``reconfigure`` would fail on, with no swap. ``reconfigure`` is the
    SIGHUP/editor hot path: validate the whole new config, then swap the
    router's own state in one block; it raises only ``ConfigError`` and,
    when it raises, has changed nothing. ``close`` runs once, at lifespan
    shutdown or when a reload drops the router.
    """

    def local_answer(
        self, method: str, path: str, headers: Mapping[str, str]
    ) -> LocalAnswer | None: ...

    def plan(self, inbound: RouteInbound) -> RoutePlan | RouteRefusal | None: ...

    def status(self) -> dict[str, Any]: ...

    def validate(self, config: Config) -> None: ...

    def reconfigure(self, config: Config) -> None: ...

    def close(self) -> None: ...


__all__ = [
    "HopDecision",
    "HopRequest",
    "HopResult",
    "LocalAnswer",
    "MAX_RESPONSE_ROWS",
    "RESPONSE_PRUNE_EVERY",
    "RouteDelivery",
    "RouteInbound",
    "RouteKind",
    "RoutePlan",
    "RouteRefusal",
    "Router",
    "SSEEvent",
    "SessionRouter",
    "Telemetry",
    "Vault",
    "VaultCipher",
    "VaultKeyError",
    "VaultManager",
]
