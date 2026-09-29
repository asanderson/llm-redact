"""Stable contract for out-of-tree plugins (the paid ``llm-redact-pro``
package registers against these names).

The open-core split (llm-redact-pro docs/licensing.md) lets a separately-distributed
package supply the paid vault/audit/session/telemetry/access implementations.
Those implementations must bind to a *supported* surface, not to private
internals that move between releases. This module is that surface: re-exports
promoted from the Free core with public names. Everything here is part of the
plugin API contract — change it deliberately, never incidentally.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
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
    import argparse
    from pathlib import Path

    from starlette.requests import HTTPConnection, Request
    from starlette.responses import Response

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

    OPTIONAL member, read via ``getattr`` so older routers keep working:
    ``is_durable(session_id) -> bool`` marks a session the live process must
    never prune (the TTL loop, ``POST /__llm-redact/sessions/prune``) because
    provider-side state still points into it — the router's equivalents of
    the configured static session, which the proxy always keeps. A pruned
    session is recreated on the next request with fresh placeholder numbers,
    so a token the provider still holds would then rehydrate to a NEWER
    value (never-wrong-value). Sessions that are merely idle stay prunable.
    Only an explicit ``False`` releases a session: any other answer, and an
    exception, keeps it, so a misbehaving router can only keep more.

    OPTIONAL ``record_object_id(object_id, session_id) -> bool | None``:
    the proxy reports the ids of objects the provider STORES for later reads
    (uploaded files, batches, message batches, stored conversations, the
    files a provider tool wrote for the request — see
    ``ProviderAdapter.object_ids_from_body``) with the session of the
    request whose answer names them (never an id that request's own body
    carries: an existing object the answer echoes), so a router can keep
    another user's later read of that object out of the creator's
    namespace. An answer to a request that READS an existing object (a
    batch's status, a fine-tuning job) names objects derived from it (the
    batch's output files, the job's result files): a router attributes
    those to the read object's creator. Reported in EVERY ``mode`` (static
    included: a router may serve unattributed traffic on the static path
    and still need to know what that shared session created). ``False``
    vetoes the durable mirror, as for response ids; a router without the
    member is never called.

    OPTIONAL ``object_access_refusal(adapter_name, method, path, body, *,
    identity) -> str | None``: asked once for EVERY forwarded HTTP request
    (matched or pass-through) after admission and the routing layer's
    ``plan`` (a plan is only ever begun after this answer) and before the
    audit START row, redaction, any upstream credential and any upstream
    contact — whatever the router's ``mode``. ``identity`` is True when the
    request reaches its provider with a credential the PROXY holds, so
    every client is one principal upstream and the request spends one it
    never presented: the provider is authorized with the proxy's own cloud
    identity (``[providers.NAME] auth = "identity"``), or the request is
    routed and its plan may send an operator key or no key at all
    (``RoutePlan.proxy_credential``). ``body`` is the parsed request body —
    None for a body that is neither JSON nor an upload, and for
    pass-through routes unless the request is sent with the proxy's
    credential: then a JSON body is parsed for this check alone (still
    forwarded byte-for-byte). A multipart/form-data upload is read for this
    check alone (``upload_view``) — on a matched route whatever the
    credential and whether or not it redacts (``detection``), on a
    pass-through route under the proxy's credential — and ``body`` is then
    a LIST of what it cites: each JSON-object line of its file parts (a
    batch input file's requests run later, with the credential the upload
    is sent with) and each form field as an object nested along its name
    (``file_ids[]`` → ``{"file_ids": [value]}``). Under the proxy's
    credential a body the check cannot read is refused before this is
    asked: content-encoded, more than one Content-Type, JSON repeating a
    key (a pass-through body; a matched one is re-serialized as checked),
    JSON beyond ``max_body_bytes``, or an upload outside the canonical
    grammar, with a transfer encoding, a form field that is not UTF-8
    text, or more JSON than ``max_body_bytes``. A string refuses the
    request with a recorded, provider-shaped 403 carrying exactly that
    text: a FIXED reason chosen by the router, never an object id, a user
    name or content. None forwards. An exception refuses too (fail closed;
    logged by exception type only). A router without the member is never
    asked, and one that keeps no ownership should return None at once.

    OPTIONAL ``sealed(session_id) -> bool | str``: asked right after
    ``resolve`` (so never in static mode, where nothing is resolved) with
    the session the request was resolved to. True — or a non-empty string,
    the fixed refusal reason the router chooses (never an id or content) —
    means that session must stay EMPTY: the proxy reads it for
    rehydration, but redacting anything into it is refused — an HTTP
    request with a value to redact gets a recorded, provider-shaped 403
    (with the router's reason, else the core's) before any upstream
    contact (nothing is written), a realtime connection is refused
    outright. A router uses it where it resolves a request to an empty
    session because what the request reads has another (or no confirmed)
    owner, or lives on provider-side where this proxy no longer knows it:
    whatever the request itself sent would otherwise share placeholder
    names with what it reads. An exception seals; a router without the
    member never seals.

    OPTIONAL ``listing_item_session(object_id) -> str | None``: for a 2xx
    listing of stored objects (``ProviderAdapter.lists_objects`` /
    ``listing_items`` — OpenAI-shaped ``{"object": "list", "data": [...]}``
    collections of files, batches, video jobs and stored chat completions),
    the vault session each listed item's placeholders should be restored
    in. The proxy rebuilds that item from the bytes the provider sent and
    rehydrates it, as a whole object, in the named session — only when
    that session exists and holds mappings (the proxy never creates one)
    and, with a vault that keeps a durable response map, only when that
    map still records the object in exactly that session (a session pruned
    and recreated since the object was created holds NEW values under the
    same token names). Otherwise the item keeps the provider's placeholders
    — name an empty session to keep an item OUT of the request's own
    session (a router that separates namespaces does so for every item it
    cannot vouch for: the request's own session may hold other values
    under the same token names). None leaves the item as the request's own
    session delivers it. An exception delivers the item exactly as the
    provider sent it: a router that cannot answer vouches for nothing. A
    listing never records ownership: nothing here reaches
    ``record_object_id``. OPTIONAL batched form
    ``listing_item_sessions(object_ids) -> Sequence[str | None]``, one
    answer per id in order, asked ONCE per listing instead when present (a
    router can then read its own records in one query); an exception or a
    miscounted answer delivers EVERY item exactly as the provider sent it.

    ``record_response_id`` MAY return ``False`` to veto the proxy's durable
    mirror of the mapping (the vault manager's response-session map): the
    router refused it (a response must never move to another namespace) or
    never reads it. ``None`` or ``True`` keep the proxy's historical
    behavior of mirroring every recorded mapping.
    """

    mode: str

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str: ...

    def record_response_id(self, response_id: str, session_id: str) -> bool | None: ...


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
    client before ``decide`` returned ``next = None``. A plan is created
    before the stored-object check and redaction, and may be abandoned
    before ``begin`` (a refusal) without the router being told.

    OPTIONAL ``proxy_credential: bool``, read via ``getattr``: whether any
    upstream this plan may send the request to — the first, or a fallback
    chain member — attaches a credential the PROXY holds (an operator key)
    or none at all, instead of forwarding the client's own: every client is
    then one principal upstream, so the session router's
    ``object_access_refusal`` is asked with ``identity=True``. A plan
    without the member counts as True (fail closed); only an explicit
    False says the client's own credential is what the provider sees.
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

    ``local_answer`` is consulted after the disabled-provider / access
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


# --- browser dashboard seam ---------------------------------------------------
# The browser ops UI (the dashboard page, its config editor and redaction
# preview) is a paid surface implemented in llm-redact-pro. The core keeps
# the machine APIs every CLI and monitor depends on (/status, /metrics,
# /healthz, /readyz, /recent, /events, /sessions, /audit, /guide)
# and dispatches ONLY its fixed dashboard paths (the bare prefix, /config,
# /preview) to a registered Dashboard — a plugin can never shadow a core
# endpoint. The core stamps the security headers on whatever it returns.


class DashboardHost(Protocol):
    """What the core exposes to a plugin-served dashboard (``ProxyState``
    satisfies it structurally).

    ``config`` is the live effective config (env overrides applied);
    ``csrf_token`` the per-process token the guarded POSTs require in the
    ``x-llm-redact-csrf`` header (the dashboard hands it to same-origin
    pages). ``config_file_path`` is the file an edit is written to.
    ``host_allowed`` / ``origin_allowed`` are the DNS-rebinding and Origin
    checks; ``guarded_post_json`` runs the POST guard chain (CSRF header,
    content type, 1 MiB cap, JSON parse) and returns ``(payload, None)`` or
    ``(None, refusal)``. ``validate_config`` is the dry run of everything
    ``apply_config`` builds (detectors, allowlists, modes, license, routing
    credentials, the router's own checks) — it raises ``ValueError`` /
    ``TypeError`` / ``re.error`` / ``ImportError`` and changes nothing.
    ``apply_config`` is the SIGHUP hot-apply path and returns the
    restart-required section names. ``preview`` runs the LIVE detection
    pipeline over caller text on a throwaway vault (no upstream, vault,
    metrics or audit write) and returns ``{redacted, detections, warnings,
    blocked}`` — ``blocked`` is ``{"type": T}`` or None, never a value.
    """

    config: Config
    csrf_token: str

    def config_file_path(self) -> Path: ...

    def host_allowed(self, request: Request) -> bool: ...

    def origin_allowed(self, request: Request) -> bool: ...

    async def guarded_post_json(
        self, request: Request
    ) -> tuple[Any, None] | tuple[None, Response]: ...

    def validate_config(self, candidate: Config) -> None: ...

    def apply_config(self, fresh: Config) -> list[str]: ...

    def preview(self, text: str) -> dict[str, Any]: ...


class Dashboard(Protocol):
    """The browser dashboard a plugin serves (``Registry.build_dashboard``;
    ``ProxyState.dashboard`` is None without one, and the core answers the
    dashboard paths with a 404 naming the package). ``handle`` receives
    only requests for the core's fixed dashboard paths and returns the
    complete reply; it must never forward anything upstream."""

    async def handle(self, request: Request, host: DashboardHost) -> Response: ...


# --- access seam ---------------------------------------------------------------
# Who may use the proxy (client authentication, named users, seats) is a
# paid subsystem implemented in llm-redact-pro; the core holds no
# credential logic at all. It asks a registered AccessGate to ADMIT each
# request before routing, applies the verdict at its fixed place in the
# request path, attributes the request to the admitted subject, and
# dispatches the gate's fixed admin paths to it. Without a gate the core
# serves the implicit single local user and guards two leak paths itself
# (every x-llm-redact-* request header is dropped before forwarding; an
# unclaimed /u/... path is answered locally, never forwarded).


@dataclass(frozen=True)
class Admission:
    """A gate's verdict on one connection.

    ``subject`` is attributed as the request's user (recent, events and
    audit rows); ``refusal``, when set, makes the core refuse the request
    — a provider-shaped 403 on HTTP, an accept-then-close on WebSocket —
    with that message. Messages must never echo a presented credential.
    """

    subject: str | None = None
    refusal: str | None = None
    # Dashboard surface only: a same-proxy path (it must start with
    # ``/__llm-redact/``) a refused BROWSER GET is sent to with a 303, e.g.
    # the gate's sign-in page. Ignored on every other surface and method,
    # and any other value is ignored (never an open redirect).
    redirect: str | None = None


class AccessGate(Protocol):
    """Client admission (``Registry.build_access_gate``).

    ``admit`` runs before routing on every HTTP request (surface ``"http"``)
    and WebSocket upgrade (``"websocket"``) outside the reserved prefix, and
    MUST remove every credential it recognizes from the connection's ASGI
    scope (path, raw path, headers) before returning: whatever it leaves is
    routed, logged and forwarded. It may return the ``Admission`` directly
    or an awaitable of one (a method that must reach a directory or an
    identity provider awaits instead of blocking the event loop).

    ``status`` is the ``users`` block of ``/status`` (metadata only).
    ``handle`` answers the core's fixed gate paths — ``ACCESS_PATHS`` (the
    admin endpoints), everything under ``AUTH_PREFIX`` (browser sign-in:
    ``AUTH_PATHS`` login, callback and sign-out, plus any page or JSON
    endpoint a sign-in method adds below the prefix, all Host- and
    Origin-checked by the core and never behind dashboard admission) and
    everything under ``SCIM_PREFIX`` — and must never forward anything
    upstream; the core stamps the security headers on its reply. ``close``
    runs at shutdown.

    Three OPTIONAL members, read with ``getattr`` so a gate written before
    them keeps working unchanged:

    - ``guards_dashboard: bool`` — when true, the core also calls
      ``admit(conn, "dashboard")`` for every reserved path except the
      monitoring probes (healthz, readyz, metrics) and the gate paths
      themselves, and refuses with a 403 — or, for a browser GET, a 303 to
      ``Admission.redirect`` — before answering it.
    - ``public_origin() -> str | None`` — the ``scheme://host[:port]`` the
      proxy is reached at from other machines (read once at startup). Its
      host joins the loopback names the reserved endpoints' DNS-rebinding
      check accepts, and exactly that origin passes the Origin check. Only
      meaningful together with ``guards_dashboard``: the core ignores it
      otherwise, so a wider Host is never accepted unauthenticated.
    - ``bind_sessions(store: SessionStore) -> None`` — called once at
      startup with the live vault's sessions, so a gate can drop the
      sessions of a user it deletes (whole sessions, through the running
      proxy's own vault manager, never a second connection).
    """

    def admit(self, conn: HTTPConnection, surface: str) -> Admission | Awaitable[Admission]: ...

    def status(self) -> dict[str, Any]: ...

    async def handle(self, request: Request, host: DashboardHost) -> Response: ...

    def close(self) -> None: ...


class SessionStore(Protocol):
    """The live vault's sessions, as handed to ``AccessGate.bind_sessions``.

    ``session_ids`` lists the sessions holding mappings. ``forget`` deletes
    whole named sessions — their mappings and response-id rows — through
    the proxy's own vault manager (so cached views are dropped too) and
    returns how many held mappings; the configured static session is never
    deleted. Whole sessions only: deleting part of one would let the next
    allocation reissue a still-referenced placeholder number for a
    different value.
    """

    def session_ids(self) -> list[str]: ...

    def forget(self, session_ids: Iterable[str]) -> int: ...


# --- upstream authorization seam ----------------------------------------------
# [providers.NAME] auth = "identity": the proxy authorizes requests to a cloud
# provider with its OWN workload identity (AWS SigV4 for Bedrock, OAuth bearer
# tokens for Vertex AI and Azure OpenAI). Credential fetching and signing are
# paid code in llm-redact-pro; the core only strips the client's credential
# channels, hands the plugin the FINAL outbound request (after redaction and
# note injection) and sends exactly the headers it returns with exactly the
# bytes it saw. Client-side SigV4 stays unsupported: a signature the CLIENT
# computed covers the unredacted body, which the proxy rewrites.


class UpstreamAuthError(Exception):
    """An ``UpstreamAuth`` could not authorize a request (no credential, a
    token endpoint refused). The message names the credential SOURCE kind
    only (for instance "AWS credential chain") — never a secret, token,
    signature or response body — because the core logs it and puts it in
    the client's provider-shaped 502."""


class UpstreamAuth(Protocol):
    """One provider's upstream authorizer (``Registry.build_upstream_auth``).

    ``authorize`` receives the request exactly as the core will send it —
    ``url`` in the form httpx puts on the wire (query included), ``headers``
    with every client credential channel already removed, ``body`` the final
    bytes — and returns the complete outbound header list (it may add
    ``host``, date and signature headers, and must not change the body or
    URL). It raises ``UpstreamAuthError`` when no credential is available;
    the core then answers a recorded 502 and forwards nothing. It must never
    block the event loop on network I/O. ``close`` runs when a reload
    displaces the authorizer or at shutdown.
    """

    async def authorize(
        self, method: str, url: str, headers: list[tuple[str, str]], body: bytes
    ) -> list[tuple[str, str]]: ...

    def close(self) -> None: ...


# --- vault database credential seam ---------------------------------------------
# ``Registry.build_db_password(vault_config)`` returns one of these (or None
# for the static password). The RDBMS vault store calls it synchronously at
# EVERY connect and reconnect, on the thread running the store — cloud
# database tokens (RDS IAM, Cloud SQL IAM, Entra ID) expire in minutes, so a
# password captured once would break the first reconnect after expiry. It
# returns the password; it raises (naming the credential SOURCE, never a
# secret) when none can be had, which fails the connect closed.
DbPasswordProvider = Callable[[], str]


# --- CLI seam -------------------------------------------------------------------
# Paid command-line subcommands register here instead of living in the core
# parser: the core adds each registered command's subparser, dispatches to
# its ``run``, and merges its completion words into the shell completions.


class CliCommand(Protocol):
    """One ``llm-redact <name>`` subcommand supplied by a plugin.

    ``completion`` is ``(subcommands, options)`` for the shell completion
    scripts; ``run`` returns the process exit code.
    """

    name: str
    help: str
    completion: tuple[tuple[str, ...], tuple[str, ...]]

    def add_arguments(self, parser: argparse.ArgumentParser) -> None: ...

    def run(self, args: argparse.Namespace) -> int: ...


# --- config seam ----------------------------------------------------------------
# A plugin's own top-level config tables: the core parses only the sections it
# knows and rejects every other key, so a plugin feature's settings (for
# instance llm-redact-pro's [auth]) are claimed here instead of being shaped
# in the core. Without the plugin such a section stays an unknown key — a
# startup ConfigError, never silently ignored.


class ConfigSection(Protocol):
    """One top-level ``[name]`` table owned by a plugin.

    ``parse`` validates the raw TOML table and returns an immutable value
    that supports ``==`` (a frozen dataclass); it raises
    ``llm_redact.config.ConfigError`` on a bad value, never echoing a
    secret. The value is stored at ``Config.extensions[name]`` only when the
    section is present in the file. ``emit`` turns that value back into a
    TOML-shaped mapping (scalars, lists, nested tables, lists of tables)
    for ``config show`` and the config editor's rewrite, so the section
    round-trips. Plugin sections are restart-only: a reload that changes
    one keeps the running value and reports ``name`` as restart-required.
    """

    name: str

    def parse(self, raw: Any, where: str) -> Any: ...

    def emit(self, value: Any) -> Mapping[str, Any]: ...


__all__ = [
    "AccessGate",
    "Admission",
    "CliCommand",
    "ConfigSection",
    "Dashboard",
    "DashboardHost",
    "DbPasswordProvider",
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
    "UpstreamAuth",
    "UpstreamAuthError",
    "Vault",
    "VaultCipher",
    "VaultKeyError",
    "VaultManager",
]
