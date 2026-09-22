"""Rule-based upstream routing: schema, pure decision logic, in-process state.

This module is the routing layer's single source of truth for its data
shapes (``UpstreamConfig``, ``RouteRule``, ``RoutingConfig``, ``ModelPrice``)
and for every decision that does not need the network: which rule a request
matches, how an inbound credential is classified, which fallback chain a
status resolves to, what the outbound headers/body/URL for an upstream are,
and how a model id is restored on the way back. ``config.py`` parses and
validates the TOML into these dataclasses; ``proxy.py`` drives the hop loop
over these functions; ``RoutingState`` holds the per-upstream cooldown and
counters that survive SIGHUP but not the process (docs/routing.md).

Load-bearing constraints:

- Nothing here logs or raises with a header value, key material, or body
  text. ``MissingCredential`` names the env VAR, never its value.
- Pure functions never mutate their inputs (``apply_body_rewrites`` returns
  a fresh dict or None; ``restore_model`` is the one documented in-place
  helper because it runs on an already-decoded event payload).
- Stdlib only: this module sits on the request hot path and the runtime
  dependency set stays httpx/starlette/uvicorn.
"""

import json
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from fnmatch import fnmatchcase
from typing import Any
from urllib.parse import quote, unquote, urlsplit

# The four wire formats a rule can match. Every other adapter/provider (azure,
# vertex, bedrock, cohere, custom:*) keeps the legacy one-upstream-per-provider
# path untouched.
PROTOCOLS: tuple[str, ...] = ("anthropic", "openai", "gemini", "ollama")

# The anthropic-beta capability token Claude Code sends on subscription (OAuth)
# traffic; its presence together with a Bearer authorization classifies the
# request as `auth = "oauth"` (R-6). Overridable via routing.oauth_beta_marker.
OAUTH_BETA_MARKER = "oauth-2025-04-20"

# Anthropic's unified rate-limit response headers whose listed values mark an
# exhausted subscription window (plan-limit) rather than a throttle (R-20).
# A config override replaces this table entirely.
DEFAULT_PLAN_LIMIT_HEADERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic-ratelimit-unified-status", ("rejected",)),
    ("anthropic-ratelimit-unified-5h-status", ("rejected",)),
    ("anthropic-ratelimit-unified-7d-status", ("rejected",)),
)

# Inbound headers that carry the client's own credential. They are forwarded
# byte-exact to a passthrough upstream and DROPPED for none/env upstreams
# (I-3: a server-held key never travels beside the client's).
CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key"})
# Gemini's header credential joins the drop set for none/env upstreams: the
# client's key must never reach an upstream the proxy authenticates itself.
_INBOUND_CREDENTIAL_HEADERS = CREDENTIAL_HEADERS | {"x-goog-api-key"}
# Gemini clients may also authenticate with `?key=` (CLAUDE.md: the query
# passes through the proxy). The same I-3 rule applies to that form: the
# parameter is dropped from the URL sent to a none/env upstream.
_CREDENTIAL_QUERY_PARAMS = frozenset({"key"})

UPSTREAM_HEADER = "x-llm-redact-upstream"
HOPS_HEADER = "x-llm-redact-hops"
REISSUE_HEADER = "x-llm-redact-reissue"

SPECIAL_STATUS_KEYS = ("plan_limit_429", "throttle_429")
CLASS_KEYS = ("4xx", "5xx")
RETRY_SAME = "retry-same"
MAX_COOLDOWN_SECONDS = 3600.0
AUTH_KINDS = ("oauth", "gateway-key", "none", "any")
CREDENTIAL_MODES = ("passthrough", "none", "env")
COSTS = ("metered", "zero")
REISSUE_POLICIES = ("stateless-only", "always", "never")
PLAN_LIMIT_DETECTION = ("headers", "off")

_GLOB_CHARS = frozenset("*?[")
_REISSUE_WINDOW_SECONDS = 3600.0


class MissingCredential(ValueError):
    """An `env:VAR` upstream whose variable is unset or empty at send time.

    The message names the VARIABLE only — never a value — so it is safe to
    log and to surface in a provider-shaped error body.
    """


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens. The single definition; pricing.py imports it."""

    input: float
    output: float
    cache_read: float
    cache_write: float


@dataclass(frozen=True)
class UpstreamConfig:
    name: str
    protocol: str  # one of PROTOCOLS
    base_url: str  # rstripped "/"
    credential: str = "passthrough"  # "passthrough" | "none" | "env:VAR"
    cost: str = "metered"  # "metered" | "zero"
    # Resolved at parse: False for passthrough (I-4), else the top-level
    # inject_system_note. The note decision is made once, for the FIRST
    # upstream of a request (the redacted body is reused on later hops).
    inject_system_note: bool = False
    # False for upstreams without /v1/messages/count_tokens (Ollama hangs on
    # it): the proxy answers 404 locally, no fallback (decision 7).
    count_tokens: bool = True
    monthly_budget_usd: float | None = None
    monthly_budget_tokens: int | None = None
    cooldown_seconds: float = 60.0
    extra_headers: tuple[tuple[str, str], ...] = ()  # (lowercased name, value), sorted
    body_defaults_json: str = "{}"  # canonical json.dumps(sort_keys=True) of the table
    # Auto-registered from [providers.NAME] when a [routing] table exists
    # (R-1); never emitted by the config writer — it re-registers.
    legacy: bool = False

    @property
    def credential_mode(self) -> str:
        if self.credential.startswith("env:"):
            return "env"
        return self.credential

    @property
    def env_var(self) -> str | None:
        if self.credential.startswith("env:"):
            return self.credential[len("env:") :]
        return None

    @property
    def is_passthrough(self) -> bool:
        return self.credential == "passthrough"

    @property
    def zero_cost(self) -> bool:
        return self.cost == "zero"

    @property
    def has_budget(self) -> bool:
        # Zero-cost upstreams ignore budgets even when one is configured
        # (parse rejects the combination on passthrough; zero-cost + budget
        # is allowed but inert).
        if self.zero_cost:
            return False
        return self.monthly_budget_usd is not None or self.monthly_budget_tokens is not None

    def body_defaults(self) -> dict[str, Any]:
        # A fresh object per call: callers merge it into request bodies and
        # must never share nested containers between requests.
        loaded = json.loads(self.body_defaults_json)
        return dict(loaded) if isinstance(loaded, dict) else {}


@dataclass(frozen=True)
class RuleMatch:
    protocol: str
    models: tuple[str, ...] = ()  # fnmatchcase globs; () = any
    headers: tuple[tuple[str, str], ...] = ()  # (lowercased name, glob) all must match
    path: str | None = None  # glob on the inbound path
    auth: str = "any"


@dataclass(frozen=True)
class RouteRule:
    id: str
    match: RuleMatch
    upstream: str
    model_rewrite: str | None = None
    # key -> chain | RETRY_SAME, in file order. Keys: an exact status
    # ("529"), a class ("4xx"/"5xx"), or a special key (SPECIAL_STATUS_KEYS).
    on_status: tuple[tuple[str, tuple[str, ...] | str], ...] = ()
    reissue_policy: str = "stateless-only"  # resolved default per R-8
    on_budget_exhausted: tuple[str, ...] = ()

    def resolve(self, status_key: str) -> tuple[str, tuple[str, ...] | str] | None:
        """The on_status entry a status key resolves through: the KEY it
        matched (the class the proxy's log line and route row report) and
        its chain (or RETRY_SAME), or None when the rule lists nothing for
        this status.

        Precedence: special key > exact status > class key. A special key
        ("plan_limit_429") falls back to "429" and then "4xx"; a transport
        fault is classified "502" by the proxy, so it matches "502"/"5xx".
        This is the ONE precedence ladder — chain_for and the proxy's hop
        loop both go through it, so the two can never drift.
        """
        candidates: tuple[str, ...]
        if status_key in SPECIAL_STATUS_KEYS:
            candidates = (status_key, "429", "4xx")
        elif status_key.isdigit():
            candidates = (status_key, f"{status_key[0]}xx")
        else:
            candidates = (status_key,)
        table = dict(self.on_status)
        for key in candidates:
            if key in table:
                return key, table[key]
        return None

    def chain_for(self, status_key: str) -> tuple[str, ...] | str | None:
        """The chain (or RETRY_SAME) for a status key, or None (`resolve`
        without the matched key)."""
        resolved = self.resolve(status_key)
        return None if resolved is None else resolved[1]


@dataclass(frozen=True)
class PricesConfig:
    table: str = "builtin"  # or a file path
    overrides: tuple[tuple[str, ModelPrice], ...] = ()  # sorted by model id


@dataclass(frozen=True)
class RoutingConfig:
    enabled: bool = False
    default_upstreams: tuple[tuple[str, str], ...] = ()  # (protocol, upstream) sorted
    max_hops: int = 3
    request_deadline_seconds: float = 600.0
    plan_limit_detection: str = "headers"  # "headers" | "off"
    plan_limit_headers: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_PLAN_LIMIT_HEADERS
    oauth_beta_marker: str = OAUTH_BETA_MARKER
    throttle_retry_max_seconds: float = 30.0
    budget_reset_day: int = 1
    debug_headers: bool = False
    expose_models: bool = False
    model_catalog: tuple[str, ...] = ()
    upstreams: tuple[UpstreamConfig, ...] = ()  # sorted by name
    rules: tuple[RouteRule, ...] = ()  # file order
    warnings: tuple[str, ...] = ()  # I-6 etc., surfaced not raised
    # True iff the TOML had a [routing] table (legacy upstreams register only
    # then; `enabled = false` is still "present").
    present: bool = False

    def upstream(self, name: str) -> UpstreamConfig:
        for upstream in self.upstreams:
            if upstream.name == name:
                return upstream
        raise KeyError(name)

    def upstream_names(self) -> tuple[str, ...]:
        return tuple(upstream.name for upstream in self.upstreams)

    def default_for(self, protocol: str) -> str | None:
        for proto, name in self.default_upstreams:
            if proto == protocol:
                return name
        return None


# The Gemini model id lives in the request PATH (decision 15b):
# /v1beta/models/{model}:generateContent. Group 2 is the id, matched on the
# raw (still percent-encoded) path so a rewrite leaves the rest byte-exact.
_GEMINI_MODEL_SEGMENT = re.compile(
    r"^(/(?:v1|v1beta)/(?:models|tunedModels)/)([^/:]+)(:[A-Za-z]+)$"
)


def gemini_path_model(path: str) -> str | None:
    """The model id between `/models/` and the `:verb` of a Gemini path, or
    None when the path has no such segment (a Vertex publisher path yields
    None too — Vertex is never routed, decision 1). THE derivation the proxy
    plans with and `routes test` dry-runs against."""
    match = _GEMINI_MODEL_SEGMENT.match(path)
    return match.group(2) if match is not None else None


def rewrite_gemini_path(raw_path: str, model: str) -> str:
    """The same raw path with its model segment replaced by `model`
    (percent-encoded); a path without the segment comes back unchanged."""
    match = _GEMINI_MODEL_SEGMENT.match(raw_path)
    if match is None:
        return raw_path
    return match.group(1) + quote(model, safe="") + match.group(3)


def is_count_tokens_path(protocol: str | None, path: str) -> bool:
    """Decision 7's gate, shared by the proxy (404 + chain-member skip) and
    `routes test`'s note: ONLY Anthropic's canonical count_tokens endpoint.
    Any other path — an OpenAI-protocol `/v1/x/count_tokens`, a non-canonical
    spelling — is forwarded like any request, so the dry-run must not claim
    a 404 the proxy would never answer."""
    return protocol == "anthropic" and path == "/v1/messages/count_tokens"


def request_protocol(adapter_name: str | None, inferred_provider: str) -> str | None:
    """The routing protocol of a request, or None for the legacy path.

    A matched adapter decides by its family name; pass-through traffic (no
    adapter) takes the proxy's inferred provider when that is itself one of
    the four protocols. Anything else (azure, vertex, bedrock, cohere,
    custom:*) is never routed.
    """
    if adapter_name is not None:
        return adapter_name if adapter_name in PROTOCOLS else None
    return inferred_provider if inferred_provider in PROTOCOLS else None


def _lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {name.lower(): value for name, value in headers.items()}


def beta_has_marker(anthropic_beta: str | None, marker: str) -> bool:
    """True when the comma-separated anthropic-beta list carries `marker`
    (whitespace-trimmed, case-insensitive)."""
    if not anthropic_beta:
        return False
    wanted = marker.strip().lower()
    return any(token.strip().lower() == wanted for token in anthropic_beta.split(","))


def strip_beta_marker(anthropic_beta: str, marker: str) -> str | None:
    """anthropic-beta without `marker`; None when nothing would remain.

    Other flags keep their own spelling and order; only the separator
    spacing between survivors is normalized to ", ".
    """
    wanted = marker.strip().lower()
    kept = [
        token.strip()
        for token in anthropic_beta.split(",")
        if token.strip() and token.strip().lower() != wanted
    ]
    return ", ".join(kept) if kept else None


def classify_auth(headers: Mapping[str, str], oauth_marker: str) -> str:
    """R-6: "oauth" | "gateway-key" | "none" for an inbound request.

    oauth = a Bearer authorization AND the OAuth capability in anthropic-beta;
    gateway-key = any non-empty authorization or x-api-key without that pair;
    none = neither credential header. Values are inspected, never returned.
    """
    lowered = _lower_headers(headers)
    authorization = lowered.get("authorization", "").strip()
    api_key = lowered.get("x-api-key", "").strip()
    is_bearer = authorization[:7].lower() == "bearer " and authorization[7:].strip() != ""
    if is_bearer and beta_has_marker(lowered.get("anthropic-beta"), oauth_marker):
        return "oauth"
    if authorization or api_key:
        return "gateway-key"
    return "none"


def _match_headers(
    wanted: tuple[tuple[str, str], ...], lowered: dict[str, str] | None, headers: Mapping[str, str]
) -> tuple[bool, dict[str, str] | None]:
    # The inbound mapping is lowercased at most once per select_rule call
    # and only when some rule constrains headers (hot path).
    if lowered is None:
        lowered = _lower_headers(headers)
    for name, glob in wanted:
        value = lowered.get(name)
        if value is None or not fnmatchcase(value, glob):
            return False, lowered
    return True, lowered


def select_rule(
    config: RoutingConfig,
    *,
    protocol: str,
    model: str | None,
    headers: Mapping[str, str],
    path: str,
    auth: str,
) -> RouteRule | None:
    """R-10: the first rule (file order) whose every match field holds.

    Globs are fnmatchcase (model, path, header values); header names compare
    case-insensitively. A rule with `auth = "any"` accepts every request;
    passing `auth="any"` (the `routes test` default — classify_auth never
    returns it) likewise matches every rule's auth constraint.
    """
    lowered: dict[str, str] | None = None
    for rule in config.rules:
        match = rule.match
        if match.protocol != protocol:
            continue
        if match.auth != "any" and auth != "any" and match.auth != auth:
            continue
        if match.models and (
            model is None or not any(fnmatchcase(model, glob) for glob in match.models)
        ):
            continue
        if match.path is not None and not fnmatchcase(path, match.path):
            continue
        if match.headers:
            ok, lowered = _match_headers(match.headers, lowered, headers)
            if not ok:
                continue
        return rule
    return None


def literal_models(config: RoutingConfig) -> list[str]:
    """The /v1/models list (R-15): `model_catalog` followed by every glob-free
    model name a rule matches, file order, deduplicated."""
    seen: dict[str, None] = dict.fromkeys(config.model_catalog)
    for rule in config.rules:
        for glob in rule.match.models:
            if not _GLOB_CHARS.intersection(glob):
                seen.setdefault(glob, None)
    return list(seen)


def is_plan_limit(
    response_headers: Mapping[str, str], table: tuple[tuple[str, tuple[str, ...]], ...]
) -> bool:
    """True when ANY header in `table` carries one of its listed values
    (case-insensitive, whitespace-trimmed)."""
    lowered = _lower_headers(response_headers)
    for name, values in table:
        value = lowered.get(name)
        if value is not None and value.strip().lower() in {v.strip().lower() for v in values}:
            return True
    return False


def status_key(
    status: int,
    *,
    protocol: str,
    response_headers: Mapping[str, str],
    config: RoutingConfig,
) -> str:
    """R-20: the on_status lookup key for an upstream response.

    Only an anthropic-protocol 429 with header detection on splits into
    "plan_limit_429" / "throttle_429"; every other status is its own digits.
    """
    if status == 429 and protocol == "anthropic" and config.plan_limit_detection == "headers":
        if is_plan_limit(response_headers, config.plan_limit_headers):
            return "plan_limit_429"
        return "throttle_429"
    return str(status)


def is_stateful_request(parsed_body: Any) -> bool:
    """R-22: an Anthropic-format body whose ASSISTANT turns carry a
    `thinking`/`redacted_thinking` block (signed to the originating
    model/credential). String content, user-turn blocks, and cache_control
    breakpoints are not stateful."""
    if not isinstance(parsed_body, dict):
        return False
    messages = parsed_body.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"):
                return True
    return False


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """Seconds from a Retry-After header (delay-seconds or HTTP-date); None
    when absent or unparsable. Never negative."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.isdigit():
        return float(int(text))
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        # "-0000" parses naive; it means UTC per RFC 5322.
        when = when.replace(tzinfo=UTC)
    current = time.time() if now is None else now
    return max(0.0, when.timestamp() - current)


def _credential_header(upstream: UpstreamConfig, environ: Mapping[str, str]) -> tuple[str, str]:
    var = upstream.env_var
    assert var is not None  # callers check credential_mode == "env"
    value = environ.get(var, "")
    if not value:
        raise MissingCredential(
            f"upstream {upstream.name!r}: environment variable {var} is unset or empty"
        )
    if upstream.protocol == "anthropic":
        return ("x-api-key", value)
    if upstream.protocol == "gemini":
        return ("x-goog-api-key", value)
    return ("authorization", f"Bearer {value}")


def outbound_headers(
    inbound: Sequence[tuple[str, str]],
    upstream: UpstreamConfig,
    *,
    environ: Mapping[str, str],
    oauth_marker: str,
) -> list[tuple[str, str]]:
    """The header list to send to `upstream` (R-13/R-14, I-3).

    passthrough: identical to `inbound` (the caller already dropped
    hop-by-hop headers). none/env: the client's credential headers are
    dropped, the OAuth capability is removed from anthropic-beta (the header
    is dropped when nothing remains), `extra_headers` are appended — each one
    REPLACING an inbound header of the same name (two X-Title values would be
    ambiguous upstream; the configured one is the operator's intent, like the
    credential header) — and an env upstream gains its provider's credential
    header from `environ`, raising MissingCredential (naming the VAR only)
    when it is unset.
    """
    if upstream.is_passthrough:
        return list(inbound)
    overridden = {name for name, _ in upstream.extra_headers}
    out: list[tuple[str, str]] = []
    for name, value in inbound:
        lowered = name.lower()
        if lowered in _INBOUND_CREDENTIAL_HEADERS or lowered in overridden:
            continue
        if lowered == "anthropic-beta":
            stripped = strip_beta_marker(value, oauth_marker)
            if stripped is None:
                continue
            value = stripped
        out.append((name, value))
    out.extend(upstream.extra_headers)
    if upstream.credential_mode == "env":
        out.append(_credential_header(upstream, environ))
    return out


def _strip_credential_query(query: str) -> str:
    # Parameter names compare after percent-decoding; everything else in the
    # query (order, encoding, repeated params) is kept byte-for-byte.
    kept = [
        part
        for part in query.split("&")
        if part and unquote(part.split("=", 1)[0]) not in _CREDENTIAL_QUERY_PARAMS
    ]
    return "&".join(kept)


def upstream_url(upstream: UpstreamConfig, path: str, query: str) -> str:
    """base_url + path (+ ?query).

    An openai-protocol base_url is the value a client would put in
    OPENAI_BASE_URL, so whenever it already carries a path (`/v1`,
    `/api/v1`, `/openai/v1`, `/v1beta/openai`) the request path's leading
    `/v1` is absorbed: https://api.openai.com/v1 + /v1/chat/completions ->
    .../v1/chat/completions, and Google's OpenAI-compatible
    .../v1beta/openai + /v1/chat/completions -> .../v1beta/openai/chat/completions
    (the same re-anchoring providers/custom.py does for Groq/OpenRouter
    bases). A host-only base (http://127.0.0.1:11434) takes the full path.
    Other protocols never fold (an anthropic base ending in /v1 stays as
    written). The query is appended verbatim for a passthrough upstream; for
    none/env upstreams the `key` credential parameter is dropped (I-3, the
    query twin of the x-goog-api-key header) and the rest stays byte-exact.
    """
    base = upstream.base_url.rstrip("/")
    if (
        upstream.protocol == "openai"
        and urlsplit(base).path.strip("/")
        and (path == "/v1" or path.startswith("/v1/"))
    ):
        path = path[len("/v1") :]
    url = base + path
    if query and not upstream.is_passthrough:
        query = _strip_credential_query(query)
    if query:
        url = f"{url}?{query}"
    return url


def apply_body_rewrites(
    body: Mapping[str, Any], *, upstream: UpstreamConfig, rule: RouteRule
) -> dict[str, Any] | None:
    """The forwarded body for this hop, or None when nothing would change.

    `model_rewrite` applies on every upstream; `body_defaults` (absent keys
    only, top-level granularity) and OpenAI `stream_options.include_usage`
    (stream = true) apply only on none/env upstreams — a passthrough body is
    never touched (R-12/R-24). The input is never mutated.

    `include_usage` is a CHAT COMPLETIONS option (R-24's `prompt_tokens`
    final-chunk rationale): the body must carry `messages`. A Responses body
    (`input`, the Codex CLI shape) already reports usage in
    `response.completed` and OpenAI rejects unknown `stream_options` keys
    with a 400 that no chain would recover from, so it is left alone.
    """
    out: dict[str, Any] | None = None
    if rule.model_rewrite is not None and body.get("model") != rule.model_rewrite:
        out = dict(body)
        out["model"] = rule.model_rewrite
    if upstream.is_passthrough:
        return out
    for key, value in upstream.body_defaults().items():
        if key not in body:
            if out is None:
                out = dict(body)
            out[key] = value
    if upstream.protocol == "openai" and body.get("stream") is True and "messages" in body:
        options = body.get("stream_options")
        if not isinstance(options, dict):
            if out is None:
                out = dict(body)
            out["stream_options"] = {"include_usage": True}
        elif "include_usage" not in options:
            if out is None:
                out = dict(body)
            out["stream_options"] = {**options, "include_usage": True}
    return out


def restore_model(payload: Any, original_model: str) -> bool:
    """R-9, in place on a decoded payload: top-level `model`, `message.model`
    (Anthropic message_start) and `response.model` (OpenAI Responses
    response.created/completed events — the Codex CLI validates it) become
    `original_model`. Returns whether anything changed. Non-dict payloads
    are left alone."""
    if not isinstance(payload, dict):
        return False
    changed = False
    if isinstance(payload.get("model"), str) and payload["model"] != original_model:
        payload["model"] = original_model
        changed = True
    for envelope in ("message", "response"):
        inner = payload.get(envelope)
        if (
            isinstance(inner, dict)
            and isinstance(inner.get("model"), str)
            and inner["model"] != original_model
        ):
            inner["model"] = original_model
            changed = True
    return changed


def restore_model_in_json_text(text: str, original_model: str) -> str:
    """restore_model over a serialized event/line; unparsable or unchanged
    text comes back byte-identical (never guess at a partial payload)."""
    try:
        payload = json.loads(text)
    except ValueError:
        return text
    if not restore_model(payload, original_model):
        return text
    return json.dumps(payload, ensure_ascii=False)


class RoutingState:
    """Per-upstream cooldown/counter state. In-process only: it survives a
    SIGHUP reload (`prune_to` drops names the new config no longer has) but
    not the process. Clocks are injectable for tests: `clock` (monotonic
    seconds) drives cooldowns and the reissue window; `wall_clock` (epoch
    seconds) only stamps `last_error_at`."""

    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        *,
        wall_clock: Callable[[], float] | None = None,
    ) -> None:
        self._clock = clock if clock is not None else time.monotonic
        self._wall = wall_clock if wall_clock is not None else time.time
        self._cooldown_until: dict[str, float] = {}
        self._requests: dict[str, int] = {}
        self._last_error: dict[str, tuple[str, float]] = {}  # name -> (class, wall ts)
        self._reissues: deque[tuple[float, str, str]] = deque()  # (mono ts, from, to)

    def healthy(self, name: str) -> bool:
        return self.cooldown_remaining(name) <= 0.0

    def cooldown_remaining(self, name: str) -> float:
        until = self._cooldown_until.get(name)
        if until is None:
            return 0.0
        return max(0.0, until - self._clock())

    def mark_unhealthy(self, name: str, seconds: float, error_class: str) -> None:
        # The caller computes R-19's min(3600, max(cooldown, retry-after));
        # the cap is re-applied here so no path can park an upstream forever.
        seconds = min(MAX_COOLDOWN_SECONDS, max(0.0, seconds))
        self._cooldown_until[name] = self._clock() + seconds
        self._last_error[name] = (error_class, self._wall())

    def record_request(self, name: str) -> None:
        self._requests[name] = self._requests.get(name, 0) + 1

    def record_reissue(self, from_name: str, to_name: str) -> None:
        self._reissues.append((self._clock(), from_name, to_name))
        self._expire_reissues()

    def _expire_reissues(self) -> None:
        cutoff = self._clock() - _REISSUE_WINDOW_SECONDS
        while self._reissues and self._reissues[0][0] <= cutoff:
            self._reissues.popleft()

    def reissues_last_hour(self, name: str | None = None) -> int:
        """Re-issues in the trailing hour: all of them, or those that left
        `name` (the upstream that failed — pairs with its last_error)."""
        self._expire_reissues()
        if name is None:
            return len(self._reissues)
        return sum(1 for _, from_name, _to in self._reissues if from_name == name)

    def prune_to(self, names: Iterable[str]) -> None:
        keep = set(names)
        for table in (self._cooldown_until, self._requests, self._last_error):
            for name in [n for n in table if n not in keep]:
                del table[name]
        self._reissues = deque(
            entry for entry in self._reissues if entry[1] in keep and entry[2] in keep
        )

    def snapshot(self, name: str) -> dict[str, Any]:
        remaining = self.cooldown_remaining(name)
        last = self._last_error.get(name)
        return {
            "state": "cooldown" if remaining > 0.0 else "healthy",
            "cooldown_remaining_seconds": round(remaining, 3),
            "requests": self._requests.get(name, 0),
            "reissues_last_hour": self.reissues_last_hour(name),
            "last_error_class": last[0] if last is not None else None,
            "last_error_at": (
                datetime.fromtimestamp(last[1], tz=UTC).isoformat() if last is not None else None
            ),
        }
