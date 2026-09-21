"""`llm-redact routes list|test` and `llm-redact spend`: offline routing tooling.

Everything here works from the config file alone — nothing is sent
upstream, no credential is resolved (`routes test` answers with the
credential MODE only), and the spend report reads the sqlite vault
file's `spend` table directly (read-only). The one network call is
`routes test`'s best-effort loopback `GET /__llm-redact/status` (the call
`llm-redact status` makes) so cooldown / budget-exhausted members can be
annotated; it never contacts an upstream. Env var names never appear in
this output; values could not (they are never read).
"""

import argparse
import json
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_redact.config import (
    Config,
    ConfigError,
    apply_env_overrides,
    load_config,
)
from llm_redact.pricing import PriceTable
from llm_redact.routing import (
    PricesConfig,
    RouteRule,
    RoutingConfig,
    RuleMatch,
    UpstreamConfig,
    literal_models,
    select_rule,
)
from llm_redact.spend import Budget, SpendRow, SqliteSpendStore, report

# The path `routes test` assumes when --path is not given: the protocol's
# primary chat endpoint (a rule with `match.path` needs an explicit --path).
DEFAULT_TEST_PATHS: dict[str, str] = {
    "anthropic": "/v1/messages",
    "openai": "/v1/chat/completions",
    "gemini": "/v1beta/models/{model}:generateContent",
    "ollama": "/api/chat",
}


def model_from_gemini_path(path: str) -> str | None:
    """Decision 15b: the Gemini model id between `/models/` and the `:verb`
    — THE proxy's own derivation (`proxy.gemini_path_model`, anchored to
    `/v1|v1beta/models|tunedModels/`), so the dry-run answers exactly what
    `_plan_route` would compute; Vertex publisher paths yield None because
    Vertex is never routed (decision 1)."""
    from llm_redact.proxy import gemini_path_model

    return gemini_path_model(path)


def build_price_table(prices: PricesConfig) -> PriceTable:
    """The effective price table: builtin or `[prices] table = PATH`, with
    `[prices.override."id"]` entries winning. Raises ConfigError for a bad
    file. Delegates to the proxy's builder so the offline CLIs and the
    ledger behind /status can never disagree on the table."""
    from llm_redact.proxy import _build_price_table

    return _build_price_table(prices)


def configured_models(routing: RoutingConfig) -> list[str]:
    """Every model id the config names literally whose price the budget
    accounting will need: the /v1/models list (catalog + glob-free rule
    models) plus model_rewrite targets — except a rewrite target that only
    ever reaches a zero-cost upstream (budgets are ignored there, so its
    missing price is not a budgeting gap)."""
    seen: dict[str, None] = dict.fromkeys(literal_models(routing))
    for rule in routing.rules:
        # parse_config rejects a rule naming an unknown upstream, so lookup is total.
        if rule.model_rewrite is not None and not routing.upstream(rule.upstream).zero_cost:
            seen.setdefault(rule.model_rewrite, None)
    return list(seen)


def unpriced_models(routing: RoutingConfig, table: PriceTable) -> list[str]:
    return [model for model in configured_models(routing) if table.lookup(model) is None]


def referenced_upstreams(routing: RoutingConfig) -> list[UpstreamConfig]:
    """The upstreams traffic can actually reach: every explicit `[upstreams.*]`
    entry plus any legacy auto-registered one that a rule, chain, budget
    chain or default names. An unreferenced legacy provider (the built-in
    defaults register four) is not a routing destination, so doctor does
    not probe it."""
    named: set[str] = set(dict(routing.default_upstreams).values())
    for rule in routing.rules:
        named.add(rule.upstream)
        named.update(rule.on_budget_exhausted)
        for _key, chain in rule.on_status:
            if isinstance(chain, tuple):
                named.update(chain)
    return [u for u in routing.upstreams if not u.legacy or u.name in named]


def budgets_for(routing: RoutingConfig) -> dict[str, Budget]:
    """Per-upstream Budget rows in the ledger's shape (passthrough upstreams
    carry no budget; zero-cost ones ignore theirs) — the proxy's own
    `_budgets_for`, so `spend` and /status agree on remaining budgets."""
    from llm_redact.proxy import _budgets_for

    return _budgets_for(routing)


def match_summary(match: RuleMatch) -> str:
    parts = [f"protocol={match.protocol}"]
    if match.models:
        parts.append("model=" + "|".join(match.models))
    for name, glob in match.headers:
        parts.append(f"{name}={glob}")
    if match.path is not None:
        parts.append(f"path={match.path}")
    if match.auth != "any":
        parts.append(f"auth={match.auth}")
    return " ".join(parts)


def _chains(rule: RouteRule) -> dict[str, list[str] | str]:
    return {
        key: list(chain) if isinstance(chain, tuple) else chain for key, chain in rule.on_status
    }


def _chain_text(chains: Mapping[str, list[str] | str]) -> str:
    if not chains:
        return "-"
    return " ".join(
        f"{key}={value if isinstance(value, str) else '>'.join(value)}"
        for key, value in chains.items()
    )


def rule_row(rule: RouteRule) -> dict[str, Any]:
    return {
        "id": rule.id,
        "protocol": rule.match.protocol,
        "match": match_summary(rule.match),
        "upstream": rule.upstream,
        "on_status": _chains(rule),
        "on_budget_exhausted": list(rule.on_budget_exhausted),
        "reissue_policy": rule.reissue_policy,
        "model_rewrite": rule.model_rewrite,
    }


def upstream_view(upstream: UpstreamConfig) -> dict[str, Any]:
    """An upstream as the offline CLIs show it: mode, never the VAR or a value."""
    return {
        "name": upstream.name,
        "protocol": upstream.protocol,
        "credential": upstream.credential_mode,
        "cost": upstream.cost,
        "legacy": upstream.legacy,
        "count_tokens": upstream.count_tokens,
        "inject_system_note": upstream.inject_system_note,
        "budget_usd": upstream.monthly_budget_usd,
        "budget_tokens": upstream.monthly_budget_tokens,
        "cooldown_seconds": upstream.cooldown_seconds,
    }


def _load(args: argparse.Namespace) -> Config | None:
    try:
        return apply_env_overrides(load_config(args.config))
    except ConfigError as problem:
        print(f"config: {problem}")
        return None


def _routing_state_line(routing: RoutingConfig) -> str:
    if not routing.present:
        return "routing: not configured (no [routing] section — protocol → provider upstream)"
    if not routing.enabled:
        return "routing: disabled ([routing] enabled = false — protocol → provider upstream)"
    defaults = ", ".join(f"{protocol}→{name}" for protocol, name in routing.default_upstreams)
    return (
        f"routing: enabled — {len(routing.rules)} rule(s), {len(routing.upstreams)}"
        f" upstream(s), default: {defaults or 'none'}"
    )


def run_routes_list(args: argparse.Namespace) -> int:
    config = _load(args)
    if config is None:
        return 1
    routing = config.routing
    if args.json:
        print(
            json.dumps(
                {
                    "present": routing.present,
                    "enabled": routing.enabled,
                    "default_upstreams": dict(routing.default_upstreams),
                    "upstreams": [upstream_view(u) for u in routing.upstreams],
                    "rules": [rule_row(rule) for rule in routing.rules],
                    "warnings": list(routing.warnings),
                },
                indent=2,
            )
        )
        return 0
    print(_routing_state_line(routing))
    for warning in routing.warnings:
        print(f"  ⚠ {warning}")
    if not routing.rules:
        if routing.present:
            print("no rules: every request takes its protocol's default_upstream")
        return 0
    rows = [rule_row(rule) for rule in routing.rules]
    headers = ("id", "protocol", "match", "upstream", "on_status", "reissue", "model_rewrite")
    table = [
        (
            row["id"],
            row["protocol"],
            row["match"],
            row["upstream"],
            _chain_text(row["on_status"]),
            row["reissue_policy"],
            row["model_rewrite"] or "-",
        )
        for row in rows
    ]
    widths = [max(len(h), *(len(r[i]) for r in table)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
    for entry in table:
        print("  ".join(v.ljust(w) for v, w in zip(entry, widths, strict=True)).rstrip())
    return 0


def _parse_headers(raw: list[str] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in raw or ():
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"--header expects NAME=VALUE, got {item!r}")
        headers[name.strip().lower()] = value
    return headers


# Per-upstream runtime state from a running proxy's /status `routing`
# block (name -> {"state", "cooldown_remaining_seconds", ...}); {} when no
# proxy answered.
LiveStates = dict[str, dict[str, Any]]

# How long `routes test` waits for the loopback /status answer: a proxy
# that is not running refuses instantly; one that is answers in ms.
_STATUS_TIMEOUT_S = 1.0


def probe_live_states(config: Config) -> LiveStates:
    """Best-effort `GET /__llm-redact/status` on the configured plain-http
    listener (the call `llm-redact status` makes). Never raises: no proxy,
    a TLS listener (client certs are `status`'s business), or an answer
    without a routing block all yield {} — the dry-run then says "not
    probed". No upstream is ever contacted."""
    if config.tls.enabled:
        return {}
    import httpx

    from llm_redact.proxy import RESERVED_PREFIX

    url = f"http://{config.host}:{config.port}{RESERVED_PREFIX}/status"
    try:
        response = httpx.get(url, timeout=_STATUS_TIMEOUT_S)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return {}
    routing = payload.get("routing") if isinstance(payload, dict) else None
    if not isinstance(routing, dict) or not routing.get("enabled"):
        return {}
    upstreams = routing.get("upstreams")
    if not isinstance(upstreams, dict):
        return {}
    return {name: entry for name, entry in upstreams.items() if isinstance(entry, dict)}


# Injectable for tests (nothing here may touch a real socket).
_status_probe: Callable[[Config], LiveStates] = probe_live_states


def _state_note(name: str, live: LiveStates) -> str | None:
    """The runtime annotation a chain member or primary gets when a running
    proxy reported it out of rotation; None when healthy or unknown."""
    entry = live.get(name)
    if entry is None:
        return None
    state = entry.get("state")
    if state == "cooldown":
        left = entry.get("cooldown_remaining_seconds") or 0
        return f"in cooldown {float(left):.0f}s — skipped"
    if state == "budget_exhausted":
        return "budget exhausted — skipped"
    return None


def _annotate_member(
    name: str, routing: RoutingConfig, rule_upstream: UpstreamConfig, live: LiveStates
) -> str:
    """Chain member with the offline-knowable annotations — passthrough
    (decision 3 — never re-issued to), zero-cost, budgeted, no count_tokens
    — plus the running proxy's cooldown / budget-exhausted state when one
    answered (`live`); without a proxy those two are simply unknown."""
    try:
        member = routing.upstream(name)
    except KeyError:
        return f"{name} (UNKNOWN upstream)"
    notes: list[str] = []
    if member.is_passthrough:
        notes.append("passthrough — never a re-issue target")
    state = _state_note(name, live)
    if state is not None:
        notes.append(state)
    if member.zero_cost:
        notes.append("zero-cost")
    elif member.has_budget:
        notes.append("budgeted")
    if not member.count_tokens:
        notes.append("no count_tokens")
    if member.protocol != rule_upstream.protocol:
        notes.append(f"protocol {member.protocol}")
    return f"{name} ({', '.join(notes)})" if notes else name


def route_decision(
    routing: RoutingConfig,
    *,
    protocol: str,
    model: str | None,
    headers: Mapping[str, str],
    path: str,
    auth: str,
    live: LiveStates | None = None,
) -> dict[str, Any]:
    """The dry-run answer `routes test` prints (also its --json shape).
    `live` is the running proxy's per-upstream state block, {} / None
    when no proxy answered."""
    live = live or {}
    rule = select_rule(
        routing, protocol=protocol, model=model, headers=headers, path=path, auth=auth
    )
    upstream_name = rule.upstream if rule is not None else routing.default_for(protocol)
    decision: dict[str, Any] = {
        "protocol": protocol,
        "model": model,
        "path": path,
        "auth": auth,
        "rule": rule.id if rule is not None else None,
        "via": "rule" if rule is not None else ("default" if upstream_name else "no_route"),
        "upstream": None,
        "chains": {},
        "on_budget_exhausted": [],
        "reissue_policy": rule.reissue_policy if rule is not None else None,
        "model_rewrite": rule.model_rewrite if rule is not None else None,
        "count_tokens_404": False,
        "state": None,
    }
    if upstream_name is None:
        return decision
    upstream = routing.upstream(upstream_name)
    decision["upstream"] = upstream_view(upstream)
    decision["count_tokens_404"] = not upstream.count_tokens and path.endswith("/count_tokens")
    live_entry = live.get(upstream_name)
    if live_entry is not None:
        decision["state"] = {
            "state": live_entry.get("state"),
            "cooldown_remaining_seconds": live_entry.get("cooldown_remaining_seconds"),
            "last_error_class": live_entry.get("last_error_class"),
        }
    if rule is not None:
        decision["chains"] = {
            key: chain
            if isinstance(chain, str)
            else [_annotate_member(name, routing, upstream, live) for name in chain]
            for key, chain in rule.on_status
        }
        decision["on_budget_exhausted"] = [
            _annotate_member(name, routing, upstream, live) for name in rule.on_budget_exhausted
        ]
    return decision


def _state_line(decision: dict[str, Any], probed: bool) -> str:
    live = decision["state"]
    if live is None:
        if probed:
            return "state:    unknown (the running proxy reports no such upstream — reload?)"
        return "state:    not probed (no proxy answered on the configured listener)"
    text = f"state:    {live['state']}"
    if live["state"] == "cooldown":
        text += f" ({float(live['cooldown_remaining_seconds'] or 0):.0f}s left)"
    if live["state"] == "budget_exhausted":
        text += " (a direct request answers 402)"
    if live["last_error_class"]:
        text += f" last error: {live['last_error_class']}"
    return text


def run_routes_test(args: argparse.Namespace) -> int:
    config = _load(args)
    if config is None:
        return 1
    routing = config.routing
    try:
        headers = _parse_headers(args.header)
    except ValueError as problem:
        print(f"routes test: {problem}")
        return 2
    protocol: str = args.protocol
    model: str | None = args.model
    path: str = args.path if args.path is not None else DEFAULT_TEST_PATHS[protocol]
    # Rules match the PATH only (`request.url.path`); a pasted `?alt=sse`
    # or `?key=` is not part of the glob (and the latter must not print).
    path = path.partition("?")[0]
    if protocol == "gemini":
        path = path.replace("{model}", model or "gemini-2.5-flash")
        if model is None:
            # Decision 15b: derived AFTER the stand-in is filled in, so the
            # dry-run matches model globs exactly as the proxy would for
            # this path and a body without `model`.
            model = model_from_gemini_path(path)
    live = _status_probe(config)
    decision = route_decision(
        routing,
        protocol=protocol,
        model=model,
        headers=headers,
        path=path,
        auth=args.auth,
        live=live,
    )
    decision["routing_enabled"] = routing.enabled
    decision["live"] = bool(live)
    if args.json:
        print(json.dumps(decision, indent=2))
        return 0
    if not routing.enabled:
        print(_routing_state_line(routing))
        print("(dry-run below shows what the rules WOULD do once enabled)")
    query = f"protocol={protocol} model={model or '-'} path={path} auth={args.auth}"
    if headers:
        query += " headers=" + ",".join(f"{k}={v}" for k, v in headers.items())
    print(f"request:  {query}")
    if decision["via"] == "no_route":
        print(
            f"result:   NO ROUTE — no rule matched and no default_upstream for"
            f" protocol {protocol}: the proxy answers 502 (never forwards by guesswork)"
        )
        return 0
    upstream = decision["upstream"]
    via = f"rule {decision['rule']}" if decision["rule"] else "default_upstream (no rule matched)"
    print(f"matched:  {via}")
    print(
        f"upstream: {upstream['name']} (protocol={upstream['protocol']}"
        f" credential={upstream['credential']} cost={upstream['cost']}"
        f"{' legacy' if upstream['legacy'] else ''})"
    )
    print(_state_line(decision, bool(live)))
    if decision["rule"]:
        chains: dict[str, list[str] | str] = decision["chains"]
        if chains:
            for key, chain in chains.items():
                target = chain if isinstance(chain, str) else " > ".join(chain)
                print(f"on {key}: {target}")
        else:
            print("on_status: none (the upstream's answer is returned as-is)")
        if decision["on_budget_exhausted"]:
            print("on budget exhausted: " + " > ".join(decision["on_budget_exhausted"]))
        print(f"reissue:  {decision['reissue_policy']}")
        print(f"model_rewrite: {decision['model_rewrite'] or 'none'}")
    if decision["count_tokens_404"]:
        print("count_tokens: this upstream does not implement it — the proxy answers 404")
    return 0


def _open_read_only(path: Path) -> sqlite3.Connection:
    """A ``mode=ro`` connection, probed once; a WAL-mode database in a
    directory this user cannot write to (a 0400 backup copy under a 0500
    directory) fails that probe because SQLite must create the ``-shm``
    side file to read WAL, so it is reopened ``immutable=1`` — sound for a
    copy nobody is writing, which is exactly that case (a live vault sits in
    the proxy's own 0700 directory and takes the normal locking path)."""
    uri = f"{path.absolute().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        conn.close()
        conn = sqlite3.connect(f"{uri}&immutable=1", uri=True)
    return conn


class ReadOnlySpendStore(SqliteSpendStore):
    """`SqliteSpendStore` over a `mode=ro` connection: the report never
    creates the vault file, its `spend` table, WAL side files, or chmods
    anything — a 0400 backup copy reads fine. `has_table` is False on a
    legacy vault the proxy has not yet recorded spend into."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn = _open_read_only(path)

    @property
    def has_table(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'spend'"
        ).fetchone()
        return row is not None

    def record(self, row: SpendRow) -> None:
        raise sqlite3.OperationalError("read-only spend store")


def _spend_store_note(config: Config) -> str | None:
    """Why no on-disk spend table exists for this config, or None when it may."""
    if config.vault.backend == "memory":
        return (
            "spend is in-process only with the memory vault backend (it resets on"
            " restart) — read it from a running proxy with `llm-redact status`, or set"
            ' [vault] backend = "sqlite" to persist it'
        )
    if config.vault.backend != "sqlite":
        return (
            f"spend is in-process only with the {config.vault.backend} vault backend —"
            " read it from a running proxy with `llm-redact status`"
        )
    return None


def _print_spend(data: dict[str, Any]) -> None:
    period = data["period"]
    current = " (current period)" if data["current"] else ""
    print(f"spend for {period}{current}: {data['start'][:10]} → {data['end'][:10]}")
    upstreams: dict[str, dict[str, Any]] = data["upstreams"]
    if not upstreams:
        print("no spend recorded")
        return
    for name, entry in upstreams.items():
        priced_rows = entry["rows"] - entry["unpriced_rows"]
        if entry["unpriced_rows"] and not priced_rows:
            usd = "unpriced"
        else:
            usd = f"${entry['usd']:.4f}"
            if entry["unpriced_rows"]:
                usd += f" (+{entry['unpriced_rows']} unpriced rows)"
        hops = entry["hops"]
        line = (
            f"{name}: in={entry['in_tokens']} out={entry['out_tokens']}"
            f" cache_read={entry['cache_read']} cache_write={entry['cache_write']}"
            f" usd={usd} rows={entry['rows']}"
            f" reissue={hops['reissue']['rows']} rows ({hops['reissue_share']:.0%} of spend)"
        )
        if entry["zero_cost"]:
            line += "  budget: zero-cost (ignored)"
        elif entry["budget_usd"] is None and entry["budget_tokens"] is None:
            line += "  budget: none"
        else:
            parts = []
            if entry["budget_usd"] is not None:
                parts.append(f"${entry['remaining_usd']:.2f} of ${entry['budget_usd']:.2f} left")
            if entry["budget_tokens"] is not None:
                parts.append(f"{entry['remaining_tokens']} of {entry['budget_tokens']} tokens left")
            line += "  budget: " + ", ".join(parts)
            if entry["exhausted"]:
                line += "  ⚠ EXHAUSTED"
        print(line)
    totals = data["totals"]
    print(
        f"total: {totals['total_tokens']} tokens, ${totals['usd']:.4f}"
        f" ({totals['unpriced_rows']} unpriced rows)"
    )


def run_spend(args: argparse.Namespace) -> int:
    """R-28: per-upstream tokens/USD, the re-issue share, remaining budget —
    read straight from the sqlite vault file's spend table (the proxy need
    not be running). Month selects a past period by label."""
    from llm_redact.vault import default_vault_path

    config = _load(args)
    if config is None:
        return 1
    routing = config.routing
    note = _spend_store_note(config)
    if note is not None:
        if args.json:
            print(json.dumps({"backend": config.vault.backend, "note": note}))
        else:
            print(note)
        return 0
    path = Path(config.vault.path).expanduser() if config.vault.path else default_vault_path()
    if not path.exists():
        # Never create the vault file from a read-only report.
        message = f"no spend recorded yet ({path} does not exist)"
        print(json.dumps({"backend": "sqlite", "note": message}) if args.json else message)
        return 0
    try:
        store = ReadOnlySpendStore(path)
    except sqlite3.Error as problem:
        print(f"spend: cannot open {path} read-only ({type(problem).__name__})")
        return 1
    try:
        if not store.has_table:
            message = f"no spend recorded yet ({path} has no spend table)"
            print(json.dumps({"backend": "sqlite", "note": message}) if args.json else message)
            return 0
        data = report(
            store,
            month=args.month,
            budgets=budgets_for(routing),
            reset_day=routing.budget_reset_day,
            now=datetime.now(tz=UTC),
        )
    except ValueError as problem:  # a malformed --month
        print(f"spend: {problem}")
        return 2
    except sqlite3.Error as problem:  # corrupted / not a sqlite file / locked
        print(f"spend: cannot read {path} ({type(problem).__name__})")
        return 1
    finally:
        store.close()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    if not routing.enabled:
        print(_routing_state_line(routing))
    _print_spend(data)
    return 0
