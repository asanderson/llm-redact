"""Emit a Config back to TOML for the /__llm-redact/config editor.

The stdlib parses TOML (tomllib) but cannot write it, and the runtime-deps
rule (httpx/starlette/uvicorn only) rules out tomli-w. The schema here is
bounded — strings, ints, bools, lists of strings, two levels of tables, and
one array-of-tables — so a small hand-rolled emitter is safe, and every
emitted document is verified by reparsing through parse_config before it is
written (see the endpoint) plus round-trip tests.

Emitted files are normalized: fixed section order, every field explicit.
Comments from a hand-edited file are NOT preserved; the editor keeps one
.bak of the previous bytes.
"""

import json
import os
import re
from pathlib import Path

from llm_redact.config import (
    AzureAuditConfig,
    Config,
    EmailConfig,
    OtelConfig,
    PricesConfig,
    RdbmsConfig,
    RouteRule,
    RoutingConfig,
    S3AuditConfig,
    UpstreamConfig,
    UsersConfig,
)
from llm_redact.detection.engine import DetectionConfig

_BARE_KEY_RE = re.compile(r"[A-Za-z0-9_-]+\Z")

_HEADER = (
    "# Written by the llm-redact config editor. Comments are not preserved;\n"
    "# the previous file is kept alongside as config.toml.bak.\n"
)


def _toml_str(value: str) -> str:
    # A JSON string with ensure_ascii=False is a valid TOML basic string:
    # both escape backslash, double quote, and control characters, and TOML
    # accepts \uXXXX escapes. The one divergence is DEL: JSON leaves U+007F
    # raw while TOML forbids it in a basic string, so it is escaped by hand
    # (a value parsed from the file must emit back as parsable TOML — the
    # routing sections carry many user strings: globs, header values,
    # model ids). Pinned by the round-trip tests.
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def _toml_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _toml_str(value)
    return str(value)


def _toml_list(values: tuple[str, ...] | list[str]) -> str:
    if not values:
        return "[]"
    items = ",\n".join(f"    {_toml_str(v)}" for v in values)
    return f"[\n{items},\n]"


def _toml_key(key: str) -> str:
    # Bare where TOML allows it (protocol names, header names, "529"),
    # quoted otherwise — a quoted key is a basic string, same escaping.
    return key if _BARE_KEY_RE.fullmatch(key) else _toml_str(key)


def _toml_inline(value: object) -> str:
    """One TOML value on one line: the routing sections live inside
    array-of-tables entries and inline tables, where the multi-line
    `_toml_list` form is not allowed. Covers exactly the JSON-representable
    scalars/containers that parse_config admits (never None/datetime)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_inline(item) for item in value) + "]"
    if isinstance(value, dict):
        return _toml_inline_table(value)
    raise TypeError(f"no TOML form for {type(value).__name__}")


def _toml_inline_table(table: dict[str, object]) -> str:
    if not table:
        return "{}"
    return "{ " + ", ".join(f"{_toml_key(k)} = {_toml_inline(v)}" for k, v in table.items()) + " }"


def _emit_upstream(lines: list[str], upstream: UpstreamConfig, *, inject_default: bool) -> None:
    lines.append(f"\n[upstreams.{upstream.name}]")
    lines.append(f"protocol = {_toml_str(upstream.protocol)}")
    lines.append(f"base_url = {_toml_str(upstream.base_url)}")
    lines.append(f"credential = {_toml_str(upstream.credential)}")
    lines.append(f"cost = {_toml_str(upstream.cost)}")
    resolved_default = False if upstream.is_passthrough else inject_default
    if upstream.inject_system_note != resolved_default:
        # Omitted when it equals the resolved default (false on passthrough,
        # the top-level switch otherwise) so the file tracks that switch.
        lines.append(f"inject_system_note = {_toml_value(upstream.inject_system_note)}")
    lines.append(f"count_tokens = {_toml_value(upstream.count_tokens)}")
    if upstream.monthly_budget_usd is not None:
        lines.append(f"monthly_budget_usd = {upstream.monthly_budget_usd}")
    if upstream.monthly_budget_tokens is not None:
        lines.append(f"monthly_budget_tokens = {upstream.monthly_budget_tokens}")
    lines.append(f"cooldown_seconds = {upstream.cooldown_seconds}")
    if upstream.extra_headers:
        lines.append(f"extra_headers = {_toml_inline_table(dict(upstream.extra_headers))}")
    body_defaults = upstream.body_defaults()
    if body_defaults:
        lines.append(f"body_defaults = {_toml_inline_table(body_defaults)}")


def _emit_rule(lines: list[str], rule: RouteRule) -> None:
    # `match` and `on_status` MUST be inline tables: a [routing.rule.match]
    # header inside an array-of-tables entry is not addressable in TOML.
    match: dict[str, object] = {"protocol": rule.match.protocol}
    if rule.match.models:
        match["model"] = list(rule.match.models)
    if rule.match.headers:
        match["headers"] = dict(rule.match.headers)
    if rule.match.path is not None:
        match["path"] = rule.match.path
    match["auth"] = rule.match.auth
    lines.append("\n[[routing.rule]]")
    lines.append(f"id = {_toml_str(rule.id)}")
    lines.append(f"match = {_toml_inline_table(match)}")
    lines.append(f"upstream = {_toml_str(rule.upstream)}")
    if rule.model_rewrite is not None:
        lines.append(f"model_rewrite = {_toml_str(rule.model_rewrite)}")
    if rule.on_status:
        on_status: dict[str, object] = {
            key: (list(chain) if isinstance(chain, tuple) else chain)
            for key, chain in rule.on_status
        }
        lines.append(f"on_status = {_toml_inline_table(on_status)}")
    lines.append(f"reissue_policy = {_toml_str(rule.reissue_policy)}")
    if rule.on_budget_exhausted:
        lines.append(f"on_budget_exhausted = {_toml_inline(list(rule.on_budget_exhausted))}")


def _emit_routing(lines: list[str], routing: RoutingConfig, *, inject_default: bool) -> None:
    for upstream in routing.upstreams:
        # Legacy upstreams are never written: they re-register from the
        # [providers.*] sections above whenever a [routing] table exists.
        if not upstream.legacy:
            _emit_upstream(lines, upstream, inject_default=inject_default)
    if not routing.present:
        return
    default_routing = RoutingConfig()
    lines.append("\n[routing]")
    lines.append(f"enabled = {_toml_value(routing.enabled)}")
    if routing.default_upstreams:
        # Always the table form; the string form folds into it at parse.
        lines.append(f"default_upstream = {_toml_inline_table(dict(routing.default_upstreams))}")
    lines.append(f"max_hops = {routing.max_hops}")
    lines.append(f"request_deadline_seconds = {routing.request_deadline_seconds}")
    lines.append(f"plan_limit_detection = {_toml_str(routing.plan_limit_detection)}")
    if routing.plan_limit_headers != default_routing.plan_limit_headers:
        # Omitted while default: the table REPLACES the built-in one, so
        # writing it out would pin today's header names against future
        # defaults (the same open-endedness as the omitted `enabled` list).
        table = {name: list(values) for name, values in routing.plan_limit_headers}
        lines.append(f"plan_limit_headers = {_toml_inline_table(dict(table))}")
    lines.append(f"oauth_beta_marker = {_toml_str(routing.oauth_beta_marker)}")
    lines.append(f"throttle_retry_max_seconds = {routing.throttle_retry_max_seconds}")
    lines.append(f"budget_reset_day = {routing.budget_reset_day}")
    lines.append(f"debug_headers = {_toml_value(routing.debug_headers)}")
    lines.append(f"expose_models = {_toml_value(routing.expose_models)}")
    if routing.model_catalog:
        lines.append(f"model_catalog = {_toml_list(routing.model_catalog)}")
    for rule in routing.rules:
        _emit_rule(lines, rule)


def _emit_prices(lines: list[str], prices: PricesConfig) -> None:
    if prices == PricesConfig():
        # Like [otel]: an all-defaults section is omitted.
        return
    lines.append("\n[prices]")
    lines.append(f"table = {_toml_str(prices.table)}")
    for model_id, price in prices.overrides:
        lines.append(f"\n[prices.override.{_toml_key(model_id)}]")
        lines.append(f"input = {price.input}")
        lines.append(f"output = {price.output}")
        lines.append(f"cache_read = {price.cache_read}")
        lines.append(f"cache_write = {price.cache_write}")


def emit_config_toml(config: Config, *, banner: bool = True) -> str:
    # banner=False for display surfaces (`config show`): the editor banner
    # talks about file-writing and .bak, which is wrong for stdout.
    lines: list[str] = [_HEADER] if banner else []
    lines.append(f"host = {_toml_str(config.host)}")
    lines.append(f"port = {config.port}")
    lines.append(f"inject_system_note = {_toml_value(config.inject_system_note)}")
    lines.append(f"max_body_bytes = {config.max_body_bytes}")

    for name in sorted(config.providers):
        # Custom upstreams ("custom:vllm") emit the canonical nested form.
        if name.startswith("custom:"):
            header = f"providers.custom.{name.removeprefix('custom:')}"
        else:
            header = f"providers.{name}"
        lines.append(f"\n[{header}]")
        lines.append(f"upstream_base_url = {_toml_str(config.providers[name].upstream_base_url)}")
        if not config.providers[name].enabled:
            # Omitted when true: enabled is the additive default.
            lines.append("enabled = false")
        if not config.providers[name].detection:
            # Omitted when true, like enabled — and loudly annotated: this
            # provider's requests are forwarded WITHOUT redaction.
            lines.append("detection = false # values go to this upstream unredacted")

    detection = config.detection
    lines.append("\n[detection]")
    if detection.enabled == DetectionConfig().enabled:
        # Omitted on purpose: an absent `enabled` means "all built-in rules,
        # including ones added in future versions"; writing the list out
        # would silently pin it.
        lines.append("# enabled omitted: all built-in rules (incl. future ones) are active")
    else:
        lines.append(f"enabled = {_toml_list(detection.enabled)}")
    if detection.languages is not None:
        # Omitted when unset: absence means all languages (all rules run).
        lines.append(f"languages = {_toml_list(detection.languages)}")
    lines.append(f"allowlist = {_toml_list(detection.allowlist)}")
    lines.append(f"allowlist_patterns = {_toml_list(detection.allowlist_patterns)}")

    if detection.mcp_exempt_servers:
        # Skipped when empty: absence means every MCP server's content is
        # detected like any other conversation content.
        lines.append("\n[detection.mcp]")
        lines.append(f"exempt_servers = {_toml_list(detection.mcp_exempt_servers)}")

    if detection.allowlist_by_type:
        # Skipped when empty, same as modes: absence means "no per-type
        # exceptions".
        lines.append("\n[detection.allowlist_by_type]")
        for detector_type, values in detection.allowlist_by_type:
            lines.append(f"{_toml_str(detector_type)} = {_toml_list(values)}")

    if detection.modes:
        # Skipped when empty: an absent table means "everything redacts",
        # the same open-ended default as the omitted `enabled` list.
        lines.append("\n[detection.modes]")
        for rule_name, mode in detection.modes:
            lines.append(f"{_toml_str(rule_name)} = {_toml_str(mode)}")

    ner = detection.ner
    lines.append("\n[detection.ner]")
    lines.append(f"enabled = {_toml_value(ner.enabled)}")
    lines.append(f"backend = {_toml_str(ner.backend)}")
    if ner.backends is not None:
        lines.append(f"backends = {_toml_list(ner.backends)}")
    lines.append(f"entities = {_toml_list(ner.entities)}")
    lines.append(f"max_chars = {ner.max_chars}")
    lines.append(f"language = {_toml_str(ner.language)}")
    if ner.model is not None:
        lines.append(f"model = {_toml_str(ner.model)}")
    if any(b in ("gliner", "presidio") for b in ner.active_backends()):
        # parse_config rejects score_threshold without a confidence backend.
        lines.append(f"score_threshold = {ner.score_threshold}")
    if ner.models:
        # Subtable LAST within the ner section: any top-level ner key
        # emitted after it would parse into the wrong table.
        lines.append("\n[detection.ner.models]")
        for backend_name, model_name in ner.models:
            lines.append(f"{backend_name} = {_toml_str(model_name)}")

    for entry in detection.deny_strings:
        # Always the canonical per-entry form: the `deny = [...]` sugar folds
        # into deny_strings at parse time, so emitting only this shape keeps
        # the round-trip exact.
        lines.append("\n[[detection.deny_strings]]")
        lines.append(f"value = {_toml_str(entry.value)}")
        lines.append(f"case_sensitive = {_toml_value(entry.case_sensitive)}")
        lines.append(f"type = {_toml_str(entry.detector_type)}")

    for rule in detection.custom_rules:
        lines.append("\n[[detection.custom_rules]]")
        lines.append(f"name = {_toml_str(rule.name)}")
        lines.append(f"type = {_toml_str(rule.detector_type)}")
        lines.append(f"pattern = {_toml_str(rule.pattern)}")
        lines.append(f"priority = {rule.priority}")
        if rule.validator is not None:
            lines.append(f"validator = {_toml_str(rule.validator)}")
        if rule.required:
            lines.append(f"required = {_toml_list(rule.required)}")
        if rule.anchors:
            lines.append(f"anchors = {_toml_list(rule.anchors)}")

    lines.append("\n[rehydration]")
    lines.append(f"fuzzy = {_toml_value(config.rehydration.fuzzy)}")

    lines.append("\n[vault]")
    lines.append(f"backend = {_toml_str(config.vault.backend)}")
    if config.vault.path is not None:
        lines.append(f"path = {_toml_str(config.vault.path)}")
    lines.append(f"session = {_toml_str(config.vault.session)}")
    lines.append(f"session_mode = {_toml_str(config.vault.session_mode)}")
    lines.append(f"encryption = {_toml_str(config.vault.encryption)}")
    if config.vault.session_ttl_days:
        lines.append(f"session_ttl_days = {config.vault.session_ttl_days}")

    if config.vault.rdbms != RdbmsConfig():
        # Like [audit.s3]: an all-defaults section is omitted. Subtable
        # AFTER [vault]'s top-level keys. The DSN may embed credentials —
        # it round-trips (this file is the operator's own config) but is
        # never logged.
        rdbms = config.vault.rdbms
        lines.append("\n[vault.rdbms]")
        if rdbms.dsn:
            lines.append(f"dsn = {_toml_str(rdbms.dsn)}")
        if rdbms.password_env != RdbmsConfig().password_env:
            lines.append(f"password_env = {_toml_str(rdbms.password_env)}")
        if rdbms.module:
            lines.append(f"module = {_toml_str(rdbms.module)}")
        if rdbms.cloud:
            lines.append(f"cloud = {_toml_str(rdbms.cloud)}")

    lines.append("\n[audit]")
    lines.append(f"enabled = {_toml_value(config.audit.enabled)}")
    if config.audit.path is not None:
        lines.append(f"path = {_toml_str(config.audit.path)}")
    lines.append(f"max_rows = {config.audit.max_rows}")
    if config.audit.tamper_evident:
        lines.append(f"tamper_evident = {_toml_value(config.audit.tamper_evident)}")
    if config.audit.required:
        lines.append(f"required = {_toml_value(config.audit.required)}")

    if config.audit.s3 != S3AuditConfig():
        # Like [otel]: an all-defaults section is omitted (the sink is off
        # unless asked for). Subtable AFTER [audit]'s top-level keys.
        s3 = config.audit.s3
        lines.append("\n[audit.s3]")
        lines.append(f"enabled = {_toml_value(s3.enabled)}")
        lines.append(f"provider = {_toml_str(s3.provider)}")
        lines.append(f"bucket = {_toml_str(s3.bucket)}")
        lines.append(f"prefix = {_toml_str(s3.prefix)}")
        lines.append(f"region = {_toml_str(s3.region)}")
        if s3.endpoint_url is not None:
            lines.append(f"endpoint_url = {_toml_str(s3.endpoint_url)}")
        lines.append(f"flush_seconds = {s3.flush_seconds}")
        if s3.encryption != "none":
            lines.append(f"encryption = {_toml_str(s3.encryption)}")

    if config.audit.azure != AzureAuditConfig():
        az = config.audit.azure
        lines.append("\n[audit.azure]")
        lines.append(f"enabled = {_toml_value(az.enabled)}")
        lines.append(f"account = {_toml_str(az.account)}")
        lines.append(f"container = {_toml_str(az.container)}")
        lines.append(f"prefix = {_toml_str(az.prefix)}")
        if az.endpoint_url is not None:
            lines.append(f"endpoint_url = {_toml_str(az.endpoint_url)}")
        lines.append(f"flush_seconds = {az.flush_seconds}")
        if az.encryption != "none":
            lines.append(f"encryption = {_toml_str(az.encryption)}")

    lines.append("\n[log]")
    lines.append(f"format = {_toml_str(config.log.format)}")

    if config.tls.certfile is not None and config.tls.keyfile is not None:
        # Omitted entirely when unset: an empty [tls] table means the same
        # as no table, and paths-only means nothing sensitive is written.
        # (parse_config guarantees certfile and keyfile come together.)
        lines.append("\n[tls]")
        lines.append(f"certfile = {_toml_str(config.tls.certfile)}")
        lines.append(f"keyfile = {_toml_str(config.tls.keyfile)}")
        if config.tls.client_ca is not None:
            lines.append(f"client_ca = {_toml_str(config.tls.client_ca)}")

    if config.otel != OtelConfig():
        # Like [tls]: an all-defaults section is omitted (the integration is
        # off unless asked for).
        lines.append("\n[otel]")
        lines.append(f"enabled = {_toml_value(config.otel.enabled)}")
        if config.otel.endpoint is not None:
            lines.append(f"endpoint = {_toml_str(config.otel.endpoint)}")
        lines.append(f"service_name = {_toml_str(config.otel.service_name)}")

    if config.users != UsersConfig():
        lines.append("\n[users]")
        if config.users.path is not None:
            lines.append(f"path = {_toml_str(config.users.path)}")

    if config.email != EmailConfig():
        # The SMTP password is env-only by design and never appears here.
        lines.append("\n[email]")
        if config.email.smtp_host is not None:
            lines.append(f"smtp_host = {_toml_str(config.email.smtp_host)}")
        lines.append(f"smtp_port = {_toml_value(config.email.smtp_port)}")
        lines.append(f"starttls = {_toml_value(config.email.starttls)}")
        if config.email.username is not None:
            lines.append(f"username = {_toml_str(config.email.username)}")
        lines.append(f"password_env = {_toml_str(config.email.password_env)}")
        if config.email.from_address is not None:
            lines.append(f"from_address = {_toml_str(config.email.from_address)}")

    if config.license.key is not None or config.license.key_file is not None:
        # Only ever the signed public token / a path — never key MATERIAL
        # beyond what the licensee was issued.
        lines.append("\n[license]")
        if config.license.key is not None:
            lines.append(f"key = {_toml_str(config.license.key)}")
        if config.license.key_file is not None:
            lines.append(f"key_file = {_toml_str(config.license.key_file)}")

    # Routing sections LAST: [[routing.rule]] is an array of tables, and any
    # top-level key emitted after one would parse into that rule.
    _emit_routing(lines, config.routing, inject_default=config.inject_system_note)
    _emit_prices(lines, config.prices)

    return "\n".join(lines) + "\n"


def write_config_atomic(path: Path, text: str) -> Path | None:
    """Write the config file atomically with 0600 perms; keep one .bak.

    Returns the backup path when a previous file existed, else None. The
    directory is created 0700 if missing (it may hold vault paths later).
    """
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)

    backup: Path | None = None
    if path.exists():
        backup = path.with_name(path.name + ".bak")
        _write_0600(backup, path.read_bytes())

    tmp = path.with_name(path.name + ".tmp")
    _write_0600(tmp, text.encode("utf-8"), fsync=True)
    os.replace(tmp, path)
    return backup


def _write_0600(path: Path, data: bytes, *, fsync: bool = False) -> None:
    path.unlink(missing_ok=True)
    # O_BINARY: without it the Windows CRT opens the fd in text mode and
    # rewrites the newlines we just emitted — the file must round-trip
    # byte-exact (POSIX has no such flag, hence the getattr).
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, data)
        if fsync:
            os.fsync(fd)
    finally:
        os.close(fd)
