"""[upstreams] / [routing] / [prices]: parsing, the §5 invariants and R-2..R-9
schema rules as specific ConfigErrors, legacy auto-registration (R-1), and
the TOML emitter round trip (docs/routing.md)."""

import datetime
import re
import tomllib
from typing import Any

import pytest

from llm_redact.config import Config, ConfigError, parse_config, resolve_credentials
from llm_redact.config_write import emit_config_toml
from llm_redact.routing import (
    DEFAULT_PLAN_LIMIT_HEADERS,
    OAUTH_BETA_MARKER,
    RETRY_SAME,
    ModelPrice,
    PricesConfig,
    RoutingConfig,
)

# The spec's §2.2 example (comments trimmed), with ONE correction: the
# spec's `anthropic-key-lane` chains to `openrouter`, an openai-protocol
# upstream, which its own R-2/I-5 forbid (no protocol translation) — here the
# chain targets `openrouter_anthropic`, OpenRouter's Anthropic-compatible
# endpoint. Every later test derives from a copy of this document.
SPEC_TOML = """
[upstreams.anthropic_oauth]
protocol   = "anthropic"
base_url   = "https://api.anthropic.com"
credential = "passthrough"
inject_system_note = false

[upstreams.anthropic_key]
protocol   = "anthropic"
base_url   = "https://api.anthropic.com"
credential = "env:ANTHROPIC_API_KEY"
monthly_budget_usd = 100
cooldown_seconds   = 60

[upstreams.openai_key]
protocol   = "openai"
base_url   = "https://api.openai.com/v1"
credential = "env:OPENAI_API_KEY"
monthly_budget_usd = 50
cooldown_seconds   = 60

[upstreams.gemini_openai_compat]
protocol   = "openai"
base_url   = "https://generativelanguage.googleapis.com/v1beta/openai"
credential = "env:GEMINI_API_KEY"
monthly_budget_usd = 30

[upstreams.openrouter]
protocol   = "openai"
base_url   = "https://openrouter.ai/api/v1"
credential = "env:OPENROUTER_API_KEY"
monthly_budget_usd = 25
extra_headers = { "HTTP-Referer" = "http://localhost", "X-Title" = "llm-redact" }
body_defaults = { provider = { only = ["anthropic", "openai", "google-vertex", "azure", "together", "deepinfra"], data_collection = "deny" } }

[upstreams.openrouter_anthropic]
protocol   = "anthropic"
base_url   = "https://openrouter.ai/api"
credential = "env:OPENROUTER_API_KEY"
monthly_budget_usd = 25

[upstreams.ollama]
protocol   = "anthropic"
base_url   = "http://host.docker.internal:11434"
credential = "none"
cost       = "zero"
count_tokens = false

[upstreams.ollama_openai]
protocol   = "openai"
base_url   = "http://host.docker.internal:11434/v1"
credential = "none"
cost       = "zero"

[routing]
enabled          = true
default_upstream = "ollama"
max_hops         = 3
request_deadline_seconds = 600
plan_limit_detection = "headers"

[[routing.rule]]
id       = "explore-local"
match    = { protocol = "anthropic", headers = { "x-claude-code-agent-id" = "Explore" } }
upstream = "ollama"
model_rewrite = "muse-64k"

[[routing.rule]]
id       = "max-lane"
match    = { protocol = "anthropic", model = "claude-*", auth = "oauth" }
upstream = "anthropic_oauth"
on_status = { plan_limit_429 = ["anthropic_key", "ollama"], 529 = ["anthropic_key"], throttle_429 = "retry-same" }
reissue_policy = "stateless-only"

[[routing.rule]]
id       = "anthropic-key-lane"
match    = { protocol = "anthropic", model = "claude-*", auth = "gateway-key" }
upstream = "anthropic_key"
on_status = { 429 = ["openrouter_anthropic", "ollama"], 5xx = ["openrouter_anthropic", "ollama"] }

[[routing.rule]]
id       = "openai-lane"
match    = { protocol = "openai", model = "gpt-*" }
upstream = "openai_key"
on_status = { 429 = ["gemini_openai_compat", "ollama_openai"], 5xx = ["gemini_openai_compat", "ollama_openai"] }

[[routing.rule]]
id       = "gemini-lane"
match    = { protocol = "openai", model = "gemini-*" }
upstream = "gemini_openai_compat"
on_status = { 429 = ["ollama_openai"] }

[[routing.rule]]
id       = "local-openai"
match    = { protocol = "openai", model = ["muse-*", "gemma4*", "gpt-oss*", "laguna*", "nomic-*", "embeddinggemma*"] }
upstream = "ollama_openai"

[prices]
table    = "builtin"
[prices.override."claude-sonnet-5"]
input = 2.0
output = 10.0
cache_read = 0.2
cache_write = 2.5
"""  # noqa: E501 — the spec's lines are kept verbatim


def _parse(text: str) -> Config:
    return parse_config(tomllib.loads(text), "<t>")


def _raw() -> dict[str, Any]:
    return tomllib.loads(SPEC_TOML)


def _round_trip(config: Config) -> Config:
    return parse_config(tomllib.loads(emit_config_toml(config)), "round-trip")


def _err(raw: dict[str, Any], pattern: str) -> None:
    with pytest.raises(ConfigError, match=pattern):
        parse_config(raw, "<t>")


# --- the spec example ---------------------------------------------------------


def test_spec_example_parses() -> None:
    config = _parse(SPEC_TOML)
    routing = config.routing
    assert routing.enabled and routing.present
    assert routing.default_upstreams == (("anthropic", "ollama"),)
    assert routing.default_for("anthropic") == "ollama"
    assert routing.default_for("openai") is None
    assert (routing.max_hops, routing.request_deadline_seconds) == (3, 600.0)
    assert routing.plan_limit_detection == "headers"
    assert routing.plan_limit_headers == DEFAULT_PLAN_LIMIT_HEADERS
    assert routing.oauth_beta_marker == OAUTH_BETA_MARKER
    # Explicit upstreams plus the legacy ones (R-1); explicit "ollama" wins
    # over [providers.ollama], so no legacy "ollama" appears.
    assert routing.upstream_names() == (
        "anthropic",
        "anthropic_key",
        "anthropic_oauth",
        "gemini",
        "gemini_openai_compat",
        "ollama",
        "ollama_openai",
        "openai",
        "openai_key",
        "openrouter",
        "openrouter_anthropic",
    )
    legacy = routing.upstream("anthropic")
    assert legacy.legacy and legacy.is_passthrough and legacy.protocol == "anthropic"
    assert legacy.base_url == "https://api.anthropic.com"
    assert not legacy.inject_system_note
    ollama = routing.upstream("ollama")
    assert not ollama.legacy and ollama.protocol == "anthropic" and ollama.zero_cost
    assert ollama.credential_mode == "none" and not ollama.count_tokens
    key = routing.upstream("anthropic_key")
    assert key.env_var == "ANTHROPIC_API_KEY"
    assert key.monthly_budget_usd == 100.0 and key.monthly_budget_tokens is None
    assert key.cooldown_seconds == 60.0 and key.has_budget
    # env upstreams inherit the top-level note switch (true by default).
    assert key.inject_system_note and not routing.upstream("anthropic_oauth").inject_system_note
    openrouter = routing.upstream("openrouter")
    assert openrouter.extra_headers == (
        ("http-referer", "http://localhost"),
        ("x-title", "llm-redact"),
    )
    assert openrouter.body_defaults() == {
        "provider": {
            "only": ["anthropic", "openai", "google-vertex", "azure", "together", "deepinfra"],
            "data_collection": "deny",
        }
    }
    assert [rule.id for rule in routing.rules] == [
        "explore-local",
        "max-lane",
        "anthropic-key-lane",
        "openai-lane",
        "gemini-lane",
        "local-openai",
    ]
    explore, max_lane, key_lane, openai_lane, _gemini, local = routing.rules
    assert explore.match.headers == (("x-claude-code-agent-id", "Explore"),)
    assert explore.match.models == () and explore.match.auth == "any"
    assert explore.model_rewrite == "muse-64k" and explore.on_status == ()
    assert explore.reissue_policy == "always"  # none-credential upstream
    assert max_lane.match.models == ("claude-*",) and max_lane.match.auth == "oauth"
    assert max_lane.on_status == (
        ("plan_limit_429", ("anthropic_key", "ollama")),
        ("529", ("anthropic_key",)),
        ("throttle_429", RETRY_SAME),
    )
    assert max_lane.reissue_policy == "stateless-only"
    assert key_lane.on_status == (
        ("429", ("openrouter_anthropic", "ollama")),
        ("5xx", ("openrouter_anthropic", "ollama")),
    )
    assert key_lane.reissue_policy == "always"
    assert openai_lane.upstream == "openai_key"
    assert local.match.models[0] == "muse-*" and len(local.match.models) == 6
    assert routing.warnings == ()
    assert config.prices == PricesConfig(
        table="builtin",
        overrides=(("claude-sonnet-5", ModelPrice(2.0, 10.0, 0.2, 2.5)),),
    )


def test_config_without_routing_sections_is_default() -> None:
    config = parse_config({}, "<t>")
    assert config.routing == RoutingConfig()
    assert config.prices == PricesConfig()
    assert not config.routing.present and not config.routing.enabled
    # [providers.*] alone registers nothing (R-1: byte-identical behavior).
    config = parse_config({"providers": {"anthropic": {"upstream_base_url": "http://a"}}}, "<t>")
    assert config.routing.upstreams == ()


# --- legacy auto-registration (R-1) ------------------------------------------


def test_legacy_registration_needs_routing_table() -> None:
    raw: dict[str, Any] = {"routing": {"enabled": False}}
    names = parse_config(raw, "<t>").routing.upstream_names()
    assert names == ("anthropic", "gemini", "ollama", "openai")
    for upstream in parse_config(raw, "<t>").routing.upstreams:
        assert upstream.legacy and upstream.is_passthrough and upstream.protocol == upstream.name
        assert upstream.cost == "metered" and not upstream.inject_system_note
    assert parse_config(raw, "<t>").routing.upstream("ollama").base_url == "http://127.0.0.1:11434"


def test_legacy_registration_explicit_wins_and_skips_disabled_or_empty() -> None:
    raw: dict[str, Any] = {
        "providers": {
            "ollama": {"enabled": False},
            "gemini": {"upstream_base_url": ""},
            "openai": {"upstream_base_url": "http://o.example/"},
        },
        "upstreams": {
            "anthropic": {
                "protocol": "openai",
                "base_url": "http://explicit.example",
                "credential": "none",
            }
        },
        "routing": {},
    }
    routing = parse_config(raw, "<t>").routing
    assert routing.upstream_names() == ("anthropic", "openai")
    explicit = routing.upstream("anthropic")
    assert (
        not explicit.legacy
        and explicit.protocol == "openai"
        and explicit.base_url.startswith("http://explicit")
    )
    assert routing.upstream("openai").base_url == "http://o.example"
    assert routing.upstream("openai").legacy
    # Cohere/azure/etc. are never protocols, so never legacy upstreams.
    raw["providers"]["cohere"] = {"upstream_base_url": "http://c.example"}
    assert "cohere" not in parse_config(raw, "<t>").routing.upstream_names()


def test_upstreams_without_routing_are_inert_with_warning() -> None:
    raw = _raw()
    del raw["routing"]
    routing = parse_config(raw, "<t>").routing
    assert not routing.present and not routing.enabled
    assert routing.rules == () and routing.default_upstreams == ()
    assert len(routing.upstreams) == 8 and not any(u.legacy for u in routing.upstreams)
    assert routing.warnings == (
        "[upstreams] is configured but there is no [routing] table: upstreams are inert",
    )
    # No upstreams, no routing: no warning either.
    assert parse_config({}, "<t>").routing.warnings == ()


# --- default_upstream forms (decision 2, I-6) ---------------------------------


def test_default_upstream_string_and_table_forms() -> None:
    raw = _raw()
    raw["routing"]["default_upstream"] = {"anthropic": "ollama", "openai": "ollama_openai"}
    routing = parse_config(raw, "<t>").routing
    assert routing.default_upstreams == (("anthropic", "ollama"), ("openai", "ollama_openai"))
    assert routing.warnings == ()
    raw["routing"]["default_upstream"] = {"openai": "ollama_openai"}
    assert parse_config(raw, "<t>").routing.default_for("anthropic") is None


def test_default_upstream_errors() -> None:
    raw = _raw()
    raw["routing"]["default_upstream"] = {"anthropic": "ollama_openai"}
    _err(raw, r"default_upstream anthropic = 'ollama_openai': that upstream speaks 'openai'")
    raw["routing"]["default_upstream"] = {"cohere": "ollama"}
    _err(raw, r"default_upstream key 'cohere' must be one of")
    raw["routing"]["default_upstream"] = {"anthropic": "nope"}
    _err(raw, r"default_upstream anthropic names unknown upstream 'nope'")
    raw["routing"]["default_upstream"] = {"anthropic": 3}
    _err(raw, r"default_upstream anthropic must be an upstream name")
    raw["routing"]["default_upstream"] = "nope"
    _err(raw, r"default_upstream names unknown upstream 'nope'")
    raw["routing"]["default_upstream"] = ["ollama"]
    _err(raw, r"default_upstream must be an upstream name or a table")


def test_i6_default_required_when_enabled_and_zero_cost_warning() -> None:
    raw = _raw()
    del raw["routing"]["default_upstream"]
    _err(raw, r"\[routing\] default_upstream is required when enabled = true")
    raw["routing"]["default_upstream"] = {}
    _err(raw, r"\[routing\] default_upstream is required when enabled = true")
    # Disabled routing may leave it unset.
    raw["routing"]["enabled"] = False
    routing = parse_config(raw, "<t>").routing
    assert routing.present and not routing.enabled and routing.default_upstreams == ()
    # A metered / passthrough default is allowed but WARNED.
    raw = _raw()
    raw["routing"]["default_upstream"] = {"anthropic": "anthropic_key", "openai": "openrouter"}
    warnings = parse_config(raw, "<t>").routing.warnings
    assert len(warnings) == 2
    assert "default_upstream for anthropic is 'anthropic_key', which is not cost" in warnings[0]
    assert "default_upstream for openai is 'openrouter'" in warnings[1]
    raw["routing"]["default_upstream"] = "anthropic_oauth"
    assert len(parse_config(raw, "<t>").routing.warnings) == 1
    # ...but not when routing is disabled (nothing would spend there).
    raw["routing"]["enabled"] = False
    assert parse_config(raw, "<t>").routing.warnings == ()


# --- upstream schema (R-2, R-3, R-4, I-4, budgets, tables) --------------------


def _upstream_raw(**overrides: Any) -> dict[str, Any]:
    section: dict[str, Any] = {
        "protocol": "anthropic",
        "base_url": "http://u.example/",
        "credential": "env:MY_KEY",
    }
    section.update(overrides)
    return {"upstreams": {"u": section}, "routing": {}}


def test_r2_protocol_enum() -> None:
    _err(_upstream_raw(protocol="cohere"), r"\[upstreams.u\] protocol must be one of")
    _err(_upstream_raw(protocol=""), r"\[upstreams.u\] protocol is required")
    raw = _upstream_raw()
    del raw["upstreams"]["u"]["protocol"]
    _err(raw, r"\[upstreams.u\] protocol is required")


def test_r3_credential_grammar() -> None:
    _err(_upstream_raw(credential="static-key"), r"credential must be 'passthrough', 'none' or")
    # The realistic mistake is pasting the key where env:VAR belongs; the
    # message reaches serve --check, doctor, logs and the editor's 400 body,
    # so the value must NEVER be echoed (R-3: values are never logged).
    pasted = "sk-ant-api03-EXAMPLEEXAMPLEEXAMPLE"
    with pytest.raises(ConfigError) as excinfo:
        parse_config(_upstream_raw(credential=pasted), "<t>")
    message = str(excinfo.value)
    assert "[upstreams.u] credential must be 'passthrough', 'none' or 'env:VAR'" in message
    assert pasted not in message and "EXAMPLE" not in message and "sk-ant" not in message
    _err(_upstream_raw(credential=3), r"\[upstreams.u\] credential must be a string, got int")
    _err(_upstream_raw(credential="env:lower"), r"env:VAR needs a variable name matching")
    _err(_upstream_raw(credential="env:"), r"env:VAR needs a variable name matching")
    _err(_upstream_raw(credential="env:1BAD"), r"env:VAR needs a variable name matching")
    for good in ("env:A", "env:_X9", "none", "passthrough"):
        section = _upstream_raw(credential=good)
        assert parse_config(section, "<t>").routing.upstream("u").credential == good
    # Absent means passthrough.
    raw = _upstream_raw()
    del raw["upstreams"]["u"]["credential"]
    assert parse_config(raw, "<t>").routing.upstream("u").is_passthrough


def test_r3_resolve_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _parse(SPEC_TOML)
    environ = {
        "ANTHROPIC_API_KEY": "sk-ant-test-not-a-real-key",
        "OPENAI_API_KEY": "k",
        "GEMINI_API_KEY": "k",
        "OPENROUTER_API_KEY": "k",
    }
    resolve_credentials(config.routing, environ)  # all present: no error
    with pytest.raises(ConfigError) as excinfo:
        resolve_credentials(config.routing, {**environ, "OPENAI_API_KEY": ""})
    message = str(excinfo.value)
    assert "OPENAI_API_KEY (upstream 'openai_key')" in message
    assert "ANTHROPIC_API_KEY" not in message and "sk-ant" not in message
    with pytest.raises(ConfigError) as excinfo:
        resolve_credentials(config.routing, {})
    assert "GEMINI_API_KEY (upstream 'gemini_openai_compat')" in str(excinfo.value)
    # Disabled routing resolves nothing (inert upstreams need no keys).
    raw = _raw()
    raw["routing"]["enabled"] = False
    resolve_credentials(parse_config(raw, "<t>").routing, {})


def test_r4_cost_enum() -> None:
    _err(_upstream_raw(cost="free"), r"\[upstreams.u\] cost must be one of")
    _err(_upstream_raw(cost=0), r"\[upstreams.u\] cost must be a string, got int")
    assert parse_config(_upstream_raw(cost="zero"), "<t>").routing.upstream("u").zero_cost


def test_i4_inject_system_note_on_passthrough() -> None:
    _err(
        _upstream_raw(credential="passthrough", inject_system_note=True),
        r"\[upstreams.u\] inject_system_note = true is not allowed on a passthrough upstream",
    )
    _err(_upstream_raw(inject_system_note="true"), r"inject_system_note must be a boolean")
    # Defaults: false on passthrough; the top-level switch otherwise; explicit
    # values are honoured either way.
    raw = _upstream_raw(credential="passthrough")
    assert not parse_config(raw, "<t>").routing.upstream("u").inject_system_note
    raw = _upstream_raw()
    assert parse_config(raw, "<t>").routing.upstream("u").inject_system_note
    raw["inject_system_note"] = False
    assert not parse_config(raw, "<t>").routing.upstream("u").inject_system_note
    raw["upstreams"]["u"]["inject_system_note"] = True
    assert parse_config(raw, "<t>").routing.upstream("u").inject_system_note


def test_budgets_on_passthrough_rejected_and_validated() -> None:
    _err(
        _upstream_raw(credential="passthrough", monthly_budget_usd=5),
        r"\[upstreams.u\] monthly_budget_\* is not allowed on a passthrough upstream",
    )
    _err(
        _upstream_raw(credential="passthrough", monthly_budget_tokens=5),
        r"monthly_budget_\* is not allowed on a passthrough upstream",
    )
    _err(_upstream_raw(monthly_budget_usd=0), r"monthly_budget_usd must be positive")
    _err(_upstream_raw(monthly_budget_usd="5"), r"monthly_budget_usd must be a number, got str")
    _err(_upstream_raw(monthly_budget_usd=True), r"monthly_budget_usd must be a number, got bool")
    _err(_upstream_raw(monthly_budget_tokens=0), r"monthly_budget_tokens must be a positive")
    _err(_upstream_raw(monthly_budget_tokens=1.5), r"monthly_budget_tokens must be an integer")
    upstream = parse_config(
        _upstream_raw(monthly_budget_usd=12.5, monthly_budget_tokens=1000), "<t>"
    ).routing.upstream("u")
    assert (upstream.monthly_budget_usd, upstream.monthly_budget_tokens) == (12.5, 1000)


def test_upstream_scalar_validation() -> None:
    _err(_upstream_raw(cooldown_seconds=-1), r"cooldown_seconds must be >= 0")
    _err(_upstream_raw(cooldown_seconds="60"), r"cooldown_seconds must be a number")
    _err(_upstream_raw(count_tokens="no"), r"count_tokens must be a boolean")
    _err(_upstream_raw(base_url="api.example"), r"base_url must start with http:// or https://")
    raw = _upstream_raw()
    del raw["upstreams"]["u"]["base_url"]
    _err(raw, r"\[upstreams.u\] base_url is required")
    _err(_upstream_raw(bogus=1), r"unknown key\(s\) \['bogus'\] in \[upstreams.u\]")
    assert parse_config(_upstream_raw(), "<t>").routing.upstream("u").base_url == "http://u.example"
    assert parse_config(_upstream_raw(cooldown_seconds=0), "<t>").routing.upstream(
        "u"
    ).cooldown_seconds == (0.0)


def test_upstream_names_and_shapes() -> None:
    _err({"upstreams": {"bad name": {}}, "routing": {}}, r"name 'bad name' must match")
    _err({"upstreams": {"-lead": {}}, "routing": {}}, r"name '-lead' must match")
    _err({"upstreams": {"u": "nope"}, "routing": {}}, r"\[upstreams.u\] must be a table")
    _err({"upstreams": ["u"], "routing": {}}, r"\[upstreams\] must contain named subtables")


def test_extra_headers_validation() -> None:
    _err(_upstream_raw(extra_headers=["a"]), r"extra_headers must be a table of header = string")
    _err(_upstream_raw(extra_headers={"x-a": 1}), r"extra_headers: header 'x-a' must map to")
    _err(_upstream_raw(extra_headers={"bad name": "v"}), r"'bad name' is not a valid header name")
    _err(_upstream_raw(extra_headers={"X-A": "1", "x-a": "2"}), r"header 'x-a' is listed twice")
    for name in ("Authorization", "x-api-key", "X-Goog-Api-Key"):
        _err(
            _upstream_raw(extra_headers={name: "Bearer static"}),
            rf"extra_headers may not set '{name.lower()}': credentials come from",
        )
    _err(
        _upstream_raw(credential="passthrough", extra_headers={"x-title": "t"}),
        r"extra_headers/body_defaults apply only to none/env upstreams",
    )
    upstream = parse_config(
        _upstream_raw(extra_headers={"X-Title": "t", "HTTP-Referer": "r"}), "<t>"
    ).routing.upstream("u")
    assert upstream.extra_headers == (("http-referer", "r"), ("x-title", "t"))


def test_body_defaults_validation() -> None:
    _err(_upstream_raw(body_defaults=["a"]), r"body_defaults must be a table")
    _err(
        _upstream_raw(body_defaults={"when": datetime.datetime(2026, 1, 1)}),
        r"body_defaults must contain only JSON-representable values",
    )
    _err(
        _upstream_raw(body_defaults={"x": float("inf")}),
        r"body_defaults must contain only JSON-representable values",
    )
    _err(
        _upstream_raw(credential="passthrough", body_defaults={"a": 1}),
        r"extra_headers/body_defaults apply only to none/env upstreams",
    )
    upstream = parse_config(
        _upstream_raw(body_defaults={"z": [1, 2.5, True, "s"], "a": {"n": {}}}), "<t>"
    ).routing.upstream("u")
    assert upstream.body_defaults_json == '{"a": {"n": {}}, "z": [1, 2.5, true, "s"]}'
    assert upstream.body_defaults() == {"z": [1, 2.5, True, "s"], "a": {"n": {}}}


# --- [routing] scalars ---------------------------------------------------------


def test_routing_scalar_validation() -> None:
    for key, value, pattern in [
        ("max_hops", 0, r"max_hops must be >= 1"),
        ("max_hops", "3", r"max_hops must be an integer, got str"),
        ("request_deadline_seconds", 0, r"request_deadline_seconds must be positive"),
        ("request_deadline_seconds", True, r"request_deadline_seconds must be a number"),
        ("plan_limit_detection", "maybe", r"plan_limit_detection must be one of"),
        ("plan_limit_detection", True, r"plan_limit_detection must be a string, got bool"),
        ("oauth_beta_marker", "  ", r"oauth_beta_marker must be a non-empty string"),
        # A wrong type is a hard error, never a coercion: str(3) would be a
        # marker no request carries and every Claude Code request would
        # silently classify as gateway-key.
        ("oauth_beta_marker", 3, r"oauth_beta_marker must be a string, got int"),
        ("throttle_retry_max_seconds", -1, r"throttle_retry_max_seconds must be >= 0"),
        ("budget_reset_day", 0, r"budget_reset_day must be between 1 and 28"),
        ("budget_reset_day", 29, r"budget_reset_day must be between 1 and 28"),
        ("debug_headers", "yes", r"debug_headers must be a boolean"),
        ("expose_models", 1, r"expose_models must be a boolean"),
        ("model_catalog", "claude", r"model_catalog must be an array of strings"),
        ("enabled", "true", r"enabled must be a boolean"),
        ("rule", {"id": "x"}, r"\[\[routing.rule\]\] must be an array of tables"),
        ("bogus", 1, r"unknown key\(s\) \['bogus'\] in \[routing\]"),
    ]:
        raw = _raw()
        raw["routing"][key] = value
        _err(raw, pattern)
    _err({"routing": "on"}, r"\[routing\] must be a table")
    raw = _raw()
    raw["routing"].update(
        {
            "max_hops": 1,
            "request_deadline_seconds": 1.5,
            "plan_limit_detection": "off",
            "oauth_beta_marker": " custom-marker ",
            "throttle_retry_max_seconds": 0,
            "budget_reset_day": 28,
            "debug_headers": True,
            "expose_models": True,
            "model_catalog": ["claude-sonnet-5", "claude-opus-5"],
        }
    )
    routing = parse_config(raw, "<t>").routing
    assert (routing.max_hops, routing.request_deadline_seconds) == (1, 1.5)
    assert routing.plan_limit_detection == "off"
    assert routing.oauth_beta_marker == "custom-marker"
    assert routing.throttle_retry_max_seconds == 0.0 and routing.budget_reset_day == 28
    assert routing.debug_headers and routing.expose_models
    assert routing.model_catalog == ("claude-sonnet-5", "claude-opus-5")


def test_plan_limit_headers_table() -> None:
    raw = _raw()
    raw["routing"]["plan_limit_headers"] = {"X-Quota": ["Exhausted"], "x-other": ["a", "b"]}
    routing = parse_config(raw, "<t>").routing
    assert routing.plan_limit_headers == (("x-quota", ("Exhausted",)), ("x-other", ("a", "b")))
    for bad, pattern in [
        ([], r"plan_limit_headers must be a table"),
        ({}, r"plan_limit_headers must not be empty"),
        ({"bad name": ["a"]}, r"'bad name' is not a valid header name"),
        ({"x-q": "a"}, r"'x-q' must map to a non-empty array of strings"),
        ({"x-q": []}, r"'x-q' must map to a non-empty array of strings"),
        ({"x-q": [""]}, r"'x-q' must map to a non-empty array of strings"),
        # Two spellings of one header would collapse in the emitter (lossy
        # round trip); the _header_table rule applies here too.
        ({"X-Q": ["a"], "x-q": ["b"]}, r"plan_limit_headers: header 'x-q' is listed twice"),
    ]:
        raw["routing"]["plan_limit_headers"] = bad
        _err(raw, pattern)


# --- rules (R-5, R-7, R-8, R-9, I-1/I-2/I-5, duplicates) ----------------------


def _with_rule(rule: dict[str, Any]) -> dict[str, Any]:
    raw = _raw()
    raw["routing"]["rule"] = [rule]
    return raw


def test_r5_match_fields() -> None:
    base = {"id": "r", "upstream": "ollama"}
    _err(_with_rule(base), r"\[\[routing.rule\]\] 'r' match is required")
    _err(_with_rule({**base, "match": "x"}), r"'r' match must be a table")
    _err(_with_rule({**base, "match": {}}), r"'r' match protocol is required")
    _err(_with_rule({**base, "match": {"protocol": "cohere"}}), r"match protocol must be one of")
    _err(
        _with_rule({**base, "match": {"protocol": "anthropic", "model": 3}}),
        r"match model must be a glob string or a non-empty array of globs",
    )
    _err(
        _with_rule({**base, "match": {"protocol": "anthropic", "model": []}}),
        r"match model must be a glob string",
    )
    _err(
        _with_rule({**base, "match": {"protocol": "anthropic", "model": ["a", ""]}}),
        r"match model must be a glob string",
    )
    _err(
        _with_rule({**base, "match": {"protocol": "anthropic", "headers": "x"}}),
        r"match headers must be a table of header = string",
    )
    _err(
        _with_rule({**base, "match": {"protocol": "anthropic", "headers": {"h": 1}}}),
        r"match headers: header 'h' must map to a string",
    )
    _err(
        _with_rule({**base, "match": {"protocol": "anthropic", "path": ""}}),
        r"match path must be a non-empty string",
    )
    _err(
        _with_rule({**base, "match": {"protocol": "anthropic", "auth": "bearer"}}),
        r"match auth must be one of",
    )
    _err(
        _with_rule({**base, "match": {"protocol": "anthropic", "auth": 1}}),
        r"'r' match auth must be a string, got int",
    )
    _err(
        _with_rule({**base, "match": {"protocol": "anthropic", "extra": 1}}),
        r"unknown key\(s\) \['extra'\] in \[\[routing.rule\]\] 'r' match",
    )
    rule = parse_config(
        _with_rule(
            {
                **base,
                "match": {
                    "protocol": "anthropic",
                    "model": "claude-*",
                    "headers": {"X-Agent": "Explore*"},
                    "path": "/v1/messages*",
                    "auth": "none",
                },
            }
        ),
        "<t>",
    ).routing.rules[0]
    assert rule.match.models == ("claude-*",)
    assert rule.match.headers == (("x-agent", "Explore*"),)
    assert (rule.match.path, rule.match.auth) == ("/v1/messages*", "none")


def test_rule_shape_ids_and_upstream() -> None:
    _err(_with_rule("x"), r"\[\[routing.rule\]\] #1 must be a table")
    _err(_with_rule({}), r"\[\[routing.rule\]\] #1 id is required")
    _err(_with_rule({"id": "bad id"}), r"#1 id 'bad id' must match")
    match = {"protocol": "anthropic"}
    _err(_with_rule({"id": "r", "match": match}), r"'r' upstream is required")
    _err(_with_rule({"id": "r", "match": match, "upstream": "nope"}), r"'r' names unknown upstream")
    _err(
        _with_rule({"id": "r", "match": match, "upstream": "ollama_openai"}),
        r"'r' upstream 'ollama_openai' speaks 'openai', not the matched protocol 'anthropic'",
    )
    _err(
        _with_rule({"id": "r", "match": match, "upstream": "ollama", "bogus": 1}),
        r"unknown key\(s\) \['bogus'\] in \[\[routing.rule\]\] #1",
    )
    raw = _raw()
    raw["routing"]["rule"].append(dict(raw["routing"]["rule"][0]))
    _err(raw, r"\[\[routing.rule\]\] id 'explore-local' is used twice")
    raw["routing"]["rule"] = []
    assert parse_config(raw, "<t>").routing.rules == ()


def test_r7_on_status_keys_and_values() -> None:
    match = {"protocol": "anthropic"}
    base = {"id": "r", "match": match, "upstream": "anthropic_oauth"}

    def err(on_status: Any, pattern: str) -> None:
        _err(_with_rule({**base, "on_status": on_status}), pattern)

    err("x", r"'r' on_status must be a table of status = chain")
    err({"abc": ["ollama"]}, r"on_status key 'abc' must be an HTTP status \(100-599\), a class")
    err({"600": ["ollama"]}, r"on_status key '600' must be an HTTP status")
    err({"42": ["ollama"]}, r"on_status key '42' must be an HTTP status")
    err({"3xx": ["ollama"]}, r"on_status key '3xx' must be an HTTP status")
    err({"429": []}, r"on_status 429 must be a non-empty array of upstream names")
    err({"429": "retry-later"}, r"on_status 429 must be a non-empty array of upstream names")
    err({"429": ["ollama", 3]}, r"on_status 429 must be a non-empty array of upstream names")
    err({"429": ["nope"]}, r"on_status 429 names unknown upstream 'nope'")
    err(
        {"5xx": ["ollama_openai"]},
        r"'r' on_status 5xx member 'ollama_openai' speaks 'openai', not the rule's protocol",
    )
    rule = parse_config(
        _with_rule(
            {
                **base,
                "on_status": {
                    "plan_limit_429": ["anthropic_key"],
                    529: ["ollama"],
                    "4xx": "retry-same",
                    "503": ["anthropic_key", "ollama"],
                },
            }
        ),
        "<t>",
    ).routing.rules[0]
    # File order preserved; int keys normalized to their digits.
    assert rule.on_status == (
        ("plan_limit_429", ("anthropic_key",)),
        ("529", ("ollama",)),
        ("4xx", RETRY_SAME),
        ("503", ("anthropic_key", "ollama")),
    )
    err({"529": ["ollama"], 529: ["ollama"]}, r"on_status key '529' is listed twice")


def test_i1_i2_no_passthrough_chain_member() -> None:
    raw = _raw()
    raw["upstreams"]["second_sub"] = {"protocol": "anthropic", "base_url": "http://s.example"}
    raw["routing"]["rule"][1]["on_status"]["plan_limit_429"] = ["second_sub"]
    _err(
        raw,
        r"\[\[routing.rule\]\] 'max-lane' on_status plan_limit_429 member 'second_sub' is a"
        r" passthrough upstream: a chain may only continue to env:/none upstreams",
    )
    # Same for the legacy passthrough upstreams and for on_budget_exhausted.
    raw = _raw()
    raw["routing"]["rule"][2]["on_status"]["429"] = ["anthropic"]
    _err(raw, r"'anthropic-key-lane' on_status 429 member 'anthropic' is a passthrough")
    raw = _raw()
    raw["routing"]["rule"][2]["on_budget_exhausted"] = ["anthropic_oauth"]
    _err(raw, r"'anthropic-key-lane' on_budget_exhausted member 'anthropic_oauth' is a passthrough")


def test_i5_chain_and_budget_members_share_protocol() -> None:
    raw = _raw()
    raw["routing"]["rule"][2]["on_budget_exhausted"] = ["openrouter"]
    _err(
        raw,
        r"'anthropic-key-lane' on_budget_exhausted member 'openrouter' speaks 'openai', not"
        r" the rule's protocol 'anthropic'",
    )
    raw["routing"]["rule"][2]["on_budget_exhausted"] = ["ollama"]
    rule = parse_config(raw, "<t>").routing.rules[2]
    assert rule.on_budget_exhausted == ("ollama",)
    raw["routing"]["rule"][2]["on_budget_exhausted"] = []
    _err(raw, r"on_budget_exhausted must be a non-empty array of upstream names")


def test_r8_reissue_policy() -> None:
    raw = _raw()
    raw["routing"]["rule"][1]["reissue_policy"] = "sometimes"
    _err(raw, r"'max-lane' reissue_policy must be one of")
    for policy in ("always", "never", "stateless-only"):
        raw["routing"]["rule"][1]["reissue_policy"] = policy
        assert parse_config(raw, "<t>").routing.rules[1].reissue_policy == policy
    # Defaults resolve per the rule's upstream credential mode.
    del raw["routing"]["rule"][1]["reissue_policy"]
    rules = parse_config(raw, "<t>").routing.rules
    assert rules[1].reissue_policy == "stateless-only"  # passthrough
    assert rules[2].reissue_policy == "always"  # env
    assert rules[0].reissue_policy == "always"  # none


def test_r9_model_rewrite() -> None:
    raw = _raw()
    raw["routing"]["rule"][0]["model_rewrite"] = ""
    _err(raw, r"'explore-local' model_rewrite must be a non-empty string")
    raw["routing"]["rule"][0]["model_rewrite"] = 3
    _err(raw, r"'explore-local' model_rewrite must be a non-empty string")
    del raw["routing"]["rule"][0]["model_rewrite"]
    assert parse_config(raw, "<t>").routing.rules[0].model_rewrite is None


def test_rule_warnings_for_dead_keys() -> None:
    raw = _raw()
    raw["routing"]["rule"][3]["on_status"]["plan_limit_429"] = ["ollama_openai"]
    raw["routing"]["rule"][0]["on_budget_exhausted"] = ["anthropic_key"]
    warnings = parse_config(raw, "<t>").routing.warnings
    assert warnings == (
        "[[routing.rule]] 'explore-local' on_budget_exhausted never applies: upstream"
        " 'ollama' has no monthly budget",
        "[[routing.rule]] 'openai-lane' on_status key 'plan_limit_429' never fires for"
        " protocol 'openai' (plan-limit classification is Anthropic-only)",
    )


# --- [prices] ------------------------------------------------------------------


def test_prices_parse_and_errors() -> None:
    assert parse_config({"prices": {}}, "<t>").prices == PricesConfig()
    prices = parse_config(
        {
            "prices": {
                "table": "/etc/llm-redact/prices.json",
                "override": {"z-model": {"input": 1, "output": 2}, "a": {"input": 0, "output": 0}},
            }
        },
        "<t>",
    ).prices
    assert prices.table == "/etc/llm-redact/prices.json"
    # cache_* default to 0.0; overrides sort by model id.
    assert prices.overrides == (
        ("a", ModelPrice(0.0, 0.0, 0.0, 0.0)),
        ("z-model", ModelPrice(1.0, 2.0, 0.0, 0.0)),
    )
    _err({"prices": "builtin"}, r"\[prices\] must be a table")
    _err({"prices": {"table": ""}}, r"\[prices\] table must be \"builtin\" or a file path")
    _err({"prices": {"bogus": 1}}, r"unknown key\(s\) \['bogus'\] in \[prices\]")
    _err({"prices": {"override": ["x"]}}, r"\[prices.override\] must contain per-model tables")
    _err({"prices": {"override": {"m": 3}}}, r"\[prices.override.'m'\] must be a table")
    _err(
        {"prices": {"override": {"m": {"input": 1}}}}, r"is missing required key\(s\) \['output'\]"
    )
    _err({"prices": {"override": {"m": {"input": "1", "output": 1}}}}, r"input must be a number")
    _err({"prices": {"override": {"m": {"input": -1, "output": 1}}}}, r"prices must be >= 0")
    _err(
        {"prices": {"override": {"m": {"input": 1, "output": 1, "x": 1}}}},
        r"unknown key\(s\) \['x'\] in \[prices.override.'m'\]",
    )


# --- emitter round trips -------------------------------------------------------

NASTY_STRINGS = [
    'quote " inside',
    "back\\slash",
    "new\nline",
    "tab\there",
    "guillemets «EMAIL_001» stay",
    "unicode: żółć 東京 🎉",
    'triple """ quotes',
    "control \x01 char",
    "  leading and trailing  ",
    "braces { } and = signs, commas",
]


def test_spec_example_round_trips() -> None:
    config = _parse(SPEC_TOML)
    assert _round_trip(config) == config
    text = emit_config_toml(config)
    # Legacy upstreams are never written (they re-register from providers);
    # routing sections come last; match/on_status are inline tables.
    assert "[upstreams.anthropic]\n" not in text
    assert "[upstreams.anthropic_oauth]" in text
    assert text.index("[routing]") > text.index("[license]" if "[license]" in text else "[log]")
    assert text.index("[prices]") > text.index("[[routing.rule]]")
    assert 'default_upstream = { anthropic = "ollama" }' in text
    assert "[routing.rule.match]" not in text
    assert 'on_status = { plan_limit_429 = ["anthropic_key", "ollama"], 529 = [' in text
    assert "plan_limit_headers" not in text  # default table is not pinned
    assert "[prices.override.claude-sonnet-5]" in text
    assert emit_config_toml(_round_trip(config)) == text


def test_default_config_emits_no_routing_sections() -> None:
    text = emit_config_toml(Config())
    assert "[upstreams" not in text and "[routing]" not in text and "[prices]" not in text
    assert _round_trip(Config()) == Config()


@pytest.mark.parametrize("nasty", NASTY_STRINGS)
def test_nasty_strings_round_trip(nasty: str) -> None:
    raw = _raw()
    raw["upstreams"]["openrouter"]["extra_headers"] = {"x-title": nasty}
    raw["upstreams"]["openrouter"]["body_defaults"] = {
        "provider": {"order": [nasty, 1, 2.5, True, {"k": nasty}], "note": nasty},
        "empty": {},
        "n": 0,
    }
    raw["routing"]["rule"][0]["match"] = {
        "protocol": "anthropic",
        "model": [nasty, "claude-*"],
        "headers": {"x-agent": nasty, "user-agent": "*"},
        "path": nasty,
        "auth": "gateway-key",
    }
    raw["routing"]["rule"][0]["model_rewrite"] = nasty
    raw["routing"]["model_catalog"] = [nasty]
    raw["routing"]["oauth_beta_marker"] = nasty.strip() or "m"
    raw["routing"]["plan_limit_headers"] = {"x-quota": [nasty, "b"]}
    raw["prices"]["override"][nasty] = {"input": 1, "output": 2, "cache_read": 0.5}
    config = parse_config(raw, "<t>")
    assert _round_trip(config) == config


def test_every_routing_field_nondefault_round_trips() -> None:
    raw = _raw()
    raw["inject_system_note"] = False
    raw["upstreams"]["anthropic_key"]["inject_system_note"] = True  # differs from the switch
    raw["upstreams"]["anthropic_key"]["monthly_budget_tokens"] = 123456
    raw["upstreams"]["anthropic_key"]["cooldown_seconds"] = 0
    raw["upstreams"]["ollama"]["count_tokens"] = False
    raw["upstreams"]["ollama"]["cooldown_seconds"] = 1.25
    raw["routing"].update(
        {
            "enabled": True,
            "default_upstream": {"anthropic": "ollama", "openai": "ollama_openai"},
            "max_hops": 5,
            "request_deadline_seconds": 30,
            "plan_limit_detection": "off",
            "plan_limit_headers": {"x-a": ["1"], "x-b": ["2", "3"]},
            "oauth_beta_marker": "oauth-2099-01-01",
            "throttle_retry_max_seconds": 5,
            "budget_reset_day": 15,
            "debug_headers": True,
            "expose_models": True,
            "model_catalog": ["claude-sonnet-5"],
        }
    )
    raw["routing"]["rule"][2]["on_budget_exhausted"] = ["ollama"]
    raw["routing"]["rule"][2]["reissue_policy"] = "never"
    raw["routing"]["rule"][1]["on_status"]["4xx"] = "retry-same"
    config = parse_config(raw, "<t>")
    assert config.routing.plan_limit_headers == (("x-a", ("1",)), ("x-b", ("2", "3")))
    rt = _round_trip(config)
    assert rt == config
    assert rt.routing.upstream("anthropic_key").inject_system_note
    assert not rt.routing.upstream("openai_key").inject_system_note


def test_disabled_and_inert_forms_round_trip() -> None:
    # [routing] present but disabled, no default.
    raw = _raw()
    raw["routing"]["enabled"] = False
    del raw["routing"]["default_upstream"]
    config = parse_config(raw, "<t>")
    assert _round_trip(config) == config
    assert "[routing]\nenabled = false\nmax_hops" in emit_config_toml(config)
    # [upstreams] without [routing]: the inert warning survives the trip.
    raw = _raw()
    del raw["routing"]
    config = parse_config(raw, "<t>")
    assert config.routing.warnings and _round_trip(config) == config
    assert "[routing]" not in emit_config_toml(config)
    # Legacy-only registration (no [upstreams]) emits no upstream tables.
    config = parse_config({"routing": {"enabled": False}}, "<t>")
    assert "[upstreams" not in emit_config_toml(config)
    assert _round_trip(config) == config


def test_warnings_round_trip_with_metered_default() -> None:
    raw = _raw()
    raw["routing"]["default_upstream"] = "anthropic_key"
    config = parse_config(raw, "<t>")
    assert len(config.routing.warnings) == 1
    assert _round_trip(config) == config


def test_emitted_routing_is_valid_toml_with_bare_and_quoted_keys() -> None:
    raw = _raw()
    raw["prices"]["override"]["openai/gpt-5"] = {"input": 1, "output": 2}
    raw["routing"]["rule"][0]["match"]["headers"] = {"x-h": "v", "X_Under.Score": "w"}
    config = parse_config(raw, "<t>")
    text = emit_config_toml(config)
    assert '[prices.override."openai/gpt-5"]' in text
    assert "[prices.override.claude-sonnet-5]" in text
    # Bare keys where TOML allows them ("-" and "_"), quoted otherwise (".").
    assert re.search(r'headers = \{ x-h = "v", "x_under.score" = "w" \}', text)
    assert _round_trip(config) == config
