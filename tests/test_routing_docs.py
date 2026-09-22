"""docs/routing.md and the config.example.toml routing block stay parseable
and honest.

Every fenced ``toml`` block in docs/routing.md is a COMPLETE config (a
user can paste it), so each one must survive `parse_config` — a doc
example that the parser rejects is worse than none. The example config's
routing block is commented out (routing is off by default); it is
delimited by ``## ROUTING-EXAMPLE-BEGIN`` / ``## ROUTING-EXAMPLE-END``
markers, inside which lines starting with ``# `` are config to uncomment
and lines starting with ``## `` are notes that stay comments. The test
uncomments it exactly that way and parses both the block alone and the
whole file with the block uncommented. The doc must also name every
routing error string verbatim (troubleshooting is keyed by them), and the
docs index must link it.
"""

import re
import tomllib
from pathlib import Path

import pytest

from llm_redact.config import ConfigError, VaultConfig, parse_config

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "routing.md"
TROUBLESHOOTING = ROOT / "docs" / "troubleshooting.md"
EXAMPLE = ROOT / "config.example.toml"

_FENCE_RE = re.compile(r"^```toml\n(.*?)^```$", re.M | re.S)
_BEGIN = "## ROUTING-EXAMPLE-BEGIN"
_END = "## ROUTING-EXAMPLE-END"

# Strings the proxy emits (proxy-generated replies and response headers);
# docs/routing.md's troubleshooting section is keyed by them verbatim.
ERROR_STRINGS = (
    "no rule matched and no default_upstream",
    "does not implement count_tokens",
    "budget exhausted for this period",
    "x-llm-redact-reissue: skipped; reason=stateful",
    "x-llm-redact-reissue: skipped; reason=no-candidate",
    "edit the file and reload",
    # The fix-stage strings: the spend-store ConfigError, the editor's
    # explicit 500, the two reissue_policy = "never" warnings, the chain
    # self-reference / duplicate refusals, and routes test's probe line.
    "spend table in the vault database",
    "could not be opened",
    "but not applied",
    "fix the cause and reload (SIGHUP)",
    'never applies: reissue_policy = "never" forbids re-issuing to another upstream',
    'on_budget_exhausted never applies: reissue_policy = "never"',
    "names the rule's own upstream",
    "twice",
    "not probed (no proxy answered on the configured listener)",
    "environment variable VAR is unset or empty",
)

# Shipped behaviours the review found undocumented; each phrase is the
# doc's own wording for one of them, so dropping the paragraph fails here.
BEHAVIOUR_PHRASES = (
    "modelVersion",  # decision 15b: Gemini path-derived model + restore
    "path segment",
    "list-price equivalent",  # passthrough USD is priced at table rates
    "1 + re-issues",  # hops semantics; retry-same is not a hop
    "not** a hop",
    "class=ok reissue=yes",
    "Id-only follow-ups carry no model",
    "referenced** upstream",  # doctor probes referenced upstreams only
    "runtime-observed",  # /status unpriced_models vs doctor's static list
    "replaces** an inbound header",  # extra_headers replace, not append
    "absorbs the request's leading `/v1`",  # base_url fold for any openai path
    "`key` parameter",  # ?key= dropped for none/env upstreams
    "whole-body string walk",  # R-12: redaction scope on the passthrough lane
    "No upstream is ever contacted",  # routes test
    "RDBMS vault backend",  # spend in-process there too
)


def _doc_blocks() -> list[str]:
    return _FENCE_RE.findall(DOC.read_text(encoding="utf-8"))


def _example_lines() -> tuple[list[str], int, int]:
    lines = EXAMPLE.read_text(encoding="utf-8").splitlines()
    return lines, lines.index(_BEGIN), lines.index(_END)


def _uncomment(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if line.startswith("## "):
            continue  # a note: stays a comment
        if line == "#" or not line.strip():
            out.append("")
            continue
        assert line.startswith("# "), f"live line inside the commented routing block: {line!r}"
        out.append(line[2:])
    return out


_DOC_BLOCKS = _doc_blocks()


@pytest.mark.parametrize(
    "text",
    [pytest.param(block, id=f"routing.md#{i}") for i, block in enumerate(_DOC_BLOCKS)],
)
def test_every_toml_block_in_routing_doc_parses(text: str) -> None:
    parse_config(tomllib.loads(text), "<docs/routing.md>")


def test_routing_doc_has_toml_blocks() -> None:
    # The parametrized test above is vacuous if the fence regex finds nothing.
    assert len(_DOC_BLOCKS) >= 3


def test_example_config_routing_block_parses_on_its_own() -> None:
    lines, begin, end = _example_lines()
    block = "\n".join(_uncomment(lines[begin + 1 : end]))
    raw = tomllib.loads(block)
    assert set(raw) == {"upstreams", "routing", "prices"}
    assert raw["routing"]["enabled"] is True
    assert raw["upstreams"]["ollama"]["count_tokens"] is False  # decision 7
    assert isinstance(raw["routing"]["default_upstream"], dict)  # decision 2, table form
    parse_config(raw, "<config.example.toml routing block>")


def test_example_config_with_routing_block_uncommented_parses_whole() -> None:
    # "A simple uncomment yields valid TOML" — for the WHOLE file, not just
    # the block: the live [providers.*] sections and the explicit
    # [upstreams.*] must coexist (explicit wins over legacy registration).
    lines, begin, end = _example_lines()
    full = lines[:begin] + _uncomment(lines[begin + 1 : end]) + lines[end + 1 :]
    parse_config(tomllib.loads("\n".join(full)), "<config.example.toml uncommented>")


def test_example_config_routing_block_is_commented_out() -> None:
    # Routing stays OFF by default: as committed, the file has no live
    # [routing] table and still parses (the existing 1.0 behavior).
    text = EXAMPLE.read_text(encoding="utf-8")
    assert not re.search(r"^\[routing\]", text, flags=re.M)
    assert not re.search(r"^\[upstreams\.", text, flags=re.M)
    parse_config(tomllib.loads(text), "<config.example.toml>")


@pytest.mark.parametrize("needle", ERROR_STRINGS)
def test_routing_doc_names_every_error_string_verbatim(needle: str) -> None:
    assert needle in DOC.read_text(encoding="utf-8"), needle


# The subset docs/troubleshooting.md carries as its own headed entries
# (the routing.md troubleshooting section is the superset).
TROUBLESHOOTING_STRINGS = (
    "no rule matched and no default_upstream",
    "does not implement count_tokens",
    "budget exhausted for this period",
    "x-llm-redact-reissue: skipped; reason=stateful",
    "edit the file and reload",
    "spend table in the vault database",
    "but not applied",
    "fix the cause and reload (SIGHUP)",
    'on_budget_exhausted never applies: reissue_policy = "never"',
    "not probed (no proxy answered on the configured listener)",
    "names the rule's own upstream",
)


@pytest.mark.parametrize("needle", TROUBLESHOOTING_STRINGS)
def test_troubleshooting_doc_names_the_routing_strings(needle: str) -> None:
    assert needle in TROUBLESHOOTING.read_text(encoding="utf-8"), needle


@pytest.mark.parametrize("phrase", BEHAVIOUR_PHRASES)
def test_routing_doc_describes_shipped_behaviour(phrase: str) -> None:
    assert phrase in DOC.read_text(encoding="utf-8"), phrase


def test_documented_runtime_strings_match_the_code() -> None:
    # The strings above that only the running proxy / CLI emit (no parser
    # probe can raise them) are pinned to their source the doc-sync way:
    # a reworded emitter fails here and points at the stale quote.
    from llm_redact.routes_cli import _state_line
    from llm_redact.routing import MissingCredential, UpstreamConfig, outbound_headers

    assert _state_line({"state": None}, probed=False) == (
        "state:    not probed (no proxy answered on the configured listener)"
    )
    proxy_source = (ROOT / "src" / "llm_redact" / "proxy.py").read_text(encoding="utf-8")
    assert "but not applied ({exc}); fix the cause and" in proxy_source
    assert " reload (SIGHUP)" in proxy_source
    assert "budget exhausted for this period" in proxy_source
    upstream = UpstreamConfig(
        name="keyed",
        protocol="anthropic",
        base_url="https://api.example.com",
        credential="env:EXAMPLE_VANISHED_KEY",
    )
    with pytest.raises(MissingCredential) as excinfo:
        outbound_headers([], upstream, environ={}, oauth_marker="oauth-2025-04-20")
    assert "environment variable EXAMPLE_VANISHED_KEY is unset or empty" in str(excinfo.value)


def test_spend_store_open_failure_is_the_documented_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from llm_redact import proxy as proxy_module

    # A locked vault file, injected the way test_routing_proxy.py does it
    # (never a real lock in the docs suite).
    def _locked(path: Path) -> object:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(proxy_module, "SqliteSpendStore", _locked)
    vault = tmp_path / "vault.db"
    with pytest.raises(ConfigError) as excinfo:
        proxy_module._build_spend_store(VaultConfig(backend="sqlite", path=str(vault)))
    assert str(excinfo.value) == (
        f"spend table in the vault database {vault} could not be opened"
        " (OperationalError: database is locked)"
    )


def test_reissue_policy_never_warnings_are_the_documented_ones() -> None:
    local = {**_LOCAL_UPSTREAM, "monthly_budget_usd": 5, "cost": "metered"}
    config = parse_config(
        {
            "upstreams": {"primary": local, "backup": _LOCAL_UPSTREAM},
            "routing": {
                "enabled": True,
                "default_upstream": "backup",
                "rule": [
                    {
                        "id": "dead-chain",
                        "match": {"protocol": "anthropic"},
                        "upstream": "primary",
                        "reissue_policy": "never",
                        "on_status": {"5xx": ["backup"]},
                        "on_budget_exhausted": ["backup"],
                    }
                ],
            },
        },
        "<never>",
    )
    warnings = list(config.routing.warnings)
    doc = DOC.read_text(encoding="utf-8")
    for fragment in (
        'on_status 5xx never applies: reissue_policy = "never" forbids re-issuing to'
        " another upstream",
        'on_budget_exhausted never applies: reissue_policy = "never" forbids re-issuing'
        " to another upstream",
    ):
        assert any(fragment in warning for warning in warnings), warnings
        # The doc quotes the key generically (KEY) but the tail verbatim.
        assert fragment.split(" never applies: ", 1)[1] in doc, fragment


def test_followup_rule_is_in_both_recommended_configs() -> None:
    # Id-only follow-ups (GET /v1/responses/{id} …) carry no model, so the
    # recommended config pins them with a model-free, path-matched rule —
    # in the doc's live block and the example file's commented block alike.
    needle = 'match    = { protocol = "openai", path = "/v1/responses/*" }'
    assert needle in DOC.read_text(encoding="utf-8")
    lines, begin, end = _example_lines()
    assert needle in "\n".join(_uncomment(lines[begin + 1 : end]))


def test_docs_index_links_routing_doc() -> None:
    # test_docs_index.py enforces this for every doc; pinned here too so the
    # routing doc's own test file fails on its own if the row is dropped.
    assert "(routing.md)" in (ROOT / "docs" / "README.md").read_text(encoding="utf-8")


def test_routing_doc_covers_the_contract_keys() -> None:
    # Every config key the routing contract defines must be documented by
    # name — a key the parser accepts but the reference omits is invisible.
    text = DOC.read_text(encoding="utf-8")
    for key in (
        "default_upstream",
        "max_hops",
        "request_deadline_seconds",
        "plan_limit_detection",
        "plan_limit_headers",
        "oauth_beta_marker",
        "throttle_retry_max_seconds",
        "budget_reset_day",
        "debug_headers",
        "expose_models",
        "model_catalog",
        "inject_system_note",
        "count_tokens",
        "monthly_budget_usd",
        "monthly_budget_tokens",
        "cooldown_seconds",
        "extra_headers",
        "body_defaults",
        "model_rewrite",
        "on_status",
        "reissue_policy",
        "on_budget_exhausted",
        "x-llm-redact-upstream",
        "x-llm-redact-hops",
        "llm_redact_routed_requests_total",
        "llm_redact_reissues_total",
        "routes test",
        "routes list",
        "llm-redact spend",
    ):
        assert key in text, key


# Parser refusals docs/routing.md quotes VERBATIM (its "What serve --check
# refuses" list). Each minimal config must raise ConfigError carrying the
# fragment AND the doc must carry it too, so a reworded parser message
# fails here and points at the stale quote (the test_api_coverage
# doc-sync pattern). Never values: base URLs are RFC-2606 hosts and the
# env var name is a placeholder.
_LOCAL_UPSTREAM = {
    "protocol": "anthropic",
    "base_url": "http://127.0.0.1:11434",
    "credential": "none",
    "cost": "zero",
}
_PASSTHROUGH_UPSTREAM = {"protocol": "anthropic", "base_url": "https://api.example.com"}

CONFIG_ERROR_PROBES = (
    pytest.param(
        {"upstreams": {"local": _LOCAL_UPSTREAM}, "routing": {"enabled": True}},
        "default_upstream is required when enabled = true",
        id="I-6-default-required-when-enabled",
    ),
    pytest.param(
        {
            "upstreams": {"pass": {**_PASSTHROUGH_UPSTREAM, "extra_headers": {"X-Title": "t"}}},
            "routing": {},
        },
        "extra_headers/body_defaults apply only to none/env upstreams",
        id="passthrough-refuses-extra-headers",
    ),
    pytest.param(
        {
            "upstreams": {"pass": {**_PASSTHROUGH_UPSTREAM, "body_defaults": {"temperature": 0}}},
            "routing": {},
        },
        "extra_headers/body_defaults apply only to none/env upstreams",
        id="passthrough-refuses-body-defaults",
    ),
    pytest.param(
        {
            "upstreams": {
                "keyed": {
                    **_PASSTHROUGH_UPSTREAM,
                    "credential": "env:EXAMPLE_API_KEY",
                    "extra_headers": {"x-api-key": "v"},
                }
            },
            "routing": {},
        },
        'credentials come from credential = "env:VAR", never the config file',
        id="extra-headers-may-not-carry-credentials",
    ),
    pytest.param(
        {
            "upstreams": {"local": _LOCAL_UPSTREAM},
            "routing": {"enabled": True, "default_upstream": "local", "plan_limit_headers": {}},
        },
        'set plan_limit_detection = "off" to disable classification instead',
        id="empty-plan-limit-headers",
    ),
    pytest.param(
        {
            "upstreams": {"local": _LOCAL_UPSTREAM, "other": _LOCAL_UPSTREAM},
            "routing": {
                "enabled": True,
                "default_upstream": "local",
                "rule": [
                    {
                        "id": "self",
                        "match": {"protocol": "anthropic"},
                        "upstream": "local",
                        "on_status": {"5xx": ["local"]},
                    }
                ],
            },
        },
        "names the rule's own upstream",
        id="chain-may-not-name-own-upstream",
    ),
    pytest.param(
        {
            "upstreams": {"local": _LOCAL_UPSTREAM, "other": _LOCAL_UPSTREAM},
            "routing": {
                "enabled": True,
                "default_upstream": "local",
                "rule": [
                    {
                        "id": "dup",
                        "match": {"protocol": "anthropic"},
                        "upstream": "local",
                        "on_status": {"5xx": ["other", "other"]},
                    }
                ],
            },
        },
        "twice",
        id="chain-may-not-list-a-member-twice",
    ),
)


@pytest.mark.parametrize(("raw", "fragment"), CONFIG_ERROR_PROBES)
def test_quoted_config_refusals_match_the_parser(raw: dict, fragment: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        parse_config(raw, "<probe>")
    assert fragment in str(excinfo.value), str(excinfo.value)
    assert fragment in DOC.read_text(encoding="utf-8"), fragment


def test_routing_doc_does_not_claim_a_per_protocol_coverage_check() -> None:
    # A protocol with neither a rule nor a default is a RUNTIME 502
    # (no_route), never a serve --check error — the gate cannot know which
    # protocols a tool will send. The parser accepts this config; the doc
    # must not say otherwise (and must say what the gate does require).
    parse_config(
        {
            "upstreams": {"local": _LOCAL_UPSTREAM},
            "routing": {"enabled": True, "default_upstream": "local"},
        },
        "<anthropic default only>",
    )
    text = DOC.read_text(encoding="utf-8")
    assert "empty upstream table" not in text
    assert "does **not** check: per-protocol coverage" in text
