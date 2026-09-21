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

from llm_redact.config import ConfigError, parse_config

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "routing.md"
EXAMPLE = ROOT / "config.example.toml"

_FENCE_RE = re.compile(r"^```toml\n(.*?)^```$", re.M | re.S)
_BEGIN = "## ROUTING-EXAMPLE-BEGIN"
_END = "## ROUTING-EXAMPLE-END"

# Strings the proxy emits (proxy-generated replies and response headers);
# docs/routing.md's troubleshooting section is keyed by them verbatim.
ERROR_STRINGS = (
    "no rule matched and no default_upstream",
    "does not implement count_tokens",
    "budget exhausted",
    "x-llm-redact-reissue: skipped; reason=stateful",
    "x-llm-redact-reissue: skipped; reason=no-candidate",
    "edit the file and reload",
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
