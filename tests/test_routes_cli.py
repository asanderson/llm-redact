"""`llm-redact routes list|test` and `llm-redact spend`: offline, value-free."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from llm_redact import routes_cli
from llm_redact.cli import main
from llm_redact.pricing import PriceTable, Usage
from llm_redact.routes_cli import (
    DEFAULT_TEST_PATHS,
    budgets_for,
    build_price_table,
    configured_models,
    model_from_gemini_path,
    run_routes_list,
    run_routes_test,
    run_spend,
    unpriced_models,
)
from llm_redact.spend import SpendRow, SqliteSpendStore, format_ts

FAKE_KEY = "sk-ant-test-not-a-real-key"

ROUTED = """
[vault]
backend = "sqlite"
path = "{vault}"

[providers.anthropic]
upstream_base_url = "https://api.anthropic.com"

[upstreams.anthropic_key]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
credential = "env:ROUTES_TEST_SECRET_VAR"
monthly_budget_usd = 100

[upstreams.ollama_local]
protocol = "anthropic"
base_url = "http://127.0.0.1:11434"
credential = "none"
cost = "zero"
count_tokens = false

[upstreams.openai_key]
protocol = "openai"
base_url = "https://api.openai.com/v1"
credential = "env:ROUTES_TEST_SECRET_VAR"

[upstreams.gemini_key]
protocol = "gemini"
base_url = "https://generativelanguage.example"
credential = "env:ROUTES_TEST_SECRET_VAR"

[routing]
enabled = true
default_upstream = {{ anthropic = "ollama_local" }}

[[routing.rule]]
id = "explore-local"
match = {{ protocol = "anthropic", headers = {{ "X-Claude-Code-Agent-Id" = "Explore" }} }}
upstream = "ollama_local"
model_rewrite = "muse-64k"

[[routing.rule]]
id = "max-lane"
match = {{ protocol = "anthropic", model = "claude-*", auth = "oauth" }}
upstream = "anthropic"
on_status = {{ plan_limit_429 = ["anthropic_key", "ollama_local"], throttle_429 = "retry-same" }}

[[routing.rule]]
id = "key-lane"
match = {{ protocol = "anthropic", model = "claude-*", auth = "gateway-key" }}
upstream = "anthropic_key"
on_status = {{ 5xx = ["ollama_local"] }}
on_budget_exhausted = ["ollama_local"]
reissue_policy = "always"

[[routing.rule]]
id = "gemini-flash"
match = {{ protocol = "gemini", model = "gemini-2.5-flash*" }}
upstream = "gemini_key"
"""


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """`routes test`'s loopback /status probe is stubbed: no test here may
    touch a socket, and a proxy that happens to run on 8787 must not
    change the output. Returns the list of configs the probe saw."""
    seen: list[Any] = []

    def none(config: Any) -> dict[str, dict[str, Any]]:
        seen.append(config)
        return {}

    monkeypatch.setattr(routes_cli, "_status_probe", none)
    return seen


def _live(monkeypatch: pytest.MonkeyPatch, states: dict[str, dict[str, Any]]) -> None:
    monkeypatch.setattr(routes_cli, "_status_probe", lambda config: states)


@pytest.fixture
def routed_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    # A secret-shaped value IS set: the CLIs must never read or print it.
    monkeypatch.setenv("ROUTES_TEST_SECRET_VAR", FAKE_KEY)
    path = tmp_path / "config.toml"
    path.write_text(ROUTED.format(vault=(tmp_path / "vault.db").as_posix()))
    return path


def _list_args(config: Path | None, json_mode: bool = False) -> argparse.Namespace:
    return argparse.Namespace(config=config, json=json_mode)


def _test_args(config: Path | None, **overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = dict(
        config=config,
        protocol="anthropic",
        model=None,
        header=None,
        path=None,
        auth="any",
        json=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _spend_args(config: Path | None, **overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = dict(config=config, month=None, json=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---- routes list ----


def test_routes_list_table(routed_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_routes_list(_list_args(routed_config)) == 0
    out = capsys.readouterr().out
    assert "routing: enabled — 4 rule(s)" in out
    assert "anthropic→ollama_local" in out
    for column in ("id", "protocol", "match", "upstream", "on_status", "reissue", "model_rewrite"):
        assert column in out
    assert "explore-local" in out and "x-claude-code-agent-id=Explore" in out
    assert "plan_limit_429=anthropic_key>ollama_local" in out
    assert "throttle_429=retry-same" in out
    assert "muse-64k" in out
    assert "stateless-only" in out and "always" in out
    assert "⚠" not in out  # a zero-cost default: nothing to warn about
    assert FAKE_KEY not in out and "ROUTES_TEST_SECRET_VAR" not in out


def test_routes_list_surfaces_config_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[upstreams.key]\nprotocol = "anthropic"\nbase_url = "https://api.anthropic.example"\n'
        'credential = "env:ROUTES_TEST_OTHER_VAR"\n'
        '[routing]\nenabled = true\ndefault_upstream = "key"\n'
    )
    assert run_routes_list(_list_args(config)) == 0
    out = capsys.readouterr().out
    assert "⚠" in out and 'not cost = "zero"' in out
    assert "ROUTES_TEST_OTHER_VAR" not in out


def test_referenced_upstreams_skips_unreferenced_legacy(routed_config: Path) -> None:
    from llm_redact.config import load_config
    from llm_redact.routes_cli import referenced_upstreams

    routing = load_config(routed_config).routing
    names = [u.name for u in referenced_upstreams(routing)]
    # Explicit ones always; legacy "anthropic" is a rule target; legacy
    # openai/gemini/ollama (built-in defaults) are reachable by nothing.
    assert names == ["anthropic", "anthropic_key", "gemini_key", "ollama_local", "openai_key"]


def test_routes_list_json(routed_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_routes_list(_list_args(routed_config, json_mode=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["enabled"] and data["present"]
    assert data["default_upstreams"] == {"anthropic": "ollama_local"}
    assert [r["id"] for r in data["rules"]] == [
        "explore-local",
        "max-lane",
        "key-lane",
        "gemini-flash",
    ]
    key_lane = data["rules"][2]
    assert key_lane["on_status"] == {"5xx": ["ollama_local"]}
    assert key_lane["on_budget_exhausted"] == ["ollama_local"]
    by_name = {u["name"]: u for u in data["upstreams"]}
    assert by_name["anthropic_key"]["credential"] == "env"  # mode only
    assert by_name["anthropic"]["legacy"] and by_name["anthropic"]["credential"] == "passthrough"
    assert by_name["ollama_local"]["count_tokens"] is False
    text = json.dumps(data)
    assert FAKE_KEY not in text and "ROUTES_TEST_SECRET_VAR" not in text


def test_routes_list_without_routing_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("port = 8787\n")
    assert run_routes_list(_list_args(config)) == 0
    out = capsys.readouterr().out
    assert "not configured" in out and "no [routing] section" in out


def test_routes_list_disabled_without_rules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[routing]\nenabled = false\n")
    assert run_routes_list(_list_args(config)) == 0
    out = capsys.readouterr().out
    assert "routing: disabled" in out
    assert "no rules" in out


def test_routes_list_config_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[routing]\nenabled = true\n")  # I-6: no default_upstream
    assert run_routes_list(_list_args(config)) == 1
    assert "config:" in capsys.readouterr().out


# ---- routes test ----


def test_routes_test_matches_header_rule_with_model_rewrite(
    routed_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _test_args(
        routed_config, model="claude-sonnet-5", header=["x-claude-code-agent-id=Explore"]
    )
    assert run_routes_test(args) == 0
    out = capsys.readouterr().out
    assert "matched:  rule explore-local" in out
    assert "upstream: ollama_local (protocol=anthropic credential=none cost=zero)" in out
    assert "model_rewrite: muse-64k" in out
    assert "on_status: none" in out
    assert "state:    not probed (no proxy answered" in out  # no running proxy


def test_routes_test_annotates_live_cooldown_and_budget_state(
    routed_config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A running proxy answered /status: the chain a re-issue would ACTUALLY
    # take is readable (cooldown / exhausted members are skipped at runtime).
    _live(
        monkeypatch,
        {
            "anthropic": {
                "state": "cooldown",
                "cooldown_remaining_seconds": 41.6,
                "last_error_class": "plan_limit_429",
            },
            "anthropic_key": {
                "state": "budget_exhausted",
                "cooldown_remaining_seconds": 0.0,
                "last_error_class": None,
            },
            "ollama_local": {
                "state": "healthy",
                "cooldown_remaining_seconds": 0.0,
                "last_error_class": None,
            },
        },
    )
    assert run_routes_test(_test_args(routed_config, model="claude-opus-5", auth="oauth")) == 0
    out = capsys.readouterr().out
    assert "state:    cooldown (42s left) last error: plan_limit_429" in out
    assert (
        "on plan_limit_429: anthropic_key (budget exhausted — skipped, budgeted)"
        " > ollama_local (zero-cost, no count_tokens)" in out
    )
    assert (
        run_routes_test(
            _test_args(routed_config, model="claude-opus-5", auth="gateway-key", json=True)
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["live"] is True
    assert data["state"] == {
        "state": "budget_exhausted",
        "cooldown_remaining_seconds": 0.0,
        "last_error_class": None,
    }
    # A member the proxy does not know (config edited, not reloaded) is unannotated.
    _live(monkeypatch, {"other": {"state": "cooldown"}})
    assert run_routes_test(_test_args(routed_config, model="claude-opus-5", auth="oauth")) == 0
    out = capsys.readouterr().out
    assert "state:    unknown (the running proxy reports no such upstream" in out
    assert "on plan_limit_429: anthropic_key (budgeted) > ollama_local" in out


def test_probe_live_states_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from llm_redact.config import parse_config
    from llm_redact.routes_cli import probe_live_states

    calls: list[str] = []

    def answer(url: str, **kwargs: Any) -> httpx.Response:
        calls.append(url)
        if answer.mode == "refused":  # type: ignore[attr-defined]
            raise httpx.ConnectError("no proxy")
        if answer.mode == "garbage":  # type: ignore[attr-defined]
            return httpx.Response(200, content=b"<html>", request=httpx.Request("GET", url))
        if answer.mode == "disabled":  # type: ignore[attr-defined]
            body: dict[str, Any] = {"routing": {"enabled": False}}
        elif answer.mode == "sparse":  # type: ignore[attr-defined]
            body = {"routing": {"enabled": True, "upstreams": {"a": {"state": "cooldown"}, "b": 1}}}
        else:
            body = {"routing": {"enabled": True, "upstreams": "nope"}}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", answer)
    config = parse_config({"port": 8790}, "<test>")
    for mode in ("refused", "garbage", "disabled", "shape"):
        answer.mode = mode  # type: ignore[attr-defined]
        assert probe_live_states(config) == {}
    answer.mode = "sparse"  # type: ignore[attr-defined]
    assert probe_live_states(config) == {"a": {"state": "cooldown"}}
    assert calls and all(u == "http://127.0.0.1:8790/__llm-redact/status" for u in calls)
    # A TLS listener needs client certs: not probed at all (no call made).
    tls_config = parse_config(
        {"tls": {"certfile": str(tmp_path / "c.pem"), "keyfile": str(tmp_path / "k.pem")}},
        "<test>",
    )
    before = len(calls)
    assert probe_live_states(tls_config) == {}
    assert len(calls) == before


def test_routes_test_auth_and_chain_annotations(
    routed_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_routes_test(_test_args(routed_config, model="claude-opus-5", auth="oauth")) == 0
    out = capsys.readouterr().out
    assert "matched:  rule max-lane" in out
    assert "credential=passthrough" in out and "legacy" in out
    assert (
        "on plan_limit_429: anthropic_key (budgeted) > ollama_local (zero-cost, no count_tokens)"
        in out
    )
    assert "on throttle_429: retry-same" in out
    assert "reissue:  stateless-only" in out
    assert FAKE_KEY not in out and "ROUTES_TEST_SECRET_VAR" not in out

    assert (
        run_routes_test(_test_args(routed_config, model="claude-opus-5", auth="gateway-key")) == 0
    )
    out = capsys.readouterr().out
    assert "matched:  rule key-lane" in out
    assert "credential=env" in out
    assert "on budget exhausted: ollama_local" in out
    assert "reissue:  always" in out


def test_routes_test_auth_any_matches_every_auth_constraint(
    routed_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # auth="any" on the query side is a wildcard: the FIRST auth-constrained
    # rule wins (max-lane precedes key-lane in file order).
    assert run_routes_test(_test_args(routed_config, model="claude-opus-5")) == 0
    assert "rule max-lane" in capsys.readouterr().out


def test_routes_test_default_and_count_tokens(
    routed_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _test_args(routed_config, model="gpt-oss", path="/v1/messages/count_tokens")
    assert run_routes_test(args) == 0
    out = capsys.readouterr().out
    assert "matched:  default_upstream (no rule matched)" in out
    assert "upstream: ollama_local" in out
    assert "count_tokens: this upstream does not implement it — the proxy answers 404" in out
    assert "reissue:" not in out  # no rule, no rule-level fields


def test_routes_test_no_route(routed_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_routes_test(_test_args(routed_config, protocol="openai", model="gpt-5")) == 0
    out = capsys.readouterr().out
    assert "NO ROUTE" in out and "no default_upstream for protocol openai" in out
    assert "502" in out


def test_routes_test_gemini_model_from_path(
    routed_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Decision 15b: --path without --model derives the id from the path.
    args = _test_args(
        routed_config,
        protocol="gemini",
        path="/v1beta/models/gemini-2.5-flash-lite:streamGenerateContent",
        json=True,
    )
    assert run_routes_test(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model"] == "gemini-2.5-flash-lite"
    assert data["rule"] == "gemini-flash" and data["via"] == "rule"
    assert data["upstream"]["name"] == "gemini_key" and data["upstream"]["credential"] == "env"
    # Without --path the default path is filled with the --model (or a stand-in).
    assert (
        run_routes_test(_test_args(routed_config, protocol="gemini", model="gemini-2.5-pro")) == 0
    )
    out = capsys.readouterr().out
    assert "path=/v1beta/models/gemini-2.5-pro:generateContent" in out
    assert "NO ROUTE" in out  # gemini has no default and the rule wants flash


def test_routes_test_gemini_without_model_or_path_derives_the_stand_in(
    routed_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Neither flag: the proxy, given the printed path and a body without
    # `model`, derives gemini-2.5-flash and matches the flash rule — so must
    # the dry-run (model derived AFTER the stand-in fills the path).
    assert run_routes_test(_test_args(routed_config, protocol="gemini", json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["path"] == "/v1beta/models/gemini-2.5-flash:generateContent"
    assert data["model"] == "gemini-2.5-flash"
    assert data["rule"] == "gemini-flash" and data["upstream"]["name"] == "gemini_key"


def test_routes_test_strips_the_query_from_path(
    routed_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Rules glob request.url.path: `?alt=sse` is not part of it, and a
    # pasted `?key=` must neither match nor print.
    args = _test_args(
        routed_config,
        protocol="gemini",
        path="/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=q-secret",
    )
    assert run_routes_test(args) == 0
    out = capsys.readouterr().out
    assert "path=/v1beta/models/gemini-2.5-flash:streamGenerateContent auth" in out
    assert "q-secret" not in out and "alt=sse" not in out
    assert "rule gemini-flash" in out


def test_model_from_gemini_path_is_the_proxys_derivation() -> None:
    # Decision 15b: ONE derivation shared with proxy._plan_route.
    from llm_redact.proxy import gemini_path_model

    cases = (
        "/v1beta/models/gemini-2.5-flash:generateContent",
        "/v1beta/tunedModels/my-tuned-1:generateContent",
        "/v1/models/gemini-2.5-pro:countTokens",
        "/v1/projects/p/locations/l/publishers/google/models/g-1:countTokens",
        "/x/models/a:b/models/c:d",
        "/v1beta/models",
        "/v1/messages",
        "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
    )
    for path in cases:
        assert model_from_gemini_path(path) == gemini_path_model(path), path
    assert model_from_gemini_path("/v1beta/models/gemini-2.5-flash:generateContent") == (
        "gemini-2.5-flash"
    )
    # tunedModels is a routed Gemini form; Vertex publisher paths are never
    # routed (decision 1); an unanchored `/models/` is not a Gemini path.
    assert model_from_gemini_path("/v1beta/tunedModels/my-tuned-1:generateContent") == (
        "my-tuned-1"
    )
    assert (
        model_from_gemini_path(
            "/v1/projects/p/locations/l/publishers/google/models/g-1:countTokens"
        )
        is None
    )
    assert model_from_gemini_path("/x/models/a:b/models/c:d") is None
    assert model_from_gemini_path("/v1beta/models") is None
    assert model_from_gemini_path("/v1/messages") is None
    assert "{model}" in DEFAULT_TEST_PATHS["gemini"]


def test_routes_test_rejects_bad_header(
    routed_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_routes_test(_test_args(routed_config, header=["no-equals-sign"])) == 2
    assert "NAME=VALUE" in capsys.readouterr().out


def test_routes_test_config_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[routing]\nenabled = true\n")
    assert run_routes_test(_test_args(config)) == 1


def test_routes_test_dry_runs_a_disabled_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[upstreams.local]\nprotocol = "anthropic"\nbase_url = "http://127.0.0.1:11434"\n'
        'credential = "none"\ncost = "zero"\n'
        '[routing]\nenabled = false\ndefault_upstream = "local"\n'
    )
    assert run_routes_test(_test_args(config, model="claude-x")) == 0
    out = capsys.readouterr().out
    assert "routing: disabled" in out and "WOULD do once enabled" in out
    assert "default_upstream (no rule matched)" in out


def test_routes_test_needs_no_env_credentials(
    routed_config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ROUTES_TEST_SECRET_VAR")
    assert run_routes_test(_test_args(routed_config, model="claude-x", auth="gateway-key")) == 0
    out = capsys.readouterr().out
    assert "rule key-lane" in out and "credential=env" in out
    assert "ROUTES_TEST_SECRET_VAR" not in out


# ---- price helpers shared with doctor ----


def test_configured_and_unpriced_models(routed_config: Path) -> None:
    from llm_redact.config import load_config

    config = load_config(routed_config)
    # Glob-free rule models: none here; the one model_rewrite target
    # (muse-64k) only ever reaches ollama_local, which is zero-cost —
    # budgets are ignored there, so its missing price is no budgeting gap.
    assert configured_models(config.routing) == []
    table = build_price_table(config.prices)
    assert isinstance(table, PriceTable) and len(table) > 0
    assert unpriced_models(config.routing, table) == []
    # The same rewrite on a METERED upstream is a real gap.
    metered = load_config(routed_config)
    text = routed_config.read_text().replace(
        'upstream = "ollama_local"\nmodel_rewrite = "muse-64k"',
        'upstream = "anthropic_key"\nmodel_rewrite = "muse-64k"',
    )
    assert text != routed_config.read_text()
    routed_config.write_text(text)
    metered = load_config(routed_config)
    assert configured_models(metered.routing) == ["muse-64k"]
    assert unpriced_models(metered.routing, table) == ["muse-64k"]
    priced = table.with_overrides({"muse-64k": table.lookup("gpt-5") or table.lookup("gpt-4o")})  # type: ignore[dict-item]
    assert unpriced_models(metered.routing, priced) == []


def test_price_table_and_budgets_are_the_proxys(routed_config: Path) -> None:
    # One implementation: `spend`/doctor and the ledger behind /status can
    # never disagree on the table or on which upstreams carry a Budget.
    from llm_redact import proxy
    from llm_redact.config import load_config

    config = load_config(routed_config)
    assert budgets_for(config.routing) == proxy._budgets_for(config.routing)
    assert set(budgets_for(config.routing)) == set(config.routing.upstream_names())
    ours, theirs = build_price_table(config.prices), proxy._build_price_table(config.prices)
    assert len(ours) == len(theirs) and ours.lookup("gpt-5") == theirs.lookup("gpt-5")


def test_build_price_table_from_file_and_overrides(tmp_path: Path) -> None:
    from llm_redact.config import parse_config

    table_file = tmp_path / "prices.json"
    table_file.write_text(
        json.dumps(
            {"models": {"muse-64k": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}}}
        )
    )
    config = parse_config(
        {
            "prices": {
                "table": str(table_file),
                "override": {"other": {"input": 1.0, "output": 2.0}},
            }
        },
        "<test>",
    )
    table = build_price_table(config.prices)
    assert table.lookup("muse-64k") is not None
    assert table.lookup("other") is not None and table.lookup("other").output == 2.0  # type: ignore[union-attr]
    assert table.lookup("gpt-5") is None  # the file REPLACES the builtin table


# ---- spend ----


def test_spend_memory_backend_is_in_process_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[vault]\nbackend = "memory"\n')
    assert run_spend(_spend_args(config)) == 0
    out = capsys.readouterr().out
    assert "in-process only" in out and "memory" in out and "llm-redact status" in out
    assert run_spend(_spend_args(config, json=True)) == 0
    assert json.loads(capsys.readouterr().out)["backend"] == "memory"


def test_spend_sqlite_without_vault_file(
    routed_config: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert run_spend(_spend_args(routed_config)) == 0
    out = capsys.readouterr().out
    assert "no spend recorded yet" in out
    assert not (tmp_path / "vault.db").exists()  # a report never creates the vault
    assert run_spend(_spend_args(routed_config, json=True)) == 0
    assert "note" in json.loads(capsys.readouterr().out)


def _seed_spend(path: Path, now: datetime) -> None:
    store = SqliteSpendStore(path)
    try:
        store.record(
            SpendRow(
                ts=format_ts(now),
                upstream="anthropic_key",
                model="claude-sonnet-5",
                hop=1,
                usage=Usage(input_tokens=1000, output_tokens=500, cache_read=100, cache_write=10),
                usd=0.5,
            )
        )
        store.record(
            SpendRow(
                ts=format_ts(now),
                upstream="anthropic_key",
                model="claude-sonnet-5",
                hop=2,
                usage=Usage(input_tokens=200, output_tokens=100),
                usd=0.25,
            )
        )
        store.record(
            SpendRow(
                ts=format_ts(now),
                upstream="ollama_local",
                model="muse-64k",
                hop=1,
                usage=Usage(input_tokens=50, output_tokens=20),
                usd=None,
            )
        )
    finally:
        store.close()


def test_spend_report_from_sqlite(
    routed_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_spend(tmp_path / "vault.db", datetime.now(tz=UTC))
    assert run_spend(_spend_args(routed_config)) == 0
    out = capsys.readouterr().out
    assert "(current period)" in out
    assert "anthropic_key: in=1200 out=600 cache_read=100 cache_write=10 usd=$0.7500" in out
    assert "reissue=1 rows (33% of spend)" in out
    assert "$99.25 of $100.00 left" in out
    assert "ollama_local:" in out and "usd=unpriced" in out and "zero-cost (ignored)" in out
    # Legacy passthrough with no rows this period: $0, not "unpriced".
    assert "anthropic: in=0 out=0 cache_read=0 cache_write=0 usd=$0.0000 rows=0" in out
    assert "anthropic: " in out and "budget: none" in out  # legacy passthrough: no budget
    assert "total:" in out and "1 unpriced rows" in out
    assert FAKE_KEY not in out


def test_spend_partially_priced_upstream_counts_both(
    routed_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = SqliteSpendStore(tmp_path / "vault.db")
    try:
        for usd in (0.5, None):
            store.record(
                SpendRow(
                    ts=format_ts(datetime.now(tz=UTC)),
                    upstream="anthropic_key",
                    model="m",
                    hop=1,
                    usage=Usage(input_tokens=1),
                    usd=usd,
                )
            )
    finally:
        store.close()
    assert run_spend(_spend_args(routed_config)) == 0
    assert "usd=$0.5000 (+1 unpriced rows)" in capsys.readouterr().out


def test_spend_is_read_only_on_a_legacy_vault(
    routed_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import sqlite3

    # A vault the proxy never recorded spend into: only the vault table.
    vault = tmp_path / "vault.db"
    conn = sqlite3.connect(vault)
    conn.execute("CREATE TABLE vault (k TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()
    assert run_spend(_spend_args(routed_config)) == 0
    assert "no spend recorded yet" in capsys.readouterr().out
    assert run_spend(_spend_args(routed_config, json=True)) == 0
    assert "no spend table" in json.loads(capsys.readouterr().out)["note"]
    conn = sqlite3.connect(vault)
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert names == {"vault"}  # no DDL from a read-only report
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] != "wal"
    conn.close()
    assert not (tmp_path / "vault.db-wal").exists()
    # A 0400 backup copy (unwritable file AND directory) still reads.
    _seed_spend(vault, datetime.now(tz=UTC))
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    backup = backup_dir / "vault-copy.db"
    backup.write_bytes(vault.read_bytes())
    backup.chmod(0o400)
    backup_dir.chmod(0o500)
    try:
        routed_config.write_text(
            routed_config.read_text().replace(vault.as_posix(), backup.as_posix())
        )
        assert run_spend(_spend_args(routed_config)) == 0
        assert "anthropic_key: in=1200" in capsys.readouterr().out
    finally:
        backup_dir.chmod(0o700)
        backup.chmod(0o600)


def test_read_only_store_falls_back_to_immutable_when_shm_cannot_be_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A WAL-mode backup copy in a directory the user cannot write to: the
    mode=ro probe fails (SQLite cannot create the -shm side file) and the
    store reopens immutable=1. Root bypasses the permission check, so the
    trigger is simulated deterministically rather than via chmod."""
    import sqlite3

    vault = tmp_path / "vault.db"
    _seed_spend(vault, datetime.now(tz=UTC))
    real_connect = sqlite3.connect
    uris: list[str] = []

    class _ShmDenied:
        def execute(self, sql: str, *args: object) -> "_ShmDenied":
            if "sqlite_master" in sql:
                raise sqlite3.OperationalError("unable to open database file")
            return self

        def fetchone(self) -> None:
            return None

        def close(self) -> None:
            pass

    def fake_connect(database: str, *args: object, **kwargs: object) -> object:
        uris.append(database)
        if "immutable=1" in database:
            return real_connect(database, *args, **kwargs)
        return _ShmDenied()

    monkeypatch.setattr(routes_cli.sqlite3, "connect", fake_connect)
    store = routes_cli.ReadOnlySpendStore(vault)
    try:
        assert store.has_table
        base = f"{vault.absolute().as_uri()}?mode=ro"
        assert uris == [base, f"{base}&immutable=1"]
    finally:
        store.close()


def test_spend_not_a_sqlite_file_is_an_error_not_a_traceback(
    routed_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "vault.db").write_bytes(b"this is not a sqlite database at all\n" * 40)
    assert run_spend(_spend_args(routed_config)) == 1
    out = capsys.readouterr().out
    # The read-only open probes the file once, so a non-database fails at
    # open time — named by exception TYPE, never a traceback.
    assert out.startswith("spend: cannot open") and "DatabaseError" in out


def test_spend_json_and_past_month(
    routed_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_spend(tmp_path / "vault.db", datetime.now(tz=UTC))
    assert run_spend(_spend_args(routed_config, json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["current"] is True
    entry = data["upstreams"]["anthropic_key"]
    assert entry["hops"]["reissue"]["rows"] == 1 and entry["remaining_usd"] == 99.25
    assert entry["exhausted"] is False
    assert data["upstreams"]["ollama_local"]["zero_cost"] is True
    # A past month: nothing there.
    assert run_spend(_spend_args(routed_config, month="2020-01", json=True)) == 0
    past = json.loads(capsys.readouterr().out)
    assert past["period"] == "2020-01" and past["current"] is False
    assert all(u["rows"] == 0 for u in past["upstreams"].values())


def test_spend_exhausted_budget_is_loud(
    routed_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = SqliteSpendStore(tmp_path / "vault.db")
    try:
        store.record(
            SpendRow(
                ts=format_ts(datetime.now(tz=UTC)),
                upstream="anthropic_key",
                model="claude-sonnet-5",
                hop=1,
                usage=Usage(input_tokens=1),
                usd=100.0,
            )
        )
    finally:
        store.close()
    assert run_spend(_spend_args(routed_config)) == 0
    assert "⚠ EXHAUSTED" in capsys.readouterr().out


def test_spend_rejects_bad_month(
    routed_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_spend(tmp_path / "vault.db", datetime.now(tz=UTC))
    assert run_spend(_spend_args(routed_config, month="September")) == 2
    assert "YYYY-MM" in capsys.readouterr().out


def test_spend_config_error_and_rdbms_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[routing]\nenabled = true\n")
    assert run_spend(_spend_args(config)) == 1
    config.write_text(
        '[vault]\nbackend = "postgresql"\n[vault.rdbms]\ndsn = "postgresql://db.example/v"\n'
    )
    assert run_spend(_spend_args(config)) == 0
    assert "postgresql vault backend" in capsys.readouterr().out


def test_spend_disabled_routing_still_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    vault = tmp_path / "vault.db"
    config.write_text(f'[vault]\nbackend = "sqlite"\npath = "{vault.as_posix()}"\n')
    _seed_spend(vault, datetime.now(tz=UTC))
    assert run_spend(_spend_args(config)) == 0
    out = capsys.readouterr().out
    assert "routing: not configured" in out
    assert "anthropic_key:" in out  # rows are reported even without budgets


# ---- argparse wiring ----


def test_main_dispatches_routes_and_spend(
    routed_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["routes", "list", "--config", str(routed_config)])
    assert excinfo.value.code == 0
    assert "explore-local" in capsys.readouterr().out
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "routes",
                "test",
                "--config",
                str(routed_config),
                "--protocol",
                "anthropic",
                "--model",
                "claude-sonnet-5",
                "--header",
                "x-claude-code-agent-id=Explore",
                "--auth",
                "oauth",
            ]
        )
    assert excinfo.value.code == 0
    assert "rule explore-local" in capsys.readouterr().out
    with pytest.raises(SystemExit) as excinfo:
        main(["spend", "--config", str(routed_config), "--json"])
    assert excinfo.value.code == 0
    with pytest.raises(SystemExit):
        main(["routes", "test", "--config", str(routed_config)])  # --protocol is required
