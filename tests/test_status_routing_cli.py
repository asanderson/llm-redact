"""`llm-redact status`'s routing summary + posture lines, and the dashboard's
routing surface — driven by literal /status payloads in the contract's shape
(the proxy is never started here)."""

import argparse
import importlib.resources
from typing import Any

import httpx
import pytest

from llm_redact.cli import _print_posture, _print_routing, _spend_summary, run_status


def _upstream(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "protocol": "anthropic",
        "credential": "passthrough",
        "cost": "metered",
        "legacy": False,
        "state": "healthy",
        "cooldown_remaining_seconds": 0.0,
        "requests": 12,
        "reissues_last_hour": 0,
        "last_error_class": None,
        "last_error_at": None,
        "spend": {
            "period": "2026-09",
            "in_tokens": 0,
            "out_tokens": 0,
            "cache_read": 0,
            "cache_write": 0,
            "usd": 0.0,
            "budget_usd": None,
            "budget_tokens": None,
            "remaining_usd": None,
            "remaining_tokens": None,
            "unpriced_rows": 0,
            "reissue_usd": 0.0,
            "reissue_tokens": 0,
        },
    }
    spend = overrides.pop("spend", {})
    base.update(overrides)
    base["spend"] = {**base["spend"], **spend}
    return base


ROUTING_BLOCK: dict[str, Any] = {
    "enabled": True,
    "default_upstreams": {"anthropic": "ollama"},
    "rules": 5,
    "reissues_last_hour": 2,
    "plan_limit_detection": "headers",
    "expose_models": False,
    "upstreams": {
        "anthropic_oauth": _upstream(spend={"in_tokens": 1000, "out_tokens": 200}),
        "anthropic_key": _upstream(
            credential="env",
            state="cooldown",
            cooldown_remaining_seconds=42.4,
            reissues_last_hour=2,
            last_error_class="plan_limit_429",
            last_error_at="2026-09-21T10:00:00+00:00",
            spend={
                "in_tokens": 10,
                "out_tokens": 5,
                "usd": 1.25,
                "budget_usd": 100.0,
                "remaining_usd": 98.75,
                "unpriced_rows": 3,
            },
        ),
        "openai_key": _upstream(
            protocol="openai",
            credential="env",
            state="budget_exhausted",
            spend={
                "usd": 50.0,
                "budget_usd": 50.0,
                "remaining_usd": 0.0,
                "budget_tokens": 1000,
                "remaining_tokens": 0,
            },
        ),
        "ollama": _upstream(credential="none", cost="zero", legacy=True),
    },
    "unpriced_models": ["muse-64k"],
    "warnings": ["[routing] default_upstream for anthropic is 'x', which is not cost = \"zero\""],
}


def _full_status_payload(routing: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": "1.1.0",
        "uptime_seconds": 1.0,
        "session": "default",
        "vault": {"backend": "sqlite", "entries": 0},
        "detections_total": {},
        "rehydrations_total": {},
        "audit": {"enabled": False, "rows": 0},
        "rehydration": {"fuzzy": True},
        "detection": {"ner_enabled": False},
        "providers": {"anthropic": "https://api.anthropic.com"},
        "license": {"tier": "free", "max_users": 1, "clouds": []},
    }
    if routing is not None:
        payload["routing"] = routing
    return payload


def _status_run(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    args = argparse.Namespace(config=None, port=8787, json=False, ca=None, cert=None, key=None)
    assert run_status(args) == 0


def test_status_prints_routing_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _status_run(monkeypatch, _full_status_payload(ROUTING_BLOCK))
    out = capsys.readouterr().out
    assert (
        "routing: enabled — 5 rules, default: anthropic→ollama, plan-limit detection: headers,"
        " re-issues last hour: 2" in out
    )
    assert (
        "routing upstream[anthropic_oauth]: protocol=anthropic credential=passthrough"
        " cost=metered state=healthy requests=12 spend $0.0000 / 1200 tokens" in out
    )
    assert "routing upstream[anthropic_key]:" in out
    assert "credential=env" in out and "state=cooldown (42s left)" in out
    assert "last error: plan_limit_429 @ 2026-09-21T10:00:00+00:00" in out
    assert "spend $1.2500 / 15 tokens (budget $100.00, $98.75 left) ⚠3 unpriced rows" in out
    assert "routing upstream[openai_key]:" in out and "state=budget_exhausted" in out
    assert "(budget 1000 tokens, 0 left)" in out
    assert "routing upstream[ollama]: protocol=anthropic credential=none cost=zero legacy" in out
    # Posture block carries the honesty lines.
    assert "⚠ routing: upstream(s) in cooldown: anthropic_key (skipped in chains)" in out
    assert "⚠ routing: budget exhausted: openai_key" in out and "402" in out
    assert "⚠ routing: unpriced models: muse-64k" in out
    assert "⚠ routing: [routing] default_upstream for anthropic" in out
    # Mode only: no env var NAME, no value-like token anywhere.
    assert "ANTHROPIC_API_KEY" not in out and "env:" not in out


def test_status_routing_disabled_and_absent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _status_run(monkeypatch, _full_status_payload({"enabled": False}))
    out = capsys.readouterr().out
    assert "routing: disabled" in out
    assert "routing upstream[" not in out
    assert "posture: all traffic redacted" in out
    # A pre-routing proxy omits the block: print nothing, invent nothing.
    _status_run(monkeypatch, _full_status_payload(None))
    out = capsys.readouterr().out
    assert "routing" not in out


def test_posture_names_the_unlisted_unpriced_count(capsys: pytest.CaptureFixture[str]) -> None:
    routing: dict[str, Any] = {
        "enabled": True,
        "upstreams": {"a": _upstream()},
        "unpriced_models": ["muse-64k"],
        "unpriced_models_dropped": 3,
        "warnings": [],
    }
    _print_posture({"detection": {}, "audit": {}, "routing": routing})
    assert "⚠ routing: unpriced models: muse-64k (+3 more not listed)" in capsys.readouterr().out
    # Every observed id was beyond the cap or not id-shaped: still loud.
    routing["unpriced_models"] = []
    _print_posture({"detection": {}, "audit": {}, "routing": routing})
    assert "⚠ routing: unpriced models: none listed (+3 more not listed)" in capsys.readouterr().out
    # A pre-cap /status (no key) prints exactly what it did before.
    del routing["unpriced_models_dropped"]
    _print_posture({"detection": {}, "audit": {}, "routing": routing})
    assert "unpriced" not in capsys.readouterr().out


def test_posture_quiet_when_routing_is_healthy(capsys: pytest.CaptureFixture[str]) -> None:
    _print_posture(
        {
            "detection": {},
            "audit": {},
            "routing": {
                "enabled": True,
                "upstreams": {"a": _upstream(), "b": _upstream(credential="env")},
                "unpriced_models": [],
                "warnings": [],
            },
        }
    )
    assert "all traffic redacted" in capsys.readouterr().out


def test_zero_cost_upstream_with_inert_budget_never_reads_as_exhausted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Decision 11: zero-cost upstreams ignore budgets; the ledger reports
    # remaining_* = None for them, which must not render as "$0.00 left".
    zero = _upstream(credential="none", cost="zero")
    zero["spend"] = dict(
        zero["spend"], budget_usd=5.0, budget_tokens=1000, remaining_usd=None, remaining_tokens=None
    )
    _print_routing({"routing": {"enabled": True, "rules": 1, "upstreams": {"local": zero}}})
    out = capsys.readouterr().out
    assert "(budget ignored: zero-cost)" in out
    assert " left" not in out and "$0.00 left" not in out
    assert _spend_summary({"usd": 0.0, "total_tokens": 3, "budget_usd": 5.0}, zero_cost=True) == (
        "spend $0.0000 / 3 tokens (budget ignored: zero-cost)"
    )
    assert _spend_summary({"usd": 0.0, "total_tokens": 3}, zero_cost=True) == (
        "spend $0.0000 / 3 tokens"
    )
    # A metered upstream at $0 remaining IS exhausted and says so.
    assert "$0.00 left" in _spend_summary(
        {"usd": 5.0, "total_tokens": 3, "budget_usd": 5.0, "remaining_usd": 0.0}
    )


def test_print_routing_tolerates_sparse_upstream_entries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Spend block absent / total_tokens present: both render without KeyError.
    _print_routing(
        {
            "routing": {
                "enabled": True,
                "rules": 0,
                "upstreams": {
                    "bare": {"protocol": "openai", "credential": "none", "cost": "zero"},
                    "totalled": {
                        "protocol": "openai",
                        "credential": "env",
                        "cost": "metered",
                        "state": "healthy",
                        "spend": {"usd": 0.5, "total_tokens": 77},
                    },
                },
            }
        }
    )
    out = capsys.readouterr().out
    assert "routing: enabled — 0 rules, default: none" in out
    assert "routing upstream[bare]: protocol=openai credential=none cost=zero state=?" in out
    assert "routing upstream[totalled]:" in out and "spend $0.5000 / 77 tokens" in out


def test_dashboard_carries_routing_pill_and_table() -> None:
    html = importlib.resources.files("llm_redact").joinpath("dashboard.html").read_text("utf-8")
    assert 'id="routing"' in html  # the pill
    assert 'id="routing-card"' in html and 'id="routing-upstreams"' in html
    assert 'id="routing-warnings"' in html
    assert "function renderRouting" in html and "renderRouting(s.routing" in html
    # Per-upstream columns the contract names, and the budget-exhausted /
    # cooldown states surfaced loudly.
    for column in ("upstream", "protocol", "credential", "cost", "state", "budget"):
        assert f"<th>{column}</th>" in html or f'<th class="num">{column}</th>' in html
    assert "budget_exhausted" in html and "cooldown" in html
    assert "unpriced_models" in html and "unpriced_models_dropped" in html
    # The budget cell keys zero-cost off the upstream's `cost` field (the
    # /status spend block carries no zero_cost key; remaining_* is null).
    assert 'routingBudget(u.spend, u.cost === "zero")' in html
    assert "if (zeroCost || spend.zero_cost)" in html
    # Self-contained and textContent-only (no innerHTML anywhere on the page).
    assert "innerHTML" not in html
    assert "http://" not in html and "https://" not in html
