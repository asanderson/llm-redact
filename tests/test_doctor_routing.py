"""`llm-redact doctor`'s routing check: value-free, network-free."""

import argparse
import re
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_redact.doctor_cli import _display_url, _probe_upstream, run_doctor

FAKE_KEY = "sk-ant-test-not-a-real-key"


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _args(config: Path, **overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = dict(config=config, json=False, offline=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


ROUTED = """
port = {port}
[vault]
backend = "{backend}"
path = "{vault}"

[upstreams.anthropic_key]
protocol = "anthropic"
base_url = "https://api.anthropic.example/?key=querysecret"
credential = "env:DOCTOR_TEST_SECRET_VAR"
monthly_budget_usd = 10

[upstreams.local]
protocol = "anthropic"
base_url = "http://127.0.0.1:{ollama}"
credential = "none"
cost = "zero"

[routing]
enabled = true
default_upstream = "{default}"
model_catalog = ["claude-sonnet-5"]

[[routing.rule]]
id = "lane"
match = {{ protocol = "anthropic", model = ["claude-*", "{literal}"] }}
upstream = "anthropic_key"
on_status = {{ 5xx = ["local"] }}
"""


def _write(
    tmp_path: Path,
    *,
    backend: str = "sqlite",
    default: str = "local",
    literal: str = "claude-opus-5",
) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        ROUTED.format(
            port=_free_port(),
            backend=backend,
            vault=(tmp_path / "vault.db").as_posix(),
            ollama=_free_port(),
            default=default,
            literal=literal,
        )
    )
    return config


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict[str, Any]]]:
    """Every httpx.request the probe makes (method, url, kwargs); the default
    fake refuses the connection. No test here may touch the network."""
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        calls.append((method, url, kwargs))
        raise httpx.ConnectError("probe-detail-must-not-print")

    monkeypatch.setattr(httpx, "request", fake_request)
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    monkeypatch.setenv("DOCTOR_TEST_SECRET_VAR", FAKE_KEY)
    return calls


def _routing_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if "  routing: " in line]


def test_routing_disabled_is_one_pass_line(
    tmp_path: Path, probes: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(f"port = {_free_port()}\n")
    assert run_doctor(_args(config)) == 0
    lines = _routing_lines(capsys.readouterr().out)
    assert len(lines) == 1 and lines[0].startswith("PASS") and "not configured" in lines[0]
    assert probes == []

    config.write_text(f"port = {_free_port()}\n[routing]\nenabled = false\n")
    assert run_doctor(_args(config)) == 0
    lines = _routing_lines(capsys.readouterr().out)
    assert len(lines) == 1 and "disabled" in lines[0]
    assert probes == []


def test_routing_enabled_healthy(
    tmp_path: Path,
    probes: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def ok(method: str, url: str, **kwargs: Any) -> httpx.Response:
        probes.append((method, url, kwargs))
        return httpx.Response(401, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", ok)
    config = _write(tmp_path)
    assert run_doctor(_args(config)) == 0
    out = capsys.readouterr().out
    lines = _routing_lines(out)
    # Explicit upstreams plus the legacy auto-registered providers.
    assert any(re.search(r"config valid: \d+ upstreams, 1 rules", line) for line in lines)
    assert any(
        "env credentials resolve for anthropic_key" in line and line.startswith("PASS")
        for line in lines
    )
    assert any("price table covers" in line for line in lines)
    assert any(
        "upstream anthropic_key answers at https://api.anthropic.example" in line for line in lines
    )
    assert any("upstream local answers at http://127.0.0.1:" in line for line in lines)
    assert not any("memory" in line for line in lines)
    # Only REACHABLE upstreams are probed: the two the rule/chain/default name,
    # not the unreferenced legacy providers the built-in defaults register.
    assert not any("upstream openai" in line for line in lines)
    # HEAD answered (401 counts), so GET was never needed; no auth, 3 s cap.
    assert [m for m, _u, _k in probes] == ["HEAD", "HEAD"]
    for _method, _url, kwargs in probes:
        assert kwargs["timeout"] == 3.0
        assert "headers" not in kwargs and "auth" not in kwargs
    # The base URL's query never reaches the terminal.
    assert "querysecret" not in out and "key=" not in out
    assert FAKE_KEY not in out


def test_routing_env_credential_missing_fails_naming_var_only(
    tmp_path: Path,
    probes: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("DOCTOR_TEST_SECRET_VAR")
    config = _write(tmp_path)
    assert run_doctor(_args(config, offline=True)) == 1
    out = capsys.readouterr().out
    fails = [line for line in _routing_lines(out) if line.startswith("FAIL")]
    assert len(fails) == 1
    assert "DOCTOR_TEST_SECRET_VAR" in fails[0] and "anthropic_key" in fails[0]
    assert "refuse to start" in fails[0]
    assert FAKE_KEY not in out


def test_routing_warnings_unpriced_and_memory_vault(
    tmp_path: Path, probes: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    # default_upstream = anthropic_key (metered) → the parser's I-6 warning;
    # literal rule model "muse-64k" has no price; memory vault → I-7.
    config = _write(tmp_path, backend="memory", default="anthropic_key", literal="muse-64k")
    assert run_doctor(_args(config, offline=True)) == 0
    out = capsys.readouterr().out
    warns = [line for line in _routing_lines(out) if line.startswith("WARN")]
    assert any('not cost = "zero"' in line for line in warns)
    assert any("no price for muse-64k" in line and "[prices.override" in line for line in warns)
    assert not any("claude-sonnet-5" in line and "no price" in line for line in warns)
    assert any('vault backend "memory"' in line and "I-7" in line for line in warns)
    assert any("probes skipped (--offline)" in line for line in _routing_lines(out))
    assert probes == []


def test_routing_probe_head_fails_get_answers(
    tmp_path: Path,
    probes: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def head_rejected(method: str, url: str, **kwargs: Any) -> httpx.Response:
        probes.append((method, url, kwargs))
        if method == "HEAD":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(404, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", head_rejected)
    config = _write(tmp_path)
    assert run_doctor(_args(config)) == 0
    lines = _routing_lines(capsys.readouterr().out)
    assert sum("answers at" in line for line in lines) == 2
    assert [m for m, _u, _k in probes] == ["HEAD", "GET", "HEAD", "GET"]


def test_routing_probe_failure_is_warn_with_type_only(
    tmp_path: Path, probes: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write(tmp_path)
    assert run_doctor(_args(config)) == 0  # unreachable upstream is WARN, never FAIL
    out = capsys.readouterr().out
    warns = [line for line in _routing_lines(out) if "not reachable" in line]
    assert len(warns) == 2 and all(line.startswith("WARN") for line in warns)
    assert all("(ConnectError)" in line for line in warns)
    assert "probe-detail-must-not-print" not in out  # the exception TEXT is never echoed
    assert "querysecret" not in out
    assert len(probes) == 4  # HEAD + GET per upstream


def test_routing_bad_price_table_file_fails(
    tmp_path: Path, probes: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write(tmp_path)
    config.write_text(
        config.read_text() + f'\n[prices]\ntable = "{(tmp_path / "missing.json").as_posix()}"\n'
    )
    assert run_doctor(_args(config, offline=True)) == 1
    out = capsys.readouterr().out
    assert any("price table" in line and line.startswith("FAIL") for line in _routing_lines(out))


def test_routing_json_mode(
    tmp_path: Path, probes: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    config = _write(tmp_path)
    assert run_doctor(_args(config, offline=True, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    routing_rows = [row for row in payload["checks"] if row["area"] == "routing"]
    assert routing_rows and all(row["level"] in ("PASS", "WARN") for row in routing_rows)
    assert FAKE_KEY not in json.dumps(payload)


def test_probe_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _display_url("https://h.example:8443/base/?key=abc#frag") == "https://h.example:8443"

    def always_fail(method: str, url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectTimeout("t")

    monkeypatch.setattr(httpx, "request", always_fail)
    assert _probe_upstream("http://127.0.0.1:1") == "ConnectTimeout"


def test_doctor_offline_flag_in_parser() -> None:
    from llm_redact.cli import build_parser

    args = build_parser().parse_args(["doctor", "--offline"])
    assert args.offline is True
    assert build_parser().parse_args(["doctor"]).offline is False
