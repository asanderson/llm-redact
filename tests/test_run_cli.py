"""`llm-redact run`: env injection, ephemeral proxy lifecycle, exit codes.

These are real-subprocess tests (the wrapper's job IS process management):
the auto-start path boots an actual `llm-redact serve` on a free port and
must tear it down; the already-running path must leave the proxy alone.
"""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from llm_redact.run_cli import run_run

CHILD_SNIPPET = (
    "import os, sys;"
    "print('ANTH=' + os.environ.get('ANTHROPIC_BASE_URL', ''));"
    "print('OLLAMA=' + os.environ.get('OLLAMA_HOST', ''));"
    "sys.exit(7)"
)


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _args(
    port: int,
    config: Path,
    tools: str | None,
    command: list[str],
    set_env: list[str] | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        config=config, port=port, tools=tools, tool_command=command, set_env=set_env
    )


def _status_ok(port: int) -> bool:
    try:
        response = httpx.get(f"http://127.0.0.1:{port}/__llm-redact/status", timeout=1.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text("")  # defaults; port comes from --port
    return path


def test_auto_start_injects_env_and_tears_down(
    config_file: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    port = _free_port()
    code = run_run(_args(port, config_file, None, ["--", sys.executable, "-c", CHILD_SNIPPET]))
    out, err = capfd.readouterr()
    assert code == 7  # child's exit code propagates
    assert f"ANTH=http://127.0.0.1:{port}" in out
    assert f"OLLAMA=http://127.0.0.1:{port}" in out
    assert "started for this run" in err
    # The ephemeral proxy is gone: nothing answers on the port anymore.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _status_ok(port):
        time.sleep(0.1)
    assert not _status_ok(port)


def test_running_proxy_is_reused_and_left_alive(
    config_file: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    port = _free_port()
    serve = [sys.executable, "-m", "llm_redact", "serve", "--config", str(config_file)]
    proxy = subprocess.Popen([*serve, "--port", str(port)])
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not _status_ok(port):
            assert proxy.poll() is None, "proxy died during startup"
            time.sleep(0.1)
        assert _status_ok(port)

        code = run_run(
            _args(port, config_file, "claude", ["--", sys.executable, "-c", CHILD_SNIPPET])
        )
        out, err = capfd.readouterr()
        assert code == 7
        assert f"ANTH=http://127.0.0.1:{port}" in out
        # \r\n: the child prints through the Windows console layer.
        assert "OLLAMA=\n" in out.replace("\r\n", "\n")  # --tools claude: Anthropic var only
        assert "already running" in err
        assert _status_ok(port)  # the wrapper never kills a proxy it didn't start
    finally:
        proxy.terminate()
        proxy.wait(timeout=10)


def test_rejects_unknown_tool_and_empty_command(config_file: Path) -> None:
    assert run_run(_args(8787, config_file, "sublime", ["--", "true"])) == 2
    assert run_run(_args(8787, config_file, None, [])) == 2
    assert run_run(_args(8787, config_file, None, ["--"])) == 2


def test_set_env_injects_extra_variables(
    config_file: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    # The escape hatch for tools not in TOOL_EXPORTS: the named variable
    # gets the proxy base URL alongside the known tools' variables.
    port = _free_port()
    snippet = (
        "import os, sys; print('EXTRA=' + os.environ.get('MY_TOOL_BASE_URL', '')); sys.exit(0)"
    )
    code = run_run(
        _args(
            port,
            config_file,
            "claude",
            ["--", sys.executable, "-c", snippet],
            set_env=["MY_TOOL_BASE_URL"],
        )
    )
    out, err = capfd.readouterr()
    assert code == 0
    assert f"EXTRA=http://127.0.0.1:{port}" in out
    assert "MY_TOOL_BASE_URL" in err  # named on the routing line


def test_set_env_rejects_non_identifier_names(
    config_file: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    code = run_run(_args(0, config_file, "claude", ["--", "true"], set_env=["not-a-var!"]))
    assert code == 2
    assert "NAMES" in capfd.readouterr().out


# --- pointed-at proxies (3.0): --proxy-url / LLM_REDACT_PROXY_URL --------------


def _proxy_args(
    config: Path, command: list[str], proxy_url: str | None = None
) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        port=None,
        tools="claude",
        tool_command=command,
        set_env=None,
        proxy_url=proxy_url,
    )


def test_proxy_url_env_reuses_running_proxy(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    port = _free_port()
    proxy = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "llm_redact",
            "serve",
            "--config",
            str(config_file),
            "--port",
            str(port),
        ]
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not _status_ok(port):
            time.sleep(0.1)
        assert _status_ok(port)
        monkeypatch.setenv("LLM_REDACT_PROXY_URL", f"http://127.0.0.1:{port}")
        code = run_run(_proxy_args(config_file, ["--", sys.executable, "-c", CHILD_SNIPPET]))
        assert code == 7  # the child's exit code propagates
        # The pointed-at proxy was reused, not shadowed by an ephemeral one.
        assert _status_ok(port)
        err = capsys.readouterr().err
        assert "existing proxy" in err
    finally:
        proxy.terminate()
        proxy.wait(timeout=10)


def test_proxy_url_dead_proxy_fails_without_spawn(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dead_port = _free_port()
    monkeypatch.setenv("LLM_REDACT_PROXY_URL", f"http://127.0.0.1:{dead_port}")
    code = run_run(_proxy_args(config_file, ["--", sys.executable, "-c", "pass"]))
    assert code == 1
    assert "no proxy answering" in capsys.readouterr().out
    assert not _status_ok(dead_port)  # nothing was auto-started


def test_proxy_url_plain_http_off_loopback_refused(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LLM_REDACT_PROXY_URL", raising=False)
    args = _proxy_args(
        config_file,
        ["--", sys.executable, "-c", "pass"],
        proxy_url="http://redact.corp.example:8787",
    )
    assert run_run(args) == 2
    assert "loopback-only" in capsys.readouterr().out


def test_validate_proxy_url_rules() -> None:
    from llm_redact.run_cli import validate_proxy_url

    assert validate_proxy_url("http://127.0.0.1:8787") is None
    assert validate_proxy_url("http://localhost:8787") is None
    assert validate_proxy_url("https://redact.corp.example:8787") is None
    assert validate_proxy_url("http://redact.corp.example:8787") is not None
    assert validate_proxy_url("ftp://x") is not None
    assert validate_proxy_url("not-a-url") is not None


def test_missing_command_exits_127_and_tears_down_ephemeral(
    config_file: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    # 3.2.1: the child spawn sat OUTSIDE the ephemeral-proxy teardown scope,
    # so the most likely first-run failure (tool not on PATH) printed a raw
    # traceback AND leaked a background `serve` that made the next attempt
    # report "already running".
    port = _free_port()
    code = run_run(_args(port, config_file, "claude", ["--", "definitely-not-a-command-xyz"]))
    _, err = capfd.readouterr()
    assert code == 127  # the shell's own command-not-found code
    assert "command not found: definitely-not-a-command-xyz" in err
    assert "Traceback" not in err
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _status_ok(port):
        time.sleep(0.1)
    assert not _status_ok(port)
