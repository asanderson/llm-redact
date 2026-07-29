"""Behavioral tests for the plugin's posture script (bin/llm-redact-posture).

The script is the SessionStart hook AND a bin/ helper: it must report
exactly one of four states (CLI missing / proxy down / session unrouted /
healthy), stay silent when healthy under --quiet-ok, never install
anything, and never echo a URL path (LLM_REDACT_PROXY_URL could embed a
/u/<key> — the run/status CLI scrubbing rule applies here too). Each
state is driven for real: stub executables on a controlled PATH and a
live loopback HTTP server standing in for the proxy."""

import http.server
import os
import stat
import subprocess
import threading
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "plugins" / "llm-redact" / "bin"
SCRIPT = SCRIPT / "llm-redact-posture"


def _run(tmp_path: Path, *, cli: bool, env: dict[str, str], args: tuple[str, ...] = ()) -> str:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    if cli:
        stub = bindir / "llm-redact"
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    full_env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        **env,
    }
    result = subprocess.run(
        ["sh", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=full_env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture()
def healthz_server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path == "/__llm-redact/healthz":
                body = b'{"status": "ok"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def test_script_is_executable_on_disk() -> None:
    assert os.access(SCRIPT, os.X_OK), "render_plugins.py must chmod bin/ files"


def test_cli_missing_reports_and_names_itself(tmp_path: Path) -> None:
    out = _run(tmp_path, cli=False, env={}, args=("--quiet-ok",))
    assert "llm-redact posture check" in out
    assert "not installed" in out


def test_proxy_down_reports(tmp_path: Path) -> None:
    # Port 1 refuses instantly on loopback: CLI present, no proxy.
    out = _run(
        tmp_path,
        cli=True,
        env={"LLM_REDACT_PROXY_URL": "http://127.0.0.1:1"},
        args=("--quiet-ok",),
    )
    assert "no proxy is answering" in out
    assert "NOT" in out  # the unprotected state is stated, not implied


def test_unrouted_session_reports(tmp_path: Path, healthz_server: str) -> None:
    out = _run(
        tmp_path,
        cli=True,
        env={"LLM_REDACT_PROXY_URL": healthz_server},
        args=("--quiet-ok",),
    )
    assert "not" in out.lower() and "routed" in out
    assert "ANTHROPIC_BASE_URL" in out
    assert "llm-redact run -- claude" in out


def test_healthy_is_silent_in_quiet_mode(tmp_path: Path, healthz_server: str) -> None:
    out = _run(
        tmp_path,
        cli=True,
        env={"LLM_REDACT_PROXY_URL": healthz_server, "ANTHROPIC_BASE_URL": healthz_server},
        args=("--quiet-ok",),
    )
    assert out == ""


def test_healthy_verbose_reports_ok(tmp_path: Path, healthz_server: str) -> None:
    out = _run(
        tmp_path,
        cli=True,
        env={"LLM_REDACT_PROXY_URL": healthz_server, "ANTHROPIC_BASE_URL": healthz_server},
    )
    assert "OK" in out


def test_url_paths_never_echo(tmp_path: Path, healthz_server: str) -> None:
    # An /u/<key> user prefix mistakenly embedded in LLM_REDACT_PROXY_URL
    # must never reach the transcript — scheme://host:port only.
    out = _run(
        tmp_path,
        cli=True,
        env={"LLM_REDACT_PROXY_URL": f"{healthz_server}/u/sekrit-user-key"},
        args=("--quiet-ok",),
    )
    assert "sekrit-user-key" not in out
    assert healthz_server in out  # the scrubbed base still names the proxy


def test_routing_comparison_ignores_paths(tmp_path: Path, healthz_server: str) -> None:
    # ANTHROPIC_BASE_URL carrying a path (some SDKs append /v1) must still
    # count as routed when the host:port matches.
    out = _run(
        tmp_path,
        cli=True,
        env={
            "LLM_REDACT_PROXY_URL": healthz_server,
            "ANTHROPIC_BASE_URL": f"{healthz_server}/v1",
        },
        args=("--quiet-ok",),
    )
    assert out == ""
