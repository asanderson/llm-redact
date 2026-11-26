"""Environment lines to paste, in the user's shell: PowerShell on Windows
(what Windows Terminal opens), POSIX sh elsewhere (byte-identical to the
lines `init` and `plugin install` always printed)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from llm_redact import shell_syntax


def test_posix_lines_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert shell_syntax.env_lines("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1") == [
        "export OPENAI_BASE_URL=http://127.0.0.1:8787/v1"
    ]


def test_windows_lines_set_the_session_and_the_user_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert shell_syntax.env_lines("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1") == [
        "$env:OPENAI_BASE_URL = 'http://127.0.0.1:8787/v1'",
        "[Environment]::SetEnvironmentVariable('OPENAI_BASE_URL',"
        " 'http://127.0.0.1:8787/v1', 'User')",
    ]


@pytest.mark.parametrize(
    ("value", "quoted"),
    [
        ("plain", "'plain'"),
        ("it's", "'it''s'"),
        ('$HOME `n "x"', "'$HOME `n \"x\"'"),  # single quotes expand nothing
        # PowerShell reads the curly single quotes as quotes too.
        ("a‘b’c‚d‛e", "'a‘‘b’’c‚‚d‛‛e'"),
    ],
)
def test_powershell_quoting_keeps_the_value_as_typed(value: str, quoted: str) -> None:
    assert shell_syntax.powershell_quote(value) == quoted


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs PowerShell")
@pytest.mark.parametrize("value", ["https://redact.corp.example:8787", "it's $x `t ‘q’"])
def test_powershell_reads_the_value_back_as_typed(value: str) -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    quoted = shell_syntax.powershell_quote(value)
    # UTF-8 out: a redirected Windows console writes the OEM code page.
    utf8 = "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)"
    script = f"{utf8}; $env:LLM_REDACT_T = {quoted}; [Console]::Out.Write($env:LLM_REDACT_T)"
    done = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        timeout=60,
        check=True,
    )
    assert done.stdout.decode("utf-8") == value


def test_init_prints_powershell_lines_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.init_cli import run_init

    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")
    args = argparse.Namespace(
        yes=True,
        tools="claude,codex",
        vault="memory",
        encryption=None,
        port=9100,
        force=True,
        encrypted_content_passthrough=False,
        accept_encrypted_content_risk=False,
    )
    assert run_init(args) == 0
    assert (tmp_path / "llm-redact" / "config.toml").is_file()
    out = capsys.readouterr().out
    assert "$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:9100'" in out
    assert "$env:OPENAI_BASE_URL = 'http://127.0.0.1:9100/v1'" in out
    assert "[Environment]::SetEnvironmentVariable('OPENAI_BASE_URL'," in out
    assert "export " not in out


def test_plugin_install_prints_powershell_lines_on_windows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.plugin_cli import proxy_setup

    monkeypatch.setattr(sys, "platform", "win32")
    code = proxy_setup(
        proxy_url="https://redact.corp.example:8787",
        install_proxy=False,
        probe=lambda url: True,
        runner=lambda command: 0,
        ask=None,
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "$env:LLM_REDACT_PROXY_URL = 'https://redact.corp.example:8787'" in out
    assert "[Environment]::SetEnvironmentVariable('LLM_REDACT_PROXY_URL'," in out
    assert "export " not in out
