"""Commands llm-redact prints for the operator to paste work in the shell
they paste them into: POSIX sh, cmd.exe and PowerShell.

- Install hints double-quote the extra (``pip install "llm-redact-proxy[x]"``):
  cmd.exe keeps single quotes as characters, so ``'llm-redact-proxy[x]'``
  named no package there; zsh needs the quotes around the brackets.
- A printed path is quoted for the shell (``overrides.shell_quote``): on
  Windows double quotes around one holding a blank or a character cmd.exe
  or PowerShell would read, with a note for what they read inside quotes
  (``%NAME%``, ``$name``)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from llm_redact.overrides import hint_config, shell_quote, windows_quote_note

SRC = Path(__file__).resolve().parent.parent / "src" / "llm_redact"
SINGLE_QUOTED_EXTRA = re.compile(r"""'llm-redact-proxy\[|install\s+'""")


def test_no_printed_install_hint_is_single_quoted() -> None:
    found = [
        f"{path.relative_to(SRC)}:{number}"
        for path in sorted(SRC.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if SINGLE_QUOTED_EXTRA.search(line)
    ]
    assert found == []


def test_the_install_hints_double_quote_the_extra() -> None:
    from llm_redact.extraction import CRYPTO_HINT
    from llm_redact.vault_rdbms import _EXTRA_HINTS

    assert _EXTRA_HINTS["postgresql"] == 'pip install "llm-redact-proxy[vault-postgres]"'
    assert CRYPTO_HINT == 'pip install "llm-redact-proxy[crypto]"'


@pytest.mark.parametrize(
    ("text", "quoted"),
    [
        ("C:\\cfg\\c.toml", "C:\\cfg\\c.toml"),
        ("C:\\my dir\\c.toml", '"C:\\my dir\\c.toml"'),
        ("C:\\a&b\\c.toml", '"C:\\a&b\\c.toml"'),
        ("C:\\Tools (x86)\\c.toml", '"C:\\Tools (x86)\\c.toml"'),
        ("C:\\o'brien\\c.toml", '"C:\\o\'brien\\c.toml"'),
        ("C:\\models dir\\", '"C:\\models dir\\\\"'),  # the closing quote stays a quote
        ("", '""'),
    ],
)
def test_windows_quoting(text: str, quoted: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert shell_quote(text) == quoted


def test_windows_notes_what_a_shell_reads_inside_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert windows_quote_note("C:\\cfg\\c.toml") is None
    for text in ("C:\\100%\\c.toml", "C:\\Users\\$me\\c.toml", "C:\\a`b\\c.toml"):
        note = windows_quote_note(text)
        assert note is not None and "inside double quotes" in note
    monkeypatch.setattr(sys, "platform", "linux")
    assert windows_quote_note("/100%/c.toml") is None


def test_the_override_hint_warns_about_a_path_a_shell_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    explicit = tmp_path / "100%" / "config.toml"
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        assert hint_config(explicit, {}) == shell_quote(str(explicit))
    assert "cmd.exe (%NAME%)" in caplog.text


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe and Windows PowerShell")
@pytest.mark.parametrize("shell", ["cmd", "pwsh"])
def test_a_quoted_path_reaches_a_program_whole(shell: str, tmp_path: Path) -> None:
    path = str(tmp_path / "llm redact & co (test)" / "config.toml")
    echo = f'"{sys.executable}" -c "import sys; print(repr(sys.argv[1:]))"'
    if shell == "cmd":
        script = tmp_path / "paste.cmd"
        script.write_text(f"@{echo} {shell_quote(path)}\r\n", encoding="utf-8")
        command = ["cmd", "/d", "/c", str(script)]
    else:
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if pwsh is None:
            pytest.skip("no PowerShell")
        script = tmp_path / "paste.ps1"
        script.write_text(f"& {echo} {shell_quote(path)}\n", encoding="utf-8-sig")
        command = [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script)]
    done = subprocess.run(command, capture_output=True, timeout=120, check=True)
    assert done.stdout.decode("utf-8", errors="replace").strip() == repr([path])
