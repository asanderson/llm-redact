"""The CLI works when its output or input is in a legacy code page.

On Windows without ``PYTHONUTF8``, output sent to a pipe or a file is
encoded with the system's code page (cp1252 on a Western install), which
has no "→", "⚠" or "Ł": printing one crashed the command with a
UnicodeEncodeError (doctor's "→", preview's echo of the text). stdout and
stderr now show what the encoding lacks as a backslash escape — except in
``lookup``, the one command that prints secret values, which prints nothing
rather than an altered value. ``preview`` reads piped input as bytes: UTF-8
(one byte-order mark allowed), else the input stream's encoding, which it
names.

Emulated with ``PYTHONIOENCODING=cp1252`` in a child process, so these run
on every OS."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from llm_redact.detection.deny import DenyEntry
from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault, open_sqlite_vault


def _cli(
    *args: str, encoding: str = "cp1252", stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": encoding}
    try:
        for arg in args:
            os.fsencode(arg)
    except UnicodeEncodeError:
        # A POSIX process under a non-UTF-8 locale cannot pass the argument
        # at all (Windows passes arguments as UTF-16: never this).
        pytest.skip("this process's file system encoding cannot pass the argument")
    return subprocess.run(
        [sys.executable, "-m", "llm_redact", *args],
        input=stdin,
        capture_output=True,
        env=env,
        timeout=120,
    )


@pytest.mark.parametrize(
    "args",
    [
        ("--version",),
        ("guide",),
        ("completions", "bash"),
        ("config", "show"),
        ("fips-check",),
        ("preview", "--text", "Łódź → jane.doe@corp.example"),
        ("doctor", "--offline"),
        ("sessions", "list"),
        ("override", "list"),
        ("models", "list"),
    ],
    ids=lambda args: " ".join(args),
)
def test_every_command_completes_on_a_legacy_code_page(args: tuple[str, ...]) -> None:
    utf8 = _cli(*args, encoding="utf-8")
    legacy = _cli(*args)
    assert b"Traceback" not in legacy.stderr, legacy.stderr.decode("cp1252", "replace")
    assert legacy.returncode == utf8.returncode


def test_what_the_code_page_lacks_is_escaped(tmp_path: Path) -> None:
    shown = _cli("preview", "--text", "Łódź → x").stdout.decode("cp1252")
    assert "\\u0141ód\\u017a \\u2192 x" in shown


def _vault(tmp_path: Path, value: str) -> Path:
    db = tmp_path / "vault.db"
    vault = open_sqlite_vault(db, "default")
    vault.placeholder_for("PERSON", value)
    vault.close()
    return db


def test_lookup_prints_a_value_the_code_page_can_show(tmp_path: Path) -> None:
    db = _vault(tmp_path, "Müller")
    found = _cli("lookup", "«PERSON_001»", "--db", str(db))
    assert found.returncode == 0
    assert found.stdout.decode("cp1252").split() == ["default", "Müller"]


def test_lookup_never_prints_an_altered_value(tmp_path: Path) -> None:
    db = _vault(tmp_path, "Łódź-hunter2")
    refused = _cli("lookup", "«PERSON_001»", "--db", str(db))
    assert refused.returncode == 2
    assert refused.stdout == b""
    message = refused.stderr.decode("cp1252")
    assert "PYTHONUTF8=1" in message and "hunter2" not in message
    shown = _cli("lookup", "«PERSON_001»", "--db", str(db), encoding="utf-8")
    assert shown.returncode == 0
    assert shown.stdout.decode("utf-8").split() == ["default", "Łódź-hunter2"]


def _expected(text: str, deny: tuple[str, ...]) -> str:
    detection = DetectionConfig(deny_strings=tuple(DenyEntry(value) for value in deny))
    redactor = Redactor(build_detectors(detection), InMemoryVault(), build_allowlist(detection))
    return redactor.redact_text(text)


@pytest.fixture
def deny_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    deny = ("Müller", "Łódź")
    config = tmp_path / "config.toml"
    config.write_text(
        f"[detection]\ndeny = {json.dumps(list(deny), ensure_ascii=False)}\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setenv("LLM_REDACT_CONFIG", str(config))
    return deny


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"], ids=["utf-8", "utf-8 with BOM"])
def test_preview_reads_piped_utf8_whatever_the_console(
    deny_config: tuple[str, ...], bom: bytes
) -> None:
    text = "ask Müller in Łódź"
    piped = _cli("preview", "--json", stdin=bom + text.encode("utf-8"))
    assert piped.returncode == 0, piped.stderr
    answer = json.loads(piped.stdout)
    assert answer["detections"] == {"DENY": 2}
    assert answer["redacted"] == _expected(text, deny_config)
    assert piped.stderr == b""


def test_preview_reads_input_that_is_not_utf8_in_the_streams_encoding(
    deny_config: tuple[str, ...],
) -> None:
    piped = _cli("preview", "--json", stdin="ask Müller".encode("cp1252"))
    assert piped.returncode == 0, piped.stderr
    assert json.loads(piped.stdout)["detections"] == {"DENY": 1}
    assert "not UTF-8" in piped.stderr.decode() and "cp1252" in piped.stderr.decode()


def test_preview_refuses_input_that_is_not_utf8_in_a_utf8_stream(
    deny_config: tuple[str, ...],
) -> None:
    piped = _cli("preview", "--json", encoding="utf-8", stdin=b"ask M\xfcller")
    assert piped.returncode == 2
    assert piped.stdout == b""
    assert "not UTF-8" in piped.stderr.decode() and "offset 5" in piped.stderr.decode()
