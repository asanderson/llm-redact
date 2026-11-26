"""The config file is read as UTF-8 on every platform, whatever the locale.

``Path.read_text()`` without an encoding decodes with the locale's code page:
cp1252 on a Western Windows install without ``PYTHONUTF8``. A deny string
saved as UTF-8 then loaded as mojibake (``MÃ¼ller``) and never matched — the
value it should have redacted went upstream. The file is decoded as UTF-8,
one leading byte-order mark (Notepad's "UTF-8 with BOM") tolerated, and a
file that is not UTF-8 is a ConfigError naming the file and the offset,
never a traceback and never a silent guess."""

from __future__ import annotations

import argparse
import pathlib
from pathlib import Path
from typing import Any

import pytest

from llm_redact.cli import _run_serve_check
from llm_redact.config import ConfigError, load_config
from llm_redact.detection.engine import build_allowlist, build_detectors
from llm_redact.doctor_cli import _check_config, _Report
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault

BOM = b"\xef\xbb\xbf"


@pytest.fixture
def windows_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text reads that name no encoding decode as Windows' cp1252, as they
    do on a Western Windows install without PYTHONUTF8."""
    real = pathlib.Path.read_text

    def read_text(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        return real(self, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(pathlib.Path, "read_text", read_text)


def _redacted(config_file: Path, text: str) -> str:
    detection = load_config(config_file).detection
    redactor = Redactor(build_detectors(detection), InMemoryVault(), build_allowlist(detection))
    return redactor.redact_text(text)


@pytest.mark.parametrize("value", ["Müller", "Łódź"])
@pytest.mark.parametrize("bom", [b"", BOM], ids=["utf-8", "utf-8 with BOM"])
@pytest.mark.usefixtures("windows_locale")
def test_a_non_ascii_deny_string_loads_and_redacts(tmp_path: Path, value: str, bom: bytes) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_bytes(bom + f'[detection]\ndeny = ["{value}"]\n'.encode())
    assert [entry.value for entry in load_config(config_file).detection.deny_strings] == [value]
    redacted = _redacted(config_file, f"ask {value} about it")
    assert value not in redacted
    assert redacted.startswith("ask «DENY_") and redacted.endswith("» about it")


def test_one_leading_byte_order_mark_only(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_bytes(BOM + BOM + b"port = 9999\n")
    with pytest.raises(ConfigError, match=str(config_file).replace("\\", "\\\\")):
        load_config(config_file)
    config_file.write_bytes(b"port = 9999\n" + BOM + b"\n")
    with pytest.raises(ConfigError):
        load_config(config_file)


@pytest.mark.parametrize(
    "data",
    [b"port = 9999\n\x81\x8d\n", b'[detection]\ndeny = ["M\xfcller"]\n', b"\xff\xfep\x00"],
    ids=["undefined bytes", "latin-1", "utf-16"],
)
def test_a_file_that_is_not_utf8_is_a_config_error(tmp_path: Path, data: bytes) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_bytes(data)
    with pytest.raises(ConfigError) as refused:
        load_config(config_file)
    message = str(refused.value)
    assert str(config_file) in message and "UTF-8" in message
    # Nothing of the file's content is echoed.
    assert "M\xfcller" not in message and "port" not in message


def test_the_offset_is_the_files_own(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    for prefix in (b"", BOM):
        config_file.write_bytes(prefix + b"port = 9999\n\x81\n")
        with pytest.raises(ConfigError, match=f"at offset {len(prefix) + 12}\\)"):
            load_config(config_file)


def _args(**values: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = dict(
        config=None, log_format=None, session=None, host=None, port=None, check=True
    )
    defaults.update(values)
    return argparse.Namespace(**defaults)


def test_doctor_and_serve_check_report_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_bytes(b'[detection]\ndeny = ["M\xfcller"]\n')
    report = _Report(json_mode=True)
    assert _check_config(report, _args(config=config_file)) is None
    assert [(row["level"], row["area"]) for row in report.rows] == [("FAIL", "config")]
    assert "UTF-8" in report.rows[0]["message"]
    assert _run_serve_check(_args(config=config_file)) == 1
    assert "serve --check: FAIL:" in capsys.readouterr().err
