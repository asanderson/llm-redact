"""doctor says where llm-redact keeps its files, how to move a Windows
install's folders from the location before ``%LOCALAPPDATA%\\llm-redact``,
and checks an encrypted vault wherever it is: on a path holding ``#`` or
``%`` (which a ``file:`` URI would read as a fragment or an escape) and on
a Windows UNC path (what a mapped network drive resolves to)."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from fake_cipher import FakeVaultCipher
from llm_redact import paths
from llm_redact.config import Config, VaultConfig
from llm_redact.doctor_cli import (
    _check_config,
    _check_paths,
    _check_vault_key_matches,
    _Report,
)
from llm_redact.registry import get_registry
from llm_redact.vault import open_sqlite_vault

XDG = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    for name in XDG:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData-Local"))
    return home


def _rows(report: _Report, area: str) -> list[dict[str, Any]]:
    return [row for row in report.rows if row["area"] == area]


@pytest.mark.parametrize("configured", [False, True])
def test_doctor_names_where_serve_logs(
    configured: bool, tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact.config import LogConfig

    monkeypatch.setattr(sys, "platform", "linux")
    file = str(tmp_path / "proxy.log") if configured else None
    report = _Report(json_mode=True)
    _check_paths(report, Config(log=LogConfig(file=file)))
    message = _rows(report, "paths")[1]["message"]
    if configured:
        assert message == f"serve logs to {tmp_path / 'proxy.log'}"
    else:
        assert "stderr" in message
        assert str(home / ".local" / "state" / "llm-redact" / "proxy.log") in message


def test_doctor_names_the_data_folder(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    report = _Report(json_mode=True)
    _check_paths(report)
    rows = _rows(report, "paths")
    assert [row["level"] for row in rows] == ["PASS"]
    assert str(home / ".local" / "share" / "llm-redact") in rows[0]["message"]


@pytest.mark.skipif(os.name != "posix", reason="a symlink stands in for a redirected folder")
def test_doctor_names_where_the_data_folder_really_is(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Microsoft Store Python keeps what it writes under AppData in its
    # package folder; other programs (an editor) do not see it there. A
    # link is the portable stand-in: doctor names the real place.
    monkeypatch.setattr(sys, "platform", "linux")
    real = tmp_path / "elsewhere"
    real.mkdir()
    (home / ".local" / "share").mkdir(parents=True)
    (home / ".local" / "share" / "llm-redact").symlink_to(real)
    report = _Report(json_mode=True)
    _check_paths(report)
    assert f"resolves to {real.resolve()}" in _rows(report, "paths")[0]["message"]


@pytest.mark.parametrize(
    ("platform", "searched"),
    [
        ("linux", ["{home}/.config/llm-redact/config.toml", "/etc/llm-redact/config.toml"]),
        ("win32", ["{local}/llm-redact/config.toml"]),
    ],
)
def test_doctor_names_where_it_looked_for_a_config_file(
    platform: str,
    searched: list[str],
    tmp_path: Path,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    # Neither file exists: the per-user one is under the throwaway home, and
    # /etc is never written by a test (a real one would only change the row).
    if Path("/etc/llm-redact/config.toml").exists():
        pytest.skip("this machine has a system-wide config")
    report = _Report(json_mode=True)
    assert _check_config(report, argparse.Namespace(config=None)) is not None
    (row,) = _rows(report, "config")
    assert row["level"] == "WARN"
    names = [
        str(Path(name.format(home=home.as_posix(), local=(tmp_path / "AppData-Local").as_posix())))
        for name in searched
    ]
    assert f"looked for {', '.join(names)});" in row["message"]


def test_doctor_warns_about_each_legacy_windows_folder_with_the_move_steps(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    legacy_config = home / ".config" / "llm-redact"
    legacy_config.mkdir(parents=True)
    (legacy_config / "config.toml").write_text("", encoding="utf-8")
    legacy_data = home / ".local" / "share" / "llm-redact"
    legacy_data.mkdir(parents=True)
    (legacy_data / "vault.db").write_bytes(b"")
    report = _Report(json_mode=True)
    _check_paths(report)
    root = tmp_path / "AppData-Local" / "llm-redact"
    warns = [row["message"] for row in _rows(report, "paths") if row["level"] == "WARN"]
    assert len(warns) == 2
    config_warn, data_warn = warns
    assert str(legacy_config / "config.toml") in config_warn and str(root) in config_warn
    assert "stop the proxy" in config_warn
    assert str(legacy_data) in data_warn and str(root / "data") in data_warn
    assert "stop the proxy" in data_warn and "-wal" in data_warn and "-shm" in data_warn
    assert not report.failed


def test_doctor_has_no_legacy_warning_once_moved(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    report = _Report(json_mode=True)
    _check_paths(report)
    rows = _rows(report, "paths")
    assert [row["level"] for row in rows] == ["PASS"]
    assert str(tmp_path / "AppData-Local" / "llm-redact" / "data") in rows[0]["message"]


def _encrypted_vault(path: Path, seed: bytes) -> None:
    vault = open_sqlite_vault(path, "default", FakeVaultCipher(seed))
    vault.placeholder_for("EMAIL", "jane@corp.example")
    vault.close()


def _key_check(monkeypatch: pytest.MonkeyPatch, path: Path, seed: bytes) -> _Report:
    monkeypatch.setattr(get_registry(), "build_cipher", lambda _vault: FakeVaultCipher(seed))
    report = _Report(json_mode=True)
    _check_vault_key_matches(
        report,
        Config(vault=VaultConfig(backend="sqlite", encryption="fernet", path=str(path))),
    )
    return report


@pytest.mark.parametrize("folder", ["vaults#1", "100%41 private", "a #b %41c"])
def test_doctor_checks_the_key_of_a_vault_whose_path_a_uri_would_misread(
    folder: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / folder / "vault.db"
    path.parent.mkdir()
    _encrypted_vault(path, b"k" * 32)
    matching = _key_check(monkeypatch, path, b"k" * 32)
    assert [row["level"] for row in _rows(matching, "vault")] == ["PASS"]
    other = _key_check(monkeypatch, path, b"x" * 32)
    assert [row["level"] for row in _rows(other, "vault")] == ["FAIL"]
    # Read-only: nothing was created beside it at a name the URI cut short.
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [folder]


def test_a_read_only_uri_escapes_what_a_uri_would_misread(tmp_path: Path) -> None:
    path = tmp_path / "a #b %41c" / "vault.db"
    path.parent.mkdir()
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x)")
    conn.close()
    uri = paths.sqlite_readonly_uri(path)
    assert uri.endswith("?mode=ro") and uri.count("?") == 1 and "#" not in uri
    reader = sqlite3.connect(uri, uri=True)
    try:
        assert reader.execute("SELECT name FROM sqlite_master").fetchall() == [("t",)]
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("CREATE TABLE u (x)")
    finally:
        reader.close()


def test_a_unc_path_gets_an_empty_uri_authority() -> None:
    # SQLite refuses ``file://server/...`` (an authority other than
    # localhost); ``file:////server/...`` is an empty authority and the
    # path ``//server/...``, which Windows opens as \\server\...
    assert (
        paths.sqlite_readonly_uri_for("//server/share/llm redact/vault#1.db")
        == "file:////server/share/llm%20redact/vault%231.db?mode=ro"
    )
    drive = paths.sqlite_readonly_uri_for("C:/Users/a/vault.db")
    assert drive == "file:C:/Users/a/vault.db?mode=ro"
    assert paths.sqlite_readonly_uri_for("/home/a/vault.db") == "file:/home/a/vault.db?mode=ro"


@pytest.mark.skipif(os.name != "posix", reason="POSIX reads a leading // as /")
def test_an_empty_uri_authority_keeps_the_path(tmp_path: Path) -> None:
    # The grammar the UNC form relies on, checked where it can be: SQLite
    # takes ``file:////tmp/...`` as an empty authority and the path
    # ``//tmp/...``, which POSIX opens as ``/tmp/...``.
    path = tmp_path / "vault.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x)")
    conn.close()
    reader = sqlite3.connect(paths.sqlite_readonly_uri_for("/" + path.as_posix()), uri=True)
    try:
        assert reader.execute("SELECT name FROM sqlite_master").fetchall() == [("t",)]
    finally:
        reader.close()


def _unc(path: Path) -> Path | None:
    """``path`` through the administrative share of its drive, or None where
    this machine has none to offer."""
    drive, rest = os.path.splitdrive(str(path))
    if len(drive) != 2 or drive[1] != ":":
        return None
    unc = Path(f"\\\\localhost\\{drive[0]}${rest}")
    try:
        return unc if unc.exists() else None
    except OSError:
        return None


@pytest.mark.skipif(sys.platform != "win32", reason="UNC paths are Windows paths")
def test_doctor_checks_the_key_of_a_vault_on_a_unc_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vault.db"
    _encrypted_vault(path, b"k" * 32)
    unc = _unc(path)
    if unc is None:
        pytest.skip("this machine offers no administrative share")
    assert paths.sqlite_readonly_uri(unc).startswith("file:////localhost/")
    matching = _key_check(monkeypatch, unc, b"k" * 32)
    assert [row["level"] for row in _rows(matching, "vault")] == ["PASS"]
    other = _key_check(monkeypatch, unc, b"x" * 32)
    assert [row["level"] for row in _rows(other, "vault")] == ["FAIL"]
