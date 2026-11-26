"""The vault, audit and override stores default to the data folder
(``llm_redact.paths.data_dir``) on every platform: ``%LOCALAPPDATA%\\llm-redact\\
data`` on a fresh Windows install, while an install from before keeps its
stores (and the tokens its vault restores) where they are.

The platform is injected (``sys.platform``), so this runs on every OS; the
last test opens the stores for real on Windows."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

from llm_redact import paths
from llm_redact.audit import default_audit_path
from llm_redact.config import VaultConfig
from llm_redact.overrides import OverrideStore, default_overrides_path
from llm_redact.vault import build_vault_manager, default_vault_path, open_sqlite_vault

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


def _defaults() -> tuple[Path, Path, Path]:
    return default_vault_path(), default_audit_path(), default_overrides_path()


def test_a_fresh_windows_install_keeps_its_stores_under_localappdata(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    data = tmp_path / "AppData-Local" / "llm-redact" / "data"
    assert _defaults() == (data / "vault.db", data / "audit.db", data / "overrides.db")


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_store_defaults_are_unchanged(
    platform: str, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    data = home / ".local" / "share" / "llm-redact"
    assert _defaults() == (data / "vault.db", data / "audit.db", data / "overrides.db")


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_a_set_xdg_data_home_holds_the_stores(
    platform: str, tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    data = tmp_path / "xdg" / "llm-redact"
    assert _defaults() == (data / "vault.db", data / "audit.db", data / "overrides.db")


@pytest.mark.parametrize("value", ["", "relative/data"])
def test_an_empty_or_relative_xdg_data_home_never_names_the_working_directory(
    value: str, tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", value)
    data = home / ".local" / "share" / "llm-redact"
    assert _defaults() == (data / "vault.db", data / "audit.db", data / "overrides.db")


def test_an_upgraded_windows_install_restores_tokens_from_its_legacy_vault(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    # What an older llm-redact wrote on Windows: its default was the POSIX
    # location under the profile.
    legacy = home / ".local" / "share" / "llm-redact"
    old = open_sqlite_vault(legacy / "vault.db", "default")
    token = old.placeholder_for("EMAIL", "jane@corp.example")
    old.close()
    assert default_vault_path() == legacy / "vault.db"
    manager = build_vault_manager(VaultConfig(backend="sqlite"))
    try:
        assert manager.get("default").original_for(token) == "jane@corp.example"
    finally:
        manager.close()
    assert not (tmp_path / "AppData-Local" / "llm-redact" / "data").exists()
    # Moved (with its -wal and -shm files, if any), it is read from the
    # new location, and the token still restores.
    new = tmp_path / "AppData-Local" / "llm-redact" / "data"
    new.mkdir(parents=True)
    for entry in legacy.iterdir():
        entry.rename(new / entry.name)
    assert default_vault_path() == new / "vault.db"
    moved = open_sqlite_vault(default_vault_path(), "default")
    try:
        assert moved.original_for(token) == "jane@corp.example"
    finally:
        moved.close()


@pytest.mark.skipif(sys.platform != "win32", reason="opens the stores where Windows keeps them")
def test_a_windows_proxy_writes_its_stores_under_localappdata(tmp_path: Path, home: Path) -> None:
    manager = build_vault_manager(VaultConfig(backend="sqlite"))
    try:
        manager.get("default").placeholder_for("EMAIL", "jane@corp.example")
    finally:
        manager.close()
    store = OverrideStore(default_overrides_path())
    store.counts()
    store.close()
    data = tmp_path / "AppData-Local" / "llm-redact" / "data"
    assert (data / "vault.db").is_file()
    conn = sqlite3.connect(data / "vault.db")
    try:
        assert conn.execute("SELECT COUNT(*) FROM mappings").fetchone() == (1,)
    finally:
        conn.close()
    assert paths.uses_legacy_location() == []
