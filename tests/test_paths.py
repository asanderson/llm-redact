"""Default locations per platform (llm_redact.paths).

Pinned for each platform: an XDG variable set (it wins everywhere), empty or
relative (both count as unset), or unset; and on Windows the
``%LOCALAPPDATA%\\llm-redact`` home with the pre-existing profile locations
kept while the new one holds nothing of their kind. The platform is injected
(``sys.platform``), so this runs on every OS."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from llm_redact import paths
from llm_redact.config import default_config_path
from llm_redact.detection import model_files

XDG = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home with no XDG variable and no LOCALAPPDATA set."""
    home = tmp_path / "home"
    home.mkdir()
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    for name in (*XDG, "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    return home


def _all() -> tuple[Path, Path, Path, Path]:
    return paths.config_dir(), paths.data_dir(), paths.log_dir(), paths.models_dir()


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_defaults_are_unchanged(
    platform: str, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    assert _all() == (
        home / ".config" / "llm-redact",
        home / ".local" / "share" / "llm-redact",
        home / ".local" / "state" / "llm-redact",
        home / ".local" / "share" / "llm-redact" / "models",
    )
    assert default_config_path() == home / ".config" / "llm-redact" / "config.toml"
    assert model_files.models_dir() == paths.models_dir()


def test_windows_defaults_live_under_localappdata(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    local = tmp_path / "AppData-Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    root = local / "llm-redact"
    assert _all() == (root, root / "data", root / "logs", root / "data" / "models")
    assert default_config_path() == root / "config.toml"
    assert paths.uses_legacy_location() == []


@pytest.mark.parametrize("value", [None, "", "relative\\AppData"])
def test_windows_without_a_usable_localappdata_uses_the_profile(
    value: str | None, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A service account may not set it; a relative (or drive-relative)
    # value would name a folder under the working directory's drive.
    monkeypatch.setattr(sys, "platform", "win32")
    if value is not None:
        monkeypatch.setenv("LOCALAPPDATA", value)
    assert paths.config_dir() == home / "AppData" / "Local" / "llm-redact"


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_a_set_xdg_variable_wins_everywhere(
    platform: str, tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    for name in XDG:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    assert _all() == (
        tmp_path / "xdg_config_home" / "llm-redact",
        tmp_path / "xdg_data_home" / "llm-redact",
        tmp_path / "xdg_state_home" / "llm-redact",
        tmp_path / "xdg_data_home" / "llm-redact" / "models",
    )


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("value", ["", "relative/xdg", "./xdg"])
def test_an_empty_or_relative_xdg_variable_counts_as_unset(
    platform: str, value: str, tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The XDG spec: such a value is ignored — a relative one would name a
    # folder under whatever the working directory is.
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    unset = _all()
    for name in XDG:
        monkeypatch.setenv(name, value)
    assert _all() == unset


def test_a_windows_install_keeps_its_legacy_locations_until_moved(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    root = tmp_path / "local" / "llm-redact"
    legacy_config = home / ".config" / "llm-redact"
    legacy_data = home / ".local" / "share" / "llm-redact"
    legacy_config.mkdir(parents=True)
    (legacy_config / "config.toml").write_text("port = 8787\n")
    legacy_data.mkdir(parents=True)
    (legacy_data / "vault.db").write_bytes(b"")
    # An upgraded install reads (and writes) where its files are.
    assert paths.config_dir() == legacy_config
    assert paths.data_dir() == legacy_data
    assert paths.models_dir() == legacy_data / "models"
    assert paths.uses_legacy_location() == [legacy_config, legacy_data]
    # A config written to the new home takes over; so do stores moved there,
    # and the config file beside them never counts as data.
    root.mkdir(parents=True)
    (root / "config.toml").write_text("port = 8787\n")
    assert paths.config_dir() == root
    assert paths.data_dir() == legacy_data
    (root / "data").mkdir()
    (root / "data" / "vault.db").write_bytes(b"")
    assert paths.data_dir() == root / "data"
    assert paths.uses_legacy_location() == []


def test_an_empty_legacy_data_folder_is_not_kept(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    (home / ".local" / "share" / "llm-redact").mkdir(parents=True)
    (home / ".config" / "llm-redact").mkdir(parents=True)  # no config.toml in it
    assert paths.data_dir() == tmp_path / "local" / "llm-redact" / "data"
    assert paths.config_dir() == tmp_path / "local" / "llm-redact"


def test_legacy_locations_are_a_windows_matter_only(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    (home / ".config" / "llm-redact").mkdir(parents=True)
    (home / ".config" / "llm-redact" / "config.toml").write_text("")
    assert paths.uses_legacy_location() == []
