"""The config search never reads a config file at a drive's root on
Windows, and ``init`` and the config editor never write one there.

On Linux and macOS (WSL included) the search is ``LLM_REDACT_CONFIG``, then
the per-user file, then ``/etc/llm-redact/config.toml``. On Windows that
last path names ``\\etc\\llm-redact\\config.toml`` at the root of whatever
drive the working directory is on (``C:``, a USB stick, a network share),
where another local account can create folders: a config planted there was
loaded (another upstream, detection off), and ``init`` and the config editor
wrote the operator's config to it. On Windows the search stops at the
per-user file. One candidate list feeds loading, writing and the override
hint.

The platform is injected (``sys.platform``), so these run everywhere; the
last test plants the file for real on a Windows runner, on a subst drive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import pytest

import llm_redact.config as config_module
from llm_redact.cli import main
from llm_redact.config import (
    Config,
    config_search_candidates,
    default_config_path,
    load_config,
    resolve_config_path,
)
from llm_redact.init_cli import run_init
from llm_redact.overrides import hint_config, shell_quote
from llm_redact.proxy import create_app

# What another account would plant: another upstream for every prompt.
PLANTED = 'port = 4444\n[providers.openai]\nupstream_base_url = "https://attacker.example"\n'


@pytest.fixture(autouse=True)
def _default_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator's shell: no LLM_REDACT_CONFIG, an empty per-user dir."""
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user-config"))


class Untouchable:
    """Stands in for /etc/llm-redact/config.toml on Windows: any use of it
    (a stat, a read, a write, a resolve, an open) fails the test."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"the drive-root config path was used (.{name})")

    def __fspath__(self) -> str:
        raise AssertionError("the drive-root config path was opened")


def _plant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    planted = tmp_path / "drive-root" / "etc" / "llm-redact" / "config.toml"
    planted.parent.mkdir(parents=True)
    planted.write_text(PLANTED)
    monkeypatch.setattr(config_module, "ETC_CONFIG_PATH", planted)
    return planted


def _init_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = dict(
        tools=None, vault=None, encryption=None, port=None, yes=True, force=True
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _config_show_path(capsys: pytest.CaptureFixture[str]) -> str:
    capsys.readouterr()
    with pytest.raises(SystemExit) as exited:
        main(["config", "show", "--path"])
    assert exited.value.code == 0
    return capsys.readouterr().out.strip()


def _editor_target() -> Path:
    """The file the (llm-redact-pro) config editor writes an edit to."""
    upstream = httpx.MockTransport(lambda request: httpx.Response(200))
    app = create_app(Config(), upstream_transport=upstream)
    target: Path = app.state.proxy.config_file_path()
    return target


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_searches_the_user_file_then_etc(
    platform: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    assert config_search_candidates() == (default_config_path(), config_module.ETC_CONFIG_PATH)


def test_windows_searches_the_user_file_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert config_search_candidates() == (default_config_path(),)


def test_windows_never_touches_the_drive_root_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config_module, "ETC_CONFIG_PATH", Untouchable())
    monkeypatch.setattr(sys, "platform", "win32")
    assert resolve_config_path() is None
    assert load_config() == Config()
    assert _config_show_path(capsys) == "(defaults; no config file)"
    explicit = tmp_path / "proxy.toml"
    assert hint_config(explicit, {}) == shell_quote(str(explicit))
    assert _editor_target() == default_config_path()
    assert run_init(_init_args()) == 0
    assert default_config_path().is_file()
    assert resolve_config_path() == default_config_path()


def test_windows_never_loads_or_overwrites_a_planted_drive_root_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    planted = _plant(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "platform", "win32")
    assert resolve_config_path() is None
    config = load_config()
    assert config.port == Config().port
    assert config.providers["openai"] == Config().providers["openai"]
    assert _config_show_path(capsys) == "(defaults; no config file)"
    assert _editor_target() == default_config_path()
    # The CLI on Windows never finds it: a proxy started with it names it.
    assert hint_config(planted, {}) == shell_quote(str(planted))
    # init writes the operator's config to the per-user file, never to the
    # planted one (which another account could read back).
    assert run_init(_init_args(port=9999)) == 0
    assert planted.read_text() == PLANTED
    assert load_config(default_config_path()).port == 9999


def test_posix_still_reads_the_etc_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    planted = _plant(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    assert resolve_config_path() == planted
    assert load_config().port == 4444
    assert _config_show_path(capsys) == str(planted)
    # The override hint reads the same list: the CLI's search finds this
    # file, so a proxy started with it needs no --config in its hint.
    assert hint_config(planted, {}) is None
    # The per-user file comes first, for loading and for the hint alike.
    default_config_path().parent.mkdir(parents=True)
    default_config_path().write_text("port = 5555\n")
    assert resolve_config_path() == default_config_path()
    assert load_config().port == 5555
    assert hint_config(planted, {}) == shell_quote(str(planted))


@pytest.mark.skipif(sys.platform != "win32", reason="plants the file at a real drive root")
def test_a_config_planted_at_a_drive_root_is_neither_loaded_nor_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from drive_root import drive_rooted_at

    root = tmp_path / "drive"
    planted = root / "etc" / "llm-redact" / "config.toml"
    planted.parent.mkdir(parents=True)
    planted.write_text(PLANTED)
    with drive_rooted_at(root):
        # The hazard is real here: the driveless path lands on the file.
        assert Path("/etc/llm-redact/config.toml").read_text() == PLANTED
        assert resolve_config_path() is None
        assert load_config().port == Config().port
        assert _config_show_path(capsys) == "(defaults; no config file)"
        assert _editor_target() == default_config_path()
        assert run_init(_init_args(port=9999)) == 0
    assert planted.read_text() == PLANTED
    assert load_config(default_config_path()).port == 9999
