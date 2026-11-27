"""A rename or delete Windows refuses while another program holds the file
(winerror 5 or 32) is tried again for about two seconds, then reported with
what to close; POSIX retries nothing (``llm_redact.held_files``).

The config file and its ``.bak`` (``write_config_atomic``, used by ``init``
and the config editor) are covered here; ``models pull --to``'s folder
replacement and the assembled GLiNER folder in test_models_cli.py and
test_model_files.py. The stand-in tests inject the error on every OS; the
real ones hold a file open on the Windows legs."""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from llm_redact import held_files
from llm_redact.config_write import write_config_atomic
from llm_redact.held_files import HeldFile, is_held, retried


class Held(PermissionError):
    """What Windows raises for a file another program holds."""

    winerror = 32


class Denied(PermissionError):
    winerror = 5


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(held_files, "DELAYS", (0.0, 0.0, 0.0))


def _failing(times: int, error: Callable[[], OSError], value: str = "done") -> Any:
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        if len(calls) <= times:
            raise error()
        return value

    operation.calls = calls  # type: ignore[attr-defined]
    return operation


@pytest.mark.usefixtures("windows")
def test_a_held_file_is_tried_again_until_it_is_released() -> None:
    waits: list[float] = []
    operation = _failing(2, lambda: Held(13, "used by another process"))
    assert retried(operation, delays=(0.05, 0.1, 0.2), sleep=waits.append) == "done"
    assert waits == [0.05, 0.1] and len(operation.calls) == 3


@pytest.mark.usefixtures("windows")
def test_a_file_held_past_the_delays_raises_the_last_error() -> None:
    operation = _failing(10, lambda: Denied(13, "access denied"))
    with pytest.raises(Denied):
        retried(operation)
    assert len(operation.calls) == len(held_files.DELAYS) + 1


@pytest.mark.usefixtures("windows")
def test_another_error_is_raised_at_once() -> None:
    operation = _failing(1, lambda: FileNotFoundError(2, "missing"))
    with pytest.raises(FileNotFoundError):
        retried(operation)
    assert len(operation.calls) == 1


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_tries_once(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    operation = _failing(1, lambda: Held(13, "permission denied"))
    with pytest.raises(Held):
        retried(operation)
    assert len(operation.calls) == 1
    assert not is_held(Held(13, "permission denied"))


def test_only_windows_sharing_and_access_errors_count_as_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert is_held(Held()) and is_held(Denied())
    assert not is_held(PermissionError(13, "no winerror"))
    assert not is_held(ValueError("not an OSError"))


def _flaky_replace(monkeypatch: pytest.MonkeyPatch, times: int) -> list[str]:
    real = os.replace
    calls: list[str] = []

    def replace(source: Any, target: Any) -> None:
        calls.append(Path(target).name)
        if len(calls) <= times:
            raise Held(13, "used by another process")
        real(source, target)

    monkeypatch.setattr(os, "replace", replace)
    return calls


@pytest.mark.usefixtures("windows")
def test_a_config_held_open_is_written_once_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.toml"
    target.write_text("port = 1\n", encoding="utf-8")
    calls = _flaky_replace(monkeypatch, 2)
    assert write_config_atomic(target, "port = 2\n") == tmp_path / "config.toml.bak"
    assert calls == ["config.toml"] * 3
    assert target.read_text(encoding="utf-8") == "port = 2\n"
    assert (tmp_path / "config.toml.bak").read_text(encoding="utf-8") == "port = 1\n"


@pytest.mark.usefixtures("windows")
def test_a_config_held_open_too_long_names_it_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.toml"
    target.write_text("port = 1\n", encoding="utf-8")
    _flaky_replace(monkeypatch, 100)
    with pytest.raises(HeldFile) as raised:
        write_config_atomic(target, "port = 2\n")
    assert str(raised.value) == (
        f"another program holds {target} open (an editor, a virus scanner or a sync client):"
        " close it and run the command again"
    )
    assert target.read_text(encoding="utf-8") == "port = 1\n"
    assert not (tmp_path / "config.toml.tmp").exists()


@pytest.mark.usefixtures("windows")
def test_a_denial_that_outlasts_the_retries_is_not_called_a_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ERROR_ACCESS_DENIED is what Windows answers for a file another program
    # holds without leave to delete it, and also for a read-only file or one
    # the account may not change: the message says both.
    target = tmp_path / "config.toml"
    target.write_text("port = 1\n", encoding="utf-8")

    def replace(source: Any, target_: Any) -> None:
        raise Denied(13, "access denied")

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(HeldFile) as raised:
        write_config_atomic(target, "port = 2\n")
    assert str(raised.value) == (
        f"Windows refused to change {target}: another program may hold it open (an editor, a"
        " virus scanner or a sync client), or your account may not change it (it is read-only,"
        " or its access control list does not let you): close such a program and run the"
        " command again"
    )


@pytest.mark.usefixtures("windows")
def test_a_backup_held_open_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "config.toml"
    target.write_text("port = 1\n", encoding="utf-8")
    backup = tmp_path / "config.toml.bak"
    backup.write_text("port = 0\n", encoding="utf-8")
    real = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self == backup:
            raise Held(13, "used by another process")
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(HeldFile, match=re.escape(f"another program holds {backup} open")):
        write_config_atomic(target, "port = 2\n")
    assert target.read_text(encoding="utf-8") == "port = 1\n"


@pytest.mark.usefixtures("windows")
def test_init_says_which_file_to_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.init_cli import run_init

    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    target = tmp_path / "llm-redact" / "config.toml"
    target.parent.mkdir()
    target.write_text("port = 1\n", encoding="utf-8")
    _flaky_replace(monkeypatch, 100)
    args = argparse.Namespace(
        yes=True,
        tools="claude",
        vault="memory",
        encryption=None,
        port=9100,
        force=True,
        encrypted_content_passthrough=False,
        accept_encrypted_content_risk=False,
    )
    assert run_init(args) == 1
    err = capsys.readouterr().err
    assert f"cannot write {target} (another program holds {target} open" in err


def test_a_config_write_that_fails_otherwise_is_raised_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full(source: Any, target: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", full)
    with pytest.raises(OSError, match="No space left") as raised:
        write_config_atomic(tmp_path / "config.toml", "port = 2\n")
    assert not isinstance(raised.value, HeldFile)
    assert not (tmp_path / "config.toml.tmp").exists()


on_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows file locks")


@on_windows
def test_a_config_another_program_holds_is_written_once_it_closes(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    target.write_text("port = 1\n", encoding="utf-8")
    holder = open(target, encoding="utf-8")  # noqa: SIM115  (Windows refuses to replace it)
    timer = threading.Timer(0.3, holder.close)
    timer.start()
    try:
        write_config_atomic(target, "port = 2\n")
    finally:
        timer.join()
        holder.close()
    assert target.read_text(encoding="utf-8") == "port = 2\n"


@on_windows
def test_a_config_another_program_keeps_open_is_reported(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    target.write_text("port = 1\n", encoding="utf-8")
    with open(target, encoding="utf-8"), pytest.raises(HeldFile):
        write_config_atomic(target, "port = 2\n")
    assert target.read_text(encoding="utf-8") == "port = 1\n"
