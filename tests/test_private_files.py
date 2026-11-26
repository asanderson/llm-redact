"""What llm-redact creates private on POSIX (a file 0600, a folder 0700) gets
a protected NTFS access control list on Windows (``winacl.make_private``):
this account, SYSTEM and Administrators only, for a folder inherited by
everything created in it, a SQLite vault's ``-wal`` and ``-shm`` included.

Before, the POSIX modes were the only control and Windows ignores them: a
vault under a folder at a drive's root inherited that folder's list, which
lets every local account read it. The stand-in tests run on every OS and pin
each site (the sqlite vault, the override store, the config file and its
``.bak``, a vault backup, the log file); the real ones run on the Windows
legs, a vault outside the user profile included."""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llm_redact import winacl
from llm_redact.winacl import ADMINISTRATORS, SYSTEM, Ace, Dacl, foreign_access, needs_private

USER = "S-1-5-21-1000-2000-3000-1001"
USERS = "S-1-5-32-545"
FULL, PASSED_ON = 0x1F01FF, 0x03  # FILE_ALL_ACCESS; OBJECT_INHERIT | CONTAINER_INHERIT


def _list(flags: int = 0, *extra: Ace, protected: bool = True) -> Dacl:
    return Dacl(
        (Ace(True, USER, FULL, flags), Ace(True, SYSTEM, FULL, flags), *extra),
        protected=protected,
    )


@pytest.mark.parametrize(
    ("dacl", "folder", "needed"),
    [
        (None, False, True),  # no list: everyone
        (_list(protected=False), False, True),  # inherited: the folder decides
        (_list(), False, False),  # protected, no one else
        (_list(0, Ace(True, USERS, FULL)), False, True),
        (_list(), True, True),  # a folder whose entries are not passed on
        (_list(PASSED_ON), True, False),
    ],
)
def test_what_is_already_private_is_left_alone(
    dacl: Dacl | None, folder: bool, needed: bool
) -> None:
    assert needs_private(dacl, USER, folder=folder) is needed


class _Binding:
    def __init__(self, current: Dacl | None | OSError) -> None:
        self.current = current
        self.set: list[tuple[str, str, bool]] = []

    def current_user_sid(self) -> str:
        return USER

    def read_dacl(self, path: str) -> Dacl | None:
        if isinstance(self.current, OSError):
            raise self.current
        return self.current

    def make_private(self, path: str, user: str, *, folder: bool) -> None:
        self.set.append((path, user, folder))


def _windows(monkeypatch: pytest.MonkeyPatch, binding: _Binding) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(winacl, "ctypes", SimpleNamespace(WinDLL=object))
    monkeypatch.setattr(winacl, "_windows", lambda: binding)


@pytest.mark.parametrize(
    ("current", "set_"),
    [
        (_list(PASSED_ON), False),
        (_list(protected=False), True),
        (OSError(None, "cannot read it"), True),
    ],
)
def test_make_private_sets_the_list_only_when_needed(
    current: Any, set_: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _Binding(current)
    _windows(monkeypatch, binding)
    winacl.make_private(Path("C:/data"), folder=True)
    assert binding.set == ([(str(Path("C:/data")), USER, True)] if set_ else [])


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_sets_no_list(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(winacl, "_windows", lambda: pytest.fail("asked Windows"))
    winacl.make_private(Path("/data"), folder=True)


def test_a_faked_windows_platform_without_the_api_sets_no_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(winacl, "ctypes", SimpleNamespace())
    monkeypatch.setattr(winacl, "_windows", lambda: pytest.fail("asked Windows"))
    winacl.make_private(Path("C:/data"))


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        winacl, "make_private", lambda path, folder=False: calls.append((Path(path).name, folder))
    )
    return calls


def test_the_sqlite_vault_and_its_folder_are_made_private(
    tmp_path: Path, made: list[tuple[str, bool]]
) -> None:
    from llm_redact.vault import _open_connection

    _open_connection(tmp_path / "data" / "vault.db").close()
    assert made == [("data", True), ("vault.db", False)]


def test_the_override_store_and_its_folder_are_made_private(
    tmp_path: Path, made: list[tuple[str, bool]]
) -> None:
    from llm_redact.overrides import OverrideStore

    store = OverrideStore(tmp_path / "data" / "overrides.db")
    try:
        assert store._open(create=True) is not None
    finally:
        store.close()
    assert made == [("data", True), ("overrides.db", False)]


def test_the_config_its_folder_and_backup_are_made_private(
    tmp_path: Path, made: list[tuple[str, bool]]
) -> None:
    from llm_redact.config_write import write_config_atomic

    target = tmp_path / "llm-redact" / "config.toml"
    write_config_atomic(target, "port = 1\n")
    assert made == [("llm-redact", True), ("config.toml.tmp", False)]
    made.clear()
    write_config_atomic(target, "port = 2\n")  # the folder exists: left as it is
    assert made == [("config.toml.bak", False), ("config.toml.tmp", False)]


def test_a_vault_backup_is_made_private(tmp_path: Path, made: list[tuple[str, bool]]) -> None:
    from llm_redact.cli import main

    db = tmp_path / "vault.db"
    sqlite3.connect(db).close()
    made.clear()
    dest = tmp_path / "vault-snap.db"
    with pytest.raises(SystemExit) as exit_:
        main(["vault", "backup", str(dest), "--db", str(db)])
    assert exit_.value.code in (0, None)
    assert ("vault-snap.db", False) in made


def test_the_log_file_and_a_folder_it_made_are_made_private(
    tmp_path: Path, made: list[tuple[str, bool]]
) -> None:
    from llm_redact.log import LogFile

    handler = LogFile(tmp_path / "logs" / "proxy.log")
    try:
        handler.emit(logging.makeLogRecord({"msg": "started", "levelno": 20, "levelname": "INFO"}))
    finally:
        handler.close()
    assert made == [("logs", True), ("proxy.log", False)]


# Real access control lists, Windows only.

on_windows = pytest.mark.skipif(sys.platform != "win32", reason="NTFS access control lists")


def _private(path: Path, user: str, *, folder: bool = False) -> None:
    dacl = winacl.read_dacl(path)
    assert dacl is not None and dacl.protected, f"{path} inherits its list"
    assert foreign_access(dacl, user) == [], f"{path} lets others in"
    assert {ace.sid for ace in dacl.aces} == {user, SYSTEM, ADMINISTRATORS}
    if folder:
        assert not needs_private(dacl, user, folder=True), f"{path} passes nothing on"


@on_windows
def test_a_vault_under_a_drive_root_folder_is_private_with_its_wal(tmp_path: Path) -> None:
    from llm_redact.vault import _open_connection

    user = winacl.current_user_sid()
    base = Path(os.environ.get("SYSTEMDRIVE", "C:") + "\\") / f"llm-redact-acl-{uuid.uuid4().hex}"
    base.mkdir()
    try:
        # Why: a folder at a drive's root lets every local account in.
        root_list = winacl.read_dacl(base)
        assert foreign_access(root_list, user) != []
        vault = base / "data" / "vault.db"
        conn = _open_connection(vault)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS probe (x)")
            conn.execute("INSERT INTO probe VALUES (1)")
            _private(vault.parent, user, folder=True)
            _private(vault, user)
            wal = vault.with_name("vault.db-wal")
            assert wal.exists()
            assert foreign_access(winacl.read_dacl(wal), user) == []  # inherited, private
        finally:
            conn.close()
    finally:
        shutil.rmtree(base, ignore_errors=True)


@on_windows
def test_the_config_and_the_log_file_are_private(tmp_path: Path) -> None:
    from llm_redact.config_write import write_config_atomic
    from llm_redact.log import LogFile

    user = winacl.current_user_sid()
    target = tmp_path / "llm-redact" / "config.toml"
    write_config_atomic(target, "port = 1\n")
    write_config_atomic(target, "port = 2\n")
    _private(target.parent, user, folder=True)
    _private(target, user)
    _private(target.with_name("config.toml.bak"), user)
    handler = LogFile(tmp_path / "logs" / "proxy.log")
    try:
        handler.emit(logging.makeLogRecord({"msg": "started", "levelno": 20, "levelname": "INFO"}))
    finally:
        handler.close()
    _private(tmp_path / "logs", user, folder=True)
    _private(tmp_path / "logs" / "proxy.log", user)
