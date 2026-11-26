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
import re
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llm_redact import winacl
from llm_redact.winacl import (
    ADMINISTRATORS,
    OWNER_RIGHTS,
    SYSTEM,
    Ace,
    Dacl,
    foreign_access,
    needs_private,
)

USER = "S-1-5-21-1000-2000-3000-1001"
OTHER = "S-1-5-21-1000-2000-3000-1002"
USERS = "S-1-5-32-545"
FULL, PASSED_ON = 0x1F01FF, 0x03  # FILE_ALL_ACCESS; OBJECT_INHERIT | CONTAINER_INHERIT
INHERIT_ONLY = 0x08


def _list(flags: int = 0, *extra: Ace, protected: bool = True, owner: str | None = USER) -> Dacl:
    return Dacl(
        (Ace(True, USER, FULL, flags), Ace(True, SYSTEM, FULL, flags), *extra),
        protected=protected,
        owner=owner,
    )


@pytest.mark.parametrize(
    ("dacl", "folder", "needed"),
    [
        (Dacl(None, owner=USER), False, True),  # no list: everyone
        (_list(protected=False), False, True),  # inherited: the folder decides
        (_list(), False, False),  # protected, no one else
        (_list(0, Ace(True, USERS, FULL)), False, True),
        (_list(), True, True),  # a folder whose entries are not passed on
        (_list(PASSED_ON), True, False),
        (_list(owner=OTHER), False, True),  # another account may rewrite it
        # Python 3.13+'s private mkdir: OWNER RIGHTS, this account owning it
        (_list(PASSED_ON, Ace(True, OWNER_RIGHTS, FULL, PASSED_ON)), True, False),
        # A folder that passes another account's entry on to what is made in it
        (_list(PASSED_ON, Ace(True, USERS, FULL, PASSED_ON | INHERIT_ONLY)), True, True),
    ],
)
def test_what_is_already_private_is_left_alone(dacl: Dacl, folder: bool, needed: bool) -> None:
    assert needs_private(dacl, USER, folder=folder) is needed


class _Binding:
    """Windows' calls: one list per path (``lists``; a missing path is a
    file that is not there). A list set is read back as set, unless the
    drive keeps none (``holds=False``)."""

    def __init__(self, lists: dict[str, Dacl | OSError], *, holds: bool = True) -> None:
        self.lists = lists
        self.holds = holds
        self.set: list[tuple[str, str, bool]] = []

    def current_user_sid(self) -> str:
        return USER

    def read_dacl(self, path: str) -> Dacl:
        found = self.lists.get(Path(path).name, OSError(None, "it is not there", path, 2))
        if isinstance(found, OSError):
            raise found
        return found

    def make_private(self, path: str, user: str, *, folder: bool) -> None:
        self.set.append((Path(path).name, user, folder))
        if self.holds:
            flags = PASSED_ON if folder else 0
            owner = self.lists[Path(path).name].owner  # type: ignore[union-attr]
            self.lists[Path(path).name] = _list(flags, owner=owner or USER)


def _windows(monkeypatch: pytest.MonkeyPatch, binding: _Binding) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(winacl, "ctypes", SimpleNamespace(WinDLL=object))
    monkeypatch.setattr(winacl, "_windows", lambda: binding)


@pytest.mark.parametrize(
    ("current", "set_"),
    [
        (_list(PASSED_ON), False),
        (_list(protected=False), True),
        (_list(PASSED_ON, owner=ADMINISTRATORS), False),  # made by an elevated process
        (_list(PASSED_ON, owner=SYSTEM), False),
    ],
)
def test_make_private_sets_the_list_only_when_needed(
    current: Dacl, set_: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _Binding({"data": current})
    _windows(monkeypatch, binding)
    winacl.make_private(Path("C:/data"), folder=True)
    assert binding.set == ([("data", USER, True)] if set_ else [])


@pytest.mark.parametrize(
    ("current", "reason"),
    [
        # Another account owns it: an owner may always change who may open
        # it, and a handle it opened before keeps its access. Taking it over
        # would leave both; it is refused instead.
        (_list(PASSED_ON, owner=OTHER), "another account owns it (PC\\eve)"),
        (Dacl(None, owner=OTHER), "another account owns it (PC\\eve)"),
        (_list(PASSED_ON, owner=None), "Windows names no owner for it"),
        (Dacl(None, owner="S-1-1-0"), "Windows names Everyone as its owner"),
        (OSError(None, "cannot read its access control list (Access is denied.)"), "cannot read"),
    ],
)
def test_a_file_another_account_owns_is_refused_never_taken_over(
    current: Any, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if isinstance(current, Dacl) and current.owner == OTHER:
        current = Dacl(current.aces, current.protected, OTHER, "PC\\eve")
    binding = _Binding({"vault.db": current})
    _windows(monkeypatch, binding)
    with pytest.raises(OSError, match=re.escape(reason)):
        winacl.make_private(Path("C:/data/vault.db"))
    assert binding.set == []


def test_a_list_that_does_not_hold_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # Read back after it is set: a drive that keeps no lists (FAT32, exFAT)
    # takes the call and stores nothing.
    binding = _Binding({"vault.db": _list(protected=False)}, holds=False)
    _windows(monkeypatch, binding)
    with pytest.raises(OSError, match="its access control list did not hold"):
        winacl.make_private(Path("D:/data/vault.db"))
    assert binding.set == [("vault.db", USER, False)]


def test_a_databases_wal_shm_and_journal_are_made_private_or_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "vault.db"
    for name in ("vault.db-wal", "vault.db-shm"):
        (tmp_path / name).write_bytes(b"")
    binding = _Binding({"vault.db-wal": _list(protected=False), "vault.db-shm": _list(PASSED_ON)})
    _windows(monkeypatch, binding)
    winacl.make_sidecars_private(database)  # no -journal: nothing to do
    assert binding.set == [("vault.db-wal", USER, False)]
    # One another account put there is refused, never opened.
    binding.lists["vault.db-shm"] = Dacl(_list().aces, True, OTHER, "PC\\eve")
    with pytest.raises(OSError, match=re.escape("another account owns it (PC\\eve)")):
        winacl.make_sidecars_private(database)


def test_a_wal_that_goes_meanwhile_is_no_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Another process closing the database removes its -wal.
    database = tmp_path / "vault.db"
    binding = _Binding({"vault.db-wal": OSError(None, "it is not there", None, 2)})
    _windows(monkeypatch, binding)
    winacl.make_sidecars_private(database)
    assert binding.set == []


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_reads_no_sidecar_list(
    tmp_path: Path, platform: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "vault.db-wal").write_bytes(b"")
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(winacl, "_windows", lambda: pytest.fail("asked Windows"))
    winacl.make_sidecars_private(tmp_path / "vault.db")


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
def made(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool | str]]:
    calls: list[tuple[str, bool | str]] = []
    monkeypatch.setattr(
        winacl, "make_private", lambda path, folder=False: calls.append((Path(path).name, folder))
    )
    monkeypatch.setattr(
        winacl, "make_sidecars_private", lambda path: calls.append((Path(path).name, "sidecars"))
    )
    return calls


def test_the_sqlite_vault_its_folder_and_sidecars_are_made_private(
    tmp_path: Path, made: list[tuple[str, bool | str]]
) -> None:
    from llm_redact.vault import _open_connection

    _open_connection(tmp_path / "data" / "vault.db").close()
    # The sidecars before SQLite opens them, and again once it has: one put
    # there meanwhile through a folder handle opened before is refused.
    assert made == [
        ("data", True),
        ("vault.db", False),
        ("vault.db", "sidecars"),
        ("vault.db", "sidecars"),
    ]


def test_a_vault_that_cannot_be_kept_private_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # serve and serve --check print it; no traceback.
    from llm_redact.config import ConfigError
    from llm_redact.vault import _open_connection

    def refused(path: Path, folder: bool = False) -> None:
        if not folder:
            raise OSError(None, "another account owns it (PC\\eve): ...", str(path))

    monkeypatch.setattr(winacl, "make_private", refused)
    vault = tmp_path / "data" / "vault.db"
    with pytest.raises(ConfigError) as refusal:
        _open_connection(vault)
    assert str(refusal.value) == (
        f"the vault {vault} cannot be kept private to this account: another account owns it"
        " (PC\\eve): ..."
    )


def test_the_override_store_its_folder_and_sidecars_are_made_private(
    tmp_path: Path, made: list[tuple[str, bool | str]]
) -> None:
    from llm_redact.overrides import OverrideStore

    store = OverrideStore(tmp_path / "data" / "overrides.db")
    try:
        assert store._open(create=True) is not None
    finally:
        store.close()
    assert made == [
        ("data", True),
        ("overrides.db", False),
        ("overrides.db", "sidecars"),
        ("overrides.db", "sidecars"),
    ]


def test_the_config_its_folder_and_backup_are_made_private(
    tmp_path: Path, made: list[tuple[str, bool | str]]
) -> None:
    from llm_redact.config_write import write_config_atomic

    target = tmp_path / "llm-redact" / "config.toml"
    write_config_atomic(target, "port = 1\n")
    assert made == [("llm-redact", True), ("config.toml.tmp", False)]
    made.clear()
    write_config_atomic(target, "port = 2\n")  # the folder exists: left as it is
    assert made == [("config.toml.bak", False), ("config.toml.tmp", False)]


def test_a_vault_backup_is_made_private(tmp_path: Path, made: list[tuple[str, bool | str]]) -> None:
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
    tmp_path: Path, made: list[tuple[str, bool | str]]
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
    # OWNER RIGHTS where Python 3.13+'s mkdir(mode=0o700) made it private,
    # this account owning it; never another principal.
    assert {ace.sid for ace in dacl.aces} <= {user, SYSTEM, ADMINISTRATORS, OWNER_RIGHTS}
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


def test_a_backup_that_cannot_be_kept_private_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.cli import main

    db = tmp_path / "vault.db"
    sqlite3.connect(db).close()

    def refused(path: Path, folder: bool = False) -> None:
        raise OSError(None, "another account owns it (PC\\eve)", str(path))

    monkeypatch.setattr(winacl, "make_private", refused)
    dest = tmp_path / "snap.db"
    with pytest.raises(SystemExit) as exit_:
        main(["vault", "backup", str(dest), "--db", str(db)])
    assert exit_.value.code == 2
    assert f"cannot keep the backup {dest} private: another account owns it" in (
        capsys.readouterr().out
    )
