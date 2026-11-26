"""doctor reads a store's NTFS access control list on Windows, and opens a
TLS file to check it can be read.

POSIX mode bits are synthetic on Windows (a regular file commonly reports
666): doctor used to WARN that it could not check anything. It now reads the
DACL (``llm_redact.winacl``) and FAILs an allow entry for anyone but this
account, SYSTEM and Administrators. ``os.access`` does not read access
control lists on Windows (only the read-only attribute), so a TLS file
serve cannot open passed; doctor now opens it.

The ACL model and doctor's decision run on every OS with stand-in lists; the
real-API tests run on the Windows legs (icacls changes the lists)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from llm_redact import winacl
from llm_redact.winacl import ADMINISTRATORS, NO_DACL, SYSTEM, Ace, Dacl, foreign_access

USER = "S-1-5-21-1000-2000-3000-1001"
USERS = "S-1-5-32-545"
AUTHENTICATED = "S-1-5-11"
CREATOR_OWNER = "S-1-3-0"
FULL, READ, SYNC_ONLY = 0x1F01FF, 0x120089, 0x120000  # FILE_ALL_ACCESS, FILE_GENERIC_READ, ...
INHERITED, INHERIT_ONLY = 0x10, 0x08


def _private(*extra: Ace, protected: bool = False) -> Dacl:
    return Dacl(
        (
            Ace(True, USER, FULL, name="RUNNER\\me"),
            Ace(True, SYSTEM, FULL, INHERITED, "NT AUTHORITY\\SYSTEM"),
            Ace(True, ADMINISTRATORS, FULL, INHERITED, "BUILTIN\\Administrators"),
            *extra,
        ),
        protected=protected,
    )


def test_a_list_of_this_account_system_and_administrators_lets_no_one_else_in() -> None:
    assert foreign_access(_private(), USER) == []


def test_an_entry_for_users_lets_them_in() -> None:
    dacl = _private(Ace(True, USERS, READ, INHERITED, "BUILTIN\\Users"))
    assert foreign_access(dacl, USER) == ["BUILTIN\\Users"]


def test_another_account_is_named_once_and_an_unnamed_one_by_its_sid() -> None:
    dacl = _private(
        Ace(True, AUTHENTICATED, READ, name="NT AUTHORITY\\Authenticated Users"),
        Ace(True, AUTHENTICATED, 0x80000000, name="NT AUTHORITY\\Authenticated Users"),
        Ace(True, "S-1-5-21-9-9-9-500", FULL),
    )
    assert foreign_access(dacl, USER) == [
        "NT AUTHORITY\\Authenticated Users",
        "S-1-5-21-9-9-9-500",
    ]


def test_entries_that_grant_nothing_here_do_not_count() -> None:
    dacl = _private(
        Ace(True, CREATOR_OWNER, FULL, INHERIT_ONLY, "CREATOR OWNER"),  # for children only
        Ace(False, USERS, FULL, name="BUILTIN\\Users"),  # a deny entry
        Ace(True, USERS, SYNC_ONLY, name="BUILTIN\\Users"),  # synchronize, read the ACL
    )
    assert foreign_access(dacl, USER) == []


def test_no_list_at_all_lets_everyone_in() -> None:
    assert foreign_access(None, USER) == [NO_DACL]


def test_the_model_says_what_is_inherited_and_what_is_protected() -> None:
    ace = Ace(True, SYSTEM, FULL, INHERITED | INHERIT_ONLY)
    assert ace.inherited and ace.inherit_only and ace.shown == SYSTEM
    assert _private(protected=True).protected and not _private().protected


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_lists_are_read_on_windows_only(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(OSError, match="Windows only"):
        winacl.read_dacl(Path("vault.db"))
    with pytest.raises(OSError, match="Windows only"):
        winacl.current_user_sid()


@pytest.mark.skipif(sys.platform == "win32", reason="the API is there on Windows")
def test_a_missing_security_api_is_an_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(OSError, match="the Windows security API is not available"):
        winacl.current_user_sid()


def _doctor(
    monkeypatch: pytest.MonkeyPatch, lists: dict[str, Any], path: Path, folder: bool = False
) -> list[dict[str, str]]:
    import llm_redact.doctor_cli as doctor

    def read(target: Path) -> Dacl | None:
        found = lists[target.name]
        if isinstance(found, OSError):
            raise found
        return found  # type: ignore[no-any-return]

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(winacl, "read_dacl", read)
    monkeypatch.setattr(winacl, "current_user_sid", lambda: USER)
    report = doctor._Report(json_mode=True)
    doctor._check_private(report, "vault", path, folder)
    return report.rows


def test_doctor_fails_a_store_others_can_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault.db"
    users = _private(Ace(True, USERS, READ, INHERITED, "BUILTIN\\Users"))
    (row,) = _doctor(monkeypatch, {"vault.db": users}, vault)
    assert row["level"] == "FAIL"
    assert row["message"].startswith(
        f"{vault} lets in BUILTIN\\Users (its NTFS access control list): remove those entries,"
        " so that only your account, SYSTEM and Administrators keep access (icacls "
    )


def test_doctor_passes_a_private_store_and_says_how_it_is_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault.db"
    (protected,) = _doctor(monkeypatch, {"vault.db": _private(protected=True)}, vault)
    assert protected == {
        "level": "PASS",
        "area": "vault",
        "message": f"{vault} is private to your account (NTFS access control list, protected)",
    }
    (inherited,) = _doctor(monkeypatch, {"vault.db": _private()}, vault)
    assert inherited["message"].endswith("(NTFS access control list, inherited from its folder)")


def test_doctor_checks_the_folder_too_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault.db"
    folder = _private(Ace(True, AUTHENTICATED, FULL, name="NT AUTHORITY\\Authenticated Users"))
    rows = _doctor(monkeypatch, {"vault.db": _private(), tmp_path.name: folder}, vault, True)
    assert [row["level"] for row in rows] == ["PASS", "FAIL"]
    assert rows[1]["message"].startswith(f"{tmp_path} lets in NT AUTHORITY\\Authenticated Users")


def test_doctor_warns_when_it_cannot_read_a_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault.db"
    error = OSError(None, "cannot read its access control list (Access is denied.)")
    (row,) = _doctor(monkeypatch, {"vault.db": error}, vault)
    assert row == {
        "level": "WARN",
        "area": "vault",
        "message": f"{vault}: cannot check its NTFS access control list (cannot read its access"
        " control list (Access is denied.))",
    }


def test_doctor_opens_a_tls_file_to_check_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llm_redact.doctor_cli as doctor

    readable = tmp_path / "cert.pem"
    readable.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    locked = tmp_path / "key.pem"
    locked.write_text("-----BEGIN PRIVATE KEY-----\n", encoding="utf-8")
    real_open = Path.open

    def open_(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == locked:
            raise PermissionError(13, "Access is denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_)
    assert doctor._readable(readable)
    assert not doctor._readable(locked)
    assert not doctor._readable(tmp_path / "missing.pem")


# Real access control lists, Windows only.

on_windows = pytest.mark.skipif(sys.platform != "win32", reason="NTFS access control lists")


def _icacls(*arguments: str) -> None:
    icacls = Path(os.environ.get("SYSTEMROOT", "C:\\Windows")) / "System32" / "icacls.exe"
    subprocess.run([str(icacls), *arguments], check=True, capture_output=True, timeout=60)


@on_windows
def test_the_real_list_of_a_private_file_and_one_users_can_read(tmp_path: Path) -> None:
    import llm_redact.doctor_cli as doctor

    user = winacl.current_user_sid()
    assert user.startswith("S-1-5-")
    vault = tmp_path / "vault.db"
    vault.write_bytes(b"")
    private = (f"*{user}:F", f"*{SYSTEM}:F", f"*{ADMINISTRATORS}:F")
    _icacls(str(vault), "/inheritance:r", "/grant:r", *private)
    dacl = winacl.read_dacl(vault)
    assert dacl is not None and dacl.protected
    assert {ace.sid for ace in dacl.aces} == {user, SYSTEM, ADMINISTRATORS}
    assert foreign_access(dacl, user) == []
    report = doctor._Report(json_mode=True)
    doctor._check_private(report, "vault", vault, False)
    assert [row["level"] for row in report.rows] == ["PASS"]

    _icacls(str(vault), "/grant", f"*{USERS}:R")
    dacl = winacl.read_dacl(vault)
    assert dacl is not None and USERS in {ace.sid for ace in dacl.aces}
    (others,) = foreign_access(dacl, user)
    assert others.endswith("\\Users") or others == USERS
    report = doctor._Report(json_mode=True)
    doctor._check_private(report, "vault", vault, False)
    assert [row["level"] for row in report.rows] == ["FAIL"]


@on_windows
def test_a_tls_file_this_account_may_not_read_fails_although_os_access_passes(
    tmp_path: Path,
) -> None:
    import llm_redact.doctor_cli as doctor

    user = winacl.current_user_sid()
    key = tmp_path / "key.pem"
    key.write_text("-----BEGIN PRIVATE KEY-----\n", encoding="utf-8")
    _icacls(str(key), "/deny", f"*{user}:R")
    try:
        assert os.access(key, os.R_OK)  # why doctor no longer asks it
        assert not doctor._readable(key)
    finally:
        _icacls(str(key), "/remove:d", f"*{user}")
