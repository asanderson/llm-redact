"""Who an NTFS access control list lets in (Windows).

POSIX mode bits are synthetic on Windows (a regular file commonly reports
666), so doctor reads a file's DACL instead: ``read_dacl`` (advapi32
through ctypes) turns it into a small model, and ``foreign_access`` names
every principal other than the current user, SYSTEM and Administrators that
the list lets read or change the file, or, for a folder, what is created in
it. The model and those decisions are plain Python and run on every OS; only
``read_dacl``, ``current_user_sid`` and the name lookup call Windows.

``make_private`` is the Windows twin of creating a file 0600 or a folder
0700: a protected DACL that lets in only this account, SYSTEM and
Administrators, inherited by everything created in a folder (a SQLite
vault's ``-wal`` and ``-shm`` files included; ``make_sidecars_private``
checks the ones already there). It does nothing elsewhere, and nothing to a
list already private (``needs_private``). It refuses a file another account
owns (an owner may always change who may open it, and a handle opened before
keeps the access it was opened with, so such a file is never taken over)
and a list that does not hold once set (a drive that keeps none).

An entry counts when it allows anything beyond reading the file's
attributes and its list (``_HARMLESS``). An entry that applies only to what
a folder passes on (INHERIT_ONLY) counts for the folder, whose new files get
it, not for a file. Deny entries are not subtracted: a principal both
allowed and denied is still named, which can only make doctor stricter. An
allow entry of a kind this module does not read (an object or compound
entry) is named by its type, never skipped.
"""

from __future__ import annotations

import ctypes
import functools
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYSTEM = "S-1-5-18"
ADMINISTRATORS = "S-1-5-32-544"
EVERYONE = "S-1-1-0"
# An entry for OWNER RIGHTS grants the file's owner (Python 3.13+ makes a
# mkdir(mode=0o700) folder private with one). CREATOR OWNER and CREATOR
# GROUP stand for what a folder passes on: inherited, CREATOR OWNER becomes
# the account that creates the file, CREATOR GROUP that account's primary
# group (Domain Users, for a domain account); on the folder itself they
# match no one.
OWNER_RIGHTS = "S-1-3-4"
CREATOR_OWNER = "S-1-3-0"
CREATOR_GROUP = "S-1-3-1"
# ACE types (winnt.h) whose SID follows the mask, read in full, and whether
# each allows: plain and callback entries.
_SID_AFTER_MASK = {0: True, 1: False, 9: True, 10: False}
# Deny entries laid out otherwise (object entries): never subtracted, so
# not read.
_OTHER_DENY_TYPES = frozenset({6, 12})
_INHERITED_ACE, _INHERIT_ONLY_ACE = 0x10, 0x08
_OBJECT_AND_CONTAINER_INHERIT = 0x01 | 0x02
# READ_CONTROL, SYNCHRONIZE, FILE_READ_ATTRIBUTES: nothing of the data,
# nothing changed. Every other right counts (deleting a folder's files,
# traversing it, extended attributes, the generic rights).
_HARMLESS = 0x20000 | 0x100000 | 0x80


@dataclass(frozen=True)
class Ace:
    """One access control entry: allow or deny, for whom, which rights."""

    allow: bool
    sid: str
    mask: int
    flags: int = 0
    name: str | None = None  # DOMAIN\\name, when the account resolves

    @property
    def inherited(self) -> bool:
        return bool(self.flags & _INHERITED_ACE)

    @property
    def inherit_only(self) -> bool:
        return bool(self.flags & _INHERIT_ONLY_ACE)

    @property
    def passed_on(self) -> bool:
        """Whether a folder passes it on to the files or folders made in it."""
        return bool(self.flags & _OBJECT_AND_CONTAINER_INHERIT)

    @property
    def shown(self) -> str:
        return self.name or self.sid


@dataclass(frozen=True)
class Dacl:
    """A discretionary ACL (``aces`` None: none at all, a NULL DACL, which
    lets everyone in), and the file's owner (who may always rewrite it).
    ``protected``: it does not inherit from the folder (SE_DACL_PROTECTED)."""

    aces: tuple[Ace, ...] | None
    protected: bool = False
    owner: str | None = None
    owner_name: str | None = None


NO_DACL = "Everyone (it has no access control list)"
_CREATOR_GROUP_SHOWN = "CREATOR GROUP (the primary group of an account creating a file there)"


def entry_for(
    ace_type: int, flags: int, mask: int, principal: Callable[[], tuple[str, str | None]]
) -> Ace | None:
    """The model of one entry of a list (``read_dacl``): a plain or callback
    allow or deny entry read in full (``principal()``: its SID and account
    name), None for a deny entry laid out otherwise (deny entries are never
    subtracted), and any other kind (an object or compound allow entry) one
    that lets in someone this module cannot name, by its type."""
    allow = _SID_AFTER_MASK.get(ace_type)
    if allow is not None:
        sid, name = principal()
        return Ace(allow, sid, mask, flags, name)
    if ace_type in _OTHER_DENY_TYPES:
        return None
    return Ace(True, f"an access control entry of type {ace_type}", mask, flags)


def foreign_access(dacl: Dacl, user_sid: str, *, folder: bool = False) -> list[str]:
    """Who besides ``user_sid``, SYSTEM and Administrators can read or
    change the file, or, for a ``folder``, what is created in it, each once:
    its owner when that is another account (an owner may always rewrite the
    list), then the principal of every allow entry, in the list's order. An
    OWNER RIGHTS entry stands for the owner, a CREATOR OWNER one for the
    account that creates a file; CREATOR GROUP counts where a folder passes
    it on. No list at all (a NULL DACL) lets everyone in."""
    trusted = {user_sid, SYSTEM, ADMINISTRATORS}
    names: list[str] = []
    if dacl.owner is not None and dacl.owner not in trusted:
        names.append(f"its owner, {dacl.owner_name or dacl.owner}")
    if dacl.aces is None:
        return [*names, NO_DACL]
    for ace in dacl.aces:
        here = not ace.inherit_only
        inherited = folder and ace.passed_on
        if not ace.allow or not ace.mask & ~_HARMLESS or not (here or inherited):
            continue
        if ace.sid in trusted or ace.sid == CREATOR_OWNER:
            continue
        if ace.sid == OWNER_RIGHTS and not (here and dacl.owner is None):
            continue  # the owner (named above when another account), or a new file's
        if ace.sid == CREATOR_GROUP and not inherited:
            continue  # matches no one on the folder itself
        shown = _CREATOR_GROUP_SHOWN if ace.sid == CREATOR_GROUP else ace.shown
        if shown not in names:
            names.append(shown)
    return names


def ownership_problem(dacl: Dacl, user_sid: str) -> str | None:
    """Why ``make_private`` refuses a file, or None: it is this account's,
    SYSTEM's or Administrators' (an elevated process gives what it creates
    to the group). Another owner may always change who may open the file,
    and a handle it opened before keeps its access whatever list is set, so
    the file is never taken over."""
    if dacl.owner in {user_sid, SYSTEM, ADMINISTRATORS}:
        return None
    if dacl.owner is None:
        return "Windows names no owner for it, so it cannot be kept private"
    if dacl.owner == EVERYONE:
        return (
            "Windows names Everyone as its owner, as on a drive that keeps no access"
            " control lists (FAT32, exFAT): every account may open what is stored there;"
            " choose a path on an NTFS drive"
        )
    return (
        f"another account owns it ({dacl.owner_name or dacl.owner}): its owner may always"
        " change who may open it, and a handle it opened keeps its access, so it is not"
        " taken over; move it aside or choose another path"
    )


def needs_private(dacl: Dacl, user_sid: str, *, folder: bool) -> bool:
    """Whether ``make_private`` must set the list: anything but a protected
    list that lets no one else in (and, for a folder, passes every entry on
    to what is created in it)."""
    if dacl.aces is None or not dacl.protected or foreign_access(dacl, user_sid, folder=folder):
        return True
    passed_on = all(
        ace.flags & _OBJECT_AND_CONTAINER_INHERIT == _OBJECT_AND_CONTAINER_INHERIT
        for ace in dacl.aces
        if ace.allow
    )
    return folder and not passed_on


def make_private(path: Path, *, folder: bool = False) -> None:
    """On Windows, let only this account, SYSTEM and Administrators into
    ``path`` (a protected DACL; for a folder, one everything created in it
    inherits). Nothing elsewhere, where the POSIX mode does it, nor where
    ctypes has no WinDLL (only a platform name faked in a test lacks it).
    OSError, so a store is never left open to others, when the list cannot
    be read, when another account owns the file (``ownership_problem``),
    when Windows refuses the new list, and when it does not hold once set
    (it is read back: a drive that keeps no lists takes the call and stores
    nothing)."""
    if not _on_windows():
        return
    binding = _windows()
    user = binding.current_user_sid()
    current = binding.read_dacl(str(path))
    refusal = ownership_problem(current, user)
    if refusal is not None:
        raise OSError(None, refusal, str(path))
    if not needs_private(current, user, folder=folder):
        return
    binding.make_private(str(path), user, folder=folder)
    if needs_private(binding.read_dacl(str(path)), user, folder=folder):
        raise OSError(
            None,
            "its access control list did not hold (a drive that keeps none, such as"
            " FAT32 or exFAT?): choose a path on an NTFS drive",
            str(path),
        )


# The files SQLite keeps beside a database while it is open.
SQLITE_SIDECARS = ("-wal", "-shm", "-journal")


def make_sidecars_private(database: Path) -> None:
    """``make_private`` for each of the database's ``-wal``, ``-shm`` and
    ``-journal`` files that is there: one left by an earlier run keeps its
    own list (a folder's new list reaches only what inherits from it), and
    one another account put there while the folder was open to it is
    refused before SQLite opens it. One that goes meanwhile (another process
    closing the database) is no refusal. Nothing but on Windows."""
    if not _on_windows():
        return
    for suffix in SQLITE_SIDECARS:
        sidecar = database.with_name(database.name + suffix)
        try:
            make_private(sidecar)
        except OSError:
            if sidecar.exists():
                raise


def _on_windows() -> bool:
    return sys.platform == "win32" and hasattr(ctypes, "WinDLL")


def read_dacl(path: Path) -> Dacl:
    """``path``'s DACL and owner. Windows only; OSError when it cannot be
    read."""
    if sys.platform != "win32":
        raise OSError("NTFS access control lists are read on Windows only")
    return _windows().read_dacl(str(path))


def current_user_sid() -> str:
    """The SID of the account this process runs as. Windows only."""
    if sys.platform != "win32":
        raise OSError("Windows SIDs are read on Windows only")
    return _windows().current_user_sid()


def _windows() -> _Windows:
    """The bound calls; OSError where Windows' security API is missing."""
    try:
        return _Windows()
    except (AttributeError, OSError) as problem:
        raise OSError(None, f"the Windows security API is not available ({problem})") from None


class _Windows:
    """The advapi32 calls, through ctypes."""

    def __init__(self) -> None:
        import ctypes as _ctypes
        from ctypes import wintypes

        # Any: the Windows-only parts of ctypes (WinDLL, FormatError,
        # get_last_error) are absent from the stubs of other platforms.
        ctypes: Any = _ctypes
        self.ctypes = ctypes
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        pointer = ctypes.POINTER(ctypes.c_void_p)

        self.get_named_security_info = advapi32.GetNamedSecurityInfoW
        self.get_named_security_info.argtypes = [
            wintypes.LPCWSTR,
            ctypes.c_int,
            wintypes.DWORD,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
        ]
        self.get_named_security_info.restype = wintypes.DWORD
        self.get_control = advapi32.GetSecurityDescriptorControl
        self.get_control.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.WORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.get_control.restype = wintypes.BOOL
        self.get_ace = advapi32.GetAce
        self.get_ace.argtypes = [ctypes.c_void_p, wintypes.DWORD, pointer]
        self.get_ace.restype = wintypes.BOOL
        self.sid_to_string = advapi32.ConvertSidToStringSidW
        self.sid_to_string.argtypes = [ctypes.c_void_p, pointer]
        self.sid_to_string.restype = wintypes.BOOL
        self.lookup_account_sid = advapi32.LookupAccountSidW
        self.lookup_account_sid.argtypes = [
            wintypes.LPCWSTR,
            ctypes.c_void_p,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lookup_account_sid.restype = wintypes.BOOL
        self.open_process_token = advapi32.OpenProcessToken
        self.open_process_token.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        self.open_process_token.restype = wintypes.BOOL
        self.get_token_information = advapi32.GetTokenInformation
        self.get_token_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.get_token_information.restype = wintypes.BOOL
        self.current_process = kernel32.GetCurrentProcess
        self.current_process.argtypes = []
        self.current_process.restype = wintypes.HANDLE
        self.close_handle = kernel32.CloseHandle
        self.close_handle.argtypes = [wintypes.HANDLE]
        self.close_handle.restype = wintypes.BOOL
        self.string_to_descriptor = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
        self.string_to_descriptor.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            pointer,
            ctypes.POINTER(wintypes.ULONG),
        ]
        self.string_to_descriptor.restype = wintypes.BOOL
        self.get_dacl = advapi32.GetSecurityDescriptorDacl
        self.get_dacl.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.BOOL),
            pointer,
            ctypes.POINTER(wintypes.BOOL),
        ]
        self.get_dacl.restype = wintypes.BOOL
        self.set_named_security_info = advapi32.SetNamedSecurityInfoW
        self.set_named_security_info.argtypes = [
            wintypes.LPWSTR,
            ctypes.c_int,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.set_named_security_info.restype = wintypes.DWORD
        self.local_free = kernel32.LocalFree
        self.local_free.argtypes = [ctypes.c_void_p]
        self.local_free.restype = ctypes.c_void_p

        class Acl(_ctypes.Structure):
            _fields_ = [
                ("revision", ctypes.c_ubyte),
                ("sbz1", ctypes.c_ubyte),
                ("size", wintypes.WORD),
                ("count", wintypes.WORD),
                ("sbz2", wintypes.WORD),
            ]

        class AceHeader(_ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_ubyte),
                ("flags", ctypes.c_ubyte),
                ("size", wintypes.WORD),
            ]

        class AllowedAce(_ctypes.Structure):  # ACCESS_ALLOWED_ACE and its kin
            _fields_ = [
                ("header", AceHeader),
                ("mask", wintypes.DWORD),
                ("sid_start", wintypes.DWORD),
            ]

        self.acl_type, self.header_type, self.ace_type = Acl, AceHeader, AllowedAce
        self.wintypes = wintypes

    def _failed(self, what: str) -> OSError:
        code = self.ctypes.get_last_error()
        return OSError(None, f"{what} failed ({self.ctypes.FormatError(code)})", None, code)

    def read_dacl(self, path: str) -> Dacl:
        ctypes = self.ctypes
        owner = ctypes.c_void_p()
        dacl = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()
        # SE_FILE_OBJECT; OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION
        status = self.get_named_security_info(
            path,
            1,
            0x1 | 0x4,
            ctypes.byref(owner),
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
        if status != 0:
            message = ctypes.FormatError(status)
            raise OSError(None, f"cannot read its access control list ({message})", path, status)
        try:
            control = self.wintypes.WORD()
            revision = self.wintypes.DWORD()
            if not self.get_control(descriptor, ctypes.byref(control), ctypes.byref(revision)):
                raise self._failed("GetSecurityDescriptorControl")
            aces: list[Ace] | None = None
            if dacl.value:
                acl = self.acl_type.from_address(dacl.value)
                aces = []
                for index in range(acl.count):
                    entry = ctypes.c_void_p()
                    if not self.get_ace(dacl, index, ctypes.byref(entry)) or not entry.value:
                        raise self._failed("GetAce")
                    header = self.header_type.from_address(entry.value)
                    # Every kind of entry has its mask right after its header.
                    body = self.ace_type.from_address(entry.value)
                    sid = ctypes.c_void_p(entry.value + self.ace_type.sid_start.offset)
                    ace = entry_for(
                        int(header.type),
                        int(header.flags),
                        int(body.mask),
                        functools.partial(self._principal, sid),
                    )
                    if ace is not None:
                        aces.append(ace)
            return Dacl(
                None if aces is None else tuple(aces),
                protected=bool(control.value & 0x1000),  # SE_DACL_PROTECTED
                owner=self._sid_text(owner) if owner.value else None,
                owner_name=self._account(owner) if owner.value else None,
            )
        finally:
            self.local_free(descriptor)

    def make_private(self, path: str, user: str, *, folder: bool) -> None:
        ctypes = self.ctypes
        inherit = "OICI" if folder else ""
        # Full access for SYSTEM, Administrators and the user, protected (P):
        # nothing inherited from the folder above.
        sddl = f"D:P(A;{inherit};FA;;;SY)(A;{inherit};FA;;;BA)(A;{inherit};FA;;;{user})"
        descriptor = ctypes.c_void_p()
        if not self.string_to_descriptor(sddl, 1, ctypes.byref(descriptor), None):
            raise self._failed("ConvertStringSecurityDescriptorToSecurityDescriptorW")
        try:
            present = self.wintypes.BOOL()
            defaulted = self.wintypes.BOOL()
            dacl = ctypes.c_void_p()
            if not self.get_dacl(
                descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)
            ):
                raise self._failed("GetSecurityDescriptorDacl")
            # SE_FILE_OBJECT; DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION
            status = self.set_named_security_info(path, 1, 0x4 | 0x80000000, None, None, dacl, None)
            if status != 0:
                message = ctypes.FormatError(status)
                raise OSError(None, f"cannot make it private ({message})", path, status)
        finally:
            self.local_free(descriptor)

    def _principal(self, sid: object) -> tuple[str, str | None]:
        return self._sid_text(sid), self._account(sid)

    def _sid_text(self, sid: object) -> str:
        ctypes = self.ctypes
        text = ctypes.c_void_p()
        if not self.sid_to_string(sid, ctypes.byref(text)):
            raise self._failed("ConvertSidToStringSidW")
        try:
            return str(ctypes.wstring_at(text.value))
        finally:
            self.local_free(text)

    def _account(self, sid: object) -> str | None:
        ctypes = self.ctypes
        name_size = self.wintypes.DWORD(0)
        domain_size = self.wintypes.DWORD(0)
        use = ctypes.c_int()
        self.lookup_account_sid(
            None,
            sid,
            None,
            ctypes.byref(name_size),
            None,
            ctypes.byref(domain_size),
            ctypes.byref(use),
        )
        if not name_size.value:
            return None  # an account Windows cannot name (a deleted one)
        name = ctypes.create_unicode_buffer(name_size.value)
        domain = ctypes.create_unicode_buffer(max(domain_size.value, 1))
        if not self.lookup_account_sid(
            None,
            sid,
            name,
            ctypes.byref(name_size),
            domain,
            ctypes.byref(domain_size),
            ctypes.byref(use),
        ):
            return None
        return f"{domain.value}\\{name.value}" if domain.value else str(name.value)

    def current_user_sid(self) -> str:
        ctypes = self.ctypes
        token = self.wintypes.HANDLE()
        # TOKEN_QUERY
        if not self.open_process_token(self.current_process(), 0x0008, ctypes.byref(token)):
            raise self._failed("OpenProcessToken")
        try:
            size = self.wintypes.DWORD(0)
            self.get_token_information(token, 1, None, 0, ctypes.byref(size))  # TokenUser
            if not size.value:
                raise self._failed("GetTokenInformation")
            buffer = ctypes.create_string_buffer(size.value)
            if not self.get_token_information(token, 1, buffer, size, ctypes.byref(size)):
                raise self._failed("GetTokenInformation")
            # TOKEN_USER starts with SID_AND_ATTRIBUTES, which starts with the SID pointer.
            sid = ctypes.c_void_p.from_buffer(buffer)
            return self._sid_text(sid)
        finally:
            self.close_handle(token)
