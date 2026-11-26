"""Who an NTFS access control list lets in (Windows).

POSIX mode bits are synthetic on Windows (a regular file commonly reports
666), so doctor reads a file's DACL instead: ``read_dacl`` (advapi32
through ctypes) turns it into a small model, and ``foreign_access`` names
every principal other than the current user, SYSTEM and Administrators that
an allow entry lets read or change the file. The model and that decision are
plain Python and run on every OS; only ``read_dacl``, ``current_user_sid``
and the name lookup call Windows.

``make_private`` is the Windows twin of creating a file 0600 or a folder
0700: a protected DACL that lets in only this account, SYSTEM and
Administrators, inherited by everything created in a folder (a SQLite
vault's ``-wal`` and ``-shm`` files included). It does nothing elsewhere,
and nothing to a list already private (``needs_private``).

An entry counts when it allows any of ``_ACCESS`` (reading or writing the
data, changing the ACL or owner, deleting, or a generic right). Entries
that apply only to what a folder's children inherit (INHERIT_ONLY) do not
count for the folder itself, and deny entries are not subtracted: a
principal both allowed and denied is still named, which can only make
doctor stricter.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYSTEM = "S-1-5-18"
ADMINISTRATORS = "S-1-5-32-544"
# An entry for OWNER RIGHTS grants the file's owner (Python 3.13+ makes a
# mkdir(mode=0o700) folder private with one); CREATOR OWNER and CREATOR
# GROUP are placeholders for what a folder's children inherit, and match no
# account.
OWNER_RIGHTS = "S-1-3-4"
_PLACEHOLDERS = frozenset({"S-1-3-0", "S-1-3-1"})
# ACE types (winnt.h): the allow and deny entries files carry.
_ALLOWED, _DENIED, _ALLOWED_CALLBACK = 0, 1, 9
_INHERITED_ACE, _INHERIT_ONLY_ACE = 0x10, 0x08
_OBJECT_AND_CONTAINER_INHERIT = 0x01 | 0x02
# FILE_READ_DATA, FILE_WRITE_DATA, FILE_APPEND_DATA, DELETE, WRITE_DAC,
# WRITE_OWNER, GENERIC_ALL, GENERIC_WRITE, GENERIC_READ.
_ACCESS = 0x1 | 0x2 | 0x4 | 0x10000 | 0x40000 | 0x80000 | 0x10000000 | 0x40000000 | 0x80000000


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
    def shown(self) -> str:
        return self.name or self.sid


@dataclass(frozen=True)
class Dacl:
    """A discretionary ACL, and the file's owner (who may always rewrite
    it). ``protected``: it does not inherit from the folder
    (SE_DACL_PROTECTED)."""

    aces: tuple[Ace, ...]
    protected: bool = False
    owner: str | None = None
    owner_name: str | None = None


NO_DACL = "Everyone (it has no access control list)"


def foreign_access(dacl: Dacl | None, user_sid: str) -> list[str]:
    """Who besides ``user_sid``, SYSTEM and Administrators can read or
    change the file, each once: its owner when that is another account (an
    owner may always rewrite the list), then the principal of every allow
    entry, in the list's order. An OWNER RIGHTS entry stands for the owner;
    the CREATOR placeholders match no account. A missing DACL (a NULL DACL)
    lets everyone in."""
    if dacl is None:
        return [NO_DACL]
    trusted = {user_sid, SYSTEM, ADMINISTRATORS}
    names: list[str] = []
    if dacl.owner is not None and dacl.owner not in trusted:
        names.append(f"its owner, {dacl.owner_name or dacl.owner}")
    for ace in dacl.aces:
        if not ace.allow or ace.inherit_only or not ace.mask & _ACCESS:
            continue
        if ace.sid in trusted or ace.sid in _PLACEHOLDERS:
            continue
        if ace.sid == OWNER_RIGHTS and dacl.owner is not None:
            continue  # the owner: named above when it is another account
        if ace.shown not in names:
            names.append(ace.shown)
    return names


def needs_private(dacl: Dacl | None, user_sid: str, *, folder: bool) -> bool:
    """Whether ``make_private`` must set the list: anything but a protected
    list that lets no one else in (and, for a folder, passes every entry on
    to what is created in it)."""
    if dacl is None or not dacl.protected or foreign_access(dacl, user_sid):
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
    OSError when Windows refuses, so a store is never left open to
    others."""
    if sys.platform != "win32" or not hasattr(ctypes, "WinDLL"):
        return
    binding = _windows()
    user = binding.current_user_sid()
    try:
        current = binding.read_dacl(str(path))
    except OSError:
        current = None
    if needs_private(current, user, folder=folder):
        binding.make_private(str(path), user, folder=folder)
        if current is None or current.owner not in {user, SYSTEM, ADMINISTRATORS}:
            # Another account owns it (or its owner is unknown): an owner
            # may always rewrite the list, so it becomes this account's.
            # The new list grants this account WRITE_OWNER to do so.
            binding.set_owner(str(path), user)


def read_dacl(path: Path) -> Dacl | None:
    """``path``'s DACL (None: a NULL DACL). Windows only; OSError when it
    cannot be read."""
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
        self.string_to_sid = advapi32.ConvertStringSidToSidW
        self.string_to_sid.argtypes = [wintypes.LPCWSTR, pointer]
        self.string_to_sid.restype = wintypes.BOOL
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

    def read_dacl(self, path: str) -> Dacl | None:
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
            if not dacl.value:
                return None
            acl = self.acl_type.from_address(dacl.value)
            aces = []
            for index in range(acl.count):
                entry = ctypes.c_void_p()
                if not self.get_ace(dacl, index, ctypes.byref(entry)) or not entry.value:
                    raise self._failed("GetAce")
                header = self.header_type.from_address(entry.value)
                if header.type not in (_ALLOWED, _DENIED, _ALLOWED_CALLBACK):
                    continue
                body = self.ace_type.from_address(entry.value)
                sid = ctypes.c_void_p(entry.value + self.ace_type.sid_start.offset)
                aces.append(
                    Ace(
                        allow=header.type != _DENIED,
                        sid=self._sid_text(sid),
                        mask=int(body.mask),
                        flags=int(header.flags),
                        name=self._account(sid),
                    )
                )
            return Dacl(
                tuple(aces),
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

    def set_owner(self, path: str, user: str) -> None:
        ctypes = self.ctypes
        sid = ctypes.c_void_p()
        if not self.string_to_sid(user, ctypes.byref(sid)):
            raise self._failed("ConvertStringSidToSidW")
        try:
            # SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION
            status = self.set_named_security_info(path, 1, 0x1, sid, None, None, None)
            if status != 0:
                message = ctypes.FormatError(status)
                raise OSError(None, f"cannot take ownership of it ({message})", path, status)
        finally:
            self.local_free(sid)

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
