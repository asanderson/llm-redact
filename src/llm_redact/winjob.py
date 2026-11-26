"""The document-extraction worker's limits on Windows: a Job Object.

On POSIX the worker caps itself with setrlimit before it reads the file
(``extract_worker.apply_limits``). Windows has no such limits. There the
proxy puts the worker in a Job Object right after it starts, before any
byte of the file reaches it (``confine``), with the limits Windows has:

- a process memory limit of ``worker_memory_mb`` (committed memory: an
  allocation past it fails, and the worker ends with nothing read);
- a CPU-time limit (Windows ends the worker past it);
- one active process (the worker can start no other);
- kill on job close: the proxy holds the job's only handle, so the worker
  ends when the proxy closes it, and when the proxy itself is killed.

Nothing stops the worker writing files or opening many: Windows has no
equivalent of RLIMIT_FSIZE or RLIMIT_NOFILE (docs/extraction.md says so).
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any

# JOB_OBJECT_LIMIT_PROCESS_TIME, _ACTIVE_PROCESS, _PROCESS_MEMORY,
# _DIE_ON_UNHANDLED_EXCEPTION, _KILL_ON_JOB_CLOSE
LIMIT_FLAGS = 0x0002 | 0x0008 | 0x0100 | 0x0400 | 0x2000
# JobObjectExtendedLimitInformation
_EXTENDED_LIMITS = 9
# PROCESS_SET_QUOTA | PROCESS_TERMINATE: what AssignProcessToJobObject needs.
_ASSIGN_ACCESS = 0x0100 | 0x0001
# CPU time is counted in 100-nanosecond units.
_TICKS_PER_SECOND = 10_000_000


def limits(memory_bytes: int, cpu_seconds: int) -> dict[str, int]:
    """The job's limits, as JOBOBJECT_EXTENDED_LIMIT_INFORMATION fields."""
    return {
        "LimitFlags": LIMIT_FLAGS,
        "PerProcessUserTimeLimit": cpu_seconds * _TICKS_PER_SECOND,
        "ActiveProcessLimit": 1,
        "ProcessMemoryLimit": memory_bytes,
    }


class Job:
    """A Job Object holding one worker; ``close`` ends whatever still runs
    in it (kill on job close)."""

    def __init__(self, handle: int, binding: _Kernel32) -> None:
        self._handle: int | None = handle
        self._binding = binding

    def close(self) -> None:
        if self._handle is not None:
            handle, self._handle = self._handle, None
            self._binding.close(handle)


def available() -> bool:
    """Whether Job Objects can be made here: Windows, with ctypes' WinDLL
    (only a platform name faked in a test lacks it)."""
    return sys.platform == "win32" and hasattr(ctypes, "WinDLL")


def confine(pid: int, *, memory_bytes: int, cpu_seconds: int) -> Job:
    """Put process ``pid`` in a new Job Object with ``limits``. OSError when
    Windows refuses (the caller ends the worker unread)."""
    if not available():
        raise OSError("Job Objects are a Windows facility")
    binding = _Kernel32()
    handle = binding.create(limits(memory_bytes, cpu_seconds))
    try:
        binding.assign(handle, pid)
    except BaseException:
        binding.close(handle)
        raise
    return Job(handle, binding)


class _Kernel32:
    """The kernel32 calls, through ctypes."""

    def __init__(self) -> None:
        from ctypes import wintypes

        # Any: WinDLL and get_last_error exist in the stubs of Windows only.
        api: Any = ctypes
        self.api = api
        kernel32 = api.WinDLL("kernel32", use_last_error=True)
        self.create_job = kernel32.CreateJobObjectW
        self.create_job.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.create_job.restype = wintypes.HANDLE
        self.set_information = kernel32.SetInformationJobObject
        self.set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.set_information.restype = wintypes.BOOL
        self.open_process = kernel32.OpenProcess
        self.open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.open_process.restype = wintypes.HANDLE
        self.assign_process = kernel32.AssignProcessToJobObject
        self.assign_process.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.assign_process.restype = wintypes.BOOL
        self.close_handle = kernel32.CloseHandle
        self.close_handle.argtypes = [wintypes.HANDLE]
        self.close_handle.restype = wintypes.BOOL

        class Basic(ctypes.Structure):  # JOBOBJECT_BASIC_LIMIT_INFORMATION
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class Counters(ctypes.Structure):  # IO_COUNTERS
            _fields_ = [
                (name, ctypes.c_uint64)
                for name in ("Reads", "Writes", "Others", "ReadBytes", "WriteBytes", "OtherBytes")
            ]

        class Extended(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
            _fields_ = [
                ("BasicLimitInformation", Basic),
                ("IoInfo", Counters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.extended_type = Extended

    def _failed(self, what: str) -> OSError:
        code = self.api.get_last_error()
        return OSError(None, f"{what} failed ({self.api.FormatError(code)})", None, code)

    def create(self, fields: dict[str, int]) -> int:
        handle = self.create_job(None, None)
        if not handle:
            raise self._failed("CreateJobObjectW")
        info = self.extended_type()
        basic = info.BasicLimitInformation
        basic.LimitFlags = fields["LimitFlags"]
        basic.PerProcessUserTimeLimit = fields["PerProcessUserTimeLimit"]
        basic.ActiveProcessLimit = fields["ActiveProcessLimit"]
        info.ProcessMemoryLimit = fields["ProcessMemoryLimit"]
        if not self.set_information(
            handle, _EXTENDED_LIMITS, ctypes.byref(info), ctypes.sizeof(info)
        ):
            error = self._failed("SetInformationJobObject")
            self.close(handle)
            raise error
        return int(handle)

    def assign(self, job: int, pid: int) -> None:
        process = self.open_process(_ASSIGN_ACCESS, False, pid)
        if not process:
            raise self._failed("OpenProcess")
        try:
            if not self.assign_process(job, process):
                raise self._failed("AssignProcessToJobObject")
        finally:
            self.close_handle(process)

    def close(self, handle: int) -> None:
        self.close_handle(handle)
