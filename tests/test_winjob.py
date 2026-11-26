"""On Windows the document-extraction worker runs in a Job Object
(``llm_redact.winjob``): POSIX caps the worker with setrlimit, and Windows
has no such limits, so before, the worker ran with none — no memory, CPU or
child-process limit, and it outlived a proxy that was killed. Its PATH was
``os.defpath``, which on Windows is ``.;C:\\bin``.

The proxy now confines the worker right after it starts, before a byte of
the file reaches it: memory ``worker_memory_mb``, CPU time, one process,
killed when the job closes. A worker that cannot be confined reads nothing.
The stand-ins run on every OS; the real Job Objects on the Windows legs."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llm_redact import extraction, winjob
from llm_redact.config import ExtractionConfig
from llm_redact.extraction import ExtractionInspector

# A worker stand-in: reads the file, answers with its length.
ECHO = (
    "import sys, json; data = sys.stdin.buffer.read();"
    " print(json.dumps({'format': 'text', 'text': str(len(data)), 'complete': True}))"
)


def test_the_job_limits_the_worker_as_setrlimit_does() -> None:
    assert winjob.limits(128 << 20, 31) == {
        # process time, one process, process memory, die on an unhandled
        # exception, kill on job close
        "LimitFlags": 0x0002 | 0x0008 | 0x0100 | 0x0400 | 0x2000,
        "PerProcessUserTimeLimit": 31 * 10_000_000,
        "ActiveProcessLimit": 1,
        "ProcessMemoryLimit": 128 << 20,
    }


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_job_objects_are_made_on_windows_only(
    platform: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    assert not winjob.available()
    with pytest.raises(OSError, match="Windows facility"):
        winjob.confine(1, memory_bytes=1 << 20, cpu_seconds=1)


def test_a_faked_windows_platform_makes_no_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(winjob, "ctypes", SimpleNamespace())
    assert not winjob.available()


class _Job:
    closed = False

    def close(self) -> None:
        self.closed = True


def _inspector(**config: Any) -> ExtractionInspector:
    return ExtractionInspector(
        ExtractionConfig(enabled=True, **config), command=(sys.executable, "-I", "-c", ECHO)
    )


async def test_the_worker_is_confined_before_it_reads_and_released_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _Job()
    asked: list[tuple[int, int, int]] = []

    def confine(pid: int, *, memory_bytes: int, cpu_seconds: int) -> _Job:
        asked.append((pid, memory_bytes, cpu_seconds))
        return job

    monkeypatch.setattr(extraction.winjob, "available", lambda: True)
    monkeypatch.setattr(extraction.winjob, "confine", confine)
    inspector = _inspector(worker_memory_mb=256, timeout_seconds=5.5)
    assert await inspector._run_worker(b"12345") == {
        "format": "text",
        "text": "5",
        "complete": True,
    }
    ((pid, memory, cpu),) = asked
    assert pid > 0 and memory == 256 << 20 and cpu == 7  # as the worker's RLIMIT_CPU
    assert job.closed


async def test_a_worker_that_cannot_be_confined_reads_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused(pid: int, **limits: int) -> None:
        raise OSError(None, "AssignProcessToJobObject failed (Access is denied.)")

    monkeypatch.setattr(extraction.winjob, "available", lambda: True)
    monkeypatch.setattr(extraction.winjob, "confine", refused)
    inspector = _inspector()
    assert await inspector._local(b"12345") is None
    assert inspector.status()["readings_total"] == {"local": {"failed": 1}}
    assert inspector.killed == 1 and inspector.running == 0  # ended unread, reaped


async def test_posix_confines_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extraction.winjob, "available", lambda: False)
    monkeypatch.setattr(extraction.winjob, "confine", lambda *a, **k: pytest.fail("confined"))
    assert (await _inspector()._run_worker(b"123"))["text"] == "3"


def test_the_memory_limit_counts_as_enforced_on_windows() -> None:
    # Only macOS ignores it now: the input bound applies there alone.
    assert extraction.MEMORY_LIMIT_ENFORCED is (sys.platform != "darwin")


VENV = "C:\\work\\.venv\\Scripts\\python.exe"
BASE = "C:\\Python312\\python.exe"


def test_in_a_virtual_environment_the_worker_runs_the_base_interpreter() -> None:
    # A virtual environment's python.exe is a launcher: it starts the base
    # interpreter as a process of its own. Confined to one process, it could
    # not (every worker exited 101 on the Windows legs), and the process
    # reading the file would not have been the one confined.
    command, env = extraction.worker_launch(
        (VENV, "-I", "-m", "llm_redact.extract_worker"),
        platform="win32",
        executable=VENV,
        base_executable=BASE,
        environ={"SYSTEMROOT": "C:\\Windows", "API_KEY": "sk-secret"},
    )
    assert command == (BASE, "-I", "-m", "llm_redact.extract_worker")
    # The environment named as multiprocessing does (bpo-35797): a path, no
    # variable of the proxy's.
    assert env == {
        "SYSTEMROOT": "C:\\Windows",
        "PATH": "C:\\Windows\\System32",
        "__PYVENV_LAUNCHER__": VENV,
    }


def test_the_launcher_is_known_whatever_the_case_and_slashes() -> None:
    command, env = extraction.worker_launch(
        ("c:/work/.venv/scripts/PYTHON.EXE", "-I"),
        platform="win32",
        executable=VENV,
        base_executable=BASE,
        environ={},
    )
    assert command == (BASE, "-I") and env["__PYVENV_LAUNCHER__"] == VENV


@pytest.mark.parametrize(
    ("platform", "first", "base"),
    [
        ("win32", BASE, BASE),  # no virtual environment
        ("win32", VENV, ""),  # an interpreter that names no base (embedded)
        ("win32", "C:\\tools\\other.exe", BASE),  # another program
        ("linux", VENV, BASE),  # POSIX: a venv's python is the interpreter
    ],
)
def test_otherwise_the_command_runs_as_given(platform: str, first: str, base: str) -> None:
    command, env = extraction.worker_launch(
        (first, "-I"), platform=platform, executable=VENV, base_executable=base, environ={}
    )
    assert command == (first, "-I") and "__PYVENV_LAUNCHER__" not in env


def test_the_worker_opens_no_console_window() -> None:
    # A proxy started without a console (pythonw.exe, a logon task) made
    # Windows open one for each worker.
    assert extraction.worker_creationflags("win32") == 0x08000000  # CREATE_NO_WINDOW
    assert extraction.worker_creationflags("linux") == 0


async def test_a_cancelled_reap_still_closes_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    job = _Job()
    monkeypatch.setattr(extraction.winjob, "available", lambda: True)
    monkeypatch.setattr(extraction.winjob, "confine", lambda pid, **limits: job)

    async def stuck(proc: Any, data: bytes, cap: int) -> bytes:
        raise asyncio.CancelledError  # the request went away mid-read

    real_wait = asyncio.subprocess.Process.wait

    async def cancelled_wait(self: Any) -> int:
        await real_wait(self)
        raise asyncio.CancelledError  # and again while the worker is reaped

    monkeypatch.setattr(extraction, "_exchange", stuck)
    monkeypatch.setattr(asyncio.subprocess.Process, "wait", cancelled_wait)
    with pytest.raises(asyncio.CancelledError):
        await _inspector()._run_worker(b"123")
    assert job.closed


# Real Job Objects, Windows only.

on_windows = pytest.mark.skipif(sys.platform != "win32", reason="Job Objects")


def _child(script: str) -> subprocess.Popen[bytes]:
    # Started as the proxy starts the worker (``worker_launch``: the base
    # interpreter in a virtual environment). Waits for a line on stdin, so it
    # runs its script only once confined.
    command, env = extraction.worker_launch(
        (sys.executable, "-I", "-c", f"import sys; sys.stdin.readline()\n{script}")
    )
    return subprocess.Popen(
        command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def _go(child: subprocess.Popen[bytes]) -> tuple[int, bytes, bytes]:
    out, err = child.communicate(b"\n", timeout=60)
    return child.returncode, out, err


# Run as the worker: reads the file, then reports the job it is in (the
# immediate one: the runner's own job may hold the proxy).
JOB_PROBE = r"""
import ctypes, json, sys
from ctypes import wintypes
sys.stdin.buffer.read()
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.c_void_p]
kernel32.QueryInformationJobObject.argtypes = [
    wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p
]
class Basic(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]
class Extended(ctypes.Structure):
    _fields_ = [
        ("Basic", Basic), ("Io", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcess", ctypes.c_size_t),
        ("PeakJob", ctypes.c_size_t),
    ]
inside = wintypes.BOOL()
kernel32.IsProcessInJob(kernel32.GetCurrentProcess(), None, ctypes.byref(inside))
info = Extended()
read = kernel32.QueryInformationJobObject(None, 9, ctypes.byref(info), ctypes.sizeof(info), None)
facts = {
    "in_job": bool(inside.value), "read": bool(read), "flags": info.Basic.LimitFlags,
    "processes": info.Basic.ActiveProcessLimit, "memory": info.ProcessMemoryLimit,
    "executable": sys.executable, "prefix": sys.prefix,
}
print(json.dumps({"format": "text", "text": json.dumps(facts), "complete": True}))
"""


@on_windows
async def test_the_process_that_reads_the_file_is_the_one_in_the_job() -> None:
    inspector = ExtractionInspector(
        ExtractionConfig(enabled=True, worker_memory_mb=256),
        command=(sys.executable, "-I", "-c", JOB_PROBE),
    )
    facts = json.loads((await inspector._run_worker(b"%PDF-1.7"))["text"])
    assert facts["in_job"] and facts["read"]
    assert facts["flags"] & winjob.LIMIT_FLAGS == winjob.LIMIT_FLAGS
    assert facts["processes"] == 1 and facts["memory"] == 256 << 20
    # Started on the base interpreter, it still runs in this environment.
    assert facts["executable"] == sys.executable and facts["prefix"] == sys.prefix


@on_windows
def test_a_confined_worker_cannot_pass_its_memory_limit() -> None:
    child = _child("x = bytearray(512 << 20); print('allocated')")
    job = winjob.confine(child.pid, memory_bytes=128 << 20, cpu_seconds=30)
    try:
        code, out, err = _go(child)
    finally:
        job.close()
    assert code != 0 and b"allocated" not in out and b"MemoryError" in err


@on_windows
def test_a_confined_worker_cannot_start_a_process() -> None:
    child = _child(
        "import subprocess; subprocess.run([sys.executable, '-c', 'pass']); print('ran')"
    )
    job = winjob.confine(child.pid, memory_bytes=256 << 20, cpu_seconds=30)
    try:
        code, out, _ = _go(child)
    finally:
        job.close()
    assert code != 0 and b"ran" not in out


@on_windows
def test_closing_the_job_ends_the_worker() -> None:
    # Ended, not finished: the exit code a job's close leaves is not one to
    # rely on, what it printed is.
    child = _child("import time; time.sleep(120); print('slept')")
    job = winjob.confine(child.pid, memory_bytes=256 << 20, cpu_seconds=30)
    assert child.stdin is not None
    child.stdin.write(b"\n")
    child.stdin.flush()
    job.close()
    out, _ = child.communicate(timeout=30)
    assert b"slept" not in out


@on_windows
def test_a_worker_ends_with_a_proxy_that_is_killed(tmp_path: Path) -> None:
    # The "proxy" confines a sleeping worker, says its pid, then is killed:
    # its handle closes with it, and the worker ends. Both start on the base
    # interpreter (``worker_launch``): a virtual environment's launcher would
    # end what it started itself, proving nothing about the job.
    src = Path(extraction.__file__).resolve().parent.parent
    command, env = extraction.worker_launch(
        (
            sys.executable,
            "-c",
            "import subprocess, sys, time\n"
            f"sys.path.insert(0, {str(src)!r})\n"
            "from llm_redact import extraction, winjob\n"
            "command, env = extraction.worker_launch(\n"
            "    (sys.executable, '-I', '-c', 'import time; time.sleep(120)')\n"
            ")\n"
            "worker = subprocess.Popen(command, env=env)\n"
            "winjob.confine(worker.pid, memory_bytes=256 << 20, cpu_seconds=30)\n"
            "print(worker.pid, flush=True)\n"
            "time.sleep(120)\n",
        )
    )
    proxy = subprocess.Popen(command, env=env, stdout=subprocess.PIPE)
    assert proxy.stdout is not None
    worker = int(proxy.stdout.readline())
    proxy.kill()
    proxy.wait(timeout=30)
    tasklist = Path(os.environ.get("SYSTEMROOT", "C:\\Windows")) / "System32"
    deadline = time.monotonic() + 30
    while True:
        listed = subprocess.run(
            [str(tasklist / "tasklist.exe"), "/FI", f"PID eq {worker}", "/NH"],
            capture_output=True,
            timeout=60,
        ).stdout
        if str(worker).encode() not in listed:
            break
        assert time.monotonic() < deadline, "the worker outlived the proxy"
        time.sleep(0.2)
