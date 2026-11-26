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


# Real Job Objects, Windows only.

on_windows = pytest.mark.skipif(sys.platform != "win32", reason="Job Objects")


def _child(script: str) -> subprocess.Popen[bytes]:
    # Waits for a line on stdin, so it runs its script only once confined.
    return subprocess.Popen(
        [sys.executable, "-I", "-c", f"import sys; sys.stdin.readline()\n{script}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _go(child: subprocess.Popen[bytes]) -> tuple[int, bytes, bytes]:
    out, err = child.communicate(b"\n", timeout=60)
    return child.returncode, out, err


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
    child = _child("import time; time.sleep(120)")
    job = winjob.confine(child.pid, memory_bytes=256 << 20, cpu_seconds=30)
    assert child.stdin is not None
    child.stdin.write(b"\n")
    child.stdin.flush()
    job.close()
    assert child.wait(timeout=30) != 0


@on_windows
def test_a_worker_ends_with_a_proxy_that_is_killed(tmp_path: Path) -> None:
    # The "proxy" confines a sleeping worker, says its pid, then is killed:
    # its handle closes with it, and the worker ends.
    src = Path(extraction.__file__).resolve().parent.parent
    proxy = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time\n"
            f"sys.path.insert(0, {str(src)!r})\n"
            "from llm_redact import winjob\n"
            "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "winjob.confine(worker.pid, memory_bytes=256 << 20, cpu_seconds=30)\n"
            "print(worker.pid, flush=True)\n"
            "time.sleep(120)\n",
        ],
        stdout=subprocess.PIPE,
    )
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
