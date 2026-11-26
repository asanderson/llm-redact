"""Which platform the proxy runs on, and where a path really is.

- ``platform_name()``: ``windows``, ``wsl``, ``linux`` or ``macos`` (else
  ``sys.platform``); the ``/status`` ``platform`` field, so a client can tell
  which proxy answered where a native and a WSL proxy both run.
- WSL is detected from ``WSL_DISTRO_NAME`` / ``WSL_INTEROP`` (set in every
  WSL process) and the kernel release (``/proc/sys/kernel/osrelease`` names
  Microsoft); its networking mode is ``wslinfo --networking-mode`` where WSL
  ships it (unknown otherwise).
- ``mount_of(path)``: the file system a path lives on, from
  ``/proc/self/mountinfo``. In WSL the Windows drives (``/mnt/c``) are 9p
  mounts (WSL 2) or drvfs (WSL 1), where SQLite's file locks and shared
  memory do not hold: ``windows_drive_problem`` names a path there.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

OSRELEASE = Path("/proc/sys/kernel/osrelease")
MOUNTINFO = Path("/proc/self/mountinfo")
WSLINFO = "/usr/bin/wslinfo"
# The Windows drives WSL mounts: 9p in WSL 2, drvfs in WSL 1.
WINDOWS_DRIVE_TYPES = frozenset({"9p", "drvfs"})
NETWORKING_MODES = frozenset({"nat", "mirrored", "virtioproxy", "bridged", "none"})
_ESCAPE = re.compile(r"\\([0-7]{3})")


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def is_wsl(environ: Mapping[str, str] | None = None, osrelease: str | None = None) -> bool:
    """Whether this Linux runs under WSL (1 or 2)."""
    if sys.platform != "linux":
        return False
    env = os.environ if environ is None else environ
    if env.get("WSL_DISTRO_NAME") or env.get("WSL_INTEROP"):
        return True
    release = _read(OSRELEASE) if osrelease is None else osrelease
    return release is not None and "microsoft" in release.lower()


def platform_name() -> str:
    """``windows``, ``wsl``, ``linux``, ``macos``, else ``sys.platform``."""
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "linux":
        return "wsl" if is_wsl() else "linux"
    return sys.platform


def wsl_distribution(environ: Mapping[str, str] | None = None) -> str | None:
    """The WSL distribution's name, when WSL says it."""
    env = os.environ if environ is None else environ
    return env.get("WSL_DISTRO_NAME") or None


def networking_mode(wslinfo: str | None = None) -> str | None:
    """WSL's networking mode (``nat``, ``mirrored`` …) as ``wslinfo
    --networking-mode`` prints it; None when it is unknown (no wslinfo,
    an older WSL, an answer this does not know). Linux only."""
    if sys.platform != "linux":
        return None
    program = wslinfo
    if program is None:
        program = WSLINFO if os.path.exists(WSLINFO) else shutil.which("wslinfo")
    if program is None:
        return None
    try:
        done = subprocess.run(
            [program, "--networking-mode"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    mode = done.stdout.decode("utf-8", errors="replace").strip().lower()
    return mode if mode in NETWORKING_MODES else None


@dataclass(frozen=True)
class Mount:
    """One mount: where, its file-system type and its source."""

    point: str
    fstype: str
    source: str


def _unescape(field: str) -> str:
    """A mountinfo field: a blank, tab, newline or backslash is \\ooo."""
    return _ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), field)


def parse_mountinfo(text: str) -> list[Mount]:
    """The mounts of a ``/proc/<pid>/mountinfo`` text, in its order (a line
    it cannot read is skipped)."""
    mounts = []
    for line in text.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split(" ")
        described = right.split(" ")
        if len(fields) < 5 or len(described) < 2:
            continue
        mounts.append(Mount(_unescape(fields[4]), described[0], _unescape(described[1])))
    return mounts


def _within(target: str, point: str) -> bool:
    if point == "/":
        return target.startswith("/")
    return target == point or target.startswith(point.rstrip("/") + "/")


def mount_of(path: Path, mountinfo: str | None = None) -> Mount | None:
    """The mount ``path`` is on (links followed as far as they exist): the
    deepest mount point holding it, the later of two at one point. None
    without a mount table (the system's is read on Linux only)."""
    if mountinfo is None:
        if sys.platform != "linux":
            return None
        mountinfo = _read(MOUNTINFO)
    text = mountinfo
    if text is None:
        return None
    target = os.path.realpath(path)
    found: Mount | None = None
    for mount in parse_mountinfo(text):
        if _within(target, mount.point) and (found is None or len(mount.point) >= len(found.point)):
            found = mount
    return found


def windows_drive_problem(path: Path) -> str | None:
    """In WSL, what to say about ``path`` on a Windows drive (a 9p or drvfs
    mount); None elsewhere, or for a path in the Linux file system."""
    if not is_wsl():
        return None
    mount = mount_of(path)
    if mount is None or mount.fstype not in WINDOWS_DRIVE_TYPES:
        return None
    return f"{path} is on a Windows drive mounted into WSL ({mount.point}, {mount.fstype})"
