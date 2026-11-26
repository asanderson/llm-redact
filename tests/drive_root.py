"""A drive letter rooted at a test directory, for the Windows tests that
plant a file where a rooted path without a drive ("/etc/...", "/dev/tty")
lands.

On Windows such a path names that location at the root of the working
directory's drive: ``\\etc\\llm-redact\\config.toml`` on ``C:``, on a USB
stick, on a network share. ``drive_rooted_at`` maps a free letter onto a
temporary directory with ``subst`` and moves the working directory to its
root, so the file there is one the test planted, never the runner's real
``C:\\etc``.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _subst() -> str:
    # System32's own subst, never one found on PATH.
    system_root = os.environ.get("SYSTEMROOT") or r"C:\Windows"
    return os.path.join(system_root, "System32", "subst.exe")


@contextmanager
def drive_rooted_at(target: Path) -> Iterator[Path]:
    """Map a free drive letter onto ``target``, make its root the working
    directory and yield that root (``X:\\``); both are undone on exit."""
    for letter in "ZYXWVUTSRQPONMLKJIHGFED":
        drive = f"{letter}:"
        if os.path.exists(drive + "\\"):
            continue
        if subprocess.run([_subst(), drive, str(target)], capture_output=True).returncode == 0:
            break
    else:
        raise RuntimeError("no free drive letter to map onto the test directory")
    previous = os.getcwd()
    try:
        os.chdir(drive + "\\")
        yield Path(drive + "\\")
    finally:
        os.chdir(previous)
        subprocess.run([_subst(), drive, "/D"], capture_output=True)
