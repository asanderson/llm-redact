"""Windows test fixtures: programs a test runs the way a user's tools run.

``cmd_shim`` writes what npm installs for a command-line tool: a ``.cmd``
batch file on PATH that starts the real program with ``%*``."""

from __future__ import annotations

import sys
from pathlib import Path

_CHILD = """import os, sys
print("SHIM=" + {label!r})
print("ANTH=" + os.environ.get("ANTHROPIC_BASE_URL", ""))
print("ARGS=" + repr(sys.argv[1:]))
sys.exit({exit_code})
"""


def cmd_shim(folder: Path, name: str, label: str, *, exit_code: int) -> Path:
    """``folder/name.cmd`` running a Python child that prints ``label``, the
    Anthropic base URL it was given and its arguments, then exits
    ``exit_code``. Returns ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    child = folder / f"{name}-child.py"
    child.write_text(_CHILD.format(label=label, exit_code=exit_code), encoding="utf-8")
    (folder / f"{name}.cmd").write_text(
        f'@"{sys.executable}" "%~dp0{child.name}" %*\r\n', encoding="utf-8", newline=""
    )
    return folder
