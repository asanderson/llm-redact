"""Environment lines a user can paste, in the syntax of the shell they use:
PowerShell on Windows (what Windows Terminal opens), POSIX sh elsewhere.

On Windows each variable gets two lines: one for the PowerShell session
(``$env:NAME = '…'``) and one that keeps it in the user's environment for
later sessions and programs (``[Environment]::SetEnvironmentVariable``),
which is what a shell profile's ``export`` does elsewhere. Values are
single-quoted, so PowerShell expands nothing in them.
"""

from __future__ import annotations

import sys

# PowerShell reads these as single quotes, the curly ones included; inside a
# single-quoted string each is doubled to stand for itself.
_POWERSHELL_QUOTES = ("'", "‘", "’", "‚", "‛")


def powershell_quote(value: str) -> str:
    """``value`` as a PowerShell single-quoted string literal."""
    for quote in _POWERSHELL_QUOTES:
        value = value.replace(quote, quote * 2)
    return f"'{value}'"


def env_lines(name: str, value: str) -> list[str]:
    """The lines that set ``name`` to ``value``: ``export NAME=value`` (as
    printed since the first release), or on Windows the PowerShell session
    and user-environment lines."""
    if sys.platform != "win32":
        return [f"export {name}={value}"]
    quoted = powershell_quote(value)
    return [
        f"$env:{name} = {quoted}",
        f"[Environment]::SetEnvironmentVariable({powershell_quote(name)}, {quoted}, 'User')",
    ]


def keep_hint() -> str:
    """How the printed lines outlive the shell they are pasted into."""
    if sys.platform != "win32":
        return "add the export line(s) above to your shell profile"
    return "the [Environment] line(s) above keep them for new terminals and programs"


def set_word() -> str:
    """The verb for setting a variable in this platform's shell."""
    return "set" if sys.platform == "win32" else "export"
