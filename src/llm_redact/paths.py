"""Where llm-redact keeps its files, by kind: config, data, logs and model
folders. Every default location comes from here, so the stores, the CLI and
doctor agree on one convention per platform. A read-only SQLite URI for
one of those files (doctor, ``override list``, the vault commands) comes
from here too, so it names the same file on every platform.

- An XDG variable that is set wins on every platform (``XDG_CONFIG_HOME``,
  ``XDG_DATA_HOME``, ``XDG_STATE_HOME``). An empty or relative value counts
  as unset, as the XDG spec says: a relative one would name a folder under
  whatever the working directory is.
- Linux, macOS and WSL: ``~/.config/llm-redact``, ``~/.local/share/llm-redact``
  and ``~/.local/state/llm-redact``, as before.
- Windows: everything under ``%LOCALAPPDATA%\\llm-redact``, which stays on
  the machine (the config holds deny strings, so it does not roam): the
  config file at its top, the stores in ``data``, logs in ``logs``.
- Windows installs from before this used the POSIX locations under the
  profile (``~\\.config\\llm-redact``, ``~\\.local\\share\\llm-redact``).
  Those stay in use until the new location holds the same kind of file — a
  vault's tokens restore whatever the version that issued them — and
  ``doctor`` says how to move them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP = "llm-redact"
CONFIG_FILE = "config.toml"


def _windows() -> bool:
    return sys.platform == "win32"


def _xdg(name: str) -> Path | None:
    """The XDG variable's folder, or None when it is unset, empty or
    relative (the XDG spec: such a value is ignored)."""
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else None


def local_app_data() -> Path:
    """``%LOCALAPPDATA%``, else the profile's ``AppData\\Local`` (a service
    account may not set it; a value that is not an absolute path is not
    used either, since a drive-relative one names a folder on whatever
    drive the working directory is on)."""
    value = os.environ.get("LOCALAPPDATA")
    if value and Path(value).is_absolute():
        return Path(value)
    return Path.home() / "AppData" / "Local"


def _has_entries(folder: Path) -> bool:
    try:
        return any(folder.iterdir())
    except OSError:
        return False


def windows_root() -> Path:
    """``%LOCALAPPDATA%\\llm-redact``: the Windows home of every kind."""
    return local_app_data() / APP


def legacy_config_dir() -> Path:
    """Where Windows installs from before this kept the config file."""
    return Path.home() / ".config" / APP


def legacy_data_dir() -> Path:
    """Where Windows installs from before this kept the stores."""
    return Path.home() / ".local" / "share" / APP


def config_dir() -> Path:
    """The folder of the per-user config file (``config.toml``)."""
    xdg = _xdg("XDG_CONFIG_HOME")
    if xdg is not None:
        return xdg / APP
    if not _windows():
        return Path.home() / ".config" / APP
    new = windows_root()
    legacy = legacy_config_dir()
    if (legacy / CONFIG_FILE).exists() and not (new / CONFIG_FILE).exists():
        return legacy
    return new


def data_dir() -> Path:
    """The folder of the vault, audit and override stores (and, below it,
    the assembled model folders)."""
    xdg = _xdg("XDG_DATA_HOME")
    if xdg is not None:
        return xdg / APP
    if not _windows():
        return Path.home() / ".local" / "share" / APP
    new = windows_root() / "data"
    legacy = legacy_data_dir()
    if _has_entries(legacy) and not _has_entries(new):
        return legacy
    return new


def log_dir() -> Path:
    """The folder of the proxy's own log file, when it writes one."""
    xdg = _xdg("XDG_STATE_HOME")
    if xdg is not None:
        return xdg / APP
    if not _windows():
        return Path.home() / ".local" / "state" / APP
    return windows_root() / "logs"


def models_dir() -> Path:
    """Where llm-redact assembles model folders: ``models`` in the data
    folder."""
    return data_dir() / "models"


def uses_legacy_location() -> list[Path]:
    """The pre-``%LOCALAPPDATA%`` folders a Windows install still reads
    (empty elsewhere, and once they are moved)."""
    if not _windows():
        return []
    found = []
    if _xdg("XDG_CONFIG_HOME") is None and config_dir() == legacy_config_dir():
        found.append(legacy_config_dir())
    if _xdg("XDG_DATA_HOME") is None and data_dir() == legacy_data_dir():
        found.append(legacy_data_dir())
    return found


def sqlite_readonly_uri_for(posix_path: str) -> str:
    """A read-only SQLite ``file:`` URI for an absolute path written with
    forward slashes (``Path.as_posix()``).

    The path is percent-encoded: a ``?`` or ``#`` would otherwise end it
    (the rest read as the query or a fragment, which drops ``mode=ro`` and
    can open, or create, another file) and a ``%`` start an escape. ``/``
    and ``:`` stay literal (separators, a Windows drive letter). A Windows
    UNC path (``//server/share/...``, also what a mapped network drive
    resolves to) gets an empty authority, ``file:////server/share/...``:
    SQLite refuses ``file://server/...``, an authority other than
    localhost."""
    from urllib.parse import quote

    scheme = "file://" if posix_path.startswith("//") else "file:"
    return f"{scheme}{quote(posix_path, safe='/:')}?mode=ro"


def sqlite_readonly_uri(path: Path) -> str:
    """A read-only SQLite ``file:`` URI for ``path`` (resolved first): see
    ``sqlite_readonly_uri_for``."""
    return sqlite_readonly_uri_for(path.resolve().as_posix())
