"""The test session's isolation from the developer's real machine.

conftest.py applies it before anything is collected: every default location
llm-redact and its CLIs read or write — the XDG config/data dirs, ``HOME``
and ``USERPROFILE`` (``Path.home()`` reads the latter on Windows: the service
unit, the agent plugins' command dirs, ``~`` in configured paths) and the
agent tools' own dir variables — points into
one throwaway directory for the whole session, subprocesses included (they
inherit the environment); a deployment's ``LLM_REDACT_*`` variables (its
config file, vault DSN and key, a proxy URL, a bind address) are dropped, so
no test can reach a real config, database or key command. The real
locations are snapshotted before the session and compared after it: any
entry created, changed or removed there fails the run
(``LLM_REDACT_TEST_ALLOW_REAL_DIR_CHANGES=1`` skips that check — a real
proxy running on the machine writes its own vault there).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path

from llm_redact.service_cli import LAUNCHD_LABEL

# Test-control variables, kept: the real-database DSNs and the Free-alone
# switch (LLM_REDACT_TEST_*), the live-API gates and model picks, and the
# prefixes the pro suite keeps (mirrored so one environment serves both).
KEPT_PREFIXES = (
    "LLM_REDACT_TEST_",
    "LLM_REDACT_PRO_",
    "LLM_REDACT_LICENSE_",
    "LLM_REDACT_LIVE",
    "LLM_REDACT_REALTIME_MODEL",
    "LLM_REDACT_BEDROCK_MODEL",
)
# Agent tools' own config dirs, which `llm-redact plugin` honors ahead of HOME.
AGENT_DIR_VARIABLES = ("CLAUDE_CONFIG_DIR", "CODEX_HOME")
ALLOW_REAL_DIR_CHANGES = "LLM_REDACT_TEST_ALLOW_REAL_DIR_CHANGES"


def deployment_variables(environ: Mapping[str, str]) -> list[str]:
    """The ``LLM_REDACT_*`` variables of ``environ`` a deployment sets (its
    config file, vault DSN/key/key command, proxy URL, bind, …): every one
    outside ``KEPT_PREFIXES``."""
    return sorted(
        name
        for name in environ
        if name.startswith("LLM_REDACT_") and not name.startswith(KEPT_PREFIXES)
    )


def watched_paths(
    environ: Mapping[str, str], home: Path, *, platform: str = sys.platform
) -> list[Path]:
    """Where llm-redact and its CLIs write by default, resolved against the
    REAL ``environ`` and ``home``: its config, data and bench-cache dirs,
    the agent plugins' command dirs (Claude Code, Codex, OpenCode, Cursor)
    and the user service unit."""
    config = Path(environ.get("XDG_CONFIG_HOME") or home / ".config")
    data = Path(environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    cache = Path(environ.get("XDG_CACHE_HOME") or home / ".cache")
    # `llm-redact plugin` prefers HOME to Path.home(); the two differ only on
    # Windows (where Path.home() reads USERPROFILE).
    tool_home = Path(environ.get("HOME") or home)
    claude = Path(environ.get("CLAUDE_CONFIG_DIR") or tool_home / ".claude")
    codex = Path(environ.get("CODEX_HOME") or tool_home / ".codex")
    opencode = Path(environ.get("XDG_CONFIG_HOME") or tool_home / ".config")
    unit = (
        home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
        if platform == "darwin"
        else home / ".config" / "systemd" / "user" / "llm-redact.service"
    )
    return [
        config / "llm-redact",
        data / "llm-redact",
        # The NER bench's dataset cache (bench/datasets/base.py): tests pass
        # their own cache_dir, so nothing may land here.
        cache / "llm-redact",
        claude / "commands",
        codex / "prompts",
        opencode / "opencode" / "commands",
        tool_home / ".cursor" / "commands",
        unit,
    ]


Snapshot = dict[str, "tuple[int, int, int] | None"]


def snapshot(paths: Iterable[Path]) -> Snapshot:
    """Every entry at or below ``paths``: (mode, size, mtime) by path, or
    None for a path that does not exist."""
    state: Snapshot = {}
    for root in paths:
        if not os.path.lexists(root):
            state[str(root)] = None
            continue
        entries = [root, *sorted(root.rglob("*"))] if root.is_dir() else [root]
        for entry in entries:
            info = entry.lstat()
            state[str(entry)] = (info.st_mode, info.st_size, info.st_mtime_ns)
    return state


def changes(before: Snapshot, after: Snapshot) -> list[str]:
    """The paths created, changed or removed between two snapshots."""
    return sorted(
        path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
    )


def isolate(environ: MutableMapping[str, str], root: Path) -> None:
    """Point ``environ``'s home and XDG dirs into ``root`` and drop the
    deployment variables (in place; ``root`` is created). The home is both
    ``HOME`` and ``USERPROFILE``: ``Path.home()`` reads the latter on Windows
    and the former everywhere else."""
    real_home = Path(environ.get("HOME") or Path.home())
    # Third-party caches (the hypothesis/spaCy/tldextract kind) stay where
    # they were: an empty cache would only make those libraries refetch. The
    # one llm-redact entry under XDG_CACHE_HOME, the NER bench's dataset
    # cache, is never written by the suite (tests pass a tmp_path cache_dir)
    # and is a watched path, so a test that did would fail the session.
    environ.setdefault("XDG_CACHE_HOME", str(real_home / ".cache"))
    for name, sub in (
        ("HOME", "home"),
        ("USERPROFILE", "home"),
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_STATE_HOME", "state"),
    ):
        path = root / sub
        path.mkdir(parents=True, exist_ok=True)
        environ[name] = str(path)
    for name in (*deployment_variables(environ), *AGENT_DIR_VARIABLES):
        environ.pop(name, None)
