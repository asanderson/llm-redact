"""The suite never touches the developer's real machine (tests/isolation.py).

conftest.py points HOME (USERPROFILE on Windows) and the XDG config/data/state
dirs into a throwaway directory for the whole session — unconditionally,
subprocesses included — drops a deployment's LLM_REDACT_* variables, and fails
the run when the real llm-redact dirs, the agent plugins' command dirs or the
user service unit changed while it ran.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

import isolation
from llm_redact import plugin_cli, service_cli
from llm_redact.audit import default_audit_path
from llm_redact.config import default_config_path
from llm_redact.vault import default_vault_path

TESTS = Path(__file__).resolve().parent


def _defaults() -> list[Path]:
    """Every default location the src resolves from HOME / XDG."""
    return [
        default_vault_path(),
        default_audit_path(),
        default_config_path(),
        service_cli._unit_path(),
        Path("~").expanduser(),
        *(plugin_cli._target_dir(tool, os.environ) for tool in plugin_cli.TOOLS),
    ]


def test_every_default_location_is_in_the_throwaway_home(isolation_root: Path) -> None:
    for name in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        assert Path(os.environ[name]).resolve().is_relative_to(isolation_root), name
    for path in _defaults():
        assert path.resolve().is_relative_to(isolation_root), path


def test_a_subprocess_inherits_the_isolation(isolation_root: Path) -> None:
    code = (
        "from llm_redact.vault import default_vault_path\n"
        "from llm_redact.config import default_config_path\n"
        "from pathlib import Path\n"
        "print(default_vault_path()); print(default_config_path()); print(Path.home())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    lines = result.stdout.split()
    assert len(lines) == 3
    assert all(Path(line).resolve().is_relative_to(isolation_root) for line in lines)


def test_no_deployment_variable_reaches_the_session() -> None:
    assert isolation.deployment_variables(os.environ) == []
    assert not set(isolation.AGENT_DIR_VARIABLES) & set(os.environ)


DROPPED = [
    "LLM_REDACT_CONFIG",
    "LLM_REDACT_VAULT_DSN",
    "LLM_REDACT_VAULT_KEY",
    "LLM_REDACT_VAULT_KEY_CMD",
    "LLM_REDACT_PROXY_URL",
    "LLM_REDACT_HOST",
    "LLM_REDACT_PORT",
    "LLM_REDACT_INSECURE_BIND",
    "LLM_REDACT_AUDIT_HMAC_KEY",
    "LLM_REDACT_USER_KEY",
]
KEPT = [
    "LLM_REDACT_TEST_PG_DSN",
    "LLM_REDACT_TEST_FREE_ALONE",
    "LLM_REDACT_TEST_ALLOW_REAL_DIR_CHANGES",
    "LLM_REDACT_PRO_ANYTHING",
    "LLM_REDACT_LICENSE_KEY",
    "LLM_REDACT_LIVE",
    "LLM_REDACT_LIVE_MODEL",
    "LLM_REDACT_REALTIME_MODEL",
    "LLM_REDACT_BEDROCK_MODEL",
]


def test_deployment_variables_are_dropped_and_test_controls_kept() -> None:
    environ = {name: "x" for name in (*DROPPED, *KEPT, "PATH", "REDACT_X")}
    assert isolation.deployment_variables(environ) == sorted(DROPPED)


def test_isolate_rewires_home_and_xdg_and_drops_deployment_settings(tmp_path: Path) -> None:
    environ = {
        "HOME": "/real/home",
        "XDG_CONFIG_HOME": "/real/config",
        "XDG_DATA_HOME": "/real/data",
        "CLAUDE_CONFIG_DIR": "/real/claude",
        "CODEX_HOME": "/real/codex",
        "LLM_REDACT_CONFIG": "/etc/llm-redact/config.toml",
        "LLM_REDACT_VAULT_KEY_CMD": "pass show llm-redact",
        "LLM_REDACT_TEST_PG_DSN": "postgresql://ci",
        "PATH": "/usr/bin",
    }
    isolation.isolate(environ, tmp_path)
    assert environ == {
        "HOME": str(tmp_path / "home"),
        "USERPROFILE": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        # Third-party caches stay where they were.
        "XDG_CACHE_HOME": str(Path("/real/home") / ".cache"),
        "LLM_REDACT_TEST_PG_DSN": "postgresql://ci",
        "PATH": "/usr/bin",
    }
    assert all((tmp_path / sub).is_dir() for sub in ("home", "config", "data", "state"))


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("tool_dirs", [False, True], ids=["home-defaults", "tool-dir-variables"])
def test_the_watched_paths_cover_every_default_writer(
    platform: str, tool_dirs: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Resolved against a stand-in for the real environment, every default
    # location the src writes lies at or under one watched path.
    monkeypatch.setattr(sys, "platform", platform)
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(tmp_path / "real-home"))
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "CLAUDE_CONFIG_DIR", "CODEX_HOME"):
        if tool_dirs:
            monkeypatch.setenv(name, str(tmp_path / name.lower()))
        else:
            monkeypatch.delenv(name, raising=False)
    watched = isolation.watched_paths(os.environ, Path.home(), platform=platform)
    for path in _defaults():
        if path == Path.home():
            continue  # `~` itself: what the configured paths expand under
        assert any(path.is_relative_to(root) for root in watched), path


def test_the_guard_sees_every_created_changed_or_removed_entry(tmp_path: Path) -> None:
    config = tmp_path / "config"
    unit = tmp_path / "unit.service"
    config.mkdir()
    for name in ("kept", "changed", "removed"):
        (config / name).write_text("a")
    before = isolation.snapshot([config, unit])
    assert isolation.changes(before, isolation.snapshot([config, unit])) == []
    (config / "changed").write_text("abc")
    (config / "removed").unlink()
    (config / "created").write_text("a")
    unit.write_text("[Unit]")
    changed = set(isolation.changes(before, isolation.snapshot([config, unit])))
    # The directory's own entry may change too (its mtime).
    assert changed - {str(config)} == {
        str(config / "changed"),
        str(config / "removed"),
        str(config / "created"),
        str(unit),
    }


# The guard end to end: a session (this conftest, in a subprocess) whose test
# writes into what that session sees as the real llm-redact data dir.
_WRITER = """
import os
from pathlib import Path

def test_writes_into_the_real_data_dir():
    target = Path(os.environ["GUARD_TARGET"])
    target.mkdir(parents=True, exist_ok=True)
    (target / "vault.db").write_text("real")
"""


@pytest.mark.parametrize("allow", [False, True], ids=["guarded", "opt-out"])
def test_a_session_that_writes_to_a_real_location_fails(tmp_path: Path, allow: bool) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for helper in ("conftest.py", "isolation.py"):
        shutil.copy(TESTS / helper, project / helper)
    (project / "test_writer.py").write_text(_WRITER)
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("XDG_") and name != isolation.ALLOW_REAL_DIR_CHANGES
    }
    env["HOME"] = env["USERPROFILE"] = str(real_home)
    env["GUARD_TARGET"] = str(real_home / ".local" / "share" / "llm-redact")
    if allow:
        env[isolation.ALLOW_REAL_DIR_CHANGES] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(project)],
        capture_output=True,
        text=True,
        cwd=project,
        env=env,
        timeout=300,
    )
    assert (real_home / ".local" / "share" / "llm-redact" / "vault.db").exists()
    if allow:
        assert result.returncode == 0, result.stdout
    else:
        assert result.returncode != 0
        assert "changed real locations outside its throwaway home" in result.stdout
        assert str(real_home / ".local" / "share" / "llm-redact") in result.stdout


@pytest.mark.parametrize(
    ("variable", "default", "tail"),
    [
        ("XDG_DATA_HOME", default_vault_path, (".local", "share", "llm-redact", "vault.db")),
        ("XDG_DATA_HOME", default_audit_path, (".local", "share", "llm-redact", "audit.db")),
        ("XDG_CONFIG_HOME", default_config_path, (".config", "llm-redact", "config.toml")),
    ],
)
def test_an_empty_xdg_variable_counts_as_unset(
    variable: str,
    default: Callable[[], Path],
    tail: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The XDG spec: an empty value is unset — never a path relative to the
    # current directory.
    monkeypatch.setenv(variable, "")
    assert default() == Path.home().joinpath(*tail)
    monkeypatch.setenv(variable, "/xdg")
    assert default() == Path("/xdg", *tail[-2:])
