"""`llm-redact plugin`: install the agent slash-command plugins.

Copies the rendered command files from plugin_assets.py into each tool's
user command directory — the pip-install path. Claude Code users cloning
the repository can instead add it as a plugin marketplace
(`/plugin marketplace add asanderson/llm-redact`), which serves the
checked-in plugins/llm-redact/ directory with proper /llm-redact:name
namespacing; the copy install here uses llm-redact-<name>.md filenames so
the commands stay namespaced in every tool's flat command space.

install never silently overwrites a file whose content differs from what
it would write (the user may have customized it) — it refuses and asks
for --force. uninstall removes exactly the files install manages, never
neighbors.
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from llm_redact.plugin_assets import (
    claude_user_files,
    codex_files,
    cursor_files,
    opencode_files,
)

TOOLS = ("claude", "codex", "opencode", "cursor")


def _default_runner(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def _default_probe(url: str) -> bool:
    from llm_redact.run_cli import _proxy_running

    return _proxy_running(url)


def _default_base_url() -> str | None:
    from llm_redact.config import apply_env_overrides, load_config

    try:
        config = apply_env_overrides(load_config(None))
        return f"http://{config.host}:{config.port}"
    except Exception:  # noqa: BLE001 — posture helpers never fail the install
        return None


def _target_dir(tool: str, env: Mapping[str, str]) -> Path:
    home = Path(env.get("HOME") or Path.home())
    if tool == "claude":
        base = Path(env["CLAUDE_CONFIG_DIR"]) if env.get("CLAUDE_CONFIG_DIR") else home / ".claude"
        return base / "commands"
    if tool == "codex":
        base = Path(env["CODEX_HOME"]) if env.get("CODEX_HOME") else home / ".codex"
        return base / "prompts"
    if tool == "opencode":
        config = Path(env["XDG_CONFIG_HOME"]) if env.get("XDG_CONFIG_HOME") else home / ".config"
        return config / "opencode" / "commands"
    if tool == "cursor":
        return home / ".cursor" / "commands"
    raise ValueError(f"unknown tool {tool!r}")


def _files_for(tool: str) -> dict[str, str]:
    if tool == "claude":
        return claude_user_files()
    if tool == "codex":
        return codex_files()
    if tool == "opencode":
        return opencode_files()
    if tool == "cursor":
        return cursor_files()
    raise ValueError(f"unknown tool {tool!r}")


def _invocation_hint(tool: str) -> str:
    if tool == "claude":
        return (
            "commands available as /llm-redact-<name> (e.g. /llm-redact-status). "
            "Repo checkouts can instead use `/plugin marketplace add "
            "asanderson/llm-redact` for namespaced /llm-redact:<name> commands."
        )
    if tool == "codex":
        return "prompts available as /prompts:llm-redact-<name> (e.g. /prompts:llm-redact-status)."
    return "commands available as /llm-redact-<name> (e.g. /llm-redact-status)."


def proxy_posture_hint() -> str:
    """One line for install output: is a proxy answering right now?

    Best-effort and read-only — the commands work whenever the proxy runs,
    so a missing proxy is a hint to start one, never an install failure.
    """
    from llm_redact.run_cli import ENV_PROXY_URL, resolve_proxy_url

    try:
        pointed_at = resolve_proxy_url(None)
        if pointed_at is not None:
            # Name the env var, never echo its value — an operator who
            # (against instructions) embedded /u/<key> must not see it logged.
            if _default_probe(pointed_at):
                return f"proxy detected via {ENV_PROXY_URL} — commands are live."
            return f"no proxy answering at {ENV_PROXY_URL} yet — commands work once it does."
        base = _default_base_url()
        if base is None:
            return "could not determine proxy status (config unreadable)."
        if _default_probe(base):
            return f"proxy detected at {base} — commands are live."
        return (
            f"no proxy answering at {base} — start one with `llm-redact serve` "
            "or `llm-redact run -- <tool>` before using the commands."
        )
    except Exception:  # noqa: BLE001 — a hint must never fail the install
        return "could not determine proxy status (config unreadable)."


def _install_local_proxy(runner: Callable[[list[str]], int]) -> int:
    """`init --yes` (only when no config exists yet) + `service install` —
    the documented local setup, run ONLY on the operator's explicit ask
    (--install-proxy or the interactive prompt), never uninvited."""
    from llm_redact.config import ConfigError, resolve_config_path

    base = [sys.executable, "-m", "llm_redact"]
    try:
        config_path = resolve_config_path()
    except ConfigError:
        config_path = None
    if config_path is None or not config_path.exists():
        code = runner([*base, "init", "--yes"])
        if code != 0:
            print("init failed; not installing the service")
            return code
    code = runner([*base, "service", "install"])
    if code == 0:
        print("proxy installed as a login service (`llm-redact service status` to inspect)")
    return code


def proxy_setup(
    *,
    proxy_url: str | None,
    install_proxy: bool,
    probe: Callable[[str], bool],
    runner: Callable[[list[str]], int],
    ask: "Callable[[str], str] | None",
) -> int:
    """The post-install proxy step: point at an existing proxy, install a
    local one, or (interactively, only when nothing answers) offer both.
    The default remains skip-with-hint — scripts never block."""
    from llm_redact.run_cli import ENV_PROXY_URL, validate_proxy_url

    if proxy_url:
        url = proxy_url.strip().rstrip("/")
        problem = validate_proxy_url(url)
        if problem is not None:
            print(f"--proxy-url: {problem}")
            return 1
        if probe(url):
            print(f"existing proxy confirmed at {url}")
        else:
            print(f"WARNING: nothing answering at {url} yet — the commands work once it does")
        print("point the CLI and your tools at it (add to your shell profile):")
        print(f"  export {ENV_PROXY_URL}={url}")
        return 0
    if install_proxy:
        return _install_local_proxy(runner)
    if ask is not None:
        base = _default_base_url()
        if base is not None and probe(base):
            return 0  # a proxy already answers — nothing to offer
        choice = (
            ask(
                "no proxy detected — [i]nstall one as a login service, [p]oint at an"
                " existing proxy URL, or [s]kip? [i/p/S] "
            )
            .strip()
            .lower()
        )
        if choice == "i":
            return _install_local_proxy(runner)
        if choice == "p":
            url = ask("existing proxy URL (e.g. https://redact.corp.example:8787): ").strip()
            if url:
                return proxy_setup(
                    proxy_url=url, install_proxy=False, probe=probe, runner=runner, ask=None
                )
    return 0


def install(
    tool: str,
    env: Mapping[str, str],
    *,
    print_only: bool,
    force: bool,
    posture_hint: "Callable[[], str] | None" = None,
    proxy_url: str | None = None,
    install_proxy: bool = False,
    probe: "Callable[[str], bool] | None" = None,
    runner: "Callable[[list[str]], int] | None" = None,
    ask: "Callable[[str], str] | None" = None,
) -> int:
    target = _target_dir(tool, env)
    files = _files_for(tool)
    if print_only:
        for name in sorted(files):
            print(f"would write {target / name}")
        return 0
    blocked = [
        name
        for name, content in sorted(files.items())
        if (target / name).exists() and (target / name).read_text(encoding="utf-8") != content
    ]
    if blocked and not force:
        for name in blocked:
            print(f"REFUSING to overwrite modified file {target / name} (use --force)")
        return 1
    target.mkdir(parents=True, exist_ok=True)
    written = unchanged = 0
    for name, content in sorted(files.items()):
        path = target / name
        if path.exists() and path.read_text(encoding="utf-8") == content:
            unchanged += 1
            continue
        path.write_text(content, encoding="utf-8")
        written += 1
    print(f"{tool}: {written} file(s) written, {unchanged} already current in {target}")
    print(f"{tool}: {_invocation_hint(tool)}")
    print(f"{tool}: {(posture_hint or proxy_posture_hint)()}")
    return proxy_setup(
        proxy_url=proxy_url,
        install_proxy=install_proxy,
        probe=probe or _default_probe,
        runner=runner or _default_runner,
        ask=ask,
    )


def uninstall(tool: str, env: Mapping[str, str]) -> int:
    target = _target_dir(tool, env)
    removed = 0
    for name in sorted(_files_for(tool)):
        path = target / name
        if path.exists():
            path.unlink()
            removed += 1
    print(f"{tool}: {removed} file(s) removed from {target}")
    return 0


def status(env: Mapping[str, str]) -> int:
    for tool in TOOLS:
        target = _target_dir(tool, env)
        files = _files_for(tool)
        current = stale = 0
        for name, content in files.items():
            path = target / name
            if not path.exists():
                continue
            if path.read_text(encoding="utf-8") == content:
                current += 1
            else:
                stale += 1
        missing = len(files) - current - stale
        state = "not installed" if current + stale == 0 else f"{current} current, {stale} stale"
        if 0 < current + stale < len(files):
            state += f", {missing} missing"
        print(f"{tool:9s} {target}  [{state}]")
    return 0


def run_plugin(args: argparse.Namespace) -> int:
    env = os.environ
    if args.plugin_command == "install":
        interactive = (
            sys.stdin.isatty()
            and not getattr(args, "print_only", False)
            and getattr(args, "proxy_url", None) is None
            and not getattr(args, "install_proxy", False)
        )
        return install(
            args.tool,
            env,
            print_only=getattr(args, "print_only", False),
            force=getattr(args, "force", False),
            proxy_url=getattr(args, "proxy_url", None),
            install_proxy=getattr(args, "install_proxy", False),
            ask=input if interactive else None,
        )
    if args.plugin_command == "uninstall":
        return uninstall(args.tool, env)
    return status(env)
