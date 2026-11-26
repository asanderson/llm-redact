"""`llm-redact run -- <command...>`: launch a tool through the proxy.

Points every routed tool's base-URL environment variable at the proxy and
runs the command. If no proxy answers on the configured port, an ephemeral
`llm-redact serve` subprocess is started for the child's lifetime and torn
down afterwards (a proxy that was already running is never touched).

Signals: SIGINT reaches the child through the terminal's foreground
process group, so the wrapper ignores it and just waits (the classic
wrapper pattern — forwarding would double-signal); SIGTERM is forwarded.
The wrapper's exit code is the child's.

Windows: a bare program name is looked up on PATH only (``resolve_program``),
never in the current directory, which CreateProcess searches first, and
with PATHEXT's extensions, so an npm ``claude.cmd`` shim is found (an
``.exe`` beside it goes first). A batch file's arguments are read again by
cmd.exe, which is said when they hold a character it treats specially.
The ephemeral proxy gets a process group of its own, so the terminal's
Ctrl+C reaches the tool and not the proxy under it, and is stopped with
CTRL_BREAK_EVENT, which uvicorn handles as SIGTERM: its shutdown drains
before it exits (a kill follows only when it does not within the bound).

WSL: a Windows program started from WSL (an ``.exe``, or a file on a
Windows drive) sees only the variables ``WSLENV`` names, so each exported
one is added there as ``NAME/w`` (shared with Win32 programs only); a
Linux program's environment is unchanged.
"""

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from types import FrameType

from llm_redact import platform_info
from llm_redact.config import apply_env_overrides, dial_url, load_config
from llm_redact.init_cli import TOOL_EXPORTS, export_value

_READY_TIMEOUT_SECONDS = 15.0
# How long the ephemeral proxy may take to stop before it is killed.
_STOP_TIMEOUT_SECONDS = 5.0
# What CreateProcess runs itself (a .cmd or .bat through cmd.exe), in the
# order one folder's matches are taken; PATHEXT decides which count.
_RUNNABLE_EXTENSIONS = (".exe", ".com", ".cmd", ".bat")
_DEFAULT_PATHEXT = ".COM;.EXE;.BAT;.CMD"
_BATCH_EXTENSIONS = (".cmd", ".bat")
# What cmd.exe reads in a batch file's arguments before the program does.
_CMD_SPECIAL = frozenset('"%!^&|<>()\r\n')

# Names an EXISTING llm-redact proxy (this machine or a team server) for
# every client-side command: `run` routes tools at it instead of spawning
# an ephemeral serve, `status` queries it, and `plugin install --proxy-url`
# writes it. Never embed credentials in this URL; a plugin that attaches an
# identity does so through Registry.tool_base_url.
ENV_PROXY_URL = "LLM_REDACT_PROXY_URL"


def validate_proxy_url(url: str) -> str | None:
    """Error text when a proxy URL is unusable, else None. Plain http is
    loopback-only: prompts to a remote proxy must ride TLS."""
    from urllib.parse import urlsplit

    from llm_redact.config import _is_loopback_host

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "proxy URL must look like http://127.0.0.1:8787 or https://host:8787"
    if parts.scheme == "http" and not _is_loopback_host(parts.hostname):
        return (
            "plain-http proxy URLs are loopback-only — a remote proxy must be"
            " https (prompts would cross the network in cleartext)"
        )
    return None


def resolve_proxy_url(explicit: str | None) -> str | None:
    """The effective pointed-at proxy: an explicit --proxy-url wins over
    the LLM_REDACT_PROXY_URL env var; empty means none configured."""
    url = (explicit or os.environ.get(ENV_PROXY_URL, "")).strip().rstrip("/")
    return url or None


def _status_url(base_url: str) -> str:
    from llm_redact.proxy import RESERVED_PREFIX

    return f"{base_url}{RESERVED_PREFIX}/status"


def _proxy_running(base_url: str, timeout: float = 1.0) -> bool:
    import httpx

    try:
        return httpx.get(_status_url(base_url), timeout=timeout).status_code == 200
    except (httpx.HTTPError, httpx.InvalidURL):
        # InvalidURL is no HTTPError: nothing answers at a host no URL carries.
        return False


def resolve_program(name: str, path: str | None, pathext: str | None) -> str | None:
    """On Windows, the program a bare ``name`` runs: the first PATH entry
    holding it with one of PATHEXT's runnable extensions (``.exe`` and
    ``.com`` before ``.cmd`` and ``.bat`` in one folder; a name that has one
    of them is looked for as is). Only absolute entries count: a relative
    one ("." or empty) names the current directory, which CreateProcess
    would search before PATH, so a program planted there never runs. A name
    with a folder or drive in it is returned as given; None when nothing on
    PATH matches."""
    if any(mark in name for mark in ("/", "\\", ":")):
        return name
    allowed = {ext.strip().lower() for ext in (pathext or _DEFAULT_PATHEXT).split(";")}
    extensions = [ext for ext in _RUNNABLE_EXTENSIONS if ext in allowed]
    lowered = name.lower()
    candidates = (
        [name]
        if any(lowered.endswith(ext) for ext in extensions)
        else [name + ext for ext in extensions]
    )
    for entry in (path or "").split(os.pathsep):
        folder = entry.strip().strip('"')
        if not folder or not os.path.isabs(folder):
            continue
        for candidate in candidates:
            program = os.path.join(folder, candidate)
            if os.path.isfile(program):
                return program
    return None


def _batch_warning(program: str, arguments: list[str]) -> str | None:
    """What to say before running a batch file whose arguments hold a
    character cmd.exe reads itself: they may not reach the program as
    typed (a ``"`` can end the quoting, and an ``&`` after it start
    another command)."""
    if not program.lower().endswith(_BATCH_EXTENSIONS):
        return None
    if not any(_CMD_SPECIAL & set(argument) for argument in arguments):
        return None
    return (
        f"llm-redact run: {os.path.basename(program)} is a batch file, so cmd.exe reads its"
        ' arguments again: a " % ! ^ & | < > ( ) in them may not reach the program as typed'
    )


def wslenv_with(current: str | None, names: Iterable[str]) -> str:
    """``WSLENV`` with each of ``names`` added as ``NAME/w`` — shared with a
    Win32 program started from WSL, not with a Linux one: an entry already
    there is kept as it is, empty entries are dropped."""
    entries = [entry for entry in (current or "").split(":") if entry]
    present = {entry.split("/", 1)[0] for entry in entries}
    for name in names:
        if name not in present:
            entries.append(f"{name}/w")
            present.add(name)
    return ":".join(entries)


def _runs_windows_program(program: str) -> bool:
    """In WSL, whether ``program`` is a Windows one: an ``.exe``, or a file
    on a Windows drive (WSL starts both as Win32 programs)."""
    if program.lower().endswith(".exe"):
        return True
    mount = platform_info.mount_of(Path(program))
    return mount is not None and mount.fstype in platform_info.WINDOWS_DRIVE_TYPES


def _stop_ephemeral(proxy: subprocess.Popen[bytes]) -> None:
    """Stop the ephemeral proxy so its shutdown drains: CTRL_BREAK_EVENT to
    its own process group on Windows (uvicorn handles it as SIGTERM; a bare
    terminate there is TerminateProcess), SIGTERM elsewhere; a kill only
    when it has not exited within the bound."""
    if sys.platform == "win32":
        try:
            proxy.send_signal(signal.CTRL_BREAK_EVENT)
        except OSError:  # no console to deliver it through
            proxy.terminate()
    else:
        proxy.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proxy.wait(timeout=_STOP_TIMEOUT_SECONDS)
    if proxy.poll() is None:
        proxy.kill()
        proxy.wait()


def _start_ephemeral_proxy(args: argparse.Namespace, base_url: str) -> subprocess.Popen[bytes]:
    command = [sys.executable, "-m", "llm_redact", "serve"]
    if args.config is not None:
        command += ["--config", str(args.config)]
    if args.port is not None:
        command += ["--port", str(args.port)]
    # The proxy's own stderr logging stays visible: it interleaves with the
    # tool's output the same way a manually started proxy would.
    if sys.platform == "win32":
        # Its own group: the terminal's Ctrl+C reaches the tool, never the
        # proxy under it, and CTRL_BREAK_EVENT can stop it alone.
        proxy = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        proxy = subprocess.Popen(command)
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if proxy.poll() is not None:
            raise RuntimeError(f"ephemeral proxy exited with code {proxy.returncode} on startup")
        if _proxy_running(base_url, timeout=0.5):
            return proxy
        time.sleep(0.1)
    _stop_ephemeral(proxy)
    raise RuntimeError(f"proxy did not answer at {base_url} within {_READY_TIMEOUT_SECONDS}s")


def run_run(args: argparse.Namespace) -> int:
    if not args.tool_command:
        print("nothing to run: llm-redact run [--tools ...] -- <command...>")
        return 2
    command = list(args.tool_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("nothing to run after --")
        return 2

    tools = [t.strip() for t in (args.tools or ",".join(TOOL_EXPORTS)).split(",") if t.strip()]
    unknown = [t for t in tools if t not in TOOL_EXPORTS]
    if unknown:
        print(f"unknown tool(s) {unknown}; known: {', '.join(sorted(TOOL_EXPORTS))}")
        return 2
    # Escape hatch for tools not in TOOL_EXPORTS: each --set-env VAR gets the
    # proxy base URL, same as the known tools' variables. Names only — the
    # VALUE is always the proxy URL, so nothing secret rides the flag.
    extra_env = list(getattr(args, "set_env", None) or ())
    bad = [name for name in extra_env if not name.isidentifier()]
    if bad:
        print(f"--set-env expects environment variable NAMES, got {bad}")
        return 2

    ephemeral: subprocess.Popen[bytes] | None = None
    pointed_at = resolve_proxy_url(getattr(args, "proxy_url", None))
    if pointed_at is not None:
        # An explicitly named proxy is used as-is: never spawn an ephemeral
        # one next to it, and require it to actually answer (a tool pointed
        # at a dead proxy would hard-fail on its first request anyway).
        problem = validate_proxy_url(pointed_at)
        if problem is not None:
            print(f"llm-redact run: {problem}")
            return 2
        if not _proxy_running(pointed_at):
            print("llm-redact run: no proxy answering at the configured proxy URL")
            return 1
        base_url = pointed_at
        origin = "existing proxy (--proxy-url/LLM_REDACT_PROXY_URL)"
    else:
        config = apply_env_overrides(load_config(args.config))
        port = args.port if args.port is not None else config.port
        if config.tls.enabled:
            # A TLS/mTLS proxy needs client-side trust configuration the
            # wrapped tool would also need; the wrapper can't inject that
            # safely — point at it with an https LLM_REDACT_PROXY_URL instead.
            print("llm-redact run supports plain-http loopback proxies only ([tls] is set)")
            return 2
        # Loopback for a wildcard bind, an IPv6 literal in brackets: the URL
        # the wrapped tools can actually dial.
        base_url = dial_url(config.host, port)

        if _proxy_running(base_url):
            origin = "already running"
        else:
            try:
                ephemeral = _start_ephemeral_proxy(args, base_url)
            except RuntimeError as problem:
                print(f"llm-redact run: {problem}")
                return 1
            origin = f"started for this run (pid {ephemeral.pid})"

    # A plugin may decorate the exported base URL (llm-redact-pro attaches
    # a named-user identity); the core exports it unchanged.
    from llm_redact.registry import get_registry

    tool_base = get_registry().tool_base_url(base_url)

    env = dict(os.environ)
    for tool in tools:
        variable = TOOL_EXPORTS[tool][0]
        # After any plugin decoration (a /u/<key> identity prefix stays
        # first): OPENAI_BASE_URL gets its /v1.
        env[variable] = export_value(variable, tool_base)
    for name in extra_env:
        env[name] = tool_base
    routed = ",".join([*tools, *extra_env])
    # base_url (never tool_base) in the log line: whatever a plugin added
    # stays out of terminals — and only scheme://host:port of a pointed-at
    # URL, in case someone embedded a path (a credential would echo).
    from urllib.parse import urlsplit

    parts = urlsplit(base_url)
    shown = f"{parts.scheme}://{parts.netloc}"
    identified = " with a client identity" if tool_base != base_url else ""
    print(f"llm-redact: routing {routed} via {shown}{identified} ({origin})", file=sys.stderr)

    try:
        if sys.platform == "win32":
            # PATH only, with PATHEXT (an npm .cmd shim is found), never the
            # current directory CreateProcess would search first.
            program = resolve_program(command[0], env.get("PATH"), env.get("PATHEXT"))
            if program is None:
                print(f"llm-redact run: command not found: {command[0]}", file=sys.stderr)
                return 127
            command = [program, *command[1:]]
            warning = _batch_warning(program, command[1:])
            if warning is not None:
                print(warning, file=sys.stderr)
        elif platform_info.is_wsl():
            target = shutil.which(command[0], path=env.get("PATH"))
            if target is not None and _runs_windows_program(target):
                exported = [TOOL_EXPORTS[tool][0] for tool in tools]
                env["WSLENV"] = wslenv_with(env.get("WSLENV"), [*exported, *extra_env])
        try:
            child = subprocess.Popen(command, env=env)
        except FileNotFoundError:
            # The most likely first-run failure: the wrapped tool isn't on
            # PATH in this shell. 127 is the shell's own command-not-found
            # code. This spawn used to sit OUTSIDE the teardown scope, so
            # the failure printed a traceback AND leaked the ephemeral
            # proxy as an orphan that made every later diagnosis wrong.
            print(f"llm-redact run: command not found: {command[0]}", file=sys.stderr)
            return 127
        except PermissionError:
            print(f"llm-redact run: permission denied running: {command[0]}", file=sys.stderr)
            return 126
        except OSError as problem:
            # A file CreateProcess cannot run (not a program: WinError 193).
            reason = problem.strerror or type(problem).__name__
            print(f"llm-redact run: cannot run {command[0]}: {reason}", file=sys.stderr)
            return 126

        def _forward_terminate(_signum: int, _frame: FrameType | None) -> None:
            child.terminate()

        previous_term = signal.signal(signal.SIGTERM, _forward_terminate)
        previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            returncode = child.wait()
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
        return returncode
    finally:
        if ephemeral is not None:
            _stop_ephemeral(ephemeral)
