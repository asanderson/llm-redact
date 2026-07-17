"""`llm-redact run -- <command...>`: launch a tool through the proxy.

Points every routed tool's base-URL environment variable at the proxy and
runs the command. If no proxy answers on the configured port, an ephemeral
`llm-redact serve` subprocess is started for the child's lifetime and torn
down afterwards (a proxy that was already running is never touched).

Signals: SIGINT reaches the child through the terminal's foreground
process group, so the wrapper ignores it and just waits (the classic
wrapper pattern — forwarding would double-signal); SIGTERM is forwarded.
The wrapper's exit code is the child's.
"""

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import time
from types import FrameType

from llm_redact.config import apply_env_overrides, load_config
from llm_redact.init_cli import TOOL_EXPORTS

_READY_TIMEOUT_SECONDS = 15.0

# Names an EXISTING llm-redact proxy (this machine or a team server) for
# every client-side command: `run` routes tools at it instead of spawning
# an ephemeral serve, `status` queries it, and `plugin install --proxy-url`
# writes it. Identity still rides LLM_REDACT_USER_KEY — never embed /u/<key>
# in this URL (log lines print the URL's host).
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
    except httpx.HTTPError:
        return False


def _start_ephemeral_proxy(args: argparse.Namespace, base_url: str) -> subprocess.Popen[bytes]:
    command = [sys.executable, "-m", "llm_redact", "serve"]
    if args.config is not None:
        command += ["--config", str(args.config)]
    if args.port is not None:
        command += ["--port", str(args.port)]
    # The proxy's own stderr logging stays visible: it interleaves with the
    # tool's output the same way a manually started proxy would.
    proxy = subprocess.Popen(command)
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if proxy.poll() is not None:
            raise RuntimeError(f"ephemeral proxy exited with code {proxy.returncode} on startup")
        if _proxy_running(base_url, timeout=0.5):
            return proxy
        time.sleep(0.1)
    proxy.terminate()
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
        base_url = f"http://{config.host}:{port}"

        if _proxy_running(base_url):
            origin = "already running"
        else:
            try:
                ephemeral = _start_ephemeral_proxy(args, base_url)
            except RuntimeError as problem:
                print(f"llm-redact run: {problem}")
                return 1
            origin = f"started for this run (pid {ephemeral.pid})"

    # Named-user identity (2.0 licensing): with LLM_REDACT_USER_KEY set, the
    # exported base URLs carry the /u/<key>/ prefix — the one knob every
    # tool has is its base URL, so identity rides it universally. The proxy
    # strips the prefix before routing and the key never appears in logs.
    tool_base = base_url
    user_key = os.environ.get("LLM_REDACT_USER_KEY", "").strip()
    if user_key:
        tool_base = f"{base_url}/u/{user_key}"

    env = dict(os.environ)
    for tool in tools:
        env[TOOL_EXPORTS[tool][0]] = tool_base
    for name in extra_env:
        env[name] = tool_base
    routed = ",".join([*tools, *extra_env])
    # base_url (never tool_base) in the log line: the key stays out of
    # terminals — and only scheme://host:port of a pointed-at URL, in case
    # someone embedded a path (an /u/<key> there would otherwise echo).
    from urllib.parse import urlsplit

    parts = urlsplit(base_url)
    shown = f"{parts.scheme}://{parts.netloc}"
    identified = " as a named user" if user_key else ""
    print(f"llm-redact: routing {routed} via {shown}{identified} ({origin})", file=sys.stderr)

    try:
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
            ephemeral.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                ephemeral.wait(timeout=5)
            if ephemeral.poll() is None:
                ephemeral.kill()
                ephemeral.wait()
