"""`llm-redact service`: run the proxy at login via launchd or systemd.

Opt-in convenience with no magic: install writes a user-level unit file,
loads it, and prints every action plus its undo. --print-only emits the
unit text without touching anything, for people who manage units by hand.
macOS uses a LaunchAgent (not a daemon — the proxy belongs to the user's
session); Linux uses a systemd user unit. Windows is unsupported, like the
rest of the project.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

LAUNCHD_LABEL = "com.llm-redact"


def _reject_control_chars(command: list[str]) -> None:
    """Refuse to emit a unit file for a command carrying a newline or other
    control character. The only variable part is the operator's own
    LLM_REDACT_CONFIG path; a newline in it would terminate the unit-file
    line and let extra directives be injected (systemd ExecStart) — so we
    fail loudly rather than write a malformed/ambiguous unit."""
    for part in command:
        if any(ord(ch) < 0x20 for ch in part):
            raise ValueError(
                "refusing to write a service unit: the serve command contains a "
                "control character (check LLM_REDACT_CONFIG)"
            )


def _command_line() -> list[str]:
    """The absolute serve command a unit file should run.

    Prefer the installed console script (uv tool / pipx / venv on PATH);
    fall back to the current interpreter's module entry so a bare
    `pip install --user` layout works too.
    """
    binary = shutil.which("llm-redact")
    command = [binary, "serve"] if binary else [sys.executable, "-m", "llm_redact", "serve"]
    config = os.environ.get("LLM_REDACT_CONFIG")
    if config:
        command += ["--config", config]
    return command


def _launchd_plist(command: list[str]) -> str:
    _reject_control_chars(command)
    # XML-escape each argv part: a config path containing &, <, or > would
    # otherwise break the plist or inject extra elements.
    array = "\n".join(f"        <string>{_xml_escape(part)}</string>" for part in command)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
{array}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
</dict>
</plist>
"""


def _systemd_unit(command: list[str]) -> str:
    # Hardening: a conservative sandbox that does not break the proxy's need to
    # write the vault/audit under XDG data. ReadWritePaths use the `-` prefix so
    # a not-yet-created data dir does not fail the unit at start. Heavy NER
    # extras (torch) may need MemoryDenyWriteExecute left OFF (it already is).
    _reject_control_chars(command)
    return f"""[Unit]
Description=llm-redact privacy proxy (127.0.0.1)
After=network.target

[Service]
ExecStart={shlex.join(command)}
Restart=on-failure
RestartSec=2

# --- sandboxing (systemd.exec) ---
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=-%h/.local/share/llm-redact -%h/.config/llm-redact -%h/.local/state
ProtectControlGroups=yes
ProtectKernelModules=yes
ProtectKernelTunables=yes
ProtectKernelLogs=yes
ProtectClock=yes
ProtectHostname=yes
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
RestrictNamespaces=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
LockPersonality=yes
CapabilityBoundingSet=
SystemCallFilter=@system-service
SystemCallErrorNumber=EPERM

[Install]
WantedBy=default.target
"""


def _unit_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    return Path.home() / ".config" / "systemd" / "user" / "llm-redact.service"


def _run(command: list[str]) -> int:
    """Thin wrapper so tests can observe the loader calls without a real
    launchd/systemd."""
    print(f"  $ {shlex.join(command)}")
    return subprocess.run(command, check=False).returncode


def run_service(args: argparse.Namespace) -> int:
    if sys.platform not in ("darwin", "linux"):
        if sys.platform == "win32":
            # Guidance, not automation: writing Task Scheduler state on the
            # operator's machine deserves an explicit, visible command they
            # run themselves (the same never-uninvited stance as plugin
            # install's proxy step). The printed command is ready to paste.
            print("service units cover macOS (launchd) and Linux (systemd).")
            print("On Windows, register a logon task yourself with Task Scheduler:")
            print()
            print(
                '  schtasks /Create /TN "llm-redact" /SC ONLOGON'
                f' /TR "{sys.executable} -m llm_redact serve"'
            )
            print()
            print('  schtasks /Delete /TN "llm-redact" /F   # to remove it again')
            print("or simply run `llm-redact serve` in a terminal (the Free-tier scope).")
            return 2
        print(f"unsupported platform {sys.platform!r}: service units cover macOS and Linux")
        return 2

    command = _command_line()
    unit_text = _launchd_plist(command) if sys.platform == "darwin" else _systemd_unit(command)
    path = _unit_path()

    if args.service_command == "install":
        if args.print_only:
            print(f"# would write {path}:\n")
            print(unit_text)
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(unit_text)
        print(f"wrote {path}")
        if sys.platform == "darwin":
            _run(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)])  # idempotent reinstall
            code = _run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)])
            print(f"undo with: llm-redact service uninstall  (or launchctl bootout + rm {path})")
        else:
            _run(["systemctl", "--user", "daemon-reload"])
            code = _run(["systemctl", "--user", "enable", "--now", "llm-redact.service"])
            print(
                "undo with: llm-redact service uninstall"
                f"  (or systemctl --user disable + rm {path})"
            )
        return 0 if code == 0 else 1

    if args.service_command == "uninstall":
        if sys.platform == "darwin":
            _run(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)])
        else:
            _run(["systemctl", "--user", "disable", "--now", "llm-redact.service"])
            _run(["systemctl", "--user", "daemon-reload"])
        if path.exists():
            path.unlink()
            print(f"removed {path}")
        else:
            print(f"nothing to remove at {path}")
        return 0

    # status
    print(f"unit file: {path} ({'present' if path.exists() else 'absent'})")
    if sys.platform == "darwin":
        _run(["launchctl", "print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"])
    else:
        _run(["systemctl", "--user", "--no-pager", "status", "llm-redact.service"])
    return 0
