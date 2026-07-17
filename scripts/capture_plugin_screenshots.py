#!/usr/bin/env python3
"""Capture the plugin commands' terminal output as SVG screenshots.

Dev-only; the committed SVGs live in docs/screenshots/plugins/ next to the
dashboard PNGs (docs/screenshots/). The agent plugin commands are terminal
UIs — each one runs an `llm-redact` CLI command — so the honest screenshot
is the REAL output of those commands, rendered into a terminal-window SVG
(text stays selectable and the asset is reviewable in a diff, unlike a
bitmap). Regenerate with:

    uv run python scripts/capture_plugin_screenshots.py

The proxy is seeded with FIXTURE traffic against an in-process fake
upstream — never run this against a proxy that has handled real secrets.
Only secret-shaped fakes appear (vendors' canonical examples,
corp.example addresses); temp paths are rewritten to a neutral
~/.config/llm-redact before rendering.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import httpx
import uvicorn

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots" / "plugins"
EMAIL = "jane.doe@corp.example"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"

# ---------------------------------------------------------------- fixture rig


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _serve(make_app: Callable[[], object], port: int) -> uvicorn.Server:
    # Build the app INSIDE the serving thread: the sqlite vault connection
    # has thread affinity, and in production serve() builds and runs in one
    # thread — the rig must match.
    holder: dict[str, uvicorn.Server] = {}
    ready = threading.Event()

    def run() -> None:
        server = uvicorn.Server(
            uvicorn.Config(make_app(), host="127.0.0.1", port=port, log_level="warning")
        )
        holder["server"] = server
        ready.set()
        server.run()

    threading.Thread(target=run, daemon=True).start()
    ready.wait()
    while not holder["server"].started:
        time.sleep(0.05)
    return holder["server"]


def _fake_upstream_app() -> object:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def messages(request: Request) -> JSONResponse:
        body = await request.json()
        text = str(body.get("messages", ""))
        tokens = [w for w in text.split() if w.startswith("«")]
        echoed = " and ".join(tokens) or "nothing"
        return JSONResponse(
            {"role": "assistant", "content": [{"type": "text", "text": f"noted {echoed}"}]}
        )

    return Starlette(routes=[Route("/v1/messages", messages, methods=["POST"])])


# ------------------------------------------------------------- SVG rendering

_WIDTH = 820
_PAD = 22
_BAR = 40
_LINE_H = 19
_FONT = 13
_MAX_CHARS = 98
_BG = "#0d1117"
_BAR_BG = "#161b22"
_TEXT = "#c9d1d9"
_PROMPT = "#7ee787"
_TITLE = "#8b949e"


def _wrap(line: str) -> list[str]:
    if len(line) <= _MAX_CHARS:
        return [line]
    out = []
    while len(line) > _MAX_CHARS:
        out.append(line[:_MAX_CHARS])
        line = "  " + line[_MAX_CHARS:]
    out.append(line)
    return out


def render_terminal_svg(command: str, output: str) -> str:
    rows: list[tuple[str, str]] = [(_PROMPT, f"$ {command}")]
    rows.extend((_TEXT, raw.rstrip()) for raw in output.rstrip("\n").splitlines())
    return _render_svg(command, rows)


def render_transcript_svg(title: str, rows: list[tuple[str, str]]) -> str:
    """A Claude Code-style session: pre-colored rows instead of $-prompt+output."""
    return _render_svg(title, rows)


def _render_svg(title: str, colored_lines: list[tuple[str, str]]) -> str:
    rows: list[tuple[str, str]] = []  # (color, wrapped text)
    for color, raw in colored_lines:
        for piece in _wrap(raw):
            rows.append((color, piece))

    height = _BAR + _PAD + len(rows) * _LINE_H + _PAD
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{height}"'
        f' viewBox="0 0 {_WIDTH} {height}" role="img"'
        # quoteattr, not escape: the command may itself contain double
        # quotes (preview --text "..."), and an unescaped quote inside a
        # quoted attribute is malformed XML that renders as a broken image.
        f" aria-label={quoteattr(f'terminal output of {title}')}>",
        f'<rect width="{_WIDTH}" height="{height}" rx="10" fill="{_BG}"/>',
        f'<path d="M0 10a10 10 0 0 1 10-10h{_WIDTH - 20}a10 10 0 0 1 10 10v{_BAR - 10}h-{_WIDTH}z"'
        f' fill="{_BAR_BG}"/>',
        f'<circle cx="24" cy="{_BAR // 2}" r="6" fill="#ff5f57"/>',
        f'<circle cx="46" cy="{_BAR // 2}" r="6" fill="#febc2e"/>',
        f'<circle cx="68" cy="{_BAR // 2}" r="6" fill="#28c840"/>',
        f'<text x="{_WIDTH // 2}" y="{_BAR // 2 + 4}" text-anchor="middle" fill="{_TITLE}"'
        f' font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"'
        f' font-size="12">{escape(title)}</text>',
    ]
    font = (
        'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"'
        f' font-size="{_FONT}"'
    )
    y = _BAR + _PAD + _FONT
    for color, text in rows:
        parts.append(
            f'<text x="{_PAD}" y="{y}" fill="{color}" {font} xml:space="preserve">'
            f"{escape(text)}</text>"
        )
        y += _LINE_H
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ------------------------------------------------------------------ captures


def _run_cli(args: list[str], env: dict[str, str]) -> str:
    executable = shutil.which("llm-redact")
    cmd = [executable] if executable else [sys.executable, "-m", "llm_redact"]
    proc = subprocess.run([*cmd, *args], capture_output=True, text=True, env=env, timeout=60)
    return proc.stdout + proc.stderr


def _truncate(output: str, keep: int) -> str:
    """Keep the first `keep` lines with an explicit elision marker — the
    shot stays real output, just visibly cut (guide is a whole document)."""
    lines = output.rstrip("\n").splitlines()
    if len(lines) <= keep:
        return output
    return "\n".join([*lines[:keep], f"… (+{len(lines) - keep} more lines)"]) + "\n"


def main() -> int:
    upstream_port = _free_port()
    proxy_port = _free_port()

    with tempfile.TemporaryDirectory(prefix="llm-redact-shots-") as tmp:
        home = Path(tmp)
        config_path = home / "config.toml"
        vault_path = home / "vault.db"
        # An (unencrypted, Free-tier) sqlite vault so `sessions list` has a
        # real row to show; everything else is unchanged by the backend.
        config_path.write_text(
            f'port = {proxy_port}\n\n[vault]\nbackend = "sqlite"\n'
            f'path = "{vault_path}"\n\n[providers.anthropic]\n'
            f'upstream_base_url = "http://127.0.0.1:{upstream_port}"\n',
            encoding="utf-8",
        )

        from llm_redact.config import load_config
        from llm_redact.proxy import create_app

        upstream = _serve(_fake_upstream_app, upstream_port)
        proxy = _serve(lambda: create_app(load_config(config_path)), proxy_port)

        # Seed fixture traffic so status shows real-looking counters.
        with httpx.Client(base_url=f"http://127.0.0.1:{proxy_port}") as client:
            for text in (
                f"email {EMAIL} the launch checklist",
                f"rotate {AWS_KEY} before the audit",
                f"cc {EMAIL} and archive the thread",
            ):
                client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-sonnet-4-5",
                        "max_tokens": 128,
                        "messages": [{"role": "user", "content": text}],
                    },
                ).raise_for_status()

        # A hermetic environment: no user config/vault, no LLM_REDACT_* env
        # (a real key or proxy URL in this shell must not shape the docs).
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LLM_REDACT_", "XDG_"))}
        env.update(
            {
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_DATA_HOME": str(home / ".local" / "share"),
            }
        )

        # One shot per plugin command (the CLI each command body runs), plus
        # the installer. `recent` is the one HTTP-fetch command: capture the
        # live endpoint's real JSON, displayed as the body's curl pipeline.
        with httpx.Client(base_url=f"http://127.0.0.1:{proxy_port}") as client:
            recent_json = json.dumps(client.get("/__llm-redact/recent").json(), indent=4)

        shots: dict[str, list[str] | str] = {
            "status": ["status", "--config", str(config_path)],
            "recent": _truncate(recent_json, 30),
            "sessions": ["sessions", "list", "--config", str(config_path)],
            "config-show": ["config", "show", "--config", str(config_path)],
            "preview": [
                "preview",
                "--config",
                str(config_path),
                "--text",
                f"Email {EMAIL} and rotate {AWS_KEY} before the launch review.",
            ],
            "doctor": ["doctor", "--config", str(config_path)],
            # audit/users are Pro subsystems: keyless without llm-redact-pro,
            # the honest Free-tier output IS the refusal naming the package.
            "audit": ["audit", "verify"],
            "users": ["users", "list"],
            "guide": ["guide"],
            "install": ["plugin", "install", "claude"],
        }
        display = {
            "status": "llm-redact status",
            "recent": "curl -sS http://127.0.0.1:8787/__llm-redact/recent | python3 -m json.tool",
            "sessions": "llm-redact sessions list",
            "config-show": "llm-redact config show",
            "preview": f'llm-redact preview --text "Email {EMAIL} and rotate {AWS_KEY} ..."',
            "doctor": "llm-redact doctor",
            "audit": "llm-redact audit verify",
            "users": "llm-redact users list",
            "guide": "llm-redact guide",
            "install": "llm-redact plugin install claude",
        }
        truncate_lines = {"guide": 14}

        OUT.mkdir(parents=True, exist_ok=True)
        for name, args in shots.items():
            output = args if isinstance(args, str) else _run_cli(args, env)
            if name in truncate_lines:
                output = _truncate(output, truncate_lines[name])
            # Neutralize sandbox specifics: temp paths and throwaway ports
            # (the fake upstream shows as the real default it stands in for).
            output = output.replace(str(config_path), "~/.config/llm-redact/config.toml")
            output = output.replace(str(vault_path), "~/.local/share/llm-redact/vault.db")
            output = output.replace(str(home), "~")
            output = output.replace(f"127.0.0.1:{proxy_port}", "127.0.0.1:8787")
            output = output.replace(f"port = {proxy_port}", "port = 8787")
            output = output.replace(
                f"http://127.0.0.1:{upstream_port}", "https://api.anthropic.com"
            )
            (OUT / f"{name}.svg").write_text(
                render_terminal_svg(display[name], output), encoding="utf-8"
            )
            print(f"wrote {OUT / f'{name}.svg'}")

        # config-edit is not one CLI call but the guarded agent WORKFLOW that
        # mirrors the web config editor, so its shots are Claude Code-style
        # sessions, one per scenario. Every step is genuinely executed, in
        # order, against a fresh demo config each time: read the path, append
        # the edit, gate with serve --check, and verify — via preview (reads
        # the config from disk) or via status against a proxy built FROM the
        # edited config (which is what the SIGHUP reload produces in place).
        dim = _TITLE

        def config_edit_session(
            svg_name: str,
            ask: str,
            edit: str,
            readback: "Callable[[Path, str], list[tuple[str, str]]]",
            closing: list[str],
        ) -> None:
            demo_port = _free_port()
            demo_config = home / f"{svg_name}.toml"
            demo_config.write_text(
                f'port = {demo_port}\n\n[vault]\nbackend = "sqlite"\n'
                f'path = "{home / f"{svg_name}.db"}"\n\n[providers.anthropic]\n'
                f'upstream_base_url = "http://127.0.0.1:{upstream_port}"\n',
                encoding="utf-8",
            )

            def neutral(text: str) -> str:
                text = text.replace(str(demo_config), "~/.config/llm-redact/config.toml")
                text = text.replace(str(home), "~")
                text = text.replace(f"127.0.0.1:{demo_port}", "127.0.0.1:8787")
                return text.replace(
                    f"http://127.0.0.1:{upstream_port}", "https://api.anthropic.com"
                )

            path_out = _run_cli(
                ["config", "show", "--path", "--config", str(demo_config)], env
            ).strip()
            demo_config.write_text(demo_config.read_text(encoding="utf-8") + edit, encoding="utf-8")
            gate_out = _run_cli(["serve", "--check", "--config", str(demo_config)], env).strip()
            edit_lines = edit.strip("\n").splitlines()
            rows: list[tuple[str, str]] = [
                (_PROMPT, f"> /llm-redact:config-edit {ask}"),
                (_TEXT, ""),
                (_TEXT, "⏺ I'll read the effective config, apply the edit, gate it with"),
                (_TEXT, "  serve --check, reload via SIGHUP, and verify the result."),
                (_TEXT, ""),
                (_TEXT, "⏺ Bash(llm-redact config show --path)"),
                (dim, f"  ⎿  {neutral(path_out)}"),
                (_TEXT, ""),
                (_TEXT, "⏺ Update(~/.config/llm-redact/config.toml)"),
                (dim, f"  ⎿  Added {len(edit_lines)} lines:"),
                *[(dim, f"       {line}") for line in edit_lines],
                (_TEXT, ""),
                (_TEXT, "⏺ Bash(llm-redact serve --check)"),
                (dim, f"  ⎿  {neutral(gate_out)}"),
                (_TEXT, ""),
                (_TEXT, "⏺ Bash(kill -HUP $(pgrep -f 'llm-redact serve'))"),
                (dim, "  ⎿  (No content)"),
                (_TEXT, ""),
                *readback(demo_config, str(demo_port)),
                (_TEXT, ""),
                *[(_TEXT, line) for line in closing],
            ]
            rows = [(color, neutral(text)) for color, text in rows]
            (OUT / f"{svg_name}.svg").write_text(
                render_transcript_svg("Claude Code — /llm-redact:config-edit", rows),
                encoding="utf-8",
            )
            print(f"wrote {OUT / f'{svg_name}.svg'}")

        def bash_rows(display_cmd: str, output: str) -> list[tuple[str, str]]:
            lines = output.rstrip("\n").splitlines()
            return [
                (_TEXT, f"⏺ Bash({display_cmd})"),
                *[(dim, ("  ⎿  " if i == 0 else "     ") + line) for i, line in enumerate(lines)],
            ]

        # Scenario 1: deny string + warn-mode trial; read back via a live
        # proxy's status posture (a warn hit is seeded through the proxy).
        def warn_readback(cfg: Path, port: str) -> list[tuple[str, str]]:
            demo = _serve(lambda: create_app(load_config(cfg)), int(port))
            with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
                client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-sonnet-4-5",
                        "max_tokens": 128,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"call +1 415-555-0173 about project aurora and {EMAIL}",
                            }
                        ],
                    },
                ).raise_for_status()
            out = _truncate(_run_cli(["status", "--config", str(cfg)], env), 10)
            demo.should_exit = True
            return bash_rows("llm-redact status", out)

        config_edit_session(
            "config-edit",
            'add "project aurora" as a deny string and trial phone_number in warn mode',
            '\n[detection]\ndeny = ["project aurora"]\n\n'
            '[detection.modes]\nphone_number = "warn"\n',
            warn_readback,
            [
                "⏺ Both changes are live without a restart. Note the posture line:",
                "  warn mode FORWARDS matched phone numbers upstream while you",
                "  measure the rule's noise — switch it to redact once it's quiet.",
            ],
        )

        # Scenario 2: per-type allowlist; read back via preview (entirely
        # local — it reads the edited config straight from disk).
        def allowlist_readback(cfg: Path, port: str) -> list[tuple[str, str]]:
            text = f"mail support@corp.example and {EMAIL} about the rollout"
            out = _run_cli(["preview", "--config", str(cfg), "--text", text], env)
            return bash_rows(f'llm-redact preview --text "{text}"', out)

        config_edit_session(
            "config-edit-allowlist",
            "stop redacting support@corp.example — it's our public address",
            '\n[detection.allowlist_by_type]\nEMAIL = ["support@corp.example"]\n',
            allowlist_readback,
            [
                "⏺ Allowlisted for the EMAIL type only: the public address passes",
                "  through while every other email still redacts — the preview",
                "  proves it without sending anything upstream.",
            ],
        )

        # Scenario 3: block mode; read back via preview showing the 400.
        def block_readback(cfg: Path, port: str) -> list[tuple[str, str]]:
            text = "deploy notes: -----BEGIN PRIVATE KEY----- MIIEvQfake -----END PRIVATE KEY-----"
            out = _run_cli(["preview", "--config", str(cfg), "--text", text], env)
            display = 'llm-redact preview --text "deploy notes: -----BEGIN PRIVATE ..."'
            return bash_rows(display, out)

        config_edit_session(
            "config-edit-block",
            "private keys must never go upstream — block them outright",
            '\n[detection.modes]\nprivate_key = "block"\n',
            block_readback,
            [
                "⏺ Block mode is live: a request carrying a private key is now",
                "  rejected with a provider-shaped 400 before any upstream contact",
                "  — fail closed, and the tool surfaces the reason directly.",
            ],
        )

        # Scenario 4: language scoping; read back via a live proxy's status —
        # the posture LOUDLY lists every national-id rule scoped out.
        def language_readback(cfg: Path, port: str) -> list[tuple[str, str]]:
            demo = _serve(lambda: create_app(load_config(cfg)), int(port))
            out = _truncate(_run_cli(["status", "--config", str(cfg)], env), 10)
            demo.should_exit = True
            return bash_rows("llm-redact status", out)

        config_edit_session(
            "config-edit-languages",
            "we only handle English and German data — scope the national-id rules",
            '\n[detection]\nlanguages = ["en", "de"]\n',
            language_readback,
            [
                "⏺ Scoping applied. The posture line is deliberately loud: every",
                "  national-id rule outside en/de is NOT BUILT and its values",
                "  would pass through — an opt-out, never silent.",
            ],
        )

        proxy.should_exit = True
        upstream.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
