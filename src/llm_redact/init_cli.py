"""`llm-redact init`: a small setup wizard, stdlib only.

Asks (or takes flags for) the handful of decisions a new user actually
faces — which tools to route, whether the vault should persist, whether to
encrypt it, which port — then writes a normalized config through the same
emitter + atomic writer the config editor uses, and prints the per-tool
environment exports and next steps. It never overwrites an existing config
without --force (or an explicit interactive confirmation).
"""

import argparse
import sys

from llm_redact.config import (
    Config,
    VaultConfig,
    default_config_path,
    resolve_config_path,
)
from llm_redact.config_write import emit_config_toml, write_config_atomic

TOOL_EXPORTS: dict[str, tuple[str, str]] = {
    # tool name -> (env var, one-line description). Shared with the
    # `llm-redact run` wrapper (run_cli.py), which injects these for the
    # wrapped command.
    "claude": ("ANTHROPIC_BASE_URL", "Claude Code and other Anthropic SDK tools"),
    "codex": ("OPENAI_BASE_URL", "Codex CLI and other OpenAI-compatible tools"),
    "gemini": ("GOOGLE_GEMINI_BASE_URL", "Gemini CLI and google-genai SDK tools"),
    "ollama": ("OLLAMA_HOST", "the ollama CLI and other native-API clients"),
}


def _ask(prompt: str, default: str) -> str:
    reply = input(f"{prompt} [{default}]: ").strip()
    return reply or default


def _ask_yes_no(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    reply = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not reply:
        return default
    return reply in ("y", "yes")


def run_init(args: argparse.Namespace) -> int:
    interactive = not args.yes and sys.stdin.isatty()

    tools = [t.strip() for t in (args.tools or "").split(",") if t.strip()]
    unknown = [t for t in tools if t not in TOOL_EXPORTS]
    if unknown:
        print(f"unknown tool(s) {unknown}; known: {', '.join(sorted(TOOL_EXPORTS))}")
        return 2
    if not tools and interactive:
        while True:
            answer = _ask(
                f"Which tools will you route through the proxy? ({'/'.join(TOOL_EXPORTS)},"
                " comma-separated)",
                "claude",
            )
            picked = [t.strip() for t in answer.split(",") if t.strip()]
            bad = [t for t in picked if t not in TOOL_EXPORTS]
            if not bad:
                tools = picked
                break
            # Same message as the --tools flag path: never silently drop a
            # typo (answering "claud" used to fall back to claude by luck).
            print(f"unknown tool(s) {bad}; known: {', '.join(sorted(TOOL_EXPORTS))}")
    if not tools:
        tools = ["claude"]

    vault = args.vault
    if vault is None:
        # sqlite is the default since 3.3: it is Free (3.2 moved the
        # unencrypted file out of Pro), and the memory backend's cost is the
        # one first-time users discover the hard way — after any restart,
        # placeholders already sitting in a long-lived conversation can never
        # be restored (the tool shows literal «EMAIL_001» junk).
        vault = (
            "memory"
            if interactive
            and not _ask_yes_no(
                "Persist placeholder mappings across restarts? (recommended —"
                " the on-disk vault is created 0600 and holds the real values;"
                " answering no means a restart makes old conversations'"
                " placeholders unrestorable)",
                True,
            )
            else "sqlite"
        )
    encryption = args.encryption
    if encryption is None:
        encryption = (
            "fernet"
            if vault == "sqlite"
            and interactive
            and _ask_yes_no(
                "Encrypt the vault at rest? (needs `pip install"
                " 'llm-redact-proxy[crypto]'` and LLM_REDACT_VAULT_KEY)",
                False,
            )
            else "none"
        )
    port = args.port
    if port is None:
        port = 8787
        while interactive:
            answer = _ask("Proxy port", "8787")
            try:
                port = int(answer)
                break
            except ValueError:
                print(f"not a port number: {answer!r}")

    config = Config(
        port=port,
        vault=VaultConfig(backend=vault, encryption=encryption),
    )

    target = resolve_config_path() or default_config_path()
    if (
        target.exists()
        and not args.force
        and not (interactive and _ask_yes_no(f"{target} exists — overwrite it?", False))
    ):
        print(f"refusing to overwrite {target} (pass --force, or edit it directly)")
        return 1

    backup = write_config_atomic(target, emit_config_toml(config))
    print(f"\nwrote {target}" + (f" (previous file kept at {backup})" if backup else ""))

    base_url = f"http://127.0.0.1:{port}"
    print("\nPoint your tools at the proxy:\n")
    for tool in tools:
        env, description = TOOL_EXPORTS[tool]
        print(f"  export {env}={base_url}    # {description}")
    print("\nNext steps:")
    first_tool = tools[0]
    print(f"  llm-redact run -- {first_tool}            # one-liner: proxy + tool together")
    print("  # or: add the export line(s) above to your shell profile and use serve:")
    print("  llm-redact serve                 # run the proxy")
    print(f"  open {base_url}/__llm-redact/   # live dashboard + config editor")
    print('  llm-redact preview --text "mail me at a@corp.example"  # see redaction work')
    if encryption == "fernet":
        print("  llm-redact vault gen-key         # then export LLM_REDACT_VAULT_KEY")
    print("  llm-redact service install       # run at login (launchd/systemd)")
    return 0
