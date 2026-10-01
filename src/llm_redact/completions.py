"""`llm-redact completions bash|zsh`: shell completion scripts.

Hand-written templates (no runtime deps, nothing to install at import
time); the COMMANDS table below is the single source of truth for the core
commands and a test cross-checks it against the real argparse parser so the
two cannot drift apart silently. Plugin-supplied commands carry their own
completion words (``CliCommand.completion``), merged in by ``all_commands``.
"""

# command -> (subcommands, options) — completion surface only.
COMMANDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "serve": ((), ("--config", "--port", "--session", "--log-format", "--check")),
    "status": ((), ("--config", "--port", "--json", "--ca", "--cert", "--key")),
    "sessions": (("list", "prune"), ("--config", "--db", "--json", "--older-than", "--yes")),
    "lookup": ((), ("--config", "--db", "--value", "--session")),
    "override": ((), ("--once", "--always", "--json", "--config", "--db")),
    "vault": (
        ("gen-key", "set-key", "verify", "rotate-key", "backup"),
        ("--config", "--db", "--yes", "--force"),
    ),
    "audit": (("verify", "decrypt"), ("--config", "--json", "--out")),
    "init": ((), ("--tools", "--vault", "--encryption", "--port", "--yes", "--force")),
    "service": (("install", "uninstall", "status"), ("--print-only",)),
    "plugin": (
        ("install", "uninstall", "status"),
        ("--print-only", "--force", "--proxy-url", "--install-proxy"),
    ),
    "run": ((), ("--config", "--port", "--tools", "--set-env", "--proxy-url")),
    "doctor": ((), ("--config", "--json", "--offline")),
    "guide": ((), ()),
    "config": (("show",), ("--config", "--path")),
    "preview": ((), ("--config", "--text", "--json")),
    "license": (("show", "verify"), ("--config", "--key", "--json")),
    "routes": (
        ("list", "test"),
        ("--config", "--json", "--protocol", "--model", "--header", "--path", "--auth"),
    ),
    "spend": ((), ("--config", "--month", "--json")),
    "completions": (("bash", "zsh", "fish"), ()),
    "fips-check": ((), ()),
}


def all_commands() -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """COMMANDS plus the completion words of plugin-supplied commands
    (llm-redact-pro's), which never shadow a core command."""
    from llm_redact.registry import get_registry

    merged = dict(COMMANDS)
    for command in get_registry().cli_commands:
        merged.setdefault(command.name, command.completion)
    return merged


def bash_script() -> str:
    commands = all_commands()
    top = " ".join(commands)
    cases = []
    for name, (subs, opts) in commands.items():
        words = " ".join((*subs, *opts))
        cases.append(f'        {name}) words="{words}" ;;')
    case_block = "\n".join(cases)
    return f"""# bash completion for llm-redact. Install:
#   llm-redact completions bash > ~/.local/share/bash-completion/completions/llm-redact
_llm_redact() {{
    local cur cmd words
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    cmd="${{COMP_WORDS[1]}}"
    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=($(compgen -W "{top} --version" -- "$cur"))
        return
    fi
    case "$cmd" in
{case_block}
        *) words="" ;;
    esac
    COMPREPLY=($(compgen -W "$words" -- "$cur"))
}}
complete -F _llm_redact llm-redact
"""


def zsh_script() -> str:
    commands = all_commands()
    top = " ".join(commands)
    cases = []
    for name, (subs, opts) in commands.items():
        words = " ".join((*subs, *opts))
        cases.append(f"        {name}) compadd -- {words} ;;")
    case_block = "\n".join(cases)
    return f"""#compdef llm-redact
# zsh completion for llm-redact. Install into any $fpath dir, e.g.:
#   llm-redact completions zsh > ~/.zfunc/_llm-redact   (with fpath+=~/.zfunc)
_llm_redact() {{
    if (( CURRENT == 2 )); then
        compadd -- {top} --version
        return
    fi
    case "$words[2]" in
{case_block}
        *) ;;
    esac
}}
_llm_redact "$@"
"""


def fish_script() -> str:
    lines = [
        "# fish completion for llm-redact. Install:",
        "#   llm-redact completions fish > ~/.config/fish/completions/llm-redact.fish",
        "complete -c llm-redact -f",
        "complete -c llm-redact -n __fish_use_subcommand -l version",
    ]
    commands = all_commands()
    for name in commands:
        lines.append(f"complete -c llm-redact -n __fish_use_subcommand -a {name}")
    for name, (subs, opts) in commands.items():
        condition = f'"__fish_seen_subcommand_from {name}"'
        for sub in subs:
            lines.append(f"complete -c llm-redact -n {condition} -a {sub}")
        for opt in opts:
            lines.append(f"complete -c llm-redact -n {condition} -l {opt.removeprefix('--')}")
    return "\n".join(lines) + "\n"


def script_for(shell: str) -> str:
    if shell == "bash":
        return bash_script()
    if shell == "zsh":
        return zsh_script()
    if shell == "fish":
        return fish_script()
    raise ValueError(f"unsupported shell {shell!r}")
