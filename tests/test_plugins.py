"""Agent-plugin assets: rendered-vs-checked-in sync and content invariants.

The checked-in Claude Code plugin (plugins/llm-redact/) and the repo-root
marketplace manifest are GENERATED from plugin_assets.py — these tests pin
the two in both directions (a content edit without re-rendering fails, and
a stale file on disk fails). The content tests keep the command set honest:
every CLI invocation a body asks an agent to run must be a real subcommand,
and `lookup` must never become a plugin command (it prints secret values,
which an agent would then send upstream).
"""

import json
import re
from pathlib import Path

from llm_redact import __version__
from llm_redact.completions import COMMANDS as CLI_COMMANDS
from llm_redact.plugin_assets import (
    COMMANDS,
    HOOKS_JSON,
    POSTURE_SCRIPT,
    claude_plugin_files,
    codex_files,
    cursor_files,
    marketplace_manifest,
    opencode_files,
    proxy_guard,
    render_claude,
)

REPO = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO / "plugins" / "llm-redact"


def test_checked_in_plugin_matches_rendered_both_directions() -> None:
    rendered = claude_plugin_files()
    for relpath, content in rendered.items():
        on_disk = (PLUGIN_DIR / relpath).read_text(encoding="utf-8")
        assert on_disk == content, f"{relpath} is stale: run scripts/render_plugins.py"
    on_disk_files = {
        # as_posix(): rendered keys use forward slashes on every platform.
        p.relative_to(PLUGIN_DIR).as_posix()
        for p in PLUGIN_DIR.rglob("*")
        if p.is_file()
    }
    assert on_disk_files == set(rendered), "stale files in plugins/llm-redact/"


def test_marketplace_manifest_matches_and_points_at_plugin() -> None:
    manifest_path = REPO / ".claude-plugin" / "marketplace.json"
    assert manifest_path.read_text(encoding="utf-8") == marketplace_manifest()
    assert '"source": "./plugins/llm-redact"' in marketplace_manifest()
    assert (PLUGIN_DIR / ".claude-plugin" / "plugin.json").is_file()


def test_plugin_version_tracks_package_version() -> None:
    assert f'"version": "{__version__}"' in (
        PLUGIN_DIR / ".claude-plugin" / "plugin.json"
    ).read_text(encoding="utf-8")


def test_every_rendering_has_wellformed_frontmatter() -> None:
    all_files = [
        *claude_plugin_files().items(),
        *codex_files().items(),
        *opencode_files().items(),
    ]
    for name, content in all_files:
        if not name.endswith(".md"):
            continue
        lines = content.splitlines()
        assert lines[0] == "---", name
        close = lines[1:].index("---") + 1
        for line in lines[1:close]:
            assert re.fullmatch(r"[a-z-]+: \S.*", line), (name, line)
        assert any(line.startswith("description: ") for line in lines[1:close]), name
        body = "\n".join(lines[close + 1 :]).strip()
        assert body, name


def test_command_names_and_argument_plumbing() -> None:
    names = [c.name for c in COMMANDS]
    assert len(names) == len(set(names))
    for command in COMMANDS:
        assert re.fullmatch(r"[a-z][a-z-]*", command.name)
        # argument-hint and $ARGUMENTS travel together: a hint without the
        # placeholder drops the user's input on the floor.
        assert (command.argument_hint is not None) == ("$ARGUMENTS" in command.body), command.name
    # config-edit has side effects: only the user may trigger it.
    config_edit = next(c for c in COMMANDS if c.name == "config-edit")
    assert config_edit.user_only
    assert "disable-model-invocation: true" in render_claude(config_edit)


def test_lookup_is_never_a_plugin_command() -> None:
    # `llm-redact lookup` prints secret VALUES; an agent that runs it would
    # carry them into its conversation and send them upstream — the exact
    # leak this proxy exists to prevent. Pinned, not accidental.
    assert "lookup" not in {c.name for c in COMMANDS}
    for command in COMMANDS:
        assert "llm-redact lookup" not in command.body, command.name


def test_bodies_reference_only_real_cli_subcommands() -> None:
    # Every `llm-redact <sub>` a body instructs an agent to run must be a
    # real subcommand (completions.COMMANDS is parser-synced by its own
    # test), so a CLI rename cannot silently strand the plugin prompts.
    for command in COMMANDS:
        # Backticked invocations only — prose like "llm-redact configuration
        # change" is not a command reference.
        for sub in re.findall(r"`llm-redact ([a-z][a-z-]*)", command.body):
            assert sub in CLI_COMMANDS, (command.name, sub)
        assert re.search(r"`llm-redact [a-z]", command.body), command.name
    # The guards' own invocations (init/service/run/serve) too — same rule.
    for tool in ("claude", "codex", "opencode", "cursor"):
        for sub in re.findall(r"`llm-redact ([a-z][a-z-]*)", proxy_guard(tool)):
            assert sub in CLI_COMMANDS, (tool, sub)


def test_non_claude_renderings_are_prefixed() -> None:
    # Codex scans only top-level prompt files; OpenCode and Cursor name
    # commands by filename — the llm-redact- prefix is the namespace.
    for name in (*codex_files(), *opencode_files(), *cursor_files()):
        assert name.startswith("llm-redact-") and name.endswith(".md"), name


def test_cursor_renderings_are_plain_markdown() -> None:
    # Cursor commands are the raw prompt: frontmatter would be injected
    # into the conversation verbatim, and $ARGUMENTS has no substitution
    # there — both must be absent.
    for name, content in cursor_files().items():
        assert content.startswith("# "), name
        assert not content.startswith("---"), name
        assert "$ARGUMENTS" not in content, name


def test_proxy_guard_present_in_every_rendering() -> None:
    # Marketplace installs can land on machines without the proxy CLI; a
    # command that guessed instead of stopping would read as "protected"
    # while nothing is. The guard asks BEFORE installing anything, offers
    # only PINNED bootstrap commands (agent-improvised install commands
    # are a supply-chain vector), and carries the per-tool routing-honesty
    # check — a live proxy is not the same as a routed session.
    for tool in ("claude", "codex", "opencode", "cursor"):
        guard = proxy_guard(tool)
        assert "ask whether to install" in guard, tool
        assert "uvx --from llm-redact-proxy" in guard, tool
        assert "Routing honesty" in guard, tool
    all_files = [
        *claude_plugin_files().items(),
        *codex_files().items(),
        *opencode_files().items(),
        *cursor_files().items(),
    ]
    for name, content in all_files:
        if not name.endswith(".md"):
            continue
        assert "proxy CLI is not installed" in content, name
        assert "after they approve" in content, name
        assert "uvx --from llm-redact-proxy" in content, name


def test_guard_routing_honesty_is_per_tool() -> None:
    # Each tool's guard must state ITS routing truth: env-var relaunch for
    # Claude Code, run-wrapper launch for the CLIs, and Cursor's
    # custom-API-key-mode limitation — implying CLI-style protection on
    # Cursor would be dishonest.
    assert "ANTHROPIC_BASE_URL" in proxy_guard("claude")
    assert "llm-redact run -- claude" in proxy_guard("claude")
    assert "OPENAI_BASE_URL" in proxy_guard("codex")
    assert "llm-redact run -- codex" in proxy_guard("codex")
    assert "OPENAI_BASE_URL" in proxy_guard("opencode")
    assert "llm-redact run -- opencode" in proxy_guard("opencode")
    assert "custom-API-key mode" in proxy_guard("cursor")
    assert "NOT protected" in proxy_guard("cursor")
    # Every tool the guards launch via `run --` must be a real
    # TOOL_EXPORTS entry, or the printed command errors out.
    from llm_redact.init_cli import TOOL_EXPORTS

    for tool in ("claude", "codex", "opencode"):
        assert tool in TOOL_EXPORTS, tool


def test_pinned_package_name_only() -> None:
    # The one permitted package name is `llm-redact-proxy`, verbatim: a
    # guard that let the agent improvise (`pip install llm-redact`, a
    # guessed name) would recreate the hallucinated-package supply-chain
    # vector the research documented. Pinned in both directions: the
    # correct literal appears, the bare wrong name never does.
    for tool in ("claude", "codex", "opencode", "cursor"):
        guard = proxy_guard(tool)
        assert "install llm-redact-proxy" in guard, tool
        assert not re.search(r"install llm-redact(?![-a-z])", guard), tool


def test_posture_script_and_hook_ship_in_plugin() -> None:
    # The SessionStart posture check exists to make the unavoidable
    # relaunch step LOUD (a running proxy with an unrouted session is the
    # silent-unprotected state). Detection only — the script must never
    # install; healthy sessions stay silent via --quiet-ok; messages name
    # this check as their source; URLs echo as scheme://host:port only.
    files = claude_plugin_files()
    assert files["bin/llm-redact-posture"] == POSTURE_SCRIPT
    assert files["hooks/hooks.json"] == HOOKS_JSON
    assert POSTURE_SCRIPT.startswith("#!/bin/sh")
    assert "llm-redact posture check" in POSTURE_SCRIPT
    assert "--quiet-ok" in POSTURE_SCRIPT
    assert "ANTHROPIC_BASE_URL" in POSTURE_SCRIPT
    for verb in ("uv tool install", "pipx install", "uvx --from"):
        assert verb not in POSTURE_SCRIPT, verb
    hook = json.loads(HOOKS_JSON)
    (entry,) = hook["hooks"]["SessionStart"]
    (command,) = entry["hooks"]
    assert command["type"] == "command"
    assert "${CLAUDE_PLUGIN_ROOT}/bin/llm-redact-posture" in command["command"]
    assert "--quiet-ok" in command["command"]
