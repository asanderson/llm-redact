"""Agent-plugin assets: rendered-vs-checked-in sync and content invariants.

The checked-in Claude Code plugin (plugins/llm-redact/) and the repo-root
marketplace manifest are GENERATED from plugin_assets.py — these tests pin
the two in both directions (a content edit without re-rendering fails, and
a stale file on disk fails). The content tests keep the command set honest:
every CLI invocation a body asks an agent to run must be a real subcommand,
and `lookup` must never become a plugin command (it prints secret values,
which an agent would then send upstream).
"""

import re
from pathlib import Path

from llm_redact import __version__
from llm_redact.completions import COMMANDS as CLI_COMMANDS
from llm_redact.plugin_assets import (
    COMMANDS,
    PROXY_GUARD,
    claude_plugin_files,
    codex_files,
    cursor_files,
    marketplace_manifest,
    opencode_files,
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
        if name.endswith(".json"):
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
    # while nothing is. The guard asks BEFORE installing anything.
    assert "ask whether to install" in PROXY_GUARD
    all_files = [
        *claude_plugin_files().items(),
        *codex_files().items(),
        *opencode_files().items(),
        *cursor_files().items(),
    ]
    for name, content in all_files:
        if name.endswith(".json"):
            continue
        assert "proxy CLI is not installed" in content, name
        assert "after they approve" in content, name
