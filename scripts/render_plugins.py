"""Re-render the checked-in agent-plugin files from plugin_assets.py.

Run after editing src/llm_redact/plugin_assets.py:

    uv run python scripts/render_plugins.py

Writes plugins/llm-redact/ (the Claude Code plugin the marketplace serves)
and .claude-plugin/marketplace.json at the repo root. tests/test_plugins.py
pins both directions: every rendered file matches disk, and no stale file
lingers on disk.
"""

from pathlib import Path

from llm_redact.plugin_assets import claude_plugin_files, marketplace_manifest

REPO = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO / "plugins" / "llm-redact"


def main() -> None:
    rendered = claude_plugin_files()
    for relpath, content in rendered.items():
        target = PLUGIN_DIR / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"wrote {target.relative_to(REPO)}")
    for stale in sorted(PLUGIN_DIR.rglob("*")):
        if stale.is_file() and str(stale.relative_to(PLUGIN_DIR)) not in rendered:
            stale.unlink()
            print(f"removed stale {stale.relative_to(REPO)}")
    marketplace = REPO / ".claude-plugin" / "marketplace.json"
    marketplace.parent.mkdir(parents=True, exist_ok=True)
    marketplace.write_text(marketplace_manifest(), encoding="utf-8")
    print(f"wrote {marketplace.relative_to(REPO)}")


if __name__ == "__main__":
    main()
