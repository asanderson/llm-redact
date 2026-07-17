"""End-to-end check for the agent plugins: install into a throwaway HOME,
then prove the files land where each tool looks and that every command a
body tells the agent to run is a real `llm-redact` subcommand that at
least parses its own `--help`.

No agent and no network — this exercises the INSTALL + CLI-reference
contract, the part that would silently rot if a command body drifted from
the CLI. Run locally or in CI:

    uv run python scripts/plugin_e2e.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from llm_redact.plugin_assets import COMMANDS
from llm_redact.plugin_cli import TOOLS, _target_dir, install


def main() -> int:
    home = Path(tempfile.mkdtemp(prefix="llm-redact-plugin-e2e-"))
    env = {"HOME": str(home)}
    expected = sorted(f"llm-redact-{c.name}.md" for c in COMMANDS)

    for tool in TOOLS:
        rc = install(tool, env, print_only=False, force=False, posture_hint=lambda: "(e2e)")
        if rc != 0:
            print(f"FAIL: install {tool} returned {rc}")
            return 1
        target = _target_dir(tool, env)
        got = sorted(p.name for p in target.iterdir())
        if got != expected:
            print(f"FAIL: {tool} wrote {got}, expected {expected}")
            return 1
        print(f"ok: {tool} -> {target} ({len(got)} commands)")

    # Every backticked `llm-redact <sub>` in a command body must be a real
    # subcommand whose --help parses (argparse exits 0). This is the live
    # version of the static sync test — it actually runs the CLI.
    subs = set()
    for command in COMMANDS:
        subs.update(re.findall(r"`llm-redact ([a-z][a-z-]*)", command.body))
    for sub in sorted(subs):
        result = subprocess.run(
            [sys.executable, "-m", "llm_redact", sub, "--help"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"FAIL: `llm-redact {sub} --help` exited {result.returncode}")
            print(result.stderr[-500:])
            return 1
        print(f"ok: llm-redact {sub} --help")

    # Re-install must be a no-op (idempotent), and a modified file must be
    # refused without --force.
    if install("cursor", env, print_only=False, force=False, posture_hint=lambda: "-") != 0:
        print("FAIL: idempotent re-install did not return 0")
        return 1
    victim = _target_dir("cursor", env) / "llm-redact-status.md"
    victim.write_text("hand-edited", encoding="utf-8")
    if install("cursor", env, print_only=False, force=False, posture_hint=lambda: "-") != 1:
        print("FAIL: modified file was overwritten without --force")
        return 1
    print("ok: idempotent + modification-guard")

    print("\nplugin e2e passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
