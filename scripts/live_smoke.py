"""Run the live-API smoke tests against real provider endpoints.

WARNING: this sends real requests and costs real API credits. It never runs
as part of the default test suite or CI on pull requests.

Usage:
    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... uv run python scripts/live_smoke.py
    uv run python scripts/live_smoke.py --provider anthropic
"""

import argparse
import os
import subprocess
import sys

_PROVIDER_SELECTORS = {
    "anthropic": "test_anthropic",
    "openai": "test_openai",
    "responses": "test_responses",
    "gemini": "test_gemini",
    "bedrock": "test_bedrock",
    "realtime": "realtime_live or gemini_live",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=[*_PROVIDER_SELECTORS, "all"],
        default="all",
        help="which provider's live tests to run",
    )
    args = parser.parse_args()

    present = [
        name
        for name, env in (
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("gemini", "GEMINI_API_KEY"),
            ("bedrock", "AWS_BEARER_TOKEN_BEDROCK"),
        )
        if os.environ.get(env)
    ]
    if not present:
        print("no API keys in the environment; nothing to run")
        print(
            "set ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY and/or AWS_BEARER_TOKEN_BEDROCK"
        )
        return 1
    print(f"keys present for: {', '.join(present)}")
    print("NOTE: these tests send real requests and cost real API credits.\n")

    command = ["uv", "run", "pytest", "-m", "live", "-rA", "tests/test_live.py"]
    if args.provider != "all":
        command += ["-k", _PROVIDER_SELECTORS[args.provider]]
    env = dict(os.environ, LLM_REDACT_LIVE="1")
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    sys.exit(main())
