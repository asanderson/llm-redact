#!/usr/bin/env python3
"""Dogfood: drive the REAL `claude` CLI through a real llm-redact proxy.

This is the one test that exercises the actual production loop end to end:
a genuine agentic tool, genuine Anthropic API traffic, a subprocess proxy
with a persistent per-conversation vault — no fakes anywhere. It costs real
API money (a few short haiku turns, well under a cent) and needs a
logged-in `claude` CLI, so it is double-gated: set LLM_REDACT_DOGFOOD=1 and
have working CLI auth. It never runs in CI by default.

    LLM_REDACT_DOGFOOD=1 uv run python scripts/dogfood_claude.py

What it proves:
  1. redaction  — /status detections_total.EMAIL grows when the canary is
     sent (deterministic: independent of model behavior);
  2. round trip — the CLI's final JSON answer contains the ORIGINAL canary
     (the model only ever saw the placeholder; one retry, echo is a model
     behavior);
  3. vault      — `llm-redact lookup --value <canary>` resolves to a token
     in the on-disk vault;
  4. compaction — a rewritten first message (what history compaction does)
     forks a fresh session that fails SAFE: no cross-session restoration,
     unowned tokens pass through verbatim.

The canary is unique per run (dogfood-<8hex>@canary.example) so the model
cannot answer from memory and stale vault state cannot mask a failure.
"""

import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def wait_for_status(port: int, proc: subprocess.Popen, deadline_s: float = 15.0) -> dict:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if proc.poll() is not None:
            sys.exit(f"proxy exited early with code {proc.returncode}")
        try:
            return http_json(f"http://127.0.0.1:{port}/__llm-redact/status")
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.2)
    sys.exit("proxy did not come up within 15s")


def run_claude(prompt: str, port: int, *, resume: str | None = None) -> dict:
    command = [
        "claude",
        "-p",
        prompt,
        "--model",
        "haiku",
        "--output-format",
        "json",
    ]
    if resume is not None:
        command += ["--resume", resume]
    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
    }
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=300)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "401" in stderr or "authentication" in stderr.lower():
            sys.exit(
                "claude CLI got an auth error — its login has likely expired;"
                " this is not an llm-redact regression.\n" + stderr
            )
        sys.exit(f"claude CLI failed ({result.returncode}):\n{stderr}\n{result.stdout}")
    return json.loads(result.stdout)


def main() -> int:
    if os.environ.get("LLM_REDACT_DOGFOOD") != "1":
        print("set LLM_REDACT_DOGFOOD=1 to run (talks to the real API, costs money)")
        return 2
    if shutil.which("claude") is None:
        sys.exit("no `claude` CLI on PATH")

    canary = f"dogfood-{secrets.token_hex(4)}@canary.example"
    # A deny-string canary that matches NO built-in rule: proves the user
    # deny list redacts and round-trips through a real model too.
    deny_canary = f"opdeny {secrets.token_hex(3)} zulu"
    port = free_port()
    tmp = Path(tempfile.mkdtemp(prefix="llm-redact-dogfood-"))
    vault_db = tmp / "vault.db"
    config = tmp / "config.toml"
    config.write_text(
        f'port = {port}\n\n[detection]\ndeny = ["{deny_canary}"]\n'
        f'\n[vault]\nbackend = "sqlite"\npath = "{vault_db}"\n'
        'session_mode = "per-conversation"\n'
    )

    print(f"proxy on 127.0.0.1:{port}, vault {vault_db}, canary {canary}")
    proc = subprocess.Popen(
        ["uv", "run", "llm-redact", "serve", "--config", str(config)],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_status(port, proc)
        failures: list[str] = []

        # --- 1+2: real CLI through the proxy, redaction + round trip -----
        prompt = (
            f"My contact email is {canary} — repeat that email address back"
            " to me exactly, and say nothing else."
        )
        answer = run_claude(prompt, port)
        text = str(answer.get("result", ""))
        session_id = str(answer.get("session_id", ""))

        status = http_json(f"http://127.0.0.1:{port}/__llm-redact/status")
        emails_redacted = status["detections_total"].get("EMAIL", 0)
        if emails_redacted < 1:
            failures.append(
                f"redaction: detections_total.EMAIL is {emails_redacted}, expected >= 1"
            )
        else:
            print(f"PASS redaction: EMAIL detections {emails_redacted}")

        if canary not in text:
            print("echo miss (model behavior); retrying once...")
            answer = run_claude(prompt, port)
            text = str(answer.get("result", ""))
        if canary in text:
            print("PASS round trip: CLI answer contains the original canary")
        else:
            failures.append(f"round trip: canary absent from CLI answer: {text!r}")

        # --- 2b: deny-string canary redacts and round-trips ---------------
        deny_answer = run_claude(
            f"Our secret project codename is {deny_canary.upper()} — repeat"
            " the codename back exactly, and say nothing else.",
            port,
        )
        deny_text = str(deny_answer.get("result", ""))
        status = http_json(f"http://127.0.0.1:{port}/__llm-redact/status")
        if status["detections_total"].get("DENY", 0) >= 1:
            print(f"PASS deny: DENY detections {status['detections_total']['DENY']}")
        else:
            failures.append("deny: detections_total.DENY did not grow")
        if deny_canary.upper() in deny_text:
            print("PASS deny round trip: CLI answer contains the original casing")
        else:
            print(f"NOTE deny echo miss (model behavior); answer: {deny_text!r}")

        # --- 3: the mapping is in the on-disk vault ----------------------
        lookup = subprocess.run(
            ["uv", "run", "llm-redact", "lookup", "--value", canary, "--db", str(vault_db)],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if lookup.returncode == 0 and "«EMAIL_" in lookup.stdout:
            print(f"PASS vault: {lookup.stdout.strip().splitlines()[-1]}")
        else:
            failures.append(f"vault lookup failed: {lookup.stdout} {lookup.stderr}")

        # --- 4: deterministic compaction probe ----------------------------
        # History compaction rewrites the first user message, so the session
        # anchor changes. Simulate exactly that with a fresh CLI conversation
        # whose prompt IS a rewritten summary containing the raw canary plus
        # a token no session ever issued. Fail-safe means: new session, the
        # unowned token passes through VERBATIM (never another session's
        # value), and the raw canary still redacts.
        sessions_before = status["vault"]["sessions"]
        emails_redacted = status["detections_total"].get("EMAIL", 0)
        probe = run_claude(
            "Summary of earlier conversation: the user's email is "
            f"{canary} and we saw the token «EMAIL_999». Repeat the"
            " token «EMAIL_999» back exactly, and say nothing else.",
            port,
        )
        status = http_json(f"http://127.0.0.1:{port}/__llm-redact/status")
        probe_text = str(probe.get("result", ""))
        if status["vault"]["sessions"] > sessions_before:
            print(f"PASS compaction: fresh session forked ({status['vault']['sessions']} total)")
        else:
            failures.append("compaction: rewritten anchor did not fork a new session")
        if status["detections_total"].get("EMAIL", 0) > emails_redacted:
            print("PASS compaction: raw canary in the rewritten summary was re-redacted")
        else:
            failures.append("compaction: canary in rewritten summary was NOT redacted")
        if "«EMAIL_999»" in probe_text:
            print("PASS compaction: unowned token echoed verbatim (no cross-session restore)")
        elif "EMAIL_999" in probe_text.replace(" ", ""):
            print("PASS compaction: unowned token echoed (model-mangled, still not restored)")
        else:
            print(f"NOTE compaction: model did not echo the token (answer: {probe_text!r})")

        # --- best effort: a real /compact through the CLI -----------------
        if session_id:
            try:
                compacted = run_claude("/compact", port, resume=session_id)
                followup = run_claude(
                    "What is my contact email address? Answer with just the address.",
                    port,
                    resume=str(compacted.get("session_id", session_id)),
                )
                follow_text = str(followup.get("result", ""))
                if canary in follow_text:
                    print("PASS real /compact: canary still round-trips after compaction")
                else:
                    # Documented fail-safe: compaction may fork the session,
                    # in which case old placeholders pass through verbatim
                    # rather than restoring — never a wrong value.
                    print(
                        "NOTE real /compact: canary not restored after compaction"
                        f" (documented fail-safe fork); answer: {follow_text!r}"
                    )
            except SystemExit:
                raise
            except Exception as exc:  # /compact via -p is undocumented
                print(f"NOTE real /compact unsupported here ({exc}); deterministic probe covers it")

        if failures:
            print("\nFAILURES:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print("\ndogfood passed: redaction, round trip, vault, compaction fail-safe")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
