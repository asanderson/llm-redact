# Quickstart: redacting your first session in five minutes

llm-redact is a local proxy: your agentic tool talks to it, it swaps
private values for `«EMAIL_001»`-style tokens before anything leaves your
machine, and it swaps them back in the responses. The tool never notices.

## 1. Install

```bash
pip install llm-redact-proxy          # or: uv tool install llm-redact-proxy
# from a checkout: uv sync, then prefix the commands below with `uv run`
```

Runtime dependencies are exactly httpx, starlette, and uvicorn. Optional
extras (`[crypto]` vault encryption, `[realtime]` WebSocket APIs, NER
backends) can come later — nothing here needs them.

## 2. Initialize

```bash
llm-redact init
```

The wizard writes `~/.config/llm-redact/config.toml` (XDG paths), picks a
vault backend, and prints the environment exports for the tools you name.
Non-interactive: `llm-redact init --yes --tools claude --vault sqlite`.

## 3. Run something through it

The one-liner — an ephemeral proxy is started if none is running, the
tool's base-URL variable is injected, and everything is torn down when the
tool exits:

```bash
llm-redact run -- claude -p "say hi"
```

Or run the proxy long-lived and point tools at it yourself:

```bash
llm-redact serve &
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787   # Claude Code
# OPENAI_BASE_URL for Codex-style tools; GOOGLE_GEMINI_BASE_URL for Gemini;
# OLLAMA_HOST for ollama; see init's output
```

For a tool whose variable llm-redact does not know:

```bash
llm-redact run --set-env MY_TOOL_BASE_URL -- my-tool ...
```

## 4. Verify it is actually protecting you

```bash
llm-redact preview --text "mail jane.doe@corp.example about AKIAIOSFODNN7EXAMPLE"
```

shows exactly what the current config would redact — entirely locally, no
proxy, no upstream. Then open the dashboard at
`http://127.0.0.1:8787/__llm-redact/` to watch live traffic: detections
and restores by type, per provider, with a recent-request table.

`llm-redact lookup «EMAIL_001»` resolves a token back to its value
(locally; the mapping never leaves the machine).

Agents can verify in-tool, too:
`llm-redact plugin install claude|codex|opencode|cursor` adds
`/llm-redact:status`, `/llm-redact:recent`, and `/llm-redact:preview`
slash commands (Claude Code can instead
`/plugin marketplace add asanderson/llm-redact`); see
[plugins.md](plugins.md).

## 5. Preflight before you change anything

```bash
llm-redact doctor            # read-only diagnostics; add --json for machines
llm-redact serve --check     # serve's full startup minus the socket
llm-redact config show       # the effective config, env overrides named
```

`doctor` dry-runs the detector build, so a config that parses but would
fail serve (an unknown rule name, a bad custom-rule validator) FAILS here
— before a deploy or a `kill -HUP` reload quietly rejects it. Run
`serve --check` as the gate before every reload.

Something not behaving? [troubleshooting.md](troubleshooting.md) is keyed
by the exact error strings you will see. The full configuration surface is
in the README; production hardening lives in [deployment.md](deployment.md).

## 6. Where to go next

`llm-redact guide` prints the packaged **user guide** — the dashboard, the
config editor, and the agent plugin commands, end to end. A running proxy
also serves it at `http://127.0.0.1:8787/__llm-redact/guide`, and agents
with the plugin installed get it as `/llm-redact-guide`. Everything else
is indexed in [docs/README.md](README.md).
