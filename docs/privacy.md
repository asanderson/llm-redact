# Privacy policy

llm-redact is local software. The proxy, the vault, the dashboard, and
the agent plugin commands all run on your machine, and the project
collects nothing.

## What we (the authors) receive from you

Nothing. llm-redact contains no telemetry, no analytics, no crash
reporting, and no update checks. License verification is offline by
design — it never phones home. The authors have no way to know you are
running it.

## What the software stores, locally

- **The vault** — the mapping between placeholder tokens and your real
  values. It exists so responses can be restored, it is written with
  owner-only file permissions, and it never leaves your machine.
- **The audit log (optional, off by default)** — metadata only: request
  paths, detection *types and counts*, durations. Never the detected
  values, never message content.

## What leaves your machine

- **Redacted traffic to your LLM provider.** The proxy's entire job is
  to substitute placeholders for detected private values before your
  agent's request reaches the provider you configured. What the provider
  receives is governed by *your* agreement with that provider.
- **Configured opt-outs are honest.** If you enable warn-mode rules,
  disable detection for a provider, or exempt an MCP server, matched
  values in that scope ARE forwarded — every such opt-out is surfaced in
  `/status`, `llm-redact doctor`, and the dashboard, never silent.
- **Optional audit sinks you configure.** The S3/Azure audit sinks
  upload the same metadata-only rows to object storage *you* control,
  optionally client-side encrypted. Off by default.

## The agent plugin commands

The slash commands (`/llm-redact:status`, `:recent`, `:preview`, …)
talk only to the local `llm-redact` CLI and the proxy's loopback
endpoints. They contact no external service, and no command exposes
vault values to the agent — `lookup` is deliberately not a plugin
command, because an agent that read a secret would send it upstream.

## Contact

Questions: open a [GitHub issue](https://github.com/asanderson/llm-redact/issues).
Vulnerabilities: see [SECURITY.md](SECURITY.md) (private reporting).
