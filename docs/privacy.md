# Privacy policy

llm-redact is local software. The proxy, the vault, the agent plugin
commands, and (with llm-redact-pro) the dashboard all run on your
machine, and the project collects nothing.

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
  `/status`, `llm-redact status`, and `llm-redact doctor`, never silent.
- **Optional audit sinks you configure.** The S3/Azure audit sinks
  upload the same metadata-only rows to object storage *you* control,
  optionally client-side encrypted. Off by default.
- **NER model downloads, only if you allow them.** The optional local
  NER models (`[detection.ner]`, off by default) load from the local
  Hugging Face cache or a local folder, and nothing is downloaded unless
  you set `allow_download = true`. Then only the proxy's startup may fetch
  a missing model from huggingface.co: the request names the model id,
  the pinned revision and the files, never anything your agent sends; a
  reload never downloads. `llm-redact models pull` is the explicit way to
  fetch them yourself, and `llm-redact doctor` warns while downloads are
  on.
  Nothing else in the NER path reaches the network: Presidio's email
  check (tldextract) reads the Public Suffix List snapshot its package
  ships instead of fetching publicsuffix.org on first use, and a CI job
  starts the proxy with its NER models inside a network namespace with no
  route, failing on any connection attempt made through Python's `socket`
  module ([air-gapped.md](air-gapped.md)).

## The agent plugin commands

The slash commands (`/llm-redact:status`, `:recent`, `:preview`, …)
talk only to the local `llm-redact` CLI and the proxy's loopback
endpoints. They contact no external service, and no command exposes
vault values to the agent — `lookup` is deliberately not a plugin
command, because an agent that read a secret would send it upstream.

## Contact

Questions: open a [GitHub issue](https://github.com/asanderson/llm-redact/issues).
Vulnerabilities: see [SECURITY.md](SECURITY.md) (private reporting).
