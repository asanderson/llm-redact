# llm-redact

[![CI](https://github.com/asanderson/llm-redact/actions/workflows/ci.yml/badge.svg)](https://github.com/asanderson/llm-redact/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llm-redact-proxy)](https://pypi.org/project/llm-redact-proxy/)
[![Container](https://img.shields.io/badge/ghcr.io-asanderson%2Fllm--redact-blue?logo=docker&logoColor=white)](https://github.com/asanderson/llm-redact/pkgs/container/llm-redact)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)

Large Language Model (LLM) information redactor that prevents private information from being sent to LLMs from agentic tools by substituting placeholders for private information on outgoing requests and then replaces the placeholders on the incoming responses seamlessly for the agentic tool users.

![System data flow: agentic tool (with the llm-redact plugin slash commands inside it), proxy, and local vault on your machine; only placeholder tokens reach the LLM provider. A dashed box marks the audit log and its object-store sinks, and the audit-row flow into them, as Pro](docs/diagrams/architecture.png)

Only placeholder tokens cross the trust boundary; the vault mapping never
leaves your machine. The dashed box marks the audit subsystem — the audit
log, its object-store sinks, and the audit-row flow into them — as
**Pro** (supplied by the proprietary `llm-redact-pro` package, like vault
at-rest encryption and the server RDBMS vault backends the diagram
doesn't draw); everything else pictured is part of this repository's
**FOSS core** (matrix in [docs/editions.md](docs/editions.md)). The full
documentation set — quickstart, deployment, security, engineering
record — is indexed in [docs/](docs/README.md).

## Contents

- [Why redact?](#why-redact)
- [What it does](#what-it-does)
- [What it does not protect](#what-it-does-not-protect)
- [Plugins](#plugins) — detail: [command reference and screenshots](docs/plugins.md)
- [Install](#install)
- [Quickstart](#quickstart) — detail: [five-minute walkthrough](docs/quickstart.md), [provider setup and coverage](docs/providers.md)
- [How it works](#how-it-works) — detail: [round trip, worked example, persistence, sessions](docs/how-it-works.md)
- [Editions and licensing](#editions-and-licensing) — detail: [tiers, keys, and the open-core boundary](docs/editions.md)
- [Containers (docker or podman)](#containers-docker-or-podman)
- [Status, dashboard, metrics, and audit log](#status-dashboard-metrics-and-audit-log) — detail: [the local ops surface](docs/dashboard.md)
- [Operations](#operations) — detail: [the deployment guide](docs/deployment.md)
- [What gets detected](#what-gets-detected) — detail: [the detection reference](docs/detection.md)
- [Benchmark and live validation](#benchmark-and-live-validation)
- [Trying it without touching a real API](#trying-it-without-touching-a-real-api)
- [Security](#security)
- [Tech stack](#tech-stack) — detail: [software bill of materials](docs/SBOM.md), [what ships and why](docs/dependencies.md)
- [Stability](#stability)
- [Development](#development)

## Why redact?

Agentic tools are built to gather context, and they are thorough about
it: to answer a prompt they read files, run commands, and paste the
results into API requests — the `.env` beside the module you asked
about, a customer's email address in a stack trace, the database DSN in
a config file, an AWS key in your shell history. You review the tool's
*answer*; you almost never review the *request*. Every one of those
requests leaves your machine for a third-party LLM provider, and once
it does:

- **It cannot be recalled.** Prompts may be retained for abuse
  monitoring or legal holds, sampled for human review, or — depending
  on the provider, plan, and settings — used for training. A value
  transmitted once has to be treated as disclosed.
- **Leaked credentials are live.** An API key or token in a prompt is a
  working credential sitting in someone else's infrastructure and logs;
  a provider-side incident or over-broad logging turns it into a
  breach. The only sound response to that kind of exposure is rotating
  the key — every key, every time it happens.
- **Personal data creates legal exposure.** Sending names, emails,
  phone numbers, national identifiers, or health data to an LLM API is
  a disclosure to a third-party processor. Privacy regimes such as
  GDPR, CCPA, and HIPAA attach conditions to that disclosure — a legal
  basis, data-processing agreements, purpose limits — and an unnoticed
  leak through an agent's context window can be a reportable incident.
- **Confidentiality obligations don't pause for tooling.** NDA-covered
  material, customer records, and unreleased work product sit in
  ordinary repositories, tickets, and logs, and an agent will forward
  them upstream without malice or notice.

llm-redact removes the trade-off between agentic productivity and
keeping that data private: recognized PII and secrets never leave your
machine — the provider sees only placeholder tokens, your tool sees
only real values, and nothing about your workflow changes. The limits
of that protection are spelled out in
[What it does not protect](#what-it-does-not-protect) below.

## What it does

A local transparent proxy sits between your agentic tool (Claude Code, or anything speaking the Anthropic Messages or OpenAI Chat Completions API) and the provider. Outbound requests are scanned for emails, credentials, API keys, and other private values; each detected value is replaced with a collision-resistant token like `«EMAIL_001»` and the mapping is kept in a local vault that never leaves your machine. Responses — including streamed SSE responses, even when a token is split across chunks — have the tokens swapped back for the originals, so the tool sees exactly what it would have seen without the proxy.

It also ships **agent plugins** that put the proxy's controls inside the
tool itself: slash commands for live status and recent traffic, a
redaction dry-run, and guarded config editing, for Claude Code, Codex,
OpenCode, and Cursor ([Plugins](#plugins) below).

## What it does not protect

Redaction substitutes *recognized, value-shaped* secrets — an email
address, an API key, a national identifier. Be equally clear about what
that cannot cover:

- **Ideas and proprietary content.** Your code's structure, business
  logic, proprietary algorithms, product plans, and everything else the
  prompt is *about* reach the provider in the clear — the model has to
  read them to work on them, and no redactor can substitute a
  placeholder for meaning. If the content itself is the secret, the
  only protection is not sending it: scope what your agentic tool may
  read, and keep that material out of its reach.
- **Inference from context.** The provider never sees a redacted value,
  but it can sometimes guess at one from the words around the
  placeholder ("the CFO's personal address «EMAIL_004»"). Only omission
  fixes that.
- **Values with no reliable shape.** Bare-digit phone numbers and SSNs,
  street addresses, and passport/driver's-license numbers are
  deliberately never matched (no grammar separates them from ordinary
  numbers), and a secret format no rule knows will not fire. You can
  extend coverage with `[[detection.custom_rules]]`, deny strings, and
  the NER extras — and `llm-redact preview` shows exactly what fires on
  a given text before you rely on it.
- **Media.** Base64 images, PDFs, and audio are never decoded and
  scanned; on realtime voice connections, only the text modality is
  redacted — what a user *says* reaches the provider as-is.
- **Anything you opt out of.** Warn-mode rules observe and *forward*
  the matched value; `[providers.NAME] detection = false`, MCP server
  exemptions, and language scoping likewise forward what they exempt.
  Every such opt-out is surfaced in `/status`, `doctor`, and the
  dashboard — never silent.

The precise boundary — assets, trust assumptions, and the full
out-of-scope table — is written down in
[docs/threat-model.md](docs/threat-model.md).

## Plugins

Ten slash commands mirror the dashboard and config editor inside
Claude Code, Codex, OpenCode, and Cursor, so the proxy can be driven
without leaving the tool:

- `/llm-redact:status`, `/llm-redact:recent`, `/llm-redact:sessions` —
  live coverage posture, the recent-request table, and the vault's
  session list;
- `/llm-redact:preview` — dry-run what the current config *would*
  redact, warn on, or block, entirely locally;
- `/llm-redact:config-show` and a guarded `/llm-redact:config-edit`
  (effective-config read → TOML edit → `serve --check` gate → SIGHUP
  reload → posture read-back);
- `/llm-redact:doctor`, `/llm-redact:audit`, `/llm-redact:users`, and
  `/llm-redact:guide`.

Install with `llm-redact plugin install claude|codex|opencode|cursor`,
or in Claude Code add this repo as a plugin marketplace:
`/plugin marketplace add asanderson/llm-redact`. Every command opens
with a proxy-presence guard — a missing `llm-redact` CLI stops the
command and asks your approval before anything is installed — and
`lookup` is deliberately not a command: an agent that read a secret
value would send it upstream. The full command reference, per-tool
install paths, and terminal screenshots are in
[docs/plugins.md](docs/plugins.md).

## Install

Requires Python 3.11+ on Linux or macOS (both CI-tested; Windows is
unsupported). The PyPI distribution is **`llm-redact-proxy`** (the
bare name belongs to an unrelated project); the CLI it installs is
`llm-redact`.

```bash
uvx --from llm-redact-proxy llm-redact serve   # zero-install run
pip install llm-redact-proxy                   # or a normal install
# Homebrew (after the first PyPI release is published):
brew tap asanderson/llm-redact https://github.com/asanderson/llm-redact
brew install asanderson/llm-redact/llm-redact
```

Or skip Python entirely and pull the prebuilt multi-arch container
image from GHCR with docker **or** podman — publish it **to loopback
only** (a bare `-p 8787:8787` would expose the proxy, and the secrets
it rehydrates, to your LAN):

```bash
docker pull ghcr.io/asanderson/llm-redact:latest      # or: podman pull …
docker run -d --name llm-redact \
  -p 127.0.0.1:8787:8787 -v llm-redact-data:/data \
  ghcr.io/asanderson/llm-redact:latest                # or: podman run …
```

Config mounts, persistence, and health probes are covered in
[Containers (docker or podman)](#containers-docker-or-podman) below.

Undecided? [`scripts/install.sh`](scripts/install.sh) is an interactive
installer: it detects what is available on your machine (uv, pipx, pip,
Homebrew, docker, podman), asks which install you prefer, prints each
command before running it, and does nothing else. Download it and read
it before running — don't pipe curl into a shell:

```bash
curl -fsSLO https://raw.githubusercontent.com/asanderson/llm-redact/main/scripts/install.sh
bash install.sh                    # or non-interactive: bash install.sh --method podman
```

Then:

- `llm-redact init` — writes a starter config interactively and prints
  the env exports for your tools.
- `llm-redact service install` — runs the proxy at login
  (launchd/systemd user unit).
- `llm-redact plugin install claude|codex|opencode|cursor` — the
  dashboard and config-editor workflows as **agent slash commands**
  (Claude Code can instead add this repo as a plugin marketplace:
  `/plugin marketplace add asanderson/llm-redact`); see
  [docs/plugins.md](docs/plugins.md).
- `llm-redact completions bash|zsh|fish` — shell completions.

## Quickstart

```bash
pip install llm-redact-proxy                # or: uv sync, from a checkout
llm-redact run -- claude -p "hello"         # env injected; ephemeral proxy if none running
```

Or run it long-lived (`llm-redact serve`, listens on `127.0.0.1:8787`)
and point tools at it via their base-URL variables (`ANTHROPIC_BASE_URL`,
`OPENAI_BASE_URL`, `GOOGLE_GEMINI_BASE_URL`, `OLLAMA_HOST`). From there:

- **Five-minute walkthrough** — install, init, run, verify, preflight:
  [docs/quickstart.md](docs/quickstart.md).
- **Providers** — the point-a-tool matrix (Anthropic, OpenAI, Azure
  OpenAI, Gemini, Vertex, Bedrock, Cohere, Ollama, custom
  OpenAI-compatible upstreams), per-provider setup, batch APIs, and the
  realtime WebSocket relay: [docs/providers.md](docs/providers.md);
  endpoint-by-endpoint coverage:
  [docs/api-coverage.md](docs/api-coverage.md).
- **Agent plugins** — drive the proxy without leaving Claude Code,
  Codex, OpenCode, or Cursor: `/llm-redact:status`,
  `/llm-redact:recent`, `/llm-redact:preview`, and a guarded
  `/llm-redact:config-edit`, installed via `llm-redact plugin install`
  (or the Claude Code plugin marketplace); see
  [docs/plugins.md](docs/plugins.md).
- **Preflight and diagnosis** — `llm-redact doctor` (read-only
  PASS/WARN/FAIL report incl. a detector-build dry-run; `--json` for
  machines) and `llm-redact serve --check` (the deploy/reload gate);
  error-by-error fixes: [docs/troubleshooting.md](docs/troubleshooting.md).
- **User guide** — the dashboard, config editor, and plugin commands,
  written for the person *using* the proxy: served at
  `/__llm-redact/guide` (also `llm-redact guide`, or packaged as
  [src/llm_redact/user_guide.md](src/llm_redact/user_guide.md)). Full
  doc index: [docs/README.md](docs/README.md).

API keys pass through untouched (and are never logged); each request is
logged as a one-line metadata summary, e.g.
`POST /v1/messages -> 200 redacted: EMAIL×1`. Configuration is optional —
[`config.example.toml`](config.example.toml) documents every setting;
`LLM_REDACT_HOST`/`_PORT`/`_CONFIG` override it, and precedence is
CLI flag > env var > config file.

## How it works

Outbound requests are scanned and each detected value is replaced with
a vault-issued token; responses — streaming included, even when a token
is split across chunk boundaries — are restored byte-for-byte.
[docs/how-it-works.md](docs/how-it-works.md) covers the mechanism end
to end:

- the request round trip, with the non-streaming and streaming sequence
  diagrams;
- a worked example with the exact records one request leaves behind —
  the placeholder body the provider sees, the injected system note, the
  restored response, the vault rows, and the metadata-only audit row
  (with the tamper-evident-versus-zero-loss audit distinction);
- vault persistence, fuzzy restoration of mangled tokens, and the
  request size limit;
- session isolation (the static default and the Pro per-conversation
  mode).

Every step is observable from inside your agent: the slash-command
plugins (`/llm-redact:status`, `/llm-redact:recent`,
`/llm-redact:preview` — [docs/plugins.md](docs/plugins.md)) show live
detections and restores without leaving the tool.

## Editions and licensing

This repository is **free and open-source software** under the
[GNU AGPL-3.0](LICENSE): use it for anything, inspect every line, modify
it, and redistribute it. The one obligation is share-alike — if you
distribute a modified version, **or run one as a network service for
others**, you must offer them your modified source under the same
license. Unmodified self-hosted use (the normal case: your tools talking
to your own proxy) carries no obligations at all. Contributions are
accepted under a [contributor license agreement](docs/CLA.md).

**Nothing in this repository is gated**: no license keys, no tiers, no
seat caps — every detection rule, redaction mode, NER backend, provider
adapter (including Bedrock/Azure/Vertex), the realtime relay,
non-loopback (mTLS) serving, and Kubernetes deployment all work keyless,
with no usage limits. Paid tiers (Pro / Team / Unlimited) exist only as
the commercial packaging of the separately-installed, **proprietary**
`llm-redact-pro` package (a per-seat subscription; not open source —
**coming soon**, not yet generally available), which supplies additional
operational subsystems — persistent server vaults and encryption at
rest, the audit log and its off-machine sinks, OTel, per-conversation
sessions, and named users. Configuring one of those without the package
fails closed naming it — never a silent downgrade. The matrix and
licensing model are in [docs/editions.md](docs/editions.md).

## Containers (docker or podman)

The same proxy runs as an OCI container; CI builds and smoke-tests both
engines, and released GHCR images are multi-arch (amd64 + arm64 — native
on Apple Silicon and Graviton). `LLM_REDACT_HOST=0.0.0.0` in the image binds only inside the
container's network namespace — the host boundary is your publish spec.
**Always publish to loopback**: a bare `-p 8787:8787` would expose the proxy
(and the secrets it rehydrates) to your LAN.

```bash
docker pull ghcr.io/asanderson/llm-redact:latest   # prebuilt (or: podman pull …)
docker build -t llm-redact .                       # or build from a checkout
docker run -d -p 127.0.0.1:8787:8787 \
  -v ./config.toml:/etc/llm-redact/config.toml:ro \
  -v llm-redact-data:/data \
  llm-redact
# or: podman run ... (rootless works; use :U on bind mounts if needed)
docker compose up   # demo stack: proxy + bundled fake upstream
```

The sqlite vault and audit log live under `/data` — mount a volume to
persist sessions across container restarts.

## Status, dashboard, metrics, and audit log

Reserved `/__llm-redact/*` paths are answered locally, never forwarded, and
carry metadata only — never redacted values. The namespace is GET-only
except the guarded config-editor, session-prune, and preview POSTs, and
every reply is hardened with a strict CSP and framing/sniffing headers.

![The dashboard: status pills, detections and restores by type, upstreams, and the recent-request table](docs/screenshots/dashboard.png)

- **Dashboard** — <http://127.0.0.1:8787/__llm-redact/>, a single
  self-contained page (no CDNs, works offline): live detection/restore
  totals by type and the recent-request table with an instant
  server-sent-event feed.
- **Config editor** — every hot-reloadable setting (rules, modes,
  allowlists, deny strings, custom rules, NER, providers), guarded with
  layered Host/Origin/CSRF protection.
- **Redaction preview** — paste text, see what the current config
  *would* redact — entirely locally, nothing sent upstream, nothing
  written.
- **Scriptable surface** — `llm-redact status`, Prometheus
  `GET /__llm-redact/metrics` (always on), DB-free `healthz`/`readyz`
  probes.
- **Agent slash commands** — the same workflows inside Claude Code,
  Codex, OpenCode, and Cursor
  (`/plugin marketplace add asanderson/llm-redact`, or
  `llm-redact plugin install`; see [docs/plugins.md](docs/plugins.md)).
- **Coverage posture, surfaced loudly** — every configured opt-out that
  lets traffic through unredacted is reported by `status`, `doctor`, and
  the dashboard; never silently.
- **Audit log** (Pro) — with its tamper-evident chain and off-machine
  object-store sinks; metadata only (types, counts, paths, durations —
  never values).

The full tour of every endpoint and screen — with screenshots — is
[docs/dashboard.md](docs/dashboard.md).

## Operations

[docs/deployment.md](docs/deployment.md) is the end-to-end deployment
guide — bind policy, containers and health probes, vault lifecycle
(backup/verify/TTL), service management, log rotation, and
observability wiring, all keyed to the mutual-TLS bind policy. The
essentials:

- **Config reload without a restart**: edit the config file, then send
  SIGHUP — `kill -HUP $(pgrep -f 'llm-redact serve')`, or for containers
  `docker kill --signal=HUP llm-redact`. Detection settings and upstream
  URLs apply immediately; vault, audit, host, port, log, TLS, OTel,
  users, and email changes are kept as-is with a "require restart"
  warning. A broken config file is logged and ignored — the running
  config stays active.
- **Agent plugins**: the ops workflows ship as slash commands —
  `/llm-redact:status`, `/llm-redact:recent`, `/llm-redact:doctor`, and
  a guarded `/llm-redact:config-edit` that mirrors the reload flow
  (effective-config read → TOML edit → `serve --check` gate → SIGHUP →
  posture read-back). Install with
  `llm-redact plugin install claude|codex|opencode|cursor`; see
  [docs/plugins.md](docs/plugins.md).
- **Logs**: one metadata-only line per request to stderr — run under
  systemd (journald owns retention) or cap the container log driver.
  `[log] format = "json"` switches to one JSON object per line for log
  shippers (same content; never values or headers).
- **Data retention**: the audit log prunes itself (`[audit] max_rows`);
  `llm-redact sessions list|prune --older-than 90d` and
  `[vault] session_ttl_days` bound vault growth — whole idle sessions
  only, since partial deletion could reuse a still-referenced token
  number.
- **Vault durability**: `llm-redact vault verify` (read-only integrity
  sweep) and `vault backup` (WAL-safe online snapshot, created `0600`)
  protect the one piece of state whose corruption would silently
  rehydrate a wrong secret.
- **Bind policy**: `serve` refuses to start on a non-loopback `host`
  without the full mutual-TLS trio
  (`[tls] certfile`/`keyfile`/`client_ca`) — a network client of the
  proxy can read rehydrated secrets, so the default and recommended
  bind is loopback. Vault encryption at rest and key rotation, and
  OpenTelemetry export are Pro features of the `llm-redact-pro` package
  (coming soon).
- **FIPS 140-3 posture**: all cryptographic uses are FIPS-approved
  algorithm selections — `llm-redact fips-check`,
  [docs/fips.md](docs/fips.md).
- **Event loop**: `pip install 'llm-redact-proxy[perf]'` adds uvloop;
  uvicorn picks it up automatically — no configuration.

## Benchmark and live validation

- `uv run python -m llm_redact.bench --check` scores every detection rule
  (precision/recall/F1) on a deterministic synthetic corpus AND scans the
  vendored false-positive corpus (`bench/fp_corpus/` — real-world code,
  RFC text, and prose) against exact per-file expected counts, so both a
  recall regression and a precision regression fail CI; the report is
  uploaded as an artifact.
- Add `--latency` for the in-process overhead benchmark. Typical numbers on
  a dev container (the prefiltered single-pass-per-anchor scan): ~1 ms
  added per small (2 KB) request end to end, ~14 ms on a secret-dense
  100 KB body, ~5-11 ms on 100 KB of ordinary prose, ~18 MB/s streaming
  rehydration.
- `uv run python scripts/live_smoke.py` runs opt-in smoke tests against the
  real provider APIs (needs API keys, spends credits, never runs in default
  test or CI runs) — including an event-shape drift detector for the
  `/v1/responses` adapter.
- `LLM_REDACT_DOGFOOD=1 uv run python scripts/dogfood_claude.py` drives the
  **real `claude` CLI** through a real proxy subprocess and asserts
  redaction, round trip, the on-disk vault mapping, and the
  history-compaction fail-safe with a per-run unique canary (needs a
  logged-in CLI; costs well under a cent).

## Trying it without touching a real API

```bash
uv run python scripts/fake_upstream.py --port 9999   # terminal 1: fake provider
# terminal 2: proxy pointed at the fake
cat > /tmp/llm-redact-demo.toml <<'EOF'
[providers.anthropic]
upstream_base_url = "http://127.0.0.1:9999"
EOF
uv run llm-redact serve --config /tmp/llm-redact-demo.toml
# terminal 3: a request containing an email and an AWS key
curl -sN http://127.0.0.1:8787/v1/messages \
  -H 'content-type: application/json' -H 'x-api-key: test' \
  -d '{"model":"claude-sonnet-4-5","stream":true,"max_tokens":100,
       "messages":[{"role":"user","content":"mail jane@corp.example key AKIAIOSFODNN7EXAMPLE"}]}'
```

The fake upstream prints the request body it received (placeholders only), and
the curl output shows the originals restored in the stream.

## What gets detected

- Emails, IP addresses, credit cards, and phone numbers — all validated
  or checksummed; bare digit runs never fire.
- Twenty-plus national identifiers (US SSN, Canadian SIN, UK NINO and
  NHS, Indian Aadhaar, French NIR, Korean RRN, and more) — every one
  matched in its grouped or signed display form **and**
  checksum-validated.
- IBANs and cryptocurrency wallet addresses (EIP-55 / base58check /
  bech32) — all checksum-vetoed.
- URL-embedded passwords; AWS keys and a wide vendor token family
  (GitHub, GitLab, Anthropic, OpenAI, Slack, Databricks, and more).
- PEM and PGP/GPG private-key blocks, plus a keyword-context + entropy
  rule for generic secrets (`password = "..."` etc.).

The complete annotated reference is
[docs/detection.md](docs/detection.md) — the full rule list, **deny
strings** (values that must always be redacted, highest precedence,
never subject to modes), **per-rule modes** (`redact`/`warn`/`block`),
custom rules with checksum validators, global and per-type allowlists,
and the optional **person-name NER backends** (spaCy, GLiNER, Presidio,
Stanza, Hugging Face). The current rule list also ships in
[`config.example.toml`](config.example.toml).

The NER backends are backed by a research survey,
[docs/ner-landscape.md](docs/ner-landscape.md) — the engines evaluated
and the bar each had to clear, why LLM-based extractors and SaaS PII
APIs are rejected as a class (the raw pre-redaction text must never
leave your machine), and the commercial self-hosted engines that would
qualify if demand materializes.

## Security

- **Threat model**: what the proxy defends against — and deliberately
  does not — is written down in
  [docs/threat-model.md](docs/threat-model.md); the request-path gates,
  with each policy decision and enforcement point mapped to code, are
  diagrammed in
  [docs/security-dataflows.md](docs/security-dataflows.md).
- **Hardened local endpoints**: the dashboard and ops endpoints carry a
  strict `Content-Security-Policy` plus
  `X-Frame-Options`/`nosniff`/`Referrer-Policy` headers, on top of the
  Host/Origin/CSRF gates on the mutating endpoints.
- **Reporting**: use GitHub private vulnerability reporting — see
  [docs/SECURITY.md](docs/SECURITY.md).
- **Supply chain**: release artifacts carry Sigstore provenance
  (`gh attestation verify <file> --repo asanderson/llm-redact`) and are
  byte-reproducible from source.
- **Measured, not just tested**: mutation testing over the load-bearing
  core gates CI on a kill-or-justify contract, and "never a wrong
  value" is one falsifiable stateful property — method and results in
  [docs/assurance.md](docs/assurance.md).

## Tech stack

The runtime is deliberately **three packages** — the entire audit
surface a security tool asks you to trust:

- **httpx** — the upstream HTTP client: async, streaming bodies, and a
  transport seam that lets the whole integration suite run in-process
  against fake upstreams.
- **starlette** — the ASGI layer (routing, WebSockets, streaming
  responses) *without* FastAPI's validation machinery: the proxy must
  forward unknown JSON fields verbatim, never validate or reshape them,
  which is why pydantic is banned from the request path.
- **uvicorn** — the ASGI server; its `loop="auto"` and WebSocket
  protocol auto-selection let the `perf` (uvloop) and `realtime`
  (websockets) extras activate with zero serve-code changes.

Everything else is stdlib or an opt-in extra: the NER backends (spaCy,
GLiNER, Presidio, Stanza, Hugging Face `transformers`), `cryptography`
for vault encryption at rest, the RDBMS vault drivers (psycopg,
PyMySQL, oracledb), `keyring`, `uvloop`, `websockets`, and the
OpenTelemetry SDK. Configuration and CLI ride the standard library
(`tomllib`, `argparse`, dataclasses); Prometheus metrics text and the
S3/Azure audit-sink signers are hand-rolled on stdlib `hmac` so no
cloud SDK ever enters the tree.

The full inventory — every extra, the runtime closure, the dev
toolchain, and how to verify the machine-readable CycloneDX SBOM
attached to each release — is [docs/SBOM.md](docs/SBOM.md) (pinned to
`pyproject.toml` by test); the why-chosen record per package is
[docs/dependencies.md](docs/dependencies.md).

## Stability

llm-redact was developed privately before its public debut; v1.0.0 is
the first public release and the public history starts there. As of
1.0.0 the token format, config keys, CLI, local `/__llm-redact`
API, proxying behavior contracts, and the three-runtime-dependency
ceiling are stable surfaces under semantic versioning — the policy,
including what MINOR releases may add and how deprecations work, is
[docs/versioning.md](docs/versioning.md).

## Development

Contribution mechanics — the gates, the testing conventions, and how to
add a detection rule — are in [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

```bash
uv run pytest                                  # tests
uv run pytest tests/test_rehydrate.py -x      # one file
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
