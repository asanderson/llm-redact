# The local ops surface: status, metrics, events, preview

Reserved `/__llm-redact/*` paths are answered locally, never forwarded,
and carry metadata only — never redacted values. The namespace is
GET-only except the guarded session-prune and user invite/revoke POSTs,
and every reply is hardened with a strict CSP and framing/sniffing
headers.

- **Browser dashboard (llm-redact-pro)**: the web page at
  <http://127.0.0.1:8787/__llm-redact/> — live status pills, detection
  and restore totals, upstreams, the recent-request table, vault
  sessions, the configuration editor (`GET/POST /__llm-redact/config`)
  and the redaction-preview card (`POST /__llm-redact/preview`) — is
  part of the separately-installed **llm-redact-pro** package (Pro
  license tier; see [editions.md](editions.md) and the llm-redact-pro
  dashboard guide; the [architecture diagram](diagrams/architecture.png)
  draws it in a dashed Pro box). Without it, those three paths answer a
  local 404 that names the package (or, with the package installed but no Pro
  key, says a key is needed) — still answered before routing, never
  forwarded. Everything the page reads is served by the free core
  below, and every workflow it offers has a free CLI or agent-plugin
  twin: `llm-redact status`, `llm-redact preview`, `llm-redact config
  show`, and the guarded `/llm-redact:config-edit` plugin command
  (effective-config read → TOML edit → `serve --check` gate → SIGHUP →
  posture read-back).
- **Live request feed**: `GET /__llm-redact/recent` returns the last
  requests (metadata-only rows, newest first) and
  `GET /__llm-redact/events` streams the same rows as server-sent events
  as they happen (15 s keepalives; a slow consumer's events are dropped,
  so poll `/recent` as the authoritative fallback). Both are
  Host-gated against DNS rebinding.
- **Status**: `llm-redact status` (or `GET /__llm-redact/status`) reports the
  live totals as JSON — including the `routing` block (per-upstream
  health, spend against budget, re-issues) when the llm-redact-pro routing
  layer is enabled (`{"enabled": false}` otherwise).
- **Routing**: with the llm-redact-pro routing layer enabled,
  `llm-redact status` prints per-upstream lines — name, protocol,
  credential mode (never the key), state (`healthy` / `cooldown` /
  `budget_exhausted`), spend against budget (the llm-redact-pro routing
  guide); the `/recent` JSON rows carry each request's `route` fields.
- **Metrics**: `GET /__llm-redact/metrics` exposes Prometheus text format
  (always on, in-memory): `llm_redact_requests_total{provider,status}`, a
  request-duration histogram labeled by `provider` and `streamed` (so
  per-provider p95 and streaming-vs-non-streaming latency are separable),
  detections/restores by type, vault entry/session gauges, start time and
  uptime. Scrape config, alert rules, and the Grafana dashboard are in
  [observability.md](observability.md).
- **Health probes**: `GET /__llm-redact/healthz` is a DB-free liveness
  check (`{"status": "ok"}`) and `/readyz` reports readiness (version +
  whether the realtime extra is available). Orchestrators should probe
  these rather than `/status`, which reads the vault on every call; the
  container and compose healthchecks already do.
- **Audit log** (`[audit] enabled = true`), its tamper-evident chain, and the
  off-machine **object-store sinks** (S3-compatible / Azure Blob, with
  client-side batch encryption) are **Pro** features of the
  `llm-redact-pro` package (**coming soon**) — a local metadata-only
  history (types, counts, paths, durations — never values).
- **Redaction preview / dry-run**: run `llm-redact preview --text "…"`
  (or pipe to stdin) to see what the current config *would* redact, warn
  on, or block — entirely local: no request is sent upstream and nothing
  is written to the vault, metrics, or audit log. Iterate on rules,
  allowlists, modes, and language scope offline. (The llm-redact-pro
  dashboard's preview card runs the same dry run against the running
  proxy's live config.)
- **Coverage posture, surfaced loudly**: `llm-redact status` and
  `llm-redact doctor` both report every configured opt-out that lets
  traffic through unredacted — warn-mode types, providers with detection
  off, MCP exempt servers, language-scoped-out national-id rules — and
  say so plainly when nothing is opted out. Protection is never quietly
  reduced.
- **Agent plugins**: the status, recent-request, preview and
  config-editing workflows are available as slash commands inside
  Claude Code, Codex, OpenCode, and Cursor —
  `/llm-redact:status`, `/llm-redact:recent`, `/llm-redact:preview`, a
  guarded `/llm-redact:config-edit` (effective-config read → TOML edit →
  `serve --check` gate → SIGHUP → posture read-back), `/llm-redact:routes`
  and `/llm-redact:spend`, and more. Claude
  Code installs the repo as a plugin marketplace
  (`/plugin marketplace add asanderson/llm-redact`); every tool can also
  use `llm-redact plugin install claude|codex|opencode|cursor`. Command
  bodies open with a proxy-presence guard: a missing `llm-redact` CLI
  stops the command and asks your approval before anything is installed. `lookup` is
  deliberately not a command — an agent that read a secret value would
  send it upstream. See [plugins.md](plugins.md).

The **user guide** — the same surface written for the person *using*
the proxy rather than deploying it — ships in the package and is served
by every running proxy at `/__llm-redact/guide` (also `llm-redact
guide`, or `/llm-redact-guide` from an agent with the plugin installed).
