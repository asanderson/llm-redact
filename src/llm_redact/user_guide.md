# llm-redact user guide

llm-redact is a local proxy that sits between your agentic tools (Claude
Code, Codex, Cursor, OpenCode, and anything else that speaks a provider
API) and the LLM providers. On the way out it replaces private values —
emails, keys, national IDs, custom patterns — with placeholder tokens like
`«EMAIL_001»`; on the way back it restores the real values. The mapping
lives in a local vault; the provider only ever sees placeholders.

This guide covers the **command line** and the **agent plugin
commands**, which are part of the free core, and the **web dashboard**
(with its config editor and redaction preview), which is part of the
separately installed llm-redact-pro package. The deeper design documents
live in the repository's `docs/` directory.

## Quick start

- `llm-redact init` — interactive setup: writes a starter config and
  prints the environment exports for your tools.
- `llm-redact run -- claude` — run a tool through the proxy (reuses a
  running proxy, or starts an ephemeral one for the command's lifetime).
- `llm-redact serve` — run the proxy in the foreground.
- `llm-redact service install` — run it at login (launchd/systemd; on
  Windows this prints a Task Scheduler command to run yourself).
- `llm-redact doctor` — read-only diagnostics: config, extras,
  permissions, ports, and every coverage opt-out.
- `llm-redact status` — the running proxy's counters and coverage
  posture (the JSON behind it is `GET /__llm-redact/status`; Prometheus
  metrics are at `/__llm-redact/metrics`).
- `llm-redact preview` — run text through the configured detectors
  locally and see what would be redacted, warned on, or blocked.

Already running a proxy elsewhere (a team server, another machine)?
Point everything at it with `LLM_REDACT_PROXY_URL` — `run`, `status`, and
the plugin commands all honor it, and
`llm-redact plugin install <tool> --proxy-url URL` sets it up for you.
Plain http is loopback-only; a remote proxy must be https.

## The web dashboard (llm-redact-pro)

The browser dashboard (config editor, redaction preview) is part of
llm-redact-pro. With it installed and a Pro license key, open
`http://127.0.0.1:8787/__llm-redact/` (your host/port may differ).
Without it, that URL answers a short JSON error naming the package and
pointing at the free surfaces: `llm-redact status`,
`/__llm-redact/status`, `/__llm-redact/metrics`, and
`llm-redact preview`. Everything on the page is served by the proxy
itself — self-contained, no external resources, no data leaves your
machine.

A sidebar switches between views (each deep-linkable, e.g.
`/__llm-redact/#config`); the top bar shows whether the live feed is
connected.

- **Overview** — the protection-posture banner (green when every
  enabled rule redacts; an amber list of every coverage opt-out
  otherwise), counters for values redacted, restored, warned (and
  forwarded) and blocked, detections by placeholder type, the runtime
  panel (version, uptime, vault, session mode, license tier — license
  warnings such as an invalid key or expiry grace appear loudly, never
  silently), and each provider's upstream with its state and upstream
  errors.
- **Traffic** — the last 200 requests (path, provider, status,
  duration, detection counts — never values), updated live over a
  server-sent event stream, with a text filter and provider / kind
  picklists. This works without the audit log.
- **Vault sessions** — vault sessions with entry counts and idle times;
  whole idle sessions can be pruned from here (sqlite backend). The
  active session is never pruned.
- **Named users** (Pro+) — invite teammates by email, see seat usage
  against the license cap, revoke access. Verification codes are
  delivered by your `[email]` SMTP settings or shown once for manual
  delivery. Per-user keys are shown once, at verification, to the
  invitee — never stored, never re-displayed.
- **Redaction preview** — paste text and see exactly what the live
  detector set would redact, without sending anything upstream or
  writing anything to the vault. Warn-mode matches are shown unmasked,
  because that is honestly what would be forwarded.
- **Configuration** — the config editor (below), with picklists and
  type-ahead: a searchable, filterable rule grid with per-rule mode
  picklists, a language picker, placeholder-type suggestions, NER
  backend chips and entity suggestions, and a save bar that tracks
  unsaved changes.
- **System** — the restart-required settings and the local endpoints.
- **Routing** — whether rule-based upstream routing is enabled and,
  when it is, one row per upstream: protocol, credential mode (never
  the key itself), state (healthy, in cooldown, or budget exhausted),
  and spend against its monthly budget.

## The config editor (llm-redact-pro)

The dashboard's Configuration view edits the proxy's TOML config file
with guardrails:

- It always merges over **file truth** — environment-variable overrides
  are never baked into the file.
- Every submit is validated by a full parse AND a dry-run detector
  build before anything is written; a rejected edit changes nothing.
- The write is atomic, keeps one `.bak`, and hot-applies without a
  restart. Comments in the file are not preserved.
- Sections that cannot hot-apply (host/port, vault, audit, TLS, log,
  OTel, users, email) are shown read-only; change those in the file and
  restart (or send SIGHUP on macOS/Linux — on Windows, where SIGHUP
  does not exist, the editor and restarts are the reload paths).
- The routing sections (`[upstreams]`, `[routing]`, `[prices]`) are
  file-only: the editor preserves them untouched and refuses an edit
  that names them. Edit the file, run `llm-redact serve --check`, and
  send SIGHUP — they hot-apply without a restart.

The free-core equivalents: `llm-redact config show` prints the
effective config; `llm-redact serve --check` runs the full startup build
without binding a socket — the deploy gate; edit the file and send
SIGHUP to apply; and the **config-edit** plugin command walks that same
flow with guardrails.

## Agent plugin commands

`llm-redact plugin install claude|codex|opencode|cursor` installs slash
commands into your agent tool (Claude Code repo checkouts can instead use
`/plugin marketplace add asanderson/llm-redact`). Invocation forms:
Claude Code and Cursor `/llm-redact-<name>`, Codex
`/prompts:llm-redact-<name>`, OpenCode `/llm-redact-<name>`.

- **status** — proxy health, tier, counters, and the honesty posture
  (anything configured to reduce protection is called out).
- **recent** — the recent-requests table, summarized by the agent.
- **sessions** — vault sessions and what pruning would remove.
- **config-show** — the effective configuration as TOML.
- **config-edit** — guided config-file editing with guardrails
  (validate with `serve --check` before a SIGHUP reload; only on your
  ask).
- **preview** — run text through the live detectors locally.
- **doctor** — the diagnostics report, interpreted.
- **audit** — audit log status, tamper-chain verification, and whether
  the zero-loss `required` mode is on.
- **users** — seat usage and invitations (never prints per-user keys).
- **routes** — the routing rules table, or a dry-run of which rule,
  upstream, and fallback chain a request would take
  (`routes test --protocol anthropic --model claude-* --auth oauth`);
  no upstream is contacted — the live `state` line comes from the
  running proxy's own listener, or reads `not probed`.
- **spend** — per-upstream tokens and USD for the month, how much came
  from fallback re-issues, and the remaining budget. On a subscription
  (passthrough) lane the USD is a list-price equivalent, not a bill —
  the token counts are the real number.
- **guide** — displays this guide.

Every command begins by checking that the `llm-redact` CLI is present
and will never install anything without asking you first. The `lookup`
command (resolving a placeholder back to its secret value) is
deliberately NOT a plugin command: an agent that read a secret would
send it upstream on its next turn.

## Routing, fallback and budgets

Routing, fallback and budgets are supplied by the llm-redact-pro
package; without it the proxy forwards each protocol to its one
provider upstream, `status` shows `routing: disabled`, and a config
with `[routing] enabled = true` refuses to start naming the package.
With it, a `[routing]` section in the config file lets one proxy serve
several upstreams per protocol — your subscription lane forwarded
byte-exact, your own API keys, a local Ollama — chosen by first-match
rules, with fallback chains on configurable statuses, per-upstream
cooldowns, Anthropic plan-limit detection, and monthly budgets metered
from the providers' usage counts. A fallback can only ever reach your
own key or a local model, never a second subscription, and a
conversation that already carries signed thinking blocks is never
swapped mid-conversation (the response says so:
`x-llm-redact-reissue: skipped; reason=stateful`). Every decision is
visible in `llm-redact status`, the `route` field of the
`/__llm-redact/recent` rows shown by the **recent** command (rule,
upstream, hops, class — the llm-redact-pro dashboard's recent table does
not render them), and the **routes** / **spend** commands. The reference, the
policy it respects, and a recommended single-user config are in the
llm-redact-pro package's `docs/routing.md`.

## Honesty surfaces

Anything that reduces coverage is surfaced, never silent. Configured
opt-outs — warn-mode rules (matches are observed and FORWARDED),
per-provider detection off, MCP server exemptions, language-scoped-out
national-ID rules, the remote-plaintext vault hatch — appear in
`/__llm-redact/status`, `llm-redact status`'s posture block, `doctor`,
and (with llm-redact-pro) the dashboard. Runtime state — audit-backup
upload failures, routing upstreams in cooldown or over budget — appears
in `/__llm-redact/status`, `llm-redact status`, and the dashboard;
`doctor` checks the routing config and its credentials, never the
running proxy's cooldown or budget state.

## Going deeper

In the repository: `README.md` (overview), `docs/editions.md` (editions
and the tier matrix), `docs/quickstart.md`, `docs/troubleshooting.md`,
`docs/deployment.md`,
`docs/threat-model.md`, and `docs/plugins.md`. Paid-feature guides (server
database vaults, named users, the paid deployment surface, and the
routing guide `docs/routing.md`) and the full licensing reference live
in the `llm-redact-pro` repo's `docs/`.
