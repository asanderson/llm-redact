# Changelog

All notable changes to llm-redact are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

Convention: entries land in `[Unreleased]` with the change that introduces
them. A release moves `[Unreleased]` into a dated version section, bumps
`__version__` in `src/llm_redact/__init__.py` (the single source of truth),
and tags `vX.Y.Z`.

## [Unreleased]

## [1.1.0] - 2026-09-21

### Added

- **Rule-based upstream routing, quota-aware fallback, and monthly
  budgets** (`docs/routing.md`), a routing layer behind the redaction
  pipeline that is OFF by default and byte-identical to 1.0 when off:
  - `[upstreams.NAME]` — named destinations, one protocol each
    (`anthropic` / `openai` / `gemini` / `ollama`), with a credential
    mode: `passthrough` (the client's auth headers and `anthropic-beta`
    forwarded byte-exact), `env:VAR` (inbound credentials stripped, the
    OAuth marker removed from `anthropic-beta`, the proxy-held key
    injected as `x-api-key` / `authorization: Bearer` / `x-goog-api-key`
    by protocol; resolved by `serve`, `serve --check` and `doctor`, never
    at parse), or `none`. Per-upstream `cost = "zero"`, `count_tokens`
    (`false` answers `/v1/messages/count_tokens` 404 locally — Ollama),
    `cooldown_seconds`, `extra_headers`, `body_defaults` (absent
    top-level keys only, never on passthrough), `monthly_budget_usd` /
    `monthly_budget_tokens`. With a `[routing]` table present the legacy
    `[providers.anthropic|openai|gemini|ollama]` sections auto-register
    as passthrough upstreams of the same name (an explicit
    `[upstreams.NAME]` wins).
  - `[routing]` — `enabled`, a fail-closed `default_upstream` (string, or
    a table keyed by protocol; a protocol with neither a rule nor a
    default answers a proxy-generated 502, never a guess), `max_hops`,
    `request_deadline_seconds`, `plan_limit_detection`,
    `plan_limit_headers`, `oauth_beta_marker`,
    `throttle_retry_max_seconds`, `budget_reset_day`, `debug_headers`,
    `expose_models`, `model_catalog`; and ordered `[[routing.rule]]`
    matchers (protocol, model globs, headers, path, `auth` ∈ oauth /
    gateway-key / none / any) with `model_rewrite` (the original id
    restored in JSON and SSE/NDJSON `model` / `message.model` /
    `response.model` fields; on protocol `gemini` the model is read from
    and rewritten in the `/models/{m}:verb` path segment — the body never
    gains one — and `modelVersion` is restored, buffered array form
    included), `on_status` chains (exact status, `4xx`/`5xx`,
    `plan_limit_429`/`throttle_429`, `retry-same`; a chain may not name
    the rule's own upstream or a member twice), `reissue_policy` (gates
    `on_status` and `on_budget_exhausted` alike; `never` makes a chain
    dead configuration, warned at parse), and `on_budget_exhausted` (the
    refused primary is hop 1, the first member hop 2). Every numeric
    routing/price key (budgets, `cooldown_seconds`,
    `request_deadline_seconds`, `throttle_retry_max_seconds`, override
    rates) must be finite: `nan`/`inf` (native TOML) and an integer
    beyond `float` range are `serve --check` errors (`must be a finite
    number (nan/inf are not accepted)`). Rules are
    selected before redaction so the first upstream's
    `inject_system_note` governs the prepared body; id-only follow-ups
    (`GET /v1/responses/{id}`, conversation/file reads, batch polls)
    carry no model and take the protocol default unless a path-matched
    rule claims them.
  - Fallback semantics: re-issue only before the first byte reaches the
    client — buffered bodies are read inside the hop loop, so a mid-body
    transport drop on a non-streaming response is chained too; only
    mid-**stream** faults propagate (`class=stream_error`); `hops` = 1 +
    re-issues (`retry-same` is not a hop, but does count as a request
    attempt in `/status`); a per-upstream cooldown (`min(3600,
    max(cooldown_seconds, retry-after))`) on every status that resolves
    through the rule's `on_status` except throttles — once a chain is
    active any non-2xx from a member continues to the next member,
    cooldown only for listed statuses; `retry-same` waits
    `max(retry-after, 2 s)` once, or returns the 429 unchanged beyond
    `throttle_retry_max_seconds`; `request_deadline_seconds` also caps
    each hop's own connect/read via a per-hop httpx timeout derived from
    the remaining budget (never above the client's 10 s / 600 s, so the
    default changes nothing): a hop that accepts and then hangs fails
    inside the deadline as `class=transport`, and a delivered stream
    silent for longer than the remaining budget is cut off
    (`class=stream_error`); transport faults — connect/send errors,
    buffered read drops, a hop timed out by the remaining deadline and an
    `env:` variable that vanished at runtime — classify as `502` for
    chain lookup and keep `class=transport` even when `reissue_policy`
    stops the chain (the class reports what happened, not the table
    key); Anthropic plan-limit 429s are
    told apart from throttles by the unified rate-limit headers
    (`anthropic-ratelimit-unified-status` / `-5h-status` / `-7d-status`
    = `rejected`, overridable). The **statelessness guard** never
    re-issues a request carrying signed `thinking` /
    `redacted_thinking` blocks under `stateless-only`, answering
    `x-llm-redact-reissue: skipped; reason=stateful` instead
    (`reason=no-candidate` when the chain had no eligible member).
    Re-issued responses carry `x-llm-redact-upstream` and
    `x-llm-redact-hops` (hop-1 responses too with `debug_headers`).
  - Policy envelope encoded as invariants, not advice: a chain may never
    contain a passthrough upstream (no subscription pooling or credential
    sharing), an `env:` upstream never sees inbound credentials, a
    passthrough upstream never sees a server-held key, and
    `inject_system_note` / budgets are config errors on passthrough
    upstreams — all refused by `serve --check` and re-asserted at
    runtime.
  - Budgets and pricing: usage parsed from Anthropic, OpenAI-compatible
    (with `stream_options.include_usage` injected into Chat Completions
    streams on `env:`/`none` upstreams when absent; Responses streams
    report usage on `response.completed`), Ollama native and Gemini responses; spend
    stored in a `spend` table inside the sqlite vault DB (in-process on
    the memory backend and on every RDBMS vault backend), attributed to
    the producing upstream with its hop number — passthrough upstreams
    included, whose USD is a list-price equivalent (tokens are the
    honest number); the spend INSERT runs synchronously in the response
    finalizer with a 250 ms lock wait (`spend.REQUEST_PATH_BUSY_TIMEOUT_MS`;
    the offline `llm-redact spend` waits 5 s) — a lock held by another
    process drops the row from the file (`spend store write failed for
    upstream NAME (Type); the row is counted in memory only`, still
    counted toward the period in memory) rather than stall the request;
    a sqlite vault whose `spend` table cannot be opened
    is a `ConfigError` (`spend table in the vault database … could not
    be opened`) at startup, `serve --check` and SIGHUP, and the config
    editor answers 500 `written to … but not applied … reload (SIGHUP)`
    when the same happens after its write; a vendored `prices.json`
    (USD per 1M tokens) with `[prices] table = "<path>"` replacement
    and `[prices.override."<model>"]`, rebuilt on every reload; an
    exhausted upstream answers 402 (`llm-redact routing: upstream NAME
    budget exhausted for this period`) and drops out of chains until
    `budget_reset_day`; unknown models cost `null` (tokens only) —
    `doctor` WARNs on the statically configured ids, `/status`
    `unpriced_models` lists the ids seen at runtime, capped at 100
    id-shaped entries (≤ 200 printable chars, no whitespace —
    `pricing.looks_like_model_id`; the `model` field is unscanned client
    text) with every other observation counted in
    `unpriced_models_dropped` (carried across SIGHUP, printed by `status`
    and the dashboard as `(+N more not listed)`).
  - Observability: the request log line gains
    `rule= upstream= hops= auth= class= reissue=` (`rule=-` for a
    default-upstream request, `class=ok reissue=yes` for a successful
    re-issue); `/recent` and
    `/events` rows gain a `route` object; `/__llm-redact/status` gains a
    `routing` block (per-upstream state healthy / cooldown /
    budget_exhausted, spend against budget, re-issues in the last hour,
    unpriced models, warnings); metrics
    `llm_redact_routed_requests_total{upstream,rule}` and
    `llm_redact_reissues_total{from_upstream,to_upstream}`; a dashboard
    routing pill and per-upstream table.
  - CLI: `llm-redact routes list [--json]` and `llm-redact routes test
    --protocol P [--model M] [--header NAME=VALUE …] [--path PATH]
    [--auth …] [--json]` (no upstream is contacted and no credential
    resolved; the live state/cooldown annotations come from a
    best-effort plain-http `GET /__llm-redact/status` of the config
    listener — `[tls]` and `LLM_REDACT_PROXY_URL` proxies show `not
    probed`; a Gemini model is derived from `--path`; the `count_tokens
    … the proxy answers 404` note fires only for Anthropic's canonical
    `/v1/messages/count_tokens` on a `count_tokens = false` upstream —
    `routing.is_count_tokens_path`, the proxy's own gate); `llm-redact spend
    [--month YYYY-MM] [--json]`, which opens the vault **read-only**
    (never creates the table or WAL files; `no spend recorded yet`
    until the proxy writes a row); `status`
    prints one routing line per upstream and posture lines for
    cooldown / budget-exhausted upstreams, unpriced models and routing
    warnings; `doctor` validates the routing config, resolves every
    `env:` credential (FAIL names the VAR only), WARNs on a non-zero-cost
    default, unpriced models and routing on the memory vault, and probes
    each **referenced** upstream base URL (`HEAD` then `GET`, 3 s, no
    credentials, printed as `scheme://host[:port]` only) unless
    `--offline` — explicit `[upstreams.*]` plus legacy auto-registrations
    a rule, chain or default names.
  - Optional model discovery: `routing.expose_models = true` answers
    `GET /v1/models` locally in the Anthropic or OpenAI shape from
    `model_catalog` plus the literal model names in rules (Claude Code
    keeps only ids containing `claude` or `anthropic`); the local answer
    sits after the disabled-provider 502 (the path infers `openai`, so
    `[providers.openai] enabled = false` refuses the Anthropic-shaped
    call too) and the named-user 403, like every proxy-generated reply
    for a real API path.
  - `scripts/fake_upstream.py` gained programmable scenarios (status,
    plan-limit headers, `retry-after`, usage blocks, mid-stream aborts,
    model echo, fail-then-succeed with a configurable `fail_status` /
    `--fail-status`) and serves `/v1/chat/completions` and
    `/v1/embeddings` alongside `/v1/messages` so the routing
    walkthroughs run in-process.
  - Plugin commands `/llm-redact:routes` (wrapping `routes list|test`)
    and `/llm-redact:spend`; `/llm-redact:status` and `/llm-redact:recent`
    report the routing fields and posture lines. Twelve commands now.

### Changed

- `[upstreams]`, `[routing]` and `[prices]` hot-reload on SIGHUP (and via
  `apply_config`), but are deliberately NOT editable in the dashboard
  config editor: the editor's merge preserves them from file truth, and
  a `POST /__llm-redact/config` naming them is a 400 ("edit the file and
  reload"). The restart-only set is unchanged.
- `AnthropicAdapter.error_body` maps 402 → `billing_error` and 404 →
  `not_found_error` for the proxy-generated routing replies (a 429 →
  `rate_limit_error` entry exists but no proxy-generated 429 does —
  every 429 a client sees is the upstream's own body); the OpenAI error
  shape is unchanged.
- The in-process latency benchmark gained `route_select_10_rules`
  (ceiling 0.2 ms) and a routing-enabled proxy-overhead macro.
- Realtime WebSocket connections, Azure, Vertex, Claude-on-Vertex,
  Bedrock, Cohere and `[providers.custom.*]` are never routed — they keep
  the legacy `upstream_base_url` path unchanged (documented).

### Docs

- New `docs/routing.md`: concepts, the Anthropic policy envelope, the
  full `[upstreams]` / `[routing]` / `[[routing.rule]]` / `[prices]`
  reference with every invariant `serve --check` enforces, the
  fallback / cooldown / plan-limit / statelessness semantics, budgets
  and the price table, observability, the recommended single-user
  config (PowerShell + docker notes), troubleshooting keyed by the
  exact error strings, and limitations. Every fenced TOML block in it
  and the commented routing block in `config.example.toml` are parsed
  by `tests/test_routing_docs.py`.
- `config.example.toml` gained the commented routing block (between
  `## ROUTING-EXAMPLE-BEGIN` / `## ROUTING-EXAMPLE-END`: `# ` lines are
  config, `## ` lines are notes), disabled by default.
- README (Operations § Routing, fallback and budgets; Plugins), docs
  index, providers, observability (metrics rows), troubleshooting,
  api-coverage (`GET /v1/models` local answer), dashboard, plugins, and
  the packaged user guide updated.

## [1.0.3] - 2026-07-29

### Added

- Plugin-first onboarding: every plugin command's proxy-presence guard
  now offers three PINNED setup tiers when the `llm-redact` CLI is
  missing — an ephemeral `uvx --from llm-redact-proxy llm-redact serve`
  run (nothing lands on PATH), a one-approval install
  (`uv tool install` / `pipx install` + `init --yes` +
  `service install`), or pointing `LLM_REDACT_PROXY_URL` at an existing
  proxy. The package name `llm-redact-proxy` is pinned verbatim in both
  directions by test (agent-improvised install commands are a
  supply-chain vector).
- Per-tool routing honesty in the guards: each tool's rendering states
  ITS routing truth — Claude Code checks `ANTHROPIC_BASE_URL` and names
  the `llm-redact run -- claude` relaunch, Codex/OpenCode check
  `OPENAI_BASE_URL` with their `run --` launch forms, and Cursor states
  plainly that its traffic is NOT protected outside custom-API-key mode
  with the base-URL override. A live proxy is never presented as a
  routed session.
- Claude Code marketplace plugin: `bin/llm-redact-posture` (read-only
  POSIX sh posture check, on the Bash tool's PATH) plus a `SessionStart`
  hook running it with `--quiet-ok` — reports CLI-missing / proxy-down /
  session-unrouted loudly, stays silent when healthy, never installs
  anything, and echoes URLs as scheme://host:port only. Behavioral
  tests drive all four states against a live loopback server
  (`tests/test_plugin_posture.py`).
- `opencode` is now a `TOOL_EXPORTS` entry: `llm-redact init --tools
  opencode` and `llm-redact run -- opencode` route OpenCode via
  `OPENAI_BASE_URL`.
- docs/plugins.md: "Plugin-first onboarding" section (the three tiers,
  the per-tool routing-honesty table, the SessionStart posture check,
  and the OpenCode JS plugin as a documented future option).

## [1.0.2] - 2026-07-19

### Added

- Privacy policy (`docs/privacy.md`): no telemetry or phone-home; what
  stays on the machine and what leaves it.

### Fixed

- Plugin command frontmatter: descriptions containing ": " rendered as
  invalid YAML, so Claude Code silently dropped ALL frontmatter fields
  (description, allowed-tools, disable-model-invocation) for the
  `recent`, `status`, and `config-edit` commands, and
  `claude plugin validate` rejected the plugin. The renderer now quotes
  YAML-unsafe values; the plugin passes official validation.

## [1.0.1] - 2026-07-17

### Added

- Interactive install script `scripts/install.sh`: detects the tools
  available on the machine (uv, pipx, pip, Homebrew, docker, podman),
  prompts for the preferred install method (`--method NAME` for
  non-interactive use), and prints every command before running it. The
  container methods pull `ghcr.io/asanderson/llm-redact:latest` and
  offer a loopback-published run.
- README Install section now shows the prebuilt-container path
  explicitly — `docker pull` / `podman pull` from
  `ghcr.io/asanderson/llm-redact` with the loopback publish spec — and
  links the install script.

## [1.0.0] - 2026-07-17

Initial public release. llm-redact was developed privately before this
debut; the public history starts here, at v1.0.0.

### Added

- **The transparent redaction proxy.** A local proxy sits between an
  agentic tool and its LLM provider: outbound requests are scanned for
  private values, each detected value is replaced with a deterministic
  `«TYPE_NNN»` placeholder whose mapping lives only in a local vault, and
  inbound responses — including streamed ones, even when a token splits
  across chunk boundaries — have the placeholders restored. Streaming is
  handled at the byte level across SSE, NDJSON, AWS eventstream, and
  WebSocket framings; unrecognized traffic passes through verbatim
  (never break the tool), and upstream faults fail closed with
  provider-shaped errors.
- **Detection**: 80+ built-in rules — vendor API keys and tokens
  (prefix-anchored), credit cards/IBANs/phones, checksum-validated
  national identifiers across 20+ countries, PGP/private-key armor, and
  checksum-vetoed crypto wallet addresses — plus user deny strings
  (always-win, tier 0), per-type allowlists, per-rule redact/warn/block
  modes, language scoping, custom rules with named validators, and
  optional person-name NER behind five interchangeable backends (spaCy,
  GLiNER, Presidio, Stanza, Hugging Face). A recall==1.0 benchmark gate
  and a real-world false-positive corpus pin detection quality in CI.
- **Providers**: Anthropic (Messages + Batches), OpenAI (Chat,
  Responses, Conversations, Files/Batches, Realtime WS, images/audio/
  video), Google Gemini (+ Live WS, context caching, batch), AWS
  Bedrock, Azure OpenAI, GCP Vertex (Gemini and Claude), Cohere, Ollama
  native, and any number of named custom OpenAI-compatible upstreams
  (vLLM, LM Studio, OpenRouter, …). Embeddings and file uploads are
  redact-only; the endpoint matrix is pinned by test in both directions.
- **Vault**: deterministic placeholder issuance — the same value always
  gets the same token within a session — with in-memory and persistent
  SQLite backends, session lifecycle CLI, and strict never-restore-
  across-sessions isolation.
- **Ops surface**: a self-contained local dashboard with a guarded
  config editor (validate → atomic write → SIGHUP hot reload), redaction
  preview dry-run, Prometheus metrics, health endpoints, a live SSE
  event feed, structured JSON logs, `doctor` diagnostics, and an honest
  posture block that surfaces every protection opt-out — warn-mode
  rules, disabled providers, MCP exemptions, language scoping — loudly.
- **Agent plugins** for Claude Code, Codex, OpenCode, and Cursor: ten
  slash commands mirroring the dashboard and config editor, with a
  proxy-presence guard and real-output screenshots in the docs.
- **Deployment**: hardened Dockerfile (multi-arch), k8s sidecar
  manifest, a Helm chart with sidecar and standalone modes plus optional
  HPA autoscaling, systemd/launchd service units, shell completions, an
  init wizard, and an env-injecting `run` wrapper. Non-loopback binds
  are fail-closed behind full mutual TLS.
- **Assurance**: split-at-every-offset streaming equivalence sweeps,
  property-based tests, differential codec fuzzing, a red-team boundary
  suite, mutation-testing gates, a complexity-coverage gate (every
  branching function executed by the suite), reproducible builds, SBOMs,
  and signed release artifacts.

### Licensing

- The core is **free and open-source software under GNU AGPL-3.0-only**,
  with nothing gated: no license keys, tiers, or seat caps anywhere in
  this repository. Contributions are accepted under `docs/CLA.md`.
- The separately-installed proprietary `llm-redact-pro` package
  (**coming soon**) will supply additional operational subsystems —
  server RDBMS vaults, vault encryption at rest, the audit log and its
  object-store sinks, OpenTelemetry export, per-conversation sessions,
  and named users. Configuring one of those without the package fails
  closed naming the feature and the package — never a silent downgrade.
