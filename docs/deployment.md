# Deployment guide

How to run llm-redact in production. This ties the operational surfaces —
bind policy, vault lifecycle, health probes, and observability — to the
single security goal in [threat-model.md](threat-model.md): *private values
must not reach the provider, and the placeholder↔value mapping must never
leave the machine.* Everything below is subordinate to that.

The proxy is designed to run **next to the tool it serves, as the same
user, on loopback**. That is not a limitation to work around — it is the
trust boundary the design depends on (see the threat model's assumptions).
This guide covers the FOSS core's whole deployment surface — including
team serving over mutual TLS and Kubernetes (the sidecar manifest, the
Helm chart, and HPA autoscaling), which are ungated parts of this
repository. The paid operational subsystems — at-rest vault encryption
and key rotation, the server RDBMS vaults, the audit log and its
off-machine sinks, and OpenTelemetry export — ship in the
`llm-redact-pro` package (**coming soon**; its operator guide ships with
it).

## Choose a bind, and know what it costs

`serve` runs `validate_bind_security` before it opens the socket. The rule
is fail-closed:

| Host | Requirement | Why |
| --- | --- | --- |
| `127.0.0.1` / `::1` (default) | none | A network client of the proxy can read *rehydrated* secrets; loopback keeps that client on the same machine, where a same-UID attacker already has the vault. |
| any non-loopback host | `[tls]` **certfile + keyfile + client_ca** (full mutual TLS) | Every connecting client must present a certificate the CA signed. Without client auth, anyone who can route to the port can read restored secrets and edit detection config. |
| unresolvable hostname | treated as non-loopback | Fail closed rather than guess. |

`serve` refuses to start on a non-loopback host without the mTLS trio.
The single documented escape hatch is `LLM_REDACT_INSECURE_BIND=1`, and it
exists for exactly one confined case: a container that binds `0.0.0.0`
*inside its own network namespace* while the operator publishes it to
loopback only. Setting it anywhere else is on you.

**Default deployment (recommended): loopback.** Point the tool's base URL
at `http://127.0.0.1:8787` (or use `llm-redact run -- <tool>`, which
injects the right env vars). Nothing else to configure.

**Remote / shared deployment** — serving clients on other hosts — means
operating a client-certificate PKI: configure the full `[tls]` trio
(`certfile`, `keyfile`, `client_ca`; the bind policy refuses a
non-loopback host without it) and query the running proxy with
`llm-redact status --ca/--cert/--key`. This is part of the FOSS core.

## Containers

The published GHCR image binds `0.0.0.0` inside the container netns and
sets `LLM_REDACT_INSECURE_BIND=1` for that confined case only. **Always
publish it to loopback:**

```bash
docker run -d --name llm-redact \
  -p 127.0.0.1:8787:8787 \
  -v llm-redact-data:/data \
  -e ANTHROPIC_BASE_URL=… \
  ghcr.io/asanderson/llm-redact:latest
```

`-p 127.0.0.1:8787:8787` — never `-p 8787:8787`, which would expose the
proxy on every interface with client auth disabled. The image ships the
`perf` (uvloop) and `realtime` (WebSocket) extras, so it runs on uvloop
and can relay OpenAI Realtime / Gemini Live. `XDG_DATA_HOME=/data` holds
the vault and audit DB — mount a volume there for persistence. Released
images are multi-arch (amd64 + arm64).

### Health probes

Orchestrators should probe the DB-free liveness endpoint, not `/status`
(which reads the vault on every call):

- `GET /__llm-redact/healthz` → `{"status": "ok"}` — liveness.
- `GET /__llm-redact/readyz` → `{"status": "ready", "version": …,
  "realtime": <bool>}` — readiness, including whether the WebSocket extra
  is importable.

The container `HEALTHCHECK` and the compose healthcheck already probe
`/healthz`. Under Kubernetes, wire `healthz` to the liveness probe and
`readyz` to the readiness probe. These endpoints are unauthenticated by
design (metadata only, no secrets) but are still served with the reserved-
path security headers and are provably never forwarded upstream.

Kubernetes deployment is part of the FOSS core: the hardened sidecar
manifest lives at `deploy/k8s-sidecar.yaml` and the Helm chart (sidecar
and standalone modes, optional HPA autoscaling) at
`deploy/helm/llm-redact/` — its `NOTES.txt` and `values.yaml` document
the modes and guardrails.

## Service management (native installs)

`llm-redact service install` writes a per-user launchd (macOS) or systemd
(Linux) unit; `service status` / `uninstall` manage it. Let the platform
own restarts and log retention:

- **systemd**: `journalctl -u llm-redact` owns retention. The generated unit
  ships a sandbox (`NoNewPrivileges`, `ProtectSystem=strict`, empty
  `CapabilityBoundingSet`, a `@system-service` syscall filter, and
  `ReadWritePaths` scoped to the XDG data dir so the vault/audit still write).
  Review it with `service install --print-only`; heavy NER extras (torch) may
  need the syscall filter loosened.
- **containers**: cap the log driver (`--log-opt max-size=10m --log-opt
  max-file=3`), since the proxy does not rotate its own logs.

Config changes apply on **SIGHUP** without dropping in-flight requests
(`kill -HUP $(pgrep -f 'llm-redact serve')`, or `docker kill
--signal=HUP`). Detection rules, allowlists, NER, fuzzy rehydration, note
injection, `max_body_bytes`, and upstream URLs hot-reload; vault, audit,
host, port, log, TLS, OTel, users, and email changes warn "require restart"
and keep the old value. A broken config file is logged and ignored — the running config
stays live. There is deliberately no HTTP reload endpoint (it would be a
CSRF-reachable mutating endpoint on loopback).

The same guarded flow is available from inside an agent: the
`/llm-redact:config-edit` plugin command reads the effective config,
edits the file, gates on `serve --check`, reloads via SIGHUP, and reads
back the coverage posture — and `/llm-redact:doctor` runs the same
read-only preflight as the CLI ([plugins.md](plugins.md)).

## Vault lifecycle in production

The sqlite vault (`[vault] backend = "sqlite"`) is the one piece of state
whose corruption is unacceptable: a lost or reused token number would
silently rehydrate the *wrong* secret. Treat it accordingly.

- **Back it up consistently.** `llm-redact vault backup <dest>` takes a
  single-file snapshot through the SQLite online-backup API — it reads
  *through* the WAL, so it is safe against a running proxy and cannot tear
  a mapping the way `cp vault.db` can. The destination is created `0600`.
- **Verify integrity.** `llm-redact vault verify` is a read-only sweep:
  it checks that token numbers are dense (`1..N` per session/type — a gap
  is what would let a reissue collide) and, for encrypted vaults, that
  every ciphertext decrypts and every HMAC index matches its plaintext.
  It prints sessions/types/counts only, never a value, and exits non-zero
  on any failure. Run it after a restore or before a key rotation.
- **Bound growth.** `[vault] session_ttl_days = N` prunes whole sessions
  idle longer than N days via a background task (never the active
  session); `0` (default) disables it. For manual control,
  `llm-redact sessions prune --older-than 90d` deletes whole idle sessions
  (partial deletion could reuse a still-referenced number). The dashboard
  and `POST /__llm-redact/sessions/prune` do the same, safe against the
  live process.

At-rest **encryption** of the vault (`[vault] encryption = "fernet"`), **key
rotation** (`vault rotate-key`), and the **server RDBMS** backends are Pro
features of the `llm-redact-pro` package (**coming soon**; documented with
the package).

`llm-redact doctor` ties these together as a read-only preflight: it
checks config parse, the bind policy, proxy reachability and version skew,
vault/audit file permissions (`0600`/`0700`), missing extras, that the
fernet key actually **matches** the vault (not merely that it is set), and
loudly reports every coverage opt-out (warn-mode types, per-provider
detection off, MCP exempt servers, language-scoped-out rules). It exits
non-zero on any FAIL and never prints a value. Run it in a pre-deploy step.

## Observability

All telemetry is **metadata only** — types, counts, paths, durations;
never a value or a placeholder id. That contract holds across every sink.

- **Prometheus**: scrape `GET /__llm-redact/metrics`. Request-duration is
  labeled by `provider` and `streamed`, so per-provider p95 and
  stream-vs-non-stream latency are visible. Detection/rehydration counters
  are per type. `llm_redact_upstream_errors_total{provider}` counts transport
  faults the proxy failed closed as 502 (the `LlmRedactUpstreamErrors` alert
  fires on a sustained rate). Ready-to-use scrape config, alert rules, and an
  importable Grafana dashboard live in [`deploy/`](../deploy) — see
  [observability.md](observability.md).
- **Live tail**: `GET /__llm-redact/recent` (last 200 rows) and the
  dashboard's SSE feed (`/__llm-redact/events`) work without the audit DB.
- **JSON logs**: `[log] format = "json"` (or `serve --log-format json`)
  switches to one JSON object per line for log shippers — content is
  unchanged (paths, statuses, counts; never values or headers).

The **audit log** (`[audit]`, with the tamper-evident chain, the zero-loss
`required` mode — "no audit row, no service" — and off-machine S3/Azure
sinks) and **OpenTelemetry** export (`[otel]`) are Pro features of the
`llm-redact-pro` package (**coming soon**; documented with the package).

## Performance and platform posture

- **Event loop**: `pip install 'llm-redact-proxy[perf]'` adds uvloop;
  uvicorn picks it up automatically — no configuration.
- **FIPS 140-3**: all cryptographic uses are FIPS-approved algorithm
  selections; run `llm-redact fips-check` on the host and see
  [fips.md](fips.md) for validated-host deployment.

## Coverage honesty

Several settings deliberately let some traffic through unredacted, and the
docs must never imply otherwise. Each is surfaced in `/status`, by
`doctor`/`status` posture output, and in the dashboard — never silently:

- `warn` mode forwards the matched value (and anything a longer warn match
  overlaps) upstream — it is observation only.
- `[providers.NAME] detection = false` forwards that whole provider's
  requests unredacted (rehydration stays on).
- `[detection.mcp] exempt_servers` bypasses detection for the named MCP
  servers' content blocks.
- `[detection] languages` does not build national-id rules scoped outside
  the listed languages.

If you enable any of these, `llm-redact doctor` and `llm-redact status`
will tell you — that is the intended way to confirm your deployment's
actual coverage.
