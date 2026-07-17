# Threat model

What llm-redact defends against, what it deliberately does not, and why.
This is the reference for [SECURITY.md](SECURITY.md)'s scope section.
The request-path gates this model implies — each policy decision point
and enforcement point, mapped to code — are diagrammed in
[security-dataflows.md](security-dataflows.md), and each boundary below
maps to the test that guards it in
[security-testing.md](security-testing.md).

## Purpose and security goal

llm-redact sits between an agentic tool (Claude Code, Codex CLI, and
similar) and an LLM provider's API. Its single security goal:

> **Private values in request bodies must not reach the provider; the
> mapping from placeholder to real value must never leave the machine.**

Everything else — availability, latency, even correctness of responses —
is subordinate to that goal. The proxy fails *closed* wherever the goal is
at stake (oversized bodies are rejected rather than forwarded unredacted;
a wrong vault key aborts startup rather than serving garbage) and fails
*open* only where it is not (unrecognized traffic passes through verbatim,
because breaking the tool teaches users to bypass the proxy).

## Assets

1. **The secret values themselves** (API keys, emails, PII) — in flight,
   in the vault, and in whatever the proxy writes to disk.
2. **The vault mapping** (placeholder ↔ value). Equivalent in sensitivity
   to the values: anyone holding it can reverse every past redaction.
3. **Traffic metadata** (what tools were used when, which detector types
   fired). Lower sensitivity, still guarded: opt-in audit, counts only.

## Trust boundaries and assumptions

- **The machine is single-user and not already compromised.** The proxy
  runs as the same user as the tools it serves. A same-UID attacker can
  read the vault file, the process memory, and the tool's own credentials
  — no useful boundary exists there, and we do not pretend to provide one.
- **The proxy binds 127.0.0.1 by default, and a wider bind is fail-closed
  behind mutual TLS.** `serve` refuses any non-loopback host unless
  `[tls]` provides certfile, keyfile, AND client_ca — every connecting
  client must present a certificate the CA signed, because a network
  client of the proxy can read rehydrated secrets and edit detection
  config. Server-only TLS (no client_ca) is allowed on loopback, where
  the sniffer it would counter already implies same-UID compromise. In
  containers the bind is 0.0.0.0 *inside the container netns* with a
  documented loopback-only publish spec (`-p 127.0.0.1:8787:8787`); the
  image sets `LLM_REDACT_INSECURE_BIND=1` for exactly that confined case,
  and setting it anywhere else is on the operator.
- **The LLM provider is honest-but-curious.** We keep values away from it;
  we do not defend against a provider actively attacking the client.
- **The agentic tool is trusted.** It holds the API credentials and the
  user's files; the proxy adds privacy, not sandboxing.
- **The browser is hostile.** Any web page can issue requests to
  127.0.0.1. This is the one boundary where an active network attacker is
  in scope — see the config editor below.

## Defenses at each boundary

### Realtime WebSocket connections

- Same trust story as HTTP: the relay listens on the same loopback/mTLS
  bind, forwards auth (headers, `?key=` queries, subprotocol keys)
  untouched and unlogged, and verifies the upstream wss certificate
  against system CAs.
- Fail-closed edges: unknown WS paths are refused (there is no default
  WS upstream), disabled providers are refused, and without the
  `realtime` extra the server cannot accept upgrades at all — realtime
  traffic can never silently bypass redaction.
- Text modality only. Voice audio is base64 media and is never decoded
  or scanned (the standing media non-goal): what a user SAYS on a
  realtime connection reaches the provider unredacted. The docs say so
  wherever realtime is described.

### Outbound requests (tool → provider)

- Detection runs over **every string value** in the JSON body via a
  generic walk — system prompts, nested content blocks, tool results —
  not a hardcoded schema. Unknown fields forward verbatim by design.
- Bodies too large to buffer and redact are rejected **413 fail-closed**.
- Per-rule `block` mode rejects requests before any upstream contact.
- Auth headers pass through untouched and are never logged.
- Batch transports are covered like chat: JSONL lines in batch creation
  and file uploads are redacted per line, results/output downloads are
  restored per line, and an upload too large to buffer is rejected 413.
- MCP connector configuration (`mcp_servers`, `tools type=mcp`) is
  deliberately NOT redacted: it is addressed to the provider, which must
  hold the real credential to call the MCP server on the model's behalf.
  MCP call content is redacted/restored like any other content.

### The vault (the mapping at rest)

- Default backend is **in-memory**: nothing on disk, dies with the
  process. Persistence is an explicit opt-in.
- SQLite vaults are created `0600` in a `0700` directory, WAL with
  `synchronous=FULL` (a lost counter write could re-issue a live token
  number and silently rehydrate the wrong secret).
- Optional **encryption at rest** (`fernet`): HKDF-split key into a
  domain-separated HMAC index key and a Fernet data key; wrong or missing
  key fails closed at open; the encrypt-in-place migration checkpoints
  and VACUUMs so plaintext does not linger in WAL/freelist pages. The
  key lives in `LLM_REDACT_VAULT_KEY`, never in the config file.
- Session isolation is **strict by construction**: token names collide
  across sessions deliberately, so there is no fallback lookup — a
  cross-session hit would silently restore someone else's secret.
  Pruning deletes whole sessions only.

### Local ops surface (`/__llm-redact/*`)

- Answered before any routing logic runs; provably never forwarded.
- GET-only, with THREE exceptions sharing one guard chain — `POST /config`,
  `POST /sessions/prune`, and `POST /preview` — defended in layers: Host
  validation (DNS rebinding), Origin validation, a per-process CSRF token
  bound to a custom header (forcing a CORS preflight that 405s with no CORS
  headers), a JSON content-type requirement, and a 1 MiB body cap.
  Config edits revalidate through the production config parser plus
  dry-run detector/mode builds before anything is written or applied;
  prune deletes whole idle sessions only and never the active one;
  `/preview` runs the live detectors over caller-supplied text on a
  throwaway vault and writes nothing (no upstream request, no vault,
  metrics, or audit write).
- Every reserved reply also carries browser-hardening response headers —
  a strict `Content-Security-Policy` (`default-src 'none'`, only inline
  script/style and same-origin `connect-src`, `frame-ancestors 'none'`),
  `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and
  `Referrer-Policy: no-referrer` — so a hostile page cannot frame the
  dashboard, run injected remote code, or leak a Referer. These are
  defense-in-depth on top of the Host/Origin/CSRF gates, not a substitute.
- Status/metrics/audit — and the `/events` live feed, which streams the
  same rows `/recent` serves — expose **types and counts only**: never
  values, never placeholder ids, never allowlist contents (the config
  editor GET is the documented exception for allowlists; it sits behind
  the same Host/Origin checks, as do `/sessions` and `/events`).

### Logging posture

- Log lines carry path, status, and detection counts. Never values,
  never headers, never bodies, never URLs with query strings (Gemini
  passes `?key=` — uvicorn access logging is disabled and the httpx
  logger is raised to WARNING for exactly this reason).
- The audit DB (opt-in) stores the same metadata plus durations, with
  `synchronous=NORMAL` — losing an audit row is acceptable, losing a
  vault row is not. `[audit] required = true` (opt-in) inverts exactly
  that stance: the DB flips to `synchronous=FULL`, a write-ahead START
  row is durably committed BEFORE any upstream contact, and a request
  whose row cannot be committed is refused 503 — audit storage joins
  the availability path by explicit choice. An optional per-row HMAC hash-chain
  (`[audit] tamper_evident`, key from `LLM_REDACT_AUDIT_HMAC_KEY` — env
  only) makes deletion or alteration of any row detectable
  (`llm-redact audit verify`); it covers the same metadata, never values,
  and `tamper_evident` without a key fails closed at startup.
- OpenTelemetry export (opt-in, `[otel]`) emits the same metadata-only
  rows as spans and counters. Pointing `endpoint` at a remote collector
  is an explicit trust decision, documented next to the config key.
  Values, headers, and placeholder ids are never attributes.
- The S3 audit sink (opt-in, `[audit.s3]`) ships the same metadata-only
  rows as NDJSON objects to a bucket you name — the same off-machine
  trust decision as a remote OTel collector. Credentials come from the
  standard AWS environment variables only, never the config file; upload
  failures warn and drop rather than blocking or crashing the proxy
  (under `[audit] required` the sinks instead spool from the audit DB
  and retry until the upload is confirmed — at-least-once, never drop).

### Deliberate protection opt-outs

Warn mode has always been observation-only (the matched value IS
forwarded). Three configuration knobs extend that family — each is a
conscious decision the operator makes, and each is surfaced rather than
silent:

- `[providers.NAME] detection = false` forwards that provider's requests
  unredacted (rehydration stays active). Logged per request, listed in
  `/status` `providers_detection_off`, marked on the dashboard. Meant
  for upstreams you own end to end (a local Ollama).
- `[detection.mcp] exempt_servers` exempts MCP content blocks addressed
  to named servers. Result blocks that cannot be correlated to an exempt
  server stay redacted (fail-closed).
- `[detection] languages` narrows which national-id rules are built;
  universal rules (emails, keys, cards, IBANs, phones) always run, and
  the scoped-out rules are listed in `/status` and the editor.

### Supply chain

- Runtime dependencies are deliberately three: httpx, starlette, uvicorn.
- CI runs a **gating pip-audit** over the exported runtime closure on
  every push/PR and weekly on schedule, and a **dependency-review** job
  fails a PR that adds a high-severity or copyleft-licensed dependency
  (Dependency Graph diff; arms on go-public).
- **OpenSSF Scorecard** runs weekly and on the default branch, publishing
  its supply-chain posture (pinned actions, token permissions, dangerous
  patterns) to the code-scanning dashboard and the public Scorecard log
  (arms on go-public).
- Actions are SHA-pinned; release artifacts carry Sigstore provenance
  attestations that the release job **self-verifies** with
  `gh attestation verify` before publishing (a broken attestation fails
  the release); PyPI uploads carry PEP 740 attestations via trusted
  publishing (no stored secrets).
- The GHCR container image is **cosign keyless-signed over its digest**
  (verifiable against the GitHub Actions OIDC issuer) and ships a BuildKit
  SBOM attestation — signature says who built it, SBOM says what is in it.

### Behavior under fault

The security goal must hold when the network drops mid-stream, the upstream
times out or 5xxs, a frame arrives truncated, or the vault's disk misbehaves.
Under every such fault the proxy **fails closed** and **never rehydrates to the
wrong value**: a buffered upstream fault becomes a recorded 502; a mid-stream
drop cuts the stream after valid bytes and still finalizes; a stream that ends
mid-token flushes the partial placeholder verbatim rather than guessing; a
vault write fault rolls back without wedging the connection or skipping a
counter; a corrupted or wrong-key vault fails closed rather than issuing or
returning a wrong secret. The full catalogue, with the suites that pin each
row, is [resilience.md](resilience.md).

## Explicitly out of scope

| Non-goal | Why |
|---|---|
| Plaintext in process memory / swap | The redactor must hold values to substitute them; encrypted swap is an OS concern |
| Same-UID attackers | No boundary exists: they can read the tool's credentials directly |
| Loopback packet sniffing | Countering it requires same-UID compromise already; loopback server-only TLS is available but optional. Non-loopback binds are supported ONLY under mutual TLS (fail-closed in `serve`) |
| Multi-user machines | Vault modes help, but the design assumes one user |
| Provider-side inference | The provider can guess redacted content from context; only omission fixes that |
| Length/timing side channels | Placeholder lengths differ from originals; smoothing them would break streaming |
| Base64 media contents | Images can't leak through text regexes; PDF parsing would need heavy deps |
| Shapes the rules exclude | Bare-digit phones, street addresses, passport/DL numbers: collision-prone with no reliable grammar |
| SigV4-signed provider traffic (AWS Bedrock via SDK credentials) | Permanent non-goal: the signature covers the payload hash, so a body-rewriting proxy can never transit it without holding the user's AWS credentials and re-signing — which this design will not do. Bearer-token Bedrock (API keys) IS supported: the proxy parses AWS's binary CRC-framed eventstream encoding natively (both CRCs validated per frame; a framing violation degrades to verbatim pass-through, so unrestored placeholders — never corrupted frames — are the worst case), and invoke-route bodies are rewritten only for positively recognized model-native shapes (Claude), with everything else forwarded verbatim |

## Residual risks

| Risk | Mitigation status |
|---|---|
| Novel secret formats the rules miss | User-extensible custom rules; NER extras; fp/recall gates keep the shipped set honest |
| LLM mangles a placeholder beyond fuzzy repair | Pass-through verbatim (never a wrong value); bracket swaps deliberately unrestored |
| History compaction rewrites the session anchor | Fails safe: fresh session, no cross-session restore — verified by the dogfood compaction probe |
| Values pre-escaped inside JSON-source strings | Captured in escaped form; documented limitation |
| A drifted provider event shape bypasses a rehydration channel | Drift detectors in live tests; unknown shapes pass through rather than corrupt |
| License enforcement circumvented by patching the source | Accepted: signed keys prevent forgery and the single chokepoint makes tampering auditable, but source-available checks are deterrence, not DRM — the license is a legal instrument, never a security boundary (the `llm-redact-pro` repo's `docs/licensing.md`) |
| Vault rows leave the machine on an RDBMS backend | Fail-closed: a non-local DSN (including recognized managed-DBMS hosts and Cloud SQL sockets) refuses startup unless the vault is Fernet-encrypted — only the HMAC index and ciphertext travel. `LLM_REDACT_VAULT_REMOTE_PLAINTEXT=1` is the explicit, surfaced opt-out; the database server and its operator join the trust boundary either way, and `backend = "dbapi"` DSNs are opaque (doctor WARNs that locality is unverifiable). Encryption mode is fixed at schema creation — server-side MVCC keeps old row versions, so an after-the-fact encrypt would be dishonest (the `llm-redact-pro` repo's `docs/vault-rdbms.md`) |
