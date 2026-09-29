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
  in scope — see "Requests from web pages" and the local ops surface
  below.

## Defenses at each boundary

### Requests from web pages (the operator's browser)

The attacker: **a web page the operator's browser visits** — any site, an
ad, a compromised page. It can reach the proxy on 127.0.0.1 even though
the operator never pointed anything at it:

- a "simple" cross-origin request (a `text/plain` POST, a GET) needs no
  CORS preflight, so it is sent without asking;
- a CORS request carrying headers (an API key) used to have its preflight
  forwarded to a CORS-friendly upstream, and the upstream's
  `Access-Control-Allow-Origin` passed back — the page could read the
  answer;
- **DNS rebinding** re-resolves the attacker's own name to 127.0.0.1, so
  the page becomes same-origin with the proxy and reads every answer;
- a **WebSocket** handshake gets no CORS at all: any page can open
  `ws://127.0.0.1:8787/v1/realtime` and read every frame.

What the proxy lends such a request, and why each matters:

- **The vault.** Every rehydrating route restores the operator's values
  into whatever the upstream sends back. A page with its OWN provider key
  that asks a model to repeat `«EMAIL_001»` (or stores a Response
  containing it and fetches it back) would read the operator's secret —
  the vault would leave the machine token by token. This holds for every
  provider and every credential mode.
- **A credential the proxy holds.** Under `auth = "identity"` (Bedrock,
  Vertex AI, Azure OpenAI) or a routed plan spending an operator key
  (llm-redact-pro routing, brokered keys), the page spends it: invocations,
  async jobs, stored objects, realtime sessions.
- **Network access.** A keyless upstream only the proxy should reach (a
  local Ollama, a vLLM or LM Studio server). The proxy also rewrites
  `Host` when forwarding, which defeats Ollama's own DNS-rebinding check.

An access gate with *ambient* credentials (client certificates, Basic
auth, an access proxy's cookie) does not stop any of this: the browser
attaches them to the page's requests by itself.

The defense (`proxy.request_origin_refusal`, applied to every HTTP request
and WebSocket upgrade bound for an upstream, before any credential fetch
or upstream contact):

- A request carrying **browser markers** — `Origin` or any Fetch Metadata
  (`Sec-Fetch-*`) header; page script can neither set nor remove either —
  must be addressed to a host name the proxy answers to (127.0.0.1,
  localhost, ::1, its bind host, `allowed_hosts`, the access gate's public
  origin), carry only its own origin (exact scheme, host and port — API
  routes have no CSRF token to cover a same-host page on another port),
  and a `Sec-Fetch-Site` of `same-origin` or `none`. The one exception is
  the operator's opt-in `allowed_origins`: a page whose `Origin` matches a
  listed origin exactly is served although it is cross-site (its Origin,
  which page script cannot forge, vouches for it) — the Host rule still
  holds, and a request without an `Origin` is never taken for a listed
  one.
- A request that would spend a credential the proxy holds must name such
  a host even without browser markers (a browser lacking Fetch Metadata
  sends none on a same-origin GET) — unless it arrived over **TLS**: a
  browser verifies the proxy's certificate against the name it resolved,
  so a rebound page never reaches a TLS listener, and a team server's
  clients may use any name its certificate covers.
- Refusals are a recorded, provider-shaped **403** over HTTP and an
  accept-then-close **1008** on a WebSocket, counted by kind in `/status`
  `request_origin_refusals_total`; the answer and the log line name the
  kind only, never the Host or Origin the request carried. The WebSocket
  check runs first, so a page learns nothing about the routes behind it.

CLI tools and SDKs send no browser markers, so they are unaffected —
including when they reach the proxy by an alias (a compose service, a
Kubernetes Service, `host.docker.internal`) on their own credential. An
alias that spends a credential the proxy holds over plain HTTP is listed in
`allowed_hosts` ([deployment.md](deployment.md#host-names-the-proxy-answers-to-allowed_hosts)).
By default, a browser-based client served from another origin (a web chat
UI, a browser extension, an Electron renderer) cannot use the proxy: it is
indistinguishable from the attacker above. An operator who trusts one lists
its exact origin in `allowed_origins`
([deployment.md](deployment.md#browser-apps-on-other-origins-allowed_origins)),
knowing what that grants: the page — and any code that runs on its origin
— reads restored values back and spends every credential the proxy holds.
Only http(s) origins can be listed, plain http only on this machine
(anyone on the network path can serve a remote http origin), never the
opaque `null` origin; the reserved `/__llm-redact/*` endpoints never
consult the list, and the proxy stays transparent to CORS (the provider's
own answer decides whether the page may read a response).

### Realtime WebSocket connections

- Same trust story as HTTP: the relay listens on the same loopback/mTLS
  bind, forwards auth (headers, `?key=` queries, subprotocol keys)
  untouched and unlogged, and verifies the upstream wss certificate
  against system CAs.
- Fail-closed edges: unknown WS paths are refused (there is no default
  WS upstream), disabled providers are refused, and without the
  `realtime` extra the server cannot accept upgrades at all — realtime
  traffic can never silently bypass redaction.
- A web page's handshake (browsers send `Origin` on every WebSocket
  upgrade and apply no CORS to it) is refused with close code 1008 before
  anything else — see "Requests from web pages" above.
- A connection never outlives the configuration it was admitted under.
  Realtime sessions run for tens of minutes, so a reload that withdraws the
  proxy's identity or tightens redaction must reach open connections, not
  only new ones. A reload that changes a connection's provider settings,
  its upstream authorizer or the `[detection]` policy closes it with code
  1012 (reconnect). No client frame the relay reads after the reload is
  forwarded under the old configuration
  ([deployment.md](deployment.md#reloads-and-open-realtime-connections)).
- Text modality only. Voice audio is base64 media and is never decoded
  or scanned (the standing media non-goal): what a user SAYS on a
  realtime connection reaches the provider unredacted. The docs say so
  wherever realtime is described.

### Outbound requests (tool → provider)

- Detection runs over **every string value** in the JSON body via a
  generic walk — system prompts, nested content blocks, tool results —
  not a hardcoded schema. Unknown fields forward verbatim by design. The
  walk skips only SCALAR values of structural keys (`model`, `role`,
  `type`, `id`, base64 `data`, …); an object or array under such a name
  is walked, and caller-supplied JSON at known tool-call / tool-result /
  document positions (Gemini `functionCall.args` and
  `functionResponse.response`, Bedrock `toolUse.input` and
  `toolResult.content[].json`, Anthropic `tool_use.input`, Cohere
  `documents`, …) is walked with no skips at all, so a tool result that
  names its own keys `id` or `data` is still redacted. The same holds for
  maps whose keys the caller chooses: `metadata` wherever it appears
  (OpenAI/Azure chat, Responses, batches, conversations, Realtime;
  Anthropic), Bedrock Converse `requestMetadata`, prompt-template
  `prompt.variables` (Responses, Realtime), and a `:predict` body's
  top-level `instances`/`parameters` (arbitrary custom-model input) — a
  metadata label or a feature column named `id`, `name` or `type` is the
  caller's data, not protocol.
- A request path with a `.`/`..` segment (any spelling) is refused 400
  before any upstream contact: matching and forwarding must address the
  same resource.
- Bodies too large to buffer and redact are rejected **413 fail-closed**.
- Per-rule `block` mode rejects requests before any upstream contact.
- Auth headers pass through untouched and are never logged.
- Batch transports are covered like chat: JSONL lines in batch creation
  and file uploads are redacted per line (and every upload's filename,
  restored on the file objects that echo it), results/output downloads
  are restored per line, and an upload too large to buffer is rejected 413.
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
- Because names collide across sessions, a request can carry tokens its
  own session never issued (a compacted history, a pasted answer). A new
  value is numbered above every such token the request carries — the
  per-type **token floor**, read from the decoded request (canonical,
  fuzzy and JSON-escaped forms; multipart parts; the Bedrock CountTokens
  blob; a realtime connection's every client frame) — so the upstream
  never reads one token name with two meanings and a foreign token's echo
  passes through verbatim. Numbers stop at 999999999: a request needing
  one past it is refused, never wrapped.

### Local ops surface (`/__llm-redact/*`)

- Answered before any routing logic runs; provably never forwarded.
- GET-only, with guarded POST exceptions sharing one guard chain —
  `POST /sessions/prune` in the core, plus llm-redact-pro's dashboard
  `POST /config` and `POST /preview` and its access gate's
  `POST /users/invite|revoke` —
  defended in layers: Host validation (DNS rebinding), Origin validation,
  a per-process CSRF token bound to a custom header (forcing a CORS
  preflight that 405s with no CORS headers), a JSON content-type
  requirement, and a 1 MiB body cap. Prune deletes whole idle sessions
  only and never the active one. The dashboard paths (`/`, `/config`,
  `/preview`) are dispatched to the llm-redact-pro plugin only when it is
  registered — otherwise they answer a local 404 — and are never
  forwarded either way; that package's config editor revalidates every
  edit through the production config parser plus dry-run detector/mode
  builds before anything is written or applied, and its preview runs the
  live detectors on a throwaway vault and writes nothing.
- Every reserved reply also carries browser-hardening response headers —
  a strict `Content-Security-Policy` (`default-src 'none'`, only inline
  script/style and same-origin `connect-src`, `frame-ancestors 'none'`),
  `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and
  `Referrer-Policy: no-referrer` — so a hostile page cannot frame a
  reserved page, run injected remote code, or leak a Referer. These are
  defense-in-depth on top of the Host/Origin/CSRF gates, not a substitute.
- With llm-redact-pro's access gate configured for it, the reserved
  endpoints also pass through the gate's **dashboard admission** before
  they answer (the core's `_admit_reserved`): the monitoring probes
  (`/healthz`, `/readyz`, `/metrics`) and the gate's own sign-in and SCIM
  paths are exempt, a refused browser GET is redirected only to a
  same-proxy `/__llm-redact/…` path (never an open redirect), and anything
  else is a 403. Only a gate that guards the dashboard may widen the Host
  and Origin checks to its one public origin, so a wider Host is never
  accepted without authentication. SCIM requests skip the Origin check
  (identity providers are not browsers) but keep the Host check.
- Status/metrics/audit — and the `/events` live feed, which streams the
  same rows `/recent` serves — expose **types and counts only**: never
  values, never placeholder ids, never allowlist contents (the
  llm-redact-pro config editor GET is the documented exception for
  allowlists; it sits behind the same Host/Origin checks, as do
  `/sessions` and `/events`).

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
  trust decision as a remote OTel collector. Credentials come from
  environment variables or, with `auth = "identity"`, from the
  workload's cloud identity at runtime — never the config file; upload
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
  `/status` `providers_detection_off` and the `status`/`doctor` posture
  output. Meant
  for upstreams you own end to end (a local Ollama).
- `[detection.mcp] exempt_servers` exempts MCP content blocks addressed
  to named servers. Result blocks that cannot be correlated to an exempt
  server stay redacted (fail-closed).
- `[detection] languages` narrows which national-id rules are built;
  universal rules (emails, keys, cards, IBANs, phones) always run, and
  the scoped-out rules are listed in `/status` and by `doctor`.

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
| Base64 media contents | Images can't leak through text regexes; PDF parsing would need heavy deps. Media blobs at their known positions (base64 `data`, Bedrock `source.bytes`) are not even scanned — scanning base64 finds nothing real, costs event-loop CPU, and could rewrite a token-shaped run inside an image |
| Values a client deliberately encodes | base64 inside a JSON string, a quoted-printable or foreign-charset multipart part: the proxy scans the bytes it receives. Under identity auth a declared multipart Content-Transfer-Encoding or charset is refused (400) instead of signed |
| Structural names | JSON object keys, header names and a multipart part's `name` are protocol, not content; values are scanned — string values, and an upload's `filename` / `filename*` |
| Shapes the rules exclude | Bare-digit phones, street addresses, passport/DL numbers: collision-prone with no reliable grammar |
| SigV4-signed provider traffic (AWS Bedrock via SDK credentials) | Permanent non-goal: the signature covers the payload hash, so a body-rewriting proxy can never transit a signature the CLIENT computed, and it never holds the user's AWS credentials to re-sign. The proxy MAY sign with its OWN identity (`[providers.bedrock] auth = "identity"`, llm-redact-pro): the client's credentials are stripped and the redacted body is signed by credentials the operator gave the proxy (a body the proxy could not redact — non-JSON, a top-level array or scalar, content-encoded (in any Content-Encoding header), sent with a repeated Content-Type, or multipart on a route it does not scan — is refused 400, never signed verbatim) — which any client that reaches the proxy can then spend, so pair it with the access gate or a loopback bind. Bearer-token Bedrock (API keys) IS supported: the proxy parses AWS's binary CRC-framed eventstream encoding natively (both CRCs validated per frame; a framing violation degrades to verbatim pass-through, so unrestored placeholders — never corrupted frames — are the worst case), and invoke-route bodies are rewritten only for positively recognized model-native shapes (Claude), with everything else forwarded verbatim |

## Residual risks

| Risk | Mitigation status |
|---|---|
| Novel secret formats the rules miss | User-extensible custom rules; NER extras; fp/recall gates keep the shipped set honest |
| LLM mangles a placeholder beyond fuzzy repair | Pass-through verbatim (never a wrong value); bracket swaps deliberately unrestored |
| History compaction rewrites the session anchor | Fails safe: fresh session, no cross-session restore — verified by the dogfood compaction probe. The fork never issues a token its summary carries: the summary is the fork's anchor, so every request of it carries the summary's tokens and the token floor numbers new values past them |
| A request carries tokens its session did not issue (a pasted answer, a foreign proxy's token) | The token floor keeps the request (and a realtime connection) that carries them from issuing those names. Residual, documented in [compaction-relink.md](compaction-relink.md): a number the session had already issued before the foreign token arrived, another request sharing the session that does not carry the token, and provider-side history (a `previous_response_id` chain, a realtime model's own output) no request of the session carries — the last is what llm-redact-pro's sealed sessions cover |
| Values pre-escaped inside JSON-source strings | Captured in escaped form; documented limitation |
| An origin listed in `allowed_origins` — or code that runs on it (an XSS, a compromised script it loads) — reads restored values back and spends the proxy's credentials | By design: listing is the operator's explicit grant, off by default. Exact origins only (normalized; no wildcard, no `null`, no non-web scheme), plain http only for this machine, the Host rule and every other rule unchanged, reserved endpoints never consult the list; `doctor` WARNs with the list, `/status` counts it and `llm-redact status` prints it |
| A web page in a browser without Fetch Metadata (older than Chrome 76, Firefox 90, Safari 16.4), after DNS rebinding, reads back a Response or file it stored with its own key through a plain-HTTP route that forwards the client's credential | Such a same-origin GET carries no browser marker, so its foreign Host is not checked (an alias host must keep working for CLI tools); every current browser sends `Sec-Fetch-Site`, which subjects it to the Host check. A request spending a credential the proxy holds is Host-checked with or without markers — over TLS the certificate does that, so a wildcard certificate covering a name the attacker controls would reopen it for such a browser |
| A drifted provider event shape bypasses a rehydration channel | Drift detectors in live tests; unknown shapes pass through rather than corrupt |
| License enforcement circumvented by patching the source | Accepted: signed keys prevent forgery and the single chokepoint makes tampering auditable, but source-available checks are deterrence, not DRM — the license is a legal instrument, never a security boundary (the `llm-redact-pro` repo's `docs/licensing.md`) |
| Vault rows leave the machine on an RDBMS backend | Fail-closed: a non-local DSN (including recognized managed-DBMS hosts and Cloud SQL sockets) refuses startup unless the vault is Fernet-encrypted — only the HMAC index and ciphertext travel. `LLM_REDACT_VAULT_REMOTE_PLAINTEXT=1` is the explicit, surfaced opt-out; the database server and its operator join the trust boundary either way, and `backend = "dbapi"` DSNs are opaque (doctor WARNs that locality is unverifiable). Encryption mode is fixed at schema creation — server-side MVCC keeps old row versions, so an after-the-fact encrypt would be dishonest (the `llm-redact-pro` repo's `docs/vault-rdbms.md`) |
