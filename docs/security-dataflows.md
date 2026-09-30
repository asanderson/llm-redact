# Security-relevant data flows

Where security decisions are *made* and where they are *enforced*, in
NIST SP 800-162 terms: each control below is split into its **PDP**
(policy decision point — what evaluates the policy, and against which
inputs) and its **PEP** (policy enforcement point — the code that acts on
the decision). The **PAP** (policy administration point) is the TOML
config file, editable by hand, by `llm-redact init`, by the guarded
`config-edit` agent command (`serve --check` gate, then SIGHUP), or —
with llm-redact-pro — through the dashboard config editor, which is
itself defended as gate ② below and revalidates every edit through the
production parser plus a dry-run detector build before anything is
written or applied.

This page maps the flows; [threat-model.md](threat-model.md) explains why
the boundaries sit where they do, and [SECURITY.md](SECURITY.md)
defines what counts as a vulnerability against them.

![Every gate a request passes: bind policy, ops guard chain, provider routing, size cap, detection modes, and the local vault](diagrams/security-gates.svg)

*Animated: the flow along the accepted path moves; refusals are static. Static [PNG](diagrams/security-gates.png) · [GIF](diagrams/security-gates.gif) · Mermaid source: [diagrams/security-gates.mmd](diagrams/security-gates.mmd).*

## The gates

In llm-redact the PDP and PEP of each gate are deliberately colocated in
one process — there is no network hop between deciding and enforcing, so
there is no decision channel to attack. The split below is therefore
about *code roles*, and it is where to look when auditing.

| # | Policy question | PDP — decided by | PEP — enforced by | On refusal |
|---|---|---|---|---|
| ① | May this socket exist at all? | `validate_bind_security` (`config.py`): non-loopback hosts require the full mTLS trio (`certfile` + `keyfile` + `client_ca`); unresolvable hostnames count as non-loopback; `LLM_REDACT_INSECURE_BIND=1` is the container-netns hatch | `serve` (`cli.py`) refuses to start **before binding**; with mTLS, uvicorn runs `ssl_cert_reqs=CERT_REQUIRED`, so unauthenticated clients fail the TLS handshake | no socket / handshake failure |
| ② | May this request use the local ops surface? | The guard chain in `proxy.py`: GET-only except the guarded POSTs (`/sessions/prune`; with llm-redact-pro also the dashboard's `/config` and `/preview` and the access gate's `/users/invite` and `/users/revoke`), each layered Host validation (DNS rebinding) → Origin → per-process CSRF token in a custom header (forces a CORS preflight that 405s with no CORS headers) → JSON content-type → 1 MiB cap; every reserved reply also carries browser-hardening headers (`_SECURITY_HEADERS`: strict CSP + `X-Frame-Options`/`nosniff`/`Referrer-Policy`); with llm-redact-pro's access gate guarding the dashboard, every reserved path except the monitoring probes and the gate's sign-in/SCIM paths first passes the gate's dashboard admission (403, or a 303 to a same-proxy sign-in page for a browser GET), and only then may the Host/Origin checks accept the gate's one public origin | Reserved `/__llm-redact/*` paths are answered by the **first statement** of `handle()` — provably never forwarded upstream; failing any layer answers 4xx; the response headers block framing, remote-code injection, and Referer leakage as defense-in-depth | 4xx, never forwarded |
| ②½ | Who is this client, and may it use the proxy? | **llm-redact-pro's access gate** (`plugin_api.AccessGate`): named users, per-user keys and seats — this repository holds no authentication code. The gate runs first (it may await a directory or identity provider) and removes every credential it recognizes from the request. Under mutual TLS, `serve` hands it the verified client certificate as the standard ASGI TLS extension (`tls_scope.py`); the core itself reads nothing from it. Without the package, every client is the implicit single local user, and two leak guards in `proxy.py` still hold: every `x-llm-redact-*` request header is dropped before forwarding (HTTP and WebSocket), and a `/u/…` identity path is answered locally, never forwarded or recorded | `handle()` applies the gate's refusal as a provider-shaped **403** after the disabled-provider 502 and before the body is read; the WebSocket relay closes **1011** with the gate's message — recorded as a 403 either way | 403 (HTTP) / close 1011 (WS) |
| ②¾ | Is this a web page's request? (every request bound for an upstream) | `request_origin_refusal` (`proxy.py`): a request with browser markers (`Origin` or any `Sec-Fetch-*` header) must be addressed to a host name the proxy answers to (127.0.0.1, localhost, ::1, the bind host, `allowed_hosts`, the access gate's public origin), carry only its own origin (exact scheme, host and port) and a `Sec-Fetch-Site` of `same-origin` or `none` — or an `Origin` the operator listed in `allowed_origins` (opt-in, default empty; exact serialized match, `config.normalize_origin`; such a page may read restored values back and spend the proxy's credentials, and still needs an answered Host; the reserved endpoints never consult the list); a request that would spend a credential the **proxy holds** — its cloud identity (known before the body), or a routed plan's operator key or no key (`RoutePlan.proxy_credential`, checked again once planned) — must name such a host even without markers, unless it arrived over TLS (a browser verifies the certificate against the name it resolved, so a rebound page never reaches a TLS listener). The rule protects the vault too: every rehydrating route would restore the operator's values into an answer a page with its own provider key could read | `handle()` answers a recorded, provider-shaped **403** right after admission — before the disabled-provider 502, the gate's refusal, the body read, any credential fetch and any upstream contact (a routed plan is dropped, never begun); the WebSocket relay closes **1008** before its own route checks, so a page learns nothing about the routes behind it; counted by kind in `/status` `request_origin_refusals_total`, logged by kind only (never the Host or Origin) | 403 (HTTP) / close 1008 (WS) |
| ③ | Is this traffic for a known, enabled provider — and does it get detection? | Adapter `matches()` (`providers/`) plus `[providers.NAME] enabled` and `detection` from config; named `[providers.custom.NAME]` upstreams route under `/custom/NAME/` and an unknown custom name has nowhere sane to go. A request no adapter recognizes goes only to the provider it is POSITIVELY attributed to (`providers/attribution.py`: a path family, else one provider's markers, else OpenAI's own key or resource prefixes) | `handle()` answers a proxy-generated **502 before the body is read** for disabled or unknown-custom providers — never falls through to pass-through; an unattributable request is a recorded local **404**, another spelling of a recognized route a recorded **400**, and that route without its `/v1` or under an extra prefix a recorded **404** (`_misaddressed`); an unrecognized route that would carry a credential the proxy holds (its cloud identity, or a routed plan's operator key or no key) is a recorded **403** before its body is read; the WebSocket relay (`realtime.py`) refuses with accept-then-close **1011** (recorded as the same 502). `detection = false` skips gate ⑤ entirely — a **deliberate per-provider opt-out** (requests reach that upstream unredacted; rehydration stays active) surfaced per request in the log, in `/status` `providers_detection_off`, and in `llm-redact status`/`doctor` posture output | 502/404/400/403 (HTTP) / close 1011 (WS) |
| ④ | Can the body be buffered and redacted? | `max_body_bytes` (default 10 MiB) and `max_body_strings` (default 100,000 strings — JSON string values, form fields, file names, uploaded JSONL lines — or multipart parts) from config | `handle()` rejects an oversized redactable body **413 fail-closed** before reading its strings; the per-request redactor copy counts every string it redacts and refuses the body 413 past the limit (multipart parts are counted before any parse); a realtime client frame over it closes the connection 1009 — nothing is sent upstream unredacted | 413, nothing upstream |
| ④¼ | Did the proxy read the body it would forward? (every recognized route where redaction applies, and every one spending a credential the proxy holds) | `_unscanned_body` (`proxy.py`): forwardable is a JSON object (read from the bytes whatever the content-type; BOM/UTF-16/32 included) or canonical multipart on a route whose `redact_multipart` scans it, every part then scanned (`require_scanned`). Anything else — a `Content-Encoding` other than identity in any header, bytes after the JSON value, invalid UTF-8 (a Latin-1 body), a top-level array or scalar, whitespace only, multipart elsewhere or outside the canonical form, a second Content-Type; inside an upload a JSONL line nesting too deep, a form field that is not UTF-8, a preamble/epilogue, a part header without one reading (a part without a header block every reader finds included), a transfer encoding or a foreign charset — would go out unscanned, and lenient upstreams (Go's JSON decoder behind Ollama, Express body-parser, Jackson) decode it anyway. Holds with `detection` on under the client's own key, and under any proxy-held credential (its cloud identity, or a routed plan's `proxy_credential` — a plan that does not say counts as the proxy's) whatever `detection` says — except the per-piece scan inside an upload, which runs with redaction: with `detection = false` under a proxy-held credential a canonical multipart upload is forwarded as sent once the stored-object check (④½) has read it, so a file part that is not JSONL (a text file) goes out unscanned. Each file part is read by its CONTENT (`upload_content.classify_file`): JSONL line by line, other text as one text, and a BINARY file (a PDF, an image, non-UTF-8 text, a known binary signature) — which cannot be redacted — is refused under a proxy-held credential and with `[detection] binary_uploads = "refuse"`, else (the client's own key, the default `"forward"`) forwarded byte-identical and UNSCANNED, counted per provider (`/status` `unscanned_uploads_total`, `llm_redact_unscanned_uploads_total`) and logged by path and count. With an upload inspector (`plugin_api.UploadInspector`, `upload_inspection.py`) each binary part is first read as text — awaited before the vault batch, bounded in count, concurrency, size and time — and scanned without issuing placeholders (`Redactor.scan_text`): a value to redact or a block-mode value refuses the upload (400, types only), a COMPLETE clean reading sends the part byte-identical (under a proxy-held credential only when the inspection allows it; counted `inspected_uploads_total`), anything else keeps the rules above | `handle()` answers a recorded, provider-shaped **400** — **415** with `Accept-Encoding: identity` for a content coding — naming the body's kind only, before the stored-object check, the session, redaction, any credential fetch, the plan's first hop, the audit START row and any upstream contact (multipart parts are counted against `max_body_strings` before the parse). `detection = false` with the client's own key, and unrecognized pass-through (reached only with the client's own credential — gate ③), keep forwarding such bodies verbatim — the opt-out gate ③ surfaces | 400/415, nothing upstream |
| ④½ | May this request reach the stored object it names? (named users only) | The session router's optional `object_access_refusal` (`plugin_api.SessionRouter`; **llm-redact-pro** keeps who created each stored object the core reports through `record_object_id`): the core passes the adapter, method, path, parsed body and whether the request reaches its provider with a credential the **proxy holds** — its own cloud identity, or a routed upstream's operator key or no key at all (`RoutePlan.proxy_credential`, read after the router plans; a plan that does not say counts as the proxy's). A pass-through route never reaches this check under such a credential (refused 403 before its body is read, gate ③), and its body is never read for it; an upload (multipart/form-data) is read for this check alone (`upload_view`: the JSON lines of its file parts — a batch input file's requests — and its form fields) on every matched route whether or not it redacts, before redaction and any upstream contact, and one the check cannot read (outside the canonical grammar, a transfer encoding, a form field that is not UTF-8 text, more JSON than `max_body_bytes`, more parts than `max_body_strings` — counted before any parse) is refused under such a credential. The Gemini API's single-request upload (`multipart/related`: the file's JSON metadata, then its media) is shown to the check as its metadata OBJECT (`upload_view.read_upload_metadata`: the first part, read like a JSON body — what Google reads as the create's body, a chosen file `name` included — a repeated key sent re-serialized, exactly as checked); metadata the check cannot read is refused under such a credential and, with the client's own key, wherever redaction applies (`detection = false` with the client's own key: unchecked, as an unparseable JSON body). llm-redact-pro refuses any reference — in the path or anywhere in the body — to another namespace's file, batch, conversation, Response, context cache, video job, stored completion or long-running job (Gemini batch/Veo, Vertex Veo, Bedrock async invocation) under a proxy-held credential (for a named user, also one nobody was recorded creating) and for unattributed requests it serves on the static path, and, in every mode, anything but a pure read (a body-less GET/HEAD) of one: writes, and requests carrying content of their own while citing it; with the client's own credential an object nobody was recorded creating is not refused here — a named user's request citing one is resolved to a **sealed** empty session (`SessionRouter.sealed`: the core refuses, 403 before any upstream contact, a request that would redact a value into it, so it stays empty). Without the package (or a router without the member) this gate does not exist | `handle()` (`proxy.py`) answers a recorded, provider-shaped **403** with the router's fixed reason — never an id — after the body is read and BEFORE the audit START row, redaction, any upstream credential (`UpstreamAuth.authorize`) and any upstream contact, routed or not (a routed request is planned first but never begun); a router that raises refuses (logged by exception type only) | 403 (400/413 — 415 for a content coding — for a body the check cannot read), nothing upstream |
| ⑤ | What in this body is private, and what happens to it? | The detection engine (`detection/`): deny strings (tier 0, win every overlap, bypass all allowlists), regex rules + checksum validators (+ optional NER), overlap resolution in `redactor.py`, per-rule modes via `build_modes` — the *winning* detection's mode governs. National-id rules outside `[detection] languages` are not built at all (`active_rule_names`, surfaced in `/status` and `doctor`) | `redact_text` substitutes vault tokens (redact); `BlockedRequest` becomes a provider-shaped **400 before any upstream contact**, carrying the detector type and never the value (block); **warn mode enforces nothing** — the value is counted and forwarded upstream; it is observation, not protection. MCP connector config blocks (`mcp_servers`, `tools type=mcp`) are stripped before this gate BY DESIGN — they are provider-directed credentials the provider must receive (docs/api-coverage.md) — and content blocks addressed to `[detection.mcp] exempt_servers` are stashed around it (a per-server opt-out; uncorrelatable Anthropic result blocks stay redacted, fail-closed) | 400 (block only) |
| ⑥ | May this request proceed without a durable audit record? (`[audit] required` only) | `AuditConfig.required` (parse-gated: needs `enabled`; startup-gated: the built `AuditLog` must have the write-ahead `begin`/`finalize` pair, else `ConfigError`) | `ProxyState.begin_audit` (`proxy.py`) durably commits a write-ahead START row AFTER redaction and BEFORE any byte leaves for the provider — HTTP and the realtime WS relay alike; on `AuditWriteError` the request is refused with a provider-shaped **503** (WS: accept-then-close) and nothing reaches the upstream. Off by default: without `required`, this gate does not exist and the audit trail stays fail-open | 503, nothing upstream |
| ⑦ | Who can read the placeholder↔value mapping? | Key resolution `vault_crypto.from_env`: env var → OS keychain → **fail closed**; keyring backend errors count as key-absent, never a silent downgrade. Session isolation is decided *by construction*: token names collide across sessions, so no fallback lookup can exist | SQLite vaults are created `0600` in `0700` dirs; a wrong or missing key aborts **at open** (`vault.py`) rather than serving garbage; per-session views never read across sessions; pruning deletes whole sessions only | startup/open failure |

Two request-path outcomes are **fail-open by design**, and both are
deliberate scope decisions rather than gaps:

- **Unrecognized traffic passes through verbatim** (gate ③) — to the
  provider it is positively attributed to (a path family, or one
  provider's markers such as `anthropic-version` or a Google key), never
  a guessed one; an unattributable request, and a misspelled or
  misaddressed variant of a recognized route, is refused locally (404/400)
  instead. Breaking the tool teaches users to remove the proxy, which
  protects nothing. The exception is a credential the proxy holds — its
  OWN identity (`auth = "identity"`) or a routed operator key: there an
  unrecognized path is a recorded 403, before its body is read. A
  RECOGNIZED route is not fail-open: it forwards only a body the proxy
  scanned (gate ④¼) — a body it cannot read is refused 400/415 instead
  of forwarded unread, unless the provider has `detection = false` and
  the request carries the client's own key (the explicit, surfaced
  opt-out). Wherever realtime frames are redacted, a non-JSON frame on
  an identity connection closes the connection 1008 unsent (other
  realtime connections relay non-JSON frames as they came: the realtime
  APIs take JSON events only), and a frame that IS JSON but cannot be
  read — nesting too deep, or an integer past the parser's digit limit —
  closes 1008 on every connection. While a session router checks frames
  (llm-redact-pro) both hold with `detection = false` too; without one,
  `detection = false` relays frames untouched (the surfaced opt-out). The
  same fail-open principle degrades a corrupt Bedrock eventstream frame
  to verbatim pass-through: unrestored placeholders are safe; guessing at
  corrupt frames is not.
- **An unrestorable placeholder stays a placeholder** (rehydration).
  Fuzzy repair only restores a candidate that the vault confirms; a miss
  passes through verbatim — never a wrong value.

## What leaves the machine, per channel

| Channel | Carries | Never carries |
|---|---|---|
| Upstream request (gate ⑤ output) | placeholders, plus warn-mode values (documented above), plus everything sent to a provider with `detection = false` or inside an exempt MCP server's blocks — both deliberate opt-outs | redacted originals |
| Logs (text or JSON) | path, status, detection counts | values, headers, bodies, URLs with query strings (`?key=` auth) |
| `/__llm-redact/*` (incl. `/recent`, `/events`, and the llm-redact-pro dashboard) | detector types + counts, session metadata | values, placeholder ids, allowlist contents (the llm-redact-pro config-editor GET is the documented exception, behind gate ②) |
| Audit DB (opt-in, local) | same metadata + durations | values, placeholder ids |
| S3 audit sink (opt-in, `[audit.s3]`) | the same audit rows as NDJSON batch objects — shipping them to a bucket is an explicit off-machine trust decision; credentials come from env vars or the workload's cloud identity (`auth = "identity"`), never the config file; failures WARN and drop (under `[audit] required`: spool from the audit DB, retry until confirmed — never drop) | values, placeholder ids, credentials |
| OpenTelemetry (opt-in) | same metadata as spans/counters — pointing `endpoint` at a remote collector is an explicit trust decision | values, headers, placeholder ids |
| Vault | — (never leaves the machine) | — |

## Realtime WebSocket connections

The same gates apply at the relay (`realtime.py`), shifted to WS
mechanics: a web page's handshake (gate ②¾ — browsers send `Origin` on
every upgrade and apply no CORS to it) is refused first, accept-then-close
**1008**, recorded as a 403; unknown paths, disabled providers, and a
missing `realtime` extra are refused accept-then-close **1011** (without the extra the
server cannot accept upgrades at all, so realtime traffic can never
silently bypass redaction); a refusal before the upstream dial — the
access gate's (403), an unusable provider's (502), an `[audit] required`
START that could not commit (503) — is recorded like its HTTP twin; the
upstream dial never follows a handshake redirect (a failed dial: 1011,
counted and recorded); a block-mode match closes **1008** carrying
the detector type only; headers, `?key=` queries, and subprotocols pass
through unlogged; and gate ⑤ walks **every** client event, skipping only
the scalar values of structural keys (enums, ids) and base64 audio —
objects under those names, and tool responses, `metadata` maps and
prompt `variables` in full, are walked. Before gate ⑤, a session router
with the optional `realtime_frame_refusal` (gate ④½'s realtime twin —
**llm-redact-pro** applies its stored-object and cloud-storage policy for
the connection's user) is asked about **every** client frame that parses
as JSON, text or binary, `detection = false` included: a refusal closes
**1008** with its fixed reason, recorded as a 403, and nothing of that
frame is redacted, numbered or sent; a check that fails closes the same
way (fail closed). While it checks frames, a frame nesting JSON too deep
is refused on every connection, and one that is not JSON under the
proxy's own identity. In the other direction a router with the optional
`realtime_server_frame` is handed every UPSTREAM frame that parses as
JSON — its own parse of the provider's bytes, before the frame is
restored or sent — read-only: it cannot change or refuse the frame, and a
fault is contained (the frame is delivered). **llm-redact-pro** records
there whose Live session a `sessionResumptionUpdate` handle belongs to,
so a later `setup` resuming it is judged on its owner.
The gates' inputs are the ones in force. A connection is admitted under
its provider's settings, the authorizer that opened it and the
`[detection]` policy. A reload that changes any of them revokes the relay
in the same synchronous step as the swap: it closes **1012** (reconnect)
and never forwards a client frame it reads after the reload. So `/status`
never reports a policy that an open connection is not following
([deployment.md](deployment.md#reloads-and-open-realtime-connections)).
Voice audio is never decoded or scanned, so speech reaches the provider unredacted (see the threat
model's media non-goal).
