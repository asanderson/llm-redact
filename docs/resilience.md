# Resilience & failure modes

The proxy sits in the live request path between an agentic tool and the
provider. Things fail there: the network drops mid-stream, the upstream times
out or 5xxs, a frame arrives truncated, the vault's disk fills. This document
is the catalogue of those faults and exactly how the proxy behaves under each.

Every entry is subordinate to the one rule in
[threat-model.md](threat-model.md): *private values must not reach the
provider, and the placeholder↔value mapping must never rehydrate to the wrong
secret.* Under fault that becomes three commitments, in priority order:

1. **Never a wrong value.** A fault must never cause a placeholder to restore
   to a different secret, or a partial token to be guessed into a value.
2. **Fail closed.** When the proxy cannot complete a request correctly, it
   errors — it never falls through to forwarding traffic unredacted.
3. **Don't break the tool more than the fault already did.** A recognizable
   error (a 502, a cut stream) beats a hang or a crash; an unrestored
   placeholder reaching the tool is safe where guessing is not.

These are enforced, not aspirational — each row below names the suite that
pins it.

## Upstream transport faults

| Fault | Behavior | Pinned by |
| --- | --- | --- |
| Connect refused / DNS / connect timeout | Proxy-generated **502** (provider-shaped body); the request is recorded and counted (`llm_redact_upstream_errors_total`). No upstream body existed, so nothing leaks. | `test_upstream_faults.py` |
| Read timeout / 5xx / mid-body drop on a **buffered** response | The open upstream response is closed, the request fails closed with a **502**, and the fault is recorded and counted. Without this the buffered `aread()` would surface a bare 500 and leak the connection. | `test_upstream_faults.py` |
| Mid-stream drop on a **streaming** response (SSE / eventstream / ndjson) | The bytes already sent are valid rehydrated output; the stream then errors (the tool sees a truncated response — honest, not a silent clean end). The generator's `finally` still closes the upstream and finalizes metrics/audit, even if `aclose()` itself raises on the broken stream. | `test_upstream_faults.py` |
| Upstream answers a **redirect** (3xx other than 304, with a `Location`) | Relayed only when the client's repeat of its original request carries nothing the proxy protects (first-party plain pass-through, or a body-less read of a recognized route, with the client's own credential — a download CDN). Otherwise — a body on a route the proxy redacts, a routed or identity-signed request, a presented proxy credential, a custom upstream — a following client would re-send its ORIGINAL request, unredacted and with its credential headers, to the `Location`: the upstream response is closed and the request answered with a recorded, provider-shaped **502** naming the status only (the `Location` is never relayed or logged: it can carry a presigned credential), counted as an upstream error; a routed hop that redirects is a hop fault the router can fail over from. | `test_upstream_redirects.py` |

The proxy's outbound client uses a generous timeout (`connect=10s`, overall
600s) so a slow-but-alive provider stream is not killed prematurely; genuine
faults surface as the transport errors above.

## Framing faults

| Fault | Behavior | Pinned by |
| --- | --- | --- |
| Malformed UTF-8 in an SSE line | The line is decoded with `errors="replace"` and forwarded; valid streams stay byte-identical. | `test_codec_fuzz.py` |
| Corrupt binary eventstream frame (bad CRC / length) | Degrade to **verbatim pass-through** of every unreturned byte and the rest of the stream — an unrestored placeholder is safe; guessing at a corrupt frame is not. | `test_provider_bedrock.py`, `test_eventstream.py` |
| ndjson line that is not valid JSON | Forwarded byte-identically. | `test_ndjson.py` |
| JSON body of very many tiny strings | Short strings are gated per string (only the rules whose literals or patterns can occur in one run on it — 300k tiny chat messages cost 1.3 s instead of 32 s of event-loop time), and a redactable body with more strings than `max_body_strings` (default 100,000; uploaded JSONL lines and multipart parts count too) is refused with a recorded, provider-shaped **413** before any upstream contact — counted as it is redacted, parts before any parse. | `test_body_string_cap.py`, `test_detector_plan.py` |
| Multipart body of very many tiny parts | Parsed in time linear in the body (each part located by offset — re-slicing the remainder per part once froze the event loop for minutes at `max_body_bytes`); more parts than `max_body_strings` is a recorded **413** before any parse (the delimiter count bounds the parts); where the scanned-body rule holds (below), a route that never scans multipart refuses it without parsing it at all. | `test_multipart.py`, `test_identity_body.py`, `test_body_string_cap.py` |
| A request body the proxy cannot read on a recognized route (content-encoded, bytes after the JSON value, invalid UTF-8, a top-level array or scalar, whitespace, multipart outside the canonical form or on a route that does not scan it, a second Content-Type; inside an upload a part it cannot scan) | A recorded, provider-shaped **400** — **415** with `Accept-Encoding: identity` for a content coding — naming the body's kind, before any upstream contact: wherever redaction applies, and under any credential the proxy holds whatever `detection` says. Forwarding it unread let a lenient upstream decode what the proxy never scanned. `detection = false` with the client's own key forwards it verbatim (the surfaced opt-out). | `test_scanned_body.py`, `test_identity_body.py`, `test_identity_multipart.py` |
| A JSON document nested deeper than `MAX_JSON_DEPTH` (128 levels of objects and arrays) | Every document the proxy reads is parsed through `jsonwalk.loads_bounded`/`loads_request`, which refuse one nesting deeper (also when the parser itself gives up: a RecursionError) — so no walk over a parsed document (redaction, MCP stripping, rehydration, re-serialization) can exhaust the interpreter's stack. A client's is unreadable: a request body, an uploaded JSONL line or JSON form field, a Bedrock `count-tokens` blob is a recorded, provider-shaped **400**, nothing forwarded (the stored-object check refuses an upload it cannot read under a credential the proxy holds); a realtime client frame closes the connection **1008** (recorded 400); a reserved endpoint's POST is a **400**. An upstream's is forwarded exactly as it came, its placeholders left in place: a buffered answer, an SSE event, an NDJSON line, a JSONL line of a file download or batch results, an eventstream payload, a realtime frame. A JWT-shaped value whose header is too deep to parse is judged by its opening (`{`: redacted). It was a bare, unrecorded 500 (`'{"messages":' + '[' * 200000`), a 502 or a cut stream. | `test_deep_json.py` |
| A JSON value holding a lone UTF-16 surrogate (a `\ud800`-style escape; surrogate-encoded bytes) | Every re-serialization (the redacted request, a restored answer, an SSE event, an NDJSON line, an eventstream frame, a realtime frame) writes it back as its escape (`jsonwalk.json_text` / `json_bytes`) — the same JSON value, every other character as before. It once failed the UTF-8 encode: an unrecorded bare 500 on the request path, a 502 or a cut stream on the answer path. | `test_lone_surrogates.py` |
| Stream **ends mid-token** (upstream closed after a prefix) | The partial placeholder held in the rehydrator buffer is flushed **verbatim** — never guessed into a value, never dropped. For every truncation point, `feed(prefix)+flush()` equals the non-streaming rehydration of exactly what arrived. | `test_stream_truncation.py` |

## Vault durability

The vault is the one piece of state whose corruption is unacceptable: a lost
or reused token number would silently rehydrate the *wrong* secret. It runs
with `synchronous=FULL` and WAL so a committed mapping survives a crash.

| Fault | Behavior | Pinned by |
| --- | --- | --- |
| Write fault mid-insert (disk full, I/O error, a cipher fault) | The open transaction is rolled back so the connection is not wedged for the next request, and the write fails closed. Caches are written only after `COMMIT` (nothing poisoned), and the number is `max(MAX(n), retired, floor)+1` with `MAX(n)` read fresh each call, so a retry of the same request reissues the **same number** — never a skipped number, never a reused token. | `test_vault_faults.py`, `test_vault_floors.py` |
| One request, many new values (the sqlite vault) | Written in **one** transaction — `BEGIN IMMEDIATE` at the first new value, one `COMMIT` (one fsync) before anything is forwarded; the request's own lookups see its uncommitted rows, the caches take them only after the `COMMIT`. A fault on any insert or on the `COMMIT`, a block-mode value, `max_body_strings`, even a write error swallowed inside the redaction: the whole transaction rolls back, nothing is cached or forwarded, and the retry reissues the same dense numbers. A vault fault (a failed insert or `COMMIT`; an RDBMS vault's driver error or an allocation that kept colliding; a read that fails while the session a request resolves to is opened) refuses the request with a recorded, provider-shaped **503** before any upstream contact — never a bare 500 — logged by exception type only and counted as `llm_redact_bookkeeping_errors_total{stage="vault"}`; a realtime frame's vault fault — or one opening the connection's session — closes the connection **1011**, recorded as 503. 10,000 new values in one body: ~0.3 s instead of ~2.5 s (10,000 fsyncs). | `test_vault_batch.py`, `test_vault_batch_e2e.py`, `test_vault_properties.py` |
| A session deleted while another instance, a live view or a provider still holds its tokens | The delete **retires** every number the session held (`retired_numbers`, in the delete's own transaction); new values are numbered above it, so a stale cache can only restore a deleted token to its own value, or pass it through. The deleting instance rebuilds every live view of the session at once (one view per session); other instances re-read the retired number within `CACHE_CHECK_SECONDS` (1 s) and drop the deleted values. A check that cannot read the database (an RDBMS blip, a failing sqlite file, a failed reconnect) keeps the view's caches and runs again a second later: restoring a cached token never needs the database. It is counted as `llm_redact_bookkeeping_errors_total{stage="vault_check"}` and logged once per outage, by exception type. | `test_vault_deletion.py`, `test_vault_properties.py`, `test_soak_concurrency.py` |
| A prune racing a new value in the same session | The idle check runs inside the delete's write transaction (sqlite `BEGIN IMMEDIATE`; a server RDBMS re-checks each session in the statement that sizes its delete and removes only the rows it saw idle) — a session that issued a value meanwhile is kept. | `test_vault_deletion.py`, `test_cli_vault.py` |
| A request carries tokens its session never issued (a compacted history, a pasted answer) | New values are numbered above every token the request carries (the token floor), so none of those names gains a second meaning; the skipped numbers are the only gaps the vault ever leaves. A request needing a number past 999999999 is refused 400 with nothing written. | `test_vault_floors.py`, `test_token_floors_e2e.py` |
| Concurrent writers (two proxies, one DB) | `PRAGMA busy_timeout` (5 s) waits on the WAL write lock — held for one request's redaction at most — and fails closed past it (nothing issued, nothing cached, the connection not wedged); a value the other writer mapped first is read inside the lock and keeps its token. | `test_vault_faults.py`, `test_vault_sqlite.py`, `test_vault_batch.py` |
| Crash between issue and use | The committed mapping is durable (WAL + `synchronous=FULL`); on reopen the counter continues from `MAX(n)` — issued tokens keep rehydrating, new values never reuse a number. | `test_vault_sqlite.py` |
| Wrong / missing encryption key | Fails closed **at open** — never silently issues fresh tokens against an unreadable store. | `test_vault_sqlite.py` |
| Corrupted at-rest ciphertext on a cold cache | `original_for` fails closed (raises) rather than returning a wrong or partial plaintext. | `test_vault_faults.py` |

## Audit write faults (`[audit] required`, Pro)

The default audit trail is fail-open (a write fault warns and continues —
see the vault rows above for why the vault is stricter). `[audit]
required = true` inverts that deliberately; its fault behavior:

| Fault | Behavior | Pinned by |
| --- | --- | --- |
| Write-ahead START row cannot be committed (disk full, IO error) | Provider-shaped **503 with ZERO upstream contact** — "no audit row, no service". Metrics and `/recent` still record the refusal. | `test_audit_required.py` |
| END-row write fails after the response is committed | Refusal is impossible; the fault logs **CRITICAL** (exception type only). The durable START row still witnesses the request. | `test_audit_required.py` (public seam) |
| Crash or kill between START and END | The next startup adopts every orphaned START as a synthetic chained `interrupted` row — a served request can lose its details, never its existence. Idempotent. | pro `test_audit_required_pro.py` |
| Off-machine sink upload fails / credentials or encryption key missing | Batches spool from the audit DB and the per-sink high-water mark does NOT advance — retained and retried (byte-identical), never dropped; `max_rows` pruning never deletes unshipped rows. | pro `test_audit_required_pro.py` |

## Faults after the upstream answered

Once the provider has answered, the proxy still restores the answer and
does session bookkeeping: it reports the response id and any stored
objects (an uploaded file, a `store: true` completion) to the session
router and mirrors them into the vault's durable map, and restores a
listing's items in their owners' sessions (llm-redact-pro named users).
A router exception, an RDBMS outage or a locked/failed sqlite write there
must not undo an answer the provider already produced — and billed.

| Fault | Behavior | Pinned by |
| --- | --- | --- |
| Recording a response id or stored object fails (router or durable map) | Contained: the answer is delivered (buffered or streamed — the stream is never cut), the fault logged by stage and exception type and counted (`llm_redact_bookkeeping_errors_total{stage}`, `/status` `bookkeeping_errors_total`). The object stays unattributed — the router's unknown-object case (an empty session: placeholders pass through), never a wrong value. | `test_bookkeeping_faults.py` |
| An open connection's access re-check fails (the access gate's recheck raises — a `CancelledError` from an awaitable another path cancelled included —, times out after min(`recheck_interval`, 10 s), or answers neither a bool, None nor a string) | Fail closed: the realtime relay closes its client with 1008 and its upstream with 1000 (its row stays 101), or the live-events stream ends. Logged by exception type only (never the gate's reason, a grant or a subject), and counted as `llm_redact_bookkeeping_errors_total{stage="recheck"}` and `llm_redact_connections_closed_total{cause="recheck_error"}`. Checks run concurrently, one pass at a time, and a pass that fails never stops the next one: only the backstop's own cancellation (shutdown) ends it, and a backstop task that ended anyway is started again with the next connection that carries a recheck. | `test_connection_control.py` |
| The session router's realtime frame check fails (`realtime_frame_refusal` raises, or answers anything but None or a non-empty string) | Fail closed: the relay closes its client with 1008 and the core's fixed reason and its upstream with 1000; nothing of the frame is redacted, numbered or sent; the connection's row records 403. Logged by exception (or answer) type only, counted as `llm_redact_bookkeeping_errors_total{stage="realtime_frame"}`. | `test_realtime_frame_check.py` |
| The session router's response observer fails (`response_observer` or the observer it returned raises) | Contained: the answer is delivered unchanged (buffered or streamed — the stream is never cut; the observer only ever holds its own parse of the provider's bytes), the fault logged by exception type and counted as `response_observer`, and that answer is observed no further. What the router would have learned stays unknown — its own fail-closed case (llm-redact-pro seals a request that cites a compaction it never saw). | `test_response_observer.py` |
| A listed item's owner session cannot be read | That item is delivered exactly as the provider sent it (placeholders in place) — never what the request's own session would make of it; counted as `listing`. | `test_bookkeeping_faults.py` |
| The session router's listing lookup fails (it raises, or its batched answer miscounts the items) | Every item it failed for — the whole listing when the batched answer failed — is delivered exactly as the provider sent it (placeholders in place): a router that cannot answer vouches for nothing, and the listing's own session may hold other values under the same token names. Logged by exception type, counted as `listing`. | `test_object_access_seams.py` |
| Restoring a **buffered** answer fails (a vault read that cannot complete, a router hook) | A recorded, provider-shaped **502** — never a bare 500, never a partial or unrestored body — its `[audit] required` END row finalized; counted as `delivery`. | `test_bookkeeping_faults.py` |
| Restoring a **streamed** answer fails | The stream is cut (its status already went out — the honest signal, as for an upstream drop); counted as `delivery`, and the row and audit END are still finalized — booked as the proxy's 502, never the upstream's 200 (a routed stream closes as `stream_error`). | `test_bookkeeping_faults.py` |

## Concurrency

Distinct conversations share the same token *names* (`«EMAIL_001»` exists in
every session), so any cross-session confusion would restore another
conversation's secret. Isolation is by construction — there is no fallback
lookup across sessions.

| Property | Behavior | Pinned by |
| --- | --- | --- |
| Many concurrent distinct sessions | Each request restores only its own session's values; no bleed through the shared token name. | `test_soak_concurrency.py` |
| Concurrent writes in one session | Distinct secrets get distinct dense tokens (no token floor involved); no counter collision. | `test_soak_concurrency.py` |
| More sessions than the view cache holds | The per-session view cache stays bounded (LRU); eviction drops only caches, never a mapping — every evicted session still rehydrates its own value. | `test_soak_concurrency.py` |

Run the concurrency/soak suite explicitly: `uv run pytest -m soak` (it is
deselected from the default run and runs as its own CI step).

## Observability of faults

- **`llm_redact_upstream_errors_total{provider}`** counts transport faults
  failed closed as 502. The `LlmRedactUpstreamErrors` Prometheus alert
  ([deploy/prometheus-alerts.yml](../deploy/prometheus-alerts.yml)) fires on a
  sustained rate. `/status` exposes the same as `upstream_errors_total`.
- **`llm_redact_bookkeeping_errors_total{stage}`** counts faults after the
  upstream answered (see above), as `stage="vault"` the vault faults that
  refused a request 503 before any upstream contact (see Vault
  durability), and as `stage="vault_check"` the staleness checks that could
  not read the vault's database (contained); `/status` exposes the same as
  `bookkeeping_errors_total`.
- Every fault path still emits a `record_request` row, so 502s appear in
  `/__llm-redact/recent`, the metrics `requests_total{status="502"}` series,
  and the audit log — a fault is never invisible.

## Deliberate non-goals under fault

- A fault during a streaming response cannot retroactively change an
  already-sent 200 status; the stream is cut instead. The tool sees a
  truncated response, which is the honest signal.
- Media (image/audio) bytes are never decoded, so a corrupt media payload is
  forwarded verbatim like any other opaque body — the text-only redaction
  scope is unchanged by faults.
