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

## Upload inspector faults (document extraction)

An upload inspector (`plugin_api.UploadInspector`) reads an upload's
binary file parts as text before redaction. The core never trusts it with
more than that: whatever goes wrong, the part counts as NOT read and keeps
the unscanned-binary rules — forwarded unscanned only with the client's
own key under `[detection] binary_uploads = "forward"` (counted), else a
recorded, provider-shaped **400** — never a pass on a part it did not read.

| Fault | Behavior | Pinned by |
| --- | --- | --- |
| The inspector raises, or answers anything but an `Inspection` | Outcome `error` (logged by exception type only, never content); the unscanned-binary rules. | `test_upload_inspection.py` |
| The inspections outlast the inspector's declared `timeout` (capped at 300 s, one deadline per request) | What still runs is cancelled and never awaited again (a result arriving later is discarded); outcome `timeout`; the unscanned-binary rules. | `test_upload_inspection.py` |
| A part over the inspector's `max_bytes`, or past 16 parts | Never handed over; outcome `not_inspected`; the unscanned-binary rules. | `test_upload_inspection.py` |
| An incomplete reading, no text, or more extracted text than the request's budgets (`max_body_bytes` characters, `max_body_strings` strings — the latter a **413**) | Outcome `incomplete`: a value found still refuses the upload, a clean scan does not clear the part. | `test_upload_inspection.py` |
| The request is cancelled while its inspections run | Every inspection is cancelled with it. | `test_upload_inspection.py` |
| The inspector's declared `timeout` or `max_bytes` is missing or not a positive number | Startup refuses (ConfigError): the bounds are never left to chance. | `test_upload_inspection.py` |

## Vault durability

The vault is the one piece of state whose corruption is unacceptable: a lost
or reused token number would silently rehydrate the *wrong* secret. It runs
with `synchronous=FULL` and WAL so a committed mapping survives a crash.

| Fault | Behavior | Pinned by |
| --- | --- | --- |
| Write fault mid-insert (disk full, I/O error, a cipher fault) | The open transaction is rolled back so the connection is not wedged for the next request, and the write fails closed. Caches are written only after `COMMIT` (nothing poisoned), and the number is `max(MAX(n), retired, floor)+1` with `MAX(n)` read fresh each call, so a retry of the same request reissues the **same number** — never a skipped number, never a reused token. | `test_vault_faults.py`, `test_vault_floors.py` |
| One request, many new values (the sqlite vault) | Written in **one** transaction — `BEGIN IMMEDIATE` at the first new value, one `COMMIT` (one fsync) before anything is forwarded; the request's own lookups see its uncommitted rows, the caches take them only after the `COMMIT`. A fault on any insert or on the `COMMIT`, a block-mode value, `max_body_strings`, even a write error swallowed inside the redaction: the whole transaction rolls back, nothing is cached or forwarded, and the retry reissues the same dense numbers. A vault fault (a failed insert or `COMMIT`; an RDBMS vault's driver error or an allocation that kept colliding; a read that fails while the session a request resolves to is opened) refuses the request with a recorded, provider-shaped **503** before any upstream contact — never a bare 500 — logged by exception type only and counted as `llm_redact_bookkeeping_errors_total{stage="vault"}`; a realtime frame's vault fault — or one opening the connection's session — closes the connection **1011**, recorded as 503. 10,000 new values in one body: ~0.3 s instead of ~2.5 s (10,000 fsyncs). | `test_vault_batch.py`, `test_vault_batch_e2e.py`, `test_vault_properties.py` |
| A session deleted while another instance, a live view or a provider still holds its tokens | The delete **retires** every number the session held (`retired_numbers`, in the delete's own transaction); new values are numbered above it, so a stale cache can only restore a deleted token to its own value, or pass it through. The deleting instance rebuilds every live view of the session at once (one view per session); other instances re-read the retired number within `CACHE_CHECK_SECONDS` (1 s) and drop the deleted values. A check that cannot read the database (an RDBMS blip, a failing sqlite file, a failed reconnect) keeps the view's caches and runs again a second later: restoring a cached token never needs the database. It is counted as `llm_redact_bookkeeping_errors_total{stage="vault_check"}` and logged once per outage, by exception type. | `test_vault_deletion.py`, `test_vault_properties.py`, `test_soak_concurrency.py` |
| The Live resumption handle map cannot be written or read (a full disk, an RDBMS blip; `record_handle_session` / `lookup_handle_session`, called by llm-redact-pro) | Contained in the vault manager: a failed write records nothing (its `replaces` deletes roll back with it) and a failed read answers None, so the handle reads as unknown and the plugin refuses the resumption — never resumes it into a session the vault no longer vouches for. Counted as `llm_redact_bookkeeping_errors_total{stage="handle_map"}` and logged once per outage, by exception type — reads and writes as separate outages (writes failing on a missing grant while reads work log one WARNING, never a false recovery); the frame that carried the handle is delivered. A whole-session delete removes the session's handle rows in its own transaction. | `test_vault_handles.py` |
| A prune racing a new value in the same session | The idle check runs inside the delete's write transaction (sqlite `BEGIN IMMEDIATE`; a server RDBMS re-checks each session in the statement that sizes its delete and removes only the rows it saw idle) — a session that issued a value meanwhile is kept. | `test_vault_deletion.py`, `test_cli_vault.py` |
| A request carries tokens its session never issued (a compacted history, a pasted answer) | New values are numbered above every token the request carries (the token floor), so none of those names gains a second meaning; the skipped numbers are the only gaps the vault ever leaves. A request needing a number past 999999999 is refused 400 with nothing written. | `test_vault_floors.py`, `test_token_floors_e2e.py` |
| Concurrent writers (two proxies, one DB) | `PRAGMA busy_timeout` (5 s) waits on the WAL write lock — held for one request's redaction at most — and fails closed past it (nothing issued, nothing cached, the connection not wedged); a value the other writer mapped first is read inside the lock and keeps its token. | `test_vault_faults.py`, `test_vault_sqlite.py`, `test_vault_batch.py` |
| Crash between issue and use | The committed mapping is durable (WAL + `synchronous=FULL`); on reopen the counter continues from `MAX(n)` — issued tokens keep rehydrating, new values never reuse a number. | `test_vault_sqlite.py` |
| Wrong / missing encryption key | Fails closed **at open** — never silently issues fresh tokens against an unreadable store. | `test_vault_sqlite.py` |
| Corrupted at-rest ciphertext on a cold cache | `original_for` fails closed (raises) rather than returning a wrong or partial plaintext. | `test_vault_faults.py` |

## Durable map writes off the event loop

After the provider answered, the proxy records which session a Responses
chain, a stored object (its owner record) and — for llm-redact-pro — a Live
resumption handle belong to (`record_response_session`,
`record_object_session`, `record_handle_session`). Each is a database write
(an fsync on sqlite, a round trip on a remote RDBMS). The proxy no longer
makes them on the event loop: the sqlite and RDBMS vault managers hand them
to one background writer thread per manager (`vault_writer.MapWriter`,
switched on at startup by the manager's optional `write_maps_in_background`),
which writes over its OWN connection — never the proxy's shared one, never
inside a request's open `run_batched` transaction — in submission order.

| Fault / situation | Behavior | Pinned by |
| --- | --- | --- |
| The map's database is slow (a slow disk, a remote RDBMS round trip) | Other requests are answered while the write is still held up: the event loop only queues it. A client that sends `previous_response_id`, or resumes a Live handle, right after the answer is served exactly as before — until the write lands, every lookup (`lookup_response_session[s]`, `lookup_handle_session`) answers from the writer's in-process overlay (a superseded handle already reads as unknown). The writer's sqlite connection runs `synchronous=NORMAL`: a map row lost to a power failure reads as unknown (refused or sealed), never as a wrong value — unlike a mapping. A sqlite redaction transaction may still wait on the writer's write lock for the length of one map write. | `test_vault_writer.py` |
| A map write fails (disk full, I/O error, an RDBMS blip, a collision past its retries) | Contained in the writer: nothing is written, the record reads as unknown — as a failed synchronous write did (an orphan session for a Responses chain, the router's unknown-object case, a refused Live resumption), never a wrong value. Counted under the write's own stage (`llm_redact_bookkeeping_errors_total{stage="response_id"|"object_ids"|"handle_map"}`) and logged once per outage and on recovery, by exception type only. Outcomes are posted to the event loop, so counters are only touched there. | `test_vault_writer.py` |
| A session is deleted (TTL prune, the prune endpoint, `SessionStore.forget`) while its writes are queued | The delete never waits for the writer (it runs on the event loop; a write held up on the writer's own connection never holds it). It turns every queued write of a deleted session into an ERASE of its key, erases a write of that session still in flight again right after it ran, and the overlay answers "absent" at once; the writer settles no write while a delete is in progress. The map reads exactly as if the writes had landed before the delete: a chain or handle into a pruned-and-recreated session stays unknown. | `test_vault_writer.py` |
| The writer falls behind (more than 10,000 writes waiting) | Bounded: the write is not queued — counted under its stage, logged once per episode — and kept in the overlay only (at most 10,000 such records, the oldest forgotten first), so this process still answers for it. After a restart, or on another replica, it reads as unknown: refused or sealed, never a wrong value. | `test_vault_writer.py` |
| Several replicas share one vault (the Helm standalone mode with autoscaling, any load-balanced set) | Read-your-writes holds within ONE process only. Another replica (or the database itself) sees a record once its write landed — normally milliseconds after the answer, but a buffered answer's row is no longer committed before the client receives the bytes. A `previous_response_id` continuation or a Live resumption that reaches another replica first reads the record as unknown: refused or sealed (an orphan session, a refused resumption), never a wrong value. A record past the 10,000-write bound is never written, so for every other replica it stays unknown for good. Route a conversation's follow-ups to the replica that answered (session affinity) where that availability matters. | `test_vault_writer.py`, llm-redact-pro `test_vault_background_writes_e2e.py` |
| Shutdown with writes queued | The lifespan waits (off the loop, at most 5 s) for them to land before the vault closes; what is still queued then is dropped, and a write still in flight is given up on once its thread has not finished within 1 s more — each counted under its write's stage (once: a given-up write's own late outcome is never counted again) and logged by NUMBER only — and reads as unknown after the restart. A drain run earlier (mid-life) never stops the final close from draining: only a drain that ran out of time with no write landing since (a stuck writer) is not waited for twice. | `test_vault_writer.py` |

Lookups (the reads) still run on the event loop; only the writes moved.

## Audit write faults (`[audit] required`, Pro)

The default audit trail is fail-open (a write fault warns and continues —
see the vault rows above for why the vault is stricter). `[audit]
required = true` inverts that deliberately; its fault behavior:

| Fault | Behavior | Pinned by |
| --- | --- | --- |
| Write-ahead START row cannot be committed (disk full, IO error), or the log's `begin` answers an awaitable (nothing is committed yet and the core never awaits it — a coroutine is closed unrun, a pending asyncio Task or Future cancelled, so no START row lands after the refusal; a member declared `async def` — `begin`, `finalize` or `amend` — refuses startup instead, a ConfigError `serve --check` reports) | Provider-shaped **503 with ZERO upstream contact** — "no audit row, no service" (a realtime upgrade: accept-then-close 1011, row 503, never dialled). Metrics and `/recent` still record the refusal. | `test_audit_required.py`, `test_realtime_refusal_rows.py` |
| An inspected upload's START row (written before its inspection) cannot be amended with the counts the redaction found (a log with the optional `amend` member; an `amend` that answers an awaitable counts as one that cannot commit — it is closed unrun), or — a log without it — the second START row carrying them cannot be committed | The same provider-shaped **503 with ZERO upstream contact**; the refusal's END row closes the START row, and a part the inspection cleared counts as `clean_refused`, never `clean`. With `amend` the request keeps ONE START row (START, its amendment, END). | `test_upload_fate.py` |
| END-row write fails after the response is committed, or the log's `finalize` answers an awaitable (closed unrun, or a pending Task or Future cancelled — never awaited, never a late END row: the same fault) | Refusal is impossible; the fault logs **CRITICAL** (exception type only). The durable START row still witnesses the request. | `test_audit_required.py` (public seam), `test_realtime_refusal_rows.py` |
| Crash or kill between START and END | The next startup adopts every orphaned START as a synthetic chained `interrupted` row — a served request can lose its details, never its existence. Idempotent. | pro `test_audit_required_pro.py` |
| Off-machine sink upload fails / credentials or encryption key missing | Batches spool from the audit DB and the per-sink high-water mark does NOT advance — retained and retried (byte-identical), never dropped; `max_rows` pruning never deletes unshipped rows. The START and AMEND rows the sinks also ship travel through a bounded in-memory buffer instead: a crash, a kill or a failed final upload can lose them, and past 10,000 queued rows they drop oldest-first, counted in `rows_dropped`. | pro `test_audit_required_pro.py` |

## Shutdown order

A clean stop (SIGTERM / Ctrl-C) drains in this order:

1. **Stop accepting.** The server closes its listening socket.
2. **In-flight requests finish.** uvicorn waits for every open connection
   to end before it runs the application's shutdown (`serve` sets no
   graceful-shutdown timeout, so a request is never cut short to make
   room for it), so every request finalizer — buffered, streaming,
   realtime — has written its END audit row first. Open realtime relays
   are closed by the server (1012), and the dashboard's
   `/__llm-redact/events` streams, which never end on their own, are
   ended by the proxy as shutdown starts (`serving.ProxyServer`; before,
   an open dashboard held the drain until the supervisor killed the
   process). This wait has no limit of its own: a streamed answer still
   running holds it until it ends. The connection re-check backstop then
   stops.
3. **Background work stops.** The sinks' periodic flush loops, the
   session-TTL prune and the license refresh are cancelled (a loop that
   had already died is logged by exception type and never cuts the
   shutdown short); the upstream client, router, upstream authorizers,
   access gate and upload inspector close (a plugin's close that fails is
   logged by exception type and never skips the steps below). The vault's
   background map writes (Responses chains, stored-object owners, Live
   handles) are drained, for at most 5 s; what is still unwritten then is
   counted and dropped when the vault closes (step 5).
4. **The off-machine audit sinks flush from the still-open audit
   database** (`aclose()` of `[audit.s3]` and `[audit.azure]`,
   concurrently): the rows spooled since the last upload, including the
   END rows of the last requests, ship now instead of at the next start.
5. **The audit database closes**, then the overrides store, then **the
   vault**, and the telemetry exporters flush last.

Step 4 is bounded so a hanging store never keeps the databases open: both
flushes share one 45 s deadline. Each upload keeps the sink's own 30 s
HTTP timeout, and the deadline sits above it on purpose: the final flush
uploads the sink's in-memory START/AMEND rows first, and rows it could
not ship then are lost from the off-machine copy, so a slow but working
store is never cut short by the core — only a stuck sink, or a long
backlog drain whose spooled rows wait in the database anyway, reaches the
deadline. A flush still running then is cancelled — logged as a
WARNING with the count only — and its unshipped spooled rows stay in the
audit database for the next start (the per-sink mark advances only after a
confirmed upload); a flush that ignores the cancellation for another
second is abandoned and the audit database and the vault close anyway.
What is bounded is reaching those closes, not the process exit: Python's
event-loop teardown cancels and then awaits every task still running, so
an abandoned flush that keeps ignoring its cancellation holds the process
open until the supervisor kills it — with both databases already closed,
nothing is lost then beyond what that flush had not shipped. A flush that
fails is logged by exception type only (its message may quote a URL or a
SAS).

**Shutdown budget.** Once the in-flight requests have finished, the
bounded steps take at most 57 s: the map-write drain (5 s), the sinks'
final flush (45 s + 1 s cancel grace) and the vault close (which drains
the map writes once more, for up to 5 s, only when a write landed after
the first drain, and then waits 1 s for the writer thread). Allow at least
the request drain plus ~60 s in a supervisor's stop timeout so the final
flush is not killed before the databases close; the request drain itself
has no bound (step 2). Kubernetes' default `terminationGracePeriodSeconds`
is 30 s: the Helm chart (`deploy/helm/llm-redact`) sets its own,
`terminationGracePeriodSeconds`, which defaults to 90 s — the 57 s plus
33 s for requests still running — on the pod in both modes (in sidecar
mode the pod's tool container shares it), validated as a non-negative
integer, with a NOTES warning below 57 s. The pod goes away as soon as the
proxy exits, so a fast stop never waits for it. systemd's default
`TimeoutStopSec` is the same 90 s. A plain manifest such as
`deploy/k8s-sidecar.yaml` keeps Kubernetes' 30 s unless you set it. No
`preStop` hook is needed for the drain: the proxy acts on SIGTERM itself.

The order is pinned by `test_shutdown_order.py` (order, the END row of a
request in flight at shutdown reaching the sink, an open events stream, a
hanging sink, a failing flush); the budget by `test_deploy_assets.py`,
which recomputes it from the constants in `proxy.py` and `vault_writer.py`
and fails when the chart's default or its NOTES threshold falls behind.

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
| Recording a response id or stored object fails (router or durable map; a background map write's fault: see Durable map writes off the event loop) | Contained: the answer is delivered (buffered or streamed — the stream is never cut), the fault logged by stage and exception type and counted (`llm_redact_bookkeeping_errors_total{stage}`, `/status` `bookkeeping_errors_total`). The object stays unattributed — the router's unknown-object case (an empty session: placeholders pass through), never a wrong value. | `test_bookkeeping_faults.py` |
| An open connection's access re-check fails (the access gate's recheck raises — a `CancelledError` from an awaitable another path cancelled included —, times out after min(`recheck_interval`, 10 s), or answers neither a bool, None nor a string) | Fail closed: the realtime relay closes its client with 1008 and its upstream with 1000 (its row stays 101), or the live-events stream ends. Logged by exception type only (never the gate's reason, a grant or a subject), and counted as `llm_redact_bookkeeping_errors_total{stage="recheck"}` and `llm_redact_connections_closed_total{cause="recheck_error"}`. Checks run concurrently, one pass at a time, and a pass that fails never stops the next one: only the backstop's own cancellation (shutdown) ends it, and a backstop task that ended anyway is started again with the next connection that carries a recheck. | `test_connection_control.py` |
| The session router's realtime frame check fails (`realtime_frame_refusal` raises, or answers anything but None or a non-empty string) | Fail closed: the relay closes its client with 1008 and the core's fixed reason and its upstream with 1000; nothing of the frame is redacted, numbered or sent; the connection's row records 403. Logged by exception (or answer) type only, counted as `llm_redact_bookkeeping_errors_total{stage="realtime_frame"}`. | `test_realtime_frame_check.py` |
| The session router's realtime server-frame observer fails (`realtime_server_frame` raises) | Contained: the upstream frame is restored and delivered as usual (the observer only ever holds its own parse of the provider's bytes, before restoration) and the connection stays open. Logged by exception type only, counted as `llm_redact_bookkeeping_errors_total{stage="realtime_server_frame"}`. What the router would have recorded stays unknown — its fail-closed case (llm-redact-pro refuses a Live `setup` resuming a handle it never recorded). | `test_realtime_server_frame.py` |
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
