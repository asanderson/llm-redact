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

The proxy's outbound client uses a generous timeout (`connect=10s`, overall
600s) so a slow-but-alive provider stream is not killed prematurely; genuine
faults surface as the transport errors above.

## Framing faults

| Fault | Behavior | Pinned by |
| --- | --- | --- |
| Malformed UTF-8 in an SSE line | The line is decoded with `errors="replace"` and forwarded; valid streams stay byte-identical. | `test_codec_fuzz.py` |
| Corrupt binary eventstream frame (bad CRC / length) | Degrade to **verbatim pass-through** of every unreturned byte and the rest of the stream — an unrestored placeholder is safe; guessing at a corrupt frame is not. | `test_provider_bedrock.py`, `test_eventstream.py` |
| ndjson line that is not valid JSON | Forwarded byte-identically. | `test_ndjson.py` |
| Multipart body of very many tiny parts | Parsed in time linear in the body (each part located by offset — re-slicing the remainder per part once froze the event loop for minutes at `max_body_bytes`); under identity auth a route that never scans multipart refuses it without parsing it at all. | `test_multipart.py`, `test_identity_body.py` |
| Stream **ends mid-token** (upstream closed after a prefix) | The partial placeholder held in the rehydrator buffer is flushed **verbatim** — never guessed into a value, never dropped. For every truncation point, `feed(prefix)+flush()` equals the non-streaming rehydration of exactly what arrived. | `test_stream_truncation.py` |

## Vault durability

The vault is the one piece of state whose corruption is unacceptable: a lost
or reused token number would silently rehydrate the *wrong* secret. It runs
with `synchronous=FULL` and WAL so a committed mapping survives a crash.

| Fault | Behavior | Pinned by |
| --- | --- | --- |
| Write fault mid-insert (disk full, I/O error, a cipher fault) | The open transaction is rolled back so the connection is not wedged for the next request, and the write fails closed. Caches are written only after `COMMIT` (nothing poisoned), and the number is `max(MAX(n), floor)+1` with `MAX(n)` read fresh each call, so a retry of the same request reissues the **same number** — never a skipped number, never a reused token. | `test_vault_faults.py`, `test_vault_floors.py` |
| A request carries tokens its session never issued (a compacted history, a pasted answer) | New values are numbered above every token the request carries (the token floor), so none of those names gains a second meaning; the skipped numbers are the only gaps the vault ever leaves. A request needing a number past 999999999 is refused 400 with nothing written. | `test_vault_floors.py`, `test_token_floors_e2e.py` |
| Concurrent writers (two proxies, one DB) | `PRAGMA busy_timeout` waits on a briefly-held WAL write lock; the unique-constraint loser re-selects the winner's placeholder. | `test_vault_faults.py`, `test_vault_sqlite.py` |
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
| A listed item's owner session cannot be read | That item is delivered exactly as the provider sent it (placeholders in place) — never what the request's own session would make of it; counted as `listing`. | `test_bookkeeping_faults.py` |
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
  upstream answered (see above); `/status` exposes the same as
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
