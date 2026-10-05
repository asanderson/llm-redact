# Observability: Prometheus + Grafana

The proxy exposes Prometheus metrics at `/__llm-redact/metrics` (always on,
no config). Everything there is **metadata only** — request counts by
provider and status, the proxy's own refusals by kind, detection /
warning / block / rehydration counts by detector *type*, request duration
and the proxy's own overhead, vault entry/session gauges and its map
writer, audit-sink batches, compaction-fork counts, routing decisions and
re-issues by upstream and rule name, NER coverage (strings read, skipped
and entities dropped, by backend), a plugin's own gauges (llm-redact-pro:
users and seats), and build info. It never carries secret values or placeholder ids (see
[threat-model.md](threat-model.md) § Logging posture), so scraping it is safe.

Ready-to-use assets live in [`deploy/`](../deploy):

| File | What it is |
|---|---|
| `deploy/prometheus-scrape.yml` | A `scrape_configs` job to merge into your `prometheus.yml`. |
| `deploy/prometheus-alerts.yml` | Alerting rules (proxy down, warn-mode value forwarding, high block rate, compaction forks, upstream errors, high proxy-overhead p95, refusals caused by the proxy's own storage, NER skipping strings longer than `max_chars`). |
| `deploy/grafana-dashboard.json` | An importable Grafana dashboard (traffic, end-to-end duration, detections/warnings/blocks by type, vault + compaction, unscanned uploads, proxy overhead, local refusals by kind, vault map writes, NER coverage: strings by outcome, strings skipped as longer than `max_chars`, windows and truncated windows, entities dropped). |

## Metrics reference

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `llm_redact_requests_total` | counter | `provider`, `status` | Requests proxied — the provider's answers AND the proxy's own refusals under one `status` label (`llm_redact_local_refusals_total` separates the latter). |
| `llm_redact_request_duration_seconds` | histogram | `provider`, `streamed` | END-TO-END duration: from admission until the answer is handed to the server (a buffered answer or a local refusal: when its whole body is handed over, before the server writes it; a stream: when the server has taken its last chunk), so it INCLUDES the provider's own time (a model thinking, a long stream) and, on a stream, the client's pace. Not a proxy-health signal on its own: alert on `llm_redact_proxy_overhead_seconds`. |
| `llm_redact_proxy_overhead_seconds` | histogram | `provider` | The proxy's OWN time per HTTP request: the request duration minus every wait on the upstream (sending the request, its answer's headers and body, each streamed chunk, a routed retry's delay) and on the client (reading its request body, a stream's consumer taking each chunk). What remains is redaction, rehydration, vault and audit writes, a `[vault] map_writes = "before_answer"` wait, an upload inspection and an identity authorizer's credential fetch, plus event-loop scheduling under load. Buckets from 1 ms to 5 s. Realtime connections are not observed (their duration is the connection's life). |
| `llm_redact_local_refusals_total` | counter | `kind`, `provider` | Responses the PROXY ITSELF generated instead of forwarding the request (realtime: instead of relaying the connection or a frame), once per request, by kind (the table below) and provider (`passthrough` when none was attributed, as in `llm_redact_requests_total`). An upstream's own answer is never counted, whatever its status. Also `/status` `local_refusals_total` (by kind). |
| `llm_redact_detections_total` | counter | `type` | Values redacted, by placeholder type. |
| `llm_redact_warnings_total` | counter | `type` | Warn-mode hits — **the value was forwarded upstream**. |
| `llm_redact_blocked_total` | counter | `type` | Requests rejected 400 by block-mode rules. |
| `llm_redact_rehydrations_total` | counter | `type` | Placeholders restored in responses. |
| `llm_redact_compaction_forks_total` | counter | — | Per-conversation sessions forked by history compaction. |
| `llm_redact_unscanned_uploads_total` | counter | `provider` | Binary file parts of uploads (a PDF, an image, an archive) **forwarded unscanned** with the client's own key under `[detection] binary_uploads = "forward"` (the default) — counted only once the upload was handed to the upstream. Also `/status` `unscanned_uploads_total` and a `llm-redact status` posture line. |
| `llm_redact_inspected_uploads_total` | counter | `provider`, `outcome` | Binary file parts read as text by an upload inspector (the core's `[extraction]`, docs/extraction.md), one outcome each: `clean` (sent byte-identical after a clean scan of the EXTRACTED text only — counted once the upload was handed to the upstream), `clean_refused` (scanned clean, the upload refused before any upstream contact), `converted`/`converted_refused` (replaced by its redacted extracted text in convert mode; sent or refused), `overridden`/`overridden_refused` (clean only because an approved refusal override let its values through: sent byte-identical WITH them, or refused — never counted `clean`), `detected`, `blocked`, `incomplete`, `not_inspected`, `timeout`, `error`. Also `/status` `inspected_uploads_total`. |
| `llm_redact_ner_strings_total` | counter | `backend`, `outcome` | Strings handed to an NER backend (`[detection.ner]`, docs/detection.md "NER coverage counters"), one outcome each: `scanned_whole` (the model read it in one call), `scanned_windowed` (in overlapping windows), `skipped_max_chars` (longer than `[detection.ner] max_chars`: **the model never read it** — the regex rules and deny strings still did). A sustained `skipped_max_chars` rate is an NER coverage gap. Rows only for the running backends; with NER off the family is empty. Also `/status` `detection.ner`, and a `llm-redact status` posture line while non-zero. |
| `llm_redact_ner_windows_total` | counter | `backend` | The windows the `scanned_windowed` strings were read in (model calls for them). |
| `llm_redact_ner_windows_truncated_total` | counter | `backend` | Windows (a string read whole counts as one) holding a single word longer than the model's encoder reads (a very long identifier), which the model may read only in part. |
| `llm_redact_ner_labels_dropped_total` | counter | `backend` | Model entities never emitted because their type cannot be a placeholder type (it starts with a digit, or is longer than 28 characters). |
| `llm_redact_ner_offsets_dropped_total` | counter | `backend` | Model entities of a requested type never redacted because the scanned string does not contain their span (or they came without one). |
| `llm_redact_overrides_used_total` | counter | `kind` | Requests (and realtime frames) that passed a refusal on an approved refusal override (docs/overrides.md), by `once`/`always` — **the overridden value, body or file part was forwarded upstream as sent**, like a warn-mode value. Also `/status` `overrides`. |
| `llm_redact_upstream_errors_total` | counter | `provider` | Transport faults failed closed as 502 (by upstream name when routing is enabled). |
| `llm_redact_bookkeeping_errors_total` | counter | `stage` | Faults in the proxy's own bookkeeping. After the upstream answered: session bookkeeping (`response_id`, `object_ids`, `listing`, and `response_observer` — a session router observing the answer — contained, the answer is still delivered) and `delivery` (restoring the answer failed: a buffered one is a recorded 502, a stream is cut). Before any upstream contact: `vault` (issuing a request's placeholders failed — a recorded 503; a realtime frame closes the connection 1011). `vault_check`: a vault view's staleness check could not read its database (contained: the view keeps serving its cache — a cached token only ever restores its own value — and checks again a second later). `recheck`: an open realtime relay's or live-events stream's access re-check raised, timed out or gave an answer that makes no sense (fail closed: the connection is closed). `realtime_frame`: the session router's per-frame realtime check (`realtime_frame_refusal`) raised or gave an answer that makes no sense (fail closed: the connection is closed 1008, the frame never sent). `realtime_server_frame`: the session router's observation of a realtime UPSTREAM frame (`realtime_server_frame`) raised (contained: the frame is delivered; what the router would have recorded stays unknown — llm-redact-pro then refuses a Live resumption it cannot attribute). `handle_map`: the vault's Live resumption handle map (written and read by llm-redact-pro) could not be written or read (contained: a failed write records nothing, a failed read answers unknown — the resumption is refused; logged once per outage, by exception type). `map_write_wait`: a `before_answer` answer sent before its map writes landed (the same answers as `llm_redact_map_write_wait_timeouts_total`, which says why). `authorization`: the access gate's optional request authorization (`authorize_request`) raised, timed out or gave an answer that makes no sense, or its detection overlay (`detection_overlay`) raised or could not be applied, or its check of what the redaction found (`authorize_content`) raised, timed out or gave an answer that makes no sense (fail closed: the request is refused 403, a realtime upgrade or frame 1008). `gate_reload`: on a configuration reload (SIGHUP or a dashboard edit) the access gate's optional policy reload (`reload`) kept its previous policy (its reason logged at WARNING, escaped and cut to 200 characters), raised or gave an answer that makes no sense (logged by type) — contained: the rest of the reload applies and the gate keeps whatever policy it holds. `plugin_metrics`: a plugin's metrics samples could not be read (a fault, a timeout, a call still running) or a sample was invalid and dropped. |
| `llm_redact_connections_closed_total` | counter | `cause` | Open long-lived connections (realtime relays, the dashboard's live-events stream) closed because their admission ended: `revoked` (the access gate revoked the user, a key or a sign-in session and closed them at once), `recheck` (the periodic access re-check refused them), `recheck_error` (the re-check failed or timed out, so the connection was closed anyway). Non-zero only with an access gate (llm-redact-pro). `/status` `connections` carries the same counts, the open connections by kind, and the re-check interval. |
| `llm_redact_routed_requests_total` | counter | `upstream`, `rule` | Requests delivered through the routing layer, by the upstream that produced the response and the rule that chose it (emitted by the core; non-zero only with the llm-redact-pro routing layer). |
| `llm_redact_reissues_total` | counter | `from_upstream`, `to_upstream` | Fallback re-issues to the next chain member (emitted by the core; non-zero only with the llm-redact-pro routing layer). |
| `llm_redact_audit_sink_batches_total` | counter | `sink` | Audit row batches an off-machine audit sink uploaded (`s3`, `azure`; `[audit.s3]`/`[audit.azure]`, llm-redact-pro). Also `/status` `audit.s3/azure.batches_uploaded`. |
| `llm_redact_audit_sink_rows_dropped_total` | counter | `sink` | Audit rows a sink dropped (an upload failed, its buffer was full, credentials or the batch key were missing) — rows missing from the off-machine copy. Also `/status` `audit.s3/azure.rows_dropped`. |
| `llm_redact_map_write_queue_depth` | gauge | — | Durable vault map writes (Responses chains, stored-object owner records, Live resumption handles) queued or in flight on the vault's background writer; 0 without one (the in-memory vault). Also `/status` `vault.map_writes_pending`. |
| `llm_redact_map_write_wait_timeouts_total` | counter | `cause` | `[vault] map_writes = "before_answer"` answers sent before their map writes landed: `bound` (the wait reached `[vault] map_write_wait_seconds`) or `stuck` (not waited for: the writer is stuck behind an earlier write a wait gave up on — one bound per episode, not per answer). Another replica may then read the record as unknown a little longer (refused or sealed, never a wrong value). The same answers also count under the bookkeeping stage `map_write_wait` (kept for existing alerts). Also `/status` `vault.map_write_wait_timeouts_total`. |
| `llm_redact_map_writes_mode` | gauge | `mode` | The effective `[vault] map_writes` (value 1): `before_answer`, `background` or `synchronous` (no background writer). Also `/status` `vault.map_writes`. |
| `llm_redact_vault_entries` / `_sessions` | gauge | — | Vault size. |
| `llm_redact_uptime_seconds` / `_start_time_seconds` | gauge | — | Process liveness. |
| `llm_redact_info` | gauge | `version` | Build info (value 1). |
| a plugin's own gauges | gauge | the plugin's | Rendered after the core's from an access gate's optional `metrics_samples` (llm-redact-pro: `llm_redact_users{state}`, `llm_redact_seats_licensed`, `llm_redact_seats_used` — see its docs/observability.md). The core enforces the SHAPE: a name under `llm_redact_` that no core family uses, at most 4 labels whose names and values are from `[a-z0-9_]` (a value at most 32 characters, with no run of 8 or more hexadecimal characters holding a digit — the shape of a key, hash, id or long number), a finite value, at most 256 samples; anything else is dropped and counted under the bookkeeping stage `plugin_metrics`. A shape check cannot tell a fixed state name from a user name that fits it (`alice_smith`): keeping every label value to a small fixed set is the plugin's obligation (llm-redact-pro's are fixed enums), and the samples are served on the open `/metrics` path. Read in a worker thread, at most 2 s per call, never two at once — scrapes arriving together (an HA pair of Prometheus servers) share the call in flight until its own 2 s bound: a slow or failing plugin never delays or fails the scrape (each scrape rendered without the gauges is counted under `plugin_metrics`, logged once per episode by exception type). |

The `llm_redact_ner_*` counters belong to the built NER detectors: they
restart from zero when a configuration reload rebuilds the detectors (any
change to `[detection]`, by SIGHUP or the dashboard editor) — Prometheus reads
that as a counter reset, which `rate()` and `increase()` handle. A reload that
leaves `[detection]` alone keeps them. A redaction preview in the
llm-redact-pro dashboard runs the live detectors and is counted like a request.

## Local refusal kinds

`llm_redact_local_refusals_total{kind}` — one per response the proxy
generated itself. Statuses are the HTTP ones; a realtime connection refused
before it was relayed is closed (1008/1011) and recorded with the same
status, a frame refused on an open relay closes it.

| Kind | Status | The proxy answered itself because |
|---|---|---|
| `request_target` | 400 | The request target is not a plain path, holds a `.`/`..` or empty segment, or would not address the configured upstream (counted without a row when the path may hold a key). |
| `identity_path` | 404 | An unclaimed `/u/<key>/` identity prefix, or a reserved path reached through a stripped one (never recorded: the path holds a key). |
| `request_origin` | 403 | A web page's request (Origin / Sec-Fetch-*), or a request lending a credential the proxy holds addressed to a host name it does not answer to. |
| `access_gate` | 403 | The access gate (llm-redact-pro) refused the client — or revoked a realtime connection before it was relayed. |
| `misaddressed` | 400/404 | Another spelling of a recognized route, the route without its `/v1`, or under an extra prefix. |
| `unattributed` | 404 | No provider can be attributed to the request (realtime: no realtime route for the path). |
| `no_upstream` | 502 | No upstream configured: an unknown `/custom/NAME/`, or a provider without a default upstream (Azure, Vertex, Bedrock) not yet set. |
| `disabled_provider` | 502 | `[providers.NAME] enabled = false`. |
| `method_override` | 400 | A matched route carrying an HTTP method override — or any route, pass-through included, when the access gate authorizes requests. |
| `identity_route` | 403 | An unrecognized route that would spend a credential the proxy holds (its cloud identity, or a routed plan's operator key). |
| `credential_protocol` | 403 | A recognized route whose protocol is not served with a credential the proxy holds (a resumable upload). |
| `too_large` | 413 | A redactable body over `max_body_bytes`. |
| `too_many_strings` | 413 | A body (or realtime frame) with more strings, parts or lines than `max_body_strings`. |
| `scanned_body` | 400 | The scanned-body rule: a body the proxy cannot read to redact (not a JSON object, invalid UTF-8, non-canonical multipart …). |
| `unsupported_encoding` | 415 | A content-encoded request body. |
| `unchecked_body` | 400 | A body the stored-object check cannot read under a credential the proxy holds, or an upload its re-reading would change. |
| `authorization` | 403 | The access gate's optional authorization (llm-redact-pro roles) refused the request or realtime upgrade (1008) — or, on Gemini/Vertex Live, the model a setup frame names, or a first frame that is no setup naming a model; on OpenAI/Azure realtime, the model a `session.update` names, or one that names no string; or what a request's (a realtime frame's) redaction found (`authorize_content`, asked after the redaction and before the audit START row and any upstream contact) — its check failed, or the requester's detection overlay could not be applied. |
| `object_access` | 403 | The session router refused a stored object of another namespace (realtime: its per-frame check). |
| `sealed_session` | 403 | The session router sealed the request's session and redaction would write to it. |
| `blocked_value` | 400 | A block-mode rule matched (realtime: the frame is refused and the connection closed 1008; the row keeps its 101). |
| `verbatim_field` | 400 | An identifier field that must be sent verbatim holds a value to redact. |
| `binary_values` | 400 | An upload inspector found values to redact in a binary file the proxy cannot rewrite. |
| `unscanned_upload` | 400 | An upload part the proxy cannot scan (a binary file under `binary_uploads = "refuse"` or a credential the proxy holds, a framing or header rule). |
| `unredactable` | 400 | A field the proxy cannot decode to redact (an undecodable Bedrock count-tokens blob; realtime: a non-JSON frame under the proxy's identity). |
| `placeholder_limit` | 400 | No placeholder number left for a new value. |
| `override_raced` / `override_fault` | 400 | A one-time refusal override another request used first, or the override store could not record the use. |
| `vault_fault` | 503 | The vault could not issue the request's placeholders or open its session. |
| `audit_unavailable` | 503 | `[audit] required`: the write-ahead START row could not be committed. |
| `upstream_auth` | 502 | The proxy's own cloud identity produced no credential. |
| `upstream_fault` | 502 | No upstream answer at all: a transport fault (connect, timeout, a drop before the answer was delivered), every routed hop failed, or a realtime dial failed. |
| `redirect_refused` | 502 | The upstream answered a redirect the proxy does not relay. |
| `delivery_fault` | 502 | Restoring the upstream's answer failed: a buffered answer is replaced by the 502, a stream is cut (its row records 502) — or a realtime relay failed (row 500). |
| `no_route` | 502 | The routing layer (llm-redact-pro) has no rule and no default for the request. |
| `route_unsupported` | 404 | The routing layer does not serve the endpoint (count_tokens). |
| `budget` | 402 | The routing layer's monthly budget is exhausted. |
| `reload` | 503 | A config reload changed a realtime connection's admission before it was relayed (an open relay is closed 1012 instead and not counted). |
| `realtime_unavailable` | — | A realtime upgrade without the `realtime` extra (never recorded). |

Not counted: answers to the proxy's own reserved paths (`/__llm-redact/*`),
`GET /` liveness answers, and the routing layer's local model-discovery
answers — none of them is a request the proxy refused to forward.

## Wiring it up

1. **Scrape.** The proxy binds `127.0.0.1` by default, so run Prometheus on
   the same host and merge `deploy/prometheus-scrape.yml` into your config. If
   the proxy runs under TLS (`[tls]`), switch the job's `scheme` to `https` and
   add a `tls_config` (never expose the metrics endpoint over a plain wider
   bind — see the bind policy in the threat model).
2. **Alert.** Add `deploy/prometheus-alerts.yml` to `rule_files:` and point it
   at your Alertmanager. The load-bearing one is
   **`LlmRedactWarnModeForwardingValues`**: warn mode is observation-only and
   sends the matched value upstream, so a sustained warn rate is a real leak
   signal, not noise. **`LlmRedactHighProxyOverheadP95`** reads the proxy's
   own time (above 0.25 s p95 for 10 minutes): it does not fire on a slow
   provider or a long stream, which the end-to-end
   `llm_redact_request_duration_seconds` includes — there is deliberately no
   end-to-end latency rule here (a provider-realistic threshold depends on
   your models; llm-redact-pro's package carries one at 30 s).
   **`LlmRedactLocalFaultRefusals`** pages when the proxy refuses requests
   because its own vault or audit storage fails.
   **`LlmRedactNerSkippingLongStrings`** warns when an NER backend keeps
   skipping strings longer than `[detection.ner] max_chars` for 30 minutes:
   the regex rules still scanned them, but names in them were not looked for
   (raise `max_chars` if such strings carry names; docs/troubleshooting.md).
3. **Visualize.** Import `deploy/grafana-dashboard.json` (Dashboards → Import),
   pick your Prometheus data source. The warn/block panels are colored to stand
   out because they represent values leaving the box or traffic being rejected.

Prometheus scraping is a Free feature and always on. **OpenTelemetry export**
(`[otel] enabled = true`) — the *same* metadata-only rows as OTLP/HTTP spans and
counters, parented into the caller's distributed trace — is a Pro feature; it
can run alongside Prometheus. Its setup and the off-machine trust decision it
represents are documented in the `llm-redact-pro` repo's
`docs/deployment-pro.md`.

The routing layer's runtime state — per-upstream health (`healthy` /
`cooldown` / `budget_exhausted`), spend against budget (a passthrough
upstream's USD is a list-price equivalent), re-issues in the last hour,
runtime-observed unpriced models — is the `routing` block of
`GET /__llm-redact/status` (`{"enabled": false}` without the
llm-redact-pro routing layer), one line per upstream in `llm-redact
status`, and the `route` field of every `/recent` and `/events` row
(`null` on the unrouted path); see the llm-redact-pro routing guide.

An access gate (llm-redact-pro's named users) reports its own state as the
`users` block of `GET /__llm-redact/status` (`{"registry": false,
"enforcement": false}` without one). Its optional `posture` list holds short
lines the gate wants seen as posture warnings (for instance `[authz] runs in
audit mode: refusals are logged, not enforced`); `llm-redact status` prints
each in its loud posture block, so the block is not silent while the gate
reports one, and `--json` shows the list. The core serves at most 8 of them,
non-empty strings only, every non-printable character escaped and each cut
to 200 characters; anything else is dropped, and so is a `posture` that is
not a list.
