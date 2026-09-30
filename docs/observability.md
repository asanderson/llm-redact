# Observability: Prometheus + Grafana

The proxy exposes Prometheus metrics at `/__llm-redact/metrics` (always on,
no config). Everything there is **metadata only** — request counts by
provider and status, detection / warning / block / rehydration counts by
detector *type*, vault entry/session gauges, compaction-fork counts,
routing decisions and re-issues by upstream and rule name, and
build info. It never carries secret values or placeholder ids (see
[threat-model.md](threat-model.md) § Logging posture), so scraping it is safe.

Ready-to-use assets live in [`deploy/`](../deploy):

| File | What it is |
|---|---|
| `deploy/prometheus-scrape.yml` | A `scrape_configs` job to merge into your `prometheus.yml`. |
| `deploy/prometheus-alerts.yml` | Alerting rules (proxy down, warn-mode value forwarding, high block rate, compaction forks, high p95). |
| `deploy/grafana-dashboard.json` | An importable Grafana dashboard (traffic, latency, detections/warnings/blocks by type, vault + compaction). |

## Metrics reference

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `llm_redact_requests_total` | counter | `provider`, `status` | Requests proxied. |
| `llm_redact_request_duration_seconds` | histogram | `provider`, `streamed` | In-proxy duration (redact + forward + rehydrate), NOT upstream RTT. |
| `llm_redact_detections_total` | counter | `type` | Values redacted, by placeholder type. |
| `llm_redact_warnings_total` | counter | `type` | Warn-mode hits — **the value was forwarded upstream**. |
| `llm_redact_blocked_total` | counter | `type` | Requests rejected 400 by block-mode rules. |
| `llm_redact_rehydrations_total` | counter | `type` | Placeholders restored in responses. |
| `llm_redact_compaction_forks_total` | counter | — | Per-conversation sessions forked by history compaction. |
| `llm_redact_unscanned_uploads_total` | counter | `provider` | Binary file parts of uploads (a PDF, an image, an archive) **forwarded unscanned** with the client's own key under `[detection] binary_uploads = "forward"` (the default) — counted only once the upload was handed to the upstream. Also `/status` `unscanned_uploads_total` and a `llm-redact status` posture line. |
| `llm_redact_inspected_uploads_total` | counter | `provider`, `outcome` | Binary file parts read as text by an upload inspector (the core's `[extraction]`, docs/extraction.md), one outcome each: `clean` (sent byte-identical after a clean scan of the EXTRACTED text only — counted once the upload was handed to the upstream), `clean_refused` (scanned clean, the upload refused before any upstream contact), `converted`/`converted_refused` (replaced by its redacted extracted text in convert mode; sent or refused), `detected`, `blocked`, `incomplete`, `not_inspected`, `timeout`, `error`. Also `/status` `inspected_uploads_total`. |
| `llm_redact_upstream_errors_total` | counter | `provider` | Transport faults failed closed as 502 (by upstream name when routing is enabled). |
| `llm_redact_bookkeeping_errors_total` | counter | `stage` | Faults in the proxy's own bookkeeping. After the upstream answered: session bookkeeping (`response_id`, `object_ids`, `listing`, and `response_observer` — a session router observing the answer — contained, the answer is still delivered) and `delivery` (restoring the answer failed: a buffered one is a recorded 502, a stream is cut). Before any upstream contact: `vault` (issuing a request's placeholders failed — a recorded 503; a realtime frame closes the connection 1011). `vault_check`: a vault view's staleness check could not read its database (contained: the view keeps serving its cache — a cached token only ever restores its own value — and checks again a second later). `recheck`: an open realtime relay's or live-events stream's access re-check raised, timed out or gave an answer that makes no sense (fail closed: the connection is closed). `realtime_frame`: the session router's per-frame realtime check (`realtime_frame_refusal`) raised or gave an answer that makes no sense (fail closed: the connection is closed 1008, the frame never sent). |
| `llm_redact_connections_closed_total` | counter | `cause` | Open long-lived connections (realtime relays, the dashboard's live-events stream) closed because their admission ended: `revoked` (the access gate revoked the user, a key or a sign-in session and closed them at once), `recheck` (the periodic access re-check refused them), `recheck_error` (the re-check failed or timed out, so the connection was closed anyway). Non-zero only with an access gate (llm-redact-pro). `/status` `connections` carries the same counts, the open connections by kind, and the re-check interval. |
| `llm_redact_routed_requests_total` | counter | `upstream`, `rule` | Requests delivered through the routing layer, by the upstream that produced the response and the rule that chose it (emitted by the core; non-zero only with the llm-redact-pro routing layer). |
| `llm_redact_reissues_total` | counter | `from_upstream`, `to_upstream` | Fallback re-issues to the next chain member (emitted by the core; non-zero only with the llm-redact-pro routing layer). |
| `llm_redact_vault_entries` / `_sessions` | gauge | — | Vault size. |
| `llm_redact_uptime_seconds` / `_start_time_seconds` | gauge | — | Process liveness. |
| `llm_redact_info` | gauge | `version` | Build info (value 1). |

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
   signal, not noise.
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
