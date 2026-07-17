# Observability: Prometheus + Grafana

The proxy exposes Prometheus metrics at `/__llm-redact/metrics` (always on,
no config). Everything there is **metadata only** — request counts by
provider and status, detection / warning / block / rehydration counts by
detector *type*, vault entry/session gauges, compaction-fork counts, and
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
