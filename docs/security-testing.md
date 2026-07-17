# Security testing

How llm-redact's security claims are *tested*, not just asserted. Every
boundary in the [threat model](threat-model.md) has a guard in a few lines of
`proxy.py` / `config.py`; the risk is that a refactor silently weakens one and
the happy-path tests stay green. This page is the map from attacker scenario to
the test that would catch the weakening.

Three complementary harnesses back the boundary map:

- **`tests/test_security_boundaries.py`** — the red-team boundary suite. One
  test per threat-model boundary, named for the attack it repels, parametrized
  over **every** guarded endpoint so a newly added one that forgets a layer
  fails. This is the single, complete boundary → guard map.
- **`tests/test_canary_leak.py`** — the self-output canary harness. Drives real
  traffic carrying planted secret-shaped canaries and asserts none surface in
  **anything the proxy emits about itself** (logs, Prometheus text, `/recent`,
  `/audit`, `/status`). A new logging or telemetry call site that echoes a
  value fails here.
- **`tests/test_codec_fuzz.py`** — adversarial codec fuzzing. Throws random and
  mutated bytes at the streaming byte codecs (SSE, AWS eventstream, NDJSON,
  multipart) and asserts their fail-safe invariants hold on inputs no fixture
  covers: parsers never raise an unexpected exception, never lose bytes, and
  the byte-faithful multipart codec round-trips.

These extend — never replace — the per-endpoint deep suites
(`test_config_endpoint.py`, `test_tls.py`, `test_proxy_integration.py`) and the
load-bearing chunk-split sweeps (`test_rehydrate.py`, `test_sse.py`).

## Attacker scenarios → coverage

| # | Attacker scenario | Boundary (threat-model §) | Guard | Test |
|---|---|---|---|---|
| B1 | A crafted request tries to make the proxy *forward* a reserved `/__llm-redact/*` path to the upstream | Local ops surface | Reserved paths answered before any routing/upstream code | `test_b1_reserved_paths_never_reach_upstream` |
| B2 | **DNS rebinding**: a hostile page re-resolves its domain to `127.0.0.1`, then reads the config editor / CSRF token | Local ops surface | Host header must be a known loopback name (403 otherwise), even for GET | `test_b2_hostile_host_rejected` (every host-gated endpoint) |
| B3 | A cross-origin page POSTs to the proxy from `evil.example` or a `null`-origin sandbox | Local ops surface | Present Origin must be a local origin; https only when the proxy serves TLS | `test_b3_hostile_origin_rejected`, `test_b3_https_origin_refused_without_tls` |
| B4 | A page issues a forged request without the per-process CSRF token | Local ops surface | Every guarded POST requires the token in a custom header (constant-time compare) | `test_b4_missing_or_wrong_csrf_rejected` (every guarded POST) |
| B5 | A cross-origin `fetch` carrying the custom header triggers a CORS preflight the attacker hopes will be waved through | Local ops surface | OPTIONS → 405 with **no** `access-control-*` headers, so the preflight fails | `test_b5_options_preflight_405_no_cors` |
| B6 | A form POST (`text/plain`, no preflight) tries to reach a mutating handler | Local ops surface | JSON content-type required (415 otherwise) | `test_b6_non_json_content_type_415` |
| B7 | A giant guarded-POST body aims to exhaust proxy memory | Local ops surface | 1 MiB cap, read incrementally so a lying Content-Length can't help | `test_b7_guarded_post_body_cap_413` |
| B8 | A redactable request body too large to buffer would otherwise be forwarded **unredacted** | Outbound requests | `max_body_bytes` fail-closed: 413, never forwarded | `test_b8_oversized_redactable_body_413_not_forwarded` |
| B9 | A hostile page tries to frame the dashboard, inject remote script, or leak a Referer | Local ops surface | Strict CSP + `X-Frame-Options: DENY` + nosniff + `Referrer-Policy: no-referrer` on every reserved reply, stamped in one place | `test_b9_security_headers_on_every_reserved_reply` |
| B10 | An operator binds the proxy to a routable interface without protecting it | Trust boundaries | `validate_bind_security` fail-closed: non-loopback demands full mutual TLS; unprovable hostnames count as non-loopback | `test_b10_*` (loopback ok, server-only-TLS refused, full-mTLS ok, insecure-bind hatch, unresolvable host) |
| B11 | A `?key=` query-string credential (Gemini and others) leaks into logs | Logging posture | Access logging off, httpx logger raised to WARNING; log lines carry path + status + counts only | `test_b11_query_auth_never_logged` (+ the canary harness across all self-output) |
| B12 | The `/status` or `/metrics` surface leaks a configured value (e.g. an allowlist entry) | Local ops surface | Metadata only: types and counts, never values; the config-editor GET is the single documented allowlist exception | `test_b12_status_and_metrics_carry_no_values` |

## Running

```bash
uv run pytest tests/test_security_boundaries.py   # boundary map
uv run pytest tests/test_canary_leak.py           # self-output leak harness
uv run pytest tests/test_codec_fuzz.py            # codec fuzzing
```

All three run in the default `uv run pytest` sweep and in CI. They need no
extras, no network, and no secrets — the canaries and fuzz inputs are
synthetic.

## What these tests deliberately do *not* cover

The [threat model's out-of-scope table](threat-model.md#explicitly-out-of-scope)
is authoritative. In particular: same-UID attackers (no boundary exists),
plaintext in process memory, loopback packet sniffing (implies same-UID
compromise), and provider-side inference are not defended and therefore not
tested here. Detection *recall and precision* are gated separately by the bench
harness (`python -m llm_redact.bench --check`) and the fp-corpus, not by this
suite — this suite is about the transport and ops-surface boundaries, not which
values the rules catch.
