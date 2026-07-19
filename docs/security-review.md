# Adversarial security review (pre-1.0 debut)

A full adversarial read of the codebase, run as five independent reviewers
each owning one attack surface: SQL construction/injection, the HTTP/WS guard
chain and routing, secret/key/DSN handling, filesystem writes and path
traversal (incl. the plugin/proxy install runner), and cryptographic
correctness plus the never-wrong-value invariant. This page records what was
found and what changed; the standing boundary→test map lives in
[security-testing.md](security-testing.md) and the trust boundaries in
[threat-model.md](threat-model.md).

Every finding below is either fixed with a dedicated red-team
regression test or recorded as a deliberate, documented limitation.
Nothing was found that lets a network attacker read conversation data or a
vault secret; the two substantive fixes close a never-wrong-value corner and
an identity-credential leak, both narrow.

## Fixed

### Never-wrong-value: unmapped Responses chain no longer collides with the static session (`sessions.py`)

In per-conversation mode the configured static session is **not** empty — the
`/v1/conversations`, Gemini batch/cache, and no-anchor fallback paths redact
into it. Routing an **unmapped** `previous_response_id` (or an unmapped
`GET /v1/responses/{id}`) there meant the provider could echo a token
(`«EMAIL_001»`) whose name collides with a *different* secret bound in the
static session, and rehydration would emit the **wrong value** — the one
guarantee the whole proxy exists to uphold. Reachable when a chain mapping is
lost (memory backend, or after the ~10k response-id LRU/`response_sessions`
eviction) and the provider echoes a token from the lost turn.

Fix: unmapped chains/GETs now route to a **unique, empty** session derived from
the response id (`orphan_session_id`, domain-separated from conversation
anchors). Nothing else ever writes to it, so an echoed stale token *misses* and
passes through verbatim — the same safe degradation as the compaction fork.
Tests: `test_orphan_session_is_unique_empty_and_never_the_static_fallback`,
`test_orphan_session_disjoint_from_conversation_anchor_derivation`, and the
updated chain/GET resolution tests in `tests/pro/test_sessions_pro.py`.

### Identity-credential leak: percent-encoded `/u/<key>` scrub (`proxy.py`)

The `/u/<key>` path scrub stripped `scope["raw_path"]` only on a byte-prefix
match against the **decoded** key. A client that percent-encoded any character
in the key segment (`/u/lrk_%41BC/...`) defeated the match, leaving the key in
`raw_path` — which the forwarder uses to build the upstream URL — so the `lrk_`
identity credential reached the provider while local logs showed it scrubbed.

Fix: the raw-path scrub is now **segment-based** (strip the first raw segment
after `/u/`, encoding-agnostic) and **fails closed** (rebuild from the scrubbed
decoded path) if `raw_path` doesn't start with `/u/`. The encoded remainder
(Bedrock ARN model ids) is preserved. Tests:
`test_percent_encoded_user_key_is_scrubbed_from_raw_path`,
`test_user_key_scrub_preserves_encoded_remainder`.

### DNS-rebinding gap: `/recent` and `/audit` now host-gated (`proxy.py`)

`/events` was host-gated but its data twins `/recent` and `/audit` — which
return the same request metadata (paths, providers, counts, user names) — were
not, so a rebinding page could poll `/recent` to defeat the `/events`
protection. Both now run the same `_host_allowed` check. `/metrics`,
`/healthz`, and `/readyz` stay open by design (Prometheus scraping and k8s
probes legitimately arrive from non-loopback with the configured host, which is
in the allowlist). Test: `test_recent_and_audit_reject_foreign_host`
(parametrized) in `test_dashboard.py`.

### Defense-in-depth hardening (all operator-input only)

These are not attacker-reachable under the local single-user threat model, but
were genuine correctness inconsistencies:

- **`cli.py run_status`** no longer echoes the raw httpx exception (which embeds
  the request URL, potentially a `/u/<key>` from `LLM_REDACT_PROXY_URL`); it
  reports the netloc plus a URL-free reason. Test:
  `test_status_error_never_echoes_url_or_user_key`.
- **`vault_cli.py` backup** opens the destination with `O_NOFOLLOW` (no-op on
  Windows) so a pre-planted symlink can't redirect the vault secrets onto its
  target, and fails gracefully; it also percent-encodes the read-only
  `file:` URI so a `?`/`#` in the vault path can't inject URI parameters.
  Tests: `test_vault_backup_refuses_to_write_through_a_symlink`,
  `test_readonly_uri_percent_encodes_query_metacharacters`.
- **`service_cli.py`** XML-escapes command parts in the launchd plist and
  refuses control characters in either the plist or the systemd unit, so a
  config path with `&`/`<`/newline can't corrupt or inject unit directives.
  Tests: `test_launchd_plist_escapes_xml_metacharacters`,
  `test_service_unit_rejects_control_chars_in_command`.

## Recorded as deliberate limitations (no code change)

- **Opaque `dbapi` DSN off-box detection** (`vault_rdbms.py`): for
  `backend = "dbapi"` the connect string is opaque, so a plaintext vault
  pointing at a genuinely remote, non-managed host can't be detected and
  refused at startup. This is documented and `doctor` WARNs
  ("backend 'dbapi' DSNs are opaque — locality cannot be verified"). The URL
  backends (postgres/mysql/oracle) fail closed conservatively. See the
  `llm-redact-pro` repo's `docs/vault-rdbms.md`.
- **Origin check ignores the port** (`proxy.py _origin_allowed`): a same-host
  different-port local origin passes the Origin check, but the per-process CSRF
  token (unreadable cross-origin, no CORS) still gates every mutating POST, so
  this is not exploitable.
- **`/u/<key>/__llm-redact/...`** is forwarded to the provider (404) rather
  than reaching the local reserved handler — it violates "reserved paths are
  never forwarded" cosmetically but reaches the provider, not the guarded local
  handler, so there is no CSRF bypass or local-state read.
- **`adapt_sql` `%`/`::` handling** (`vault_rdbms.py`): the paramstyle adapter
  assumes vault SQL templates carry no literal `%`, `LIKE` pattern, or `::`
  cast (none currently do). A future template with such syntax would produce a
  driver error, never injection — values are always bound.

## What was verified clean

SQL: every value is a bound parameter and every identifier a hardcoded literal
or allowlisted config key; the dense-counter `MAX(n)+1` allocation is
TOCTOU-safe (UNIQUE constraint + bounded retry, cache-after-commit). Crypto:
the vendored Ed25519 verifier rejects malleable/non-canonical signatures and
verifies before trusting the payload; the dev-key gate can't be bypassed in
production; vault HMAC-index and Fernet keys are domain-separated; audit batch
encryption never falls back to plaintext when the key vanishes. Rehydration is
strictly session-scoped with a verbatim pass-through on any miss (the
never-wrong-value core). Guard chain: no SSRF/open-proxy, provider-disabled
fails closed, reserved paths never forwarded to the local handler, auth
headers/`?key=`/subprotocols never logged.
