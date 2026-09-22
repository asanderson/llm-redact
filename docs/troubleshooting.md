# Troubleshooting: what the error means and what to do

Keyed by the exact strings llm-redact emits. First stop, always:

```bash
llm-redact doctor          # read-only diagnostics; --json for machines
llm-redact serve --check   # serve's full startup, minus the socket
```

If doctor is green and `serve --check` exits 0, serve will start and a
`kill -HUP` reload will apply.

## "the {name} provider is disabled in llm-redact config"

A 502 from the proxy itself (never the upstream): the request matched a
provider with `[providers.NAME] enabled = false`. This is fail-closed by
design — a disabled provider must never fall through to unredacted
pass-through. Re-enable the provider or stop sending its traffic.

## "no [providers.custom.NAME] upstream is configured"

A request arrived under `/custom/NAME/` but the config has no matching
`[providers.custom.NAME]` section. Same fail-closed rule as above. Check
the prefix your tool uses against `llm-redact config show`.

## "request body exceeds llm-redact max_body_bytes"

A 413: the redactable body is bigger than the cap (default ~10 MiB), and
forwarding it unscanned is never an option. Batch/file uploads legitimately
exceed chat-sized caps — raise `max_body_bytes` in the config.

## "config parses but does not BUILD: …"

From `doctor`: the file is valid TOML but the detector build refuses it —
an unknown rule name in `[detection]`/`[detection.modes]`, an unknown
custom-rule `validator`, or two rules sharing a detector type with
conflicting modes. serve would refuse this config at startup, and a SIGHUP
reload would keep the current one (with only a log line saying so). The
message names the exact offender; fix it and re-run `serve --check`.

## "config reload failed; keeping current config" / "changes require restart"

Log lines from a `kill -HUP`. The first means the new file failed to parse
or build — the proxy deliberately keeps serving the old config rather than
crash; fix the file (`serve --check` shows the error) and HUP again. The
second lists fields (host, port, vault, audit, log, tls, otel) that only
apply on a full restart.

## "the vault at {path} is encrypted; set [vault] encryption = \"fernet\" …"

The vault was migrated to encrypted form (schema v3, one-way) but the
current config opens it without a cipher. Set `[vault] encryption =
"fernet"` and provide the key (`LLM_REDACT_VAULT_KEY`, key command, or the
OS keychain via `llm-redact vault set-key`).

## "LLM_REDACT_VAULT_KEY does not match the vault at {path}"

The key resolves but is not the one this vault was encrypted under —
fail-closed at open, never at the first request. `llm-redact doctor`
checks key-match without starting anything. If the key was rotated, make
sure the NEW key is what resolves; `llm-redact vault rotate-key` is the
supported way to change it.

## "non-loopback bind … requires mutual TLS" (bind refused at startup)

`host` is set to something other than 127.0.0.1 without the full
`[tls]` trio (certfile + keyfile + client_ca). Non-loopback is fail-closed
behind mutual TLS; keep the proxy on loopback unless you operate a client
certificate PKI (`docs/threat-model.md` explains why). The container's
documented publish spec (`-p 127.0.0.1:8787:8787`) keeps loopback
semantics without any of this.

## "tamper_evident = true but LLM_REDACT_AUDIT_HMAC_KEY not set"

The audit hash-chain needs its HMAC key from the environment (a keyless
chain would be attacker-recomputable, so the proxy refuses to start).
Export the key or disable `tamper_evident`.

## "[audit] required = true needs [audit] enabled = true"

Zero-loss mode is a property OF the audit log, so it cannot be requested
without one. Enable the audit log (`[audit] enabled = true`, Pro) or drop
`required`.

## "[audit] required = true needs a llm-redact-pro version with write-ahead audit support"

The installed `llm-redact-pro` package predates the write-ahead
`begin`/`finalize` pair, and the proxy refuses to run a config that
promises zero loss on a log that cannot deliver it. Upgrade the pro
package (or drop `required` to run the classic fail-open audit log).

## "llm-redact: audit log unavailable and [audit] required is enabled" (HTTP 503)

`[audit] required` is doing its job: the write-ahead audit row could not
be durably committed (typically a full disk or an IO error on the audit
DB), so the request was refused BEFORE contacting the provider — "no
audit row, no service". Free disk space or repair the audit DB path; the
matching CRITICAL log line names the exception type. If availability
matters more than a guaranteed-complete trail, disable `required`.

## "llm-redact routing: no rule matched and no default_upstream for protocol X"

A proxy-generated 502 (log `class=no_route`): `[routing] enabled = true`
and this protocol has neither a matching `[[routing.rule]]` nor an entry
in `default_upstream` — routing never forwards by guesswork. Add a default
for the protocol (the table form covers several:
`default_upstream = { anthropic = "ollama", ollama = "ollama_native" }`)
or a catch-all rule. An explicit `[upstreams.NAME]` named after a legacy
provider replaces its auto-registration *protocol included*, which is the
usual way native Ollama traffic loses its route. `llm-redact routes test
--protocol X` reproduces the decision without contacting any upstream
(its `state` line is a plain-http probe of the configured listener).
Reference: [routing.md](routing.md).

## "llm-redact routing: upstream NAME does not implement count_tokens"

A local 404 (no upstream contact, no fallback): the rule selected an
upstream with `count_tokens = false` — Ollama, which serves
`/v1/messages` but hangs on `/v1/messages/count_tokens` — and the tool
asked for a token count. Expected on the local lane; tools treat it as
"no count available".

## "llm-redact routing: upstream NAME budget exhausted for this period" (HTTP 402)

The named upstream's `monthly_budget_usd` / `monthly_budget_tokens` is
spent for the current period (`[budget_reset_day of this month,
budget_reset_day of next month)` in UTC). `llm-redact spend` shows the
period, totals and the
share that came from re-issues; raise the cap, wait for the reset, or
give the rule an `on_budget_exhausted` chain (gated by the rule's
`reissue_policy`: `never` keeps the 402, `stateless-only` keeps it for a
request carrying signed thinking blocks and adds
`x-llm-redact-reissue: skipped; reason=stateful`). Chains already skip
exhausted members, so only *direct* requests see the 402.

## "spend store write failed for upstream NAME (Type); the row is counted in memory only"

A WARNING (exception type only) from the response finalizer: the spend
INSERT into the sqlite vault file could not commit. It runs
synchronously on the event loop with a **250 ms** lock wait
(`REQUEST_PATH_BUSY_TIMEOUT_MS` — the offline `llm-redact spend` waits
5 s), so the usual cause is another process holding the vault's write
lock (a second proxy on the same file, a tool inside a transaction) or a
full/unwritable disk. The request finished normally and the row still
counts toward this period's budget in memory until restart; it is just
not persisted, so `llm-redact spend` and the next proxy start under-read
by it. Free the lock (one proxy per vault file) or the disk.

## "[SECTION] KEY must be a finite number (nan/inf are not accepted)"

`serve --check` / `serve` / `doctor` / SIGHUP refuse a routing or price
number that is `nan` or `inf` (TOML spells both natively:
`monthly_budget_usd = inf`, `cooldown_seconds = nan`,
`[prices.override."m"] input = inf`) or an integer literal too large
for a `float` (a `monthly_budget_tokens` of hundreds of digits). Such a value would pass
every other check and then never trip a threshold, break `/status` JSON
and the editor's reparse guard; write a real number.

## `x-llm-redact-reissue: skipped; reason=stateful` / `reason=no-candidate`

The upstream's error was returned unchanged and the header says why.
`stateful`: the request carries an assistant `thinking` /
`redacted_thinking` block and the rule is `reissue_policy =
"stateless-only"` — signed thinking is bound to the originating model and
credential, so the proxy refuses to swap lanes mid-conversation; start
a fresh conversation on the fallback lane, or set `"always"` if you
accept signature rejections and cache loss. `no-candidate`: the chain's
members were all ineligible (passthrough — never allowed —, in
cooldown, budget-exhausted, or `count_tokens = false` on that path);
`/__llm-redact/status` → `routing.upstreams.*.state` says which.

## "edit the file and reload" (HTTP 400 from the config editor)

The editor POST named `[upstreams]`, `[routing]` or `[prices]`. Those
sections are deliberately file-only (the editor preserves them from file
truth): edit the TOML, run `llm-redact serve --check`, then `kill -HUP`.

## "spend table in the vault database PATH could not be opened (Type: message)"

A `ConfigError` from `serve`, `serve --check`, a SIGHUP reload or the
config editor: routing is enabled on `[vault] backend = "sqlite"` and
the vault file's `spend` table could not be opened — the file or its
directory is unwritable, another process holds a lock, or the file is
not a sqlite database (the exception type and sqlite's message are in
the parentheses). A reload logs `config reload failed; keeping current
config` and keeps serving on the old config. Fix the file (`0600` /
`0700`, the lock, the path) and reload again.

## "written to PATH but not applied (…); fix the cause and reload (SIGHUP)" (HTTP 500 from the config editor)

The POST validated and the TOML was written (with its `.bak`), but
hot-applying it failed after the dry run — typically the spend-table
error above. The file on disk is the new config, the running proxy still
has the old one; remove the cause, then `kill -HUP`.

## "on_status KEY never applies: reissue_policy = "never" …" / "on_budget_exhausted never applies: reissue_policy = "never" …"

Parse-time WARNINGs (build log, `routes list`): the rule lists a chain
but `reissue_policy = "never"` means no request ever leaves the primary,
so the chain is dead configuration — `retry-same` still works, it stays
on the same upstream. Drop the chain or set `stateless-only` / `always`.
The sibling `on_budget_exhausted never applies: upstream 'X' has no
monthly budget` means the chain can never be entered because nothing
exhausts.

## "state:    not probed (no proxy answered on the configured listener)" (from `routes test`)

The routing decision is complete; only the live state and the
cooldown/budget annotations are missing. They come from a best-effort
plain-http `GET /__llm-redact/status` of the config file's `host`/`port`
(1 s), which is skipped for a `[tls]` listener and never uses
`LLM_REDACT_PROXY_URL` — so a proxy that is not running, an mTLS
listener, or a pointed-at proxy all read `not probed`. `llm-redact
status` (with `--ca/--cert/--key`) reports the live state in those
cases. No upstream is ever contacted by `routes test`.

## `serve --check` refuses `[upstreams]` / `[routing]` / `[prices]`

The message names the section, rule id or key (never a value). The
invariants it enforces: a chain may never contain a passthrough upstream
(no subscription pooling); `inject_system_note = true`,
`monthly_budget_*`, `extra_headers` and `body_defaults` are errors on
passthrough upstreams; a rule's upstream and every chain member and
default must share the rule's protocol; every referenced upstream must
exist; a chain may not name the rule's own upstream
(`names the rule's own upstream 'X'` — use `"retry-same"`) or a member
twice (`lists 'X' twice`); `env:VAR` must resolve in the *proxy's*
environment (a container
needs `-e VAR`); and `enabled = true` requires `default_upstream`
(`[routing] default_upstream is required when enabled = true`). A
metered or passthrough `default_upstream` is a WARNING, not an error.
What it does NOT check is per-protocol coverage: a protocol with neither
a rule nor a default is the runtime 502 `no_route` above — the gate
cannot know which protocols your tools will send. Probe each one with
`llm-redact routes test --protocol X` before traffic arrives. The full
refusal list is in [routing.md](routing.md).

## Tool sees `«EMAIL_001»`-style tokens in responses

A placeholder reached the tool unrestored. Almost always one of: the
response came through a DIFFERENT session than the request (per-conversation
mode after a history compaction — visible as `compaction_forks` in
`/__llm-redact/status`), or the tool mangled the token beyond the fuzzy
grammar (bracket swaps like `[EMAIL_001]` are deliberately never restored).
An unrestored token is the fail-safe outcome — the value it hides was
never exposed.

## Nothing is being redacted

Check the posture block in `llm-redact status` (or `doctor`): warn-mode
rules, `[providers.NAME] detection = false`, MCP exempt servers, and
language-scoped-out rules all deliberately forward values and are loudly
listed there. If posture is clean, confirm the tool actually points at the
proxy: `llm-redact run -- <tool>` injects the variable for you, and the
dashboard's recent-request table shows whether traffic is arriving at all.
