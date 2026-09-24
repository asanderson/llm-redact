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
second lists fields (host, port, vault, audit, log, tls, otel, users,
email) that only apply on a full restart.

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

## "[routing] enabled = true requires the llm-redact-pro package"

Rule-based upstream routing, fallback chains and budgets are an
llm-redact-pro feature; the core parses and validates the
`[upstreams]`/`[routing]`/`[prices]` sections but never runs them. Three
surfaces say so, each naming the package: `serve`, `serve --check`, a
SIGHUP reload (`config reload failed; keeping current config: …`) and
the llm-redact-pro config editor's dry-run (a 400) refuse a config with
`[routing] enabled = true` (`… requires the llm-redact-pro package
(0.3+) …`); `llm-redact routes …` and `llm-redact spend` print
`routing tooling requires the llm-redact-pro package 0.3 or newer …` and
exit 1; and `doctor` FAILs its `routing` line (`… the proxy will refuse to
start …`). An installed llm-redact-pro older than 0.3 has no routing layer
and gets the same messages: upgrade it.
Install the package (see [editions.md](editions.md)) or set
`enabled = false` — `[upstreams]` and `[prices]` are then inert and the
proxy forwards each protocol to its one `[providers.NAME]` upstream, as
`llm-redact status` reports (`routing: disabled`). The routing layer's
own runtime messages are documented in that package's routing guide.

## "[SECTION] KEY must be a finite number (nan/inf are not accepted)"

`serve --check` / `serve` / `doctor` / SIGHUP refuse a routing or price
number that is `nan` or `inf` (TOML spells both natively:
`monthly_budget_usd = inf`, `cooldown_seconds = nan`,
`[prices.override."m"] input = inf`) or an integer literal too large
for a `float` (a `monthly_budget_tokens` of hundreds of digits). Such a value would pass
every other check and then never trip a threshold, break `/status` JSON
and the (llm-redact-pro) config editor's reparse guard; write a real number.

## "edit the file and reload" (HTTP 400 from the config editor)

(llm-redact-pro dashboard only.) The editor POST named `[upstreams]`,
`[routing]` or `[prices]`. Those sections are deliberately file-only
(the editor preserves them from file truth): edit the TOML, run `llm-redact serve --check`, then `kill -HUP`.

## "written to PATH but not applied (…); fix the cause and reload (SIGHUP)" (HTTP 500 from the config editor)

(llm-redact-pro dashboard only.) The POST validated and the TOML was
written (with its `.bak`), but hot-applying it failed after the dry run — a fault only the routing
layer's live swap can hit (the spend table in a locked vault file, say;
the message carries the exception type). The file on disk is the new
config, the running proxy still has the old one; remove the cause, then
`kill -HUP`.

## "on_status KEY never applies: reissue_policy = "never" …" / "on_budget_exhausted never applies: reissue_policy = "never" …"

Parse-time WARNINGs (build log, `routes list`): the rule lists a chain
but `reissue_policy = "never"` means no request ever leaves the primary,
so the chain is dead configuration — `retry-same` still works, it stays
on the same upstream. Drop the chain or set `stateless-only` / `always`.
The sibling `on_budget_exhausted never applies: upstream 'X' has no
monthly budget` means the chain can never be entered because nothing
exhausts.

## `serve --check` refuses `[upstreams]` / `[routing]` / `[prices]`

The message names the section, rule id or key (never a value). The
invariants it enforces: a chain may never contain a passthrough upstream
(no subscription pooling; `is a passthrough upstream: a chain may only
continue to env:/none upstreams`); `inject_system_note = true`,
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
a rule nor a default is a runtime 502 `no_route` — the gate cannot know
which protocols your tools will send. Probe each one with `llm-redact
routes test --protocol X` (llm-redact-pro) before traffic arrives. The
full refusal list is in the llm-redact-pro routing guide.

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
recent-request feed (`GET /__llm-redact/recent`, or `/llm-redact:recent`
in an agent) shows whether traffic is arriving at all.
