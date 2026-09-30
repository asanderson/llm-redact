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

## "this path lacks the API's /v1 segment" (HTTP 404)

The tool's OpenAI-compatible base URL is missing `/v1`: the OpenAI SDKs,
Codex and OpenCode append `/responses` or `/chat/completions` to a base
that already includes it. Set `OPENAI_BASE_URL=http://127.0.0.1:8787/v1`
(`llm-redact run` exports that). Nothing was forwarded. An upstream that
serves its OpenAI-compatible API without `/v1` is a
`[providers.custom.NAME]` upstream.

## "is not an API path llm-redact can attribute to a provider" (HTTP 404)

The request matched no route and nothing names its provider: no path
family (`/v1beta/…` is Gemini's, `/api/…` Ollama's, …) and no header only
one provider's clients send (`anthropic-version`, a Google API key, an
`openai-*` header, …) — or headers of two providers at once. The proxy
never forwards such a request to a guessed provider (that would hand one
provider's key and your content to another). Check the tool's base URL
against [providers.md](providers.md); `/recent` shows the path.

## "this path carries an extra prefix" (HTTP 404) / "must be spelled exactly" (HTTP 400) / "must not contain an empty segment" (HTTP 400)

The base URL is off: it repeats the API version (`/v1/v1/messages` —
`ANTHROPIC_BASE_URL` takes no `/v1`), ends in `/` and is joined naively
(`//`), or the tool changed the path's case, added a trailing `/`, a `\`,
a `;param`, trailing spaces or dots, or encoded the path twice. A spelling
the API does not define is never forwarded: an upstream (or a front end
before it) that ignores case or a trailing slash, or normalizes such
spellings, would run it unredacted.

## "the upstream answered a redirect (NNN), which llm-redact does not relay" (HTTP 502)

The upstream answered a 3xx with a `Location` to a request whose repeat
would leak — a body on a route the proxy redacts, a routed or
identity-signed request, a request carrying a credential for the proxy, or
any request to a custom upstream: a following client would re-send your
original, unredacted request — credentials included — to wherever it
points, so the proxy refuses to relay it. Usually `upstream_base_url` is
an `http://` URL whose server redirects to `https://`, or an old path:
set it to the API's final `https` URL. Counted in
`llm_redact_upstream_errors_total`.

## "this proxy does not answer to that host name" / "a web page on another origin sent this request"

A 403 (WebSocket: close 1008) from the proxy itself, before any upstream
contact, counted in `/status` `request_origin_refusals_total`. The proxy
refuses requests a web page in a browser could have sent (CSRF, DNS
rebinding, cross-site WebSockets — see the threat model's "Requests from
web pages"):

- `host`: a browser request, or a request that would spend a credential
  the proxy holds (`auth = "identity"`, a routed operator key) over plain
  HTTP, was addressed to a name other than 127.0.0.1, localhost, ::1, the
  bind host or `allowed_hosts`. If a tool reaches the proxy as a compose
  service, `host.docker.internal` or a Kubernetes Service, list that name
  in `allowed_hosts` and restart
  ([deployment.md](deployment.md#host-names-the-proxy-answers-to-allowed_hosts)).
- `origin` / `fetch_site`: the request carried an `Origin` other than the
  proxy's own or one listed in `allowed_origins`, or a `Sec-Fetch-Site` of
  `cross-site`/`same-site` without an `Origin` to check. A browser app on
  another origin is served only when you list its exact origin in
  `allowed_origins` (`scheme://host[:port]`, then restart) — and a listed
  page can read your restored values back
  ([deployment.md](deployment.md#browser-apps-on-other-origins-allowed_origins)).
  Otherwise run the tool outside the browser.

## "llm-redact config reload changed this connection's …; reconnect"

A realtime (WebSocket) connection closed by the proxy with code 1012
(Service Restart). A config reload (SIGHUP, the config editor) changed
something the connection was admitted with. The reason names what changed:
the provider's `[providers.NAME] settings`, the `upstream authorizer`, or
the `[detection] policy`. Nothing is wrong: reconnect, and the new
connection runs under the new configuration in the same vault session.
Frames the client sent after the reload were not forwarded; resend them on
the new connection. A client that was still writing may see a connection
reset instead of this close frame. A reload that changes nothing a
connection depends on leaves it open
([deployment.md](deployment.md#reloads-and-open-realtime-connections)).

## "an uploaded file is binary (not text llm-redact can redact)"

A 400 on a file upload (`/v1/files` and its Azure/custom twins): the file
is not text — a PDF, an image, an archive, or text in an encoding other
than UTF-8 (UTF-16/32 with a byte-order mark is fine) — so llm-redact
cannot scan it. It is refused when the request would be sent with a
credential the proxy holds (`auth = "identity"`, a routing rule's operator
key), or when `[detection] binary_uploads = "refuse"`. With your own key
and the default `binary_uploads = "forward"` such a file is forwarded
unscanned instead (counted in `/status` `unscanned_uploads_total`).
Convert a Latin-1/Windows-1252 text file to UTF-8 to have it redacted.
With llm-redact-pro's document extraction a PDF or Office file read
completely as clean text is sent instead; one it could not read completely
(a scanned page, an embedded image) keeps this refusal — `/status`
`inspected_uploads_total` counts each part's outcome (`clean` only for a
part that went out; `clean_refused` for one that read clean in an upload
refused for another reason).

## "an uploaded binary file holds values it must redact (…, found in the file's extracted text)"

A 400 on a file upload with an upload inspector (llm-redact-pro's document
extraction): the text read out of a PDF or Office file holds values of the
named types that llm-redact would redact — and it cannot redact inside the
file. Remove them from the document (or send its text instead, which is
redacted), allowlist a value that is not sensitive, or set that rule's
mode to `warn` if forwarding it is acceptable.

## "request body exceeds llm-redact max_body_bytes"

A 413: the redactable body is bigger than the cap (default ~10 MiB), and
forwarding it unscanned is never an option. Batch/file uploads legitimately
exceed chat-sized caps — raise `max_body_bytes` in the config.

## "request body exceeds llm-redact max_body_strings"

A 413 of the same kind: the body carries more strings to redact than the
cap (default 100,000 — JSON string values, form fields, file names, lines
of an uploaded JSONL file) or more multipart parts. Redaction costs per
string on the proxy's event loop, so the cap keeps one request from
stalling the others. Agent conversations stay far below it; a large batch
upload of short prompts can exceed it — raise `max_body_strings` (and
usually `max_body_bytes`) in the config. A realtime connection whose
client frame exceeds it is closed with code 1009.

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
second lists fields (host, port, allowed_hosts, allowed_origins, vault,
audit, log, tls, otel, users, email) that only apply on a full restart.

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
