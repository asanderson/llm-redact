# Routing, fallback and budgets

Rule-based upstream selection, quota-aware fallback, and monthly budgets —
a small routing layer that sits **behind** the redaction pipeline so the
proxy can be the single local gateway for one power user: the
subscription (Max) lane stays a byte-exact OAuth pass-through, overflow
goes to your **own** API keys or a local Ollama under explicit rules,
cooldowns and budgets. It is **off by default**: a config with no
`[routing]` table behaves byte-identically to 1.0 (protocol → provider
upstream, credential pass-through, no fallback, no budget). Every setting
below also appears, commented out, in
[`config.example.toml`](../config.example.toml).

Redaction is unchanged by routing. Requests are redacted once; the rule
is selected first (its inputs — headers, path, the body's `model` — are
the same either way) because the first upstream's `inject_system_note`
governs the prepared body, and that redacted body is what every hop
forwards. Redaction is the existing whole-body string walk of
`jsonwalk` — every string value, `system` and `tools` strings included,
structural keys (`model`, `role`, `type`, ids, base64 `data`) skipped —
on every lane, passthrough included; it is **not** narrowed to
`messages[].content`. Warn mode, `[providers.NAME] detection = false`,
MCP exemptions and language scoping keep exactly their documented
meaning — routing only chooses the destination.

## Concepts

| Term | Meaning |
|---|---|
| **protocol** | The wire format of the request, taken from the adapter that matched it: `anthropic` (`/v1/messages*`), `openai` (`/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, …), `gemini`, `ollama` (native `/api/*`). Pass-through requests with no adapter take the protocol the proxy infers from the path. Every other adapter — Azure, Vertex, Claude-on-Vertex, Bedrock, Cohere, `[providers.custom.*]` — and the realtime WebSocket relay keep the legacy path (`upstream_base_url`, no hops, no routing headers). |
| **upstream** | A named destination (`[upstreams.NAME]`): protocol, base URL, credential mode, optional budget and cooldown. Replaces the one-upstream-per-provider model. |
| **rule** | An ordered matcher (`[[routing.rule]]`) on protocol, model glob, headers, path and auth kind that picks an upstream, an optional model rewrite and fallback chains. First match wins. |
| **lane** | Informal: the end-to-end path a class of traffic takes (Max lane, API-key lane, local lane). Not a config object. |
| **passthrough** | Credential mode: the client's own auth-bearing headers are forwarded byte-exact; the proxy holds no key for that upstream. |
| **plan-limit 429** | An Anthropic 429 whose unified rate-limit headers say the subscription's 5-hour or weekly window is exhausted — as opposed to a transient throttle. |
| **stateless request** | An Anthropic request whose `messages` contain no assistant turn with a `thinking` or `redacted_thinking` block (nothing signed to a prior model or credential). `cache_control` breakpoints do **not** make a request stateful — cache loss is tolerable, a signature mismatch is not. |

## The policy envelope (Anthropic, as of September 2026)

Consumer OAuth (Pro/Max) is for Claude Code and native Anthropic apps;
third parties may not route requests through, collect, store or
intermediate those credentials. A **base-URL-only pass-through gateway**
that forwards the OAuth capability in `anthropic-beta` unchanged is the
documented pattern. A gateway credential *replaces* the subscription for
that session and is billed per token. Routing Claude Code to non-Claude
models through a gateway is *unsupported* (features may break) — that is
not the same thing as credential intermediation, which is *prohibited*.

Consequently the proxy may re-issue a failed request to the user's
**own** API key or to **local** Ollama; it must never re-issue to a
second subscription and must never present one upstream's credential to
another. The design encodes that as config-time errors and runtime
asserts, not advice:

- **A chain never contains a passthrough upstream.** Any `on_status` or
  `on_budget_exhausted` member with `credential = "passthrough"` is a
  `serve --check` error naming the rule id, the key and the member; the
  runtime re-checks before every re-issue and skips such a candidate
  with a WARNING. This single rule is what makes subscription pooling
  and credential sharing impossible (invariants I-1 and I-2).
- **Credential isolation.** An `env:`/`none` upstream never receives the
  inbound `authorization` / `x-api-key` / `x-goog-api-key` values (nor a
  `?key=` query parameter — dropped from the forwarded URL, the rest of
  the query kept byte-exact) and has the OAuth marker removed from
  `anthropic-beta` (other beta flags kept); a passthrough upstream never
  receives a server-held key (I-3).
- **Passthrough bodies are not touched.** `inject_system_note` is
  `false` on every passthrough upstream and may not be enabled there
  (I-4); `system`, `tools`, `tool_choice`, `metadata`, `cache_control`,
  thinking blocks and their signatures, `stream` and `max_tokens` are
  forwarded as redaction left them — redaction is the whole-body string
  walk described above (a string inside `system` or a tool description
  is substituted like any other; keys, types and ids are not). Only
  `model_rewrite` rewrites a passthrough body. `extra_headers` and
  `body_defaults` may not even be configured on a passthrough upstream
  (a config error), and the streaming-usage option is never applied
  there.
- **Headers forwarded byte-exact on passthrough**: `authorization`,
  `x-api-key`, `anthropic-beta`, `anthropic-version`, `user-agent`,
  `x-app`, `x-claude-code-*`, `x-stainless-*` and everything else that
  is not hop-by-hop; `anthropic-ratelimit-*`, `retry-after`,
  `request-id`, `x-request-id`, `content-type` and SSE `ping` events
  are relayed back. Nothing in the routing layer buffers a stream
  beyond the existing chunk-boundary token reassembly.

## Turning it on

The smallest honest config: one zero-cost local upstream as the
fail-closed default. With routing enabled, a request whose protocol has
**no matching rule and no default** is answered with a proxy-generated
502 — never forwarded by guesswork — so give every protocol your tools
speak a default or a catch-all rule.

```toml
[upstreams.ollama]
protocol   = "anthropic"            # Ollama serves /v1/messages natively (>= 0.14)
base_url   = "http://127.0.0.1:11434"
credential = "none"
cost       = "zero"
count_tokens = false                # Ollama hangs on /v1/messages/count_tokens

[routing]
enabled = true
default_upstream = "ollama"         # string form: applies to that upstream's protocol
```

Recommend `[vault] backend = "sqlite"` for any lane that talks to
Anthropic: placeholder numbering must survive restarts for prompt-cache
prefixes and preserved-thinking turns to stay consistent, and the sqlite
static session is stable across restarts (dense counters, `MAX(n)+1`).
`llm-redact doctor` WARNs when routing is enabled on the memory backend.

## Configuration reference

### `[upstreams.NAME]`

| Key | Default | Meaning |
|---|---|---|
| `protocol` | required | `anthropic`, `openai`, `gemini` or `ollama`. A rule, a chain member and a default may only name an upstream of the rule's (or key's) protocol — there is no translation. |
| `base_url` | required | Trailing `/` stripped. The request path is appended verbatim (percent-encoding preserved), with one fold: for `protocol = "openai"` **any** base URL that carries a path — `/v1`, `/api/v1`, `/openai/v1`, `/v1beta/openai` — absorbs the request's leading `/v1` (`https://api.openai.com/v1` + `/v1/chat/completions` → `https://api.openai.com/v1/chat/completions`; `https://generativelanguage.googleapis.com/v1beta/openai` + `/v1/chat/completions` → `…/v1beta/openai/chat/completions`), a host-only base takes the full path, and other protocols never fold. The query string is appended as received on a passthrough upstream; on `none`/`env:` upstreams the `key` parameter (Gemini's query credential) is dropped and the rest kept byte-exact. |
| `credential` | `"passthrough"` | `passthrough` (forward the client's auth headers byte-exact), `none` (strip them, send nothing), or `env:VAR` (strip them and inject the key held in environment variable `VAR`, which must match `[A-Z_][A-Z0-9_]*`). The provider-appropriate header is chosen by protocol: `x-api-key` for `anthropic`, `authorization: Bearer` for `openai`/`ollama`, `x-goog-api-key` for `gemini`. The variable is **not** resolved at parse — `serve`, `serve --check` and `doctor` resolve it and fail closed naming the VAR (never its value). A variable that **vanishes at runtime** is not a refusal: that hop is treated as a transport fault — status key `"502"`, so the rule's `502`/`5xx` chain takes over, the upstream enters cooldown and counts in `upstream_errors_total`, `class=transport` when nothing delivers — and the WARNING names the VAR only. |
| `cost` | `"metered"` | `metered` or `zero`. Zero-cost upstreams ignore budgets and are always chain-eligible. |
| `inject_system_note` | see meaning | Per-upstream override of the top-level `inject_system_note`. Defaults to `false` on passthrough upstreams (where `true` is a config error) and to the top-level value otherwise. The decision is made **once, for the first upstream** of a request: later hops reuse the redacted body verbatim. |
| `count_tokens` | `true` | `false` answers `POST /v1/messages/count_tokens` locally with a 404 (`llm-redact routing: upstream NAME does not implement count_tokens`) — no upstream contact, no fallback, recorded as a 404. Set it on Ollama, which serves `/v1/messages` but hangs on `count_tokens`. |
| `monthly_budget_usd` | none | Monthly USD cap. Any budget key on a passthrough upstream is a config error (subscription usage is the vendor's to meter); zero-cost upstreams ignore budgets. |
| `monthly_budget_tokens` | none | Monthly total-token cap (input + output + cache). Either configured limit exhausts the upstream. |
| `cooldown_seconds` | `60` | Circuit-breaker duration after a chain-resolved failure (see below); `>= 0`. |
| `extra_headers` | `{}` | String table added to the outbound request on `env:`/`none` upstreams (OpenRouter's `HTTP-Referer`/`X-Title`, for instance); each entry **replaces** an inbound header of the same name (an inbound `X-Title` is dropped, the configured value wins — never two values). Names are validated as header tokens and lowercased (two spellings of one name are a duplicate). It may not name `authorization`, `x-api-key` or `x-goog-api-key` (`credentials come from credential = "env:VAR", never the config file`), and setting it on a passthrough upstream is a config error (`extra_headers/body_defaults apply only to none/env upstreams`) — a passthrough request is forwarded byte-exact. |
| `body_defaults` | `{}` | Table deep-merged into the JSON body **only for top-level keys absent from the request** — a present key is never touched. Values must be JSON-representable (no TOML datetimes). Setting it on a passthrough upstream is the same config error as `extra_headers`. |

**Legacy auto-registration.** When a `[routing]` table is present,
each enabled legacy provider among `anthropic`, `openai`, `gemini` and
`ollama` with a non-empty `upstream_base_url` — and all four have
built-in defaults, so they are registered whether or not the file names
their `[providers.*]` section — becomes an upstream of the same name
(`protocol` = the provider name, `credential = "passthrough"`, no
system note) so rules can reference `anthropic` without duplicating the
URL — unless an explicit `[upstreams.NAME]` of that name exists, which
wins **entirely, protocol included**. The
spec's `[upstreams.ollama] protocol = "anthropic"` therefore replaces
the native-Ollama registration: give native `/api/*` traffic its own
upstream and default (`ollama_native` in the example below) or it is a
502 `no_route`. Legacy upstreams are marked `"legacy": true` in
`/status` and are never written back by the config emitter — they
re-register from `[providers.*]` every time. `[upstreams]` without
`[routing]` is parsed, kept and inert (a warning says so).

### `[routing]`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch. `enabled = true` **requires** `default_upstream` — enabling without one (or with an empty table) is a config error (`[routing] default_upstream is required when enabled = true`, I-6): the fail-closed destination for unmatched traffic must be explicit whenever rules are live. |
| `default_upstream` | required when `enabled = true` | The fail-closed destination for requests no rule matches. A **string** applies to that upstream's protocol only; a **table keyed by protocol** covers several: `default_upstream = { anthropic = "ollama", openai = "ollama_openai" }`. A default whose upstream protocol differs from its key is a config error. Each default that is not `cost = "zero"` produces a loud WARNING (logged at build, printed by `serve --check`, `doctor` and `status`). A protocol with neither a rule nor a default answers a **runtime** 502 (`llm-redact routing: no rule matched and no default_upstream for protocol X`) — `serve --check` cannot know which protocols your tools will send, so it does not check coverage; probe each protocol with `llm-redact routes test --protocol P`. |
| `max_hops` | `3` | Bounds the chain **including** the original attempt (`>= 1`): `hops` = 1 + re-issues, so `3` allows two re-issues. A `retry-same` attempt is **not** a hop (`max_hops = 1` still permits it); see *Hops and attempts*. |
| `request_deadline_seconds` | `600` | Total wall time across hops (`> 0`): checked before every re-issue and before a `retry-same` wait, **and** each hop's own connect/read is capped by the remaining budget (never above the client's 10 s connect / 600 s read, so the default changes nothing), so a hop that accepts the connection and then hangs fails **inside** the deadline as a transport fault (`"502"` key, `class=transport` when nothing else delivers) rather than after the client's 600 s. A delivered stream keeps that per-read cap for its lifetime: under a short deadline a stream that falls silent for longer than the remaining budget is cut off and finalized as `class=stream_error` (never re-issued — the first-byte rule); a stream that keeps sending is not bounded by the deadline. |
| `plan_limit_detection` | `"headers"` | `headers` classifies an Anthropic 429 as `plan_limit_429` or `throttle_429` from the response headers; `off` makes every 429 a plain `429`. |
| `plan_limit_headers` | see below | Header → accepted-values table; a 429 is a plan limit when ANY listed header carries one of its values. An override **replaces the default table entirely**. |
| `oauth_beta_marker` | `"oauth-2025-04-20"` | The `anthropic-beta` token (comma-separated list, trimmed, case-insensitive) whose presence with a Bearer `authorization` classifies the request as `auth = "oauth"`. |
| `throttle_retry_max_seconds` | `30` | `retry-same` waits at most this long; a longer `retry-after` returns the 429 unchanged (the client owns long retries). |
| `budget_reset_day` | `1` | Day of month (1–28) the budget period starts; periods are `[reset day this month, reset day next month)` in UTC. |
| `debug_headers` | `false` | Also add `x-llm-redact-upstream` / `x-llm-redact-hops` to hop-1 responses (they are always added on re-issued responses). |
| `expose_models` | `false` | Answer `GET /v1/models` locally (see *Model discovery*). |
| `model_catalog` | `[]` | Extra model ids to list in `GET /v1/models`, ahead of the literal names found in rules. |

The default plan-limit table — Anthropic's unified rate-limit family —
is equivalent to:

```toml
[upstreams.ollama]
protocol   = "anthropic"
base_url   = "http://127.0.0.1:11434"
credential = "none"
cost       = "zero"
count_tokens = false

[routing]
enabled = true
default_upstream = "ollama"

[routing.plan_limit_headers]
"anthropic-ratelimit-unified-status"    = ["rejected"]
"anthropic-ratelimit-unified-5h-status" = ["rejected"]
"anthropic-ratelimit-unified-7d-status" = ["rejected"]
```

The related headers Anthropic sends (`anthropic-ratelimit-unified-reset`,
unix seconds; `-representative-claim` ∈ `five_hour`/`seven_day`/…;
`-overage-status`) are relayed to the client untouched but not
consulted. Names and values are both compared trimmed and
case-insensitively: a header value of `Rejected` matches the table's
`rejected`.

### `[[routing.rule]]`

Rules are evaluated in file order, before redaction (the match inputs
are the inbound headers, the path and the body's `model` — none of them
changes with redaction); the first whose present `match` fields ALL
hold wins. Ids must be unique; an empty rule list is allowed
(everything goes to the defaults).

| Key | Default | Meaning |
|---|---|---|
| `id` | required | Appears in logs, `/status`, `recent` rows and metrics. |
| `match.protocol` | required | One of the four protocols; the rule's upstream and every chain member must share it. |
| `match.model` | any | A glob or list of globs (`fnmatch`, case-sensitive) matched against the request body's `model` — or, for `protocol = "gemini"` when the body carries none, the id in the path segment `/v1|v1beta/models|tunedModels/{model}:verb` (decision 15b; `routes test --path` derives it the same way). A request with **no model at all** — id-only follow-ups such as `GET /v1/responses/{id}`, Conversations item reads, `GET /v1/files/{id}/content`, Anthropic batch polls/results, and a batch **submission** whose models sit inside `requests[].params` — never matches a rule that has `match.model`; only a model-free rule (a `match.path` glob, say) or the protocol's default can take it. |
| `match.headers` | any | Table of header name → glob (names case-insensitive, values `fnmatch`); all must match. |
| `match.path` | any | Glob on the inbound path. |
| `match.auth` | `"any"` | `oauth` (Bearer `authorization` **and** the `oauth_beta_marker` in `anthropic-beta`), `gateway-key` (`x-api-key` or `authorization` present without the marker), `none` (neither), `any`. |
| `upstream` | required | The primary destination. |
| `model_rewrite` | none | Replace the body's `model` before forwarding (on every credential mode, passthrough included). For `protocol = "gemini"` the rewrite targets the `/models/{model}:verb` **path segment** instead (percent-encoded; the body never gains a `model` key — Google rejects one). The response carries the **original** id restored — top-level `model`, `message.model` (Anthropic `message_start`) and `response.model` (OpenAI Responses events) in JSON bodies and inside SSE/NDJSON events (OpenAI chunks, Ollama lines), plus Gemini's `modelVersion` in both the SSE form and the buffered JSON-array form — so clients that echo or validate the model name keep working. Spend is priced at the id the upstream saw. |
| `on_status` | `{}` | Table of status key → chain. Keys: an integer status (`529`, or `"529"`), a class (`4xx`, `5xx`), or the special keys `plan_limit_429` / `throttle_429`. Values: an ordered list of upstream names, or the string `"retry-same"`. Precedence: special key > exact status > class. A chain may not name the rule's own upstream (use `"retry-same"`) or list a member twice. |
| `reissue_policy` | `stateless-only` when the rule's upstream is passthrough, else `always` | Consulted before **any** chain is entered — `on_status` and `on_budget_exhausted` alike. `stateless-only` never re-issues a stateful request (see *Statelessness guard*; the refused response carries `x-llm-redact-reissue: skipped; reason=stateful`, the budget 402 included); `always`; `never` (the chain is dead — a parse-time warning says so — and the response is returned with `reissue=no`). `llm-redact routes list` prints every rule's resolved policy. |
| `on_budget_exhausted` | `[]` | Chain to continue with when the primary upstream's budget is exhausted (otherwise a 402). Same membership rules as `on_status` chains, gated by the same `reissue_policy`. The refused primary counts as hop 1, so the first member that answers is hop 2 (its response carries `x-llm-redact-hops: 2`). |

### `[prices]`

| Key | Default | Meaning |
|---|---|---|
| `table` | `"builtin"` | The vendored `prices.json` (USD per 1M tokens: `input`, `output`, `cache_read`, `cache_write`; best-effort list prices, versioned) — or a path to a `.json`/`.toml` file of the same shape that replaces it. |
| `[prices.override."<model id>"]` | none | Per-model override applied on top of the table: `input` and `output` are required, `cache_read` and `cache_write` default to `0`; every price is `>= 0` USD per 1M tokens and no other keys are accepted. |

```toml
[prices]
table = "builtin"

[prices.override."claude-sonnet-5"]
input = 2.0
output = 10.0
cache_read = 0.2
cache_write = 2.5
```

Lookup order for a model id: exact match; then with a `vendor/` prefix
stripped (`anthropic/claude-sonnet-5`); then with a trailing `-YYYYMMDD`
or `-latest` removed; then the **longest table key that is a prefix** of
the id (`claude-sonnet-5-20260401` → `claude-sonnet-5`). An unknown model
costs `null`: it is counted in tokens only (`unpriced_rows` in the spend
row). Two lists report unpriced ids, and they answer different
questions: `doctor` WARNs from the **static config** — `model_catalog`,
the glob-free `match.model` names and `model_rewrite` targets, before
any traffic and excluding rewrite targets that only reach a zero-cost
upstream (budgets are ignored there) — while `unpriced_models` in
`/status` (and the `status`/dashboard posture line) is **runtime-observed**:
ids the price table failed to price on real 2xx usage, carried across
SIGHUP (minus any the reloaded table now prices). The list is bounded:
at most **100** distinct ids are kept (`pricing.UNKNOWN_MODELS_CAP`),
and only ids that look like one — non-empty, at most 200 characters,
printable, no whitespace (`looks_like_model_id`) — are ever listed,
because the `model` field is client text that detection never scans. An
id beyond the cap or outside that shape is still costed `null` and
still counted, in `unpriced_models_dropped` (carried across SIGHUP, reset
on restart), which `status` and the dashboard print as `(+N more not
listed)`. A `claude-*` glob request for an unpriced id appears only in
`/status`; a catalog id nothing ever sends appears only in `doctor`.

### What `serve --check` refuses

Every violation is a `ConfigError` naming the section, rule id or key —
never a value — and `serve --check` / `doctor` report it before any
socket opens. The list, so the messages are not a surprise:

- **`[upstreams.NAME]`**: a name outside `[A-Za-z0-9_][A-Za-z0-9_-]{0,63}`;
  an unknown key; a missing or unknown `protocol`; a `base_url` that does
  not start with `http://` or `https://`; a `credential` that is not
  `passthrough`, `none` or `env:VAR` with `VAR` matching
  `[A-Z_][A-Z0-9_]*`; an unknown `cost`; a non-positive
  `monthly_budget_usd` or a `monthly_budget_tokens` that is not a
  positive integer; `cooldown_seconds < 0`; any of those that is not
  finite (`must be a finite number (nan/inf are not accepted)` — TOML
  spells `nan`/`inf` natively, and an integer beyond `float` range
  counts as `inf`; every numeric routing/price key shares the check);
  a malformed `extra_headers`
  table (non-string value, invalid header name, one name spelled twice)
  or one naming `authorization`, `x-api-key` or `x-goog-api-key`
  (`credentials come from credential = "env:VAR", never the config file`);
  a `body_defaults` that is not a table or holds values with no JSON
  form.
- **Passthrough upstreams** additionally refuse `inject_system_note =
  true` (I-4), any `monthly_budget_*` (subscription usage is the
  vendor's to meter), and `extra_headers` / `body_defaults`
  (`extra_headers/body_defaults apply only to none/env upstreams`).
- **`[routing]`**: an unknown key; `enabled = true` without
  `default_upstream` or with an empty table
  (`[routing] default_upstream is required when enabled = true`, I-6); a
  default naming an unknown upstream, keyed by something that is not a
  protocol, or whose upstream's protocol differs from its key;
  `max_hops < 1`;
  `request_deadline_seconds <= 0`; an unknown `plan_limit_detection`; a
  malformed or **empty** `plan_limit_headers` table
  (`set plan_limit_detection = "off" to disable classification instead`);
  an empty `oauth_beta_marker`; `throttle_retry_max_seconds < 0`;
  `budget_reset_day` outside 1–28; a non-finite
  `request_deadline_seconds` / `throttle_retry_max_seconds`
  (`must be a finite number (nan/inf are not accepted)`).
- **`[[routing.rule]]`**: an unknown key; an id outside the name grammar
  above or used twice; a missing `match`; an unknown `match.protocol` or
  `match.auth`; a `match.model` that is neither a glob string nor a
  non-empty list of globs; a malformed `match.headers` table; an unknown
  `upstream` or chain member, or one whose protocol differs from
  `match.protocol` (I-5); an `on_status` key outside `100–599` / `4xx` /
  `5xx` / the two special keys, listed twice (`529` and `"529"` are the
  same key), or with a value that is neither a non-empty name list nor
  `"retry-same"`; a chain (`on_status` or `on_budget_exhausted`) that
  names the rule's own upstream (`names the rule's own upstream 'X': a
  chain continues to other upstreams (use "retry-same" to retry this
  one)`) or lists a member twice (`lists 'X' twice` — a wasted hop
  against `max_hops`); a passthrough upstream in any chain (I-1, I-2);
  an unknown `reissue_policy`.
- **`[prices]`**: an unknown key; an empty `table`; an override that is
  not a table, lacks `input` or `output`, carries an unknown key, or
  holds a negative or non-finite price.
- **At `serve`, `serve --check` and `doctor` only** (`parse_config` never
  reads the environment): an `env:VAR` credential whose variable is
  unset or empty in the proxy's environment — the message names the VAR
  and its upstream, never a value.
- **At `serve`, `serve --check` and SIGHUP** (and the config editor's
  apply): with `[vault] backend = "sqlite"`, a vault file whose `spend`
  table cannot be opened — locked, unwritable — is `spend table in the
  vault database PATH could not be opened (Type: message)`; a reload
  logs `config reload failed; keeping current config` and the editor
  answers 500 (see *Troubleshooting*).

Warnings, not errors: a default that is not `cost = "zero"`;
`[upstreams]` without a `[routing]` table (`[upstreams] is configured but
there is no [routing] table: upstreams are inert`); a `plan_limit_429` /
`throttle_429` key on a non-`anthropic` rule (it never fires); an
`on_budget_exhausted` chain on an upstream with no budget
(`on_budget_exhausted never applies: upstream 'X' has no monthly
budget`); and, under `reissue_policy = "never"`, every chain — `on_status
KEY never applies: reissue_policy = "never" forbids re-issuing to another
upstream` and `on_budget_exhausted never applies: reissue_policy =
"never" forbids re-issuing to another upstream` (`retry-same` stays
live: it never leaves the upstream). Where they show: the build log —
which `serve --check` and every SIGHUP run — and `llm-redact routes
list` always; `doctor` and the `status` posture block only while
routing is **enabled** (both return after one line otherwise), so the
inert-`[upstreams]` warning is visible in the build log and `routes
list` alone.

What the gate does **not** check: per-protocol coverage. A protocol with
neither a matching rule nor a default is a **runtime** 502 `no_route`
(`llm-redact routing: no rule matched and no default_upstream for
protocol X`) — the gate cannot know which protocols your tools will
send. Probe each one before traffic arrives with
`llm-redact routes test --protocol P`.

### Reload and the config editor

`[upstreams]`, `[routing]` and `[prices]` are **hot**: a SIGHUP (or the
same `apply_config` path) rebuilds the price table, re-budgets the
ledger and prunes cooldown state for upstreams that disappeared, without
dropping in-flight requests. They are deliberately **not editable in the
dashboard editor** — the editor's merge preserves them from file truth,
and a `POST /__llm-redact/config` that names one of them is a 400
(`edit the file and reload`). The restart-only set (host, port, vault,
audit, log, TLS, OTel, users, email) is unchanged.

## How a request is routed

1. **Match.** Before redaction: `auth` is classified from the inbound
   headers, `model` read from the body (for `gemini`, from the
   `/models/{model}:verb` path segment when the body has none), and the
   first matching rule selected. Primary = the rule's upstream, else the
   protocol's default, else a 502 `no_route`. **Id-only follow-ups carry
   no model** — `GET /v1/responses/{id}` (and `/input_items`),
   Conversations reads, `GET /v1/files/{id}/content`, Anthropic batch
   polls and `/results`, and the batch submission itself (models inside
   `requests[].params`) — so every rule with `match.model` misses them
   and they land on the **protocol default** unless a model-free,
   path-matched rule claims them. With the recommended config below the
   openai default is local Ollama, so a Codex `GET /v1/responses/{id}`
   would go there — not to the upstream that produced the object — and
   404; the `openai-followups` rule in that config pins the path to
   `openai_key`.
2. **Prepare once.** The system note is injected only if the *first*
   upstream allows it; the resulting redacted body is reused verbatim on
   every later hop (no second redaction pass).
3. **Local gates.** `count_tokens = false` on the selected upstream
   answers the count_tokens path 404. A primary whose budget is exhausted
   answers 402 unless the rule has an `on_budget_exhausted` chain **and**
   its `reissue_policy` allows a re-issue for this request (`never`, or
   `stateless-only` on a stateful body → the 402, the latter with
   `x-llm-redact-reissue: skipped; reason=stateful`); the refused primary
   is hop 1 and the chain continues from hop 2.
4. **Hop loop.** Per hop the outbound headers (`passthrough` → identical
   to inbound; `env:`/`none` → credentials dropped, marker stripped,
   `extra_headers` added (replacing same-named inbound ones), key
   injected) and body rewrites
   (`model_rewrite`, `body_defaults`, and for `openai` Chat Completions
   streams `stream_options.include_usage = true` when absent — never on
   passthrough; Responses API streams already report usage on
   `response.completed`) are computed for the target and the request sent.
   A response `_deliver` would **buffer** (anything that is not SSE or
   NDJSON, 2xx included) is read to completion here, while no byte has
   reached the client; a streamed body is left untouched. The outcome
   is classified into a **status key**:
   - 2xx → deliver (`class=ok`);
   - 429 from an `anthropic` upstream → `plan_limit_429` when any
     configured header carries a listed value, else `throttle_429`
     (`plan_limit_detection = "off"`, and other protocols: plain `429`);
   - any other status → its number, also matching its `4xx`/`5xx` class;
   - a transport fault — connect/send error, a drop while reading a
     buffered body (error or 2xx), a hop timed out by the remaining
     `request_deadline_seconds` budget, or an `env:` variable that
     vanished since startup → `"502"` (matches `502` and `5xx`), counted
     in `upstream_errors_total` for that upstream, `class=transport` when
     it is what gets delivered (also when `reissue_policy` stops the
     chain: the class reports what happened, not the table key).

   With **no chain active** the rule's chain for that key decides:
   **deliver** (no chain — the upstream response is returned as-is),
   **retry-same**, or **re-issue** to the first eligible chain member
   (after the `reissue_policy` gate; the failed upstream enters cooldown
   unless the key is `throttle_429`). **Once a chain is active, any
   non-2xx from a member continues to the next member** — a status the
   rule does not list, a member's own throttle, a `retry-same` key (that
   action belongs to the primary) — and cooldown is applied to the member
   only when its status resolves through the rule's `on_status`
   (throttles excepted). Candidates are the chain in order, skipping
   unknown names, passthrough upstreams (with a WARNING), upstreams in
   cooldown, budget-exhausted upstreams, and `count_tokens = false`
   upstreams on the count_tokens path. An empty candidate set returns the
   last upstream's response unchanged plus
   `x-llm-redact-reissue: skipped; reason=no-candidate`. A response that
   is not delivered is always closed before the next hop. `max_hops` and
   `request_deadline_seconds` bound the loop.
5. **Deliver.** The existing streaming / buffered branches run with three
   wrappers: model restoration (when the rule rewrote it), a usage
   tracker feeding the budget ledger, and the response headers —
   `x-llm-redact-upstream: <name>` and `x-llm-redact-hops: <n>` on every
   re-issued (hops ≥ 2) response, on hop-1 responses only with
   `debug_headers = true`. The delivering upstream's own rate-limit
   headers are relayed, not the failed one's.

### Hops and attempts

`hops` — the debug header, the log line, the spend row's hop number and
the `max_hops` bound all use the same count — is **1 + re-issues**: the
original attempt is hop 1, each re-issue to another upstream adds one,
and the exhausted primary of a budget chain counts as hop 1 with the
first member at hop 2. A `retry-same` attempt is **not** a hop: a
throttled-then-served request is still hop 1, reports `reissue=no`,
records no re-issue spend and gets no debug headers — but it **is** a
request attempt for `/status` → `routing.upstreams.NAME.requests`, which
counts every attempt sent to that upstream (two for a retried request).
`class` names the delivered outcome, not the trigger: a re-issue that
succeeds logs `class=ok reissue=yes` (the trigger is the failed
upstream's `last_error_class` in `/status`); a chain that gives up on a
listed status reports the **resolved** key (`5xx`, `529`,
`plan_limit_429`), an unlisted delivered status its bare number.

### retry-same (throttle)

`throttle_429 = "retry-same"` retries the **same** upstream once after
`max(retry-after, 2 s)`, bounded by the request deadline. When that wait
exceeds `throttle_retry_max_seconds` the proxy does not wait: the 429 is
returned unchanged. A second 429 is returned to the client. A throttle
never re-issues to another upstream and never triggers a cooldown; the
retry is not a hop (see above).

### Cooldown (circuit breaker)

Every status that resolved through the rule's `on_status` — except
`throttle_429` / `retry-same` — puts the **failed** upstream into
cooldown for `min(3600, max(cooldown_seconds, retry-after))` seconds:
the primary when its chain is entered, and a chain member whose own
failure is a listed key. A member that fails with a status the rule does
**not** list is skipped for this request without a cooldown (the chain
moves on regardless). A zero `cooldown_seconds` still records
`last_error_class` / `last_error_at`. Upstreams in cooldown are skipped
as chain candidates; if every candidate is unavailable the last upstream
error is returned unchanged. State is in-process (no DB): it survives
SIGHUP (names removed by the reload are dropped) and is lost on restart;
`/status` shows `state`, `cooldown_remaining_seconds`,
`last_error_class` and `last_error_at` per upstream.

### Statelessness guard

With `reissue_policy = "stateless-only"` a re-issue happens only if the
redacted request carries no assistant-role message with a `thinking` or
`redacted_thinking` block. Otherwise the upstream's response (typically
the plan-limit 429) is returned unchanged **plus**
`x-llm-redact-reissue: skipped; reason=stateful`, so the harness or user
can switch lanes deliberately. Rationale: signed thinking blocks are
bound to the originating model and credential; swapping mid-conversation
risks `bound to a different conversation` rejections and discards the
prompt cache. Claude Code's main thread is stateful once it has thought;
its subagents' fresh requests are not.

### The first-byte rule

Re-issue is possible only before the first byte of an upstream response
body has reached the client. Non-streaming responses are buffered inside
the hop loop, so status-based fallback always applies to them — and so
does a transport drop in the middle of such a body, chained as `"502"`.
A stream (SSE/NDJSON) is never pre-read: one that fails after its first
event is propagated as-is (error event or connection close), logged
`class=stream_error`, and never re-issued. Only mid-**stream** faults
are outside fallback.

## Budgets and the price table

**Accounting source.** After every 2xx response with a parsable usage
block the spend is recorded against the upstream that produced it, with
the hop number that reached it: Anthropic `usage` (`input_tokens`,
`output_tokens`, `cache_creation_input_tokens`,
`cache_read_input_tokens`; on SSE `message_start` then the cumulative
`message_delta`), OpenAI-compatible `usage` (`prompt_tokens`,
`completion_tokens`, `prompt_tokens_details.cached_tokens` — cached tokens
are subtracted from the input count; on SSE the last chunk carrying
`usage`, which is why `stream_options.include_usage` is injected into
Chat Completions bodies on `env:`/`none` upstreams when absent — never on
passthrough; Responses API streams carry `usage` on `response.completed`
and are read from there), Ollama native
(`prompt_eval_count`/`eval_count` on the `done: true` line), Gemini
`usageMetadata`. **Passthrough upstreams are recorded too**, priced
through the same table: the USD figure `spend`, `/status`, `llm-redact
status` and the dashboard show for the subscription (OAuth) lane is a
**list-price equivalent** — what those tokens would cost at the table's
API rates, for information only; nobody is billed it. The token counts
are the honest number there. Passthrough upstreams just have no budget.

**Storage.** With `[vault] backend = "sqlite"` the rows live in a `spend`
table inside the vault DB file (own connection, WAL, `0600`). The INSERT
runs synchronously in the response finalizer on the event loop (the
vault's own writes make the same call) with a **250 ms** lock wait
(`spend.REQUEST_PATH_BUSY_TIMEOUT_MS`, against the 5 s the offline
`llm-redact spend` can afford): a write lock held by another process on
the shared vault file (a second proxy, a tool holding a transaction)
costs the request milliseconds, not seconds — that row is **not
persisted** (`spend store write failed for upstream NAME (Type); the row
is counted in memory only`, exception type only) but stays counted in
the in-process period totals, so budget enforcement is unaffected until
restart while `llm-redact spend` and the next proxy start will not see
it. A failed write never fails the request, while a vault file whose
`spend` table cannot be **opened** is a
startup/`serve --check`/reload `ConfigError` (`spend table in the vault
database … could not be opened`). On the memory backend — and on every
RDBMS vault backend (postgresql/mysql/oracle/dbapi; the server-side
schema is the pro package's, spend is a local operator ledger) — spend is
in-process only, reset on restart, and `llm-redact spend` says so.

**Period and enforcement.** The period is `[budget_reset_day of this
month, budget_reset_day of next month)` in UTC and rolls over lazily. An
upstream is **exhausted** when its USD spent ≥ `monthly_budget_usd` OR
its total tokens ≥ `monthly_budget_tokens` (whichever limit is set).
Exhausted upstreams are skipped in chains; a direct request to one is a
provider-shaped **402**
(`llm-redact routing: upstream NAME budget exhausted for this period`,
`class=budget_exhausted`, hops=1), unless
the rule's `on_budget_exhausted` chain continues — subject to the rule's
`reissue_policy` like any chain — in which case the refused primary is
hop 1 and the first member that answers is hop 2. Zero-cost upstreams
ignore budgets; the check is made at request time, so a request already
in flight can overshoot the cap by its own usage.

## Model discovery (`expose_models`)

With `routing.expose_models = true`, `GET /v1/models` is answered
locally (never forwarded) so
`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` works against the proxy:
Anthropic shape when the request carries an `anthropic-version` header
(`{"data":[{"type":"model","id":…,"display_name":…,"created_at":…}],"has_more":false,…}`),
OpenAI shape otherwise (`{"object":"list","data":[{"id":…,"object":"model","created":0,"owned_by":"llm-redact"}]}`).
The list is `model_catalog` followed by the literal (glob-free) model
names from rules in file order, deduplicated; a request is recorded with
provider `routing`. The local answer sits where every other
proxy-generated reply for a real API path sits — **after** the
`[providers.NAME] enabled = false` 502 and the named-user 403: the path
infers provider `openai` whatever the headers say, so with
`[providers.openai] enabled = false` the discovery call is refused 502
(the Anthropic-shaped one carrying `anthropic-version` included), and
with 2+ verified named users an unauthenticated client gets the 403
(nothing about the catalog is learnt past a gate). Claude Code keeps only ids containing `claude` or
`anthropic` — name your local aliases accordingly (or use
`model_rewrite`) if you want them to appear in its picker. With
`expose_models = false` the endpoint passes through to the upstream as
in the [API coverage matrix](api-coverage.md).

## Observability

**Log line.** A routed request's line gains the route fields; unrouted
requests keep today's line exactly:

```text
POST /v1/messages -> 200 rule=max-lane upstream=anthropic_oauth hops=1 auth=oauth class=ok reissue=no redacted: EMAIL×1
```

`class` ∈ `ok`, `plan_limit_429`, `throttle_429`, a bare status, the
`4xx`/`5xx` key that resolved, `transport`, `stream_error`,
`budget_exhausted`, `no_route`; `reissue` ∈ `yes`, `no`,
`skipped:stateful`, `skipped:no-candidate`. `class` is the **delivered**
outcome: a successful re-issue logs `class=ok reissue=yes` (hops ≥ 2);
a throttled-then-served `retry-same` logs `hops=1 reissue=no`. A request
that took the protocol default prints `rule=-`; a `no_route` refusal
prints `upstream=-`. Header values, model text and key material are
never logged.

**`/__llm-redact/recent` and `/events` rows** gain one key, `route`:
`null` when unrouted, else `{"rule": id-or-null, "upstream": name,
"hops": n, "auth": …, "class": …, "reissue": …}`. Audit rows are
unchanged.

**`/__llm-redact/status`** carries a `routing` block — `{"enabled":
false}` when routing is off, otherwise:

```json
{"enabled": true, "default_upstreams": {"anthropic": "ollama"}, "rules": 5, "reissues_last_hour": 2,
 "plan_limit_detection": "headers", "expose_models": false,
 "upstreams": {"anthropic_oauth": {"protocol": "anthropic", "credential": "passthrough", "cost": "metered",
   "legacy": false, "state": "healthy", "cooldown_remaining_seconds": 0.0,
   "requests": 12, "reissues_last_hour": 0, "last_error_class": null, "last_error_at": null,
   "spend": {"period": "2026-09", "in_tokens": 0, "out_tokens": 0, "cache_read": 0, "cache_write": 0, "usd": 0.0,
             "budget_usd": null, "budget_tokens": null, "remaining_usd": null, "remaining_tokens": null,
             "unpriced_rows": 0, "reissue_usd": 0.0, "reissue_tokens": 0}}},
 "unpriced_models": [], "unpriced_models_dropped": 0, "warnings": []}
```

`state` is `healthy`, `cooldown` or `budget_exhausted`; `requests`
counts every attempt sent to the upstream (a `retry-same` counts twice);
`spend.usd` on a passthrough upstream is the list-price equivalent
described under *Budgets*; `unpriced_models` is the runtime-observed
list and `unpriced_models_dropped` the ids seen unpriced but never
listed (see *`[prices]`*); `credential` is the **mode** only
(`passthrough` / `none` / `env`) — variable names and values never
appear in `/status`.

**Metrics** (Prometheus text, alongside the existing ones):

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `llm_redact_routed_requests_total` | counter | `upstream`, `rule` | Requests delivered through routing, by the upstream that produced the response and the rule that chose it. |
| `llm_redact_reissues_total` | counter | `from_upstream`, `to_upstream` | Re-issues to the next chain member. |

Transport faults are counted in the existing
`llm_redact_upstream_errors_total{provider}` by upstream name.

**CLI.**

- `llm-redact routes list [--config PATH] [--json]` — one row per rule:
  id, protocol, match summary, upstream, chains, reissue policy, model
  rewrite.
- `llm-redact routes test --protocol P [--model M] [--header NAME=VALUE …]
  [--path PATH] [--auth oauth|gateway-key|none|any] [--config PATH]
  [--json]` — the matched rule id (or the default), the upstream
  (protocol, credential mode, cost, state), the chains per key with
  passthrough/cooldown/budget annotations, the reissue policy and model
  rewrite, and — only for Anthropic's canonical
  `/v1/messages/count_tokens` on a `count_tokens = false` upstream, the
  one path the proxy gates (`routing.is_count_tokens_path`, shared with
  the proxy) — the note `count_tokens: this upstream does not implement
  it — the proxy answers 404`. **No upstream is ever contacted** and no `env:` variable
  needs to resolve. The `state` and the cooldown/budget annotations come
  from one best-effort `GET /__llm-redact/status` (1 s timeout) of the
  running proxy on the **config file's `host`/`port` over plain http** —
  the same call `llm-redact status` makes; a proxy that is not running,
  a `[tls]` listener (client certs are `status`'s business) and a proxy
  named only by `LLM_REDACT_PROXY_URL` are not probed, and the line reads
  `state:    not probed (no proxy answered on the configured listener)`
  — the decision itself is still complete; use `llm-redact status
  [--ca --cert --key]` for the live state in those cases. For
  `--protocol gemini` without `--model`, the model is derived from
  `--path` (or the default path's stand-in) exactly as the proxy does;
  a `?query` on `--path` is stripped before matching.
- `llm-redact spend [--month YYYY-MM] [--json] [--config PATH]` —
  per-upstream tokens and USD, the hop-origin breakdown (how much came
  from re-issues), and remaining budget, read from the sqlite vault file
  **read-only** (the proxy need not be running; it never creates the
  table, WAL side files or the file — `no spend recorded yet` until the
  proxy has written a row; a locked or non-sqlite file is `spend: cannot
  open … read-only (Type)` / `spend: cannot read … (Type)`, exit 1). On
  the memory backend, and on any RDBMS vault backend, it prints that
  spend is in-process only and points at `llm-redact status`. The USD
  column of a passthrough upstream is the list-price equivalent.
- `llm-redact status` prints one routing line per upstream (name,
  protocol, credential mode, state, spend/budget — `(budget ignored:
  zero-cost)` on zero-cost upstreams); its posture block adds upstreams
  in cooldown or budget-exhausted, the runtime-observed unpriced models
  and the routing warnings — only while routing is enabled.
- `llm-redact doctor [--offline]` reports `routing config valid: N
  upstreams, M rules` (or the exact violation), that every `env:`
  credential resolves (FAIL names the VAR only), the routing warnings
  (WARN), the **statically** unpriced ids (WARN, listing ids — the
  config's literal models and metered rewrite targets, not what traffic
  has sent), routing on the memory vault (WARN), and — unless
  `--offline` — a `HEAD` then `GET` probe (3 s timeout) of each
  **referenced** upstream's base URL: every explicit `[upstreams.*]`
  entry plus a legacy auto-registered one only when a rule, chain or
  default names it (an unreferenced legacy `openai`/`gemini`
  registration is not a destination and is not probed). WARN on failure,
  PASS on any HTTP answer; credentials are never sent, and the URL is
  printed as `scheme://host[:port]` only. With routing disabled `doctor`
  prints one PASS line and stops.

**Dashboard and plugins.** The dashboard header gets a `routing` pill
(enabled/disabled) and, when enabled, a per-upstream table (name,
protocol, credential mode, state, tokens, USD — the list-price
equivalent on a passthrough upstream — and budget). The dashboard's
recent-requests table does not render the `route` fields; they are in
the `/recent` and `/events` JSON rows. The agent plugins gain
`/llm-redact:routes` (wrapping `routes list|test`) and
`/llm-redact:spend`, and `/llm-redact:status` / `/llm-redact:recent`
show the routing fields ([plugins.md](plugins.md)).

## Recommended single-user config

The deployment the design targets: a laptop running Ollama natively,
llm-redact in Docker Desktop published on `127.0.0.1:8787` (so Ollama is
`host.docker.internal:11434` from inside the container; use
`127.0.0.1:11434` for a native install), `vault.backend = "sqlite"` on
the `/data` volume. Every `env:` variable is read by the **proxy
process** — pass them to the container with `-e`, or set them in the
service unit — never by the client tools.

```toml
[vault]
backend = "sqlite"                  # placeholder numbering survives restarts (I-7)

# ------------------------------------------------------------------
# Upstreams: named destinations. Each carries exactly one protocol.
# ------------------------------------------------------------------
[upstreams.anthropic_oauth]
protocol   = "anthropic"
base_url   = "https://api.anthropic.com"
credential = "passthrough"          # forward client auth + anthropic-beta byte-exact
inject_system_note = false          # never mutate `system` on this upstream (the default here)

[upstreams.anthropic_key]
protocol   = "anthropic"
base_url   = "https://api.anthropic.com"
credential = "env:ANTHROPIC_API_KEY"   # strip inbound auth, send x-api-key
monthly_budget_usd = 100
cooldown_seconds   = 60

[upstreams.openai_key]
protocol   = "openai"
base_url   = "https://api.openai.com/v1"
credential = "env:OPENAI_API_KEY"
monthly_budget_usd = 50
cooldown_seconds   = 60

[upstreams.gemini_openai_compat]
protocol   = "openai"
base_url   = "https://generativelanguage.googleapis.com/v1beta/openai"
credential = "env:GEMINI_API_KEY"
monthly_budget_usd = 30

[upstreams.openrouter]
protocol   = "openai"
base_url   = "https://openrouter.ai/api/v1"
credential = "env:OPENROUTER_API_KEY"
monthly_budget_usd = 25
extra_headers = { "HTTP-Referer" = "http://localhost", "X-Title" = "llm-redact" }
# OpenRouter's provider allowlist is its own body field; applied only when
# the request does not already carry `provider`.
body_defaults = { provider = { only = ["anthropic", "openai", "google-vertex", "azure", "together", "deepinfra"], data_collection = "deny" } }

[upstreams.ollama]
protocol   = "anthropic"            # Ollama serves /v1/messages natively (>= 0.14)
base_url   = "http://host.docker.internal:11434"
credential = "none"
cost       = "zero"                 # never counts against any budget; always eligible
count_tokens = false                # Ollama hangs on count_tokens: answer 404 locally

[upstreams.ollama_openai]
protocol   = "openai"
base_url   = "http://host.docker.internal:11434/v1"
credential = "none"
cost       = "zero"

[upstreams.ollama_native]
protocol   = "ollama"               # the native /api/* surface (OLLAMA_HOST clients)
base_url   = "http://host.docker.internal:11434"
credential = "none"
cost       = "zero"

# ------------------------------------------------------------------
# Routing: ordered rules. First match wins. Unmatched -> default_upstream.
# ------------------------------------------------------------------
[routing]
enabled          = true
# FAIL-CLOSED: unknown traffic stays local, per protocol. A protocol with
# neither a rule nor a default answers 502 rather than guessing.
default_upstream = { anthropic = "ollama", openai = "ollama_openai", ollama = "ollama_native" }
max_hops         = 3                # original + up to 2 re-issues
request_deadline_seconds = 600      # total wall time across hops
plan_limit_detection = "headers"    # "headers" (default) | "off"
throttle_retry_max_seconds = 30     # longer retry-after: return the 429 unchanged
budget_reset_day = 1                # UTC calendar month by default
debug_headers    = false            # x-llm-redact-* also on hop-1 responses
expose_models    = false            # answer GET /v1/models locally (model discovery)

# Claude Code subagent named Explore -> local model, zero quota use.
[[routing.rule]]
id       = "explore-local"
match    = { protocol = "anthropic", headers = { "x-claude-code-agent-id" = "Explore" } }
upstream = "ollama"
model_rewrite = "muse-64k"

# Max lane: byte-exact OAuth pass-through, with guarded fallback.
[[routing.rule]]
id       = "max-lane"
match    = { protocol = "anthropic", model = "claude-*", auth = "oauth" }
upstream = "anthropic_oauth"
on_status = { plan_limit_429 = ["anthropic_key", "ollama"], 529 = ["anthropic_key"], throttle_429 = "retry-same" }
reissue_policy = "stateless-only"   # never re-issue a request that carries signed thinking blocks

# API-key Anthropic lane (a second Claude Code, or Cline, given a gateway token).
# Chains stay inside the rule's protocol: OpenRouter speaks openai, so it can
# only back the openai lanes below (no protocol translation).
[[routing.rule]]
id       = "anthropic-key-lane"
match    = { protocol = "anthropic", model = "claude-*", auth = "gateway-key" }
upstream = "anthropic_key"
on_status = { 429 = ["ollama"], 5xx = ["ollama"] }
on_budget_exhausted = ["ollama"]

[[routing.rule]]
id       = "openai-lane"
match    = { protocol = "openai", model = "gpt-*" }
upstream = "openai_key"
on_status = { 429 = ["openrouter", "gemini_openai_compat", "ollama_openai"], 5xx = ["openrouter", "gemini_openai_compat", "ollama_openai"] }

[[routing.rule]]
id       = "gemini-lane"
match    = { protocol = "openai", model = "gemini-*" }
upstream = "gemini_openai_compat"
on_status = { 429 = ["ollama_openai"] }

# Id-only follow-ups carry no model (GET /v1/responses/{id}, its
# /input_items, conversation item reads, GET /v1/files/{id}/content), so
# every model-globbed rule above misses them and they would take the
# openai default — local Ollama, which never produced the object. A
# path-matched, model-free rule pins them to the lane that did; add
# siblings for /v1/conversations/* and /v1/files/* if those live on a key.
[[routing.rule]]
id       = "openai-followups"
match    = { protocol = "openai", path = "/v1/responses/*" }
upstream = "openai_key"

[[routing.rule]]
id       = "local-openai"
match    = { protocol = "openai", model = ["muse-*", "gemma4*", "gpt-oss*", "laguna*", "nomic-*", "embeddinggemma*"] }
upstream = "ollama_openai"

# ------------------------------------------------------------------
# Prices: vendored table + user overrides. USD per 1M tokens.
# ------------------------------------------------------------------
[prices]
table = "builtin"                   # or a path to a JSON/TOML file of the same shape

[prices.override."claude-sonnet-5"]
input = 2.0
output = 10.0
cache_read = 0.2
cache_write = 2.5
```

Client wiring (PowerShell; the bash form is the same variables):

```powershell
# Max lane: Claude Code on the subscription, through the proxy, no credential variable.
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787"

# Overflow lane (a separate shell): any non-OAuth token classifies as gateway-key;
# the proxy discards it and substitutes ANTHROPIC_API_KEY on the anthropic_key upstream.
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787"; $env:ANTHROPIC_AUTH_TOKEN = "gateway-local"

# OpenAI-format clients (Cline, Continue, Open WebUI, Codex wire_api=chat).
$env:OPENAI_BASE_URL = "http://127.0.0.1:8787/v1"
```

The overflow lane's placeholder token is used **only** for `auth`
classification and is dropped before the request leaves the proxy —
never hand clients a real key for it. Docker notes: publish to loopback
only (`-p 127.0.0.1:8787:8787`), mount `/data` for the sqlite vault (which
also holds the `spend` table), pass the `env:` variables with `-e`, and
on Linux hosts without Docker Desktop replace `host.docker.internal`
with the host address (`--add-host=host.docker.internal:host-gateway`).
Windows itself remains an unsupported host for the native proxy; the
container path is the supported one there.

## Troubleshooting

Keyed by the exact strings, like [troubleshooting.md](troubleshooting.md).

**`llm-redact routing: no rule matched and no default_upstream for protocol X`** —
a proxy-generated 502 (`class=no_route`): routing is enabled and this
protocol has neither a matching rule nor a default. Add a default for it
(`default_upstream` table form) or a catch-all rule. Remember that an
explicit `[upstreams.NAME]` named after a legacy provider replaces its
auto-registration, protocol included. `llm-redact routes test --protocol X`
reproduces the decision offline.

**`llm-redact routing: upstream NAME does not implement count_tokens`** —
a local 404: the selected upstream has `count_tokens = false` (Ollama)
and the tool called `/v1/messages/count_tokens`. Expected for the local
lane; there is deliberately no fallback for this path. Tools treat the
404 as "no count available".

**`llm-redact routing: upstream NAME budget exhausted for this period`**
(HTTP 402, `class=budget_exhausted`) — the upstream's
`monthly_budget_usd` / `monthly_budget_tokens` is spent for the current
period. `llm-redact spend` shows the period and totals; raise the cap,
wait for `budget_reset_day`, or give the rule an `on_budget_exhausted`
chain — and check the rule's `reissue_policy`: under `never` the chain
never runs, under `stateless-only` a stateful request still gets the
402 (with `x-llm-redact-reissue: skipped; reason=stateful`). Chains
already skip exhausted members.

**`spend table in the vault database PATH could not be opened (Type: message)`**
(`ConfigError` from `serve`, `serve --check`, a SIGHUP reload or the
config editor) — `[vault] backend = "sqlite"` and routing is enabled,
but the vault file's `spend` table could not be opened: the file or its
directory is unwritable, locked by another process, or not a sqlite
database. The exception type and sqlite's message follow in
parentheses. A reload logs `config reload failed; keeping current
config` and the proxy keeps serving with the previous config; fix the
file (permissions `0600`/`0700`, the lock holder, the path) and reload
again.

**`written to PATH but not applied (…); fix the cause and reload (SIGHUP)`**
(HTTP 500 from the config editor) — the POST validated, and the TOML
was written (with its `.bak`), but hot-applying it failed after the dry
run — typically the spend-table error above. The file on disk is the
new config; the running proxy still has the old one. Remove the cause,
then `kill -HUP` to apply what was written.

**`on_status KEY never applies: reissue_policy = "never" forbids re-issuing to another upstream`** /
**`on_budget_exhausted never applies: reissue_policy = "never" forbids re-issuing to another upstream`**
(parse-time WARNINGs, in the build log and `routes list`) — the rule
lists a chain but its `reissue_policy = "never"` means no request ever
leaves the primary, so the chain is dead configuration (`retry-same`
still works: it stays on the same upstream). Either drop the chain or
set the policy to `stateless-only` / `always`.

**`names the rule's own upstream 'X'`** / **`lists 'X' twice`** (config
errors on `on_status` / `on_budget_exhausted`) — a chain continues to
*other* upstreams: to retry the primary use `"retry-same"`, and a member
listed twice would only burn a hop against `max_hops`.

**`state:    not probed (no proxy answered on the configured listener)`**
(from `routes test`) — the decision is complete, but the live state and
cooldown/budget annotations come from a plain-http `GET /__llm-redact/status`
of the config file's `host`/`port`, and nothing answered there: the
proxy is not running, it listens with `[tls]`, or it is only reachable
through `LLM_REDACT_PROXY_URL`. Use `llm-redact status` (with
`--ca/--cert/--key` for mTLS) for the live state.

**`x-llm-redact-reissue: skipped; reason=stateful`** — the upstream's
error was returned unchanged because the request carries signed thinking
blocks and the rule is `stateless-only`. Start a fresh conversation on
the fallback lane (the overflow shell above), or set
`reissue_policy = "always"` if you accept signature rejections and cache
loss.

**`x-llm-redact-reissue: skipped; reason=no-candidate`** — the chain
existed but every member was ineligible: passthrough (never allowed),
in cooldown, budget-exhausted, or `count_tokens = false` on that path.
`/status` → `routing.upstreams.*.state` says which.

**`edit the file and reload`** (HTTP 400 from the config editor) — the
editor POST named `[upstreams]`, `[routing]` or `[prices]`. Those
sections are file-only; edit the TOML, run `serve --check`, send SIGHUP.

**`serve --check` refuses a routing config** — read the message: it
names the section, rule id or key. The common ones are `enabled = true`
without `default_upstream`, a passthrough upstream inside a chain,
`inject_system_note = true`, a budget, `extra_headers` or `body_defaults`
on a passthrough upstream, a cross-protocol reference, and an `env:`
variable that does not resolve in the proxy's environment (the
container needs `-e VAR`). The full list is under *What `serve --check`
refuses*. A protocol with no rule and no default is **not** a check
error — it is the runtime 502 at the top of this section.

**A 429 keeps coming back instead of falling over** — the 429 classified
as `throttle_429` (no plan-limit header value matched): `retry-same`
retried once, then returned it. Check the response's
`anthropic-ratelimit-unified-*` headers against `plan_limit_headers`,
or the `class=` field of the log line.

**A `GET /v1/responses/{id}` (or conversation/file read) 404s on the
local lane** — id-only follow-ups carry no model, so they take the
protocol default rather than the upstream that created the object. Add
a model-free, path-matched rule (`match = { protocol = "openai", path =
"/v1/responses/*" }`, the `openai-followups` rule in the recommended
config) pointing at that upstream.

**`upstream NAME unavailable: … environment variable VAR is unset or empty`**
(WARNING at request time) — the `env:` variable resolved at startup but
is gone now. The hop is treated as a transport fault: status key `502`,
the rule's `502`/`5xx` chain if any, cooldown for that upstream,
`class=transport` otherwise. Restore the variable and reload.

## Limitations

- **Realtime WebSocket connections are never routed** — they keep the
  legacy `[providers.*]` upstream (no first-message anchor, no hop
  loop). Azure, Vertex, Claude-on-Vertex, Bedrock, Cohere and
  `[providers.custom.*]` traffic likewise stays on the legacy path.
- **No protocol translation.** Clients keep their native protocol; a
  chain crosses vendors only through the vendor's own compatible
  endpoint (OpenRouter, Gemini's OpenAI-compatible surface, Ollama's
  `/v1/messages`).
- **The system note is decided for the first hop only**; the redacted
  body is reused verbatim on re-issue, so a fallback upstream that would
  have wanted the note does not get one (a missing note only weakens
  token preservation; it never leaks).
- **Transport faults are classified as 502** for chain lookup — connect
  and send errors, a drop while reading a buffered body, a hop timed out
  by the remaining `request_deadline_seconds` budget, and an `env:`
  variable that vanished at runtime alike — and, when no chain applies,
  returned as the existing provider-shaped 502.
- **`request_deadline_seconds` is also a per-hop read timeout**: it
  bounds a hung hop (which fails inside the deadline as a transport
  fault) and cuts a delivered stream that stays silent longer than the
  remaining budget (`class=stream_error`); it does not bound a stream
  that keeps sending. The default equals the client's own 600 s read
  timeout, so only a shorter deadline changes stream behaviour.
- **Mid-stream failures are never re-issued** (the first-byte rule);
  buffered bodies, including a mid-body drop, are.
- **Cooldown and reissue counters are in-process**: kept across SIGHUP,
  lost on restart. Spend persists only with the sqlite vault (memory and
  RDBMS vault backends keep it in-process).
- **Budgets are checked before the request**, not reserved — one
  in-flight request can overshoot. Unknown models count in tokens only,
  and a passthrough upstream's USD is a list-price equivalent, not a
  bill.
- **Id-only follow-ups carry no model** and take the protocol default
  unless a path-matched rule claims them.
- **Model restoration** covers the `model` fields listed above; a model
  id echoed inside free text is not rewritten.
- **`/v1/models` is local only with `expose_models = true`**, and Claude
  Code filters the list to ids containing `claude` or `anthropic`.
- **No caching, no multi-tenant virtual keys**, and no Pro-tier gate:
  everything here is FOSS and works with `vault.backend = "sqlite"`,
  `session_mode = "static"`.
