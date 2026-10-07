# Online playgrounds for llm-redact and llm-redact-pro: plan

> **Status: proposal, 2026-10-07.** Nothing described here is built or
> deployed. This is the plan for the owner to approve, change or reject; the
> decisions it needs are listed in [section 16](#16-owner-decisions). Code
> references are to `main` at `130516f` (version 1.11.0). Platform facts were
> checked on 2026-10-06 and 2026-10-07; their sources are listed at the end.

## 1. Summary

Two public playgrounds that anonymous visitors can use without installing
anything, each linked from its own GitHub repository:

- **The llm-redact playground** (this repository). The real proxy runs
  inside the visitor's browser tab: the released wheel's `create_app`, with
  every built-in rule configured, under Pyodide (CPython compiled to
  WebAssembly), in front of a simulated provider that also runs in the tab.
  Visitors type or pick a sample, switch the 11 rule categories (87 rules)
  on and off, choose modes, add deny strings, and see what a provider would
  receive and what their tool gets back, streaming included. There is no
  server: nothing a visitor types leaves the page, every tab has its own
  proxy and vault, and the site is static files on GitHub Pages, deployed
  from this repository's releases.
- **The llm-redact-pro playground** (the llm-redact-pro repository). Each
  visitor gets a private, short-lived sandbox server running the real proxy
  with llm-redact-pro and a demo licence: the browser dashboard (status
  view, config editor, redaction preview), the audit log, per-conversation
  sessions, routing between simulated upstreams, and person-name detection,
  which needs a model no browser can run. It is server-side because the
  proprietary package must never be shipped to visitors' browsers. Each
  sandbox is one Cloudflare container that no other visitor can reach, and
  it is deleted after 10 idle minutes or 30 minutes in total.

The load-bearing decision is that **no two visitors ever share a proxy, a
vault or a configuration**. One llm-redact proxy serves one static vault
session to all of its clients, so on a shared demo proxy any visitor could
type «EMAIL_001» into a prompt that the simulated provider echoes and read
back the email another visitor sent
([section 9.1](#91-why-a-shared-demo-proxy-is-unacceptable)).

## 2. Goals and non-goals

Goals:

1. A visitor with no install and no account sees, within seconds, what
   llm-redact would redact from text or from a real provider request, and
   what comes back.
2. Category toggles (per category and per rule), modes (redact, warn,
   block), deny strings and allowlist values, with the effect visible
   immediately.
3. Fidelity: the playgrounds run the code the proxy runs, not a
   reimplementation. The Free playground runs `create_app` itself; a Pro
   sandbox runs the released image plus the released pro package.
4. Privacy by construction: the Free playground keeps visitor text in the
   tab; a Pro sandbox keeps it in one container's memory for at most 30
   minutes.
5. Honesty: each page says what it does not show (person names in the Free
   playground, binary documents, real model behaviour) and that warn mode
   forwards values.
6. Linked from each repository; no running cost for Free; a bounded,
   alerting cost for Pro.

Non-goals:

- A hosted proxy for real traffic. Neither playground contacts a real
  provider or holds a provider credential.
- Accounts, persistence or sharing of visitor text, content analytics.
- The realtime (WebSocket) relay, document extraction of binary uploads,
  and voice.
- Any change to the product's threat model, bind policy, local API or
  security headers for the playgrounds' sake.

## 3. What was verified while planning

These checks ran on 2026-10-07 against the 1.11.0 wheel built from this
tree at `130516f`; [appendix C](#appendix-c-reproducing-the-checks) shows how to repeat
them. In M1 their scripts become the seed of the playground's test suite
([section 14](#14-testing-and-gates)).

| Check | Result |
| --- | --- |
| Library-level round trip under Pyodide 314.0.7 (CPython 3.14.2) in Node 22: redact a request, have a simulated provider echo it in 7-character chunks, rehydrate. Run for Anthropic Messages, OpenAI Chat Completions, OpenAI Responses, Gemini, Bedrock Converse stream (binary eventstream) and Ollama (NDJSON). | All six restore every value. Mangled tokens restore with fuzzy matching on and pass through verbatim with it off. Switching off `credit_card` leaves the card in clear. A second vault does not restore the first vault's «EMAIL_001». Import about 0.4 s, run about 0.45 s. The chain loads 37 `llm_redact` modules and no third-party package. |
| The full proxy under Pyodide: `create_app(config, upstream_transport=…)` with an in-process simulated provider, driven by a 30-line ASGI client. | Status 200 with 13 streamed chunks. The provider received placeholders plus the injected system note; the client received the original values. `ProxyState.apply_config` switched `credit_card` off live, and `/__llm-redact/status` counted the detections. Import about 1.8 s, `create_app` about 35 ms, first request about 80 ms. |
| Headless Chromium, page CSP `script-src 'self' 'wasm-unsafe-eval'` delivered by a `<meta>` tag, Pyodide inside a module Web Worker. | Pyodide booted and ran without `'unsafe-eval'`. A worker bootstrapped from a `blob:` URL inherits the page's CSP: its cross-origin `fetch` was blocked and reported. A worker loaded from its own URL, on a host that sends no CSP header, ran with no CSP, and its cross-origin `fetch` was not blocked. Boot 3.1 s, first redaction 263 ms, on localhost. |
| Download size of a cold visit, gzip. | Pyodide: 3.54 MB WebAssembly, 2.50 MB standard library, 0.29 MB JavaScript and lock file. The llm-redact wheel: 0.80 MB. The proxy's locked runtime closure for Python 3.14, without uvicorn and click, which the in-tab proxy never imports (httpx, httpcore, h11, anyio, idna, certifi, starlette): 0.59 MB. Total about 7.7 MB. |
| Python version in the browser. | Pyodide 314.0.7 ships CPython 3.14.2, a version the core's CI already tests (3.11 to 3.14). |
| Pyodide's package index (357 packages). | No spaCy, torch, transformers, GLiNER, Presidio, Stanza or onnxruntime. Person-name detection cannot run in a browser. |
| Rule inventory. | 87 built-in rules over 84 placeholder types. |

Preparing the samples also surfaced four detection gaps that the bench does
not see; they are listed in [appendix A](#appendix-a-detection-gaps-found-while-preparing-samples).

## 4. Free playground: what the visitor sees

One page at `https://asanderson.github.io/llm-redact/playground/` (a custom
domain is decision 3).

**Loading.** The page renders at once and starts the worker. A progress
line names the cost ("Loading the proxy: about 8 MB, once"). When the
browser reports `navigator.connection.saveData`, loading waits for a click.
The Text tab works as soon as the library has imported; the full proxy for
the Request tab imports in the background. On a fast desktop connection the
first result should arrive in about five seconds (an estimate from the
measurements in section 3; M1 measures it on real networks). The page needs
module workers and WebAssembly, so current Chrome, Edge, Firefox or Safari;
an older browser gets a static explanation instead.

**Text tab.** A text box prefilled from a samples menu, scanned again as
the visitor types (250 ms debounce). It shows:

- the original text with every detection marked: an underline plus a label
  with the placeholder type, a colour per category (never colour alone),
  and a tooltip naming the category, the mode and the placeholder issued;
- the text as a provider would receive it;
- counts by type. A warn-mode value is flagged "sent to the provider: warn
  mode observes, it does not protect". A block-mode value shows "this
  request would be refused with a 400 before anything is sent", naming the
  type only.

Each scan uses the current settings and a throwaway vault, as
`ProxyState.preview` does, so the numbering always reflects the text on
screen. The Request tab uses the in-tab proxy's own vault instead, so
tokens stay stable from one request to the next, as in a real session.

**Samples.** Only secret-shaped fakes of the kinds docs/CONTRIBUTING.md
asks for: vendors' canonical examples, documentation address ranges,
reserved domains and the checksum examples the detection docs already pin.
Each was checked with `llm-redact preview` on 2026-10-07:

| Sample | Shows |
| --- | --- |
| An agent tool call with `AKIAIOSFODNN7EXAMPLE` and `postgres://app:hunter2@db-1:5432/app` | AWS_KEY; URL_PASSWORD, where only the password is replaced |
| A support thread with `jane.doe@corp.example`, `+1 212-555-0100` and the card `4111 1111 1111 1111` | EMAIL, PHONE, CREDIT_CARD |
| National IDs `RSSMRA85T10A562S`, `131052-308T`, `943 476 5919`, `046 454 286`, `S1234567D` | IT_CF, FI_HETU, NHS_NUMBER, CA_SIN, SG_NRIC |
| An infrastructure note with `203.0.113.7`, `2001:db8::1` and `127.0.0.1` | IPV4 and IPV6 replaced; loopback kept by the default allowlist |
| A stack trace containing an OpenSSH private-key block | PRIVATE_KEY |
| "Nothing to find" | No change, and no system note injected |

**Request tab.** The visitor picks a provider shape (Anthropic Messages,
OpenAI Chat Completions, OpenAI Responses, Gemini, Bedrock Converse, Ollama
chat), streaming on or off, "the model mangles tokens" (the
`scripts/fake_upstream.py --mangle` grammar: lowercase, hyphens, no zero
padding), fuzzy restoration on or off, and a chunk size of 1 to 64
characters. The visitor's text is placed into that provider's request body,
which is shown as editable JSON. Sending it through the in-tab proxy fills
three panes:

1. **What the provider received**: the outbound body, with placeholders and
   the injected system note highlighted. Each provider gets its own note
   shape: a `system` string (Anthropic), a leading system message (OpenAI
   Chat), `instructions` (Responses), `systemInstruction` (Gemini), a
   Converse `system` block (Bedrock), or nothing for Ollama unless the
   request already has a system message.
2. **What the provider sent back**: the simulated reply chunk by chunk,
   with tokens split across chunks.
3. **What your tool received**: the proxy's output chunk by chunk, with the
   held-back partial token visible ("«EMAIL_" waits for "001»").

A status line gives the HTTP status, the proxy's counts and, for a refusal,
the provider-shaped error body the proxy returned.

**Proxy internals.** A drawer shows the in-tab proxy's
`/__llm-redact/status`, `/__llm-redact/recent` and `/__llm-redact/metrics`:
metadata only, as on a real deployment.

**Settings.** The toggle panel of [section 6](#6-toggle-model).

**Run it yourself.** "Export config.toml" writes the visitor's detection
and rehydration settings, on top of the defaults, through the core's own
emitter `emit_config_toml` (config_write.py:327), the code behind
`llm-redact config show`. The playground's simulated provider addresses are
left out of the export. Next to it are the commands that run the file
locally with `uvx --from llm-redact-proxy llm-redact serve`. "Copy link"
puts the settings, never the text, in the URL fragment.

**Footer.** "Runs entirely in your browser. Nothing you type is sent
anywhere." The llm-redact and Pyodide versions, the source link to the
exact release tag (AGPL-3.0), third-party licences, and the privacy note.

**Accessibility and small screens.** The target is WCAG 2.2 AA: native
checkboxes grouped in fieldsets per category, full keyboard operation,
visible focus, marks that do not rely on colour, results in an
`aria-live="polite"` region, no stream animation under
`prefers-reduced-motion`, and light and dark themes. Below 900 px the panes
stack, the settings become a drawer and the Request tab's three panes
become tabs.

## 5. Free playground: architecture

```text
GitHub Pages (static, one origin)               visitor's browser tab
  playground/index.html, app.mjs, ...     -->   page: UI thread, meta CSP
  pyodide/314.0.7/* (pinned, checksummed)         |  postMessage (JSON)
  wheels/llm_redact_proxy-X.Y.Z-*.whl             v
  wheels/<locked runtime closure>         -->   module Worker (blob:-bootstrapped,
  wheels/llm_redact_playground-*.whl              inherits the page CSP)
                                                  Pyodide + llm_redact_playground
                                                    Session: create_app(config,
                                                      upstream_transport=FakeProviders())
                                                    AsgiDriver: request -> body chunks
                                                    FakeProviders: six provider shapes
                                                    explain(): spans for highlighting
```

Everything lives under a new top-level `playground/` directory, outside
the wheel ([section 10](#10-repository-changes-llm-redact)):

- `playground/web/`: `index.html`, `app.mjs`, `worker.mjs`, `style.css`. No
  framework, no build step, no CDN, no inline script or style.
- `playground/py/llm_redact_playground/`, a pure-Python package built into
  a small wheel when the site is built:
  - `session.py`: `PlaygroundSession` builds the base config, calls
    `parse_config` (config.py:2392) and `create_app(config,
    upstream_transport=FakeProviders())` (proxy.py:8158), and keeps
    `app.state.proxy` (proxy.py:8293). `configure(settings)` turns the
    settings into a raw config dict and runs `parse_config`, then
    `validate_config` (proxy.py:1907), then `apply_config` (proxy.py:1745):
    the same path SIGHUP and the pro editor use. `reset()` builds a new app
    with a new vault.
  - `driver.py`: an ASGI client that hands each `http.response.body` chunk
    to the page as the proxy sends it. It never opens a socket.
  - `upstream.py`: `FakeProviders`, an `httpx.AsyncBaseTransport` that
    answers the six provider shapes, buffered or streamed. It records the
    body it received, echoes the user's last message as received (user
    content only, so the system note's example tokens are not echoed back),
    optionally mangles tokens, and cuts the reply into chunks of the chosen
    size. Its event shapes come from fixtures the adapter tests already pin
    ([section 7.2](#72-the-simulated-providers)).
  - `catalog.py`: the 11 categories of [section 6](#6-toggle-model).
  - `explain.py`: spans for highlighting through `Redactor.explain`
    ([section 7.3](#73-core-addition-redactorexplain)), over detectors
    built from the current settings with `build_detectors`,
    `build_allowlist` and `build_modes` and cached per settings, so the Text
    tab works before the full app has imported.
  - `samples.py`, `presets.py`, and `api.py` (the worker message handlers
    of [section 7.1](#71-page-and-worker-messages)).
- `playground/build.py` assembles `_site/` and writes `SHA256SUMS` plus the
  third-party licence page.
- `playground/pins.toml` holds the Pyodide version and the SHA-256 of each
  runtime file.
- `playground/package.json` and `package-lock.json` are development-only:
  the `pyodide` npm package for the Node test runner, plus Playwright and
  axe-core for browser tests. None of it is shipped.

**The in-tab configuration.** All 87 rules, every mode `redact`, fuzzy
restoration on, the system note on, `max_body_bytes = 1048576`,
`max_body_strings = 10000`, the memory vault, and every provider pointed at
a `*.playground.invalid` base URL that `FakeProviders` answers for. The
`.invalid` names never resolve, Pyodide has no sockets, and the CSP blocks
other origins anyway. `[providers.bedrock]` is set, since Bedrock is a 502
until it is. There is no audit log, NER, extraction or overrides.

**Isolation.** One `create_app` per tab. Its `InMemoryVault` (vault.py:274)
lives in the worker and dies with the tab; "Reset" builds a new app. Only
the settings may be stored in `localStorage`, and only when the visitor
saves them.

**Why the whole app and not only the library.** The full app produces the
bytes a deployment produces: provider-shaped errors, the 400 for a
block-mode value, the 413 for an oversized body, the real streaming
delivery and hold-back, `/status`, and toggles through `apply_config`. The
library-only path (verified above, with no third-party imports) stays as
the fallback and serves the Text tab while the full app is still importing.
The full app costs 0.59 MB and about 1.3 seconds of import more than the
library alone, which is why the Text tab does not wait for it.

**The core interfaces it relies on.** `create_app(config, *,
upstream_transport=)`, `app.state.proxy`, the `DashboardHost` members
`config`, `validate_config`, `apply_config` and `preview` (plugin_api.py:586,
pinned by tests/test_plugin_api_surface.py), `parse_config` and
`emit_config_toml`. A test in this repository pins exactly this surface
([section 14](#14-testing-and-gates)), so a refactor that would break the
playground fails CI here instead of on the live site.

## 6. Toggle model

Every built-in rule is in exactly one category, and no placeholder type
spans two categories, so a mode chosen per category can never put two modes
on one type, which `build_modes` (engine.py:587) refuses. The playground's
catalogue test enforces both properties against `BUILTIN_RULES`, so a new
rule fails CI until it is placed.

| Category | Rules (placeholder type) |
| --- | --- |
| Contact | email (EMAIL), phone_number (PHONE) |
| Network | ipv4 (IPV4), ipv6 (IPV6), url_credentials (URL_PASSWORD: only the password) |
| Payment cards and bank accounts | credit_card (CREDIT_CARD), iban (IBAN) |
| Crypto wallets | eth_address (ETH_ADDRESS), btc_address and btc_bech32 (BTC_ADDRESS) |
| Government IDs (23, shown with country and language) | us_ssn (SSN), canadian_sin (CA_SIN), uk_nino (UK_NINO), aadhaar (AADHAAR), australian_tfn (AU_TFN), spanish_dni (ES_DNI), french_nir (FR_NIR), german_steuer_id (DE_STEUER_ID), brazilian_cpf (BR_CPF), italian_codice_fiscale (IT_CF), swiss_ahv (CH_AHV), swedish_personnummer (SE_PNR), belgian_nn (BE_NN), finnish_hetu (FI_HETU), nhs_number (NHS_NUMBER), norwegian_fnr (NO_FNR), korean_rrn (KR_RRN), singapore_nric (SG_NRIC), chinese_resident_id (CN_RESIDENT_ID), japanese_my_number (JP_MY_NUMBER), thai_id (TH_ID), irish_pps (IE_PPS), mexican_curp (MX_CURP) |
| Cloud and infrastructure secrets (10) | aws_access_key_id (AWS_KEY), aws_secret_key (AWS_SECRET), google_api_key (GOOGLE_API_KEY), gcp_private_key_id (GCP_KEY_ID), google_oauth_client_secret (GOOGLE_OAUTH_SECRET), azure_storage_key (AZURE_STORAGE_KEY), digitalocean_token (DO_TOKEN), hashicorp_vault_token (VAULT_TOKEN), tailscale_key (TAILSCALE_KEY), doppler_token (DOPPLER_TOKEN) |
| Developer platform tokens (13) | github_token and github_fine_grained_pat (GITHUB_TOKEN), gitlab_pat and gitlab_token (GITLAB_TOKEN), bitbucket_app_password (BITBUCKET_TOKEN), atlassian_api_token (ATLASSIAN_TOKEN), linear_api_key (LINEAR_KEY), notion_token (NOTION_TOKEN), figma_pat (FIGMA_PAT), postman_key (POSTMAN_KEY), sentry_token (SENTRY_TOKEN), new_relic_key (NEW_RELIC_KEY), grafana_service_account (GRAFANA_TOKEN) |
| AI and agent-stack keys (16) | anthropic_api_key (ANTHROPIC_KEY), openai_api_key (OPENAI_KEY), openrouter_key (OPENROUTER_KEY), groq_key (GROQ_KEY), xai_key (XAI_KEY), perplexity_key (PERPLEXITY_KEY), huggingface_token (HF_TOKEN), replicate_token (REPLICATE_TOKEN), langsmith_key (LANGSMITH_KEY), langfuse_secret_key (LANGFUSE_KEY), pinecone_key (PINECONE_KEY), jina_key (JINA_KEY), tavily_key (TAVILY_KEY), firecrawl_key (FIRECRAWL_KEY), nvidia_key (NVIDIA_KEY), cerebras_key (CEREBRAS_KEY) |
| SaaS, messaging and payments (7) | slack_token (SLACK_TOKEN), stripe_key (STRIPE_KEY), sendgrid_key (SENDGRID_KEY), twilio_id (TWILIO_ID), telegram_bot_token (TELEGRAM_BOT_TOKEN), shopify_token (SHOPIFY_TOKEN), airtable_pat (AIRTABLE_PAT) |
| Package registries and data platforms (5) | npm_token (NPM_TOKEN), pypi_token (PYPI_TOKEN), databricks_token (DATABRICKS_TOKEN), supabase_key (SUPABASE_KEY), planetscale_token (PLANETSCALE_TOKEN) |
| Private keys, JWTs and generic secrets (3) | private_key (PRIVATE_KEY), jwt (JWT), generic_secret (SECRET) |
| Person names (NER) | Shown disabled in the Free playground, with the reason; live in a Pro sandbox |

How settings become configuration:

- **On and off.** The category switch and the per-rule checkboxes become
  `[detection] enabled`: the checked rules in `BUILTIN_RULES` order, never
  in screen order, because detector order breaks ties (engine.py:312).
- **Modes.** Chosen per category, with a per-type override in an advanced
  view, and written to `[detection.modes]` for every rule of the type.
- **Languages.** A preset that checks or unchecks Government IDs, not the
  `[detection] languages` key, so the checkboxes stay the single source of
  truth.
- **Deny strings.** `[detection] deny`, validated by `parse_config` (no
  guillemets), capped at 50 entries of 200 characters.
- **Allowlist.** Exact values (`[detection] allowlist`) and per-type values
  (`[detection.allowlist_by_type]`). Regular-expression allowlist patterns
  and custom rules are not offered in v1: a pattern in a shared link could
  hang the tab of whoever opens it.
- **Rehydration.** The system note on or off (`inject_system_note`) and
  fuzzy restoration (`[rehydration] fuzzy`).

None of these keys is restart-only (config.py:203), so every change is one
`apply_config` call. Rebuilding the detectors costs well under a
millisecond: 118 µs for all 87 rules plus 211 µs for the per-string plan,
measured natively.

## 7. Interfaces

### 7.1 Page and worker messages

JSON messages, each carrying a request id:

| Page sends | Worker answers |
| --- | --- |
| `boot {settings}` | `ready {version, pyodide, categories, rules: [{name, type, category, languages}], timings}` |
| `explain {id, text}` | `explained {id, spans: [{start, end, type, category, mode, deny, placeholder}], redacted, blocked, counts, warnings}` |
| `configure {id, settings}` | `configured {id, ok, error}`, where `error` is the `ConfigError` text, which never echoes a value |
| `send {id, preset, body, stream, mangle, chunk}` | `upstream {id, body}`, then `upstream_chunk {id, n, text}` and `client_chunk {id, n, text}` as they happen, then `done {id, status, headers, ms}` |
| `internals {id}` | `internals {id, status, recent, metrics}` |
| `export {id}` | `exported {id, toml}` |
| `reset` | `ready …` |

Text is capped at 200,000 characters. After 3 seconds without an answer the
page offers "Stop", which terminates and restarts the worker, so even a
pathological input can only stall the visitor's own tab.

### 7.2 The simulated providers

| Preset | Proxy route | Streamed reply | Where the shapes come from |
| --- | --- | --- | --- |
| Anthropic Messages | `POST /v1/messages` | SSE `content_block_delta` text deltas | scripts/fake_upstream.py |
| OpenAI Chat Completions | `POST /v1/chat/completions` | data-only SSE chunks, `finish_reason`, `[DONE]` | tests/test_provider_openai.py |
| OpenAI Responses | `POST /v1/responses` | `response.output_text.delta`, `.done`, `response.completed` | tests/test_provider_openai_responses.py |
| Gemini | `POST /v1beta/models/{m}:streamGenerateContent?alt=sse`, and `:generateContent` | SSE candidates, `finishReason` | tests/test_provider_gemini.py |
| Bedrock Converse | `POST /model/{id}/converse-stream`, and `/converse` | binary eventstream frames | scripts/fake_upstream.py |
| Ollama chat | `POST /api/chat` | NDJSON lines, `done: true` | scripts/fake_upstream.py, tests/test_provider_ollama.py |

### 7.3 Core addition: `Redactor.explain`

The Text tab needs the offsets of what was found. No public API returns
them today: `ProxyState.preview` (proxy.py:1943) and `llm-redact preview`
return the redacted text and counts. The proposal is one small public
method in `redactor.py`:

```python
@dataclass(frozen=True)
class ExplainedSpan:
    start: int
    end: int
    detector_type: str
    mode: str                # "redact" | "warn" | "block"
    deny: bool               # a deny string (tier 0): always redacted
    placeholder: str | None  # the token issued; None for warn and block


def explain(self, text: str) -> list[ExplainedSpan]: ...
```

It uses the same winners as `redact_text` (`_winners`, redactor.py:314),
the same mode decision, and the same placeholder numbering through
`_placeholder`, floors included. It reports a block-mode winner as a span
instead of raising, and charges the string budget once. A differential
test proves that rebuilding the text from the spans equals `redact_text`
over the recall and false-positive corpora, a hypothesis test covers random
text, and the mutation and complexity gates cover it like the rest of
`redactor.py`. `llm-redact preview --json` gains an additive `spans` list
(offsets, type and mode, no values: the caller has the text).
`DashboardHost.preview` and its pinned return shape do not change;
llm-redact-pro can adopt `explain` in its own time.

If the owner declines this (decision 2), the playground keeps the same
logic in `explain.py` over the private `_winners`, pinned to the release it
ships with and covered by the interface test.

## 8. Pro playground

### 8.1 What the visitor sees

The landing page explains the sandbox and asks for a Cloudflare Turnstile
check before "Start my sandbox". The visitor then lands in the real Pro
dashboard of their own proxy, under a banner: "Your sandbox: deleted after
10 idle minutes or 30 minutes in total. Simulated providers only. Do not
paste real secrets."

- **Dashboard tour.** The status view, the config editor, where category
  toggles are `[detection]` edits through the editor's own validation and
  hot apply, and the redaction preview card.
- **Console.** The Free playground's Request tab, sending real HTTP to the
  visitor's own sandbox on the same origin. The body the provider received
  is read from the sandbox's simulated providers through a sandbox-local
  endpoint.
- **Pro features, live.** Audit log rows (metadata only) with the
  tamper-evident chain on; per-conversation sessions, where two
  conversations get separate token namespaces; routing between two
  simulated upstreams, with fallback when one answers 429; and person names
  through spaCy `en_core_web_sm`, labelled as a free core feature that is
  shown here because a sandbox can hold a model.

Named users and seats are left out of v1: a sandbox has one anonymous
operator.

### 8.2 Architecture

A Cloudflare Worker is the broker, and each visitor gets one Cloudflare
container. Cloudflare Containers became generally available on 2026-04-13.

- **`POST /start`** verifies Turnstile, applies a per-IP limit kept in a
  Durable Object keyed by the client address (3 sandbox starts an hour),
  and enforces a global cap through a counter Durable Object (50 live
  sandboxes; above it, "busy, try again in a minute"). It mints a 256-bit
  sandbox id, sets the cookie `__Host-sbx` (Secure, HttpOnly,
  SameSite=Strict, Path=/, Max-Age=1800), and redirects to
  `/__llm-redact/`.
- **Every other request** needs that cookie (without it, the landing page)
  and goes to `env.SANDBOX.getByName(id).fetch(request)`. The Worker caps
  bodies at 64 KiB, applies a per-IP request rate limit, and refuses
  WebSocket upgrades. `/console/*` is served from the Worker's static
  assets with its own CSP.
- **The container class** sets `sleepAfter = "10m"` (a container that
  sleeps restarts with a fresh disk, and so with a fresh vault), an alarm
  that stops it 30 minutes after creation and retires its id, `enableInternet
  = false` (no outbound traffic: the simulated providers run inside the
  container), and the `basic` instance type (1/4 vCPU, 1 GiB, 4 GB disk).
- **The image** starts `FROM ghcr.io/asanderson/llm-redact:X.Y.Z@sha256:…`,
  verified with cosign in CI, and adds the llm-redact-pro wheel, spaCy with
  `en_core_web_sm` installed at build time (no download at run time), the
  simulated providers on 127.0.0.1 (the Free playground's `FakeProviders`,
  built from the core repository at the same tag; they never log bodies),
  and `sandbox.toml`.
- **`sandbox.toml`**:
  - `allowed_hosts = ["<playground host>"]` and `allowed_origins =
    ["https://<playground host>"]`. TLS ends at Cloudflare, so the proxy
    sees plain HTTP and the page's `https` origin is never its own origin;
    listing it is what lets the dashboard and the console pass the
    request-origin rule.
  - `max_body_bytes = 65536` and `max_body_strings = 512`.
  - Every provider pointed at the in-container simulation, and two named
    `[upstreams]` behind a `[routing]` rule for the routing demo.
  - The memory vault, the audit log on the container's ephemeral disk,
    `[log] format = "json"`, NER with `allow_download = false`, and no
    overrides.
  - No `[tls]`. The image's `LLM_REDACT_INSECURE_BIND=1` stays: the
    container's only ingress is the Worker binding, which is the confined
    case that setting exists for.
- **Demo licence.** `LLM_REDACT_LICENSE_KEY` is a Worker secret passed into
  the container at start, never built into the image or committed. Pro
  tier, issued to "llm-redact playground". Rotation is decision 7.

### 8.3 Isolation

One container per sandbox id. An id is never reused, cannot be guessed,
and is routed only for the visitor holding its cookie. Containers have no
public ingress and no egress. Inside its sandbox the visitor is the single
operator the product's threat model assumes, so every product guarantee
holds there unchanged.

### 8.4 Cost

At Cloudflare's published rates, assuming 20 billed minutes per sandbox
(10 active, then 10 idle before it sleeps), 1 GiB and 10 % CPU use:

| Sandboxes a month | Approximate monthly cost |
| --- | --- |
| 1,000 | $8, including the $5 Workers Paid plan |
| 10,000 | $44 |

A spend alert at $50 and the 50-sandbox cap bound the worst case. Fly.io
Machines, the alternative (shared-cpu-1x with 1 GB at $6.70 a month,
billed per second), cost about the same but need their own lifecycle code
and abuse controls; Cloudflare supplies per-name container routing,
Turnstile, rate limiting and an egress switch.

### 8.5 Where it lives

Everything is in llm-redact-pro
([section 11](#11-repository-changes-llm-redact-pro)). This repository
gains nothing for it beyond the playground package itself, whose simulated
providers the Pro image reuses.

## 9. Security and privacy

### 9.1 Why a shared demo proxy is unacceptable

- The Free core's only session router resolves every request to one
  session (sessions.py:30), so a proxy shares one vault among all its
  clients.
- Tokens are numbered per type from 001, so the first email any visitor
  sends is «EMAIL_001», and the fuzzy grammar also accepts «email-1».
- Rehydration restores any token the vault knows. A simulated provider
  that echoes the prompt hands visitor B's typed «EMAIL_001» back to the
  proxy, which restores visitor A's email.
- A token floor only numbers new values; it never stops a restore. Worse,
  a visitor who sends «EMAIL_999999998» pushes the shared counter to its
  limit, and every other visitor's next new email is refused.
- llm-redact-pro's per-conversation sessions are keyed by a hash of the
  first user message, so two visitors who send the same sample share a
  namespace.
- docs/threat-model.md assumes a single-user machine and puts multi-user
  machines out of scope, while docs/SECURITY.md puts cross-session
  placeholder restoration in scope.

Hence one proxy per tab for Free and one container per visitor for Pro.

### 9.2 Free playground controls

- **No server.** Every file is static and served from one origin.
- **CSP**, delivered by `<meta>` because GitHub Pages cannot set response
  headers: `default-src 'none'; script-src 'self' 'wasm-unsafe-eval';
  worker-src blob:; connect-src 'self'; style-src 'self'; img-src 'self';
  base-uri 'none'; form-action 'none'`. The worker is bootstrapped from a
  `blob:` URL so it inherits this policy. That was verified in Chromium
  with `worker-src 'self' blob:`; M1 tightens it to `blob:` alone, which
  forbids a URL-loaded worker, the kind that would run with no CSP on a
  host that sends no headers, and repeats the probe. `frame-ancestors`
  cannot be set by `<meta>`, so the page refuses to run inside a frame.
- **Supply chain.** The Pyodide files are pinned by version and SHA-256 in
  `pins.toml`, fetched from the npm registry tarball whose integrity hash
  is in `package-lock.json`, and verified at build. The Python wheels are
  the core's own `uv.lock` runtime closure, downloaded with
  `--require-hashes`. The llm-redact wheel is built from the release tag in
  the same job. Nothing is loaded from a CDN at run time.
- **No collection.** Visitor text never goes in the URL; share links carry
  settings only. No cookies, no analytics, no third-party requests: a
  browser test asserts that every request in a session is same-origin.
  GitHub, the host, sees ordinary requests for the page's files under its
  own privacy statement.
- **Licence notices.** AGPL-3.0 for llm-redact, with the exact tag's
  source; MPL-2.0 for Pyodide, with its release source; and the bundled
  components' licences on a page generated from the wheels' metadata:
  httpx, httpcore, starlette and idna (BSD-3-Clause), anyio and h11 (MIT),
  certifi (MPL-2.0), and the CPython standard library (PSF).

### 9.3 Pro playground controls

- Turnstile, the per-IP and global caps, the `__Host-` cookie, no egress, a
  64 KiB body cap, no WebSocket, the 30-minute lifetime, a pinned and
  verified image, the licence as a secret, and simulated providers that
  never log bodies.
- Logs stay value-free: the proxy logs in its JSON format, and the
  Worker's logs are disabled or kept for the shortest period the plan
  allows. Only Cloudflare's aggregate request counts are used.
- A visitor's custom rules run only in their own container, which has 1/4
  vCPU and a 30-minute life. The regex worst-case test recommended in
  [section 14](#14-testing-and-gates) should still land before launch.
- The config editor in a public sandbox needs a pro-side review, listed in
  [section 11](#11-repository-changes-llm-redact-pro): no editable key may
  read a file path into a response (`[license] key_file` first), and the
  demo licence key must never appear in any dashboard or `/status` answer.

### 9.4 Notices for visitors

Free: "This playground runs llm-redact entirely in your browser. Nothing
you type is sent anywhere: the page has no server, and its security policy
blocks connections to other sites. No real AI model is involved; the
provider is a simulation that echoes what it received. The playground
cannot detect person names, which needs a model that does not run in a
browser, and a clean result is not a guarantee for your data. Use
synthetic values."

Pro: "Your sandbox is a private copy of llm-redact with llm-redact-pro.
What you type is processed in that sandbox's memory and deleted with it,
at the latest 30 minutes after you start. Simulated providers only;
nothing is sent to an AI provider. Starting a sandbox runs a Cloudflare
Turnstile check. Records kept: aggregate request counts and Cloudflare's
request logs, which include your IP address. Do not paste real secrets or
personal data." The controller's name, contact and the log
retention period come from the owner, and counsel reviews both texts
(decision 11).

### 9.5 Documentation updates

- docs/threat-model.md gains a "Public playgrounds" section: both are
  separate deployments; the Free one runs in the visitor's browser; the
  Pro one gives each visitor a dedicated sandbox; neither changes the
  product's single-operator model.
- docs/SECURITY.md: playground isolation failures are in scope.
- docs/privacy.md: what each hosted demo receives.

### 9.6 Licensing (not legal advice)

Serving the AGPL wheel to browsers conveys it. The wheel is itself
source, and the footer links the tag and keeps the notices. The
playground's own code lives in this repository under AGPL-3.0. A Pro
sandbox runs the unmodified core beside the proprietary package, so AGPL
section 13, which concerns modified versions, asks nothing more; the
banner still links the core's source. The owner holds copyright in the
playground UI through the CLA, so llm-redact-pro may reuse it. Counsel
confirms (decision 11).

## 10. Repository changes: llm-redact

| Path | Change | Kind |
| --- | --- | --- |
| `src/llm_redact/redactor.py` | `ExplainedSpan` and `Redactor.explain` | Core code; owner confirms (decision 2) |
| `src/llm_redact/cli.py` | `preview --json` gains an additive `spans` key | Core code; owner confirms (decision 2) |
| `tests/test_redactor_explain.py` | Differential test against `redact_text`, hypothesis test, budget and floors | Tests |
| `playground/web/*`, `playground/py/llm_redact_playground/*`, `playground/build.py`, `playground/pins.toml`, `playground/package.json`, `playground/package-lock.json`, `playground/README.md` | The Free playground | Playground only, never in the wheel |
| `tests/test_playground_*.py` | Catalogue drift, simulated providers against every adapter, chunk-size sweeps, `configure` and `reset`, export round trip, the interface pin | Tests |
| `pyproject.toml` | ruff `src`, mypy `files` and pytest `pythonpath` add `playground/py`; mutmut `also_copy` adds `playground` | Config |
| `.github/workflows/ci.yml` | The complexity job runs `complexity_gate.py` a second time with `--src playground/py` | CI |
| `.github/workflows/playground.yml` | Build, Pyodide tests, browser tests, Pages deploy | CI |
| `docs/playground.md` | User page: what it is, the privacy note, what it does not show, running it locally | Docs |
| `docs/README.md` | Index rows | Docs |
| `README.md` | Badge, a paragraph in "Trying it without touching a real API", a Contents entry, once the site is live | Docs |
| `docs/threat-model.md`, `docs/SECURITY.md`, `docs/privacy.md` | [Section 9.5](#95-documentation-updates) | Docs |
| `docs/CONTRIBUTING.md` | "Adding a detection rule" adds: place it in a playground category | Docs |
| `CHANGELOG.md` | One entry per milestone | Docs |

Repository settings, done by the owner: Pages source "GitHub Actions"; the
`github-pages` environment restricted to release tags; About, Website set
to the playground URL; optionally a social preview image.

Unchanged: the wheel's contents, the runtime dependencies,
`/__llm-redact/*`, `_SECURITY_HEADERS`, `DASHBOARD_PATHS`, the session
router and the bind policy. The `playground/` sources ship in the sdist as
`deploy/` and `plugins/` do (decision 5); no Pyodide binary is ever
committed.

## 11. Repository changes: llm-redact-pro

Written against the seams, since the pro repository was not available
while planning:

| Path | Change |
| --- | --- |
| `playground/worker/` | TypeScript Worker (the broker), the Container class, the Turnstile and rate-limit logic, `wrangler.toml` |
| `playground/image/` | `Dockerfile`, `sandbox.toml`, the simulated providers (the core's `llm_redact_playground.upstream` served by uvicorn), the start script |
| `playground/console/` | The Free playground's web UI with a remote HTTP transport in place of the in-tab worker |
| `playground/tests/` | Isolation end to end (visitor B cannot restore visitor A's token; separate containers), cookie and routing, rate limits, lifetime, no egress, image smoke test |
| `.github/workflows/playground.yml` | Build and push the image, `wrangler deploy` to staging and then production with a scoped API token, smoke test, rollback with `wrangler rollback` |
| `README.md` | "Try it online" link to the Pro playground |

To verify in the pro code before launch:

- The dashboard serves a non-loopback peer through the Worker without an
  access gate, and its guarded POSTs pass with the sandbox's
  `allowed_hosts` and `allowed_origins`.
- The demo licence key never appears in any dashboard, `/status` or error
  answer.
- In playground mode the config editor keeps `[license]` read-only, beside
  the restart-only keys it already keeps read-only.
- `/__llm-redact/events` streams through the Worker.

## 12. Hosting, deployment and CI/CD

### 12.1 Free

`.github/workflows/playground.yml`, with SHA-pinned actions like every
workflow here:

1. **build**: `uv sync --frozen`; `uv build --wheel`; `uv export --frozen
   --no-dev --no-emit-project` piped into `pip download --require-hashes
   --only-binary=:all: --no-deps`, with the environment markers evaluated
   for Python 3.14, Pyodide's version, so `typing-extensions` (below 3.13
   only) and `colorama` (Windows only) drop out, and uvicorn and click left
   out because the in-tab proxy never imports them; `npm ci` in
   `playground/` for the pinned Pyodide; `playground/build.py`; upload
   `_site/` as an artifact.
2. **test-pyodide**: the Node runner loads the built wheels under Pyodide
   and runs the probe suite: the library round trip for six providers, the
   full-app round trip, toggles and isolation.
3. **test-browsers**: serve `_site/` and run Playwright on Chromium,
   Firefox and WebKit: boot, spans, toggles, streaming, the block 400 and
   export; zero CSP violations; the in-worker cross-origin probe blocked;
   every request same-origin; no serious axe-core findings.
4. **deploy**: only for a published release, or a manual dispatch naming a
   tag; `actions/upload-pages-artifact` and `actions/deploy-pages`, with
   `pages: write` and `id-token: write` granted to this job alone.

Pull requests that touch `playground/`, `src/llm_redact/` or `uv.lock` run
jobs 1 to 3, so a core change that stops importing under Pyodide, such as
a new top-level import of a module Pyodide lacks (`fcntl`, `pwd`,
`resource`), is caught before a release (decision 9).

Rollback is the manual dispatch with the previous tag. GitHub Pages allows
a 1 GB site, 100 GB of bandwidth a month (a soft limit) and 10 builds an
hour, and serves `cache-control: max-age=600` with ETags. At 7.7 MB a cold
visit, the bandwidth limit covers about 13,000 first visits a month, and
repeat visits revalidate cheaply. If traffic outgrows that, or response
headers become necessary, the same `_site/` moves to Cloudflare Pages with
a `_headers` file. Running cost: none. M1 also checks that Pages serves
`.wasm` as `application/wasm`; if it does not, Pyodide still loads, with a
slower non-streaming compile.

### 12.2 Pro

Built and deployed from llm-redact-pro: `wrangler deploy` to a staging
Worker, a smoke test, then production; a spend alert at $50. The container
image is pinned by digest and rebuilt for every core or pro release.

## 13. Linking from GitHub

llm-redact:

- **README badge**, after the Container badge:
  `[![Playground](https://img.shields.io/badge/playground-try_it_in_your_browser-2ea44f)](https://asanderson.github.io/llm-redact/playground/)`.
- **README paragraph**, first in "Trying it without touching a real API":
  "**In your browser, with nothing installed:** the [llm-redact
  playground](https://asanderson.github.io/llm-redact/playground/) runs
  this proxy, the released wheel under Pyodide, inside your browser tab in
  front of a simulated provider. Pick a sample or type your own text,
  switch rule categories and modes, and see what a provider would receive
  and what your tool gets back. Nothing you type leaves the page. It
  cannot detect person names, which needs a model that does not run in a
  browser." The Contents entry for that section mentions the playground.
- **Repository About**: Website set to the playground URL, in Settings.
- **docs/README.md**: docs/playground.md under "Getting started".
- **The playground page** links back to the repository, the install
  section, the exact release tag and the issue tracker for feedback.

llm-redact-pro:

- **README**: a "Try it online" section linking the Pro playground.
- **The Pro landing page** links the pro documentation, the Free
  playground and the core repository.
- If the pro repository stays private, visitors cannot read its README;
  docs/editions.md in this repository then links the Pro playground once
  Pro is generally available (decision 8).

## 14. Testing and gates

In this repository:

- **Core.** The `Redactor.explain` differential test over the recall and
  fp corpora, a hypothesis test, budget and floor tests, and mutation
  coverage through `redactor.py`'s existing mutmut entry.
- **Playground in CPython**, in the normal `test` job: catalogue drift; the
  simulated providers against each of the six adapters with the chunk size
  swept from 1 to 64 (the streamed result must equal the buffered one);
  block, warn and oversized-body refusals; `configure`, `reset` and the
  isolation of two sessions; export through `emit_config_toml` and back
  through `parse_config`; the interface pin on `create_app(...,
  upstream_transport=)`, `app.state.proxy` and the `DashboardHost` members.
- **Pyodide and browsers**: the `playground.yml` jobs of section 12.1.
- **Gates**: ruff already lints the whole tree; mypy (strict), pytest and
  the complexity gate are extended to `playground/py`;
  `tests/test_docs_index.py` requires the index rows; a CHANGELOG entry per
  milestone.
- **Recommended before the Pro launch**: a regex worst-case test that runs
  every built-in rule on adversarial near-misses at the body cap within a
  time bound. It protects production as much as the sandboxes.

In llm-redact-pro: the isolation, routing, rate-limit, lifetime and egress
tests of section 11.

## 15. Milestones

| Milestone | Scope | Effort | Done when |
| --- | --- | --- | --- |
| M0 Core prerequisite | `Redactor.explain` and the `preview --json` spans | 2 days | Gates green; decision 2 taken |
| M1 Free Text tab | Worker, Pyodide boot, Text tab, toggles, samples, CSP, build and deploy pipeline, privacy note, README link | 7 days | Live on Pages from a release; browser tests green on three engines |
| M2 Free Request tab | Full in-tab proxy, six provider presets, streaming panes, mangling and fuzzy restoration, internals drawer, export | 7 days | Every preset restores at every chunk size, in CI and in browsers |
| M3 Free polish | Accessibility audit, small screens, Save-Data, docs/playground.md, threat-model and privacy sections | 4 days | axe clean; docs merged |
| M4 Pro playground | Image, Worker broker, Turnstile, limits, licence rotation, console, NER, dashboard tour, pro README link | 12 days | Two-visitor isolation test green; spend alert on; one week on staging |
| M5 Launch | Domains if any, repository Website fields, announcement, a month of watching bandwidth, sandbox counts and cost | 2 days | Owner sign-off |

## 16. Owner decisions

Each comes with the recommended default.

1. **The Free playground in this repository.** It shows detection toggles
   and a preview in a browser, which the Pro dashboard sells for a live
   proxy. The plan keeps them inside a demo proxy that exists only in a
   browser tab and can never front a real tool, and it leaves the core's
   HTTP surface as it is. Default: approve on that basis.
2. **`Redactor.explain` and the `preview --json` spans** as core code.
   Default: approve. The fallback is a playground-local shim over private
   methods.
3. **Free URL.** Default: `https://asanderson.github.io/llm-redact/playground/`
   now, a custom domain later.
4. **Free code location.** Default: `playground/` in this repository rather
   than a separate repository, so the gates and the release pin it.
5. **sdist.** Default: ship the `playground/` sources in the sdist, as
   `deploy/` and `plugins/` are shipped.
6. **Pro platform.** Default: Cloudflare Containers behind a Worker;
   fallback Fly.io Machines.
7. **Demo licence.** A leaked key works anywhere until it expires, plus the
   14-day grace period, so its validity should be short; but status warns
   during the last 30 days before expiry. Default: Pro-tier keys for a
   dedicated "llm-redact playground" licensee, valid for 45 days and
   rotated every 14 days. The key in use then always has more than 30 days
   left, so no warning shows, and a leaked key stops working at most 59
   days after it was issued: 45 days of validity plus the 14-day grace
   period.
8. **Linking the Pro playground from this repository** before Pro is
   generally available. Default: no. The core README keeps "coming soon"
   and links the Pro playground at GA.
9. **Pyodide compatibility as a core gate.** Default: the Pyodide job runs
   on every core pull request, informational through M1 and blocking from
   M2.
10. **Analytics.** Default: none for Free; only Cloudflare's aggregate
    counts for Pro.
11. **Legal review** of the notices, the controller's identity, the log
    retention and the licence reading, by counsel.
12. **Budget.** Default: the $50 alert and the 50-sandbox cap, both set by
    the owner.

## 17. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| A core change stops importing under Pyodide | The Pyodide job on core pull requests; the playground deploys a pinned release |
| A Pyodide upgrade changes behaviour | Pinned version and checksums; upgrades are reviewed pull requests with the full browser suite |
| Download size on mobile networks | Size shown before loading; click to load under Save-Data; the Text tab is ready first |
| Visitors read a clean result as a guarantee | The notice, the NER banner and the warn-mode labels |
| The GitHub Pages bandwidth soft limit | Move `_site/` unchanged to Cloudflare Pages |
| A browser engine does not pass the page CSP to a `blob:` worker | The cross-origin probe runs on all three engines in CI; where it fails, the worker runs on the page thread instead |
| A Pro sandbox is abused for CPU, scraping or relaying | Turnstile, per-IP and global caps, no egress, a 30-minute life, 1/4 vCPU |
| The demo licence key leaks | No answer ever shows it (verified in pro); short validity; rotation |
| A cost spike | The spend alert, the concurrency cap and `sleepAfter` |
| The playground relies on internals that get refactored | The interface pin test in this repository |
| Visitors paste real secrets | Free: they stay in the tab. Pro: they stay in one container's memory for at most 30 minutes. Both notices ask for synthetic values |

## Appendix A: detection gaps found while preparing samples

Building the sample set surfaced detection gaps that the bench's recall
gate does not see, because its corpus never puts a value at the end of a
sentence. A scan of all 82 value-generated rules in eight punctuation
contexts (plain, sentence end, comma, parentheses, semicolon, colon after,
quoted, end of text) found:

| Rule | Missed when | Example (missed, then found) |
| --- | --- | --- |
| `belgian_nn` | the value ends a sentence | `NN 93.05.18-223.61.` missed; `NN 93.05.18-223.61, done` found |
| `swiss_ahv` | the value ends a sentence | `AHV 756.1234.5678.97. Done` missed; `AHV 756.1234.5678.97 done` found |
| `ipv6` | the value ends a sentence, or a colon follows it | `Ship to 2001:db8::1. Done` missed; `Ship to 2001:db8::1 now` found |
| `iban` | it is written in the spaced paper format | `GB82 WEST 1234 5698 7654 32` missed; `GB82WEST12345698765432` found |

The first three come from a negative lookahead that rejects a following
`.` (and `:` for IPv6). Each is a leak in ordinary prose and deserves its
own change: a narrower lookahead, such as a period not followed by a
digit, sentence-final contexts in `bench/corpus.py`, and a run of the
fp-corpus gate. The spaced IBAN form needs a grammar decision first. None
of the playground work depends on them.

Also observed: in `postgres://app:hunter2@db.corp.example:5432/app` the
email rule claims `hunter2@db.corp.example`, which is longer than the URL
password span, so the password and host become one «EMAIL_001». Nothing
leaks, but the type is surprising, and the playground will show it.

## Appendix B: housekeeping found while planning

- CLAUDE.md lists `scripts/issue_license.py gen-key|sign`, but that script
  is not in this repository; licence signing lives in llm-redact-pro.
- CLAUDE.md's ops bullet says release.yml builds and signs an `-rdbms`
  image variant; release.yml notes, near line 171, that the variant moved to
  the llm-redact-pro release.
- docs/RELEASING.md says the image is not signed yet; release.yml signs it
  with cosign by digest.

## Appendix C: reproducing the checks

From a checkout at `130516f` (1.11.0), with Node 22, in an empty scratch directory:

1. `uv build --wheel` in the checkout.
2. Download the runtime closure's wheels at the versions in `uv.lock`,
   leaving out uvicorn and click:
   `pip download --no-deps --only-binary=:all: httpx==0.28.1 httpcore==1.0.9 h11==0.16.0 anyio==4.14.2 idna==3.18 certifi==2026.6.17 starlette==1.3.1`.
3. `npm install pyodide@314.0.7`. In Node, `loadPyodide({indexURL})` from
   the package directory, unpack each wheel with
   `pyodide.unpackArchive(new Uint8Array(bytes), "wheel")` (a Node `Buffer`
   is refused), and run the probe below with `runPythonAsync`.

<details>
<summary>The full-app probe, condensed</summary>

```python
import json
import re

import httpx

from llm_redact.config import parse_config
from llm_redact.proxy import create_app


class FakeUpstream(httpx.AsyncBaseTransport):
    """Anthropic-shaped echo of the tokens it received, in 7-character SSE chunks."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(await request.aread())
        seen = json.dumps(body["messages"], ensure_ascii=False)
        reply = "Upstream saw " + " and ".join(re.findall("«[A-Z0-9_]+»", seen)) + "."

        async def stream():
            for i in range(0, len(reply), 7):
                event = {"type": "content_block_delta", "index": 0,
                         "delta": {"type": "text_delta", "text": reply[i : i + 7]}}
                data = json.dumps(event, ensure_ascii=False)
                yield f"event: content_block_delta\ndata: {data}\n\n".encode()
            yield b'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream())


config = parse_config({"providers": {"anthropic": {"upstream_base_url": "https://fake.invalid"}}}, "probe")
app = create_app(config, upstream_transport=FakeUpstream())
# Drive `app` with an ASGI client that records each "http.response.body"
# message. httpx.ASGITransport also works, but it joins the chunks into one.
```

</details>

The CSP check: serve a page carrying the `<meta>` policy of
[section 9.2](#92-free-playground-controls), start a module worker from a
`blob:` URL whose only line imports the worker script by absolute URL, and
`fetch` a foreign URL from inside the worker. Chromium blocks the fetch and
fires `securitypolicyviolation` in the worker.

## Sources

Checked on 2026-10-06 and 2026-10-07:

- Pyodide 314.0.7 on npm (`npm view pyodide`), its
  [WebAssembly constraints](https://pyodide.org/en/stable/usage/wasm-constraints.html)
  and [changelog](https://pyodide.org/en/stable/project/changelog.html).
- MDN, [CSP `script-src` and `'wasm-unsafe-eval'`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Content-Security-Policy/script-src).
- GitHub, [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits);
  the `cache-control` value was read from a live Pages response.
- Cloudflare, [Containers pricing](https://developers.cloudflare.com/containers/pricing/),
  [limits](https://developers.cloudflare.com/containers/platform-details/limits/),
  [outbound traffic](https://developers.cloudflare.com/containers/platform-details/outbound-traffic/),
  [FAQ: ephemeral disk and `sleepAfter`](https://developers.cloudflare.com/containers/faq/),
  [general availability, 2026-04-13](https://developers.cloudflare.com/changelog/2026-04-13-containers-sandbox-ga),
  and [Pages custom headers](https://developers.cloudflare.com/pages/configuration/headers/).
- Fly.io, [pricing](https://docs.fly.io/about/pricing) and
  [restarting Machines](https://fly.io/docs/apps/restart/).
- Precedent: Microsoft Presidio's [hosted demo](https://huggingface.co/spaces/presidio/presidio_demo).
