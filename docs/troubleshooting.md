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
With document extraction (`[extraction]`, [extraction.md](extraction.md)) a PDF or Office file read
completely as clean text is sent instead — with your own key always, under
a credential the proxy holds only when `[extraction] proxy_credential =
true` (off by default); one it could not read completely
(a scanned page, an embedded image) keeps this refusal — `/status`
`inspected_uploads_total` counts each part's outcome (`clean` only for a
part of an upload handed to the upstream; `clean_refused` for one that read
clean in an upload refused before that, for another reason).

## "an uploaded binary file holds values it must redact (…, found in the file's extracted text)"

A 400 on a file upload with an upload inspector (the core's document
extraction): the text read out of a PDF or Office file holds values of the
named types that llm-redact would redact — and it cannot redact inside the
file. Remove them from the document (or send its text instead, which is
redacted), allowlist a value that is not sensitive, set that rule's
mode to `warn` if forwarding it is acceptable, or opt in to
`[extraction] convert`, which sends the document's REDACTED text in place
of the file where the route takes a text file (the model then sees text,
not the original).

## "to allow: llm-redact override CODE --once | --always"

Refusal overrides are on for this proxy (`[overrides] enabled = true`; they
are off by default, and then no refusal carries this line, `llm-redact
override CODE` and `override list` exit 1 naming the setting — `override
revoke ID` still revokes, names it on stderr and exits 0 — and
`/__llm-redact/overrides` answers 404). A proxy started with `serve --config PATH` (or
`LLM_REDACT_CONFIG`) on a file the command would not find by itself prints
the line as `llm-redact override --config PATH CODE --once | --always`:
run it as printed. If the command still says overrides are off in another
file (or that it found no config file) — `list`/`revoke`, or a realtime
close reason whose path did not fit — pass it the proxy's `--config PATH`. A detection refusal you may override for your own data: a block-mode value,
values in a binary upload, a verbatim field, a body that is not JSON, or a
binary part under `binary_uploads = "refuse"`. Run the command yourself in
a terminal. It shows what was refused and waits for `allow` typed there. It
never reads stdin, so an agent cannot confirm for you. After that, send the
request again. `--once` covers the next such request within
`[overrides] ttl_minutes` (default 15). `--always` covers every request
until `llm-redact override revoke`. The value is FORWARDED unredacted. The
error "no pending refusal matches (it is unknown, used or expired)" means
the code was already used or is older than the TTL: repeat the request for
a new one. "belongs to a named user" means the refusal is a named user's,
who approves it in the llm-redact-pro dashboard. See
[overrides.md](overrides.md).

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

## "backend \"NAME\" needs torch, which is not installed" / "needs torch >= 2.6 (CVE-2025-32434), but torch VERSION is installed"

From `doctor`, for an enabled `gliner`, `stanza` or `hf` NER backend: these
backends run on torch, and their library can be present without it
(transformers imports fine without torch and fails only when a model
loads), so serve could not build the backend. torch older than 2.6 is
refused because CVE-2025-32434 lets a crafted checkpoint run code through
`torch.load(weights_only=True)`, which GLiNER uses for `pytorch_model.bin`
files. Install the backend's extra (`uv sync --extra hf`, or
`pip install 'llm-redact-proxy[hf]'`), which requires `torch>=2.6`; on a
CPU-only host take torch from the PyTorch CPU index first
([dependencies.md](dependencies.md)).

## "config reload failed; keeping current config" / "changes require restart"

Log lines from a `kill -HUP`. The first means the new file failed to parse
or build — the proxy deliberately keeps serving the old config rather than
crash; fix the file (`serve --check` shows the error) and HUP again. The
second lists sections that only apply on a full restart: host, port,
allowed_hosts, allowed_origins, vault, audit, log, tls, otel, users, email,
extraction, and overrides changes are kept as they were and named in the
"require restart" line.

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

## `[detection.ner] entities: "PER" is emitted as PER now and as PERSON from 2.0.0; write "PERSON" to switch now, or set [detection.ner.labels] PER = "PER" to keep PER`

A deprecation WARNING logged at startup (also by `serve --check`) and shown
as a `doctor` WARN row, one per affected entity; the quoted entity and types
are yours. The entity names a model's own label rather than a placeholder
type, and llm-redact 2.0.0 will fold such labels into the shared type
(`PER` → `PERSON`, `"phone number"` → `PHONE`, `EMAIL_ADDRESS` → `EMAIL`), so
the same value gets the same token whichever backend finds it. Nothing changes
until you upgrade to 2.0.0. Settle it now in one of two ways:

- write the type the message names (`entities = ["PERSON"]`): detections
  switch to that type now (new placeholders such as «PERSON_001»; metrics and
  alerts keyed on the old type move with them);
- or keep today's type for good with the override the message names
  (`[detection.ner.labels] PER = "PER"`).

Either way the warning goes away. See docs/detection.md "Placeholder types
from NER models".

## `[detection.ner] entities: "…" can never match: no active backend emits it (…)`

A WARNING logged at startup (also by `serve --check`) and after a reload that
rebuilds the detectors, once per entity; the parentheses name each active
backend and its model. Every loaded model was asked which labels it has
(Hugging Face models through their `id2label` table, spaCy pipelines through
their `ner` component, Presidio through the entities its analyzer supports),
and none of them is ever emitted as this entity's type. GLiNER (zero-shot) and
Stanza publish no label set, so with either of them active this never fires.
Common causes:

- a typo (`PERSONS`), or a label this model does not use — check the model
  card (`PER` vs `PERSON` no longer matters: write `PERSON`, the type);
- a type the model was not trained for (`ADDRESS` with a CoNLL-2003 model);
- while raw entities keep their own type (until 2.0.0), a raw entry such as
  `PER` claims the model's `PER` label, which then no longer serves a
  `PERSON` entry beside it — list one of the two;
- a `[detection.ner.labels]` override that maps the type to `""`, or a type
  that cannot be a placeholder (it starts with a digit or is longer than 28
  characters).

Nothing breaks: the entity simply never redacts anything. Fix the entry or
pick a model that covers it. While it stands, `llm-redact status` repeats it
as the posture line `NER entities no backend can emit: … (never detected)`,
and `/status` lists it in `detection.ner.unmatched_entities`.

## "NER skipped … string(s) longer than max_chars (…) — regex rules still applied"

A `llm-redact status` posture line: since the detectors were built, the named
NER backends were handed strings longer than `[detection.ner] max_chars` and
never read them (`llm_redact_ner_strings_total{outcome="skipped_max_chars"}`
counts them, `/status` `detection.ner` per backend). The regex rules, deny
strings and custom rules still scanned those strings; only names and other
contextual values the models would have found in them were not looked for.
`max_chars` is a latency cap: the time a model takes grows with the length of
what it reads. If your traffic carries long strings that
hold names (pasted documents, large tool results), raise `max_chars`; if the
skipped strings are logs or files you do not need NER on, the line is the
expected cost of the cap.

## "[detection.ner.labels] LABEL: the type must match [A-Z][A-Z0-9_]* and be at most 20 characters, or be "" to drop the label"

A `[detection.ner.labels]` value is not a placeholder type. Write the type in
uppercase letters, digits and `_`, starting with a letter and at most 20
characters (`CITY = "ADDRESS"`), or `""` to drop the label. Related
messages from the same table: "a label needs at least one letter or digit"
(a key such as `"--"`), "… name the same label (X) with different types" (two
spellings of one label, such as `"first name"` and `FIRST_NAME`, mapped to
different types: keep one), and "must be a table of LABEL = TYPE".

## "[detection.ner.revisions] BACKEND must be a full 40-character lowercase hex commit id; branch and tag names (such as main) are refused because they move"

A `[detection.ner.revisions]` value is not a commit id. A pin must name one
commit for good, so write the full 40-character lowercase hex id from the
model's page on the Hugging Face Hub ("Files and versions" → the commit, or
`https://huggingface.co/api/models/ORG/MODEL/revision/main` → `sha`), not
`main`, a tag or a shortened id. Related messages from the same table: "…
BACKEND: a revision pins a Hugging Face Hub model, so only the gliner and hf
backends take one" (an entry for `spacy`, `presidio` or `stanza`, whose
models are not Hub snapshots: remove it) and "must be a table of BACKEND =
"<40-character commit id>"". The message names the backend, never the value.

## "[detection.ner] entities […] match no entity the Presidio analyzer supports for language '…'"

From `serve` / `serve --check` with the `presidio` backend: none of the
configured `entities` is something Presidio can find in that language, so
every request would fail. The message lists what the analyzer supports. Name
a placeholder type Presidio covers (`PERSON`, `EMAIL`, `PHONE`, `SSN`, `IBAN`,
`CREDIT_CARD`) or one of the listed Presidio entities (`LOCATION`,
`IP_ADDRESS`, …); `PER` is a spaCy/Hugging Face label Presidio does not use —
write `PERSON`.

## `backend = "hf" but torch is not installed; install the hf extra`

From `serve` / `serve --check` with `[detection.ner]` using the `hf` backend:
the `transformers` package is installed but `torch` is not, so no model can
run. Install the extra, which brings both: `uv sync --extra hf` (or
`pip install 'llm-redact-proxy[hf]'`), then re-run `serve --check`.

## "[detection.ner] hf model '…' has no fast tokenizer; character offsets are required"

Startup (and `serve --check`) refused an `hf` backend model whose tokenizer
is a slow (pure-Python) one. Only a fast tokenizer reports where each entity
sits in the text — without those offsets an entity could never be redacted,
and the pipeline could not read a long string in windows. Most Hub models
ship a fast tokenizer (`tokenizer.json`), and transformers builds one for
many others (from a WordPiece `vocab.txt`, for one). If the model has none,
pick another model, or save a converted fast tokenizer beside the model once
and point `[detection.ner.models] hf` at that local folder.

## "failed to load Hugging Face token-classification model '…': …"

From `serve` / `serve --check`: torch is installed and the model's files are
in place, but loading the named `hf` model failed; the message ends with the
exception type. Common causes: a model that is not a token-classification
model, an architecture the installed transformers does not know, damaged
files in the cache, or a full disk. Load the folder once by hand to see the
library's full error: `uv run python -c "from transformers import pipeline;
pipeline('token-classification', model='/path/to/the/model/folder')"` (the
folder: the Hugging Face cache's `models--ORG--MODEL/snapshots/<revision>`).

## "[detection.ner] hf model '…' has no safetensors weights; set allow_pickle_weights = true to load pytorch_model.bin"

The `hf` model ships its weights only as `pytorch_model.bin`, a Python
pickle — and loading a pickle can run code — so llm-redact loads only
safetensors weights by default. Pick a model that ships `model.safetensors`
(most do), convert the checkpoint once into a local folder and point
`[detection.ner.models] hf` at it, or, if you trust the model's publisher,
set `[detection.ner] allow_pickle_weights = true`. A model that ships both
always loads its safetensors.

## "[detection.ner] … model '…' at revision … is not (completely) in the local Hugging Face cache, and downloads are off …"

The named model (or a GLiNER model's base model) is not in the local Hugging
Face cache at the revision llm-redact loads, and `[detection.ner]
allow_download` is `false` (the default), so nothing is fetched. A copy of
another revision does not count: the pin is the commit the message names
(`[detection.ner.revisions]`, else the model catalog's pin; "no revision
pinned" means the newest cached revision of the default branch). Fetch the
model once with `llm-redact models pull`, or set `allow_download = true` to
let a startup fetch the pinned files (with a `HF_HOME` the proxy can write).
A reload (SIGHUP, the dashboard editor) never downloads, whatever
`allow_download` says: after a reload naming a new model, the running
configuration is kept — fetch the model, then reload again, or restart.
"(completely)": the cache holds the revision but not every file the loader
needs, such as after an interrupted download.

## "[detection.ner] … model '…' … could not be fetched from the Hugging Face Hub: …"

`allow_download = true`, but fetching the model's files failed; the message
ends with the exception type. Common causes: no network access to
huggingface.co, a model id or revision that does not exist, a gated model
without a token, a full disk or a cache the proxy cannot write.

## "[detection.ner] … model '…' is a local directory; a revision in [detection.ner.revisions] applies only to a Hugging Face model id"

`[detection.ner.models]` (or `model`) names a local folder, and
`[detection.ner.revisions]` pins that backend too. A folder is whatever it
holds, so a commit id cannot apply to it: remove the backend's revision.

## "[detection.ner] … model '…' is neither a local directory nor a Hugging Face model id"

The model value is not a folder that exists and not of the form `ORG/NAME`.
Check the path (it is read relative to the proxy's working directory unless
absolute) or the model id.

## "[detection.ner] … model '…' needs code from its repository (… names auto_map); llm-redact never runs model code"

The model's `config.json`, `tokenizer_config.json` or `gliner_config.json`
names Python classes to import from the model repository (`auto_map`).
llm-redact never loads with `trust_remote_code`, so such a model cannot be
used; pick one built on an architecture transformers ships.

## "[detection.ner] … model '…': config.json is not a JSON configuration" / "… cannot be read (…)"

A configuration file in the model's folder is not a UTF-8 JSON object (or is
larger than 4 MiB), or cannot be opened. Re-fetch the model
(`llm-redact models pull`) or fix the local folder.

## "failed to load GLiNER model '…': …"

From `serve` / `serve --check`: the model's files are in place, but GLiNER
could not load them; the message ends with the exception type. Common
causes: damaged files in the cache (fetch them again: `llm-redact models
pull`), a checkpoint the installed gliner version cannot read, or too little
memory.

## "[detection.ner] gliner model '…' has no gliner_config.json" / "… names no base model (model_name) for its tokenizer and encoder configuration"

The folder or repository is not a GLiNER checkpoint (`gliner_config.json`
is missing), or its configuration ships no tokenizer and no `encoder_config`
and does not name the base model to take them from. Check the model id or
folder.

## "[detection.ner] gliner … '…': … names a model type transformers does not know; llm-redact never runs model code"

The GLiNER checkpoint's `encoder_config` (or its base model's `config.json`)
names an architecture the installed transformers does not ship. GLiNER
would build that encoder with `trust_remote_code`, which could run code from
the model repository, so it is refused. Upgrade transformers if the type is
newer than your version (`uv sync --extra gliner`), or pick another model.

## "[detection.ner] gliner model '…' … has no ONNX file '…' ([detection.ner.onnx] gliner)" / "[detection.ner.onnx] gliner needs onnxruntime, which the gliner extra installs"

`[detection.ner.onnx] gliner` names a file the model does not ship (check the
model's "Files and versions" page: Knowledgator's GLiNER-PII models ship
`onnx/model.onnx`, `onnx/model_quint8.onnx` and, except `-large`,
`onnx/model_fp16.onnx`), or onnxruntime is missing — reinstall the extra:
`uv sync --extra gliner`. Related: "[detection.ner.onnx] BACKEND: only the
gliner backend loads ONNX weights" and "… must be a .onnx file inside the
model" (no wildcard, no `..`, no absolute path).

## "[detection.ner] gliner base model '…' has no config.json"

The base model a GLiNER checkpoint names lacks the configuration llm-redact
embeds into the assembled folder. Check the `model_name` in the checkpoint's
`gliner_config.json`.

## "[detection.ner] gliner model '…' ships no tokenizer or encoder_config, and the model catalog pins no revision of its base model '…': the newest cached revision of its default branch loads"

A startup warning for a GLiNER checkpoint llm-redact's model catalog does
not know (or knows with another base model): its base model's tokenizer and
configuration load at whatever revision of its default branch the cache
holds. Prefer a catalogued model or a self-contained checkpoint (one that
ships its tokenizer and an `encoder_config`), or a folder written by
`llm-redact models pull --to`.

## "[detection.ner] gliner model '…': cannot assemble its local folder under … (…)"

llm-redact could not write the self-contained GLiNER folder it builds from
the checkpoint and its base model (under `$XDG_DATA_HOME/llm-redact/models/`).
Make that directory writable for the proxy's user (the systemd unit and the
Helm chart already allow the data directory), or free disk space.

## "[detection.ner] … model '…' needs huggingface_hub, which the hf and gliner extras install; install the backend's extra" / "backend = \"gliner\" needs transformers, which the gliner extra installs"

The `huggingface_hub` package is missing, although the `hf` and `gliner`
extras install it (through transformers and gliner). Re-install the
backend's extra: `uv sync --extra hf` or `uv sync --extra gliner`.

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

## NER bench: "the NER bench needs [detection.ner] enabled = true in the config it scores"

`python -m llm_redact.bench.ner --config PATH` measures NER models, so the
config it scores must enable them (`[detection.ner] enabled = true`, with
the backends and entities to measure). The regex rules alone are measured by
`python -m llm_redact.bench`. See [ner-bench.md](ner-bench.md).

## NER bench: "no thresholds for [CONFIG.DATASET] in bench/ner_thresholds.toml; record a baseline from this run's report"

`--check` found no recorded floors and ceilings for this configuration and
dataset; a run with nothing recorded fails rather than passing unmeasured.
Record a baseline as [ner-bench.md](ner-bench.md#recording-a-baseline)
describes, or drop `--check` to only print the report.

## NER bench: "recall floor for TYPE cannot be checked: the run holds no gold spans of TYPE"

The thresholds entry sets a floor for a type the scored samples never
contain (a `--limit` too small, or a dataset without that type). Raise
`--limit`, or remove the floor from that dataset's entry.

## NER bench: "--dump-errors must name a file outside any git work tree"

`--dump-errors` writes dataset text, so it refuses a path inside a git
repository, where the file could be committed. Write it under `/tmp` or your
home directory, and delete it when done.

## NER bench: "dataset 'NAME' holds real data; --dump-errors would write its text to disk: add --allow-real-data-dump to confirm"

The dataset holds real text (real prompts, code or documents). Its errors
are written only when you confirm with `--allow-real-data-dump`; the file is
private (mode 0600) and belongs outside every repository.

## NER bench: "no NER ceilings for [CONFIG] in bench/ner_ceilings.toml; record a baseline from this run's report"

`--fp-corpus … --check` found no section for this configuration in the
ceilings file; a run with nothing recorded fails. Run it without `--check`,
read the per-file counts, and record a section as
[ner-bench.md](ner-bench.md#false-positives-on-agent-traffic---fp-corpus)
describes.

## NER bench: "FILE: TYPE found N, ceiling M (lines …)" / "bench/ner_ceilings.toml [CONFIG] names FILE, which is not in the corpus"

The model adds more detections of that type to that negatives file than
its recorded ceiling allows (a file or type with no ceiling allows none):
read the lines named, decide whether the hits are legitimate (raise the
ceiling, with the reason in the commit) or a regression (fix the
configuration). The second message names a ceiling for a file that no
longer exists: remove the stale entry.

## NER bench: "downloading a dataset needs huggingface_hub; install the bench-data extra: uv sync --extra bench-data" / "reading a parquet dataset needs pyarrow; …"

Published datasets (`openpii`, `nemotron`, …) are fetched and read with the
`bench-data` extra's packages, which a default install does not carry. Run
`uv sync --extra bench-data` (or `pip install 'llm-redact-proxy[bench-data]'`)
and retry. The generated datasets (`synthetic`, `rules`) need neither.

## NER bench: "could not download FILE of REPO at revision REVISION: ERRORTYPE"

The bench asked the Hugging Face Hub for one file of a dataset at its pinned
revision and the download failed (no network, a proxy refusing the host, a
full disk). Files already in the cache directory are reused, so a machine
without network access can run a dataset whose cache was filled elsewhere:
copy `${XDG_CACHE_HOME:-~/.cache}/llm-redact/bench-datasets` across, or
point `--cache-dir` at the copy.

## NER bench: "--cache-dir must be outside any git work tree"

Downloaded datasets are never committed, so the bench refuses a cache inside
a git repository. Use the default cache or a directory outside the
repository.

## NER bench: "--language: dataset 'NAME' has no language to filter on"

`--language` keeps the rows of one language and works only for datasets that
record one (`--list-datasets` marks them). Drop it for the others.

## NER bench: "privy: cannot read privy-dataset.zip: …" / "privy: a data file …" / "PUPA: cannot read FILE: …"

A downloaded dataset file is not what its pinned revision holds: a truncated
download, a damaged cache entry or a file edited by hand. Delete the
dataset's entry under the cache directory
(`${XDG_CACHE_HOME:-~/.cache}/llm-redact/bench-datasets`, or your
`--cache-dir`) and run again to download it afresh.

## NER bench: "dataset 'creddata' reads a local checkout: pass --data-dir" / "CredData: --data-dir must name a CredData checkout with a meta/ directory"

CredData is not downloaded by the bench: clone it, run its own
`download_data.py`, and pass the checkout with `--data-dir`
([ner-bench.md](ner-bench.md#creddata)). `--data-dir` must point at the
CredData directory itself (the one holding `meta/` and, after the
download, `data/`). Rows whose files the download did not produce are
skipped and counted.

## NER bench: "--data-dir applies only to datasets read from a local checkout"

Only `creddata` reads a local checkout; the other datasets are generated or
downloaded. Drop `--data-dir`, or add `--dataset creddata`.

## NER bench: "CredData: cannot read meta/FILE.csv: …"

A CredData metadata file lacks the columns the adapter reads (`FilePath`,
`LineStart`, `LineEnd`, `GroundTruth`, `ValueStart`, `ValueEnd`) or is not
valid UTF-8 CSV — a checkout of a CredData version whose format changed, or
a damaged file. Check out the commit named in
[ner-bench.md](ner-bench.md#creddata).

## NER bench: "dataset 'agent-eval' reads a local file: pass --path" / "agent-eval: --path must name the frozen set …"

The agent-traffic evaluation set is private and never downloaded: the bench
reads it from the file `scripts/pii_corpus/review.py freeze` wrote
([ner-bench.md](ner-bench.md#the-agent-traffic-evaluation-set)). Pass that
file with `--path` (for example
`--path ~/.local/share/llm-redact/pii-corpus/agent-eval.jsonl`); a path
that does not exist or is a directory is refused the same way.

## NER bench: "--path applies only to datasets read from a local file"

Only `agent-eval` reads a local file; the other datasets are generated,
downloaded or read from a checkout (`--data-dir`). Drop `--path`, or add
`--dataset agent-eval`.

## NER bench: "agent-eval: cannot read FILE.manifest.json …" / "agent-eval: FILE.manifest.json is not a llm-redact-agent-eval/1 manifest"

A frozen set is scored only beside the manifest `review.py freeze` wrote
for it (`FILE.manifest.json`, format `llm-redact-agent-eval/1`). The file
named by `--path` is not a frozen set — a `generate.py` output or a
`review.py` verified file, which have no such manifest — or its manifest was
moved, renamed or replaced. Point `--path` at the frozen file, or freeze the
verified rows again (`review.py freeze VERIFIED.jsonl --out FROZEN.jsonl`).

## NER bench: "agent-eval: FILE does not match the SHA-256 in FILE.manifest.json; a frozen set changes only by freezing it again"

The frozen file changed after it was frozen (an edit by hand, a line-ending
conversion, a partial copy), so scores would no longer describe the set its
manifest records. Restore the frozen file from where you keep it, or freeze
the verified rows again with `review.py freeze … --force`, which writes a
new file and manifest together.

## NER bench: "--fp-corpus and --latency are separate runs; pick one" / "--dump-errors applies to dataset and --fp-corpus runs"

`--latency` times NER; it scores nothing and dumps nothing. Run it on its
own, and run `--fp-corpus` or a dataset (with `--dump-errors` if wanted)
separately.

## NER bench: "p50_ms at N characters: X ms is above the ceiling Y ms"

The full pipeline's median per-string time crossed the ceiling recorded in
`[CONFIG.latency]` of the thresholds file. Latency depends on the CPU the
report names: compare like with like before treating it as a regression,
and raise the ceiling (with the CPU in its `note`) when the machine changed.
