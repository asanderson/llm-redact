# Provider setup and coverage

How to point each supported provider surface through the proxy, and
what is covered on each. The endpoint-by-endpoint table, including
documented gaps, is [api-coverage.md](api-coverage.md); every setting
below lives in [`config.example.toml`](../config.example.toml).

At a glance:

| Provider | Point at the proxy | Covered surface |
|---|---|---|
| Anthropic | `ANTHROPIC_BASE_URL` | Messages (+streaming), count_tokens, Message Batches, beta Files |
| OpenAI | `OPENAI_BASE_URL` (ending in `/v1`) | Chat Completions, Responses, Conversations, legacy completions, embeddings, Files+Batches, Realtime WS |
| Azure OpenAI | `[providers.azure]` + tool's Azure endpoint | same OpenAI surface incl. Responses/Conversations/Realtime, legacy completions, image/speech prompts, files/batches, model listings |
| Google Gemini | `GOOGLE_GEMINI_BASE_URL` | generateContent/stream, countTokens, embeddings, cachedContents, batch, Live WS, the OpenAI-compatible `/v1beta/openai/` surface |
| Vertex AI | `[providers.vertex]` | Gemini-on-Vertex + Claude-on-Vertex (`rawPredict`/`streamRawPredict`), context caching, computeTokens/embedContent, Imagen/Veo, model listings |
| AWS Bedrock | `[providers.bedrock]` (bearer keys, or the proxy's own identity) | converse(+stream), invoke(+response-stream), binary eventstream, count-tokens, ApplyGuardrail, async invoke |
| Cohere | `[providers.cohere]` | v2 chat (+streaming), embed, rerank, legacy v1 chat/generate |
| Ollama (native) | `OLLAMA_HOST` | /api/chat, /api/generate (+NDJSON streaming), /api/embed |
| Any OpenAI-compatible | `[providers.custom.NAME]` → `/custom/NAME/` | full OpenAI surface per named upstream, several side by side |

One upstream per provider is the default model. Several upstreams per
protocol — a subscription lane, your own API keys, a local Ollama —
selected by rules with fallback chains, cooldowns and monthly budgets
is the routing layer of the llm-redact-pro package (the llm-redact-pro
routing guide).

## Anthropic, OpenAI, Gemini, Ollama (env-var providers)

These need no configuration — point the tool's base-URL variable at the
proxy and the default upstreams apply:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude -p "hello"
# OpenAI-compatible tools (chat completions and /v1/responses — its compaction
# and input-token count too — e.g. Codex CLI, OpenCode, the OpenAI SDKs); the
# /v1 is part of the base URL:
OPENAI_BASE_URL=http://127.0.0.1:8787/v1 <your-tool>
# Gemini (generateContent / streamGenerateContent / countTokens):
GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8787 <your-tool>
# Ollama's native API (/api/chat, /api/generate, /api/embed):
OLLAMA_HOST=http://127.0.0.1:8787 <your-tool>
```

`llm-redact run -- <tool>` injects the right variable(s) for you. An
agent with the llm-redact plugin installed can confirm its traffic is
actually flowing through the proxy with `/llm-redact:status` and
`/llm-redact:recent` ([plugins.md](plugins.md)).

The OpenAI SDKs (and Codex and OpenCode) append endpoint paths such as
`/responses` to a base URL that already includes `/v1` — their default is
`https://api.openai.com/v1`. With the old `OPENAI_BASE_URL=http://127.0.0.1:8787`
the proxy receives `/responses`, which is no provider's path: it answers a
404 naming the fix and forwards nothing. (Releases up to 1.8.0 sent such a
request to the anthropic upstream with the OpenAI key and the prompt,
unredacted.)

The proxy forwards a request no route matches only to a provider it can
positively attribute it to — a path family, or a header only that
provider's clients send (`anthropic-version`, a Google API key, …) —
never to a guessed one; anything else is a recorded local 404. A path
spelled differently from the API's own route (an empty `//` segment, a
trailing `/`, another case, or a spelling a normalizing front end reads as
the route: `\` for `/`, `;params`, trailing spaces or dots, a second
encoding) is refused 400, a recognized route under an extra prefix
(`/v1/v1/messages`: a base URL repeating the version) is a 404, and
`GET`/`HEAD /` is answered locally. See
[api-coverage.md](api-coverage.md#requests-no-route-matches). Point each
`upstream_base_url` at the API's final `https` URL: the proxy never relays
an upstream redirect that a following client would answer by re-sending
a redacted request's unredacted original to the `Location` — it answers
502.

## Azure OpenAI

Set `[providers.azure] upstream_base_url` to your resource URL and point
the tool's Azure endpoint at the proxy. The full OpenAI surface is
covered on Azure paths too (both the `/openai/deployments/{d}/…`
api-version form and the `/openai/v1/…` API) — Chat Completions, legacy
completions, Responses, Conversations, embeddings, image-generation/edit
prompts, text-to-speech input, files/batches, and Realtime; the model,
deployment and file listings are recognized as well. The tool's `api-key` (or Entra ID
bearer token) is forwarded as is, unless the proxy authorizes with its
own identity (see [the proxy's own cloud identity](#the-proxys-own-cloud-identity)).

## Vertex AI (Gemini and Claude models)

Set `[providers.vertex] upstream_base_url` to your regional
`https://{region}-aiplatform.googleapis.com` host and Vertex
`generateContent`/`streamGenerateContent`/`countTokens` traffic (Bearer
auth, so body rewriting is safe) is redacted like Gemini. **Claude
models on Vertex** are covered too: their
`publishers/anthropic/models/{m}:rawPredict` / `:streamRawPredict` paths
carry Anthropic Messages bodies, so they reuse the Anthropic
redaction/rehydration and the same `[providers.vertex]` upstream (other
publishers' `rawPredict` traffic is deliberately left untouched).
Context caches (`projects/{p}/locations/{l}/cachedContents`: the create is
redacted, get/list/patch/delete recognized), `:computeTokens`,
`:embedContent`, Imagen/Veo (`:predict`, `:predictLongRunning`,
`:fetchPredictOperation`) and the publisher/Model Registry model listings
are covered too, and the Vertex AI Live API WebSocket is relayed like
Gemini Live ([realtime](#realtime-websocket-apis)). The tool's bearer token
is forwarded as is, unless the proxy authorizes with its own identity
(below).

## AWS Bedrock

Bedrock's bearer-token API keys are supported: set
`[providers.bedrock] upstream_base_url` to your
`https://bedrock-runtime.{region}.amazonaws.com` host and the four
runtime routes (`converse`, `converse-stream`, `invoke`,
`invoke-with-response-stream`) are redacted, including AWS's binary
eventstream response framing, which the proxy parses and re-frames
natively. `count-tokens` (including the base64 `input.invokeModel.body`
form, decoded and redacted like an invoke body), ApplyGuardrail
(`/guardrail/{id}/version/{v}/apply` — content redacted, the guardrail's
rewritten output restored; note the guardrail itself therefore judges the
redacted text, placeholders and all) and
StartAsyncInvoke (`POST /async-invoke`; its output lands in S3 with the
placeholders in place) are redacted too, and the async-invoke status reads
are recognized. Base64 media blocks (`image`/`document`/`video`
`source.bytes`) are forwarded byte-identical on every route — never
scanned, like base64 `data` elsewhere (the media non-goal); the text
beside them is redacted. A signature the CLIENT computed (SigV4-signed SDK traffic)
remains a permanent non-goal: it covers the payload hash of the
unredacted body, so no body-rewriting proxy can transit it (see
[threat-model.md](threat-model.md)). The proxy can instead sign each
request itself, after redaction, with a cloud identity it holds (below).

## Request bodies llm-redact cannot read

A recognized route (the chat and redact-only rows of
[api-coverage.md](api-coverage.md)) forwards only a request body the
proxy scanned — wherever redaction applies (`detection` on, which is the
default, with the tool's own key) and wherever the request is sent with a
credential the PROXY holds (its own cloud identity, below, or an operator
key a routing rule spends), whatever `detection` says. A non-empty body
must be a JSON object (read from the bytes whatever the content-type; a
UTF-8 BOM or UTF-16/32 encoding is fine) or canonical multipart on a
route whose multipart form llm-redact scans (Files uploads, image edits,
video jobs). Anything else would be forwarded verbatim, unread — and
common upstreams decode it anyway: Go's JSON decoder behind Ollama and
Jackson take the first JSON value and ignore what follows, Express's
body-parser substitutes invalid bytes and inflates gzip. So non-JSON
bytes, invalid UTF-8 (Windows PowerShell 5.1 sends a string `-Body`
without a charset as ISO-8859-1), bytes after the JSON value (a trailing
NUL, a second value), a top-level JSON array or scalar (`null` included),
JSON nesting deeper than 128 levels of objects and arrays (no walk could
read it), a whitespace-only body, multipart on any other route or outside the
canonical form, and a repeated `Content-Type` header (a singleton field;
a second one could name a multipart boundary the proxy never parsed with)
are refused with a recorded, provider-shaped **400** naming the body's
kind, before any upstream contact. A `Content-Encoding` coding other than
`identity`, in any of the request's Content-Encoding headers, is refused
**415** with `Accept-Encoding: identity`: llm-redact does not decode
request bodies — send it uncompressed. An empty body (a GET, DELETE or
body-less POST) is forwarded as before, and every SDK sends UTF-8 JSON
objects, so only clients that were already sending malformed bodies see
the refusal instead of a silent leak.

Inside an accepted multipart upload every piece must be scanned too, or
the whole request is refused the same way. An uploaded file is read by
its CONTENT (the declared type and file name are only the client's
guess): a JSONL file (every non-blank line a JSON object — a batch input
or fine-tuning file) is redacted line by line as JSON — every value of
every line, whatever its key (a data file's `id`, `name`, `type` or
`data` is content), except in what a provider runs as a request: the
`body` of a line in an OpenAI Files upload of purpose `batch`, and the
conversation of a `fine-tune` example, keep a request's protocol fields
(`role`, `model`, a tool call's `id` and `name`) as sent; any other text
file — strict UTF-8 (a byte-order mark kept), or UTF-16/UTF-32 opened by
its byte-order mark — is redacted as one text and re-encoded exactly as it
came (a CSV, a log, notes; a file mixing JSON lines with other lines is
text), except that a text file which is ONE JSON document (a pretty-printed
service-account key) is redacted escape-aware: each string literal, keys
included, is decoded, redacted and written back only when it changed, and
what a raw reading of the result still finds inside a literal holding no
escape (a `private_key_id` known by its key) is redacted in place — the
vault holds each value as it decodes; a document where a raw reading finds
anything else (a card number written as a JSON number, a value spanning
literals, one only the escaped source spells) is redacted as one raw text
instead; a BINARY file — bytes that do not decode, text holding a NUL, or a
known binary signature even when the bytes would decode (an all-ASCII PDF:
rewriting it would break its byte offsets) — cannot be redacted at all. A
binary file part is the one piece that may leave unscanned, and only with
the tool's own key: `[detection] binary_uploads = "forward"` (the default)
forwards it byte-identical — its file NAME still redacted — counts it
(`/status` `unscanned_uploads_total`, `llm_redact_unscanned_uploads_total`,
an INFO log line with the path and count, the `llm-redact status` posture
block) and says so in `doctor`; `"refuse"` answers 400 instead. **What is
inside a forwarded PDF, image or archive reaches the provider as-is** —
exactly like base64 media in a chat body. Under a credential the proxy
holds (its cloud identity, or a routing rule's operator key) a binary file
is always refused: the proxy vouches only for what it read. A JSONL line
nesting JSON deeper than 128 levels is refused (a JSONL reader would
decode what no walk can read). Plain form fields (`purpose`, `user`,
`size`, …) are scanned as UTF-8 text (a field that is not UTF-8 is
refused), and bytes outside every part (a multipart preamble or epilogue)
are refused. So is
a part header without a single reading — a folded or repeated header
line, a filename holding a backslash that is not a `\"` or `\\` escape, a
malformed `filename*` or one in a charset other than UTF-8 — and a part
the proxy could not read as its plain bytes: a Content-Transfer-Encoding
other than `7bit`/`8bit`/`binary` on any part (RFC 7578 deprecates them),
or, on a part whose content is scanned, a declared charset other than
the one its content decoded as — UTF-8/US-ASCII, or a UTF-16/32 text
file's own (its Content-Type `charset`, or the RFC 7578 `_charset_`
field). `GET /v1/files/{id}/content` reads a download the same way: a
text file this proxy uploaded redacted as one raw text that a download
would read as JSON (JSON Lines only once redacted, or a JSON document the
escape-aware reading gave way on) is remembered by a digest of the bytes
sent — once the whole upload was redacted (a refused request records
nothing), the newest 1024, in the running process — and restored as the
text it was (a value lands exactly as it was redacted, never
JSON-escaped); a JSONL file is
restored line by line as JSON, every value of every line; any other text
file that is ONE JSON document (a JSON file a model or code wrote around a
placeholder) is restored over its source text with each restored value
JSON-escaped, keys included and formatting kept, so it stays valid JSON;
any other text file as one text; each re-encoded as it came; a binary file
is left untouched. A download is read this way whatever Content-Type the
provider serves it with (a JSON file served as `application/json`
included). A JSON document uploaded escape-aware needs no remembering: its
JSON-escaped restoration is the exact inverse in any process — another
replica over a shared vault, another `llm-redact run` proxy, a restart.
One residual, for the remembered kind only: the record is per process and
shared by every caller, so a download through another process, or after
1024 newer such uploads (any caller's), reads the file like one a model
wrote — valid JSON, but a value whose source form held an escape (`\\`,
`\n`) comes back escaped once more. The image and mask parts of an image edit (and a video job's
reference image) are media — the documented non-goal, as base64 media in
a JSON body — and are sent as they came (their filenames redacted).

`[providers.NAME] detection = false` with the tool's own key is the
explicit opt-out: nothing is scanned and such bodies — any upload
included — are forwarded verbatim, surfaced like every other opt-out
(`/status` `providers_detection_off`, `llm-redact status`, `doctor`).
Traffic on a route llm-redact does not recognize is forwarded verbatim
as before (pass-through).

## The proxy's own cloud identity

With the llm-redact-pro package installed, Bedrock, Vertex AI and Azure
OpenAI can be authorized by the PROXY instead of the tool:

```toml
[providers.bedrock]
upstream_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
auth = "identity"          # default "passthrough": forward the tool's credential
# region = "us-east-1"     # bedrock only; when the host names no region
```

With `auth = "identity"` the proxy removes every credential the tool
sent (`Authorization`, `x-api-key`, `api-key`, `x-goog-api-key`, any
other `*api-key` or `*authorization*` header, `x-amz-*` signing
headers, cookies, `password`/`passwd`, Google's `x-goog-user-project`,
`x-goog-quota-user` and IAM selector headers; the `key=` / `$key=` /
`api-key=` / `access_token=` / `oauth_token=` / `userProject=` /
`quotaUser=` / `subscription-key=` / `password=` / `passwd=` /
`*authorization*` / `X-Amz-*` query parameters, compared
case-insensitively with a leading `$` ignored; and — on realtime
WebSocket upgrades — every subprotocol except the known non-credential
`realtime` and `openai-beta.*` offers, so `openai-insecure-api-key.<key>`
and a secret offered as the entry after a bare `bearer` marker never
reach the provider), then authorizes the final, redacted request with its
own workload identity: AWS SigV4 for Bedrock, a Google OAuth token for
Vertex AI (Gemini and Claude models alike), a Microsoft Entra ID token
for Azure OpenAI. If no credential can be obtained, the proxy answers a
502 and forwards nothing. The setting is valid only for these three
providers; without llm-redact-pro it is a startup error, and so is an
`http://` `upstream_base_url` on a non-loopback host (every request
carries the proxy's credential, so the upstream must be https — realtime
dials the wss twin). The signed URL must be exactly the configured
upstream — scheme, host, port, and the `upstream_base_url` PATH (an
API-management base such as `https://gw.example/my-api` is honored on
HTTP and realtime alike) followed by the request's own path — or the
request is refused before signing. The identity signs only a body the
proxy actually redacted — the scanned-body rule every recognized route
follows (above, "Request bodies llm-redact cannot read"), which under the
proxy's own identity holds whatever `detection` says: `detection = false`
turns redaction off, not this rule (the ownership check of
llm-redact-pro's named users reads the parsed body too, so a gzip or
non-JSON body it could not read is refused either way). With `detection`
on, an upload holding a binary file (a PDF, an image) cannot be sent
with the proxy's identity; a text file is redacted and sent. Realtime
WebSocket connections are authorized the same way — Azure OpenAI
Realtime and the Vertex AI Live API (below): the upgrade request is
authorized as the HTTP GET it is and the upstream is dialled with
exactly the proxy's headers — and never redirected: a 3xx answer to the
upgrade is a failed dial (1011, counted and recorded), so the proxy's
credential cannot follow a `Location` to another host or path (nor, on
key-authorized connections, the client's own key). Only those documented realtime paths are
authorized (any other WebSocket path to such a provider is refused
1011 and recorded as a 403), a missing credential closes the connection 1011 naming the
credential source, a client frame that is not JSON (text or binary —
Gemini Live's JSON-in-binary frames stay allowed) closes it 1008 unsent
and records the connection as a 400, and such a provider is never
routed. Only the HTTP
routes llm-redact recognizes (the Vertex AI, Azure OpenAI and Bedrock
tables in [api-coverage.md](api-coverage.md)) are forwarded with that
identity; any other path to the provider is refused with a recorded 403,
never signed. Credential sources and the IAM
permissions to grant are in llm-redact-pro's provider-identity guide.

Any client that can reach the proxy can then spend that identity: keep
the proxy on 127.0.0.1, or require a client identity with
llm-redact-pro's access gate (`[auth] require = true`). `llm-redact
doctor` warns about a non-loopback bind without one, and `llm-redact
status` lists the providers the proxy holds credentials for.

A web page in your browser is not such a client (unless you list its
origin in `allowed_origins`, which lends it the identity too): a request
carrying browser markers (`Origin`, `Sec-Fetch-*`) from another origin or site —
a cross-site "simple" POST, a WebSocket from any page — is refused with a
recorded 403 (WebSocket: close 1008) before any credential is fetched,
and so is any request to an identity-authorized provider that names a
host the proxy does not answer to (DNS rebinding) when it arrives over
plain HTTP. The proxy answers to 127.0.0.1, localhost, ::1 and its bind
host; a tool that reaches a plain-HTTP proxy under another name (a
compose service, a Kubernetes Service) needs that name in
`allowed_hosts` ([deployment.md](deployment.md#host-names-the-proxy-answers-to-allowed_hosts)).
An access gate with ambient credentials (client certificates, Basic auth,
an access proxy's cookie) does not change this: the browser attaches
those to a page's requests by itself. The same browser rule holds on
every other route too — see the threat model's "Requests from web pages".

## Ollama's native API

Supported out of the box (`OLLAMA_HOST=http://127.0.0.1:8787`, or point
the tool at the proxy): `/api/chat` and `/api/generate` are redacted and
rehydrated including their newline-delimited-JSON streaming, and
`/api/embed`/`/api/embeddings` inputs are scrubbed. The model inventory
(`/api/tags`, `/api/ps`, `/api/show`) and `/api/version` are recognized,
and the ollama CLI's `HEAD /` heartbeat is answered by the proxy itself.
The default upstream is the local daemon at `http://127.0.0.1:11434`.

Ollama (like a local vLLM or LM Studio server) needs no key, so the proxy
lends whoever reaches it access to the model — and it rewrites `Host` when
forwarding, which defeats Ollama's own DNS-rebinding check. The proxy
therefore refuses a web page's request itself (a foreign `Origin`, a
cross-site `Sec-Fetch-Site`, or a browser request to a host name the proxy
does not answer to) on these routes as on every other; see the threat
model's "Requests from web pages". Tools keep working under any name. A
browser app you list in `allowed_origins` is served, and Ollama's own
policy then applies to it: allow the origin in `OLLAMA_ORIGINS` too.

## Local and custom OpenAI-compatible servers

Other local OpenAI-compatible servers (vLLM, LM Studio — and Ollama's
own `/v1` endpoints) are covered by pointing
`[providers.openai] upstream_base_url` at them, so even self-hosted
model traffic can be redacted — or run **several side by side** as
named custom upstreams (`[providers.custom.NAME]`, served under
`/custom/NAME/` with the full OpenAI surface, including Responses). A
custom upstream's endpoints are recognized under whatever base path it
serves them — `/v1`, `/openai/v1`, `/api/v1`, `/inference`, `/models`,
`/api/paas/v4`, a Cloudflare AI Gateway's `/v1/{account}/{gateway}/openai`
— and the path after `/custom/NAME` is forwarded byte-for-byte: point the
tool at `http://127.0.0.1:8787/custom/NAME` plus the base path the
upstream's own docs give. Gemini's OpenAI-compatible surface needs no
custom provider: `http://127.0.0.1:8787/v1beta/openai/` reaches
`[providers.gemini]`, redacted like OpenAI's.

## Embeddings

Embeddings endpoints (`/v1/embeddings`, Azure embeddings, Gemini
`embedContent`) are redacted too — the input is scrubbed and the vector
response passes through untouched.

## Batch APIs

Anthropic Message Batches (creation redacted per entry, the JSONL
results stream restored line by line) and OpenAI Files + Batches (the
uploaded JSONL file part — batch inputs and fine-tuning examples — is
redacted line by line with every other byte of the multipart body
preserved; batch output downloads are restored the same way) are
covered. An upload's file NAME is content too (`jane.doe@corp.example
notes.jsonl`): every part's Content-Disposition `filename` and RFC 8187
`filename*` (UTF-8) is redacted on every multipart route — only those
value bytes change, the part `name` and every other header stay as
sent — and the file object the provider echoes it in (the upload
response, `GET /v1/files`, `GET /v1/files/{id}`, and Azure's
`/openai/files` twins) comes back with the name restored. A filename
with no single reading (a bare backslash, a folded header) is left as
sent with key auth and refused with identity auth. Batch flows use the static vault session (with llm-redact-pro's
named users, the submitting user's own copy of it), and uploads larger
than `max_body_bytes` — or carrying more lines, strings or parts than
`max_body_strings` — are rejected 413 fail-closed: raise the caps for
large batch files (`llm-redact doctor` reminds you).

MCP connector configuration (Anthropic `mcp_servers`, OpenAI
`tools type=mcp`) passes through unredacted by design — the provider
must hold the real credential to call your MCP server — while MCP call
arguments and output are redacted and restored like any other content.

## Stored-object APIs: fine-tuning, vector stores, containers

OpenAI's fine-tuning jobs (`/v1/fine_tuning/jobs`: create, the list, one
job, its cancel/pause/resume, events and checkpoints — on Azure's
`/openai/v1` and api-version `/openai` families and custom providers too)
are recognized, so a credential the proxy holds (a routed operator key,
Azure's identity auth) may reach them. The caller's free-form `metadata`
is redacted and restored in every echo of the job, and the provider's
event messages are restored; the training data itself is the file,
redacted at its upload. Some fields the provider keeps exactly as sent:
the `suffix` becomes part of the fine-tuned model's name — the name
every later request cites in its `model`, which is never rewritten — and
the file ids and the Weights & Biases `integrations` name things that
exist elsewhere. Those VERBATIM fields are scanned but never rewritten:
a value llm-redact would redact there refuses the request with a 400
naming the field (a placeholder would name a model, file or project that
does not exist). Choose a suffix that holds nothing private. The created
job, and the files a finished job wrote (`result_files`), are reported
to a session router that tracks stored objects (llm-redact-pro's named
users). Checkpoint permissions (an admin key sharing a checkpoint across
projects) stay pass-through.

Vector stores (`/v1/vector_stores`: the store, its search, its files and
file batches — OpenAI, both Azure families, custom providers) are
recognized the same way. A store's `description` and `metadata`, a file's
`attributes` (a map keyed by the caller, walked like `metadata`: a key
named `name` or `id` is data), the store's `name` (a label: the store is
addressed by its id), a search's `query` and its attribute-filter values
are redacted; every answer echoing them, and the stored files'
content a search or a file's content read returns, is restored. Vector
store traffic uses the static vault session (with llm-redact-pro's named
users, the user's own copy of it) — the session the files were uploaded
and the attributes redacted in — so a filter value's placeholder is the
stored attribute's and the filter still matches (the vault is
deterministic). The file ids a store, attach or file batch names, and a
filter's attribute `key`, are verbatim, as above. The
created store is reported to a session router, and the store list is a
listing it attributes per item; a store's own files are read as the
store's.

Code interpreter containers (`/v1/containers`: the container and its
files — OpenAI, Azure's v1 API, custom providers) are recognized too. A
container file upload is redacted exactly as a `/v1/files` upload is (the
file part read by its content, its filename and every form field; a
binary file goes out unscanned only with the client's own key), a JSON
container-file create names a stored file (verbatim), the container file
object's `path` (the filename) is restored, and a download is restored
like a Files API download — a text file line by line, a binary file
untouched. A container's `name` is redacted and restored like a store's;
its starting `file_ids` are verbatim. Created containers — and the container a
Response's code interpreter call ran in (or a file it cites was written
in), unless the request named it — and container files are reported to a
session router.

The Uploads API (`/v1/uploads`: a large file sent in parts) stays
unrecognized — a part is an opaque byte range whose boundaries can split
a line or a value, and the total size is declared before the first part —
so a credential the proxy holds is never lent to it; upload through
`/v1/files` instead (docs/api-coverage.md, honest gaps).

## Realtime WebSocket APIs

With `pip install 'llm-redact-proxy[realtime]'`: OpenAI Realtime
(`/v1/realtime`), Azure OpenAI Realtime (`/openai/realtime` preview and
`/openai/v1/realtime` GA, to `[providers.azure]`), Gemini Live
(`BidiGenerateContent`) and the Vertex AI Live API
(`/ws/google.cloud.aiplatform.{v1,v1beta1}.LlmBidiService/BidiGenerateContent`,
to `[providers.vertex]`) connections are
relayed over wss with text events redacted outbound and restored
inbound — tokens split across streaming frames reassemble exactly, and
base64 audio passes through untouched (audio is not scanned, the same
stance as images). Without the extra, WebSocket upgrades are refused
outright, so nothing silently bypasses redaction. Realtime connections
use the static vault session — the per-conversation mode's
first-message anchor does not exist at connection time. With
llm-redact-pro's named users, each user's connection uses that user's own
copy of the static session ([per-user namespaces](how-it-works.md#session-isolation)).
A connection keeps a running token floor: a token any of its client
frames carried (a restored conversation, a pasted answer) is never issued
to a new value later on that connection ([the vault records](how-it-works.md#the-vault-records)).
A config reload (SIGHUP, the config editor) reaches open connections too.
A connection whose provider settings, upstream authorizer or `[detection]`
policy the reload changed is closed on both sides with code 1012
(reconnect), and no frame it reads after the reload is forwarded under the
old configuration. A client that reconnects gets the new configuration and
the same vault session ([details](deployment.md#reloads-and-open-realtime-connections)).
The Azure and Vertex routes work with the proxy's own cloud identity
([above](#the-proxys-own-cloud-identity)); the full list of accepted
WebSocket paths is in [api-coverage.md](api-coverage.md#realtime-websocket-routes).

## Routing, fallback and budgets (llm-redact-pro)

`[upstreams.NAME]` + `[routing]` (off by default; the llm-redact-pro
routing guide) replaces the one-upstream-per-provider model for the
four env-var protocols — `anthropic`, `openai`, `gemini`, and native
`ollama`: named destinations with a credential mode (`passthrough`
byte-exact, `env:VAR` injected by the proxy, `none`), first-match rules
on protocol / model glob / headers / path / auth kind, `on_status`
fallback chains with cooldowns and Anthropic plan-limit detection, and
per-upstream monthly budgets. When a `[routing]` table is present the
legacy `[providers.anthropic|openai|gemini|ollama]` sections
auto-register as passthrough upstreams of the same name, so existing
configs keep working and rules can reference them. The core parses and
validates these sections; `[routing] enabled = true` refuses to start
without the llm-redact-pro package. Azure, Vertex, Bedrock, Cohere,
custom providers and the realtime relay keep the `upstream_base_url`
path described above and are never routed.

## Disabling providers, and the deliberate opt-outs

Any provider can be disabled (`[providers.NAME] enabled = false`); its
routes then answer 502 rather than ever passing traffic through
unredacted. Each provider also has a deliberate detection off-switch
(`detection = false`): its requests are forwarded **unredacted** —
nothing is protected, like warn mode — while rehydration stays active;
use it only for upstreams you own end to end, such as a local Ollama.
It turns off redaction only: a body with a repeated JSON key is still
forwarded as the proxy parsed it (its last occurrence — what the session
router's ownership check and stored-object tracking read), and an
identity-authorized provider still refuses a body the proxy cannot read.
Detection can also be scoped by language
(`[detection] languages = ["en"]` skips other countries' national-id
rules; universal rules always run) and per MCP server
(`[detection.mcp] exempt_servers` exempts a trusted server's MCP
content blocks). Every such opt-out is surfaced in `/status`, `doctor`,
and `llm-redact status` — never silent.
