# Provider setup and coverage

How to point each supported provider surface through the proxy, and
what is covered on each. The endpoint-by-endpoint table, including
documented gaps, is [api-coverage.md](api-coverage.md); every setting
below lives in [`config.example.toml`](../config.example.toml).

At a glance:

| Provider | Point at the proxy | Covered surface |
|---|---|---|
| Anthropic | `ANTHROPIC_BASE_URL` | Messages (+streaming), count_tokens, Message Batches, beta Files |
| OpenAI | `OPENAI_BASE_URL` | Chat Completions, Responses, Conversations, legacy completions, embeddings, Files+Batches, Realtime WS |
| Azure OpenAI | `[providers.azure]` + tool's Azure endpoint | same OpenAI surface incl. Responses/Conversations/Realtime, legacy completions, image/speech prompts, files/batches, model listings |
| Google Gemini | `GOOGLE_GEMINI_BASE_URL` | generateContent/stream, countTokens, embeddings, cachedContents, batch, Live WS |
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
# OpenAI-compatible tools (chat completions and /v1/responses, e.g. Codex CLI):
OPENAI_BASE_URL=http://127.0.0.1:8787 <your-tool>
# Gemini (generateContent / streamGenerateContent / countTokens):
GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8787 <your-tool>
# Ollama's native API (/api/chat, /api/generate, /api/embed):
OLLAMA_HOST=http://127.0.0.1:8787 <your-tool>
```

`llm-redact run -- <tool>` injects the right variable(s) for you. An
agent with the llm-redact plugin installed can confirm its traffic is
actually flowing through the proxy with `/llm-redact:status` and
`/llm-redact:recent` ([plugins.md](plugins.md)).

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
are recognized. A signature the CLIENT computed (SigV4-signed SDK traffic)
remains a permanent non-goal: it covers the payload hash of the
unredacted body, so no body-rewriting proxy can transit it (see
[threat-model.md](threat-model.md)). The proxy can instead sign each
request itself, after redaction, with a cloud identity it holds (below).

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
proxy actually redacted: a non-empty request body on a recognized route
must be a JSON object (read from the bytes whatever the content-type; a
UTF-8 BOM or UTF-16/32 encoding is fine) or canonical multipart on a
route whose multipart form llm-redact scans (Azure Files uploads and
image edits). Anything else — non-JSON bytes, invalid UTF-8, a top-level
JSON array or scalar (`null` included), a whitespace-only body,
multipart on any other route or outside the canonical form, and any
`Content-Encoding` other than `identity` (the proxy never decompresses
a request, so it cannot see what the upstream would) — is refused with a
recorded, provider-shaped 400 naming the body's kind, before any
credential is fetched or the upstream contacted. An empty body (a GET,
DELETE or body-less POST) is forwarded as before; `detection = false`
stays the explicit unredacted opt-out, and key-authorized providers keep
forwarding such bodies verbatim. Inside an accepted multipart upload every
piece must be scanned too, or the whole request is refused the same way:
each non-blank line of an uploaded file must be a JSON object (so a text,
PDF or other non-JSONL file cannot be uploaded with the proxy's identity —
use key auth for those), plain form fields (`purpose`, `user`, `size`, …)
are scanned as UTF-8 text (a field that is not UTF-8 is refused), and
bytes outside every part (a multipart preamble or epilogue) are refused.
So is a part header without a single reading — a folded or repeated
header line, a filename holding a backslash that is not a `\"` or `\\`
escape, a malformed `filename*` or one in a charset other than UTF-8 —
and a part the proxy could not read as its plain bytes: a
Content-Transfer-Encoding other than `7bit`/`8bit`/`binary` on any part
(RFC 7578 deprecates them), or, on a part whose content is scanned, a
declared charset other than UTF-8/US-ASCII (its Content-Type `charset`,
or the RFC 7578 `_charset_` field). The image and mask parts of an
image edit are media — the documented non-goal, as base64 media in a
JSON body — and are signed as sent (their filenames redacted).
Key-authorized uploads are unchanged: unscanned pieces forward verbatim,
plain form fields are not scanned, a filename without a single reading
is left as sent, and declared encodings are the encoding non-goal (the
proxy scans the bytes it receives). Realtime
WebSocket connections are authorized the same way — Azure OpenAI
Realtime and the Vertex AI Live API (below): the upgrade request is
authorized as the HTTP GET it is and the upstream is dialled with
exactly the proxy's headers. Only those documented realtime paths are
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

## Ollama's native API

Supported out of the box (`OLLAMA_HOST=http://127.0.0.1:8787`, or point
the tool at the proxy): `/api/chat` and `/api/generate` are redacted and
rehydrated including their newline-delimited-JSON streaming, and
`/api/embed`/`/api/embeddings` inputs are scrubbed. The default
upstream is the local daemon at `http://127.0.0.1:11434`.

## Local and custom OpenAI-compatible servers

Other local OpenAI-compatible servers (vLLM, LM Studio — and Ollama's
own `/v1` endpoints) are covered by pointing
`[providers.openai] upstream_base_url` at them, so even self-hosted
model traffic can be redacted — or run **several side by side** as
named custom upstreams (`[providers.custom.NAME]`, served under
`/custom/NAME/` with the full OpenAI surface, including Responses).

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
than `max_body_bytes` are rejected 413 fail-closed — raise the cap for
large batch files (`llm-redact doctor` reminds you).

MCP connector configuration (Anthropic `mcp_servers`, OpenAI
`tools type=mcp`) passes through unredacted by design — the provider
must hold the real credential to call your MCP server — while MCP call
arguments and output are redacted and restored like any other content.

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
Detection can also be scoped by language
(`[detection] languages = ["en"]` skips other countries' national-id
rules; universal rules always run) and per MCP server
(`[detection.mcp] exempt_servers` exempts a trusted server's MCP
content blocks). Every such opt-out is surfaced in `/status`, `doctor`,
and `llm-redact status` — never silent.
