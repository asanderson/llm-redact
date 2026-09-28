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
| Azure OpenAI | `[providers.azure]` + tool's Azure endpoint | same OpenAI surface incl. Responses/Realtime, files/batches |
| Google Gemini | `GOOGLE_GEMINI_BASE_URL` | generateContent/stream, countTokens, embeddings, cachedContents, batch, Live WS |
| Vertex AI | `[providers.vertex]` | Gemini-on-Vertex + Claude-on-Vertex (`rawPredict`/`streamRawPredict`) |
| AWS Bedrock | `[providers.bedrock]` (bearer keys, or the proxy's own identity) | converse(+stream), invoke(+response-stream), binary eventstream |
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
covered on Azure paths too — Chat Completions, Responses, embeddings,
files/batches, and Realtime. The tool's `api-key` (or Entra ID
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
publishers' `rawPredict` traffic is deliberately left untouched), and
the Vertex AI Live API WebSocket is relayed like Gemini Live
([realtime](#realtime-websocket-apis)). The
tool's bearer token is forwarded as is, unless the proxy authorizes with
its own identity (below).

## AWS Bedrock

Bedrock's bearer-token API keys are supported: set
`[providers.bedrock] upstream_base_url` to your
`https://bedrock-runtime.{region}.amazonaws.com` host and the four
runtime routes (`converse`, `converse-stream`, `invoke`,
`invoke-with-response-stream`) are redacted, including AWS's binary
eventstream response framing, which the proxy parses and re-frames
natively. A signature the CLIENT computed (SigV4-signed SDK traffic)
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
headers, cookies, `key=` / `api-key=` / `access_token=` /
`*authorization*` / `X-Amz-*` query parameters, and — on realtime
WebSocket upgrades — credential-bearing subprotocols such as
`openai-insecure-api-key.<key>`), then authorizes the final, redacted request with its
own workload identity: AWS SigV4 for Bedrock, a Google OAuth token for
Vertex AI (Gemini and Claude models alike), a Microsoft Entra ID token
for Azure OpenAI. If no credential can be obtained, the proxy answers a
502 and forwards nothing. The setting is valid only for these three
providers; without llm-redact-pro it is a startup error. Realtime
WebSocket connections are authorized the same way — Azure OpenAI
Realtime and the Vertex AI Live API (below): the upgrade request is
authorized as the HTTP GET it is and the upstream is dialled with
exactly the proxy's headers. Only those documented realtime paths are
authorized (any other WebSocket path to such a provider is refused
1011), a missing credential closes the connection 1011 naming the
credential source, and such a provider is never routed. Credential sources and the IAM
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
covered. Batch flows use the static vault session (with llm-redact-pro's
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
