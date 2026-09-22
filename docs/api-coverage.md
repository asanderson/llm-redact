# API coverage matrix

Every documented Anthropic and OpenAI endpoint, with the classification
the proxy applies. Enumerated against docs.anthropic.com and
platform.openai.com as of **2026-07**; `tests/test_api_coverage.py` pins
each row's routing (and checks this table and the test table against each
other in both directions), so an endpoint silently drifting to
pass-through fails CI. Live drift tests additionally assert observed
stream event names ⊆ the adapters' known sets.

Classifications:

- **chat** — request redacted AND response rehydrated (streaming included)
- **redact-only** — request redacted; response has nothing to restore
- **pass-through** — deliberately forwarded verbatim (metadata/ids only,
  or a documented non-goal); the disabled-provider 502 still applies
- **websocket** — relayed by `realtime.py` (see the realtime sections of
  the README and threat model)

## Anthropic

| Endpoint | Classification | Notes |
|---|---|---|
| `POST /v1/messages` | chat | MCP connector: `mcp_servers[]` blocks pass through unredacted BY DESIGN — the provider must hold the real `authorization_token` to call the MCP server; everything else in the body is redacted |
| `POST /v1/messages/count_tokens` | redact-only | note counted too, keeping counts honest |
| `POST /v1/messages/batches` | redact-only | each `requests[].params` redacted + noted |
| `GET /v1/messages/batches` | pass-through | processing metadata only |
| `GET /v1/messages/batches/{id}` | pass-through | processing metadata only |
| `GET /v1/messages/batches/{id}/results` | chat | JSONL restored line by line |
| `POST /v1/messages/batches/{id}/cancel` | pass-through | no content either way |
| `DELETE /v1/messages/batches/{id}` | pass-through | no content either way |
| `GET /v1/models` | pass-through | model listings carry no user content |
| `GET /v1/models/{id}` | pass-through | |
| `POST /v1/complete` | chat | legacy Text Completions: prompt redacted, completion restored (streaming included); no system note (the body has no system field) |
| `GET /v1/organizations/...` (Admin API) | pass-through | org metadata |
| WebSocket realtime | websocket | not offered by Anthropic today |

Anthropic's beta Files API shares its paths (`/v1/files...`) with
OpenAI's. Routing is header-aware here: requests carrying an
`anthropic-version` header pass through to the ANTHROPIC upstream
(their uploads are documents — the media non-goal — so pass-through is
the correct handling, but they must reach the right host); everything
else takes the OpenAI files handling below.

## OpenAI

| Endpoint | Classification | Notes |
|---|---|---|
| `POST /v1/chat/completions` | chat | streaming `delta.content`, tool-call arguments, and reasoning-model chain-of-thought (`delta.reasoning_content` / `delta.reasoning`) are all rehydrated per choice |
| `GET /v1/chat/completions/{id}` | chat | stored-completion retrieval restored |
| `POST /v1/responses` | chat | MCP connector: `tools[].type == "mcp"` entries (server_url, headers) pass through unredacted BY DESIGN — the provider needs the real credential; `mcp_call` arguments/output in responses are rehydrated, streaming included |
| `GET /v1/responses/{id}` | chat | stored responses rehydrated |
| `GET /v1/responses/{id}/input_items` | chat | input-item echoes restored |
| `DELETE /v1/responses/{id}` | pass-through | |
| `POST /v1/conversations` | chat | create: item content redacted, echoed response restored |
| `POST /v1/conversations/{id}/items` | chat | add items: content redacted + echo restored |
| `GET /v1/conversations/{id}` | chat | retrieve conversation, restored |
| `GET /v1/conversations/{id}/items` | chat | list items, stored content restored (list-envelope walk) |
| `GET /v1/conversations/{id}/items/{item_id}` | chat | single item restored |
| `DELETE /v1/conversations/{id}` (and `/items/{item_id}`) | pass-through | ids only |
| `POST /v1/embeddings` | redact-only | vectors come back verbatim |
| `POST /v1/files` | redact-only | multipart upload; JSONL file-part lines (batch + fine-tune) redacted, all other bytes preserved |
| `GET /v1/files` | pass-through | metadata only |
| `GET /v1/files/{id}` | pass-through | metadata only |
| `GET /v1/files/{id}/content` | chat | batch output JSONL restored line by line |
| `DELETE /v1/files/{id}` | pass-through | |
| `POST /v1/batches` | pass-through | file ids + metadata only |
| `GET /v1/batches` | pass-through | |
| `GET /v1/batches/{id}` | pass-through | |
| `POST /v1/batches/{id}/cancel` | pass-through | |
| `GET /v1/models` | pass-through | |
| `POST /v1/completions` | chat | legacy text completions: prompt redacted, choices[].text restored (streaming included); no system note |
| `POST /v1/moderations` | pass-through | DOCUMENTED GAP: moderation input is user text; redacting it would change moderation results, so it is deliberately untouched |
| `POST /v1/audio/transcriptions` | pass-through | audio media non-goal (multipart audio is never decoded) |
| `POST /v1/audio/translations` | pass-through | audio media non-goal |
| `POST /v1/audio/speech` | redact-only | the text-to-speech `input` is user text and is redacted; the audio response is bytes forwarded verbatim |
| `POST /v1/images/generations` | redact-only | the OUTPUT is media, but the `prompt` is plain text and is redacted; the response (`b64_json`/`url`) comes back verbatim — a dall-e-3 `revised_prompt` echo may carry placeholder tokens (fail-safe: the value it hides was never exposed) |
| `POST /v1/images/edits` | redact-only | multipart: the `prompt` form FIELD is redacted; image/mask file parts are media and stay byte-identical |
| `POST /v1/images/variations` | pass-through | image in, images out — no text anywhere in the request |
| `POST /v1/videos` | chat | Sora job create: the `prompt` (JSON or multipart form field) is redacted, and the returned job object's prompt ECHO is restored; multipart `input_reference` media stays byte-identical |
| `GET /v1/videos` | chat | job list: echoed prompts restored via the list-envelope walk |
| `GET /v1/videos/{id}` | chat | job retrieve: echoed prompt restored |
| `POST /v1/videos/{id}/remix` | chat | remix prompt redacted; echo restored |
| `GET /v1/videos/{id}/content` | pass-through | the rendered video: media bytes verbatim |
| `DELETE /v1/videos/{id}` | pass-through | id only |
| `POST /v1/fine_tuning/jobs` | pass-through | file ids only; the training FILE is covered at upload via `/v1/files` |
| `GET /v1/fine_tuning/jobs` | pass-through | |
| WebSocket `/v1/realtime` | websocket | beta + GA event vocabularies; MCP tool config preserved, MCP arguments rehydrated |

## MCP (Model Context Protocol)

MCP itself is a local protocol between the agentic tool and its MCP
servers — that traffic never transits this proxy. What does transit is
the providers' **MCP connector** surfaces, covered above: connector
CONFIGURATION (`mcp_servers[]`, `tools[].type == "mcp"`) passes through
unredacted by design (the provider must receive the real credential to
call the MCP server on the model's behalf — it is addressed to the
provider, not conversation content), while MCP call CONTENT — arguments
the model writes and output the server returns — is redacted outbound
and rehydrated inbound like any other content, on Messages, Responses
(streaming included), and Realtime.

## Other providers

Gemini **context caching** (`POST /v1beta/cachedContents`) and **Batch
Mode** (`models/{m}:batchGenerateContent`) are redact-only: the cached
prompt and the inlined batch requests are content that must not reach the
provider in the clear, while their responses carry only a cache/operation
name (nothing to rehydrate). Both are stored/async — the cache is reused
and batch results are fetched later through the operations API with no
first-message anchor — so they use the STATIC vault session (the batch
stance), keeping redact/rehydrate always in agreement. The per-cache
GET/PATCH/DELETE and list return metadata only and pass through.

Gemini **Imagen** (`models/{m}:predict`) and **Veo**
(`models/{m}:predictLongRunning`) are redact-only: `instances[].prompt`
is user text and is redacted, while the responses carry image bytes or a
long-running operation name — nothing to rehydrate. The same two verbs
are covered on Vertex paths, where the matcher stays provably disjoint
from Claude-on-Vertex's `rawPredict`/`streamRawPredict` (the colon
anchors the verb).

**Cohere** (`[providers.cohere]`, default `https://api.cohere.com`) covers
`POST /v2/chat` (CHAT — messages redacted, the response text / `tool_plan` /
tool-call `arguments` rehydrated non-streaming and per-channel on the SSE
stream), `POST /v2/embed` and `POST /v2/rerank` (redact-only — the input
content is scanned, the vector/rank responses have nothing to restore), and
the deprecated `POST /v1/chat` and `POST /v1/generate` (CHAT; the v1
`text-generation` stream has its own channel). Streaming shapes are pinned by
fixtures + a live drift test; an unrecognized event forwards verbatim.

Gemini, Vertex, Azure OpenAI, Bedrock, Cohere, and Ollama route coverage is
pinned by their adapter test suites (`tests/test_provider_*.py`); their
matched routes appear in the README's provider section. **Claude models
on Vertex** are covered separately from Gemini-on-Vertex: their
`publishers/anthropic/models/{m}:rawPredict` / `:streamRawPredict` paths
carry Anthropic Messages bodies (`anthropic_version: vertex-2023-10-16`,
no `model` field), so `ClaudeVertexAdapter` reuses the Anthropic
redaction/rehydration and routes to the same `[providers.vertex]`
upstream; its matcher is proven disjoint from the Gemini Vertex adapter's
(`rawPredict` vs `generateContent` verbs), and other publishers' rawPredict
traffic (Llama, etc.) is deliberately not matched. Azure files
uploads (`POST /openai/files`) and content downloads reuse the OpenAI
multipart/JSONL handling on Azure's path shapes; `/openai/batches` and
file metadata pass through. **Azure OpenAI Responses**
(`POST /openai/responses` and the `/openai/v1/responses` preview, plus the
stored-response and input-item GETs) reuses `OpenAIResponsesAdapter`
wholesale via `AzureResponsesAdapter` — identical event vocabulary, delta
channels, and note injection; only routing differs (matcher disjoint from
the Azure chat adapter's, proven by test). **Azure Realtime**
(`/openai/realtime`) likewise reuses the OpenAI Realtime WS adapter via
`AzureRealtimeWs`; both route to the customer's `[providers.azure]`
resource URL. Named custom providers
(`[providers.custom.NAME]`, served under `/custom/NAME/`) expose the
full OpenAI surface above per upstream. Their inner path is normalized
before matching (`_canonical`): OpenAI-compatible upstreams serve those
same endpoints under varied base paths — Groq `/openai/v1`, OpenRouter
`/api/v1`, Fireworks `/inference/v1` — and some tools bake `/v1` into
`upstream_base_url` so the inner path omits it. The path is re-anchored at
the last `/v1/` (or `/v1` is prepended) so a known endpoint always routes;
an unknown tail still falls through to pass-through via the exact matcher.

## Known uncovered content surfaces (honest gaps)

These carry user content but are **not** redacted today — a request to one
is forwarded verbatim, the same honesty posture as warn mode and
per-provider `detection = false`. Documented so nobody assumes protection
that is not there:

- **OpenAI Uploads API** (`POST /v1/uploads`, `/parts`, `/complete`) — the
  large-file sibling of `/v1/files`. Each part is an opaque byte range and a
  secret can straddle a part boundary, so per-line scanning cannot be applied
  safely; real coverage would need stateful cross-part buffering. Pass-through,
  routed to the OpenAI upstream (pinned by test — it previously
  fell through to the anthropic default).
- **OpenAI Assistants / Threads / vector-store search** — on OpenAI's
  announced deprecation path (Responses/Conversations is the successor), so
  not built.
- **OpenAI WebRTC realtime** (`POST /v1/realtime/calls`, SDP offer/answer) —
  after setup, media and the event data channel flow peer-to-peer and never
  transit this HTTP/WS proxy at all: structurally unreachable, not merely
  unimplemented. The WebSocket realtime transport IS covered.

Closing any of these is additive future work; each reuses the existing
redaction/rehydration machinery except where noted.
