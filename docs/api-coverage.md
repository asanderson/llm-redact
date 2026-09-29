# API coverage matrix

Every documented Anthropic and OpenAI endpoint, and the commonly used
Vertex AI, Azure OpenAI and Bedrock runtime endpoints, with the
classification the proxy applies. Enumerated against docs.anthropic.com and
platform.openai.com as of **2026-07**, and against the Vertex AI v1 REST
reference, the Azure OpenAI REST reference (2024-10-21 GA + the `OpenAI.v1`
spec in Azure/azure-rest-api-specs) and the Bedrock runtime API reference as
of **2026-09**; `tests/test_api_coverage.py` pins
each row's routing (and checks this table and the test table against each
other in both directions), so an endpoint silently drifting to
pass-through fails CI. Live drift tests additionally assert observed
stream event names ⊆ the adapters' known sets.

Classifications:

- **chat** — request redacted AND response rehydrated (streaming included)
- **redact-only** — request redacted; response has nothing to restore
- **pass-through** — deliberately forwarded verbatim (metadata/ids only,
  or a documented non-goal); the disabled-provider 502 still applies. On a
  provider configured `auth = "identity"` (Bedrock, Vertex AI, Azure — the
  proxy signs with its OWN cloud identity) a pass-through route is instead
  REFUSED with a recorded local 403: the proxy lends its identity only to
  the routes it recognizes
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

`GET /v1/models` stays pass-through in both provider tables (the
Anthropic row above, the OpenAI row below). With the llm-redact-pro
routing layer and `[routing] expose_models = true`, `GET /v1/models` is
answered **locally** instead (never forwarded) — Anthropic shape when
the request carries `anthropic-version`, OpenAI shape otherwise — so
Claude Code's gateway model discovery works; the adapter matrix row
stays pass-through. The local answer sits behind the same gates as every
proxy-generated reply for a real API path: the path infers provider
`openai`, so `[providers.openai] enabled = false` refuses it 502 (the
Anthropic-shaped call too), and an llm-redact-pro access gate refuses an
unadmitted client 403, before the catalog is consulted. The row
does not change: that answer is a routing feature, not a redaction
classification. Under `[routing]` the id-only rows here and below —
`GET /v1/responses/{id}`, conversation item reads,
`GET /v1/files/{id}/content`, batch polls and results — carry no model,
so they take the protocol's `default_upstream` unless a path-matched
rule names another; their classification is unchanged.

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

## Google Vertex AI

`{p}`/`{l}` are the project and location; every `/v1/` row also matches
`/v1beta1/`, and the `publishers/{pub}/models/{m}` rows also match with no
`projects/{p}/locations/{l}/` prefix (express mode) and on
`endpoints/{id}`. Metadata GETs are **redact-only** — a body-less request
makes that a no-op, and it makes the route RECOGNIZED, so identity auth
forwards it (a pass-through row would be refused 403 there).

| Endpoint | Classification | Notes |
|---|---|---|
| `POST /v1/projects/{p}/locations/{l}/publishers/google/models/{m}:generateContent` | chat | Gemini wire format, inherited from the Gemini adapter |
| `POST /v1/projects/{p}/locations/{l}/publishers/google/models/{m}:streamGenerateContent` | chat | SSE (`alt=sse`) and the buffered JSON-array form |
| `POST /v1/projects/{p}/locations/{l}/endpoints/{id}:generateContent` | chat | tuned/deployed endpoints |
| `POST /v1/publishers/google/models/{m}:generateContent` | chat | express mode (no project prefix) |
| `POST /v1/projects/{p}/locations/{l}/publishers/google/models/{m}:countTokens` | redact-only | note counted too |
| `POST /v1/projects/{p}/locations/{l}/publishers/google/models/{m}:computeTokens` | redact-only | answers token ids/bytes — nothing to restore |
| `POST /v1/projects/{p}/locations/{l}/publishers/google/models/{m}:embedContent` | redact-only | vectors come back verbatim |
| `POST /v1/projects/{p}/locations/{l}/publishers/google/models/{m}:predict` | redact-only | Imagen / text-embedding `instances[]` |
| `POST /v1/projects/{p}/locations/{l}/publishers/google/models/{m}:predictLongRunning` | redact-only | Veo: answers an operation name |
| `POST /v1/projects/{p}/locations/{l}/publishers/google/models/{m}:fetchPredictOperation` | redact-only | Veo poll: the body is an operation name |
| `POST /v1/projects/{p}/locations/{l}/publishers/anthropic/models/{m}:rawPredict` | chat | Claude on Vertex: Anthropic Messages bodies |
| `POST /v1/projects/{p}/locations/{l}/publishers/anthropic/models/{m}:streamRawPredict` | chat | Anthropic SSE |
| `POST /v1/projects/{p}/locations/{l}/publishers/meta/models/{m}:rawPredict` | pass-through | other publishers' rawPredict bodies are not Messages-shaped — deliberately unmatched |
| `POST /v1/projects/{p}/locations/{l}/cachedContents` | redact-only | context-cache create: `contents`/`systemInstruction`/`tools` redacted; STATIC vault session (the batch stance); the returned cache `name` is reported to the session router as `cachedContents/<id>` |
| `GET /v1/projects/{p}/locations/{l}/cachedContents` | redact-only | cache list: metadata (cached contents are input-only, never returned) |
| `GET /v1/projects/{p}/locations/{l}/cachedContents/{id}` | redact-only | metadata |
| `PATCH /v1/projects/{p}/locations/{l}/cachedContents/{id}` | redact-only | ttl / expireTime only |
| `DELETE /v1/projects/{p}/locations/{l}/cachedContents/{id}` | redact-only | no content either way |
| `GET /v1beta1/publishers/google/models` | redact-only | publisher model list (google-genai `models.list`) |
| `GET /v1beta1/projects/{p}/locations/{l}/publishers/google/models/{m}` | redact-only | publisher model metadata (google-genai `models.get`) |
| `GET /v1/projects/{p}/locations/{l}/models` | redact-only | Model Registry list (tuned models) |
| `GET /v1/projects/{p}/locations/{l}/models/{m}` | redact-only | Model Registry metadata |
| `POST /v1/projects/{p}/locations/{l}/batchPredictionJobs` | pass-through | Vertex batch prediction reads its input from GCS/BigQuery — out of scope (see Gemini Batch Mode below) |

## Azure OpenAI

Both path families: the api-version form (`/openai/deployments/{d}/…`,
`/openai/files`, `?api-version=` in the query) and the v1 API
(`/openai/v1/…`, the model in the body). Classifications mirror the OpenAI
table above, with one deliberate difference: the id/metadata routes OpenAI
leaves as pass-through (file list/metadata/delete, batches, model and
deployment listings, response/conversation delete) are RECOGNIZED on Azure,
so `[providers.azure] auth = "identity"` does not refuse them. On a
body-less request redact-only is a no-op; batch objects are **chat**
because they echo the user `metadata` a batch create carries.

| Endpoint | Classification | Notes |
|---|---|---|
| `POST /openai/deployments/{d}/chat/completions` | chat | inherited from the OpenAI chat adapter |
| `POST /openai/v1/chat/completions` | chat | |
| `POST /openai/deployments/{d}/completions` | chat | legacy completions: prompt redacted, `choices[].text` restored (streaming included); no system note |
| `POST /openai/v1/completions` | chat | |
| `POST /openai/deployments/{d}/embeddings` | redact-only | |
| `POST /openai/v1/embeddings` | redact-only | |
| `POST /openai/deployments/{d}/images/generations` | redact-only | the prompt is redacted; image output verbatim |
| `POST /openai/deployments/{d}/images/edits` | redact-only | multipart: the `prompt` form field is redacted, image parts byte-identical |
| `POST /openai/deployments/{d}/audio/speech` | redact-only | text-to-speech `input` redacted; audio bytes verbatim |
| `POST /openai/deployments/{d}/audio/transcriptions` | pass-through | audio media non-goal (identity auth refuses it) |
| `POST /openai/responses` | chat | Responses on Azure, inherited from the OpenAI Responses adapter |
| `POST /openai/v1/responses` | chat | |
| `GET /openai/responses/{id}` | chat | stored responses restored |
| `GET /openai/v1/responses/{id}/input_items` | chat | input-item echoes restored |
| `POST /openai/v1/responses/{id}/cancel` | chat | answers the Response object, restored |
| `DELETE /openai/responses/{id}` | redact-only | ids only |
| `POST /openai/v1/conversations` | chat | item content redacted, echo restored; STATIC vault session |
| `POST /openai/v1/conversations/{id}/items` | chat | |
| `GET /openai/v1/conversations/{id}` | chat | |
| `GET /openai/v1/conversations/{id}/items` | chat | list-envelope walk |
| `DELETE /openai/v1/conversations/{id}` | redact-only | ids only |
| `POST /openai/files` | redact-only | multipart JSONL upload, lines redacted (+ note on chat-shaped lines) |
| `POST /openai/v1/files` | redact-only | |
| `GET /openai/files` | redact-only | metadata only |
| `GET /openai/files/{id}` | redact-only | metadata only |
| `DELETE /openai/files/{id}` | redact-only | |
| `GET /openai/files/{id}/content` | chat | batch output JSONL restored line by line |
| `GET /openai/v1/files/{id}/content` | chat | |
| `POST /openai/batches` | chat | file ids + user `metadata` (redacted out, restored in the echo) |
| `GET /openai/batches` | redact-only | the batch LIST is never restored in the reader's vault namespace: it spans batches other users created, so that could hand one user another's value (pro named users). A session router that attributes listed items (`listing_item_session`, llm-redact-pro named users) restores each batch the READER created in the session it was created in; every other item's `metadata` keeps its placeholders |
| `GET /openai/v1/batches/{id}` | chat | |
| `POST /openai/batches/{id}/cancel` | chat | |
| `GET /openai/models` | redact-only | model listing |
| `GET /openai/v1/models` | redact-only | |
| `GET /openai/v1/models/{id}` | redact-only | |
| `GET /openai/deployments` | redact-only | deployment listing |
| `GET /openai/deployments/{d}` | redact-only | |
| `POST /openai/v1/fine_tuning/jobs` | pass-through | file ids only; the training FILE is covered at upload |

## AWS Bedrock (runtime)

`{m}` may be a percent-encoded ARN (matching runs on the decoded path; the
raw path is forwarded). A path with a `.` or `..` segment — in any
spelling, `%2E` included — is refused with a 400 before any upstream
contact, on every provider: matching and forwarding must address the
same resource, and servers resolve dot segments.

| Endpoint | Classification | Notes |
|---|---|---|
| `POST /model/{m}/invoke` | chat | model-native bodies; note only into recognized Claude/Converse shapes |
| `POST /model/{m}/invoke-with-response-stream` | chat | binary event stream; Claude `chunk` payloads rehydrated |
| `POST /model/{m}/converse` | chat | |
| `POST /model/{m}/converse-stream` | chat | binary event stream, per-block channels |
| `POST /model/{m}/count-tokens` | redact-only | `input.converse` content redacted; answers a count. The `input.invokeModel.body` form is base64 of the model's native JSON prompt — text, not media — so it is decoded, redacted exactly like an `/invoke` body, and re-encoded; a blob that is not base64 of UTF-8 JSON is refused with a 400 (never forwarded unredacted) |
| `POST /guardrail/{id}/version/{v}/apply` | chat | ApplyGuardrail: `content[]` redacted; `outputs[].text` (the submitted text as the guardrail rewrote it) and the assessments' quoted `match` values carry the placeholders sent up, so they are restored — the client gets back its OWN text. The guardrail therefore evaluates the REDACTED text: its verdict is on placeholders, so a guardrail policy keyed on the values llm-redact redacts (a sensitive-information filter matching emails, say) never sees them |
| `POST /async-invoke` | redact-only | StartAsyncInvoke: `modelInput` redacted; the output is written to S3 and never passes through the proxy, so it keeps its placeholders (the batch stance) |
| `GET /async-invoke` | redact-only | ListAsyncInvokes: metadata |
| `GET /async-invoke/{id}` | redact-only | GetAsyncInvoke: metadata |

## Realtime WebSocket routes

Every WebSocket path the realtime relay (`realtime.py`) accepts, pinned
both directions by `tests/test_api_coverage.py` like the tables above
(the OpenAI row sits in its table). Any other WebSocket path is refused
(accept-then-close 1011): unlike HTTP there is no default upstream to pass
an unknown path through to. The Azure and Vertex routes also work with the
proxy's own cloud identity (`[providers.azure|vertex] auth = "identity"`,
llm-redact-pro): only the exact paths below are authorized — a subpath is
refused (and recorded as a 403) — and every client credential channel
(upgrade headers, the same query parameters as HTTP — `key=`/`$key=`,
`api-key=`, `access_token=`, `userProject=`, `quotaUser=`,
`subscription-key=`, `Authorization=`, … — and every subprotocol except
the known non-credential `realtime` / `openai-beta.*` offers) is stripped
before the proxy's credential is added. The relay dials the configured
`upstream_base_url` INCLUDING its path (an API-management base such as
`https://gw.example/my-api` is honored), exactly like HTTP. Bedrock has no WebSocket
API, so no realtime route reaches `[providers.bedrock]`.

| Endpoint | Classification | Notes |
|---|---|---|
| WebSocket `/openai/realtime` | websocket | Azure OpenAI Realtime, preview form (`?api-version=…&deployment=…`): the OpenAI Realtime event tables, `[providers.azure]`; identity auth supported |
| WebSocket `/openai/v1/realtime` | websocket | Azure OpenAI Realtime, GA form (`?model=<deployment>`, no api-version): same handling; identity auth supported |
| WebSocket `/ws/google.ai.generativelanguage.{version}.GenerativeService.BidiGenerateContent` | websocket | Gemini Live (v1alpha/v1beta), JSON over text or binary frames, `[providers.gemini]` |
| WebSocket `/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent` | websocket | Vertex AI Live API: Gemini Live message handling, `[providers.vertex]` (the regional `https://{region}-aiplatform.googleapis.com` host); identity auth supported |
| WebSocket `/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent` | websocket | Vertex AI Live API, v1beta1 (the google-genai SDK's Vertex default): same handling; identity auth supported |

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
stance; with llm-redact-pro's named users, the user's own copy of it),
keeping redact/rehydrate always in agreement. The per-cache
GET/PATCH/DELETE and list return metadata only and pass through.

A pass-through request under `/v1/` that carries a **Google API key**
(`x-goog-api-key`, or a `key=`/`$key=` query parameter) is forwarded to
the Gemini upstream, not inferred as OpenAI: Gemini's v1 surface
(`GET /v1/models[/{m}]`, …) shares OpenAI's prefix, and the Google key must
never be sent to api.openai.com.

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

Vertex AI, Azure OpenAI, and Bedrock routes are pinned by the tables above
(and `tests/test_cloud_routes.py`, which also proves the matchers disjoint
and that identity auth signs every recognized route); Gemini, Cohere, and
Ollama route coverage is pinned by their adapter test suites
(`tests/test_provider_*.py`); their matched routes appear in the README's
provider section. **Claude models
on Vertex** are covered separately from Gemini-on-Vertex: their
`publishers/anthropic/models/{m}:rawPredict` / `:streamRawPredict` paths
carry Anthropic Messages bodies (`anthropic_version: vertex-2023-10-16`,
no `model` field), so `ClaudeVertexAdapter` reuses the Anthropic
redaction/rehydration and routes to the same `[providers.vertex]`
upstream; its matcher is proven disjoint from the Gemini Vertex adapter's
(`rawPredict` vs `generateContent` verbs), and other publishers' rawPredict
traffic (Llama, etc.) is deliberately not matched. Azure files
uploads (`POST /openai/files`, `/openai/v1/files`) and content downloads
reuse the OpenAI multipart/JSONL handling on Azure's path shapes; batches
and file metadata are recognized (see the Azure table). **Azure OpenAI Responses**
(`POST /openai/responses` and the `/openai/v1/responses` preview, plus the
stored-response and input-item GETs) reuses `OpenAIResponsesAdapter`
wholesale via `AzureResponsesAdapter` — identical event vocabulary, delta
channels, and note injection; only routing differs (matcher disjoint from
the Azure chat adapter's, proven by test). **Azure Realtime**
(`/openai/realtime` and the GA `/openai/v1/realtime`) likewise reuses the
OpenAI Realtime WS adapter via `AzureRealtimeWs`; both route to the
customer's `[providers.azure]` resource URL. The **Vertex AI Live API**
(`LlmBidiService/BidiGenerateContent`, v1 and v1beta1) reuses the Gemini
Live adapter via `VertexLiveWs` on `[providers.vertex]` (see the realtime
table above). Named custom providers
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
  not built. The same holds for their Azure v1 twins (`/openai/v1/threads`,
  `/openai/v1/vector_stores/{id}/search`), and Azure's `/openai/v1/evals`
  and `/openai/v1/containers` are not covered either — pass-through, and
  refused under `auth = "identity"`.
- **OpenAI WebRTC realtime** (`POST /v1/realtime/calls`, SDP offer/answer) —
  after setup, media and the event data channel flow peer-to-peer and never
  transit this HTTP/WS proxy at all: structurally unreachable, not merely
  unimplemented. The WebSocket realtime transport IS covered.

Closing any of these is additive future work; each reuses the existing
redaction/rehydration machinery except where noted.
