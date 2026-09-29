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
- **redact-only** — request redacted; response has nothing to restore. A
  body-less id or metadata route (a model listing, a batch poll, a delete)
  is redact-only too: a no-op that makes the route RECOGNIZED, so a
  credential the proxy holds may reach it (below)
- **pass-through** — deliberately forwarded verbatim (a documented
  non-goal, or content not covered yet — see the honest gaps below), to
  the provider the request is positively attributed to ([requests no route
  matches](#requests-no-route-matches)); the disabled-provider 502 still
  applies. A request that would reach its provider with a credential the
  PROXY holds is instead REFUSED with a recorded local 403 on a
  pass-through route: on a provider configured `auth = "identity"`
  (Bedrock, Vertex AI, Azure — the proxy signs with its OWN cloud
  identity), and on a routed request (the llm-redact-pro routing layer)
  whose plan sends an operator key or no key at all — the proxy lends such
  a credential only to the routes it recognizes, and the refusal comes
  before the body is read
- **websocket** — relayed by `realtime.py` (see the realtime sections of
  the README and threat model)

A chat or redact-only route forwards only a request body the proxy
scanned — wherever redaction applies (`detection` on, the client's own
key included) and wherever the request spends a credential the PROXY
holds (`auth = "identity"`, or a routed plan's operator key), whatever
`detection` says. A non-empty body that is not a JSON object (JSON is
read from the bytes whatever the content-type; a UTF-8 BOM or UTF-16/32
is fine) or canonical multipart on a route whose multipart form is
scanned — bytes after the JSON value, invalid UTF-8 (a Latin-1 body), a
top-level array or scalar, JSON nesting deeper than 128 levels, whitespace
only, multipart anywhere else — or
that is sent with a repeated `Content-Type`, is refused with a recorded,
provider-shaped 400 instead of being forwarded verbatim (a lenient
upstream would decode what the proxy never read: take the first JSON
value, substitute bad bytes). A `Content-Encoding` other than `identity`,
in any of the request's Content-Encoding headers, is refused **415** with
`Accept-Encoding: identity`: llm-redact does not decode request bodies —
send it uncompressed. Inside an accepted upload every piece must be
scanned too wherever redaction applies (the `/v1/files` rows below; with
`detection = false` an upload's file parts go out as sent, a proxy-held
credential included, once the stored-object check has read the upload). `[providers.NAME] detection =
false` with the client's own key forwards such a body as sent, and so does
every pass-through route (a route this table does not claim) — reached
only with the client's own credential, since a credential the proxy holds
is never lent to one (above).

## Anthropic

| Endpoint | Classification | Notes |
|---|---|---|
| `POST /v1/messages` | chat | MCP connector: `mcp_servers[]` blocks pass through unredacted BY DESIGN — the provider must hold the real `authorization_token` to call the MCP server; everything else in the body is redacted. The files a code execution run WROTE (`code_execution_output` / `bash_code_execution_output` entries of a code execution tool result, streaming included) are reported to a session router as the requester's |
| `POST /v1/messages/count_tokens` | redact-only | note counted too, keeping counts honest |
| `POST /v1/messages/batches` | redact-only | each `requests[].params` redacted + noted |
| `GET /v1/messages/batches` | redact-only | processing metadata only (a body-less no-op: recognized) |
| `GET /v1/messages/batches/{id}` | redact-only | processing metadata only |
| `GET /v1/messages/batches/{id}/results` | chat | JSONL restored line by line |
| `POST /v1/messages/batches/{id}/cancel` | redact-only | no content either way |
| `DELETE /v1/messages/batches/{id}` | redact-only | no content either way |
| `GET /v1/models` | redact-only | with `anthropic-version` (every Anthropic SDK request carries it): the model listing, metadata only |
| `GET /v1/models/{id}` | redact-only | with `anthropic-version` |
| `POST /v1/complete` | chat | legacy Text Completions: prompt redacted, completion restored (streaming included); no system note (the body has no system field) |
| `POST /v1/files` (with `anthropic-version`) | pass-through | beta Files API: the uploaded document is media (the non-goal) |
| `GET /v1/files/{id}/content` (with `anthropic-version`) | pass-through | the document back, verbatim |
| `GET /v1/organizations/...` (Admin API) | pass-through | org metadata |
| WebSocket realtime | websocket | not offered by Anthropic today |

`GET /v1/models` is shared: the Anthropic row above when the request
carries `anthropic-version` (and no other provider's marker), the OpenAI
row below otherwise — a request carrying a Google API key or the Cohere
SDK's header goes to that provider instead. With the llm-redact-pro
routing layer and `[routing] expose_models = true`, `GET /v1/models` is
answered **locally** instead (never forwarded) — Anthropic shape when
the request carries `anthropic-version`, OpenAI shape otherwise — so
Claude Code's gateway model discovery works. The local answer sits
behind the same gates as every proxy-generated reply for a real API
path: the provider the request is addressed to (`anthropic` for the
Anthropic-shaped call, `openai` otherwise) refuses it 502 when disabled,
and an llm-redact-pro access gate refuses an unadmitted client 403,
before the catalog is consulted. The rows do not change: that answer is
a routing feature, not a redaction classification. Under `[routing]` the id-only rows here and below —
`GET /v1/responses/{id}`, conversation item reads,
`GET /v1/files/{id}/content`, batch polls and results — carry no model,
so they take the protocol's `default_upstream` unless a path-matched
rule names another; their classification is unchanged.

Anthropic's beta Files API shares its paths (`/v1/files...`) with
OpenAI's. Routing is header-aware here: requests carrying an
`anthropic-version` header pass through to the ANTHROPIC upstream
(their uploads are documents — the media non-goal — so pass-through is
the correct handling, but they must reach the right host); everything
else takes the OpenAI files handling below. Any other request with
`anthropic-version` that no route matches is Anthropic's too (a newer
Anthropic API such as `/v1/skills`) — see [requests no route
matches](#requests-no-route-matches).

## OpenAI

| Endpoint | Classification | Notes |
|---|---|---|
| `POST /v1/chat/completions` | chat | streaming `delta.content`, tool-call arguments, and reasoning-model chain-of-thought (`delta.reasoning_content` / `delta.reasoning`) are all rehydrated per choice |
| `GET /v1/chat/completions/{id}` | chat | stored-completion retrieval restored |
| `POST /v1/responses` | chat | MCP connector: `tools[].type == "mcp"` entries (server_url, headers) pass through unredacted BY DESIGN — the provider needs the real credential; `mcp_call` arguments/output in responses are rehydrated, streaming included. The files the code interpreter WROTE into its container (`container_file_citation` annotations, a code interpreter call's output files; streaming included) are reported to a session router as the requester's |
| `GET /v1/responses/{id}` | chat | stored responses rehydrated |
| `GET /v1/responses/{id}/input_items` | chat | input-item echoes restored |
| `DELETE /v1/responses/{id}` | redact-only | id only (a body-less no-op: recognized) |
| `POST /v1/responses/compact` | chat | compaction (Codex CLI compacts long sessions: the whole conversation): the window redacted, the note joins its `instructions` (the compacted state must carry every token exactly); the compacted window (`response.compaction`: retained messages and tool calls, then one opaque, encrypted compaction item) restored — the compaction item goes back as sent. Never streams; stateless, so nothing is reported as a stored object |
| `POST /v1/responses/input_tokens` | redact-only | a create body, redacted with the note like the request it counts (Anthropic `count_tokens`' stance); the count forwarded as sent |
| `POST /v1/conversations` | chat | create: item content redacted, echoed response restored |
| `POST /v1/conversations/{id}/items` | chat | add items: content redacted + echo restored |
| `GET /v1/conversations/{id}` | chat | retrieve conversation, restored |
| `GET /v1/conversations/{id}/items` | chat | list items, stored content restored (list-envelope walk) |
| `GET /v1/conversations/{id}/items/{item_id}` | chat | single item restored |
| `DELETE /v1/conversations/{id}` | redact-only | ids only |
| `DELETE /v1/conversations/{id}/items/{item_id}` | redact-only | ids only |
| `POST /v1/embeddings` | redact-only | vectors come back verbatim |
| `POST /v1/files` | chat | multipart upload; JSONL file-part lines (batch + fine-tune), every part's `filename` / `filename*` and every plain form field (the structural `purpose`, `expires_after[…]` too, scanned as text) redacted, all other bytes preserved; a file that is not JSONL (a PDF, an image, a text file), a line that is not a JSON object, a form field that is not UTF-8, a preamble or epilogue, a part header without one reading, a Content-Transfer-Encoding or a declared charset other than UTF-8/US-ASCII refuses the upload (400: llm-redact cannot scan it — `detection = false` forwards it as sent); the file object answering it echoes the filename, restored |
| `GET /v1/files` | chat | the file list: each echoed filename restored in the request's own session |
| `GET /v1/files/{id}` | chat | the file object: its echoed filename restored |
| `GET /v1/files/{id}/content` | chat | batch output JSONL restored line by line |
| `DELETE /v1/files/{id}` | redact-only | id only |
| `POST /v1/batches` | chat | the caller's free-form `metadata` values are redacted out and restored in the echoed batch object; structural fields (`input_file_id`, `endpoint`, `completion_window`) carry nothing a detector matches and are forwarded byte-identical; no system note |
| `GET /v1/batches` | chat | the LIST is restored in the request's own session, like a single batch: one shared namespace holds every batch's tokens. With llm-redact-pro named users a listing resolves to an EMPTY session, and the session router's `listing_item_session` restores only the batches the READER created, each in the session it was created in; every other item keeps its placeholders. No system note |
| `GET /v1/batches/{id}` | chat | the batch object echoes `metadata`: restored |
| `POST /v1/batches/{id}/cancel` | chat | the cancelled batch object echoes `metadata`: restored |
| `GET /v1/models` | redact-only | the model listing, metadata only |
| `GET /v1/models/{id}` | redact-only | |
| `POST /v1/completions` | chat | legacy text completions: prompt redacted, choices[].text restored (streaming included); no system note |
| `POST /v1/moderations` | pass-through | DOCUMENTED GAP: moderation input is user text; redacting it would change moderation results, so it is deliberately untouched |
| `POST /v1/audio/transcriptions` | pass-through | audio media non-goal (multipart audio is never decoded) |
| `POST /v1/audio/translations` | pass-through | audio media non-goal |
| `POST /v1/audio/speech` | redact-only | the text-to-speech `input` is user text and is redacted; the audio response is bytes forwarded verbatim |
| `POST /v1/images/generations` | redact-only | the OUTPUT is media, but the `prompt` is plain text and is redacted; the response (`b64_json`/`url`) comes back verbatim — a dall-e-3 `revised_prompt` echo may carry placeholder tokens (fail-safe: the value it hides was never exposed) |
| `POST /v1/images/edits` | redact-only | multipart: the `prompt` form FIELD is redacted, and so is every other non-structural plain field (`user`, like its JSON twin); structural fields (`model`, `size`, `n`, `quality`, `response_format`, …) are forwarded as sent; image/mask file parts are media and stay byte-identical |
| `POST /v1/images/variations` | pass-through | image in, images out — no text anywhere in the request |
| `POST /v1/videos` | chat | Sora job create: the `prompt` (JSON or multipart form field) is redacted, and the returned job object's prompt ECHO is restored; multipart `input_reference` media stays byte-identical |
| `GET /v1/videos` | chat | job list: echoed prompts restored via the list-envelope walk |
| `GET /v1/videos/{id}` | chat | job retrieve: echoed prompt restored |
| `POST /v1/videos/{id}/remix` | chat | remix prompt redacted; echo restored |
| `GET /v1/videos/{id}/content` | redact-only | the rendered video: media bytes verbatim (a body-less no-op) |
| `DELETE /v1/videos/{id}` | redact-only | id only |
| `POST /v1/fine_tuning/jobs` | pass-through | file ids only; the training FILE is covered at upload via `/v1/files`; the created job is reported to a session router as its creator's |
| `GET /v1/fine_tuning/jobs` | pass-through | |
| `GET /v1/fine_tuning/jobs/{id}` | pass-through | the job's `result_files` are reported to a session router (like a batch's output files on its status; also on the job's `cancel`, `pause` and `resume`) |
| `POST /v1/uploads` | pass-through | DOCUMENTED GAP: the Uploads API (see the honest gaps below) |
| `POST /v1/uploads/{id}/parts` | pass-through | an opaque byte range of the file |
| `POST /v1/vector_stores` | pass-through | DOCUMENTED GAP: names and attributes are forwarded as sent |
| `POST /v1/assistants` | pass-through | DOCUMENTED GAP: Assistants (deprecated) |
| `POST /v1/threads/{id}/messages` | pass-through | DOCUMENTED GAP: Threads (deprecated) carry message content |
| `GET /v1/containers/{id}/files/{file_id}/content` | pass-through | a code-interpreter container's file, verbatim |
| `GET /v1/evals` | pass-through | |
| `POST /v1/realtime/client_secrets` | pass-through | an ephemeral Realtime key for a browser client; the WebSocket session itself is covered below |
| `GET /v1/organization/...` (Admin API) | pass-through | org metadata (singular: `/v1/organizations/` is Anthropic's) |
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
leaves as pass-through (file delete, model and deployment listings,
response/conversation delete) are RECOGNIZED on Azure, so
`[providers.azure] auth = "identity"` does not refuse them. On a body-less
request redact-only is a no-op; batch objects and the batch list are
**chat** on both providers because they echo the user `metadata` a batch
create carries, and file objects (the upload response, the list and one
file) because they echo the upload's redacted filename.

| Endpoint | Classification | Notes |
|---|---|---|
| `POST /openai/deployments/{d}/chat/completions` | chat | inherited from the OpenAI chat adapter |
| `POST /openai/v1/chat/completions` | chat | |
| `POST /openai/deployments/{d}/completions` | chat | legacy completions: prompt redacted, `choices[].text` restored (streaming included); no system note |
| `POST /openai/v1/completions` | chat | |
| `POST /openai/deployments/{d}/embeddings` | redact-only | |
| `POST /openai/v1/embeddings` | redact-only | |
| `POST /openai/deployments/{d}/images/generations` | redact-only | the prompt is redacted; image output verbatim |
| `POST /openai/deployments/{d}/images/edits` | redact-only | multipart: the `prompt` and every other non-structural form field (`user`) redacted, structural fields as sent, image parts byte-identical; under identity auth the structural fields are scanned as text too |
| `POST /openai/deployments/{d}/audio/speech` | redact-only | text-to-speech `input` redacted; audio bytes verbatim |
| `POST /openai/deployments/{d}/audio/transcriptions` | pass-through | audio media non-goal (identity auth refuses it) |
| `POST /openai/responses` | chat | Responses on Azure, inherited from the OpenAI Responses adapter |
| `POST /openai/v1/responses` | chat | |
| `GET /openai/responses/{id}` | chat | stored responses restored |
| `GET /openai/v1/responses/{id}/input_items` | chat | input-item echoes restored |
| `POST /openai/v1/responses/{id}/cancel` | chat | answers the Response object, restored |
| `DELETE /openai/responses/{id}` | redact-only | ids only |
| `POST /openai/v1/responses/compact` | chat | compaction, as on OpenAI |
| `POST /openai/responses/compact` | chat | the api-version form (Azure documents `/openai/v1`) |
| `POST /openai/v1/responses/input_tokens` | redact-only | not documented by Azure: a client that sends it gets the body redacted all the same |
| `POST /openai/v1/conversations` | chat | item content redacted, echo restored; STATIC vault session |
| `POST /openai/v1/conversations/{id}/items` | chat | |
| `GET /openai/v1/conversations/{id}` | chat | |
| `GET /openai/v1/conversations/{id}/items` | chat | list-envelope walk |
| `DELETE /openai/v1/conversations/{id}` | redact-only | ids only |
| `POST /openai/files` | chat | multipart JSONL upload, lines and filenames redacted (+ note on chat-shaped lines), the echoed filename restored; a non-JSON-object line, a non-JSONL file, a non-UTF-8 form field, a part header without one reading (a `filename*` outside UTF-8 included), a Content-Transfer-Encoding, or a declared charset other than UTF-8/US-ASCII refuses the upload (400) — under identity auth, and under key auth wherever redaction applies — and every form field is scanned as text |
| `POST /openai/v1/files` | chat | |
| `GET /openai/files` | chat | the file list: echoed filenames restored |
| `GET /openai/files/{id}` | chat | the file object: echoed filename restored |
| `GET /openai/v1/files` | chat | |
| `GET /openai/v1/files/{id}` | chat | |
| `DELETE /openai/files/{id}` | redact-only | |
| `GET /openai/files/{id}/content` | chat | batch output JSONL restored line by line |
| `GET /openai/v1/files/{id}/content` | chat | |
| `POST /openai/batches` | chat | file ids + user `metadata` (redacted out, restored in the echo) |
| `GET /openai/batches` | chat | the LIST is restored in the request's own session (both path families); with llm-redact-pro named users only the reader's own batches are restored (an empty listing session + `listing_item_session`), every other item keeps its placeholders |
| `GET /openai/v1/batches/{id}` | chat | |
| `POST /openai/batches/{id}/cancel` | chat | |
| `GET /openai/models` | redact-only | model listing |
| `GET /openai/v1/models` | redact-only | |
| `GET /openai/v1/models/{id}` | redact-only | |
| `GET /openai/deployments` | redact-only | deployment listing |
| `GET /openai/deployments/{d}` | redact-only | |
| `POST /openai/v1/fine_tuning/jobs` | pass-through | file ids only; the training FILE is covered at upload; the job, and later its `result_files`, are reported like OpenAI's |

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
| `POST /model/{m}/count-tokens` | redact-only | `input.converse` content redacted; answers a count. The `input.invokeModel.body` form is base64 of the model's native JSON prompt — text, not media — so it is decoded, redacted exactly like an `/invoke` body, and re-encoded; a blob that is not base64 of UTF-8 JSON, or whose JSON repeats a key, is refused with a 400 (never forwarded unredacted) |
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

The Gemini **Files API** passes through — files are media, the documented
non-goal: the upload (`POST /upload/v1beta/files`), the metadata-only
create (`POST /v1beta/files`), `files:register`, a file's metadata, delete
and download (`GET /v1beta/files/{id}[:download]`, `DELETE`), the list,
and `GET /download/v1beta/files/{id}:download` (a batch's output file),
all forwarded to the Gemini upstream. What llm-redact reads is who owns
what: the file a create answers with (`files/<id>`), and the output file a
finished batch's status names (`GET /v1beta/batches/{id}`), are reported to
a session router that tracks stored objects (llm-redact-pro's named users;
see [how-it-works.md](how-it-works.md)). The google-genai SDKs upload with
the resumable protocol and send the data chunks — the last one answers
with the file — to the upload URL Google returns, not through the proxy:
such a file is created without the proxy ever seeing its name.

A pass-through request that carries a **Google API key**
(`x-goog-api-key`, or a `key=`/`$key=` query parameter) or any other
`x-goog-*` header is forwarded to the Gemini upstream, never another
provider's: Gemini's v1 surface (`GET /v1/models[/{m}]`, …) shares OpenAI's
prefix, and the Google key must never be sent to api.openai.com. An
explicit Vertex path family (`/v1/projects/…`, `/v1/publishers/…`,
`/v1beta1/…`) stays Vertex's whatever key it carries (express mode and
service-account API keys authorize Vertex that way) — never sent to the
Gemini API's host. The Gemini model listing and one model's metadata
(`GET /v1beta/models[/{m}]`) are recognized, redact-only (a body-less
no-op).

The Gemini API's **OpenAI-compatible surface** (`/v1beta/openai/…`: the
OpenAI SDK with `base_url` `…/v1beta/openai/`) is the OpenAI surface
above under that prefix, on `[providers.gemini]`: chat completions
(streaming included), embeddings, files and batches are redacted and
restored exactly as OpenAI's (`GeminiOpenAIAdapter`,
`GeminiOpenAIResponsesAdapter`; the raw path is forwarded unchanged).

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
reuse the OpenAI multipart/JSONL/filename handling on Azure's path shapes;
batches are recognized and file objects restored (see the Azure table). **Azure OpenAI Responses**
(`POST /openai/responses` and the `/openai/v1/responses` preview, plus the
stored-response and input-item GETs, compaction and the input-token count)
reuses `OpenAIResponsesAdapter`
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
`upstream_base_url` so the inner path omits it — and others serve them
without any `/v1/` segment: Gemini `/v1beta/openai`, GitHub Models
`/inference`, Azure AI `/models`, Zhipu `/api/paas/v4`, a Cloudflare AI
Gateway `/v1/{account}/{gateway}/openai`. The endpoint is the path's TAIL:
the longest tail, at a segment boundary, that is a known OpenAI endpoint
(as is when it starts `/v1/`, else under `/v1`) is matched, so a known
endpoint routes under any base path; the stripped inner path is forwarded
byte-for-byte. An unknown tail still falls through to pass-through (to
that custom upstream) via the exact matcher.

## Requests no route matches

A request no route above matches is forwarded verbatim (pass-through) —
but only to a provider the proxy can POSITIVELY attribute it to
(`providers/attribution.py`), never to a guessed default (a guess hands
one provider's credential and the client's unredacted content to another:
the old anthropic default sent an OpenAI key and prompt to
api.anthropic.com):

1. an explicit path family, whatever the request carries: `/v1beta/…`,
   `/upload/v1beta/…`, `/download/v1beta/…` (Gemini API), `/v1/projects/…`,
   `/v1/publishers/…`, `/v1beta1/…` (Vertex AI), `/openai/…` (Azure),
   `/model/…`, `/guardrail/…`, `/async-invoke…` (Bedrock), `/api/…` (Ollama),
   `/v2/…` (Cohere), `/custom/NAME/…` (that custom upstream), and
   Anthropic's own `/v1/messages…`, `/v1/complete`, `/v1/organizations/…`;
2. otherwise the markers only one provider's clients send:
   `anthropic-version` (every Anthropic SDK request carries it) →
   Anthropic; a Google API key (`x-goog-api-key`, `key=`/`$key=`) or any
   other `x-goog-*` header → Gemini; an `openai-*` header → OpenAI; the
   Cohere SDK's `x-fern-sdk-name` → Cohere. Markers of two providers
   attribute nothing;
3. otherwise OpenAI: an `Authorization: Bearer sk-…` key that is not an
   Anthropic token (`sk-ant-…`), or a path under one of OpenAI's own API
   prefixes (`/v1/chat`, `/v1/files`, `/v1/fine_tuning`, `/v1/assistants`,
   `/v1/organization`, …).

Anything else is answered locally with a **recorded 404** that names the
path (never the query) and why, and nothing is forwarded. So are paths
that only look unrecognized — each would otherwise carry its body
unredacted to an upstream that may serve it as the recognized route
(routers that ignore a trailing `/` or case, front ends that normalize
paths):

- an **empty path segment** (`//`, in the raw or decoded path, `\`
  counted as `/`) is refused with a 400 before admission, like a `.`/`..`
  segment — never recorded or logged with its path (it may still hold an
  identity-prefix key). A base URL ending in `/` joined onto an endpoint
  path is the usual cause;
- **another spelling** of a recognized route is refused with a recorded,
  provider-shaped 400: the path that was matched must be the path that is
  forwarded, byte for byte. A spelling is the path as a router or a
  normalizing front end reads it: without a trailing `/`; in another case
  (lower case, and .NET's case-insensitive comparison, where a dotless `ı`
  is `I`); with `\` (or `%5C`) as `/` (IIS, Azure API Management, Envoy);
  with each segment's `;params` dropped (Tomcat, Jetty, Spring) and its
  trailing spaces, tabs and dots trimmed (IIS); in Unicode compatibility
  form (a full-width `ｃ` is `c`); and percent-decoded a second time, IIS's
  `%uXXXX` escapes included (a gateway that decodes before the app server
  does). Only a path one of whose spellings matches a route is refused: a
  Gemini `:method` or a Bedrock ARN (`%3A`, `%2F`) is matched as sent, and
  a spelling of a pass-through route stays pass-through;
- a recognized route **without its `/v1`** (`/chat/completions`,
  `/responses`, `/models`, … — an OpenAI-compatible base URL that lacks
  `/v1`) is a recorded 404 whose message gives the fix
  (`OPENAI_BASE_URL=http://127.0.0.1:8787/v1`);
- a recognized route **under an extra prefix** of up to eight segments
  (`/v1/v1/messages` — a base URL that repeats the API version) is a
  recorded 404. Azure's `/openai/…` and custom `/custom/NAME/…` paths
  embed OpenAI routes by design and are exempt.

These checks, like every routing step, cost time linear in the path's
length, and they run only for a request the request-origin rule and the
access gate admit (a refused request gets its own refusal). An
OpenAI-compatible prefix (`/custom/NAME/…`, `/v1beta/openai/…`) is matched
on the endpoint's tail: tails of at most eight segments after `/v1` (every
OpenAI endpoint has four or fewer), and the one at the first OpenAI
resource name.

`GET /` and `HEAD /` — the proxy's base URL itself, no provider's API —
are answered locally with a 200 (a client's liveness probe: the ollama
CLI's heartbeat), never forwarded or recorded; like every other answer
outside the reserved prefix, only after the request-origin check (a web
page on another site gets its recorded 403 instead).

An upstream **redirect** (a 3xx other than 304 carrying a `Location`) is
relayed only when the client's repeat of its original request would carry
nothing the proxy protects. A following client repeats its ORIGINAL
request at the `Location` — the unredacted body, and the credential
headers its HTTP library keeps across hosts — and a relative `Location`
resolves against the proxy (without a `/custom/NAME` prefix). So a redirect
answering a request with a body on a route the proxy redacts, a routed or
identity-signed request, a request that presented a credential for the
proxy itself (an identity path prefix, an `x-llm-redact-*` header, an
access-gate subject), or any request to a custom upstream is answered with
a recorded, provider-shaped 502 naming the status only (the `Location` is
never relayed or logged) and counted as an upstream error; a routed hop
that redirects is a hop fault the router can fail over from. A
first-party provider's redirect is still relayed for plain pass-through
(the body already went to that provider verbatim) and for a body-less
read of a recognized route (a download CDN — its bytes then bypass
rehydration, placeholders intact), with the client's own credential.

## Known uncovered content surfaces (honest gaps)

These carry user content but are **not** redacted today — a request to one
is forwarded verbatim, the same honesty posture as warn mode and
per-provider `detection = false`. Documented so nobody assumes protection
that is not there:

- **OpenAI Uploads API** (`POST /v1/uploads`, `/parts`, `/complete`) — the
  large-file sibling of `/v1/files`. Each part is an opaque byte range and a
  secret can straddle a part boundary, so per-line scanning cannot be applied
  safely; real coverage would need stateful cross-part buffering. Pass-through,
  routed to the OpenAI upstream with the client's own credential (pinned by
  test — it previously fell through to the anthropic default; a credential
  the proxy holds is never lent to it: a recorded 403). For the same reason
  the File a completed Upload creates is reported to a session router as its
  creator's only when its `purpose` is stated and is not `batch`: a batch
  input file's requests would be run with the upload's credential, and the
  stored objects they cite were never checked.
- **OpenAI Assistants / Threads / vector-store search** — on OpenAI's
  announced deprecation path (Responses/Conversations is the successor), so
  not built. The same holds for their Azure v1 twins (`/openai/v1/threads`,
  `/openai/v1/vector_stores/{id}/search`), and Azure's `/openai/v1/evals`
  and `/openai/v1/containers` are not covered either — pass-through, and
  refused wherever the proxy's own credential would carry them
  (`auth = "identity"`, or a routed operator key). OpenAI's own
  `/v1/containers` (the code interpreter's containers and their files)
  passes through to the OpenAI upstream too, on the same terms; the files a
  Response's code wrote into a container are reported to a session router
  as that Response's creator's.
- **OpenAI WebRTC realtime** (`POST /v1/realtime/calls`, SDP offer/answer) —
  after setup, media and the event data channel flow peer-to-peer and never
  transit this HTTP/WS proxy at all: structurally unreachable, not merely
  unimplemented. The WebSocket realtime transport IS covered.

Closing any of these is additive future work; each reuses the existing
redaction/rehydration machinery except where noted.
