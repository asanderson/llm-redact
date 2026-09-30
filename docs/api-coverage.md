# API coverage matrix

Every documented Anthropic and OpenAI endpoint, and the commonly used
Gemini API, Vertex AI, Azure OpenAI and Bedrock runtime endpoints, with the
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
credential included, once the stored-object check has read the upload) —
except a BINARY file part (a PDF, an image, an archive, text that is not
UTF-8): it cannot be redacted, so with the client's own key it is forwarded
UNSCANNED (`[detection] binary_uploads = "forward"`, the default; counted in
`/status` `unscanned_uploads_total`) or refused 400 (`"refuse"`), and under a
credential the proxy holds it is always refused 400. `[providers.NAME] detection =
false` with the client's own key forwards such a body as sent, and so does
every pass-through route (a route this table does not claim) — reached
only with the client's own credential, since a credential the proxy holds
is never lent to one (above).

## Anthropic

| Endpoint | Classification | Notes |
|---|---|---|
| `POST /v1/messages` | chat | MCP connector: `mcp_servers[]` blocks pass through unredacted BY DESIGN — the provider must hold the real `authorization_token` to call the MCP server; everything else in the body is redacted. The files a code execution run WROTE (`code_execution_output` / `bash_code_execution_output` entries of a code execution tool result, streaming included) and the code execution CONTAINER the answer names (`container.id`, streamed in `message_start` / `message_delta`; later requests reuse it by id) are reported to a session router as the requester's — a container the request itself named is not |
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
| `POST /v1/files` (with `anthropic-version`) | chat | beta Files API upload (multipart/form-data): read by content exactly as the OpenAI Files upload (the same code): a JSONL file's lines redacted as JSON, any other text file as one text re-encoded as it came, every part's file name redacted; a BINARY document (a PDF — even an all-ASCII one —, an image) is forwarded unscanned with the client's own key (`binary_uploads = "forward"`, counted) and refuses the upload (400) under `"refuse"` or a credential the proxy holds, as does anything it cannot read; the file object answering it echoes the filename, restored, and is reported to a session router as its creator's |
| `GET /v1/files` (with `anthropic-version`) | chat | the file list (`{"data": [...]}`): each echoed filename restored in the request's own session, or — with a session router attributing listed items — each file in its creator's |
| `GET /v1/files/{id}` (with `anthropic-version`) | chat | the file's metadata: its filename restored |
| `GET /v1/files/{id}/content` (with `anthropic-version`) | chat | the file back (downloadable for files a tool created), restored like an OpenAI file download: a JSONL file line by line as JSON, one JSON document JSON-escaped, any other text file as one text, a binary file never read |
| `DELETE /v1/files/{id}` (with `anthropic-version`) | redact-only | the id only |
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
OpenAI's. Routing is header-aware here: a request carrying the
`anthropic-version` header (and no other provider's marker) takes the
Anthropic rows above and reaches the ANTHROPIC upstream; everything else
takes the OpenAI files handling below (a request carrying the markers of
both providers is neither's: a recorded 404). Any other request with
`anthropic-version` that no route matches is Anthropic's too (a newer
Anthropic API such as `/v1/skills`) — see [requests no route
matches](#requests-no-route-matches).

## OpenAI

| Endpoint | Classification | Notes |
|---|---|---|
| `POST /v1/chat/completions` | chat | streaming `delta.content`, tool-call arguments, and reasoning-model chain-of-thought (`delta.reasoning_content` / `delta.reasoning`) are all rehydrated per choice |
| `GET /v1/chat/completions/{id}` | chat | stored-completion retrieval restored |
| `POST /v1/responses` | chat | MCP connector: `tools[].type == "mcp"` entries (server_url, headers) pass through unredacted BY DESIGN — the provider needs the real credential; `mcp_call` arguments/output in responses are rehydrated, streaming included. The files the code interpreter WROTE into its container (`container_file_citation` annotations, a code interpreter call's output files; streaming included), and the container they were written in (a call's or a citation's `container_id`) unless the request names it, are reported to a session router as the requester's |
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
| `POST /v1/files` | chat | multipart upload, each file part decided by its CONTENT: a JSONL file's lines redacted as JSON — every value, whatever its key, except the request a line carries under purpose `batch` (its `body`) or `fine-tune` (the conversation), which keeps a request's protocol fields —, any other text file (UTF-8, or UTF-16/32 with a byte-order mark) redacted as one text and re-encoded as it came; every part's `filename` / `filename*` and every plain form field (the structural `purpose`, `expires_after[…]` too, scanned as text) redacted, all other bytes preserved; a BINARY file (a PDF — even an all-ASCII one —, an image, an archive, text that is not UTF-8) is forwarded byte-identical and UNSCANNED with the client's own key (`[detection] binary_uploads = "forward"`, the default, counted) and refuses the upload otherwise (400: `"refuse"`, or a credential the proxy holds); a JSONL line nesting JSON too deep, a form field that is not UTF-8, a preamble or epilogue, a part header without one reading, a Content-Transfer-Encoding or a declared charset other than the content's own refuses the upload (400 — `detection = false` forwards it as sent); the file object answering it echoes the filename, restored |
| `GET /v1/files` | chat | the file list: each echoed filename restored in the request's own session |
| `GET /v1/files/{id}` | chat | the file object: its echoed filename restored |
| `GET /v1/files/{id}/content` | chat | a text file this process uploaded redacted as one text (remembered by digest, newest 1024) restored as one text — a value lands exactly as it was redacted, never JSON-escaped —; otherwise a JSONL file (batch output) line by line as JSON, one JSON document over its source text JSON-escaping each restored value, any other text file (a fine-tune results CSV) as one text; re-encoded as it came; a binary file untouched |
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
| `POST /v1/fine_tuning/jobs` | chat | the caller's free-form `metadata` redacted out and restored in the echoed job; the training data is the FILE, redacted at its upload (`/v1/files`). VERBATIM fields — `suffix` (it becomes part of the fine-tuned model's name, which later requests cite in their structural `model`), `training_file` / `validation_file` (file ids) and `integrations` (a W&B project, entity, run name and tags) — are scanned but never rewritten: a value llm-redact would redact there refuses the request (400) instead of forwarding it or naming a model, file or project that does not exist. `hyperparameters` / `method` go through the walk (numbers and enums); a reinforcement grader's `name` (every `name` under `method`, nested multi-graders included) is a LABEL — redacted, and restored in every echo of the job (create, list, read, cancel/pause/resume). The created job is reported to a session router as its creator's; no system note |
| `GET /v1/fine_tuning/jobs` | chat | the job list: each job's echoed `metadata` restored in the request's own session; a listing a session router attributes per item (`listing_item_session`) |
| `GET /v1/fine_tuning/jobs/{id}` | chat | the job object, restored; its `result_files` are reported to a session router (like a batch's output files on its status; also on the job's `cancel`, `pause` and `resume`) |
| `POST /v1/fine_tuning/jobs/{id}/cancel` | chat | answers the job object, restored |
| `POST /v1/fine_tuning/jobs/{id}/pause` | chat | answers the job object, restored |
| `POST /v1/fine_tuning/jobs/{id}/resume` | chat | answers the job object, restored |
| `GET /v1/fine_tuning/jobs/{id}/events` | chat | the provider's messages about the job, restored |
| `GET /v1/fine_tuning/jobs/{id}/checkpoints` | redact-only | checkpoint metadata (a body-less no-op: recognized) |
| `POST /v1/fine_tuning/checkpoints/{id}/permissions` | pass-through | an admin key sharing a checkpoint across projects (org administration, like the Admin API) |
| `POST /v1/uploads` | pass-through | DOCUMENTED GAP: the Uploads API cannot be redacted statelessly (see the honest gaps below); never lent a credential the proxy holds |
| `POST /v1/uploads/{id}/parts` | pass-through | an opaque byte range of the file |
| `POST /v1/vector_stores` | chat | create: the store's `name` (a label — the store is addressed by id), `description` and the caller's `metadata` redacted, echoed back restored (the `name` of every vector store object, alone or listed); VERBATIM (scanned, never rewritten — a value llm-redact would redact refuses the request, 400): its `file_ids`. The store is reported to a session router as its creator's; no system note on any vector store route |
| `GET /v1/vector_stores` | chat | the store list, restored in the request's own session; a listing a session router attributes per item |
| `GET /v1/vector_stores/{id}` | chat | the store, restored |
| `POST /v1/vector_stores/{id}` | chat | modify: `name` and `metadata` redacted; the echo restored |
| `DELETE /v1/vector_stores/{id}` | redact-only | id only |
| `POST /v1/vector_stores/{id}/search` | chat | the `query` redacted, and so are attribute-filter VALUES — in the same (static) session the files' `attributes` were redacted in, so a filter's placeholder is the stored attribute's (the vault is deterministic); a filter's `key` names an attribute KEY (never rewritten where it was set) and is verbatim. The results — file content chunks, filenames, attributes, the echoed query — are restored |
| `POST /v1/vector_stores/{id}/files` | chat | attach a file: its `attributes` (values under keys the caller chooses — walked like `metadata`, a key named `id` or `name` included) redacted, `file_id` verbatim; the vector-store file object restored |
| `GET /v1/vector_stores/{id}/files` | chat | the store's files (attributes restored), read as the store's: not a listing attributed per item |
| `GET /v1/vector_stores/{id}/files/{file_id}` | chat | restored |
| `POST /v1/vector_stores/{id}/files/{file_id}` | chat | update `attributes`: redacted, the echo restored |
| `DELETE /v1/vector_stores/{id}/files/{file_id}` | redact-only | ids only |
| `GET /v1/vector_stores/{id}/files/{file_id}/content` | chat | the file's parsed content chunks and filename, restored |
| `POST /v1/vector_stores/{id}/file_batches` | chat | `attributes` redacted; `file_ids` and each `files[].file_id` verbatim |
| `GET /v1/vector_stores/{id}/file_batches/{batch_id}` | chat | the batch's status, restored |
| `POST /v1/vector_stores/{id}/file_batches/{batch_id}/cancel` | chat | |
| `GET /v1/vector_stores/{id}/file_batches/{batch_id}/files` | chat | the batch's vector-store files, restored like the store's files |
| `POST /v1/assistants` | pass-through | DOCUMENTED GAP: Assistants (deprecated) |
| `POST /v1/threads/{id}/messages` | pass-through | DOCUMENTED GAP: Threads (deprecated) carry message content |
| `POST /v1/containers` | chat | a code interpreter container: its `name` (a label) redacted and restored on every container object, alone or listed; its starting `file_ids` are VERBATIM (scanned, never rewritten — a value llm-redact would redact refuses the request, 400); `expires_after` / `memory_limit` go through the walk. The container is reported to a session router as its creator's (so is one a Response's code interpreter call created, `container_id`); no system note on any container route |
| `GET /v1/containers` | chat | the container list; a listing a session router attributes per item |
| `GET /v1/containers/{id}` | chat | |
| `DELETE /v1/containers/{id}` | redact-only | id only |
| `POST /v1/containers/{id}/files` | chat | multipart: read and redacted as a `/v1/files` upload is (by content: a JSONL or text file redacted, a binary file forwarded unscanned with the client's own key and refused under a credential the proxy holds or `binary_uploads = "refuse"`; its filename and every form field redacted); JSON: the stored `file_id` it copies in is verbatim. The container file object (its `path` echoes the filename) restored; the new container file is reported to a session router |
| `GET /v1/containers/{id}/files` | chat | the container's files (paths restored), read as the container's: not a listing attributed per item |
| `GET /v1/containers/{id}/files/{file_id}` | chat | restored |
| `GET /v1/containers/{id}/files/{file_id}/content` | chat | the file, uploaded or written by the code, restored as a `/v1/files` download is: a JSONL file line by line as JSON, one JSON document JSON-escaped, any other text file as one text, a binary file untouched |
| `DELETE /v1/containers/{id}/files/{file_id}` | redact-only | ids only |
| `GET /v1/evals` | pass-through | |
| `POST /v1/realtime/client_secrets` | pass-through | an ephemeral Realtime key for a browser client; the WebSocket session itself is covered below |
| `GET /v1/organization/...` (Admin API) | pass-through | org metadata (singular: `/v1/organizations/` is Anthropic's) |
| WebSocket `/v1/realtime` | websocket | beta + GA event vocabularies; MCP tool config preserved, MCP arguments rehydrated |

## Google Gemini API

The Gemini API (`generativelanguage.googleapis.com`, `[providers.gemini]`);
every `/v1beta/models/…` verb row also matches under `/v1/` and
`/v1beta/tunedModels/{t}`. Batch Mode and context caches are stored or
async (read back later with no first-message anchor), so they use the
STATIC vault session — the batch stance; with llm-redact-pro's named users,
the user's own copy of it — and no route here but the generate verbs and
`countTokens` ever carries the system note.

| Endpoint | Classification | Notes |
|---|---|---|
| `POST /v1beta/models/{m}:generateContent` | chat | text, thoughts, function-call args, code and grounding restored |
| `POST /v1beta/models/{m}:streamGenerateContent` | chat | the SSE stream (`alt=sse`) and the buffered JSON-array form, per-candidate channels |
| `POST /v1beta/models/{m}:countTokens` | redact-only | note counted too, keeping counts honest |
| `POST /v1beta/models/{m}:embedContent` | redact-only | vectors back |
| `POST /v1beta/models/{m}:batchEmbedContents` | redact-only | vectors back |
| `POST /v1beta/models/{m}:predict` | redact-only | Imagen: `instances[].prompt` redacted, image bytes back |
| `POST /v1beta/models/{m}:predictLongRunning` | redact-only | Veo: prompt redacted; the operation is reported to a session router as its creator's |
| `GET /v1beta/models/{m}/operations/{id}` | pass-through | the Veo job's status |
| `GET /v1beta/models` | redact-only | the model listing, metadata only (a body-less no-op: recognized) |
| `GET /v1beta/models/{m}` | redact-only | one model's metadata |
| `POST /v1beta/cachedContents` | redact-only | the cached prompt redacted; the cache (`cachedContents/<id>`, which later bodies cite as `cachedContent`) is reported to a session router |
| `GET /v1beta/cachedContents` | pass-through | metadata only (the cached content is never returned) |
| `GET /v1beta/cachedContents/{id}` | pass-through | |
| `PATCH /v1beta/cachedContents/{id}` | pass-through | the expiry |
| `DELETE /v1beta/cachedContents/{id}` | pass-through | |
| `POST /v1beta/models/{m}:batchGenerateContent` | redact-only | Batch Mode: the inlined requests (and the display name) redacted, no note; the answer is the batch operation (`batches/<id>`), reported to a session router as its creator's — its echo keeps the placeholders (restored on the status read) |
| `POST /v1beta/models/{m}:asyncBatchEmbedContent` | redact-only | async batch embeddings: as `:batchGenerateContent` |
| `GET /v1beta/batches` | chat | the batch list (`{"operations": [...]}`): every batch's echo restored in the request's own session, or — with a session router attributing listed items (llm-redact-pro's named users) — each item, by its `name`, in its creator's |
| `GET /v1beta/batches/{id}` | chat | a batch's status: the display name and a finished batch's INLINED responses (model output carrying placeholders) restored; a file-output batch names its output file, reported to a session router |
| `POST /v1beta/batches/{id}:cancel` | redact-only | the name only |
| `DELETE /v1beta/batches/{id}` | redact-only | the name only |
| `PATCH /v1beta/batches/{id}:updateGenerateContentBatch` | pass-through | a pending batch's update: never sent with a credential the proxy holds |
| `POST /upload/v1beta/files` | chat | the Files upload. The SINGLE-REQUEST protocol (`X-Goog-Upload-Protocol: multipart`, a `multipart/related` body: the JSON metadata, then the media) is redacted part by part, EVERY part read as a file by its content with the OpenAI Files upload's part loop — the metadata (one JSON line: its `displayName`), a JSONL file (a batch input file's requests) line by line as JSON, any other text file as one text re-encoded as it came, every part's file name; a BINARY file is forwarded unscanned with the client's own key (`binary_uploads = "forward"`, counted) and refuses the upload (400) under `"refuse"` or a credential the proxy holds, as does anything it cannot read. The RESUMABLE protocol's start (JSON metadata) is redacted with the client's own key, and its upload URL (Google's; the data chunks go there directly, never through llm-redact — the media gap below) relayed; under a credential the proxy holds the start is REFUSED (403) — that URL would let the client store unread bytes as the proxy's principal — and an `X-Goog-Upload-URL` answer header is never relayed. A data chunk sent to the proxy (`upload_id`, an `X-Goog-Upload-Command` other than `start`) and the raw protocol are pass-through. The File answering it: `displayName` restored, reported to a session router as its creator's. A session router's stored-object check is shown the single-request upload's metadata part as the create's JSON body (a chosen `file.name` included), before anything is sent |
| `POST /v1beta/files` | chat | the metadata-only create: `displayName` redacted and restored; reported as the creator's |
| `POST /v1beta/files:register` | pass-through | registers Cloud Storage objects the provider reads as the caller: never sent with a credential the proxy holds |
| `GET /v1beta/files` | chat | the file list (`{"files": [...]}`): each `displayName` restored in the request's own session, or — with a session router attributing listed items — each file, by its `name`, in its creator's |
| `GET /v1beta/files/{id}` | chat | a file's metadata: `displayName` restored |
| `GET /v1beta/files/{id}:download` | chat | the file back, restored like an OpenAI file download: a JSONL file line by line as JSON (re-escaped), one JSON document JSON-escaped, any other text file as one text, a binary file never read |
| `GET /download/v1beta/files/{id}:download` | chat | the media download a batch's output file (JSONL: model output carrying placeholders) is fetched from: as above |
| `DELETE /v1beta/files/{id}` | redact-only | the name only |

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
| `POST /openai/files` | chat | multipart upload read by content like OpenAI's: JSONL lines, text files and filenames redacted (+ note on chat-shaped lines), the echoed filename restored; a binary file is forwarded unscanned with the client's own key (`binary_uploads = "forward"`, counted) and refused (400) under identity auth or `"refuse"`; a too-deep JSONL line, a non-UTF-8 form field, a part header without one reading (a `filename*` outside UTF-8 included), a Content-Transfer-Encoding, or a declared charset other than the content's own refuses the upload (400) — under identity auth, and under key auth wherever redaction applies — and every form field is scanned as text |
| `POST /openai/v1/files` | chat | |
| `GET /openai/files` | chat | the file list: echoed filenames restored |
| `GET /openai/files/{id}` | chat | the file object: echoed filename restored |
| `GET /openai/v1/files` | chat | |
| `GET /openai/v1/files/{id}` | chat | |
| `DELETE /openai/files/{id}` | redact-only | |
| `GET /openai/files/{id}/content` | chat | a JSONL file restored line by line as JSON, one JSON document JSON-escaped, any other text file as one text, a binary file untouched |
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
| `POST /openai/v1/fine_tuning/jobs` | chat | as on OpenAI: `metadata` redacted and restored, the verbatim fields scanned but never rewritten; the job, and later its `result_files`, are reported like OpenAI's |
| `POST /openai/fine_tuning/jobs` | chat | the api-version form |
| `GET /openai/fine_tuning/jobs` | chat | |
| `GET /openai/v1/fine_tuning/jobs/{id}` | chat | |
| `POST /openai/fine_tuning/jobs/{id}/cancel` | chat | also `pause` / `resume` |
| `GET /openai/v1/fine_tuning/jobs/{id}/events` | chat | |
| `GET /openai/fine_tuning/jobs/{id}/checkpoints` | redact-only | |
| `POST /openai/v1/vector_stores` | chat | vector stores, as on OpenAI (both path families) |
| `GET /openai/vector_stores` | chat | |
| `POST /openai/v1/vector_stores/{id}/search` | chat | |
| `POST /openai/vector_stores/{id}/files` | chat | |
| `GET /openai/v1/vector_stores/{id}/files/{file_id}/content` | chat | |
| `POST /openai/v1/vector_stores/{id}/file_batches` | chat | |
| `DELETE /openai/vector_stores/{id}` | redact-only | |
| `POST /openai/v1/containers` | chat | code interpreter containers, as on OpenAI (the v1 API only) |
| `GET /openai/v1/containers/{id}` | chat | |
| `POST /openai/v1/containers/{id}/files` | chat | |
| `GET /openai/v1/containers/{id}/files/{file_id}/content` | chat | |
| `DELETE /openai/v1/containers/{id}/files/{file_id}` | redact-only | |

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
Mode** (`models/{m}:batchGenerateContent`, `:asyncBatchEmbedContent`) are
redact-only: the cached prompt and the inlined batch requests are content
that must not reach the provider in the clear, while their responses carry
only a cache/operation name. Both are stored/async — the cache is reused
and batch results are read later through the batch's status
(`GET /v1beta/batches/{id}`) with no first-message anchor — so they use the
STATIC vault session (the batch stance; with llm-redact-pro's named users,
the user's own copy of it), keeping redact/rehydrate always in agreement:
the status restores a finished batch's inlined responses in that session
(see the [Gemini API table](#google-gemini-api)). The per-cache
GET/PATCH/DELETE and list return metadata only and pass through.

The Gemini **Files API** is recognized (the [Gemini API
table](#google-gemini-api)): the single-request upload and the
metadata-only create are redacted, every echo of a file (the create, its
metadata, the list) restored and its download restored when it is text,
so a credential the proxy holds may reach it. What llm-redact reads is also
who owns what: the file a create answers with (`files/<id>`), and the
output file a finished batch's status names (`GET /v1beta/batches/{id}`),
are reported to a session router that tracks stored objects
(llm-redact-pro's named users; see [how-it-works.md](how-it-works.md)). The
google-genai SDKs upload with the RESUMABLE protocol and send the data
chunks — the last one answers with the file — to the upload URL Google
returns, not through the proxy: with the client's own key such a file's
data never passes llm-redact (its display name does, redacted), and under
a credential the proxy holds the resumable start is refused — send the
file in one multipart request instead.

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
and that identity auth signs every recognized route); the Gemini API table
is pinned by `tests/test_api_coverage.py` like the others; Cohere and
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
reuse the OpenAI multipart/content-classified file/filename handling on Azure's path shapes;
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
on the endpoint's tail: tails of at most ten segments after `/v1` (every
OpenAI endpoint has five or fewer), and the one at the first OpenAI
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

- **OpenAI Uploads API** (`POST /v1/uploads`, `/parts`, `/complete`,
  `/cancel`) — the large-file sibling of `/v1/files`, and deliberately NOT
  recognized: it cannot be redacted without state the proxy does not keep.
  The create declares the file's total `bytes` (and `mime_type`) before any
  content is sent, and redaction changes lengths; each part is an opaque
  byte range (up to 64 MB) whose boundaries can split a JSONL line, a
  multi-byte character or a value in two; parts may be sent in parallel and
  are put in order only by the `part_ids` list at `/complete` (with an
  optional checksum of the whole file). Scanning each part alone would miss
  a secret that straddles a boundary and could corrupt the file; redacting
  it correctly would mean buffering every part of every Upload across
  requests — the file's plaintext held by the proxy between requests, and
  replayed to the provider only at `/complete` — which the proxy never
  does. So it stays pass-through, routed to the OpenAI upstream with the
  client's own credential (pinned by test — it previously fell through to
  the anthropic default), and a credential the proxy holds (identity, a
  routed operator key) is never lent to any Uploads route: a recorded 403
  before the body is read (pinned by `tests/test_lent_credentials.py`).
  Upload through `POST /v1/files` instead, whose file part is scanned.
  For the same reason the File a completed Upload creates is reported to a
  session router as its creator's only when its `purpose` is stated and is
  not `batch`: a batch input file's requests would be run with the
  upload's credential, and the stored objects they cite were never
  checked.
- **OpenAI Assistants / Threads** — on OpenAI's announced deprecation path
  (Responses/Conversations is the successor), so not built. The same holds
  for their Azure v1 twins (`/openai/v1/threads`), and Azure's
  `/openai/v1/evals` is not covered either — pass-through, and refused
  wherever the proxy's own credential would carry them (`auth =
  "identity"`, or a routed operator key).
- **Gemini API resumable and raw uploads** — a resumable upload's data
  chunks go to the upload URL Google returns (its host, not the proxy's),
  and a data chunk or a raw-protocol upload sent to the proxy is forwarded
  as sent (pass-through, with the client's own key; refused under a
  credential the proxy holds, where the resumable start is refused too).
  Only the single-request multipart upload's content is scanned.
- **OpenAI WebRTC realtime** (`POST /v1/realtime/calls`, SDP offer/answer) —
  after setup, media and the event data channel flow peer-to-peer and never
  transit this HTTP/WS proxy at all: structurally unreachable, not merely
  unimplemented. The WebSocket realtime transport IS covered.

Closing any of these is additive future work; each reuses the existing
redaction/rehydration machinery except where noted.
