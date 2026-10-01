# How it works

The redaction mechanism end to end: the request round trip, one worked
example with the exact records it leaves behind, how the vault persists
and restores mangled tokens, and how sessions isolate placeholder
namespaces.

## Contents

- [The request round trip](#the-request-round-trip)
- [A worked example](#a-worked-example)
  - [The vault records](#the-vault-records)
  - [The audit record](#the-audit-record)
- [Persistence, mangled tokens, and size limits](#persistence-mangled-tokens-and-size-limits)
- [Session isolation](#session-isolation)

## The request round trip

A request round trip, using an email as the private value:

![Sequence diagram of a non-streaming request: detect, issue a vault token, forward placeholders, reverse-lookup on the response](diagrams/sequence-chat.svg)

*Animated: the messages appear in order. Static [PNG](diagrams/sequence-chat.png) · [GIF](diagrams/sequence-chat.gif) · [Mermaid source](diagrams/sequence-chat.mmd).*

Streaming is the hard case — a placeholder can be split across chunk
boundaries. The rehydrator holds back only a viable token prefix and flushes
any leftover before the stream-end event, so streamed output is byte-for-byte
identical to the non-streaming result (the same machinery serves SSE, NDJSON,
Bedrock's binary eventstream, and realtime WebSocket deltas). Where a stream
chunk is a whole JSON object (Gemini/Vertex, Gemini Live), everything but
its streamed text is restored exactly as the buffered response would be,
walked from the chunk's root — a tool call's arguments at their opaque
position included:

![Sequence diagram of streaming rehydration reassembling a placeholder split across two SSE deltas](diagrams/sequence-streaming.svg)

*Animated. Static [PNG](diagrams/sequence-streaming.png) · [GIF](diagrams/sequence-streaming.gif) · [Mermaid source](diagrams/sequence-streaming.mmd).*

A request body with nothing to redact is forwarded as its original bytes
(no parse-and-reserialize round trip). The one exception is a JSON object
that repeats a key, at any depth: the parser keeps the last occurrence,
so the redactor never sees the earlier ones — while the provider's parser
might keep the first. Such a body (and such an uploaded JSONL line) is
always re-serialized from the walked object, so the earlier occurrences
never leave the machine — on a `detection = false` provider too, where it
is re-serialized unredacted, so the upstream acts on exactly the value the
session router's ownership check read; realtime frames are always
re-serialized anyway, and a Bedrock `count-tokens` base64 body that
repeats a key is refused with a 400.

Watch the round trip live: `llm-redact status` and the `/recent` feed
([dashboard.md](dashboard.md)) show detections and restores, and an
agent with the slash-command plugins installed can do the same in-tool with
`/llm-redact:status`, `/llm-redact:recent`, and `/llm-redact:preview`
([plugins.md](plugins.md)).

## A worked example

One request through the proxy, with the records it actually leaves
behind. (The private values here are the vendors' canonical
documentation fakes.) The tool sends:

```json
POST /v1/messages
{
  "model": "claude-sonnet-4-5",
  "max_tokens": 256,
  "messages": [{
    "role": "user",
    "content": "Email jane.doe@corp.example that the key AKIAIOSFODNN7EXAMPLE was found in the repo and must be rotated."
  }]
}
```

The provider receives placeholders in place of both values, plus an
injected system note telling the model to reproduce tokens exactly:

```json
{
  "model": "claude-sonnet-4-5",
  "max_tokens": 256,
  "messages": [{
    "role": "user",
    "content": "Email «EMAIL_001» that the key «AWS_KEY_001» was found in the repo and must be rotated."
  }],
  "system": "Some values in this conversation have been replaced with privacy tokens of the form «TYPE_NNN» (for example «EMAIL_000»). Treat each token as an opaque identifier for the real value and reproduce every token exactly, character for character, whenever you refer to it."
}
```

The model answers in terms of the tokens — *"Done - I emailed
«EMAIL_001» and flagged «AWS_KEY_001» for rotation."* — and the proxy
restores the real values on the way back, so the tool reads:

```json
{"content": [{"type": "text",
  "text": "Done - I emailed jane.doe@corp.example and flagged AKIAIOSFODNN7EXAMPLE for rotation."}]}
```

### The vault records

`vault.db`, created `0600`; this is the file that never leaves your
machine. One row per detected value, and the same
`(session, type, value)` always maps to the same token, so the email is
still `«EMAIL_001»` ten turns later:

```json
{"session_id": "default", "detector_type": "EMAIL",   "original": "jane.doe@corp.example",
 "placeholder": "«EMAIL_001»",   "n": 1, "created_at": "2026-07-14T11:57:05Z"}
{"session_id": "default", "detector_type": "AWS_KEY", "original": "AKIAIOSFODNN7EXAMPLE",
 "placeholder": "«AWS_KEY_001»", "n": 1, "created_at": "2026-07-14T11:57:05Z"}
```

With `[vault] encryption = "fernet"` the `original` column holds an
HMAC index and ciphertext instead of plaintext.

`n` counts per `(session, type)`, so every session has its own
`«EMAIL_001»`. A request can therefore carry tokens its session never
issued — a compacted history quoting the conversation's earlier session,
an answer pasted from another conversation. A new value is never numbered
onto one of those: before redacting, the proxy reads the highest
placeholder number per type the request already carries (canonical,
fuzzy-mangled and JSON-escaped forms alike — the **token floor**) and
numbers each new value `max(MAX(n), floor) + 1`. The request's foreign
tokens stay unissued in this session, so their echoes pass through
verbatim instead of restoring the new value; skipped numbers are a gap,
never a reuse. Values already mapped keep their tokens. Multipart uploads
are read part by part for the floor, the Bedrock CountTokens blob is
decoded for it, and a realtime connection keeps a running floor over
everything its client has sent. The record of what a per-request floor
cannot see is in [compaction-relink.md](compaction-relink.md).

Deleting a session (the `session_ttl_days` prune, `POST
/__llm-redact/sessions/prune`, `llm-redact sessions prune`, an access
gate's purge) deletes its rows but **retires its numbers**: one row per
deleted session records the highest number it held, and a new value in
that session is numbered above it, forever. So no `(session, type, n)`
ever carries two values — a provider history, another proxy instance's
cached copy of the session, or a live realtime connection still holding a
deleted token can only restore it to its own value, or pass it through,
never to a value issued after the delete:

```json
{"session_id": "conv-4f1c…", "n": 7}
```

Every view of a session re-reads that number at most once a second and
drops the deleted values from memory when it moved; the instance that
deleted the session drops them at once. A re-read that fails (the database
is briefly unreachable) keeps what the view holds and tries again a second
later: a cached token only ever restores its own value, so restoring one
never waits on the database.

A request's new values are written in **one transaction**, committed once
before anything is forwarded (the sqlite vault runs `synchronous=FULL`, so
each commit is an fsync — once per request, not once per value). A refused
request (a block-mode value, `max_body_strings`) and a failed commit roll
the whole transaction back: nothing it issued is kept, cached or sent. A
vault fault answers the request with a recorded 503 (a realtime frame: the
connection closes 1011), never a bare 500.

### The audit record

`audit.db`, an opt-in Pro feature: one metadata-only row per request
(`detections`/`rehydrations` are stored as JSON text columns):

```json
{"id": 1, "ts": "2026-07-14T11:57:05+00:00", "session": "default",
 "provider": "anthropic", "method": "POST", "path": "/v1/messages",
 "status": 200, "duration_ms": 19.8, "streamed": 0,
 "detections":   {"EMAIL": 1, "AWS_KEY": 1},
 "rehydrations": {"EMAIL": 1, "AWS_KEY": 1},
 "chain_hash": "7e5efb5c…e04abf5b", "user": null, "warned": null}
```

Note what is absent from the audit row: no values, no placeholder ids
(only detector types and counts), no headers, no bodies, no query
strings. That is why the audit row is safe to copy off the machine —
the S3/GCS/Azure sinks and the OTel export ship exactly these rows —
while the vault row is the secret store and is never exported.

> **Tamper-evident is not zero-loss.** The optional audit hash chain
> (`[audit] tamper_evident`) links each row to its predecessor with an
> HMAC (the `chain_hash` above), so `llm-redact audit verify` detects
> any later alteration or deletion of *stored* rows. It does not
> guarantee completeness: the audit trail is deliberately fail-open —
> a write error, a full disk, or an unreachable backup sink warns and
> drops rather than blocking traffic, so under fault a request can be
> proxied without leaving an audit row, and the chain stays valid
> across that gap. Losing an audit row is acceptable by design; losing
> a vault row is not (the vault uses stricter write durability for
> exactly that reason). Every drop is counted and surfaced —
> `rows_dropped` in `/status`, doctor, and the shipped Prometheus
> alerts — so an incomplete trail is always visible. If your compliance
> regime requires a guaranteed-complete audit record, that is what
> `[audit] required = true` (Pro, opt-in) provides: a write-ahead START
> row is durably committed *before* any upstream contact and a request
> that cannot be recorded is refused with a 503 — "no audit row, no
> service", at the explicit cost that audit storage joins the
> availability path.

## Persistence, mangled tokens, and size limits

- **Persistent vault** (`[vault] backend = "sqlite"`, no subscription needed): token
  mappings survive proxy restarts, so a conversation started before a restart
  keeps working and provider prompt caches stay coherent. The database holds the
  real secret values — it is created `0600` in a `0700` directory, but enable it
  only if you accept secrets on disk. Separate workspaces with `--session NAME`.
  Encrypting it at rest (`[vault] encryption = "fernet"`, below) or moving to a
  server RDBMS backend needs Pro.
- **Vault encryption at rest** and the **server RDBMS backends**
  (`[vault] encryption = "fernet"`, `[vault] backend = "postgresql"|…`) are
  **Pro** features that ship in the separately-installed `llm-redact-pro`
  package (**coming soon**) — Fernet at-rest encryption with
  keychain/key-command resolution, and the PostgreSQL/MySQL/Oracle/DB-API
  server vault; its operator guides ship with the package.
- **Fuzzy rehydration** (`[rehydration] fuzzy`, on by default): models
  sometimes rewrite placeholders (`«email_001»`, `«EMAIL-1»`). Recognized
  mangles are restored after a vault check; unknown tokens always pass through
  verbatim. Bracket-swapped forms like `[EMAIL_001]` are deliberately not
  restored — code legitimately contains such identifiers.
- **Request size limit** (`max_body_bytes`, 10 MiB default): oversized
  redactable requests are rejected with 413 before anything goes upstream —
  the proxy never silently forwards unredacted content.
- **Request string limit** (`max_body_strings`, 100,000 default): redaction
  runs string by string on the proxy's event loop, so a body of hundreds of
  thousands of tiny strings would stall every other request. A redactable
  request with more strings than this (JSON string values, form fields,
  file names, uploaded JSONL lines) or more multipart parts is rejected with
  413 the same way; a realtime client frame over it closes the connection
  (1009). A long agent conversation carries a few thousand strings; raise
  it together with `max_body_bytes` for large batch uploads.

## Session isolation

- **Default**: all traffic shares one placeholder namespace
  (`[vault] session_mode = "static"`), so the same value redacts to the
  same token everywhere.
- **Per-conversation isolation** (`session_mode = "per-conversation"`,
  **Pro**): a separate vault namespace per conversation, derived from a
  salted hash of its first user message, with strict
  never-restore-across-sessions behavior. Ships in the `llm-redact-pro`
  package (**coming soon**), with setup and the history-compaction
  limitation documented alongside it.
- **Per-user namespaces** (named users, **Pro**): every request the
  `llm-redact-pro` access gate attributes to a named user resolves inside
  that user's own copy of the session — static and per-conversation mode
  alike, HTTP and realtime — so a placeholder one user's traffic produced
  can never be restored for another user. Wherever these docs say a flow
  uses "the static vault session" (realtime connections, batches, the
  Conversations API, Gemini context caching), read "the user's own copy of
  it" when named users are on; unattributed traffic keeps the configured
  behavior. The live prune (`session_ttl_days`, `POST
  /__llm-redact/sessions/prune`) keeps each user's copy like the static
  session itself.
- **Stored objects** (named users, **Pro**): the core reports the ids of
  objects the provider stores for later reads — uploaded files (OpenAI,
  Azure, custom providers, Anthropic's Files API, a completed OpenAI
  Upload — except one of `purpose` batch, or of no stated purpose: the
  Uploads API's parts are forwarded unread, an opaque byte range can split
  a line, so the requests such a file holds were never checked for the
  stored objects they cite, and it stays an object nobody is recorded
  creating), batches, message batches, stored conversations, Gemini context
  caches, Gemini API Files (`files/<id>`: the multipart upload, a
  resumable upload's finalizing chunk when it is sent through the proxy,
  the metadata-only create and `files:register`; and a finished Gemini
  batch's output file, named on the status its creator reads), video jobs
  (OpenAI/Azure `/videos`), stored chat completions, OpenAI fine-tuning
  jobs (and a job's `result_files`, named when the job is read), vector
  stores, code interpreter containers (created directly, or by a
  Response's code interpreter call: its `container_id`) and the files
  uploaded into one, the files
  a provider tool WROTE for a request — an Anthropic code execution run's
  output files (Files API ids, from a Messages answer, streamed ones read
  from the `content_block_start` that carries the result) and the files
  OpenAI's code interpreter wrote into its container
  (`container_file_citation` annotations and a code interpreter call's
  output files, streamed ones from whichever event names them first) —
  and the long-running jobs read back by name: Gemini API batches
  (generation and async embeddings) and Veo
  operations, Vertex Veo operations (`:predictLongRunning`, polled through
  `:fetchPredictOperation`) and Bedrock async invocations (`POST
  /async-invoke`, polled by ARN) — with the session that created them.
  Only what the answer to THAT request names is reported, each id once
  per answer, and never an id the request body itself carries (a file it
  attached or uploaded into a container, an earlier answer it resends):
  that is an existing object echoed, not one this request created
  (`SessionRouter.record_object_id`, in every vault mode: a router may
  serve unattributed traffic on the static path next to named users, and
  that shared session's objects are no user's). With a sqlite or RDBMS
  vault the record is mirrored in the durable map as an owner record
  (`record_object_session`), bounded APART from the Responses chain rows:
  past its newest 10,000 records, only one whose creating session holds
  no mappings can be dropped, and Responses traffic never pushes one out
  (an RDBMS user that may not `ALTER` the table keeps the old shared
  bound, with a warning — surfaced value-free as `/status`
  `vault.owner_bound_shared`, in `llm-redact status` posture and as a
  `doctor` WARN read from the running proxy). A session router may then
  refuse a request that reaches another namespace's object
  (`object_access_refusal`): the core answers a recorded, provider-shaped
  **403** before the audit START row, redaction, any upstream credential
  and any upstream contact, and is told whether the request reaches its
  provider with a credential the PROXY holds (`identity`): the proxy's
  own cloud identity, or — the routing layer plans before this check — a
  routed upstream that sends an operator key or no key at all
  (`RoutePlan.proxy_credential`; a plan that does not say counts as the
  proxy's). Under such a credential a request no adapter recognizes
  (pass-through) never reaches the check: it is refused with a recorded
  403 before its body is read — the proxy lends a credential it holds
  only to the routes it recognizes. A matched route's parsed JSON body is
  what the check reads, and a body the proxy cannot read — content-encoded
  (415), with more than one Content-Type, not a JSON object — is refused
  before the check by the scanned-body rule wherever it holds (redaction
  applies, or a proxy-held credential is spent). An upload
  (multipart/form-data) on a matched route is read for the check alone
  (`upload_view`), whether or not its route redacts (`detection = false`
  included): the router is asked with the list of what it cites — the
  JSON lines of its file parts (a batch input file's requests, run later
  with the upload's credential) and its form fields, nested along their
  names (`file_ids[]` → `{"file_ids": [...]}`) — before redaction and any
  upstream contact. An upload the check cannot read (outside the
  canonical grammar, a part header without one reading, a transfer
  encoding, a form field that is not UTF-8 text, more JSON than
  `max_body_bytes`, more parts than `max_body_strings`) is refused under
  a proxy-held credential; with `detection = false` a line repeating a
  key is sent re-serialized, exactly as checked. The Gemini API's
  single-request upload (`multipart/related`: the file's JSON metadata,
  then its media) is shown to the router as its metadata OBJECT — what
  Google reads as the create's body, so a file name the upload CHOOSES is
  judged before anything is sent, as the metadata-only JSON create's is.
  The first part (declared `application/json`, or undeclared) is read
  like a JSON body (strict UTF-8, the last
  occurrence of a repeated key — the part then sent re-serialized, exactly
  as checked — at most 128 levels deep); metadata it cannot read is
  refused under a proxy-held credential and, with the client's own key,
  wherever redaction applies (only `detection = false` with the client's
  own key sends it unchecked, as an unparseable JSON body).
  llm-redact-pro refuses every reference to another namespace's stored
  object (in a JSON body, an upload's lines and fields or its metadata)
  under a proxy-held credential (and, for a named user, a reference to an
  object no user is recorded creating), and in every mode anything but a pure read (a
  body-less `GET`/`HEAD`): writes, and requests that carry content of
  their own while citing the object (a chat's file part, a code
  interpreter's `file_ids`, a `previous_response_id` continuation, a
  remix) — their own new values would share placeholder names with the
  object's in the one session the answer is restored from. A router may also mark the session
  it resolved a request to as **sealed** (`SessionRouter.sealed`): the
  core then reads it for rehydration but never writes to it — a request
  that would redact a value there gets a recorded 403 before any upstream
  contact (a realtime connection is refused), so an empty session a
  router hands out for an object of unconfirmed ownership stays empty.
  A realtime connection is one long request whose client frames can
  cite the same objects: a router may check each of them
  (`realtime_frame_refusal`) — every client frame that parses as JSON,
  before it is redacted, numbered or sent, told the adapter, the path,
  whether the connection spends the proxy's own identity and its
  session — and a refusal closes the connection 1008 with the router's
  fixed reason (recorded 403); a check that fails closes it too.
  The router may also observe what the provider sends back on the
  connection (`realtime_server_frame`): every upstream frame that parses
  as JSON, as the provider sent it and before it is restored or sent —
  read-only, a fault contained — so it can record whose Live session a
  resumption handle belongs to before the client can present it. With
  such a router every upstream frame (audio chunks included) is parsed
  once more for it, CPU on the event loop only. A sqlite or RDBMS vault
  offers such a plugin a durable handle map (`record_handle_session` /
  `lookup_handle_session`: a digest of each handle and the session it was
  issued in, bounded per session and in all, deleted with its session on
  every prune), so a handle can be honoured after a restart and on every
  replica sharing the vault — and never once its session was deleted.
  On an OpenAI-shaped listing
  (`GET` files, batches, video jobs, stored chat completions — OpenAI,
  Azure and `/custom/<name>/` alike) the router may name the session
  each listed object was created in (`listing_item_session`, or the
  batched `listing_item_sessions` — one call, one query, per listing):
  the core rebuilds that item from the provider's own bytes and restores
  it there — only when that session exists and holds mappings (the core
  never creates one) and, with a sqlite or RDBMS vault, only when the
  durable map still records the object in exactly that session: a session
  pruned and recreated since holds NEW values under the same token names,
  so the item keeps the provider's placeholders. Items the router names
  nothing for stay as the listing's own session delivers them; if its
  answer fails (an exception, a miscounted batch), every item it failed
  for goes out exactly as the provider sent it — a router that cannot
  answer vouches for nothing. llm-redact-pro reads a named user's listing in
  an empty session and names each item's creator session or an empty
  one — an item nobody is recorded creating included, while named users
  are in play — so each user sees their own items restored and nobody
  else's, whatever session the listing itself is read in. The
  batch list itself is a **chat** route: with one shared namespace it is
  restored like a single batch. A router may also OBSERVE answers
  (`SessionRouter.response_observer`, read-only and provider-neutral):
  for each upstream answer the proxy delivers it is told the adapter,
  method, path, status, content type, the request body as sent upstream,
  the session and whether a proxy-held credential was spent, and may
  return a callable that the proxy then hands each JSON value of the
  answer AS THE PROVIDER SENT IT — before rehydration, so placeholders
  only, never a restored value; each its own parse, so nothing it does
  changes what the client receives: a buffered JSON body once, an SSE
  stream per event, an NDJSON stream per line, an AWS eventstream per
  frame (realtime frames are not observed). A fault there is contained
  like the other bookkeeping (stage `response_observer`); a router
  without the member costs one attribute test per answer.
  llm-redact-pro learns from it which session an OpenAI compaction's
  opaque encrypted item was issued in.
- `compaction_forks` counts only a session first seen by this process
  whose history carries placeholders it cannot own: a persisted session
  resumed after a restart (its vault already holds the tokens) and a
  per-user copy of the static session are not forks. A fork never issues
  one of those placeholders to a new value: the compacted summary is the
  fork's first message, so every request of it carries the summary's
  tokens and the token floor (see "The vault records") numbers new values
  past them.
- The engineering record for why compaction-fork relinking was rejected
  (it cannot meet the never-restore-a-wrong-value bar) stays public in
  [compaction-relink.md](compaction-relink.md).

---

The security-relevant flows — every gate a request passes, with each
policy decision point (PDP) and enforcement point (PEP) mapped to code —
are diagrammed in [security-dataflows.md](security-dataflows.md); what
the proxy defends against, and deliberately does not, is
[threat-model.md](threat-model.md). Mermaid sources for all diagrams
live in [diagrams/](diagrams/); regenerate the PNGs and the animated
SVG/GIF versions with `scripts/render_diagrams.sh`.
