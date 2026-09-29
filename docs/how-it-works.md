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
Bedrock's binary eventstream, and realtime WebSocket deltas):

![Sequence diagram of streaming rehydration reassembling a placeholder split across two SSE deltas](diagrams/sequence-streaming.svg)

*Animated. Static [PNG](diagrams/sequence-streaming.png) · [GIF](diagrams/sequence-streaming.gif) · [Mermaid source](diagrams/sequence-streaming.mmd).*

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
  "system": "Some values in this conversation have been replaced with privacy tokens of the form «TYPE_NNN» (for example «EMAIL_001»). Treat each token as an opaque identifier for the real value and reproduce every token exactly, character for character, whenever you refer to it."
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
  objects the provider stores for later reads — uploaded files, batches,
  message batches, stored conversations, Gemini context caches, video
  jobs and stored chat completions — with the session that created them
  (`SessionRouter.record_object_id`, in every vault mode: a router may
  serve unattributed traffic on the static path next to named users, and
  that shared session's objects are no user's). A session router may then
  refuse a request that reaches another namespace's object
  (`object_access_refusal`): the core answers a recorded, provider-shaped
  **403** before the audit START row, redaction, any upstream credential
  and any upstream contact, and is told whether the provider is
  authorized with the proxy's own cloud identity. llm-redact-pro refuses
  every such reference under identity auth, and in every mode anything
  but a pure read (a body-less `GET`/`HEAD`): writes, and requests that
  carry content of their own while citing the object (a chat's file
  part, a `previous_response_id` continuation, a remix) — their own new
  values would share placeholder names with the object's in the one
  session the answer is restored from. A router may also mark the session
  it resolved a request to as **sealed** (`SessionRouter.sealed`): the
  core then reads it for rehydration but never writes to it — a request
  that would redact a value there gets a recorded 403 before any upstream
  contact (a realtime connection is refused), so an empty session a
  router hands out for an object of unconfirmed ownership stays empty.
  On an OpenAI-shaped listing
  (`GET` files, batches, video jobs, stored chat completions — OpenAI,
  Azure and `/custom/<name>/` alike) the router may name the session
  each listed object was created in (`listing_item_session`): the core
  rebuilds that item from the provider's own bytes and restores it there
  (a session that does not exist, or holds nothing, restores nothing —
  the item keeps the provider's placeholders; the core never creates
  one). Items the router names nothing for stay as the listing's own
  session delivers them. llm-redact-pro reads a named user's listing in
  an empty session and names each item's creator session or an empty
  one, so each user sees their own items restored and nobody else's. The
  batch list itself is a **chat** route: with one shared namespace it is
  restored like a single batch.
- `compaction_forks` counts only a session first seen by this process
  whose history carries placeholders it cannot own: a persisted session
  resumed after a restart (its vault already holds the tokens) and a
  per-user copy of the static session are not forks.
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
