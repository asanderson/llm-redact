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

![Sequence diagram of a non-streaming request: detect, issue a vault token, forward placeholders, reverse-lookup on the response](diagrams/sequence-chat.png)

Streaming is the hard case — a placeholder can be split across chunk
boundaries. The rehydrator holds back only a viable token prefix and flushes
any leftover before the stream-end event, so streamed output is byte-for-byte
identical to the non-streaming result (the same machinery serves SSE, NDJSON,
Bedrock's binary eventstream, and realtime WebSocket deltas):

![Sequence diagram of streaming rehydration reassembling a placeholder split across two SSE deltas](diagrams/sequence-streaming.png)

Watch the round trip live: the dashboard ([dashboard.md](dashboard.md))
shows detections and restores per request, and an agent with the
slash-command plugins installed can do the same in-tool with
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
- The engineering record for why compaction-fork relinking was rejected
  (it cannot meet the never-restore-a-wrong-value bar) stays public in
  [compaction-relink.md](compaction-relink.md).

---

The security-relevant flows — every gate a request passes, with each
policy decision point (PDP) and enforcement point (PEP) mapped to code —
are diagrammed in [security-dataflows.md](security-dataflows.md); what
the proxy defends against, and deliberately does not, is
[threat-model.md](threat-model.md). Mermaid sources for all diagrams
live in [diagrams/](diagrams/); regenerate the PNGs with
`scripts/render_diagrams.sh`.
