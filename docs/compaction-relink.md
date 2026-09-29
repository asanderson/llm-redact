# History compaction and session relinking: the spike

**The problem.** In `per-conversation` mode, a conversation's vault
namespace is derived from a hash of its first user message. When an
agentic tool compacts history (e.g. `claude` `/compact`), it replaces the
transcript with a synthetic summary — the first message changes, the hash
changes, and the conversation forks into a fresh session. Tokens issued
by the original session («EMAIL_001», …) appear in the new session's
requests but are unknown to its vault, so they pass through verbatim:
the model keeps seeing consistent placeholders, but responses that echo
them are no longer restored for the user. (Pass-through holds only as
long as the fork never issues one of those names itself — the token
floor below is what guarantees that.)

**The bar.** The 1.0 behavior contract is *pass-through-verbatim, never
wrong-value*: an unrestored placeholder is a visible, recoverable
annoyance (`llm-redact lookup` still resolves it); restoring another
conversation's secret is a silent cross-conversation leak. Any relink
mechanism must be **provably correct**, not probabilistically right.

## Designs considered — and why each fails the bar

1. **Token-name lookup across sessions.** Impossible by construction:
   every session numbers its own tokens from 001, so «EMAIL_001» exists
   in essentially all of them. A name identifies nothing.

2. **Token-set fingerprinting.** Match the *set* of token names (and
   their maxima) in the compacted summary against sessions that could
   have issued them. Rich sets are probabilistically strong, but small
   ones ({«EMAIL_001»}) are shared by every session, and "probabilistic"
   is exactly what the bar excludes: a wrong match restores another
   conversation's secrets into this one.

3. **Prefix/anchor hash chains.** Store hashes of every user-message
   prefix and relink when a new conversation extends a known prefix.
   Compaction rewrites the transcript into a *synthetic* summary — no
   prefix survives, so there is never a chain to follow. (This mechanism
   already exists for the Responses API via `response_id`, where the
   provider hands us a durable, exact link.)

4. **Session marker tokens.** Inject a per-session marker («SESSION_xxx»)
   via the system note and relink when it appears in a new conversation's
   first message. Two failures: models do not reliably copy system-note
   content into summaries (so the mechanism silently degrades to the fork
   anyway), and the marker is not binding — a user pasting a snippet of an
   old answer into a genuinely new conversation would relink the whole new
   namespace to the old session, after which hallucinated token names
   («EMAIL_002» the new conversation never saw) restore the *old*
   conversation's values into it. That is the wrong-value leak class.

**Verdict: no relink ships.** The fork stays the deliberate, fail-safe
behavior.

## What ships instead: fork observability

The fork was previously visible only as a generic "new conversation
session" INFO line. A new per-conversation session whose first message
*already contains* placeholder-shaped tokens is the compaction signature
— a genuinely new conversation cannot know the token grammar. The proxy
now counts these (`compaction_forks` in `/status`,
`llm_redact_compaction_forks_total` in metrics, a `llm-redact status`
posture line, and a specific INFO line), so a user wondering why tokens
stopped restoring mid-conversation has the answer in front of them,
along with the recovery path: `llm-redact lookup «TOKEN»` resolves any token from any
session, and sqlite-backed vaults keep the original session's mappings
until pruned.

## What also ships: the token floor (a collision-free fork)

"Pass-through-verbatim" used to hold only until the fork issued a number
of its own. The fork's session is new and empty, so it numbers from 001:
a summary quoting «EMAIL_001» (the original session's alice@…) plus a new
turn mentioning bob@… would redact bob to «EMAIL_001» as well. The
upstream then read one token with two meanings, and every later echo of
«EMAIL_001» restored bob — including where the model meant alice. That
is the wrong-value class, and it was not specific to forks: any request
carrying tokens its session did not issue (an answer pasted into another
conversation, a token pasted from another proxy or user into the static
session, a router's orphan or foreign session) could collide the same way.

The fix is a floor on NEW numbers. Before a request is redacted, the proxy
takes, per type, the highest placeholder number the request already
carries — in every form the rehydrator could restore: canonical
(`«EMAIL_003»`), fuzzy-mangled (`«email-2»`, `« EMAIL_0003 »`), and with
JSON-escaped guillemets inside JSON-source strings (a tool call's
`arguments`). A value the session has not mapped yet is numbered
`max(MAX(n), floor) + 1`, so it can never take a name the request
already holds; a value the session already mapped keeps its token. In the
example, bob becomes «EMAIL_002»: «EMAIL_001» stays unissued in the fork,
its echo still passes through verbatim, and «EMAIL_002» restores bob.
Numbers skipped this way are a gap, never a reuse (the density invariant
becomes "dense except where a floor asked"); a normal session, whose
history carries only its own tokens, has a floor at or below its own
maximum and is unaffected.

Where the floor is read from, per request:

- **JSON bodies**: every key and string of the decoded body — including
  fields the redactor never rewrites (structural scalars, MCP blocks) —
  behind a byte gate that skips the walk when the body cannot hold a
  guillemet in any encoding (no `0xAB` byte, no `\u00` escape, no NUL,
  no `%AB`).
- **Nested encodings an adapter decodes**: the Bedrock CountTokens
  `input.invokeModel.body` blob is decoded first and its tokens join the
  floor before any of its values is numbered.
- **Multipart uploads** (OpenAI/Azure files, image and video prompts):
  the whole upload, read the way the part loop reads it — every file name
  (`filename*` decoded), every JSONL line as the JSON it parses to (or as
  text), every other non-media part as text — before any part is redacted,
  so a token in a later line bounds an earlier line's values.
- **Realtime connections**: a running floor per connection, raised by
  every frame the client sends (the provider holds the conversation; the
  client never resends history).

The floor ends at 999999999 (nine digits — the fuzzy grammar's reach and
a 32-bit column): a request carrying a token numbered exactly that, plus a
new value of that type, is refused with a 400 naming the type — never
numbered past the limit or onto the token.

**Within a fork the guarantee holds for the fork's whole life**: the
compacted summary IS the fork's first message — its session anchor — so
every request of the forked conversation carries it, and the floor always
covers the original session's tokens.

**The precise residual** — what a per-request floor cannot see:

- **Tokens that never appear in any request of the session**:
  provider-side history the client does not resend — a Responses
  `previous_response_id` chain or stored conversation continued in a
  fresh session, Gemini Live session resumption, and a realtime model's
  own output (a token it invents or echoes). llm-redact-pro's sealed
  sessions cover exactly this case: a session that reaches content the
  proxy cannot attribute is never written to at all.
- **A number the session had already issued before the foreign token
  arrived** (pasting another conversation's «EMAIL_001» into a session
  whose own «EMAIL_001» already means someone): the name was taken first;
  the floor governs only new numbers.
- **Other requests sharing the session**: the floor belongs to the
  request (or realtime connection) that carries the token. In a session
  several conversations share — the static session, a named user's copy
  of it, the batch/realtime/Conversations flows — a request that does NOT
  carry the foreign token can still reach its number by ordinary counting.
- **Tokens hidden in encodings the proxy does not decode**: images, audio,
  base64 media, percent-encoding outside multipart file names.
