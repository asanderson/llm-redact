# History compaction and session relinking: the spike

**The problem.** In `per-conversation` mode, a conversation's vault
namespace is derived from a hash of its first user message. When an
agentic tool compacts history (e.g. `claude` `/compact`), it replaces the
transcript with a synthetic summary — the first message changes, the hash
changes, and the conversation forks into a fresh session. Tokens issued
by the original session («EMAIL_001», …) appear in the new session's
requests but are unknown to its vault, so they pass through verbatim:
the model keeps seeing consistent placeholders, but responses that echo
them are no longer restored for the user.

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
`llm_redact_compaction_forks_total` in metrics, a dashboard pill, and a
specific INFO line), so a user wondering why tokens stopped restoring
mid-conversation has the answer in front of them, along with the
recovery path: `llm-redact lookup «TOKEN»` resolves any token from any
session, and sqlite-backed vaults keep the original session's mappings
until pruned.
