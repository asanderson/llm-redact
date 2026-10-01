# Refusal overrides

Some requests are **refused** by detection rather than redacted: a value a
rule in `mode = "block"` matches, values found inside a binary upload the
proxy cannot rewrite, an identifier field the provider uses exactly as sent,
a body the proxy cannot read. When the refused data is yours and you want it
sent anyway, the refusal message carries a short single-use code:

```
llm-redact: request blocked; a EMAIL value was detected and this rule is
configured with mode = "block"; to allow: llm-redact override 7K3M9QX2HD4P --once | --always
```

A **person** approves it, on the terminal:

```bash
llm-redact override 7K3M9QX2HD4P --once     # the next such request passes once
llm-redact override 7K3M9QX2HD4P --always   # these exact values pass every time
llm-redact override list [--json]           # pending codes and live rules (no values)
llm-redact override revoke r12              # drop a rule (r…) or a pending code (p…)
```

The command shows what was refused (kind, detector types, route, requester)
and waits for `allow` typed on the controlling terminal (`/dev/tty`). It never
reads the answer from stdin and refuses without a terminal, so an agent that
runs the command cannot approve its own refusal. With llm-redact-pro, the
dashboard shows the same pending refusals with **Allow once** and **Always
allow** buttons, and your rules with **Revoke**.

**An override forwards the value.** Detection still runs and still decides the
refusal. The override is only asked after that, and a value it lets through
goes upstream **unredacted**, exactly like a `warn`-mode value (an overridden
body or binary part goes out **unscanned**). Every use is counted and marked
on the request's row, the same way warn mode is.

## What can be overridden

| Refusal | Kind | What an approval matches |
| --- | --- | --- |
| A block-mode value (HTTP 400, realtime close 1008) | `block` | each refused value and its detector type |
| Values found in an inspected binary upload's extracted text | `binary_values` | each refused value and its detector type |
| A verbatim identifier field (fine-tune `suffix`, file ids, …) holding a value llm-redact redacts | `verbatim_field` | each refused value and its detector type |
| A body that is not a JSON object (plain text, a top-level array or scalar, invalid UTF-8), with the client's own key | `unscanned_body` | the kind, provider, method and exact path |
| A binary upload part under `binary_uploads = "refuse"`, with the client's own key | `binary_upload` | the kind, provider, method and exact path |

A value approval is an allowlist entry for that exact value and type, stored
as an HMAC. It applies wherever that value would refuse a request (block
mode, a binary upload, a verbatim field). It does not stop a value from being
redacted where the proxy can redact it. Deny strings are values like any
other: in a verbatim field or a binary upload they refuse, and an approval
lets them through.

A block refusal names the first block-mode value the proxy meets. When a
request holds more than one, approving the first leads to a new refusal (and
code) for the next.

## What can never be overridden

These refusals carry no code:

- access control: the stored-object ownership check, sealed sessions, storage
  allowlists, unknown owners, the access gate itself;
- request origin and DNS rebinding (`allowed_hosts`, `allowed_origins`);
- the request target checks: dot or empty segments, a non-origin-form target,
  an unattributable request, a method override;
- a credential the proxy holds (identity auth, a routed operator key): every
  refusal under one, including an unscannable body or binary part;
- framing where two readers could disagree: a content coding (415), a
  repeated `Content-Type`, JSON nested too deep, multipart outside the
  canonical form or on a route that does not scan it, an unreadable part
  header, a non-UTF-8 form field;
- a non-JSON body while a stored-object check is active (llm-redact-pro named
  users): that check reads the body, so an unread one could cite another
  user's stored object;
- size and cost caps (`max_body_bytes`, `max_body_strings`, 413) and vault
  faults (503).

## Who approves

The requester approves their own refusal, and only their own requests use the
approval:

- **No access gate** (a single local user): the requester is the local
  operator. The CLI works on the override store file directly (like `lookup`),
  so run it where the proxy's data directory is.
- **llm-redact-pro named users**: the requester is the admitted user. Their
  refusal codes, one-time grants and every-time rules belong to them. Another
  user can neither use nor approve them. They approve in the dashboard, signed
  in as themselves (`[auth.dashboard]`). The CLI, as the local operator,
  refuses to approve a named user's refusal. It can still list and revoke
  anything.

The endpoints behind the dashboard buttons are core:

- `GET /__llm-redact/overrides` returns the admitted requester's entries.
- `POST /__llm-redact/overrides/approve` takes `{"id": "p7", "scope": "once"|"always"}`.
- `POST /__llm-redact/overrides/revoke` takes `{"id": "r12"}`.

The POSTs sit behind the same guard chain as the configuration editor
(Host, Origin, and a CSRF token that only the llm-redact-pro dashboard page
hands out).

This is a guard against an agent approving by accident, not a security
boundary against local software. Any process running as the operator's user
can edit the store file, just as it can edit the configuration. Keep
`[overrides] enabled = false` where every refusal must stay final.

## Lifetimes and storage

- A code is single-use and expires after `ttl_minutes` (default 15). At most
  256 pending codes are kept; the oldest are dropped first.
- `--once` grants the next matching request of the same requester, within
  `ttl_minutes` of the approval. It is consumed atomically as that request
  passes, so of two parallel requests exactly one uses it. A realtime
  approval applies to the next connection's matching frame.
- `--always` lasts until `llm-redact override revoke`.
- The store is `$XDG_DATA_HOME/llm-redact/overrides.db` (0600, its directory
  0700), or `[overrides] path`.
  - It holds no values: each refused value is an HMAC-SHA256 under a random
    per-install key kept in the same file. Someone who can read the file
    can therefore test guesses of a short value, such as a national ID
    number, against it. The file is as private as the vault, which holds
    the values themselves.
  - It holds no codes, only their SHA-256 hashes.
  - The file is created with the first code.
- The running proxy reads the store only when detection is deciding a
  refusal, so an approval applies to the very next request, with no reload.

```toml
[overrides]
enabled = true      # false: no codes, every refusal is final
ttl_minutes = 15    # 1..1440
# path = "/var/lib/llm-redact/overrides.db"
```

`[overrides]` is restart-only.

## Where overrides show up

- The error message of an overridable refusal carries the code (a realtime
  close reason carries it too, shortened to fit 123 bytes).
- Each request that passed on an override is marked on its `/__llm-redact/recent`
  row, its live-events row and its audit row: `"override": "once"` or
  `"always"`. Only the kind of use is recorded, never what passed.
- `/__llm-redact/status` has an `overrides` block with counts: `pending`,
  `once`, `always`, and `used_total` by `once`/`always`.
- `/__llm-redact/metrics` has `llm_redact_overrides_used_total{kind="once"|"always"}`.
- `llm-redact status` prints a posture line while every-time rules exist or
  overrides were used.
- `llm-redact doctor` prints a WARN with the counts while approved overrides
  exist.

None of these show a value, a digest or a code. Log lines name the path and
the kind only.
