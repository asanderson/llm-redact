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

The command shows what was refused (kind, detector types, route, requester;
the route and requester come from the request, so every non-printable
character is escaped and either is cut to 120 characters, its length named),
repeats the kind and types on the line it asks you to confirm, and waits for
`allow` typed on the controlling terminal (`/dev/tty`). It never
reads the answer from stdin and refuses without a terminal, so an agent that
merely pipes `allow` into the command does not approve anything. With
llm-redact-pro and a dashboard sign-in (`[auth.dashboard]`), the dashboard
shows the same pending refusals with **Allow once** and **Always allow**
buttons, and your rules with **Revoke**.

**Approval guards against accidents, not against local software.** An agent
that has a shell running as you can approve its own refusal: it can open a
pseudo-terminal and type `allow` into it, or edit the store file, just as it
can edit the configuration. Keep `[overrides] enabled = false` where every
refusal must stay final, or where an agent with a shell works next to the
proxy unsupervised.

**An override forwards the value.** Detection still runs and still decides the
refusal. The override is only asked after that, and a value it lets through
goes upstream **unredacted**, exactly like a `warn`-mode value (an overridden
body or binary part goes out **unscanned**). As with warn mode, that includes
anything the approved value overlaps: a value inside its span that another
rule would have redacted on its own is forwarded with it. Every use is counted and marked
on the request's row, the same way warn mode is.

## What can be overridden

| Refusal | Kind | What an approval matches |
| --- | --- | --- |
| A block-mode value (HTTP 400, realtime close 1008) | `block` | each refused value and its detector type |
| Values found in an inspected binary upload's extracted text | `binary_values` | each refused value and its detector type |
| A verbatim identifier field (fine-tune `suffix`, file ids, …) holding a value llm-redact redacts | `verbatim_field` | each refused value and its detector type |
| A plain-text body (valid UTF-8 that is not a JSON object and that no JSON reader could take for one), with the client's own key | `unscanned_body` | the kind, provider, method and exact path |
| A binary upload part under `binary_uploads = "refuse"`, with the client's own key | `binary_upload` | the kind, provider, method and exact path |

A value approval is an allowlist entry for that exact value and type, stored
as an HMAC. It applies wherever that value would refuse a request (block
mode, a binary upload, a verbatim field). It does not stop a value from being
redacted where the proxy can redact it — a binary file that `[extraction]
convert` replaces by its redacted text included: the values of such a file are
redacted, never put to an approval, never part of a code, and never use a
one-time grant. A deny string where the proxy cannot redact it (a binary
upload, a verbatim field) makes the whole refusal final: no code. A binary
part that reads clean only because an approval let its values through is
counted `overridden` in `inspected_uploads_total`, never `clean`.

A block refusal names the first block-mode value the proxy meets. When a
request holds more than one, approving the first leads to a new refusal (and
code) for the next.

## What can never be overridden

These refusals carry no code:

- deny strings (`[detection] deny`): the operator's always-redact list. In a
  verbatim field or a binary upload they refuse the request, and no approval
  lets them through;
- access control: the stored-object ownership check, sealed sessions, storage
  allowlists, unknown owners, the access gate itself;
- request origin and DNS rebinding (`allowed_hosts`, `allowed_origins`);
- the request target checks: dot or empty segments, a non-origin-form target,
  an unattributable request, a method override;
- a credential the proxy holds (identity auth, a routed operator key): every
  refusal under one, including an unscannable body or binary part and a
  realtime frame on a connection authorized with the proxy's identity. An
  every-time rule approved for a value under the client's own key does not
  pass it there either;
- framing where two readers could disagree: a content coding (415), a
  repeated `Content-Type`, JSON nested too deep, multipart outside the
  canonical form or on a route that does not scan it, an unreadable part
  header, a non-UTF-8 form field;
- a body some reader could still take for a request: invalid UTF-8, a body
  whose first character past spaces, byte-order marks and control characters
  is `{` or `[` (JSON with a trailing byte, a top-level array), or a body sent
  as multipart or form data. A lenient upstream reads JSON with one trailing
  byte as an ordinary chat request, so an approval of a plain-text body never
  reaches one;
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
  in as themselves (`[auth.dashboard]`), so their refusal says `to allow:
  Allow once | Always allow under Refusal overrides in the llm-redact
  dashboard` instead of naming the CLI. A user the access gate does not let
  approve (llm-redact-pro: anyone who cannot sign in to the dashboard, which
  admits administrators) gets no code and no hint, since they could not act
  on it. The CLI, as the local operator, refuses to approve a named user's
  refusal. It can still list and revoke anything. Records are kept under the
  user's stable id when the access gate supplies one (llm-redact-pro: the
  user's namespace, shown as `id:…` by `override list`), so a renamed user
  keeps their approvals and someone who later takes their old name inherits
  none. The gate is asked about a user (can they approve, what is their id)
  only once a refusal of theirs is being decided, never per request.

The endpoints behind the dashboard buttons are core:

- `GET /__llm-redact/overrides` returns the admitted requester's entries.
- `POST /__llm-redact/overrides/approve` takes `{"id": "p7", "scope": "once"|"always"}`.
- `POST /__llm-redact/overrides/revoke` takes `{"id": "r12"}`.

The POSTs sit behind the same guard chain as the configuration editor
(Host, Origin, and a CSRF token that only the llm-redact-pro dashboard page
hands out), and are served only to a requester a person signed in to the
dashboard IN A BROWSER: the access gate must say the dashboard connection
rests on its browser sign-in (its optional `browser_signed_in` member;
llm-redact-pro: the web session cookie). An API key, a per-user key, a bearer
token or a client certificate the gate also admits to the dashboard is no
sign-in: an agent can hold one. Without a browser sign-in any local client
can fetch the CSRF token, so it proves no person: the POSTs answer 403 and
point to the CLI, and the listing says `"can_approve": false`.

## Lifetimes and storage

- A code is single-use and expires after `ttl_minutes` (default 15). At most
  256 pending codes are kept per requester (4096 in all); the oldest are
  dropped first, so one requester retrying a refused request never drops
  another's codes.
- `--once` grants the next matching request of the same requester, within
  `ttl_minutes` of the approval. It is consumed atomically as that request
  passes, so of two parallel requests exactly one uses it. A request that
  is then refused before it reaches the upstream (an `[audit] required`
  failure, no upstream configured, the cloud authorizer, a routed budget
  refusal) hands the grant back for the next one. A realtime approval
  applies to the next connection's matching frame.
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
  A request passing on an every-time rule only reads it: its use is counted
  in memory and written at most once a minute and at shutdown, so another
  process's `override list` can show a use count up to a minute late. Only
  a one-time grant needs a write as its request passes; if the store is
  busy (another process holds its lock for more than 0.2 s) the request is
  refused saying the use could not be recorded.

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
  `"always"`. Only the kind of use is recorded, never what passed. A request
  refused after it passed (no upstream configured, the upstream authorizer, a
  routed budget refusal) is not marked: nothing went out on the override, and
  a one-time grant it used is handed back. With `[audit] required`, the
  write-ahead START row written right before the send already carries the
  marker — an uploaded file's too.
- `/__llm-redact/status` has an `overrides` block with counts: `pending`,
  `once`, `always`, and `used_total` by `once`/`always`.
- `/__llm-redact/metrics` has `llm_redact_overrides_used_total{kind="once"|"always"}`.
- `llm-redact status` prints a posture line while every-time rules exist or
  overrides were used.
- `llm-redact doctor` prints a WARN with the counts while approved overrides
  exist.

None of these show a value, a digest or a code. Log lines name the path and
the kind only.
