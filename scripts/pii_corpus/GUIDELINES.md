# Labeling guidelines for the agent-traffic evaluation set

Guideline version **1** (`review.py` records it in every review record;
`freeze` refuses rows reviewed under another version, so a changed rule
means bumping `GUIDELINE_VERSION` and reviewing again).

The set measures what llm-redact must keep out of an LLM provider's hands in
coding-agent traffic. A value is gold when **the proxy should redact it if it
were real**. The proxy cannot tell an invented value from a real one, so a
realistic invented value is labeled exactly as a real one would be.

## The decision per row

`review.py review` shows the row with every span tagged
(`<pii type="TYPE">value</pii>`) and a list of the tagged values.

- **Accept** (`a`) when every personal value is tagged, with the right type
  and exact boundaries, and nothing else is tagged.
- **Edit** (`e`) when a tag is missing, wrong or misplaced: fix the tags in
  the editor (add, remove, retype, move boundaries; you may also fix the
  text). The edit is grounded again: tags must be well formed, values
  unpadded and on one line (an address may span lines), one value one type.
  Untagged whole-token repeats of a tagged value are tagged for you.
- **Reject** (`r`) when the row cannot be fixed with a few tag edits:
  - it is not a plausible coding-agent artifact (prose about the task,
    instructions to the model, a refusal, an explanation around the
    artifact, truncated or garbled output);
  - a value looks REAL rather than invented (a public figure, a real
    company's headquarters address, a well-known phone number or account) —
    the teacher can reproduce memorized data, and the set must hold none;
  - it is a hard negative (prompt id ending `.negative`, shown with "no
    spans") that contains personal data: reject it rather than tagging it,
    so negatives stay negatives;
  - it near-duplicates a row you already accepted.
- **Skip** (`s`) to decide later; **quit** (`q`) to stop (a later run
  resumes with the undecided rows).

## What each type covers

| Type | Tag | Do not tag |
|---|---|---|
| `PERSON` | A person's name, full or partial, wherever it appears: prose, JSON values, test fixtures, seed data, commit authors and trailers, log lines. Titles stay outside the span (`Dr. <pii>Ann Lee</pii>`). A name split over fields (`"first": "Ann", "last": "Lee"`) is two spans. | Names inside code identifiers (`AnnLeeFactory`, `ann_lee_fixture`), tool, library and product names (Jenkins, Hudson, Django), bot accounts (`dependabot[bot]`), company names, fictional characters used as product names. |
| `ADDRESS` | A street address: number and street, with unit, city, region and postcode when they are written contiguously with it (one span, may span lines). | A city, region, country or postcode on its own; IP addresses and URLs (other types or none). |
| `DATE_OF_BIRTH` | A date given as someone's birth date (`dob`, `birth_date`, "born on"). | Any other date: timestamps, release dates, due dates, ages. |
| `PASSPORT` | The passport number only. | The label ("Passport No."), the issuing country. |
| `DRIVER_LICENSE` | The licence number only. | The label, the issuing state. |
| `USERNAME` | A person's login name or handle (`annlee90`, `@annlee`, the user part of a home path such as `/home/annlee`). | Service and system accounts (`postgres`, `root`, `www-data`, `ci-bot`), role names, team names. |
| `ACCOUNT_NUMBER` | A bank or customer account number belonging to a person. | Order, invoice, ticket, request and transaction ids; AWS account ids of a service. |
| `EMAIL` | Every email address of a person, including work addresses and commit-trailer addresses. | — (role addresses such as `noreply@` are tagged too: the proxy redacts every address.) |
| `PHONE` | Every phone number, in any format. | Port numbers, version strings, ids that look numeric. |

Other placeholder types (`SSN`, `CREDIT_CARD`, `IBAN`, `IPV4`, `SECRET`, …)
may be tagged by an edit when a row happens to hold one; the generator does
not ask for them.

## Recurring questions

- **Is a fixture name like "John Doe" in test code `PERSON`?** Yes. The proxy
  cannot know a name is a placeholder, and real customer names end up in
  fixtures and seed files; tagging it costs a model nothing that matters,
  missing a real one is a leak. The same holds for "Jane Doe", "Alice" and
  "Bob" when they stand for people (not for a protocol role inside an
  identifier).
- **Are commit author names `PERSON`?** Yes, in `Author:`,
  `Signed-off-by:` and `Co-authored-by:` lines alike; their addresses are
  `EMAIL`. A GitHub handle there is `USERNAME`.
- **A name inside a URL or path** (`/users/annlee/profile`,
  `https://example.com/~annlee`): tag the name part as `USERNAME` (or
  `PERSON` when it is a written name), not the URL.
- **Placeholders already redacted** (`«PERSON_001»`, `[REDACTED]`, `xxx`):
  never tagged; the generator drops rows holding a guillemet.
- **Boundaries**: the value only — no quotes, field names, trailing
  punctuation or surrounding spaces.

## Privacy of the set

The verified and frozen files hold text: keep them outside every git work
tree (the scripts refuse a path inside one), on an encrypted disk, and
never paste rows into an issue, a chat or a model prompt. The NER bench
reports only counts for this set, and `--dump-errors` on it needs
`--allow-real-data-dump`.
