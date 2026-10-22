# What gets detected

The complete detection surface: the built-in rule families, deny
strings, per-rule modes, allowlists, and the optional NER backends and
models. The full, current rule list with per-rule comments is in
[`config.example.toml`](../config.example.toml); how each rule is
scored and gated is in [assurance.md](assurance.md) and the benchmark
section of the [README](../README.md#benchmark-and-live-validation).

## Encrypted-content passthrough (opt-in)

By default, `encrypted_content` strings are scanned like other content.
Detector collisions can therefore rewrite ciphertext or refuse a request
that carries it. To preserve opaque protocol fields when replaying provider
responses, you can opt into a structural-key exemption:

```toml
[detection]
# WARNING: plaintext under this key can bypass detection and scan capture.
# Enable only after accepting the risk described below; both default false.
# encrypted_content_passthrough = true
# acknowledge_encrypted_content_risk = true
```

Both settings must be `true` to enable it. For a new configuration,
`llm-redact init --encrypted-content-passthrough` displays the warning and
requires explicit interactive acceptance. Headless setup also requires
`--accept-encrypted-content-risk`; `--yes` alone cannot give consent.

The exemption is equivalent to adding `encrypted_content` to the scalar
structural-key skip policy across supported JSON protocol walks. Skipped
values are preserved on replay and bypass regex rules, NER, deny strings,
block/warn policy and scan capture. The exemption is selected by field name alone:
the proxy does not authenticate provider origin or verify encryption
with a key. Plaintext placed under that name can therefore reach the
provider unscanned.

Known caller-data contexts such as `metadata`, tool results and documents
still scan every string, including `encrypted_content`. Objects and arrays
under that key still recurse; outside protected caller-data contexts, a
nested scalar also named `encrypted_content` is skipped. Uploaded DATA
(full-string upload content) keeps its scan. This option does not establish
that a value is valid Fernet or any other encryption format.

While enabled, the coverage opt-out remains visible at startup, in
`/status`, `llm-redact status` and `llm-redact doctor`.

## The built-in rules

Emails, IPv4 and IPv6 addresses (both parser-validated; IPv6 additionally
gated so Python slices and cert-serial hex pairs never fire), credit cards
(Luhn-validated), phone numbers (E.164 and separator-punctuated national
formats — bare digit runs never fire), US SSNs (hyphenated form, invalid
ranges vetoed), Canadian SINs (grouped form, Luhn-checked), UK National
Insurance numbers (HMRC grammar with the invalid-prefix blacklist), Indian
Aadhaar numbers (grouped form, Verhoeff-checked), Australian TFNs (grouped
form, ATO checksum), Spanish DNI/NIE (control letter), French NIR social
security numbers (spaced form, mod-97 key), German Steuer-IDs (spaced
form, ISO 7064 check), Brazilian CPFs (dotted form, dual mod-11), Italian
codici fiscali (mod-26 check letter), Swiss AHV (EAN-13 check), Swedish
personnummer (Luhn), Belgian Rijksregisternummer (mod-97), Finnish HETU
(mod-31 check character), UK NHS numbers (spaced form, mod-11), Norwegian
fødselsnummer (double mod-11), Korean RRNs (hyphenated form, mod-11),
Chinese Resident IDs (solid 18-char GB 11643 form: province gate + real
calendar date + MOD 11-2 check char),
Singapore NRIC/FIN (checksum letter), Japanese My Numbers (grouped 4-4-4
form, ordinance mod-11 check), Thai Citizen IDs (dashed display form,
mod-11), Irish PPS numbers (mod-23 check letter), and Mexican CURPs
(18-char grammar, state gate, real date, mod-10 check) — every
national-id rule matches its
grouped or signed display form and is checksum-validated, so bare digit
runs never fire. IBANs
(mod-97 checksum; the solid electronic form, and the ISO 13616 paper format
of the SWIFT registry's countries), cryptocurrency wallet addresses (Ethereum with the
EIP-55 checksum, Bitcoin base58check, and bech32/taproot), URL-embedded
passwords
(`postgres://user:pass@host` — only the password is redacted), AWS
access/secret keys, GitHub/GitLab/Bitbucket/Atlassian/Databricks tokens,
Anthropic/OpenAI/xAI/Perplexity/Slack API keys, Google OAuth client
secrets, Sentry tokens, agent-stack keys (Tavily, Firecrawl, NVIDIA NIM,
Cerebras, Langfuse, Figma), PEM and PGP/GPG private-key blocks, and a
keyword-context + entropy rule for generic secrets (`password = "..."`
etc.).

Detection is regex-based for proxy-hot-path latency; the detector
interface is pluggable and the NER backends below ride it. Add your own
rules and allowlists in the config file — global exact values and
regexes, or scoped to a single placeholder type
(`[detection.allowlist_by_type] EMAIL = ["support@corp.example"]`).
Custom rules (`[[detection.custom_rules]]`) can also name a built-in
validator (`luhn`, `mod97`, `verhoeff`, `jwt`, `entropy`) so a loose
pattern only fires on checksum-valid matches. A match of no characters —
where a pattern that can match the empty string (`\b`, a lookahead such
as `(?=acme)`, `x*`) matches nothing — is no value and is ignored, from
any detector.

### Values next to punctuation

A value may end a sentence. `belgian_nn` and `swiss_ahv` find a number
followed by a run of dots, and `ipv6` an address followed by a run of dots
or one `:`, when one of these follows that run: whitespace, a closing quote
or bracket (`` " ' ` ) ] } > » ” ’ `` and the German and French `“ ‘ « ‹ ›`),
the `*` of Markdown emphasis, the `<` of an HTML tag, a backslash (a JSON
string's escape: a tool call's arguments are redacted as JSON source, where
a line break after the `.` is written `\n`), or the end of the text
(`AHV 756.9217.0769.85.`, `2001:db8::1: unreachable`, `(2001:db8::.)`,
`**AHV 756.9217.0769.85.**`); the punctuation stays outside the redacted
span. A value glued into a longer token is not one: `756.9217.0769.85.3`,
`93.05.18-223.61..x` and `2001:db8::1:x` are not matched.
`planetscale_token` takes the dots after a token into it, as it did while
the token's body and the dots were at most 64 characters, now also when
they run past that and one of the same characters follows them (not when
another PlanetScale token starts in the first 64: that one is read on its
own).

The `iban` rule reads the solid form as before (the length checked for 19
common countries, any other country code on the checksum alone) and the
paper format of the 89 countries of the SWIFT IBAN registry, each at its
registry length: groups of four after the check digits, the last of one to
four, separated by one space, no-break space or narrow no-break space
(`GB82 WEST 1234 5698 7654 32`). The checksum runs on the compacted value;
the redacted span is the text as written. IBANs of countries outside the
registry are read in the solid form only (Guernsey, Jersey, the Isle of
Man and the French overseas departments use GB and FR IBANs, which are
read in both forms).

What these readings find — a number or an IPv6 address before a run of dots
(or `:`), a PlanetScale token whose dots take it past 64 characters, an IBAN
in the paper format — no release before 1.12.1 read, right next to text
another value may hold. So none of them wins an overlap: each is joined with
the values it overlaps ([below](#values-joined-with-the-ones-they-overlap)),
and a value inside one keeps its own mode (a generic secret inside a
PlanetScale token's run in block mode refuses the request; with `iban` in
warn mode a card number inside a spaced IBAN is still redacted). A value
read as 1.12.0 read it (a number or address not followed by a run of dots
or `:`, a PlanetScale token of at most 64 characters with its dots, an IBAN
in the solid form) resolves its overlaps as it did.

Known residuals:

- A paper-format IBAN is not read when one of `-`, `.`, `:`, `@`, `%` or
  `+`, a separator and a digit, or a separator and a one-letter word `A`
  to `D` (the last letter of a National Insurance number) follows it
  (another rule's value could run on from inside it), except a run of dots
  or one `:` followed by one of the characters above: in
  `GB82 WEST 1234 5698 7654 32-x` the IBAN goes upstream as sent.
- Upper-case hex printed in groups of four holds runs shaped like a
  registry country's paper IBAN: one random `gpg --fingerprint` line in
  about 9,300, and one random `xxd -u` line in about 7,500, has a run that
  passes the checksum and is redacted as an IBAN (twenty million random
  lines of each).
- An allowlist entry is an exact value: one in the solid form does not
  cover the same IBAN in the paper format. List both, or use
  `allowlist_patterns`.
- A sentence's final `.` after a PlanetScale token or a generic secret is
  redacted with it (it may be the value's own last character).

### A card next to another digit group

The `credit_card` rule reads 13 to 19 digits, separated by single spaces
or hyphens, and checks them with the Luhn algorithm. Up to 1.12.1 a card
written next to another digit group (a quantity or a year before it, a
CVV or an expiry after it) was read together with that group, failed the
check, and went upstream as sent: `Order 12 4111 1111 1111 1111`,
`4111 1111 1111 1111 123`, `4111 1111 1111 1111 12 27`. Since 1.13.0,
when the check fails, the rule looks for a card inside what it read: a
stretch from the start of a group to the end of one, 13 to 19 digits,
written in a card's own grouping (solid; groups of four, the last of one
to four digits; or 4-6-5 and 4-6-4 as American Express and Diners Club
print them; one separator throughout) that passes the check. Of several,
it takes the one of most digits (of equal length, the one starting last).
The stretch may run on past what the rule read first when that stopped at
its 19th digit (`2024 4111 1111 1111 1111`). It must end where the run of
groups ends; before one more group when the card and that group are at
most 19 digits together (`4111 1111 1111 1111 12/27`); or before an
expiry, a month (`01` to `12`) then a year of two or four digits, each
after the card's own separator (either one, the same twice, after a solid
card): `4111 1111 1111 1111 12 27`, `4111-1111-1111-1111-12-2027`. No
other built-in rule's value can then start inside the card and run on past
it (a test tries every built-in pattern from every place inside cards of
every grouping). What the rule found before is found exactly as before,
and no card it finds inside a run overlaps another of its detections.

On 5,000 synthetic negatives (runs of 13 to 40 random digits in groups of
one to four, separated by one space or one hyphen) the rule finds a card
in 13.72% of them, against 13.44% in 1.12.1 (Luhn passes one random number
in ten); the expiry reading adds none there. On 5,000 such runs followed
by an expiry-shaped pair of groups it finds one in 14.58%, against 14.48%
in 1.12.1 and 14.50% without the expiry reading. Looking only for a card
after a leading group it would be 13.54% of the first sample, and without
the grouping requirement 36.96% (26.94% after a leading group only).
`tests/test_credit_card_groups.py` pins the shipped figures.

Known residuals:

- A card in a longer run is not read when groups follow it that are no
  expiry (`4111 1111 1111 1111 13 27`, `4111 1111 1111 1111 12 27 123`, a
  card right before a SIN or an NHS number), an expiry written with
  another separator (`4111 1111 1111 1111-12-27`), or one group that
  brings the two past 19 digits (`Order 12 4111 1111 1111 1111 5678`):
  another rule's value could start inside the card and run on past it (an
  Aadhaar number's three groups; a US SSN's shape, `111-12-2027` in
  `4111 1111 1111 111-12-2027`).
- Nor when the run of groups is followed by one of `-`, `.`, `:`, `@`,
  `%` or `+`, as a paper IBAN is not, except a run of dots or one `:`
  followed by whitespace, a closing quote or bracket, `*`, `<`, a
  backslash or the end of the text (`12 4111 1111 1111 1111@example.com`:
  an email address takes in the last group), nor when the card is written
  in another grouping or with two separators.
- The Luhn check passes one random number in ten, so a stretch of a long
  run of digit groups that is no card can be redacted as one.

### Values joined with the ones they overlap

Some values are read where earlier releases read none, written right next
to text another value may hold: a card read inside a longer run of digit
groups (1.13.0), an IBAN in the paper format, and a value ending before a
run of dots — a `belgian_nn` or `swiss_ahv` number, an IPv6 address (also
before one `:`), a PlanetScale token whose dots take it past 64 characters
(1.12.1; [Values next to punctuation](#values-next-to-punctuation)). So none
of them takes characters from another value: each is joined with every
value it overlaps, directly or through other overlapping values — a custom
rule's value, an NER span, another rule's value, a deny string. A deny
string is joined the same way with every value it overlaps
([below](#deny-strings-values-that-must-always-be-redacted)). In such a
run of overlapping values:

- A value in block mode refuses the request (an approved override lets
  it through as sent, where nothing redacted covers it).
- Every value that is redacted (redact mode, a deny string) is redacted.
  Values overlapping one another become one placeholder over their union,
  typed after the one starting first (of those, the longest):
  `Order 12 4111 1111 1111 1111 Main Street`, with `1111 Main Street` an
  address a custom rule or an NER model reads, goes upstream as
  `Order 12 «CREDIT_CARD_001»`, and `AHV 756.9217.0769.85. Done`, with
  `85. Done` a custom rule's value, as `AHV «CH_AHV_001»`; the answer gets
  the whole union back. A value covering all the others stays itself (a
  paper IBAN holding a card number is redacted as the IBAN).
- A value in warn mode is forwarded only where no redacted value covers
  it, and counted as a warning also when a redacted value overlaps it, as
  long as part of it lies outside the redacted values: that part goes
  upstream (the rule for every warn-mode value,
  [Per-rule modes](#per-rule-modes-redact-warn-block)). One lying wholly
  inside them (a warn-mode rule's `9217` in
  `AHV 756.9217.0769.85. Done`) sends nothing and is not counted — decided
  once everything is redacted, so one whose rest an access gate's added
  deny string or an `llm` backend finding redacts is not counted either. Of
  values sent that overlap one another, the one the overlap sweep reads
  first is counted, as everywhere — so one value two detectors find (a
  custom rule and an NER model reading the same address) counts once. A
  run refused for a block-mode value counts what it would with the value
  approved.
- Each value is checked against the allowlists before it is joined: an
  allowlisted value joins nothing, and the joined text itself is no value
  an allowlist entry names.
- A joined span is counted once, under its type. Where the proxy cannot
  rewrite the text (a verbatim field, an upload's extracted text), a span
  joining two or more values refuses the request unless every one of them
  is approved ([overrides.md](overrides.md)): the refusal's code names them
  all, one approval of it lets the span through as sent, and each value is
  approved on its own (an every-time approval of the card passes it in
  another span too). A span holding a deny string refuses with no code, as
  a deny string does.
- An `llm` backend's finding is never joined: as everywhere, it keeps only
  what lies outside the other values, so the joined value keeps its own
  placeholder (`Order «PERSON_001» «CREDIT_CARD_001» «PERSON_002» Road`
  when the model reads `12 4111 1111 1111 1111 Main` as a name).

Everywhere else overlaps resolve as before: the value starting first wins,
of those the longest, and under `[detection.modes]` the winner's mode
applies to the characters it takes in.

## Deny strings: values that must always be redacted

Name the strings you never want to leave the machine — project codenames,
internal hostnames, a specific password — and they are redacted with the
highest precedence in the pipeline:

```toml
[detection]
deny = ["project aurora"]        # case-insensitive substring, type DENY

[[detection.deny_strings]]       # per-entry options
value = "Aurora"
case_sensitive = true
type = "PROJECT"                 # -> «PROJECT_001»
```

A deny match always redacts: it is never subject to per-rule modes and
bypasses the allowlist. It is joined with every value it overlaps, directly
or through other overlapping values, like the values
[above](#values-joined-with-the-ones-they-overlap):

- Every value among them that redacts — another deny match, a value in
  redact mode — is redacted with it, values overlapping one another as ONE
  placeholder over their union typed after the one starting first (of
  those, the longest): with `deny = ["acme"]`, `mail bob@acme.example`
  goes upstream as `mail «EMAIL_001»`, and with `deny = ["acme", "mecorp"]`,
  `acmecorp` as `«DENY_001»`. The placeholder, the detection counts and
  the type a refusal names are the first value's: a deny string of type
  `PROJECT` inside an email that starts before it is counted as `EMAIL`.
- A value in block mode among them refuses the request (an approved
  refusal override lets it through as sent where nothing redacted covers
  it) — also one that would lose the overlap to a value redacted in its
  place without the deny string, and one the deny string covers whole.
- A value forwarded as sent (warn mode) goes only where nothing redacted
  covers it.
- An `llm` backend finding is never joined: as everywhere it keeps only what
  lies outside the other values, so where a warn-mode value the join keeps
  covers part of it, that part goes upstream with that value.

Before 1.13.0 a deny match won every overlap, and the rest of what it
overlapped went upstream: `acme` inside an email sent
`bob@«DENY_001».example`, two overlapping deny strings sent the rest of
one of them, and a block-mode value a deny string overlapped was not
refused — its rest was sent. An access gate's added deny strings
(llm-redact-pro roles) are taken in after the configured policy instead, so
they only tighten what it decided. Matching is literal substring ("Auroras"
gets its "Aurora" redacted); each casing variant round-trips back to
exactly what was sent. The llm-redact-pro dashboard's config editor has a
deny-strings table.

## Per-rule modes: redact, warn, block

Every rule defaults to **redact** (substitute a placeholder). Two other
modes are available per rule in `[detection.modes]`:

```toml
[detection.modes]
phone_number = "warn"    # count + log the TYPE only; the VALUE still goes
                         # upstream unredacted — use to trial a noisy rule
private_key = "block"    # reject the whole request with a 400 before
                         # anything is sent upstream (fail closed)
```

Warn-mode hits show up in `/status` (`warnings_total`), Prometheus
(`llm_redact_warnings_total`), and `llm-redact status`, so you can measure a
rule's noise on your real traffic before trusting it with redaction. Be
aware warn is *observation only* — the matched value (and anything a longer
warn-mode match overlaps) is sent to the provider.

Warnings are counted by one rule, which depends only on the values found in
a text and on the characters replaced by placeholders in the end. Every
value found that is forwarded as sent, or would be — a warn-mode value, and a
block-mode value whether or not an override approves it — is a value sent
when at least one of its characters is not replaced. The values sent are
read in the order overlap resolution reads values (the one starting first,
of those the longest), each one overlapping none taken before it is taken,
and each warn-mode value taken is one warning; an `llm` backend finding is
read after the other detectors' values, as overlap resolution reads it
(below). So:

- A warn-mode value part of which is redacted after all counts once, when
  part of it is still sent, and not at all when what is replaced covers it
  whole — whatever redacts that part: a deny string (a configured one, or
  one an access gate adds for its requester: an llm-redact-pro role's
  detection overlay, which only tightens this policy), a value redacted in
  its place that the overlap sweep prefers (one starting earlier, or
  longer), a value it is joined with
  ([above](#values-joined-with-the-ones-they-overlap)), or an `llm` backend
  finding. With `deny = ["acme"]` and `email = "warn"`,
  `bob.private@acme-corp.example` goes upstream as
  `bob.private@«DENY_001»-corp.example` and counts as one warning — as
  when an access gate adds `acme`.
- Of values sent that overlap one another, only those the sweep takes
  count: one value two detectors find counts once, and a warn-mode value
  overlapping one taken before it — a longer warn-mode match, a block-mode
  value — is not counted, however much of it is sent, even all of it. With
  `email = "warn"` and a warn-mode custom rule reading `.example now`,
  `mail bob.private@acme-corp.example now` counts the email alone when
  `acme` is denied; when the whole address is, nothing of the email is
  sent and the custom rule's value is counted instead. The value taken
  before it is counted as what it is: a warn-mode value as a warning, an
  approved block as an override used — and a block-mode value that lost
  the overlap resolution to a value redacted in its place, whose rest goes
  upstream without a refusal (the winner's mode governs), as neither: the
  warn-mode value it hides goes upstream uncounted, whole or in part.
- A deny string counts the same whether it is configured or added by an
  access gate, wherever it replaces the same characters: the values come
  from the configured detectors, the characters from what is replaced in
  the end — never from the way overlap resolution reached it (a configured
  deny string is joined with the values it overlaps, an added one is taken
  in after them).
- An `llm` backend finding is one value: it loses no overlap but is cut
  around the other values ("Additive only",
  [below](#a-model-server-you-run-the-llm-backend)), and counts once
  however many of its pieces are sent; one the model's other finding
  displaces counts when part of it is still sent. It is read after every
  other detector's values sent and counts for what it sends beyond them
  and beyond what is replaced: as it never takes another detector's value's
  place in the redaction, it never takes one's warning either — a crafted
  span starting just before a warn-mode email leaves the email counted.
- A text refused for a block-mode value counts what it would count with
  every refusing value approved, wherever the refusing value stands —
  before the warn-mode values, after them, inside a run of joined values,
  among the values an access gate's added deny strings are taken in with.
  An approved value counts as an override used
  (`llm_redact_overrides_used_total`, the content facts'
  `overridden_types`), never as a warning.
- Text the proxy cannot rewrite (a verbatim field, an upload's extracted
  text) counts as text it rewrites would: against what the redaction would
  replace, whether the text is refused, redacted after all (convert mode)
  or sent as it is — every value that would refuse it approved by a
  refusal override. So approving values never changes what is counted
  there either: a redact-mode value an approval lets through still cuts a
  warn-mode value it overlaps, which counts when part of it lies outside —
  and one it covers whole goes upstream uncounted as a warning, the
  approved value counted as an override used.

Block-mode rejections
return a provider-shaped 400 whose message names the rule type, which
agentic tools surface directly. `llm-redact preview` shows which mode a
given text would trigger; the llm-redact-pro dashboard's config editor has
a three-way selector per rule.

## Person-name detection (optional NER)

Regex catches structured values (emails, keys) but misses most person names,
and has no reliable grammar for street addresses, dates of birth, usernames,
account numbers or passport and driver's-licence numbers. Optional NER
backends close that gap (`[detection.ner]`): names by default (`entities =
["PERSON"]`) and, with a PII-trained model and the types listed in
`entities`, the contextual types too — best effort, measured per model
([ner-landscape.md](ner-landscape.md#verdicts-at-a-glance)), not a guarantee
([threat-model.md](threat-model.md#contextual-values-rules-and-models)):

```bash
# spaCy (default backend, ~tens of MB, ~1-5 ms/string):
uv sync --extra ner
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
# or GLiNER (heavy: torch + transformers; more robust on unusual names,
# supports score_threshold):
uv sync --extra gliner
# or GLiNER2 (Fastino; the same weight class; zero-shot with character
# spans, supports score_threshold):
uv sync --extra gliner2
# or Microsoft Presidio (FOSS PII analyzer: recognizers + checksums +
# context scoring over the same spaCy model; supports score_threshold):
uv sync --extra presidio
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
# or any Hugging Face token-classification model, PII models included
# (pulls torch; supports score_threshold), or Stanza (60+ languages):
uv sync --extra hf        # or: --extra stanza
# or a generative model on a server you run (Ollama, vLLM, llama.cpp's
# server, LM Studio): no extra; see "A model server you run" below.
```

Models for the `gliner`, `gliner2` and `hf` backends come from the Hugging Face
Hub and are fetched once, explicitly: `llm-redact models pull` (the startup
downloads nothing unless `allow_download = true`; "Model sources" below).

Presidio entity types that overlap the built-in regex rules
(`EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `IBAN_CODE`, `CREDIT_CARD`)
are folded into the built-in placeholder names, so a value gets the same
`«EMAIL_NNN»` identity whichever detector finds it.

Presidio's email recognizer asks tldextract whether an address's domain ends
in a public suffix, and tldextract fetches the Public Suffix List from the
internet the first time it is asked — on a request. The presidio backend
replaces tldextract's extractor with one that reads the list snapshot the
tldextract package ships and writes no cache, so no request opens a
connection and an air-gapped host gets the same answers; a tldextract it
cannot set up that way stops the startup.

`[detection.ner] language` sets the analyzer language (wired through
Presidio; the Stanza model to load; implied by the model for spaCy), and
`model` overrides the default model: a spaCy package name for
spacy/presidio (default `en_core_web_sm`), a Hugging Face model id for
GLiNER (default `urchade/gliner_small-v2.1`) or GLiNER2 (default
`fastino/gliner2-base-v1`) or, for the `hf` backend, a
Hugging Face `token-classification` model id (default
`OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` since 1.13.0,
`dslim/bert-base-NER` before; see "The `hf` default's change in 1.13.0"
below); Stanza ignores it. `score_threshold` (a number
greater than 0 and at most 1) drops entities below that confidence on the backends that report one — gliner, gliner2,
presidio and hf; spaCy and Stanza report none, so the key is a config error
when only they are active. Left unset, each of those backends runs at its
model's **catalog default** — the threshold the model catalog records for the
model that backend loads (its configured model, else the backend's default; a
local folder by the model id its `llm-redact-model.json` names), recorded
only where the llm-redact bench measured one: 0.9 for
`fastino/gliner2-privacy-filter-PII-multi` (measured below), and no other
entry has one (the Knowledgator GLiNER-PII cards use 0.3 in their examples;
the bench has not measured it, so it is not adopted) — else 0.5. A set value always wins and applies to every backend, 0.5
included (configuration files written before 1.12.0 by `config show` or the
config editor carry one). `/status` (`detection.ner.backends.<backend>`)
reports each backend's `score_threshold` and its `score_threshold_source`
(`config`, `catalog` or `default`), and `llm-redact doctor` names a catalog
default under `models` and WARNs when a set value overrides a different one; `config show` writes the key only when it is set.

**GLiNER2 (`gliner2`).** Fastino's GLiNER2 is a schema-driven successor of
GLiNER: zero-shot like GLiNER (it is prompted with the same natural-language
prompts, "person", "street address", …), and it reports each entity's
character span, which llm-redact uses as given — a value that occurs twice is
redacted at both places. It runs on the `gliner2` extra (the gliner2 package
with torch, transformers and peft). The default model,
`fastino/gliner2-base-v1` (Apache-2.0, English), is catalogued as not yet
measured by the llm-redact bench. Fastino's PII model,
`fastino/gliner2-privacy-filter-PII-multi` (Apache-2.0; 42 labels; English,
French, Spanish, German, Italian, Portuguese and Dutch; trained on synthetic
texts its card says GPT-5.4 generated), is catalogued with the label
spellings it was trained on ("street_address", "date_of_birth",
"passport_number", "drivers_license_number", …), which llm-redact sends for
the type requests. Its card reports a precision of 0.35–0.37 on the SPY
benchmark; on llm-redact's bench a `score_threshold` of 0.9 keeps its recall
(`PERSON` 0.998, character leak 0.009 for the five contextual types) and
cuts its agent-traffic false positives from 96 to 26 per 50 KB against 0.5,
at 453 ms per 500-character string — "caution", with the numbers in [ner-landscape.md](ner-landscape.md#pii-models-measured-by-the-llm-redact-bench).
0.9 is the model's catalog default, so a configuration naming the model
runs at it without a `score_threshold` of its own (one set wins). The default
trades recall on real text for fewer false positives on agent traffic: on the
published OpenPII data it finds 0.786 of the English names at 0.9 (the
default `hf` model finds 0.989, measured asked for `PERSON` before
word-by-word decoding; see
[ner-landscape.md](ner-landscape.md#fastino-gliner2-pii-gliner2)), so a
deployment that prefers recall sets `score_threshold` lower:

```toml
[detection.ner]
enabled = true
backend = "gliner2"
entities = ["PERSON", "ADDRESS", "DATE_OF_BIRTH", "USERNAME", "ACCOUNT_NUMBER"]

[detection.ner.models]
gliner2 = "fastino/gliner2-privacy-filter-PII-multi"
```

The gliner2 package also contains a client for Fastino's hosted API, which
llm-redact never uses: models run locally. Multiple backends can run
concurrently (`backends = ["spacy", "presidio"]`), and the multilingual
Stanza and Hugging Face `token-classification` backends are available the
same way — the survey behind the lineup is
[ner-landscape.md](ner-landscape.md).

### A model server you run: the `llm` backend

The `llm` backend (opt-in, no extra) asks a generative model on a server you
run to list the contextual values in each scanned string: Ollama's own API
(`api = "ollama"`, the default) or an OpenAI-compatible one (`api =
"openai"`: vLLM, llama.cpp's server, LM Studio). It is configured in its own
table:

```toml
[detection.ner]
enabled = true
backends = ["llm"]            # alone, or beside other backends
entities = ["PERSON", "ADDRESS", "DATE_OF_BIRTH", "USERNAME"]

[detection.ner.llm]
url = "http://127.0.0.1:11434"   # the default: Ollama on loopback
api = "ollama"
model = "MODEL:TAG"            # the model as the server names it: `ollama pull` it first
digest = "<64 hex characters>"   # Ollama only: its digest as /api/tags reports it
timeout_ms = 30000            # one string's time, every window of it
on_error = "block"            # or "skip"
breaker_seconds = 30          # the circuit breaker's cooldown; 0 = off
breaker_failures = 3          # server-down failures in a row that open it (1 to 100)
# api_key_env = "VLLM_API_KEY"   # a server that requires a key: the variable holding it
```

**The server receives every scanned string unredacted** — that is what it is
asked to read, and it includes the secrets in it: the API keys and passwords
the regex rules redact before the provider sees them reach the model server
as written. `url` must therefore be on loopback; a server off loopback is
refused unless `allow_remote = true` and the scheme is `https`. Requests go
exactly to `url`: no proxy or netrc from the environment, no redirect
followed, answers asked for uncompressed (a compressed one is a failure)
and read no further than 1 MiB. `url` may not carry a credential; the
backend sends one only when `api_key_env` is set (below). In a container,
loopback is the container's own: run the model server in the same network
namespace (beside llm-redact in its Kubernetes pod, or with `docker run
--network container:NAME`), or put https in front of it and set
`allow_remote = true` — a plain-http `host.docker.internal` URL is refused.

**`url`, `allow_remote` and `api_key_env` take a restart.** `[detection]`
reloads hot, but where the backend sends every scanned string, and which
environment variable's value it sends there as its key, do not: whoever can
change `[detection]` — a file edit and SIGHUP, the llm-redact-pro config
editor — can neither point the backend at another server nor make it send
another variable's value. A reload keeps the running values of these three
keys, applies the rest of `[detection]` (the model, `timeout_ms`, the
entities, ...), and names each kept key in its "require restart" warning
(`detection.ner.llm.url`, `detection.ner.llm.allow_remote`,
`detection.ner.llm.api_key_env`; the editor's answer lists the same names).
The running values are the ones the proxy started with: no reload moves
them. A reload starts the backend only at the values the proxy started
with, also on a proxy started without it. One that would run it while the
running configuration does not — `"llm"` added to `backends`, NER switched
on with it, its table added, or any of these after a reload stopped it
(NER switched off, or `"llm"` removed from `backends`, with or without its
table) — is applied when the file names the `url`, `allow_remote` and
`api_key_env` the proxy started with: on a proxy started running the
backend, the ones it ran at; on one started without it, those of a
`[detection.ner.llm]` table its startup file held while the backend did
not run (NER off, or `"llm"` not among `[detection.ner] backends`: a
dormant table), or the defaults (Ollama on loopback, no key) when the file
held none. The build
then reads the key's variable and checks the server and its model exactly
as at startup (any failure keeps the running configuration), `/status`
shows the backend, and an open realtime connection is closed 1012 to
reconnect under the new detection policy, as for any `[detection]` change.
A start with any of the three changed is refused (`a reload cannot start
the llm backend at other values than the proxy started with`, naming the
changed keys) and the running configuration kept: kept, the running values
would start it at a destination the file does not name. So is a reload the
kept values make invalid (a changed `api` whose paths the running `url`
already ends with). Each refusal names keys, never a value. Restart the
proxy to apply any of them. The startup and `serve --check` read the file
as written. A running proxy therefore never begins to send scanned text,
or a credential, to a destination its startup configuration does not name.
Whoever can change `[detection]` can switch the backend on at those
values, though — on a proxy started without it, at a dormant table's `url`
and key variable, or at the loopback defaults — and from then on every
scanned string goes to that server
([threat-model.md](threat-model.md#residual-risks)): a dormant table is a
destination a reload can switch on, so leave it out of a configuration
that is not to run the backend. A config editor save that only switches
NER on starts it too when the file's backends hold `"llm"`. A start by
reload is logged once applied, a WARNING naming the server's origin
(`config reload starts the llm NER backend: model server … now receives
every string it scans, unredacted`), and `llm-redact doctor` WARNs about a
dormant table that names a server off loopback or a key variable.

**A server that requires a key** (vLLM started with `--api-key`, a hosted
https endpoint, Ollama behind an authenticating reverse proxy or Ollama's
own hosted API — all read the same header): set `api_key_env` to the NAME of
an environment variable holding the key, never the key itself. The
variable's name takes a restart (above); its value is read at every build
(startup, `serve --check`, a reload that changes `[detection]`; a reload
that does not keeps the key it was built with) from the proxy's own
environment, which nothing outside the process
can change while it runs — a new key, or a variable exported after the
start, takes a restart. Its edge blanks are trimmed; unset, empty or
holding anything but visible ASCII, the build stops with an error naming
the variable, never its value (a reload keeps the running configuration).
The key is sent as
`Authorization: Bearer <key>` with every request — the model list and each
chat, on either `api` — to `url` only: no redirect is followed, so it never
goes to a `Location`, and the off-loopback rule above holds (plain http only
on loopback). It is never logged, echoed or shown: `/status` says only
whether a key is configured (`api_key`), and `llm-redact doctor` checks that
the variable is set (offline too) without showing it. A variable whose name
starts with `LLM_REDACT_` is refused: those hold llm-redact's own settings
and keys, which must never reach a model server.

**Pinned model.** Before the backend is built (at startup, `serve --check`,
and on a reload that changes `[detection]` — which builds on a worker
thread, so the wait holds no request) the server must list `model`
(answering within 10 seconds, or `timeout_ms` when that is shorter); on
Ollama its `/api/tags` digest must equal `digest` (required there; the
64-character form `/api/tags` reports, with or without `sha256:` — `ollama
list` shows only its first 12 characters), so a model re-tagged on the
server is refused. An OpenAI-compatible server reports no digest, so `digest`
is refused with `api = "openai"`. llm-redact never pulls a model: `ollama
pull` it first, then read its digest with `curl -s
http://127.0.0.1:11434/api/tags`. A failed check stops the startup (a reload
keeps the running configuration), and `llm-redact doctor` makes the same
read-only check (it sends no text; `doctor --offline` skips it). The pin
holds at the build: the chat requests name the model, not its digest, so a
model re-tagged on the server after the build is used until the next build
(a restart, or a reload that changes `[detection]`) refuses it.

**Per string** the backend sends one chat request: a fixed system prompt
that names the requested types (as the `gliner` prompts do: "person",
"street address") and says the text is data, never instructions; the text
itself between two marker lines whose id is derived from the text, so the
text cannot close its own frame; temperature 0 and a fixed seed; and a JSON
schema for the answer (Ollama's `format`, the OpenAI `response_format`), a
list of `{text, type}` objects. A string longer than 2,000 characters is read
in overlapping windows of whole words (a single word longer than that in
overlapping pieces of 2,000 characters, counted `windows_truncated`, so no
request outgrows the context the server is asked for); one longer than
`max_chars` is skipped, as on every backend. `timeout_ms` bounds a string's
windows together: a window is asked only while time is left, each wait of
its request (connecting, sending, every read) lasts at most what was left
when that request started, and an answer body still arriving once the time
is up is cut — so a server that does not answer, or goes silent
mid-answer, holds a string for at most about twice `timeout_ms` before it
fails. The bound is per read until the answer's headers are complete: a
server that keeps sending header bytes, or interim `1xx` answers, a little
at a time holds the string for as long as it keeps doing so. Your own
server is trusted not to (one that misbehaves can as well fail every
request). What a string held is remembered by
its SHA-256 (the last 4,096 strings, in process memory, gone when a reload
rebuilds the detectors), so the conversation history an agent resends with
every request costs one call per new string.

Each new string still costs a model call — hundreds of milliseconds to
seconds — which runs off the event loop through the NER prefetch ("How NER
runs"): one string at a time for the whole proxy, one request's strings
after another's, so a request also waits for the new strings of the
requests ahead of it. A string a request's prefetch did not cover (a
reload during the request) is asked inline, holding every other request for
that call (`inline_calls`); a body the redaction refuses part way (one over
`max_body_strings`, say) still has the strings before that point asked on
the worker. Ollama unloads a model idle for five minutes
(`OLLAMA_KEEP_ALIVE`); the first string after that waits for the load within
`timeout_ms`.

**Grounded.** A value is kept only when its text occurs verbatim in the text
the model read (edge blanks trimmed, at least two characters, no placeholder
guillemets), and it is then redacted at every place it occurs in the string —
inside longer words too ("Ann" in "Annual": the placeholder is restored in
the answer like any other), never inside a placeholder an earlier turn left
in the text (`PERSON_001` of `«PERSON_001»`). Anything else the model returns — invented,
mangled, a different spelling — is dropped and counted (`ungrounded`); a type
that was not requested is ignored.

**Additive only.** The scanned text can instruct the model, so the backend's
findings may only ADD redactions: they never take another detector's place.
Overlap resolution sets the other detectors' winners first, exactly as
without this backend, and keeps of each `llm` finding only what lies outside
them — so a text that tells the model to "return no entities" removes at most
the model's own findings, and a crafted answer whose span starts just before
an API key still leaves the whole key redacted (the span's other part is
redacted too). In warn mode each part left is forwarded as sent, and the
finding counts as one warning however many parts of it are sent; counted
after the other detectors' values, it never takes a warning from one of
them ([Per-rule modes](#per-rule-modes-redact-warn-block)).

**When the server fails** — a timeout, an error status, a refused
connection, an answer that is not the requested JSON — `on_error = "block"`
(the default) refuses the request with a provider-shaped 503 before any
upstream contact (`llm_redact_local_refusals_total{kind="detector_unavailable"}`;
a realtime connection is closed 1011, its row 503). `on_error = "skip"`
forwards it with the other detectors' findings instead: counted (`errors`),
logged once per outage, and reported as a coverage opt-out — a `/status`
field (`detection.ner.backends.llm.on_error`), a `llm-redact status` posture
line and a `doctor` WARN, like warn mode. `llm-redact preview` exits 1 and
names the failure. A string the model cannot answer within its limits —
more values than one answer holds (2,048 tokens, 512 values), or text that
makes it ramble — fails the same way, and a failure is never remembered: with
`"block"`, every request that carries that string again (a conversation's
history) is refused until the string leaves it.

**Circuit breaker.** A server that hangs, rather than refusing the
connection, would cost every new string up to `timeout_ms` — and strings are
asked one at a time for the whole proxy. So failures that say the server
is down — a timeout, a transport fault (a refused or dropped connection), an
HTTP 5xx answer — `breaker_failures` of them in a row (default 3, from 1 to
100; 1 opens it at the first), with no answer from the server between them,
open a circuit breaker for `breaker_seconds` (default 30, 0 = off, at most
3600). Below that count each failing string fails on its own (as below) and
the next one asks the server again; any answer from the server — even an
unusable one — starts the count again. Once open, until the cooldown has
passed every string that needs a call fails at once without contacting the
server, handled exactly as a failure
(`on_error = "block"`: its request is refused 503 with the reason `circuit
breaker open`; `"skip"`: it goes on with the other detectors' findings),
and is counted in `errors` and `breaker_short_circuits`. After the cooldown
one string's call goes out as a probe: an answer closes the breaker, a
failure opens it again at once for another `breaker_seconds` — no new run
of failures needed (`breaker_opens` counts each opening). A failure from a
server that answers — an error status below 500, a redirect, an answer that
is not the requested JSON — fails only its own string, never counts toward
an opening and starts the count again (as the probe's answer, even an
unusable one, it closes the breaker: the server is up); so does a timeout
once the server has answered an earlier window of the same string, which is
that long string's own `timeout_ms` running out. A string the backend
already read (the cache) or one with nothing to read needs no call and is
unaffected. `/status` shows the breaker
(`detection.ner.backends.llm.breaker`: `state`, `failures` — the failures in
a row so far — and `threshold`, the failure `reason` and
`retry_in_seconds` while it is open) and `llm-redact status` prints a
posture line while it is open; the breaker belongs to the built backend, so
a reload that rebuilds it starts a closed one with no failures counted. With `"skip"` an open breaker forwards requests without the
backend's findings for the whole cooldown, even if the server recovers
sooner — the coverage gap the posture line and `errors` count. Keep
`timeout_ms` well below what your clients wait for an answer: the string
that opens the breaker still waits it out.

The failures that open the breaker can come from the text: `breaker_failures`
texts in a row that make the model ramble past `timeout_ms`, or that the
server answers with a 5xx, open it for every string, not just theirs — and
the scanned text is not only yours (a web page or file an agent read lands
in a request; one such text resent with every request of a conversation, or
a request carrying several, fails that many in a row). With `"block"` every
request that needs the server is then refused for the cooldown; with
`"skip"` every request goes on without the backend's findings for it, again
after each cooldown while such texts keep coming. Keep `"block"` where that
matters, raise `breaker_failures`, or set `breaker_seconds = 0` (each such
string then fails alone, at a whole `timeout_ms` each).

Like every NER backend it is best effort, measured per model, never a
guarantee ([threat-model.md](threat-model.md#contextual-values-rules-and-models)).
Logs name the model, the server's host and a fixed failure kind, never text.

### Model sources: pinned revisions, downloads and pickle weights

![Flowchart of where NER model weights come from: the model catalog's pins and the Hugging Face Hub feed llm-redact models pull (or a startup with allow_download = true), which fills the local Hugging Face cache; models verify checks the files offline; a model-load policy, when a plugin installs one (llm-redact-pro's [models]), is shown the cached files before anything loads and may refuse the build; once allowed, the load assembles a GLiNER base model into a local folder when the checkpoint needs one, and the startup loads local files only (safetensors or ONNX, never repository code) and reloads never download. Beside it the air-gap path: models pull --to writes portable folders with a SHA-256 manifest, which are carried into the enclave, checked with models verify --dir and loaded with downloads off](diagrams/model-supply-chain.png)

*Static diagram. [Mermaid source](diagrams/model-supply-chain.mmd).*

The `gliner`, `gliner2` and `hf` backends load models from the Hugging Face
Hub. Three `[detection.ner]` keys say where those models may come from:

```toml
[detection.ner]
allow_download = false        # the default
allow_pickle_weights = false  # the default; hf only

[detection.ner.revisions]
hf = "a2360d3f42526fc660ac3b2b2301e1c2d94eba61"   # a full commit id
```

- `revisions` pins a backend's model to one commit: a full 40-character
  lowercase hex commit id per backend, `gliner`, `gliner2` or `hf` (the other
  backends' models are not Hub snapshots and take none). A branch or tag name such as
  `main` is a config error, because it moves. A backend with no entry uses
  the pin llm-redact's model catalog records for the model it loads, when
  there is one: every model in the vetted and caution tables below, the
  defaults (`urchade/gliner_small-v2.1`,
  `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1`, `fastino/gliner2-base-v1`)
  included, is pinned to the commit its `main`
  branch pointed at when its entry was checked (2026-10-05 to 2026-10-07; the
  date is in `llm-redact models list --json`, under `facts`). An entry for a
  backend that is not active is kept and ignored, like a
  `[detection.ner.models]` entry.
- `allow_download` (default `false`) decides whether the proxy's startup
  (`serve`, `serve --check`) may fetch a model's pinned files from the Hub
  into the Hugging Face cache. With `false` every model loads from the
  local cache or a model folder, and a model that is not there stops the
  startup with an error naming it, its revision and `llm-redact models
  pull`. Only the startup ever downloads: a reload (SIGHUP, the dashboard
  editor), the editor's dry run and `llm-redact preview` read local files
  only, so a reload naming a model that is not cached is refused and the
  running configuration is kept. Whenever a build may not download, the
  Hugging Face libraries' own offline switches (`HF_HUB_OFFLINE`,
  `TRANSFORMERS_OFFLINE`) are set before they are first imported, so no
  library code reaches the network either.
- `allow_pickle_weights` (default `false`, `hf` only) permits loading
  `pytorch_model.bin`, a pickle, for an `hf` model that ships no safetensors
  weights. GLiNER loads its `pytorch_model.bin` through torch's
  `weights_only` loader whatever this key says.

`llm-redact doctor` reports these settings under `models` while NER is on,
without loading a model or touching the network: a WARN while
`allow_download` or `allow_pickle_weights` is on and for a model with no pin,
each model's pin and where it comes from, and whether every file its load
reads is in the local Hugging Face cache at that pin (a GLiNER model's base
model included) or in its folder — a FAIL, naming `llm-redact models pull`,
when a model the startup needs is missing while downloads are off.

**How the `hf` backend loads a model.** A Hub model is looked up in the local
Hugging Face cache at its pinned revision (fetched only when
`allow_download = true`), with an explicit list of top-level files: its
`config.json`, safetensors weights and tokenizer files — never the
TensorFlow, Flax, ONNX or `original/` copies a repository may also hold. A
model missing from the cache is a startup error that names the model and the
revision. A local folder is loaded as it is; a revision for it is a config
error, and it never takes the catalog's pin, even when its path reads like a
catalogued model id. A value written as a path (absolute, `./…`, `~/…`, or
with more than one `/`) is always a folder, never a Hub id: when nothing is
there (an empty or unmounted volume) the startup stops with an error naming
`llm-redact models pull --to`. Every load checks that the folder or cached snapshot
holds what it reads — `config.json`, the weights (every shard a weight index
names) and a tokenizer — so an interrupted download counts as a model not in
the cache. A model whose `config.json` or `tokenizer_config.json` names code
to import from its repository (`auto_map`) is refused, and nothing is ever
loaded with `trust_remote_code`. The weights must be safetensors: a model
with only `pytorch_model.bin` is refused unless `allow_pickle_weights = true`,
and a model with both always loads its safetensors. Every model runs in
float32, whatever precision its weights are stored in: a model stored in
bfloat16 runs slower on a CPU, and loaded in float32 it takes twice the
memory of its weight files (size a container or Helm memory limit for
that). The load needs transformers 4.56 or newer (the first whose
pipeline takes the precision); an older one stops the startup with
`[detection.ner] backend = "hf" needs transformers >= 4.56 …`.

**How the `gliner` backend loads a model.** The same way: the GLiNER
checkpoint at its pinned revision, from the local cache unless
`allow_download = true`, with an explicit list of files (`gliner_config.json`,
`model.safetensors`, the tokenizer files, and `pytorch_model.bin` only for a
checkpoint without safetensors — GLiNER loads it with torch's `weights_only`
loader). A checkpoint that ships its own tokenizer and an `encoder_config`
(Knowledgator's) loads from that folder. One that does not (the urchade v2.1
models, the default included) would make GLiNER fetch its base model's
tokenizer and configuration from the Hub at every load, at no fixed
revision; llm-redact instead fetches the base model's configuration and
tokenizer (never its weights) at the revision the model catalog pins and
assembles a self-contained folder under `$XDG_DATA_HOME/llm-redact/models/gliner/`:
links to the checkpoint's weights and the base model's tokenizer, and a
`gliner_config.json` that embeds the base model's configuration as
`encoder_config` and names no absolute path. A base model the catalog does
not pin loads at its newest cached revision, with a startup warning. A base
model that is a local folder (the checkpoint's configuration names a path,
or a folder under the working directory is named like the base model's id)
is read from that folder as it is, with no pin and no warning, like a local
model folder. A configuration naming code to import (`auto_map`) or a model
type transformers does not know is refused.

**How the `gliner2` backend loads a model.** A GLiNER2 checkpoint is
self-contained: its `config.json`, its encoder's configuration
(`encoder_config/config.json`), its tokenizer and `model.safetensors` are
looked up at the pinned revision, from the local cache unless `allow_download
= true` (`pytorch_model.bin` only for a checkpoint without safetensors —
gliner2 loads it with torch's `weights_only` loader). A checkpoint missing
one of the three configuration files is refused, and so is one whose
configuration names code to import (`auto_map`) or whose encoder is a model
type transformers does not know (gliner2 builds the encoder from that
configuration with `trust_remote_code`, so an unknown type could run code).
The model is loaded from that folder alone; nothing is fetched at load time.

**ONNX weights for `gliner`.** `[detection.ner.onnx]` loads a GLiNER model's
ONNX export through onnxruntime (the `gliner` extra lists it itself: gliner
0.2.29 and later no longer install it) instead
of its torch weights: name the file inside the model, and only that file —
never `model.safetensors` or `pytorch_model.bin` — is fetched or read:

```toml
[detection.ner.models]
gliner = "knowledgator/gliner-pii-base-v1.0"

[detection.ner.onnx]
gliner = "onnx/model_quint8.onnx"   # the int8 export (Knowledgator ships model.onnx,
                                    # model_fp16.onnx and model_quint8.onnx)
```

Only `gliner` takes an entry; the value must be a `.onnx` path inside the
model (no wildcard, no `..`). A model that lacks the file is a startup error.

### The model catalog

llm-redact keeps a catalog of the Hugging Face models the `gliner`, `gliner2`
and `hf` backends may load (`src/llm_redact/detection/model_catalog.py`): each model's
license, a one-line statement of facts with a link to its model card and the
date they were checked, the commit it is pinned to, and, for a GLiNER model
that ships no tokenizer, the base model and the commit that base model is
pinned to. The catalog only states facts; llm-redact never refuses a model
for its catalog status (a policy plugin may).

- **vetted**: a known-good choice, pinned to a commit. A model the
  llm-redact bench measures is vetted only when it clears every admission
  bar: an OSI-approved weights license and no known restricted
  training-data lineage, PERSON recall of at least 0.85 and a character
  leak of at most 0.15 on the synthetic corpus, at most one false positive
  per 50 KB of agent-traffic negatives, and a p50 of at most 100 ms per
  500-character string ([CONTRIBUTING.md](CONTRIBUTING.md#adding-an-ner-model-or-backend),
  step 6). The two backend defaults and the `urchade` models are vetted as
  the models users already run, from before the bar existed; none of the
  models the bar has been applied to reached vetted.
- **caution**: configurable and pinned, with the reason shown — for example,
  not yet measured by the llm-redact bench, or measured below a bar (the
  reason then quotes the numbers).
- **restricted**: never suggested. A configured restricted model logs a
  startup WARNING with the catalog's facts:
  `[detection.ner] BACKEND model 'ID' has model catalog status "restricted": …`.

What the catalog says about each running backend's model is in `/status`
(`detection.ner.backends.BACKEND`: `source`, `model_id`, `revision`, `pinned`,
`catalog`, `license`, see "NER coverage counters" below), in `llm-redact
doctor` under `models` (a WARN for a restricted model and for a model nothing
pins, and a FAIL when the installed library is older than a model needs) and
in `llm-redact models list`. A model the catalog does not know loads as
configured; pin it in `[detection.ner.revisions]`. A local folder is looked up
by the model its `llm-redact-model.json` names.

<!-- model-catalog:vetted -->
| Model | Backend | License | Pinned revision | Base model (pinned revision) |
|---|---|---|---|---|
| `dslim/bert-base-NER` | hf | MIT | `d1a3e8f13f8c` | — |
| `urchade/gliner_small-v2.1` | gliner | Apache-2.0 | `4e091416cf7c` | `microsoft/deberta-v3-small` (`a36c739020e0`) |
| `urchade/gliner_medium-v2.1` | gliner | Apache-2.0 | `40ec419335d0` | `microsoft/deberta-v3-base` (`8ccc9b6f3619`) |
| `urchade/gliner_multi-v2.1` | gliner | Apache-2.0 | `443d26d654e0` | `microsoft/mdeberta-v3-base` (`a0484667b223`) |
| `urchade/gliner_multi_pii-v1` | gliner | Apache-2.0 | `1fcf13e85f4e` | `microsoft/mdeberta-v3-base` (`a0484667b223`) |
<!-- /model-catalog -->

The `gliner` default, `urchade/gliner_small-v2.1`, is vetted; the `hf`
default since 1.13.0, `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1`, and
the `gliner2` default, `fastino/gliner2-base-v1`, are listed under caution
below. `dslim/bert-base-NER`, the `hf` default before 1.13.0, stays vetted and
pinned: a configuration that names it keeps loading it. A base model is
pinned where the GLiNER checkpoint ships no tokenizer or encoder
configuration of its own.

Configurable, status caution (their cards do not name the training data):
Knowledgator's GLiNER-PII models — `-edge` and `-base` with the bench numbers
that keep them below the "vetted" bars (see "Choosing a GLiNER model" below),
`-small` and `-large` not yet measured by the llm-redact bench — and the
`gliner2` backend's default, `fastino/gliner2-base-v1`, not yet measured,
and Fastino's PII model `fastino/gliner2-privacy-filter-PII-multi`, measured
below the bars ("GLiNER2" above).
Each ships its tokenizer and encoder configuration, so no base model is
fetched; the Knowledgator `-edge` and `-small` need transformers 4.48 or
newer. Also caution, with their bench numbers: the `hf` PII models trained
on NVIDIA Nemotron-PII (CC BY 4.0), OpenMed-PII Small 44M and
ettin-68m-nemotron-pii, and `openai/privacy-filter` ("PII models for the
`hf` backend" below; the ModernBERT ettin model needs transformers 4.48 or
newer too, the privacy filter 5.6.0):

<!-- model-catalog:caution -->
| Model | Backend | License | Pinned revision | Base model (pinned revision) |
|---|---|---|---|---|
| `knowledgator/gliner-pii-edge-v1.0` | gliner | Apache-2.0 | `9b7f39b0a2da` | `jhu-clsp/ettin-encoder-32m` (—) |
| `knowledgator/gliner-pii-small-v1.0` | gliner | Apache-2.0 | `d21aad5b4a7e` | `jhu-clsp/ettin-encoder-68m` (—) |
| `knowledgator/gliner-pii-base-v1.0` | gliner | Apache-2.0 | `61726e0ad791` | `microsoft/deberta-v3-small` (—) |
| `knowledgator/gliner-pii-large-v1.0` | gliner | Apache-2.0 | `f847f54fbc97` | `microsoft/deberta-v3-large` (—) |
| `fastino/gliner2-base-v1` | gliner2 | Apache-2.0 | `f9634218e535` | `microsoft/deberta-v3-base` (—) |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | hf | Apache-2.0 | `a2360d3f4252` | `microsoft/deberta-v3-small` (—) |
| `kalyan-ks/ettin-68m-nemotron-pii` | hf | MIT | `500262a2aaf9` | `jhu-clsp/ettin-encoder-68m` (—) |
| `openai/privacy-filter` | hf | Apache-2.0 | `7ffa9a043d54` | — |
| `fastino/gliner2-privacy-filter-PII-multi` | gliner2 | Apache-2.0 | `1cb4166094dc` | `microsoft/mdeberta-v3-base` (—) |
<!-- /model-catalog -->

Restricted (a startup WARNING names the facts; no pin):

<!-- model-catalog:restricted -->
| Model | Backend | License | Pinned revision | Base model (pinned revision) |
|---|---|---|---|---|
| `iiiorg/piiranha-v1-detect-personal-information` | hf | CC-BY-NC-ND-4.0 | — | — |
| `Isotonic/deberta-v3-base_finetuned_ai4privacy_v2` | hf | CC-BY-NC-4.0 | — | — |
| `Isotonic/distilbert_finetuned_ai4privacy_v2` | hf | CC-BY-NC-4.0 | — | — |
| `urchade/gliner_base` | gliner | CC-BY-NC-4.0 | — | — |
| `nvidia/gliner-PII` | gliner | LicenseRef-NVIDIA-Open-Model-License | — | — |
| `bigcode/starpii` | hf | LicenseRef-bigcode-starpii-terms-of-use | — | — |
| `ai4privacy/llama-ai4privacy-*` | hf | MIT | — | — |
| `knowledgator/gliner-stream-pii-v1.0` | gliner | Apache-2.0 | — | `Qwen/Qwen3-0.6B` (—) |
| `perplexity-ai/PII-Tracer` | hf | MIT | — | — |
| `OpenMed/privacy-filter-multilingual` | hf | Apache-2.0 | — | — |
| `llm-semantic-router/mmbert32k-pii-detector-merged` | hf | MIT | — | — |
<!-- /model-catalog -->

`*` marks an id prefix: every model whose id starts with it. The reasons, with
their links and check dates, are what the startup warning, `doctor` and
`llm-redact models list --json` print.

### Choosing a GLiNER model

The `gliner` backend loads `urchade/gliner_small-v2.1` unless
`[detection.ner.models] gliner` names another model. Knowledgator's
GLiNER-PII models (`knowledgator/gliner-pii-edge-v1.0`, `-small-`, `-base-`,
`-large-v1.0`, Apache-2.0, developed with Wordcab) are GLiNER checkpoints
trained on PII labels; each ships its own tokenizer and encoder
configuration, so nothing beyond the checkpoint is fetched, and an int8
ONNX export of each can be loaded instead of the PyTorch weights. The
catalog records the prompt each was trained on and llm-redact sends it for
a type request: "name" for `PERSON` (the generic prompt is "person"),
"location address" for `ADDRESS`, "dob" for `DATE_OF_BIRTH`, "passport
number", "driver license", "username" and "account number".
`urchade/gliner_multi_pii-v1` (Apache-2.0, multilingual) is the default
model of Presidio's own GLiNER recognizer; it is in the catalog but not
measured here.

A working configuration (fetch the model once with `llm-redact models
pull`, then start; downloads stay off):

```toml
[detection.ner]
enabled = true
backend = "gliner"
entities = ["PERSON", "ADDRESS", "DATE_OF_BIRTH", "USERNAME", "ACCOUNT_NUMBER"]

[detection.ner.models]
gliner = "knowledgator/gliner-pii-base-v1.0"

[detection.ner.onnx]
gliner = "onnx/model_quint8.onnx"   # the int8 export; leave the table out for the PyTorch weights
```

What the NER bench measured (2026-10-07, one shared 4-core
`Intel(R) Xeon(R) Processor @ 2.80GHz`, threshold 0.5; the synthetic corpus
with the five entities above requested; false positives are NER detections
the rules alone do not make in the four agent-traffic files of
`bench/fp_corpus`, 19,808 bytes; p50 is the whole pipeline per
500-character string; [ner-bench.md](ner-bench.md) explains each metric,
[ner-landscape.md](ner-landscape.md#pii-models-measured-by-the-llm-redact-bench)
has the full record):

| Model (weights) | PERSON recall | Character leak | Over-redaction | Agent-traffic false positives per 50 KB | p50, 500 characters |
|---|---|---|---|---|---|
| `urchade/gliner_small-v2.1` (the default) | 0.79 | 0.05 | 0.050 | 34 | — (189 ms asked for `PERSON` only) |
| `knowledgator/gliner-pii-edge-v1.0` (PyTorch) | 0.99 | 0.08 | 0.100 | 109 | 93 ms |
| `knowledgator/gliner-pii-edge-v1.0` (int8 ONNX) | 0.77 | 0.27 | 0.050 | 16 | 97 ms |
| `knowledgator/gliner-pii-base-v1.0` (PyTorch) | 0.97 | 0.02 | 0.055 | 8 | 234 ms |
| `knowledgator/gliner-pii-base-v1.0` (int8 ONNX) | 0.96 | 0.03 | 0.028 | 13 | 182 ms |

- `-base` leaks the least of the five types and draws the fewest false
  positives on agent traffic; its int8 export keeps that at about three
  quarters of the PyTorch latency. `-edge` is the fast one, but its int8
  export loses a quarter of the names (PERSON recall 0.99 → 0.77).
- Most false positives on agent traffic are `USERNAME` (identifiers and
  commit authors' handles read as user names). Asked for `PERSON` only,
  `-edge` (PyTorch) finds every name of the corpus and makes no detection
  in the agent-traffic files.
- Knowledgator's card suggests a threshold of 0.3: on the int8 exports it
  raises recall (`-base` PERSON 0.998, leak 0.003) and the false positives
  with it (from 13 to 57 per 50 KB for `-base`, from 16 to 114 for
  `-edge`), and on `-edge` it costs four email addresses their exact
  match (a wider model span over the address wins the overlap).

None of them meets every bar of the catalog's "vetted" status (agent-traffic
false positives above one per 50 KB; `-base` above 100 ms too), so they stay
"caution" with these numbers in their catalog reasons; the default model is
unchanged. `-small` and `-large` are not measured yet.

### PII models for the `hf` backend

The `hf` backend's default since 1.13.0 is
`OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` (Apache-2.0,
DeBERTa-v3-small, 566 MB of weights, read in windows of 384 tokens, its
card's sequence length), a PII model trained on NVIDIA's Nemotron-PII (CC BY
4.0, which asks for attribution: "Trained on NVIDIA Nemotron-PII, CC BY
4.0"). Before 1.13.0 it was `dslim/bert-base-NER` (MIT, 433 MB), a general
NER model (people, organizations, places), which stays catalogued and pinned:
name it to keep it. `kalyan-ks/ettin-68m-nemotron-pii` (MIT, a ModernBERT
encoder: transformers 4.48 or newer) is trained on the same data. Both PII
models label 54–55 kinds of data; their `first_name`/`last_name`,
`street_address`, `date_of_birth`, `user_name` and `account_number` labels
fold into `PERSON`, `ADDRESS`, `DATE_OF_BIRTH`, `USERNAME` and
`ACCOUNT_NUMBER`, and their sensitive-attribute labels (gender, race or
ethnicity, religious belief, political view, sexuality) are never folded. The
default configuration asks for `PERSON`; to have the default model find the
other contextual types too:

```toml
[detection.ner]
enabled = true
backend = "hf"
entities = ["PERSON", "ADDRESS", "DATE_OF_BIRTH", "USERNAME", "ACCOUNT_NUMBER"]
```

Measured by the NER bench (2026-10-07, the same machine and rules as the
GLiNER table above; the two PII models re-measured that day with
word-by-word decoding, while other jobs ran on the machine, so their
latencies are indicative only; the default's accuracy re-measured with the
same results on 2026-10-14 as the `hf-default` bench configuration, its
latency taken side by side with `dslim/bert-base-NER` on 2026-10-07):

| Model, entities | PERSON recall (exact) | Character leak | Over-redaction | Agent-traffic false positives per 50 KB | p50, 500 characters |
|---|---|---|---|---|---|
| OpenMed-PII Small 44M (the default), `PERSON` | 1.00 (1.00) | 0.39 | 0.002 | 0 | 175 ms |
| OpenMed-PII Small 44M, five types | 1.00 (1.00) | 0.05 | 0.006 | 18 | 136 ms |
| `dslim/bert-base-NER` (the default before 1.13.0), `PERSON` | 0.97 (0.95) | 0.40 | 0.003 | 0 | 176 ms |
| ettin-68m-nemotron-pii, five types | 0.98 (0.87) | 0.14 | 0.005 | 16 | 127 ms |

`kalyan-ks/ettin-68m-nemotron-pii` tags every sub-word piece of a value as
the start of a value; read word by word ("BIO taggers without word-piece
marks" above), its names are whole values, but most of its account numbers
carry labels that are not folded into `ACCOUNT_NUMBER` (`customer_id`,
`unique_id`, …), so a quarter of the synthetic corpus's account numbers is
found and three quarters of their digits leak. Both PII models are catalogued
"caution" with their five-type numbers (agent-traffic false positives above
one per 50 KB, latency above 100 ms). The `PERSON`-only rows request what the
default configuration requests, so their character leak counts every other
labelled value as leaked; asked for `PERSON` only, OpenMed-PII Small 44M finds
more names than `dslim/bert-base-NER` with no agent-traffic false positive
for either ([ner-landscape.md](ner-landscape.md#the-hf-default)).

#### The `hf` default's change in 1.13.0

A configuration that names no model for the `hf` backend (no
`[detection.ner.models] hf`, and no `[detection.ner] model` with `hf` the only
backend) loads the new default after the upgrade. Three things can stop its
startup:

- **The model is not in the cache.** Downloads are off by default, so the
  startup (and `serve --check`) stops with "hf model
  'OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1' at revision a2360d3f42526fc660ac3b2b2301e1c2d94eba61 is
  not (completely) in the local Hugging Face cache … run `llm-redact models
  pull`". Run `llm-redact models pull` once (566 MB), or write the model's
  folder for an air-gapped host with `models pull --to` ([air-gapped.md](air-gapped.md)).
- **An entity only the previous default served.** While raw entities keep
  their own type (until 2.0.0), the previous default's CoNLL-2003 labels
  served a raw `PER`, `ORG`, `LOC` or `MISC` in `entities`; the new default
  emits none of them. Rather than redact less, every detector build — the
  startup, `serve --check`, a reload (which keeps the running configuration)
  and the config editor's check — refuses such a configuration, naming the
  entities (never a value) and both ways out: keep the previous default with

  ```toml
  [detection.ner.models]
  hf = "dslim/bert-base-NER"
  ```

  or change `entities` to types the new default emits (`PERSON` instead of
  `PER`; `["PERSON"]`, the default, keeps the `PERSON` placeholder type).
  The decision uses the backend's own label policy (above), so a
  `[detection.ner.labels]` override counts: `PER = "PERSON"` makes `PER` a
  `PERSON` request, which passes. `llm-redact doctor` shows the same line as a
  FAIL without loading a model (from the labels the model catalog records for
  the new default).
- **A revision pin written for the previous default.** `[detection.ner.revisions]
  hf = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"` (the commit the model catalog
  records for `dslim/bert-base-NER`; these docs showed it before 1.13.0) with no
  `hf` model named would pin the new default to a commit of another repository.
  Every detector build refuses it before any file is looked up, naming both ways
  out: name `dslim/bert-base-NER` in `[detection.ner.models] hf` to keep it at
  that pin, or remove the `hf` entry of `[detection.ner.revisions]` to load the
  new default at its own catalog pin. `llm-redact doctor` FAILs it the same way.

`openai/privacy-filter` (Apache-2.0; a 1.5B-parameter mixture of experts,
50M parameters active, 2.8 GB of weights) labels `private_person`,
`private_address`, `account_number`, `private_email`, `private_phone`,
`private_url`, `private_date` and `secret` with BIOES tags, which the `hf`
backend decodes with the constrained Viterbi decoder and the
`viterbi_calibration.json` the repository ships (fetched with the model). It
needs transformers 5.6.0 or newer: an older one stops the startup with
`[detection.ner] hf: openai/privacy-filter needs transformers >= 5.6.0 (model
catalog), …` before any weights load. Ask it for `PERSON`, `ADDRESS` and
`ACCOUNT_NUMBER`; its `secret` label is better left to the anchored secret
rules. Measured on the same corpora: `PERSON` recall 0.99, character leak
0.12, 23 agent-traffic false positives per 50 KB, and about a second per
500-character string on a 4-core CPU (1.1 s; 19.5 s for 10,000 characters),
so it stays "caution" and its bench configuration is measured by hand
(`bench/configs/manual/`, [ner-bench.md](ner-bench.md)); CI pulls it only
for a smoke test on a few short strings. Those latencies were
measured with its weights in bfloat16, as the repository stores them; the
`hf` backend now loads every model in float32, which measured about 30%
faster on that CPU for the model alone, and which takes twice the memory of
bfloat16 weights: about 6 GB of RAM for its 2.8 GB of weights, so a
container or Helm memory limit sized for bfloat16 is too small.

### Model-load policies (plugins)

The core loads the model the configuration names; it holds no policy that
refuses one. A plugin package may install one through the registry seam
`Registry.build_model_policy(config, tier)` (`plugin_api.ModelPolicy`; the
Free default returns None, so nothing is asked). llm-redact-pro's `[models]`
section is such a policy: it can restrict models to a list of licenses, refuse
lineage tags from the catalog and require every model to come from a verified
offline bundle.

The policy is built once at startup and asked, synchronously, about every NER
backend's model in every detector build of the process: the startup (`serve`,
`serve --check`), a reload that rebuilds the detectors, the config editor's
dry run and `llm-redact preview`. For the `gliner`, `gliner2` and `hf`
backends it is asked after the model's files are resolved (from the local
cache, a local folder, or downloaded at startup when `allow_download` lets
it) and before its weights load, and it is shown a `plugin_api.ModelLoad`:
the backend, the configured model, the Hub id and commit (a local folder's
from its `llm-redact-model.json`), the folder and every file the load reads
(path inside the model folder and the absolute path it is read from; for a
GLiNER checkpoint assembled with its base model, the base model's tokenizer
and configuration too), the ONNX file, and the catalog's status, license,
lineage tags and attribution. A spaCy, Presidio or Stanza model is not a Hub
snapshot: the policy sees its backend and configured model only (a Stanza
model is its language).

`None` lets the model load. A reason refuses it: the build fails with
`[detection.ner] BACKEND model 'MODEL' refused by the model-load policy:
REASON`, so the startup refuses to serve, a reload keeps the running
configuration and the editor answers 400. Nothing fails open: an exception,
an empty string or any other answer refuses too, naming only the exception
or answer type.

### Fetching and checking models: `llm-redact models`

`llm-redact models` works on the models of the `gliner`, `gliner2` and `hf`
backends the configuration names (`--config PATH`, like `serve` and `doctor`; NER need not
be enabled yet). `pull` is the one subcommand that downloads; `list` and
`verify` read local files only and never touch the network:

```bash
llm-redact models pull              # fetch each model (and GLiNER base model) at its revision
llm-redact models pull --to DIR [--as /models]   # ... and write portable folders + a manifest
llm-redact models list [--json]     # each model: revision, catalog status, license, files, base model
llm-redact models verify            # exit 1 unless every model is complete at its revision
llm-redact models verify --dir DIR  # check a folder written by `models pull --to` (no config)
```

- `pull` fetches each model at the revision its load asks for (the
  `[detection.ner.revisions]` pin, else the catalog's), with exactly the file
  names the loader uses — never TensorFlow, Flax, ONNX or `original/` copies,
  a `pytorch_model.bin` only where the loader would take it — and a GLiNER
  model's base model at the catalog's pin, into the Hugging Face cache (the
  same `HF_HOME` the proxy reads). With `allow_download = false`, the default,
  this is how a model gets there. For a model nothing pins it prints the
  commit it fetched and the `[detection.ner.revisions]` line that pins it;
  for a model the catalog lists as restricted it prints the catalog's facts.
  A model configured as a local folder is read, never fetched — except that
  for a GLiNER folder that ships no tokenizer or encoder configuration of its
  own (a clone of an urchade model), which loads with its base model's from
  the Hugging Face cache, `pull` fetches that base model (at the catalog's
  pin when the folder's `llm-redact-model.json` names a catalogued model); a
  folder that cannot load fails with the startup's message.
  Downloads need network access to huggingface.co (and `HF_TOKEN` for a gated
  model); exit 1 when a model cannot be fetched.
- `pull --to DIR` also writes one self-contained folder per model into `DIR`
  (`hf-OpenMed--OpenMed-PII-SuperClinical-Small-44M-v1`,
  `gliner-urchade--gliner_small-v2.1`,
  `gliner2-fastino--gliner2-base-v1`): its files copied, a GLiNER model's
  base-model tokenizer and configuration included (the folder loads with no
  base model and no assembly; a GLiNER2 checkpoint is self-contained, its
  `encoder_config/config.json` kept in its subfolder), an
  `llm-redact-model.json` naming the model and revision, and beside the
  folders a manifest, `llm-redact-models.json`, listing every file with its
  size and SHA-256. It then prints the `[detection.ner.models]` lines that
  load the folders — as they are, or as mounted elsewhere with `--as PATH`
  (`--as /models` for a volume mounted at `/models`; an absolute path, since
  the proxy reads a relative one against its working directory, and one such
  as `models/hf-…` as a Hugging Face model id). A folder loads with downloads
  off and no network at all; it takes no `[detection.ner.revisions]` entry
  (its `llm-redact-model.json` records the revision, so the catalog and
  `/status` still know which model it is). Only the Hub models `pull` fetches
  are written: a model configured as a local folder is not copied (carry it
  yourself — a GLiNER folder without a tokenizer of its own also needs its
  base model in the Hugging Face cache where it loads). Nothing is written to
  `DIR` unless every model was fetched; a folder of the same name that `pull
  --to` did not write is never replaced, and nothing is replaced until every
  new folder is written beside the old ones (should replacing them still fail
  part-way, `DIR` is left without its manifest, so `verify --dir` fails until
  a pull completes). Use an empty `DIR`: `verify --dir` checks every file
  inside the model folders. This is the way to carry models into an air-gapped
  network: pull on a connected machine, copy `DIR`, run `llm-redact models
  verify --dir` there, and point the configuration at the folders.

- `list` prints, per model, the revision it loads, its catalog status and
  license, whether its files are there (`cached`, `folder` for a local
  folder, `missing`, `incomplete`, `error`) and its base model (a GLiNER
  model without a tokenizer of its own); `--json` adds the catalog's facts,
  the libraries a model needs and the problem text. spaCy, Presidio and
  Stanza models are not Hugging Face snapshots: their install commands are
  printed instead.
- `verify` exits 1 unless every model's files are complete where its load
  reads them, at its revision: its configuration (a GLiNER2 model's encoder
  configuration too), every weight file and its tokenizer, and a GLiNER
  model's base model. The Hugging Face cache can hold
  a revision only in part (after an interrupted download) and still report
  it as present; `verify` checks what the loader needs, exactly as the
  proxy's startup does. Exit 2: the configuration cannot be read.
- `verify --dir DIR` checks a folder of portable models written by
  `llm-redact models pull --to` against the manifest beside them,
  `llm-redact-models.json`: every file's size and SHA-256, no file the
  manifest does not list inside a model folder (a loader could read it), each
  folder's `llm-redact-model.json` naming the model and revision the manifest
  does, and each folder complete for its loader. It needs no configuration
  and no network, so it runs inside an air-gapped enclave before the proxy
  loads the folder; entries beside the model folders (`lost+found`) are only
  noted.

## How NER runs

![Flowchart of one string through one NER backend: the max_chars gate, one call or overlapping windows, the model (an hf BIOES/BILOU tagger's, or a BIO tagger without word pieces', spans decoded by llm-redact), the label policy, the placeholder-type guard, threshold and offset checks, duplicate removal, part merging, rule toggles, the allowlist, overlap resolution with the regex rules and deny strings, and the mode that sends the winner to the vault](diagrams/ner-pipeline.png)

*Static diagram. [Mermaid source](diagrams/ner-pipeline.mmd).*

Every NER backend handles each string the redaction scans the same way:

1. **The `max_chars` gate.** A string longer than `[detection.ner] max_chars`
   (default 20,000 characters) is read by no model; it is counted as
   `skipped_max_chars`, and the regex rules, deny strings and custom rules
   still scan it.
2. **One call or windows.** A string that fits the model's window is read in
   one call, as sent. A longer one is read in overlapping windows (the `hf`,
   `gliner`, `gliner2` and `llm` backends; below).
3. **The model** reports entities with a label, a score and character offsets.
4. **The label policy** turns the label into a placeholder type and keeps it
   only when that type was requested (see "Placeholder types from NER
   models" below).
5. **The type guard** drops a type that cannot be a placeholder type
   (`labels_dropped`).
6. **Score and offsets.** An entity scored below the backend's score
   threshold (`score_threshold`, else its model's catalog default; on the
   backends that report a score) is dropped;
   one whose span the string does not contain is never redacted
   (`offsets_dropped`): only the exact text sent can be restored.
7. **Duplicates and parts.** An entity two windows both report is kept once,
   and parts of one name or address one or two blanks apart become one span.
8. **Rule toggles.** A type whose built-in rule is disabled (or scoped out by
   `[detection] languages`) is suppressed for NER too.
9. **Allowlists, overlaps and modes.** Allowlisted values are left alone; the
   rest meet the regex rules and deny strings in overlap resolution (the
   longest span wins; a regex rule wins an exact tie; a deny string is
   joined with what it overlaps and always redacted,
   [Deny strings](#deny-strings-values-that-must-always-be-redacted); an
   `llm` finding never displaces another detection and keeps only what
   lies outside the others' winners), and the winner's mode redacts it
   into the vault, forwards it (warn) or refuses the request (block). A span of any other backend overlapping a card read inside a
   run of digit groups, a paper IBAN or a value before a run of dots is
   joined with it instead
   ([Values joined with the ones they overlap](#values-joined-with-the-ones-they-overlap));
   an `llm` finding is not: it keeps only what lies outside that value.

**Long strings.** A model reads a bounded number of tokens at a time and, left
alone, ignores the rest of a longer string. The `hf`, `gliner` and `gliner2`
backends read a longer string in overlapping windows, so a name anywhere in it
is found at its exact offsets:

- `hf`: a window is the model's limit (the window the model catalog records
  for it — 384 tokens for the default,
  `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1`, its card's sequence
  length — else the smaller of its tokenizer's `model_max_length` and its
  config's `max_position_embeddings`, 512 when neither says; 512 tokens for
  `dslim/bert-base-NER`), and consecutive windows share a quarter of it.
  Windowing needs a fast tokenizer — the only kind that reports character
  offsets — so an `hf` model without one is refused at startup. A model whose tokenizer marks word pieces (WordPiece, as BERT and
  `dslim/bert-base-NER` use) labels each word by its first piece, so a name
  is reported as whole words, never cut inside one ("Angela Merk"). Other
  tokenizers (SentencePiece, byte-level BPE) do not tell the pipeline where
  words end, so llm-redact decodes a BIO model with such a tokenizer itself,
  word by word (below).
- `gliner2`: GLiNER2 sets no word limit of its own, but its encoder was
  trained on 512 positions (Fastino's DeBERTa-v3 encoders) and its cost grows
  with the square of the length. A window holds at most 200 of GLiNER2's own
  words — an e-mail address, a URL or an @handle is one word; every other
  brace, quote, colon and comma is one — and, with the model's fast
  tokenizer, no more subword tokens than the encoder reads beside the entity
  prompt; windows overlap and count like GLiNER's below.
- `gliner`: GLiNER reads at most `max_len` words of a text (384 for the urchade
  v2.1 models) and drops the rest. A window holds at most 200 of GLiNER's own
  words, fewer when the entity prompts leave less room within `max_len`, and —
  with the model's fast tokenizer — no more subword tokens than its encoder
  reads beside the prompt (the tokenizer's limit, else the encoder's position
  limit, else 512). GLiNER counts every JSON brace, quote, colon and comma as
  a word, so JSON-dense text makes many words. Consecutive windows share a
  fifth of a window. A single word longer than the encoder reads (a very long
  identifier) gets a window of its own, which the model may read only in part:
  counted as `windows_truncated`.

**BIOES and BILOU taggers (`hf`).** Most token-classification models tag a
span's first token `B-` and the rest `I-` (BIO), which the transformers
pipeline reads. A model whose labels also mark a span's last token —
`E-`nd and `S-`ingle (BIOES), or `L-`ast and `U-`nit (BILOU) — would have every
span cut at its last token by the pipeline, so llm-redact reads such a model
itself: the same token windows, the model's per-token label scores, and its
own span decoder. When llm-redact's model catalog lists the model's
calibration file (transition biases its publisher ships, fetched with the
model), spans are decoded with a constrained Viterbi decoder: the best label
sequence in which every span opens with `B`/`S`, continues with `I` of the
same entity and closes with `E`/`S`. The file is then part of the model:
`llm-redact models pull` fetches it, `models verify` (and `verify --dir`) and
doctor require it, and a snapshot or folder without it is refused at startup
like one missing its weights (a snapshot pulled before the catalog listed
the file: run `models pull` again) — a model whose catalog entry lists a
calibration file is never decoded greedily instead. For a model the catalog
lists no calibration file for, each token takes its most likely label and
spans are read greedily: `B` … `E`, a single `S`, `I`
continuing; a tag that cannot continue the open span starts a new one, and a
span left open is kept as it is, so no token the model marked is dropped. A
span's score is the mean probability of its tokens' labels (`score_threshold`
applies), and its offsets leave out blanks at either edge. With a tokenizer
that marks word pieces (WordPiece), the decoder reads words instead of tokens,
each scored by its first piece — the piece such a model is trained to label —
as the pipeline reads a BIO model, so a span covers whole words and is never
cut inside one. The scheme comes
from the model's labels; a model whose labels mix BIOES and BILOU tags is
refused at startup.

**BIO taggers without word-piece marks (`hf`).** A SentencePiece or
byte-level BPE tokenizer (DeBERTa-v3, XLM-R, RoBERTa, ModernBERT) does not
tell the transformers pipeline where words end: labelled piece by piece, a
name became one value per piece (`kalyan-ks/ettin-68m-nemotron-pii` tags
every piece `B-`, so "Zbigniew Brzezinski" was six values), and the
pipeline's whitespace fallback would read `{"name":"Angela` as one word
labelled by its brace. llm-redact reads such a model itself, over the same
token windows, word by word:

- the words are the text's own: a word ends at a blank, a quote or a
  bracket of any script, and at `,` `;` `:` `=` `|` `/` `\` and the ASCII
  quotes and brackets; each character of a script written without spaces
  between words (Chinese, Japanese kana, Thai, Lao, Khmer, Myanmar,
  Tibetan, Yi) is a word of its own, with the combining marks after it;
  punctuation at either end of a word is left out ("Paris." is "Paris"),
  punctuation inside it stays ("1985-03-12", "j.doe" and "dev_jo42" are one
  word each). The words decide which piece labels what; what a value covers
  can grow past them (below);
- a word is labelled by its first piece, the piece a model trained the
  Hugging Face way labels (its later pieces were never trained and may say
  anything); a piece of blanks only right before the word — SentencePiece's
  lone "▁" before a character its vocabulary has no "▁"-piece for, such as
  "Ę" or a CJK character — is that first piece, wherever the word stands.
  For a model llm-redact's model catalog lists as trained on every piece
  (`kalyan-ks/ettin-68m-nemotron-pii`), which may leave a word's first piece
  untagged and tag a later one, a word is labelled by the piece tagged most
  confidently, else by its first piece;
- a window reads the words whose first piece it holds. A word no window
  holding its first piece reads to its end (longer than the windows'
  overlap) is read in parts, each from its own first piece in the window
  that holds it, so every piece of a text is read;
- labels are read greedily, `B` opening a span and `I` continuing it; a span
  is cut where anything but one or two blanks, a comma and a space or a
  slash separates two of its words — a quote, a bracket, a colon, a
  semicolon, an equals sign, a newline. A span continues across a comma and
  a space or a slash only where the model continues it ("March 3, 1985",
  "03/12/1985"; also two numbers of a list, "4417123456, 5512345678", when
  the model tags the second as the first one's continuation);
- a span then grows over the characters beside it that no word holds and
  that a piece the model tags with the span's entity covers: the symbols at
  a password's edges ("$ecret!"), a colon or an equals sign inside one
  ("p@ss:w0rd=x9"), the colons of a MAC address — every character the model
  tags as part of a value is redacted with it. It never grows over a blank,
  a quote or a bracket: a quote inside a value (a password holding `"`)
  goes upstream as sent, between the two parts of the value, each its own
  placeholder. Neighbouring spans of one label then separated by nothing or
  a slash are one value;
- a span's score is the mean probability of the words the model tagged as
  one span (`score_threshold` applies), also for each part a cut leaves.

A label without a tag (`PER`) is read as `I-PER`, as the pipeline reads it.
WordPiece models keep the pipeline, unchanged.

An entity two windows both report counts once; one cut by a window's edge is
also reported whole by the next window, and the longer span wins. spaCy,
Stanza and Presidio read each string whole. Windows make a long string cost
more model time; `max_chars` caps that time per string, and the strings it
skips are counted (see "NER coverage counters" below).

## Placeholder types from NER models

Each NER model names what it finds in its own words: spaCy says `PERSON`,
`dslim/bert-base-NER` says `PER`, a PII model may say `first_name` and
`last_name`. llm-redact turns every model label into one placeholder type, so
a value gets the same token whichever backend finds it — the vault is keyed on
(session, type, value), and two names for one type would issue two tokens for
one person.

A label is first normalized (a leading `B-`/`I-`/`E-`/`S-`/`L-`/`U-` tag is
dropped, letters are uppercased, any other run of characters becomes `_`:
`"B-first_name"` → `FIRST_NAME`, `"street address"` → `STREET_ADDRESS`), then
folded into its placeholder type:

| Type | Model labels that fold into it |
|---|---|
| `PERSON` | PER, PERSON, NAME, FULL_NAME, FIRST_NAME, FIRSTNAME, GIVENNAME, GIVEN_NAME, MIDDLE_NAME, MIDDLENAME, LAST_NAME, LASTNAME, SURNAME, FAMILY_NAME, PRIVATE_PERSON |
| `ADDRESS` | ADDRESS, STREET_ADDRESS, STREETADDRESS, STREET, LOCATION_STREET, LOCATION_ADDRESS, BUILDINGNUM, BUILDING_NUMBER, BUILDINGNUMBER, PRIVATE_ADDRESS |
| `DATE_OF_BIRTH` | DATE_OF_BIRTH, DATEOFBIRTH, DOB, BIRTH_DATE, BIRTHDATE |
| `PASSPORT` | PASSPORT, PASSPORT_NUMBER, PASSPORTNUM, PASSPORTNUMBER |
| `DRIVER_LICENSE` | DRIVER_LICENSE, DRIVERS_LICENSE, DRIVER_LICENSE_NUMBER, DRIVERS_LICENSE_NUMBER, DRIVERLICENSENUM, DRIVER_LICENCE |
| `USERNAME` | USERNAME, USER_NAME |
| `ACCOUNT_NUMBER` | ACCOUNT_NUMBER, ACCOUNTNUM, BANK_ACCOUNT, BANK_ACCOUNT_NUMBER |
| `EMAIL` | EMAIL, EMAIL_ADDRESS, PRIVATE_EMAIL |
| `PHONE` | PHONE, PHONE_NUMBER, TELEPHONE, TELEPHONENUM, PRIVATE_PHONE |
| `SSN` | SSN, US_SSN |
| `IBAN` | IBAN, IBAN_CODE |
| `CREDIT_CARD` | CREDIT_CARD, CREDIT_CARD_NUMBER, CREDITCARDNUMBER, CREDIT_DEBIT_CARD, CARD_NUMBER, PAYMENT_CARD |
| `IPV4` / `IPV6` | IPV4 / IPV6 |
| `SECRET` (the type of `generic_secret`) | PASSWORD, SECRET, API_KEY, ACCESS_TOKEN |

Deliberately not folded, and kept only when listed raw in `entities`:
`IP_ADDRESS` (it covers v4 and v6), place names and plain dates (`LOC`,
`LOCATION`, `GPE`, `CITY`, `STATE`, `COUNTRY`, `ZIPCODE`, `POSTCODE`, `DATE`,
`PRIVATE_DATE`, `TIME`), organisations (`ORG`, `COMPANY_NAME`), URLs, the
country-ambiguous `SOCIALNUM`/`TAXNUM`/`IDCARDNUM` (the national-id rules own
those types), and every label describing a sensitive attribute (race or
ethnicity, religion, political view, sexuality, gender, Presidio's `NRP`):
those are detected only when you list them yourself.

**Type requests.** An entry of `[detection.ner] entities` whose normalized form
is a placeholder type — `PERSON`, `ADDRESS`, `DATE_OF_BIRTH`, `PASSPORT`,
`DRIVER_LICENSE`, `USERNAME`, `ACCOUNT_NUMBER`, or any built-in rule's type
such as `EMAIL` — requests that type from every backend, whatever the model
calls it. The default `entities = ["PERSON"]` therefore works with spaCy,
Stanza, Presidio and with a `PER`-emitting `hf` model alike, and a name two
backends both find gets one token. GLiNER and GLiNER2 are prompted in natural
language for a type request (`PERSON` → "person", `ADDRESS` → "street address",
`DATE_OF_BIRTH` → "date of birth", `PASSPORT` → "passport number",
`DRIVER_LICENSE` → "driver license number", `USERNAME` → "username",
`ACCOUNT_NUMBER` → "account number", `EMAIL` → "email address", `PHONE` →
"phone number"; other built-in types send their name in lowercase words) —
unless the model catalog records the prompt the loaded model was trained on
for that type, which is sent instead (Knowledgator's GLiNER-PII models are
asked for "name", not "person": "Choosing a GLiNER model" above). A model
folder takes the prompts of the model its `llm-redact-model.json` names.

**Raw requests.** Any other entry (`PER`, `ORG`, `"job title"`) is a raw
request: GLiNER and GLiNER2 are sent the text as written, and the backend emits the label's
normalized form (`"job title"` → `JOB_TITLE`) — as before, Presidio's
`EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `IBAN_CODE` and `CREDIT_CARD`
included, which the presidio backend emits as the built-in `EMAIL`, `PHONE`,
`SSN`, `IBAN` and `CREDIT_CARD`.

**Deprecated: raw requests fold from 2.0.0.** From 2.0.0 a raw request folds
like a model label: `entities = ["PER"]` will request and emit `PERSON`,
`"phone number"` `PHONE`, `EMAIL_ADDRESS` (outside Presidio) `EMAIL`. Each
configured entity whose type will change logs one WARNING at startup and shows
a `doctor` WARN row naming both types. Write the type (`"PERSON"`) to switch
now, or keep the old type with an override (`[detection.ner.labels] PER =
"PER"`, below).

**Overrides.** `[detection.ner.labels]` maps a model label (normalized as
above, so `"first name"` and `FIRST_NAME` are one key) to a placeholder type,
or to `""` to drop it. An override comes before the fold table, and it applies
to the `entities` list as well as to model output:

```toml
[detection.ner]
entities = ["PERSON", "ADDRESS"]

[detection.ner.labels]
CITY = "ADDRESS"   # a model's CITY detections are redacted as addresses
TIME = ""          # the model's TIME label is never emitted
```

A target type uses the deny-string type grammar: an uppercase letter, then
uppercase letters, digits and `_`, at most 20 characters. `PER = "PER"` keeps
`entities = ["PER"]` emitting `PER` after raw requests start folding in 2.0.0.

Presidio is asked only for the entities its analyzer supports for the
configured language and the policy keeps: a type request becomes the Presidio
entities that fold into it (`EMAIL` asks for `EMAIL_ADDRESS`, `PHONE` for
`PHONE_NUMBER`, `SSN` for `US_SSN`, `PERSON` for `PERSON`), a raw request is
asked for as written, and a type Presidio has no entity for (`ADDRESS`) is left
out. An `entities` list Presidio supports none of stops the startup with an
error, rather than failing every request.

A type the token format cannot carry (one that does not start with a letter,
or longer than 28 characters) is never emitted.

After the models load, each entity is checked against the labels they can
emit (an `hf` model's `id2label`, a spaCy pipeline's `ner` labels, the
entities Presidio supports; zero-shot GLiNER and GLiNER2, and Stanza, can emit
anything).
An entity no active backend can ever emit logs one WARNING naming the entity,
the backends and their models, so a typo or a label the model lacks no
longer detects nothing in silence.

Models trained to tag `first_name` and `last_name` separately report "Jane
Doe" as two parts. When one backend finds two `PERSON` (or two `ADDRESS`)
spans separated by one or two spaces, tabs or no-break spaces, they become
one span, so a full name gets one token. Parts are never joined across a
newline, punctuation, a quote, a comma or a JSON delimiter.

`[detection.allowlist_by_type]` keys name the type a detection carries.
Besides the rule and deny types, a key may name any type the NER entities are
emitted as (`JOB_TITLE` for the GLiNER entity `"job title"`) or an entity as
written; a key that only NER emits is read as the type NER emits for it
(`"job title"` → `JOB_TITLE`; `PER` → `PERSON` once raw entities fold), and
the startup log names each such key once.

A folded built-in type follows its rule's toggle: with `generic_secret`
disabled, a model's `PASSWORD` detections (type `SECRET`) are suppressed too,
and with `email` disabled so are the `EMAIL_ADDRESS` ones.

## NER coverage counters

A model reads a bounded amount of text, and text it never reads is covered by
the regex rules alone. Each NER backend therefore counts what it read and what
it dropped, so a gap shows instead of staying silent. `GET
/__llm-redact/status` carries the counts in `detection.ner` (beside the
existing `detection.ner_enabled`):

```json
"ner": {
  "enabled": true,
  "max_chars": 20000,
  "backends": {
    "hf": {
      "model": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
      "source": "hub", "model_id": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
      "revision": "a2360d3f42526fc660ac3b2b2301e1c2d94eba61", "pinned": true,
      "catalog": "caution", "license": "Apache-2.0",
      "score_threshold": 0.5, "score_threshold_source": "default",
      "counters": {"scanned_whole": 812, "scanned_windowed": 14, "skipped_max_chars": 3,
                   "windows": 61, "windows_truncated": 0, "labels_dropped": 0,
                   "offsets_dropped": 0, "ungrounded": 0, "errors": 0,
                   "breaker_opens": 0, "breaker_short_circuits": 0,
                   "inline_calls": 0, "prefetch_misses": 0}
    }
  },
  "unmatched_entities": []
}
```

| Counter | Counts |
|---|---|
| `scanned_whole` | strings the model read in one call |
| `scanned_windowed` | strings the model read in overlapping windows (`hf`, `gliner`, `gliner2`) |
| `skipped_max_chars` | strings longer than `[detection.ner] max_chars`, which the model never read (the regex rules and deny strings still scan them) |
| `windows` | the windows the windowed strings were read in |
| `windows_truncated` | windows (a string read whole counts as one) holding a single word longer than the model's encoder reads, which it may read only in part (`gliner`, `gliner2`); for `llm`, the pieces a word longer than its 2,000-character window is cut into |
| `labels_dropped` | model entities whose type cannot be a placeholder type (never emitted) |
| `offsets_dropped` | model entities of a requested type whose span the scanned string does not contain, or that came without one (never redacted: only the exact text sent can be restored) |
| `ungrounded` | `llm` only: values the model server returned whose text does not occur in the text it read, is shorter than two characters or holds a placeholder's guillemet (never redacted) |
| `errors` | `llm` only: strings its model server failed on (a timeout, an error status, a connection fault, an answer that is not the requested JSON) and strings its open circuit breaker failed at once; each refused its request with `on_error = "block"`, and went on without the backend's findings with `"skip"` |
| `breaker_opens` | `llm` only: times its circuit breaker opened (`breaker_failures` timeouts, transport faults or HTTP 5xx answers in a row; a failed probe opens it again at once) |
| `breaker_short_circuits` | `llm` only: strings failed at once, without asking its model server, because its circuit breaker was open (each also in `errors`) |
| `inline_calls` | strings the backend was run on by the redaction itself instead of ahead of it on the NER worker thread, every call attempted, whether it returned or raised: on a request's path that is the event loop, every other request waiting for that string's inference; a redaction preview in the llm-redact-pro dashboard counts too — on a worker thread of its own in llm-redact-pro releases after 0.17.1, on the event loop in 0.17.1 and earlier |
| `prefetch_misses` | strings a request's redaction did not find among its precomputed NER results (or found computed for detectors a reload has since replaced), run inline instead: counted only where the backend ran on that string, so each is counted in `inline_calls` too — a backend listed after a detector that raised on the string (the `llm` backend's `on_error = "block"` failure refusing its request, any other fault) never runs on it and counts neither |

Each string a backend is handed counts once, under `scanned_whole`,
`scanned_windowed` or `skipped_max_chars`; with several backends each one
counts the strings it was handed. `unmatched_entities` lists the configured
entities no active backend can ever emit (the startup warning above), and
`model` the model each backend loaded. For the `gliner`, `gliner2` and `hf` backends,
`source` says whether it came from the Hugging Face Hub (`hub`) or a local
folder (`local`), `model_id` names the Hub model (a folder's: the one its
`llm-redact-model.json` names, else `null`), `revision` the commit it loads
(the `[detection.ner.revisions]` pin, else the model catalog's; a folder's from
its `llm-redact-model.json`), `pinned` whether there is one, and `catalog` and
`license` what the model catalog records (`vetted`, `caution`, `restricted`,
or `null` for a model it does not list; see "The model catalog"). These six
are `null` for the spaCy, Presidio and Stanza backends, whose models are not
Hub snapshots. `score_threshold` is the confidence threshold a backend runs at
and `score_threshold_source` where it comes from: `config` (`[detection.ner]
score_threshold`), `catalog` (the model's catalog default) or `default` (0.5);
both are `null` for spaCy and Stanza, which report no confidences. The `llm`
backend's entry adds `server` (its model server's `scheme://host:port`), `api`,
`on_error`, `api_key` (whether a key is configured: never the key or its
variable) and `breaker` (its circuit breaker: `state` — `closed`, `open`,
`half_open` once the cooldown has passed, `off` with `breaker_seconds = 0` —,
`seconds`, `failures` — the calls failed in a row because the server is
down, since its last answer — and `threshold` (`breaker_failures`, the
failures in a row that open it), the `reason` that opened it and, while
open, `retry_in_seconds`).

The same counts are Prometheus counters (`llm_redact_ner_strings_total` by
backend and outcome, `llm_redact_ner_windows_total`,
`llm_redact_ner_windows_truncated_total`, `llm_redact_ner_labels_dropped_total`,
`llm_redact_ner_offsets_dropped_total`, `llm_redact_ner_ungrounded_total`,
`llm_redact_ner_errors_total`, `llm_redact_ner_breaker_opens_total`,
`llm_redact_ner_breaker_short_circuits_total`, `llm_redact_ner_inline_calls_total`,
`llm_redact_ner_prefetch_misses_total`;
see [observability.md](observability.md)), and `llm-redact status` prints a
posture line while a backend has skipped strings longer than `max_chars` or an
entity can never match, and whenever the `llm` backend runs with `on_error =
"skip"`:

```
posture:
  ⚠ NER skipped hf×3 string(s) longer than max_chars (20000) — regex rules still applied
  ⚠ NER entities no backend can emit: PERSONS (never detected)
  ⚠ NER llm on_error = "skip": 2 string(s) forwarded without its findings after its model server failed (the other detectors still applied)
```

The counters belong to the built detectors: they start at zero at startup and
again when a reload rebuilds the detectors (any change to `[detection]`); a
reload that leaves `[detection]` alone keeps them. A redaction preview in the
llm-redact-pro dashboard runs the live detectors and is counted like a
request.
