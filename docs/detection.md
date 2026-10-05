# What gets detected

The complete detection surface: the built-in rule families, deny
strings, per-rule modes, allowlists, and the optional person-name NER
backends. The full, current rule list with per-rule comments is in
[`config.example.toml`](../config.example.toml); how each rule is
scored and gated is in [assurance.md](assurance.md) and the benchmark
section of the [README](../README.md#benchmark-and-live-validation).

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
(mod-97 checksum), cryptocurrency wallet addresses (Ethereum with the
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
pattern only fires on checksum-valid matches.

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

A deny match wins any overlap against rule matches (even longer ones),
bypasses the allowlist, and is never subject to per-rule modes — it always
redacts. Matching is literal substring ("Auroras" gets its "Aurora"
redacted); each casing variant round-trips back to exactly what was sent.
The llm-redact-pro dashboard's config editor has a deny-strings table.

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
warn-mode match overlaps) is sent to the provider. Block-mode rejections
return a provider-shaped 400 whose message names the rule type, which
agentic tools surface directly. `llm-redact preview` shows which mode a
given text would trigger; the llm-redact-pro dashboard's config editor has
a three-way selector per rule.

## Person-name detection (optional NER)

Regex catches structured values (emails, keys) but misses most person names.
Optional NER backends close that gap (`[detection.ner]`):

```bash
# spaCy (default backend, ~tens of MB, ~1-5 ms/string):
uv sync --extra ner
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
# or GLiNER (heavy: torch + transformers; more robust on unusual names,
# supports score_threshold):
uv sync --extra gliner
# or Microsoft Presidio (FOSS PII analyzer: recognizers + checksums +
# context scoring over the same spaCy model; supports score_threshold):
uv sync --extra presidio
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Presidio entity types that overlap the built-in regex rules
(`EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `IBAN_CODE`, `CREDIT_CARD`)
are folded into the built-in placeholder names, so a value gets the same
`«EMAIL_NNN»` identity whichever detector finds it.

`[detection.ner] language` sets the analyzer language (wired through
Presidio; the Stanza model to load; implied by the model for spaCy), and
`model` overrides the default model: a spaCy package name for
spacy/presidio (default `en_core_web_sm`), a Hugging Face model id for
GLiNER (default `urchade/gliner_small-v2.1`) or, for the `hf` backend, a
Hugging Face `token-classification` model id (default
`dslim/bert-base-NER`); Stanza ignores it. `score_threshold` (default 0.5)
drops entities below that confidence on the backends that report one —
gliner, presidio and hf; spaCy and Stanza report none, so the key is a
config error when only they are active. Multiple backends can run
concurrently (`backends = ["spacy", "presidio"]`), and the multilingual
Stanza and Hugging Face `token-classification` backends are available the
same way — the survey behind the lineup is
[ner-landscape.md](ner-landscape.md).

### Model sources: pinned revisions, downloads and pickle weights

The `gliner` and `hf` backends load models from the Hugging Face Hub. Three
`[detection.ner]` keys say where those models may come from:

```toml
[detection.ner]
allow_download = false        # the default
allow_pickle_weights = false  # the default; hf only

[detection.ner.revisions]
hf = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"   # a full commit id
```

- `revisions` pins a backend's model to one commit: a full 40-character
  lowercase hex commit id per backend, `gliner` or `hf` (the other backends'
  models are not Hub snapshots and take none). A branch or tag name such as
  `main` is a config error, because it moves. A backend with no entry uses
  the pin llm-redact's model catalog records for the model it loads, when
  there is one: the default models `urchade/gliner_small-v2.1` and
  `dslim/bert-base-NER`, `urchade/gliner_medium-v2.1`,
  `urchade/gliner_multi-v2.1`, `urchade/gliner_multi_pii-v1` and the four
  `knowledgator/gliner-pii-*-v1.0` sizes are pinned to the commit their
  `main` branch pointed at on 2026-10-05. An entry for a backend that is not
  active is kept and ignored, like a `[detection.ner.models]` entry.
- `allow_download` (default `false`) decides whether a startup may fetch the
  pinned model files from the Hub; with `false`, models are meant to load
  only from the local Hugging Face cache or a model folder.
- `allow_pickle_weights` (default `false`, `hf` only) permits loading
  `pytorch_model.bin`, a pickle, for an `hf` model that ships no safetensors
  weights. GLiNER loads its `pytorch_model.bin` through torch's
  `weights_only` loader whatever this key says.

**Not enforced yet.** The three keys are parsed, validated and written back
(`llm-redact config show`, the dashboard editor keeps them from the file),
but the model loaders do not read them yet: whatever `allow_download` and
`revisions` say, a model still loads as it did before — the newest revision
from the Hub, downloaded on first use and cached — and `allow_pickle_weights`
changes nothing.

## How NER runs

![Flowchart of one string through one NER backend: the max_chars gate, one call or overlapping windows, the model, the label policy, the placeholder-type guard, threshold and offset checks, duplicate removal, part merging, rule toggles, the allowlist, overlap resolution with the regex rules and deny strings, and the mode that sends the winner to the vault](diagrams/ner-pipeline.png)

*Static diagram. [Mermaid source](diagrams/ner-pipeline.mmd).*

Every NER backend handles each string the redaction scans the same way:

1. **The `max_chars` gate.** A string longer than `[detection.ner] max_chars`
   (default 20,000 characters) is read by no model; it is counted as
   `skipped_max_chars`, and the regex rules, deny strings and custom rules
   still scan it.
2. **One call or windows.** A string that fits the model's window is read in
   one call, as sent. A longer one is read in overlapping windows (the `hf`
   and `gliner` backends; below).
3. **The model** reports entities with a label, a score and character offsets.
4. **The label policy** turns the label into a placeholder type and keeps it
   only when that type was requested (see "Placeholder types from NER
   models" below).
5. **The type guard** drops a type that cannot be a placeholder type
   (`labels_dropped`).
6. **Score and offsets.** An entity scored below `score_threshold` (on the
   backends that report a score) is dropped;
   one whose span the string does not contain is never redacted
   (`offsets_dropped`): only the exact text sent can be restored.
7. **Duplicates and parts.** An entity two windows both report is kept once,
   and parts of one name or address one or two blanks apart become one span.
8. **Rule toggles.** A type whose built-in rule is disabled (or scoped out by
   `[detection] languages`) is suppressed for NER too.
9. **Allowlists, overlaps and modes.** Allowlisted values are left alone; the
   rest meet the regex rules and deny strings in overlap resolution (deny
   strings win every overlap, then the longest span; a regex rule wins an
   exact tie), and the winner's mode redacts it into the vault, forwards it
   (warn) or refuses the request (block).

**Long strings.** A model reads a bounded number of tokens at a time and, left
alone, ignores the rest of a longer string. The `hf` and `gliner` backends
read a longer string in overlapping windows, so a name anywhere in it is found
at its exact offsets:

- `hf`: a window is the model's limit (the smaller of its tokenizer's
  `model_max_length` and its config's `max_position_embeddings`, 512 when
  neither says; 512 tokens for `dslim/bert-base-NER`), and consecutive windows
  share a quarter of it. Windowing needs a fast tokenizer — the only kind that
  reports character offsets — so an `hf` model without one is refused at
  startup. A model whose tokenizer marks word pieces (WordPiece, as BERT and
  `dslim/bert-base-NER` use) labels each word by its first piece, so a name
  is reported as whole words, never cut inside one ("Angela Merk"). Other
  tokenizers (SentencePiece, byte-level BPE) do not tell the pipeline where
  words end, so their models are labelled piece by piece, and a span can
  still end inside a word.
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
backends both find gets one token. GLiNER is prompted in natural language for
a type request (`PERSON` → "person", `ADDRESS` → "street address",
`DATE_OF_BIRTH` → "date of birth", `PASSPORT` → "passport number",
`DRIVER_LICENSE` → "driver license number", `USERNAME` → "username",
`ACCOUNT_NUMBER` → "account number", `EMAIL` → "email address", `PHONE` →
"phone number"; other built-in types send their name in lowercase words).

**Raw requests.** Any other entry (`PER`, `ORG`, `"job title"`) is a raw
request: GLiNER is sent the text as written, and the backend emits the label's
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
entities Presidio supports; zero-shot GLiNER and Stanza can emit anything).
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
      "model": "dslim/bert-base-NER",
      "revision": null, "catalog": null, "license": null,
      "counters": {"scanned_whole": 812, "scanned_windowed": 14, "skipped_max_chars": 3,
                   "windows": 61, "windows_truncated": 0, "labels_dropped": 0,
                   "offsets_dropped": 0}
    }
  },
  "unmatched_entities": []
}
```

| Counter | Counts |
|---|---|
| `scanned_whole` | strings the model read in one call |
| `scanned_windowed` | strings the model read in overlapping windows (`hf`, `gliner`) |
| `skipped_max_chars` | strings longer than `[detection.ner] max_chars`, which the model never read (the regex rules and deny strings still scan them) |
| `windows` | the windows the windowed strings were read in |
| `windows_truncated` | windows (a string read whole counts as one) holding a single word longer than the model's encoder reads, which it may read only in part (`gliner`) |
| `labels_dropped` | model entities whose type cannot be a placeholder type (never emitted) |
| `offsets_dropped` | model entities of a requested type whose span the scanned string does not contain, or that came without one (never redacted: only the exact text sent can be restored) |

Each string a backend is handed counts once, under `scanned_whole`,
`scanned_windowed` or `skipped_max_chars`; with several backends each one
counts the strings it was handed. `unmatched_entities` lists the configured
entities no active backend can ever emit (the startup warning above), and
`model` the model each backend loaded; `revision`, `catalog` and `license` are
`null`.

The same counts are Prometheus counters (`llm_redact_ner_strings_total` by
backend and outcome, `llm_redact_ner_windows_total`,
`llm_redact_ner_windows_truncated_total`, `llm_redact_ner_labels_dropped_total`,
`llm_redact_ner_offsets_dropped_total`;
see [observability.md](observability.md)), and `llm-redact status` prints a
posture line while a backend has skipped strings longer than `max_chars` or an
entity can never match:

```
posture:
  ⚠ NER skipped hf×3 string(s) longer than max_chars (20000) — regex rules still applied
  ⚠ NER entities no backend can emit: PERSONS (never detected)
```

The counters belong to the built detectors: they start at zero at startup and
again when a reload rebuilds the detectors (any change to `[detection]`); a
reload that leaves `[detection]` alone keeps them. A redaction preview in the
llm-redact-pro dashboard runs the live detectors and is counted like a
request.
