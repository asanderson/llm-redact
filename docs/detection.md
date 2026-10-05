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
Presidio; implied by the model for spaCy), and `model` overrides the
default pipeline: a spaCy package name for spacy/presidio (default
`en_core_web_sm`) or a Hugging Face model id for GLiNER. Multiple
backends can run concurrently (`backends = ["spacy", "presidio"]`), and
the multilingual Stanza and Hugging Face `token-classification` backends
are available the same way — the survey behind the lineup is
[ner-landscape.md](ner-landscape.md).

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
`SSN`, `IBAN` and `CREDIT_CARD`. From 2.0.0 a raw request folds like a model
label (`entities = ["PER"]` then requests and emits `PERSON`); write the type
(`"PERSON"`) to get that behavior now.

A type the token format cannot carry (one that does not start with a letter,
or longer than 28 characters) is never emitted.

A folded built-in type follows its rule's toggle: with `generic_secret`
disabled, a model's `PASSWORD` detections (type `SECRET`) are suppressed too,
and with `email` disabled so are the `EMAIL_ADDRESS` ones.
