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
# or GLiNER2 (Fastino; the same weight class; zero-shot with character
# spans, supports score_threshold):
uv sync --extra gliner2
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
GLiNER (default `urchade/gliner_small-v2.1`) or GLiNER2 (default
`fastino/gliner2-base-v1`) or, for the `hf` backend, a
Hugging Face `token-classification` model id (default
`dslim/bert-base-NER`); Stanza ignores it. `score_threshold` (default 0.5)
drops entities below that confidence on the backends that report one —
gliner, gliner2, presidio and hf; spaCy and Stanza report none, so the key is
a config error when only they are active.

**GLiNER2 (`gliner2`).** Fastino's GLiNER2 is a schema-driven successor of
GLiNER: zero-shot like GLiNER (it is prompted with the same natural-language
prompts, "person", "street address", …), and it reports each entity's
character span, which llm-redact uses as given — a value that occurs twice is
redacted at both places. It runs on the `gliner2` extra (the gliner2 package
with torch, transformers and peft). The default model,
`fastino/gliner2-base-v1` (Apache-2.0, English), is catalogued as not yet
measured by the llm-redact bench. The gliner2 package also contains a client
for Fastino's hosted API, which llm-redact never uses: models run locally. Multiple backends can run
concurrently (`backends = ["spacy", "presidio"]`), and the multilingual
Stanza and Hugging Face `token-classification` backends are available the
same way — the survey behind the lineup is
[ner-landscape.md](ner-landscape.md).

### Model sources: pinned revisions, downloads and pickle weights

The `gliner`, `gliner2` and `hf` backends load models from the Hugging Face
Hub. Three `[detection.ner]` keys say where those models may come from:

```toml
[detection.ner]
allow_download = false        # the default
allow_pickle_weights = false  # the default; hf only

[detection.ner.revisions]
hf = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"   # a full commit id
```

- `revisions` pins a backend's model to one commit: a full 40-character
  lowercase hex commit id per backend, `gliner`, `gliner2` or `hf` (the other
  backends' models are not Hub snapshots and take none). A branch or tag name such as
  `main` is a config error, because it moves. A backend with no entry uses
  the pin llm-redact's model catalog records for the model it loads, when
  there is one: the default models `urchade/gliner_small-v2.1` and
  `dslim/bert-base-NER`, `urchade/gliner_medium-v2.1`,
  `urchade/gliner_multi-v2.1`, `urchade/gliner_multi_pii-v1` and the four
  `knowledgator/gliner-pii-*-v1.0` sizes are pinned to the commit their
  `main` branch pointed at on 2026-10-05, and the `gliner2` default
  `fastino/gliner2-base-v1` to its `main` commit of 2026-10-06. An entry for a backend that is not
  active is kept and ignored, like a `[detection.ner.models]` entry.
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
catalogued model id. Every load checks that the folder or cached snapshot
holds what it reads — `config.json`, the weights (every shard a weight index
names) and a tokenizer — so an interrupted download counts as a model not in
the cache. A model whose `config.json` or `tokenizer_config.json` names code
to import from its repository (`auto_map`) is refused, and nothing is ever
loaded with `trust_remote_code`. The weights must be safetensors: a model
with only `pytorch_model.bin` is refused unless `allow_pickle_weights = true`,
and a model with both always loads its safetensors.

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

- **vetted**: a known-good choice, pinned to a commit.
- **caution**: configurable and pinned, with the reason shown — for example,
  not yet measured by the llm-redact bench.
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

The defaults are `urchade/gliner_small-v2.1` (`gliner`) and
`dslim/bert-base-NER` (`hf`); the `gliner2` default, `fastino/gliner2-base-v1`,
is listed under caution below. A base model is pinned where the GLiNER
checkpoint ships no tokenizer or encoder configuration of its own.

Configurable, status caution (`not yet measured by the llm-redact bench`;
their cards do not name the training data): Knowledgator's GLiNER-PII models
and the `gliner2` backend's default, `fastino/gliner2-base-v1`. Each ships its
tokenizer and encoder configuration, so no base model is fetched; the
Knowledgator `-edge` and `-small` need transformers 4.48 or newer:

<!-- model-catalog:caution -->
| Model | Backend | License | Pinned revision | Base model (pinned revision) |
|---|---|---|---|---|
| `knowledgator/gliner-pii-edge-v1.0` | gliner | Apache-2.0 | `9b7f39b0a2da` | `jhu-clsp/ettin-encoder-32m` (—) |
| `knowledgator/gliner-pii-small-v1.0` | gliner | Apache-2.0 | `d21aad5b4a7e` | `jhu-clsp/ettin-encoder-68m` (—) |
| `knowledgator/gliner-pii-base-v1.0` | gliner | Apache-2.0 | `61726e0ad791` | `microsoft/deberta-v3-small` (—) |
| `knowledgator/gliner-pii-large-v1.0` | gliner | Apache-2.0 | `f847f54fbc97` | `microsoft/deberta-v3-large` (—) |
| `fastino/gliner2-base-v1` | gliner2 | Apache-2.0 | `f9634218e535` | `microsoft/deberta-v3-base` (—) |
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
  (`hf-dslim--bert-base-NER`, `gliner-urchade--gliner_small-v2.1`,
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

![Flowchart of one string through one NER backend: the max_chars gate, one call or overlapping windows, the model (an hf BIOES/BILOU tagger's spans decoded by llm-redact), the label policy, the placeholder-type guard, threshold and offset checks, duplicate removal, part merging, rule toggles, the allowlist, overlap resolution with the regex rules and deny strings, and the mode that sends the winner to the vault](diagrams/ner-pipeline.png)

*Static diagram. [Mermaid source](diagrams/ner-pipeline.mmd).*

Every NER backend handles each string the redaction scans the same way:

1. **The `max_chars` gate.** A string longer than `[detection.ner] max_chars`
   (default 20,000 characters) is read by no model; it is counted as
   `skipped_max_chars`, and the regex rules, deny strings and custom rules
   still scan it.
2. **One call or windows.** A string that fits the model's window is read in
   one call, as sent. A longer one is read in overlapping windows (the `hf`,
   `gliner` and `gliner2` backends; below).
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
alone, ignores the rest of a longer string. The `hf`, `gliner` and `gliner2`
backends read a longer string in overlapping windows, so a name anywhere in it
is found at its exact offsets:

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
"phone number"; other built-in types send their name in lowercase words).

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
      "model": "dslim/bert-base-NER",
      "source": "hub", "model_id": "dslim/bert-base-NER",
      "revision": "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc", "pinned": true,
      "catalog": "vetted", "license": "MIT",
      "counters": {"scanned_whole": 812, "scanned_windowed": 14, "skipped_max_chars": 3,
                   "windows": 61, "windows_truncated": 0, "labels_dropped": 0,
                   "offsets_dropped": 0, "inline_calls": 0, "prefetch_misses": 0}
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
| `inline_calls` | strings the backend ran on the event loop itself, holding up every other request for that string's inference, instead of ahead of the redaction on the NER worker thread |
| `prefetch_misses` | strings a request's redaction did not find among its precomputed NER results (or found computed for detectors a reload has since replaced), run inline instead and counted in `inline_calls` too |

Each string a backend is handed counts once, under `scanned_whole`,
`scanned_windowed` or `skipped_max_chars`; with several backends each one
counts the strings it was handed. `unmatched_entities` lists the configured
entities no active backend can ever emit (the startup warning above), and
`model` the model each backend loaded. For the `gliner` and `hf` backends,
`source` says whether it came from the Hugging Face Hub (`hub`) or a local
folder (`local`), `model_id` names the Hub model (a folder's: the one its
`llm-redact-model.json` names, else `null`), `revision` the commit it loads
(the `[detection.ner.revisions]` pin, else the model catalog's; a folder's from
its `llm-redact-model.json`), `pinned` whether there is one, and `catalog` and
`license` what the model catalog records (`vetted`, `caution`, `restricted`,
or `null` for a model it does not list; see "The model catalog"). These six
are `null` for the spaCy, Presidio and Stanza backends, whose models are not
Hub snapshots.

The same counts are Prometheus counters (`llm_redact_ner_strings_total` by
backend and outcome, `llm_redact_ner_windows_total`,
`llm_redact_ner_windows_truncated_total`, `llm_redact_ner_labels_dropped_total`,
`llm_redact_ner_offsets_dropped_total`, `llm_redact_ner_inline_calls_total`,
`llm_redact_ner_prefetch_misses_total`;
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
