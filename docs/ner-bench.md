# The NER bench: measuring models on labelled text

`python -m llm_redact.bench.ner` scores the whole detection pipeline — the
regex rules, the configured NER backends, overlap resolution, no allowlist —
on a labelled dataset, and with `--check` gates the result against recorded
floors and ceilings. It answers "what does this model add, what does it
miss, what does it wrongly redact, and does it break a rule?" for a given
configuration.

It is separate from the deterministic gate, `python -m llm_redact.bench
--check` (recall == 1.0 per rule on generated positives, exact counts on the
false-positive corpus), which is unchanged: a rule either matches its own
grammar or it is broken, while a model is measured statistically. See
[assurance.md](assurance.md#statistical-gates-for-ner-models) for how the two
gates differ.

![Flowchart of the NER bench: datasets that are never committed (generated, downloaded at pinned revisions, or local files) pass through adapters and label maps into the full detection pipeline of a bench configuration; metrics feed metadata-only reports and the --check gate against the recorded thresholds and ceilings, which the ner-models CI job runs on every pull request and the weekly ner-eval workflow reports; beside it the unchanged deterministic gate on the vendored negatives](diagrams/ner-bench.png)

*Static diagram. [Mermaid source](diagrams/ner-bench.mmd).*

## Running it

```bash
uv run python -m llm_redact.bench.ner --list-datasets
uv run python -m llm_redact.bench.ner --config my-ner.toml --out /tmp/ner-report   # the synthetic corpus
uv run python -m llm_redact.bench.ner --config my-ner.toml --check   # gate against bench/ner_thresholds.toml
```

| Option | Meaning |
|---|---|
| `--config PATH` | The llm-redact config to score (parsed like `serve`'s; `[detection.ner] enabled = true` is required). Models load exactly as at proxy startup. |
| `--name NAME` | The config's name in the thresholds file (default: the config file's stem, so `bench/configs/hf-default.toml` is `hf-default`). |
| `--dataset NAME[:SPLIT]` | What to score (default `synthetic`); `--list-datasets` prints every dataset with its splits, license and attribution. |
| `--limit N` | Score at most N samples (default 2000; `0` = all). |
| `--seed N` | Seed of generated datasets (same seed, same corpus). |
| `--language CODE` | Keep only rows in that language (datasets that record one: `openpii`). The thresholds key becomes `NAME@CODE`. |
| `--data-dir DIR` | The local checkout a dataset is read from (`creddata`: a CredData directory after its own download script ran). |
| `--path FILE` | The local file a dataset is read from (`agent-eval`: the frozen set written by `scripts/pii_corpus/review.py freeze`). |
| `--cache-dir DIR` | Where downloaded datasets are kept (default `${XDG_CACHE_HOME:-~/.cache}/llm-redact/bench-datasets`; refused inside a git work tree). |
| `--out DIR` | Write `report.md` and `report.json` there instead of printing the report. |
| `--thresholds PATH` | The gate file (default `bench/ner_thresholds.toml`). |
| `--check` | Exit 1 when a floor or ceiling is crossed, or when the run has no recorded entry. |
| `--fp-corpus DIR` | Count NER's detections on a negatives corpus (`bench/fp_corpus`) instead of scoring a dataset (below). |
| `--latency` | Time NER instead of scoring a dataset (below). `--latency-iterations N` (default 20) and `--many-small-strings N` (default 20,000) size the run. |
| `--ceilings PATH` | The `--fp-corpus` gate file (default `bench/ner_ceilings.toml`). |
| `--dump-errors PATH` | Write misses, leaks, false positives and regressions WITH their text to PATH (below). |
| `--allow-real-data-dump` | Let `--dump-errors` write the text of a real-data dataset. |

Exit status: 0 done (or gate passed), 1 gate failed, 2 an input problem (a
missing config, NER off, an unknown dataset, a malformed thresholds file).

## What a dataset's labels mean

Each dataset maps its own labels to what is scored:

- a **placeholder type** (`PERSON`, `ADDRESS`, `EMAIL`, …): scored per type
  and counted by the leak metric;
- **leak**: counted only by the type-agnostic character-leak metric — a
  value the proxy should not let through although no placeholder type
  matches it (a country-specific tax number);
- **not scored**: a city, a plain date. A detection over it is neither a
  false positive nor over-redaction, and missing it is not a leak;
- **not personal data**: the dataset marks the value as non-PII, so a
  detection over it is a false positive and over-redaction.

A label missing from the map is not scored and is counted in the report, so
a dataset that changes its labels shows up.

Parts of one name or address that a dataset labels separately (`GIVENNAME`
and `SURNAME`, `BUILDINGNUM` and `STREET`) and that map to the same type are
scored as ONE span, joined by the rule that joins a model's parts
(separated by one or two spaces, tabs or no-break spaces): a model that
finds "Jane Doe" whole is right, as the proxy would redact it.

## Metrics

Per type, two matchings:

- **exact** — a detection with the gold span's start, end and type;
- **overlap-typed** — a gold span is found when a detection of its type
  overlaps it, and a detection is right when it overlaps a gold span of its
  type. Models often draw a span a word wider or narrower than the
  annotation; overlap-typed scores count those as found.

For each, precision, recall and F1. A detection that overlaps no typed gold
span but only a leak or not-scored one is **neutral**: not counted for
precision (reported as a count).

Type-agnostic:

- **character-leak rate** — gold characters (typed or leak) that no
  detection covers, over all such characters. A wrong type still covers the
  value (the vault replaces it), so this is the number that says how much
  personal data would reach the provider.
- **over-redaction rate** — detected characters outside every gold span,
  over the characters outside every gold span: how much ordinary text the
  model hides from the provider.

Per type: the **character-leak rate of a type** — characters of that
type's gold spans that no detection (of any type) covers, over those
characters. The type-agnostic rate also counts every type the
configuration does not ask for (always leaked), so it is diluted when a
configuration requests only some of a dataset's types; a gate on the
requested types reads the per-type rate.

**Structured-regression check.** The bench also runs the same configuration
with NER off. Every gold span of a regex rule's type (an email, a phone
number, an IBAN) that the rules alone find exactly must still be found
exactly with NER on. A model drawing a wider span over a structured value
wins overlap resolution (the longest span wins) and costs the rule its
match; such a span is counted as a regression.

Reports are metadata only: types, counts, rates, label names, model ids and
dataset attributions — never text from a dataset.

## The gate: `bench/ner_thresholds.toml`

One table per config name and dataset, `[<config name>.<dataset>]`; the
dataset part is its name, plus `:SPLIT` when the split is not the default
and `@LANGUAGE` with `--language` (quote such keys:
`[hf-default."openpii:train"]`, `[hf-default."openpii@de"]`).

| Key | Meaning |
|---|---|
| `recall = { TYPE = floor }` | overlap-typed recall floors per type |
| `exact_recall = { TYPE = floor }` | exact-match recall floors per type |
| `leak_max` | character-leak rate ceiling, over all gold characters |
| `type_leak_max = { TYPE = ceiling }` | character-leak rate ceilings per type |
| `over_redaction_max` | over-redaction rate ceiling |
| `structured_regressions_max` | regex-type spans NER may cost (default 0) |
| `recorded`, `note` | when and with which models and revisions the baseline was measured |

A run with no entry fails `--check`, and so does a floor or a
`type_leak_max` on a type the run holds no gold spans of (it cannot be
checked). Unknown keys are refused.

### Recording a baseline

Run the bench on the configuration and dataset without `--check`, read the
report, and add an entry: each recall floor at the measured value minus
0.05, each ceiling at the measured value plus 0.05 (a tighter over-redaction
ceiling is fine when the measured rate is near zero: the recorded baselines
use twice the measured rate), `recorded` set to the date and `note` naming
the model ids and revisions and the measured values. Commit the entry with
the change that motivated it. Floors round down and ceilings round up, to
two decimals.

## False positives on agent traffic: `--fp-corpus`

```bash
uv run python -m llm_redact.bench.ner --config my-ner.toml --fp-corpus bench/fp_corpus --check
```

`bench/fp_corpus` holds real-world and authored NEGATIVES — files the
deterministic gate pins to exact regex counts in its `MANIFEST.toml`. Four
of them look like coding-agent traffic and contain no personal data:
`synthetic_agent_tool_results.json` (tool results as JSON text),
`synthetic_git_log.txt`, `synthetic_ci_log.txt` and
`synthetic_python_module.py`, full of tool names that are also surnames
(Jenkins, Jackson, Hudson). With `--fp-corpus`, the bench scans every
corpus file in chunks of whole lines of at most 2,000 characters (the size
of a message or a tool result; a whole file in one string would exceed
NER's `max_chars` and be skipped) and counts what NER ADDS: detections the
full pipeline makes and the same configuration's rules alone do not (a
model span displacing a rule's match counts). The report lists them per
file and type, with the total per 100 KB of corpus.

`--check` compares against `bench/ner_ceilings.toml` (outside the corpus
directory, whose every file is scanned):

```toml
[hf-default]
recorded = "2026-10-05"
note = "dslim/bert-base-NER at <revision>"
per_100kb_max = 2.0              # NER hits per 100 KB of the whole corpus

[hf-default."rfc_excerpt.txt"]
PERSON = 4                       # maximum count of that type in that file
```

A config without a section fails; a file or type not listed allows no NER
detection at all; a ceiling naming a file the corpus does not hold fails
(a stale entry). Some files hold real names on purpose (the RFC's authors,
the characters of *Alice's Adventures in Wonderland*): a model finding them
is right, and their ceilings say so. A baseline's count ceilings are the
measured counts plus 5%, rounded up and at least one more than measured, so
the same model on another CPU does not fail on one borderline span;
`per_100kb_max` likewise.

## Latency: `--latency`

```bash
uv run python -m llm_redact.bench.ner --config my-ner.toml --latency --out /tmp/ner-latency
```

Times detection with the configured NER backends, per string, at 50, 500,
2,000 and 10,000 characters of synthetic-corpus text: each NER backend's
own `detect` (one row per backend and model) and the full pipeline (every
detector plus overlap resolution). It also redacts, end to end, the
many-small-strings body the regex latency bench uses (20,000 short chat
messages): NER runs once per string, which is what such a request costs the
event loop. The report gives p50 and p95 in milliseconds and names the CPU
model (Linux `/proc/cpuinfo`, else what the platform reports), the Python
version and the platform — every number depends on them.

It is report-only by default. With `--check`, optional ceilings in the
thresholds file apply; without an entry the run passes and says nothing was
recorded:

```toml
[hf-default.latency]
p50_ms = { "500" = 100.0 }      # full pipeline, per string length
p95_ms = { "2000" = 400.0 }
many_small_ms = 30000.0         # p50 of the many-small body
recorded = "2026-10-05"
note = "CPU model, model ids and revisions"
```

## CI

Two GitHub Actions workflows run real models; the regular test jobs never
do (the suite fakes every model, and the `real_model` tests are deselected
by default).

**`ner-models`** (a job of `.github/workflows/ci.yml`, on pull requests,
pushes to `main` and the weekly CI schedule):

1. installs the `hf`, `gliner`, `gliner2` and `bench-data` extras with
   `scripts/ner_ci_env.sh`: every package at the version `uv.lock` pins,
   each wheel checked against the lock's sha256, except that torch is the
   PyTorch CPU wheel of the locked version, checked against the CPU wheels'
   SHA-256 that `scripts/cpu_torch.py` records (the locked PyPI wheel
   brings CUDA libraries no runner uses, so `cpu_torch.py` takes it and
   torch's GPU-only packages out of the export; it is the same recipe the
   `-ner` image and the `airgap` job use, docs/dependencies.md), then
   `cpu_torch.py check` refuses any CUDA or triton package and
   `uv pip check` the result's consistency;
2. restores the Hugging Face cache, keyed on the model catalog and the
   configurations below, and runs `llm-redact models pull --config` on each
   of them, so every model is fetched at its catalog pin (a GLiNER
   checkpoint's pinned base model included);
3. runs `pytest -m real_model` with
   `LLM_REDACT_TEST_REAL_MODELS_REQUIRED=1`, which turns a real-model test
   that would skip (a model not pulled, an extra missing) into a failure: a
   green job means every one of them ran. Then it removes
   `openai/privacy-filter` from the Hugging Face cache (below), whatever the
   tests did;
4. for every `bench/configs/*.toml`, runs the bench three times with
   `--check`: the synthetic corpus against `bench/ner_thresholds.toml`, the
   negatives corpus (`--fp-corpus bench/fp_corpus`) against
   `bench/ner_ceilings.toml`, and `--latency` with a many-small body of
   1,000 strings instead of 20,000, which would take an hour or more per
   model on a runner (report only until a `[<config>.latency]` entry
   exists). Every run completes before the step
   fails, so one crossed gate does not hide another report; the reports are
   uploaded as the `ner-report` artifact.

The models load offline in steps 3 and 4, as a proxy with
`allow_download = false` loads them: only step 2 contacts the Hub.

**`ner-eval`** (`.github/workflows/ner-eval.yml`, weekly and by
`workflow_dispatch`): the same environment, then one job per bench
configuration (its matrix lists every `bench/configs/*.toml`), each pulling
its own model and scoring it on 2,000-row slices of OpenPII 1.5M
(`validation`) and Nemotron-PII (`test`), downloaded at their pinned
revisions into a cached directory. One job per configuration keeps each
within its 120-minute timeout: the slowest, `gliner2-fastino`, needs an
estimated 40 minutes for both slices on the reference CPU. It is report only — no
`--check` — until baselines for those datasets are recorded; each job
uploads its reports as the `ner-eval-<configuration>` artifact and writes
them to the run's summary.

| Configuration | What it scores | Gated by |
|---|---|---|
| `bench/configs/hf-default.toml` | the `hf` backend's default model (`dslim/bert-base-NER`) at its catalog pin, default entities (`PERSON`) | `[hf-default.synthetic]`, `[hf-default]` |
| `bench/configs/gliner-default.toml` | the `gliner` backend's default model (`urchade/gliner_small-v2.1`, assembled with its pinned base model) at its catalog pin, default entities | `[gliner-default.synthetic]`, `[gliner-default]` |
| `bench/configs/gliner-knowledgator-edge.toml`, `-edge-onnx.toml` | `knowledgator/gliner-pii-edge-v1.0` at its catalog pin, PyTorch weights and the int8 ONNX export, asked for `PERSON`, `ADDRESS`, `DATE_OF_BIRTH`, `USERNAME` and `ACCOUNT_NUMBER` | `[gliner-knowledgator-edge.synthetic]`, `[gliner-knowledgator-edge]`, and the same for `-edge-onnx` |
| `bench/configs/gliner-knowledgator-base.toml`, `-base-onnx.toml` | `knowledgator/gliner-pii-base-v1.0`, the same way | `[gliner-knowledgator-base.synthetic]`, `[gliner-knowledgator-base]`, and the same for `-base-onnx` |
| `bench/configs/hf-openmed-pii-small.toml`, `hf-ettin-68m-nemotron-pii.toml` | `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` and `kalyan-ks/ettin-68m-nemotron-pii` at their catalog pins, asked for the same five types | `[hf-openmed-pii-small.synthetic]`, `[hf-openmed-pii-small]`, and the same for `hf-ettin-68m-nemotron-pii` |
| `bench/configs/gliner2-fastino.toml` | `fastino/gliner2-privacy-filter-PII-multi` at its catalog pin (the `gliner2` backend), asked for the same five types with `score_threshold = 0.9` | `[gliner2-fastino.synthetic]`, `[gliner2-fastino]` |
| `bench/configs/manual/hf-openai-privacy-filter.toml` | not run by CI: `openai/privacy-filter` at its catalog pin, Viterbi-decoded, asked for `PERSON`, `ADDRESS` and `ACCOUNT_NUMBER`; measured by hand (below) | `[hf-openai-privacy-filter.synthetic]`, `[hf-openai-privacy-filter]`, checked by hand |
| `tests/real_model_configs/*.toml` | not scored: the other models the `real_model` tests load (the `gliner2` default, a Knowledgator ONNX checkpoint, `openai/privacy-filter` for a smoke test), pulled by `ner-models` | — |

A configuration added to `bench/configs/` is pulled, tested and gated by
`ner-models` with no workflow change, and fails it until its baselines are
recorded; `tests/test_ner_ci.py` fails until `ner-eval.yml`'s matrix lists
it. The recorded entries say when, on which model revisions and with which
measured values they were taken; with the default `PERSON` entities, the
synthetic corpus's `ADDRESS`, `DATE_OF_BIRTH`, `USERNAME` and
`ACCOUNT_NUMBER` values are not requested, so their characters (about 38%
of the corpus's gold characters) count toward the type-agnostic leak rate
(the configurations of other models request the contextual types their
models are recommended for that the corpus labels). A ceiling on that rate alone would let
`PERSON` leakage grow about fivefold before failing, so each entry also
carries a `type_leak_max` for every entity its configuration requests.

**Configurations measured by hand.** `bench/configs/manual/` holds
configurations of models too slow for a CI runner; neither workflow reads
that directory (`ner-models`' globs are not recursive, and `ner-eval`'s
matrix lists only `bench/configs/*.toml`). `openai/privacy-filter`
(`bench/configs/manual/hf-openai-privacy-filter.toml`) takes about a second
per 500-character string on a 4-core CPU, and its latency run alone takes
twenty minutes. Its baselines are recorded like any other's and checked the
same way, by hand:

```bash
uv run --no-sync llm-redact models pull --config bench/configs/manual/hf-openai-privacy-filter.toml
uv run --no-sync python -m llm_redact.bench.ner --config bench/configs/manual/hf-openai-privacy-filter.toml --check
uv run --no-sync python -m llm_redact.bench.ner --config bench/configs/manual/hf-openai-privacy-filter.toml --fp-corpus bench/fp_corpus --check
```

`ner-models` still loads the model, for a smoke test only:
`tests/real_model_configs/hf-openai-privacy-filter.toml` asks for what the
manual configuration asks for, the job pulls it with the others, and
`tests/test_hf_bioes.py` builds it offline through the proxy's own build
(the catalog pin, the BIOES tagger and the `viterbi_calibration.json` it
lists), checks that the decoder got that file's biases and keeps a span
whole where the greedy reading would cut it, and reads four short strings:
a name, an address and an account number in a sentence, the same sentence
inside a JSON string, an empty and a blank string. Its 2.8 GB are pulled again on every run instead of being
saved with the model cache: GitHub keeps 10 GB of caches per repository
and evicts the least recently used beyond that, and this entry, saved anew
whenever the catalog or a configuration changes, would push the other
jobs' caches out. The job removes it right after the `real_model` tests,
so the bench runs have the disk space and the saved cache never holds it,
and frees disk before anything else by removing preinstalled toolchains it
does not use (`df -h /` in the log shows what is left).

To reproduce the job locally (it creates a CPython 3.13 venv; Linux,
x86_64 or aarch64, where the PyTorch CPU index has a `+cpu` wheel; the
script runs the venv's `bin/python`, so it does not run on Windows, which
is unsupported), install as it does; the script replaces `.venv`, so run it in a
checkout you do not develop in, and the models go to the Hugging Face
cache. When `uv.lock` moves torch to another version, the script refuses
to run until `CPU_WHEELS` in `scripts/cpu_torch.py` records that
version's CPU wheels (a test fails until then too):

```bash
scripts/ner_ci_env.sh
uv run --no-sync llm-redact models pull --config bench/configs/hf-default.toml
uv run --no-sync pytest -m real_model
uv run --no-sync python -m llm_redact.bench.ner --config bench/configs/hf-default.toml --check
```

## Seeing the errors: `--dump-errors`

The report never shows text. To look at what a model missed or wrongly
found, `--dump-errors PATH` writes one JSON line per error — kind (`miss`,
`leak`, `false_positive`, `regression`), type, offsets, the span text and
40 characters on each side — to PATH, created with mode 0600 (an existing
file is truncated and set to 0600; a symlink is refused). PATH must be
outside every git work tree, so dataset text cannot be committed by
accident. A dataset marked as real data (real prompts, real code, real
documents) is refused unless `--allow-real-data-dump` is also given. Delete
the file when you are done.

## Datasets

| Name | What it is | License | Real data |
|---|---|---|---|
| `synthetic` (default) | Agent-traffic-shaped text generated from the seed at run time (below). | generated (part of llm-redact) | no |
| `openpii` | [OpenPII 1.5M](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m): synthetic PII text in 30 languages; splits `validation` (default), `train`. | CC BY 4.0 — credit "Ai4Privacy / Ai Suisse SA" | no |
| `nemotron` | [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII): synthetic English documents (forms, invoices, emails, notes) across 50+ industries; splits `test` (default), `train`. | CC BY 4.0 — by NVIDIA Corporation | no |
| `privy` | [beki/privy](https://huggingface.co/datasets/beki/privy): synthetic PII inside JSON, SQL, HTML and XML payloads generated from OpenAPI specifications — the closest public analogue of tool-call bodies; splits `test` (default), `dev`, `train` and their `-large` variants. | MIT — Benjamin Kilimnik | no |
| `pupa` | [PUPA](https://huggingface.co/datasets/Columbia-NLP/PUPA): 901 real user prompts (WildChat) with the personal-data strings an LLM extracted from each; splits `all` (default), `tnb`, `new`. | MIT — Columbia NLP (PAPILLON, Li et al. 2024) | **yes** |
| `mapa` | [MAPA](https://huggingface.co/datasets/joelniklaus/mapa): human-annotated EUR-Lex legal text in 21 languages; splits `test` (default), `validation`, `train`; `--language` filters it. | CC BY 4.0 — de Gibert Bonet et al. (LREC 2022), converted by Joel Niklaus and Veton Matoshi | **yes** |
| `creddata` | [CredData](https://github.com/Samsung/CredData): labelled (obfuscated) secrets in lines of real code and configuration, read from a local checkout (`--data-dir`); splits `all` (default), `src`, `test`, `other`. | labels Apache-2.0; each code file keeps its project's license | **yes** |
| `agent-eval` | The private, hand-verified agent-traffic evaluation set: coding-agent artifacts with invented personal data, read from a local file (`--path`); split `all` (below). | private, not distributed | **yes** |
| `rules` | The regex bench's generated positives and decoys for every built-in rule, built from the seed at run time. Structured values only: it shows what a model costs the rules (structured regressions, over-redaction). | generated (part of llm-redact) | no |

### The synthetic corpus

`synthetic` (`src/llm_redact/bench/ner_corpus.py`) is generated from the
seed at every run and never committed. Its 1,200 samples take turns over
twelve contexts. Six carry gold spans, in the shapes coding-agent traffic
has: prose, chat turns, JSON tool results carried as text, code comments,
log lines and git output, with `PERSON` (a first and a last name),
`ADDRESS` (number, street, suffix), `DATE_OF_BIRTH` (four date formats),
`USERNAME`, `ACCOUNT_NUMBER` (8–12 digits) and, for the
structured-regression check, `EMAIL` and `PHONE` values the regex rules own.
The other six are hard negatives with no gold at all — UUIDs, commit hashes,
CamelCase identifiers (many built from tool names that are also surnames,
such as Jenkins or Jackson), file paths, stack traces and timestamps — so
every detection there counts as over-redaction. Names, streets and handles
come from small embedded lists combined at random and describe no real
person. The regex rules alone find exactly the corpus's `EMAIL` and `PHONE`
values and nothing else (pinned by `tests/test_ner_corpus.py`).

### Published datasets

Published datasets are downloaded at run time with
`huggingface_hub.hf_hub_download` at a pinned revision (a commit hash) into
the cache directory and never committed; they need the `bench-data` extra
(`uv sync --extra bench-data`: `huggingface_hub` and `pyarrow`). They are
used for EVALUATION only — the bench never trains on anything. Rows whose
gold spans do not match their text are skipped and counted in the report,
never repaired; each report repeats the dataset's license and attribution.
A `--limit` slice reads the first rows of the split in file order.

| Dataset | Hub id at revision | Card checked | Label map |
|---|---|---|---|
| `openpii` | `ai4privacy/pii-masking-openpii-1.5m` at `a785eb528e28be2693c3718a27e066970de5dadb` | 2026-10-05 | `GIVENNAME`, `SURNAME` → `PERSON`; `STREET`, `BUILDINGNUM` → `ADDRESS`; `PASSPORTNUM` → `PASSPORT`; `DRIVERLICENSENUM` → `DRIVER_LICENSE`; `EMAIL` → `EMAIL`; `TELEPHONENUM` → `PHONE`; `CREDITCARDNUMBER` → `CREDIT_CARD`; `SOCIALNUM`, `TAXNUM`, `IDCARDNUM` → leak; `DATE`, `AGE`, `TITLE`, `GENDER`, `SEX`, `CITY`, `ZIPCODE` → not scored |
| `nemotron` | `nvidia/Nemotron-PII` at `b70ffaf5ff39e079776134c5bf4381f00a9fd1ed` | 2026-10-05 | `first_name`, `last_name` → `PERSON`; `street_address` → `ADDRESS`; `date_of_birth` → `DATE_OF_BIRTH`; `user_name` → `USERNAME`; `account_number` → `ACCOUNT_NUMBER`; `email` → `EMAIL`; `phone_number` → `PHONE`; `ssn` → `SSN`; `credit_debit_card` → `CREDIT_CARD`; `ipv4` → `IPV4`; `ipv6` → `IPV6`; `password`, `api_key` → `SECRET`; the other 40 labels → not scored |
| `privy` | `beki/privy` at `dc137a6a976f6b5bb8768e9bb51ec58df930ccd1` | 2026-10-05 | `PER`, `PERSON` → `PERSON`; `US_PASSPORT` → `PASSPORT`; `US_DRIVER_LICENSE` → `DRIVER_LICENSE`; `US_BANK_NUMBER` → `ACCOUNT_NUMBER`; `EMAIL_ADDRESS` → `EMAIL`; `PHONE_NUMBER` → `PHONE`; `US_SSN` → `SSN`; `IBAN_CODE` → `IBAN`; `CREDIT_CARD` → `CREDIT_CARD`; `PASSWORD` → `SECRET`; `US_ITIN`, `IP_ADDRESS` → leak; `O` → not personal data; the other 18 labels → not scored |
| `pupa` | `Columbia-NLP/PUPA` at `9981b49b6ced0033988a224b6712895ebf119294` | 2026-10-05 | every PII unit → leak (the units carry no type) |
| `mapa` | `joelniklaus/mapa` at `bbb2a0157b760465002fd12a61af81b475cd387a` | 2026-10-05 | fine-grained `FAMILY NAME`, `INITIAL NAME` → `PERSON`; the other 20 fine-grained labels → not scored |

Facts found when the adapters were checked against the data (2026-10-05):

- **OpenPII 1.5M** (`data/validation.jsonl`, `data/train.jsonl`; fields
  `source_text`, `privacy_mask` with `value`/`start`/`end`/`label`,
  `language`). The card lists 19 labels; the rows also carry a few labels
  it does not list (`TIME`, `CURRENCY`, `ACCOUNTNUM`, `AMOUNT`, `COUNTRY`,
  `ORGANISATION`, `URL`: about 80 spans in 3,926 rows sampled across the
  validation file). They are not in the label map, so reports list them as
  unmapped and do not score them. The card's metadata says
  `license: other` with `license_name: cc-by-4.0`; its text grants CC BY
  4.0 and asks for the credit "Ai4Privacy / Ai Suisse SA". The validation
  file is about 1 GB, downloaded whole on first use. Its rows are grouped
  by language: the first 2,000 (a default `--limit` slice) hold Chinese,
  Japanese, Vietnamese, Tagalog, Indonesian, Malay, Korean and 236 English
  rows (checked 2026-10-07), so an English-only model is measured mostly
  on languages it was not trained on; `--language en` scores English rows
  only.
- **Nemotron-PII** (`data/test-00000-of-00001.parquet`,
  `data/train-00000-of-00001.parquet`, about 150 MB each). The `spans`
  column is a string holding a Python-literal list (single quotes, not
  JSON), read with `ast.literal_eval`, which evaluates literals only. Each
  file holds 100,000 rows — 50,000 records twice, once with locale `us`
  (the first half) and once `intl` (the second half), with different
  text — so a `--limit` slice reads US-locale rows only. About 7% of rows
  have a span whose text differs from the text at its offsets (76 of the
  first 1,000); they are skipped and counted.
- **privy** (`privy-dataset.zip`, about 300 MB, holding
  `{train,dev,test}-{small,large}.json`, each one JSON array of rows with
  `full_text` and `spans`). The repository's loading script is never run:
  the archive is read directly and its arrays are parsed a row at a time.
  The `small` files label people `PER`, the `large` ones `PERSON`; both
  label non-PII payload values `O`, scored as not personal data, so a
  detection on them is over-redaction. Empty values (an empty field in a
  payload, start equal to end) are labelled too and are dropped.
- **PUPA** (`PUPA_TNB.csv`, 237 rows; `PUPA_New.csv`, 664 rows). Real
  prompts: `--dump-errors` needs `--allow-real-data-dump`. The
  `pii_units` column holds lowercased, `||`-separated strings an LLM
  extracted — no types, no offsets — so the adapter marks every whole-word,
  case-insensitive occurrence of each unit in its prompt and scores them
  with the leak metric only; per-type scores mean nothing here. A row with
  a unit that does not occur in its prompt is skipped (77 of 901 at this
  revision).
- **MAPA** (`train.jsonl`, `validation.jsonl`, `test.jsonl`; `language`,
  `tokens`, `coarse_grained`, `fine_grained` IOB tags). The rows hold
  tokens without whitespace information, so the text is the tokens joined
  with single spaces (`Article 3 ( 1 )`). The coarse `PERSON` tag also
  covers roles, professions and nationalities, so only the fine-grained
  family-name and initial tags are scored; the data carries no given-name,
  street, email or identifier tags at this revision, although the card
  lists some. EUR-Lex decisions name real parties, so the dataset is marked
  as real data.

### CredData

[CredData](https://github.com/Samsung/CredData) measures the secret rules on
real code. The repository ships only metadata — `meta/<repo>.csv`, one row
per suspicious line with `FilePath`, `LineStart`, `LineEnd`, `GroundTruth`
(`T` true, `F`/`X` false), `ValueStart`, `ValueEnd` and `Category` — plus
its own `download_data.py`, which fetches the labelled files from their
source repositories and obfuscates the credential values. The bench never
runs or vendors any of it; prepare a checkout yourself, outside this
repository (it fetches hundreds of repositories, several GB):

```bash
git clone https://github.com/Samsung/CredData ~/datasets/CredData
cd ~/datasets/CredData   # its README: Linux, Python 3.10 recommended
python download_data.py
uv run python -m llm_redact.bench.ner --config my-ner.toml --dataset creddata \
  --data-dir ~/datasets/CredData
```

Each distinct line range of the metadata becomes one sample (those lines,
joined by newlines); every `T` row's value is gold and is scored by the
character-leak metric only. The splits `src`, `test` and `other` keep the
files of that directory. Lines whose rows are all `F`/`X` hold look-alikes, so a
detection there counts as over-redaction. Rows whose file is missing,
whose lines or value offsets fall outside the file, or that are true but
carry no offsets are skipped and counted. The metadata format was checked
on 2026-10-05 at commit `0b1940e171725ad8937311120b191602608a4801`.
Licensing: the labels are Apache-2.0; every code file keeps its own
project's license (the checkout's `license` directory holds them by
repository), which is why nothing from it is ever committed here. The code
is real, so `--dump-errors` needs `--allow-real-data-dump`.

### The agent-traffic evaluation set

`agent-eval` scores the shapes no public dataset has: tool-call JSON
arguments and results, diffs, logs, config files, commit messages, test
fixtures, SQL and tracebacks, with invented personal data, and hard
negatives full of name-like identifiers that are not people. It is built out
of band with the dev-only tooling in
[`scripts/pii_corpus/`](../scripts/pii_corpus/README.md): a local
Apache-2.0 teacher served by Ollama generates tagged rows (`generate.py`),
a human accepts, edits or rejects every row following
[`GUIDELINES.md`](../scripts/pii_corpus/GUIDELINES.md) (`review.py
review`), and `review.py freeze` writes the frozen set with a manifest
beside it. The set is private: it is never committed or published, and the
bench reads it by path:

```bash
uv run python -m llm_redact.bench.ner --config my-ner.toml --dataset agent-eval \
  --path ~/.local/share/llm-redact/pii-corpus/agent-eval.jsonl
```

The frozen file holds one JSON object per line (`id`, `text`, `spans` with
`start`/`end`/`type`, `teacher`, `prompt_id`, `seed` and the `review`
record); every span type is a placeholder type, scored as itself. The
adapter first checks `FILE.manifest.json` (format
`llm-redact-agent-eval/1`): a file whose SHA-256 differs from the
manifest's is refused — a frozen set changes only by freezing it again.
Rows that are malformed or whose spans fall outside their text are skipped
and counted. A row's context in the report is its prompt id (a hard
negative's as `<artifact>-negative`). The text is generated and verified by
hand rather than real people's data, but the set is private and marked as
real data, so `--dump-errors` needs `--allow-real-data-dump`.
