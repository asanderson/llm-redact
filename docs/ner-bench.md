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
| `--out DIR` | Write `report.md` and `report.json` there instead of printing the report. |
| `--thresholds PATH` | The gate file (default `bench/ner_thresholds.toml`). |
| `--check` | Exit 1 when a floor or ceiling is crossed, or when the run has no recorded entry. |
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
(quote such keys: `[hf-default."openpii:train"]`).

| Key | Meaning |
|---|---|
| `recall = { TYPE = floor }` | overlap-typed recall floors per type |
| `exact_recall = { TYPE = floor }` | exact-match recall floors per type |
| `leak_max` | character-leak rate ceiling |
| `over_redaction_max` | over-redaction rate ceiling |
| `structured_regressions_max` | regex-type spans NER may cost (default 0) |
| `recorded`, `note` | when and with which models and revisions the baseline was measured |

A run with no entry fails `--check`, and so does a floor on a type the run
holds no gold spans of (it cannot be checked). Unknown keys are refused.

### Recording a baseline

Run the bench on the configuration and dataset without `--check`, read the
report, and add an entry: each recall floor at the measured value minus
0.05, each ceiling at the measured value plus 0.05 (a tighter over-redaction
ceiling is fine when the measured rate is near zero), `recorded` set to the
date and `note` naming the model ids and revisions. Commit the entry with
the change that motivated it.

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
