---
license: TO FILL (the student's license; at most as permissive as its inputs allow)
base_model: {{base_model}}
tags: [token-classification, pii, ner, llm-redact]
---

# {{model_name}}

A small personal-data detector for coding-agent traffic, trained to run inside
the llm-redact proxy through its `{{base_backend}}` NER backend. It detects
contextual values the regex rules cannot (names, street addresses, dates of
birth, usernames, account and document numbers) and is used **beside** the
rules, never instead of them. Planned {{date}}.

## Base model

| Field | Value |
|---|---|
| Model | `{{base_model}}` |
| Revision | `{{base_revision}}` |
| License | {{base_license}} |
| Lineage tags | {{base_lineage}} |

## Training data

| Source | Hub id | Revision | License | Lineage |
|---|---|---|---|---|
{{training_data_rows}}

OpenPII 1.5M written confirmation: {{openpii_confirmation}}.

Never trained on: the NER bench's evaluation splits (Nemotron-PII `test`,
privy `test`, OpenPII `validation`), the frozen agent-traffic evaluation set,
or any real-data dataset (PUPA, MAPA, CredData).

### Attribution

{{attributions}}

## Training procedure

{{hyperparameters}}

- Hardware: TO FILL (GPU model, count, memory)
- Wall-clock time per epoch: TO FILL
- Training code: scripts/pii_corpus/TRAINING.md at llm-redact commit TO FILL

## Labels

Placeholder types emitted: `PERSON`, `ADDRESS`, `DATE_OF_BIRTH`, `PASSPORT`,
`DRIVER_LICENSE`, `USERNAME`, `ACCOUNT_NUMBER` (and `EMAIL`, `PHONE`, which the
regex rules own). Sensitive attributes (race, religion, politics, sexuality,
gender) are not labels of this model.

## Evaluation (TO FILL after training)

Measured with `python -m llm_redact.bench.ner` (docs/ner-bench.md) on the
exported checkpoint at its pinned revision.

| Dataset | PERSON recall | ADDRESS recall | leak rate | over-redaction | structured regressions |
|---|---|---|---|---|---|
| synthetic | | | | | |
| nemotron (test) | | | | | |
| privy (test) | | | | | |
| agent-eval (private, counts only) | | | | | |

| False positives (`--fp-corpus bench/fp_corpus`) | Latency p50 / p95, 500 characters (`--latency`, CPU named) |
|---|---|
| | |

Admission: vetted only with an OSI-approved license, clean lineage,
PERSON recall ≥ 0.85 and leak rate ≤ 0.15 on the synthetic corpus, ≤ 1 false
positive per 50 KB of agent-traffic negatives and p50 ≤ 100 ms for 500
characters on the reference CPU; otherwise caution, with these numbers shown.

## Intended use and limits

- Intended: detecting personal data in text a coding agent sends to an LLM
  provider, inside llm-redact, as an addition to its regex rules.
- Not intended: a sole control, legal or compliance decisions, sensitive
  attribute inference, or text far from the training distribution.
- Known limits: TO FILL from the evaluation (languages, formats, types with
  low recall).
