# Student-model training recipe

How to fine-tune a small personal-data detector — a GLiNER checkpoint or a
DeBERTa-v3 token classifier — on license-clean data, so it can run inside the
proxy through the core `gliner` or `hf` NER backend. This is a recipe and a
skeleton (plan T43): `train_student.py plan` checks the data and writes the
run's data manifest and model card; **no step here downloads data or trains a
model**, and the `train` command deliberately refuses. Running the training
is an owner decision.

## 1. Plan the run

```bash
uv run python scripts/pii_corpus/train_student.py plan --name acme-pii-student \
  --base microsoft/deberta-v3-small \
  --sources nemotron,gretel-pii-masking-en-v1,privy,kiji,agent-corpus \
  --agent-corpus ~/.local/share/llm-redact/pii-corpus/train-share.verified.jsonl \
  --agent-eval ~/.local/share/llm-redact/pii-corpus/agent-eval.jsonl \
  --out ~/student-runs/acme-pii-student
```

`plan` refuses, with the reason:

- a source missing from the data manifest,
  [`training_sources.toml`](training_sources.toml) — every dataset with its
  license, attribution, pinned revision and lineage as read on its card
  (checked 2026-10-05);
- an evaluation-only source: PUPA (real prompts), MAPA (real legal
  decisions), CredData (real code);
- **OpenPII 1.5M** until the confirmation is recorded. Its card text says
  CC BY 4.0 while its metadata says `license: other`, and it does not say
  how the data was generated (its predecessors were generated with Llama
  models). Plan decision D9: it may be used for evaluation now and for
  training only after AI4Privacy confirms in writing. The confirmation is
  recorded by an owner commit that adds `confirmation = "REF"` (for example
  the confirmation's date and archive location) under `[sources.openpii]`
  in `training_sources.toml`, where review sees it; until then openpii is
  refused whatever is passed. Once recorded, `--openpii-confirmation REF`
  must repeat it exactly, and `REF` is written into the data manifest and
  the model card ("recorded in training_sources.toml");
- an `agent-corpus` file that overlaps the frozen `agent-eval` set
  (evaluation only: training on it would void every score measured on it):
  `--agent-eval FROZEN` names the frozen set (checked against its manifest)
  and is required with `agent-corpus`, and a training row is refused when
  its id, its text (SHA-256 after whitespace normalisation) or its
  generator run (teacher and seed) is one of the frozen set's — so the
  verified file the set was frozen from, a copy of the set without its
  manifest and a re-run with the eval run's seed are all refused; so is a
  file holding unverified rows. Train on a separately reviewed share
  (`review.py review` on a generator run with another seed); the data
  manifest records the frozen set's file name, SHA-256 and row count the
  share was checked against;
- a base model that is neither an allowed encoder
  (`microsoft/deberta-v3-small`, `microsoft/deberta-v3-base`, MIT) nor a
  GLiNER checkpoint in llm-redact's model catalog with an Apache-2.0 or MIT
  license, a pinned revision and a status other than restricted (the
  `knowledgator/gliner-pii-*-v1.0` sizes qualify; their training data is
  undisclosed, which the manifest records as lineage).

The run directory (outside every git work tree, mode 0700) then holds
`data-manifest.json` — every source's license, attribution, revision, split,
lineage and notes, the base model's license, revision and lineage, the
hyperparameters, and the OpenPII confirmation reference when one was given;
no text — and `MODEL_CARD.md`, filled from
[`MODEL_CARD_TEMPLATE.md`](MODEL_CARD_TEMPLATE.md): `plan` fills every
`{{field}}` from the manifest, and the parts marked `TO FILL` (the student's
license, hardware, time, evaluation, known limits) are completed after
training. Keep every statement a verifiable fact (plan D14): licenses,
revisions, datasets, measured numbers.

## 2. Data

| Source | License | What it adds | Train on |
|---|---|---|---|
| `nemotron` | CC BY 4.0 | English documents across 50+ industries, 55+ PII types | `train` (the bench scores `test`) |
| `gretel-pii-masking-en-v1` | Apache-2.0 | documents generated with Mistral NeMo, PII and PHI types | `train` |
| `gretel-synthetic-pii-finance-multilingual` | Apache-2.0 | financial documents in several languages | `train` |
| `privy` | MIT | PII inside JSON, SQL, HTML and XML payloads | `train` (the bench scores `test`) |
| `kiji` | Apache-2.0 | six languages, 26 types | `train` |
| `agent-corpus` | private | coding-agent artifacts, verified by hand | a verified training share, never the frozen set |
| `openpii` | see D9 | 1.5M rows, 30 languages | `train`, only with the confirmation |

Preparation, in order:

1. Download each source at its pinned revision (`huggingface_hub`, the
   `bench-data` extra) into a cache outside the repository, as the NER bench
   does.
2. Map every label to the placeholder types with llm-redact's own label
   policy: the bench's label maps where a dataset has one
   (`llm_redact.bench.datasets.<name>.SPEC.label_map`), otherwise
   `labels.normalize_label` and the default folds
   (`llm_redact.detection.labels`). Labels that fold to no type become
   outside (`O`); sensitive attributes (race, religion, politics, sexuality,
   gender; plan D10) are always outside.
3. Drop rows whose span text differs from the text at its offsets (as the
   bench skips them) and rows whose text equals, after whitespace
   normalisation, any row of an evaluation split (dedupe by SHA-256).
4. Hold out 5% of the training rows, stratified by source, as the dev set;
   seed every split (`seed` in the manifest).

## 3. Train

Hyperparameters recorded by `plan` (adjust and record the change):
3 epochs, learning rate 3e-5 with 10% warm-up, batch 16, 512 tokens.

- **Token classifier (`hf` backend)**: `transformers`
  `AutoModelForTokenClassification` with BIO tags over the label set, a
  fast tokenizer (the `hf` backend refuses a slow one), long rows split into
  overlapping 512-token windows with a stride of a quarter window — the same
  windowing the backend uses at inference.
- **GLiNER (`gliner` backend)**: the `gliner` package's training loop,
  prompting each type with the natural-language prompt the backend sends
  (`labels.gliner_prompt`), so inference and training see the same prompts.

Expected hardware and time (estimates scaled from the plan's figure of about
4 to 5 hours per epoch for DeBERTa-v3-base over 1.5M rows on a desktop RTX
4090-class GPU; not measured here): the license-clean sources above, a few
hundred thousand rows together, take roughly an hour per epoch for a base
model and about half that for `deberta-v3-small` or `gliner-pii-edge` on
such a GPU; a laptop GPU is several times slower; CPU-only training is not
practical. Record the real numbers in the model card.

## 4. Export

- Weights as safetensors (`save_pretrained(..., safe_serialization=True)`;
  the `hf` backend refuses pickle-only checkpoints by default), with
  `tokenizer.json`.
- A GLiNER student as a self-contained folder: tokenizer and
  `encoder_config` included, so loading never fetches a base model.
- ONNX: `optimum` for a token classifier; GLiNER's own conversion for a
  GLiNER model (the `gliner` backend can load `onnx/*.onnx`).
- A local folder carries an `llm-redact-model.json` sidecar (model id and
  revision) so llm-redact can identify it.

## 5. Evaluate

Run the NER bench on the exported checkpoint (docs/ner-bench.md):
`synthetic`, `nemotron:test`, `privy:test`, `agent-eval --path FROZEN`
(private: counts only), `--fp-corpus bench/fp_corpus` and `--latency`.
Fill the model card's evaluation tables from the reports.

## 6. Release checklist

- [ ] Data manifest and model card complete; every attribution line present
      (CC BY 4.0 sources require it).
- [ ] No OpenPII rows unless the D9 confirmation reference is recorded.
- [ ] No evaluation split, no frozen `agent-eval` row and no real-data
      dataset in the training data.
- [ ] Weights in safetensors (and ONNX), uploaded to the Hub; the release
      commit recorded as a 40-character revision.
- [ ] A catalog entry in `src/llm_redact/detection/model_catalog.py`
      (license, lineage, revision pin, recommended entities, window) with
      the admission decision under plan D11 (vetted or caution, numbers
      shown), and the docs table updated.
- [ ] Bench thresholds recorded for the new configuration
      (`bench/ner_thresholds.toml`, `bench/ner_ceilings.toml`).
- [ ] CHANGELOG entry.
