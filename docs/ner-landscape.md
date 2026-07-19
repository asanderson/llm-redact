# FOSS NER landscape: candidates for future optional backends

> **Update (v1.10.0):** the two top recommendations from this survey —
> **Stanza** (`stanza` extra, `backend = "stanza"`) and a generic
> **Hugging Face token-classification** backend (`hf` extra,
> `backend = "hf"`, any Hub NER model via `[detection.ner.models] hf = "..."`)
> — are now **shipped**. Stanza is the multilingual person-name backend (60+
> languages, no confidences); the HF backend gives power users any
> fine-tuned/multilingual checkpoint and emits confidences, so
> `score_threshold` applies. Both slot into `[detection.ner] backends = [...]`
> beside spaCy/GLiNER/Presidio and run concurrently. The survey below is kept
> as the rationale of record.

Research survey (2026-07) for expanding beyond the three shipped NER
backends (spaCy, GLiNER, Presidio). Method: evaluation against the
criteria that actually gate inclusion here — license, install footprint,
CPU latency class, language coverage, fit with the `Detector` protocol,
and whether the test suite can inject a fake model so CI never needs the
extra installed. Package facts should be re-verified against the
project's current releases before implementation (this survey is a
snapshot; the conclusions, not the version numbers, are the deliverable).

## The bar a new backend must clear

1. OSI-approved license compatible with the project's AGPL-3.0
   distribution (permissive and Apache-2.0 backends all qualify).
2. Meaningful capability the current three do not cover — for us that is
   principally **multilingual person-name recall** (spaCy small English
   model is the default; GLiNER is zero-shot but heavy; Presidio layers
   on spaCy).
3. Constructor-injectable model object (the `ner` test convention: unit
   tests inject fakes, the real model loads only behind the extra).
4. Latency compatible with per-string scanning under `max_chars`.

## Candidates

| Engine | License | Footprint | Latency class | Languages | Verdict |
|---|---|---|---|---|---|
| **Stanza** (Stanford NLP) | Apache-2.0 | torch + per-language models (~100-500 MB) | ~10-50 ms/string CPU | 70+ official language packages, consistent NER for ~a dozen | **RECOMMENDED next** — the strongest multilingual story; clean `Pipeline(lang, processors="tokenize,ner")` API that wraps naturally in the Detector protocol; model object injectable |
| **HF token-classification pipeline** (transformers) | Apache-2.0 (library; model licenses vary) | torch + transformers (GB-class, same as gliner) | model-dependent, ~10-100 ms CPU | any HF NER checkpoint (multilingual BERT variants, WikiNeural, ...) | **RECOMMENDED as a power-user backend** — anyone already paying the gliner extra's torch cost gets arbitrary-model flexibility nearly free; model id validated at startup like gliner's |
| **Flair** | MIT | torch + embeddings (~0.5-2 GB for best models) | slow on CPU (contextual string embeddings) | good for EN/DE + multilingual models | viable but third: excellent F1, painful CPU latency for a per-request proxy |
| **spaCy larger/multilingual pipelines** | MIT (code) / model licenses vary | tens-to-hundreds of MB | ~1-10 ms | per-language pipelines (de/fr/es/…, `xx` multi) | **already reachable today** — `[detection.ner] language` + per-backend `models` accept any installed spaCy pipeline; document rather than build |
| NLTK `ne_chunk` | Apache-2.0 | small | fast | EN only | rejected: dated accuracy well below spaCy small; no capability gained |
| DeepPavlov | Apache-2.0 | very heavy, TF/torch mix | slow | RU-strong, multilingual | rejected: heavy, moves fast, overlaps HF pipeline path |
| Apache OpenNLP | Apache-2.0 | JVM | n/a | several | rejected: Java runtime is out of the question for a pip extra |
| SpanMarker | Apache-2.0 | torch + transformers | ~HF-class | model-dependent | fold into the HF-pipeline backend rather than a dedicated extra |
| **LangExtract** (google/langextract) | Apache-2.0 | thin library + an LLM (Gemini API, or local via Ollama) | seconds/string (an LLM generation pass) | any (LLM-dependent) | **rejected — structural**: LLM-driven extraction inverts the threat model; see the section below |

## Recommendation

1. **Stanza** as the fourth backend (`stanza` extra) when multilingual
   demand materializes — it clears every bar and its per-language model
   downloads mirror the spaCy-model install step users already know.
   With P13.7's multi-backend config, it would slot in as one more name
   in `[detection.ner] backends` with its own `models` entry.
2. **A generic HF token-classification backend** (`hf-ner` extra) for
   power users — highest flexibility, no new dependency class beyond
   what gliner already pulls.
3. Do **not** add Flair/NLTK/DeepPavlov/OpenNLP — each fails latency,
   accuracy, or runtime-class bars without adding coverage the first two
   don't.

## LLM-based extractors (LangExtract and its class) — rejected as detectors

LangExtract (google/langextract) and similar prompted-extraction tools
find entities by sending the text to an LLM with few-shot instructions
and returning grounded spans. As request-path detectors they fail
structurally, not tunably:

1. **Threat-model inversion.** A detector runs on the RAW,
   pre-redaction request text. A cloud-backed extractor (Gemini API,
   LangExtract's default) would transmit every secret to a cloud LLM in
   order to decide what to hide from cloud LLMs — the exact leak this
   proxy exists to prevent. All five shipped backends run in-process on
   local weights precisely so the raw text never leaves the machine.
2. **Latency.** The local-model escape hatch (Ollama) fixes privacy but
   is a generation pass per scanned string — seconds, where bar 4 above
   is the ~1–100 ms class and the bench gates in-process overhead at
   ~1 ms. jsonwalk multiplies that per string in every request body.
3. **Nondeterminism.** `python -m llm_redact.bench --check` gates
   recall == 1.0 per rule, deterministically. Sampling-based extraction
   cannot pin a recall gate, and a detector whose recall varies silently
   is a silent-leak risk (the same reasoning that machine-checks
   prefilter literals).

LLM-based extraction is legitimate BESIDE the proxy, out of band:
auditing a document corpus for PII before ingestion, or growing
`bench/fp_corpus` — batch jobs where the text is already destined for an
LLM or never touches the request path.

## Running NER on non-English text

All five backends key their model on `[detection.ner] language` and the
per-backend `[detection.ner.models]` overrides. Three ways to cover another
language, cheapest first:

```toml
# 1. A language-specific spaCy pipeline (tens of MB, ~1-10 ms). Install the
#    model, then name it. entities uses spaCy's labels (PER for many non-EN
#    pipelines).
[detection.ner]
enabled = true
backend = "spacy"
language = "de"
model = "de_core_news_sm"     # uv run python -m spacy download de_core_news_sm
entities = ["PER"]

# 2. Stanza — one line per language, 60+ supported (pulls torch).
[detection.ner]
enabled = true
backend = "stanza"
language = "fr"               # python -c "import stanza; stanza.download('fr')"
entities = ["PERSON"]

# 3. Any multilingual Hugging Face NER checkpoint (pulls transformers+torch),
#    with score_threshold since it emits confidences.
[detection.ner]
enabled = true
backend = "hf"
score_threshold = 0.6
[detection.ner.models]
hf = "Davlan/xlm-roberta-base-ner-hrl"   # 10 languages
```

Backends compose: `backends = ["spacy", "stanza"]` runs an English spaCy
model and a per-language Stanza model at once, and same-span same-type hits
dedupe in overlap resolution. Entity labels differ across models — check the
model card and set `entities` to match (a wrong label silently detects
nothing).

This document is also the decision record for when a user asks for a
language the current backends serve poorly.
