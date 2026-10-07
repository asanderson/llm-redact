# FOSS NER landscape: candidates for future optional backends

> **Update:** the two top recommendations from this survey —
> **Stanza** (`stanza` extra, `backend = "stanza"`) and a generic
> **Hugging Face token-classification** backend (`hf` extra,
> `backend = "hf"`, any Hub NER model via `[detection.ner.models] hf = "..."`)
> — are now **shipped**. Stanza is the multilingual person-name backend (60+
> languages, no confidences); the HF backend gives power users any
> fine-tuned/multilingual checkpoint and emits confidences, so
> `score_threshold` applies. Both slot into `[detection.ner] backends = [...]`
> beside spaCy/GLiNER/Presidio and run concurrently. The survey below is kept
> as the rationale of record. The lineup has grown since: a GLiNER2 backend
> (`gliner2` extra), ONNX weights for GLiNER, and a model catalog of
> PII-specific open models, each measured by the llm-redact bench and given a
> verdict in "PII models measured by the llm-redact bench" below — the current
> decision record.

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
5. Weight provenance: the model loads at a pinned revision (and, for a
   GLiNER checkpoint that ships no tokenizer, a pinned base model), as
   safetensors or ONNX weights (a pickle only through torch's
   `weights_only` loader or an explicit opt-in), without running code from
   the model's repository, and from local files with downloads off, so an
   air-gapped host can use it ([detection.md](detection.md#model-sources-pinned-revisions-downloads-and-pickle-weights)).
6. Long-text coverage: a string longer than the model's window is read in
   overlapping windows instead of being silently truncated, and every string
   a backend does not read (longer than `max_chars`) is counted and
   surfaced in `/status`, `/metrics` and the posture output
   ([detection.md](detection.md#how-ner-runs)).

The Hub backends (`hf`, `gliner`, `gliner2`) meet items 5 and 6 for every
model they run, catalogued or not: explicit file names, no repository code,
safetensors unless the operator opts in, and local files with downloads off;
windows and counters. A pin comes from the catalog or the operator, and a
model with neither loads at the newest cached revision with a `doctor` WARN.
spaCy, Presidio and Stanza read each string whole. Items 1 to 4 and the
measured bars below are what a catalog status records.

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
   With the multi-backend config, it would slot in as one more name
   in `[detection.ner] backends` with its own `models` entry.
2. **A generic HF token-classification backend** (`hf-ner` extra) for
   power users — highest flexibility, no new dependency class beyond
   what gliner already pulls.
3. Do **not** add Flair/NLTK/DeepPavlov/OpenNLP — each fails latency,
   accuracy, or runtime-class bars without adding coverage the first two
   don't.

## PII models measured by the llm-redact bench

Dated decision record (2026-10-07) of the open PII models measured with the
NER bench ([ner-bench.md](ner-bench.md)) and the catalog status each was
given ([detection.md](detection.md#the-model-catalog)). Every number comes
from one shared machine: an `Intel(R) Xeon(R) Processor @ 2.80GHz` with 4
cores (other work ran beside some of the accuracy runs, never beside a
latency run), torch 2.13.0 (CPU build), transformers 5.10.1, gliner 0.2.28,
models loaded offline at their catalog pins, `score_threshold` 0.5 unless a
row says otherwise. Accuracy numbers are deterministic for a model and
revision; latency is this machine's.

**The admission bar ("vetted").** An OSI-approved
weights license and no known restricted training-data lineage (undisclosed
training data is stated as a fact, not disqualifying); on the synthetic
corpus PERSON recall ≥ 0.85 and a character-leak rate ≤ 0.15; at most one
false positive per 50 KB on the agent-traffic negatives; p50 ≤ 100 ms for a
500-character string. A model that misses a bar stays "caution", with its
numbers in its catalog reason. How each is measured here:

- each model runs with its recommended entities that the synthetic corpus
  labels (`PERSON`, `ADDRESS`, `DATE_OF_BIRTH`, `USERNAME`,
  `ACCOUNT_NUMBER`; fewer when the model recommends fewer), so the
  character-leak rate covers what the model is asked for plus the corpus's
  `EMAIL` and `PHONE` values, which the rules find. A model also
  recommended for `PASSPORT` and `DRIVER_LICENSE`, which the corpus does
  not label, is not asked for them, and its catalog reason says so: its
  false-positive count is for the narrower request;
- false positives are the NER detections the rules alone do not make in the
  four agent-traffic files of `bench/fp_corpus`
  (`synthetic_agent_tool_results.json`, `synthetic_git_log.txt`,
  `synthetic_ci_log.txt`, `synthetic_python_module.py`; 19,808 bytes),
  scaled to 50 KB (51,200 bytes). "Whole corpus" is every file of
  `bench/fp_corpus` per 100 KB, real names included (a novel, an RFC's
  authors), so it is not a false-positive rate;
- latency is the whole pipeline's p50 per 500-character string
  (`--latency`).

### Verdicts at a glance

Every model the bench measured, one row per configuration, with the license
and training data its catalog entry states and the status it was given
(`llm-redact models list` prints status and license; the catalog tables are
in [detection.md](detection.md#the-model-catalog)). Recall is `PERSON`
overlap-typed recall on the synthetic corpus with the exact-match figure in
brackets; leak, false positives and latency are as defined above. Each
model's own section below has the full tables and the published-data results.
"Good for" says what the numbers support; none of it is a guarantee on
text the bench did not see.

| Model (backend), as measured | License; training data | Status | PERSON recall (exact); character leak | Agent-traffic FP per 50 KB; p50, 500 chars | Good for |
|---|---|---|---|---|---|
| `urchade/gliner_small-v2.1` (`gliner`, the default), asked for `PERSON` | Apache-2.0; `urchade/pile-mistral-v0.1` (Apache-2.0) | vetted | 1.000 (1.000); 0.324 | 52; 189 ms | More names on published data than the Knowledgator models (OpenPII 0.708, Nemotron 0.874, against 0.485 and 0.706 for `-edge`), at about half their precision. Its leak counts the types it was not asked for |
| `dslim/bert-base-NER` (`hf`, the default), asked for `PERSON` | MIT; CoNLL-2003 (Reuters news) | vetted | 0.974 (0.951); 0.398 | 0; 176 ms | English person names with no detection in the agent-traffic files; `PERSON` only (it also tags organizations and places). Its leak counts the types it was not asked for |
| `knowledgator/gliner-pii-edge-v1.0`, PyTorch, five types | Apache-2.0; the card does not name them | caution | 0.992 (0.992); 0.082 | 109; 93 ms | Names and addresses (ADDRESS 0.994) at the lowest latency here; `USERNAME` draws most of its false positives |
| same, int8 ONNX | same | caution | 0.771 (0.757); 0.270 | 16; 97 ms | Fewer false positives than the PyTorch weights (16 against 109); finds almost no dates of birth (0.053) and few usernames (0.273) |
| `knowledgator/gliner-pii-base-v1.0`, PyTorch, five types | same | caution | 0.966 (0.966); 0.021 | 8; 234 ms | The lowest leak and the fewest agent-traffic false positives of the Knowledgator runs, at 234 ms |
| same, int8 ONNX | same | caution | 0.964 (0.964); 0.029 | 13; 182 ms | The same coverage on the synthetic corpus at 182 ms; loses recall on published data (OpenPII 0.321) |
| `knowledgator/gliner-pii-small-v1.0`, `-large-v1.0` | same | caution | not measured | — | — |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` (`hf`), five types | Apache-2.0; `microsoft/deberta-v3-small` (MIT) fine-tuned on `nvidia/Nemotron-PII` (CC BY 4.0) | caution | 1.000 (1.000); 0.048 | 18; 136 ms | Names, addresses, account numbers and usernames (ADDRESS 1.000, ACCOUNT_NUMBER 0.991, USERNAME 0.894); dates of birth are its weak type (0.509). The better `hf` default candidate, below |
| `kalyan-ks/ettin-68m-nemotron-pii` (`hf`), five types | MIT; `jhu-clsp/ettin-encoder-68m` (MIT) fine-tuned on `nvidia/Nemotron-PII` (CC BY 4.0) | caution | 0.984 (0.866); 0.143 | 16; 127 ms | Names and addresses; it finds few dates of birth (0.377) and account numbers (0.243: most carry labels that are not folded into that type) |
| `openai/privacy-filter` (`hf`), `PERSON`, `ADDRESS`, `ACCOUNT_NUMBER` | Apache-2.0; the card does not name them | caution | 0.994 (0.994); 0.120 | 23; 1,108 ms | Names, addresses (0.930) and account numbers (1.000) from a 1.5B-parameter mixture of experts, at about a second per string |
| `fastino/gliner2-privacy-filter-PII-multi` (`gliner2`), five types, threshold 0.9 | Apache-2.0; 4,910 synthetic texts the card says GPT-5.4 generated | caution | 0.998 (0.998); 0.009 | 26; 453 ms | All five types with the lowest leak measured, in seven languages; recall on real text is lower at its 0.9 default (OpenPII English 0.786) |
| `fastino/gliner2-base-v1` (`gliner2`, the default) | Apache-2.0; multi-domain datasets (its card) | caution | not measured | — | — |
| `urchade/gliner_multi_pii-v1`, `-medium-v2.1`, `-multi-v2.1` (`gliner`) | Apache-2.0; `urchade/pile-mistral-v0.1` and `urchade/synthetic-pii-ner-mistral-v1` (both Apache-2.0) | vetted | not measured | — | Catalogued and pinned without a bench measurement of their own |

Eleven further catalog entries are "restricted" (a non-commercial, gated or
non-OSI-approved license, training data with restrictive terms, a Qwen
backbone, or a model that needs code from its repository to load): none was
measured, none is suggested, and a configured one logs the catalog's facts at
startup (the table in [detection.md](detection.md#the-model-catalog)).

**What each model finds, by type.** Overlap-typed recall of each type, with
the type's character leak in brackets, on the synthetic corpus (the values
recorded in `bench/ner_thresholds.toml`). A cell is empty where the
configuration did not ask for the type.

| Model, as measured | `PERSON` | `ADDRESS` | `DATE_OF_BIRTH` | `USERNAME` | `ACCOUNT_NUMBER` |
|---|---|---|---|---|---|
| Knowledgator `-edge`, PyTorch | 0.992 (0.007) | 0.994 (0.008) | 0.447 (0.563) | 0.629 (0.380) | 0.930 (0.068) |
| Knowledgator `-edge`, int8 ONNX | 0.771 (0.224) | 0.646 (0.334) | 0.053 (0.958) | 0.273 (0.705) | 0.896 (0.096) |
| Knowledgator `-base`, PyTorch | 0.966 (0.034) | 1.000 (0.000) | 0.991 (0.010) | 0.879 (0.106) | 1.000 (0.000) |
| Knowledgator `-base`, int8 ONNX | 0.964 (0.037) | 0.994 (0.006) | 0.982 (0.018) | 0.788 (0.181) | 1.000 (0.000) |
| OpenMed-PII Small 44M | 1.000 (0.000) | 1.000 (0.007) | 0.509 (0.511) | 0.894 (0.072) | 0.991 (0.007) |
| ettin-68m-nemotron-pii | 0.984 (0.054) | 1.000 (0.012) | 0.377 (0.799) | 0.856 (0.124) | 0.243 (0.744) |
| `openai/privacy-filter` | 0.994 (0.005) | 0.930 (0.070) | | | 1.000 (0.000) |
| Fastino GLiNER2-PII, threshold 0.9 | 0.998 (0.002) | 0.994 (0.026) | 1.000 (0.000) | 0.924 (0.046) | 1.000 (0.000) |

Recall without precision flatters a type that draws many false positives:
`USERNAME` precision on the same runs is 0.13 for Knowledgator `-edge` (0.12
int8), 0.59 for `-base` (0.61 int8), 0.75 for OpenMed and ettin and 0.52 for
Fastino, and username detections are most of the agent-traffic false
positives of the Knowledgator and OpenMed runs below. `PASSPORT` and
`DRIVER_LICENSE` are requested from the models that list them (the catalog's
prompts and labels), but no bench configuration asks for them and the
synthetic corpus labels neither, so no recorded baseline covers them
(OpenPII labels both; the weekly evaluation is report-only).

The two default models, measured the same way for comparison (their
recorded baselines request `PERSON` only):

| Configuration | PERSON recall (exact) | Precision | Character leak | Over-redaction | Agent-traffic FP per 50 KB | Whole corpus per 100 KB | p50, 500 chars |
|---|---|---|---|---|---|---|---|
| `urchade/gliner_small-v2.1`, `PERSON` (`gliner-default`) | 1.000 (1.000) | 0.639 | 0.324 | 0.015 | 52 | 448 | 189 ms |
| `urchade/gliner_small-v2.1`, five types | 0.791 (0.791) | 0.916 | 0.049 | 0.050 | 34 | 457 | — |
| `dslim/bert-base-NER`, `PERSON` (`hf-default`) | 0.974 (0.951) | 0.913 | 0.398 | 0.003 | 0 | 210 | 176 ms |

### Knowledgator GLiNER-PII (`gliner`)

`knowledgator/gliner-pii-{edge,small,base,large}-v1.0` (Apache-2.0,
Knowledgator with Wordcab, created 2025-09-24; the card does not name the
training data). Self-contained checkpoints with int8 ONNX exports. Measured
with the prompts the card lists ("name", "location address", "dob",
"username", "account number"); before this measurement llm-redact sent the
generic prompts ("person", …), with which `-edge` (int8) found 0.27 of the
names instead of 0.77.

| Configuration | PERSON recall (exact) | Precision | Character leak | Over-redaction | Agent-traffic FP per 50 KB | Whole corpus per 100 KB | p50, 500 chars |
|---|---|---|---|---|---|---|---|
| `-edge`, PyTorch (`gliner-knowledgator-edge`) | 0.992 (0.992) | 0.961 | 0.082 | 0.100 | 109 | 138 | 93 ms |
| `-edge`, int8 ONNX (`gliner-knowledgator-edge-onnx`) | 0.771 (0.757) | 0.934 | 0.270 | 0.050 | 16 | 10 | 97 ms |
| `-base`, PyTorch (`gliner-knowledgator-base`) | 0.966 (0.966) | 0.921 | 0.021 | 0.055 | 8 | 76 | 234 ms |
| `-base`, int8 ONNX (`gliner-knowledgator-base-onnx`) | 0.964 (0.964) | 0.915 | 0.029 | 0.028 | 13 | 93 | 182 ms |
| `-edge`, PyTorch, `PERSON` only | 1.000 (1.000) | 0.862 | 0.368 | 0.006 | 0 | 166 | — |
| `-base`, PyTorch, `PERSON` only | 1.000 (1.000) | 0.734 | 0.330 | 0.007 | 8 | 141 | — |
| `-edge`, int8 ONNX, threshold 0.3 | 0.968 (0.929) | 0.868 | 0.118 | 0.141 | 114 | 200 | — |
| `-base`, int8 ONNX, threshold 0.3 | 0.998 (0.998) | 0.854 | 0.003 | 0.114 | 57 | 303 | — |

On published data (2,000-row slices in file order: OpenPII 1.5M `validation`,
whose first 2,000 rows hold eight languages — Chinese, Japanese,
Vietnamese, Tagalog, Indonesian, Malay, Korean and 236 English rows —
and, with `--language en`, its first 2,000 English rows; Nemotron-PII
`test`, US-locale English documents, 142 rows skipped for spans that
disagree with their text):

| Configuration | OpenPII PERSON recall / precision | OpenPII character leak | Nemotron PERSON recall / precision | Nemotron character leak |
|---|---|---|---|---|
| `urchade/gliner_small-v2.1`, `PERSON` (`gliner-default`) | 0.708 / 0.788 | 0.617 | 0.874 / 0.538 | 0.317 |
| `-edge`, PyTorch | 0.485 / 0.848 | 0.362 | 0.706 / 0.890 | 0.140 |
| `-edge`, int8 ONNX | 0.243 / 0.765 | 0.618 | 0.358 / 0.834 | 0.365 |
| `-base`, PyTorch | 0.551 / 0.946 | 0.459 | 0.682 / 0.943 | 0.192 |
| `-base`, int8 ONNX | 0.321 / 0.957 | 0.520 | 0.669 / 0.961 | 0.207 |

The int8 exports lose more on published data than on the synthetic corpus
(`-base` int8: OpenPII PERSON recall 0.32 against 0.55 for PyTorch). On
these slices the default `urchade/gliner_small-v2.1` (asked for `PERSON`
only, so its leak counts every other type) finds more names than
Knowledgator's models (Nemotron 0.874 against 0.706 for `-edge`), at about
half their precision (0.538 against 0.890).

**Verdict: caution** for `-edge` and `-base`: both clear the license,
recall and leak bars, and `-edge` the latency bar, but neither clears one
false positive per 50 KB of agent traffic with its recommended entities
(most are `USERNAME` detections of identifiers and handles; `-base` is also
above 100 ms). `-small` and `-large` were not measured. The bench configs
`bench/configs/gliner-knowledgator-{edge,edge-onnx,base,base-onnx}.toml`
are gated in CI. On these numbers `-base` (int8 ONNX for speed) leaks less
and draws fewer agent-traffic false positives than the default
`urchade/gliner_small-v2.1` asked for the same five types, at the same
latency class; the `gliner` default model is unchanged here. Asked for
`PERSON` only, `-edge` (PyTorch) would clear every bar if the leak bar were read for the requested types
(its `PERSON` leak is 0.000; the all-type leak counts the corpus's other
types, never requested); the bar is read here over every type the corpus
labels.

### OpenMed-PII Small 44M and ettin-68m-nemotron-pii (`hf`)

`OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` (Apache-2.0;
microsoft/deberta-v3-small, MIT, fine-tuned on nvidia/Nemotron-PII, CC BY
4.0; 54 entity types; BIO tags, first sub-token labelled; its card's
sequence length of 384 tokens is the catalog window) and
`kalyan-ks/ettin-68m-nemotron-pii` (MIT; jhu-clsp/ettin-encoder-68m, MIT,
fine-tuned on the same dataset; 55 entity types). Both fold every
recommended type from their labels; their sensitive-attribute labels
(gender, race or ethnicity, religious belief, political view, sexuality)
are never folded.
Measured after two fixes found here: the `hf` backend leaves out the blank a
SentencePiece or byte-level BPE token's offsets take in before its word
(OpenMed's " Jane" + " Doe" were two placeholders that swallowed the
spaces; exact `PERSON` recall rose from 0.00 to 0.67), and — since 1.12.0 —
it decodes a BIO tagger whose tokenizer does not mark word pieces word by
word ([detection.md](detection.md), "BIO taggers without word-piece marks"),
in float32 (ettin's weights are stored in bfloat16). The numbers below are
2026-10-07's re-measurement; the latencies were taken while other jobs ran
on the same 4-core machine, so they are indicative only.

| Configuration | PERSON recall (exact) | Precision | Character leak | Over-redaction | Agent-traffic FP per 50 KB | Whole corpus per 100 KB | p50, 500 chars |
|---|---|---|---|---|---|---|---|
| OpenMed, five types (`hf-openmed-pii-small`) | 1.000 (1.000) | 0.937 | 0.048 | 0.006 | 18 | 191 | 136 ms |
| OpenMed, `PERSON` only | 1.000 (1.000) | 0.937 | 0.385 | 0.002 | 0 | 186 | — |
| ettin, five types (`hf-ettin-68m-nemotron-pii`) | 0.984 (0.866) | 0.938 | 0.143 | 0.005 | 16 | 68 | 127 ms |
| ettin, `PERSON` only | 0.984 (0.866) | 0.938 | 0.413 | 0.003 | 0 | 63 | — |

Against the token-level aggregation they were first measured with (same
corpora; ettin then in bfloat16): exact `PERSON` recall rose from 0.674 to
1.000 (OpenMed) and from 0.030 to 0.866 (ettin), exact recall rose for every
type, and the character leak fell (OpenMed 0.051 to 0.048, ettin 0.158 to
0.143; ettin's `PERSON` leak 0.077 to 0.054, `USERNAME` 0.180 to 0.124,
OpenMed's `ACCOUNT_NUMBER` 0.036 to 0.007). What got worse: OpenMed's
`USERNAME` recall (0.924 to 0.894, leak 0.069 to 0.072: four usernames whose
first piece the model leaves untagged or unsure), ettin's `DATE_OF_BIRTH` and
`USERNAME` recall (0.404 to 0.377, 0.902 to 0.856), precision (OpenMed
`PERSON` 0.950 to 0.937, ettin `PERSON` 0.968 to 0.938) and over-redaction
(OpenMed 0.005 to 0.006, ettin 0.003 to 0.005): a word is now redacted whole
where a piece of it was before. ettin's lost recall is recall the
token-level decoder was credited for one piece of a value: it found ettin's
six lost usernames by 1 to 6 of their 8 to 13 characters and its three lost
dates of birth by 1 or 2 of theirs, and the characters of those types left
uncovered fell (`USERNAME` 241 to 166 of 1,340, `DATE_OF_BIRTH` 1,169 to
1,153 of 1,443); OpenMed's is real (five usernames lost, one gained, five
more characters uncovered).
The negatives corpus draws fewer detections (OpenMed 551 to 514, ettin 286
to 183). Re-measured after the decoder's review fixes (a span grows over the
characters the model tags with it, so a password's symbols or a MAC
address's colons are never sent upstream between its parts; scripts written
without spaces are read character by character; a word longer than the
windows' overlap is read in parts): the same numbers, with one more OpenMed
`USERNAME` detection (precision 0.752 to 0.747).

How a word is labelled was chosen on these numbers. Cutting words at every
punctuation mark, as BERT's tokenizer does, left the parts of
"1985-03-12", "j.doe" and "dev_jo42" to pieces OpenMed never labels (its
`USERNAME` leak rose to 0.135): words are cut only at blanks, quotes,
brackets and value delimiters. Reading a word by its most confidently tagged
piece leaked least for ettin (`PERSON` 0.054 against 0.085 for its first
piece), but for OpenMed, which labels first pieces only, its untrained later
pieces turned seven hyphenated reference numbers of the national-id
negatives into account numbers: the model catalog records which pieces a
model labels (`piece_labels`), and only ettin is read by its most confident
piece. A span continues across a comma and a space where the model
continues it ("March 3, 1985"; OpenMed's exact `DATE_OF_BIRTH` recall 0.307
without, 0.465 with).

On published data (the slices of the Knowledgator section; Nemotron-PII is
the training distribution of both models, so its numbers favour them):

| Configuration | OpenPII PERSON recall / precision (leak) | OpenPII English PERSON recall / precision (leak) | Nemotron PERSON recall / precision (leak) |
|---|---|---|---|
| `dslim/bert-base-NER`, `PERSON` (`hf-default`) | 0.527 / 0.757 (0.389) | 0.911 / 0.994 (0.097) | 0.888 / 0.965 (0.092) |
| OpenMed, `PERSON` only | 0.829 / 0.793 (0.227) ¹ | 0.989 / 0.995 (0.030) ¹ | — |
| OpenMed, five types | 0.829 / 0.793 (0.166) ¹ | 0.980 / 0.994 (0.017) | 0.995 / 0.996 (0.006) ¹ |
| ettin, `PERSON` only | 0.581 / 0.922 (0.452) ¹ | 0.951 / 0.997 (0.141) ¹ | — |
| ettin, five types | 0.580 / 0.922 (0.428) ¹ | 0.948 / 0.993 (0.092) ² | 0.987 / 0.988 (0.032) ¹ |

(Leak in brackets is the `PERSON` characters left uncovered; the
five-type rows cover some names under another type. ¹ Measured with the
token-level aggregation, before word-by-word decoding; not re-run. Re-run on
OpenPII English, the five-type rows were 0.989 / 0.995 (0.028) for OpenMed
and 0.951 / 0.997 (0.138) for ettin before. ² With one structured
regression: a value a rule finds that a wider span took over.)

**ettin's labels.** The model labels every sub-word piece `B-…`; read word
by word, "Zbigniew Brzezinski" is one value, no longer six. Its account
numbers still leak (synthetic `ACCOUNT_NUMBER` recall 0.24, 74% of their
characters leaked): the model labels most of them `customer_id`,
`unique_id`, `credit_debit_card` or `phone_number`, which are not folded
into `ACCOUNT_NUMBER`, so decoding cannot recover them. Some of its
probabilities are low (the pieces of a surname scoring 0.3 to 0.4), so the
default `score_threshold` of 0.5 drops some names it finds.

**Verdict: caution** for both. OpenMed clears the license, recall and leak
bars but not one false positive per 50 KB of agent traffic with its five
types (all seven are `USERNAME` detections, six of them in the tool-result
JSON) nor 100 ms per 500 characters (136 ms); ettin now clears the recall
and leak bars too (`PERSON` recall 0.98, leak 0.143) but not the
false-positive bar (16 per 50 KB) nor latency (127 ms). Bench configs
`bench/configs/hf-openmed-pii-small.toml` and
`bench/configs/hf-ettin-68m-nemotron-pii.toml` are gated in CI.

**The `hf` default.** Compared as the default
configuration runs (`entities = ["PERSON"]`), OpenMed-PII Small 44M finds
more names than `dslim/bert-base-NER` everywhere it was measured outside its
own training distribution — synthetic corpus 1.000 against 0.974 (leaking
none of the name characters against 0.037), OpenPII English 0.989 against
0.911, OpenPII's first 2,000 rows 0.829 against 0.527 (both measured before
word-by-word decoding) — at equal false positives (none in the
agent-traffic files for either; 186 against 210 detections per 100 KB of
the whole negatives corpus) and similar latency (175 against 176 ms when
measured side by side; 136 ms in the busier re-measurement). Read word by
word, its exact-span `PERSON` recall is 1.000 (0.951 for
`dslim/bert-base-NER`; 0.674 before, when its spans were often a word wider
or narrower than the annotation). The bench therefore supports OpenMed-PII
Small 44M as a better default for the `hf` backend than
`dslim/bert-base-NER` (better `PERSON` recall at equal false positives and
equal latency). The default is **unchanged in 1.x**, and the switch is
planned together with raw-entity folding in the next major release (2.0.0):
until raw entities fold, a configured `entities = ["PER"]` would match
nothing from OpenMed, which never emits `PER`. The weights are 566 MB
(`dslim/bert-base-NER`: 433 MB), and redistributing them carries the
Nemotron-PII attribution.

### OpenAI Privacy Filter (`hf`)

`openai/privacy-filter` (Apache-2.0; created 2026-04-17; 1.5B parameters,
50M active per token — a sparse mixture of experts — with banded attention;
the card does not name the training data). It loads on the locked
transformers 5.10.1 without remote code (transformers knows the model type
from 5.6.0; the catalog records that minimum and every build checks it).
Only `model.safetensors` (2.8 GB), `config.json`, `tokenizer.json`,
`tokenizer_config.json` and `viterbi_calibration.json` are fetched, never
the `original/` copy or the ONNX exports. Eight span labels, BIOES-tagged;
`private_person`, `private_address`, `account_number` fold into `PERSON`,
`ADDRESS` and `ACCOUNT_NUMBER`, the types it is recommended for; its
`secret` label is left to the anchored secret rules (its card lists
over-redaction of hashes, placeholders and sample credentials among its
failure modes), and `private_date` and `private_url` are not folded.

| Configuration | PERSON recall (exact) | Precision | Character leak | Over-redaction | Agent-traffic FP per 50 KB | Whole corpus per 100 KB | p50, 500 chars |
|---|---|---|---|---|---|---|---|
| Viterbi decoding (`bench/configs/manual/hf-openai-privacy-filter.toml`) | 0.994 (0.994) | 0.768 | 0.120 | 0.046 | 23 | 27 | 1,108 ms |
| greedy decoding (the calibration left out) | 0.994 (0.994) | 0.717 | 0.120 | 0.047 | 57 | — | — |

Latency per string: 261 ms at 50 characters, 2.1 s at 2,000, 19.5 s at
10,000; the 1,000-string many-small body took 57 s. These latencies were
measured with the weights in their stored precision (bfloat16), as is the
catalog's p50; a float32 load measured about 30% faster on this CPU (738
against 1,025 ms per 500 characters for the model alone) and takes twice
the memory (about 6 GB of RAM for the 2.8 GB of weights), and since 1.12.0
the `hf` backend loads every model in float32. Its
`ACCOUNT_NUMBER` precision is 0.46: it calls many numeric identifiers
account numbers (23 in the national-id log of the negatives corpus). On
OpenPII's first 2,000 English rows: `PERSON` recall 0.898, precision 0.933
(`PERSON` leak 0.082; character leak 0.145 over the types asked for and
the rest), with two structured regressions (a wider span over a value a
rule finds); the other published slices were not run (each would take
hours on this CPU).

**Verdict: caution.** It clears the license, recall and leak bars (the leak
counts the corpus's `DATE_OF_BIRTH` and `USERNAME` values, which it is not
asked for) but neither the false-positive bar nor the latency bar: about a
second per 500-character string on this CPU, five times the other models.
Its bench configuration is measured by hand (`bench/configs/manual/`), not
in CI; the CI `ner-models` job pulls the model only for a `real_model`
smoke test on a few short strings (since 2026-10-07).
The constrained Viterbi decoder with the repository's calibration is what
the `hf` backend uses for it; greedy decoding finds the same names with more
than twice the false positives. At the pinned revision the calibration's
default operating point sets all six transition biases to 0, so the
difference is the decoder's span constraint (a span opens with `B` or `S`,
continues with `I` of its entity and closes with `E` or `S`), not a bias.

### Fastino GLiNER2-PII (`gliner2`)

`fastino/gliner2-privacy-filter-PII-multi` (Apache-2.0; created
2026-05-10; GLiNER2, 205M parameters, on `microsoft/mdeberta-v3-base`;
fine-tuned on 4,910 synthetic texts its card says GPT-5.4 generated;
English, French, Spanish, German, Italian, Portuguese, Dutch; 42 labels).
Self-contained (`config.json`, `encoder_config/config.json`, tokenizer,
`model.safetensors`, 1.2 GB). The card's label spellings for the contextual
types — `person`, `street_address`, `date_of_birth`, `passport_number`,
`drivers_license_number`, `username`, `account_number` — are its catalog
prompts. Its card reports a precision of 0.35–0.37 on SPY and suggests
raising the threshold for names; measured at three thresholds (the
threshold does not change the model's work, so one latency run covers
them):

| Configuration | PERSON recall (exact) | Precision | Character leak | Over-redaction | Agent-traffic FP per 50 KB | Whole corpus per 100 KB | p50, 500 chars |
|---|---|---|---|---|---|---|---|
| five types, threshold 0.9 (`gliner2-fastino`) | 0.998 (0.998) | 0.955 | 0.009 | 0.022 | 26 | 360 | 453 ms |
| five types, threshold 0.7 | 0.998 (0.998) | 0.882 | 0.004 | 0.051 | 72 | 445 | — |
| five types, threshold 0.5 | 1.000 (1.000) | 0.825 | 0.003 | 0.091 | 96 | 503 | 453 ms |
| `PERSON` only, threshold 0.5 | 1.000 (1.000) | 0.459 | 0.320 | 0.049 | 80 | 503 | — |

On published data (the slices of the Knowledgator section):

| Configuration | OpenPII PERSON recall / precision (leak) | OpenPII English PERSON recall / precision (leak) | Nemotron PERSON recall / precision (leak) |
|---|---|---|---|
| five types, threshold 0.9 | 0.614 / 0.997 (0.298) | 0.786 / 0.997 (0.197) | 0.759 / 0.853 (0.186) |
| `PERSON` only, threshold 0.5 | — | 0.977 / 0.884 (0.025) | — |

(Leak in brackets is the `PERSON` characters left uncovered.) The 0.9
threshold that holds its synthetic-corpus recall costs it a fifth of
OpenPII's English names (0.786 against 0.977 at 0.5), where the default
`hf` and `gliner` models find 0.911 and 0.922: the threshold trades recall
on real text for false positives on agent traffic.

**Verdict: caution.** It clears the license and lineage bar (its training
data is synthetic text the card says GPT-5.4 generated: stated as a fact),
recall and leak bars (at 0.5 its leak, 0.003, ties the lowest measured
here), but neither the
false-positive bar nor the latency bar (453 ms per 500 characters; its
1,000-string body takes 206 s). `bench/configs/gliner2-fastino.toml`
(threshold 0.9) is gated in CI. Update 2026-10-07: the catalog records 0.9
as this model's default threshold, so a configuration naming it runs at 0.9
unless it sets `[detection.ner] score_threshold` (which wins, for every
backend); the bench configuration now relies on that default, and its
baselines stay the 0.9 measurement above.

### What the measurements decided

- **No PII model reached "vetted".** Each of them misses at least one
  admission bar (false positives on agent traffic, latency, or both), so
  each is "caution" with its numbers in its catalog reason. The leak bar is
  read over every type the synthetic corpus labels, so a configuration that
  asks for `PERSON` only (the default) shows a large character leak that is
  the unrequested types, not its names.
- **The `hf` default stays `dslim/bert-base-NER` in 1.x.** OpenMed-PII Small
  44M finds more names at equal false positives and similar latency (above);
  the switch is planned together with raw-entity folding in the next major
  release (2.0.0), and will be listed in its migration notes.
- **The `gliner` default stays `urchade/gliner_small-v2.1`.** On published
  data it finds more names than the Knowledgator models (Nemotron-PII 0.874
  against 0.706 for `-edge`, OpenPII 0.708 against 0.485), at about half
  their precision, while `-base` leaks less and draws fewer agent-traffic
  false positives on the synthetic corpus than the default asked for the same
  five types (int8: leak 0.029 against 0.049, 13 against 34 false positives
  per 50 KB). The Knowledgator models stay a configurable catalog option,
  pinned, with the prompts they were trained on; `-small` and `-large` are
  not measured yet.
- **Fastino's PII model ships with a catalog threshold of 0.9**, the only
  per-model default: at the generic 0.5 it makes about four times the
  agent-traffic false positives (96 against 26 per 50 KB).
- **Contextual types are best effort.** What the models add for addresses,
  dates of birth, usernames and account numbers is measured on a generated
  corpus and on published slices, never guaranteed; the structured types
  stay with the rules
  ([threat-model.md](threat-model.md#contextual-values-rules-and-models)).

## LLM-based extractors (LangExtract and its class) — rejected as detectors

LangExtract (google/langextract) and similar prompted-extraction tools
find entities by sending the text to an LLM with few-shot instructions
and returning grounded spans. As request-path detectors they fail
structurally, not tunably:

1. **Threat-model inversion.** A detector runs on the RAW,
   pre-redaction request text. A cloud-backed extractor (Gemini API,
   LangExtract's default) would transmit every secret to a cloud LLM in
   order to decide what to hide from cloud LLMs — the exact leak this
   proxy exists to prevent. All six shipped backends run in-process on
   local weights precisely so the raw text never leaves the machine.
2. **Latency.** The local-model escape hatch (Ollama) fixes privacy but
   is a generation pass per scanned string — seconds, where bar 4 above
   is the 1–100 ms class (the slowest model the NER bench measured, with
   1.5B parameters, already takes about a second per 500 characters and
   is catalogued "caution" for it). jsonwalk multiplies
   that per string in every request body.
3. **No portable recall guarantee.** `python -m llm_redact.bench --check`
   gates recall == 1.0 per rule, deterministically, and the NER bench gates
   a model's recall at floors measured for one model file, revision and
   decoding ([ner-bench.md](ner-bench.md)). Sampling-based generation
   pins neither: the same prompt returns other spans on another server,
   quantization, driver or version, so a figure measured on one machine
   says nothing about another, and a detector whose recall varies
   silently is a silent-leak risk (the same reasoning that machine-checks
   prefilter literals).
4. **The scanned text can instruct the detector (prompt injection).** An
   instruction-following model reads instructions inside the text it is
   asked to scan, and that text is attacker-influenced by construction
   (web pages, issue bodies, dependency READMEs, a file the agent read): a
   line telling the extractor to report nothing lowers its recall with the
   very content the proxy exists to protect. An encoder tagger reads the
   text as data and has no instruction channel to talk to.

LLM-based extraction is legitimate BESIDE the proxy, out of band: batch jobs
where the text is already destined for an LLM or never touches the request
path. Two such roles are built, as dev-only tooling in `scripts/pii_corpus/`
(not part of the wheel; its README has the details):

- **Corpus teacher.** A local LLM served by Ollama writes coding-agent
  artifacts (tool-call JSON, diffs, logs, configuration, commit messages,
  SQL, tracebacks) with invented personal data and tags every value
  inline (`generate.py`); only models published under Apache-2.0 may teach,
  by exact name and size tag, on a loopback server, with no cloud tags. A
  human accepts, edits or rejects every row (`review.py`), and the frozen,
  hand-verified set is scored by the bench as the private `agent-eval`
  dataset. The same teacher can read `bench/fp_corpus` beside the detectors
  and list candidate misses and false positives (`audit.py`; a report that
  never edits the corpus or its manifest, and the teacher is told the text is
  data, not instructions, so a corpus file that talks it out of a finding
  only costs a candidate).
- **Student.** Teaching a small encoder from that data is a recipe only:
  `train_student.py plan` checks the training sources against a license
  manifest and writes the data manifest and a model card; it trains
  nothing, and no student model exists.

## Why SaaS NER providers are not considered

Cloud PII-detection APIs — AWS Comprehend PII, Google Cloud Sensitive
Data Protection (DLP), Azure AI Language's hosted PII endpoint,
Nightfall, Skyflow, and the rest of the category — are excluded as a
class, for the same root cause as cloud-backed LLM extraction: a
detector's input is the RAW, pre-redaction request text. Sending it to
a SaaS endpoint transmits exactly the secrets this proxy exists to keep
local, trading the LLM provider for a second third-party data
processor. The trust boundary grows; the product's one promise — the
value never leaves your machine — becomes false. Three secondary
failures follow even where a user would accept that trade:

1. **Availability coupling**: a detector sits in the hot path of every
   request; a vendor outage forces a choice between failing open
   (scanning less, silently) and failing closed (your agent is down).
2. **Per-request cost and quota** on a path that scans every string of
   every body.
3. **Posture honesty**: `docs/privacy.md` states that nothing is
   transmitted anywhere except the redacted provider traffic; a SaaS
   detector would falsify it.

This is a data-locality objection, not a commercial one — the same
vendors' engines become acceptable the moment they run on your own
hardware, which is the next section.

## Commercial self-hosted engines (paid, structurally acceptable)

Paid engines that deploy as containers/services on the user's own
infrastructure clear the data-locality bar; they differ from the FOSS
backends only in licensing and in shipping as a LOCAL SERVICE rather
than an importable model. Verified candidates (2026-07; re-check
vendor packaging before implementation):

| Engine | Deployment | Capability added | Caveats |
|---|---|---|---|
| **Private AI** | self-hosted container (AWS Marketplace et al.) | purpose-built PII: 50+ types, 50+ languages, file formats | the strongest fit; commercial license |
| **Azure AI Language PII (container)** | on-prem Docker container | Microsoft-maintained PII taxonomy, multilingual | standard containers meter usage back to Azure for billing (data stays local); air-gap needs the disconnected-container commitment tier |
| **Babel Street Analytics (Rosette)** | Analytics Server (Docker/Helm) | enterprise multilingual NER, tunable (gazetteers + patterns + models) | Java service, ~16 GB heap recommended, up to ~90 GB disk — enterprise footprint |
| **John Snow Labs Healthcare NLP** | local Spark/JVM | best-in-class clinical de-identification | Spark+JVM runtime; healthcare-niche; external-service integration only |

**Integration shape, if demand materializes**: not one backend per
vendor but ONE generic `http` NER backend — POST the string to a
configured **loopback-only** URL and map the vendor's entity labels in
config. Loopback would be enforced the way the off-box vault rule is
(non-local URL = startup ConfigError, explicit env hatch), making
"self-hosted only" structural rather than advisory. It is
fake-injectable for tests (HTTP client), and localhost latency sits in
the existing 10–100 ms class. The new failure mode to design
deliberately: a down sidecar must degrade LOUDLY (doctor FAIL, /status,
posture warnings) — never silently scan less. Like the rest of this
document: demand-driven, not built speculatively.

## Running NER on non-English text

The backends pick their model from `[detection.ner] language` (spaCy,
Presidio, Stanza) and the per-backend `[detection.ner.models]` overrides
(every backend). Three ways to cover another language, cheapest first:

```toml
# 1. A language-specific spaCy pipeline (tens of MB, ~1-10 ms). Install the
#    model, then name it. Many non-English pipelines label people PER;
#    the PERSON type request covers it (docs/detection.md).
[detection.ner]
enabled = true
backend = "spacy"
language = "de"
model = "de_core_news_sm"     # uv run python -m spacy download de_core_news_sm
entities = ["PERSON"]

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
dedupe in overlap resolution. Entity labels differ across models; a
placeholder type such as `PERSON` requests every label that folds into it
(docs/detection.md), and an entity no loaded model can emit logs a startup
warning instead of silently detecting nothing.

This document is also the decision record for when a user asks for a
language the current backends serve poorly.
