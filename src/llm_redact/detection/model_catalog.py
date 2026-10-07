"""What llm-redact knows about NER models: licenses, pins, prompts, windows.

One frozen :class:`CatalogEntry` per Hugging Face model (or model-id prefix)
the NER backends may load. The catalog states facts and never refuses a
model (the core warns; a policy plugin may enforce):

* ``status``: ``vetted`` (a known-good choice), ``caution`` (configurable,
  with the reason shown: for example not yet measured by the llm-redact
  bench) or ``restricted`` (never suggested; a startup warning names the
  reason);
* ``reason``: one neutral, verifiable line (owner decision D14: a license
  identifier, "not OSI-approved", "backbone: ...", "trained on <dataset>
  (<license>)"; never a legal or procurement conclusion), shown with a link
  to the model card and the date the facts were checked
  (:meth:`CatalogEntry.describe`);
* ``revision``: the commit a vetted or caution model loads at unless
  ``[detection.ner.revisions]`` names another. Each pin is the model's
  ``main`` commit on the check date, so a cache that an online load
  refreshed since that commit already holds the pinned snapshot;
* ``backbone`` / ``backbone_revision``: the base model a GLiNER checkpoint
  names in ``gliner_config.json``. A pin is recorded only where the
  checkpoint ships no tokenizer and no ``encoder_config`` (the backbone
  then supplies both); a self-contained checkpoint needs no backbone files;
* ``prompts``: per placeholder type, the GLiNER prompt the model was trained
  on, sent instead of the generic one (:data:`labels.GLINER_PROMPTS`);
* ``window``: the longest input one model call takes: tokens for ``hf``,
  GLiNER words for ``gliner``. None: read it from the model's own config;
* ``tagging`` / ``viterbi_calibration``: an ``hf`` model's token-tagging
  scheme, and the repository file holding the transition biases its BIOES
  or BILOU spans are decoded with (the constrained Viterbi decoder,
  detection/tagging.py; without one they are decoded greedily).

Local model directories are identified by an optional sidecar file,
:data:`SIDECAR_NAME` (``{"model_id": ..., "revision": ...}``), which
``llm-redact models pull --to`` and model bundles write beside the files.

Lookups are case-insensitive: the Hub resolves an id in any letter case to
the same repository (``NVIDIA/GLINER-PII`` loads ``nvidia/gliner-PII``), so
a case variant must not escape its entry. This module is data plus pure
functions; it imports nothing heavy and logs nothing.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from llm_redact.jsonwalk import json_text, loads_bounded

# The date every fact below was read from the model cards and the Hub API
# (https://huggingface.co/api/models/<id>/revision/main).
CHECKED = "2026-10-05"

Status = Literal["vetted", "caution", "restricted"]
STATUSES: tuple[Status, ...] = ("vetted", "caution", "restricted")
# Token-tagging schemes of `hf` models (BIO models run the transformers
# pipeline's aggregation; the hf backend decodes BIOES/BILOU itself,
# detection/tagging.py).
TAGGING_SCHEMES = ("bio", "bioes", "bilou")
# CatalogEntry.piece_labels: which pieces of a word a BIO tagger labels.
PIECE_LABELS = ("first", "every")

# Value-free provenance tags a policy can match on (llm-redact-pro's model
# policy). Each is a neutral fact about the weights or their training data.
LINEAGE_TAGS = frozenset(
    {
        "noncommercial",  # the weights' license permits non-commercial use only
        "non-osi",  # the weights' license or terms are not OSI-approved
        "llama-derived",  # training data generated with Llama models
        "qwen-backbone",  # built on a Qwen model
        "ai4privacy-restricted",  # trained on an AI4Privacy release with a restrictive license
        "remote-code",  # loading needs trust_remote_code (llm-redact never enables it)
        "undisclosed-training-data",  # the card does not name the training data
        "nemotron-cc-by",  # trained on NVIDIA Nemotron-PII (CC BY 4.0: attribution)
        "conll2003",  # trained on CoNLL-2003 (Reuters news)
    }
)

# The NER backends whose models are Hugging Face Hub snapshots: the only
# ones a revision pin or this catalog applies to (spaCy, Presidio and Stanza
# load pip-installed or library-managed models).
HUB_BACKENDS = ("gliner", "gliner2", "hf")
# The model each Hub backend loads when the configuration names none (the
# backends' own defaults; tests/test_model_catalog.py pins them equal).
DEFAULT_MODELS: Mapping[str, str] = MappingProxyType(
    {
        "gliner": "urchade/gliner_small-v2.1",
        "gliner2": "fastino/gliner2-base-v1",
        "hf": "dslim/bert-base-NER",
    }
)

# A full commit id. Branch and tag names move, so a pin is always this.
REVISION_RE = re.compile(r"[0-9a-f]{40}")
# A Hugging Face model id: an owner, "/", a name (letters, digits, "-",
# "_", "."). The owner part is optional on the Hub for a few legacy models.
MODEL_ID_RE = re.compile(r"(?:[A-Za-z0-9][A-Za-z0-9._-]*/)?[A-Za-z0-9][A-Za-z0-9._-]*")

# The model directory's identity file and the most of it ever read.
SIDECAR_NAME = "llm-redact-model.json"
MAX_SIDECAR_BYTES = 64 * 1024

# The reason a configurable model carries until the bench has measured it.
UNMEASURED = "not yet measured by the llm-redact bench"
# The day the llm-redact bench measured the models whose reasons quote it
# (docs/ner-landscape.md, "PII models measured by the llm-redact bench").
MEASURED = "2026-10-07"


def measured(
    recall: float,
    leak: float,
    false_positives: float,
    p50_ms: float,
    *,
    unrequested: tuple[str, ...] = (),
) -> str:
    """The bench's numbers as a catalog reason states them (owner decision
    D11: a model that misses an admission bar stays "caution", with its
    numbers shown). Measured with the entities the model's bench
    configuration requests (bench/configs): its recommended entities that
    the synthetic corpus labels. ``unrequested`` names the recommended ones
    left out (the corpus labels no PASSPORT or DRIVER_LICENSE), so the
    reason says every number — the false positives too — is for that
    narrower request. PERSON recall and the character-leak rate on the
    synthetic corpus, the false positives per 50 KB of the agent-traffic
    negatives, and the full pipeline's p50 for a 500-character string."""
    scope = f", {_joined(unrequested)} not requested" if unrequested else ""
    return (
        f"llm-redact bench {MEASURED}{scope}: synthetic-corpus PERSON recall {recall:.2f},"
        f" character leak {leak:.2f}; {false_positives:.0f} false positives per 50 KB of"
        f" agent-traffic negatives; p50 {p50_ms:.0f} ms per 500 characters"
    )


def _joined(names: tuple[str, ...]) -> str:
    """Names joined for a sentence: A; A and B; A, B and C."""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


# The seven contextual types a PII model is recommended for (structured
# types stay with the regex rules: models draw wider spans, and the longest
# span wins overlap resolution).
_CONTEXTUAL = (
    "PERSON",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "PASSPORT",
    "DRIVER_LICENSE",
    "USERNAME",
    "ACCOUNT_NUMBER",
)


# The recommended contextual types the synthetic corpus labels no value of:
# a bench configuration cannot score them, so it does not request them.
_UNSCORED = ("PASSPORT", "DRIVER_LICENSE")


@dataclass(frozen=True)
class CatalogEntry:
    """The facts about one model (or, with ``prefix``, every model whose id
    starts with ``model_id``)."""

    model_id: str
    backends: tuple[str, ...]
    # SPDX identifier, or LicenseRef-... for a license SPDX does not list.
    license: str
    status: Status
    # One neutral line (D14); describe() adds the card link and check date.
    reason: str
    prefix: bool = False
    # The card the reason is read from (default: the model's own page).
    card: str = ""
    checked: str = CHECKED
    revision: str | None = None
    backbone: str | None = None
    backbone_revision: str | None = None
    # What a redistribution or a model list should credit.
    attribution: str = ""
    lineage: tuple[str, ...] = ()
    recommended_entities: tuple[str, ...] = ()
    # Extra label map: normalized model label -> placeholder type, before
    # the default folds (detection/labels.py).
    labels: tuple[tuple[str, str], ...] = ()
    # GLiNER prompt overrides: placeholder type -> the prompt sent for it.
    prompts: tuple[tuple[str, str], ...] = ()
    # Repo-relative ONNX weight files the gliner backend can load.
    onnx_files: tuple[str, ...] = ()
    tagging: str | None = None
    # Which sub-word pieces a BIO tagger was trained to label, for a
    # tokenizer without word-piece marks (hf_ner._word_row): "first" (the
    # Hugging Face convention: a word's first piece, the others never
    # trained) or "every" (every piece, so a later piece's tag is evidence
    # too). An uncatalogued model is read as "first".
    piece_labels: str = "first"
    # A BIOES/BILOU tagger's calibration file (repo-relative, fetched with
    # the model): transition biases for the constrained Viterbi decoder.
    viterbi_calibration: str | None = None
    window: int | None = None
    # (distribution, minimum version) the model needs beyond the extras'.
    min_versions: tuple[tuple[str, str], ...] = ()

    @property
    def card_url(self) -> str:
        return self.card or f"https://huggingface.co/{self.model_id}"

    def prompt_for(self, type_name: str) -> str | None:
        """The GLiNER prompt this model was trained on for ``type_name``,
        or None (the generic prompt applies)."""
        return dict(self.prompts).get(type_name)

    def extra_files(self, backend: str) -> tuple[str, ...]:
        """The repository files beyond ``backend``'s own file list that a
        load of this model reads, so a pull fetches them and a check
        requires them: an ``hf`` tagger's calibration file."""
        if backend == "hf" and "hf" in self.backends and self.viterbi_calibration is not None:
            return (self.viterbi_calibration,)
        return ()

    def describe(self) -> str:
        """The reason as a warning or a docs row shows it: the facts, the
        card they come from and the date they were checked."""
        who = f"{self.model_id}*" if self.prefix else self.model_id
        return f"{who}: {self.reason} ({self.card_url}, checked {self.checked})"


def _urchade_v21(size: str, backbone: str, backbone_revision: str, revision: str) -> CatalogEntry:
    # urchade's v2.1 GLiNER checkpoints ship gliner_config.json and weights
    # only: no tokenizer and no encoder_config, so the backbone named in
    # gliner_config.json (model_name) supplies both and is pinned here.
    model_id = f"urchade/gliner_{size}-v2.1"
    default = " the gliner backend's default model;" if model_id == DEFAULT_MODELS["gliner"] else ""
    return CatalogEntry(
        model_id=model_id,
        backends=("gliner",),
        license="Apache-2.0",
        status="vetted",
        reason=(
            f"Apache-2.0;{default} trained on urchade/pile-mistral-v0.1 (Apache-2.0); ships"
            f" no tokenizer or encoder_config, so its backbone {backbone} (MIT) is pinned too"
        ),
        revision=revision,
        backbone=backbone,
        backbone_revision=backbone_revision,
        attribution=f"{model_id} (Apache-2.0); GLiNER, arXiv:2311.08526",
        recommended_entities=("PERSON",),
    )


# Knowledgator GLiNER-PII (developed with Wordcab), all four sizes created
# 2025-09-24. Each checkpoint is self-contained: tokenizer files and an
# encoder_config in gliner_config.json. Trained-prompt names from the card's
# label lists, one per contextual type ("location address" is the card's
# "street addresses" label; "location street" means street names only).
_KNOWLEDGATOR_PROMPTS = (
    ("PERSON", "name"),
    ("ADDRESS", "location address"),
    ("DATE_OF_BIRTH", "dob"),
    ("PASSPORT", "passport number"),
    ("DRIVER_LICENSE", "driver license"),
    ("USERNAME", "username"),
    ("ACCOUNT_NUMBER", "account number"),
)
# The card's GLiNER.cpp example runs these models with a 512 limit
# (`gliner::Config{12, 512}`); gliner_config.json's max_len is larger
# (2048 words; 768 for -large), and the deberta-v3 encoders of -base and
# -large take 512 positions. The gliner backend's windows also check the
# encoder's subword limit.
_KNOWLEDGATOR_WINDOW = 512
_ONNX_ALL = ("onnx/model.onnx", "onnx/model_fp16.onnx", "onnx/model_quint8.onnx")


def _knowledgator(
    size: str,
    revision: str,
    backbone: str,
    *,
    onnx_files: tuple[str, ...] = _ONNX_ALL,
    min_versions: tuple[tuple[str, str], ...] = (),
    bench: str = UNMEASURED,
) -> CatalogEntry:
    return CatalogEntry(
        model_id=f"knowledgator/gliner-pii-{size}-v1.0",
        backends=("gliner",),
        license="Apache-2.0",
        status="caution",
        reason=(
            f"Apache-2.0; backbone {backbone}; the card does not name the training data; {bench}"
        ),
        checked="2026-10-07",
        revision=revision,
        backbone=backbone,
        attribution="GLiNER-PII by Knowledgator and Wordcab (Apache-2.0)",
        lineage=("undisclosed-training-data",),
        recommended_entities=_CONTEXTUAL,
        prompts=_KNOWLEDGATOR_PROMPTS,
        onnx_files=onnx_files,
        window=_KNOWLEDGATOR_WINDOW,
        min_versions=min_versions,
    )


# transformers learned ModernBERT (the ettin encoders' model_type) in 4.48.0.
_MODERNBERT = (("transformers", "4.48.0"),)
_AI4PRIVACY_400K = (
    "ai4privacy/pii-masking-400k, whose license permits academic and non-commercial use only"
)

CATALOG: tuple[CatalogEntry, ...] = (
    # --- vetted: the models users already run, pinned to main ---------------
    CatalogEntry(
        # main since 2024-10-08 (README edits; weights unchanged since 2024).
        model_id="dslim/bert-base-NER",
        backends=("hf",),
        license="MIT",
        status="vetted",
        reason=(
            "MIT; the hf backend's default model; bert-base-cased fine-tuned on CoNLL-2003"
            " (Reuters news), labels PER, ORG, LOC, MISC; model.safetensors; no tokenizer.json"
            " (the fast tokenizer is built from vocab.txt)"
        ),
        revision="d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc",
        attribution=(
            "dslim/bert-base-NER (MIT); trained on CoNLL-2003 (Tjong Kim Sang and De Meulder, 2003)"
        ),
        lineage=("conll2003",),
        recommended_entities=("PERSON",),
        # CoNLL-2003's IOB1 variant: an entity may open with I-, B- only
        # separates two adjacent entities of one type. The pipeline's
        # aggregation reads it like BIO.
        tagging="bio",
    ),
    # main since 2024-04-10; pytorch_model.bin only (GLiNER loads it with
    # torch.load(weights_only=True)).
    _urchade_v21(
        "small",
        "microsoft/deberta-v3-small",
        "a36c739020e01763fe789b4b85e2df55d6180012",
        "4e091416cf7c3481db542c2a3d26156916f3a47f",
    ),
    # main since 2024-08-21 (model.safetensors added beside the .bin).
    _urchade_v21(
        "medium",
        "microsoft/deberta-v3-base",
        "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        "40ec419335d09393f298636f471328b722c6da9e",
    ),
    # main since 2025-12-08 (model.safetensors added); a cache last
    # refreshed before then holds 853ce23e47e5 instead.
    _urchade_v21(
        "multi",
        "microsoft/mdeberta-v3-base",
        "a0484667b22365f84929a935b5e50a51f71f159d",
        "443d26d654e0324125a96bebd8e796c14ff2efe6",
    ),
    CatalogEntry(
        # main since 2024-04-20; pytorch_model.bin only.
        model_id="urchade/gliner_multi_pii-v1",
        backends=("gliner",),
        license="Apache-2.0",
        status="vetted",
        reason=(
            "Apache-2.0; urchade/gliner_multi-v2.1 fine-tuned on"
            " urchade/synthetic-pii-ner-mistral-v1 (Apache-2.0); no updates since 2024-04-20;"
            " ships no tokenizer or encoder_config, so its backbone microsoft/mdeberta-v3-base"
            " (MIT) is pinned too"
        ),
        revision="1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d",
        backbone="microsoft/mdeberta-v3-base",
        backbone_revision="a0484667b22365f84929a935b5e50a51f71f159d",
        attribution="urchade/gliner_multi_pii-v1 (Apache-2.0); GLiNER, arXiv:2311.08526",
        recommended_entities=_CONTEXTUAL,
    ),
    # --- caution: configurable (owner decision D13); -edge and -base measured
    # by the bench below every D11 bar but agent-traffic false positives
    # (-base: latency too); -small and -large not yet measured ----------
    _knowledgator(
        # main since 2026-03-26 (README edit).
        "edge",
        "9b7f39b0a2da971a5beea78d35f1539d4009c891",
        "jhu-clsp/ettin-encoder-32m",
        min_versions=_MODERNBERT,
        # PyTorch weights; the int8 ONNX export measures lower (PERSON
        # recall 0.77, leak 0.27: docs/ner-landscape.md).
        bench=measured(
            recall=0.99, leak=0.08, false_positives=109, p50_ms=93, unrequested=_UNSCORED
        ),
    ),
    _knowledgator(
        # main since 2025-09-27.
        "small",
        "d21aad5b4a7ec82b3d0970fd1ac74a12c087d85e",
        "jhu-clsp/ettin-encoder-68m",
        min_versions=_MODERNBERT,
    ),
    _knowledgator(
        # main since 2025-09-27.
        "base",
        "61726e0ad791dcab3e29339bbec3ad42ded65641",
        "microsoft/deberta-v3-small",
        # PyTorch weights; the int8 ONNX export: PERSON recall 0.96, leak
        # 0.03, 13 false positives per 50 KB, p50 182 ms.
        bench=measured(
            recall=0.97, leak=0.02, false_positives=8, p50_ms=234, unrequested=_UNSCORED
        ),
    ),
    _knowledgator(
        # main since 2026-05-07 (README edit); ships no fp16 ONNX file.
        "large",
        "f847f54fbc97ad6e78bfa20ed9c5e5d5c43327b9",
        "microsoft/deberta-v3-large",
        onnx_files=("onnx/model.onnx", "onnx/model_quint8.onnx"),
    ),
    # --- caution: the gliner2 backend's default, not yet measured --------
    CatalogEntry(
        # main since 2026-09-28 (card edits; model.safetensors unchanged since
        # 2025-07-02). Self-contained: config.json, encoder_config/config.json
        # (deberta-v2), tokenizer files and model.safetensors.
        model_id="fastino/gliner2-base-v1",
        backends=("gliner2",),
        license="Apache-2.0",
        status="caution",
        reason=(
            "Apache-2.0; the gliner2 backend's default model; backbone"
            " microsoft/deberta-v3-base; the card describes its training data only as"
            f" multi-domain datasets; {UNMEASURED}"
        ),
        checked="2026-10-06",
        revision="f9634218e53580c56edf0de97ca1a7d3f1c2354e",
        backbone="microsoft/deberta-v3-base",
        attribution="GLiNER2 by Fastino AI (Apache-2.0); arXiv:2507.18546",
        lineage=("undisclosed-training-data",),
        recommended_entities=("PERSON",),
    ),
    # --- caution: PII models the bench measured below a D11 bar (their
    # numbers in the reason; docs/ner-landscape.md) ------------------------
    CatalogEntry(
        # main since 2026-01-13. BIO tags, first sub-token labelled only;
        # 54 entity types, 5 of them sensitive attributes (D10: not folded).
        model_id="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
        backends=("hf",),
        license="Apache-2.0",
        status="caution",
        reason=(
            "Apache-2.0; microsoft/deberta-v3-small (MIT) fine-tuned on nvidia/Nemotron-PII"
            " (CC BY 4.0); 54 entity types; "
            + measured(recall=1.00, leak=0.05, false_positives=18, p50_ms=175)
        ),
        checked="2026-10-07",
        revision="a2360d3f42526fc660ac3b2b2301e1c2d94eba61",
        backbone="microsoft/deberta-v3-small",
        attribution=(
            "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 (Apache-2.0); trained on NVIDIA"
            " Nemotron-PII, CC BY 4.0"
        ),
        lineage=("nemotron-cc-by",),
        recommended_entities=("PERSON", "ADDRESS", "DATE_OF_BIRTH", "USERNAME", "ACCOUNT_NUMBER"),
        tagging="bio",
        # The card's "Max Sequence Length: 384 tokens" (the model takes 512
        # positions).
        window=384,
    ),
    CatalogEntry(
        # main since 2026-05-22. BIO tags; 55 entity types, 5 of them
        # sensitive attributes (D10: not folded).
        model_id="kalyan-ks/ettin-68m-nemotron-pii",
        backends=("hf",),
        license="MIT",
        status="caution",
        reason=(
            "MIT; jhu-clsp/ettin-encoder-68m (MIT) fine-tuned on nvidia/Nemotron-PII"
            " (CC BY 4.0); 55 entity types; every sub-word piece tagged B-, read as separate"
            " values; " + measured(recall=0.99, leak=0.16, false_positives=21, p50_ms=277)
        ),
        checked="2026-10-07",
        revision="500262a2aaf913825ef750ef255c3fe437cd8e64",
        backbone="jhu-clsp/ettin-encoder-68m",
        attribution=(
            "kalyan-ks/ettin-68m-nemotron-pii (MIT); trained on NVIDIA Nemotron-PII, CC BY 4.0"
        ),
        lineage=("nemotron-cc-by",),
        recommended_entities=("PERSON", "ADDRESS", "DATE_OF_BIRTH", "USERNAME", "ACCOUNT_NUMBER"),
        tagging="bio",
        # Measured 2026-10-07: it tags every sub-word piece (each B-), and a
        # word's first piece is often untagged or unsure while a later one
        # is sure.
        piece_labels="every",
        # tokenizer_config.json's max_length (the tokenizer reports 8192,
        # the encoder takes 7999 positions).
        window=1024,
        min_versions=_MODERNBERT,
    ),
    CatalogEntry(
        # main since 2026-04-22 (created 2026-04-17). 1.5B parameters, 50M
        # active (sparse mixture of experts); model.safetensors is 2.8 GB
        # (the repository also holds original/ and ONNX copies, never
        # fetched). Eight span labels, each B-/I-/E-/S- tagged.
        model_id="openai/privacy-filter",
        backends=("hf",),
        license="Apache-2.0",
        status="caution",
        reason=(
            "Apache-2.0; 1.5B parameters, 50M active (sparse mixture of experts); BIOES"
            " tags; the card does not name the training data; "
            + measured(recall=0.99, leak=0.12, false_positives=23, p50_ms=1108)
        ),
        checked="2026-10-07",
        revision="7ffa9a043d54d1be65afb281eddf0ffbe629385b",
        attribution="OpenAI Privacy Filter by OpenAI (Apache-2.0)",
        lineage=("undisclosed-training-data",),
        # Its `secret` label is left to the anchored secret rules: the card
        # lists over-redaction of hashes, placeholders and sample
        # credentials among its failure modes.
        recommended_entities=("PERSON", "ADDRESS", "ACCOUNT_NUMBER"),
        tagging="bioes",
        viterbi_calibration="viterbi_calibration.json",
        # The card's 128,000-token context window.
        window=128000,
        # transformers learned the model type (openai_privacy_filter) in 5.6.0.
        min_versions=(("transformers", "5.6.0"),),
    ),
    CatalogEntry(
        # main since 2026-09-28 (created 2026-05-10). Self-contained:
        # config.json, encoder_config/config.json, tokenizer and
        # model.safetensors (1.2 GB). 42 labels; the prompts below are the
        # card's spellings of the contextual types.
        model_id="fastino/gliner2-privacy-filter-PII-multi",
        backends=("gliner2",),
        license="Apache-2.0",
        status="caution",
        reason=(
            "Apache-2.0; GLiNER2 (backbone microsoft/mdeberta-v3-base) fine-tuned on 4,910"
            " synthetic texts the card says GPT-5.4 generated; English, French, Spanish,"
            " German, Italian, Portuguese, Dutch; at score_threshold 0.9, "
            + measured(
                recall=1.00,
                leak=0.01,
                false_positives=26,
                p50_ms=453,
                unrequested=_UNSCORED,
            )
        ),
        checked="2026-10-07",
        revision="1cb4166094dc58fa8d836429f060d6c95f62b495",
        backbone="microsoft/mdeberta-v3-base",
        attribution="GLiNER2-PII by Fastino AI (Apache-2.0); arXiv:2605.09973",
        recommended_entities=_CONTEXTUAL,
        prompts=(
            ("PERSON", "person"),
            ("ADDRESS", "street_address"),
            ("DATE_OF_BIRTH", "date_of_birth"),
            ("PASSPORT", "passport_number"),
            ("DRIVER_LICENSE", "drivers_license_number"),
            ("USERNAME", "username"),
            ("ACCOUNT_NUMBER", "account_number"),
        ),
    ),
    # --- restricted: never suggested; a warning names the reason ----------
    CatalogEntry(
        model_id="iiiorg/piiranha-v1-detect-personal-information",
        backends=("hf",),
        license="CC-BY-NC-ND-4.0",
        status="restricted",
        reason=f"CC-BY-NC-ND-4.0 (non-commercial, no derivatives); trained on {_AI4PRIVACY_400K}",
        lineage=("noncommercial", "ai4privacy-restricted"),
    ),
    CatalogEntry(
        model_id="Isotonic/deberta-v3-base_finetuned_ai4privacy_v2",
        backends=("hf",),
        license="CC-BY-NC-4.0",
        status="restricted",
        reason=(
            "CC-BY-NC-4.0 (non-commercial); trained on ai4privacy/pii-masking-200k, whose"
            " license requires a company license for organizations above 3 staff"
        ),
        lineage=("noncommercial", "ai4privacy-restricted"),
    ),
    CatalogEntry(
        model_id="Isotonic/distilbert_finetuned_ai4privacy_v2",
        backends=("hf",),
        license="CC-BY-NC-4.0",
        status="restricted",
        reason=(
            "CC-BY-NC-4.0 (non-commercial); trained on ai4privacy/pii-masking-200k, whose"
            " license requires a company license for organizations above 3 staff"
        ),
        lineage=("noncommercial", "ai4privacy-restricted"),
    ),
    CatalogEntry(
        model_id="urchade/gliner_base",
        backends=("gliner",),
        license="CC-BY-NC-4.0",
        status="restricted",
        reason="CC-BY-NC-4.0 (non-commercial)",
        lineage=("noncommercial",),
    ),
    CatalogEntry(
        model_id="nvidia/gliner-PII",
        backends=("gliner",),
        license="LicenseRef-NVIDIA-Open-Model-License",
        status="restricted",
        reason=(
            "NVIDIA Open Model License: not OSI-approved; its text includes termination"
            " clauses; built on urchade/gliner_large-v2.1, trained on nvidia/Nemotron-PII"
            " (CC BY 4.0)"
        ),
        lineage=("non-osi", "nemotron-cc-by"),
    ),
    CatalogEntry(
        model_id="bigcode/starpii",
        backends=("hf",),
        license="LicenseRef-bigcode-starpii-terms-of-use",
        status="restricted",
        reason=(
            "gated: access requires accepting the model's terms of use (use limited to"
            " removing PII from datasets; the model may not be shared); no license declared"
        ),
        lineage=("non-osi",),
    ),
    CatalogEntry(
        model_id="ai4privacy/llama-ai4privacy-",
        prefix=True,
        backends=("hf",),
        license="MIT",
        status="restricted",
        reason=(
            "MIT weights trained on ai4privacy/open-pii-masking-500k-ai4privacy, which was"
            " generated with Llama 3.1 and 3.3; that dataset's card applies the Llama 3.1/3.3"
            " Community License (model naming, attribution) to models trained on it"
        ),
        card="https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy",
        lineage=("llama-derived",),
    ),
    CatalogEntry(
        model_id="knowledgator/gliner-stream-pii-v1.0",
        backends=("gliner",),
        license="Apache-2.0",
        status="restricted",
        reason="Apache-2.0; backbone Qwen/Qwen3-0.6B",
        backbone="Qwen/Qwen3-0.6B",
        lineage=("qwen-backbone",),
    ),
    CatalogEntry(
        model_id="perplexity-ai/PII-Tracer",
        backends=("hf",),
        license="MIT",
        status="restricted",
        reason=(
            "MIT; loading requires trust_remote_code (config.json auto_map names"
            " modeling_pii_masking.py), which llm-redact never enables; Qwen3 encoder"
        ),
        lineage=("remote-code", "qwen-backbone"),
    ),
    CatalogEntry(
        model_id="OpenMed/privacy-filter-multilingual",
        backends=("hf",),
        license="Apache-2.0",
        status="restricted",
        reason=(
            f"Apache-2.0; openai/privacy-filter fine-tuned on AI4Privacy releases including"
            f" {_AI4PRIVACY_400K}, and ai4privacy/open-pii-masking-500k-ai4privacy, generated"
            f" with Llama 3.1 and 3.3"
        ),
        lineage=("ai4privacy-restricted", "llama-derived"),
    ),
    CatalogEntry(
        model_id="llm-semantic-router/mmbert32k-pii-detector-merged",
        backends=("hf",),
        license="MIT",
        status="restricted",
        reason=f"MIT; its card lists among its training data {_AI4PRIVACY_400K}",
        lineage=("ai4privacy-restricted",),
    ),
)

_EXACT: Mapping[str, CatalogEntry] = MappingProxyType(
    {entry.model_id.casefold(): entry for entry in CATALOG if not entry.prefix}
)
# Longest prefix first, so a more specific prefix wins.
_PREFIXES = tuple(
    sorted((entry for entry in CATALOG if entry.prefix), key=lambda e: -len(e.model_id))
)


def lookup(model_id: str) -> CatalogEntry | None:
    """The catalog entry for a Hub model id: an exact match, else the
    longest matching id prefix, else None. Case-insensitive, as the Hub
    resolves ids."""
    folded = model_id.casefold()
    entry = _EXACT.get(folded)
    if entry is not None:
        return entry
    for candidate in _PREFIXES:
        if folded.startswith(candidate.model_id.casefold()):
            return candidate
    return None


def pinned_revision(model_id: str) -> str | None:
    """The commit the catalog pins ``model_id`` to (None: not catalogued, or
    catalogued without a pin, as restricted models are)."""
    entry = lookup(model_id)
    return entry.revision if entry is not None else None


@dataclass(frozen=True)
class ModelIdentity:
    """Which Hub model, at which commit, a set of model files is."""

    model_id: str
    revision: str | None = None


class SidecarError(ValueError):
    """A model directory's sidecar file exists but does not say which model
    the directory holds. The message names the file and the problem only."""


def read_sidecar(directory: str | Path) -> ModelIdentity | None:
    """The identity recorded in ``directory``'s :data:`SIDECAR_NAME` file;
    None when there is no such file (or ``directory`` is no directory).
    Keys other than ``model_id`` and ``revision`` are ignored. A file that
    exists but cannot be read as that identity raises SidecarError."""
    path = Path(directory) / SIDECAR_NAME
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_SIDECAR_BYTES + 1)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        raise SidecarError(f"{path}: cannot be read ({type(exc).__name__})") from exc
    if len(raw) > MAX_SIDECAR_BYTES:
        raise SidecarError(f"{path}: larger than {MAX_SIDECAR_BYTES} bytes")
    try:
        data = loads_bounded(raw.decode("utf-8"))
    except ValueError as exc:  # UnicodeDecodeError, JSONDecodeError, JsonTooDeep
        raise SidecarError(f"{path}: not a UTF-8 JSON document") from exc
    if not isinstance(data, dict):
        raise SidecarError(f"{path}: not a JSON object")
    model_id = data.get("model_id")
    if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id):
        raise SidecarError(f"{path}: model_id must be a Hugging Face model id (owner/name)")
    revision = data.get("revision")
    if revision is not None and (
        not isinstance(revision, str) or not REVISION_RE.fullmatch(revision)
    ):
        raise SidecarError(
            f"{path}: revision must be a 40-character lowercase hex commit id, or null"
        )
    return ModelIdentity(model_id, revision)


def sidecar_text(identity: ModelIdentity) -> str:
    """The :data:`SIDECAR_NAME` file content for ``identity`` (what
    ``models pull --to`` and bundles write)."""
    return json_text({"model_id": identity.model_id, "revision": identity.revision}) + "\n"


def identify(model: str) -> ModelIdentity | None:
    """Which model a configured model value names: a local directory by its
    sidecar file (None without one: an unidentified local model), anything
    else as the Hub id it is (its revision comes from the configuration)."""
    if Path(model).is_dir():
        return read_sidecar(model)
    return ModelIdentity(model)
