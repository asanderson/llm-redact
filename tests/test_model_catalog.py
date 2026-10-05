"""The NER model catalog (detection/model_catalog.py): lookups, the shape and
neutrality of every entry, pins, GLiNER prompt overrides and the model
directory sidecar. Pure data: no network, no model."""

import json
import re
from pathlib import Path

import pytest

from llm_redact.detection import gliner_ner, hf_ner, model_catalog
from llm_redact.detection.labels import (
    CANONICAL_NER_TYPES,
    DEFAULT_FOLDS,
    SENSITIVE_LABELS,
    TYPE_NAMES,
    LabelPolicy,
    normalize_label,
)
from llm_redact.detection.model_catalog import (
    CATALOG,
    CHECKED,
    DEFAULT_MODELS,
    HUB_BACKENDS,
    LINEAGE_TAGS,
    MAX_SIDECAR_BYTES,
    MODEL_ID_RE,
    REVISION_RE,
    SIDECAR_NAME,
    STATUSES,
    TAGGING_SCHEMES,
    UNMEASURED,
    CatalogEntry,
    ModelIdentity,
    SidecarError,
    identify,
    lookup,
    pinned_revision,
    read_sidecar,
    sidecar_text,
)
from llm_redact.placeholders import is_placeholder_type

_ENTRY_IDS = [entry.model_id for entry in CATALOG]
_KNOWLEDGATOR = [
    f"knowledgator/gliner-pii-{size}-v1.0" for size in ("edge", "small", "base", "large")
]
_SHA = "0123456789abcdef0123456789abcdef01234567"


# --- lookups -------------------------------------------------------------


def test_exact_lookup() -> None:
    entry = lookup("dslim/bert-base-NER")
    assert entry is not None
    assert entry.model_id == "dslim/bert-base-NER"
    assert entry.status == "vetted"
    assert not entry.prefix


@pytest.mark.parametrize(
    ("asked", "found"),
    [
        ("NVIDIA/GLINER-PII", "nvidia/gliner-PII"),
        ("nvidia/gliner-pii", "nvidia/gliner-PII"),
        ("Dslim/Bert-Base-NER", "dslim/bert-base-NER"),
        ("AI4PRIVACY/LLAMA-AI4PRIVACY-english-anonymiser-openpii", "ai4privacy/llama-ai4privacy-"),
    ],
)
def test_lookup_ignores_letter_case(asked: str, found: str) -> None:
    # The Hub resolves an id in any case to the same repository, so a case
    # variant must not escape its entry (a restricted model's warning).
    entry = lookup(asked)
    assert entry is not None
    assert entry.model_id == found


@pytest.mark.parametrize(
    "model_id",
    [
        "ai4privacy/llama-ai4privacy-english-anonymiser-openpii",
        "ai4privacy/llama-ai4privacy-multilingual-anonymiser-openpii",
        "ai4privacy/llama-ai4privacy-",
    ],
)
def test_prefix_lookup(model_id: str) -> None:
    entry = lookup(model_id)
    assert entry is not None
    assert entry.prefix
    assert entry.status == "restricted"


@pytest.mark.parametrize(
    "model_id",
    ["", "org/unknown-model", "ai4privacy/llama-ai4priv", "ai4privacy/other-model", "dslim"],
)
def test_unknown_ids_are_not_found(model_id: str) -> None:
    assert lookup(model_id) is None
    assert pinned_revision(model_id) is None


def test_exact_beats_prefix_and_longer_prefix_beats_shorter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = CatalogEntry("org/model-x", ("hf",), "MIT", "vetted", "exact")
    short = CatalogEntry("org/model-", ("hf",), "MIT", "restricted", "short", prefix=True)
    long = CatalogEntry("org/model-y", ("hf",), "MIT", "caution", "long", prefix=True)
    monkeypatch.setattr(model_catalog, "_EXACT", {"org/model-x": exact})
    monkeypatch.setattr(model_catalog, "_PREFIXES", (long, short))
    assert lookup("org/model-x") is exact
    assert lookup("org/model-yz") is long
    assert lookup("org/model-z") is short


def test_pinned_revision() -> None:
    assert pinned_revision("dslim/bert-base-NER") == "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
    # Restricted models are catalogued without a pin.
    assert lookup("nvidia/gliner-PII") is not None
    assert pinned_revision("nvidia/gliner-PII") is None


# --- every entry ---------------------------------------------------------


def test_ids_are_unique_ignoring_case() -> None:
    folded = [model_id.casefold() for model_id in _ENTRY_IDS]
    assert len(folded) == len(set(folded))


@pytest.mark.parametrize("entry", CATALOG, ids=_ENTRY_IDS)
def test_entry_shape(entry: CatalogEntry) -> None:
    assert MODEL_ID_RE.fullmatch(entry.model_id)
    assert entry.status in STATUSES
    assert entry.backends and set(entry.backends) <= set(HUB_BACKENDS)
    assert re.fullmatch(r"LicenseRef-[A-Za-z0-9.-]+|[A-Za-z0-9.+-]+", entry.license)
    assert set(entry.lineage) <= LINEAGE_TAGS
    assert len(set(entry.lineage)) == len(entry.lineage)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry.checked)
    assert entry.checked == CHECKED
    assert entry.card_url.startswith("https://huggingface.co/")
    for type_name in entry.recommended_entities:
        # D10: a sensitive attribute is never recommended.
        assert type_name in TYPE_NAMES
        assert type_name not in SENSITIVE_LABELS
    for label, type_name in entry.labels:
        assert normalize_label(label) == label
        assert type_name == "" or is_placeholder_type(type_name)
    if entry.tagging is not None:
        assert entry.tagging in TAGGING_SCHEMES
        assert entry.backends == ("hf",)
    for onnx_file in entry.onnx_files:
        assert entry.backends == ("gliner",)
        assert onnx_file.endswith(".onnx")
        assert not onnx_file.startswith("/")
        assert ".." not in onnx_file.split("/")
    if entry.prompts:
        assert entry.backends == ("gliner",)
    assert entry.window is None or entry.window > 0
    for distribution, version in entry.min_versions:
        assert re.fullmatch(r"[a-z0-9-]+", distribution)
        assert re.fullmatch(r"\d+(\.\d+)*", version)


@pytest.mark.parametrize("entry", CATALOG, ids=_ENTRY_IDS)
def test_pins_are_full_commits_where_the_catalog_vouches(entry: CatalogEntry) -> None:
    if entry.status == "restricted" or entry.prefix:
        assert entry.revision is None
    else:
        assert entry.revision is not None
        assert REVISION_RE.fullmatch(entry.revision)
    if entry.backbone_revision is not None:
        assert entry.backbone is not None
        assert REVISION_RE.fullmatch(entry.backbone_revision)
    # A prefix names many models: it can carry no single model's facts.
    if entry.prefix:
        assert entry.status == "restricted"
        assert not (entry.prompts or entry.onnx_files or entry.window or entry.backbone)


# Words that would turn a fact into a conclusion (D14: neutral, verifiable
# facts only in the public core).
_CONCLUSIONS = re.compile(
    r"\b(must not|do not use|don't use|avoid|illegal|unlawful|unsafe|dangerous|risky|"
    r"forbidden|prohibited|banned|violat\w*|infring\w*|compliant|non-compliant|recommend\w*)\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("entry", CATALOG, ids=_ENTRY_IDS)
def test_reasons_are_neutral_dated_facts_with_a_link(entry: CatalogEntry) -> None:
    assert entry.reason.strip() == entry.reason
    assert entry.reason
    assert "\n" not in entry.reason
    assert not _CONCLUSIONS.search(entry.reason)
    line = entry.describe()
    assert entry.card_url in line
    assert f"checked {CHECKED}" in line
    assert entry.reason in line


def test_describe_marks_a_prefix() -> None:
    entry = lookup("ai4privacy/llama-ai4privacy-x")
    assert entry is not None
    assert entry.describe().startswith("ai4privacy/llama-ai4privacy-*: ")
    exact = lookup("urchade/gliner_base")
    assert exact is not None
    assert exact.describe().startswith("urchade/gliner_base: CC-BY-NC-4.0")
    assert "(https://huggingface.co/urchade/gliner_base, checked 2026-10-05)" in exact.describe()


def test_section_4_6_statuses() -> None:
    vetted = {e.model_id for e in CATALOG if e.status == "vetted"}
    caution = {e.model_id for e in CATALOG if e.status == "caution"}
    assert vetted == {
        "dslim/bert-base-NER",
        "urchade/gliner_small-v2.1",
        "urchade/gliner_medium-v2.1",
        "urchade/gliner_multi-v2.1",
        "urchade/gliner_multi_pii-v1",
    }
    assert caution == set(_KNOWLEDGATOR)
    restricted = {e.model_id for e in CATALOG if e.status == "restricted"}
    assert {
        "iiiorg/piiranha-v1-detect-personal-information",
        "urchade/gliner_base",
        "nvidia/gliner-PII",
        "bigcode/starpii",
        "ai4privacy/llama-ai4privacy-",
        "knowledgator/gliner-stream-pii-v1.0",
        "perplexity-ai/PII-Tracer",
    } <= restricted


# --- defaults, backbones and the Knowledgator options ---------------------


def test_default_models_are_the_backends_defaults() -> None:
    assert dict(DEFAULT_MODELS) == {"gliner": gliner_ner._MODEL_NAME, "hf": hf_ner._MODEL_NAME}
    for model_id in DEFAULT_MODELS.values():
        entry = lookup(model_id)
        assert entry is not None
        assert entry.status == "vetted"
        assert entry.revision is not None
        assert "default model" in entry.reason


@pytest.mark.parametrize(
    ("model_id", "backbone"),
    [
        ("urchade/gliner_small-v2.1", "microsoft/deberta-v3-small"),
        ("urchade/gliner_medium-v2.1", "microsoft/deberta-v3-base"),
        ("urchade/gliner_multi-v2.1", "microsoft/mdeberta-v3-base"),
        ("urchade/gliner_multi_pii-v1", "microsoft/mdeberta-v3-base"),
    ],
)
def test_checkpoints_without_a_tokenizer_pin_their_backbone(model_id: str, backbone: str) -> None:
    entry = lookup(model_id)
    assert entry is not None
    assert entry.backbone == backbone
    assert entry.backbone_revision is not None


@pytest.mark.parametrize("model_id", _KNOWLEDGATOR)
def test_knowledgator_gliner_pii_is_a_configurable_option(model_id: str) -> None:
    entry = lookup(model_id)
    assert entry is not None
    assert entry.status == "caution"
    assert UNMEASURED in entry.reason
    assert entry.revision is not None
    # Self-contained checkpoints: no backbone snapshot to pin.
    assert entry.backbone is not None
    assert entry.backbone_revision is None
    assert {type_name for type_name, _ in entry.prompts} == set(CANONICAL_NER_TYPES)
    assert entry.recommended_entities == CANONICAL_NER_TYPES
    assert entry.window == 512
    assert {"onnx/model.onnx", "onnx/model_quint8.onnx"} <= set(entry.onnx_files)
    assert entry.lineage == ("undisclosed-training-data",)
    modernbert = entry.backbone.startswith("jhu-clsp/ettin-encoder-")
    assert (("transformers", "4.48.0") in entry.min_versions) == modernbert


def test_prompt_for() -> None:
    entry = lookup("knowledgator/gliner-pii-base-v1.0")
    assert entry is not None
    assert entry.prompt_for("PERSON") == "name"
    assert entry.prompt_for("DATE_OF_BIRTH") == "dob"
    assert entry.prompt_for("EMAIL") is None


_PROMPTS = [(e, t, p) for e in CATALOG for t, p in e.prompts]


@pytest.mark.parametrize(
    ("entry", "type_name", "prompt"),
    _PROMPTS,
    ids=[f"{e.model_id}:{t}" for e, t, _ in _PROMPTS],
)
@pytest.mark.parametrize("fold_raw", [False, True])
def test_every_prompt_override_maps_back_to_its_type(
    entry: CatalogEntry, type_name: str, prompt: str, fold_raw: bool
) -> None:
    assert type_name in TYPE_NAMES
    # A label GLiNER returns as the prompt folds back into the requesting
    # type, through the entry's own label map or the default folds ...
    label = normalize_label(prompt)
    assert dict(entry.labels).get(label, DEFAULT_FOLDS.get(label, label)) == type_name
    # ... and the label policy keeps it as that type when the type is
    # requested, in both raw-entity modes.
    policy = LabelPolicy([type_name], backend="gliner", fold_raw=fold_raw)
    assert policy.classify_gliner(prompt) == type_name
    assert policy.classify(prompt) == type_name


@pytest.mark.parametrize("entry", [e for e in CATALOG if e.prompts], ids=lambda e: e.model_id)
def test_prompts_are_one_per_type_and_distinct(entry: CatalogEntry) -> None:
    types = [type_name for type_name, _ in entry.prompts]
    prompts = [normalize_label(prompt) for _, prompt in entry.prompts]
    assert len(types) == len(set(types))
    assert len(prompts) == len(set(prompts))


# --- the model directory sidecar ------------------------------------------


@pytest.mark.parametrize("revision", [_SHA, None])
def test_sidecar_round_trips(tmp_path: Path, revision: str | None) -> None:
    identity = ModelIdentity("knowledgator/gliner-pii-base-v1.0", revision)
    (tmp_path / SIDECAR_NAME).write_text(sidecar_text(identity))
    assert read_sidecar(tmp_path) == identity
    assert read_sidecar(str(tmp_path)) == identity
    assert json.loads(sidecar_text(identity)) == {
        "model_id": "knowledgator/gliner-pii-base-v1.0",
        "revision": revision,
    }


def test_sidecar_ignores_other_keys(tmp_path: Path) -> None:
    payload = {"model_id": "dslim/bert-base-NER", "revision": _SHA, "files": ["x"], "v": 2}
    (tmp_path / SIDECAR_NAME).write_text(json.dumps(payload))
    assert read_sidecar(tmp_path) == ModelIdentity("dslim/bert-base-NER", _SHA)


def test_no_sidecar_is_no_identity(tmp_path: Path) -> None:
    assert read_sidecar(tmp_path) is None
    assert read_sidecar(tmp_path / "missing") is None
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("x")
    assert read_sidecar(not_a_directory) is None


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        (b"{not json", "not a UTF-8 JSON document"),
        (b"\xff\xfe{}", "not a UTF-8 JSON document"),
        (b"[" * 200 + b"]" * 200, "not a UTF-8 JSON document"),
        (b'["dslim/bert-base-NER"]', "not a JSON object"),
        (b"{}", "model_id must be a Hugging Face model id"),
        (b'{"model_id": 7}', "model_id must be a Hugging Face model id"),
        (b'{"model_id": "../../etc"}', "model_id must be a Hugging Face model id"),
        (b'{"model_id": "a/b", "revision": "main"}', "revision must be a 40-character"),
        (b'{"model_id": "a/b", "revision": "' + _SHA.upper().encode() + b'"}', "revision must"),
        (b'{"model_id": "a/b", "revision": 1}', "revision must be a 40-character"),
    ],
)
def test_a_malformed_sidecar_is_an_error_naming_the_file(
    tmp_path: Path, content: bytes, problem: str
) -> None:
    (tmp_path / SIDECAR_NAME).write_bytes(content)
    with pytest.raises(SidecarError, match=re.escape(problem)) as caught:
        read_sidecar(tmp_path)
    assert str(tmp_path / SIDECAR_NAME) in str(caught.value)
    # Never the file's content.
    assert "main" not in str(caught.value).replace(str(tmp_path), "")


def test_an_oversized_sidecar_is_refused_unread(tmp_path: Path) -> None:
    (tmp_path / SIDECAR_NAME).write_bytes(b" " * (MAX_SIDECAR_BYTES + 1))
    with pytest.raises(SidecarError, match="larger than"):
        read_sidecar(tmp_path)


def test_an_unreadable_sidecar_is_an_error(tmp_path: Path) -> None:
    (tmp_path / SIDECAR_NAME).mkdir()
    with pytest.raises(SidecarError, match=r"cannot be read \(IsADirectoryError\)"):
        read_sidecar(tmp_path)


def test_identify(tmp_path: Path) -> None:
    assert identify("dslim/bert-base-NER") == ModelIdentity("dslim/bert-base-NER")
    # A local directory is identified by its sidecar only.
    assert identify(str(tmp_path)) is None
    (tmp_path / SIDECAR_NAME).write_text(sidecar_text(ModelIdentity("urchade/gliner_base")))
    assert identify(str(tmp_path)) == ModelIdentity("urchade/gliner_base")
