"""The ``gliner2`` NER backend (Fastino GLiNER2, ``gliner2`` extra).

GLiNER2 is zero-shot like GLiNER: it is prompted with the label policy's
prompts and answers each entity with a confidence and its character span,
which the detector uses as given — never a search for the surface text, which
would put a repeated value at its first occurrence. Long strings are read in
windows of the model's own words. The model loads from a self-contained local
folder at a pinned revision. A fake model stands in (tests/ner_fakes.py); the
one ``real_model`` test needs the gliner2 extra and a cached
fastino/gliner2-base-v1 and is deselected by default.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from llm_redact.config import ConfigError, load_config
from llm_redact.config_write import emit_config_toml
from llm_redact.detection import gliner2_ner
from llm_redact.detection.engine import DetectionConfig, NerConfig, build_detectors
from llm_redact.detection.gliner2_ner import (
    Gliner2Detector,
    build_gliner2_detector,
    gliner2_words,
)
from llm_redact.detection.labels import LabelPolicy
from llm_redact.detection.model_files import GLINER2_PATTERNS, gliner2_model_dir
from llm_redact.detection.ner import NER_PRIORITY
from ner_fakes import DEFAULT_REPO, FakeGliner2, FakeHub, install_gliner2, install_hub
from real_models import cached_snapshot, offline_hub

BASE = "fastino/gliner2-base-v1"
BASE_PIN = "f9634218e53580c56edf0de97ca1a7d3f1c2354e"


def _detector(
    findings: list[tuple[str, str, float]],
    entities: tuple[str, ...] = ("PERSON",),
    *,
    max_chars: int = 100_000,
    threshold: float = 0.5,
    model: Any = None,
) -> tuple[Gliner2Detector, FakeGliner2]:
    fake = model if model is not None else FakeGliner2(findings)
    detector = Gliner2Detector(
        fake,
        frozenset(entities),
        max_chars,
        threshold,
        policy=LabelPolicy(entities, backend="gliner2"),
    )
    return detector, fake


# --- spans, labels and the threshold ----------------------------------------------


def test_spans_come_from_the_model_not_a_text_search() -> None:
    # The same name twice: each detection is at its own offsets.
    text = "Jane Roe wrote to Jane Roe."
    detector, fake = _detector([("Jane Roe", "person", 0.9)])
    found = detector.detect(text)
    assert [(d.start, d.end) for d in found] == [(0, 8), (18, 26)]
    assert all(d.value == "Jane Roe" and d.detector_type == "PERSON" for d in found)
    assert all(d.priority == NER_PRIORITY for d in found)
    # A type request sends its natural-language prompt.
    assert fake.calls == [["person"]]


def test_the_threshold_reaches_the_model() -> None:
    detector, _ = _detector([("Jane Roe", "person", 0.6)], threshold=0.7)
    assert detector.detect("hi Jane Roe") == []
    detector, _ = _detector([("Jane Roe", "person", 0.6)], threshold=0.5)
    assert [d.value for d in detector.detect("hi Jane Roe")] == ["Jane Roe"]


def test_a_type_request_and_its_prompt_spelled_raw_send_one_prompt() -> None:
    # "phone number" is PHONE's prompt: sent once, emitted as PHONE.
    detector, fake = _detector([("555 0100", "phone number", 0.9)], ("PHONE", "phone number"))
    assert [d.detector_type for d in detector.detect("call 555 0100")] == ["PHONE"]
    assert fake.calls == [["phone number"]]


def test_a_raw_request_keeps_its_own_type_until_2_0() -> None:
    detector, _ = _detector([("CTO", "job title", 0.9)], ("job title",))
    assert [d.detector_type for d in detector.detect("the CTO said")] == ["JOB_TITLE"]


def test_first_and_last_name_parts_merge() -> None:
    detector, _ = _detector([("Jane", "person", 0.9), ("Roe", "person", 0.9)])
    assert [d.value for d in detector.detect("hi Jane Roe!")] == ["Jane Roe"]


def test_nothing_requested_never_calls_the_model() -> None:
    detector, fake = _detector([("Jane", "person", 0.9)], ())
    assert detector.detect("Jane") == []
    assert fake.calls == []


def test_a_span_outside_the_text_or_without_offsets_is_never_redacted() -> None:
    class Odd(FakeGliner2):
        def extract_entities(self, text: str, entity_types: list[str], **kwargs: Any) -> Any:
            return {
                "entities": {
                    "person": [
                        {"text": "x", "start": 0, "end": len(text) + 5},
                        {"text": "x", "start": 3, "end": 3},
                        {"text": "x", "start": "0", "end": 2},
                        "Jane",  # a span-less form
                        {"text": "Jane", "start": 3, "end": 7},
                    ],
                    "street address": None,
                }
            }

    detector, _ = _detector([], model=Odd([]))
    assert [d.value for d in detector.detect("hi Jane")] == ["Jane"]
    assert detector.stats.offsets_dropped == 4


class Punctuating(FakeGliner2):
    """gliner2 2.0.0 adds a "." to a text ending in none of ".!?" and reads
    its spans on that text; this model's one span runs from ``start`` to
    the end of that longer text, the added "." included."""

    def __init__(self, start: int) -> None:
        super().__init__([])
        self.start = start

    def extract_entities(self, text: str, entity_types: list[str], **kwargs: Any) -> Any:
        read = text if text.endswith((".", "!", "?")) else text + "."
        found = {"text": read[self.start :], "start": self.start, "end": len(read)}
        return {"entities": {"person": [found]}}


@pytest.mark.parametrize(
    ("text", "start", "value"),
    [
        ("hi Jane Roe", 3, "Jane Roe"),  # "Jane Roe." in gliner2's text
        ("hi Jane Roe ", 3, "Jane Roe"),  # "Jane Roe ." : the blank too
    ],
)
def test_a_span_taking_in_the_added_period_ends_at_the_text(
    text: str, start: int, value: str
) -> None:
    # Dropped as "outside the text", the name inside it went upstream.
    detector, _ = _detector([], model=Punctuating(start))
    found = detector.detect(text)
    assert [(d.start, d.value) for d in found] == [(start, value)]
    assert detector.stats.offsets_dropped == 0


@pytest.mark.parametrize("text", ["hi Jane Roe.", "hi Jane Roe!"])
def test_no_period_is_added_to_a_text_ending_a_sentence(text: str) -> None:
    # Nothing was added, so a span past the end is outside the text.
    class Past(FakeGliner2):
        def extract_entities(self, text: str, entity_types: list[str], **kwargs: Any) -> Any:
            return {"entities": {"person": [{"text": "x", "start": 3, "end": len(text) + 1}]}}

    detector, _ = _detector([], model=Past([]))
    assert detector.detect(text) == []
    assert detector.stats.offsets_dropped == 1


def test_a_span_of_only_the_added_period_is_dropped() -> None:
    detector, _ = _detector([], model=Punctuating(len("hi Jane")))
    assert detector.detect("hi Jane") == []
    assert detector.stats.offsets_dropped == 1
    # A blank before it leaves nothing either.
    detector, _ = _detector([], model=Punctuating(len("hi Jane")))
    assert detector.detect("hi Jane ") == []
    assert detector.stats.offsets_dropped == 1


def test_a_type_outside_the_placeholder_grammar_is_dropped_and_counted() -> None:
    entity = "a" * 40  # a raw request whose type is too long to be a placeholder
    detector, _ = _detector([("Jane", entity, 0.9)], (entity,))
    assert detector.detect("hi Jane") == []
    assert detector.stats.labels_dropped == 1


def test_max_chars_skips_before_any_call() -> None:
    detector, fake = _detector([("Jane", "person", 0.9)], max_chars=10)
    assert detector.detect("hello there Jane") == []
    assert fake.calls == []
    assert detector.stats.skipped_max_chars == 1


# --- long strings -------------------------------------------------------------------


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_a_short_text_is_one_call_as_sent() -> None:
    detector, fake = _detector([("Jane", "person", 0.9)])
    text = '{"name": "Jane"}'
    assert [d.value for d in detector.detect(text)] == ["Jane"]
    assert fake.texts == [text]
    assert detector.stats.scanned_whole == 1
    assert detector.stats.windows == 0


def test_a_long_text_is_read_in_windows_with_absolute_offsets() -> None:
    detector, fake = _detector([("Jane Roe", "person", 0.9)])
    text = _words(1500) + " Jane Roe " + _words(300)
    found = detector.detect(text)
    start = text.index("Jane Roe")
    assert [(d.start, d.end, d.value) for d in found] == [(start, start + 8, "Jane Roe")]
    assert detector.stats.scanned_windowed == 1
    assert detector.stats.windows == len(fake.texts) > 1
    # Every call holds at most 200 of GLiNER2's words.
    assert all(len(gliner2_words(piece)) <= 200 for piece in fake.texts)


def test_an_entity_in_an_overlap_is_reported_once() -> None:
    detector, fake = _detector([("Jane Roe", "person", 0.9)])
    # Words 160-199 are in the first window and the second.
    text = _words(170) + " Jane Roe " + _words(100)
    assert [d.value for d in detector.detect(text)] == ["Jane Roe"]
    assert sum("Jane Roe" in piece for piece in fake.texts) == 2


def test_json_is_windowed_by_gliner2_words() -> None:
    # Every brace, quote, colon and comma is a word; an e-mail address or a
    # URL is one.
    assert len(gliner2_words('{"a": "b"}')) == 9
    assert gliner2_words("mail jane@example.com or https://x.example/a?b=c") == [
        (0, 4),
        (5, 21),
        (22, 24),
        (25, 48),
    ]
    detector, fake = _detector([])
    detector.detect(", ".join(['{"k": "v"}'] * 40))
    assert len(fake.texts) > 1
    assert all(len(gliner2_words(piece)) <= 200 for piece in fake.texts)


class FakeEncoding:
    def __init__(self, word_ids: list[int | None]) -> None:
        self._word_ids = word_ids
        self.input_ids = list(range(len(word_ids)))

    def word_ids(self) -> list[int | None]:
        return self._word_ids

    def __getitem__(self, key: str) -> list[int]:
        assert key == "input_ids"
        return self.input_ids


class FakeFastTokenizer:
    """Each word costs one token per 3 characters (at least one); every
    prompt word is one token. Records the words it was asked to count."""

    is_fast = True

    def __init__(self, model_max_length: int = int(1e30)) -> None:
        self.model_max_length = model_max_length
        self.counted: list[list[str]] = []

    def num_special_tokens_to_add(self) -> int:
        return 2

    def __call__(self, words: list[str], **kwargs: Any) -> FakeEncoding:
        assert kwargs == {"is_split_into_words": True, "add_special_tokens": False}
        self.counted.append(words)
        ids: list[int | None] = []
        for index, word in enumerate(words):
            ids += [index] * max(1, len(word) // 3)
        return FakeEncoding(ids)


def _with_processor(tokenizer: Any, positions: int | None = 64) -> FakeGliner2:
    model = FakeGliner2([("Jane Roe", "person", 0.9)])
    model.processor = types.SimpleNamespace(tokenizer=tokenizer)  # type: ignore[attr-defined]
    config = types.SimpleNamespace(max_position_embeddings=positions)
    model.encoder = types.SimpleNamespace(config=config)  # type: ignore[attr-defined]
    return model


def test_windows_fit_the_encoder_beside_the_prompt() -> None:
    tokenizer = FakeFastTokenizer()
    detector, _ = _detector([], model=_with_processor(tokenizer))
    # ( [P] entities ( [E] person ) ) [SEP_TEXT]: 13 tokens in the fake
    # (a token per 3 characters); 64 positions - 13 - 2 specials.
    assert tokenizer.counted[0] == [
        "(",
        "[P]",
        "entities",
        "(",
        "[E]",
        "person",
        ")",
        ")",
        "[SEP_TEXT]",
    ]
    assert detector.subword_budget == 49
    model = _with_processor(FakeFastTokenizer(), positions=None)
    assert _detector([], model=model)[0].subword_budget == 512 - 13 - 2
    assert _detector([], model=_with_processor(FakeFastTokenizer(40)))[0].subword_budget == 25


def test_word_costs_are_counted_on_lowercased_words_and_bound_windows() -> None:
    tokenizer = FakeFastTokenizer()
    model = _with_processor(tokenizer)
    detector, _ = _detector([], model=model)
    text = " ".join(["Abcdefghi"] * 30) + " Jane Roe"  # 3 tokens a word
    found = detector.detect(text)
    assert [d.value for d in found] == ["Jane Roe"]
    assert tokenizer.counted[-1][0] == "abcdefghi"  # gliner2 lowercases each word
    assert all(sum(max(1, len(w) // 3) for w in t.split()) <= 49 for t in model.texts)
    assert detector.stats.scanned_windowed == 1


def test_one_word_longer_than_the_encoder_reads_is_counted() -> None:
    detector, _ = _detector([], model=_with_processor(FakeFastTokenizer()))
    detector.detect("short " + "x" * 400 + " tail")
    assert detector.stats.windows_truncated == 1


def test_a_slow_tokenizer_bounds_windows_by_words_only() -> None:
    slow = FakeFastTokenizer()
    slow.is_fast = False
    detector, _ = _detector([], model=_with_processor(slow))
    assert detector.subword_budget is None


def test_the_models_own_word_splitter_is_used() -> None:
    model = FakeGliner2([])
    seen: list[tuple[str, bool]] = []

    def splitter(text: str, lower: bool = True) -> Any:
        seen.append((text, lower))
        return iter([("a", 0, 1)])

    model.processor = types.SimpleNamespace(word_splitter=splitter)  # type: ignore[attr-defined]
    detector, _ = _detector([], model=model)
    detector.detect("a")
    assert seen == [("a", False)]


# --- building -----------------------------------------------------------------------


def test_the_default_model_loads_pinned_from_its_local_folder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = FakeHub()
    model = FakeGliner2([("Jane Roe", "person", 0.9)])
    install_gliner2(monkeypatch, model, hub)
    detector = build_gliner2_detector(NerConfig(enabled=True, backend="gliner2"))
    (call,) = hub.calls
    assert call["repo_id"] == BASE
    assert call["revision"] == BASE_PIN
    assert call["allow_patterns"] == list(GLINER2_PATTERNS)
    assert call["local_files_only"] is True
    folder = hub.snapshot_download(BASE, revision=BASE_PIN)
    assert model.loaded_with == [
        {"model_id": folder, "local_files_only": True, "map_location": "cpu"}
    ]
    # gliner2's load banner never reaches stdout (`--json` output).
    assert capsys.readouterr().out == ""
    assert detector.model_name == BASE
    assert [d.value for d in detector.detect("hi Jane Roe")] == ["Jane Roe"]


def test_the_builder_hands_the_overrides_and_threshold_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeGliner2([("Springfield", "CITY", 0.8)])
    install_gliner2(monkeypatch, model)
    ner = NerConfig(
        enabled=True,
        backend="gliner2",
        entities=("ADDRESS", "CITY"),
        labels=(("CITY", "ADDRESS"),),
        score_threshold=0.75,
        model="org/gliner2-pii",
    )
    detector = build_gliner2_detector(ner)
    assert [d.detector_type for d in detector.detect("in Springfield")] == ["ADDRESS"]
    assert detector.model_name == "org/gliner2-pii"


def test_a_missing_extra_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gliner2", None)
    with pytest.raises(ConfigError) as caught:
        build_gliner2_detector(NerConfig(enabled=True, backend="gliner2"))
    assert str(caught.value) == (
        '[detection.ner] backend = "gliner2" but the gliner2 extra is not installed;'
        " install it: uv sync --extra gliner2"
    )


def test_a_failed_load_names_only_the_exception_type(monkeypatch: pytest.MonkeyPatch) -> None:
    install_gliner2(monkeypatch, FakeGliner2([]))

    def broken(name: str, **kwargs: Any) -> Any:
        raise RuntimeError("secret detail")

    monkeypatch.setattr(sys.modules["gliner2"].AutoExtractor, "from_pretrained", broken)
    with pytest.raises(ConfigError) as caught:
        build_gliner2_detector(NerConfig(enabled=True, backend="gliner2", model="org/m"))
    assert str(caught.value) == "failed to load GLiNER2 model 'org/m': RuntimeError"


def test_a_module_gliner2_imports_at_load_is_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # gliner2 imports peft only when it loads a model: its import succeeds
    # without it, the load does not.
    install_gliner2(monkeypatch, FakeGliner2([]))

    def without_peft(name: str, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("No module named 'peft'", name="peft")

    monkeypatch.setattr(sys.modules["gliner2"].AutoExtractor, "from_pretrained", without_peft)
    with pytest.raises(ConfigError) as caught:
        build_gliner2_detector(NerConfig(enabled=True, backend="gliner2", model="org/m"))
    assert str(caught.value) == (
        '[detection.ner] backend = "gliner2" but the gliner2 extra is not installed;'
        " install it: uv sync --extra gliner2"
    )


def test_gliner2s_own_logs_are_silenced(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # gliner2 names a word of the text it reads at WARNING (a word making no
    # subword) and logs a failed extraction's traceback at ERROR.
    import logging

    logger = logging.getLogger("gliner2")
    level = logger.level
    install_gliner2(monkeypatch, FakeGliner2([]))
    try:
        build_gliner2_detector(NerConfig(enabled=True, backend="gliner2"))
        assert logging.getLogger("gliner2.processor").getEffectiveLevel() > logging.CRITICAL
        with caplog.at_level(logging.DEBUG):
            logging.getLogger("gliner2.processor").warning("word %r made no subwords", "x")
            logging.getLogger("gliner2.inference.runtime").error("extraction failed")
        assert caplog.records == []
    finally:
        logger.setLevel(level)


def test_build_detectors_dispatches_and_goes_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    install_gliner2(monkeypatch, FakeGliner2([("Jane Roe", "person", 0.9)]))
    import os

    (detector,) = build_detectors(
        DetectionConfig(enabled=(), ner=NerConfig(enabled=True, backend="gliner2"))
    )
    assert detector.inner.name == "gliner2"  # type: ignore[attr-defined]
    assert [d.value for d in detector.detect("hi Jane Roe")] == ["Jane Roe"]
    # A build that may not download set the Hugging Face offline switches.
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_gliner2_runs_beside_other_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    from ner_fakes import FakeGliner, install_gliner

    install_gliner(monkeypatch, FakeGliner([("Jane Roe", "person", 0.9)]))
    install_gliner2(monkeypatch, FakeGliner2([("Jane Roe", "person", 0.9)]))
    detectors = build_detectors(
        DetectionConfig(enabled=(), ner=NerConfig(enabled=True, backends=("gliner", "gliner2")))
    )
    assert [d.inner.name for d in detectors] == ["gliner", "gliner2"]  # type: ignore[attr-defined]


# --- model files ----------------------------------------------------------------------


def _hub_with(files: dict[str, str]) -> FakeHub:
    return FakeHub(repos={"org/g2": files})


def test_a_checkpoint_without_its_encoder_configuration_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {k: v for k, v in DEFAULT_REPO.items() if k != "encoder_config/config.json"}
    install_hub(monkeypatch, _hub_with(files))
    with pytest.raises(ConfigError) as caught:
        gliner2_model_dir("org/g2", revision=None, allow_download=False)
    assert str(caught.value) == (
        "[detection.ner] gliner2 model 'org/g2' has no encoder_config/config.json; a GLiNER2"
        " checkpoint ships its configuration, its encoder configuration and its tokenizer"
    )


def test_a_checkpoint_without_its_tokenizer_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {k: v for k, v in DEFAULT_REPO.items() if k != "tokenizer_config.json"}
    install_hub(monkeypatch, _hub_with(files))
    with pytest.raises(ConfigError, match="has no tokenizer_config.json"):
        gliner2_model_dir("org/g2", revision=None, allow_download=False)


@pytest.mark.parametrize("name", ["config.json", "encoder_config/config.json"])
def test_a_configuration_naming_code_is_refused(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    files = {**DEFAULT_REPO, name: json.dumps({"model_type": "deberta-v2", "auto_map": {}})}
    install_hub(monkeypatch, _hub_with(files))
    with pytest.raises(ConfigError, match=f"needs code from its repository \\({name} names"):
        gliner2_model_dir("org/g2", revision=None, allow_download=False)


def test_an_encoder_type_transformers_does_not_know_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ner_fakes import fake_transformers

    fake_transformers(monkeypatch)
    files = {**DEFAULT_REPO, "encoder_config/config.json": json.dumps({"model_type": "evil"})}
    install_hub(monkeypatch, _hub_with(files))
    with pytest.raises(ConfigError) as caught:
        gliner2_model_dir("org/g2", revision=None, allow_download=False)
    assert str(caught.value) == (
        "[detection.ner] gliner2 model 'org/g2': encoder_config/config.json names a model type"
        " transformers does not know; llm-redact never runs model code"
    )


def test_the_model_type_check_needs_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    install_hub(monkeypatch, _hub_with(dict(DEFAULT_REPO)))
    with pytest.raises(ConfigError) as caught:
        gliner2_model_dir("org/g2", revision=None, allow_download=False)
    assert str(caught.value) == (
        '[detection.ner] backend = "gliner2" needs transformers, which the gliner2 extra'
        " installs; install it: uv sync --extra gliner2"
    )


def test_a_checkpoint_without_safetensors_fetches_its_weights_only_bin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ner_fakes import fake_transformers

    fake_transformers(monkeypatch)
    files = {k: v for k, v in DEFAULT_REPO.items() if k != "model.safetensors"}
    hub = install_hub(monkeypatch, _hub_with({**files, "pytorch_model.bin": ""}))
    path = gliner2_model_dir("org/g2", revision=None, allow_download=False)
    assert (path / "pytorch_model.bin").is_file()
    assert hub.calls[1]["allow_patterns"] == [*GLINER2_PATTERNS, "pytorch_model.bin"]


def test_a_local_folder_loads_as_it_is(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ner_fakes import fake_transformers

    fake_transformers(monkeypatch)
    hub = install_hub(monkeypatch)
    for name, content in DEFAULT_REPO.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(content)
    assert gliner2_model_dir(str(tmp_path), revision=None, allow_download=False) == tmp_path
    assert hub.calls == []


# --- configuration ------------------------------------------------------------------


def _load(tmp_path: Path, ner: str) -> NerConfig:
    path = tmp_path / "config.toml"
    path.write_text(f"[detection.ner]\n{ner}\n")
    return load_config(path).detection.ner


def test_the_parser_knows_the_backend(tmp_path: Path) -> None:
    ner = _load(
        tmp_path,
        'enabled = true\nbackends = ["gliner2"]\nscore_threshold = 0.8\n'
        '[detection.ner.models]\ngliner2 = "org/g2"\n'
        f'[detection.ner.revisions]\ngliner2 = "{BASE_PIN}"\n',
    )
    assert ner.active_backends() == ("gliner2",)
    assert ner.score_threshold == 0.8
    assert ner.model_for("gliner2") == "org/g2"
    assert ner.revision_for("gliner2") == BASE_PIN


def test_a_gliner2_only_threshold_round_trips(tmp_path: Path) -> None:
    _load(tmp_path, 'backend = "gliner2"\nscore_threshold = 0.8\n')
    path = tmp_path / "config.toml"
    emitted = emit_config_toml(load_config(path))
    assert "score_threshold = 0.8" in emitted
    path.write_text(emitted)
    assert load_config(path).detection.ner.score_threshold == 0.8


def test_gliner2_loads_no_onnx_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="only the gliner backend loads ONNX weights"):
        _load(tmp_path, '[detection.ner.onnx]\ngliner2 = "onnx/model.onnx"\n')


# --- a real model (deselected by default) ---------------------------------------------

BASE_FILES = list(GLINER2_PATTERNS)


@pytest.mark.real_model
def test_real_model_finds_names_at_their_own_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    # fastino/gliner2-base-v1 (Apache-2.0) at the catalog pin, checked 2026-10-06.
    pytest.importorskip("torch")
    pytest.importorskip("gliner2")
    offline_hub(monkeypatch)
    cached_snapshot(BASE, BASE_PIN, BASE_FILES)
    detector = build_gliner2_detector(NerConfig(enabled=True, backend="gliner2"))
    text = 'Contact Angela Merkel at {"name":"Zbigniew Brzezinski"} today. Angela Merkel again.'
    found = detector.detect(text)
    assert sorted((d.start, d.value) for d in found) == [
        (8, "Angela Merkel"),
        (34, "Zbigniew Brzezinski"),
        (63, "Angela Merkel"),
    ]
    long_text = "word " * 3000 + "Angela Merkel met Barack Obama."
    found = detector.detect(long_text)
    assert [d.value for d in found] == ["Angela Merkel", "Barack Obama"]
    assert all(long_text[d.start : d.end] == d.value for d in found)
    assert detector.stats.scanned_windowed == 1


def test_module_constants() -> None:
    assert gliner2_ner._MODEL_NAME == BASE
