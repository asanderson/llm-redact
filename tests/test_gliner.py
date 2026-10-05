"""GlinerDetector tests via an injectable fake model — no gliner install needed."""

from typing import Any

import pytest

from llm_redact.detection.engine import (
    Allowlist,
    DetectionConfig,
    NerConfig,
    build_detectors,
    detect_all,
)
from llm_redact.detection.gliner_ner import GlinerDetector
from llm_redact.detection.ner import NER_PRIORITY

NO_ALLOW = Allowlist(exact=frozenset(), patterns=())


class FakeModel:
    """Recognizes 'Jane Doe' when prompted "person" (score 0.9) and 'Acme'
    when prompted "ORG" (0.3). A PERSON type request sends the
    natural-language prompt "person"; the raw request ORG is sent verbatim."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], float]] = []

    def predict_entities(
        self, text: str, labels: list[str], threshold: float
    ) -> list[dict[str, Any]]:
        self.calls.append((text, labels, threshold))
        entities = []
        for name, label, score in (("Jane Doe", "person", 0.9), ("Acme", "ORG", 0.3)):
            index = text.find(name)
            if index != -1 and label in labels and score >= threshold:
                entities.append(
                    {
                        "start": index,
                        "end": index + len(name),
                        "label": label,
                        "text": name,
                        "score": score,
                    }
                )
        return entities


def test_entity_mapping_and_priority() -> None:
    detector = GlinerDetector(FakeModel(), frozenset({"PERSON"}), max_chars=1000, threshold=0.5)
    detections = list(detector.detect("ask Jane Doe about Acme"))
    assert len(detections) == 1
    d = detections[0]
    assert (d.detector_type, d.value, d.priority) == ("PERSON", "Jane Doe", NER_PRIORITY)


def test_threshold_filters_low_scores() -> None:
    detector = GlinerDetector(
        FakeModel(), frozenset({"PERSON", "ORG"}), max_chars=1000, threshold=0.5
    )
    types = [d.detector_type for d in detector.detect("Jane Doe works at Acme")]
    assert types == ["PERSON"]  # ORG scored 0.3 < 0.5

    permissive = GlinerDetector(
        FakeModel(), frozenset({"PERSON", "ORG"}), max_chars=1000, threshold=0.2
    )
    types = [d.detector_type for d in permissive.detect("Jane Doe works at Acme")]
    assert types == ["PERSON", "ORG"]


def test_type_request_sends_its_natural_language_prompt() -> None:
    model = FakeModel()
    detector = GlinerDetector(model, frozenset({"PERSON", "ORG"}), max_chars=1000, threshold=0.5)
    list(detector.detect("Jane Doe"))
    ((_text, labels, threshold),) = model.calls
    assert labels == ["person", "ORG"]  # type request's prompt, then the raw one
    assert threshold == 0.5


def test_max_chars_gate() -> None:
    model = FakeModel()
    detector = GlinerDetector(model, frozenset({"PERSON"}), max_chars=10, threshold=0.5)
    assert list(detector.detect("Jane Doe " + "x" * 100)) == []
    assert model.calls == []


def test_label_case_normalized() -> None:
    class WeirdLabelModel:
        def predict_entities(
            self, text: str, labels: list[str], threshold: float
        ) -> list[dict[str, Any]]:
            return [{"start": 0, "end": 4, "label": "job title", "text": "misc", "score": 0.9}]

    detector = GlinerDetector(
        WeirdLabelModel(), frozenset({"job title"}), max_chars=100, threshold=0.5
    )
    assert next(iter(detector.detect("misc"))).detector_type == "JOB_TITLE"


def test_allowlist_applies() -> None:
    detectors: list[Any] = [
        GlinerDetector(FakeModel(), frozenset({"PERSON"}), max_chars=1000, threshold=0.5)
    ]
    assert [d.value for d in detect_all(detectors, "ask Jane Doe", NO_ALLOW)] == ["Jane Doe"]
    allow = Allowlist(exact=frozenset({"Jane Doe"}), patterns=())
    assert detect_all(detectors, "ask Jane Doe", allow) == []


def test_enabled_without_gliner_config_error() -> None:
    try:
        import gliner  # noqa: F401

        pytest.skip("gliner installed; missing-dependency path not testable")
    except ImportError:
        pass
    from llm_redact.config import ConfigError

    with pytest.raises(ConfigError, match="uv sync --extra gliner"):
        build_detectors(DetectionConfig(ner=NerConfig(enabled=True, backend="gliner")))


def test_config_rejects_threshold_for_spacy(tmp_path: Any) -> None:
    from llm_redact.config import ConfigError, load_config

    config_file = tmp_path / "c.toml"
    config_file.write_text("[detection.ner]\nenabled = false\nscore_threshold = 0.7\n")
    with pytest.raises(ConfigError, match="score_threshold"):
        load_config(config_file)


def test_config_accepts_threshold_for_gliner(tmp_path: Any) -> None:
    from llm_redact.config import load_config

    config_file = tmp_path / "c.toml"
    config_file.write_text(
        '[detection.ner]\nenabled = false\nbackend = "gliner"\nscore_threshold = 0.7\n'
    )
    config = load_config(config_file)
    assert config.detection.ner.backend == "gliner"
    assert config.detection.ner.score_threshold == 0.7


# --- long strings: GLiNER word windows ------------------------------------------

import json  # noqa: E402
import re  # noqa: E402
import types  # noqa: E402

from llm_redact.config import ConfigError  # noqa: E402
from llm_redact.detection.gliner_ner import (  # noqa: E402
    _subword_budget,
    _window_words,
    build_gliner_detector,
)
from llm_redact.detection.windows import gliner_words, word_windows  # noqa: E402
from llm_redact.redactor import Redactor  # noqa: E402
from llm_redact.vault import InMemoryVault  # noqa: E402
from real_models import cached_snapshot, offline_hub  # noqa: E402


def _filler(words: int) -> str:
    return " ".join(f"w{i}" for i in range(words))


class WindowModel:
    """GLiNER's real call shape — predict_entities(text, labels, flat_ner,
    threshold): a threshold passed by position would land in flat_ner. It
    reports "Jane Doe" (prompt "person", 0.9) wherever it occurs in the text
    it is handed, and the part of it inside that text when the text cuts it
    (a window edge)."""

    def __init__(self, processor: Any = None, config: Any = None) -> None:
        self.calls: list[str] = []
        self.thresholds: list[float] = []
        if processor is not None:
            self.data_processor = processor
        if config is not None:
            self.config = config

    def predict_entities(
        self, text: str, labels: list[str], flat_ner: bool = True, threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        self.calls.append(text)
        self.thresholds.append(threshold)
        found = []
        for surface in ("Jane Doe", "Jane", "Doe"):
            for match in re.finditer(re.escape(surface), text):
                found.append({"start": match.start(), "end": match.end(), "label": "person"})
        # Whole names win; a part counts only where the whole is cut off.
        whole = [(f["start"], f["end"]) for f in found if f["end"] - f["start"] == 8]
        return [
            {**f, "text": text[f["start"] : f["end"]], "score": 0.9}
            for f in found
            if f["end"] - f["start"] == 8
            or not any(s <= f["start"] and f["end"] <= e for s, e in whole)
            and (f["start"] == 0 or f["end"] == len(text))
        ]


def _detector(model: Any, max_chars: int = 1_000_000) -> GlinerDetector:
    return GlinerDetector(model, frozenset({"PERSON"}), max_chars, 0.5)


def test_a_long_string_is_read_in_gliner_word_windows() -> None:
    model = WindowModel()
    detector = _detector(model)
    assert detector.window_words == 200  # min(200, 384 - (2 * 1 + 1) - 16)
    text = _filler(1500) + " Jane Doe " + _filler(300)
    (found,) = detector.detect(text)
    start = text.index("Jane Doe")
    assert (found.start, found.end, found.value, found.detector_type) == (
        start,
        start + 8,
        "Jane Doe",
        "PERSON",
    )
    assert max(len(gliner_words(call)) for call in model.calls) <= 200
    assert detector.stats.scanned_windowed == 1
    assert detector.stats.windows == len(model.calls) == 12  # 1,802 words, 160 a step
    assert detector.stats.scanned_whole == 0


def test_an_entity_in_an_overlap_is_reported_once() -> None:
    model = WindowModel()
    # Windows: words 0-199, then 160-359 — words 160-199 are in both.
    text = _filler(170) + " Jane Doe " + _filler(250)
    assert [d.value for d in _detector(model).detect(text)] == ["Jane Doe"]
    assert sum("Jane Doe" in call for call in model.calls) == 2


def test_a_name_cut_by_a_seam_is_redacted_whole() -> None:
    # The first window ends after "Jane" (word 199); the second holds the
    # whole name. The redactor's overlap resolution keeps the longer span.
    model = WindowModel()
    text = _filler(199) + " Jane Doe " + _filler(100)
    assert {d.value for d in _detector(model).detect(text)} == {"Jane", "Jane Doe"}
    redacted = Redactor([_detector(WindowModel())], InMemoryVault(), NO_ALLOW).redact_text(text)
    assert "Jane" not in redacted and "Doe" not in redacted
    assert redacted.count("«PERSON_001»") == 1


def test_json_dense_text_is_windowed_by_gliner_words() -> None:
    # No whitespace at all, yet over a thousand GLiNER words: every brace,
    # quote, colon and comma counts, as GLiNER itself counts them.
    model = WindowModel()
    text = json.dumps({f"k{i}": i for i in range(300)}, separators=(",", ":"))
    assert len(text.split()) == 1 and len(gliner_words(text)) > 1000
    detector = _detector(model)
    detector.detect(text)
    assert detector.stats.scanned_windowed == 1
    assert max(len(gliner_words(call)) for call in model.calls) <= 200


def test_a_short_text_is_one_call_with_the_text_as_sent() -> None:
    model = WindowModel()
    detector = _detector(model)
    text = "  ask Jane Doe  "
    assert [d.value for d in detector.detect(text)] == ["Jane Doe"]
    assert model.calls == [text]
    assert detector.stats.scanned_whole == 1
    assert detector.stats.windows == 0


def test_text_without_words_is_one_call() -> None:
    model = WindowModel()
    detector = _detector(model)
    assert detector.detect("   ") == []
    assert model.calls == ["   "]
    assert detector.stats.scanned_whole == 1


def test_the_threshold_reaches_gliner_by_keyword() -> None:
    # GLiNER's third parameter is flat_ner: a positional threshold was
    # silently ignored (every model call used GLiNER's default 0.5).
    model = WindowModel()
    GlinerDetector(model, frozenset({"PERSON"}), 1000, 0.8).detect("ask Jane Doe")
    assert model.thresholds == [0.8]


def test_offsets_outside_a_window_are_dropped() -> None:
    class OutOfRange(WindowModel):
        def predict_entities(
            self, text: str, labels: list[str], flat_ner: bool = True, threshold: float = 0.5
        ) -> list[dict[str, Any]]:
            super().predict_entities(text, labels, flat_ner, threshold)
            return [{"start": 0, "end": len(text) + 1, "label": "person", "score": 0.9}]

    detector = _detector(OutOfRange())
    assert detector.detect(_filler(500)) == []
    assert detector.stats.offsets_dropped == detector.stats.windows == 3


def _whitespace_splitter(text: str) -> Any:
    for match in re.finditer(r"\S+", text):
        yield match.group(), match.start(), match.end()


def test_the_models_own_words_splitter_is_used() -> None:
    # A model configured with another splitter (here: whitespace runs) is
    # windowed by ITS words: the JSON-dense text is one word, one window.
    processor = types.SimpleNamespace(words_splitter=_whitespace_splitter)
    model = WindowModel(processor=processor)
    detector = _detector(model)
    text = json.dumps({f"k{i}": i for i in range(300)}, separators=(",", ":"))
    detector.detect(text)
    assert model.calls == [text]
    assert detector.stats.scanned_whole == 1


@pytest.mark.parametrize(
    ("max_len", "labels", "words"),
    [
        (None, 1, 200),  # GLiNER's default max_len 384
        (384, 30, 200),  # 384 - 61 - 16 = 307, capped at 200
        (100, 2, 79),  # 100 - 5 - 16
        (10, 5, 1),  # never below one word
    ],
)
def test_window_words(max_len: int | None, labels: int, words: int) -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace(max_len=max_len))
    assert _window_words(model, [f"label {i}" for i in range(labels)]) == words


class _Encoding(dict[str, Any]):
    def __init__(self, word_ids: list[int]) -> None:
        super().__init__(input_ids=list(range(len(word_ids))))
        self._word_ids = word_ids

    def word_ids(self) -> list[int]:
        return self._word_ids


class SubwordTokenizer:
    """A fast tokenizer: a word costs one subword token per 4 characters
    (at least one); GLiNER's <<ENT>> / <<SEP>> markers one each."""

    def __init__(self, model_max_length: int = 512, is_fast: bool = True) -> None:
        self.model_max_length = model_max_length
        self.is_fast = is_fast
        self.calls = 0

    def __call__(
        self, words: list[str], *, is_split_into_words: bool, add_special_tokens: bool
    ) -> _Encoding:
        assert is_split_into_words and not add_special_tokens
        self.calls += 1
        ids: list[int] = []
        for index, word in enumerate(words):
            marker = word in ("<<ENT>>", "<<SEP>>")
            ids += [index] * (1 if marker else max(1, -(-len(word) // 4)))
        return _Encoding(ids)

    def num_special_tokens_to_add(self) -> int:
        return 2


def _processor(tokenizer: Any) -> Any:
    return types.SimpleNamespace(
        transformer_tokenizer=tokenizer, ent_token="<<ENT>>", sep_token="<<SEP>>"
    )


def _config(max_len: int = 384, positions: int | None = 512) -> Any:
    encoder = types.SimpleNamespace(max_position_embeddings=positions)
    return types.SimpleNamespace(max_len=max_len, encoder_config=encoder)


@pytest.mark.parametrize(
    ("tokenizer_max", "positions", "budget"),
    [
        (256, 512, 256 - 4 - 2),  # the tokenizer's limit
        (int(1e30), 512, 512 - 4 - 2),  # sentinel: the encoder's positions
        (int(1e30), None, 512 - 4 - 2),  # neither: 512
        (8, 512, 2),  # the prompt + specials leave 2
        (5, 512, 1),  # never below one token
    ],
)
def test_subword_budget(tokenizer_max: int, positions: int | None, budget: int) -> None:
    # Prompt block for "person": <<ENT>> person <<SEP>> = 1 + 2 + 1 tokens.
    tokenizer = SubwordTokenizer(tokenizer_max)
    model = WindowModel(processor=_processor(tokenizer), config=_config(positions=positions))
    assert _subword_budget(model, ["person"]) == (tokenizer, budget)


def test_no_subword_budget_without_a_fast_tokenizer() -> None:
    slow = WindowModel(processor=_processor(SubwordTokenizer(is_fast=False)), config=_config())
    assert _subword_budget(slow, ["person"]) == (None, None)
    assert _subword_budget(WindowModel(), ["person"]) == (None, None)
    assert _detector(slow).subword_budget is None


def test_a_bare_prompt_block_costs_nothing() -> None:
    # A processor without the uni-encoder markers puts no prompt in the text.
    tokenizer = SubwordTokenizer(100)
    processor = types.SimpleNamespace(transformer_tokenizer=tokenizer)
    assert _subword_budget(WindowModel(processor=processor), ["person"]) == (tokenizer, 98)


def test_windows_also_respect_the_subword_budget() -> None:
    # 100 words of 40 characters (10 subwords each) fit max_len, not the
    # 506-token budget: windows of at most 50 words each.
    tokenizer = SubwordTokenizer(512)
    model = WindowModel(processor=_processor(tokenizer), config=_config())
    detector = _detector(model)
    text = " ".join("x" * 39 + str(i % 10) for i in range(100))
    detector.detect(text)
    assert detector.stats.scanned_windowed == 1
    assert detector.stats.windows == len(model.calls) == 3
    assert all(len(call.split()) <= 50 for call in model.calls)
    assert detector.stats.windows_truncated == 0
    assert tokenizer.calls == 2  # the prompt at build, then one call per text


def test_a_word_longer_than_the_model_reads_is_counted_truncated() -> None:
    tokenizer = SubwordTokenizer(512)
    model = WindowModel(processor=_processor(tokenizer), config=_config())
    detector = _detector(model)
    huge = "a" * 4000  # one GLiNER word of 1,000 subword tokens
    detector.detect("Jane Doe " + huge + " Jane Doe")
    assert detector.stats.windows_truncated == 1
    assert huge in model.calls  # read alone, in a window of its own
    detector.detect(huge)  # a short text whose one word is too long
    assert detector.stats.scanned_whole == 1
    assert detector.stats.windows_truncated == 2


def test_word_windows() -> None:
    assert list(word_windows([1] * 10, 10, None)) == [(0, 10, False)]
    assert list(word_windows([1] * 10, 5, None)) == [(0, 5, False), (4, 9, False), (8, 10, False)]
    assert list(word_windows([], 5, None)) == [(0, 0, False)]
    assert list(word_windows([1, 1, 9, 1, 1], 10, 3)) == [
        (0, 2, False),
        (2, 3, True),
        (3, 5, False),
    ]


def test_gliner_words_match_gliners_whitespace_splitter() -> None:
    text = '{"name": "Jane-Doe_x", "n": 3}'
    assert [text[s:e] for s, e in gliner_words(text)] == [
        "{",
        '"',
        "name",
        '"',
        ":",
        '"',
        "Jane-Doe_x",
        '"',
        ",",
        '"',
        "n",
        '"',
        ":",
        "3",
        "}",
    ]


# --- a real model (deselected by default) -----------------------------------------

# urchade/gliner_small-v2.1 (Apache-2.0), the default gliner model, at the
# revision checked on 2026-10-05. Its tokenizer and encoder config come from
# its backbone, microsoft/deberta-v3-small, which must be cached too.
GLINER_SMALL = "urchade/gliner_small-v2.1"
GLINER_SMALL_REVISION = "4e091416cf7c3481db542c2a3d26156916f3a47f"


@pytest.mark.real_model
def test_real_model_finds_a_name_past_max_len(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("gliner")
    offline_hub(monkeypatch)
    path = cached_snapshot(
        GLINER_SMALL, GLINER_SMALL_REVISION, ["gliner_config.json", "pytorch_model.bin"]
    )
    try:
        detector = build_gliner_detector(NerConfig(enabled=True, backend="gliner", model=path))
    except ConfigError:
        pytest.skip("the gliner backbone (microsoft/deberta-v3-small) is not cached")
    assert detector.window_words == 200
    assert detector.subword_budget is not None
    # Past GLiNER's own 384 words, which it would truncate with a warning.
    text = _filler(600) + " My colleague Jane Doe joined the call."
    found = detector.detect(text)
    assert [(d.detector_type, d.value) for d in found] == [("PERSON", "Jane Doe")]
    assert all(text[d.start : d.end] == d.value for d in found)
    assert detector.stats.scanned_windowed == 1
    assert detector.stats.windows_truncated == 0
    # score_threshold reaches the model (the name scores about 0.94).
    strict = build_gliner_detector(
        NerConfig(enabled=True, backend="gliner", model=path, score_threshold=0.99)
    )
    assert strict.detect("My colleague Jane Doe joined the call.") == []
