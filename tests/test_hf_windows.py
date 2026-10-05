"""The ``hf`` backend reads whole strings in overlapping token windows.

Without ``stride`` the transformers token-classification pipeline truncates at
the tokenizer's maximum length, so a name past the first window was never
read. The builder now requires a fast tokenizer (character offsets), sizes the
window from the tokenizer and model config, and builds the pipeline with
``stride`` once; the detector counts windowed strings and windows, and keeps
one copy of an entity two windows both report.

The fakes count one token per whitespace word (ner_fakes.FakeTokenizer); the
one ``real_model`` test needs transformers, torch and a cached
dslim/bert-base-NER and is deselected by default (``pytest -m real_model``).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from llm_redact.config import ConfigError
from llm_redact.detection.base import Detection
from llm_redact.detection.engine import Allowlist, NerConfig
from llm_redact.detection.hf_ner import (
    HfDetector,
    aggregation_for,
    build_hf_detector,
    model_window,
    window_counter,
)
from llm_redact.detection.ner import NER_PRIORITY
from llm_redact.detection.windows import drop_exact_duplicates
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault
from ner_fakes import FakeHfPipe, FakeTokenizer, install_transformers
from real_models import cached_snapshot, offline_hub

SENTINEL = int(1e30)  # transformers' VERY_LARGE_INTEGER: "limit unknown"


def _filler(words: int) -> str:
    return " ".join(f"w{i}" for i in range(words))


class WindowedPipe:
    """The strided pipeline WITHOUT its own overlap aggregation: each window
    (FakeTokenizer chunks of whitespace words) reports the findings inside
    it at absolute offsets, a finding cut by the window's edge as the part
    inside — so overlaps report entities twice and seams cut them."""

    def __init__(self, findings: list[tuple[str, str]], window: int, stride: int) -> None:
        self.findings = findings
        self.window = window
        self.stride = stride
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[dict[str, Any]]:
        self.calls.append(text)
        words = [m.span() for m in re.finditer(r"\S+", text)]
        content = self.window - 2
        out: list[dict[str, Any]] = []
        first = 0
        while True:
            last = min(first + content, len(words))
            lo, hi = words[first][0], words[last - 1][1]
            for surface, label in self.findings:
                for match in re.finditer(re.escape(surface), text):
                    start, end = max(match.start(), lo), min(match.end(), hi)
                    if start < end:
                        entity = {"entity_group": label, "score": 0.9}
                        out.append({**entity, "start": start, "end": end})
            if last >= len(words):
                return out
            first += content - self.stride


def _windowed_detector(pipe: WindowedPipe, max_chars: int = 1_000_000) -> HfDetector:
    tokenizer = FakeTokenizer(model_max_length=pipe.window)
    return HfDetector(
        pipe,
        frozenset({"PERSON"}),
        max_chars,
        0.5,
        windows_of=window_counter(tokenizer, pipe.stride),
    )


# --- the window ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("tokenizer_max", "positions", "catalog", "window"),
    [
        (512, 512, None, 512),  # dslim/bert-base-NER
        (SENTINEL, 1024, None, 1024),  # tokenizer does not know: the model does
        (SENTINEL, None, None, 512),  # neither says: the BERT default
        (256, 512, None, 256),  # the tokenizer's limit is lower
        (1024, 512, None, 512),  # the model reads less than its tokenizer claims
        (0, 384, None, 384),  # a non-limit
        (512, 512, 128, 128),  # a catalogued window wins
    ],
)
def test_model_window(
    tokenizer_max: int, positions: int | None, catalog: int | None, window: int
) -> None:
    pipe = FakeHfPipe([], max_position_embeddings=positions)
    tokenizer = FakeTokenizer(model_max_length=tokenizer_max)
    assert model_window(tokenizer, pipe.model, catalog) == window


def test_window_counter_matches_the_pipeline_chunking() -> None:
    # Hugging Face fast tokenizers: windows of model_max_length tokens, the
    # specials included, each sharing `stride` tokens with the previous one.
    # Verified against transformers 5.10.1 + dslim/bert-base-NER: 20, 310,
    # 519-610, 1010 and 3010 tokens made 1, 1, 2, 3 and 8 chunks.
    count = window_counter(FakeTokenizer(model_max_length=512), 128)
    assert [count(_filler(n)) for n in (10, 510, 511, 892, 893, 3010)] == [1, 1, 2, 2, 3, 8]


# --- building -------------------------------------------------------------------


def test_stride_is_set_once_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = FakeHfPipe([("Jane Doe", "PER", 0.9)])
    install_transformers(monkeypatch, pipe)
    detector = build_hf_detector(NerConfig(enabled=True, backend="hf"))
    loaded, strided = pipe.built_with
    assert loaded == {"model": "dslim/bert-base-NER", "aggregation_strategy": "simple"}
    assert strided == {
        "model": pipe.model,
        "tokenizer": pipe.tokenizer,
        "aggregation_strategy": "first",
        "stride": 128,
    }
    # Calls keep their one-argument signature: the stride is not per call.
    assert [d.value for d in detector.detect("hi Jane Doe")] == ["Jane Doe"]
    assert pipe.calls == ["hi Jane Doe"]


# --- whole words --------------------------------------------------------------


@pytest.mark.parametrize(
    ("prefix", "strategy"),
    [
        ("##", "first"),  # WordPiece (BERT, dslim/bert-base-NER): real word boundaries
        ("", "simple"),  # byte-level BPE (RoBERTa): the whitespace fallback
        (None, "simple"),  # SentencePiece Unigram (XLM-R, DeBERTa-v3): no prefix at all
    ],
)
def test_the_aggregation_keeps_whole_words_where_the_tokenizer_knows_them(
    monkeypatch: pytest.MonkeyPatch, prefix: str | None, strategy: str
) -> None:
    # A word-level strategy only where transformers trusts its word
    # boundaries; elsewhere its fallback glues '{"name":"Angela' into one
    # "word" labelled by its first piece, which would lose the name.
    tokenizer = FakeTokenizer(subword_prefix=prefix)
    assert aggregation_for(tokenizer) == strategy
    pipe = FakeHfPipe([("Jane Doe", "PER", 0.9)], tokenizer=tokenizer)
    install_transformers(monkeypatch, pipe)
    build_hf_detector(NerConfig(enabled=True, backend="hf"))
    loaded, strided = pipe.built_with
    # The first load only fetches the model and its tokenizer: a word-level
    # strategy there would refuse a slow tokenizer before the clearer
    # fast-tokenizer check below runs.
    assert loaded["aggregation_strategy"] == "simple"
    assert strided["aggregation_strategy"] == strategy


def test_a_tokenizer_without_a_backend_model_keeps_token_aggregation() -> None:
    assert aggregation_for(object()) == "simple"


def test_a_tokenizer_without_a_known_limit_gets_the_model_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = FakeHfPipe([], max_position_embeddings=384, tokenizer=FakeTokenizer(SENTINEL))
    install_transformers(monkeypatch, pipe)
    detector = build_hf_detector(NerConfig(enabled=True, backend="hf"))
    assert pipe.tokenizer.model_max_length == 384
    assert pipe.built_with[1]["stride"] == 96
    list(detector.detect(_filler(1000)))
    assert detector.stats.scanned_windowed == 1
    assert detector.stats.windows == 4  # 382 tokens a window, 96 shared


def test_a_slow_tokenizer_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = FakeHfPipe([], tokenizer=FakeTokenizer(is_fast=False))
    install_transformers(monkeypatch, pipe)
    with pytest.raises(ConfigError) as caught:
        build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/slow-ner"))
    assert str(caught.value) == (
        "[detection.ner] hf model 'org/slow-ner' has no fast tokenizer;"
        " character offsets are required"
    )
    assert len(pipe.built_with) == 1  # refused before the windowed build


def test_windowing_settings_transformers_refuses_are_a_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    pipe = FakeHfPipe([])

    def pipeline(*args: Any, **kwargs: Any) -> FakeHfPipe:
        if "stride" in kwargs:
            raise ValueError("`stride` must be less than `tokenizer.model_max_length`")
        return pipe

    module = types.ModuleType("transformers")
    module.pipeline = pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)
    with pytest.raises(ConfigError) as caught:
        build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/ner"))
    assert str(caught.value) == (
        "failed to load Hugging Face token-classification model 'org/ner': ValueError"
    )


# --- reading in windows -----------------------------------------------------------


def test_an_entity_past_the_first_window_keeps_its_absolute_offsets() -> None:
    pipe = WindowedPipe([("Jane Doe", "PER")], window=512, stride=128)
    text = _filler(1500) + " met Jane Doe today " + _filler(400)
    detector = _windowed_detector(pipe)
    found = detector.detect(text)
    start = text.index("Jane Doe")
    assert found == [
        Detection(
            start=start,
            end=start + 8,
            detector_type="PERSON",
            value="Jane Doe",
            priority=NER_PRIORITY,
        )
    ]
    assert detector.stats.scanned_windowed == 1
    assert detector.stats.windows == 5  # 1,904 words: 510 a window, 128 shared
    assert detector.stats.scanned_whole == 0


def test_an_entity_in_an_overlap_is_reported_once() -> None:
    pipe = WindowedPipe([("Jane Doe", "PER")], window=512, stride=128)
    # Words 382-509 are in both the first and the second window.
    text = _filler(450) + " Jane Doe " + _filler(400)
    raw = pipe(text)
    assert len(raw) == 2  # the pipeline stand-in reports it twice
    assert [d.value for d in _windowed_detector(pipe).detect(text)] == ["Jane Doe"]


def test_a_name_cut_by_a_seam_is_redacted_whole() -> None:
    # The first window ends between "Jane" and "Doe": it reports "Jane", the
    # second window the whole name. Both reach the redactor, whose overlap
    # resolution keeps the longest span: one token for the whole name.
    pipe = WindowedPipe([("Jane Doe", "PER")], window=512, stride=128)
    text = _filler(509) + " Jane Doe " + _filler(10)
    assert {d.value for d in _windowed_detector(pipe).detect(text)} == {"Jane", "Jane Doe"}
    redactor = Redactor([_windowed_detector(pipe)], InMemoryVault(), Allowlist())
    redacted = redactor.redact_text(text)
    assert "Jane" not in redacted
    assert redacted.count("«PERSON_001»") == 1
    assert "«PERSON_002»" not in redacted


def test_a_short_text_is_one_whole_call() -> None:
    pipe = WindowedPipe([("Jane Doe", "PER")], window=512, stride=128)
    detector = _windowed_detector(pipe)
    assert [d.value for d in detector.detect("hi Jane Doe")] == ["Jane Doe"]
    assert detector.stats.scanned_whole == 1
    assert detector.stats.scanned_windowed == detector.stats.windows == 0
    assert pipe.calls == ["hi Jane Doe"]


def test_max_chars_still_skips_before_any_window() -> None:
    pipe = WindowedPipe([("Jane Doe", "PER")], window=512, stride=128)
    detector = _windowed_detector(pipe, max_chars=100)
    assert detector.detect(_filler(50) + " Jane Doe") == []
    assert pipe.calls == []
    assert detector.stats.skipped_max_chars == 1
    assert detector.stats.windows == 0


def test_drop_exact_duplicates_keeps_order_and_distinct_types() -> None:
    a = Detection(0, 4, "PERSON", "Jane")
    b = Detection(0, 4, "ADDRESS", "Jane")
    c = Detection(5, 8, "PERSON", "Doe")
    assert drop_exact_duplicates([a, b, a, c, b]) == [a, b, c]


# --- a real model (deselected by default) -----------------------------------------

# dslim/bert-base-NER (MIT; trained on CoNLL-2003 Reuters news), the default hf
# model, at the revision checked on 2026-10-05.
DSLIM = "dslim/bert-base-NER"
DSLIM_REVISION = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
DSLIM_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
]


@pytest.mark.real_model
def test_real_model_reports_a_name_as_whole_words(monkeypatch: pytest.MonkeyPatch) -> None:
    # Token-level aggregation reported "Angela Merk": the word piece "##el"
    # was labelled O, so "el" went upstream as sent.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    offline_hub(monkeypatch)
    path = cached_snapshot(DSLIM, DSLIM_REVISION, DSLIM_FILES)
    detector = build_hf_detector(NerConfig(enabled=True, backend="hf", model=path))
    for text, names in [
        ("Yesterday Angela Merkel met the press.", ["Angela Merkel"]),
        ('{"name":"Angela Merkel","role":"chancellor"}', ["Angela Merkel"]),
        (
            "Contact Zbigniew Brzezinski or Kateryna Shevchenko today.",
            ["Zbigniew Brzezinski", "Kateryna Shevchenko"],
        ),
        ("The file was written by Xu Wenjing.", ["Xu Wenjing"]),
    ]:
        found = detector.detect(text)
        assert [(d.detector_type, d.value) for d in found] == [("PERSON", n) for n in names]
        for d in found:
            # No word is cut: the characters around the span are no letters.
            assert not text[d.start - 1 : d.start].isalnum()
            assert not text[d.end : d.end + 1].isalnum()


@pytest.mark.real_model
def test_real_model_finds_a_name_after_3000_words(monkeypatch: pytest.MonkeyPatch) -> None:
    # Offline: the model loads from its cached local folder only.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    offline_hub(monkeypatch)
    path = cached_snapshot(DSLIM, DSLIM_REVISION, DSLIM_FILES)
    detector = build_hf_detector(NerConfig(enabled=True, backend="hf", model=path))
    text = "word " * 3000 + "Angela Merkel met Barack Obama in Berlin."
    found = detector.detect(text)
    assert [(d.detector_type, d.value) for d in found] == [
        ("PERSON", "Angela Merkel"),
        ("PERSON", "Barack Obama"),
    ]
    assert all(text[d.start : d.end] == d.value for d in found)
    assert detector.stats.scanned_windowed == 1
    assert detector.stats.windows == 8  # 3,010 tokens in windows of 512, 128 shared
