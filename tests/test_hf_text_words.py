"""BIO taggers whose tokenizer does not mark word pieces, decoded word by word.

A byte-level BPE (RoBERTa, ModernBERT) or SentencePiece (DeBERTa-v3, XLM-R)
tokenizer gives transformers no word boundaries: its token-level aggregation
(``"simple"``) made each sub-word piece its own value — kalyan-ks/ettin-68m-
nemotron-pii labels EVERY piece ``B-``, so "Zbigniew Brzezinski" became six
PERSON values — and its whitespace fallback glues ``{"name":"Angela`` into one
"word" labelled by its brace. The ``hf`` backend now decodes such a model
itself (hf_ner.TaggerPipe) over the text's own words (hf_ner.text_words: cut at
blanks, quotes, brackets and value delimiters, each CJK ideograph alone,
trimmed of punctuation), each labelled by its first piece — or by its first
piece the model tags, when the first is background. A span never takes in a
quote, a bracket, a colon or a newline; it continues across a comma and a
space or a slash only where the model continues the entity. WordPiece models
keep the transformers pipeline, unchanged.

The fakes cut text into pieces the way the two tokenizer families do (a
piece's offsets take in the blank before its word); nothing here needs torch,
transformers or a network.
"""

from __future__ import annotations

import itertools
import math
import re
import types
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_redact.bench import corpus as recall_corpus
from llm_redact.detection import model_catalog
from llm_redact.detection.engine import NerConfig
from llm_redact.detection.hf_ner import (
    HfDetector,
    TaggerPipe,
    build_hf_detector,
    text_words,
)
from llm_redact.detection.labels import LabelPolicy
from llm_redact.detection.model_catalog import CatalogEntry
from llm_redact.detection.tagging import TagSet, decoder_for
from ner_fakes import FakeHfPipe, FakeTokenizer, install_torch, install_transformers

LABELS = [
    "O",
    "B-first_name",
    "I-first_name",
    "B-last_name",
    "I-last_name",
    "B-account_number",
    "I-account_number",
    "B-user_name",
    "I-user_name",
]
ENTITIES = ("PERSON", "ACCOUNT_NUMBER", "USERNAME")

# --- the words -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("Angela Merkel", ["Angela", "Merkel"]),
        ('{"name":"Angela Merkel"}', ["name", "Angela", "Merkel"]),
        ("{'name': 'Angela'}", ["name", "Angela"]),
        ("O'Brien, Mary-Jane", ["O", "Brien", "Mary-Jane"]),
        ("acct=4417123456789;", ["acct", "4417123456789"]),
        ("  tab\tnew\nline  ", ["tab", "new", "line"]),
        ("王小明的电话？", ["王", "小", "明", "的", "电", "话"]),
        ("李华abc", ["李", "华", "abc"]),
        ("«Jane»—Doe", ["Jane", "Doe"]),
        ("「田中」さん", ["田", "中", "さん"]),
        ("a_b $5 x+y", ["a_b", "5", "x+y"]),
        (
            "1985-03-12 j.doe dev_jo42 (Paris.) @jdoe",
            ["1985-03-12", "j.doe", "dev_jo42", "Paris", "jdoe"],
        ),
        (
            "/home/jdoe/x C:\\Users\\jdoe 03/12/1985",
            ["home", "jdoe", "x", "C", "Users", "jdoe", "03", "12", "1985"],
        ),
        ("café São Paulo", ["café", "São", "Paulo"]),
        ("...", []),
        ("", []),
    ],
)
def test_text_words(text: str, words: list[str]) -> None:
    assert [text[start:end] for start, end in text_words(text)] == words


_IDEOGRAPHS = [
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
]  # fmt: skip


def _reference_words(text: str) -> list[tuple[int, int]]:
    """The word cut written character by character: a blank, a quote or a
    bracket of any script (Unicode Ps, Pe, Pi, Pf) or one of the ASCII
    value delimiters ends a word; a CJK ideograph is a word of its own;
    each word loses the punctuation (Unicode P*, ASCII symbols) at its
    ends, and a word of punctuation only is none."""
    delimiters = set("\"'`()[]{}<>,;:=|/\\")
    words, start = [], None
    for index, char in enumerate(text + " "):
        alone = index < len(text) and any(low <= ord(char) <= high for low, high in _IDEOGRAPHS)
        ends = char.isspace() or char in delimiters
        ends = ends or unicodedata.category(char) in ("Ps", "Pe", "Pi", "Pf")
        if alone or ends:
            if start is not None:
                words.append((start, index))
                start = None
            if alone:
                words.append((index, index + 1))
        elif start is None:
            start = index

    def punctuation(char: str) -> bool:
        ascii_symbol = char.isascii() and not char.isalnum() and not char.isspace()
        return ascii_symbol or unicodedata.category(char).startswith("P")

    trimmed = []
    for start, end in words:
        while start < end and punctuation(text[start]):
            start += 1
        while end > start and punctuation(text[end - 1]):
            end -= 1
        if start < end:
            trimmed.append((start, end))
    return trimmed


@settings(max_examples=500, deadline=None)
@given(
    st.text(max_size=80)
    | st.text(
        alphabet="ab1 _-.,/:'\"\u00a0\u3000\u4e00\u9fff\u3400\uff01\u00e9\u0301\u20ac\u00ab\u300c",
        max_size=40,
    )
)
def test_text_words_match_the_character_by_character_definition(text: str) -> None:
    assert text_words(text) == _reference_words(text)


# --- fake tokenizers and a fake model ---------------------------------------------


class PieceEncoding(dict[str, list[Any]]):
    """What a fast tokenizer returns: input ids, offsets, special-tokens mask."""


class PieceTokenizer:
    """A fast tokenizer that cuts text the way byte-level BPE (``"bpe"``: a
    pre-token is a run of letters, of digits or of other symbols, with the
    blank before it) or SentencePiece (``"spm"``: a pre-token is a
    blank-separated word, punctuation and all) does, then cuts each pre-token
    into pieces of at most ``size`` characters; the first piece's offsets take
    in the blank before its word. Windows of ``model_max_length`` tokens, the
    two specials included, share ``stride`` tokens. A token's id is its index
    in the text's tokens (``spans``)."""

    is_fast = True
    CLS, SEP = -1, -2
    _PRE = {
        "bpe": re.compile(r"\s?[^\W\d_]+|\s?\d+|\s?[^\w\s]+|\s+(?!\S)|\s+"),
        "spm": re.compile(r"\s?\S+|\s+"),
    }

    def __init__(self, style: str, model_max_length: int = 512, size: int = 3) -> None:
        self.style = style
        self.model_max_length = model_max_length
        self.size = size
        self.spans: list[tuple[int, int]] = []
        self.text = ""

    @property
    def _tokenizer(self) -> Any:
        # Neither family marks a word's later pieces (no "##").
        prefix = "" if self.style == "bpe" else None
        return types.SimpleNamespace(model=types.SimpleNamespace(continuing_subword_prefix=prefix))

    def pieces(self, text: str) -> list[tuple[int, int]]:
        spans = []
        for match in self._PRE[self.style].finditer(text):
            start, end = match.span()
            if not text[start:end].strip():
                continue  # a run of blanks alone: no token
            body = start + (1 if text[start].isspace() else 0)
            at = start
            while at < end:
                cut = min(end, (body if at == start else at) + self.size)
                spans.append((at, cut))
                at = cut
        return spans

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        return_overflowing_tokens: bool = False,
        stride: int = 0,
        **_kwargs: Any,
    ) -> PieceEncoding:
        assert truncation and return_overflowing_tokens
        self.text = text
        self.spans = self.pieces(text)
        out = PieceEncoding(input_ids=[], offset_mapping=[], special_tokens_mask=[])
        content = self.model_max_length - 2
        start = 0
        while True:
            chunk = list(range(start, min(start + content, len(self.spans))))
            out["input_ids"].append([self.CLS, *chunk, self.SEP])
            out["offset_mapping"].append([(0, 0), *(self.spans[i] for i in chunk), (0, 0)])
            out["special_tokens_mask"].append([1, *([0] * len(chunk)), 1])
            if start + content >= len(self.spans):
                return out
            start += content - stride


# A tagger: (text, piece start, piece end, whether the piece opens a
# pre-token) -> the label it scores highest.
Labeler = Callable[[str, int, int, bool], str]


class PieceModel:
    """A BIO token-classification model over PieceTokenizer ids: each piece
    scores the label ``labeler`` names (6.0 against 0.0 for the rest)."""

    def __init__(
        self, tokenizer: PieceTokenizer, labeler: Labeler, labels: list[str] = LABELS
    ) -> None:
        self.tokenizer = tokenizer
        self.labeler = labeler
        self.labels = labels
        self.config = types.SimpleNamespace(id2label=dict(enumerate(labels)))
        self.device = "cpu"

    def rows(self, ids: list[int]) -> list[list[float]]:
        """Log-probabilities per token of the text last encoded."""
        text, out = self.tokenizer.text, []
        for token in ids:
            label = "O"
            if token >= 0:
                start, end = self.tokenizer.spans[token]
                opens = token == 0 or self.tokenizer.spans[token - 1][1] != start
                opens = opens or text[start].isspace()
                label = self.labeler(text, start, end, opens)
            raw = [6.0 if name == label else 0.0 for name in self.labels]
            total = math.log(sum(math.exp(x) for x in raw))
            out.append([x - total for x in raw])
        return out

    def __call__(self, *, input_ids: Any) -> Any:
        # Called by hf_ner.torch_scorer (with ner_fakes.install_torch).
        logits = [self.rows(row) for row in input_ids.data]
        return types.SimpleNamespace(logits=type(input_ids)(logits))


def _detector(
    style: str,
    labeler: Labeler,
    *,
    window: int = 512,
    size: int = 3,
    threshold: float = 0.5,
    every_piece: bool = False,
) -> HfDetector:
    tokenizer = PieceTokenizer(style, window, size)
    model = PieceModel(tokenizer, labeler)
    tags = TagSet.from_labels(model.config.id2label)
    pipe = TaggerPipe(
        model,
        tokenizer,
        window // 4,
        tags,
        decoder_for(tags, None),
        model.rows,
        by_text_word=True,
        every_piece=every_piece,
    )
    return HfDetector(
        pipe,
        frozenset(ENTITIES),
        1_000_000,
        threshold,
        policy=LabelPolicy(ENTITIES, backend="hf"),
    )


def _every_piece(surfaces: dict[str, str]) -> Labeler:
    """ettin-like: every piece that takes in a character of an occurrence of
    a surface scores B- of its label (a SentencePiece piece such as '"Zb'
    included)."""

    def label(text: str, start: int, end: int, _opens: bool) -> str:
        for surface, name in surfaces.items():
            for match in re.finditer(re.escape(surface), text):
                if start < match.end() and match.start() < end:
                    return f"B-{name}"
        return "O"

    return label


def _found(detector: HfDetector, text: str) -> list[tuple[str, str]]:
    found = detector.detect(text)
    assert all(d.value == text[d.start : d.end] for d in found)
    return [(d.detector_type, d.value) for d in found]


# --- every piece B- (kalyan-ks/ettin-68m-nemotron-pii) -------------------------


@pytest.mark.parametrize("style", ["bpe", "spm"])
def test_a_model_labelling_every_piece_b_gives_whole_values(style: str) -> None:
    text = "Please email Zbigniew Brzezinski about account 4417123456789 today."
    detector = _detector(
        style,
        _every_piece(
            {"Zbigniew": "first_name", "Brzezinski": "last_name", "4417123456789": "account_number"}
        ),
    )
    # Token-level aggregation made "Zbi", "gni", "ew", … six PERSON values
    # and cut the account number into digit pieces.
    assert _found(detector, text) == [
        ("PERSON", "Zbigniew Brzezinski"),
        ("ACCOUNT_NUMBER", "4417123456789"),
    ]


@pytest.mark.parametrize("style", ["bpe", "spm"])
def test_a_word_is_labelled_by_its_first_piece(style: str) -> None:
    # A model trained the Hugging Face way labels a word on its first piece;
    # the later pieces are never trained and may say anything: another
    # type here, which neither cuts the word nor relabels it.
    def label(text: str, start: int, end: int, opens: bool) -> str:
        if "Brzezinski" not in text[max(0, start - 9) : end + 9]:
            return "O"
        return "B-last_name" if opens else "B-user_name"

    assert _found(_detector(style, label), "Ask Brzezinski now") == [("PERSON", "Brzezinski")]


@pytest.mark.parametrize("style", ["bpe", "spm"])
def test_a_word_whose_first_piece_is_background_takes_a_tagged_piece_of_an_every_piece_model(
    style: str,
) -> None:
    # A model trained on every piece may leave a word's first piece O and
    # tag a later one: reading the word as background would send it
    # upstream. A model trained on first pieces only says O there, and its
    # later pieces were never trained: the word stays background (reading
    # them turned hyphenated reference numbers into account numbers).
    def label(text: str, start: int, end: int, opens: bool) -> str:
        if "Brzezinski" not in text[max(0, start - 9) : end + 9] or opens:
            return "O"
        return "B-last_name"

    found = _found(_detector(style, label, every_piece=True), "Ask Brzezinski now")
    assert found == [("PERSON", "Brzezinski")]
    assert _found(_detector(style, label), "Ask Brzezinski now") == []


def test_the_most_confident_tagged_piece_decides_a_word_of_an_every_piece_model() -> None:
    # " Brz" scores B-user_name a little, "ezi" B-last_name surely, "nsk" O
    # surer still: the word is a PERSON with the sure tagged piece's
    # confidence (over the threshold the first piece alone would miss).
    tokenizer = PieceTokenizer("spm", size=3)
    sure = {" Brz": ("B-user_name", 0.6), "ezi": ("B-last_name", 5.0), "nsk": ("O", 9.0)}

    def scorer(ids: list[int]) -> list[list[float]]:
        out = []
        for token in ids:
            piece = "" if token < 0 else tokenizer.text[slice(*tokenizer.spans[token])]
            name, logit = sure.get(piece, ("O", 3.0))
            raw = [logit if label == name else 0.0 for label in LABELS]
            total = math.log(sum(math.exp(x) for x in raw))
            out.append([x - total for x in raw])
        return out

    model = PieceModel(tokenizer, lambda *args: "O")
    tags = TagSet.from_labels(model.config.id2label)
    for every_piece, expected in [(True, [("PERSON", "Brzezinski")]), (False, [])]:
        pipe = TaggerPipe(
            model,
            tokenizer,
            128,
            tags,
            decoder_for(tags, None),
            scorer,
            by_text_word=True,
            every_piece=every_piece,
        )
        detector = HfDetector(
            pipe, frozenset(ENTITIES), 1_000_000, 0.5, policy=LabelPolicy(ENTITIES, backend="hf")
        )
        # A first-piece model: the unsure first piece decides, under the
        # threshold.
        assert _found(detector, "Ask Brzezinski now") == expected
    (entity,) = pipe("Ask Brzezinski now")
    assert entity["entity_group"] == "user_name"
    every = TaggerPipe(
        model,
        tokenizer,
        128,
        tags,
        decoder_for(tags, None),
        scorer,
        by_text_word=True,
        every_piece=True,
    )
    (entity,) = every("Ask Brzezinski now")
    assert entity["score"] == pytest.approx(math.exp(5.0) / (math.exp(5.0) + 8))


@pytest.mark.parametrize("style", ["bpe", "spm"])
def test_punctuation_inside_a_word_is_part_of_it(style: str) -> None:
    text = "login dev_jo42 or j.doe, born 1985-03-12"
    detector = _detector(style, _every_piece({"dev_jo42": "user_name", "j.doe": "user_name"}))
    assert _found(detector, text) == [("USERNAME", "dev_jo42"), ("USERNAME", "j.doe")]


# --- punctuation, quotes, JSON -----------------------------------------------------


@pytest.mark.parametrize("style", ["bpe", "spm"])
def test_a_span_never_takes_in_json_punctuation_or_quotes(style: str) -> None:
    text = '{"name":"Angela Merkel","acct":"88123456"}'

    # A model that tags every piece touching the values, the quotes and
    # colons beside them included (a SentencePiece piece such as '"Ang'
    # carries the quote in it), and continues one span into the next value.
    def label(text: str, start: int, end: int, _opens: bool) -> str:
        if end <= text.index("Angela") - 2 or start >= text.index("88123456") - 1:
            return "I-first_name" if start >= text.index("88123456") - 1 else "O"
        return "B-first_name" if start < text.index("Angela") else "I-first_name"

    assert _found(_detector(style, label), text) == [
        ("PERSON", "Angela Merkel"),
        ("PERSON", "acct"),
        ("PERSON", "88123456"),
    ]


@pytest.mark.parametrize("style", ["bpe", "spm"])
@pytest.mark.parametrize(
    "separator", [",", " ,", ",  ", "\n", ". ", " - ", '" "', "   ", ": ", "; ", " = ", " | ", "\\"]
)
def test_a_span_is_cut_where_anything_else_than_its_gaps_separates_words(
    style: str, separator: str
) -> None:
    text = f"Doe{separator}Jane said"

    def label(text: str, start: int, end: int, _opens: bool) -> str:
        piece = text[start:end].strip().lstrip('",;:=|-.\\ ')
        return {"Do": "B-last_name", "Ja": "I-last_name"}.get(piece[:2], "O")

    # The model continues the span across the separator; the decoder cuts
    # it there, so the separator is never part of a value.
    assert _found(_detector(style, label), text) == [("PERSON", "Doe"), ("PERSON", "Jane")]


@pytest.mark.parametrize("style", ["bpe", "spm"])
@pytest.mark.parametrize("gap", [" ", "  ", "\t", "\u00a0", ", ", "/"])
def test_a_span_the_model_continues_keeps_these_gaps(style: str, gap: str) -> None:
    def label(text: str, start: int, end: int, _opens: bool) -> str:
        piece = text[start:end].strip().lstrip("/,")
        return {"Ma": "B-first_name", "19": "I-first_name"}.get(piece[:2], "O")

    text = f"born March{gap}1985 here"
    found = _found(_detector(style, label), text)
    assert found == [("PERSON", f"March{gap}1985")]


@pytest.mark.parametrize("style", ["bpe", "spm"])
def test_spans_of_one_label_join_across_a_slash(style: str) -> None:
    # A model that labels every word B- ("03", "12", "1985") gives one value
    # for a slashed date, not three placeholders with the slashes between.
    text = "born 03/12/1985 in 12 Oak Street"
    detector = _detector(style, _every_piece({"03/12/1985": "account_number"}))
    assert _found(detector, text) == [("ACCOUNT_NUMBER", "03/12/1985")]
    # Across a comma and a space, B- after B- stays two values (the model
    # says the second one starts anew).
    text = "accounts 1234, 5678"
    detector = _detector(style, _every_piece({"1234": "account_number", "5678": "account_number"}))
    assert _found(detector, text) == [("ACCOUNT_NUMBER", "1234"), ("ACCOUNT_NUMBER", "5678")]


def test_a_cut_span_keeps_the_confidence_of_the_span_the_model_tagged() -> None:
    # The model tags "Doe" "Jane" as one span; cut at the newline, each part
    # scores the span's mean probability, so the part the model is less
    # sure of alone is not dropped by the threshold.
    tokenizer = PieceTokenizer("spm", size=8)
    rows = {"Doe": {"B-last_name": 4.0}, "Jane": {"I-last_name": 1.5}}

    def scorer(ids: list[int]) -> list[list[float]]:
        out = []
        for token in ids:
            word = "" if token < 0 else tokenizer.text[slice(*tokenizer.spans[token])].strip()
            raw = [rows.get(word, {}).get(name, 1.0 if name == "O" else 0.0) for name in LABELS]
            total = math.log(sum(math.exp(x) for x in raw))
            out.append([x - total for x in raw])
        return out

    model = PieceModel(tokenizer, lambda *args: "O")
    tags = TagSet.from_labels(model.config.id2label)
    pipe = TaggerPipe(
        model, tokenizer, 128, tags, decoder_for(tags, None), scorer, by_text_word=True
    )
    text = "Doe\nJane"
    found = pipe(text)
    assert [(text[e["start"] : e["end"]]) for e in found] == ["Doe", "Jane"]
    assert found[0]["score"] == found[1]["score"]
    # Seven labels score 0 (e^0 = 1) and O scores 1 beside the tagged one.
    doe = math.exp(4.0) / (math.exp(4.0) + math.e + 7)
    jane = math.exp(1.5) / (math.exp(1.5) + math.e + 7)
    assert found[0]["score"] == pytest.approx((doe + jane) / 2)
    assert jane < 0.5 < found[1]["score"]


def test_one_or_two_blanks_keep_a_span_whole() -> None:
    def label(text: str, start: int, end: int, opens: bool) -> str:
        word = text[start:end].strip()
        return {"Doe": "B-last_name", "Jan": "I-last_name"}.get(word[:3], "O") if opens else "O"

    for gap in (" ", "  ", "\t", "\u00a0"):
        text = f"Doe{gap}Jane said"
        assert _found(_detector("spm", label), text) == [("PERSON", f"Doe{gap}Jane")]


# --- scripts without spaces ----------------------------------------------------------


def test_an_unspaced_cjk_sentence_is_read_character_by_character() -> None:
    # transformers' whitespace fallback made this whole clause one "word"
    # labelled by its first character: 李华 after 请联系 went upstream.
    text = "王小明的电话是多少？请联系李华。"
    detector = _detector("bpe", _every_piece({"王小明": "first_name", "李华": "last_name"}), size=1)
    found = _found(detector, text)
    # Each character is a word; neighbours of one label join into a value.
    assert found == [("PERSON", "王小明"), ("PERSON", "李华")]


# --- windows ---------------------------------------------------------------------------


def test_a_word_a_window_edge_cuts_is_reported_whole_once() -> None:
    filler = " ".join(f"w{i}" for i in range(4))  # one piece each
    text = f"{filler} Brzezinski ok"
    # 6 tokens a window, 1 shared: [w0 w1 w2 w3 " Br" "zez"] ["zez" "ins" "ki" " ok"]:
    # the first window holds the word's first piece and reports the word
    # whole; the second opens inside the word and does not read it.
    detector = _detector("bpe", _every_piece({"Brzezinski": "last_name"}), window=8)
    detector._pipe._stride = 1  # type: ignore[attr-defined]
    assert _found(detector, text) == [("PERSON", "Brzezinski")]

    # The second window would call its later pieces something else: they
    # are not read there.
    def label(text: str, start: int, end: int, opens: bool) -> str:
        if text[start:end].strip().startswith("Br"):
            return "B-last_name"
        return "B-user_name" if text.index("Br") < start < text.index(" ok") else "O"

    detector = _detector("bpe", label, window=8)
    detector._pipe._stride = 1  # type: ignore[attr-defined]
    assert _found(detector, text) == [("PERSON", "Brzezinski")]


@pytest.mark.parametrize("style", ["bpe", "spm"])
def test_a_name_past_the_first_window_keeps_its_absolute_offsets(style: str) -> None:
    text = " ".join(f"w{i}" for i in range(300)) + " met Angela Merkel here"
    detector = _detector(
        style, _every_piece({"Angela": "first_name", "Merkel": "last_name"}), window=64
    )
    assert _found(detector, text) == [("PERSON", "Angela Merkel")]


# --- every span, any text (hypothesis) ----------------------------------------------------

_ALPHABET = "abcXYZ019 \t\n\u00a0\"'{}[]:,.-_@#=/王李é"


@settings(max_examples=300, deadline=None)
@given(
    text=st.text(alphabet=_ALPHABET, max_size=60),
    picks=st.lists(st.sampled_from(LABELS), min_size=1, max_size=40),
    style=st.sampled_from(["bpe", "spm"]),
    window=st.sampled_from([6, 9, 512]),
    every_piece=st.booleans(),
)
def test_spans_cover_whole_words_and_nothing_else(
    text: str, picks: list[str], style: str, window: int, every_piece: bool
) -> None:
    def label(_text: str, start: int, _end: int, _opens: bool) -> str:
        return picks[start % len(picks)]

    tokenizer = PieceTokenizer(style, window)
    model = PieceModel(tokenizer, label)
    tags = TagSet.from_labels(model.config.id2label)
    pipe = TaggerPipe(
        model,
        tokenizer,
        window // 4,
        tags,
        decoder_for(tags, None),
        model.rows,
        by_text_word=True,
        every_piece=every_piece,
    )
    words = text_words(text)
    starts = {start for start, _end in words}
    ends = {end for _start, end in words}
    entities = pipe(text)
    for entity in entities:
        start, end = entity["start"], entity["end"]
        assert 0 <= start < end <= len(text)
        # A span starts and ends on word boundaries …
        assert start in starts and end in ends
        # … and between its words only nothing, one or two blanks, a comma
        # and a space, or a slash: never a quote, a bracket, a colon, an
        # equals sign or a newline.
        inside = [w for w in words if start <= w[0] and w[1] <= end]
        for (_a, left_end), (right_start, _b) in itertools.pairwise(inside):
            assert text[left_end:right_start] in ("", " ", "  ", "\t", "\u00a0", ", ", "/") or (
                len(text[left_end:right_start]) <= 2
                and set(text[left_end:right_start]) <= {" ", "\t", "\u00a0"}
            )
    # Every word whose first piece scores an entity label is covered.
    covered = {i for e in entities for i in range(e["start"], e["end"])}
    spans = tokenizer.pieces(text)
    first_pieces: dict[tuple[int, int], int] = {}
    for index, (start, end) in enumerate(spans):
        for word in words:
            if start < word[1] and end > word[0] and word not in first_pieces:
                first_pieces[word] = index
    for (word_start, word_end), index in first_pieces.items():
        if label(text, *spans[index], True) != "O":
            assert set(range(word_start, word_end)) <= covered


# --- building ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["", None])
def test_a_bio_model_without_word_pieces_is_decoded_word_by_word(
    monkeypatch: pytest.MonkeyPatch, prefix: str | None
) -> None:
    install_torch(monkeypatch)
    tokenizer = PieceTokenizer("bpe" if prefix == "" else "spm")
    model = PieceModel(tokenizer, _every_piece({"Zbigniew": "first_name"}))
    pipe = FakeHfPipe([], tokenizer=tokenizer)  # type: ignore[arg-type]
    pipe.model = model
    install_transformers(monkeypatch, pipe)
    monkeypatch.setattr("llm_redact.detection.model_catalog.lookup", lambda model_id: None)
    detector = build_hf_detector(
        NerConfig(enabled=True, backend="hf", model="org/bio-bpe", entities=("PERSON",))
    )
    # Loaded once; no strided pipeline: the backend decodes the model itself.
    assert len(pipe.built_with) == 1
    assert isinstance(detector._pipe, TaggerPipe)  # type: ignore[attr-defined]
    assert _found(detector, '{"to":"Zbigniew"}') == [("PERSON", "Zbigniew")]
    # An uncatalogued model is read as labelling first pieces only.
    assert detector._pipe._every_piece is False  # type: ignore[attr-defined]


def test_the_catalog_says_which_models_label_every_piece(monkeypatch: pytest.MonkeyPatch) -> None:
    install_torch(monkeypatch)
    tokenizer = PieceTokenizer("bpe")
    pipe = FakeHfPipe([], tokenizer=tokenizer)  # type: ignore[arg-type]
    pipe.model = PieceModel(tokenizer, _every_piece({"Zbigniew": "first_name"}))
    install_transformers(monkeypatch, pipe)
    entry = CatalogEntry(
        model_id="org/every",
        backends=("hf",),
        license="MIT",
        status="caution",
        reason="test",
        tagging="bio",
        piece_labels="every",
    )
    monkeypatch.setattr("llm_redact.detection.model_catalog.lookup", lambda model_id: entry)
    detector = build_hf_detector(
        NerConfig(enabled=True, backend="hf", model="org/every", entities=("PERSON",))
    )
    assert detector._pipe._every_piece is True  # type: ignore[attr-defined]
    assert model_catalog.lookup(ETTIN) is entry  # (patched) …
    monkeypatch.undo()
    assert model_catalog.lookup(ETTIN).piece_labels == "every"  # type: ignore[union-attr]
    every = [e.model_id for e in model_catalog.CATALOG if e.piece_labels == "every"]
    assert every == [ETTIN]
    assert {e.piece_labels for e in model_catalog.CATALOG} <= set(model_catalog.PIECE_LABELS)


def test_a_bio_label_without_a_tag_is_read_as_inside_of_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pipeline reads "PER" as I-PER; so does the decoder.
    assert TagSet.from_bio_labels({"0": "O", "1": " PER ", "2": "B-LOC"}) == TagSet(
        ("O", "I", "B"), ("", "PER", "LOC")
    )
    install_torch(monkeypatch)
    tokenizer = PieceTokenizer("spm")
    model = PieceModel(tokenizer, _every_piece({"Zbigniew Brzezinski": "x"}), ["O", "PER"])
    model.labeler = lambda *args: (
        "PER" if _every_piece({"Zbigniew Brzezinski": "x"})(*args) != "O" else "O"
    )
    pipe = FakeHfPipe([], tokenizer=tokenizer)  # type: ignore[arg-type]
    pipe.model = model
    install_transformers(monkeypatch, pipe)
    monkeypatch.setattr("llm_redact.detection.model_catalog.lookup", lambda model_id: None)
    detector = build_hf_detector(
        NerConfig(enabled=True, backend="hf", model="org/untagged", entities=("PERSON",))
    )
    assert _found(detector, "ask Zbigniew Brzezinski") == [("PERSON", "Zbigniew Brzezinski")]


def test_a_wordpiece_bio_model_keeps_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = FakeHfPipe(
        [("Jane Doe", "PER", 0.9)],
        id2label={0: "O", 1: "B-PER", 2: "I-PER"},
        tokenizer=FakeTokenizer(subword_prefix="##"),
    )
    install_transformers(monkeypatch, pipe)
    detector = build_hf_detector(NerConfig(enabled=True, backend="hf"))
    assert len(pipe.built_with) == 2
    assert pipe.built_with[1]["aggregation_strategy"] == "first"
    assert detector._pipe is pipe  # type: ignore[attr-defined]


# --- WordPiece models: exactly as before (differential) ----------------------------------

_FP_CORPUS = Path(__file__).resolve().parent.parent / "bench" / "fp_corpus"


def _corpus_texts() -> list[str]:
    texts = [sample.text for sample in recall_corpus.generate()]
    for path in sorted(_FP_CORPUS.iterdir()):
        if path.is_file() and path.name != "MANIFEST.toml":
            texts.extend(chunk for chunk in path.read_text(encoding="utf-8").split("\n\n") if chunk)
    return texts


def test_a_wordpiece_model_detects_exactly_as_before_over_the_corpora(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The builder hands a WordPiece BIO model to the same strided
    # transformers pipeline as before this change; the detector over it
    # reports what the pre-change detector (HfDetector straight over the
    # pipeline, the pre-change construction) reports, on every text of the
    # recall corpus and the false-positive corpus.
    findings = [
        ("Jane Doe", "PER", 0.9),
        ("Smith", "PER", 0.6),
        ("@", "MISC", 0.9),
        ("e", "PER", 0.4),
    ]
    pipe = FakeHfPipe(findings, id2label={0: "O", 1: "B-PER", 2: "I-PER", 3: "B-MISC"})
    install_transformers(monkeypatch, pipe)
    built = build_hf_detector(NerConfig(enabled=True, backend="hf", score_threshold=0.5))
    reference = HfDetector(pipe, frozenset({"PERSON"}), NerConfig().max_chars, 0.5)
    texts = _corpus_texts()
    assert len(texts) > 100
    for text in texts:
        assert built.detect(text) == reference.detect(text)


# --- a real model (deselected by default) -----------------------------------------

# kalyan-ks/ettin-68m-nemotron-pii (MIT; ModernBERT byte-level BPE tokenizer,
# tags every piece B-) at its catalog pin, checked on 2026-10-07.
ETTIN = "kalyan-ks/ettin-68m-nemotron-pii"
ETTIN_REVISION = "500262a2aaf913825ef750ef255c3fe437cd8e64"
ETTIN_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


@pytest.mark.real_model
def test_real_model_every_piece_b_tagger_gives_whole_names(monkeypatch: pytest.MonkeyPatch) -> None:
    # Token-level aggregation made "Y", "us", "uf", "Lind", "gren" five
    # PERSON values here; the quotes around the name never join it.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from real_models import cached_snapshot, offline_hub

    offline_hub(monkeypatch)
    cached_snapshot(ETTIN, ETTIN_REVISION, ETTIN_FILES)
    detector = build_hf_detector(
        NerConfig(enabled=True, backend="hf", model=ETTIN, entities=("PERSON",))
    )
    for text, names in [
        ('billing: charge declined for "Yusuf Lindgren" account=994053910371', ["Yusuf Lindgren"]),
        ("# Reported by Mei Navarro; reproduces only for account", ["Mei Navarro"]),
    ]:
        assert _found(detector, text) == [("PERSON", name) for name in names]
