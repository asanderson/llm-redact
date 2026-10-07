"""BIOES and BILOU taggers in the ``hf`` backend: decoded by llm-redact.

The transformers pipeline's aggregation understands B-/I- tags only; a model
whose labels also mark a span's end (E-, S- in BIOES; L-, U- in BILOU) is run
without it and decoded by tagging.py — the constrained Viterbi decoder when
the model catalog lists the model's calibration file, else greedily. BIO
models keep the pipeline.

The Viterbi decoder is linear in tokens x labels; it is checked against a
brute-force decoder over the full transition matrix, written from OpenAI's
reference decoder for openai/privacy-filter (opf/_core/decoding.py,
checked 2026-10-06). Fakes count one token per whitespace word; no test needs
torch, transformers or a network.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from llm_redact.config import ConfigError
from llm_redact.detection import model_catalog
from llm_redact.detection.engine import NerConfig
from llm_redact.detection.hf_ner import HfDetector, TaggerPipe, build_hf_detector, torch_scorer
from llm_redact.detection.model_catalog import SIDECAR_NAME, TAGGING_SCHEMES, CatalogEntry
from llm_redact.detection.tagging import (
    BIAS_KEYS,
    BILOU,
    BIO,
    BIOES,
    ZERO_BIASES,
    TaggingError,
    TagSet,
    argmax_path,
    decoder_for,
    spans_of,
    tagging_scheme,
    viterbi_biases,
    viterbi_path,
)
from ner_fakes import FakeHfPipe, FakeHub, install_torch, install_transformers
from real_models import cached_snapshot, offline_hub

# openai/privacy-filter's labels, in its config.json order (checked 2026-10-06).
PRIVACY_LABELS = ["O"] + [
    f"{tag}-{entity}"
    for entity in (
        "account_number",
        "private_address",
        "private_date",
        "private_email",
        "private_person",
        "private_phone",
        "private_url",
        "secret",
    )
    for tag in "BIES"
]
BIOES_PERSON = ["O", "B-person", "I-person", "E-person", "S-person"]
BILOU_PERSON = ["O", "B-PER", "I-PER", "L-PER", "U-PER"]


def _tags(labels: list[str]) -> TagSet:
    return TagSet.from_labels(dict(enumerate(labels)))


def _rows(labels: list[str], picks: list[dict[str, float]]) -> list[list[float]]:
    """One row of scores per token: the given labels' scores, 0 elsewhere."""
    return [[pick.get(label, 0.0) for label in labels] for pick in picks]


# --- the scheme ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "scheme"),
    [
        (["O", "B-PER", "I-PER", "B-LOC", "I-LOC"], BIO),  # dslim/bert-base-NER
        (["O", "PER", "LOC"], BIO),  # no tags at all
        (PRIVACY_LABELS, BIOES),
        (["O", "b-per", "i-per", "e-per", "s-per"], BIOES),  # any case
        (["O", "S-PER"], BIOES),
        (BILOU_PERSON, BILOU),
        (["O", "U-PER"], BILOU),
        (["O", "E-2", "S-3", "B-PER"], BIO),  # a tag needs a letter after it
    ],
)
def test_the_scheme_comes_from_the_label_tags(labels: list[str], scheme: str) -> None:
    assert tagging_scheme(labels) == scheme


def test_mixed_bioes_and_bilou_tags_are_refused() -> None:
    with pytest.raises(TaggingError, match="mix BIOES"):
        tagging_scheme(["O", "B-PER", "E-PER", "U-PER"])


def test_the_catalog_and_the_decoder_name_the_same_schemes() -> None:
    assert TAGGING_SCHEMES == (BIO, BIOES, BILOU)


def test_a_tag_set_reads_labels_by_logit_index() -> None:
    tags = TagSet.from_labels({"1": "B-person", "0": "O", "2": "L-person", "3": "U-person"})
    assert tags.tags == ("O", "B", "E", "S")  # BILOU's L and U read as E and S
    assert tags.entities == ("", "person", "person", "person")
    assert len(tags) == 4
    untagged = TagSet.from_labels({0: " o ", 1: "PER", 2: "S-PER"})
    assert untagged.tags == ("O", None, "S")


@pytest.mark.parametrize(
    "id2label",
    [{"zero": "O"}, {0: "O", 2: "S-PER"}, {1: "O"}],
)
def test_a_tag_set_needs_every_logit_index(id2label: dict[Any, str]) -> None:
    with pytest.raises(TaggingError):
        TagSet.from_labels(id2label)


# --- greedy reading ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "path", "spans"),
    [
        # Well-formed BIOES: B…E, and S alone.
        (BIOES_PERSON, ["B", "I", "E", "O", "S"], [(0, 2), (4, 4)]),
        (BIOES_PERSON, ["S", "S"], [(0, 0), (1, 1)]),
        (BIOES_PERSON, ["O", "O"], []),
        # BILOU reads the same: L ends, U is one token.
        (BILOU_PERSON, ["B", "I", "L", "U"], [(0, 2), (3, 3)]),
        # Malformed: I or E with no span open starts one there.
        (BIOES_PERSON, ["I", "E"], [(0, 1)]),
        (BIOES_PERSON, ["E", "O"], [(0, 0)]),
        # A span still open at O or at the end is kept as it is.
        (BIOES_PERSON, ["B", "I", "O", "B"], [(0, 1), (3, 3)]),
        # B (or S) inside an open span closes it and starts anew.
        (BIOES_PERSON, ["B", "B", "E"], [(0, 0), (1, 2)]),
        (BIOES_PERSON, ["B", "S", "I"], [(0, 0), (1, 1), (2, 2)]),
    ],
)
def test_the_greedy_reading(
    labels: list[str], path: list[str], spans: list[tuple[int, int]]
) -> None:
    index = {label[0]: i for i, label in enumerate(labels)}
    tags = _tags(labels)
    found = spans_of([index[tag] for tag in path], tags)
    assert [(first, last) for first, last, _ in found] == spans
    assert {entity for *_, entity in found} <= {"person", "PER"}


def test_a_tag_of_another_entity_starts_a_new_span() -> None:
    tags = _tags(["O", "B-person", "I-person", "E-person", "I-address", "E-address"])
    assert spans_of([1, 4, 5], tags) == [(0, 0, "person"), (1, 2, "address")]
    assert spans_of([1, 2, 3], tags) == [(0, 2, "person")]


def test_an_untagged_label_closes_a_span_like_o() -> None:
    tags = _tags(["O", "B-person", "PER"])
    assert spans_of([1, 2, 1], tags) == [(0, 0, "person"), (2, 2, "person")]


def test_argmax_takes_the_first_best_and_never_a_nan() -> None:
    assert argmax_path([[0.1, 0.9, 0.9], [math.nan, 0.2, 0.1]]) == [1, 1]


# --- Viterbi ----------------------------------------------------------------------


def _reference_path(
    rows: Sequence[Sequence[float]], tags: TagSet, biases: dict[str, float]
) -> tuple[float, list[int]] | None:
    """Brute-force constrained Viterbi over the full transition matrix: the
    best (score, path), or None when no path is valid."""
    neg = -math.inf
    labels = range(len(tags))

    def closed(j: int) -> bool:
        return tags.tags[j] in ("O", "E", "S")

    def edge(i: int, j: int) -> float:
        a, b = tags.tags[i], tags.tags[j]
        if a is None or b is None:
            return neg
        if closed(i):
            if b == "O":
                key = "background_stay" if a == "O" else "end_to_background"
            elif b in ("B", "S"):
                key = "background_to_start" if a == "O" else "end_to_start"
            else:
                return neg
            return biases[f"transition_bias_{key}"]
        if b in ("I", "E") and tags.entities[i] == tags.entities[j]:
            key = "inside_to_continue" if b == "I" else "inside_to_end"
            return biases[f"transition_bias_{key}"]
        return neg

    score = [rows[0][j] if tags.tags[j] in ("O", "B", "S") else neg for j in labels]
    paths: list[list[int]] = [[j] for j in labels]
    for row in rows[1:]:
        new_score, new_paths = [], []
        for j in labels:
            best, best_i = neg, -1
            for i in labels:
                candidate = score[i] + edge(i, j)
                if candidate > best:
                    best, best_i = candidate, i
            new_score.append(best + row[j])
            new_paths.append([*paths[best_i], j] if best_i >= 0 else [])
        score, paths = new_score, new_paths
    finals = [(score[j], j) for j in labels if closed(j) and score[j] > neg]
    if not finals:
        return None
    best, last = max(finals)
    return best, paths[last]


def _valid(path: list[int], tags: TagSet) -> bool:
    found = _reference_path(
        [[0.0 if k == label else -math.inf for k in range(len(tags))] for label in path],
        tags,
        dict(ZERO_BIASES),
    )
    return found is not None


TAG_SETS = [
    BIOES_PERSON,
    BILOU_PERSON,
    ["O", "B-a", "I-a", "E-a", "S-a", "B-b", "I-b", "E-b", "S-b"],
    ["B-a", "I-a", "E-a", "S-a", "O"],  # O last
    ["O", "B-a", "E-a", "S-a", "PER"],  # no I, an untagged label
    ["B-a", "I-a", "E-a", "S-a"],  # no O
]


@pytest.mark.parametrize("labels", TAG_SETS)
@pytest.mark.parametrize("seed", range(12))
def test_viterbi_finds_the_best_valid_path(labels: list[str], seed: int) -> None:
    generator = random.Random(seed)
    tags = _tags(labels)
    biases = {key: generator.uniform(-2, 2) for key in BIAS_KEYS} if seed % 3 else dict(ZERO_BIASES)
    rows = [[generator.uniform(-6, 0) for _ in labels] for _ in range(generator.randint(1, 9))]
    reference = _reference_path(rows, tags, biases)
    assert reference is not None
    path = viterbi_path(rows, tags, biases)
    assert path is not None
    assert _valid(path, tags)
    ours = _scored(path, rows, tags, biases)
    assert ours == pytest.approx(reference[0])


def _scored(
    path: list[int], rows: Sequence[Sequence[float]], tags: TagSet, biases: dict[str, float]
) -> float:
    """``path``'s score: the reference decoder's best over that path alone."""
    pinned = [
        [score if k == label else -math.inf for k, score in enumerate(row)]
        for label, row in zip(path, rows, strict=True)
    ]
    found = _reference_path(pinned, tags, biases)
    assert found is not None
    return found[0]


def test_viterbi_spans_are_always_well_formed() -> None:
    generator = random.Random(7)
    tags = _tags(TAG_SETS[2])
    for _ in range(200):
        rows = [[generator.uniform(-5, 0) for _ in range(len(tags))] for _ in range(6)]
        path = viterbi_path(rows, tags)
        assert path is not None
        for first, last, entity in spans_of(path, tags):
            opened, closed = tags.tags[path[first]], tags.tags[path[last]]
            if first == last:
                assert opened == "S"
            else:
                assert (opened, closed) == ("B", "E")
            assert all(tags.entities[path[t]] == entity for t in range(first, last + 1))


def test_viterbi_repairs_what_the_argmax_cuts() -> None:
    # Token 2 is a little more likely O than I: the argmax reads two spans,
    # "Jane" and "Doe"; the best VALID path joins them.
    rows = _rows(
        BIOES_PERSON,
        [{"B-person": 4.0}, {"O": 2.0, "I-person": 1.9}, {"E-person": 4.0}],
    )
    tags = _tags(BIOES_PERSON)
    assert spans_of(argmax_path(rows), tags) == [(0, 0, "person"), (2, 2, "person")]
    path = viterbi_path(rows, tags)
    assert path is not None
    assert spans_of(path, tags) == [(0, 2, "person")]


def test_the_biases_move_the_decision() -> None:
    rows = _rows(BIOES_PERSON, [{"O": 1.0, "S-person": 0.8}])
    tags = _tags(BIOES_PERSON)
    assert viterbi_path(rows, tags) == [0]
    # A cheaper span entry (here: a dearer background) tips it.
    eager = dict(ZERO_BIASES, transition_bias_background_stay=-1.0)
    rows2 = _rows(BIOES_PERSON, [{"O": 1.0}, {"O": 1.0, "S-person": 0.8}])
    assert viterbi_path(rows2, tags, eager) == [0, 4]
    assert viterbi_path(rows2, tags) == [0, 0]


def test_viterbi_without_a_valid_path_falls_back_to_the_greedy_reading() -> None:
    # No O and no S: a single token cannot be tagged at all.
    labels = ["B-a", "I-a", "E-a"]
    tags = _tags(labels)
    rows = [[0.0, -1.0, -2.0]]
    assert viterbi_path(rows, tags) is None
    decode = decoder_for(tags, dict(ZERO_BIASES))
    assert decode(rows) == [0]
    assert spans_of(decode(rows), tags) == [(0, 0, "a")]


def test_viterbi_of_nothing_is_nothing() -> None:
    assert viterbi_path([], _tags(BIOES_PERSON)) == []


def test_without_biases_the_decoder_is_greedy() -> None:
    assert decoder_for(_tags(BIOES_PERSON), None) is argmax_path


def test_viterbi_never_takes_a_nan() -> None:
    rows = _rows(BIOES_PERSON, [{"O": 1.0}])
    rows[0][4] = math.nan
    assert viterbi_path(rows, _tags(BIOES_PERSON)) == [0]


# --- calibration -------------------------------------------------------------------


def _calibration(**biases: Any) -> dict[str, Any]:
    values: dict[str, Any] = dict.fromkeys(BIAS_KEYS, 0.0)
    values.update(biases)
    return {"operating_points": {"default": {"biases": values}}}


def test_the_calibration_biases_are_read() -> None:
    biases = viterbi_biases(_calibration(transition_bias_background_stay=-0.5))
    assert biases == dict(ZERO_BIASES, transition_bias_background_stay=-0.5)
    assert viterbi_biases(_calibration(transition_bias_end_to_start=1)) == dict(
        ZERO_BIASES, transition_bias_end_to_start=1.0
    )


@pytest.mark.parametrize(
    ("calibration", "message"),
    [
        ({}, "exactly one key, operating_points"),
        ({"operating_points": [], "x": 1}, "exactly one key, operating_points"),
        ({"operating_points": {"high_recall": {}}}, "exactly one key, default"),
        ({"operating_points": {"default": {"biases": [], "x": 1}}}, "exactly one key, biases"),
        ({"operating_points": {"default": {"biases": {"x": 0}}}}, "must hold exactly"),
        (_calibration(transition_bias_inside_to_end="1"), "inside_to_end must be a number"),
        (_calibration(transition_bias_inside_to_end=True), "inside_to_end must be a number"),
        (_calibration(transition_bias_end_to_start=math.inf), "end_to_start must be finite"),
    ],
)
def test_a_malformed_calibration_is_refused(calibration: dict[str, Any], message: str) -> None:
    with pytest.raises(TaggingError, match=message):
        viterbi_biases(calibration)


# --- the tagger pipe ----------------------------------------------------------------


class TagEncoding(dict[str, list[Any]]):
    """What a fast tokenizer returns for overflowing windows: the lists by
    key, and each window's word ids (indices into the whole text's words)."""

    def __init__(self, words: list[list[int | None]]) -> None:
        super().__init__(input_ids=[], offset_mapping=[], special_tokens_mask=[])
        self._words = words

    def word_ids(self, batch_index: int = 0) -> list[int | None]:
        return self._words[batch_index]


class TagTokenizer:
    """A fast tokenizer for the tagger: one token per whitespace word (its
    id an index into ``vocab``) — or, for a word in ``pieces``, one token
    per piece, the later ones marked ``##`` as WordPiece marks them — a
    special token at each end of a window, character offsets into the whole
    text and each window's word ids; windows of ``model_max_length`` tokens
    sharing ``stride``, like the Hugging Face fast tokenizers."""

    is_fast = True
    CLS, SEP = -1, -2

    def __init__(
        self, model_max_length: int = 512, pieces: dict[str, list[str]] | None = None
    ) -> None:
        self.model_max_length = model_max_length
        self.vocab: list[str] = []
        self.pieces = pieces or {}

    @property
    def _tokenizer(self) -> Any:
        # The `tokenizers` model a fast tokenizer wraps; WordPiece names the
        # mark of a word's later pieces (hf_ner.aggregation_for reads it).
        model = types.SimpleNamespace(continuing_subword_prefix="##" if self.pieces else None)
        return types.SimpleNamespace(model=model)

    def _id(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab.append(word)
        return self.vocab.index(word)

    def _tokens(self, text: str) -> list[tuple[int, tuple[int, int], int]]:
        """Each token's id, character offsets and word index."""
        tokens = []
        for word, match in enumerate(re.finditer(r"\S+", text)):
            at = match.start()
            for piece in self.pieces.get(match[0], [match[0]]):
                size = len(piece.removeprefix("##"))
                tokens.append((self._id(piece), (at, at + size), word))
                at += size
        return tokens

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        return_overflowing_tokens: bool = False,
        stride: int = 0,
        return_offsets_mapping: bool = False,
        return_special_tokens_mask: bool = False,
    ) -> TagEncoding:
        assert truncation and return_overflowing_tokens
        tokens = self._tokens(text)
        content = self.model_max_length - 2
        words: list[list[int | None]] = []
        out = TagEncoding(words)
        start = 0
        while True:
            chunk = tokens[start : start + content]
            out["input_ids"].append([self.CLS, *(token for token, _, _ in chunk), self.SEP])
            out["offset_mapping"].append([(0, 0), *(offsets for _, offsets, _ in chunk), (0, 0)])
            out["special_tokens_mask"].append([1, *([0] * len(chunk)), 1])
            words.append([None, *(word for _, _, word in chunk), None])
            if start + content >= len(tokens):
                return out
            start += content - stride


class TagModel:
    """A token-classification model over TagTokenizer ids: each word in
    ``picks`` gets those label scores, every other token (specials included)
    scores O. Called like a transformers model: ``model(input_ids=...)``."""

    def __init__(
        self, labels: list[str], tokenizer: TagTokenizer, picks: dict[str, dict[str, float]]
    ) -> None:
        self.labels = labels
        self.tokenizer = tokenizer
        self.picks = picks
        self.config = type("Config", (), {"id2label": dict(enumerate(labels))})()
        self.device = "cpu"
        self.calls = 0

    def rows(self, input_ids: list[int]) -> list[list[float]]:
        out = []
        for token in input_ids:
            word = self.tokenizer.vocab[token] if token >= 0 else ""
            pick = self.picks.get(word, {"O": 4.0})
            out.append([pick.get(label, 0.0) for label in self.labels])
        return out

    def __call__(self, *, input_ids: Any) -> Any:
        self.calls += 1
        logits = [self.rows(row) for row in input_ids.data]
        return type("Output", (), {"logits": type(input_ids)(logits)})()


def _log_softmax(rows: list[list[float]]) -> list[list[float]]:
    out = []
    for row in rows:
        total = math.log(sum(math.exp(x) for x in row))
        out.append([x - total for x in row])
    return out


NAME = {"Jane": {"B-person": 6.0}, "Q.": {"O": 2.0, "I-person": 1.9}, "Doe": {"E-person": 6.0}}


def _pipe(
    picks: dict[str, dict[str, float]],
    *,
    biases: dict[str, float] | None = None,
    window: int = 512,
    labels: list[str] = BIOES_PERSON,
) -> TaggerPipe:
    tokenizer = TagTokenizer(window)
    model = TagModel(labels, tokenizer, picks)
    tags = _tags(labels)
    return TaggerPipe(
        model,
        tokenizer,
        window // 4,
        tags,
        decoder_for(tags, biases),
        lambda ids: _log_softmax(model.rows(ids)),
    )


def test_the_pipe_reports_spans_like_the_pipeline() -> None:
    text = "Ask Jane Q. Doe or Kim today"
    found = _pipe({**NAME, "Kim": {"S-person": 6.0}}, biases=dict(ZERO_BIASES))(text)
    assert [(e["entity_group"], text[e["start"] : e["end"]]) for e in found] == [
        ("person", "Jane Q. Doe"),
        ("person", "Kim"),
    ]
    assert all(0.5 < e["score"] <= 1.0 for e in found)


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_a_text_without_tokens_never_reaches_the_model(text: str) -> None:
    # openai/privacy-filter's tokenizer adds no special tokens: an empty
    # string is one window of no token ids, and a model call on it raised
    # (torch made the empty id list a float tensor), failing the request.
    class NoSpecials(TagTokenizer):
        def __call__(self, text: str, **kwargs: Any) -> TagEncoding:  # type: ignore[override]
            out = super().__call__(text, **kwargs)
            for key in ("input_ids", "offset_mapping", "special_tokens_mask"):
                out[key] = [row[1:-1] for row in out[key]]
            return out

    def scorer(ids: list[int]) -> list[list[float]]:
        raise AssertionError("scored a window without tokens")

    tags = _tags(BIOES_PERSON)
    pipe = TaggerPipe(None, NoSpecials(512), 128, tags, decoder_for(tags, None), scorer)
    assert pipe(text) == []


def test_without_a_calibration_the_pipe_reads_greedily() -> None:
    text = "Ask Jane Q. Doe today"
    found = _pipe(NAME)(text)
    assert [text[e["start"] : e["end"]] for e in found] == ["Jane", "Doe"]


def test_the_pipe_reads_every_window_at_absolute_offsets() -> None:
    words = [f"w{i}" for i in range(1500)]
    text = " ".join(words) + " met Jane Q. Doe here " + " ".join(words[:300])
    pipe = _pipe(NAME, biases=dict(ZERO_BIASES))
    found = pipe(text)
    start = text.index("Jane")
    assert {(e["start"], e["end"]) for e in found} == {(start, start + len("Jane Q. Doe"))}
    assert pipe.model.calls == 0  # the pipe calls the scorer, not the model


def test_blanks_at_a_span_edge_are_left_out() -> None:
    tokenizer = TagTokenizer()
    labels = BIOES_PERSON
    tags = _tags(labels)

    class Spaced:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[Any]]:
            # a byte-level BPE token's offsets include the blank before it,
            # and a token of blanks only is a span of nothing
            return {
                "input_ids": [[0, 1]],
                "offset_mapping": [[(3, 8), (8, 10)]],
                "special_tokens_mask": [[0, 0]],
            }

    rows = [[0.0, 0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, 0.0, 5.0]]
    pipe = TaggerPipe(
        TagModel(labels, tokenizer, {}),
        Spaced(),
        0,
        tags,
        decoder_for(tags, None),
        lambda ids: rows,
    )
    text = "hi  Jane    x"
    assert [(e["start"], e["end"]) for e in pipe(text)] == [(4, 8)]


# WordPiece cuts "Merkel" into Me ##rk ##el; a tagger trained the Hugging Face
# way labels each word on its first piece only (the later pieces are never
# trained), so they score O here.
MERKEL = {"Merkel": ["Me", "##rk", "##el"]}
FIRST_PIECES = {"Angela": {"B-person": 6.0}, "Me": {"E-person": 6.0}}


def _word_pipe(
    picks: dict[str, dict[str, float]],
    *,
    window: int = 512,
    biases: dict[str, float] | None = None,
) -> TaggerPipe:
    tokenizer = TagTokenizer(window, pieces=MERKEL)
    model = TagModel(BIOES_PERSON, tokenizer, picks)
    tags = _tags(BIOES_PERSON)
    return TaggerPipe(
        model,
        tokenizer,
        window // 4,
        tags,
        decoder_for(tags, biases),
        lambda ids: _log_softmax(model.rows(ids)),
        by_word=True,
    )


@pytest.mark.parametrize("biases", [None, dict(ZERO_BIASES)])
def test_word_pieces_are_decoded_word_by_word(biases: dict[str, float] | None) -> None:
    text = "Ask Angela Merkel today"
    found = _word_pipe(FIRST_PIECES, biases=biases)(text)
    assert [(e["entity_group"], text[e["start"] : e["end"]]) for e in found] == [
        ("person", "Angela Merkel")
    ]
    assert found[0]["score"] > 0.9  # the two words' first pieces, not the O pieces
    # A later piece's label is never read: one tagged S does not cut the name.
    found = _word_pipe({**FIRST_PIECES, "##el": {"S-person": 9.0}}, biases=biases)(text)
    assert [text[e["start"] : e["end"]] for e in found] == ["Angela Merkel"]


def test_a_word_a_window_edge_cuts_is_read_by_its_first_piece_and_reported_whole() -> None:
    text = "w0 w1 w2 Angela Merkel w3"
    # 5 tokens a window, 1 shared: [w0 w1 w2 Angela Me] [Me ##rk ##el w3].
    # The first window holds only the first piece of "Merkel", which it
    # still reports whole.
    found = _word_pipe(FIRST_PIECES, window=7)(text)
    assert [text[e["start"] : e["end"]] for e in found] == ["Angela Merkel", "Merkel"]
    # 4 tokens a window: [w0 w1 w2 Angela] [Angela Me ##rk ##el] [##el w3].
    # The last window opens inside "Merkel", whose first piece it lacks: the
    # word is not read there, whatever its later piece scores.
    found = _word_pipe({**FIRST_PIECES, "##el": {"S-person": 9.0}}, window=6)(text)
    assert [text[e["start"] : e["end"]] for e in found] == ["Angela", "Angela Merkel"]


def test_the_torch_scorer_returns_log_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = install_torch(monkeypatch)
    tokenizer = TagTokenizer()
    tokenizer._id("Jane")
    model = TagModel(BIOES_PERSON, tokenizer, {"Jane": {"S-person": 3.0}})
    rows = torch_scorer(model)([TagTokenizer.CLS, 0])
    assert len(rows) == 2
    assert max(range(5), key=lambda j: rows[1][j]) == 4
    assert sum(math.exp(x) for x in rows[0]) == pytest.approx(1.0)
    # Integer token ids, whatever the list holds (an empty list made a
    # float tensor the embedding refused).
    assert torch.made == [{"data": [[TagTokenizer.CLS, 0]], "dtype": "torch.long", "device": "cpu"}]


# --- building ----------------------------------------------------------------------


def _tagger_pipe(
    picks: dict[str, dict[str, float]], labels: list[str] = BIOES_PERSON
) -> FakeHfPipe:
    tokenizer = TagTokenizer()
    pipe = FakeHfPipe([], tokenizer=tokenizer)  # type: ignore[arg-type]
    pipe.model = TagModel(labels, tokenizer, picks)
    return pipe


def _entry(model_id: str, **fields: Any) -> CatalogEntry:
    return CatalogEntry(
        model_id=model_id,
        backends=("hf",),
        license="Apache-2.0",
        status="caution",
        reason="test",
        **fields,
    )


def test_a_bio_model_keeps_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = FakeHfPipe([("Jane Doe", "PER", 0.9)], id2label={0: "O", 1: "B-PER", 2: "I-PER"})
    install_transformers(monkeypatch, pipe)
    detector = build_hf_detector(NerConfig(enabled=True, backend="hf"))
    assert len(pipe.built_with) == 2  # loaded, then the strided pipeline
    assert [d.value for d in detector.detect("hi Jane Doe")] == ["Jane Doe"]


def test_a_bioes_model_is_decoded_without_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_torch(monkeypatch)
    pipe = _tagger_pipe({**NAME, "Kim": {"S-person": 6.0}})
    install_transformers(monkeypatch, pipe)
    monkeypatch.setattr(model_catalog, "lookup", lambda model_id: None)
    detector = build_hf_detector(
        NerConfig(enabled=True, backend="hf", model="org/bioes-ner", entities=("PERSON",))
    )
    assert len(pipe.built_with) == 1  # loaded only: no strided pipeline
    assert detector.emittable_types == frozenset({"PERSON"})
    text = "Ask Jane Q. Doe or Kim"
    # No calibration in the catalog: greedy, so the O between cuts the name.
    assert [d.value for d in detector.detect(text)] == ["Jane", "Doe", "Kim"]
    assert {d.detector_type for d in detector.detect(text)} == {"PERSON"}


def test_a_bilou_model_is_decoded_too(monkeypatch: pytest.MonkeyPatch) -> None:
    install_torch(monkeypatch)
    picks = {"Jane": {"B-PER": 6.0}, "Doe": {"L-PER": 6.0}, "Kim": {"U-PER": 6.0}}
    install_transformers(monkeypatch, _tagger_pipe(picks, BILOU_PERSON))
    monkeypatch.setattr(model_catalog, "lookup", lambda model_id: None)
    detector = build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/bilou-ner"))
    assert [d.value for d in detector.detect("Jane Doe and Kim")] == ["Jane Doe", "Kim"]


def test_a_wordpiece_tagger_reports_whole_words(monkeypatch: pytest.MonkeyPatch) -> None:
    # Decoded piece by piece, "Me" (E) would close the name and "rkel" would
    # go upstream as sent.
    install_torch(monkeypatch)
    tokenizer = TagTokenizer(pieces=MERKEL)
    pipe = FakeHfPipe([], tokenizer=tokenizer)  # type: ignore[arg-type]
    pipe.model = TagModel(BIOES_PERSON, tokenizer, FIRST_PIECES)
    install_transformers(monkeypatch, pipe)
    monkeypatch.setattr(model_catalog, "lookup", lambda model_id: None)
    detector = build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/bert-bioes"))
    found = detector.detect("Ask Angela Merkel today")
    assert [(d.detector_type, d.value) for d in found] == [("PERSON", "Angela Merkel")]


def test_a_catalogued_calibration_selects_viterbi(monkeypatch: pytest.MonkeyPatch) -> None:
    install_torch(monkeypatch)
    hub = FakeHub()
    assert hub.default is not None
    hub.repos["org/bioes-ner"] = {
        **hub.default,
        "viterbi_calibration.json": json.dumps(_calibration()),
    }
    pipe = _tagger_pipe(NAME)
    install_transformers(monkeypatch, pipe, hub)
    entry = _entry("org/bioes-ner", tagging="bioes", viterbi_calibration="viterbi_calibration.json")
    monkeypatch.setattr(model_catalog, "lookup", lambda model_id: entry)
    detector = build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/bioes-ner"))
    assert "viterbi_calibration.json" in hub.calls[0]["allow_patterns"]
    assert [d.value for d in detector.detect("Ask Jane Q. Doe today")] == ["Jane Q. Doe"]
    # Windows are counted as for the pipeline.
    assert detector.stats.scanned_whole == 1


def test_a_long_text_is_read_in_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    install_torch(monkeypatch)
    pipe = _tagger_pipe(NAME)
    install_transformers(monkeypatch, pipe)
    monkeypatch.setattr(model_catalog, "lookup", lambda model_id: _entry("org/x", window=64))
    detector = build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/x"))
    text = " ".join(f"w{i}" for i in range(500)) + " Jane Doe " + " ".join(["x"] * 30)
    found = detector.detect(text)
    start = text.index("Jane")
    assert [(d.start, d.end) for d in found] == [(start, start + 8)]
    assert detector.stats.scanned_windowed == 1
    assert detector.stats.windows == 12  # 62 tokens a window, 16 shared


def test_a_missing_calibration_file_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A file the catalog lists is part of the model: missing, the model is
    # not (completely) cached — never decoded greedily instead.
    from llm_redact.detection.model_files import ModelNotCached

    hub = FakeHub()
    install_transformers(monkeypatch, _tagger_pipe(NAME), hub)
    entry = _entry("org/bioes-ner", viterbi_calibration="viterbi_calibration.json")
    monkeypatch.setattr(model_catalog, "lookup", lambda model_id: entry)
    with pytest.raises(ModelNotCached) as caught:
        build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/bioes-ner"))
    assert caught.value.missing == ("viterbi_calibration.json",)
    assert str(caught.value).endswith("; missing: viterbi_calibration.json")
    # A repository that lacks it, with downloads on.
    with pytest.raises(ConfigError) as caught:
        build_hf_detector(
            NerConfig(enabled=True, backend="hf", model="org/bioes-ner", allow_download=True)
        )
    assert str(caught.value) == (
        "[detection.ner] hf model 'org/bioes-ner' (no revision pinned): its repository lacks"
        " what the loader needs: viterbi_calibration.json"
    )
    # A local folder its sidecar identifies as the catalogued model.
    assert hub.default is not None
    for name, text in {
        **hub.default,
        SIDECAR_NAME: json.dumps({"model_id": "org/bioes-ner", "revision": None}),
    }.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    with pytest.raises(ConfigError) as caught:
        build_hf_detector(NerConfig(enabled=True, backend="hf", model=str(tmp_path)))
    assert str(caught.value) == (
        f"[detection.ner] hf model {str(tmp_path)!r} is a local directory that lacks what the"
        " loader needs: viterbi_calibration.json"
    )


def test_a_malformed_calibration_file_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub()
    assert hub.default is not None
    hub.repos["org/bioes-ner"] = {**hub.default, "calibration.json": json.dumps({"x": 1})}
    install_transformers(monkeypatch, _tagger_pipe(NAME), hub)
    entry = _entry("org/bioes-ner", viterbi_calibration="calibration.json")
    monkeypatch.setattr(model_catalog, "lookup", lambda model_id: entry)
    with pytest.raises(ConfigError) as caught:
        build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/bioes-ner"))
    assert str(caught.value) == (
        "[detection.ner] hf model 'org/bioes-ner': calibration.json must hold exactly one key,"
        " operating_points, an object"
    )


def test_mixed_tags_refuse_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transformers(monkeypatch, _tagger_pipe({}, ["O", "B-PER", "E-PER", "U-PER"]))
    monkeypatch.setattr(model_catalog, "lookup", lambda model_id: None)
    with pytest.raises(ConfigError) as caught:
        build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/odd"))
    assert str(caught.value) == (
        "[detection.ner] hf model 'org/odd': its labels mix BIOES (E-, S-) and BILOU (L-, U-) tags"
    )


def test_a_tagger_needs_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch fails
    install_transformers(monkeypatch, _tagger_pipe(NAME))
    monkeypatch.setattr(model_catalog, "lookup", lambda model_id: None)
    with pytest.raises(ConfigError, match="torch is not installed"):
        build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/bioes-ner"))


def test_the_privacy_filter_labels_fold_into_placeholder_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_torch(monkeypatch)
    picks = {"Jane": {"S-private_person": 6.0}, "Elm": {"S-private_address": 6.0}}
    install_transformers(monkeypatch, _tagger_pipe(picks, PRIVACY_LABELS))
    monkeypatch.setattr(model_catalog, "lookup", lambda model_id: None)
    detector = build_hf_detector(
        NerConfig(enabled=True, backend="hf", model="org/privacy", entities=("PERSON", "ADDRESS"))
    )
    assert detector.emittable_types == frozenset({"PERSON", "ADDRESS"})
    found = detector.detect("Jane lives on Elm")
    assert [(d.detector_type, d.value) for d in found] == [("PERSON", "Jane"), ("ADDRESS", "Elm")]


def test_a_library_older_than_the_catalog_says_refuses_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # openai/privacy-filter's model type exists from transformers 5.6.0
    # (the hf extra allows older releases): the build says so, naming the
    # versions, instead of failing inside the load.
    from llm_redact.detection import model_sources
    from llm_redact.detection.engine import DetectionConfig, build_detectors

    install_torch(monkeypatch)
    install_transformers(monkeypatch, _tagger_pipe(NAME))
    installed = {"transformers": "5.5.4"}
    monkeypatch.setattr(model_sources.importlib.metadata, "version", installed.__getitem__)
    ner = NerConfig(enabled=True, backend="hf", model="openai/privacy-filter")
    with pytest.raises(ConfigError) as caught:
        build_detectors(DetectionConfig(enabled=(), ner=ner))
    assert str(caught.value) == (
        "[detection.ner] hf: openai/privacy-filter needs transformers >= 5.6.0 (model"
        " catalog), but transformers 5.5.4 is installed; upgrade it: uv sync --extra hf"
        " --upgrade-package transformers"
    )
    # The version it names, or a later one, builds.
    hub = FakeHub()
    assert hub.default is not None
    hub.repos["openai/privacy-filter"] = {
        **hub.default,
        "viterbi_calibration.json": json.dumps(_calibration()),
    }
    install_transformers(monkeypatch, _tagger_pipe(NAME), hub)
    for version in ("5.6.0", "5.10.1"):
        installed["transformers"] = version
        assert build_detectors(DetectionConfig(enabled=(), ner=ner))


def test_every_tag_combination_of_two_tokens_reads_without_error() -> None:
    tags = _tags(TAG_SETS[2])
    for path in itertools.product(range(len(tags)), repeat=2):
        for first, last, _ in spans_of(list(path), tags):
            assert 0 <= first <= last <= 1


# --- a real tokenizer and model (deselected by default) ----------------------------

# dslim/bert-base-NER's WordPiece tokenizer (MIT), at the revision the catalog
# pins (checked 2026-10-05): a real fast tokenizer's windows and offsets.
DSLIM = "dslim/bert-base-NER"
DSLIM_REVISION = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
DSLIM_TOKENIZER = [
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
]
REAL_TEXTS = [
    '{"name":"Angela Merkel","role":"chancellor"}',
    "x " * 900 + "Hi Angela Merkel. " + "y " * 300,
    "Angela Merkel",
]


def _constructed_tagger(torch: Any, transformers: Any, tokenizer: Any) -> Any:
    """A BERT token classifier with no encoder layer whose label is fixed per
    token id: the pieces of "Angela Merkel" in REAL_TEXTS tag B, I…, E;
    every other token O."""
    size = len(BIOES_PERSON)
    config = transformers.BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=size,
        num_hidden_layers=0,
        num_attention_heads=1,
        intermediate_size=4,
        id2label=dict(enumerate(BIOES_PERSON)),
        label2id={label: i for i, label in enumerate(BIOES_PERSON)},
    )
    model = transformers.BertForTokenClassification(config)
    embeddings = model.bert.embeddings
    with torch.no_grad():
        embeddings.word_embeddings.weight.zero_()
        embeddings.word_embeddings.weight[:, 0] = 10.0
        embeddings.position_embeddings.weight.zero_()
        embeddings.token_type_embeddings.weight.zero_()
        embeddings.LayerNorm.weight.fill_(1.0)
        embeddings.LayerNorm.bias.zero_()
        model.classifier.weight.copy_(torch.eye(size))
        model.classifier.bias.zero_()
        for text in REAL_TEXTS:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            start = text.index("Angela")
            end = start + len("Angela Merkel")
            pieces = [
                token
                for token, (s, e) in zip(
                    encoded["input_ids"], encoded["offset_mapping"], strict=True
                )
                if s < end and e > start
            ]
            for token, label in zip(pieces, [1] + [2] * (len(pieces) - 2) + [3], strict=True):
                embeddings.word_embeddings.weight[token].zero_()
                embeddings.word_embeddings.weight[token, label] = 10.0
    return model


@pytest.mark.real_model
def test_a_real_tokenizer_and_model_decode_whole_spans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    offline_hub(monkeypatch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cached_snapshot(DSLIM, DSLIM_REVISION, DSLIM_TOKENIZER)
    )
    _constructed_tagger(torch, transformers, tokenizer).save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    detector = build_hf_detector(NerConfig(enabled=True, backend="hf", model=str(tmp_path)))
    for text in REAL_TEXTS:
        found = detector.detect(text)
        start = text.index("Angela")
        assert [(d.detector_type, d.start, d.end) for d in found] == [
            ("PERSON", start, start + len("Angela Merkel"))
        ]
    # The long text was read in windows of the real tokenizer.
    assert detector.stats.scanned_windowed == 1
    assert detector.stats.windows == 3


# openai/privacy-filter (Apache-2.0) at the catalog pin checked on 2026-10-07,
# pulled by the CI ner-models job (tests/real_model_configs/hf-openai-
# privacy-filter.toml): 2.8 GB of weights and about a second per
# 500-character string on a 4-core CPU, so only short strings here.
PRIVACY = "openai/privacy-filter"
PRIVACY_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
PRIVACY_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "viterbi_calibration.json",
]
PRIVACY_CONFIG = Path(__file__).parent / "real_model_configs" / "hf-openai-privacy-filter.toml"
PRIVACY_SENTENCE = "Wire it to account 00123456789 for Maria Gonzalez, 42 Elm Street, Springfield."


@pytest.mark.real_model
def test_real_privacy_filter_finds_whole_spans_with_its_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from llm_redact.config import load_config
    from llm_redact.detection.engine import build_detectors, ner_backends
    from llm_redact.detection.model_sources import version_problems

    entry = model_catalog.lookup(PRIVACY)
    if problems := version_problems(entry, "hf", PRIVACY):
        pytest.skip(problems[0])  # an older transformers than the model needs
    offline_hub(monkeypatch)
    cached_snapshot(PRIVACY, PRIVACY_REVISION, PRIVACY_FILES)
    # The CI config, through the proxy's own build (downloads off): the
    # catalog pin, the BIOES tagger and the calibration file it lists.
    detection = load_config(PRIVACY_CONFIG).detection
    (detector,) = ner_backends(build_detectors(detection))
    assert isinstance(detector, HfDetector)
    assert detector.model_name == PRIVACY
    pipe = detector._pipe
    assert isinstance(pipe, TaggerPipe)
    assert pipe._decode is not argmax_path  # the constrained Viterbi decoder
    expected = [
        ("ACCOUNT_NUMBER", "00123456789"),
        ("PERSON", "Maria Gonzalez"),
        ("ADDRESS", "42 Elm Street, Springfield"),
    ]
    # The same sentence as plain text and inside a JSON string: each value
    # at its own offsets, no quote or blank taken in.
    for text in (PRIVACY_SENTENCE, json.dumps({"note": PRIVACY_SENTENCE})):
        found = sorted(detector.detect(text), key=lambda d: d.start)
        assert [(d.detector_type, d.value) for d in found] == expected
        assert [(d.start, d.end) for d in found] == [
            (text.index(value), text.index(value) + len(value)) for _type, value in expected
        ]
    # An empty text is no model call on no token ids (which fails), and a
    # blank one no span (span edges lose their blanks).
    assert detector.detect("") == []
    assert detector.detect("   \n\t ") == []
