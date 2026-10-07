"""Optional Hugging Face token-classification NER backend (`hf` extra).

Any `token-classification` model on the Hub (multilingual XLM-R NER, biomedical
NER, domain-tuned checkpoints, …) becomes a detector, which is the escape
hatch for teams that already have a fine-tuned model. Uses the `transformers`
pipeline, which merges a model's per-token labels into entity spans with a
confidence that `score_threshold` gates.

A span must cover whole words: the pipeline's token-level aggregation
(``"simple"``) ends a span wherever a word piece's label differs, so
dslim/bert-base-NER reported "Angela Merk" in "Angela Merkel" and the rest
of the name went upstream as sent. A tokenizer that marks word pieces (a
continuing-subword prefix such as WordPiece's ``##``: BERT and its family)
gives the pipeline real word boundaries, and the word-level ``"first"``
aggregation then labels every word by its first piece (:func:`aggregation_for`).
Other tokenizers (SentencePiece, byte-level BPE) give it none: transformers
falls back to a whitespace heuristic that glues ``{"name":"Angela`` — or a
whole sentence in a script written without spaces — into one "word" labelled
by its first piece, and its token-level aggregation makes every piece its own
value (kalyan-ks/ettin-68m-nemotron-pii tags every piece ``B-``: "Zbigniew
Brzezinski" was six values). Such a BIO model is decoded by llm-redact
itself (:class:`TaggerPipe`) over the text's own words (:func:`text_words`:
cut at blanks, quotes, brackets and value delimiters, each character of a
script written without spaces alone), each labelled by its first piece (by
its most confidently tagged piece for a model the catalog says labels every
piece); a span is cut at a quote, a bracket, a colon or a newline between
its words (:func:`_cut_spans`), then grows over the characters beside it
that the model tags with its entity but never over a blank, a quote or a
bracket (:func:`_grown`): the words choose which piece labels a value,
never which of its tagged characters go upstream.

Long strings are read whole, in overlapping token windows: without `stride`
the pipeline truncates at the tokenizer's maximum length and never reads the
rest. `stride` needs a fast tokenizer — which also reports the character
offsets every detection needs — so a model without one is refused at startup.

A model tagging BIOES or BILOU (its labels carry `E-`/`S-` or `L-`/`U-` tags,
which the pipeline's aggregation does not understand) runs without the
pipeline too: :class:`TaggerPipe` reads the same token windows, takes the
model's per-token log-probabilities and decodes its spans itself (tagging.py)
— with the constrained Viterbi decoder when the model catalog lists the
model's calibration file, else greedily.

Import-lazy: loads only when an `hf` backend is enabled; the model load
happens at proxy startup (fail fast, no first-request latency spike), in
float32 whatever precision the checkpoint stores (a bfloat16 model runs
slower on a CPU). The files come from a local directory at a pinned revision (model_files.py):
safetensors weights unless `allow_pickle_weights` is set, and never code
from the model's repository (`trust_remote_code=False`).
"""

import bisect
import importlib.metadata
import importlib.util
import math
import re
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.labels import LabelPolicy, is_part_gap, merge_adjacent_parts
from llm_redact.detection.ner import NER_PRIORITY
from llm_redact.detection.stats import NerStats
from llm_redact.detection.tagging import (
    BIO,
    Decoder,
    TaggingError,
    TagSet,
    argmax_path,
    decoder_for,
    spans_of,
    tagging_scheme,
    viterbi_biases,
)
from llm_redact.detection.windows import drop_exact_duplicates

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig
    from llm_redact.detection.model_catalog import CatalogEntry

_MODEL_NAME = "dslim/bert-base-NER"
# A tokenizer that does not know its model's limit reports a huge sentinel
# (transformers' VERY_LARGE_INTEGER); below this it is a real limit.
_SENTINEL_MAX_LENGTH = 1_000_000
# What an encoder reads when its config does not say (BERT and most others).
_DEFAULT_WINDOW = 512
# The precision every hf model runs in (transformers' `dtype`, which it
# resolves to torch.float32): the pipeline and the tagger both use the one
# model the first pipeline() call loads.
MODEL_DTYPE = "float32"
# transformers' pipeline() takes `dtype` from 4.56 on (`torch_dtype` before
# it, and an older pipeline hands the unknown keyword to the task pipeline,
# which refuses it: the load fails with a bare TypeError). The hf extra
# requires it; an environment that kept an older transformers is refused by
# name (transformers_problem).
TRANSFORMERS_MINIMUM = "4.56"
# The pipeline's aggregation strategies: per word (a word-aware tokenizer)
# or per token (aggregation_for).
WORD_AGGREGATION = "first"
TOKEN_AGGREGATION = "simple"


class _PipelineLike(Protocol):
    """The sliver of the transformers token-classification pipeline used."""

    def __call__(self, text: str) -> list[dict[str, Any]]: ...


def _model_types(pipe: _PipelineLike, policy: LabelPolicy) -> frozenset[str] | None:
    """The placeholder types the pipeline's model can emit under ``policy``:
    its ``config.id2label`` labels, classified. None when the pipeline does
    not say (any type may come)."""
    id2label = getattr(getattr(getattr(pipe, "model", None), "config", None), "id2label", None)
    if not isinstance(id2label, Mapping):
        return None
    types = (policy.classify(str(label)) for label in id2label.values())
    return frozenset(t for t in types if t is not None)


class HfDetector:
    name = "hf"
    # A model runs per string: the detector plan detects it ahead of the
    # redaction, on the NER worker thread (DetectorPlan, ner_prefetch).
    heavy = True

    # Read by the never-match check (engine.build_detectors): the placeholder
    # types this model can emit (None = unknown, e.g. zero-shot), the model
    # for messages, and the configured entities no active backend emits.
    emittable_types: frozenset[str] | None = None
    model_name: str | None = None
    unmatched_entities: tuple[str, ...] = ()

    def __init__(
        self,
        pipe: _PipelineLike,
        entities: frozenset[str],
        max_chars: int,
        threshold: float,
        *,
        policy: LabelPolicy | None = None,
        windows_of: Callable[[str], int] | None = None,
    ) -> None:
        self._pipe = pipe
        # How many windows the pipeline reads a text in (build_hf_detector
        # counts them with the pipeline's own tokenizer); None: one.
        self._windows_of = windows_of
        # The label policy (labels.py) turns the model's labels (`PER`,
        # `B-PER` without aggregation) into placeholder types; by default
        # it is built from `entities`.
        self.label_policy = (
            policy if policy is not None else LabelPolicy(entities, backend=self.name)
        )
        self.emittable_types = _model_types(pipe, self.label_policy)
        self._max_chars = max_chars
        self._threshold = threshold
        # Coverage counters (stats.py): strings read or skipped, entities
        # dropped.
        self.stats = NerStats()

    def detect(self, text: str) -> list[Detection]:
        # An entity two overlapping windows both report counts once; parts
        # of one name or address reported separately join into one span
        # (labels.merge_adjacent_parts).
        return merge_adjacent_parts(drop_exact_duplicates(self._found(text)), text)

    def _found(self, text: str) -> Iterator[Detection]:
        if len(text) > self._max_chars:
            self.stats.skipped_max_chars += 1
            return
        windows = self._windows_of(text) if self._windows_of is not None else 1
        if windows > 1:
            self.stats.scanned_windowed += 1
            self.stats.windows += windows
        else:
            self.stats.scanned_whole += 1
        for ent in self._pipe(text):
            label = self.label_policy.classify(
                str(ent.get("entity_group", ent.get("entity", ""))), self.stats
            )
            if label is None:
                continue
            if float(ent.get("score", 1.0)) < self._threshold:
                continue
            start, end = ent.get("start"), ent.get("end")
            if start is None or end is None or not 0 <= int(start) < int(end) <= len(text):
                # A span the text does not contain cannot be redacted (or
                # restored) faithfully: skip it rather than guess.
                self.stats.offsets_dropped += 1
                continue
            start, end = int(start), int(end)
            # A SentencePiece or byte-level BPE token's offsets take in the
            # blank before its word (" Jane"): the span is the value, never
            # the blank around it, so the two parts of a name stay adjacent
            # parts (one placeholder) and the text keeps its spaces.
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start == end:
                continue
            yield Detection(
                start=start,
                end=end,
                detector_type=label,
                # The source slice, never the pipeline's decoded `word`
                # (convert_tokens_to_string can differ from what the user
                # sent: casing, spacing, [UNK]) — the vault must map the
                # exact sent text, or rehydration restores a value the user
                # never wrote.
                value=text[start:end],
                priority=NER_PRIORITY,
            )


def transformers_problem() -> str | None:
    """Why the installed transformers cannot load an hf model the way
    :func:`build_hf_detector` does — it is older than
    :data:`TRANSFORMERS_MINIMUM` — or None. Distribution metadata only:
    nothing is imported, and a transformers without metadata (a source
    tree) is not judged."""
    from llm_redact.detection.model_sources import version_tuple

    try:
        installed = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return None
    if version_tuple(installed) < version_tuple(TRANSFORMERS_MINIMUM):
        return (
            f"needs transformers >= {TRANSFORMERS_MINIMUM} (its pipeline's dtype),"
            f" but transformers {installed} is installed"
        )
    return None


def model_window(tokenizer: Any, model: Any, catalog_window: int | None = None) -> int:
    """How many tokens (special tokens included) the model reads at once:
    ``catalog_window`` when known, else the smaller of the tokenizer's
    ``model_max_length`` (unless it is the unknown-limit sentinel) and the
    model config's ``max_position_embeddings`` (512 when absent)."""
    if catalog_window is not None:
        return catalog_window
    limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    window = limit if isinstance(limit, int) and limit > 0 else _DEFAULT_WINDOW
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < _SENTINEL_MAX_LENGTH:
        window = min(window, tokenizer_limit)
    return window


def aggregation_for(tokenizer: Any) -> str:
    """The pipeline aggregation that keeps whole words for ``tokenizer``:
    word-level ``"first"`` when the fast tokenizer's model marks word
    pieces with a continuing-subword prefix — the test transformers itself
    makes before it trusts its word boundaries — else ``"simple"`` (a BIO
    model with such a tokenizer is decoded by :class:`TaggerPipe` over the
    text's words instead; ``"simple"`` is left for a model whose labels
    the builder cannot read).

    Measured on dslim/bert-base-NER (WordPiece) with transformers 5.10.1:
    ``"simple"`` cut "Angela Merk", "Ngoz", "Xu Wen"; ``"first"``,
    ``"max"`` and ``"average"`` kept every word whole, and ``"first"``
    alone kept McAllister and DiCaprio (``"max"`` lost the one, ``"average"``
    both, and scored Venkataraman 0.33, under the default threshold)."""
    model = getattr(getattr(tokenizer, "_tokenizer", None), "model", None)
    if getattr(model, "continuing_subword_prefix", None):
        return WORD_AGGREGATION
    return TOKEN_AGGREGATION


def window_counter(tokenizer: Any, stride: int) -> Callable[[str], int]:
    """How many windows the strided pipeline reads a text in: the chunks
    its fast tokenizer makes with the pipeline's own overflow settings
    (truncation at ``model_max_length``, ``stride`` tokens of overlap)."""

    def windows_of(text: str) -> int:
        encoded = tokenizer(text, truncation=True, return_overflowing_tokens=True, stride=stride)
        return len(encoded["input_ids"])

    return windows_of


Scorer = Callable[[list[int]], Sequence[Sequence[float]]]


def torch_scorer(model: Any) -> Scorer:
    """The model's label log-probabilities for one window of token ids, one
    row per token (torch, imported here: only a loaded model needs it)."""
    import torch

    def score(input_ids: list[int]) -> Sequence[Sequence[float]]:
        with torch.inference_mode():
            # Token ids are integers, whatever the list holds.
            ids = torch.tensor([input_ids], dtype=torch.long, device=getattr(model, "device", None))
            logits = model(input_ids=ids).logits[0]
            rows: Sequence[Sequence[float]] = torch.log_softmax(logits.float(), dim=-1).tolist()
            return rows

    return score


# What a tagger decodes in one window: per unit — a token, or a word scored
# by its first piece — the index of its row of label scores, and its start
# and end in the text.
Units = list[tuple[int, int, int]]
# One window as the tokenizer returns it: token ids, character offsets and
# the special-tokens mask.
Window = tuple[Sequence[int], Sequence[Sequence[int]], Sequence[int]]


def _token_units(windows: Sequence[Window]) -> list[Units]:
    """Per window, its tokens: every token that is not special and covers a
    character, at its own offsets."""
    return [
        [
            (index, start, end)
            for index, (start, end) in enumerate(offsets)
            if not special[index] and end > start
        ]
        for _ids, offsets, special in windows
    ]


def _word_units(encoded: Any, windows: Sequence[Window]) -> list[Units]:
    """Per window, its words (the tokenizer's ``word_ids``), each scored by
    the row of its first piece — the piece a tagger is trained to label —
    at the offsets of the whole word: a word a window's edge cuts after its
    first piece is still reported whole, and a window that opens after a
    word's first piece does not read that word (the window before does)."""
    word_ids = [encoded.word_ids(index) for index in range(len(windows))]
    extents: dict[int, tuple[int, int]] = {}
    for words, (_ids, offsets, special) in zip(word_ids, windows, strict=True):
        for word, (start, end), is_special in zip(words, offsets, special, strict=True):
            if word is not None and not is_special and end > start:
                low, high = extents.get(word, (start, end))
                extents[word] = (min(low, start), max(high, end))
    units: list[Units] = []
    for words, (_ids, offsets, special) in zip(word_ids, windows, strict=True):
        found: Units = []
        for index, word in enumerate(words):
            start, end = offsets[index]
            if word is None or special[index] or end <= start or start != extents[word][0]:
                continue  # a special token, or not a word's first piece
            found.append((index, *extents[word]))
        units.append(found)
    return units


# How the BIO decoder of a tokenizer without word-piece marks cuts a text
# into words (text_words). A word is labelled by one piece, so what a word
# holds decides what one label covers: transformers' whitespace fallback
# made `{"name":"Angela` one word, labelled by its brace, and an unspaced
# sentence one word, labelled by its first character. Here a word ends at a
# blank and at every character that delimits values — a quote or a bracket
# of any script (Unicode Ps, Pe, Pi, Pf) and the ASCII delimiters below —
# and each character of a script written without spaces between words is a
# word of its own (_UNSPACED). Punctuation inside a word stays:
# "1985-03-12", "j.doe" and "dev_jo42" are one word each, as the model was
# trained to label them (cutting them there left their later parts to
# pieces no model labels). A slash ends a word too, so each part of a path
# is read on its own. The words only decide which piece labels what: the
# characters a span covers grow past them (_grown).
_DELIMITERS = frozenset("\"'`()[]{}<>,;:=|/\\")
_BRACKETS_AND_QUOTES = frozenset({"Ps", "Pe", "Pi", "Pf"})
# The ASCII quotes and brackets: with a blank and the Unicode quotes and
# brackets, what a span never grows over (_grown).
_ASCII_QUOTES_AND_BRACKETS = frozenset("\"'`()[]{}<>")
# Scripts written without spaces between words (Unicode blocks, first and
# last code point, sorted): each character — with the combining marks that
# follow it — is a word of its own, so a name inside a sentence is labelled
# by its own pieces, as BERT's tokenizer reads CJK ideographs. A run of
# them made one word was labelled by its first piece, and a name inside it
# was never redacted.
_UNSPACED = (
    (0x0E00, 0x0EFF),  # Thai, Lao
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x1950, 0x19FF),  # Tai Le, New Tai Lue, Khmer Symbols
    (0x1A20, 0x1AAF),  # Tai Tham
    (0x2E80, 0x2FDF),  # CJK Radicals Supplement, Kangxi Radicals
    (0x3005, 0x3007),  # ideographic iteration mark, closing mark, number zero
    (0x3021, 0x3029),  # Hangzhou numerals
    (0x3031, 0x3035),  # kana repeat marks
    (0x3038, 0x303C),  # Hangzhou numerals, ideographic marks
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x3100, 0x312F),  # Bopomofo
    (0x31A0, 0x31BF),  # Bopomofo Extended
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xA000, 0xA4CF),  # Yi Syllables, Yi Radicals
    (0xA9E0, 0xA9FF),  # Myanmar Extended-B
    (0xAA60, 0xAADF),  # Myanmar Extended-A, Tai Viet
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),  # Halfwidth Katakana
    (0x1AFF0, 0x1B16F),  # Kana Extended-B, Kana Supplement, Kana Extended-A, Small Kana
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0x2A700, 0x2EE5F),  # CJK Extensions C, D, E, F, I
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    (0x30000, 0x3347F),  # CJK Extensions G, H, J
)
_UNSPACED_LOWS = tuple(low for low, _high in _UNSPACED)
# Runs of characters that are neither blank nor an ASCII delimiter; a run
# that is not pure ASCII is cut further (_cut_run).
_RUN_RE = re.compile(r"[^\s\"'`()\[\]{}<>,;:=|/\\]+")


def _unspaced(char: str) -> bool:
    """Whether ``char`` belongs to a script written without spaces between
    words (:data:`_UNSPACED`)."""
    code = ord(char)
    block = bisect.bisect_right(_UNSPACED_LOWS, code) - 1
    return block >= 0 and code <= _UNSPACED[block][1]


def _punctuation(char: str) -> bool:
    """A Unicode punctuation character or an ASCII symbol (what a word is
    trimmed of at either end)."""
    return (char.isascii() and not char.isalnum() and not char.isspace()) or (
        unicodedata.category(char).startswith("P")
    )


def _trimmed(text: str, start: int, end: int) -> tuple[int, int] | None:
    """``text[start:end]`` without punctuation at either end ("Paris." is
    "Paris", "@jdoe" is "jdoe"); None when nothing else is left."""
    while start < end and _punctuation(text[start]):
        start += 1
    while end > start and _punctuation(text[end - 1]):
        end -= 1
    return (start, end) if start < end else None


def _cut_run(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    """The parts of ``text[start:end]`` (a run without blanks or ASCII
    delimiters) between quotes and brackets of other scripts, each
    character of a script written without spaces a part of its own with
    the combining marks that follow it."""
    begin = index = start
    while index < end:
        char = text[index]
        if char.isascii():
            index += 1
        elif _unspaced(char):
            if begin < index:
                yield begin, index
            begin = index + 1
            while begin < end and unicodedata.category(text[begin]).startswith("M"):
                begin += 1
            yield index, begin
            index = begin
        else:
            if unicodedata.category(char) in _BRACKETS_AND_QUOTES:
                if begin < index:
                    yield begin, index
                begin = index + 1
            index += 1
    if begin < end:
        yield begin, end


def text_words(text: str) -> list[tuple[int, int]]:
    """The (start, end) of each word of ``text`` (see _DELIMITERS above):
    runs of characters between blanks, quotes, brackets and the ASCII
    delimiters, each character of a script written without spaces a word of
    its own, every word without punctuation at either end. Nothing but word
    characters and the punctuation inside a word is ever part of one."""
    words: list[tuple[int, int]] = []
    for match in _RUN_RE.finditer(text):
        parts = [match.span()] if match[0].isascii() else _cut_run(text, *match.span())
        for start, end in parts:
            trimmed = _trimmed(text, start, end)
            if trimmed is not None:
                words.append(trimmed)
    return words


# A word unit: the rows of its pieces in one window (the piece that labels
# it first), and its start and end in the text.
WordUnits = list[tuple[tuple[int, ...], int, int]]
# A piece of a word in one window: its row, start and end.
_Piece = tuple[int, int, int]


def _word_pieces(
    text: str, words: Sequence[tuple[int, int]], starts: Sequence[int], window: Window
) -> dict[int, list[_Piece]]:
    """The pieces of each word (by index into ``words``) in one window, in
    token order: every token covering a character of the word, and a token
    of blanks only right before the word — SentencePiece's lone "▁" before
    a character its vocabulary has no "▁"-word for is the first piece of
    that word wherever it stands (at the start of a text its offsets take in
    the character itself). Special tokens and tokens covering no character
    are no piece."""
    _ids, offsets, special = window
    pieces: dict[int, list[_Piece]] = {}
    for row, (start, end) in enumerate(offsets):
        if special[row] or end <= start:
            continue
        word = max(bisect.bisect_right(starts, start) - 1, 0)
        if word < len(words) and words[word][1] <= start:
            word += 1  # the token starts after this word ends
        if text[start:end].isspace():
            if word < len(words) and words[word][0] == end:
                pieces.setdefault(word, []).append((row, start, end))
            continue
        while word < len(words) and words[word][0] < end:
            pieces.setdefault(word, []).append((row, start, end))
            word += 1
    return pieces


def _text_word_units(
    text: str, words: Sequence[tuple[int, int]], windows: Sequence[Window]
) -> list[WordUnits]:
    """Per window, the words of ``text`` (:func:`text_words`) it decodes,
    each with the rows of its pieces there (:func:`_word_pieces`). A window
    decodes a word when it holds the word's first piece, so a window that
    opens after that piece does not read the word (the window before does)
    and a word a window's edge cuts after its first piece is still reported
    whole. Unless no window holding the first piece reaches the word's last
    piece (a word longer than the windows' overlap): then the part those
    windows read is a unit of its own, and the rest is decoded the same way
    from its own first piece, in the window that holds it — so every piece
    of a text is read by some window."""
    starts = [start for start, _end in words]
    # Each word's pieces in each window that holds any, windows in order.
    found: dict[int, list[tuple[int, list[_Piece]]]] = {}
    for index, window in enumerate(windows):
        for word, pieces in _word_pieces(text, words, starts, window).items():
            found.setdefault(word, []).append((index, pieces))
    units: list[WordUnits] = [[] for _window in windows]
    for word in sorted(found):
        word_start, word_end = words[word]
        begin = word_start
        while begin < word_end:
            # The pieces of the part not decoded yet, per window (the lone
            # blank piece before the word ends where the word starts).
            rest = [
                (index, kept)
                for index, pieces in found[word]
                if (kept := [piece for piece in pieces if piece[2] > begin])
            ]
            if not rest:
                break  # no token covers the rest of the word
            first = min(kept[0][1] for _index, kept in rest)
            holders = [(index, kept) for index, kept in rest if kept[0][1] == first]
            reach = max(piece[2] for _index, kept in holders for piece in kept)
            later = any(piece[2] > reach for _index, kept in rest for piece in kept)
            end = reach if later and reach < word_end else word_end
            for index, kept in holders:
                rows = tuple(row for row, _start, _end in kept)
                if begin == word_start:
                    # The blank piece before the word comes first.
                    lead = dict(found[word])[index]
                    rows = (*(row for row, _s, stop in lead if stop <= begin), *rows)
                units[index].append((rows, begin, end))
            begin = end
    return units


def _word_row(
    rows: Sequence[Sequence[float]],
    pieces: Sequence[int],
    tagset: TagSet,
    every_piece: bool,
) -> Sequence[float]:
    """The label scores a word is decoded by. A model trained the Hugging
    Face way labels a word on its first piece only (its later pieces are
    never trained and may say anything): the first piece's. A model trained
    on every piece (``every_piece``: the model catalog's ``piece_labels =
    "every"``, kalyan-ks/ettin-68m-nemotron-pii) may leave a word's first
    piece untagged or unsure and tag a later one surely: the scores of the
    piece it tags with an entity label most confidently (the first such
    piece on a tie), else the first piece's.

    Measured on the NER bench (docs/ner-landscape.md): for the every-piece
    model the most confidently tagged piece leaked the fewest characters;
    for a first-piece model (OpenMed-PII) reading its untrained later pieces
    turned hyphenated reference numbers into account numbers."""
    if every_piece:
        best: Sequence[float] | None = None
        best_score = -math.inf
        for piece in pieces:
            row = rows[piece]
            label = argmax_path([row])[0]
            if tagset.tags[label] not in (None, "O") and row[label] > best_score:
                best, best_score = row, row[label]
        if best is not None:
            return best
    return rows[pieces[0]]


# What may separate two units of one span: nothing (characters of a script
# written without spaces), one or two blanks (labels.is_part_gap), or a
# comma and a space ("March 3, 1985") or a slash ("03/12/1985") where the
# model tags the second unit as the span's continuation. Anything else — a
# quote, a bracket, a colon, a newline — cuts the span (_cut_spans); a
# character the model tags as part of the value is taken back in (_grown).
_SPAN_GAPS = frozenset({", ", "/"})
# What may separate two spans of the same label that become one value: a
# model that labels every word B- ("03" "12" "1985" of "03/12/1985").
_JOINED_GAPS = frozenset({"", "/"})


@dataclass(slots=True)
class _Span:
    """A value decoded in one window: where it starts and ends in the text,
    its label, and the summed confidence of its units and their number (a
    span scores the mean of its units)."""

    start: int
    end: int
    label: str
    confidence: float
    units: int

    def join(self, other: "_Span") -> None:
        self.end = other.end
        self.confidence += other.confidence
        self.units += other.units


def _cut_spans(
    spans: Sequence[tuple[int, int, str]],
    kept: WordUnits,
    confidence: Sequence[float],
    text: str,
) -> list[_Span]:
    """``spans`` (unit indices into ``kept``) cut wherever two of their
    units are separated by anything else than :data:`_SPAN_GAPS` or one or
    two blanks; each part keeps the confidence of the units the model
    tagged."""
    cut: list[_Span] = []
    for first, last, label in spans:
        start = first
        for unit in range(first, last + 1):
            if unit < last:
                gap = text[kept[unit][2] : kept[unit + 1][1]]
                if not gap or gap in _SPAN_GAPS or is_part_gap(gap):
                    continue
            cut.append(
                _Span(
                    kept[start][1],
                    kept[unit][2],
                    label,
                    sum(confidence[start : unit + 1]),
                    unit + 1 - start,
                )
            )
            start = unit + 1
    return cut


def _tag_chars(
    tagged: dict[int, set[str]], rows: Sequence[Sequence[float]], window: Window, tagset: TagSet
) -> None:
    """Record in ``tagged`` (a character position: the entities it is
    tagged with) every character of each piece of ``window`` whose best
    label tags an entity."""
    _ids, offsets, special = window
    for row, (start, end) in enumerate(offsets):
        if special[row] or end <= start:
            continue
        label = argmax_path([rows[row]])[0]
        if tagset.tags[label] in (None, "O"):
            continue
        for position in range(start, end):
            tagged.setdefault(position, set()).add(tagset.entities[label])


def _grows(
    text: str,
    position: int,
    label: str,
    words: Sequence[tuple[int, int]],
    starts: Sequence[int],
    tagged: Mapping[int, set[str]],
) -> bool:
    """Whether a span labelled ``label`` takes in the character at
    ``position`` beside it: a character no word holds (a word is decoded by
    its own label), neither a blank nor a quote or a bracket of any script,
    inside a piece the model tags with the span's entity."""
    char = text[position]
    if (
        char.isspace()
        or char in _ASCII_QUOTES_AND_BRACKETS
        or unicodedata.category(char) in _BRACKETS_AND_QUOTES
    ):
        return False
    word = bisect.bisect_right(starts, position) - 1
    if word >= 0 and words[word][1] > position:
        return False
    return label in tagged.get(position, ())


def _grown(
    cut: Sequence[_Span],
    text: str,
    words: Sequence[tuple[int, int]],
    starts: Sequence[int],
    tagged: Mapping[int, set[str]],
) -> list[_Span]:
    """The spans of one window grown over the characters beside them that
    :func:`_grows` lets them take in — a symbol at a word's edge
    ("$ecret!"), a delimiter inside a value ("p@ss:w0rd") — each character
    to the first span that reaches it; then neighbouring spans of the same
    label joined over :data:`_JOINED_GAPS`. The words only decide which
    piece labels what: a character the model tags as part of a value is
    redacted with it, never sent upstream as part of it."""
    joined: list[_Span] = []
    taken = 0  # where the span before ends
    for span in cut:
        while span.start > taken and _grows(
            text, span.start - 1, span.label, words, starts, tagged
        ):
            span.start -= 1
        while span.end < len(text) and _grows(text, span.end, span.label, words, starts, tagged):
            span.end += 1
        taken = span.end
        if (
            joined
            and joined[-1].label == span.label
            and text[joined[-1].end : span.start] in _JOINED_GAPS
        ):
            joined[-1].join(span)
        else:
            joined.append(span)
    return joined


def _joined_across(found: Sequence[_Span]) -> list[_Span]:
    """The spans every window decoded, sorted, those of one label that meet
    (one ends where the other starts: the parts of a word no single window
    reads whole) joined."""
    spans: list[_Span] = []
    ending: dict[tuple[str, int], _Span] = {}
    for span in sorted(found, key=lambda span: (span.start, span.end)):
        before = ending.pop((span.label, span.start), None)
        if before is None:
            spans.append(span)
            before = span
        else:
            before.join(span)
        ending[(before.label, before.end)] = before
    return spans


class TaggerPipe:
    """A token-classification model llm-redact decodes itself, reading the
    windows the strided pipeline reads: the fast tokenizer's overlapping
    windows (``model_max_length`` tokens, ``stride`` shared), and per window
    one model call whose label scores ``decode`` turns into a label path
    (tagging.decoder_for). Used for a BIOES/BILOU tagger, whose tags the
    pipeline's aggregation does not understand, and for a BIO tagger whose
    tokenizer does not mark word pieces, which the pipeline cannot read word
    by word.

    The path runs over units: with ``by_word`` — a tokenizer that marks word
    pieces (:func:`aggregation_for`), whose models are trained to label a
    word on its first piece — the window's words, each scored by its first
    piece as the pipeline's word-level aggregation reads a BIO model; with
    ``by_text_word`` the words of the text itself (:func:`text_words`), each
    scored by :func:`_word_row` (``every_piece``: the model labels every
    piece of a word), a span cut by :func:`_cut_spans`, grown over the
    characters beside it the model tags with its entity and joined by
    :func:`_grown`, and the parts of a word no window reads whole joined by
    :func:`_joined_across`; otherwise the window's tokens. Each span is
    reported like a pipeline entity — its entity label (the tag dropped),
    the mean probability of the labels of the units the model tagged as one
    span, and character offsets into the whole text, without the whitespace
    at either edge. Special tokens and tokens covering no character are not
    decoded."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        stride: int,
        tagset: TagSet,
        decode: Decoder,
        scorer: Scorer,
        *,
        by_word: bool = False,
        by_text_word: bool = False,
        every_piece: bool = False,
    ) -> None:
        # Read by HfDetector: the model's config.id2label names what it emits.
        self.model = model
        self._tokenizer = tokenizer
        self._stride = stride
        self._tagset = tagset
        self._decode = decode
        self._scorer = scorer
        self._by_word = by_word
        self._by_text_word = by_text_word
        self._every_piece = every_piece

    def __call__(self, text: str) -> list[dict[str, Any]]:
        encoded = self._tokenizer(
            text,
            truncation=True,
            return_overflowing_tokens=True,
            stride=self._stride,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        windows: list[Window] = list(
            zip(
                encoded["input_ids"],
                encoded["offset_mapping"],
                encoded["special_tokens_mask"],
                strict=True,
            )
        )
        if self._by_text_word:
            return self._text_word_entities(text, windows)
        units = _word_units(encoded, windows) if self._by_word else _token_units(windows)
        entities: list[dict[str, Any]] = []
        for (ids, _offsets, _special), kept in zip(windows, units, strict=True):
            if not kept:
                # No token covers a character (an empty or blank text, a
                # tokenizer without special tokens): nothing to decode, and
                # a model call on no token ids fails.
                continue
            rows = self._scorer(list(ids))
            scores = [rows[row] for row, _start, _end in kept]
            path = self._decode(scores)
            for first, last, label in spans_of(path, self._tagset):
                probabilities = [math.exp(scores[t][path[t]]) for t in range(first, last + 1)]
                _add_entity(entities, text, label, probabilities, kept[first][1], kept[last][2])
        return entities

    def _text_word_entities(self, text: str, windows: Sequence[Window]) -> list[dict[str, Any]]:
        words = text_words(text)
        if not words:
            return []  # nothing a span could start at: no model call
        starts = [start for start, _end in words]
        # Every character a piece the model tags covers, in any window: a
        # span grows over them (_grown) once every window is read.
        tagged: dict[int, set[str]] = {}
        cut: list[list[_Span]] = []
        for window, kept in zip(windows, _text_word_units(text, words, windows), strict=True):
            ids, offsets, special = window
            if all(special[row] or end <= start for row, (start, end) in enumerate(offsets)):
                continue  # no token covers a character (see __call__)
            rows = self._scorer(list(ids))
            _tag_chars(tagged, rows, window, self._tagset)
            if not kept:
                continue  # no word decoded here: its pieces only tag characters
            scores = [
                _word_row(rows, pieces, self._tagset, self._every_piece)
                for pieces, _start, _end in kept
            ]
            path = self._decode(scores)
            # Each unit scores the mean probability of the span the model
            # tagged it in, whatever cutting and joining makes of the span.
            confidence = [0.0] * len(kept)
            spans = spans_of(path, self._tagset)
            for first, last, _label in spans:
                probabilities = [math.exp(scores[t][path[t]]) for t in range(first, last + 1)]
                confidence[first : last + 1] = [sum(probabilities) / len(probabilities)] * (
                    last + 1 - first
                )
            cut.append(_cut_spans(spans, kept, confidence, text))
        found = [span for spans in cut for span in _grown(spans, text, words, starts, tagged)]
        return [
            {
                "entity_group": span.label,
                "score": span.confidence / span.units,
                "start": span.start,
                "end": span.end,
            }
            for span in _joined_across(found)
        ]


def _add_entity(
    entities: list[dict[str, Any]],
    text: str,
    label: str,
    probabilities: Sequence[float],
    start: int,
    end: int,
) -> None:
    """Append the entity ``text[start:end]`` scored by the mean of
    ``probabilities``, without the blanks at either edge (a token's offsets
    may take in the blank before a word); nothing when only blanks are
    left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end:
        return
    entities.append(
        {
            "entity_group": label,
            "score": sum(probabilities) / len(probabilities),
            "start": start,
            "end": end,
        }
    )


def _catalog_entry(model: str) -> "CatalogEntry | None":
    """The model catalog's ``hf`` entry for ``model`` (a Hub id, or a local
    directory its sidecar file identifies); None when there is none."""
    from llm_redact.detection.model_files import catalog_entry

    return catalog_entry(model, "hf")


def catalog_window(model: str) -> int | None:
    """The token window the model catalog records for ``model`` (a Hub id,
    or a local directory its sidecar file identifies) on the ``hf``
    backend; None when it records none."""
    entry = _catalog_entry(model)
    return entry.window if entry is not None else None


def _tagger_biases(path: Path, model: str, calibration: str | None) -> dict[str, float] | None:
    """The Viterbi transition biases of the model's calibration file the
    catalog lists (``viterbi_calibration``); None when it lists none. The
    file is there: :func:`model_files.hf_files` requires every file the
    catalog lists for the model, as ``llm-redact models verify`` does."""
    from llm_redact.config import ConfigError
    from llm_redact.detection.model_files import read_config

    if calibration is None:
        return None
    try:
        return viterbi_biases(read_config(path / calibration, what="hf model", model=model))
    except TaggingError as exc:
        raise ConfigError(f"[detection.ner] hf model {model!r}: {calibration} {exc}") from exc


def _tagger(
    loaded: Any,
    tokenizer: Any,
    stride: int,
    biases: dict[str, float] | None,
    model: str,
    *,
    every_piece: bool = False,
) -> TaggerPipe | None:
    """A :class:`TaggerPipe` for a model whose labels tag BIOES or BILOU,
    or tag BIO with a tokenizer that does not mark word pieces (read word
    by word over the text's own words); None for a BIO model whose
    tokenizer marks word pieces (the pipeline reads it word by word)."""
    from llm_redact.config import ConfigError

    id2label = getattr(getattr(loaded.model, "config", None), "id2label", None)
    if not isinstance(id2label, Mapping):
        return None
    by_word = aggregation_for(tokenizer) == WORD_AGGREGATION
    try:
        bio = tagging_scheme(str(label) for label in id2label.values()) == BIO
        if bio and by_word:
            return None
        tagset = TagSet.from_bio_labels(id2label) if bio else TagSet.from_labels(id2label)
    except TaggingError as exc:
        raise ConfigError(f"[detection.ner] hf model {model!r}: {exc}") from exc
    try:
        scorer = torch_scorer(loaded.model)
    except ImportError as exc:
        raise ConfigError(
            '[detection.ner] backend = "hf" but torch is not installed;'
            " install the hf extra: uv sync --extra hf"
        ) from exc
    return TaggerPipe(
        loaded.model,
        tokenizer,
        stride,
        tagset,
        # A BIO path is read greedily: the constrained Viterbi decoder and
        # its calibration are for taggers that mark a span's end.
        decoder_for(tagset, None if bio else biases),
        scorer,
        # Whole words where the tokenizer marks them, as for a BIO model.
        by_word=by_word,
        by_text_word=bio,
        every_piece=every_piece,
    )


def build_hf_detector(config: "NerConfig") -> HfDetector:
    from llm_redact.config import ConfigError
    from llm_redact.detection.model_files import SAFETENSORS_FILES, has_files, hf_model_dir

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ConfigError(
            '[detection.ner] backend = "hf" but the hf extra is not installed;'
            " install it: uv sync --extra hf"
        ) from exc
    problem = transformers_problem()
    if problem is not None:
        raise ConfigError(
            f'[detection.ner] backend = "hf" {problem}; upgrade it:'
            " uv sync --extra hf --upgrade-package transformers"
        )
    model_name = config.model or _MODEL_NAME
    entry = _catalog_entry(model_name)
    calibration = entry.viterbi_calibration if entry is not None else None
    # The model's files, local at their pinned revision: configuration,
    # tokenizer and safetensors weights (a pickle only with the hatch), and
    # the calibration file of a BIOES/BILOU tagger the catalog lists one for
    # (required: a listed file that is missing refuses the model, never a
    # silent greedy fallback).
    path = hf_model_dir(
        model_name,
        revision=config.revision_for("hf"),
        allow_download=config.allow_download,
        allow_pickle_weights=config.allow_pickle_weights,
        extra_files=entry.extra_files("hf") if entry is not None else (),
    )
    biases = _tagger_biases(path, model_name, calibration)
    try:
        # Any: transformers' own types are not part of the checked surface.
        # From the local directory only, never with model code, and from
        # safetensors whenever the directory holds them (True refuses any
        # other weights; None, reached only with allow_pickle_weights,
        # lets transformers read pytorch_model.bin — False would skip
        # safetensors altogether).
        loaded: Any = pipeline(
            "token-classification",
            model=str(path),
            tokenizer=str(path),
            aggregation_strategy=TOKEN_AGGREGATION,
            trust_remote_code=False,
            # Every weight in float32, whatever precision the checkpoint
            # stores: transformers' default ("auto") keeps it, and on a CPU
            # bfloat16 is slower (openai/privacy-filter: about 30%).
            dtype=MODEL_DTYPE,
            model_kwargs={"use_safetensors": True if has_files(path, SAFETENSORS_FILES) else None},
        )
    except Exception as exc:  # load can fail many ways; name only what is known
        if importlib.util.find_spec("torch") is None:
            # transformers imports without torch but cannot run a model.
            raise ConfigError(
                '[detection.ner] backend = "hf" but torch is not installed;'
                " install the hf extra: uv sync --extra hf"
            ) from exc
        raise ConfigError(
            f"failed to load Hugging Face token-classification model {model_name!r}:"
            f" {type(exc).__name__}"
        ) from exc
    tokenizer = loaded.tokenizer
    # Only a fast tokenizer reports character offsets (a slow one's
    # entities come back without them and could never be redacted) and
    # lets the pipeline read past its first window.
    if not getattr(tokenizer, "is_fast", False):
        raise ConfigError(
            f"[detection.ner] hf model {model_name!r} has no fast tokenizer;"
            " character offsets are required"
        )
    window = model_window(tokenizer, loaded.model, entry.window if entry is not None else None)
    # The pipeline windows at the tokenizer's limit: make it the model's
    # (a tokenizer that does not know its limit would hand the model the
    # whole text, past its position embeddings).
    if tokenizer.model_max_length != window:
        tokenizer.model_max_length = window
    stride = window // 4
    # A BIOES/BILOU tagger, and a BIO tagger whose tokenizer does not mark
    # word pieces, read the same windows but decode their own spans.
    every_piece = entry is not None and entry.piece_labels == "every"
    pipe: Any = _tagger(loaded, tokenizer, stride, biases, model_name, every_piece=every_piece)
    if pipe is None:
        pipe = _strided_pipeline(pipeline, loaded, tokenizer, stride, model_name)
    detector = HfDetector(
        pipe,
        frozenset(config.entities),
        config.max_chars,
        config.score_threshold,
        policy=LabelPolicy(config.entities, backend="hf", overrides=config.labels),
        windows_of=window_counter(tokenizer, stride),
    )
    detector.model_name = model_name
    return detector


def _strided_pipeline(
    pipeline: Callable[..., Any], loaded: Any, tokenizer: Any, stride: int, model: str
) -> Any:
    """The transformers pipeline over the loaded BIO model, windowed."""
    from llm_redact.config import ConfigError

    try:
        # The same model and tokenizer, read in overlapping windows: `stride`
        # is a construction parameter, so every call reads the whole text.
        # Each entity covers whole words where the tokenizer knows them.
        return pipeline(
            "token-classification",
            model=loaded.model,
            tokenizer=tokenizer,
            aggregation_strategy=aggregation_for(tokenizer),
            stride=stride,
        )
    except Exception as exc:  # transformers refuses the windowing settings
        raise ConfigError(
            f"failed to load Hugging Face token-classification model {model!r}:"
            f" {type(exc).__name__}"
        ) from exc
