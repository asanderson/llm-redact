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
by its first piece, which would lose names token-level aggregation finds, so
those models keep ``"simple"``.

Long strings are read whole, in overlapping token windows: without `stride`
the pipeline truncates at the tokenizer's maximum length and never reads the
rest. `stride` needs a fast tokenizer — which also reports the character
offsets every detection needs — so a model without one is refused at startup.

A model tagging BIOES or BILOU (its labels carry `E-`/`S-` or `L-`/`U-` tags,
which the pipeline's aggregation does not understand) runs without the
pipeline: :class:`TaggerPipe` reads the same token windows, takes the model's
per-token log-probabilities and decodes its spans itself (tagging.py) — with
the constrained Viterbi decoder when the model catalog lists the model's
calibration file, else greedily.

Import-lazy: loads only when an `hf` backend is enabled; the model load
happens at proxy startup (fail fast, no first-request latency spike). The
files come from a local directory at a pinned revision (model_files.py):
safetensors weights unless `allow_pickle_weights` is set, and never code
from the model's repository (`trust_remote_code=False`).
"""

import importlib.util
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.labels import LabelPolicy, merge_adjacent_parts
from llm_redact.detection.ner import NER_PRIORITY
from llm_redact.detection.stats import NerStats
from llm_redact.detection.tagging import (
    BIO,
    Decoder,
    TaggingError,
    TagSet,
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
    makes before it trusts its word boundaries — else ``"simple"``.

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
            ids = torch.tensor([input_ids], device=getattr(model, "device", None))
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


class TaggerPipe:
    """A BIOES/BILOU token-classification model read the way the strided
    pipeline reads a BIO one: the fast tokenizer's overlapping windows
    (``model_max_length`` tokens, ``stride`` shared), and per window one
    model call whose label scores ``decode`` turns into a label path
    (tagging.decoder_for). With ``by_word`` — a tokenizer that marks word
    pieces (:func:`aggregation_for`), whose models are trained to label a
    word on its first piece — the path runs over the window's words, each
    scored by its first piece as the pipeline's word-level aggregation
    reads a BIO model, so a span always covers whole words; otherwise over
    its tokens. Each span it marks is reported like a pipeline entity — its
    entity label (the tag dropped), the mean probability of its units'
    labels, and character offsets into the whole text, without the
    whitespace at either edge. Special tokens and tokens covering no
    character are not decoded."""

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
    ) -> None:
        # Read by HfDetector: the model's config.id2label names what it emits.
        self.model = model
        self._tokenizer = tokenizer
        self._stride = stride
        self._tagset = tagset
        self._decode = decode
        self._scorer = scorer
        self._by_word = by_word

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
        units = _word_units(encoded, windows) if self._by_word else _token_units(windows)
        entities: list[dict[str, Any]] = []
        for (ids, _offsets, _special), kept in zip(windows, units, strict=True):
            rows = self._scorer(list(ids))
            scores = [rows[row] for row, _start, _end in kept]
            path = self._decode(scores)
            for first, last, label in spans_of(path, self._tagset):
                start, end = kept[first][1], kept[last][2]
                # A token's offsets may take in the blank before a word.
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                if start == end:
                    continue
                probabilities = [math.exp(scores[t][path[t]]) for t in range(first, last + 1)]
                entities.append(
                    {
                        "entity_group": label,
                        "score": sum(probabilities) / len(probabilities),
                        "start": start,
                        "end": end,
                    }
                )
        return entities


def _catalog_entry(model: str) -> "CatalogEntry | None":
    """The model catalog's ``hf`` entry for ``model`` (a Hub id, or a local
    directory its sidecar file identifies); None when there is none."""
    from llm_redact.config import ConfigError
    from llm_redact.detection.model_catalog import SidecarError, identify, lookup

    try:
        identity = identify(model)
    except SidecarError as exc:
        raise ConfigError(f"[detection.ner] hf model: {exc}") from exc
    entry = lookup(identity.model_id) if identity is not None else None
    return entry if entry is not None and "hf" in entry.backends else None


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
    loaded: Any, tokenizer: Any, stride: int, biases: dict[str, float] | None, model: str
) -> TaggerPipe | None:
    """A :class:`TaggerPipe` for a model whose labels tag BIOES or BILOU;
    None for a BIO model (the pipeline reads it)."""
    from llm_redact.config import ConfigError

    id2label = getattr(getattr(loaded.model, "config", None), "id2label", None)
    if not isinstance(id2label, Mapping):
        return None
    try:
        if tagging_scheme(str(label) for label in id2label.values()) == BIO:
            return None
        tagset = TagSet.from_labels(id2label)
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
        decoder_for(tagset, biases),
        scorer,
        # Whole words where the tokenizer marks them, as for a BIO model.
        by_word=aggregation_for(tokenizer) == WORD_AGGREGATION,
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
    # A BIOES/BILOU tagger reads the same windows but decodes its own spans.
    pipe: Any = _tagger(loaded, tokenizer, stride, biases, model_name)
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
