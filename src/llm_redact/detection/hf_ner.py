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

Import-lazy: loads only when an `hf` backend is enabled; the model load
happens at proxy startup (fail fast, no first-request latency spike). The
files come from a local directory at a pinned revision (model_files.py):
safetensors weights unless `allow_pickle_weights` is set, and never code
from the model's repository (`trust_remote_code=False`).
"""

import importlib.util
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.labels import LabelPolicy, merge_adjacent_parts
from llm_redact.detection.ner import NER_PRIORITY
from llm_redact.detection.stats import NerStats
from llm_redact.detection.windows import drop_exact_duplicates

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig

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


def catalog_window(model: str) -> int | None:
    """The token window the model catalog records for ``model`` (a Hub id,
    or a local directory its sidecar file identifies) on the ``hf``
    backend; None when it records none."""
    from llm_redact.config import ConfigError
    from llm_redact.detection.model_catalog import SidecarError, identify, lookup

    try:
        identity = identify(model)
    except SidecarError as exc:
        raise ConfigError(f"[detection.ner] hf model: {exc}") from exc
    entry = lookup(identity.model_id) if identity is not None else None
    return entry.window if entry is not None and "hf" in entry.backends else None


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
    # The model's files, local at their pinned revision: configuration,
    # tokenizer and safetensors weights (a pickle only with the hatch).
    path = hf_model_dir(
        model_name,
        revision=config.revision_for("hf"),
        allow_download=config.allow_download,
        allow_pickle_weights=config.allow_pickle_weights,
    )
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
    window = model_window(tokenizer, loaded.model, catalog_window(model_name))
    # The pipeline windows at the tokenizer's limit: make it the model's
    # (a tokenizer that does not know its limit would hand the model the
    # whole text, past its position embeddings).
    if tokenizer.model_max_length != window:
        tokenizer.model_max_length = window
    stride = window // 4
    try:
        # The same model and tokenizer, read in overlapping windows: `stride`
        # is a construction parameter, so every call reads the whole text.
        # Each entity covers whole words where the tokenizer knows them.
        pipe: Any = pipeline(
            "token-classification",
            model=loaded.model,
            tokenizer=tokenizer,
            aggregation_strategy=aggregation_for(tokenizer),
            stride=stride,
        )
    except Exception as exc:  # transformers refuses the windowing settings
        raise ConfigError(
            f"failed to load Hugging Face token-classification model {model_name!r}:"
            f" {type(exc).__name__}"
        ) from exc
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
