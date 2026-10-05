"""Optional GLiNER-backed NER detection.

Heavier but more robust than the spaCy backend on unusual names: the gliner
package pulls in torch and transformers (gigabytes installed), so it is a
separate opt-in extra and never part of default installs, CI, or the
container image. Unlike spaCy, GLiNER emits per-entity confidence scores —
this is the backend the reserved ``score_threshold`` config key exists for.

GLiNER reads at most ``max_len`` words of a text (384 for the urchade v2.1
models) and drops the rest with a warning; the subword sequence it hands its
encoder — the prompt block of entity labels first, then the words — is
bounded too. A longer string is therefore read in overlapping windows of
GLiNER's own words (its words splitter: every JSON brace, quote, colon and
comma is a word), each small enough for both limits (windows.word_windows).

The model loads from a local folder at a pinned revision (model_files.py):
a checkpoint without its own tokenizer and encoder configuration gets them
from its pinned base model in a folder assembled once, so no load asks the
Hub for anything.
"""

import importlib.util
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.labels import LabelPolicy, merge_adjacent_parts
from llm_redact.detection.ner import NER_PRIORITY
from llm_redact.detection.stats import NerStats
from llm_redact.detection.windows import drop_exact_duplicates, gliner_words, word_windows

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig

_MODEL_NAME = "urchade/gliner_small-v2.1"
# gliner's GLiNERConfig default: the words of a text a model reads.
_DEFAULT_MAX_LEN = 384
# A window holds at most this many words, and leaves this many of max_len
# beside the prompt block (two items per label, one separator).
_MAX_WINDOW_WORDS = 200
_WORD_MARGIN = 16
# A tokenizer that does not know its model's limit reports a huge sentinel;
# the encoder's position limit applies then (512 when its config is silent).
_SENTINEL_MAX_LENGTH = 1_000_000
_DEFAULT_ENCODER_LIMIT = 512

Words = Callable[[str], list[tuple[int, int]]]


class _ModelLike(Protocol):
    """The sliver of GLiNER's interface the detector uses."""

    def predict_entities(
        self, text: str, labels: list[str], threshold: float
    ) -> list[dict[str, Any]]: ...


def _words_of(model: Any) -> Words:
    """The (start, end) offsets of the words the model's own splitter makes
    of a text (``data_processor.words_splitter``); GLiNER's default
    whitespace splitter when the model does not expose one."""
    splitter = getattr(getattr(model, "data_processor", None), "words_splitter", None)
    if not callable(splitter):
        return gliner_words

    def words(text: str) -> list[tuple[int, int]]:
        return [(int(start), int(end)) for _word, start, end in splitter(text)]

    return words


def _window_words(model: Any, labels: Sequence[str]) -> int:
    """Words a window holds: at most 200, and room for the prompt block
    plus a margin within the model's ``max_len``."""
    max_len = getattr(getattr(model, "config", None), "max_len", None)
    if not isinstance(max_len, int) or max_len <= 0:
        max_len = _DEFAULT_MAX_LEN
    prompt_items = 2 * len(labels) + 1
    return max(1, min(_MAX_WINDOW_WORDS, max_len - prompt_items - _WORD_MARGIN))


def _subword_budget(model: Any, labels: Sequence[str]) -> tuple[Any, int | None]:
    """The model's fast tokenizer and the subword tokens left for a window's
    words: the encoder's limit (the tokenizer's ``model_max_length``, else
    the encoder config's ``max_position_embeddings``, else 512) minus the
    prompt block and the special tokens. (None, None) without a fast
    tokenizer: windows are then bounded by words only."""
    processor = getattr(model, "data_processor", None)
    tokenizer = getattr(processor, "transformer_tokenizer", None)
    if tokenizer is None or not getattr(tokenizer, "is_fast", False):
        return None, None
    limit = getattr(tokenizer, "model_max_length", None)
    if not isinstance(limit, int) or not 0 < limit < _SENTINEL_MAX_LENGTH:
        encoder = getattr(getattr(model, "config", None), "encoder_config", None)
        positions = getattr(encoder, "max_position_embeddings", None)
        limit = (
            positions if isinstance(positions, int) and positions > 0 else _DEFAULT_ENCODER_LIMIT
        )
    prompt: list[str] = []
    ent, sep = getattr(processor, "ent_token", None), getattr(processor, "sep_token", None)
    if isinstance(ent, str) and isinstance(sep, str):
        # How GLiNER prefixes the text: <<ENT>> label ... <<SEP>>.
        for label in labels:
            prompt += [ent, label]
        prompt.append(sep)
    prompt_tokens = len(_token_ids(tokenizer, prompt)) if prompt else 0
    count_specials = getattr(tokenizer, "num_special_tokens_to_add", None)
    specials = count_specials() if callable(count_specials) else 2
    return tokenizer, max(1, limit - prompt_tokens - specials)


def _token_ids(tokenizer: Any, words: list[str]) -> Any:
    return tokenizer(words, is_split_into_words=True, add_special_tokens=False)["input_ids"]


class GlinerDetector:
    name = "gliner"

    # Read by the never-match check (engine.build_detectors): the placeholder
    # types this model can emit (None = unknown, e.g. zero-shot), the model
    # for messages, and the configured entities no active backend emits.
    emittable_types: frozenset[str] | None = None
    model_name: str | None = None
    unmatched_entities: tuple[str, ...] = ()

    def __init__(
        self,
        model: _ModelLike,
        entities: frozenset[str],
        max_chars: int,
        threshold: float,
        *,
        policy: LabelPolicy | None = None,
    ) -> None:
        self._model = model
        # Zero-shot prompts come from the label policy (labels.py): a type
        # request sends a natural-language prompt ("person", "street
        # address"), a raw request its own text; by default the policy is
        # built from `entities` (sorted: a set has no order to keep).
        self.label_policy = (
            policy if policy is not None else LabelPolicy(sorted(entities), backend=self.name)
        )
        self._labels = list(self.label_policy.prompts)
        self._max_chars = max_chars
        self._threshold = threshold
        # Coverage counters (stats.py): strings read whole or in windows or
        # skipped, windows, entities dropped.
        self.stats = NerStats()
        # How a long string is cut: the model's own words, at most
        # `window_words` of them a window, and — with a fast tokenizer —
        # at most `subword_budget` subword tokens of them.
        self._words = _words_of(model)
        self.window_words = _window_words(model, self._labels)
        self._tokenizer, self.subword_budget = _subword_budget(model, self._labels)

    def detect(self, text: str) -> list[Detection]:
        # An entity two overlapping windows both report counts once; parts
        # of one name or address reported separately join into one span
        # (labels.merge_adjacent_parts).
        return merge_adjacent_parts(drop_exact_duplicates(self._found(text)), text)

    def _costs(self, text: str, words: list[tuple[int, int]]) -> list[int]:
        """Each word's size in subword tokens, as the model's tokenizer
        counts it (one call for the whole text); 1 each without one."""
        if self._tokenizer is None or not words:
            return [1] * len(words)
        encoded = self._tokenizer(
            [text[start:end] for start, end in words],
            is_split_into_words=True,
            add_special_tokens=False,
        )
        costs = [0] * len(words)
        for word in encoded.word_ids():
            if word is not None:
                costs[word] += 1
        return costs

    def _found(self, text: str) -> Iterator[Detection]:
        if not self._labels:
            return  # nothing requested: the model is never called
        if len(text) > self._max_chars:
            self.stats.skipped_max_chars += 1
            return
        words = self._words(text)
        windows = list(
            word_windows(self._costs(text, words), self.window_words, self.subword_budget)
        )
        if len(windows) == 1:
            # A text that fits is read whole, as it came.
            self.stats.scanned_whole += 1
            spans = [(0, len(text), windows[0][2])]
        else:
            self.stats.scanned_windowed += 1
            self.stats.windows += len(windows)
            spans = [(words[first][0], words[end - 1][1], over) for first, end, over in windows]
        for start, end, over in spans:
            if over:
                # One word longer than the model reads: it sees only part.
                self.stats.windows_truncated += 1
            yield from self._read(text, start, end)

    def _read(self, text: str, start: int, end: int) -> Iterator[Detection]:
        """The model's entities in ``text[start:end]``, at offsets into
        ``text``."""
        piece = text[start:end]
        # threshold by keyword: GLiNER's third parameter is flat_ner.
        for entity in self._model.predict_entities(piece, self._labels, threshold=self._threshold):
            label = self.label_policy.classify_gliner(str(entity["label"]), self.stats)
            if label is None:
                continue
            found_start, found_end = int(entity["start"]), int(entity["end"])
            if not 0 <= found_start < found_end <= len(piece):
                # A span the text does not contain is never redacted.
                self.stats.offsets_dropped += 1
                continue
            yield Detection(
                start=start + found_start,
                end=start + found_end,
                detector_type=label,
                # The source slice: what the vault maps is exactly what the
                # user sent.
                value=text[start + found_start : start + found_end],
                priority=NER_PRIORITY,
            )


def build_gliner_detector(config: "NerConfig") -> GlinerDetector:
    from llm_redact.config import ConfigError

    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise ConfigError(
            '[detection.ner] backend = "gliner" but the gliner extra is not installed;'
            " install it: uv sync --extra gliner"
        ) from exc
    from llm_redact.detection.model_files import gliner_model_dir

    model_name = config.model or _MODEL_NAME
    # A self-contained local folder at the pinned revision (model_files.py):
    # loading it reads no file from the Hub, the base model's included.
    onnx_file = config.onnx_for("gliner")
    load: dict[str, Any] = {"local_files_only": True, "map_location": "cpu"}
    if onnx_file is not None:
        # [detection.ner.onnx]: ONNX weights through onnxruntime, which the
        # gliner package depends on.
        if importlib.util.find_spec("onnxruntime") is None:
            raise ConfigError(
                "[detection.ner.onnx] gliner needs onnxruntime, which the gliner extra"
                " installs; install it: uv sync --extra gliner"
            )
        load.update(load_onnx_model=True, onnx_model_file=onnx_file)
    folder = gliner_model_dir(
        model_name,
        revision=config.revision_for("gliner"),
        allow_download=config.allow_download,
        onnx_file=onnx_file,
    )
    try:
        model = GLiNER.from_pretrained(str(folder), **load)
    except Exception as exc:  # model load can fail many ways; name only what is known
        raise ConfigError(
            f"failed to load GLiNER model {model_name!r}: {type(exc).__name__}"
        ) from exc
    detector = GlinerDetector(
        model,
        frozenset(config.entities),
        config.max_chars,
        config.score_threshold,
        policy=LabelPolicy(config.entities, backend="gliner", overrides=config.labels),
    )
    detector.model_name = model_name
    return detector
