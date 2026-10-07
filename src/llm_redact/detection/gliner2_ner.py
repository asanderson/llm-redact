"""Optional GLiNER2-backed NER detection (`gliner2` extra).

GLiNER2 (Fastino, Apache-2.0) is GLiNER's schema-driven successor: a model
is prompted with the entity types to extract — zero-shot, like GLiNER — and
reports each entity with a confidence (``score_threshold`` applies) and its
character span. The detector uses those spans (``include_spans=True``) and
never searches the text for an entity's surface form, which would mislocate
a value that occurs twice.

GLiNER2 does not truncate a text, but its encoder was trained on a bounded
length (512 positions for the DeBERTa-v3 encoders of Fastino's checkpoints)
and its cost grows with the square of the length. A longer string is read in
overlapping windows of the model's own words (its word splitter: a URL, an
e-mail address or an @handle is one word, and every other non-space
character outside a word is one, so each JSON brace and quote counts), each
holding at most 200 of them and — with the model's fast tokenizer — no more
subword tokens than the encoder reads beside the entity prompt
(windows.word_windows, as for GLiNER).

The model loads from a local folder at a pinned revision
(model_files.gliner2_model_dir): a GLiNER2 checkpoint ships its
configuration, its encoder configuration and its tokenizer, so a load reads
nothing from the Hub. The gliner2 package also holds a client for Fastino's
hosted API; llm-redact never uses it.
"""

import contextlib
import io
import logging
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.labels import LabelPolicy, merge_adjacent_parts
from llm_redact.detection.ner import NER_PRIORITY
from llm_redact.detection.stats import NerStats
from llm_redact.detection.windows import drop_exact_duplicates, word_windows

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig

_MODEL_NAME = "fastino/gliner2-base-v1"
# A window holds at most this many words (GLiNER2 itself sets no word limit
# when it extracts).
_MAX_WINDOW_WORDS = 200
# A tokenizer that does not know its model's limit reports a huge sentinel;
# the encoder's position limit applies then (512 when its config is silent).
_SENTINEL_MAX_LENGTH = 1_000_000
_DEFAULT_ENCODER_LIMIT = 512
# How gliner2 2.0.0 prefixes a text for an entity schema, word by word
# (processor.SchemaTransformer): ( [P] entities ( [E] label ... ) ) [SEP_TEXT].
_SCHEMA_HEAD = ("(", "[P]", "entities", "(")
_ENTITY_MARK = "[E]"
_SCHEMA_TAIL = (")", ")", "[SEP_TEXT]")

# gliner2 2.0.0 WhitespaceTokenSplitter (processing/word_splitter.py), the
# default word splitter: a URL, an e-mail address or an @handle is one word;
# otherwise like GLiNER's (windows.GLINER_WORD_RE).
GLINER2_WORD_RE = re.compile(
    r"(?:https?://[^\s]+|www\.[^\s]+)"
    r"|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
    r"|@[a-z0-9_]+"
    r"|\w+(?:[-_]\w+)*"
    r"|\S",
    re.IGNORECASE,
)

Words = Callable[[str], list[tuple[int, int]]]

# Above every level: no record of the gliner2 package's loggers is emitted.
_SILENT = logging.CRITICAL + 1
# gliner2 2.0.0 appends a "." to a text ending in none of these.
_ENDS_SENTENCE = (".", "!", "?")


class _ModelLike(Protocol):
    """The sliver of GLiNER2's interface the detector uses."""

    def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        *,
        threshold: float,
        include_confidence: bool,
        include_spans: bool,
    ) -> Mapping[str, Any]: ...


def gliner2_words(text: str) -> list[tuple[int, int]]:
    """The (start, end) offsets of the words GLiNER2's default splitter
    makes of ``text``."""
    return [match.span() for match in GLINER2_WORD_RE.finditer(text)]


def _words_of(model: Any) -> Words:
    """The (start, end) offsets of the words the model's own splitter makes
    of a text (``processor.word_splitter``); GLiNER2's default splitter when
    the model does not expose one."""
    splitter = getattr(getattr(model, "processor", None), "word_splitter", None)
    if not callable(splitter):
        return gliner2_words

    def words(text: str) -> list[tuple[int, int]]:
        return [(int(start), int(end)) for _word, start, end in splitter(text, lower=False)]

    return words


def _subword_budget(model: Any, labels: Sequence[str]) -> tuple[Any, int | None]:
    """The model's fast tokenizer and the subword tokens left for a window's
    words: the encoder's limit (the tokenizer's ``model_max_length``, else
    the encoder config's ``max_position_embeddings``, else 512) minus the
    entity prompt and the special tokens. (None, None) without a fast
    tokenizer: windows are then bounded by words only."""
    tokenizer = getattr(getattr(model, "processor", None), "tokenizer", None)
    if tokenizer is None or not getattr(tokenizer, "is_fast", False):
        return None, None
    limit = getattr(tokenizer, "model_max_length", None)
    if not isinstance(limit, int) or not 0 < limit < _SENTINEL_MAX_LENGTH:
        encoder = getattr(getattr(model, "encoder", None), "config", None)
        positions = getattr(encoder, "max_position_embeddings", None)
        limit = (
            positions if isinstance(positions, int) and positions > 0 else _DEFAULT_ENCODER_LIMIT
        )
    prompt = [*_SCHEMA_HEAD]
    for label in labels:
        prompt += [_ENTITY_MARK, label]
    prompt += _SCHEMA_TAIL
    prompt_tokens = len(
        tokenizer(prompt, is_split_into_words=True, add_special_tokens=False)["input_ids"]
    )
    count_specials = getattr(tokenizer, "num_special_tokens_to_add", None)
    specials = count_specials() if callable(count_specials) else 2
    return tokenizer, max(1, limit - prompt_tokens - specials)


class Gliner2Detector:
    name = "gliner2"
    # A model runs per string: the detector plan detects it ahead of the
    # redaction, on the NER worker thread (DetectorPlan, ner_prefetch).
    heavy = True

    # Read by the never-match check (engine.build_detectors): zero-shot, so
    # the types it can emit are unknown (None); the model for messages; the
    # configured entities no active backend emits.
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
        # Zero-shot prompts come from the label policy (labels.py), as for
        # GLiNER: a type request sends a natural-language prompt ("person",
        # "street address"), a raw request its own text.
        self.label_policy = (
            policy if policy is not None else LabelPolicy(sorted(entities), backend=self.name)
        )
        self._labels = list(self.label_policy.prompts)
        self._max_chars = max_chars
        self._threshold = threshold
        # Coverage counters (stats.py).
        self.stats = NerStats()
        # How a long string is cut: the model's own words, at most
        # `window_words` of them a window, and — with a fast tokenizer — at
        # most `subword_budget` subword tokens of them.
        self._words = _words_of(model)
        self.window_words = _MAX_WINDOW_WORDS
        self._tokenizer, self.subword_budget = _subword_budget(model, self._labels)

    def detect(self, text: str) -> list[Detection]:
        # An entity two overlapping windows both report counts once; parts
        # of one name or address reported separately join into one span.
        return merge_adjacent_parts(drop_exact_duplicates(self._found(text)), text)

    def _costs(self, text: str, words: list[tuple[int, int]]) -> list[int]:
        """Each word's size in subword tokens as the model's tokenizer counts
        it — GLiNER2 tokenizes each word lowercased — in one call for the
        whole text; 1 each without a tokenizer."""
        if self._tokenizer is None or not words:
            return [1] * len(words)
        encoded = self._tokenizer(
            [text[start:end].lower() for start, end in words],
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
                # One word longer than the encoder reads: it sees only part.
                self.stats.windows_truncated += 1
            yield from self._read(text, start, end)

    def _read(self, text: str, start: int, end: int) -> Iterator[Detection]:
        """The model's entities in ``text[start:end]``, at offsets into
        ``text``."""
        piece = text[start:end]
        # gliner2 adds a "." to a text that does not end in ".", "!" or "?"
        # and reads its spans on that longer text (processor.py,
        # collate_fn_inference), so a span may take in that one added
        # character: it ends at the piece's end then.
        limit = len(piece) + (0 if piece.endswith(_ENDS_SENTENCE) else 1)
        result = self._model.extract_entities(
            piece,
            self._labels,
            threshold=self._threshold,
            include_confidence=True,
            include_spans=True,
        )
        for label, found in result.get("entities", {}).items():
            for entity in found or ():
                type_name = self.label_policy.classify_gliner(str(label), self.stats)
                if type_name is None:
                    continue
                spanned = isinstance(entity, Mapping)
                found_start = entity.get("start") if spanned else None
                found_end = entity.get("end") if spanned else None
                if isinstance(found_end, int) and len(piece) < found_end <= limit:
                    found_end = len(piece)
                    while found_end > 0 and piece[found_end - 1].isspace():
                        found_end -= 1  # the blank before the added "."
                if (
                    not isinstance(found_start, int)
                    or not isinstance(found_end, int)
                    or not 0 <= found_start < found_end <= len(piece)
                ):
                    # A span the text does not contain is never redacted.
                    self.stats.offsets_dropped += 1
                    continue
                yield Detection(
                    start=start + found_start,
                    end=start + found_end,
                    detector_type=type_name,
                    # The source slice: what the vault maps is exactly what
                    # the user sent.
                    value=text[start + found_start : start + found_end],
                    priority=NER_PRIORITY,
                )


_EXTRA_MISSING = (
    '[detection.ner] backend = "gliner2" but the gliner2 extra is not installed;'
    " install it: uv sync --extra gliner2"
)


def build_gliner2_detector(config: "NerConfig") -> Gliner2Detector:
    from llm_redact.config import ConfigError
    from llm_redact.detection.model_files import catalog_entry, gliner2_model_dir

    # gliner2 logs words of the text it reads (processor.py: a word that
    # makes no subword is named at WARNING, a failed extraction's traceback
    # at ERROR); logs never carry request content.
    logging.getLogger("gliner2").setLevel(_SILENT)
    try:
        from gliner2 import AutoExtractor
    except ImportError as exc:  # gliner2 itself, or torch/transformers below it
        raise ConfigError(_EXTRA_MISSING) from exc
    model_name = config.model or _MODEL_NAME
    # A self-contained local folder at the pinned revision (model_files.py).
    folder = gliner2_model_dir(
        model_name,
        revision=config.revision_for("gliner2"),
        allow_download=config.allow_download,
    )
    entry = catalog_entry(model_name, "gliner2")
    try:
        # gliner2 prints its model configuration to stdout while it builds
        # a model; stdout belongs to the command (`--json` output).
        with contextlib.redirect_stdout(io.StringIO()):
            model = AutoExtractor.from_pretrained(
                str(folder), local_files_only=True, map_location="cpu"
            )
    except ImportError as exc:
        # A module gliner2 imports only when it loads a model (peft, in its
        # extraction runtime) is part of the extra too.
        raise ConfigError(_EXTRA_MISSING) from exc
    except Exception as exc:  # model load can fail many ways; name only what is known
        raise ConfigError(
            f"failed to load GLiNER2 model {model_name!r}: {type(exc).__name__}"
        ) from exc
    detector = Gliner2Detector(
        model,
        frozenset(config.entities),
        config.max_chars,
        config.score_threshold_for("gliner2")[0],
        policy=LabelPolicy(
            config.entities,
            backend="gliner2",
            overrides=config.labels,
            # The prompts the model was trained on, where the catalog knows them.
            prompts=entry.prompts if entry is not None else (),
        ),
    )
    detector.model_name = model_name
    return detector
