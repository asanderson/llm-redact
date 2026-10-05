"""Optional Stanford Stanza NER backend (`stanza` extra).

Stanza ships accurate neural NER models for 60+ languages, which makes it the
multilingual complement to the English-first spaCy default. Like spaCy it
emits no per-entity confidence, so `score_threshold` does not apply. Heavier
than spaCy (pulls in torch), so it is a separate opt-in extra, never in the
default install or the container image.

Import-lazy: this module loads only when a `stanza` backend is enabled, and
the model load happens at proxy startup (fail fast, no first-request spike).
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.labels import LabelPolicy, merge_adjacent_parts
from llm_redact.detection.ner import NER_PRIORITY

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig


class _PipelineLike(Protocol):
    """The sliver of stanza's Pipeline interface the detector uses."""

    def __call__(self, text: str) -> Any: ...


class StanzaDetector:
    name = "stanza"

    # Read by the never-match check (engine.build_detectors): the placeholder
    # types this model can emit (None = unknown, e.g. zero-shot), the model
    # for messages, and the configured entities no active backend emits.
    emittable_types: frozenset[str] | None = None
    model_name: str | None = None
    unmatched_entities: tuple[str, ...] = ()

    def __init__(
        self,
        nlp: _PipelineLike,
        entities: frozenset[str],
        max_chars: int,
        *,
        policy: LabelPolicy | None = None,
    ) -> None:
        self._nlp = nlp
        # Stanza's labels (PER in most languages, PERSON in English) become
        # placeholder types through the label policy (labels.py).
        self.label_policy = (
            policy if policy is not None else LabelPolicy(entities, backend="stanza")
        )
        self._max_chars = max_chars

    def detect(self, text: str) -> list[Detection]:
        # Parts of one name or address reported separately join into one
        # span (labels.merge_adjacent_parts).
        return merge_adjacent_parts(self._found(text), text)

    def _found(self, text: str) -> Iterator[Detection]:
        if len(text) > self._max_chars:
            return
        for ent in self._nlp(text).ents:
            label = self.label_policy.classify(str(ent.type))
            start, end = int(ent.start_char), int(ent.end_char)
            if label is None or not 0 <= start < end <= len(text):
                continue
            yield Detection(
                start=start,
                end=end,
                detector_type=label,
                value=text[start:end],
                priority=NER_PRIORITY,
            )


def build_stanza_detector(config: "NerConfig") -> StanzaDetector:
    from llm_redact.config import ConfigError

    try:
        import stanza
    except ImportError as exc:
        raise ConfigError(
            '[detection.ner] backend = "stanza" but the stanza extra is not installed;'
            " install it: uv sync --extra stanza"
        ) from exc
    language = config.language or "en"
    try:
        # download_method=None: never fetch at request/startup time — a missing
        # model is an actionable config error, not a silent multi-GB download.
        nlp = stanza.Pipeline(
            lang=language,
            processors="tokenize,ner",
            download_method=None,
            verbose=False,
        )
    except Exception as exc:  # model load can fail many ways (absent, corrupt)
        raise ConfigError(
            f"Stanza {language!r} NER model is not available; download it:"
            f" uv run python -c \"import stanza; stanza.download('{language}')\""
        ) from exc
    detector = StanzaDetector(
        nlp,
        frozenset(config.entities),
        config.max_chars,
        policy=LabelPolicy(config.entities, backend="stanza", overrides=config.labels),
    )
    detector.model_name = f"stanza {language}"
    return detector
