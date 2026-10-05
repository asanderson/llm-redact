"""Optional NER-based detection (spaCy).

Regex collapses on person names (F1 ~0.13-0.19 on informal text in published
benchmarks); a small NER model recovers most of that at ~1-5 ms per string on
CPU. spaCy's en_core_web_sm was chosen over GLiNER-class models because the
gliner package hard-depends on torch/transformers (gigabytes); spaCy is tens
of megabytes and MIT-licensed, and the ``Detector`` protocol keeps a heavier
backend addable later (a score threshold is reserved for that future backend
— spaCy NER emits no per-entity confidences).

Everything here is import-lazy: this module is only imported when
``[detection.ner] enabled = true``, and the spaCy import happens at proxy
startup (fail fast with an actionable error, no first-request latency spike).
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.labels import LabelPolicy, merge_adjacent_parts

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig

# Above regex rules (structured detectors win equal-length overlap ties).
NER_PRIORITY = 120


class _NlpLike(Protocol):
    """The sliver of spaCy's Language interface the detector uses."""

    def __call__(self, text: str) -> Any: ...


def _pipeline_types(nlp: _NlpLike, policy: LabelPolicy) -> frozenset[str] | None:
    """The placeholder types the pipeline's ``ner`` component can emit
    under ``policy``. None when the pipeline does not say (no ``ner``
    component: an entity ruler may still add entities)."""
    get_pipe = getattr(nlp, "get_pipe", None)
    if get_pipe is None:
        return None
    try:
        labels = get_pipe("ner").labels
    except KeyError:
        return None
    types = (policy.classify(str(label)) for label in labels)
    return frozenset(t for t in types if t is not None)


class NerDetector:
    name = "ner"

    # Read by the never-match check (engine.build_detectors): the placeholder
    # types this model can emit (None = unknown, e.g. zero-shot), the model
    # for messages, and the configured entities no active backend emits.
    emittable_types: frozenset[str] | None = None
    model_name: str | None = None
    unmatched_entities: tuple[str, ...] = ()

    def __init__(
        self,
        nlp: _NlpLike,
        entities: frozenset[str],
        max_chars: int,
        *,
        policy: LabelPolicy | None = None,
    ) -> None:
        self._nlp = nlp
        # spaCy labels (PERSON, ORG, a non-English pipeline's PER) become
        # placeholder types through the label policy (labels.py).
        self.label_policy = policy if policy is not None else LabelPolicy(entities, backend="spacy")
        self.emittable_types = _pipeline_types(nlp, self.label_policy)
        self._max_chars = max_chars

    def detect(self, text: str) -> list[Detection]:
        # Parts of one name or address reported separately join into one
        # span (labels.merge_adjacent_parts).
        return merge_adjacent_parts(self._found(text), text)

    def _found(self, text: str) -> Iterator[Detection]:
        # Latency gate: giant tool results (whole files, logs) are skipped.
        # Regex rules still cover structured values inside them.
        if len(text) > self._max_chars:
            return
        for ent in self._nlp(text).ents:
            label = self.label_policy.classify(str(ent.label_))
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


def build_ner_detector(config: "NerConfig") -> NerDetector:
    from llm_redact.config import ConfigError

    try:
        import spacy
    except ImportError as exc:
        raise ConfigError(
            "[detection.ner] is enabled but spaCy is not installed;"
            " install the extra: uv sync --extra ner"
        ) from exc
    model_name = config.model or "en_core_web_sm"
    try:
        nlp = spacy.load(model_name, disable=["parser", "tagger", "lemmatizer"])
    except OSError as exc:
        raise ConfigError(
            f"spaCy model {model_name} is not available;"
            f" download it: uv run python -m spacy download {model_name}"
        ) from exc
    detector = NerDetector(
        nlp,
        frozenset(config.entities),
        config.max_chars,
        policy=LabelPolicy(config.entities, backend="spacy", overrides=config.labels),
    )
    detector.model_name = model_name
    return detector
