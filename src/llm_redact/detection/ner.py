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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig

# Above regex rules (structured detectors win equal-length overlap ties).
NER_PRIORITY = 120


class _NlpLike(Protocol):
    """The sliver of spaCy's Language interface the detector uses."""

    def __call__(self, text: str) -> Any: ...


class NerDetector:
    name = "ner"

    def __init__(self, nlp: _NlpLike, entities: frozenset[str], max_chars: int) -> None:
        self._nlp = nlp
        self._entities = entities
        self._max_chars = max_chars

    def detect(self, text: str) -> Iterable[Detection]:
        # Latency gate: giant tool results (whole files, logs) are skipped.
        # Regex rules still cover structured values inside them.
        if len(text) > self._max_chars:
            return
        for ent in self._nlp(text).ents:
            if ent.label_ in self._entities:
                yield Detection(
                    start=ent.start_char,
                    end=ent.end_char,
                    detector_type=ent.label_,
                    value=ent.text,
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
    return NerDetector(nlp, frozenset(config.entities), config.max_chars)
