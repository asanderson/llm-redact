"""Optional Hugging Face token-classification NER backend (`hf` extra).

Any `token-classification` model on the Hub (multilingual XLM-R NER, biomedical
NER, domain-tuned checkpoints, …) becomes a detector, which is the escape
hatch for teams that already have a fine-tuned model. Uses the `transformers`
pipeline with `aggregation_strategy="simple"` so sub-word tokens are merged
into whole entity spans with a confidence, which `score_threshold` gates.

Import-lazy: loads only when an `hf` backend is enabled; the model load
happens at proxy startup (fail fast, no first-request latency spike).
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.ner import NER_PRIORITY

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig

_MODEL_NAME = "dslim/bert-base-NER"


class _PipelineLike(Protocol):
    """The sliver of the transformers token-classification pipeline used."""

    def __call__(self, text: str) -> list[dict[str, Any]]: ...


class HfDetector:
    name = "hf"

    def __init__(
        self,
        pipe: _PipelineLike,
        entities: frozenset[str],
        max_chars: int,
        threshold: float,
    ) -> None:
        self._pipe = pipe
        self._entities = entities
        self._max_chars = max_chars
        self._threshold = threshold

    def detect(self, text: str) -> Iterable[Detection]:
        if len(text) > self._max_chars:
            return
        for ent in self._pipe(text):
            label = str(ent.get("entity_group", ent.get("entity", ""))).upper()
            if label not in self._entities:
                continue
            if float(ent.get("score", 1.0)) < self._threshold:
                continue
            start, end = ent.get("start"), ent.get("end")
            if start is None or end is None:
                continue
            yield Detection(
                start=int(start),
                end=int(end),
                detector_type=label,
                value=str(ent.get("word", text[int(start) : int(end)])),
                priority=NER_PRIORITY,
            )


def build_hf_detector(config: "NerConfig") -> HfDetector:
    from llm_redact.config import ConfigError

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ConfigError(
            '[detection.ner] backend = "hf" but the hf extra is not installed;'
            " install it: uv sync --extra hf"
        ) from exc
    model_name = config.model or _MODEL_NAME
    try:
        pipe = pipeline("token-classification", model=model_name, aggregation_strategy="simple")
    except Exception as exc:  # download/load can fail many ways
        raise ConfigError(
            f"failed to load Hugging Face token-classification model {model_name!r};"
            " check network access and disk space"
        ) from exc
    return HfDetector(pipe, frozenset(config.entities), config.max_chars, config.score_threshold)
