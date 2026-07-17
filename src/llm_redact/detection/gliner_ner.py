"""Optional GLiNER-backed NER detection.

Heavier but more robust than the spaCy backend on unusual names: the gliner
package pulls in torch and transformers (gigabytes installed), so it is a
separate opt-in extra and never part of default installs, CI, or the
container image. Unlike spaCy, GLiNER emits per-entity confidence scores —
this is the backend the reserved ``score_threshold`` config key exists for.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.ner import NER_PRIORITY

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig

_MODEL_NAME = "urchade/gliner_small-v2.1"


class _ModelLike(Protocol):
    """The sliver of GLiNER's interface the detector uses."""

    def predict_entities(
        self, text: str, labels: list[str], threshold: float
    ) -> list[dict[str, Any]]: ...


class GlinerDetector:
    name = "gliner"

    def __init__(
        self, model: _ModelLike, entities: frozenset[str], max_chars: int, threshold: float
    ) -> None:
        self._model = model
        self._entities = sorted(entities)
        self._max_chars = max_chars
        self._threshold = threshold

    def detect(self, text: str) -> Iterable[Detection]:
        if len(text) > self._max_chars:
            return
        for entity in self._model.predict_entities(text, self._entities, self._threshold):
            yield Detection(
                start=int(entity["start"]),
                end=int(entity["end"]),
                detector_type=str(entity["label"]).upper().replace(" ", "_"),
                value=str(entity["text"]),
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
    model_name = config.model or _MODEL_NAME
    try:
        model = GLiNER.from_pretrained(model_name)
    except Exception as exc:  # model download/load can fail many ways
        raise ConfigError(
            f"failed to load GLiNER model {model_name!r}; check network access and disk space"
        ) from exc
    return GlinerDetector(
        model, frozenset(config.entities), config.max_chars, config.score_threshold
    )
