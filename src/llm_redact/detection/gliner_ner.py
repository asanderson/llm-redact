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
from llm_redact.detection.labels import LabelPolicy
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
        self._policy = (
            policy if policy is not None else LabelPolicy(sorted(entities), backend=self.name)
        )
        self._labels = list(self._policy.prompts)
        self._max_chars = max_chars
        self._threshold = threshold

    def detect(self, text: str) -> Iterable[Detection]:
        if len(text) > self._max_chars or not self._labels:
            return
        for entity in self._model.predict_entities(text, self._labels, self._threshold):
            label = self._policy.classify_gliner(str(entity["label"]))
            if label is None:
                continue
            start, end = int(entity["start"]), int(entity["end"])
            if not 0 <= start < end <= len(text):
                continue  # a span the text does not contain is never redacted
            yield Detection(
                start=start,
                end=end,
                detector_type=label,
                # The source slice: what the vault maps is exactly what the
                # user sent.
                value=text[start:end],
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
        model,
        frozenset(config.entities),
        config.max_chars,
        config.score_threshold,
        policy=LabelPolicy(config.entities, backend="gliner", overrides=config.labels),
    )
