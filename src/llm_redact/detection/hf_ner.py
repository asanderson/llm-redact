"""Optional Hugging Face token-classification NER backend (`hf` extra).

Any `token-classification` model on the Hub (multilingual XLM-R NER, biomedical
NER, domain-tuned checkpoints, …) becomes a detector, which is the escape
hatch for teams that already have a fine-tuned model. Uses the `transformers`
pipeline with `aggregation_strategy="simple"` so sub-word tokens are merged
into whole entity spans with a confidence, which `score_threshold` gates.

Import-lazy: loads only when an `hf` backend is enabled; the model load
happens at proxy startup (fail fast, no first-request latency spike).
"""

import importlib.util
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.labels import LabelPolicy
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
        *,
        policy: LabelPolicy | None = None,
    ) -> None:
        self._pipe = pipe
        # The label policy (labels.py) turns the model's labels (`PER`,
        # `B-PER` without aggregation) into placeholder types; by default
        # it is built from `entities`.
        self._policy = policy if policy is not None else LabelPolicy(entities, backend=self.name)
        self._max_chars = max_chars
        self._threshold = threshold

    def detect(self, text: str) -> Iterable[Detection]:
        if len(text) > self._max_chars:
            return
        for ent in self._pipe(text):
            label = self._policy.classify(str(ent.get("entity_group", ent.get("entity", ""))))
            if label is None:
                continue
            if float(ent.get("score", 1.0)) < self._threshold:
                continue
            start, end = ent.get("start"), ent.get("end")
            if start is None or end is None:
                continue
            start, end = int(start), int(end)
            if not 0 <= start < end <= len(text):
                # A span the text does not contain cannot be redacted (or
                # restored) faithfully: skip it rather than guess.
                continue
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
    return HfDetector(
        pipe,
        frozenset(config.entities),
        config.max_chars,
        config.score_threshold,
        policy=LabelPolicy(config.entities, backend="hf"),
    )
