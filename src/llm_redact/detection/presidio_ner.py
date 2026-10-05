"""Optional Presidio-backed NER detection (Microsoft's FOSS PII analyzer).

presidio-analyzer layers pattern recognizers, checksum validators, and
context-word scoring on top of a spaCy pipeline, and emits per-entity
confidence scores — like GLiNER, it honors ``score_threshold``. It is a
separate opt-in extra (pulls pydantic and spaCy; extras never touch the
request path, so the no-pydantic rule for body handling is unaffected).

Several Presidio recognizers overlap with the built-in regex rules. Their
entity types become placeholder types through the shared label policy
(labels.py), so «EMAIL_001» means the same thing whichever detector found
the value — the vault is keyed on (session, type, value), and two names for
one type would issue two tokens for the same secret. A type request is
translated into the Presidio entities that fold into it (``EMAIL`` asks for
``EMAIL_ADDRESS``); the analyzer is asked only for entities it supports,
since it raises on every request for an unsupported-only list.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.labels import LEGACY_FOLDS, LabelPolicy
from llm_redact.detection.ner import NER_PRIORITY

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig

# Presidio entity type -> built-in placeholder type: the five folds this
# backend has always applied (labels.LEGACY_FOLDS, a subset of the policy's
# default folds). IP_ADDRESS stays unfolded: it covers v4 and v6, so either
# built-in name would mislabel the other.
PRESIDIO_TYPE_MAP = dict(LEGACY_FOLDS)


class _AnalyzerLike(Protocol):
    """The sliver of presidio_analyzer.AnalyzerEngine this detector uses."""

    def get_supported_entities(self, language: str | None = ...) -> list[str]: ...

    def analyze(
        self,
        text: str,
        language: str,
        entities: list[str] | None = ...,
        score_threshold: float = ...,
    ) -> list[Any]: ...


class PresidioDetector:
    name = "presidio"

    def __init__(
        self,
        analyzer: _AnalyzerLike,
        entities: frozenset[str],
        max_chars: int,
        threshold: float,
        language: str = "en",
        *,
        policy: LabelPolicy | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._policy = (
            policy if policy is not None else LabelPolicy(sorted(entities), backend="presidio")
        )
        # Ask only for supported entities the policy keeps: every Presidio
        # entity a requested type folds from (EMAIL -> EMAIL_ADDRESS), raw
        # requests as written. An empty list would make analyze() raise on
        # every request, so it is a startup error instead.
        supported = analyzer.get_supported_entities(language)
        self._entities = sorted(
            {name for name in supported if self._policy.classify(name) is not None}
        )
        if not self._entities:
            from llm_redact.config import ConfigError

            raise ConfigError(
                f"[detection.ner] entities {list(self._policy.entities)!r} match no entity the"
                f" Presidio analyzer supports for language {language!r}; it supports"
                f" {sorted(supported)!r}"
            )
        self._max_chars = max_chars
        self._threshold = threshold
        self._language = language

    def detect(self, text: str) -> Iterable[Detection]:
        # Latency gate, same as the other NER backends: giant tool results
        # are skipped; regex rules still cover structured values in them.
        if len(text) > self._max_chars:
            return
        for result in self._analyzer.analyze(
            text, language=self._language, entities=self._entities, score_threshold=self._threshold
        ):
            label = self._policy.classify(str(result.entity_type))
            start, end = int(result.start), int(result.end)
            if label is None or not 0 <= start < end <= len(text):
                continue
            yield Detection(
                start=start,
                end=end,
                detector_type=label,
                value=text[start:end],
                priority=NER_PRIORITY,
            )


def build_presidio_detector(config: "NerConfig") -> PresidioDetector:
    from llm_redact.config import ConfigError

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:
        raise ConfigError(
            '[detection.ner] backend = "presidio" but the presidio extra is not installed;'
            " install it: uv sync --extra presidio"
        ) from exc
    try:
        # Pin the same small model the spacy backend uses instead of
        # Presidio's en_core_web_lg default: tens of MB, and one download
        # serves both backends.
        model_name = config.model or "en_core_web_sm"
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": config.language, "model_name": model_name}],
            }
        )
        analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(), supported_languages=[config.language]
        )
    except Exception as exc:  # model load / engine build can fail many ways
        raise ConfigError(
            "failed to build the Presidio analyzer; is the spaCy model available?"
            " download it: uv run python -m spacy download en_core_web_sm"
        ) from exc
    return PresidioDetector(
        analyzer,
        frozenset(config.entities),
        config.max_chars,
        config.score_threshold,
        language=config.language,
        policy=LabelPolicy(config.entities, backend="presidio", overrides=config.labels),
    )
