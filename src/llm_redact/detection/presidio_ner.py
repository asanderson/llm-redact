"""Optional Presidio-backed NER detection (Microsoft's FOSS PII analyzer).

presidio-analyzer layers pattern recognizers, checksum validators, and
context-word scoring on top of a spaCy pipeline, and emits per-entity
confidence scores — like GLiNER, it honors ``score_threshold``. It is a
separate opt-in extra (pulls pydantic and spaCy; extras never touch the
request path, so the no-pydantic rule for body handling is unaffected).

Several Presidio recognizers overlap with the built-in regex rules.
``PRESIDIO_TYPE_MAP`` folds those entity types into the built-in
placeholder names, so «EMAIL_001» means the same thing whichever detector
found the value — the vault is keyed on (session, type, value), and two
names for one type would issue two tokens for the same secret. Unmapped
entity types (PERSON, LOCATION, NRP, ...) pass through as-is.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.ner import NER_PRIORITY

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig

# Presidio entity type -> built-in placeholder type. Only types the regex
# rules also emit are folded; everything else keeps Presidio's name
# (IP_ADDRESS included: it covers v4 and v6, so folding it into either
# built-in name would mislabel the other).
PRESIDIO_TYPE_MAP = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
    "IBAN_CODE": "IBAN",
    "CREDIT_CARD": "CREDIT_CARD",
}


class _AnalyzerLike(Protocol):
    """The sliver of presidio_analyzer.AnalyzerEngine this detector uses."""

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
    ) -> None:
        self._analyzer = analyzer
        self._entities = sorted(entities)
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
            start, end = int(result.start), int(result.end)
            entity_type = str(result.entity_type)
            yield Detection(
                start=start,
                end=end,
                detector_type=PRESIDIO_TYPE_MAP.get(entity_type, entity_type),
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
    )
