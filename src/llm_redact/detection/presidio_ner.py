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

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.labels import LEGACY_FOLDS, LabelPolicy, merge_adjacent_parts
from llm_redact.detection.ner import NER_PRIORITY
from llm_redact.detection.stats import NerStats

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
    # A model runs per string: the detector plan detects it ahead of the
    # redaction, on the NER worker thread (DetectorPlan, ner_prefetch).
    heavy = True

    # Read by the never-match check (engine.build_detectors): the placeholder
    # types this model can emit (None = unknown, e.g. zero-shot), the model
    # for messages, and the configured entities no active backend emits.
    emittable_types: frozenset[str] | None = None
    model_name: str | None = None
    unmatched_entities: tuple[str, ...] = ()

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
        self.label_policy = (
            policy if policy is not None else LabelPolicy(sorted(entities), backend="presidio")
        )
        # Ask only for supported entities the policy keeps: every Presidio
        # entity a requested type folds from (EMAIL -> EMAIL_ADDRESS), raw
        # requests as written. An empty list would make analyze() raise on
        # every request, so it is a startup error instead.
        supported = analyzer.get_supported_entities(language)
        self._entities = sorted(
            {name for name in supported if self.label_policy.classify(name) is not None}
        )
        if not self._entities:
            from llm_redact.config import ConfigError

            raise ConfigError(
                f"[detection.ner] entities {list(self.label_policy.entities)!r} match no entity the"
                f" Presidio analyzer supports for language {language!r}; it supports"
                f" {sorted(supported)!r}"
            )
        # Exactly the types the asked entities classify as.
        self.emittable_types = frozenset(
            t for t in map(self.label_policy.classify, self._entities) if t is not None
        )
        self._max_chars = max_chars
        self._threshold = threshold
        self._language = language
        # Coverage counters (stats.py); the analyzer reads a whole string
        # in one call.
        self.stats = NerStats()

    def detect(self, text: str) -> list[Detection]:
        # Parts of one name or address reported separately join into one
        # span (labels.merge_adjacent_parts).
        return merge_adjacent_parts(self._found(text), text)

    def _found(self, text: str) -> Iterator[Detection]:
        # Latency gate, same as the other NER backends: giant tool results
        # are skipped (and counted); regex rules still cover structured
        # values in them.
        if len(text) > self._max_chars:
            self.stats.skipped_max_chars += 1
            return
        self.stats.scanned_whole += 1
        for result in self._analyzer.analyze(
            text, language=self._language, entities=self._entities, score_threshold=self._threshold
        ):
            label = self.label_policy.classify(str(result.entity_type), self.stats)
            if label is None:
                continue
            start, end = int(result.start), int(result.end)
            if not 0 <= start < end <= len(text):
                self.stats.offsets_dropped += 1
                continue
            yield Detection(
                start=start,
                end=end,
                detector_type=label,
                value=text[start:end],
                priority=NER_PRIORITY,
            )


def offline_suffix_list() -> None:
    """Keep Presidio's email check off the network (AD11).

    Presidio's EmailRecognizer asks tldextract whether an address's domain
    ends in a public suffix, and tldextract's default extractor fetches the
    Public Suffix List from the internet the first time it is asked — on a
    request, from inside the proxy — then caches it under ``~/.cache``. The
    extractor is replaced, process-wide, by one that reads the list snapshot
    the tldextract package ships and writes no cache: no request ever opens a
    connection, and an air-gapped host gets the same answers as a connected
    one. A tldextract laid out otherwise (no ``TLDExtract`` / module-level
    extractor) fails the build instead of leaving the fetch in place.
    """
    import importlib

    from llm_redact.config import ConfigError

    try:
        module: Any = importlib.import_module("tldextract.tldextract")
        module.TLD_EXTRACTOR = module.TLDExtract(
            cache_dir=None, suffix_list_urls=(), fallback_to_snapshot=True
        )
    except Exception as exc:  # absent, or not the layout this relies on
        raise ConfigError(
            "[detection.ner] presidio: cannot keep tldextract (which Presidio's email check"
            " uses) from fetching the public suffix list over the network"
            f" ({type(exc).__name__}); reinstall the presidio extra: uv sync --extra presidio"
        ) from exc


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
    offline_suffix_list()
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
    detector = PresidioDetector(
        analyzer,
        frozenset(config.entities),
        config.max_chars,
        config.score_threshold_for("presidio")[0],
        language=config.language,
        policy=LabelPolicy(config.entities, backend="presidio", overrides=config.labels),
    )
    detector.model_name = model_name
    return detector
