"""Optional Stanford Stanza NER backend (`stanza` extra).

Stanza ships accurate neural NER models for 60+ languages, which makes it the
multilingual complement to the English-first spaCy default. Like spaCy it
emits no per-entity confidence, so `score_threshold` does not apply. Heavier
than spaCy (pulls in torch), so it is a separate opt-in extra, never in the
default install or the container image.

Import-lazy: this module loads only when a `stanza` backend is enabled, and
the model load happens at proxy startup (fail fast, no first-request spike).
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from llm_redact.detection.base import Detection
from llm_redact.detection.ner import NER_PRIORITY

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig


class _PipelineLike(Protocol):
    """The sliver of stanza's Pipeline interface the detector uses."""

    def __call__(self, text: str) -> Any: ...


class StanzaDetector:
    name = "stanza"

    def __init__(self, nlp: _PipelineLike, entities: frozenset[str], max_chars: int) -> None:
        self._nlp = nlp
        self._entities = entities
        self._max_chars = max_chars

    def detect(self, text: str) -> Iterable[Detection]:
        if len(text) > self._max_chars:
            return
        for ent in self._nlp(text).ents:
            if ent.type in self._entities:
                yield Detection(
                    start=int(ent.start_char),
                    end=int(ent.end_char),
                    detector_type=str(ent.type),
                    value=str(ent.text),
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
    return StanzaDetector(nlp, frozenset(config.entities), config.max_chars)
