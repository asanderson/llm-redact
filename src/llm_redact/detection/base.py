from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    start: int
    end: int
    detector_type: str
    value: str
    # Lower sorts first when detections start at the same offset with the
    # same length; lets specific rules (sk-ant-) beat generic ones (sk-).
    priority: int = 100
    # 0 = user deny strings: wins ANY overlap regardless of span length or
    # start position, always redacts, bypasses the allowlist. 1 = everything
    # else (rules, NER) — the pre-tier behavior, byte for byte.
    tier: int = 1


class Detector(Protocol):
    """Finds values in one string. A detector that runs a model may also
    say ``heavy = True`` (an optional attribute, read with getattr: every
    NER backend does): the detector plan then lets it be detected ahead of
    the redaction, off the event loop (engine.DetectorPlan, ner_prefetch),
    and runs it under one lock per model."""

    name: str

    def detect(self, text: str) -> Iterable[Detection]: ...
