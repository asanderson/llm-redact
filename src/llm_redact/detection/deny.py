"""User-configured deny strings: literal values that must always be redacted.

The strongest signal in the pipeline: a deny match is tier 0, so it wins any
overlap against rule/NER detections regardless of span length, is never
subject to [detection.modes], and bypasses the allowlist. Matching is plain
substring (no word boundaries — "Auroras" gets its "Aurora" redacted),
case-insensitive unless the entry says otherwise.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from llm_redact.detection.base import Detection


@dataclass(frozen=True)
class DenyEntry:
    value: str
    case_sensitive: bool = False
    detector_type: str = "DENY"


class DenyDetector:
    """One detector over all configured entries.

    Each entry compiles to its own escaped-literal pattern rather than one
    combined alternation: Python alternation is first-alternative-wins, and
    longest-deny-wins between overlapping entries ("aurora" vs "project
    aurora") needs every entry's independent match stream. Case-insensitive
    entries use re.IGNORECASE, NOT a lowered haystack — str.lower() can
    change string length ('İ' lowers to two characters), which would corrupt
    match offsets.
    """

    name = "deny_strings"

    def __init__(self, entries: Sequence[DenyEntry]) -> None:
        self._compiled = [
            (
                re.compile(re.escape(entry.value), 0 if entry.case_sensitive else re.IGNORECASE),
                entry.detector_type,
            )
            for entry in entries
        ]

    def detect(self, text: str) -> Iterable[Detection]:
        for pattern, detector_type in self._compiled:
            for match in pattern.finditer(text):
                yield Detection(
                    start=match.start(),
                    end=match.end(),
                    detector_type=detector_type,
                    value=match.group(0),
                    priority=0,
                    tier=0,
                )
