from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

# Detection.tier of a detection that may only ADD to what the other
# detectors find (the llm NER backend's: a model the scanned text can
# instruct). Overlap resolution never lets one displace another detection
# (redactor._resolve_overlaps): the others are resolved without them, and
# each additive detection keeps only the part of its span no winner holds.
ADDITIVE_TIER = 2


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
    # else (rules, NER) — the pre-tier behavior, byte for byte. 2
    # (ADDITIVE_TIER) = additive only: never displaces a tier-0 or tier-1
    # detection, cut around the ones it overlaps instead.
    tier: int = 1
    # A value read where an earlier release read none, right next to text
    # another value may hold (a card read inside a longer run of digit
    # groups, an IBAN in the paper format, a value ending before a run of
    # dots since 1.12.1): it must not take another value's characters, so
    # the redactor JOINS it with every value it overlaps, directly or through
    # others, instead of letting the overlap sweep hand them to one
    # (Redactor._join): what any of them redacts is redacted.
    joins: bool = False
    # The values a span joining overlapping ones stands for (a tier-0 union
    # typed after the first, redactor._joined), in start order; empty for
    # every other detection. Where the proxy cannot rewrite the text
    # (Redactor.scan) each is put to the requester's overrides on its own.
    # Left out of equality, hashing and repr, which stay the span's.
    members: "tuple[Detection, ...]" = field(default=(), compare=False, repr=False)


class DetectorUnavailable(Exception):
    """A detector could not read a string, and its configuration says no
    request may go out without it (the llm NER backend's ``on_error =
    "block"``: its model server timed out, failed or answered something
    unreadable). The request is refused — 503 over HTTP, a 1011 close on a
    realtime connection — before any upstream contact. Carries the backend
    name and a fixed failure kind only, never the text it was reading."""

    def __init__(self, backend: str, reason: str) -> None:
        super().__init__(f"the {backend} NER backend could not read a string ({reason})")
        self.backend = backend
        self.reason = reason


class Detector(Protocol):
    """Finds values in one string. A detector that runs a model may also
    say ``heavy = True`` (an optional attribute, read with getattr: every
    NER backend does): the detector plan then lets it be detected ahead of
    the redaction, off the event loop (engine.DetectorPlan, ner_prefetch),
    and runs it under one lock per model."""

    name: str

    def detect(self, text: str) -> Iterable[Detection]: ...
