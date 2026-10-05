"""Reading long strings with fixed-window NER models.

Encoder models read a bounded number of tokens; past it they truncate,
silently. The ``hf`` backend therefore reads a long string in overlapping
windows (the token-classification pipeline's own ``stride``), so a value
anywhere in the string can be found. An entity inside an overlap can be
reported by both windows: :func:`drop_exact_duplicates` keeps one copy of
each (start, end, type), and partial overlaps (a name cut at one window's
edge, whole in the next) are left to the redactor's overlap resolution,
where the longest span wins.

Nothing here sees more than offsets and types; nothing logs.
"""

from collections.abc import Iterable

from llm_redact.detection.base import Detection


def drop_exact_duplicates(detections: Iterable[Detection]) -> list[Detection]:
    """``detections`` without repeats of one (start, end, type): what two
    overlapping windows both report. First occurrence kept, order kept."""
    seen: set[tuple[int, int, str]] = set()
    kept: list[Detection] = []
    for detection in detections:
        key = (detection.start, detection.end, detection.detector_type)
        if key not in seen:
            seen.add(key)
            kept.append(detection)
    return kept
