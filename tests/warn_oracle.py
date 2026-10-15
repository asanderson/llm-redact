"""The warnings the counting rule predicts for warn-mode values a joined run
drops (``Redactor._join``), worked out without the redactor's own helpers.

A warn-mode value is forwarded as sent and counted as a warning. One a
joined run drops — it overlaps a value the run redacts — is counted only
when part of it is still sent (some character no placeholder replaced in
the end, an access gate's added deny strings and llm findings taken in),
once per value the start-ordered overlap sweep reads among such values. The
winners themselves are counted by ``redact_text`` and ``scan``; shared by
tests/test_joined_values.py and tests/test_overlay_deny.py.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from llm_redact.detection.base import ADDITIVE_TIER, Detection


def dropped_warnings(
    detections: list[Detection],
    winners: list[Detection],
    modes: Mapping[str, str],
    redacted: set[int],
) -> Counter[str]:
    """The warnings predicted for the warn-mode values a joined run drops:
    in a group of ``detections`` (start-sorted, the longest first, then by
    priority) overlapping one another directly or through others that holds
    a joining one, a warn-mode value overlapping a value the configured
    ``winners`` redact (an llm finding aside: it is only cut around the
    others, never joined) is dropped; it is a warning when a character of
    it lies outside ``redacted`` (the offsets finally redacted), and of such
    values overlapping one another only the first the sweep takes counts."""
    joined = [d for d in detections if d.tier != ADDITIVE_TIER]
    group = list(range(len(joined)))

    def root(i: int) -> int:
        while group[i] != i:
            i = group[i]
        return i

    for i, a in enumerate(joined):
        for j in range(i):
            b = joined[j]
            if a.start < b.end and b.start < a.end:
                group[root(i)] = root(j)
    joining = {root(i) for i, d in enumerate(joined) if d.joins}
    fixed = [
        w
        for w in winners
        if w.tier != ADDITIVE_TIER
        and (w.tier == 0 or modes.get(w.detector_type, "redact") == "redact")
    ]
    counted: Counter[str] = Counter()
    end = -1
    for i, d in enumerate(joined):
        if root(i) not in joining or modes.get(d.detector_type) != "warn":
            continue
        if not any(d.start < w.end and w.start < d.end for w in fixed):
            continue  # kept: a winner, or swept out among those kept
        if set(range(d.start, d.end)) <= redacted:
            continue  # sends nothing
        if d.start >= end:
            counted[d.detector_type] += 1
            end = d.end
    return counted
