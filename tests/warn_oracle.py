"""The warnings the counting rule predicts for a text, worked out without the
redactor's own helpers.

The rule (docs/detection.md "Per-rule modes"): a text's warnings depend only
on the values found in it and on the characters its fate replaces by
placeholders. Its CANDIDATES are the configured detections the redaction
read (allowlists applied) that are forwarded as sent, or would be: those in
warn or block mode — a block-mode value whether or not an override approves
it — never a deny string (configured; an added one or a union is no
detection the configured detectors make), an llm finding (``ADDITIVE_TIER``)
as found, never the pieces the resolution cuts of it. Where the text goes
upstream as sent (``Redactor.scan`` of text the proxy cannot rewrite, every
value that would refuse it approved) every value found is a candidate and
nothing is replaced. A candidate the replaced characters cover whole sends
nothing and is dropped; the others are read in the resolution's order (by
start, the longest first, then by priority) by a greedy sweep that takes a
value only where it overlaps none taken before it, and each warn-mode value
taken is one warning under its type.

What is replaced is read from an output (``replaced``): ``redact_text``'s —
of a text it refuses, the output of the same text with every refusing value
approved, as a refused text counts what it would with them approved.
``sent_as_is`` decides, from the winners, whether a scan sends its text as
it is. Shared by tests/test_overlay_deny.py, tests/test_joined_values.py and
tests/test_warn_counting.py.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence

from llm_redact.detection.base import Detection
from llm_redact.placeholders import PLACEHOLDER_RE
from llm_redact.vault import InMemoryVault


class Every:
    """An override check approving every value: a refused text's twin is
    redacted under it, and what that twin replaces is what the refused text
    counts against."""

    def allows(self, detector_type: str, value: str) -> bool:
        return True

    def unoverridable(self) -> None:
        return None


def replaced(original: str, output: str, vault: InMemoryVault) -> set[int]:
    """The positions of ``original`` that ``output`` replaced by a
    placeholder (aligned through the vault: the original holds no
    guillemet, so every one in the output starts a placeholder)."""
    positions: set[int] = set()
    cursor = last = 0
    for match in PLACEHOLDER_RE.finditer(output):
        literal = output[last : match.start()]
        assert original.startswith(literal, cursor)
        cursor += len(literal)
        value = vault.original_for(match.group(0))
        assert value is not None and original.startswith(value, cursor)
        positions.update(range(cursor, cursor + len(value)))
        cursor += len(value)
        last = match.end()
    assert original[cursor:] == output[last:]
    return positions


def sent_as_is(
    winners: Sequence[Detection],
    modes: Mapping[str, str],
    approved: Collection[str],
    *,
    redactable: bool,
) -> bool:
    """Whether a scan of text the proxy cannot rewrite (``Redactor.scan``)
    sends it upstream as it is, with ``winners`` the values the resolution
    acts on (an access gate's added deny strings taken in): not
    ``redactable``, and every winner that would refuse it approved (its
    value in ``approved``) — one in block or redact mode, a span joining
    values (``Detection.members``) when each of them is; a deny string
    never, nor a span holding one. A warn-mode value never refuses."""
    if redactable:
        return False
    for d in winners:
        mode = "redact" if d.tier == 0 else modes.get(d.detector_type, "redact")
        if mode == "warn":
            continue
        values = d.members if d.tier == 0 else (d,)
        if not values or not all(v.tier != 0 and v.value in approved for v in values):
            return False
    return True


def expected_warnings(
    found: Sequence[Detection],
    modes: Mapping[str, str],
    redacted: set[int],
    *,
    sent: bool = False,
) -> Counter[str]:
    """The warnings the rule predicts: ``found`` the configured detections
    the redaction read (the configured detectors', allowlists applied, as
    the plan hands them over), ``redacted`` the positions replaced (empty
    where the text is sent as it is), ``sent`` whether it is (then every
    value found is a candidate, in whatever mode)."""
    candidates = [
        d
        for d in found
        if d.tier != 0 and (sent or modes.get(d.detector_type) in ("warn", "block"))
    ]
    candidates.sort(key=lambda d: (d.start, -(d.end - d.start), d.priority))
    counted: Counter[str] = Counter()
    taken: set[int] = set()
    for d in candidates:
        span = set(range(d.start, d.end))
        if span <= redacted:
            continue  # sends nothing, and hides nothing
        if span & taken:
            continue  # overlaps a value taken before it
        taken |= span
        if modes.get(d.detector_type) == "warn":
            counted[d.detector_type] += 1
    return counted
