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
value that would refuse it approved) nothing is replaced, and the values the
overrides let through are candidates too (``let_through``). A candidate the
replaced characters cover whole sends nothing and is dropped. The others are
read as overlap resolution reads values, by a greedy sweep in the
resolution's order (by start, the longest first, then by priority) that
takes a value only where it overlaps none taken before it: every detector's
values but the llm backend's first, then the llm findings, each read only
where a character of it lies outside what is replaced and every value taken
before (an llm finding never hides another detector's value). Each
warn-mode value taken is one warning under its type.

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

from llm_redact.detection.base import ADDITIVE_TIER, Detection
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


def _order(d: Detection) -> tuple[int, int, int]:
    return (d.start, -(d.end - d.start), d.priority)


def let_through(
    found: Sequence[Detection], winners: Sequence[Detection], modes: Mapping[str, str]
) -> list[Detection]:
    """In text a scan sends upstream as it is: the values of ``found`` the
    overrides let through, read from ``winners`` (the values the resolution
    acts on). Each winner that would refuse the text (in block or redact
    mode, or a span joining values) was approved; it stands for itself,
    for each value it joins (``Detection.members``), or — a piece of an llm
    finding — for the finding it was cut from: of that type, the one
    holding it among the llm findings a greedy sweep takes (computed here).
    A value no winner stands for was asked about by no one."""
    swept: list[Detection] = []
    taken: set[int] = set()
    for d in sorted((d for d in found if d.tier == ADDITIVE_TIER), key=_order):
        span = set(range(d.start, d.end))
        if not span & taken:
            taken |= span
            swept.append(d)
    through: list[Detection] = []
    for w in winners:
        mode = "redact" if w.tier == 0 else modes.get(w.detector_type, "redact")
        if mode == "warn":
            continue
        if w.tier == 0:
            through += w.members
        elif w.tier == ADDITIVE_TIER:
            through += [
                s
                for s in swept
                if s.detector_type == w.detector_type and s.start <= w.start and w.end <= s.end
            ]
        else:
            through.append(w)
    return through


def expected_warnings(
    found: Sequence[Detection],
    modes: Mapping[str, str],
    redacted: set[int],
    *,
    through: Collection[Detection] = (),
) -> Counter[str]:
    """The warnings the rule predicts: ``found`` the configured detections
    the redaction read (the configured detectors', allowlists applied, as
    the plan hands them over), ``redacted`` the positions replaced (empty
    where the text is sent as it is), ``through`` the values the overrides
    let through where the text is sent as it is (``let_through``)."""
    candidates = [
        d
        for d in found
        if d.tier != 0 and (modes.get(d.detector_type) in ("warn", "block") or d in through)
    ]
    candidates.sort(key=_order)
    counted: Counter[str] = Counter()
    held = set(redacted)
    for llm in (False, True):
        taken: set[int] = set()
        for d in candidates:
            if (d.tier == ADDITIVE_TIER) != llm:
                continue
            span = set(range(d.start, d.end))
            if span <= held:
                continue  # sends nothing of its own, and hides nothing
            if span & taken:
                continue  # overlaps a value taken before it
            taken |= span
            if modes.get(d.detector_type) == "warn":
                counted[d.detector_type] += 1
        # The llm findings are read against the other values taken too.
        held |= taken
    return counted
