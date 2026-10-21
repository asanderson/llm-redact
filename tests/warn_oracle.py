"""The warnings the counting rule predicts for a redacted text, worked out
without the redactor's own helpers.

A warn-mode value is forwarded as sent and counted as a warning. One the
redaction acts on (a winner, left whole as sent) is counted by
``redact_text`` and ``scan``. Any other is a warning when the redaction CUT
it: some of its characters were replaced by a placeholder and some are still
sent — whatever it lost to: a deny string (configured, or added by an access
gate), a value redacted in its place (a longer one, an earlier one), a
joined run, an added deny string taking a winner's place, an llm finding
redacting part of it. One the placeholders cover whole sends nothing; one
that lost only to a value forwarded as sent has no character replaced:
neither is a warning. Of the cut values overlapping one another, only the
first the start-ordered overlap sweep reads counts (two detectors finding
one value count once). A deny string takes no mode; an llm finding
(``ADDITIVE_TIER``) is cut around the others, never dropped, so it counts
only through the pieces the resolution keeps of it. Which characters were
replaced is read from the output itself (``replaced``). Where the proxy
cannot rewrite the text (``Redactor.scan``: a verbatim field, an upload's
extracted text) nothing is replaced: a value is cut there by the values the
text is refused for — or redacted for, when the caller redacts it after all
(convert mode) — and a value an approved refusal override lets through is
forwarded as sent, so one that lost only to such a value is no warning
either (``scanned``). Shared by tests/test_joined_values.py and
tests/test_overlay_deny.py.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence

from llm_redact.detection.base import ADDITIVE_TIER, Detection
from llm_redact.placeholders import PLACEHOLDER_RE
from llm_redact.vault import InMemoryVault


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


def scanned(
    winners: Sequence[Detection],
    modes: Mapping[str, str],
    approved: Collection[str],
    *,
    redactable: bool,
) -> set[int] | None:
    """The positions a scan of text the proxy cannot rewrite
    (``Redactor.scan``) refuses that text for — or, ``redactable`` (convert
    mode), redacts it for — among ``winners`` (the values the resolution
    acts on, an access gate's added deny strings taken in): the counterpart
    of what ``replaced`` reads from a redacted text. A warn-mode value is
    forwarded as sent; so is one the scan puts to the requester's overrides
    when they approve it (its value in ``approved``): a value in block mode
    (None when one is not approved: the scan refuses the text) and, where
    the text is not ``redactable``, a value in redact mode or a span joining
    values (``Detection.members``: when each of them is approved; one
    holding a deny string never). A deny string never is."""
    positions: set[int] = set()
    for d in winners:
        if d.tier == 0:
            members = d.members
            if (
                not redactable
                and members
                and all(m.tier != 0 and m.value in approved for m in members)
            ):
                continue
        else:
            mode = modes.get(d.detector_type, "redact")
            if mode == "warn":
                continue
            if mode == "block" and d.value not in approved:
                return None
            if (mode == "block" or not redactable) and d.value in approved:
                continue
        positions.update(range(d.start, d.end))
    return positions


def cut_warnings(
    found: Sequence[Detection],
    configured: Sequence[Detection],
    modes: Mapping[str, str],
    redacted: set[int],
) -> Counter[str]:
    """The warnings predicted for the warn-mode values the redaction cut:
    among ``found`` (the configured detectors' detections, start-sorted,
    the longest first, then by priority) — a deny string and an llm finding
    aside — and the llm findings' pieces among the ``configured`` winners
    (the resolution's without an access gate's added deny strings), a
    warn-mode value with a character in ``redacted`` (the offsets finally
    replaced, or refused for) and one outside it; of such values overlapping
    one another, only the first the sweep takes counts."""
    values = [d for d in found if d.tier not in (0, ADDITIVE_TIER)]
    values += [d for d in configured if d.tier == ADDITIVE_TIER]
    values.sort(key=lambda d: (d.start, -(d.end - d.start), d.priority))
    counted: Counter[str] = Counter()
    end = -1
    for d in values:
        if modes.get(d.detector_type) != "warn":
            continue
        span = set(range(d.start, d.end))
        if not span & redacted:
            continue  # a winner, or it lost only to values forwarded as sent
        if span <= redacted:
            continue  # sends nothing
        if d.start >= end:
            counted[d.detector_type] += 1
            end = d.end
    return counted


def expected_warnings(
    found: Sequence[Detection],
    configured: Sequence[Detection],
    modes: Mapping[str, str],
    redacted: set[int],
) -> Counter[str]:
    """Every warning predicted for the text: one per warn-mode winner of
    the ``configured`` resolution no character of which was replaced (left
    whole as sent; an added deny string overlapping one cuts it instead),
    plus those the redaction cut (``cut_warnings``). ``redacted``: the
    positions replaced (``replaced``), or a scan refuses the text for
    (``scanned``)."""
    whole = Counter(
        d.detector_type
        for d in configured
        if d.tier
        and modes.get(d.detector_type) == "warn"
        and not set(range(d.start, d.end)) & redacted
    )
    return whole + cut_warnings(found, configured, modes, redacted)
