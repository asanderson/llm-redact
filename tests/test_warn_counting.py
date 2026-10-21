"""Warn-mode values are counted by ONE rule over the values found and what is
replaced (docs/detection.md "Per-rule modes"; ``Redactor._count_warned``).

The candidates are the configured detections the redaction read that are
forwarded as sent, or would be — in warn or block mode (a block-mode value
whether or not an override approves it), an llm finding as found, never a
deny string — or, in text a scan sends upstream as it is, every value
found. Those with a character outside what the text's fate replaces are read
by the overlap sweep, and each warn-mode value it takes counts once. Counted
once per text, before anything acts on the winners: a text refused for a
block — in the winner loop or while resolving — counts what it counts with
every block approved.

So a deny string counts alike configured or added wherever it replaces the
same characters, an llm finding counts once however many pieces of it are
sent (and when the additive sweep drops it, when part of it is sent), and a
refusing value never changes what is counted, wherever it stands. Nothing
else the redaction does changes: a differential against the redactor before
the rule (tests/redactor_before_warn_rule.py, a frozen copy) pins output,
refusals, issued placeholders, ``Redactor.counts``, the overrides consulted
(in order), the recorder's calls and the string budget.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from collections.abc import Sequence
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

import redactor_before_warn_rule as before
from llm_redact.detection.base import ADDITIVE_TIER, Detection
from llm_redact.detection.deny import DenyDetector, DenyEntry
from llm_redact.detection.engine import Allowlist, DetectorPlan
from llm_redact.redactor import BlockedRequest, Redactor
from llm_redact.vault import InMemoryVault
from warn_oracle import Every, expected_warnings, replaced

# Every character once, none a case-insensitive match of another: a string
# occurs at one place, so spans and deny strings are positions.
TEXT = "ABCDEFGHIJKLMNOPQRSTUVWX"
TYPES = ("EMAIL", "PHONE", "IBAN")


class Asked:
    """Approves exactly ``values``; logs every consultation, in order."""

    def __init__(self, *values: str) -> None:
        self.values = set(values)
        self.log: list[tuple[str, ...]] = []

    def allows(self, detector_type: str, value: str) -> bool:
        self.log.append(("allows", detector_type, value))
        return value in self.values

    def unoverridable(self) -> None:
        self.log.append(("unoverridable",))


class Blocks:
    """Approves every block-mode value, and only those."""

    def __init__(self, modes: dict[str, str]) -> None:
        self.modes = modes

    def allows(self, detector_type: str, value: str) -> bool:
        return self.modes.get(detector_type) == "block"

    def unoverridable(self) -> None:
        return None


class Recorded:
    """A scan sink's recorder: every call kept."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Any]]] = []

    def record(self, text: str, found: Sequence[Any]) -> None:
        self.calls.append((text, list(found)))


class Spans:
    """A stand-in detector finding exactly ``found`` in any text."""

    name = "spans"

    def __init__(self, found: Sequence[Detection]) -> None:
        self.found = list(found)

    def detect(self, text: str) -> list[Detection]:
        return list(self.found)


def _at(start: int, end: int, kind: str, tier: int = 1, *, joins: bool = False) -> Detection:
    return Detection(start, end, kind, TEXT[start:end], 120 if tier else 0, tier, joins)


def _deny(*strings: str) -> DetectorPlan | None:
    return DetectorPlan([DenyDetector([DenyEntry(s) for s in strings])]) if strings else None


def _redactor(
    found: Sequence[Detection],
    modes: dict[str, str],
    *,
    added: tuple[str, ...] = (),
    configured: tuple[str, ...] = (),
    final_blocks: frozenset[str] = frozenset(),
    vault: InMemoryVault | None = None,
    cls: Any = Redactor,
) -> Any:
    detectors: list[Any] = [Spans(found)]
    if configured:
        detectors.append(DenyDetector([DenyEntry(s) for s in configured]))
    return cls(
        detectors,
        vault if vault is not None else InMemoryVault(),
        Allowlist(),
        modes=modes,
        added_deny=_deny(*added),
        final_blocks=final_blocks,
    )


# --- the consequences -----------------------------------------------------------------

# Two overlapping warn-mode values, W (starting first) and X.
W = _at(2, 10, "W")
X = _at(6, 14, "X")
WARN_WX = {"W": "warn", "X": "warn"}


@pytest.mark.parametrize(
    ("deny", "warned"),
    [
        # Inside W only: a configured string leaves X a winner of its own
        # (W left the resolution), an added one is taken in after W beat X.
        # Both are forwarded as sent and overlap: W, read first, counts.
        ("DE", Counter({"W": 1})),
        # Covering W whole: nothing of it is sent; X is the value sent.
        ("CDEFGHIJ", Counter({"X": 1})),
        # Covering both whole: nothing counts.
        ("BCDEFGHIJKLMNO", Counter()),
    ],
)
@pytest.mark.parametrize("scanning", [False, True])
def test_a_deny_string_counts_alike_configured_or_added(
    deny: str, warned: Counter[str], scanning: bool
) -> None:
    outputs = []
    for configured in (True, False):
        vault = InMemoryVault()
        redactor = _redactor(
            [W, X],
            WARN_WX,
            configured=(deny,) if configured else (),
            added=() if configured else (deny,),
            vault=vault,
        )
        if scanning:
            assert redactor.scan(TEXT).found == Counter({"DENY": 1})
        else:
            outputs.append(redactor.redact_text(TEXT))
        assert redactor.warn_counts == warned
    # The same characters replaced either way.
    assert len(set(outputs)) <= 1


@pytest.mark.parametrize("scanning", [False, True])
def test_an_llm_finding_cut_into_pieces_counts_once(scanning: bool) -> None:
    # A deny string in the middle of a warn-mode llm finding: two pieces of
    # it go upstream, one value.
    finding = _at(2, 12, "PERSON", ADDITIVE_TIER)
    redactor = _redactor([finding], {"PERSON": "warn"}, configured=("GH",))
    if scanning:
        redactor.scan(TEXT, redactable=True)
    else:
        assert redactor.redact_text(TEXT) == "ABCDEF«DENY_001»IJKLMNOPQRSTUVWX"
    assert redactor.warn_counts == Counter({"PERSON": 1})


@pytest.mark.parametrize("scanning", [False, True])
def test_an_llm_finding_the_additive_sweep_drops_counts_when_part_of_it_is_sent(
    scanning: bool,
) -> None:
    # A redact-mode llm finding starting first wins the additive sweep; the
    # warn-mode one it overlaps is dropped, not cut — and its tail goes
    # upstream: a warning.
    person = _at(2, 8, "PERSON", ADDITIVE_TIER)
    email = _at(5, 12, "EMAIL", ADDITIVE_TIER)
    redactor = _redactor([person, email], {"EMAIL": "warn"})
    if scanning:
        assert redactor.scan(TEXT, redactable=True).found == Counter({"PERSON": 1})
    else:
        assert redactor.redact_text(TEXT) == "AB«PERSON_001»IJKLMNOPQRSTUVWX"
    assert redactor.warn_counts == Counter({"EMAIL": 1})


# A block-mode value (K) before the warn-mode one (W2), or after it.
EARLY = (_at(0, 3, "K"), _at(6, 10, "W"))
LATE = (_at(6, 10, "W"), _at(14, 18, "K"))


@pytest.mark.parametrize("found", [EARLY, LATE])
@pytest.mark.parametrize("approve", [False, True])
@pytest.mark.parametrize("scanning", [False, True])
def test_a_block_counts_alike_before_or_after_a_warn_value(
    found: tuple[Detection, ...], approve: bool, scanning: bool
) -> None:
    # Wherever the refusing value stands, approved or not: the warn-mode
    # value counts (it counted only when the block came after it).
    block = next(d for d in found if d.detector_type == "K")
    asked = Asked(*((block.value,) if approve else ()))
    redactor = _redactor(found, {"K": "block", "W": "warn"}).with_overrides(asked)
    run = (lambda: redactor.scan(TEXT)) if scanning else (lambda: redactor.redact_text(TEXT))
    if approve:
        run()
    else:
        with pytest.raises(BlockedRequest, match="K"):
            run()
    assert redactor.warn_counts == Counter({"W": 1})
    # The block was asked about once, as before; nothing else.
    assert asked.log == [("allows", "K", block.value)]


# A warn-mode value (W, AB..F), a joining one (J, CD..J) in redact mode and
# a block-mode value (K, IJ..L) overlapping the joining one: one run.
JOINED = (_at(0, 6, "W"), _at(2, 10, "J", joins=True), _at(8, 12, "K"))


@pytest.mark.parametrize("scanning", [False, True])
def test_a_block_inside_a_joined_run_counts_what_it_counts_approved(scanning: bool) -> None:
    # Refused while resolving (Redactor._join): counted as with K approved —
    # the joining value replaced, W's first characters sent — and the
    # override consulted exactly once, by the refusal itself.
    asked = Asked()
    redactor = _redactor(JOINED, {"W": "warn", "K": "block"}).with_overrides(asked)
    with pytest.raises(BlockedRequest, match="K"):
        if scanning:
            redactor.scan(TEXT)
        else:
            redactor.redact_text(TEXT)
    assert redactor.warn_counts == Counter({"W": 1})
    assert asked.log == [("allows", "K", "IJKL")]
    approved = _redactor(JOINED, {"W": "warn", "K": "block"}).with_overrides(Asked("IJKL"))
    assert approved.redact_text(TEXT) == "AB«J_001»KLMNOPQRSTUVWX"
    assert approved.warn_counts == Counter({"W": 1})
    # A check ahead of the redaction counts nothing, refusing or not.
    checked = _redactor(JOINED, {"W": "warn", "K": "block"})
    assert checked.blocked_type(TEXT) == "K" and checked.warn_counts == Counter()


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("scanning", [False, True])
def test_a_block_an_added_deny_string_meets_counts_what_it_counts_approved(
    final: bool, scanning: bool
) -> None:
    # An access gate's added deny string makes the redaction take its
    # matches in (Redactor._absorb), where the block-mode winner K refuses
    # the text while resolving. The warn-mode value W the added string cuts
    # counts, as with K approved — an overlay-added (final) block too, its
    # refusal consulting nothing but unoverridable(), once, as before.
    found = (_at(0, 3, "K"), _at(10, 16, "W"))
    asked = Asked()
    redactor = _redactor(
        found,
        {"K": "block", "W": "warn"},
        added=("MN",),
        final_blocks=frozenset({"K"}) if final else frozenset(),
    ).with_overrides(asked)
    with pytest.raises(BlockedRequest, match="K"):
        if scanning:
            redactor.scan(TEXT)
        else:
            redactor.redact_text(TEXT)
    assert redactor.warn_counts == Counter({"W": 1})
    assert asked.log == ([("unoverridable",)] if final else [("allows", "K", "ABC")])


def test_a_value_forwarded_as_sent_hides_a_warn_value_it_overlaps() -> None:
    # Of values forwarded as sent that overlap, the sweep counts the ones it
    # takes: a block-mode value read first hides the warn-mode one, approved
    # or refusing. A redact-mode value is no such value, even one sent in
    # part (cut by another one replaced): it hides nothing.
    block = (_at(2, 10, "K"), _at(6, 14, "W"))
    for asked in (Asked("CDEFGHIJ"), Asked()):
        redactor = _redactor(block, {"K": "block", "W": "warn"}).with_overrides(asked)
        with contextlib.suppress(BlockedRequest):
            redactor.redact_text(TEXT)
        assert redactor.warn_counts == Counter()
    # R1 replaced; R2 (redact mode) cut by it and sent in part; W after R2.
    cut = (_at(0, 6, "R"), _at(3, 9, "R2"), _at(7, 12, "W"))
    redactor = _redactor(cut, {"W": "warn"})
    assert redactor.redact_text(TEXT) == "«R_001»GHIJKLMNOPQRSTUVWX"
    assert redactor.warn_counts == Counter({"W": 1})
    # Nor in a text refused while resolving (a block-mode value K in a run
    # of values joined around J, apart from them).
    refused = (*cut, _at(14, 18, "J", joins=True), _at(16, 20, "K"))
    redactor = _redactor(refused, {"W": "warn", "K": "block"})
    with pytest.raises(BlockedRequest, match="K"):
        redactor.redact_text(TEXT)
    assert redactor.warn_counts == Counter({"W": 1})
    # Nor a deny string another one wins against, sent in part: a deny
    # string is no value forwarded as sent.
    denied = (_at(0, 6, "D", 0), _at(4, 10, "D", 0), _at(8, 14, "W"))
    redactor = _redactor(denied, {"W": "warn"})
    assert redactor.redact_text(TEXT) == "«D_001»GHIJKLMNOPQRSTUVWX"
    assert redactor.warn_counts == Counter({"W": 1})


def test_a_value_covered_whole_hides_nothing() -> None:
    # V lies wholly inside a deny string; W, overlapping it and read after
    # it, is sent in part and counts.
    found = (_at(4, 8, "V"), _at(6, 12, "W"))
    redactor = _redactor(found, {"V": "warn", "W": "warn"}, configured=("DEFGH",))
    assert redactor.redact_text(TEXT) == "ABC«DENY_001»IJKLMNOPQRSTUVWX"
    assert redactor.warn_counts == Counter({"W": 1})


def test_two_detectors_finding_one_value_count_once() -> None:
    found = (_at(4, 9, "W"), _at(4, 9, "V"))
    redactor = _redactor(found, {"W": "warn", "V": "warn"}, configured=("F",))
    redactor.redact_text(TEXT)
    assert redactor.warn_counts == Counter({"W": 1})


def test_touching_spans_cover_a_value_together() -> None:
    # Two deny strings touching inside a warn-mode value cover it whole.
    found = (_at(4, 10, "W"),)
    redactor = _redactor(found, {"W": "warn"}, configured=("EFG", "HIJ"))
    assert redactor.redact_text(TEXT) == "ABCD«DENY_001»«DENY_002»KLMNOPQRSTUVWX"
    assert redactor.warn_counts == Counter()
    # One character left between them: sent, a warning.
    redactor = _redactor(found, {"W": "warn"}, configured=("EF", "HIJ"))
    redactor.redact_text(TEXT)
    assert redactor.warn_counts == Counter({"W": 1})


@pytest.mark.parametrize(
    ("approve", "warned"),
    [
        # Sent as it is (the redact-mode value approved): every value found
        # is forwarded as sent and read as such — R, read first, hides W.
        (("ABCDEF",), Counter()),
        # Refused by the caller (R found): counted as redact_text counts
        # it, R replaced and W's tail sent.
        ((), Counter({"W": 1})),
    ],
)
def test_a_scan_sent_as_it_is_reads_every_value_found(
    approve: tuple[str, ...], warned: Counter[str]
) -> None:
    found = (_at(0, 6, "R"), _at(3, 9, "W"))
    redactor = _redactor(found, {"W": "warn"}).with_overrides(Asked(*approve))
    redactor.scan(TEXT)
    assert redactor.warn_counts == warned


def test_no_warn_mode_no_count_and_no_second_resolution() -> None:
    # Without a rule in warn mode nothing is counted and a text refused
    # while resolving is not resolved again (nothing consulted twice).
    asked = Asked()
    redactor = _redactor(JOINED, {"K": "block"}).with_overrides(asked)
    with pytest.raises(BlockedRequest):
        redactor.redact_text(TEXT)
    assert asked.log == [("allows", "K", "IJKL")]
    assert redactor.warn_counts == Counter()


# --- the properties ---------------------------------------------------------------------

SPANS = st.lists(
    st.tuples(
        st.integers(0, len(TEXT) - 1),  # start
        st.integers(1, 12),  # length
        st.sampled_from(TYPES),
        # A configured deny string (tier 0), a value, an llm finding.
        st.sampled_from([0, 1, 1, 1, ADDITIVE_TIER]),
        st.booleans(),  # joins (values only)
        st.sampled_from([110, 120]),  # priority
    ),
    max_size=7,
)
VALUES = st.lists(
    st.tuples(
        st.integers(0, len(TEXT) - 1),
        st.integers(1, 12),
        st.sampled_from(TYPES),
        st.sampled_from([1, 1, 1, ADDITIVE_TIER]),
        st.booleans(),
        st.sampled_from([110, 120]),
    ),
    max_size=7,
)
STRINGS = st.lists(st.tuples(st.integers(0, len(TEXT) - 1), st.integers(1, 10)), max_size=3)
MODES = st.dictionaries(st.sampled_from(TYPES), st.sampled_from(["warn", "block", "redact"]))
APPROVE = st.sets(st.integers(0, 6))


def _found(spans: Sequence[tuple[int, int, str, int, bool, int]]) -> list[Detection]:
    found = []
    for start, length, kind, tier, joins, priority in spans:
        end = min(start + length, len(TEXT))
        found.append(
            Detection(start, end, kind, TEXT[start:end], priority, tier, joins and tier == 1)
        )
    return found


def _strings(spans: Sequence[tuple[int, int]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(TEXT[start : start + length] for start, length in spans))


def _twin_replaced(found: Sequence[Detection], modes: dict[str, str], **kwargs: Any) -> set[int]:
    """What the same text replaces with every refusing value approved."""
    vault = InMemoryVault()
    twin = _redactor(found, modes, vault=vault, **kwargs).with_overrides(Every())
    return replaced(TEXT, twin.redact_text(TEXT), vault)


@settings(deadline=None, max_examples=1500)
@given(
    spans=VALUES,
    deny=st.lists(
        st.tuples(st.integers(0, len(TEXT) - 1), st.integers(1, 6)), min_size=1, max_size=3
    ),
    modes=MODES,
    approve=APPROVE,
    redactable=st.booleans(),
)
@example(
    spans=[(2, 8, "EMAIL", 1, False, 110), (6, 8, "PHONE", 1, False, 110)],
    deny=[(3, 2)],
    modes={"EMAIL": "warn", "PHONE": "warn"},
    approve=set(),
    redactable=False,
)
def test_deny_strings_count_alike_configured_or_added_where_they_replace_alike(
    spans: list[tuple[int, int, str, int, bool, int]],
    deny: list[tuple[int, int]],
    modes: dict[str, str],
    approve: set[int],
    redactable: bool,
) -> None:
    # The same detections, the same deny strings configured or added by an
    # access gate (joins, llm findings, random modes and approvals): where
    # the two replace the same characters, they count the same — in
    # redact_text and in a scan, refused or not.
    found = _found(spans)
    strings = _strings(deny)
    values = {found[i].value for i in approve if i < len(found)}
    counted = {}
    replacing = {}
    for configured in (True, False):
        where = {"configured": strings} if configured else {"added": strings}
        replacing[configured] = _twin_replaced(found, modes, **where)
        redacting = _redactor(found, modes, **where).with_overrides(Asked(*values))
        scanning = _redactor(found, modes, **where).with_overrides(Asked(*values))
        with contextlib.suppress(BlockedRequest):
            redacting.redact_text(TEXT)
        with contextlib.suppress(BlockedRequest):
            scanning.scan(TEXT, redactable=redactable)
        counted[configured] = (redacting.warn_counts, scanning.warn_counts)
        # Both follow the rule (every deny string is found: never sent as is).
        detections = _redactor(found, modes).plan.detect(TEXT, Allowlist())
        expected = expected_warnings(detections, modes, replacing[configured])
        assert counted[configured] == (expected, expected)
    if replacing[True] == replacing[False]:
        assert counted[True] == counted[False]


@settings(deadline=None, max_examples=1500)
@given(
    spans=SPANS,
    added=STRINGS,
    modes=MODES,
    approve=APPROVE,
    final=st.sets(st.sampled_from(TYPES)),
    redactable=st.booleans(),
)
# A block before a warn-mode value and after it; in a joined run; among the
# winners an added deny string is taken in with.
@example(
    spans=[(0, 3, "PHONE", 1, False, 110), (6, 4, "EMAIL", 1, False, 110)],
    added=[],
    modes={"PHONE": "block", "EMAIL": "warn"},
    approve=set(),
    final=set(),
    redactable=False,
)
@example(
    spans=[
        (0, 6, "EMAIL", 1, False, 110),
        (2, 8, "IBAN", 1, True, 110),
        (8, 4, "PHONE", 1, False, 110),
    ],
    added=[],
    modes={"PHONE": "block", "EMAIL": "warn"},
    approve=set(),
    final=set(),
    redactable=True,
)
@example(
    spans=[(0, 3, "PHONE", 1, False, 110), (10, 6, "EMAIL", 1, False, 110)],
    added=[(12, 2)],
    modes={"PHONE": "block", "EMAIL": "warn"},
    approve=set(),
    final={"PHONE"},
    redactable=False,
)
def test_a_refused_text_counts_what_it_counts_with_every_block_approved(
    spans: list[tuple[int, int, str, int, bool, int]],
    added: list[tuple[int, int]],
    modes: dict[str, str],
    approve: set[int],
    final: set[str],
    redactable: bool,
) -> None:
    found = _found(spans)
    strings = _strings(added)
    values = {found[i].value for i in approve if i < len(found)}
    final_blocks = frozenset(t for t in final if modes.get(t) == "block")
    approved = _redactor(found, modes, added=strings).with_overrides(Blocks(modes))
    approved.redact_text(TEXT)  # never refuses: every block approved
    redacting = _redactor(found, modes, added=strings, final_blocks=final_blocks).with_overrides(
        Asked(*values)
    )
    try:
        redacting.redact_text(TEXT)
    except BlockedRequest:
        assert redacting.warn_counts == approved.warn_counts
    scanning = _redactor(found, modes, added=strings, final_blocks=final_blocks).with_overrides(
        Asked(*values)
    )
    try:
        scanning.scan(TEXT, redactable=redactable)
    except BlockedRequest:
        # As redact_text counts the text with every block approved...
        assert scanning.warn_counts == approved.warn_counts
        # ...and as the scan does, unless it then sends the text as it is
        # (nothing replaced: a text that goes upstream, not a refused one).
        twin = _redactor(found, modes, added=strings).with_overrides(Blocks(modes))
        scan = twin.scan(TEXT, redactable=redactable)
        if redactable or scan.found:
            assert scanning.warn_counts == twin.warn_counts


def _run(cls: Any, call: str, found: list[Detection], modes: dict[str, str], **kw: Any) -> Any:
    """One call of a ``cls`` redactor, everything it touches recorded."""
    values, recorded, redactable = kw.pop("values"), kw.pop("recorded"), kw.pop("redactable")
    floors = kw.pop("floors")
    vault = InMemoryVault()
    asked = Asked(*values)
    recorder = Recorded()
    redactor = _redactor(found, modes, vault=vault, cls=cls, **kw).with_floors(floors)
    # Recorded before the budget and override copies: they keep it (and
    # the recorder's copy keeps the floors).
    if recorded:
        redactor = redactor.with_recorder(recorder)
    redactor = redactor.with_budget(10).with_overrides(asked)
    try:
        if call == "redact_text":
            result: Any = redactor.redact_text(TEXT)
        elif call == "scan":
            result = tuple(redactor.scan(TEXT, redactable=redactable))
        else:
            result = redactor.blocked_type(TEXT)
    except (BlockedRequest, before.BlockedRequest) as refused:
        result = ("refused", type(refused).__name__, refused.detector_type)
    return (
        result,
        dict(redactor.counts),
        asked.log,
        recorder.calls,
        redactor._budget.used,
        dict(vault._reverse),
    )


@settings(deadline=None, max_examples=2000)
@given(
    spans=SPANS,
    added=STRINGS,
    modes=MODES,
    approve=APPROVE,
    final=st.sets(st.sampled_from(TYPES)),
    redactable=st.booleans(),
    recorded=st.booleans(),
    floors=st.dictionaries(st.sampled_from(TYPES), st.integers(1, 5)),
    call=st.sampled_from(["redact_text", "scan", "blocked_type"]),
)
def test_nothing_but_the_warn_counts_differs_from_before(
    spans: list[tuple[int, int, str, int, bool, int]],
    added: list[tuple[int, int]],
    modes: dict[str, str],
    approve: set[int],
    final: set[str],
    redactable: bool,
    recorded: bool,
    floors: dict[str, int],
    call: str,
) -> None:
    # Against the redactor before the rule (a frozen copy): output, refusal,
    # placeholders issued, counts, overrides consulted in order, the
    # recorder's calls and the budget are the same — the counting resolution
    # of a refused text consults nothing.
    found = _found(spans)
    kw = {
        "added": _strings(added),
        "final_blocks": frozenset(t for t in final if modes.get(t) == "block"),
        "values": {found[i].value for i in approve if i < len(found)},
        "recorded": recorded,
        "redactable": redactable,
        "floors": floors,
    }
    now = _run(Redactor, call, found, modes, **dict(kw))
    then = _run(before.Redactor, call, found, modes, **dict(kw))
    assert now == then
