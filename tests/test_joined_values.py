"""A value overlapping a newly read one is joined with it, never cut.

1.13.0 reads a card inside a longer run of digit groups
(regex_rules._card_inside) and 1.12.1 the IBAN's paper format: values no
earlier release read, right next to text another value may hold. A custom
rule's value or an NER span starting inside such a card or IBAN and running
on past it lost its tail to it, since the earlier-starting value won the
start-ordered overlap sweep (redactor._resolve_overlaps). Both readings now
JOIN (Detection.joins): in a run of values overlapping one, directly or
through others, Redactor._join redacts every value that redacts (values
overlapping one another as one placeholder over their union, typed after
the first), refuses for every value in block mode and forwards a warn-mode
value only where nothing redacted covers it.

The differential at the end pins it against 1.12.1's pipeline over the
recall and fp corpora and generated texts, with the same custom rules and
fake NER spans (ner_fakes.FakeSpacy): every character 1.12.1 redacted stays
redacted, every request it refused is refused, under the default modes and
random ones.
"""

import random
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_redact.bench.corpus import VALUE_GENERATORS, generate
from llm_redact.detection.base import ADDITIVE_TIER, Detection, Detector
from llm_redact.detection.deny import DenyDetector, DenyEntry
from llm_redact.detection.engine import (
    Allowlist,
    CustomRule,
    DetectionConfig,
    DetectorPlan,
    build_allowlist,
    build_detectors,
)
from llm_redact.detection.ner import NerDetector
from llm_redact.detection.regex_rules import RegexDetector
from llm_redact.placeholders import PLACEHOLDER_RE
from llm_redact.redactor import BlockedRequest, Redactor, _resolve_overlaps, _sweep
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault
from ner_fakes import FakeSpacy
from test_no_loss_since_1_12_0 import ENDINGS, JOINERS, as_1_12_1, card_run, luhn_card

CARD = "4111 1111 1111 1111"
# A paper IBAN whose digits after the bank code are a Luhn-valid card number.
IBAN = "GB39 WEST 3607 5983 8675 65"
IBAN_CARD = "3607 5983 8675 65"
STREET = CustomRule(name="street", detector_type="ADDRESS", pattern=r"\d{2,4} [A-Z][a-z]+ Street")
ORDER = CustomRule(name="order", detector_type="ORDER_REF", pattern=r"Order \d{1,4} \d{4}")
FP_CORPUS = Path(__file__).resolve().parent.parent / "bench" / "fp_corpus"


class Approved:
    """Allows exactly ``values``; records every question and every refusal
    made final."""

    def __init__(self, *values: str) -> None:
        self.values = set(values)
        self.asked: list[tuple[str, str]] = []
        self.final = 0

    def allows(self, detector_type: str, value: str) -> bool:
        self.asked.append((detector_type, value))
        return value in self.values

    def unoverridable(self) -> None:
        self.final += 1


def _redactor(
    *rules: CustomRule,
    modes: Mapping[str, str] | None = None,
    ner: Iterable[tuple[str, str]] = (),
    deny: tuple[str, ...] = (),
    allowlist: tuple[str, ...] = (),
    by_type: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> Redactor:
    """The production Redactor with ``rules``, a fake NER backend finding
    ``ner``'s (surface, label) pairs, ``deny`` strings and allowlists."""
    config = DetectionConfig(
        custom_rules=rules,
        deny_strings=tuple(DenyEntry(value) for value in deny),
        allowlist=allowlist,
        allowlist_by_type=by_type,
    )
    detectors: list[Detector] = list(build_detectors(config))
    findings = [(surface, label, 1.0) for surface, label in ner]
    detectors.append(NerDetector(FakeSpacy(findings), frozenset({"PERSON", "ADDRESS"}), 10_000))
    return Redactor(detectors, InMemoryVault(), build_allowlist(config), modes=modes)


def _redacted(redactor: Redactor, text: str) -> str:
    """``text`` through ``redactor``; its output restores exactly to it."""
    output = redactor.redact_text(text)
    assert Rehydrator(redactor._vault).rehydrate_text(output) == text
    return output


# --- what is joined -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "rules", "ner", "expected"),
    [
        # A custom rule's value starting inside the card and running on
        # past it: the card and it are one placeholder, typed after the card.
        (f"Order 12 {CARD} Main Street", (STREET,), (), "Order 12 «CREDIT_CARD_001»"),
        # An NER span doing the same.
        (
            f"Order 12 {CARD} Main Road",
            (),
            (("1111 Main", "PERSON"),),
            "Order 12 «CREDIT_CARD_001» Road",
        ),
        # A value starting before the card and ending inside it: typed after it.
        (f"Order 12 {CARD} paid", (ORDER,), (), "«ORDER_REF_001» paid"),
        (f"Ship to Rue 12 {CARD}", (), (("Rue 12 4111", "ADDRESS"),), "Ship to «ADDRESS_001»"),
        # Both at once, through the card.
        (f"Order 12 {CARD} Main Street", (ORDER, STREET), (), "«ORDER_REF_001»"),
        # A paper IBAN with a value starting in its last group.
        (f"pay {IBAN} Main Street", (STREET,), (), "pay «IBAN_001»"),
        # The changelog's.
        ("pay GB82 WEST 1234 5698 7654 32 Main Street", (STREET,), (), "pay «IBAN_001»"),
        (f"pay {IBAN} now", (), (("65 now", "PERSON"),), "pay «IBAN_001»"),
        # A value only overlapping something that overlaps the card.
        (
            f"Order 12 {CARD} Main Street Jane",
            (STREET,),
            (("Street Jane", "PERSON"),),
            "Order 12 «CREDIT_CARD_001»",
        ),
    ],
)
def test_a_value_overlapping_a_joining_one_is_redacted_with_it(
    text: str, rules: tuple[CustomRule, ...], ner: tuple[tuple[str, str], ...], expected: str
) -> None:
    redactor = _redactor(*rules, ner=ner)
    assert _redacted(redactor, text) == expected
    # Counted once, under the placeholder's type.
    assert sum(redactor.counts.values()) == 1
    # Nothing in block mode: nothing to refuse for, the union built alike.
    assert redactor.blocked_type(text) is None


def test_without_a_joining_value_the_sweep_decides_as_before() -> None:
    # A card the rule reads alone (finditer's) does not join: the custom
    # value starting inside it loses its tail, as in every release.
    redactor = _redactor(STREET)
    assert _redacted(redactor, f"card {CARD} Main Street") == "card «CREDIT_CARD_001» Main Street"
    # Nor does an IBAN in the solid form.
    solid = IBAN.replace(" ", "")
    assert _redacted(_redactor(), f"x {solid} y") == "x «IBAN_001» y"


def test_values_before_and_after_a_joined_run_keep_their_order() -> None:
    redactor = _redactor(STREET)
    text = f"mail a@b.example, Order 12 {CARD} Main Street, call +1 415 555 0132"
    assert _redacted(redactor, text) == (
        "mail «EMAIL_001», Order 12 «CREDIT_CARD_001», call «PHONE_001»"
    )


def test_two_joined_runs_and_a_value_touching_one() -> None:
    qty = CustomRule(name="qty", detector_type="QTY", pattern=r"Qty \d+ ")
    redactor = _redactor(STREET, qty)
    text = f"Order 12 {CARD} Main Street; Qty 3 5500 0000 0000 0004 Elm Street"
    # A value ending where the card starts is no part of it.
    assert _redacted(redactor, text) == "Order 12 «CREDIT_CARD_001»; «QTY_001»«CREDIT_CARD_002»"
    assert redactor.counts == Counter({"CREDIT_CARD": 2, "QTY": 1})


def test_a_forwarded_value_touching_a_redacted_one_is_kept() -> None:
    # Forwarded values in the joined run: one overlapping the card yields to
    # it, one only touching it (ending where it starts, starting where it
    # ends) is forwarded and counted as before.
    rules = (
        CustomRule(name="qty", detector_type="QTY", pattern=r"Qty \d+ "),
        CustomRule(name="lead", detector_type="LEAD", pattern=r"\d+ 4111"),
        CustomRule(name="tail", detector_type="TAIL", pattern=r"1111 Main"),
        CustomRule(name="rest", detector_type="REST", pattern=r"(?<=\d) Main Street"),
    )
    modes = {"QTY": "warn", "LEAD": "warn", "TAIL": "warn", "REST": "warn"}
    redactor = _redactor(*rules, modes=modes)
    text = f"Qty 12 {CARD} Main Street"
    assert _redacted(redactor, text) == "Qty 12 «CREDIT_CARD_001» Main Street"
    assert redactor.warn_counts == Counter({"QTY": 1, "REST": 1})


def test_with_an_overlay_a_forwarded_value_never_takes_the_card() -> None:
    # An access gate's added deny strings (Redactor._absorb) sweep the
    # winners again: a warn-mode value the joined run dropped stays dropped.
    config = DetectionConfig()
    detectors: list[Detector] = list(build_detectors(config))
    detectors.append(
        NerDetector(FakeSpacy([("Qty 12 4111", "PERSON", 1.0)]), frozenset({"PERSON"}), 10_000)
    )
    redactor = Redactor(
        detectors,
        InMemoryVault(),
        build_allowlist(config),
        modes={"PERSON": "warn"},
        added_deny=DetectorPlan([DenyDetector([DenyEntry("paid")])]),
    )
    assert _redacted(redactor, f"Qty 12 {CARD} paid") == "Qty 12 «CREDIT_CARD_001» «DENY_001»"


def test_a_joining_value_alone_is_redacted_as_before() -> None:
    redactor = _redactor(modes={"CREDIT_CARD": "warn"})
    assert _redacted(redactor, f"Order 12 {CARD}") == f"Order 12 {CARD}"
    assert redactor.warn_counts == Counter({"CREDIT_CARD": 1})
    with pytest.raises(BlockedRequest, match="CREDIT_CARD"):
        _redactor(modes={"CREDIT_CARD": "block"}).redact_text(f"Order 12 {CARD}")


def test_a_paper_iban_holding_a_card_number_is_the_iban() -> None:
    # The IBAN covers the card: redacted as the IBAN, as before — and as a
    # tier-1 IBAN, put to the overrides in text the proxy cannot rewrite.
    redactor = _redactor()
    assert _redacted(redactor, f"pay {IBAN} now") == "pay «IBAN_001» now"
    approved = Approved(IBAN)
    scan = redactor.with_overrides(approved).scan(f"pay {IBAN} now")
    assert scan.overridden and scan.found == Counter() and approved.asked == [("IBAN", IBAN)]


# --- modes, deny strings, allowlists, overrides --------------------------------------


@pytest.mark.parametrize(
    ("modes", "expected", "warned"),
    [
        # A warn-mode card never keeps the other value from being redacted.
        ({"CREDIT_CARD": "warn"}, "Order 12 4111 1111 1111 «ADDRESS_001»", {}),
        # Nor a warn-mode value the card.
        ({"ADDRESS": "warn"}, "Order 12 «CREDIT_CARD_001» Main Street", {}),
        # Both forwarded: the sweep picks the one counted.
        (
            {"ADDRESS": "warn", "CREDIT_CARD": "warn"},
            f"Order 12 {CARD} Main Street",
            {"CREDIT_CARD": 1},
        ),
    ],
)
def test_a_warn_mode_value_yields_to_one_redacted(
    modes: dict[str, str], expected: str, warned: dict[str, int]
) -> None:
    redactor = _redactor(STREET, modes=modes)
    assert _redacted(redactor, f"Order 12 {CARD} Main Street") == expected
    assert redactor.warn_counts == Counter(warned)


@pytest.mark.parametrize("blocked", ["CREDIT_CARD", "ADDRESS"])
def test_a_block_mode_value_on_either_side_refuses(blocked: str) -> None:
    redactor = _redactor(STREET, modes={blocked: "block"})
    text = f"Order 12 {CARD} Main Street"
    with pytest.raises(BlockedRequest, match=blocked):
        redactor.redact_text(text)
    assert redactor.blocked_type(text) == blocked
    assert redactor.blocked_type(f"Order 12 {CARD} paid") == (
        "CREDIT_CARD" if blocked == "CREDIT_CARD" else None
    )


def test_a_block_mode_card_inside_a_paper_iban_refuses() -> None:
    # 1.12.1 redacted the IBAN and let it hide the card's block mode.
    text = f"pay {IBAN} now"
    redactor = _redactor(modes={"CREDIT_CARD": "block"})
    with pytest.raises(BlockedRequest, match="CREDIT_CARD"):
        redactor.redact_text(text)
    assert redactor.blocked_type(text) == "CREDIT_CARD"
    with pytest.raises(BlockedRequest, match="CREDIT_CARD"):
        redactor.scan(text, redactable=True)
    # A warn-mode IBAN forwards what the card number does not cover.
    warned = _redactor(modes={"IBAN": "warn"})
    assert _redacted(warned, text) == "pay GB39 WEST «CREDIT_CARD_001» now"
    assert warned._vault.original_for("«CREDIT_CARD_001»") == IBAN_CARD
    assert warned.warn_counts == Counter()


def test_an_approved_block_mode_value_is_forwarded_where_nothing_redacts() -> None:
    approved = Approved(CARD)
    redactor = _redactor(STREET, modes={"CREDIT_CARD": "block"}).with_overrides(approved)
    text = f"Order 12 {CARD} Main Street"
    assert _redacted(redactor, text) == "Order 12 4111 1111 1111 «ADDRESS_001»"
    assert redactor.blocked_type(text) is None
    assert ("CREDIT_CARD", CARD) in approved.asked


def test_a_deny_string_joined_with_a_card_is_never_put_to_the_overrides() -> None:
    redactor = _redactor(deny=("1111 Main",))
    text = f"Order 12 {CARD} Main Street"
    assert _redacted(redactor, text) == "Order 12 «CREDIT_CARD_001» Street"
    # In text the proxy cannot rewrite, their union refuses for good.
    approved = Approved(CARD, "1111 Main", f"{CARD} Main")
    scan = redactor.with_overrides(approved).scan(text)
    assert scan.found == Counter({"CREDIT_CARD": 1}) and not scan.overridden
    assert approved.asked == [] and approved.final == 1
    # A deny string inside the card: the card's span, still final.
    inside = _redactor(deny=("1111 1111",))
    assert _redacted(inside, f"Order 12 {CARD}") == "Order 12 «CREDIT_CARD_001»"
    approved = Approved(CARD)
    assert inside.with_overrides(approved).scan(f"Order 12 {CARD}").found == Counter(
        {"CREDIT_CARD": 1}
    )
    assert approved.final == 1
    # A deny string covering the card stays the deny string.
    covering = _redactor(deny=(f"12 {CARD}",))
    assert _redacted(covering, f"Order 12 {CARD}") == "Order «DENY_001»"


def test_a_joined_union_is_never_put_to_the_overrides() -> None:
    # No one approval names a card and an address together: a span joining
    # two values refuses for good in text the proxy cannot rewrite.
    redactor = _redactor(STREET)
    text = f"Order 12 {CARD} Main Street"
    approved = Approved(CARD, "1111 Main Street", f"{CARD} Main Street")
    scan = redactor.with_overrides(approved).scan(text)
    assert scan.found == Counter({"CREDIT_CARD": 1}) and approved.final == 1
    assert approved.asked == []
    # Convert mode redacts it: nothing made final.
    approved = Approved()
    assert redactor.with_overrides(approved).scan(text, redactable=True).found == Counter(
        {"CREDIT_CARD": 1}
    )
    assert approved.final == 0


def test_each_value_is_allowlisted_before_it_is_joined() -> None:
    text = f"Order 12 {CARD} Main Street"
    # An allowlisted value joins nothing.
    assert (
        _redacted(_redactor(STREET, allowlist=("1111 Main Street",)), text)
        == "Order 12 «CREDIT_CARD_001» Main Street"
    )
    assert (
        _redacted(_redactor(STREET, by_type=(("CREDIT_CARD", (CARD,)),)), text)
        == "Order 12 4111 1111 1111 «ADDRESS_001»"
    )
    # Their union is no value an allowlist entry names.
    assert (
        _redacted(_redactor(STREET, allowlist=(f"{CARD} Main Street",)), text)
        == "Order 12 «CREDIT_CARD_001»"
    )


# --- any detections ------------------------------------------------------------------

SPAN = st.tuples(
    st.integers(0, 40),  # start
    st.integers(1, 12),  # length
    st.sampled_from(("A", "B", "C", "DENY")),  # type; DENY is tier 0
    st.booleans(),  # joins
    st.integers(0, 2),  # priority
)
MODES = st.dictionaries(st.sampled_from(("A", "B", "C")), st.sampled_from(("warn", "block")))


def _sorted(spans: Iterable[tuple[int, int, str, bool, int]]) -> list[Detection]:
    """``spans`` as the detection plan hands detections on: start-sorted,
    the longest first, then by priority."""
    text = "x" * 60
    found = [
        Detection(a, a + n, kind, text[a : a + n], prio, 0 if kind == "DENY" else 1, joins)
        for a, n, kind, joins, prio in spans
    ]
    return sorted(found, key=lambda d: (d.start, -(d.end - d.start), d.priority))


def _redacts(winners: Iterable[Detection], modes: Mapping[str, str]) -> set[int] | None:
    """The offsets ``winners`` redact under ``modes`` (None: one refuses)."""
    offsets: set[int] = set()
    for d in winners:
        mode = "redact" if d.tier == 0 else modes.get(d.detector_type, "redact")
        if mode == "block":
            return None
        if mode == "redact":
            offsets.update(range(d.start, d.end))
    return offsets


@settings(deadline=None, max_examples=500)
@given(st.lists(SPAN, max_size=10), MODES, st.data())
def test_joining_never_redacts_or_refuses_less_than_a_sweep_without_some(
    spans: list[tuple[int, int, str, bool, int]], modes: dict[str, str], data: st.DataObject
) -> None:
    # Against the overlap sweep over the same detections less any joining
    # ones (a release reading fewer, as 1.12.1 did no card inside a run and
    # 1.12.0 no paper IBAN): every character it redacts is redacted, every
    # refusal it makes is made, every joining value that redacts is redacted
    # whole; the winners are disjoint and in order; and with no joining
    # value the result is the sweep's own.
    detections = _sorted(spans)
    dropped = data.draw(st.sets(st.sampled_from(range(len(detections))))) if detections else set()
    fewer = [d for i, d in enumerate(detections) if not (d.joins and i in dropped)]
    redactor = Redactor([], InMemoryVault(), Allowlist(), modes=modes)
    before = _redacts(_resolve_overlaps(fewer), modes)
    try:
        winners = redactor._resolved("x" * 60, detections)
    except BlockedRequest:
        return
    assert all(a.end <= b.start for a, b in zip(winners, winners[1:], strict=False))
    if not any(d.joins for d in detections):
        assert winners == _resolve_overlaps(detections)
    now = _redacts(winners, modes)
    if now is None:
        return
    assert before is not None and before <= now
    for d in detections:
        if d.joins and (d.tier == 0 or modes.get(d.detector_type, "redact") == "redact"):
            assert set(range(d.start, d.end)) <= now


class _Additive:
    """A stand-in for the llm NER backend: finds ``surface`` as an additive
    (``ADDITIVE_TIER``) PERSON span."""

    name = "llm"

    def __init__(self, surface: str) -> None:
        self.surface = surface

    def detect(self, text: str) -> list[Detection]:
        at = text.find(self.surface)
        if at < 0:
            return []
        end = at + len(self.surface)
        return [Detection(at, end, "PERSON", self.surface, 120, ADDITIVE_TIER)]


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        # Covering the card: the card keeps its own placeholder, the model's
        # span only what lies outside it.
        (f"12 {CARD} Main", "Order «PERSON_001» «CREDIT_CARD_001» «PERSON_002» Road"),
        # Starting inside the card and running on: its tail is still redacted.
        ("1111 Main", "Order 12 «CREDIT_CARD_001» «PERSON_001» Road"),
    ],
)
def test_an_additive_span_is_cut_around_a_joining_value(surface: str, expected: str) -> None:
    config = DetectionConfig()
    detectors: list[Detector] = [*build_detectors(config), _Additive(surface)]
    redactor = Redactor(detectors, InMemoryVault(), build_allowlist(config))
    assert _redacted(redactor, f"Order 12 {CARD} Main Road") == expected
    assert redactor.counts["CREDIT_CARD"] == 1


@settings(deadline=None, max_examples=500)
@given(
    st.lists(SPAN, max_size=10),
    st.lists(st.tuples(st.integers(0, 40), st.integers(1, 14)), max_size=4),
    MODES,
)
def test_an_additive_detection_never_joins(
    spans: list[tuple[int, int, str, bool, int]],
    added: list[tuple[int, int]],
    modes: dict[str, str],
) -> None:
    # The llm backend's detections next to joining values: the other
    # winners (and refusals) are exactly those without them, and each keeps
    # only what lies outside the winners, as _resolve_overlaps treats them.
    text = "x" * 60
    base = _sorted(spans)
    extra = [Detection(a, a + n, "PERSON", text[a : a + n], 120, ADDITIVE_TIER) for a, n in added]
    both = sorted(base + extra, key=lambda d: (d.start, -(d.end - d.start), d.priority))
    redactor = Redactor([], InMemoryVault(), Allowlist(), modes=modes)
    try:
        without = redactor._resolved(text, base)
    except BlockedRequest as refused:
        with pytest.raises(BlockedRequest) as again:
            redactor._resolved(text, both)
        assert again.value.detector_type == refused.detector_type
        return
    winners = redactor._resolved(text, both)
    assert [d for d in winners if d.tier != ADDITIVE_TIER] == without
    assert all(a.end <= b.start for a, b in zip(winners, winners[1:], strict=False))
    covered = {i for d in winners for i in range(d.start, d.end)}
    ordered = sorted(extra, key=lambda d: (d.start, -(d.end - d.start), d.priority))
    for d in _sweep(ordered):
        assert set(range(d.start, d.end)) <= covered


# --- the differential --------------------------------------------------------------

CUSTOM_RULES = (
    STREET,
    ORDER,
    # Digit groups, which start inside cards and IBANs and run on past them.
    CustomRule(name="groups", detector_type="ACCOUNT", pattern=r"\b\d{2,4}(?:[ -]\d{2,4}){2,5}\b"),
    # An IBAN's last groups and a word.
    CustomRule(name="code", detector_type="CODE", pattern=r"\b[A-Z0-9]{2,4} \d{2,4} [a-z]{3,}\b"),
)
TYPES = ("CREDIT_CARD", "IBAN", "ADDRESS", "ORDER_REF", "ACCOUNT", "CODE", "PERSON", "DENY")
WORDS = (" Main Street", " now", " 12 Main Street", "-x", " Jane Doe", ", paid", " Rue 3")


def _pipelines(config: DetectionConfig, nlp: FakeSpacy) -> tuple[DetectorPlan, DetectorPlan]:
    """This release's detectors and 1.12.1's (its rules without the fields
    1.13.0 added, test_no_loss_since_1_12_0.as_1_12_1), each with the same
    custom rules, deny strings and fake NER backend."""
    release: list[Detector] = list(build_detectors(config))
    previous: list[Detector] = [
        RegexDetector(as_1_12_1(det.rule)) if isinstance(det, RegexDetector) else det
        for det in release
    ]
    ner = NerDetector(nlp, frozenset({"PERSON", "ADDRESS"}), 1_000_000)
    return DetectorPlan([*release, ner]), DetectorPlan([*previous, ner])


def _by_1_12_1(
    plan: DetectorPlan, allowlist: Allowlist, text: str, modes: Mapping[str, str]
) -> set[int] | None:
    """What 1.12.1's Redactor redacted in ``text`` (offsets), transcribed:
    the overlap sweep's winners, each by its mode — None when a block-mode
    winner refused the text."""
    offsets: set[int] = set()
    for d in _resolve_overlaps(plan.detect(text, allowlist)):
        mode = "redact" if d.tier == 0 else modes.get(d.detector_type, "redact")
        if mode == "block":
            return None
        if mode == "redact":
            offsets.update(range(d.start, d.end))
    return offsets


def _by_release(redactor: Redactor, vault: InMemoryVault, text: str) -> set[int] | None:
    """The offsets the release's Redactor replaces (None: it refused),
    read back from its output: each placeholder stands for its vault value
    at that place."""
    try:
        redacted = redactor.redact_text(text)
    except BlockedRequest:
        return None
    offsets: set[int] = set()
    pos = cursor = 0
    for token in PLACEHOLDER_RE.finditer(redacted):
        pos += token.start() - cursor
        value = vault.original_for(token.group())
        assert value is not None and text.startswith(value, pos)
        offsets.update(range(pos, pos + len(value)))
        pos, cursor = pos + len(value), token.end()
    return offsets


def _crossing(rng: random.Random, text: str, joining: list[Detection]) -> list[tuple[str, str]]:
    """NER findings (surface, label) crossing each joining value, one or
    two of: starting inside it and running on past it, starting before it
    and ending inside it, inside it, around it; and one anywhere."""
    found = []
    for d in joining:
        inner = rng.randrange(d.start + 1, d.end)
        spans = [
            (inner, min(len(text), d.end + rng.randrange(1, 12))),
            (max(0, d.start - rng.randrange(1, 10)), rng.randrange(d.start + 1, d.end + 1)),
            (inner, rng.randrange(inner + 1, d.end + 1)),
            (max(0, d.start - 2), min(len(text), d.end + 2)),
        ]
        chosen = rng.sample(spans, rng.choice((1, 1, 2)))
        found += [(text[a:b], rng.choice(("PERSON", "ADDRESS"))) for a, b in chosen if a < b]
    if len(text) > 2:
        a = rng.randrange(len(text) - 1)
        found.append((text[a : rng.randrange(a + 1, min(len(text), a + 30) + 1)], "PERSON"))
    return [(surface, label) for surface, label in found if surface.strip()]


def _generated(rng: random.Random) -> Iterable[str]:
    """Cards in runs of digit groups (some before an expiry) and paper
    IBANs, next to words and values the custom rules and NER spans take
    in."""
    iban = VALUE_GENERATORS["iban"][1]
    values = [gen(rng) for _type, gen in VALUE_GENERATORS.values()]
    for _ in range(1500):
        value = rng.choice(values)
        yield card_run(rng, value) + rng.choice(WORDS)
        yield card_run(rng, value, expiry=True) + rng.choice(WORDS)
        yield f"Order {rng.randrange(1, 100)} " + card_run(rng, "") + rng.choice(WORDS)
        spaced = iban(rng).replace(" ", "")
        paper = " ".join(spaced[i : i + 4] for i in range(0, len(spaced), 4))
        yield rng.choice(("pay ", "", "IBAN ")) + paper + rng.choice(WORDS + ENDINGS)
        yield value + rng.choice(JOINERS) + paper + rng.choice(WORDS)
        # A card the rule reads only inside its run: a quantity before it.
        card = luhn_card(rng)
        fours = rng.choice(" -").join(card[i : i + 4] for i in range(0, len(card), 4))
        word = rng.choice(("Order", "Qty", "Item", "ref", ""))
        yield f"{word} {rng.randrange(1, 100)} {fours}" + rng.choice(WORDS + ENDINGS)


def _texts(seed: int) -> list[str]:
    rng = random.Random(seed)
    texts = [sample.text for sample in generate(seed=seed)]
    for path in sorted(FP_CORPUS.iterdir()):
        if path.suffix != ".toml" and path.name != "README.md":
            texts += path.read_text(encoding="utf-8").splitlines()
    return texts + list(_generated(rng))


@pytest.mark.parametrize("seed", [0, 1])
def test_nothing_1_12_1_redacted_or_refused_is_forwarded(seed: int) -> None:
    rng = random.Random(seed)
    nlp = FakeSpacy([])
    # The digit-group rule (CUSTOM_RULES[2]) in half of them: it takes in
    # whole runs of groups, which hides what the others do.
    configs = [
        DetectionConfig(custom_rules=rules, deny_strings=deny)
        for rules in (CUSTOM_RULES, CUSTOM_RULES[:2] + CUSTOM_RULES[3:])
        for deny in ((), (DenyEntry("Main"),))
    ]
    built = [(config, build_allowlist(config), *_pipelines(config, nlp)) for config in configs]
    lost = []
    checked = joined = cut_without_joining = refused_now = random_modes = 0
    for index, text in enumerate(_texts(seed)):
        config, allowlist, release, previous = built[index % 2 * 2 + (index % 7 == 0)]
        nlp.findings = []
        joining = [d for d in release.detect(text, allowlist) if d.joins]
        nlp.findings = [(s, label, 1.0) for s, label in _crossing(rng, text, joining)]
        modes: dict[str, str] = {}
        if index % 3 == 0:
            random_modes += 1
            modes = {t: rng.choice(("redact", "warn", "block")) for t in rng.sample(TYPES, 3)}
        vault = InMemoryVault()
        redactor = Redactor(release, vault, allowlist, modes=modes)
        before = _by_1_12_1(previous, allowlist, text, modes)
        now = _by_release(redactor, vault, text)
        checked += 1
        joined += bool(joining)
        if now is None:
            refused_now += before is not None
            continue
        if before is None or before - now:
            lost.append((text, modes, None if before is None else sorted(before - now)))
            continue
        detections = release.detect(text, allowlist)
        # Without joining, this release's own detections would lose some.
        swept = _by_1_12_1(release, allowlist, text, modes)
        cut_without_joining += swept is not None and bool(before - swept)
        # A joining value that redacts is redacted whole.
        for d in detections:
            if d.joins and modes.get(d.detector_type, "redact") == "redact":
                assert set(range(d.start, d.end)) <= now, text
    assert lost == []
    # Not vacuous: joining values in thousands of texts, values the sweep
    # would cut without joining, requests refused for a block-mode value
    # joined with one.
    assert checked > 15_000 and joined > 3_000 and random_modes > 5_000
    assert cut_without_joining > 100 and refused_now > 100
