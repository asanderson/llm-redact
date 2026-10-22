"""A value overlapping a newly read one is joined with it, never cut.

1.13.0 reads a card inside a longer run of digit groups
(regex_rules._card_inside), and 1.12.1 the IBAN's paper format and a value
ending before a run of dots (belgian_nn, swiss_ahv, ipv6 — also before one
":" — and a PlanetScale token's run past 64 characters): values no earlier
release read, right next to text another value may hold. A custom rule's
value or an NER span starting inside such a value and running on past it
lost its tail to it, since the earlier-starting value won the start-ordered
overlap sweep (redactor._resolve_overlaps). These readings now JOIN
(Detection.joins): in a run of values overlapping one, directly or through
others, Redactor._join redacts every value that redacts (values overlapping
one another as one placeholder over their union, typed after the first),
refuses for every value in block mode and forwards a warn-mode value only
where nothing redacted covers it (counted as a warning when part of it goes
upstream, once however many detectors read it). Where
the proxy cannot rewrite the text (Redactor.scan), a union is put to the
requester's overrides value by value.

The differential at the end pins it against 1.12.1's and 1.12.0's pipelines
over the recall and fp corpora and generated texts, with the same custom
rules and fake NER spans (ner_fakes.FakeSpacy): every character either
redacted stays redacted, every request either refused is refused, under the
default modes and random ones.
"""

import random
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
from hypothesis import example, given, settings
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
from test_no_loss_since_1_12_0 import (
    ENDINGS,
    FROZEN_1_12_0,
    JOINERS,
    Frozen,
    as_1_12_1,
    card_run,
    luhn_card,
)
from warn_oracle import Every, expected_warnings

CARD = "4111 1111 1111 1111"
# A paper IBAN whose digits after the bank code are a Luhn-valid card number.
IBAN = "GB39 WEST 3607 5983 8675 65"
IBAN_CARD = "3607 5983 8675 65"
STREET = CustomRule(name="street", detector_type="ADDRESS", pattern=r"\d{2,4} [A-Z][a-z]+ Street")
ORDER = CustomRule(name="order", detector_type="ORDER_REF", pattern=r"Order \d{1,4} \d{4}")
# A value's last characters, a run of dots or one ":", and a capitalized
# word: starts inside a value ending before a run of dots and runs on.
TAIL = CustomRule(
    name="tail", detector_type="TAIL", pattern=r"[0-9A-Za-z]{1,4}(?:\.+|:) [A-Z][a-z]+"
)
# A PlanetScale token whose body is 64 characters: with the dots after it, a
# run past 64 (read since 1.12.1).
TOKEN64 = "pscale_tkn_" + "a1B2c3D4" * 8
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


QTY = CustomRule(name="qty", detector_type="QTY", pattern=r"Qty \d+ ")
LEAD = CustomRule(name="lead", detector_type="LEAD", pattern=r"\d+ 4111")
TAIL_1111 = CustomRule(name="tail", detector_type="TAIL", pattern=r"1111 Main")
REST = CustomRule(name="rest", detector_type="REST", pattern=r"(?<=\d) Main Street")


@pytest.mark.parametrize(
    ("rules", "warned"),
    [
        # Overlapping the card: each yields to it and is still counted (what
        # lies outside the card goes upstream).
        ((LEAD, TAIL_1111), Counter({"LEAD": 1, "TAIL": 1})),
        # Only touching it (ending where it starts, starting where it ends):
        # forwarded whole and counted.
        ((QTY, REST), Counter({"QTY": 1, "REST": 1})),
        # All four: the lead overlaps the quantity and the tail the rest, each
        # pair forwarded as sent, so of each the one the sweep reads first
        # counts — as without the card.
        ((QTY, LEAD, TAIL_1111, REST), Counter({"QTY": 1, "TAIL": 1})),
    ],
)
def test_a_forwarded_value_overlapping_a_redacted_one_yields_and_is_counted(
    rules: tuple[CustomRule, ...], warned: Counter[str]
) -> None:
    modes = dict.fromkeys(("QTY", "LEAD", "TAIL", "REST"), "warn")
    redactor = _redactor(*rules, modes=modes)
    text = f"Qty 12 {CARD} Main Street"
    assert _redacted(redactor, text) == "Qty 12 «CREDIT_CARD_001» Main Street"
    assert redactor.warn_counts == warned
    # A check ahead of the redaction counts nothing.
    redactor.warn_counts.clear()
    assert redactor.blocked_type(text) is None and redactor.warn_counts == Counter()
    # Nor does a scan count a value twice: once per value.
    assert redactor.scan(text, redactable=True).found == Counter({"CREDIT_CARD": 1})
    assert redactor.warn_counts == warned


def test_a_value_forwarded_as_sent_touching_the_union_stays_a_winner() -> None:
    # In a run joined around J (redacted), an approved block-mode value K
    # ending where J starts is clear of it: a winner, so the redaction puts
    # it to the overrides once more, as every block-mode winner; W, which
    # overlaps J, yields to it.
    text = "ABCDEFGHIJKLMNOP"
    k = Detection(0, 4, "K", text[0:4])
    w = Detection(2, 6, "W", text[2:6])
    j = Detection(4, 10, "J", text[4:10], joins=True)
    approved = Approved(k.value)
    redactor = Redactor(
        [_Given(k, w, j)], InMemoryVault(), Allowlist(), modes={"K": "block", "W": "warn"}
    ).with_overrides(approved)
    assert redactor.redact_text(text) == "ABCD«J_001»KLMNOP"
    assert approved.asked == [("K", "ABCD"), ("K", "ABCD")]
    # Of the values sent, K is read first and W overlaps it: no warning.
    assert redactor.warn_counts == Counter()


def test_a_warn_mode_value_overlapping_two_redacted_ones_is_counted_once() -> None:
    # One warn-mode value across the card and an address apart from it:
    # both redacted, the value dropped, counted once.
    span = CustomRule(name="span", detector_type="SPAN", pattern=r"12 4111.*Street")
    redactor = _redactor(span, STREET, modes={"SPAN": "warn"})
    text = f"Order 12 {CARD} via 22 Main Street"
    assert _redacted(redactor, text) == "Order 12 «CREDIT_CARD_001» via «ADDRESS_001»"
    assert redactor.warn_counts == Counter({"SPAN": 1})
    # An approved block-mode value dropped alike is no warning.
    value = f"12 {CARD} via 22 Main Street"
    approved = Approved(value)
    blocked = _redactor(span, STREET, modes={"SPAN": "block"}).with_overrides(approved)
    assert _redacted(blocked, text) == "Order 12 «CREDIT_CARD_001» via «ADDRESS_001»"
    assert blocked.warn_counts == Counter() and approved.asked == [("SPAN", value)]


AHV_DOT = "AHV 756.9217.0769.85. Done"
# Redacted where it touches the AHV number (starting where the number ends).
DOT = CustomRule(name="dot", detector_type="DOT", pattern=r"\. Done")


@pytest.mark.parametrize(
    ("text", "pattern", "rules", "expected", "warned"),
    [
        # Wholly inside the joined value: nothing of it goes upstream, so
        # it is no warning — as with the same number read without the
        # sentence's dot (1.12.0's reading, which joins nothing).
        (AHV_DOT, r"\b9217\b", (), "AHV «CH_AHV_001». Done", 0),
        ("AHV 756.9217.0769.85 Done", r"\b9217\b", (), "AHV «CH_AHV_001» Done", 0),
        (f"pay {IBAN} now", r"\b5983\b", (), "pay «IBAN_001» now", 0),
        # Starting where it starts, ending where it ends.
        (AHV_DOT, r"\b756\b", (), "AHV «CH_AHV_001». Done", 0),
        (AHV_DOT, r"\b85\b", (), "AHV «CH_AHV_001». Done", 0),
        # Inside two redacted values touching each other.
        (AHV_DOT, r"85\. Do", (DOT,), "AHV «CH_AHV_001»«DOT_001»", 0),
        (AHV_DOT, r"85\. Done", (DOT,), "AHV «CH_AHV_001»«DOT_001»", 0),
        # A character past them is part of it going upstream: counted.
        (AHV_DOT, r" 756\b", (), "AHV «CH_AHV_001». Done", 1),
        (AHV_DOT, r"\b85\.", (), "AHV «CH_AHV_001». Done", 1),
        (f"{AHV_DOT}e", r"85\. Donee", (DOT,), "AHV «CH_AHV_001»«DOT_001»e", 1),
    ],
)
def test_a_warn_mode_value_inside_the_redacted_values_is_no_warning(
    text: str, pattern: str, rules: tuple[CustomRule, ...], expected: str, warned: int
) -> None:
    pin = CustomRule(name="pin", detector_type="PIN", pattern=pattern)
    redactor = _redactor(pin, *rules, modes={"PIN": "warn"})
    assert any(d.detector_type == "PIN" for d in redactor.plan.detect(text, Allowlist()))
    expect = Counter({"PIN": warned} if warned else {})
    assert _redacted(redactor, text) == expected
    assert redactor.warn_counts == expect
    # A scan reads it alike; a check ahead of the redaction counts nothing.
    redactor.warn_counts.clear()
    redactor.scan(text, redactable=True)
    assert redactor.warn_counts == expect
    redactor.warn_counts.clear()
    assert redactor.blocked_type(text) is None and redactor.warn_counts == Counter()


@pytest.mark.parametrize(
    ("rules", "ner"),
    [
        # Two custom rules reading the same address.
        (
            (STREET, CustomRule(name="street2", detector_type="ADDRESS", pattern=STREET.pattern)),
            (),
        ),
        # A custom rule and an NER model reading it.
        ((STREET,), (("1111 Main Street", "ADDRESS"),)),
        # Two values overlapping each other: the one the sweep keeps.
        ((STREET, CustomRule(name="part", detector_type="PART", pattern=r"1111 Main")), ()),
    ],
)
def test_one_warn_mode_value_two_detectors_find_is_counted_once(
    rules: tuple[CustomRule, ...], ner: tuple[tuple[str, str], ...]
) -> None:
    # A warn-mode value a joined run drops counts as the sweep without the
    # card counts it: once.
    modes = {"ADDRESS": "warn", "PART": "warn"}
    redactor = _redactor(*rules, ner=ner, modes=modes)
    text = f"Order 12 {CARD} Main Street"
    assert _redacted(redactor, text) == "Order 12 «CREDIT_CARD_001» Main Street"
    assert redactor.warn_counts == Counter({"ADDRESS": 1})
    alone = _redactor(*rules, ner=ner, modes=modes)
    assert _redacted(alone, "Ship to 1111 Main Street") == "Ship to 1111 Main Street"
    assert alone.warn_counts == Counter({"ADDRESS": 1})


def test_with_an_overlay_a_forwarded_value_never_takes_the_card() -> None:
    # An access gate's added deny strings (Redactor._absorb) sweep the
    # winners again: a warn-mode value the joined run dropped stays dropped
    # (counted once, by the join).
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
    assert redactor.warn_counts == Counter({"PERSON": 1})


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


# --- values ending before a run of dots (1.12.1) -------------------------------------


@pytest.mark.parametrize(
    ("text", "rules", "ner", "expected"),
    [
        # A custom rule's value starting inside the value and running on
        # past its run of dots (1.12.0 read no value there, so it redacted
        # the custom one whole; 1.12.1 forwarded ". Done").
        ("AHV 756.9217.0769.85. Done", (TAIL,), (), "AHV «CH_AHV_001»"),
        ("NN 93.05.18-223.61... Paid", (TAIL,), (), "NN «BE_NN_001»"),
        ("gw 2001:db8::1: Unreachable", (TAIL,), (), "gw «IPV6_001»"),
        ("gw 2001:db8::1. Down", (TAIL,), (), "gw «IPV6_001»"),
        (f"token {TOKEN64}. Done", (TAIL,), (), "token «PLANETSCALE_TOKEN_001»"),
        # An NER span doing the same.
        (
            "AHV 756.9217.0769.85. Jane Doe",
            (),
            (("85. Jane", "PERSON"),),
            "AHV «CH_AHV_001» Doe",
        ),
        # Starting before the value and ending inside it: typed after it.
        ("Ship to 2001:db8::1. Done", (), (("to 2001", "ADDRESS"),), "Ship «ADDRESS_001». Done"),
        (
            f"user Jane {TOKEN64}.",
            (),
            (("Jane pscale", "PERSON"),),
            "user «PERSON_001»",
        ),
    ],
)
def test_a_value_overlapping_one_ending_before_dots_is_redacted_with_it(
    text: str, rules: tuple[CustomRule, ...], ner: tuple[tuple[str, str], ...], expected: str
) -> None:
    redactor = _redactor(*rules, ner=ner)
    detections = redactor.plan.detect(text, Allowlist())
    assert any(d.joins for d in detections)
    assert _redacted(redactor, text) == expected
    assert sum(redactor.counts.values()) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # What 1.12.0 read joins nothing: the custom value starting inside
        # it loses its tail, as in every release.
        ("AHV 756.9217.0769.85 Done", "AHV «CH_AHV_001» Done"),
        ("gw 2001:db8::1 Down", "gw «IPV6_001» Down"),
        # Nor a token whose body and dots stop at 64 characters.
        (f"token {TOKEN64[:-2]}.. Done", "token «PLANETSCALE_TOKEN_001» Done"),
    ],
)
def test_what_1_12_0_read_joins_nothing(text: str, expected: str) -> None:
    redactor = _redactor(
        CustomRule(name="tail", detector_type="TAIL", pattern=r"\w{2}\.* Done|1 Down")
    )
    assert not any(d.joins for d in redactor.plan.detect(text, Allowlist()))
    assert _redacted(redactor, text) == expected


def test_a_value_ending_before_dots_and_modes() -> None:
    text = "AHV 756.9217.0769.85. Done"
    # A block-mode value it overlaps refuses (1.12.1's sweep let the AHV
    # number take its place, and the block never fired).
    for blocked in ("TAIL", "CH_AHV"):
        redactor = _redactor(TAIL, modes={blocked: "block"})
        with pytest.raises(BlockedRequest, match=blocked):
            redactor.redact_text(text)
        assert redactor.blocked_type(text) == blocked
    # A warn-mode number is forwarded where the custom value does not cover
    # it, and counted.
    warned = _redactor(TAIL, modes={"CH_AHV": "warn"})
    assert _redacted(warned, text) == "AHV 756.9217.0769.«TAIL_001»"
    assert warned.warn_counts == Counter({"CH_AHV": 1})
    # A warn-mode custom value yields to the number, counted.
    warned = _redactor(TAIL, modes={"TAIL": "warn"})
    assert _redacted(warned, text) == "AHV «CH_AHV_001». Done"
    assert warned.warn_counts == Counter({"TAIL": 1})


def test_a_value_inside_a_planetscale_run_keeps_its_mode() -> None:
    # A generic secret read with the same span as a PlanetScale token's run
    # past 64 characters: the run joins, so the secret's block mode refuses
    # (1.12.1 let the token's mode govern it). A token of 64 characters
    # with its dots joins nothing: it wins the overlap, as in 1.12.0.
    blocking = _redactor(modes={"SECRET": "block"})
    with pytest.raises(BlockedRequest, match="SECRET"):
        blocking.redact_text(f"password = {TOKEN64}. Done")
    assert blocking.blocked_type(f"password = {TOKEN64}. Done") == "SECRET"
    assert _redacted(blocking, f"password = {TOKEN64} Done") == (
        "password = «PLANETSCALE_TOKEN_001» Done"
    )
    assert _redacted(_redactor(), f"password = {TOKEN64}. Done") == (
        "password = «PLANETSCALE_TOKEN_001» Done"
    )


def test_a_union_with_a_value_ending_before_dots_is_approved_value_by_value() -> None:
    redactor = _redactor(TAIL)
    text = "gw 2001:db8::1: Unreachable"
    members = [("IPV6", "2001:db8::1"), ("TAIL", "1: Unreachable")]
    approved = Approved("2001:db8::1")
    assert redactor.with_overrides(approved).scan(text) == (Counter({"IPV6": 1}), False)
    assert approved.asked == members
    approved = Approved("2001:db8::1", "1: Unreachable")
    assert redactor.with_overrides(approved).scan(text) == (Counter(), True)


# --- modes, deny strings, allowlists, overrides --------------------------------------


@pytest.mark.parametrize(
    ("modes", "expected", "warned"),
    [
        # A warn-mode card never keeps the other value from being redacted;
        # it is counted (what lies outside the address goes upstream).
        ({"CREDIT_CARD": "warn"}, "Order 12 4111 1111 1111 «ADDRESS_001»", {"CREDIT_CARD": 1}),
        # Nor a warn-mode value the card.
        ({"ADDRESS": "warn"}, "Order 12 «CREDIT_CARD_001» Main Street", {"ADDRESS": 1}),
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
    assert warned.warn_counts == Counter({"IBAN": 1})


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
    # Final (an idempotent flag: said before anything is asked, and again
    # at the union).
    assert approved.asked == [] and approved.final >= 1
    # A deny string inside the card: the card's span, still final.
    inside = _redactor(deny=("1111 1111",))
    assert _redacted(inside, f"Order 12 {CARD}") == "Order 12 «CREDIT_CARD_001»"
    approved = Approved(CARD)
    assert inside.with_overrides(approved).scan(f"Order 12 {CARD}").found == Counter(
        {"CREDIT_CARD": 1}
    )
    assert approved.asked == [] and approved.final >= 1
    # A deny string covering the card stays the deny string.
    covering = _redactor(deny=(f"12 {CARD}",))
    assert _redacted(covering, f"Order 12 {CARD}") == "Order «DENY_001»"


def test_a_joined_union_is_put_to_the_overrides_value_by_value() -> None:
    # In text the proxy cannot rewrite, a span joining a card and an
    # address is forwarded as sent only when both values are approved; each
    # is asked, none skipped, so a refusal's code names them all.
    redactor = _redactor(STREET)
    text = f"Order 12 {CARD} Main Street"
    both = [("CREDIT_CARD", CARD), ("ADDRESS", "1111 Main Street")]
    approved = Approved(CARD, "1111 Main Street")
    assert redactor.with_overrides(approved).scan(text) == (Counter(), True)
    assert redactor.with_overrides(approved).scan_text(text) == Counter()
    assert approved.asked == both * 2 and approved.final == 0
    # One of them approved: refused, both asked (the first refusal too).
    for one in (CARD, "1111 Main Street"):
        approved = Approved(one)
        scan = redactor.with_overrides(approved).scan(text)
        assert scan == (Counter({"CREDIT_CARD": 1}), False)
        assert approved.asked == both and approved.final == 0
    # Their union is no value an approval names.
    approved = Approved(f"{CARD} Main Street", f"12 {CARD} Main Street")
    assert redactor.with_overrides(approved).scan(text).found == Counter({"CREDIT_CARD": 1})
    # Convert mode redacts it: nothing asked, nothing made final.
    approved = Approved(CARD, "1111 Main Street")
    assert redactor.with_overrides(approved).scan(text, redactable=True).found == Counter(
        {"CREDIT_CARD": 1}
    )
    assert approved.asked == [] and approved.final == 0
    # Rewritable text redacts it whatever is approved.
    copy = redactor.with_overrides(approved)
    assert _redacted(copy, text) == "Order 12 «CREDIT_CARD_001»" and approved.asked == []


def test_a_union_holding_a_value_the_overlay_made_final_is_never_put_to_the_overrides() -> None:
    # A type whose block mode a detection overlay added refuses for good
    # (Redactor.final_blocks): no grant consulted, no code. Its block mode
    # refuses the run before any union forms...
    text = f"Order 12 {CARD} Main Street"
    approved = Approved(CARD, "1111 Main Street")
    config = DetectionConfig(custom_rules=(STREET,))
    blocking = Redactor(
        build_detectors(config),
        InMemoryVault(),
        build_allowlist(config),
        modes={"ADDRESS": "block"},
        final_blocks=frozenset({"ADDRESS"}),
    ).with_overrides(approved)
    with pytest.raises(BlockedRequest, match="ADDRESS"):
        blocking.scan(text)
    assert approved.asked == [] and approved.final == 1
    # ...and a union holding such a type is never approved either.
    approved = Approved(CARD, "1111 Main Street")
    union = Redactor(
        build_detectors(config),
        InMemoryVault(),
        build_allowlist(config),
        final_blocks=frozenset({"ADDRESS"}),
    ).with_overrides(approved)
    assert union.scan(text) == (Counter({"CREDIT_CARD": 1}), False)
    assert approved.asked == [] and approved.final == 1


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


def _replaced(winners: Iterable[Detection], modes: Mapping[str, str]) -> set[int]:
    """The offsets ``redact_text`` replaces of ``winners`` under ``modes``:
    a deny string's or a union's (tier 0), a redact-mode value's."""
    offsets: set[int] = set()
    for d in winners:
        if d.tier == 0 or modes.get(d.detector_type, "redact") == "redact":
            offsets.update(range(d.start, d.end))
    return offsets


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
    # 1.12.0 no paper IBAN; a configuration with fewer deny strings) — and
    # over all of them, where a deny string won every overlap until it was
    # joined with what it overlaps: every character it redacts is
    # redacted, every refusal it makes is made, every joining value that
    # redacts (a deny string always) is redacted whole; the winners are
    # disjoint and in order; and with no joining value the result is the
    # sweep's own.
    detections = _sorted(spans)
    dropped = data.draw(st.sets(st.sampled_from(range(len(detections))))) if detections else set()
    fewer = [d for i, d in enumerate(detections) if not (_joining(d) and i in dropped)]
    redactor = Redactor([], InMemoryVault(), Allowlist(), modes=modes)
    before = _redacts(_resolve_overlaps(fewer), modes)
    try:
        winners = redactor._resolved("x" * 60, detections)
    except BlockedRequest:
        return
    assert all(a.end <= b.start for a, b in zip(winners, winners[1:], strict=False))
    if not any(_joining(d) for d in detections):
        assert winners == _resolve_overlaps(detections)
    now = _redacts(winners, modes)
    if now is None:
        return
    assert before is not None and before <= now
    swept = _redacts(_resolve_overlaps(detections), modes)
    assert swept is not None and swept <= now
    for d in detections:
        if _joining(d) and (d.tier == 0 or modes.get(d.detector_type, "redact") == "redact"):
            assert set(range(d.start, d.end)) <= now


def _joining(d: Detection) -> bool:
    """Whether ``d`` is joined with what it overlaps: a value that says so,
    or a deny string."""
    return d.joins or d.tier == 0


LLM_SPAN = st.tuples(st.integers(0, 40), st.integers(1, 12), st.sampled_from(("A", "L")))
LLM_MODES = st.dictionaries(
    st.sampled_from(("A", "B", "C", "L")), st.sampled_from(("redact", "warn", "block"))
)


# A warn-mode value Q wins the sweep against a redact-mode R and another
# warn-mode value P; an llm finding L overlaps P past Q's end. Without the
# deny string inside R, the sweep keeps Q and redacts L's tail; with it,
# R joins the deny string, Q yields to their union, P is kept — and L's
# tail goes upstream with P (tests/test_deny.py's real-rule property has no
# llm finding).
RESIDUAL = (
    [
        (1, 19, "A", False, 0),
        (2, 6, "B", False, 0),
        (12, 13, "A", False, 0),
        (3, 2, "DENY", False, 0),
    ],
    [(18, 7, "L")],
    {"A": "warn"},
)


def test_the_llm_residual_is_real() -> None:
    spans, findings, modes = RESIDUAL
    text = "x" * 60
    detections = _with_findings(spans, findings)
    winners = Redactor([], InMemoryVault(), Allowlist(), modes=modes)._resolved(text, detections)
    now = _redacts(winners, modes)
    without_deny = [d for d in detections if d.tier]
    before = _redacts(_resolve_overlaps(without_deny), modes)
    assert now is not None and before is not None
    assert before - now == set(range(20, 25))


def _with_findings(
    spans: Iterable[tuple[int, int, str, bool, int]], findings: Iterable[tuple[int, int, str]]
) -> list[Detection]:
    """``spans`` (``_sorted``) and llm findings (``ADDITIVE_TIER``) in the
    detection plan's order."""
    text = "x" * 60
    llm = [
        Detection(a, a + n, kind, text[a : a + n], 120, ADDITIVE_TIER) for a, n, kind in findings
    ]
    return sorted(_sorted(spans) + llm, key=lambda d: (d.start, -(d.end - d.start), d.priority))


@settings(deadline=None, max_examples=500)
@example(*RESIDUAL)
@given(st.lists(SPAN, max_size=10), st.lists(LLM_SPAN, max_size=3), LLM_MODES)
def test_an_llm_finding_a_kept_warn_value_covers_is_the_one_residual(
    spans: list[tuple[int, int, str, bool, int]],
    findings: list[tuple[int, int, str]],
    modes: dict[str, str],
) -> None:
    # The llm backend's findings (ADDITIVE_TIER) are never joined: they are
    # cut around the winners, as everywhere — the values forwarded as sent
    # among them. The join may keep a different one of two overlapping
    # warn-mode values than the overlap sweep does, so a different part of
    # a finding goes upstream with that value. That is the one way the join
    # redacts less than the sweep, with or without the joining values (a
    # deny string included): every character lost lies inside an llm
    # finding AND inside a value the join forwards as sent; and a refusal
    # lost is one the sweep made for an llm finding alone. Against the sweep
    # over every detection, and over all of them but the joining ones.
    text = "x" * 60
    detections = _with_findings(spans, findings)
    fewer = [d for d in detections if not _joining(d)]
    redactor = Redactor([], InMemoryVault(), Allowlist(), modes=modes)
    try:
        winners = redactor._resolved(text, detections)
    except BlockedRequest:
        return
    now = _redacts(winners, modes)
    if now is None:
        return
    forwarded = {
        i
        for d in winners
        if d.tier and modes.get(d.detector_type, "redact") == "warn"
        for i in range(d.start, d.end)
    }
    llm = {i for d in detections if d.tier == ADDITIVE_TIER for i in range(d.start, d.end)}
    for swept in (fewer, detections):
        before = _redacts(_resolve_overlaps(swept), modes)
        if before is None:
            values = [d for d in swept if d.tier != ADDITIVE_TIER]
            assert _redacts(_resolve_overlaps(values), modes) is not None
            continue
        assert before - now <= llm & forwarded


@settings(deadline=None, max_examples=500)
@given(st.lists(SPAN, max_size=10), MODES)
def test_a_dropped_warn_mode_value_is_counted_only_when_part_of_it_goes_upstream(
    spans: list[tuple[int, int, str, bool, int]], modes: dict[str, str]
) -> None:
    # What the redaction counts as warnings: exactly what the rule predicts
    # (tests/warn_oracle.py) from the values found and what is replaced — a
    # warn-mode value a joined run drops, or a deny string or a value
    # redacted in its place wins an ordinary overlap against, counts when a
    # character of it is still sent, each span once however many detections
    # read it. A text a block refuses while resolving (a block-mode value in
    # a joined run) counts what it would with every block approved.
    text = "x" * 60
    detections = _sorted(spans)
    redactor = Redactor([], InMemoryVault(), Allowlist(), modes=modes)
    approving = Redactor([], InMemoryVault(), Allowlist(), modes=modes).with_overrides(Every())
    passing = approving._resolved(text, detections)
    try:
        winners = redactor._settled(text, detections, None, unrewritable=False)
    except BlockedRequest:
        winners = passing
    else:
        assert winners == passing
    assert redactor.warn_counts == expected_warnings(detections, modes, _replaced(winners, modes))


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


class _Given:
    """A stand-in detector finding exactly ``found`` in any text."""

    name = "given"

    def __init__(self, *found: Detection) -> None:
        self.found = found

    def detect(self, text: str) -> list[Detection]:
        return list(self.found)


def test_a_dropped_warn_value_an_llm_finding_covers_the_rest_of_is_no_warning() -> None:
    # A warn-mode value a joining one overlaps: its part outside the joining
    # value is sent — a warning — unless an llm finding (ADDITIVE_TIER, cut
    # around the winners) redacts that part, so nothing of it is sent.
    text = "x" * 30
    warn = Detection(5, 15, "A", text[5:15])
    joining = Detection(10, 20, "B", text[10:20], joins=True)
    person = Detection(0, 10, "PERSON", text[:10], 120, ADDITIVE_TIER)
    modes = {"A": "warn"}
    sent = Redactor([_Given(warn, joining)], InMemoryVault(), Allowlist(), modes=modes)
    assert sent.redact_text(text) == "x" * 10 + "«B_001»" + "x" * 10
    assert sent.warn_counts == Counter({"A": 1})
    covered = Redactor([_Given(warn, joining, person)], InMemoryVault(), Allowlist(), modes=modes)
    assert covered.redact_text(text) == "«PERSON_001»«B_001»" + "x" * 10
    assert covered.warn_counts == Counter()


def test_a_recorder_sees_the_same_joined_redaction() -> None:
    # A scan sink's recorder (Redactor.with_recorder) changes nothing the
    # redaction does, a joined run's union included.
    class Recorder:
        def __init__(self) -> None:
            self.texts: list[str] = []

        def record(self, text: str, found: object) -> None:
            self.texts.append(text)

    text = f"Order 12 {CARD} Main Street"
    recorder = Recorder()
    recorded = _redactor(STREET).with_recorder(recorder)
    assert _redacted(recorded, text) == _redacted(_redactor(STREET), text)
    assert _redacted(recorded, text) == "Order 12 «CREDIT_CARD_001»"
    assert recorder.texts == [text, text]


def test_a_recorder_counts_the_warn_values_a_joined_run_drops() -> None:
    # The recorded path (_recorded_winners) counts the warn-mode values a
    # joined run drops exactly as the plain one does.
    class Recorder:
        def record(self, text: str, found: object) -> None:
            pass

    rules = (QTY, LEAD, TAIL_1111)
    modes = {"QTY": "warn", "LEAD": "warn", "TAIL": "warn"}
    text = f"Qty 12 {CARD} Main Street"
    plain = _redactor(*rules, modes=modes)
    recorded = _redactor(*rules, modes=modes).with_recorder(Recorder())
    assert _redacted(recorded, text) == _redacted(plain, text)
    # The lead overlaps the quantity, both sent: one of them counts.
    assert recorded.warn_counts == plain.warn_counts == Counter({"QTY": 1, "TAIL": 1})


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
    # The warnings follow the rule: an llm finding read as found, a
    # warn-mode value one redacts the rest of not counted.
    assert redactor._settled(text, both, None, unrewritable=False) == winners
    assert redactor.warn_counts == expected_warnings(both, modes, _replaced(winners, modes))


# --- the differential --------------------------------------------------------------

CUSTOM_RULES = (
    STREET,
    ORDER,
    # Digit groups, which start inside cards and IBANs and run on past them.
    CustomRule(name="groups", detector_type="ACCOUNT", pattern=r"\b\d{2,4}(?:[ -]\d{2,4}){2,5}\b"),
    # An IBAN's last groups and a word.
    CustomRule(name="code", detector_type="CODE", pattern=r"\b[A-Z0-9]{2,4} \d{2,4} [a-z]{3,}\b"),
    # A value's last characters before a run of dots or one ":", and a word.
    TAIL,
)
TYPES = (
    "CREDIT_CARD", "IBAN", "CH_AHV", "BE_NN", "IPV6", "PLANETSCALE_TOKEN", "ADDRESS", "ORDER_REF",
    "ACCOUNT", "CODE", "TAIL", "PERSON", "DENY",
)  # fmt: skip
WORDS = (" Main Street", " now", " 12 Main Street", "-x", " Jane Doe", ", paid", " Rue 3")
# The types of the values 1.12.1 reads before a run of dots.
DOTTED_TYPES = frozenset({"CH_AHV", "BE_NN", "IPV6", "PLANETSCALE_TOKEN"})
# What may follow such a run (regex_rules._CLOSE: "" goes on into WORDS),
# and what may not (".x", "-").
CLOSES = ("", "", " ", '"', ")", "**", "<br>", "\\n", ".x", "-")
ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _pipelines(
    config: DetectionConfig, nlp: FakeSpacy
) -> tuple[DetectorPlan, DetectorPlan, DetectorPlan]:
    """This release's detectors, 1.12.1's (its rules without the fields
    1.13.0 added, test_no_loss_since_1_12_0.as_1_12_1) and 1.12.0's (the six
    rules changed since frozen as 1.12.0 defined them,
    test_no_loss_since_1_12_0.Frozen), each with the same custom rules, deny
    strings and fake NER backend."""
    release: list[Detector] = list(build_detectors(config))
    previous: list[Detector] = [
        RegexDetector(as_1_12_1(det.rule)) if isinstance(det, RegexDetector) else det
        for det in release
    ]
    old: list[Detector] = [
        Frozen(det.name) if isinstance(det, RegexDetector) and det.name in FROZEN_1_12_0 else det
        for det in release
    ]
    ner = NerDetector(nlp, frozenset({"PERSON", "ADDRESS"}), 1_000_000)
    return DetectorPlan([*release, ner]), DetectorPlan([*previous, ner]), DetectorPlan([*old, ner])


def _by_sweep(
    plan: DetectorPlan, allowlist: Allowlist, text: str, modes: Mapping[str, str]
) -> set[int] | None:
    """What 1.12.1's or 1.12.0's Redactor redacted in ``text`` (offsets)
    with ``plan``, transcribed: the overlap sweep's winners, each by its
    mode — None when a block-mode winner refused the text."""
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
    """Cards in runs of digit groups (some before an expiry), paper IBANs
    and values before a run of dots (a PlanetScale token's run past 64
    characters among them), next to words and values the custom rules and
    NER spans take in."""
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
        # A value before a run of dots (an IPv6 address: or one ":"), then
        # what may, or may not, follow such a run.
        name = rng.choice(("swiss_ahv", "belgian_nn", "ipv6"))
        dots = rng.choice((".", "...", ":") if name == "ipv6" else (".", ".."))
        before = rng.choice(("", "AHV ", "NN ", "gw ", "Jane ", "Order 12 "))
        dotted = before + VALUE_GENERATORS[name][1](rng) + dots
        yield dotted + rng.choice(CLOSES) + rng.choice(WORDS + ENDINGS)
        body = "".join(rng.choice(ALNUM) for _ in range(64))
        token = f"pscale_{rng.choice(('tkn', 'oauth', 'pw'))}_{body}" + "." * rng.randrange(1, 4)
        yield rng.choice(("", "token ", "Jane ")) + token + rng.choice(CLOSES) + rng.choice(WORDS)


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
    checked = joined = joined_dotted = cut_without_joining = refused_now = random_modes = 0
    lost_by_1_12_1 = 0
    for index, text in enumerate(_texts(seed)):
        config, allowlist, release, previous, oldest = built[index % 2 * 2 + (index % 7 == 0)]
        nlp.findings = []
        joining = [d for d in release.detect(text, allowlist) if d.joins]
        nlp.findings = [(s, label, 1.0) for s, label in _crossing(rng, text, joining)]
        modes: dict[str, str] = {}
        if index % 3 == 0:
            random_modes += 1
            modes = {t: rng.choice(("redact", "warn", "block")) for t in rng.sample(TYPES, 3)}
        vault = InMemoryVault()
        redactor = Redactor(release, vault, allowlist, modes=modes)
        before = _by_sweep(previous, allowlist, text, modes)
        old = _by_sweep(oldest, allowlist, text, modes)
        now = _by_release(redactor, vault, text)
        checked += 1
        joined += bool(joining)
        joined_dotted += any(d.detector_type in DOTTED_TYPES for d in joining)
        if old is not None and before is not None:
            # What 1.12.1's sweep forwarded of what 1.12.0 redacted.
            lost_by_1_12_1 += bool(old - before)
        if now is None:
            refused_now += before is not None
            continue
        if before is None or old is None or (before | old) - now:
            lost.append(
                (text, modes, None if None in (before, old) else sorted((before | old) - now))
            )
            continue
        detections = release.detect(text, allowlist)
        # Without joining, this release's own detections would lose some.
        swept = _by_sweep(release, allowlist, text, modes)
        cut_without_joining += swept is not None and bool(before - swept)
        # A joining value that redacts is redacted whole.
        for d in detections:
            if d.joins and modes.get(d.detector_type, "redact") == "redact":
                assert set(range(d.start, d.end)) <= now, text
    assert lost == []
    # Not vacuous: joining values in thousands of texts (values before a
    # run of dots in thousands too), values the sweep would cut without
    # joining, requests refused for a block-mode value joined with one, and
    # values 1.12.0 redacted that 1.12.1 forwarded.
    assert checked > 15_000 and joined > 3_000 and random_modes > 5_000
    assert joined_dotted > 1_000
    assert cut_without_joining > 100 and refused_now > 100 and lost_by_1_12_1 > 100


# Values 1.12.1 reads before a run of dots (an IPv6 address: or one ":"; a
# PlanetScale token with its run past 64 characters), and the pipelines of
# the property below: a custom rule taking in a value's last characters and
# a word after them, an address rule, and a fake NER backend.
DOTTED = ("756.9217.0769.85", "93.05.18-223.61", "2001:db8::1", "2001:db8:1a2b::", TOKEN64)
_DOTTED_CONFIG = DetectionConfig(custom_rules=(TAIL, STREET))
_DOTTED_NLP = FakeSpacy([])
_DOTTED_PLANS = _pipelines(_DOTTED_CONFIG, _DOTTED_NLP)


@settings(deadline=None, max_examples=400)
@given(
    st.sampled_from(("", "AHV ", "Jane ", "x ", "Order 12 ")),
    st.sampled_from(DOTTED),
    st.sampled_from(("", ".", "...", ":", "::")),
    st.sampled_from(CLOSES),
    st.sampled_from(WORDS + (" Done", " 12 Main Street Jane")),
    st.lists(
        st.tuples(st.integers(0, 110), st.integers(1, 30), st.sampled_from(("PERSON", "ADDRESS"))),
        max_size=3,
    ),
    st.dictionaries(st.sampled_from(TYPES), st.sampled_from(("redact", "warn", "block"))),
)
def test_a_value_before_a_run_of_dots_never_takes_another_values_characters(
    before: str,
    value: str,
    dots: str,
    close: str,
    word: str,
    spans: list[tuple[int, int, str]],
    modes: dict[str, str],
) -> None:
    # A value 1.12.1 reads before a run of dots, next to custom rule values
    # and NER spans anywhere, under any modes: every character 1.12.0 or
    # 1.12.1 redacted is redacted, every text either refused is refused, and
    # a joining value that redacts is redacted whole.
    release, previous, oldest = _DOTTED_PLANS
    allowlist = build_allowlist(_DOTTED_CONFIG)
    text = before + value + dots + close + word
    _DOTTED_NLP.findings = [
        (text[a : a + n], label, 1.0) for a, n, label in spans if text[a : a + n].strip()
    ]
    vault = InMemoryVault()
    now = _by_release(Redactor(release, vault, allowlist, modes=modes), vault, text)
    if now is None:
        return
    old = _by_sweep(oldest, allowlist, text, modes)
    earlier = _by_sweep(previous, allowlist, text, modes)
    assert old is not None and earlier is not None
    assert old | earlier <= now
    for d in release.detect(text, allowlist):
        if d.joins and modes.get(d.detector_type, "redact") == "redact":
            assert set(range(d.start, d.end)) <= now
