"""A card written next to another digit group is redacted.

1.12.1 forwarded these unredacted: the credit_card regex reads up to 19
digits in groups, so a card with a quantity before it or a CVV after it was
read together with that group, failed the Luhn check, and the scan went on
after it. Since 1.13.0 a vetoed candidate is looked inside
(regex_rules._card_inside, RegexRule.inside): a Luhn-valid stretch from a
group start to a group end, in a card's own grouping, ending where no other
rule's value can start inside it and run on. That nothing 1.12.1 redacted
is lost is test_no_loss_since_1_12_0.py (credit_card is frozen there).
"""

import dataclasses
import random
from collections.abc import Callable

import pytest

from llm_redact.bench.corpus import VALUE_GENERATORS
from llm_redact.detection.engine import DetectionConfig, DetectorPlan, build_detectors
from llm_redact.detection.regex_rules import BUILTIN_RULES, RegexDetector
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault
from test_value_punctuation import ALLOWLIST, DETECTORS, _seconds, redacted

RULE = next(rule for rule in BUILTIN_RULES if rule.name == "credit_card")
CARD = RegexDetector(RULE)
# The rule as 1.12.1 ran it: finditer, the Luhn check, nothing looked inside.
CARD_1_12_1 = RegexDetector(dataclasses.replace(RULE, inside=None))
AADHAAR = VALUE_GENERATORS["aadhaar"][1](random.Random(5))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The reported shapes: a group before the card, a group after it.
        ("Order 12 4111 1111 1111 1111", "Order 12 «CREDIT_CARD_001»"),
        ("card 4111 1111 1111 1111 12", "card «CREDIT_CARD_001» 12"),
        # A year before it: the candidate stops at its 19th digit, and the
        # card runs on past it.
        ("Order 2024 4111 1111 1111 1111", "Order 2024 «CREDIT_CARD_001»"),
        ("Order 1234 5678 4111 1111 1111 1111", "Order 1234 5678 «CREDIT_CARD_001»"),
        ("Item 3-4111 1111 1111 1111", "Item 3-«CREDIT_CARD_001»"),
        ("Order 12 4111111111111111", "Order 12 «CREDIT_CARD_001»"),
        # A CVV or an expiry month after it, both sides at once.
        ("card 4111111111111111 123", "card «CREDIT_CARD_001» 123"),
        ("exp: 4111 1111 1111 1111 12/27", "exp: «CREDIT_CARD_001» 12/27"),
        ("card 4111-1111-1111-1111-123.", "card «CREDIT_CARD_001»-123."),
        ("Order 12 4111 1111 1111 1111 123", "Order 12 «CREDIT_CARD_001» 123"),
        # American Express's 4-6-5, with a group before or a CID after.
        ("Amex 1 3782 822463 10005", "Amex 1 «CREDIT_CARD_001»"),
        ("Amex 3782 822463 10005 1234", "Amex «CREDIT_CARD_001» 1234"),
        # At the end of a sentence, a clause, a JSON string.
        ("Order 12 4111 1111 1111 1111.", "Order 12 «CREDIT_CARD_001»."),
        ("Order 12 4111 1111 1111 1111: paid", "Order 12 «CREDIT_CARD_001»: paid"),
        ('{"card": "12 4111 1111 1111 1111"}', '{"card": "12 «CREDIT_CARD_001»"}'),
    ],
)
def test_a_card_next_to_a_digit_group_is_redacted(text: str, expected: str) -> None:
    assert redacted(text) == expected
    # 1.12.1 forwarded each of them.
    assert list(CARD_1_12_1.detect(text)) == []


@pytest.mark.parametrize(
    "text",
    [
        "Order 12 4111 1111 1111 1112",  # no stretch passes the check
        "Order 12 41 11 11 11 11 11 11 11",  # not a card's grouping
        "Order 12 4111 1111-1111 1111",  # two separators
        "x4111 1111 1111 1111 12",  # its first group glued to a word
        "Order 12 4111 1111 1111 1111.5",  # glued to a number
        "Order 12 4111 1111 1111 1111_x",
        "card 4111 1111 1111 1111 12-x",
    ],
)
def test_near_misses_stay_unredacted(text: str) -> None:
    assert redacted(text) == text


@pytest.mark.parametrize(
    "text",
    [
        # Two groups after the card, or one that makes the two more than
        # 19 digits: an Aadhaar number (4-4-4) could start inside the card
        # and run on past it. docs/detection.md lists these residuals.
        "card 4111 1111 1111 1111 12 34",
        "Order 12 4111 1111 1111 1111 5678",
        # An email address could start in its last group.
        "Order 12 4111 1111 1111 1111@example.com",
    ],
)
def test_the_documented_residuals_are_not_read(text: str) -> None:
    assert "«CREDIT_CARD" not in redacted(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("call 212-555-0100 4111 1111 1111 1111", "call «PHONE_001» «CREDIT_CARD_001»"),
        (
            "+1 212 555 0100 card 4111 1111 1111 1111 12",
            "«PHONE_001» card «CREDIT_CARD_001» 12",
        ),
        (
            "IBAN GB82 WEST 1234 5698 7654 32, card 12 4111 1111 1111 1111",
            "IBAN «IBAN_001», card 12 «CREDIT_CARD_001»",
        ),
        # A spaced IBAN's digit groups hold card-shaped runs: the IBAN
        # starts first and is redacted whole.
        ("pay DE89 3704 0044 0532 0130 00 now", "pay «IBAN_001» now"),
        (f"{AADHAAR} 4111 1111 1111 1111", "«AADHAAR_001» «CREDIT_CARD_001»"),
        (f"4111 1111 1111 1111 {AADHAAR}", "«CREDIT_CARD_001» «AADHAAR_001»"),
        # The card is not read before an Aadhaar number's three groups (its
        # last two and the next one could be one), and the Aadhaar number
        # stays redacted whole, as in 1.12.1.
        (f"12 4111 1111 1111 1111 {AADHAAR}", "12 4111 1111 1111 1111 «AADHAAR_001»"),
        ("4111 1111 1111 1111 046 454 286", "4111 1111 1111 1111 «CA_SIN_001»"),
        ("card 4111 1111 1111 1111 943 476 5919", "card 4111 1111 1111 1111 «NHS_NUMBER_001»"),
    ],
)
def test_other_values_beside_a_card(text: str, expected: str) -> None:
    assert redacted(text) == expected


def _runs(seed: int, count: int) -> list[str]:
    """Runs of 13 to 40 random digits in groups of one to four, separated
    by one space or one hyphen."""
    rng = random.Random(seed)
    texts = []
    for _ in range(count):
        groups, left = [], rng.randint(13, 40)
        while left:
            size = min(left, rng.randint(1, 4))
            groups.append("".join(rng.choice("0123456789") for _ in range(size)))
            left -= size
        texts.append("ref " + rng.choice(" -").join(groups) + " done")
    return texts


def test_what_1_12_1_found_is_found_as_before() -> None:
    # The scan is finditer's, and what the check accepted alone is found
    # exactly as before: only what is found inside a vetoed candidate is
    # added, and it never overlaps another detection of the rule.
    texts = _runs(11, 3000)
    rng = random.Random(12)
    texts += [
        "4111 1111 1111 1111 " * rng.randint(1, 4) + rng.choice(("", "12", "1 2"))
        for _ in range(200)
    ]
    added = 0
    for text in texts:
        now, before = list(CARD.detect(text)), list(CARD_1_12_1.detect(text))
        assert set(before) <= set(now), text
        added += len(now) - len(before)
        spans = sorted((d.start, d.end) for d in now)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False)), text
    assert added > 0


def test_random_digit_runs_hardly_gain_false_positives() -> None:
    # The grouping and the end the stretch must have keep the added false
    # positives small: measured on 5,000 such runs, 13.58% of them held a
    # Luhn-valid candidate before and 13.78% now; without the grouping
    # requirement it would be 60.40%.
    texts = _runs(7, 3000)
    now = sum(bool(list(CARD.detect(text))) for text in texts)
    before = sum(bool(list(CARD_1_12_1.detect(text))) for text in texts)
    assert before > 300  # Luhn passes one random run in ten
    assert now - before <= len(texts) // 200


def test_the_detector_plan_reads_cards_inside_runs() -> None:
    # credit_card declares no prefilter literal: the plan's combined
    # alternation finds the candidate, and the plan runs the rule's detect.
    plan = DetectorPlan(build_detectors(DetectionConfig()))
    text = "Order 12 4111 1111 1111 1111"
    assert [d.value for d in plan.detect(text, ALLOWLIST)] == ["4111 1111 1111 1111"]
    assert plan.detect(text, ALLOWLIST) == plan.detect_each(text, ALLOWLIST)


# --- cost ---------------------------------------------------------------------

DIGIT_RUNS = (
    "1 ", "1-", "12 ", "1 2 ", "1234 ", "12345 ", "1111 1111 1111 1111 ",
    "4111 1111 1111 1111 12 ", "12 4111 1111 1111 1111 ", "1 4111111111111111 ",
    "1234 123456 12345 ", "4111 1111 1111 1111 12.",
)  # fmt: skip


def test_no_adversarial_digit_run_costs_more_than_1_12_1s_worst() -> None:
    # The rule is linear on every run (ten times the input, under thirty
    # times the time: about ten), and the whole Redactor takes on each less
    # than twice what the 1.12.1 pipeline took on the slowest of them:
    # measured, the slowest ties with it. Over that bound both are timed
    # again, so a burst of other load does not fail it.
    from test_no_loss_since_1_12_0 import OLD_PLAN

    release = Redactor(DETECTORS, InMemoryVault(), ALLOWLIST).redact_text
    before = Redactor(OLD_PLAN, InMemoryVault(), ALLOWLIST).redact_text
    scan: Callable[[str], object] = lambda text: list(CARD.detect(text))  # noqa: E731

    def reference() -> float:
        return max(_seconds(before, unit) for unit in DIGIT_RUNS)

    worst = reference()
    for unit in DIGIT_RUNS:
        small, large = _seconds(scan, unit, 10_000), _seconds(scan, unit)
        assert large < 30 * max(small, 1e-4), (unit, small, large)
        large = _seconds(release, unit)
        if large >= 2 * worst:
            worst, large = reference(), _seconds(release, unit)
        assert large < 2 * worst, (unit, large, worst)
