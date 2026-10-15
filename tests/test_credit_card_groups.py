"""A card written next to another digit group is redacted.

1.12.1 forwarded these unredacted: the credit_card regex reads up to 19
digits in groups, so a card with a quantity before it, a CVV or an expiry
after it was read together with those groups, failed the Luhn check, and the
scan went on after it. Since 1.13.0 a vetoed candidate is looked inside
(regex_rules._card_inside, RegexRule.inside): a Luhn-valid stretch from a
group start to a group end, in a card's own grouping, ending where no other
built-in rule's value can start inside it and run on (where the run ends,
before one group, or before an expiry: test_no_rule_runs_on_from_inside_a_card
tries every built-in pattern). That nothing 1.12.1 redacted is lost is
test_no_loss_since_1_12_0.py (credit_card is frozen there, and 1.12.1's
pipeline is checked too); a custom rule's value or an NER span that overlaps
such a card is joined with it (test_joined_values.py).
"""

import dataclasses
import random
import re
from collections.abc import Callable, Iterator

import pytest

from llm_redact.bench.corpus import VALUE_GENERATORS
from llm_redact.detection.engine import DetectionConfig, DetectorPlan, build_detectors
from llm_redact.detection.regex_rules import (
    _CARD_END,
    _CARD_GROUPING,
    BUILTIN_RULES,
    RegexDetector,
    _card_inside,
)
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
    ("text", "expected"),
    [
        # An expiry after the card: a month, then a year of two or four
        # digits, each after the card's own separator.
        ("card 4111 1111 1111 1111 12 27", "card «CREDIT_CARD_001» 12 27"),
        ("card 4111-1111-1111-1111-12-27", "card «CREDIT_CARD_001»-12-27"),
        ("card 4111 1111 1111 1111 01 2031", "card «CREDIT_CARD_001» 01 2031"),
        ("card 4111-1111-1111-1111-09-2031", "card «CREDIT_CARD_001»-09-2031"),
        # A solid card's expiry takes either separator, the same twice.
        ("card 4111111111111111 12 27", "card «CREDIT_CARD_001» 12 27"),
        ("card 4111111111111111-12-2027", "card «CREDIT_CARD_001»-12-2027"),
        # American Express and Diners Club; a group before the card too.
        ("Amex 3782 822463 10005 12 27", "Amex «CREDIT_CARD_001» 12 27"),
        ("Diners 3056 930902 5904 10 29", "Diners «CREDIT_CARD_001» 10 29"),
        ("Order 12 4111 1111 1111 1111 12 27", "Order 12 «CREDIT_CARD_001» 12 27"),
        # At the end of a sentence, a clause, a JSON string.
        ("card 4111 1111 1111 1111 12 27.", "card «CREDIT_CARD_001» 12 27."),
        ("4111 1111 1111 1111 12 27: paid", "«CREDIT_CARD_001» 12 27: paid"),
        ('{"card": "4111 1111 1111 1111 12 27"}', '{"card": "«CREDIT_CARD_001» 12 27"}'),
    ],
)
def test_a_card_before_an_expiry_is_redacted(text: str, expected: str) -> None:
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
        # Two groups after the card that are no expiry, more than two, or
        # one that makes the two more than 19 digits: an Aadhaar number
        # (4-4-4) could start inside the card and run on past it.
        # docs/detection.md lists these residuals.
        "card 4111 1111 1111 1111 13 27",  # no month 13
        "card 4111 1111 1111 1111 12 345",  # nor a three-digit year
        "card 4111 1111 1111 1111 12 27 123",
        "card 4111 1111 1111 1111-12-27",  # not the card's separator
        "card 4111111111111111 12-27",  # nor one separator twice
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


def _luhn(digits: str) -> bool:
    """The Luhn check, transcribed apart from the rule's (bench/corpus.py
    has a third): not all zeros, and the doubled-digit sum ends in 0."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch) * (1 + i % 2)
        total += d // 10 + d % 10
    return total % 10 == 0 and any(int(ch) for ch in digits)


def _inside_by_brute_force(match: re.Match[str], limit: int) -> tuple[int, int] | None:
    """What _card_inside's docstring says it returns, searched for by
    trying every stretch: from a group start inside the vetoed candidate to
    a group end before ``limit``, digit groups joined by single separators,
    13 to 19 digits, the candidate itself aside; in a card's grouping,
    ending where _CARD_END lets it, passing the check; the one of most
    digits, of equal length the one starting last."""
    text = match.string
    best: tuple[int, int, int] | None = None
    for start in range(match.start(), match.end()):
        if start > match.start() and text[start - 1] not in " -":
            continue
        digits, separator = "", None
        for end in range(start + 1, limit + 1):
            last = text[end - 1]
            if last in " -":
                if text[end - 2] in " -":
                    break
                separator = separator or last
                continue
            if not last.isdecimal():
                break
            digits += last
            if len(digits) > 19:
                break
            if end < len(text) and re.match(r"\w", text[end]):
                continue  # inside a group
            # After the run's end, one group (the two at most 19 digits) or
            # an expiry after the card's separator (a solid card's: either).
            after = _CARD_END.match(text, end)
            if (
                len(digits) >= 13
                and (start, end) != match.span()
                and _CARD_GROUPING.fullmatch(text[start:end])
                and after is not None
                and (after.group(1) is None or len(digits) + len(after.group(1)) <= 19)
                and after.group(2) in (None, separator or after.group(2))
                and _luhn(digits)
                and (best is None or (len(digits), start) > best[:2])
            ):
                best = (len(digits), start, end)
    return None if best is None else best[1:]


def test_the_card_read_inside_is_the_one_its_definition_names() -> None:
    # Every vetoed candidate of random runs of digit groups (cards and
    # card-shaped groups among them, other separators and words beside
    # them): _card_inside finds what a search of every stretch finds.
    rng = random.Random(21)
    pieces = ("4111", "1111", "12", "123", "3782", "822463", "10005", "x") + (
        "4111 1111 1111 1111", "4111-1111-1111-1111", "3782 822463 10005", "4111111111111111",
        "12 27", "09-2031", "13 27", "27", "2031",
    )  # fmt: skip
    joins = (" ", "-") * 6 + ("", ".", ". ", ":", ": ", "/", "@", "\u00a0", "  ")
    card = VALUE_GENERATORS["credit_card"][1]
    texts = [
        "".join(
            (rng.choice(pieces) if rng.random() < 0.7 else str(rng.randrange(10**6)))
            + rng.choice(joins)
            for _ in range(rng.randrange(1, 10))
        )
        for _ in range(4000)
    ]
    # Two stretches pass: a card, and the card with the group after it.
    for _ in range(300):
        number = card(rng)
        while not _luhn(number + (after := str(rng.randrange(1, 1000)))):
            pass
        fours = " ".join(number[i : i + 4] for i in range(0, 16, 4))
        texts.append(f"{rng.randrange(1, 100)} {fours} {after}")
    checked = found = 0
    for text in texts:
        matches = list(RULE.pattern.finditer(text))
        for i, match in enumerate(matches):
            if RULE.validator is not None and RULE.validator(match):
                continue
            limit = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            card = _card_inside(match, limit)
            assert (card and card.span()) == _inside_by_brute_force(match, limit), text
            checked += 1
            found += card is not None
    assert checked > 1500 and found > 300


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


def _cards_before_tails() -> Iterator[tuple[str, int, int]]:
    """Cards in every grouping _card_inside reads (solid, fours, 4-6-5,
    4-6-4) at every length, with each separator, before every tail
    _card_ends lets them have (the run's end, one group, an expiry) and
    what may follow it: (text, card start, card end). The digits are 2s
    (every built-in rule's first digit may be one) or random."""
    rng = random.Random(3)
    groupings: list[list[int]] = [[n] for n in range(13, 20)]
    groupings += [[4] * (n // 4) + ([n % 4] if n % 4 else []) for n in range(13, 20)]
    groupings += [[4, 6, 5], [4, 6, 4]]
    for sizes in groupings:
        for separator in " -":
            digits = sum(sizes)
            tails = ["", *(separator + "2" * n for n in range(1, min(4, 20 - digits)))]
            for expiry_separator in " -" if len(sizes) == 1 else separator:
                tails += [
                    expiry_separator + month + expiry_separator + year
                    for month in ("02", "12")
                    for year in ("22", "2222")
                ]
            for tail in tails:
                for fill in ("2", None):
                    groups = [
                        "".join(fill or rng.choice("0123456789") for _ in range(size))
                        for size in sizes
                    ]
                    card = separator.join(groups)
                    for before in ("", "x "):
                        for after in ("", ". x", ": x", " x", "\\n"):
                            text = f"{before}{card}{tail}{after}"
                            yield text, len(before), len(before) + len(card)


def test_no_rule_runs_on_from_inside_a_card() -> None:
    # Where _card_inside lets a card end, no other built-in rule's pattern
    # matches from a place inside the card (its own context around it) to
    # one past its end: no built-in value can lose characters to the card
    # (a custom rule's or an NER span is joined with it, test_joined_values).
    patterns = [rule.pattern for rule in BUILTIN_RULES if rule.name != "credit_card"]
    tried = 0
    for text, start, end in _cards_before_tails():
        for pattern in patterns:
            for pos in range(start + 1, end):
                found = pattern.match(text, pos)
                assert found is None or found.end() <= end, (text, pattern.pattern)
                tried += 1
    assert tried > 500_000


def _expiry_runs(seed: int, count: int) -> list[str]:
    """Runs of 13 to 40 random digits in groups of one to four, then an
    expiry-shaped tail (a month 01 to 12, then a year of two or four
    random digits), all separated by one space or one hyphen."""
    rng = random.Random(seed)
    texts = []
    for _ in range(count):
        groups, left = [], rng.randint(13, 40)
        while left:
            size = min(left, rng.randint(1, 4))
            groups.append("".join(rng.choice("0123456789") for _ in range(size)))
            left -= size
        separator = rng.choice(" -")
        month = f"{rng.randint(1, 12):02d}"
        year = f"{rng.randrange(10000):04d}" if rng.random() < 0.5 else f"{rng.randrange(100):02d}"
        texts.append("ref " + separator.join([*groups, month, year]) + " done")
    return texts


def test_random_digit_runs_hardly_gain_false_positives() -> None:
    # The grouping and the end the stretch must have keep the added false
    # positives small: on these 3,000 runs 12.93% held a Luhn-valid
    # candidate before and 13.13% now.
    texts = _runs(7, 3000)
    now = sum(bool(list(CARD.detect(text))) for text in texts)
    before = sum(bool(list(CARD_1_12_1.detect(text))) for text in texts)
    assert before > 300  # Luhn passes one random run in ten
    assert now - before <= len(texts) // 200


def test_the_documented_false_positive_figures() -> None:
    # docs/detection.md's figures: 5,000 runs of digit groups, and 5,000
    # such runs ending in an expiry-shaped tail, before (1.12.1) and now.
    for texts, before, now in ((_runs(0, 5000), 672, 686), (_expiry_runs(1, 5000), 724, 729)):
        assert sum(bool(list(CARD_1_12_1.detect(text))) for text in texts) == before
        assert sum(bool(list(CARD.detect(text))) for text in texts) == now


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
    # measured, about one and a half times (the rule alone about three
    # times 1.12.1's on these runs, where nearly every candidate is looked
    # inside). Over that bound both are timed again, so a burst of other
    # load does not fail it.
    from test_no_loss_since_1_12_0 import PLAN_1_12_1

    release = Redactor(DETECTORS, InMemoryVault(), ALLOWLIST).redact_text
    before = Redactor(PLAN_1_12_1, InMemoryVault(), ALLOWLIST).redact_text
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
