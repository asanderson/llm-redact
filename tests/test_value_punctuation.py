"""Values at the end of a sentence, and the spaced IBAN, are redacted.

1.12.0 forwarded these unredacted: belgian_nn and swiss_ahv refused a value
followed by ".", ipv6 an address followed by "." or ":", planetscale_token a
token whose body and the dots after it pass 64 characters, and the iban rule
read only the solid electronic form. Since 1.12.1 a value may also be
followed by a run of dots (an IPv6 address also by one ":") when whitespace,
a closing quote or bracket, "*", "<", a backslash (a JSON string's escape, in
a tool call's arguments) or the end of the text follows the run, and the
iban rule reads the ISO 13616 paper format of the SWIFT registry's
countries. A value glued into a longer token is still not one. That nothing
1.12.0 redacted is lost is test_no_loss_since_1_12_0.py.
"""

import json
import time
from collections.abc import Callable

import pytest

from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault

CONFIG = DetectionConfig()
DETECTORS = build_detectors(CONFIG)
ALLOWLIST = build_allowlist(CONFIG)
TOKEN64 = "pscale_tkn_" + "a1B2c3D4" * 8


def redacted(text: str) -> str:
    """``text`` through the production Redactor; its output restores
    exactly to ``text``."""
    vault = InMemoryVault()
    output = Redactor(DETECTORS, vault, ALLOWLIST).redact_text(text)
    assert Rehydrator(vault).rehydrate_text(output) == text
    return output


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("NN 93.05.18-223.61.", "NN «BE_NN_001»."),
        ("NN 93.05.18-223.61... (twice)", "NN «BE_NN_001»... (twice)"),
        ("AHV 756.9217.0769.85. Done", "AHV «CH_AHV_001». Done"),
        ('"AHV 756.9217.0769.85."', '"AHV «CH_AHV_001»."'),
        ("Ship to 2001:db8::1.", "Ship to «IPV6_001»."),
        ("2001:db8::1: unreachable", "«IPV6_001»: unreachable"),
        ("net 2001:db8::: down", "net «IPV6_001»: down"),
        ("(gw 2001:db8::1.)", "(gw «IPV6_001».)"),
        ("pay GB82 WEST 1234 5698 7654 32 now", "pay «IBAN_001» now"),
        # A token's run of dots is part of it, as 1.12.0 read a shorter one.
        (f"token {TOKEN64}. Done", "token «PLANETSCALE_TOKEN_001» Done"),
        (f"token {TOKEN64[:-2]}...", "token «PLANETSCALE_TOKEN_001»"),
    ],
)
def test_reported_values_are_redacted_whole(text: str, expected: str) -> None:
    assert redacted(text) == expected


# Every closing quote or bracket, the German and French ones included, the
# "*" of Markdown emphasis, the "<" of an HTML tag, and the backslash of a
# JSON string's escape.
CLOSERS = "\"'`)]}>»”’“‘«‹›*<\\"


@pytest.mark.parametrize("close", CLOSERS)
def test_a_sentence_dot_may_come_before_any_closer(close: str) -> None:
    assert redacted(f"NN 93.05.18-223.61.{close}") == f"NN «BE_NN_001».{close}"
    assert redacted(f"AHV 756.9217.0769.85.{close}") == f"AHV «CH_AHV_001».{close}"
    assert redacted(f"gw 2001:db8::1.{close}") == f"gw «IPV6_001».{close}"
    assert redacted(f"gw 2001:db8::1:{close}") == f"gw «IPV6_001»:{close}"
    assert redacted(f"pay DE89 3704 0044 0532 0130 00.{close}") == f"pay «IBAN_001».{close}"
    assert redacted(f"token {TOKEN64}.{close}") == f"token «PLANETSCALE_TOKEN_001»{close}"


@pytest.mark.parametrize(
    "text",
    [
        # Glued into a longer dotted or colon token, a version, a MAC, a
        # timestamp, a ratio: not the value, as in 1.12.0.
        "build 756.9217.0769.85.3 done",
        "id 93.05.18-223.61.7 done",
        "id 93.05.18-223.61-2 done",
        "id 93.05.18-223.61..x done",
        "addr 2001:db8::1.5 done",
        "addr 2001:db8::1..user done",
        "addr 2001:db8::1:x done",
        "addr 2001:db8::1:: done",
        "ver 1:2:3.",
        "mac 00:1b:44:11:3a:b7.",
        "serial 04:9f:86:d0:81:88:4c:7d:",
        "at 08:00:01: done",
        "ratio a:b:c.",
        "slice x[::2].",
        f"token {TOKEN64}.x done",
        f"token {TOKEN64}a. done",
    ],
)
def test_glued_and_vetoed_shapes_stay_unredacted(text: str) -> None:
    assert redacted(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\u201eNN 93.05.18-223.61.\u201c", "\u201eNN «BE_NN_001».\u201c"),
        ("It is **AHV 756.9217.0769.85.**", "It is **AHV «CH_AHV_001».**"),
        ("gw 2001:db8::1.<br>", "gw «IPV6_001».<br>"),
    ],
)
def test_typography_and_markup_end_a_sentence(text: str, expected: str) -> None:
    assert redacted(text) == expected


def test_the_reported_values_are_redacted_in_a_tool_calls_arguments() -> None:
    # A tool call's arguments are JSON source, redacted as one string: an
    # escaped line break (backslash, "n") follows each sentence's ".".
    lines = [
        "Ship to 2001:db8::1.",
        "NN 93.05.18-223.61.",
        "AHV 756.9217.0769.85.",
        "IBAN GB82 WEST 1234 5698 7654 32.",
        f"token {TOKEN64}.",
        "End",
    ]
    arguments = json.dumps({"content": "\n".join(lines)})
    output = redacted(arguments)
    for value in ("2001:db8::1", "93.05.18-223.61", "756.9217.0769.85", "GB82 WEST", TOKEN64):
        assert value not in output
    assert json.loads(output)["content"].count("«") == 5


PAPER_IBANS = [
    "GB82 WEST 1234 5698 7654 32",
    "BE68 5390 0754 7034",
    "DE89 3704 0044 0532 0130 00",
    "PL61 1090 1014 0000 0712 1981 2874",
    "FR14 2004 1010 0505 0001 3M02 606",
    "MT84 MALT 0110 0001 2345 MTLC AST0 01S",
    "NO93 8601 1117 947",
]


@pytest.mark.parametrize("iban", PAPER_IBANS)
@pytest.mark.parametrize(
    "context",
    [
        "IBAN {v}",
        "IBAN {v}.",
        "IBAN: {v}; thanks",
        "(IBAN {v})",
        "**{v}**",
        "IBAN {v} BIC GKCCBEBB",
        "IBAN {v} y BIC CAIXESBBXXX",
        "IBAN {v} a week ago",
        "IBAN {v} I think",
        "IBAN {v}, +32 2 555 12 12",
        "IBAN {v}, card 4111 1111 1111 1111",
    ],
)
def test_a_paper_format_iban_is_redacted_whole(iban: str, context: str) -> None:
    # Whole: an all-digit BBAN holds card- and phone-shaped runs, and the
    # IBAN starts first.
    text = context.replace("{v}", iban)
    assert redacted(text).startswith(context.split("{v}")[0] + "«IBAN_001»")


@pytest.mark.parametrize("separator", ["\u00a0", "\u202f"])
def test_a_no_break_space_separates_groups_like_a_space(separator: str) -> None:
    iban = "GB82 WEST 1234 5698 7654 32".replace(" ", separator, 3)
    assert redacted(f"pay {iban}.") == "pay «IBAN_001»."


def test_an_iban_after_a_vetoed_candidate_is_found() -> None:
    # "IS12" and the GB IBAN's groups are an Icelandic IBAN's length: that
    # candidate fails the checksum, and the scan goes on at its next group.
    assert redacted("ref IS12 GB82 WEST 1234 5698 7654 32") == "ref IS12 «IBAN_001»"


@pytest.mark.parametrize(
    "text",
    [
        "pay GB82 WEST 1234 5698 7654 33 now",  # checksum off by one
        "pay GB82  WEST 1234 5698 7654 32 now",  # a double space
        "pay GB82 WEST 1234 5698 7654 3 now",  # too short
        "pay GB82 WEST 1234 5698 7654 32X now",  # glued to a word
        "pay GB82 WEST 1234 5698 7654 32.5 now",  # glued to a number
        "pay GB82 WEST 1234 5698 7654 32 99 now",  # a digit group after
        "pay GB82 WEST 1234 5698 7654 32 A now",  # a NINO's last letter after
        "pay CI93 CI00 8011 1301 1342 9120 0589 now",  # outside the registry
    ],
)
def test_paper_format_near_misses_are_not_ibans(text: str) -> None:
    assert "«IBAN" not in redacted(text)


def test_the_paper_format_is_read_only_at_the_registry_length() -> None:
    from llm_redact.detection.regex_rules import _IBAN_PAPER_LENGTHS
    from test_no_loss_since_1_12_0 import with_check_digits

    for country, length in _IBAN_PAPER_LENGTHS.items():
        for size in range(15, 35):
            value = with_check_digits(country, ("XYZW" + "1234567890" * 3)[: size - 4])
            paper = " ".join(value[i : i + 4] for i in range(0, len(value), 4))
            text = f"IBAN {paper} ok"
            if size == length:
                assert redacted(text) == "IBAN «IBAN_001» ok"
            else:
                # Not an IBAN. Its digit groups may hold a Luhn-valid card
                # since 1.13.0 (test_credit_card_groups.py): exactly the
                # cards the credit_card rule reads are redacted.
                assert redacted(text) == _cards_redacted(text)


def _cards_redacted(text: str) -> str:
    """``text`` with what the credit_card rule alone reads in it replaced,
    numbered as the vault numbers them."""
    card = next(det for det in DETECTORS if det.name == "credit_card")
    numbers: dict[str, int] = {}
    output, pos = [], 0
    for found in card.detect(text):
        number = numbers.setdefault(found.value, len(numbers) + 1)
        output += [text[pos : found.start], f"«CREDIT_CARD_{number:03d}»"]
        pos = found.end
    return "".join(output) + text[pos:]


# --- cost ---------------------------------------------------------------------

SIZE = 100_000
# 1.12.0's slowest known inputs: an IPv6 address every four characters.
WORST_1_12_0 = ("::a ", "a:: ")
ADVERSARIAL = (
    "::a ", " ::", "1:.", "a::.", "2001:db8::1: ", "2001:db8::1.. ", "2001:db8::1..x ",
    "756.", "756.9217.0769.85. ", "756.9217.0769.85.3 ", "93.05.18-223.61. ",
    "LC12 ", "LC53 ", "XK29 ", "BE35 ", "LC12\u00a0", "RU12\u202f", "GB82 WEST 1234 5698 7654 32 ",
    "pscale_tkn_", "pscale_tkn_" + "a" * 40 + "." * 40 + " ",
)  # fmt: skip


def _seconds(scan: Callable[[str], object], unit: str, size: int = SIZE) -> float:
    text = unit * (size // len(unit) + 1)
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        scan(text)
        best = min(best, time.perf_counter() - started)
    return best


def test_no_adversarial_input_costs_more_than_1_12_0s_worst() -> None:
    # Each changed rule is linear on every input (ten times the input,
    # under thirty times the time: about ten), and the whole Redactor takes
    # on each less than twice what 1.12.0's took on its slowest known one:
    # measured, the slowest of them ties with it, so the bound leaves room
    # for a shared runner's noise. Over that bound both are timed again, so
    # a burst of other load does not fail it.
    from test_no_loss_since_1_12_0 import FROZEN_1_12_0, OLD_PLAN

    changed = [det for det in DETECTORS if det.name in FROZEN_1_12_0]
    release = Redactor(DETECTORS, InMemoryVault(), ALLOWLIST).redact_text
    before = Redactor(OLD_PLAN, InMemoryVault(), ALLOWLIST).redact_text

    def reference() -> float:
        return max(_seconds(before, unit) for unit in WORST_1_12_0)

    worst = reference()
    for unit in ADVERSARIAL:
        for det in changed:
            scan = lambda text, det=det: list(det.detect(text))  # noqa: E731
            small, large = _seconds(scan, unit, SIZE // 10), _seconds(scan, unit)
            assert large < 30 * max(small, 1e-4), (det.name, unit, small, large)
        large = _seconds(release, unit)
        if large >= 2 * worst:
            worst, large = reference(), _seconds(release, unit)
        assert large < 2 * worst, (unit, large, worst)
