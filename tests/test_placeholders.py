import pytest

from llm_redact.placeholders import (
    MAX_PLACEHOLDER_LEN,
    MAX_TOKEN_NUMBER,
    MAX_TYPE_NAME_LEN,
    PLACEHOLDER_RE,
    canonicalize,
    format_placeholder,
    is_placeholder_type,
    viable_prefix_start,
)


def test_format_and_regex_round_trip() -> None:
    token = format_placeholder("EMAIL", 1)
    assert token == "«EMAIL_001»"
    assert PLACEHOLDER_RE.fullmatch(token)


def test_counter_rollover_past_999() -> None:
    token = format_placeholder("EMAIL", 1000)
    assert token == "«EMAIL_1000»"
    assert PLACEHOLDER_RE.fullmatch(token)


def test_regex_rejects_lowercase_and_unclosed() -> None:
    assert not PLACEHOLDER_RE.search("«email_001»")
    assert not PLACEHOLDER_RE.search("«EMAIL_001")
    assert not PLACEHOLDER_RE.search("«EMAIL»")


def test_viable_prefix_detected() -> None:
    assert viable_prefix_start("hello «EM") == 6
    assert viable_prefix_start("hello «EMAIL_00") == 6
    assert viable_prefix_start("«") == 0


def test_no_prefix_when_closed_or_absent() -> None:
    assert viable_prefix_start("no guillemets here") is None
    assert viable_prefix_start("done «EMAIL_001»") is None


def test_invalid_body_char_is_not_a_prefix() -> None:
    # French prose: « followed by lowercase can never become a token.
    assert viable_prefix_start("il a dit «bonjour") is None


def test_overlong_prefix_released() -> None:
    assert viable_prefix_start("«" + "A" * MAX_PLACEHOLDER_LEN) is None


# ---- fuzzy grammar ----


@pytest.mark.parametrize(
    ("mangled", "canonical"),
    [
        ("«EMAIL_001»", "«EMAIL_001»"),  # canonical is a fixed point
        ("«email_001»", "«EMAIL_001»"),
        ("«Email_001»", "«EMAIL_001»"),
        ("«EMAIL_1»", "«EMAIL_001»"),
        ("«EMAIL_0001»", "«EMAIL_001»"),
        ("«EMAIL-001»", "«EMAIL_001»"),
        ("«CREDIT-CARD-001»", "«CREDIT_CARD_001»"),
        ("« EMAIL_001»", "«EMAIL_001»"),
        ("«EMAIL_001 »", "«EMAIL_001»"),
        ("« EMAIL_001 »", "«EMAIL_001»"),
        ("«email-1 »", "«EMAIL_001»"),
        ("«AWS_KEY_1234»", "«AWS_KEY_1234»"),  # >999 keeps its digits
    ],
)
def test_canonicalize_in_scope(mangled: str, canonical: str) -> None:
    assert canonicalize(mangled) == canonical


@pytest.mark.parametrize(
    "not_a_token",
    [
        "«bonjour»",  # no trailing digits
        "«peut-être»",  # accented char outside grammar
        "[EMAIL_001]",  # bracket swap: deliberately out of scope
        "<EMAIL_001>",
        "EMAIL_001",  # bare identifier
        "«EMAIL_»",  # separator but no digits
        "«_001»",  # no type name
        "«   EMAIL_001»",  # >2 pad chars
        "«EMAIL 001»",  # interior space
    ],
)
def test_canonicalize_out_of_scope(not_a_token: str) -> None:
    assert canonicalize(not_a_token) is None


def test_fuzzy_prefix_holdback_and_release() -> None:
    assert viable_prefix_start("x «email_00", fuzzy=True) == 2
    assert viable_prefix_start("x « EMAIL_001", fuzzy=True) == 2
    # Out-of-language characters release the run.
    assert viable_prefix_start("il a dit «bonjour et", fuzzy=True) is None
    assert viable_prefix_start("«peut-êt", fuzzy=True) is None
    # Pad after letters (not digits) is not a viable prefix.
    assert viable_prefix_start("«email ", fuzzy=True) is None
    # Length cap still applies.
    assert viable_prefix_start("«" + "a" * MAX_PLACEHOLDER_LEN, fuzzy=True) is None


def test_max_type_name_len_keeps_every_token_within_the_holdback_bound() -> None:
    assert MAX_TYPE_NAME_LEN == 28
    longest = format_placeholder("T" * MAX_TYPE_NAME_LEN, MAX_TOKEN_NUMBER)
    assert len(longest) == MAX_PLACEHOLDER_LEN
    assert PLACEHOLDER_RE.fullmatch(longest)
    assert canonicalize(longest) == longest


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("PERSON", True),
        ("A", True),
        ("A1_B2", True),
        ("A_", True),
        ("T" * 28, True),
        ("T" * 29, False),
        ("", False),
        ("1A", False),
        ("_A", False),
        ("a", False),
        ("Ab", False),
        ("A-B", False),
        ("A B", False),
        ("A\n", False),
        ("É", False),
    ],
)
def test_is_placeholder_type(name: str, ok: bool) -> None:
    assert is_placeholder_type(name) is ok
