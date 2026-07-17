"""Named validators for custom rules + their config/build wiring."""

import re

import pytest

from llm_redact.detection.engine import (
    Allowlist,
    CustomRule,
    DetectionConfig,
    build_detectors,
)
from llm_redact.detection.validators import VALIDATORS
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault


def _match(value: str) -> re.Match[str]:
    m = re.fullmatch(r".+", value, re.S)
    assert m is not None
    return m


def test_luhn_validator() -> None:
    luhn = VALIDATORS["luhn"]
    assert luhn(_match("4111 1111 1111 1111"))  # valid test Visa
    assert not luhn(_match("4111 1111 1111 1112"))  # bad check digit
    assert not luhn(_match("0000 0000 0000 0000"))  # all-zero placeholder
    # A two-digit valid-Luhn number pins the `len(digits) >= 2` floor: a
    # tighter floor (>2 / >=3) would reject it.
    assert luhn(_match("18"))


def test_verhoeff_validator() -> None:
    verhoeff = VALIDATORS["verhoeff"]
    # Find the valid Verhoeff check digit for an 11-digit base, then confirm
    # the 12-digit number validates and a one-digit change breaks it.
    base = "23456789012"
    valid = next(base + str(c) for c in range(10) if verhoeff(_match(base + str(c))))
    assert verhoeff(_match(valid))
    wrong = valid[:-1] + str((int(valid[-1]) + 1) % 10)
    assert not verhoeff(_match(wrong))
    # A FIXED known-valid number: the dynamic search above self-adjusts to a
    # mutated accumulator seed or final-comparison constant, so it cannot pin
    # them — this literal does (2363 has Verhoeff checksum 0).
    assert verhoeff(_match("2363"))
    # No digits at all must be rejected, not silently accepted.
    assert not verhoeff(_match("no digits here"))


def test_mod97_validator() -> None:
    mod97 = VALIDATORS["mod97"]
    assert mod97(_match("GB82 WEST 1234 5698 7654 32"))  # valid IBAN
    assert not mod97(_match("GB82 WEST 1234 5698 7654 33"))
    # Exactly four alnum chars: 98 % 97 == 1 pins the `len(alnum) < 4` floor
    # (a tighter <=4 / <5 would reject it).
    assert mod97(_match("0098"))
    # Fewer than four chars must be rejected.
    assert not mod97(_match("12"))
    # A plain MOD-97-10 value that validates via the AS-IS path only (its
    # four-char rotation does NOT satisfy the check) — pins the as_is branch
    # independently of the IBAN move-first-4 rearrangement.
    assert mod97(_match("10089"))


def test_jwt_validator() -> None:
    jwt = VALIDATORS["jwt"]
    # {"alg":"HS256"} . {"sub":"1"} . sig
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2ln"
    assert jwt(_match(token))
    assert not jwt(_match("not.a.jwt!!!"))
    assert not jwt(_match("onlytwo.segments"))
    # A well-formed header but a payload that decodes to non-JSON must be
    # rejected: BOTH segments must be JSON objects (pins the `and`, and the
    # decode/parse failure returning False rather than True).
    assert not jwt(_match("eyJhbGciOiJIUzI1NiJ9.aaaa.c2ln"))


def test_entropy_validator() -> None:
    entropy = VALIDATORS["entropy"]
    assert entropy(_match("Zx9Kq2Lm7Wp4Rt8Vn3Bc6"))  # high-entropy token
    assert not entropy(_match("the quick brown fox jumps"))  # prose
    assert not entropy(_match("short"))  # below the length floor
    # Exactly 16 chars, high entropy: pins the `len(value) < 16` floor (a
    # tighter <=16 / <17 would reject it).
    assert entropy(_match("Zx9Kq2Lm7Wp4Rt8V"))
    # 16 chars, no whitespace, but LOW entropy (~0.34 bits/char): the length
    # floor passes, so the Shannon computation itself must reject it. A
    # mutated per-term weight (c * n instead of c / n) inflates the estimate
    # past the threshold and would wrongly accept this.
    assert not entropy(_match("aaaaaaaaaaaaaaab"))


def _detectors(rule: CustomRule):
    return build_detectors(DetectionConfig(enabled=(), custom_rules=(rule,)))


def test_custom_rule_with_validator_gates_matches() -> None:
    # A loose "16ish digits" pattern that only fires when Luhn passes.
    rule = CustomRule(
        name="cardish",
        detector_type="CARDISH",
        pattern=r"\d[\d ]{14,18}\d",
        validator="luhn",
    )
    vault = InMemoryVault()
    redactor = Redactor(_detectors(rule), vault, Allowlist(exact=frozenset(), patterns=()))
    good = redactor.redact_text("pay 4111 1111 1111 1111 now")
    assert "4111" not in good and "«CARDISH_001»" in good
    # A same-shape non-Luhn number is left alone.
    bad = redactor.redact_text("ref 4111 1111 1111 1112 here")
    assert "4111 1111 1111 1112" in bad


def test_unknown_validator_is_build_time_error() -> None:
    rule = CustomRule(name="x", detector_type="X", pattern=r"\d+", validator="nope")
    with pytest.raises(ValueError, match="unknown validator 'nope'"):
        _detectors(rule)


def test_custom_rule_required_prefilter_still_matches() -> None:
    # `required` is a hot-path skip hint; a correct one does not change results.
    rule = CustomRule(
        name="ticket",
        detector_type="TICKET",
        pattern=r"PROJ-\d+",
        required=("PROJ-",),
        anchors=("PROJ-",),
    )
    vault = InMemoryVault()
    redactor = Redactor(_detectors(rule), vault, Allowlist(exact=frozenset(), patterns=()))
    assert "«TICKET_001»" in redactor.redact_text("see PROJ-4821 for details")
    # A text lacking the required literal is skipped (and matches nothing).
    assert redactor.redact_text("no ticket here") == "no ticket here"
