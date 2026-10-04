"""Deny strings ADDED to the configured policy (``Redactor(added_deny=...)``:
an access gate's detection overlay, ``authorization.OverlayBuilds``) only
ever TIGHTEN it.

They are detected apart from the configured detectors and taken in on top of
the configured winners (``Redactor._absorb``): a value the configured policy
redacts keeps its effect and, with every added match overlapping it, the
UNION of their spans is redacted as one placeholder; a configured block
still refuses; an added match beats only a value the policy forwards as sent
(warn mode, an approved block). So the characters redacted with the added
strings are a SUPERSET of those redacted without them, and every occurrence
of an added string is redacted — pinned by the hypothesis property below.
(Among the configured detectors a deny string wins every overlap, and the
value it cut into left in part: a block never fired.)
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_redact.detection.deny import DenyDetector, DenyEntry
from llm_redact.detection.engine import (
    DetectionConfig,
    DetectorPlan,
    build_allowlist,
    build_detectors,
    build_modes,
)
from llm_redact.placeholders import PLACEHOLDER_RE
from llm_redact.redactor import BlockedRequest, Redactor
from llm_redact.vault import InMemoryVault

EMAIL = "bob.private@acme-corp.example"


class Approved:
    """Allows exactly ``values``; records every question."""

    def __init__(self, *values: str) -> None:
        self.values = set(values)
        self.asked: list[tuple[str, str]] = []
        self.final = 0

    def allows(self, detector_type: str, value: str) -> bool:
        self.asked.append((detector_type, value))
        return value in self.values

    def unoverridable(self) -> None:
        self.final += 1


def _plan(added: tuple[str, ...]) -> DetectorPlan | None:
    return DetectorPlan([DenyDetector([DenyEntry(value) for value in added])]) if added else None


def _redactor(
    config: DetectionConfig | None = None,
    added: tuple[str, ...] = (),
    vault: InMemoryVault | None = None,
) -> Redactor:
    config = config or DetectionConfig()
    return Redactor(
        build_detectors(config),
        vault if vault is not None else InMemoryVault(),
        build_allowlist(config),
        modes=build_modes(config),
        added_deny=_plan(added),
    )


def _block(rule: str = "email") -> DetectionConfig:
    return DetectionConfig(modes=((rule, "block"),))


# --- a value the configured policy redacts --------------------------------------------


def test_inside_a_redacted_value_the_configured_token_is_kept() -> None:
    vault = InMemoryVault()
    redactor = _redactor(added=("acme",), vault=vault)
    # Without the added string: exactly the same.
    assert _redactor().redact_text(f"mail {EMAIL}") == "mail «EMAIL_001»"
    assert redactor.redact_text(f"mail {EMAIL}") == "mail «EMAIL_001»"
    assert vault.original_for("«EMAIL_001»") == EMAIL and len(vault) == 1
    assert redactor.counts == Counter({"EMAIL": 1})


def test_at_the_same_start_the_configured_value_names_the_union() -> None:
    redactor = _redactor(added=("bob.private",))
    assert redactor.redact_text(f"mail {EMAIL}") == "mail «EMAIL_001»"


def test_straddling_a_redacted_value_their_union_is_redacted() -> None:
    vault = InMemoryVault()
    redactor = _redactor(added=("example now",), vault=vault)
    assert redactor.redact_text(f"mail {EMAIL} now!") == "mail «EMAIL_001»!"
    # One placeholder over the union, typed after its first span.
    assert vault.original_for("«EMAIL_001»") == f"{EMAIL} now"
    assert redactor.counts == Counter({"EMAIL": 1})


def test_a_configured_deny_string_and_an_added_one_are_redacted_as_their_union() -> None:
    configured = DetectionConfig(
        deny_strings=(DenyEntry("corp secret plan", detector_type="CODENAME"),)
    )
    text = "status of acme corp secret plan today"
    assert _redactor(configured).redact_text(text) == "status of acme «CODENAME_001» today"
    vault = InMemoryVault()
    redactor = _redactor(configured, added=("acme corp",), vault=vault)
    # Neither the configured string's tail nor the added one's head leaves.
    assert redactor.redact_text(text) == "status of «DENY_001» today"
    assert vault.original_for("«DENY_001»") == "acme corp secret plan"


def test_a_union_bridging_two_values_is_one_placeholder() -> None:
    vault = InMemoryVault()
    redactor = _redactor(added=("example ann",), vault=vault)
    text = f"{EMAIL} ann@two.example"
    assert _redactor().redact_text(text) == "«EMAIL_001» «EMAIL_002»"
    assert redactor.redact_text(text) == "«EMAIL_001»"
    assert vault.original_for("«EMAIL_001»") == text


def test_overlapping_added_matches_join_and_adjacent_ones_do_not() -> None:
    joined = InMemoryVault()
    assert _redactor(added=("zetaom", "omega"), vault=joined).redact_text("zetaomega") == (
        "«DENY_001»"
    )
    assert joined.original_for("«DENY_001»") == "zetaomega"
    assert _redactor(added=("zeta", "omega")).redact_text("zetaomega") == "«DENY_001»«DENY_002»"


def test_a_run_keeps_its_furthest_end() -> None:
    # "private" and "@acme-corp" both lie inside the email: the run ends
    # where the email does, not where its last member does.
    redactor = _redactor(added=("private", "acme-corp"))
    assert redactor.redact_text(f"mail {EMAIL} today") == "mail «EMAIL_001» today"


def test_an_added_string_matching_nothing_changes_nothing() -> None:
    redactor = _redactor(added=("zeta",))
    assert redactor.redact_text(f"mail {EMAIL}") == "mail «EMAIL_001»"
    assert redactor.redact_text("nothing here") == "nothing here"


def test_the_thin_copies_keep_the_added_strings() -> None:
    base = _redactor(added=("zeta",))
    for copy in (
        base.with_budget(10),
        base.with_floors({"DENY": 7}),
        base.with_overrides(Approved()),
    ):
        assert "zeta" not in copy.redact_text("about zeta")


# --- a configured block ---------------------------------------------------------------


def test_a_configured_block_still_refuses() -> None:
    with pytest.raises(BlockedRequest) as configured:
        _redactor(_block()).redact_text(f"mail {EMAIL}")
    redactor = _redactor(_block(), added=("acme",))
    with pytest.raises(BlockedRequest) as caught:
        redactor.redact_text(f"mail {EMAIL}")
    assert caught.value.detector_type == configured.value.detector_type == "EMAIL"
    # Ahead of the redaction, the same verdict.
    assert redactor.blocked_type(f"mail {EMAIL}") == "EMAIL"
    assert redactor.blocked_type("about acme") is None
    with pytest.raises(BlockedRequest):
        redactor.scan(f"mail {EMAIL}", redactable=True)


def test_the_first_refused_value_is_the_configured_policys() -> None:
    # The phone (block) comes first; the email, which the added string
    # overlaps, second: the refusal names the phone, as without it.
    config = DetectionConfig(modes=(("email", "block"), ("phone_number", "block")))
    text = f"call +1 415 555 0132 or mail {EMAIL}"
    approved = Approved()
    redactor = _redactor(config, added=("acme",)).with_overrides(approved)
    with pytest.raises(BlockedRequest) as caught:
        redactor.redact_text(text)
    assert caught.value.detector_type == "PHONE"
    assert approved.asked == [("PHONE", "+1 415 555 0132")]


# --- a value forwarded as sent --------------------------------------------------------


def test_an_added_string_beats_a_warn_value() -> None:
    config = DetectionConfig(modes=(("email", "warn"),))
    plain = _redactor(config)
    assert plain.redact_text(f"mail {EMAIL}") == f"mail {EMAIL}"
    assert plain.warn_counts == Counter({"EMAIL": 1})
    redactor = _redactor(config, added=("acme",))
    # The added string is redacted; the rest of the value goes as it would.
    assert redactor.redact_text(f"mail {EMAIL}") == "mail bob.private@«DENY_001»-corp.example"
    assert redactor.counts == Counter({"DENY": 1}) and redactor.warn_counts == Counter()


def test_an_added_string_beats_an_approved_block() -> None:
    other = "bob.private@x.example"
    approved = Approved(EMAIL, other)
    redactor = _redactor(_block(), added=("acme",)).with_overrides(approved)
    assert redactor.redact_text(f"mail {EMAIL}") == "mail bob.private@«DENY_001»-corp.example"
    assert approved.asked == [("EMAIL", EMAIL)]
    # An approved block no added string touches goes as sent.
    assert redactor.redact_text(f"mail {other}") == f"mail {other}"
    assert redactor.counts == Counter({"DENY": 1})


def test_a_value_forwarded_as_sent_yields_to_a_union_it_starts_before() -> None:
    # The warn-mode email starts before the union of an added string and the
    # configured deny string it reaches into: the union is redacted whole,
    # the email — forwarded as sent — yields to it.
    config = DetectionConfig(
        modes=(("email", "warn"),),
        deny_strings=(DenyEntry("plan secret", detector_type="CODENAME"),),
    )
    vault = InMemoryVault()
    redactor = _redactor(config, added=("example plan",), vault=vault)
    assert redactor.redact_text(f"mail {EMAIL} plan secret") == (
        "mail bob.private@acme-corp.«DENY_001»"
    )
    assert vault.original_for("«DENY_001»") == "example plan secret"
    assert redactor.warn_counts == Counter()


def test_a_warn_value_away_from_every_added_match_is_still_counted() -> None:
    config = DetectionConfig(modes=(("email", "warn"),))
    redactor = _redactor(config, added=("zeta",))
    assert redactor.redact_text(f"zeta {EMAIL}") == f"«DENY_001» {EMAIL}"
    assert redactor.warn_counts == Counter({"EMAIL": 1})


# --- detection only (text the proxy cannot rewrite) -----------------------------------


def test_a_scan_refuses_the_union_for_good() -> None:
    # The email alone is an overridable finding; with the added string in
    # it the proxy could not redact their union in an unrewritable text.
    approved = Approved(EMAIL)
    redactor = _redactor(added=("acme",)).with_overrides(approved)
    scan = redactor.scan(f"mail {EMAIL}")
    assert scan.found == Counter({"EMAIL": 1}) and not scan.overridden
    assert approved.asked == [] and approved.final == 1


def test_a_scan_still_puts_a_lone_value_to_the_overrides() -> None:
    approved = Approved(EMAIL)
    redactor = _redactor(added=("zeta",)).with_overrides(approved)
    scan = redactor.scan(f"zeta {EMAIL}")
    assert scan.found == Counter({"DENY": 1}) and scan.overridden
    assert approved.asked == [("EMAIL", EMAIL)] and approved.final == 1


def test_a_redactable_scan_finds_the_union_without_refusing_it() -> None:
    approved = Approved()
    redactor = _redactor(added=("acme",)).with_overrides(approved)
    scan = redactor.scan(f"mail {EMAIL}", redactable=True)
    assert scan.found == Counter({"EMAIL": 1}) and not scan.overridden
    assert approved.asked == [] and approved.final == 0


# --- the invariant ----------------------------------------------------------------------


def _replaced(original: str, output: str, vault: InMemoryVault) -> set[int]:
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


_WORDS = st.sampled_from(
    ["bob", ".", "private", "@", "acme", "-", "corp", ".example", " ", "secret", "plan", "zeta"]
)
_CONFIGURED = st.lists(
    st.sampled_from(["corp secret", "secret plan", "acme", "zeta pl"]), max_size=2, unique=True
)
_ADDED = st.lists(
    st.sampled_from(["acme", "corp", "e c", "private@acme", "secret", "zeta", "an", "example "]),
    min_size=1,
    max_size=3,
    unique=True,
)


@settings(deadline=None, max_examples=300)
@given(
    words=st.lists(_WORDS, max_size=14),
    configured=_CONFIGURED,
    added=_ADDED,
    mode=st.sampled_from(["redact", "warn", "block"]),
    approve=st.booleans(),
)
def test_added_strings_redact_a_superset_and_refuse_alike(
    words: list[str], configured: list[str], added: list[str], mode: str, approve: bool
) -> None:
    text = "".join(words)
    config = DetectionConfig(
        modes=(("email", mode),) if mode != "redact" else (),
        deny_strings=tuple(DenyEntry(value, detector_type="CODENAME") for value in configured),
    )
    base_vault, vault = InMemoryVault(), InMemoryVault()
    base = _redactor(config, vault=base_vault)
    tightened = _redactor(config, added=tuple(added), vault=vault)
    if approve:

        class All:
            def allows(self, detector_type: str, value: str) -> bool:
                return True

            def unoverridable(self) -> None:
                return None

        base, tightened = base.with_overrides(All()), tightened.with_overrides(All())
    try:
        expected = base.redact_text(text)
    except BlockedRequest as configured_block:
        # Every configured block still refuses, for the same value.
        with pytest.raises(BlockedRequest) as caught:
            tightened.redact_text(text)
        assert caught.value.detector_type == configured_block.detector_type
        return
    got = tightened.redact_text(text)
    redacted = _replaced(text, got, vault)
    # A SUPERSET of what the configured policy redacts...
    assert _replaced(text, expected, base_vault) <= redacted
    # ...and every occurrence of every added string.
    for value in added:
        for match in re.finditer(re.escape(value), text, re.IGNORECASE):
            assert set(range(match.start(), match.end())) <= redacted
