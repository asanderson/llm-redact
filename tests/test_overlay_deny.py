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

from llm_redact.detection.base import ADDITIVE_TIER, Detection
from llm_redact.detection.deny import DenyDetector, DenyEntry
from llm_redact.detection.engine import (
    Allowlist,
    CustomRule,
    DetectionConfig,
    DetectorPlan,
    build_allowlist,
    build_detectors,
    build_modes,
)
from llm_redact.plugin_api import Inspection
from llm_redact.redactor import BlockedRequest, Redactor
from llm_redact.upload_inspection import judge
from llm_redact.vault import InMemoryVault
from warn_oracle import expected_warnings, replaced

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
    # The added string is redacted; the rest of the value goes as it would,
    # still counted as a warning: part of it goes upstream.
    assert redactor.redact_text(f"mail {EMAIL}") == "mail bob.private@«DENY_001»-corp.example"
    assert redactor.counts == Counter({"DENY": 1})
    assert redactor.warn_counts == Counter({"EMAIL": 1})


def test_an_added_string_beats_an_approved_block() -> None:
    other = "bob.private@x.example"
    approved = Approved(EMAIL, other)
    redactor = _redactor(_block(), added=("acme",)).with_overrides(approved)
    assert redactor.redact_text(f"mail {EMAIL}") == "mail bob.private@«DENY_001»-corp.example"
    assert approved.asked == [("EMAIL", EMAIL)]
    # An approved block no added string touches goes as sent.
    assert redactor.redact_text(f"mail {other}") == f"mail {other}"
    assert redactor.counts == Counter({"DENY": 1})
    # Neither is a warning: an approved block is no warn-mode value.
    assert redactor.warn_counts == Counter()


class _Found:
    """A stand-in detector finding ``surface`` as ``detector_type`` at
    ``tier`` (``ADDITIVE_TIER``: the llm NER backend's)."""

    name = "found"

    def __init__(self, found: list[tuple[str, str, int]]) -> None:
        self.found = found

    def detect(self, text: str) -> list[Detection]:
        detections = []
        for surface, detector_type, tier in self.found:
            at = text.find(surface)
            if at >= 0:
                end = at + len(surface)
                detections.append(Detection(at, end, detector_type, surface, 120, tier))
        return detections


@pytest.mark.parametrize("tier", [1, ADDITIVE_TIER])
def test_an_approved_or_warn_value_yields_to_a_union_at_any_tier(tier: int) -> None:
    # An llm finding (ADDITIVE_TIER) forwarded as sent yields to an added
    # match exactly as any other value does: dropped, never cut into
    # fragments put to the overrides again.
    value = "bob at example dot com"
    text = f"mail {value} please"
    detector = _Found([(value, "EMAIL", tier)])
    approved = Approved(value)
    blocking = Redactor(
        [detector],
        InMemoryVault(),
        Allowlist(),
        modes={"EMAIL": "block"},
        added_deny=_plan(("example",)),
    ).with_overrides(approved)
    assert blocking.blocked_type(text) is None
    assert blocking.redact_text(text) == "mail bob at «DENY_001» dot com please"
    assert {value for _type, value in approved.asked} == {value}
    warning = Redactor(
        [detector],
        InMemoryVault(),
        Allowlist(),
        modes={"EMAIL": "warn"},
        added_deny=_plan(("example",)),
    )
    assert warning.redact_text(text) == "mail bob at «DENY_001» dot com please"
    # The rest of it goes upstream: a warning, once.
    assert warning.warn_counts == Counter({"EMAIL": 1})


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
    assert redactor.warn_counts == Counter({"EMAIL": 1})


def test_a_warn_value_away_from_every_added_match_is_still_counted() -> None:
    config = DetectionConfig(modes=(("email", "warn"),))
    redactor = _redactor(config, added=("zeta",))
    assert redactor.redact_text(f"zeta {EMAIL}") == f"«DENY_001» {EMAIL}"
    assert redactor.warn_counts == Counter({"EMAIL": 1})


WARN = DetectionConfig(modes=(("email", "warn"),))


@pytest.mark.parametrize(
    ("added", "expected"),
    [
        # Cut into twice: one value, one warning.
        (("private", "corp"), "mail bob.«DENY_001»@acme-«DENY_002».example now"),
        # Sticking out of the added match by one character at either edge.
        (("ob.private@acme-corp.example",), "mail b«DENY_001» now"),
        (("bob.private@acme-corp.exampl",), "mail «DENY_001»e now"),
        # An added match starting before it and ending inside it.
        (("mail bob",), "«DENY_001».private@acme-corp.example now"),
    ],
)
def test_a_warn_value_partly_forwarded_is_counted_once(
    added: tuple[str, ...], expected: str
) -> None:
    # The access gate's added strings take part of the warn-mode value; the
    # rest goes upstream as sent, so it is a warning — once, as without them.
    redactor = _redactor(WARN, added=added)
    assert redactor.redact_text(f"mail {EMAIL} now") == expected
    assert redactor.warn_counts == Counter({"EMAIL": 1})


@pytest.mark.parametrize(
    ("added", "expected"),
    [
        # The very value.
        ((EMAIL,), "mail «DENY_001» now"),
        # A longer string holding it.
        ((f"{EMAIL} now",), "mail «DENY_001»"),
        # Two added matches touching inside it: together they cover it.
        (("bob.private@", "acme-corp.example"), "mail «DENY_001»«DENY_002» now"),
    ],
)
def test_a_warn_value_the_added_matches_cover_whole_is_no_warning(
    added: tuple[str, ...], expected: str
) -> None:
    # Nothing of it goes upstream: nothing to warn about (the added
    # strings' redactions are counted as such).
    redactor = _redactor(WARN, added=added)
    assert redactor.redact_text(f"mail {EMAIL} now") == expected
    assert redactor.warn_counts == Counter()
    assert redactor.counts == Counter({"DENY": len(added)})


def test_a_warn_value_covered_by_a_union_and_a_touching_match_is_no_warning() -> None:
    # A configured deny string and an added match reaching into the email
    # are redacted as one union; another added match touches it and covers
    # the rest of the email.
    config = DetectionConfig(
        modes=(("email", "warn"),),
        deny_strings=(DenyEntry("secret", detector_type="CODENAME"),),
    )
    vault = InMemoryVault()
    redactor = _redactor(config, added=("et bob.private@", "acme-corp.example"), vault=vault)
    assert redactor.redact_text(f"secret {EMAIL}") == "«CODENAME_001»«DENY_001»"
    assert vault.original_for("«CODENAME_001»") == "secret bob.private@"
    assert redactor.warn_counts == Counter()
    # Without the second match, the rest of the email goes upstream.
    redactor = _redactor(config, added=("et bob.private@",))
    assert redactor.redact_text(f"secret {EMAIL}") == "«CODENAME_001»acme-corp.example"
    assert redactor.warn_counts == Counter({"EMAIL": 1})


def test_warn_values_are_counted_each_and_an_approved_block_never() -> None:
    # Two warn-mode values the added strings cut into count once each; an
    # approved block-mode value they cut into is forwarded in part as well,
    # but it is no warning.
    other = "ann.roe@acme-corp.example"
    config = DetectionConfig(modes=(("email", "warn"), ("ipv4", "block")))
    approved = Approved("10.1.2.3")
    redactor = _redactor(config, added=("acme", "1.2")).with_overrides(approved)
    text = f"mail {EMAIL} or {other} at 10.1.2.3"
    assert redactor.redact_text(text) == (
        "mail bob.private@«DENY_001»-corp.example or ann.roe@«DENY_001»-corp.example"
        " at 10.«DENY_002».3"
    )
    assert redactor.warn_counts == Counter({"EMAIL": 2})
    assert approved.asked == [("IPV4", "10.1.2.3")]


@pytest.mark.parametrize(
    ("added", "found", "warned"),
    [
        (("acme",), Counter({"DENY": 1}), Counter({"EMAIL": 1})),
        ((EMAIL,), Counter({"DENY": 1}), Counter()),
    ],
)
@pytest.mark.parametrize("redactable", [False, True])
def test_a_scan_counts_alike(
    added: tuple[str, ...], found: Counter[str], warned: Counter[str], redactable: bool
) -> None:
    # Detection only (an upload's extracted text, a verbatim field): the
    # same warning, once.
    redactor = _redactor(WARN, added=added)
    scan = redactor.scan(f"mail {EMAIL}", redactable=redactable)
    assert scan.found == found and not scan.overridden
    assert redactor.warn_counts == warned


def test_a_check_ahead_of_the_redaction_counts_nothing() -> None:
    # blocked_type takes no added string in, and counts nothing.
    redactor = _redactor(WARN, added=("acme",))
    assert redactor.blocked_type(f"mail {EMAIL}") is None
    assert redactor.warn_counts == Counter()
    # Nor does the overlay's own step: the redaction counts the warn-mode
    # winner it cuts once the added strings are taken in (_settled).
    text = f"mail {EMAIL}"
    winners = redactor._winners(text)
    assert redactor.warn_counts == Counter({"EMAIL": 1})
    plain = _redactor(WARN)
    configured = plain._winners(text)
    added = _plan(("acme",))
    assert added is not None
    found = added.detect(text, Allowlist())
    assert redactor._absorb(text, configured, found) == winners
    assert redactor.warn_counts == Counter({"EMAIL": 1})


class _Recorder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def record(self, text: str, found: object) -> None:
        self.texts.append(text)


@pytest.mark.parametrize(
    ("added", "warned"),
    [(("acme",), Counter({"EMAIL": 1})), ((EMAIL,), Counter()), (("zeta",), Counter({"EMAIL": 1}))],
)
def test_a_recorder_counts_alike(added: tuple[str, ...], warned: Counter[str]) -> None:
    # A scan sink's recorder (with_recorder, _recorded_winners) changes
    # nothing the redaction does or counts.
    text = f"mail {EMAIL} now"
    plain = _redactor(WARN, added=added)
    recorder = _Recorder()
    recorded = _redactor(WARN, added=added).with_recorder(recorder)
    assert recorded.redact_text(text) == plain.redact_text(text)
    assert recorded.warn_counts == plain.warn_counts == warned
    assert recorder.texts == [text]
    recorded.scan(text)
    assert recorded.warn_counts == warned + warned


@pytest.mark.parametrize(
    ("convertible", "outcome", "warned"),
    # Converted: the text sent in the file's place is redacted (and its
    # warning counted) later, so the scan's count is taken back out.
    [(True, "converted", Counter()), (False, "detected", Counter({"EMAIL": 1}))],
)
def test_an_inspected_upload_counts_alike(
    convertible: bool, outcome: str, warned: Counter[str]
) -> None:
    text = f"mail {EMAIL}"
    redactor = _redactor(WARN, added=("acme",))
    verdict = judge(
        {0: Inspection(text, True, "x", convert_text=text)},
        redactor,
        identity=False,
        text_budget=1000,
        convertible=convertible,
    )
    assert verdict.outcomes == Counter({outcome: 1})
    assert redactor.warn_counts == warned
    if convertible:
        # ... where it is counted once.
        assert redactor.redact_text(verdict.converted[0]) == (
            "mail bob.private@«DENY_001»-corp.example"
        )
        assert redactor.warn_counts == Counter({"EMAIL": 1})


IBAN = "GB39 WEST 3607 5983 8675 65"


@pytest.mark.parametrize(
    ("added", "expected", "warned"),
    [
        # None: the card number the paper IBAN holds is redacted, the rest
        # of the warn-mode IBAN sent — a warning.
        ((), "pay GB39 WEST «CREDIT_CARD_001» now", Counter({"IBAN": 1})),
        # One taking the rest of it, joined with the card into one union:
        # nothing of the IBAN is sent, no warning.
        (("GB39 WEST 3607",), "pay «DENY_001» now", Counter()),
        # One touching the card: nothing sent either.
        (("GB39 WEST ",), "pay «DENY_001»«CREDIT_CARD_001» now", Counter()),
        # One taking only part of the rest: what lies between is sent.
        (("GB39",), "pay «DENY_001» WEST «CREDIT_CARD_001» now", Counter({"IBAN": 1})),
        (("pay GB39 WEST",), "«DENY_001» «CREDIT_CARD_001» now", Counter({"IBAN": 1})),
    ],
)
@pytest.mark.parametrize("recorded", [False, True])
def test_a_warn_value_a_joined_run_drops_is_counted_after_the_added_strings(
    added: tuple[str, ...], expected: str, warned: Counter[str], recorded: bool
) -> None:
    # A warn-mode paper IBAN is dropped by the run joining it with the card
    # number it holds (redacted); whether part of it is still sent is
    # decided once the added strings are taken in, not before.
    config = DetectionConfig(modes=(("iban", "warn"),))
    redactor = _redactor(config, added=added)
    if recorded:
        redactor = redactor.with_recorder(_Recorder())
    assert redactor.redact_text(f"pay {IBAN} now") == expected
    assert redactor.warn_counts == warned
    # Detection only, the same warning.
    redactor.scan(f"pay {IBAN} now", redactable=True)
    assert redactor.warn_counts == warned + warned


# --- a warn-mode value an ordinary overlap cuts ----------------------------------------


@pytest.mark.parametrize(
    ("deny", "expected", "warned"),
    [
        # Part of it: the rest is sent, a warning.
        (("acme",), "mail bob.private@«DENY_001»-corp.example now", Counter({"EMAIL": 1})),
        # Cut twice: one value, one warning.
        (
            ("private", "corp"),
            "mail bob.«DENY_001»@acme-«DENY_002».example now",
            Counter({"EMAIL": 1}),
        ),
        # A deny string starting before it and ending inside it.
        (("mail bob",), "«DENY_001».private@acme-corp.example now", Counter({"EMAIL": 1})),
        # The very value, a longer string holding it, two touching inside it:
        # nothing of it is sent, nothing to warn about.
        ((EMAIL,), "mail «DENY_001» now", Counter()),
        ((f"{EMAIL} now",), "mail «DENY_001»", Counter()),
        (("bob.private@", "acme-corp.example"), "mail «DENY_001»«DENY_002» now", Counter()),
    ],
)
def test_a_warn_value_a_configured_deny_string_cuts_is_counted_as_an_added_one_is(
    deny: tuple[str, ...], expected: str, warned: Counter[str]
) -> None:
    # Among the configured detectors a deny string wins every overlap: the
    # warn-mode value loses it, and what of it lies outside the deny strings
    # is sent as it was — a warning, once, exactly as when an access gate
    # adds the same strings: the same bytes upstream, the same count.
    text = f"mail {EMAIL} now"
    configured = DetectionConfig(
        modes=(("email", "warn"),), deny_strings=tuple(DenyEntry(value) for value in deny)
    )
    redactor = _redactor(configured)
    assert redactor.redact_text(text) == expected
    assert redactor.warn_counts == warned
    added = _redactor(WARN, added=deny)
    assert added.redact_text(text) == expected
    assert added.warn_counts == warned
    # Detection only, a recorder attached: the same warning; a check ahead
    # of the redaction counts nothing.
    for redactable in (False, True):
        scanning = _redactor(configured)
        scanning.scan(text, redactable=redactable)
        assert scanning.warn_counts == warned
    recorder = _Recorder()
    recorded = _redactor(configured).with_recorder(recorder)
    assert recorded.redact_text(text) == expected
    assert recorded.warn_counts == warned and recorder.texts == [text]
    checked = _redactor(configured)
    assert checked.blocked_type(text) is None and checked.warn_counts == Counter()


MAILER = "mail bob.private"


@pytest.mark.parametrize(
    ("pattern", "expected", "warned"),
    [
        # Starting before it and ending inside it: the rest of it is sent.
        (r"mail bob\.private", "«LEAD_001»@acme-corp.example now", Counter({"EMAIL": 1})),
        # Starting where it starts and longer: it covers the value whole.
        (r"bob\.private@acme-corp\.example now", "mail «LEAD_001»", Counter()),
    ],
)
def test_a_warn_value_a_value_redacted_in_its_place_cuts_is_counted_when_partly_sent(
    pattern: str, expected: str, warned: Counter[str]
) -> None:
    lead = CustomRule(name="lead", detector_type="LEAD", pattern=pattern)
    config = DetectionConfig(modes=(("email", "warn"),), custom_rules=(lead,))
    redactor = _redactor(config)
    assert redactor.redact_text(f"mail {EMAIL} now") == expected
    assert redactor.warn_counts == warned
    assert redactor.counts == Counter({"LEAD": 1})


def test_a_warn_value_that_lost_only_to_a_value_forwarded_as_sent_is_not_counted() -> None:
    # The email loses the overlap to a value starting before it that is
    # forwarded as sent: nothing of the email is redacted, so the winner is
    # what is counted (a warn-mode one) — or nothing (an approved block).
    text = f"mail {EMAIL} now"
    lead = CustomRule(name="lead", detector_type="LEAD", pattern=r"mail bob\.private")
    warning = _redactor(
        DetectionConfig(modes=(("email", "warn"), ("lead", "warn")), custom_rules=(lead,))
    )
    assert warning.redact_text(text) == text
    assert warning.warn_counts == Counter({"LEAD": 1})
    approved = Approved(MAILER)
    blocking = _redactor(
        DetectionConfig(modes=(("email", "warn"), ("lead", "block")), custom_rules=(lead,))
    ).with_overrides(approved)
    assert blocking.redact_text(text) == text
    assert blocking.warn_counts == Counter()
    assert approved.asked == [("LEAD", MAILER)]
    # One that lost to such a value but that a deny string cuts elsewhere
    # is cut all the same: a warning besides the winner's.
    cut = _redactor(
        DetectionConfig(
            modes=(("email", "warn"), ("lead", "warn")),
            custom_rules=(lead,),
            deny_strings=(DenyEntry("acme"),),
        )
    )
    assert cut.redact_text(text) == "mail bob.private@«DENY_001»-corp.example now"
    assert cut.warn_counts == Counter({"LEAD": 1, "EMAIL": 1})


@pytest.mark.parametrize(("priority", "counted"), [(100, "EMAIL"), (90, "MAIL")])
def test_one_warn_value_two_detectors_find_is_counted_once_when_cut(
    priority: int, counted: str
) -> None:
    # The built-in email rule and a custom rule read the same value; a deny
    # string cuts it: one warning, under the type the overlap sweep reads
    # first (by priority; at a tie, the detector listed first).
    mail = CustomRule(
        name="mail",
        detector_type="MAIL",
        pattern=r"bob\.private@acme-corp\.example",
        priority=priority,
    )
    config = DetectionConfig(
        modes=(("email", "warn"), ("mail", "warn")),
        custom_rules=(mail,),
        deny_strings=(DenyEntry("acme"),),
    )
    redactor = _redactor(config)
    assert (
        redactor.redact_text(f"mail {EMAIL} now") == "mail bob.private@«DENY_001»-corp.example now"
    )
    assert redactor.warn_counts == Counter({counted: 1})


def test_a_deny_string_another_one_wins_against_is_no_warning_whatever_its_type() -> None:
    # Two overlapping deny strings typed after a warn-mode rule's type: the
    # one the deny sweep drops is cut too, but a deny string takes no mode.
    config = DetectionConfig(
        modes=(("email", "warn"),),
        deny_strings=(
            DenyEntry("acme corp", detector_type="EMAIL"),
            DenyEntry("corp secret", detector_type="EMAIL"),
        ),
    )
    redactor = _redactor(config)
    assert (
        redactor.redact_text("status: acme corp secret plan") == "status: «EMAIL_001» secret plan"
    )
    assert redactor.warn_counts == Counter()
    assert redactor.counts == Counter({"EMAIL": 1})


def test_an_llm_finding_a_configured_deny_string_cuts_counts_as_before() -> None:
    # An llm finding (ADDITIVE_TIER) is never dropped by the ordinary
    # overlap: it is cut around the winners and its pieces are what the
    # redaction acts on — each a warn-mode winner, counted as such. One the
    # additive sweep drops for another llm finding is no warning either.
    value = "bob at example dot com"
    text = f"mail {value} please"
    deny = DetectorPlan([DenyDetector([DenyEntry("example")])])

    def redactor(found: list[tuple[str, str, int]]) -> Redactor:
        return Redactor(
            [_Found(found), *deny.detectors],
            InMemoryVault(),
            Allowlist(),
            modes={"EMAIL": "warn", "PERSON": "warn"},
        )

    pieces = redactor([(value, "EMAIL", ADDITIVE_TIER)])
    assert pieces.redact_text(text) == "mail bob at «DENY_001» dot com please"
    assert pieces.warn_counts == Counter({"EMAIL": 2})
    swept = redactor([("mail bob at", "PERSON", ADDITIVE_TIER), (value, "EMAIL", ADDITIVE_TIER)])
    assert swept.redact_text(text) == "mail bob at «DENY_001» dot com please"
    assert swept.warn_counts == Counter({"PERSON": 1})


def test_a_warn_value_a_configured_deny_string_cuts_in_a_converted_upload_counts_once() -> None:
    # Converted: the text sent in the file's place is redacted (and its
    # warning counted) later, so the scan's count is taken back out.
    configured = DetectionConfig(modes=(("email", "warn"),), deny_strings=(DenyEntry("acme"),))
    text = f"mail {EMAIL}"
    redactor = _redactor(configured)
    verdict = judge(
        {0: Inspection(text, True, "x", convert_text=text)},
        redactor,
        identity=False,
        text_budget=1000,
        convertible=True,
    )
    assert verdict.outcomes == Counter({"converted": 1})
    assert redactor.warn_counts == Counter()
    assert redactor.redact_text(verdict.converted[0]) == "mail bob.private@«DENY_001»-corp.example"
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


_WORDS = st.sampled_from(
    # The whole email too, so that most texts hold one for the deny strings
    # and the custom rule to overlap.
    [EMAIL, "bob", ".", "private", "@", "acme", "-", "corp", ".example", " ", "secret", "plan"]
    + ["zeta"]
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


# A custom rule starting inside an email and running on past it: it loses
# the overlap sweep to the email (which starts earlier), and to a deny string.
TAIL = CustomRule(name="tail", detector_type="TAIL", pattern=r"\.example [a-z]+")
_MODE = st.sampled_from(["redact", "warn", "block"])


@settings(deadline=None, max_examples=300)
@given(
    words=st.lists(_WORDS, max_size=14),
    configured=_CONFIGURED,
    added=_ADDED,
    mode=_MODE,
    tail_mode=_MODE,
    approve=st.booleans(),
)
def test_added_strings_redact_a_superset_and_refuse_alike(
    words: list[str],
    configured: list[str],
    added: list[str],
    mode: str,
    tail_mode: str,
    approve: bool,
) -> None:
    text = "".join(words)
    config = DetectionConfig(
        modes=tuple(
            (rule, m) for rule, m in (("email", mode), ("tail", tail_mode)) if m != "redact"
        ),
        deny_strings=tuple(DenyEntry(value, detector_type="CODENAME") for value in configured),
        custom_rules=(TAIL,),
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
    redacted = replaced(text, got, vault)
    # A SUPERSET of what the configured policy redacts...
    assert replaced(text, expected, base_vault) <= redacted
    # ...and every occurrence of every added string.
    for value in added:
        for match in re.finditer(re.escape(value), text, re.IGNORECASE):
            assert set(range(match.start(), match.end())) <= redacted
    # A warn-mode value is a warning exactly when part of it is still sent,
    # once — with the added strings and without them: one a configured deny
    # string or a value redacted in its place cuts as well (the plain path).
    plain = _redactor(config)
    configured_winners = plain._winners(text)
    found = plain.plan.detect(text, build_allowlist(config))
    modes = build_modes(config)
    assert tightened.warn_counts == expected_warnings(found, configured_winners, modes, redacted)
    assert base.warn_counts == expected_warnings(
        found, configured_winners, modes, replaced(text, expected, base_vault)
    )


_SPANS = st.lists(
    st.tuples(
        st.integers(0, 40),
        st.integers(1, 12),
        st.sampled_from(["EMAIL", "PHONE"]),
        st.sampled_from([1, ADDITIVE_TIER]),
    ),
    max_size=5,
)


@settings(deadline=None, max_examples=300)
@given(
    words=st.lists(_WORDS, min_size=1, max_size=14),
    spans=_SPANS,
    added=_ADDED,
    modes=st.dictionaries(st.sampled_from(["EMAIL", "PHONE"]), st.sampled_from(["warn", "block"])),
    approve=st.sets(st.integers(0, 4)),
)
def test_added_strings_put_no_new_value_to_the_overrides_and_blocked_type_agrees(
    words: list[str],
    spans: list[tuple[int, int, str, int]],
    added: list[str],
    modes: dict[str, str],
    approve: set[int],
) -> None:
    text = "".join(words)
    found = []
    for start, length, detector_type, tier in spans:
        surface = text[start : start + length]
        if surface.strip():
            found.append((surface, detector_type, tier))
    # The values resolution hands on without the added strings (an llm
    # finding is already cut around the others there, by design).
    plain = Redactor([_Found(found)], InMemoryVault(), Allowlist(), modes=modes)
    configured = plain._winners(text)
    values = {d.value for d in configured}
    approved = Approved(*(found[i][0] for i in approve if i < len(found)))
    vault = InMemoryVault()
    redactor = Redactor(
        [_Found(found)], vault, Allowlist(), modes=modes, added_deny=_plan(tuple(added))
    ).with_overrides(approved)
    expected = redactor.blocked_type(text)
    # A check ahead of the redaction counts nothing.
    assert redactor.warn_counts == Counter()
    try:
        got = redactor.redact_text(text)
    except BlockedRequest as refused:
        assert refused.detector_type == expected
    else:
        assert expected is None
        # A warn-mode value — an llm one too, never cut into fragments — is
        # a warning exactly when part of it is still sent, once; an
        # approved block never.
        found = plain.plan.detect(text, Allowlist())
        assert redactor.warn_counts == expected_warnings(
            found, configured, modes, replaced(text, got, vault)
        )
    # The added strings never make the overrides hear a new value (a
    # fragment of one cut around them).
    assert {value for _type, value in approved.asked} <= values


class _Spans:
    """A stand-in detector finding exactly ``found`` in any text."""

    name = "spans"

    def __init__(self, found: list[Detection]) -> None:
        self.found = found

    def detect(self, text: str) -> list[Detection]:
        return list(self.found)


# Every character once (and none a case-insensitive match of another): a
# string occurs at one place, so spans and added strings are positions.
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"
_JOINING_SPANS = st.lists(
    st.tuples(
        st.integers(0, len(_LETTERS) - 1),
        st.integers(1, 12),
        st.sampled_from(["EMAIL", "PHONE", "IBAN"]),
        # A configured deny string (tier 0: never in a mode, whatever its
        # type), a value, an llm finding.
        st.sampled_from([0, 1, 1, 1, ADDITIVE_TIER]),
        st.booleans(),  # joins (never a deny string or an llm finding)
        st.sampled_from([110, 120]),  # priority
    ),
    max_size=6,
)
_ADDED_SPANS = st.lists(
    st.tuples(st.integers(0, len(_LETTERS) - 1), st.integers(1, 10)), min_size=1, max_size=3
)


@settings(deadline=None, max_examples=1000)
@given(
    spans=_JOINING_SPANS,
    added=_ADDED_SPANS,
    warned=st.sets(st.sampled_from(["EMAIL", "PHONE", "IBAN"])),
)
def test_a_warn_value_is_a_warning_exactly_when_part_of_it_is_still_sent(
    spans: list[tuple[int, int, str, int, bool, int]],
    added: list[tuple[int, int]],
    warned: set[str],
) -> None:
    # Configured deny strings, joined runs (Detection.joins: a card read
    # inside a run of digit groups, a paper IBAN), llm findings and added
    # strings together, in redact and warn mode only, so nothing refuses: a
    # warn-mode value is a warning exactly when part of it is still sent — a
    # winner, one an added string takes part of, one a joined run drops, one
    # a deny string or a value redacted in its place wins an ordinary
    # overlap against — and never when the redacted values cover it whole,
    # nor when it lost only to values forwarded as sent; decided after the
    # added strings are taken in; with them and without.
    text = _LETTERS
    found = []
    for start, length, kind, tier, joins, priority in spans:
        end = min(start + length, len(text))
        joining = joins and tier == 1
        found.append(Detection(start, end, kind, text[start:end], priority, tier, joining))
    modes = dict.fromkeys(warned, "warn")
    plain = Redactor([_Spans(found)], InMemoryVault(), Allowlist(), modes=modes)
    configured = plain._winners(text)
    detections = plain.plan.detect(text, Allowlist())
    values = tuple(text[start : start + length] for start, length in added)
    for added_deny in (None, _plan(values)):
        vault = InMemoryVault()
        redactor = Redactor([_Spans(found)], vault, Allowlist(), modes=modes, added_deny=added_deny)
        got = redactor.redact_text(text)
        redacted = replaced(text, got, vault)
        assert redactor.warn_counts == expected_warnings(detections, configured, modes, redacted)
