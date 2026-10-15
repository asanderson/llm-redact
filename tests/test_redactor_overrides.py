"""``Redactor.with_overrides``: the requester's approved overrides are
asked only where detection refuses — a block-mode winner in
``redact_text``/``blocked_type``, every refusing winner in ``scan_text``
(a span joining values each of them, so one code names them all) — never
for a value the redaction replaces or a warn-mode one. An approved value
stays in place (forwarded as sent, uncounted); the thin copies keep the
overrides; a redactor without them behaves exactly as before."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from llm_redact.detection.engine import (
    DetectionConfig,
    build_allowlist,
    build_detectors,
    build_modes,
)
from llm_redact.overrides import OverrideScope, OverrideStore
from llm_redact.redactor import BlockedRequest, Redactor
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"
OTHER = "john.roe@corp.example"
PHONE = "+1 415 555 0132"


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


def _redactor(modes: tuple[tuple[str, str], ...] = (("email", "block"),)) -> Redactor:
    config = DetectionConfig(modes=modes)
    return Redactor(
        build_detectors(config),
        InMemoryVault(),
        build_allowlist(config),
        modes=build_modes(config),
    )


def test_redact_text_passes_only_an_approved_block_value() -> None:
    base = _redactor()
    approved = Approved(EMAIL)
    redactor = base.with_overrides(approved)
    assert redactor.redact_text(f"mail {EMAIL}") == f"mail {EMAIL}"
    assert redactor.counts == Counter() and redactor.warn_counts == Counter()
    assert approved.asked == [("EMAIL", EMAIL)]
    with pytest.raises(BlockedRequest):
        redactor.redact_text(f"mail {OTHER}")
    with pytest.raises(BlockedRequest):
        base.redact_text(f"mail {EMAIL}")  # the original asks nothing


def test_redaction_and_warn_never_ask() -> None:
    approved = Approved(EMAIL, PHONE)
    redactor = _redactor((("phone_number", "warn"),)).with_overrides(approved)
    assert redactor.redact_text(f"{EMAIL} {PHONE}") == f"«EMAIL_001» {PHONE}"
    assert approved.asked == []


def test_thin_copies_keep_the_overrides() -> None:
    redactor = _redactor().with_overrides(Approved(EMAIL))
    copy = redactor.with_budget(10).with_floors({"EMAIL": 5})
    assert copy.redact_text(EMAIL) == EMAIL


def test_with_overrides_keeps_the_floors_and_the_budget() -> None:
    """The proxy adds the overrides LAST, onto the request's floored and
    budgeted copy: the overrides copy keeps both — a new value is still
    numbered above the request's tokens, and the string budget still
    refuses the body."""
    from llm_redact.redactor import TooManyStrings

    floored = _redactor(modes=()).with_floors({"EMAIL": 7}).with_overrides(Approved())
    assert floored.redact_text(f"mail {EMAIL}") == "mail «EMAIL_008»"
    budgeted = _redactor(modes=()).with_budget(1).with_overrides(Approved())
    budgeted.redact_text("first")
    with pytest.raises(TooManyStrings):
        budgeted.redact_text("second")


def test_blocked_type_skips_an_approved_value() -> None:
    approved = Approved(EMAIL)
    redactor = _redactor().with_overrides(approved)
    assert redactor.blocked_type(f"mail {EMAIL}") is None
    assert redactor.blocked_type(f"mail {OTHER}") == "EMAIL"
    assert _redactor().blocked_type(f"mail {EMAIL}") == "EMAIL"
    assert redactor.blocks


def test_scan_text_asks_for_every_refusing_winner() -> None:
    approved = Approved(EMAIL)
    redactor = _redactor(()).with_overrides(approved)
    assert redactor.scan_text(f"{EMAIL} {OTHER}") == Counter({"EMAIL": 1})
    assert approved.asked == [("EMAIL", EMAIL), ("EMAIL", OTHER)]
    blocking = _redactor().with_overrides(Approved(EMAIL))
    assert blocking.scan_text(f"mail {EMAIL}") == Counter()
    with pytest.raises(BlockedRequest):
        blocking.scan_text(f"mail {OTHER}")
    warned = _redactor((("email", "warn"),)).with_overrides(approved := Approved())
    assert warned.scan_text(EMAIL) == Counter() and approved.asked == []
    assert warned.warn_counts == Counter({"EMAIL": 1})


def test_a_deny_string_is_never_put_to_the_overrides() -> None:
    """Deny strings are the operator's always-redact list: where one would
    refuse a request (``scan_text``: a verbatim field, a binary upload's
    text) no requester's override is asked, so none can let it through."""
    from llm_redact.detection.deny import DenyEntry

    config = DetectionConfig(deny_strings=(DenyEntry("project aurora"),))
    approved = Approved("project aurora", EMAIL)
    redactor = Redactor(
        build_detectors(config),
        InMemoryVault(),
        build_allowlist(config),
        modes=build_modes(config),
    ).with_overrides(approved)
    assert redactor.scan_text(f"project aurora {EMAIL}") == Counter({"DENY": 1})
    assert approved.asked == [("EMAIL", EMAIL)]
    # The refusal it causes is final: no value code (OverrideScope).
    assert approved.final == 1
    # Each deny string found is counted.
    twice = redactor.scan("project aurora, again project aurora", redactable=True)
    assert twice.found == Counter({"DENY": 2})
    # In text the caller redacts (convert mode) it refuses nothing.
    assert redactor.scan("project aurora", redactable=True).found == Counter({"DENY": 1})
    assert approved.final == 1
    # Without overrides there is nobody to tell.
    plain = Redactor(
        build_detectors(config), InMemoryVault(), build_allowlist(config), modes=build_modes(config)
    )
    assert plain.scan_text("project aurora") == Counter({"DENY": 1})


def test_scan_reports_what_an_override_let_through() -> None:
    approved = Approved(EMAIL)
    redactor = _redactor(()).with_overrides(approved)
    assert redactor.scan(f"mail {EMAIL}") == (Counter(), True)
    assert redactor.scan(f"mail {OTHER}") == (Counter({"EMAIL": 1}), False)
    assert redactor.scan("nothing") == (Counter(), False)
    blocking = _redactor().with_overrides(Approved(EMAIL))
    assert blocking.scan(f"mail {EMAIL}") == (Counter(), True)


def test_a_redactable_scan_puts_only_block_winners_to_the_overrides() -> None:
    """Convert mode: a reading whose text replaces its file is redacted, so
    a redact-mode value there is found (to be redacted), never asked — an
    approval must not turn it into a file sent as it came — while a
    block-mode value still refuses unless approved."""
    approved = Approved(EMAIL, PHONE)
    redactor = _redactor((("phone_number", "block"),)).with_overrides(approved)
    scan = redactor.scan(f"{EMAIL} {OTHER} {PHONE}", redactable=True)
    assert scan == (Counter({"EMAIL": 2}), True)
    assert approved.asked == [("PHONE", PHONE)]
    with pytest.raises(BlockedRequest):
        redactor.scan("+1 212 555 0198", redactable=True)
    assert _redactor(()).scan(EMAIL, redactable=True) == (Counter({"EMAIL": 1}), False)


# --- a block an access gate's detection overlay added ---------------------------------


def _overlay_block(*final: str) -> Redactor:
    """The configured policy redacts emails and blocks phones; an overlay
    escalated ``final`` (detector types) to block on top of it."""
    config = DetectionConfig(modes=(("phone_number", "block"),))
    return Redactor(
        build_detectors(config),
        InMemoryVault(),
        build_allowlist(config),
        modes={**build_modes(config), "EMAIL": "block"},
        final_blocks=frozenset(final),
    )


def test_a_block_the_overlay_added_is_never_put_to_the_overrides() -> None:
    # An approval would forward, as sent, a value the configured policy
    # REDACTS (which never mints a code): the refusal carries none, and no
    # grant is consulted or consumed.
    approved = Approved(EMAIL)
    redactor = _overlay_block("EMAIL").with_overrides(approved)
    with pytest.raises(BlockedRequest) as caught:
        redactor.redact_text(f"mail {EMAIL}")
    assert caught.value.detector_type == "EMAIL"
    assert redactor.blocked_type(f"mail {EMAIL}") == "EMAIL"
    for redactable in (False, True):
        with pytest.raises(BlockedRequest):
            redactor.scan(f"mail {EMAIL}", redactable=redactable)
    assert approved.asked == [] and approved.final == 4


def test_a_configured_block_beside_it_stays_overridable() -> None:
    approved = Approved(PHONE)
    redactor = _overlay_block("EMAIL").with_overrides(approved)
    assert redactor.redact_text(f"call {PHONE}") == f"call {PHONE}"
    assert redactor.blocked_type(f"call {PHONE}") is None
    assert approved.asked == [("PHONE", PHONE), ("PHONE", PHONE)] and approved.final == 0


def test_the_overlays_block_without_overrides_refuses_alike() -> None:
    with pytest.raises(BlockedRequest):
        _overlay_block("EMAIL").redact_text(f"mail {EMAIL}")
    assert _overlay_block("EMAIL").blocked_type(f"mail {EMAIL}") == "EMAIL"


def test_the_thin_copies_keep_the_overlays_blocks() -> None:
    approved = Approved(EMAIL)
    base = _overlay_block("EMAIL")
    for copy in (base.with_budget(10), base.with_floors({"EMAIL": 5})):
        with pytest.raises(BlockedRequest):
            copy.with_overrides(approved).redact_text(f"mail {EMAIL}")
    assert approved.asked == [] and approved.final == 2


# --- a span joining values (Redactor._join) -------------------------------------------

CARD = "4111 1111 1111 1111"
UNION = f"Order 12 {CARD} Main Street"


def _joining(tmp_path: Path) -> tuple[Redactor, OverrideStore]:
    """A card read inside a run of digit groups joined with the address a
    custom rule reads from inside it: ONE span in text the proxy cannot
    rewrite; and the override store a scope reads."""
    from llm_redact.detection.engine import CustomRule

    street = CustomRule(name="street", detector_type="ADDRESS", pattern=r"\d{4} [A-Z][a-z]+ Street")
    config = DetectionConfig(custom_rules=(street,))
    redactor = Redactor(
        build_detectors(config), InMemoryVault(), build_allowlist(config), modes=build_modes(config)
    )
    return redactor, OverrideStore(tmp_path / "overrides.db")


def _refused_code(scope: OverrideScope) -> str:
    code = scope.refusal_code("verbatim_field", "openai", "POST", "/v1/fine_tuning/jobs")
    assert code is not None
    return code


def test_one_code_approves_every_value_a_joined_span_stands_for(tmp_path: Path) -> None:
    redactor, store = _joining(tmp_path)
    refused = OverrideScope(store, "")
    assert redactor.with_overrides(refused).scan_text(UNION) == Counter({"CREDIT_CARD": 1})
    store.approve("once", approver=None, code=_refused_code(refused))
    # The next scan passes the span as sent; each value counted once in the
    # content facts, however often it is asked.
    scope = OverrideScope(store, "")
    passing = redactor.with_overrides(scope)
    assert passing.scan(UNION) == (Counter(), True)
    assert passing.scan(f"again: {UNION}") == (Counter(), True)
    assert scope.allowed_types() == {"CREDIT_CARD": 1, "ADDRESS": 1}
    assert scope.commit() == (True, "once")
    scope.settle(sent=True)
    # The one-time grant is used up: refused again.
    again = OverrideScope(store, "")
    assert redactor.with_overrides(again).scan_text(UNION) == Counter({"CREDIT_CARD": 1})
    assert again.allowed_types() == {}


def test_an_every_time_approval_is_per_value(tmp_path: Path) -> None:
    redactor, store = _joining(tmp_path)
    refused = OverrideScope(store, "")
    assert redactor.with_overrides(refused).scan_text(UNION) == Counter({"CREDIT_CARD": 1})
    store.approve("always", approver=None, code=_refused_code(refused))
    # Each value is a rule of its own: the same card joined with another
    # address refuses, and its code names that address only.
    other = UNION.replace("Main", "Elm")
    scope = OverrideScope(store, "")
    assert redactor.with_overrides(scope).scan_text(other) == Counter({"CREDIT_CARD": 1})
    store.approve("once", approver=None, code=_refused_code(scope))
    entries = [e for e in store.entries() if e.state == "once"]
    assert [e.types for e in entries] == [("ADDRESS",)]
    passing = OverrideScope(store, "")
    assert redactor.with_overrides(passing).scan(other) == (Counter(), True)
    assert passing.allowed_types() == {"CREDIT_CARD": 1, "ADDRESS": 1}


def test_a_partly_approved_span_still_refuses(tmp_path: Path) -> None:
    redactor, store = _joining(tmp_path)
    # Approve the card alone, as a block-mode refusal's code would.
    alone = OverrideScope(store, "")
    assert not alone.allows("CREDIT_CARD", CARD)
    store.approve("always", approver=None, code=_refused_code(alone))
    scope = OverrideScope(store, "")
    assert redactor.with_overrides(scope).scan(UNION) == (Counter({"CREDIT_CARD": 1}), False)
    # The refusal's code names the address the card's rule does not pass.
    store.approve("always", approver=None, code=_refused_code(scope))
    assert redactor.with_overrides(OverrideScope(store, "")).scan(UNION).overridden
