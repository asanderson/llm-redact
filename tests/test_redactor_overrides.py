"""``Redactor.with_overrides``: the requester's approved overrides are
asked only where detection refuses — a block-mode winner in
``redact_text``/``blocked_type``, every refusing winner in ``scan_text`` —
never for a value the redaction replaces or a warn-mode one. An approved
value stays in place (forwarded as sent, uncounted); the thin copies keep
the overrides; a redactor without them behaves exactly as before."""

from __future__ import annotations

from collections import Counter

import pytest

from llm_redact.detection.engine import (
    DetectionConfig,
    build_allowlist,
    build_detectors,
    build_modes,
)
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
