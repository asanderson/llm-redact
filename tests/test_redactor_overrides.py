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

    def allows(self, detector_type: str, value: str) -> bool:
        self.asked.append((detector_type, value))
        return value in self.values


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
