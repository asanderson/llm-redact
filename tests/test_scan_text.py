"""``Redactor.scan_text``: detection without redaction.

The text of a binary upload read by an upload inspector cannot be
rewritten, so it is scanned instead: exactly the detections ``redact_text``
would act on (allowlists, per-type allowlists, deny strings, overlap
resolution, modes), no placeholder issued, nothing written to the vault. A
block-mode winner raises, a warn-mode winner is counted and left alone,
every other winner — a deny string whatever the modes — is returned by
type. Charged against the body's string budget as one string.

``Redactor.blocked_type`` is the block check ahead of a redaction (an
upload's pieces, before its binary parts are inspected): the type
``redact_text`` would refuse the text for, else None — nothing issued,
counted or charged. ``Redactor.blocks`` says whether it can find one.
"""

from __future__ import annotations

from collections import Counter

import pytest

from llm_redact.config import parse_config
from llm_redact.detection.engine import (
    DetectionConfig,
    build_allowlist,
    build_detectors,
    build_modes,
)
from llm_redact.redactor import BlockedRequest, Redactor, TooManyStrings
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"
OTHER = "john.roe@corp.example"
IP = "10.20.30.40"


def _redactor(config: DetectionConfig, vault: InMemoryVault | None = None) -> Redactor:
    return Redactor(
        build_detectors(config),
        vault if vault is not None else InMemoryVault(),
        build_allowlist(config),
        modes=build_modes(config),
    )


def _detection(**raw: object) -> DetectionConfig:
    return parse_config({"detection": raw}, "t").detection


def test_returns_the_types_it_would_redact_with_their_counts() -> None:
    vault = InMemoryVault()
    redactor = _redactor(DetectionConfig(), vault)
    found = redactor.scan_text(f"mail {EMAIL} and {OTHER} from {IP}")
    assert found == Counter({"EMAIL": 2, "IPV4": 1})
    # Nothing issued, nothing counted as redacted.
    assert len(vault) == 0 and redactor.counts == {} and redactor.warn_counts == {}
    assert redactor.scan_text("nothing to see") == Counter()


def test_the_same_detections_as_redaction() -> None:
    redactor = _redactor(DetectionConfig())
    text = f"key sk-ant-api03-abcdefghijklmnopqrstuv, mail {EMAIL}"
    found = redactor.scan_text(text)
    redacted = _redactor(DetectionConfig()).redact_text(text)
    assert found == Counter({"ANTHROPIC_KEY": 1, "EMAIL": 1})
    assert "«ANTHROPIC_KEY_001»" in redacted and "OPENAI" not in redacted


def test_block_mode_raises_with_the_type() -> None:
    vault = InMemoryVault()
    redactor = _redactor(_detection(modes={"email": "block"}), vault)
    with pytest.raises(BlockedRequest) as caught:
        redactor.scan_text(f"from {IP} mail {EMAIL}")
    assert caught.value.detector_type == "EMAIL"
    assert len(vault) == 0


def test_warn_mode_is_counted_per_value_and_not_returned() -> None:
    redactor = _redactor(_detection(modes={"email": "warn"}))
    found = redactor.scan_text(f"{EMAIL} {OTHER} {IP}")
    assert found == Counter({"IPV4": 1})
    assert redactor.warn_counts == Counter({"EMAIL": 2})


def test_a_warn_before_a_block_is_counted_like_redact_text() -> None:
    redactor = _redactor(_detection(modes={"email": "warn", "ipv4": "block"}))
    with pytest.raises(BlockedRequest) as caught:
        redactor.scan_text(f"{EMAIL} then {IP}")
    assert caught.value.detector_type == "IPV4"
    assert redactor.warn_counts == Counter({"EMAIL": 1})


@pytest.mark.parametrize("mode", ["warn", "block", "redact"])
def test_a_deny_string_is_always_found_whatever_its_types_mode(mode: str) -> None:
    config = _detection(
        deny_strings=[{"value": "project nightjar", "type": "EMAIL"}], modes={"email": mode}
    )
    redactor = _redactor(config)
    assert redactor.scan_text("the Project Nightjar budget") == Counter({"EMAIL": 1})
    assert redactor.warn_counts == {}


def test_a_deny_string_wins_the_overlap() -> None:
    redactor = _redactor(_detection(deny=["doe@corp"]))
    assert redactor.scan_text(f"mail {EMAIL}") == Counter({"DENY": 1})


def test_allowlists_apply() -> None:
    assert _redactor(DetectionConfig(allowlist=(EMAIL,))).scan_text(EMAIL) == Counter()
    by_type = DetectionConfig(allowlist_by_type=(("EMAIL", (EMAIL,)),))
    assert _redactor(by_type).scan_text(f"{EMAIL} {OTHER}") == Counter({"EMAIL": 1})


def test_each_scan_is_one_string_against_the_budget() -> None:
    redactor = _redactor(DetectionConfig()).with_budget(2)
    redactor.scan_text(EMAIL)
    redactor.scan_text(EMAIL)
    with pytest.raises(TooManyStrings):
        redactor.scan_text("a third")
    # A shared redactor (no budget) never refuses.
    unbounded = _redactor(DetectionConfig())
    for _ in range(3):
        unbounded.scan_text(EMAIL)


# --- blocked_type: the block check ahead of the redaction -----------------------------


def test_blocks_says_whether_a_rule_is_in_block_mode() -> None:
    assert not _redactor(DetectionConfig()).blocks
    assert not _redactor(_detection(modes={"email": "warn"})).blocks
    assert _redactor(_detection(modes={"email": "warn", "ipv4": "block"})).blocks


def test_blocked_type_is_the_type_redact_text_refuses_for() -> None:
    vault = InMemoryVault()
    config = _detection(modes={"email": "warn", "ipv4": "block"})
    redactor = _redactor(config, vault).with_budget(1)
    text = f"{EMAIL} then {IP}"
    assert redactor.blocked_type(text) == "IPV4"
    assert redactor.blocked_type(text) == "IPV4"  # never charged (a budget of one)
    # Nothing issued or counted — not the warn value either.
    assert len(vault) == 0 and redactor.counts == {} and redactor.warn_counts == {}
    with pytest.raises(BlockedRequest) as caught:
        redactor.redact_text(text)
    assert caught.value.detector_type == "IPV4"


def test_blocked_type_is_none_where_redact_text_goes_through() -> None:
    redactor = _redactor(_detection(modes={"email": "block", "ipv4": "warn"}))
    assert redactor.blocked_type(f"from {IP}") is None  # warn mode
    assert redactor.blocked_type("nothing to see") is None
    other = _redactor(_detection(modes={"ipv4": "block"}))
    assert other.blocked_type(f"mail {EMAIL}") is None  # redact mode


def test_blocked_type_never_blocks_a_deny_string() -> None:
    # A deny string (tier 0) always redacts, whatever its type's mode...
    config = _detection(
        deny_strings=[{"value": "project nightjar", "type": "EMAIL"}], modes={"email": "block"}
    )
    redactor = _redactor(config)
    assert redactor.blocked_type("the Project Nightjar budget") is None
    assert redactor.redact_text("the Project Nightjar budget") == "the «EMAIL_001» budget"
    # ...and wins the overlap with a block-mode match it covers.
    covering = _redactor(_detection(deny=["doe@corp"], modes={"email": "block"}))
    assert covering.blocked_type(f"mail {EMAIL}") is None
    assert covering.redact_text(f"mail {EMAIL}") == "mail jane.«DENY_001».example"


def test_blocked_type_applies_the_allowlists() -> None:
    modes = (("email", "block"),)
    allowed = DetectionConfig(allowlist=(EMAIL,), modes=modes)
    assert _redactor(allowed).blocked_type(EMAIL) is None
    by_type = DetectionConfig(allowlist_by_type=(("EMAIL", (EMAIL,)),), modes=modes)
    assert _redactor(by_type).blocked_type(EMAIL) is None
    assert _redactor(by_type).blocked_type(OTHER) == "EMAIL"
