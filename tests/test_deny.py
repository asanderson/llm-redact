"""User deny strings: detector, precedence, config surfaces."""

import pytest

from llm_redact.config import ConfigError, parse_config
from llm_redact.detection.deny import DenyDetector, DenyEntry
from llm_redact.detection.engine import (
    DetectionConfig,
    build_allowlist,
    build_detectors,
)
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault

# ---- DenyDetector ----


def _detections(entries: list[DenyEntry], text: str) -> list[tuple[int, int, str, str]]:
    return [(d.start, d.end, d.detector_type, d.value) for d in DenyDetector(entries).detect(text)]


def test_case_insensitive_by_default() -> None:
    found = _detections([DenyEntry("aurora")], "AURORA and Aurora and aurora")
    assert [(s, e) for s, e, _t, _v in found] == [(0, 6), (11, 17), (22, 28)]
    # The matched value keeps the original casing so rehydration is exact.
    assert [v for _s, _e, _t, v in found] == ["AURORA", "Aurora", "aurora"]


def test_case_sensitive_when_asked() -> None:
    found = _detections([DenyEntry("Aurora", case_sensitive=True)], "AURORA Aurora aurora")
    assert [(s, e) for s, e, _t, _v in found] == [(7, 13)]


def test_substring_semantics_no_word_boundary() -> None:
    found = _detections([DenyEntry("aurora")], "Auroras are pretty")
    assert [(s, e) for s, e, _t, _v in found] == [(0, 6)]


def test_literal_not_regex() -> None:
    found = _detections([DenyEntry("a.c")], "abc a.c")
    assert [(s, e) for s, e, _t, _v in found] == [(4, 7)]


def test_custom_type_and_tier() -> None:
    detections = list(DenyDetector([DenyEntry("x1z", detector_type="CODE")]).detect("a x1z"))
    assert detections[0].detector_type == "CODE"
    assert detections[0].tier == 0
    assert detections[0].priority == 0


def test_turkish_dotted_i_offsets_stay_correct() -> None:
    # str.lower() would change this string's length; IGNORECASE must not.
    text = "prefix İstanbul suffix"
    found = _detections([DenyEntry("İstanbul")], text)
    assert [(s, e) for s, e, _t, _v in found] == [(7, 15)]
    assert text[7:15] == "İstanbul"


# ---- precedence through the full redactor ----


def _redactor(vault: InMemoryVault, config: DetectionConfig, **kwargs) -> Redactor:
    return Redactor(build_detectors(config), vault, build_allowlist(config), **kwargs)


def test_deny_wins_inside_longer_rule_match(vault: InMemoryVault) -> None:
    # The email is longer and starts earlier; the deny substring must still
    # win the overlap ("highest precedence").
    config = DetectionConfig(deny_strings=(DenyEntry("corp.example"),))
    out = _redactor(vault, config).redact_text("mail jane@corp.example now")
    assert "«DENY_001»" in out
    assert "corp.example" not in out
    assert "jane@" in out  # the non-deny remainder of the email is not an email anymore


def test_deny_wins_over_warn_mode_engulfing(vault: InMemoryVault) -> None:
    # A warn-mode EMAIL match overlapping a deny span is dropped entirely:
    # the deny redacts, and no warn is counted for the dropped match.
    config = DetectionConfig(deny_strings=(DenyEntry("corp.example"),))
    redactor = _redactor(vault, config, modes={"EMAIL": "warn"})
    out = redactor.redact_text("mail jane@corp.example now")
    assert "«DENY_001»" in out
    assert redactor.warn_counts == {}


def test_deny_suppresses_block_mode_match(vault: InMemoryVault) -> None:
    # Deliberate pin: the deny span wins the overlap, so the block-mode
    # EMAIL match never reaches dispatch and the request goes through with
    # the deny span redacted.
    config = DetectionConfig(deny_strings=(DenyEntry("corp.example"),))
    redactor = _redactor(vault, config, modes={"EMAIL": "block"})
    out = redactor.redact_text("mail jane@corp.example now")
    assert "«DENY_001»" in out


def test_deny_never_subject_to_modes_even_on_type_collision(vault: InMemoryVault) -> None:
    # A deny entry whose type collides with a warn-moded rule type must
    # still redact: tier-0 dispatch is structural, not type-keyed.
    config = DetectionConfig(deny_strings=(DenyEntry("hunter2", detector_type="SECRET"),))
    redactor = _redactor(vault, config, modes={"SECRET": "warn"})
    out = redactor.redact_text("value hunter2 here")
    assert "«SECRET_001»" in out
    assert redactor.warn_counts == {}


def test_deny_bypasses_allowlist(vault: InMemoryVault) -> None:
    config = DetectionConfig(
        allowlist=("keep@corp.example",),
        deny_strings=(DenyEntry("keep@corp.example"),),
    )
    out = _redactor(vault, config).redact_text("mail keep@corp.example now")
    assert "keep@corp.example" not in out


def test_overlapping_deny_entries_longest_wins(vault: InMemoryVault) -> None:
    config = DetectionConfig(deny_strings=(DenyEntry("aurora"), DenyEntry("project aurora")))
    out = _redactor(vault, config).redact_text("the project aurora launch")
    assert out == "the «DENY_001» launch"
    assert vault.original_for("«DENY_001»") == "project aurora"


def test_casing_variants_round_trip_exactly(vault: InMemoryVault) -> None:
    config = DetectionConfig(deny_strings=(DenyEntry("aurora"),))
    redactor = _redactor(vault, config)
    rehydrator = Rehydrator(vault)
    text = "Aurora then AURORA"
    out = redactor.redact_text(text)
    assert "Aurora" not in out and "AURORA" not in out
    assert rehydrator.rehydrate_text(out) == text


def test_no_deny_entries_leaves_pipeline_untouched(vault: InMemoryVault) -> None:
    config = DetectionConfig()
    out = _redactor(vault, config).redact_text("mail jane@corp.example now")
    assert out == "mail «EMAIL_001» now"


# ---- config parsing ----


def test_deny_sugar_folds_to_entries() -> None:
    config = parse_config({"detection": {"deny": ["beta", "Alpha"]}}, "t")
    assert config.detection.deny_strings == (
        DenyEntry("Alpha", False, "DENY"),
        DenyEntry("beta", False, "DENY"),
    )


def test_deny_strings_full_form() -> None:
    config = parse_config(
        {
            "detection": {
                "deny_strings": [{"value": "Aurora", "case_sensitive": True, "type": "PROJECT"}]
            }
        },
        "t",
    )
    assert config.detection.deny_strings == (DenyEntry("Aurora", True, "PROJECT"),)


def test_deny_both_surfaces_merge() -> None:
    config = parse_config({"detection": {"deny": ["a"], "deny_strings": [{"value": "b"}]}}, "t")
    assert {e.value for e in config.detection.deny_strings} == {"a", "b"}


@pytest.mark.parametrize(
    "raw",
    [
        {"deny": [""]},  # empty value
        {"deny": ["has«guillemet"]},  # would collide with the token grammar
        {"deny_strings": [{"value": "x", "type": "lower"}]},  # bad type grammar
        {"deny_strings": [{"value": "x", "type": "A" * 21}]},  # type too long
        {"deny_strings": [{"nope": "x"}]},  # missing value key
        {"deny_strings": [{"value": "x", "extra": 1}]},  # unknown key
        {"deny": ["dup", "dup"]},  # duplicate
        {"deny": "not-a-list"},  # wrong shape (editor JSON path)
        {"deny_strings": {"value": "x"}},  # wrong shape
    ],
)
def test_bad_deny_config_rejected(raw: dict) -> None:
    with pytest.raises(ConfigError):
        parse_config({"detection": raw}, "t")


def test_deny_error_messages_never_carry_the_value() -> None:
    secret = "SuperSecretCodename«"
    with pytest.raises(ConfigError) as excinfo:
        parse_config({"detection": {"deny": [secret]}}, "t")
    assert "SuperSecretCodename" not in str(excinfo.value)


# ---- sweep equivalence when no deny entries exist ----


def test_all_tier1_resolution_identical_to_plain_sweep() -> None:
    import random

    from llm_redact.detection.base import Detection
    from llm_redact.redactor import _resolve_overlaps, _sweep

    rng = random.Random(7)
    for _ in range(200):
        detections = []
        for _n in range(rng.randrange(12)):
            start = rng.randrange(50)
            detections.append(
                Detection(
                    start=start,
                    end=start + rng.randrange(1, 12),
                    detector_type="T",
                    value="v",
                    priority=rng.choice([1, 10, 100]),
                )
            )
        detections.sort(key=lambda d: (d.start, -(d.end - d.start), d.priority))
        assert _resolve_overlaps(detections) == _sweep(detections)
