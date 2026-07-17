"""Per-rule detection modes: parsing, build_modes resolution, redactor dispatch."""

import pytest

from llm_redact.config import ConfigError, parse_config
from llm_redact.detection.engine import (
    CustomRule,
    DetectionConfig,
    build_allowlist,
    build_detectors,
    build_modes,
)
from llm_redact.redactor import BlockedRequest, Redactor
from llm_redact.vault import InMemoryVault

# ---- parse_config ----


def test_modes_parse_sorted_canonical() -> None:
    config = parse_config(
        {"detection": {"modes": {"us_ssn": "block", "phone_number": "warn"}}}, "t"
    )
    assert config.detection.modes == (("phone_number", "warn"), ("us_ssn", "block"))


def test_modes_bad_value_rejected() -> None:
    with pytest.raises(ConfigError, match="'redact', 'warn' or 'block'"):
        parse_config({"detection": {"modes": {"email": "drop"}}}, "t")


def test_modes_absent_is_empty() -> None:
    assert parse_config({}, "t").detection.modes == ()


# ---- build_modes ----


def test_build_modes_empty_default() -> None:
    assert build_modes(DetectionConfig()) == {}


def test_build_modes_maps_rule_name_to_detector_type() -> None:
    config = DetectionConfig(modes=(("phone_number", "warn"), ("us_ssn", "block")))
    assert build_modes(config) == {"PHONE": "warn", "SSN": "block"}


def test_build_modes_drops_explicit_redact() -> None:
    config = DetectionConfig(modes=(("email", "redact"), ("phone_number", "warn")))
    assert build_modes(config) == {"PHONE": "warn"}


def test_build_modes_unknown_rule_name() -> None:
    config = DetectionConfig(modes=(("no_such_rule", "warn"),))
    with pytest.raises(ValueError, match="no_such_rule"):
        build_modes(config)


def test_build_modes_shared_type_conflict_names_both_rules() -> None:
    # github_token and github_fine_grained_pat both emit GITHUB_TOKEN.
    config = DetectionConfig(modes=(("github_fine_grained_pat", "block"), ("github_token", "warn")))
    with pytest.raises(ValueError) as excinfo:
        build_modes(config)
    assert "github_token" in str(excinfo.value)
    assert "github_fine_grained_pat" in str(excinfo.value)
    assert "GITHUB_TOKEN" in str(excinfo.value)


def test_build_modes_shared_type_same_mode_ok() -> None:
    config = DetectionConfig(
        modes=(("github_fine_grained_pat", "block"), ("github_token", "block"))
    )
    assert build_modes(config) == {"GITHUB_TOKEN": "block"}


def test_build_modes_shared_type_redact_plus_other_ok() -> None:
    # An explicit redact is the default, not a conflicting assignment.
    config = DetectionConfig(
        modes=(("github_fine_grained_pat", "redact"), ("github_token", "warn"))
    )
    assert build_modes(config) == {"GITHUB_TOKEN": "warn"}


def test_build_modes_accepts_custom_rule_names() -> None:
    config = DetectionConfig(
        custom_rules=(CustomRule(name="jira", detector_type="TICKET", pattern=r"PROJ-\d+"),),
        modes=(("jira", "warn"),),
    )
    assert build_modes(config) == {"TICKET": "warn"}


# ---- redactor dispatch ----


def _redactor(vault: InMemoryVault, modes: dict[str, str]) -> Redactor:
    config = DetectionConfig()
    return Redactor(build_detectors(config), vault, build_allowlist(config), modes=modes)


def test_warn_leaves_value_and_counts(vault: InMemoryVault) -> None:
    redactor = _redactor(vault, {"PHONE": "warn"})
    out = redactor.redact_text("call +1 415 555 0100 today")
    assert out == "call +1 415 555 0100 today"
    assert redactor.warn_counts["PHONE"] == 1
    assert redactor.counts == {}  # warn is not a redaction
    assert len(vault) == 0  # and issues no placeholder


def test_warn_counts_increment_per_occurrence(vault: InMemoryVault) -> None:
    # Two warn-mode matches of the same type must count as 2, not be
    # overwritten to 1 — the honesty metric reflects every forwarded value.
    redactor = _redactor(vault, {"PHONE": "warn"})
    redactor.redact_text("call +1 415 555 0100 or +1 415 555 0199")
    assert redactor.warn_counts["PHONE"] == 2


def test_block_raises_with_type_only(vault: InMemoryVault) -> None:
    redactor = _redactor(vault, {"SSN": "block"})
    with pytest.raises(BlockedRequest) as excinfo:
        redactor.redact_text("ssn 219-09-9999")
    assert excinfo.value.detector_type == "SSN"
    assert "219-09-9999" not in str(excinfo.value)


def test_mixed_modes_in_one_text(vault: InMemoryVault) -> None:
    redactor = _redactor(vault, {"PHONE": "warn"})
    out = redactor.redact_text("mail jane@corp.example or call +1 415 555 0100")
    assert "«EMAIL_001»" in out
    assert "+1 415 555 0100" in out
    assert redactor.counts["EMAIL"] == 1
    assert redactor.warn_counts["PHONE"] == 1


def test_mode_dispatch_follows_overlap_winner(vault: InMemoryVault) -> None:
    # The anthropic key wins the overlap against the generic sk- rule, so a
    # block mode on the loser must NOT fire: the winner's mode governs.
    redactor = _redactor(vault, {"OPENAI_KEY": "block"})
    out = redactor.redact_text("key sk-ant-api03-abcdefghijklmnopqrstuv end")
    assert "«ANTHROPIC_KEY_001»" in out


def test_block_in_json_body(vault: InMemoryVault) -> None:
    redactor = _redactor(vault, {"EMAIL": "block"})
    with pytest.raises(BlockedRequest):
        redactor.redact_json({"messages": [{"role": "user", "content": "reach jane@corp.example"}]})


def test_modes_non_table_rejected() -> None:
    # TOML cannot produce this, but the /config editor feeds arbitrary JSON
    # through parse_config: it must be a clean ConfigError, not a 500.
    with pytest.raises(ConfigError, match="must be a table"):
        parse_config({"detection": {"modes": ["email", "warn"]}}, "t")
