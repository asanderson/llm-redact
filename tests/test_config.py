from pathlib import Path

import pytest

from llm_redact.config import (
    Config,
    ConfigError,
    apply_env_overrides,
    load_config,
    parse_config,
)


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_REDACT_HOST", "0.0.0.0")
    monkeypatch.setenv("LLM_REDACT_PORT", "9000")
    config = apply_env_overrides(Config())
    assert config.host == "0.0.0.0"
    assert config.port == 9000


def test_env_overrides_absent_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_REDACT_HOST", raising=False)
    monkeypatch.delenv("LLM_REDACT_PORT", raising=False)
    assert apply_env_overrides(Config()) == Config()


def test_env_invalid_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_REDACT_PORT", "not-a-port")
    with pytest.raises(ConfigError, match="LLM_REDACT_PORT"):
        apply_env_overrides(Config())


def test_llm_redact_config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "c.toml"
    config_file.write_text("port = 9123\n")
    monkeypatch.setenv("LLM_REDACT_CONFIG", str(config_file))
    assert load_config().port == 9123


def test_llm_redact_config_missing_is_hard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_REDACT_CONFIG", str(tmp_path / "missing.toml"))
    with pytest.raises(ConfigError, match="missing file"):
        load_config()


def test_defaults_when_nothing_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # no config there
    assert load_config() == Config()


def test_session_ttl_days_parse() -> None:
    assert parse_config({}, "<t>").vault.session_ttl_days == 0  # default: disabled
    assert parse_config({"vault": {"session_ttl_days": 30}}, "<t>").vault.session_ttl_days == 30
    with pytest.raises(ConfigError, match="session_ttl_days must be >= 0"):
        parse_config({"vault": {"session_ttl_days": -1}}, "<t>")


# --- 3.2.1: type validation (the silent-protection-loss class) ---------------
#
# `tuple("^192.168")` iterates a string into single characters, and the
# one-char allowlist patterns `.` / `^` matched EVERY value — a config typo
# that turned all redaction off while --check, doctor, and posture reported
# full protection. Wrong types are hard errors now, never coercions.


@pytest.mark.parametrize(
    "detection_key",
    ["enabled", "allowlist", "allowlist_patterns"],
)
def test_string_where_list_expected_is_rejected(detection_key: str) -> None:
    with pytest.raises(ConfigError, match=rf"{detection_key} must be an array of strings"):
        parse_config({"detection": {detection_key: "^192.168"}}, "<t>")


def test_ner_entities_and_backends_reject_strings() -> None:
    with pytest.raises(ConfigError, match="entities must be an array of strings"):
        parse_config({"detection": {"ner": {"entities": "PERSON"}}}, "<t>")
    with pytest.raises(ConfigError, match="backends must be an array of strings"):
        parse_config({"detection": {"ner": {"backends": "spacy"}}}, "<t>")


def test_custom_rule_required_and_anchors_reject_strings() -> None:
    rule = {"name": "x", "type": "X", "pattern": "abc", "required": "abc"}
    with pytest.raises(ConfigError, match="required must be an array of strings"):
        parse_config({"detection": {"custom_rules": [rule]}}, "<t>")


def test_empty_allowlist_pattern_rejected() -> None:
    # An empty pattern compiles and matches everything — the same
    # protection-loss failure as the string-iteration one.
    with pytest.raises(ConfigError, match="entries must be non-empty"):
        parse_config({"detection": {"allowlist_patterns": [""]}}, "<t>")


@pytest.mark.parametrize(
    ("raw", "where"),
    [
        ({"audit": {"enabled": "false"}}, r"\[audit\] enabled"),
        ({"audit": {"tamper_evident": "false"}}, r"\[audit\] tamper_evident"),
        ({"rehydration": {"fuzzy": "false"}}, r"\[rehydration\] fuzzy"),
        ({"otel": {"enabled": "false"}}, r"\[otel\] enabled"),
        ({"detection": {"ner": {"enabled": "false"}}}, r"\[detection.ner\] enabled"),
        ({"providers": {"openai": {"detection": "false"}}}, r"\[providers.openai\] detection"),
        ({"providers": {"openai": {"enabled": 1}}}, r"\[providers.openai\] enabled"),
        ({"inject_system_note": "no"}, "inject_system_note"),
        ({"email": {"starttls": "true"}}, r"\[email\] starttls"),
    ],
)
def test_quoted_booleans_are_rejected_not_coerced(raw: dict, where: str) -> None:
    # bool("false") is True: a quoted boolean used to mean its OPPOSITE —
    # [audit] enabled = "false" silently turned request-history logging ON.
    with pytest.raises(ConfigError, match=f"{where} must be a boolean"):
        parse_config(raw, "<t>")


def test_deny_strings_case_sensitive_rejects_string() -> None:
    entry = {"value": "hunter2", "case_sensitive": "yes"}
    with pytest.raises(ConfigError, match="case_sensitive must be a boolean"):
        parse_config({"detection": {"deny_strings": [entry]}}, "<t>")


def test_custom_rule_missing_required_keys_is_named_error() -> None:
    # A KeyError here was a raw traceback in serve --check and doctor.
    with pytest.raises(ConfigError, match=r"#1 is missing required key\(s\) \['type'\]"):
        parse_config({"detection": {"custom_rules": [{"name": "x", "pattern": "a"}]}}, "<t>")
    with pytest.raises(ConfigError, match="must be an array of tables"):
        parse_config({"detection": {"custom_rules": "oops"}}, "<t>")


def test_valid_bools_and_lists_still_parse() -> None:
    config = parse_config(
        {
            "audit": {"enabled": False},
            "rehydration": {"fuzzy": True},
            "detection": {"allowlist_patterns": [r"^192\.168\."], "allowlist": ["a@b.example"]},
        },
        "<t>",
    )
    assert config.audit.enabled is False
    assert config.rehydration.fuzzy is True
    assert config.detection.allowlist_patterns == (r"^192\.168\.",)


def test_toml_syntax_error_names_the_file(tmp_path: Path) -> None:
    # Line/column alone is confusing when the effective file came from
    # LLM_REDACT_CONFIG or /etc — the error names the resolved path.
    bad = tmp_path / "config.toml"
    bad.write_text("backend = sqlite\n")  # unquoted string: invalid TOML
    with pytest.raises(ConfigError, match=str(bad).replace("\\", "\\\\")):
        load_config(bad)


def test_unknown_provider_error_points_at_custom() -> None:
    # The most likely author of [providers.vllm] wants [providers.custom.vllm].
    with pytest.raises(ConfigError, match=r"\[providers.custom.vllm\]"):
        parse_config({"providers": {"vllm": {"upstream_base_url": "http://x"}}}, "<t>")


def test_allowlist_by_type_unknown_type_is_named_error() -> None:
    # A typo'd TYPE key ("EMIAL") was silently inert — the user believed the
    # value was allowlisted while it kept being redacted.
    from llm_redact.detection.engine import build_allowlist

    config = parse_config(
        {"detection": {"allowlist_by_type": {"EMIAL": ["noreply@corp.example"]}}}, "<t>"
    ).detection
    with pytest.raises(ValueError, match=r"unknown placeholder type\(s\) \['EMIAL'\]"):
        build_allowlist(config)
    # The correctly spelled type builds.
    ok = parse_config(
        {"detection": {"allowlist_by_type": {"EMAIL": ["noreply@corp.example"]}}}, "<t>"
    ).detection
    assert build_allowlist(ok).allows_for("EMAIL", "noreply@corp.example")
