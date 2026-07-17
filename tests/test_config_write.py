"""The hand-rolled TOML emitter must round-trip through the real parser."""

import os
import tomllib
from pathlib import Path

import pytest

from llm_redact.config import (
    AuditConfig,
    Config,
    LogConfig,
    ProviderConfig,
    RehydrationConfig,
    VaultConfig,
    parse_config,
)
from llm_redact.config_write import emit_config_toml, write_config_atomic
from llm_redact.detection.deny import DenyEntry
from llm_redact.detection.engine import CustomRule, DetectionConfig, NerConfig

NASTY_STRINGS = [
    'quote " inside',
    "back\\slash",
    "new\nline",
    "tab\there",
    "guillemets «EMAIL_001» stay",
    "unicode: żółć 東京 🎉",
    'triple """ quotes',
    "control \x01 char",
    "windows C:\\Users\\jane",
    "  leading and trailing  ",
]


def _round_trip(config: Config) -> Config:
    return parse_config(tomllib.loads(emit_config_toml(config)), "round-trip")


def test_default_config_round_trips() -> None:
    assert _round_trip(Config()) == Config()


def test_default_enabled_stays_open_ended() -> None:
    # An emitted default config must NOT pin the rule list: absent `enabled`
    # means all built-ins including future ones.
    text = emit_config_toml(Config())
    assert "enabled = [" not in text.split("[detection.ner]")[0].split("[detection]")[1]


@pytest.mark.parametrize("nasty", NASTY_STRINGS)
def test_nasty_strings_round_trip(nasty: str) -> None:
    config = Config(
        detection=DetectionConfig(
            enabled=("email",),
            allowlist=(nasty,),
            allowlist_patterns=(nasty,),
            custom_rules=(
                CustomRule(name=nasty, detector_type="TICKET", pattern=nasty, priority=7),
            ),
            modes=((nasty, "warn"),),
        ),
        vault=VaultConfig(backend="sqlite", path=nasty, session=nasty),
    )
    assert _round_trip(config) == config


def test_every_field_nondefault_round_trips() -> None:
    config = Config(
        host="0.0.0.0",
        port=1234,
        inject_system_note=False,
        max_body_bytes=99,
        providers={
            "anthropic": ProviderConfig(upstream_base_url="http://a.example"),
            "openai": ProviderConfig(upstream_base_url="http://o.example"),
            "gemini": ProviderConfig(upstream_base_url="http://g.example"),
            "cohere": ProviderConfig(upstream_base_url="http://co.example"),
            "azure": ProviderConfig(upstream_base_url="http://az.example"),
            "vertex": ProviderConfig(upstream_base_url="http://vx.example"),
            "bedrock": ProviderConfig(upstream_base_url="http://br.example"),
            "ollama": ProviderConfig(upstream_base_url="http://ol.example", enabled=False),
        },
        detection=DetectionConfig(
            enabled=("email", "ipv4"),
            allowlist=("keep@example.com",),
            allowlist_patterns=(r"^10\.",),
            custom_rules=(
                CustomRule(name="jira", detector_type="TICKET", pattern=r"PROJ-\d+", priority=90),
                CustomRule(name="two", detector_type="ID", pattern=r"ID\d+"),
            ),
            ner=NerConfig(enabled=False, backend="spacy", entities=("PERSON", "ORG"), max_chars=5),
            modes=(("email", "warn"), ("us_ssn", "block")),
        ),
        vault=VaultConfig(
            backend="sqlite",
            path="/tmp/v.db",
            session="s1",
            session_mode="per-conversation",
            session_ttl_days=30,
        ),
        rehydration=RehydrationConfig(fuzzy=False),
        audit=AuditConfig(enabled=True, required=True, path="/tmp/a.db", max_rows=5),
        log=LogConfig(format="json"),
    )
    assert _round_trip(config) == config


def test_custom_rule_validator_and_prefilter_round_trip() -> None:
    config = Config(
        detection=DetectionConfig(
            custom_rules=(
                CustomRule(
                    name="cardish",
                    detector_type="CARDISH",
                    pattern=r"\d[\d ]{14,18}\d",
                    priority=80,
                    validator="luhn",
                    required=("card", "no"),
                    anchors=("4", "5"),
                ),
                CustomRule(name="plain", detector_type="P", pattern=r"P\d+"),
            )
        )
    )
    rt = _round_trip(config)
    assert rt == config
    # The gate/prefilter fields survived on the first rule and stayed empty on
    # the second.
    first, second = rt.detection.custom_rules
    assert (first.validator, first.required, first.anchors) == ("luhn", ("card", "no"), ("4", "5"))
    assert (second.validator, second.required, second.anchors) == (None, (), ())


def test_gliner_score_threshold_round_trips() -> None:
    config = Config(
        detection=DetectionConfig(
            ner=NerConfig(enabled=True, backend="gliner", score_threshold=0.75)
        )
    )
    assert _round_trip(config) == config


def test_allowlist_by_type_round_trips() -> None:
    config = Config(
        detection=DetectionConfig(
            allowlist_by_type=(
                ("EMAIL", ("ceo@corp.example", "support@corp.example")),
                ("IPV4", ("192.0.2.1",)),
            )
        )
    )
    emitted = emit_config_toml(config)
    assert "[detection.allowlist_by_type]" in emitted
    assert _round_trip(config) == config
    # Absent when empty, like modes.
    assert "[detection.allowlist_by_type]" not in emit_config_toml(Config())


def test_disabled_provider_round_trips() -> None:
    config = Config(
        providers={
            **Config().providers,
            "openai": ProviderConfig(upstream_base_url="https://api.openai.com", enabled=False),
        }
    )
    emitted = emit_config_toml(config)
    assert 'upstream_base_url = "https://api.openai.com"\nenabled = false' in emitted
    # `enabled = true` is the additive default and never written out: only
    # the one disabled provider carries the key.
    providers_block = emitted.split("[detection]")[0]
    assert providers_block.count("enabled = false") == 1
    assert _round_trip(config) == config


def test_presidio_score_threshold_round_trips() -> None:
    config = Config(
        detection=DetectionConfig(
            ner=NerConfig(enabled=True, backend="presidio", score_threshold=0.35)
        )
    )
    assert _round_trip(config) == config


def test_ner_language_and_model_round_trip() -> None:
    config = Config(
        detection=DetectionConfig(
            ner=NerConfig(enabled=True, backend="presidio", language="de", model="de_core_news_sm")
        )
    )
    assert _round_trip(config) == config
    # model is omitted when unset (backend default applies).
    assert "model =" not in emit_config_toml(Config())


def test_empty_modes_table_omitted() -> None:
    # An absent [detection.modes] means "everything redacts", the same
    # open-ended default as the omitted `enabled` list.
    assert "[detection.modes]" not in emit_config_toml(Config())
    with_modes = emit_config_toml(
        Config(detection=DetectionConfig(modes=(("phone_number", "warn"),)))
    )
    assert "[detection.modes]" in with_modes
    assert '"phone_number" = "warn"' in with_modes


def test_emit_is_idempotent() -> None:
    config = Config(
        detection=DetectionConfig(enabled=("email",), allowlist=("a@b.example",)),
        rehydration=RehydrationConfig(fuzzy=False),
    )
    once = emit_config_toml(config)
    again = emit_config_toml(_round_trip(config))
    assert once == again


def test_write_config_atomic_creates_0600_and_bak(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "config.toml"
    assert write_config_atomic(target, "port = 1\n") is None  # first write: no backup
    assert target.read_text() == "port = 1\n"
    if os.name == "posix":
        assert (target.stat().st_mode & 0o777) == 0o600
        assert (target.parent.stat().st_mode & 0o777) == 0o700

    backup = write_config_atomic(target, "port = 2\n")
    assert backup is not None
    assert backup.read_text() == "port = 1\n"
    if os.name == "posix":
        assert (backup.stat().st_mode & 0o777) == 0o600
    assert target.read_text() == "port = 2\n"
    assert not target.with_name(target.name + ".tmp").exists()


def test_deny_strings_round_trip() -> None:
    # Canonical (parse-produced) order: entries sort by value codepoints.
    config = Config(
        detection=DetectionConfig(
            deny_strings=(
                DenyEntry("Blue Harvest", case_sensitive=True, detector_type="PROJECT"),
                DenyEntry("aurora"),
            )
        )
    )
    assert _round_trip(config) == config
    text = emit_config_toml(config)
    assert "[[detection.deny_strings]]" in text
    # No deny entries -> no table at all.
    assert "deny_strings" not in emit_config_toml(Config())


@pytest.mark.parametrize("nasty", NASTY_STRINGS)
def test_nasty_deny_values_round_trip(nasty: str) -> None:
    if "«" in nasty or "»" in nasty:
        pytest.skip("guillemets are rejected by design")
    config = Config(detection=DetectionConfig(deny_strings=(DenyEntry(nasty),)))
    assert _round_trip(config) == config
