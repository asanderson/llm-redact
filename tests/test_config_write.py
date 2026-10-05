"""The hand-rolled TOML emitter must round-trip through the real parser."""

import os
import tomllib
from pathlib import Path

import pytest

from llm_redact.config import (
    RETRY_SAME,
    AuditConfig,
    Config,
    LogConfig,
    ModelPrice,
    PricesConfig,
    ProviderConfig,
    RehydrationConfig,
    RouteRule,
    RoutingConfig,
    RuleMatch,
    UpstreamConfig,
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
        max_body_strings=77,
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
            ner=NerConfig(
                enabled=False,
                backend="spacy",
                entities=("PERSON", "ORG"),
                max_chars=5,
                labels=(("CITY", "ADDRESS"), ("EMAIL", ""), ("FIRST_NAME", "PERSON")),
                revisions=(("gliner", "a" * 40), ("hf", "0123456789abcdef" * 2 + "01234567")),
                onnx=(("gliner", "onnx/model_quint8.onnx"),),
                allow_download=True,
                allow_pickle_weights=True,
            ),
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
        routing=_ROUTING,
        prices=PricesConfig(
            table="/tmp/prices.json",
            overrides=(
                ("fast-large", ModelPrice(input=1.0, output=2.0, cache_read=0.5, cache_write=0.25)),
                ("slow-small", ModelPrice(input=3.0, output=4.0, cache_read=0.0, cache_write=0.0)),
            ),
        ),
    )
    assert _round_trip(config) == config


def _legacy(name: str, base_url: str) -> UpstreamConfig:
    # R-1: with a [routing] table present, the enabled [providers.*] of the
    # four protocols auto-register as passthrough upstreams. The emitter never
    # writes them (they re-register at parse), so the constructed config must
    # carry them for the round trip to compare equal — `ollama` is disabled in
    # the providers above and therefore absent here.
    return UpstreamConfig(name=name, protocol=name, base_url=base_url, legacy=True)


# Every routing/prices field off its default, built by hand (parse-canonical
# order: upstreams sorted by name, headers lowercased+sorted, default table
# sorted, `present` set). The top-level switch above is False, so the one
# `inject_system_note = true` upstream is the non-default form.
_ROUTING = RoutingConfig(
    enabled=True,
    present=True,
    default_upstreams=(("anthropic", "local"),),  # zero-cost: no metered-default warning
    max_hops=5,
    request_deadline_seconds=30.0,
    plan_limit_detection="off",
    plan_limit_headers=(("x-a", ("1",)), ("x-b", ("2", "3"))),
    oauth_beta_marker="oauth-2099-01-01",
    throttle_retry_max_seconds=5.0,
    budget_reset_day=15,
    debug_headers=True,
    expose_models=True,
    model_catalog=("fast-large",),
    upstreams=(
        _legacy("anthropic", "http://a.example"),
        UpstreamConfig(
            name="anthropic_key",
            protocol="anthropic",
            base_url="https://api.anthropic.com",
            credential="env:ANTHROPIC_API_KEY",
            inject_system_note=True,
            monthly_budget_usd=100.0,
            monthly_budget_tokens=123456,
            cooldown_seconds=0.0,
            extra_headers=(("x-title", "llm-redact"),),
            body_defaults_json='{"provider": {"order": ["a"]}}',
        ),
        _legacy("gemini", "http://g.example"),
        UpstreamConfig(
            name="local",
            protocol="anthropic",
            base_url="http://127.0.0.1:11434",
            credential="none",
            cost="zero",
            count_tokens=False,
            cooldown_seconds=1.25,
        ),
        _legacy("openai", "http://o.example"),
        UpstreamConfig(
            name="openai_key",
            protocol="openai",
            base_url="https://api.openai.com/v1",
            credential="env:OPENAI_API_KEY",
            monthly_budget_usd=50.0,
        ),
    ),
    rules=(
        RouteRule(
            id="max-lane",
            match=RuleMatch(
                protocol="anthropic",
                models=("fast-*",),
                headers=(("x-agent", "Explore"),),
                path="/v1/messages",
                auth="oauth",
            ),
            upstream="anthropic",
            on_status=(
                ("plan_limit_429", ("anthropic_key", "local")),
                ("529", ("anthropic_key",)),
                ("throttle_429", RETRY_SAME),
                ("5xx", ("local",)),
            ),
            reissue_policy="stateless-only",
        ),
        RouteRule(
            id="key-lane",
            match=RuleMatch(protocol="anthropic", models=("fast-*",), auth="gateway-key"),
            upstream="anthropic_key",
            on_status=(("429", ("local",)), ("4xx", RETRY_SAME)),
            reissue_policy="always",
            on_budget_exhausted=("local",),
        ),
        RouteRule(
            id="openai-lane",
            match=RuleMatch(protocol="openai", models=("fast-*", "slow-?")),
            upstream="openai_key",
            model_rewrite="fast-large",
            reissue_policy="never",
        ),
    ),
)


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


@pytest.mark.parametrize(
    "ner",
    [
        NerConfig(enabled=True, backend="hf", score_threshold=0.8),
        NerConfig(enabled=True, backends=("spacy", "hf"), score_threshold=0.8),
    ],
    ids=["hf", "spacy+hf"],
)
def test_hf_score_threshold_round_trips(ner: NerConfig) -> None:
    # hf emits confidences, so the parser accepts its threshold; an emitter
    # that knew only gliner/presidio dropped it (config show, editor saves).
    config = Config(detection=DetectionConfig(ner=ner))
    assert "score_threshold = 0.8" in emit_config_toml(config)
    assert _round_trip(config) == config


def test_ner_labels_round_trip_after_models() -> None:
    config = Config(
        detection=DetectionConfig(
            ner=NerConfig(
                enabled=True,
                backends=("gliner", "hf"),
                models=(("hf", "org/model"),),
                labels=(("CITY", "ADDRESS"), ("PER", "PER"), ("TIME", "")),
                score_threshold=0.6,
            )
        )
    )
    emitted = emit_config_toml(config)
    # Both subtables follow every [detection.ner] scalar.
    ner_section = emitted.split("[detection.ner]")[1]
    assert ner_section.index("score_threshold") < ner_section.index("[detection.ner.labels]")
    assert '"CITY" = "ADDRESS"' in emitted
    assert '"TIME" = ""' in emitted
    assert _round_trip(config) == config
    assert "[detection.ner.labels]" not in emit_config_toml(Config())


def test_ner_model_sources_round_trip_after_the_scalars() -> None:
    config = Config(
        detection=DetectionConfig(
            ner=NerConfig(
                enabled=True,
                backends=("gliner", "hf"),
                models=(("hf", "org/model"),),
                revisions=(("hf", "b" * 40),),
                labels=(("CITY", "ADDRESS"),),
                allow_download=True,
                allow_pickle_weights=True,
            )
        )
    )
    emitted = emit_config_toml(config)
    ner_section = emitted.split("[detection.ner]")[1]
    # Both switches are [detection.ner] scalars: before every subtable.
    assert ner_section.index("allow_download = true") < ner_section.index("[detection.ner.models]")
    assert ner_section.index("allow_pickle_weights = true") < ner_section.index(
        "[detection.ner.revisions]"
    )
    assert f'hf = "{"b" * 40}"' in emitted
    assert _round_trip(config) == config


@pytest.mark.parametrize(
    "ner",
    [
        NerConfig(allow_download=True),
        NerConfig(allow_pickle_weights=True),
        NerConfig(revisions=(("gliner", "c" * 40),)),
        NerConfig(onnx=(("gliner", "onnx/model.onnx"),)),
    ],
    ids=["allow_download", "allow_pickle_weights", "revisions", "onnx"],
)
def test_each_ner_model_source_round_trips_alone(ner: NerConfig) -> None:
    config = Config(detection=DetectionConfig(ner=ner))
    assert _round_trip(config) == config


def test_ner_model_sources_at_their_defaults_are_not_written() -> None:
    # The switches loosen a safe default: written only when turned on (like
    # [audit] tamper_evident), so a default configuration emits none of them.
    emitted = emit_config_toml(Config())
    assert "allow_download" not in emitted
    assert "allow_pickle_weights" not in emitted
    assert "[detection.ner.revisions]" not in emitted
    assert "[detection.ner.onnx]" not in emitted


def test_score_threshold_is_not_emitted_without_a_confidence_backend() -> None:
    # The parser rejects the key for spacy/stanza-only configs, so the
    # emitter must not write it there either.
    for ner in (NerConfig(backend="spacy"), NerConfig(backends=("spacy", "stanza"))):
        config = Config(detection=DetectionConfig(ner=ner))
        assert "score_threshold" not in emit_config_toml(config)
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
