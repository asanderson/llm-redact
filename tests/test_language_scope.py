"""Language-scoped detection ([detection] languages): tagged national-id
rules outside the configured scope are not built; universal rules always
run; NER language mismatches are loud."""

import tomllib

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from llm_redact.config import Config, ConfigError, ProviderConfig, parse_config
from llm_redact.config_write import emit_config_toml
from llm_redact.detection.engine import (
    DetectionConfig,
    active_rule_names,
    build_allowlist,
    build_detectors,
)
from llm_redact.detection.regex_rules import BUILTIN_RULES
from llm_redact.proxy import create_app
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault

# Checksum-valid synthetic examples (same values test_detectors.py pins).
NIR = "1 85 03 75 116 384 32"
SSN = "536-22-8726"
EMAIL = "jane.doe@corp.example"


def _redact(text: str, config: DetectionConfig) -> str:
    redactor = Redactor(build_detectors(config), InMemoryVault(), build_allowlist(config))
    return redactor.redact_text(text)


# --- config -----------------------------------------------------------------


def test_parse_languages_normalizes_and_sorts() -> None:
    config = parse_config({"detection": {"languages": ["FR", "en", " en "]}}, "<test>")
    assert config.detection.languages == ("en", "fr")
    assert parse_config({}, "<test>").detection.languages is None


def test_parse_rejects_bad_languages() -> None:
    with pytest.raises(ConfigError, match="non-empty strings"):
        parse_config({"detection": {"languages": [""]}}, "<t>")
    with pytest.raises(ConfigError, match="non-empty strings"):
        parse_config({"detection": {"languages": "en"}}, "<t>")
    with pytest.raises(ConfigError, match="must not be empty"):
        parse_config({"detection": {"languages": []}}, "<t>")


def test_ner_language_outside_scope_is_loud() -> None:
    with pytest.raises(ConfigError, match="not in \\[detection\\] languages"):
        parse_config(
            {"detection": {"languages": ["fr"], "ner": {"enabled": True, "language": "en"}}},
            "<t>",
        )
    # NER off: the mismatch is irrelevant, no error.
    parse_config({"detection": {"languages": ["fr"], "ner": {"language": "en"}}}, "<t>")
    # Case-normalized like the list itself.
    parse_config(
        {"detection": {"languages": ["EN"], "ner": {"enabled": False, "language": "en"}}}, "<t>"
    )


def test_emitter_round_trips_languages() -> None:
    config = parse_config({"detection": {"languages": ["en", "sv"]}}, "<test>")
    text = emit_config_toml(config)
    assert "languages" in text
    reparsed = parse_config(tomllib.loads(text), "<reparse>")
    assert reparsed.detection.languages == ("en", "sv")
    # Unset stays unset (omitted = open-ended default).
    assert (
        parse_config(tomllib.loads(emit_config_toml(Config())), "<r>").detection.languages is None
    )


# --- rule filtering -----------------------------------------------------------


def test_every_tagged_rule_is_a_national_id() -> None:
    tagged = {rule.name for rule in BUILTIN_RULES if rule.languages is not None}
    assert tagged == {
        "us_ssn",
        "canadian_sin",
        "uk_nino",
        "aadhaar",
        "australian_tfn",
        "spanish_dni",
        "french_nir",
        "german_steuer_id",
        "brazilian_cpf",
        "italian_codice_fiscale",
        "swiss_ahv",
        "swedish_personnummer",
        "belgian_nn",
        "finnish_hetu",
        "nhs_number",
        "norwegian_fnr",
        "korean_rrn",
        "chinese_resident_id",
        "singapore_nric",
        "japanese_my_number",
        "thai_id",
        "irish_pps",
        "mexican_curp",
    }


def test_scope_filters_tagged_rules_only() -> None:
    english = set(active_rule_names(DetectionConfig(languages=("en",))))
    assert "us_ssn" in english and "aadhaar" in english  # en / en+hi
    assert "french_nir" not in english and "swedish_personnummer" not in english
    assert "email" in english and "credit_card" in english  # universal

    swedish = set(active_rule_names(DetectionConfig(languages=("sv",))))
    assert "swedish_personnummer" in swedish
    assert "us_ssn" not in swedish
    assert "email" in swedish

    multilingual = set(active_rule_names(DetectionConfig(languages=("de", "fr"))))
    assert {"french_nir", "german_steuer_id", "swiss_ahv", "canadian_sin"} <= multilingual
    assert "spanish_dni" not in multilingual

    # No scope = every enabled rule, the exact historical default.
    assert active_rule_names(DetectionConfig()) == list(DetectionConfig().enabled)


def test_explicit_enabled_list_intersects_with_scope() -> None:
    config = DetectionConfig(enabled=("french_nir", "email"), languages=("en",))
    assert active_rule_names(config) == ["email"]


def test_scoped_out_rule_does_not_fire() -> None:
    scoped = DetectionConfig(languages=("en",))
    text = f"NIR {NIR} SSN {SSN} mail {EMAIL}"
    result = _redact(text, scoped)
    assert NIR in result  # french_nir not built
    assert SSN not in result and "«SSN_001»" in result
    assert EMAIL not in result
    # Without the scope the same NIR is redacted (recall unchanged).
    assert NIR not in _redact(text, DetectionConfig())


def test_language_scope_suppresses_folded_ner_types() -> None:
    # The rule toggles are the single source of truth per placeholder type;
    # language scoping participates: us_ssn scoped out (languages=["sv"])
    # must suppress SSN-typed NER emissions the same way disabling does.
    from llm_redact.detection.base import Detection
    from llm_redact.detection.engine import TypeFilteredDetector

    class FakeNer:
        name = "fake-ner"
        priority = 120

        def detect(self, text: str) -> list[Detection]:
            return [Detection(0, 3, "SSN", "abc"), Detection(4, 7, "PERSON", "Jan")]

    config = DetectionConfig(languages=("sv",))
    active_types = {
        rule.detector_type for rule in BUILTIN_RULES if rule.name in active_rule_names(config)
    }
    suppressed = frozenset(
        rule.detector_type for rule in BUILTIN_RULES if rule.detector_type not in active_types
    )
    filtered = TypeFilteredDetector(FakeNer(), suppressed)
    kept = {d.detector_type for d in filtered.detect("abc Jan")}
    assert kept == {"PERSON"}  # SSN suppressed, PERSON never suppressed


# --- integration + surfacing ----------------------------------------------------

received: dict[str, object] = {}


def _fake_upstream() -> Starlette:
    async def messages(request: Request) -> Response:
        received["body"] = await request.json()
        return JSONResponse({"content": [{"type": "text", "text": "ok"}]})

    return Starlette(routes=[Route("/v1/messages", messages, methods=["POST"])])


@pytest.mark.anyio
async def test_scope_end_to_end_and_surfaced() -> None:
    received.clear()
    config = Config(
        providers={**Config().providers, "anthropic": ProviderConfig("http://upstream")},
        detection=DetectionConfig(languages=("en",)),
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")

    body = {"model": "m", "messages": [{"role": "user", "content": f"NIR {NIR} SSN {SSN}"}]}
    response = await client.post("/v1/messages", json=body)
    assert response.status_code == 200
    seen = received["body"]["messages"][0]["content"]  # type: ignore[index]
    assert NIR in seen and SSN not in seen

    status = (await client.get("/__llm-redact/status")).json()
    assert status["detection"]["languages"] == ["en"]
    assert "french_nir" in status["detection"]["language_inactive_rules"]
    assert "us_ssn" not in status["detection"]["language_inactive_rules"]

    editor = (await client.get("/__llm-redact/config")).json()
    assert editor["editable"]["detection"]["languages"] == ["en"]
    assert editor["builtin_rule_languages"]["french_nir"] == ["fr"]
    assert "email" not in editor["builtin_rule_languages"]  # universal: untagged
    assert "french_nir" in editor["language_inactive_rules"]
    await client.aclose()
