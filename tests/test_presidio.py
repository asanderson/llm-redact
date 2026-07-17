"""PresidioDetector tests via an injectable fake engine — no presidio install
needed. The type map must fold overlapping recognizers into the built-in
placeholder names so one value never gets two token identities."""

from types import SimpleNamespace
from typing import Any

import pytest

from llm_redact.config import ConfigError, load_config
from llm_redact.detection.engine import (
    Allowlist,
    DetectionConfig,
    NerConfig,
    build_detectors,
    detect_all,
)
from llm_redact.detection.ner import NER_PRIORITY
from llm_redact.detection.presidio_ner import PRESIDIO_TYPE_MAP, PresidioDetector
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault

NO_ALLOW = Allowlist(exact=frozenset(), patterns=())

EMAIL = "jane.doe@corp.example"


class FakeAnalyzer:
    """Recognizes one email (score 0.9), one PERSON (0.6), one ORG (0.3)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str] | None, float]] = []

    def analyze(
        self,
        text: str,
        language: str,
        entities: list[str] | None = None,
        score_threshold: float = 0.0,
    ) -> list[Any]:
        self.calls.append((text, language, entities, score_threshold))
        results = []
        for value, entity_type, score in (
            (EMAIL, "EMAIL_ADDRESS", 0.9),
            ("Jane Doe", "PERSON", 0.6),
            ("Acme", "ORG", 0.3),
        ):
            index = text.find(value)
            wanted = entities is None or entity_type in entities
            if index != -1 and wanted and score >= score_threshold:
                results.append(
                    SimpleNamespace(
                        entity_type=entity_type,
                        start=index,
                        end=index + len(value),
                        score=score,
                    )
                )
        return results


def _detector(
    entities: frozenset[str], *, max_chars: int = 1000, threshold: float = 0.5
) -> PresidioDetector:
    return PresidioDetector(FakeAnalyzer(), entities, max_chars, threshold)


def test_type_map_folds_into_builtin_names() -> None:
    detector = _detector(frozenset({"EMAIL_ADDRESS", "PERSON"}))
    detections = {d.detector_type: d for d in detector.detect(f"ask Jane Doe at {EMAIL}")}
    assert set(detections) == {"EMAIL", "PERSON"}  # EMAIL_ADDRESS folded
    assert detections["EMAIL"].value == EMAIL
    assert detections["EMAIL"].priority == NER_PRIORITY
    assert detections["PERSON"].value == "Jane Doe"  # unmapped: identity


def test_threshold_passed_and_filters() -> None:
    detector = _detector(frozenset({"PERSON", "ORG"}), threshold=0.5)
    types = [d.detector_type for d in detector.detect("Jane Doe works at Acme")]
    assert types == ["PERSON"]  # ORG scored 0.3 < 0.5

    permissive = _detector(frozenset({"PERSON", "ORG"}), threshold=0.2)
    types = [d.detector_type for d in permissive.detect("Jane Doe works at Acme")]
    assert types == ["PERSON", "ORG"]


def test_entities_forwarded_to_analyzer() -> None:
    analyzer = FakeAnalyzer()
    detector = PresidioDetector(analyzer, frozenset({"PERSON"}), 1000, 0.5)
    list(detector.detect("hello"))
    ((_text, language, entities, threshold),) = analyzer.calls
    assert language == "en"
    assert entities == ["PERSON"]
    assert threshold == 0.5


def test_language_forwarded_to_analyzer() -> None:
    analyzer = FakeAnalyzer()
    detector = PresidioDetector(analyzer, frozenset({"PERSON"}), 1000, 0.5, language="de")
    list(detector.detect("hallo"))
    ((_text, language, _entities, _threshold),) = analyzer.calls
    assert language == "de"


def test_max_chars_gate() -> None:
    analyzer = FakeAnalyzer()
    detector = PresidioDetector(analyzer, frozenset({"PERSON"}), 10, 0.5)
    assert list(detector.detect("Jane Doe " + "x" * 100)) == []
    assert analyzer.calls == []


def test_overlap_with_builtin_email_rule_issues_one_token() -> None:
    # The built-in email rule and Presidio's EMAIL_ADDRESS both claim the
    # same span; overlap resolution keeps one winner and the folded type
    # means the vault sees a single (EMAIL, value) identity either way.
    config = DetectionConfig(enabled=("email",))
    detectors: list[Any] = build_detectors(config)
    detectors.append(_detector(frozenset({"EMAIL_ADDRESS"})))
    vault = InMemoryVault()
    redactor = Redactor(detectors, vault, NO_ALLOW)
    redacted = redactor.redact_text(f"mail {EMAIL} now")
    assert redacted == "mail «EMAIL_001» now"
    assert redactor.counts["EMAIL"] == 1
    assert len(vault) == 1


def test_allowlist_applies() -> None:
    detectors: list[Any] = [_detector(frozenset({"PERSON"}))]
    allow = Allowlist(exact=frozenset({"Jane Doe"}), patterns=())
    assert detect_all(detectors, "ask Jane Doe", allow) == []


def test_enabled_without_presidio_config_error() -> None:
    try:
        import presidio_analyzer  # noqa: F401

        pytest.skip("presidio installed; missing-dependency path not testable")
    except ImportError:
        pass
    with pytest.raises(ConfigError, match="uv sync --extra presidio"):
        build_detectors(DetectionConfig(ner=NerConfig(enabled=True, backend="presidio")))


def test_config_accepts_threshold_for_presidio(tmp_path: Any) -> None:
    config_file = tmp_path / "c.toml"
    config_file.write_text(
        '[detection.ner]\nenabled = false\nbackend = "presidio"\nscore_threshold = 0.7\n'
    )
    config = load_config(config_file)
    assert config.detection.ner.backend == "presidio"
    assert config.detection.ner.score_threshold == 0.7


def test_config_rejects_unknown_backend(tmp_path: Any) -> None:
    config_file = tmp_path / "c.toml"
    config_file.write_text('[detection.ner]\nbackend = "watson"\n')
    with pytest.raises(ConfigError, match="presidio"):
        load_config(config_file)


def test_type_map_targets_are_builtin_types() -> None:
    from llm_redact.detection.regex_rules import BUILTIN_RULES

    builtin_types = {rule.detector_type for rule in BUILTIN_RULES}
    assert set(PRESIDIO_TYPE_MAP.values()) <= builtin_types


def test_real_analyzer_smoke() -> None:
    """The one test that exercises a real presidio AnalyzerEngine (the rest
    inject fakes). Runs in the ner-extra CI job; skips wherever the presidio
    extra or the spaCy model is absent."""
    pytest.importorskip("presidio_analyzer")
    spacy_util = pytest.importorskip("spacy.util")
    if not spacy_util.is_package("en_core_web_sm"):
        pytest.skip("en_core_web_sm model not installed")

    # Not the module-level EMAIL: presidio's EmailRecognizer validates the
    # TLD against the public-suffix list, and reserved ".example" domains
    # score zero (verified against the real recognizer).
    email = "jane.doe@corp-example.com"
    # The email rule must be ENABLED for the EMAIL fold to flow: rule
    # toggles are the single source of truth per placeholder type, and a
    # type disabled at the rule level is suppressed for NER too.
    config = DetectionConfig(
        enabled=("email",),
        ner=NerConfig(
            enabled=True,
            backend="presidio",
            entities=("EMAIL_ADDRESS", "PERSON"),
            score_threshold=0.4,
        ),
    )
    _email_rule, detector = build_detectors(config)
    text = f"Please contact Jane Doe at {email} about the invoice."
    detections = list(detector.detect(text))
    by_type = {d.detector_type: d.value for d in detections}
    # The real engine folds EMAIL_ADDRESS into the built-in EMAIL name.
    assert by_type.get("EMAIL") == email
    assert "EMAIL_ADDRESS" not in by_type
    assert by_type.get("PERSON") == "Jane Doe"

    # And with every rule disabled the fold is suppressed (single source
    # of truth) while PERSON — which has no built-in rule — still flows.
    (suppressed_detector,) = build_detectors(
        DetectionConfig(
            enabled=(),
            ner=NerConfig(
                enabled=True,
                backend="presidio",
                entities=("EMAIL_ADDRESS", "PERSON"),
                score_threshold=0.4,
            ),
        )
    )
    suppressed_types = {d.detector_type for d in suppressed_detector.detect(text)}
    assert "EMAIL" not in suppressed_types
    assert "PERSON" in suppressed_types
