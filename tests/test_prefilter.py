"""The required-literal prefilter must be invisible: identical detections,
just cheaper. Soundness is machine-checked per rule; equivalence is checked
differentially against the unfiltered pipeline."""

import random

import pytest

from llm_redact.bench.corpus import generate
from llm_redact.detection.base import Detection
from llm_redact.detection.engine import (
    Allowlist,
    DetectionConfig,
    build_detectors,
    detect_all,
)
from llm_redact.detection.regex_rules import BUILTIN_RULES, PreparedText, RegexDetector

_NO_ALLOW = Allowlist(exact=frozenset(), patterns=())

_FILTERED_RULES = [rule for rule in BUILTIN_RULES if rule.required]


def _naive_detect_all(detectors, text: str) -> list[Detection]:
    """The pre-prefilter pipeline: every detector's plain finditer scan."""
    detections = [d for det in detectors for d in det.detect(text)]
    detections.sort(key=lambda d: (d.start, -(d.end - d.start), d.priority))
    return detections


# ---- soundness: every recall-corpus match satisfies its rule's CNF ----


@pytest.mark.parametrize("rule", _FILTERED_RULES, ids=lambda r: r.name)
def test_required_literals_are_necessary_conditions(rule) -> None:
    # If a declared literal group were NOT a necessary condition of the
    # regex, some generated positive would match while failing the CNF —
    # exactly the case where the prefilter would silently drop a detection.
    samples = generate(seed=11, samples_per_rule=50)
    matches = 0
    for sample in samples:
        for match in rule.pattern.finditer(sample.text):
            matches += 1
            matched = match.group(0)
            haystack = matched.lower() if rule.required_ci else matched
            for group in rule.required:
                assert any(lit in haystack for lit in group), (
                    f"{rule.name}: a real match failed required group {group!r} —"
                    " the prefilter would drop it"
                )
    assert matches > 0, f"{rule.name}: corpus generated no matches; soundness unverified"


def test_prefilter_skips_when_literals_absent() -> None:
    # Sanity that the fast path actually engages: a text with no "@" never
    # runs the email regex (observable only via identical-but-empty output
    # here; the latency bench shows the win).
    detector = next(RegexDetector(rule) for rule in BUILTIN_RULES if rule.name == "email")
    assert list(detector.detect_prepared(PreparedText("no at sign here"))) == []
    assert [d.value for d in detector.detect_prepared(PreparedText("a@b.example"))] == [
        "a@b.example"
    ]


# ---- differential: filtered pipeline == naive pipeline ----


def _texts() -> list[str]:
    texts = [sample.text for sample in generate(seed=42, samples_per_rule=10)]
    # Prose with tempting fragments but few full matches.
    rng = random.Random(3)
    words = ["deploy", "token", "ticket", "at", "secret:", "ok", "10.0", "call", "+", "sk"]
    texts.append(" ".join(rng.choice(words) for _ in range(2000)))
    # Unicode traps: lowered length differs ('İ'), guillemets, emoji.
    texts.append("İstanbul mail jane@corp.example «EMAIL_001» 🎉 SECRET: Abc123def456ghi7")
    texts.append("")
    return texts


def test_differential_equivalence_builtin_rules() -> None:
    detectors = build_detectors(DetectionConfig())
    for text in _texts():
        fast = detect_all(detectors, text, _NO_ALLOW)
        naive = _naive_detect_all(detectors, text)
        assert fast == naive


def test_differential_equivalence_fp_corpus() -> None:
    from pathlib import Path

    detectors = build_detectors(DetectionConfig())
    root = Path(__file__).resolve().parent.parent / "bench" / "fp_corpus"
    for path in sorted(root.iterdir()):
        if path.name == "MANIFEST.toml" or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert detect_all(detectors, text, _NO_ALLOW) == _naive_detect_all(detectors, text)


def test_lowered_haystack_shared_not_recomputed() -> None:
    prepared = PreparedText("ABC")
    assert prepared.lower == "abc"
    assert prepared.lower is prepared.lower  # cached, same object


# ---- anchored scan ----

_ANCHORED_RULES = [rule for rule in BUILTIN_RULES if rule.anchors]


@pytest.mark.parametrize("rule", _ANCHORED_RULES, ids=lambda r: r.name)
def test_anchors_prefix_every_match(rule) -> None:
    # The anchored scan only attempts pattern.match at anchor positions: if
    # some real match did NOT start with a declared anchor, it would be
    # silently dropped. Prove the premise on the recall corpus.
    samples = generate(seed=13, samples_per_rule=50)
    matches = 0
    for sample in samples:
        for match in rule.pattern.finditer(sample.text):
            matches += 1
            matched = match.group(0).lower() if rule.anchors_ci else match.group(0)
            assert matched.startswith(tuple(rule.anchors)), (
                f"{rule.name}: match does not start with a declared anchor"
            )
    assert matches > 0, f"{rule.name}: corpus generated no matches"


def test_match_at_pos_respects_word_boundary() -> None:
    # pattern.match(text, pos) evaluates \b against the REAL neighboring
    # character — "xAKIA…" must not match at pos 1 even though the slice
    # "AKIA…" would.
    detector = next(
        RegexDetector(rule) for rule in BUILTIN_RULES if rule.name == "aws_access_key_id"
    )
    glued = "xAKIAIOSFODNN7EXAMPLE"
    assert list(detector.detect_prepared(PreparedText(glued))) == []
    spaced = "x AKIAIOSFODNN7EXAMPLE"
    assert [d.value for d in detector.detect_prepared(PreparedText(spaced))] == [
        "AKIAIOSFODNN7EXAMPLE"
    ]


def test_ci_anchor_falls_back_when_lowering_changes_length() -> None:
    # 'İ'.lower() is two characters: the lowered haystack no longer maps
    # offsets 1:1, so the CI anchored scan must fall back to plain finditer
    # rather than match at shifted positions.
    detector = next(RegexDetector(rule) for rule in BUILTIN_RULES if rule.name == "generic_secret")
    text = "İİİİ password = Xk29QmPl40Vt85Zw"
    found = list(detector.detect_prepared(PreparedText(text)))
    assert [d.value for d in found] == ["Xk29QmPl40Vt85Zw"]
    assert text[found[0].start : found[0].end] == "Xk29QmPl40Vt85Zw"


def test_nested_anchor_occurrence_skipped_like_finditer() -> None:
    # An "sk-" occurrence INSIDE an already-matched key is consumed by the
    # previous match in both scans.
    detector = next(RegexDetector(rule) for rule in BUILTIN_RULES if rule.name == "openai_api_key")
    text = "key sk-abcdefghijsk-klmnopqrstuvwx end"
    fast = [(d.start, d.end) for d in detector.detect_prepared(PreparedText(text))]
    naive = [(d.start, d.end) for d in detector.detect(text)]
    assert fast == naive
