"""The required-literal prefilter must be invisible: identical detections,
just cheaper. Soundness is machine-checked per rule; equivalence is checked
differentially against the unfiltered pipeline."""

import random
import re

import pytest

from llm_redact.bench.corpus import generate
from llm_redact.detection.base import Detection
from llm_redact.detection.engine import (
    Allowlist,
    DetectionConfig,
    build_detectors,
    detect_all,
)
from llm_redact.detection.regex_rules import (
    BUILTIN_RULES,
    PreparedText,
    RegexDetector,
    fold_digits,
    lower_for_ci,
)

_NO_ALLOW = Allowlist(exact=frozenset(), patterns=())

_FILTERED_RULES = [rule for rule in BUILTIN_RULES if rule.required]

# Decimal-digit sets `\d` matches beyond 0-9: full-width (CJK text),
# Arabic-Indic, Extended Arabic-Indic (Persian/Urdu), Devanagari.
_DIGIT_SCRIPTS = tuple(
    "".join(chr(zero + i) for i in range(10)) for zero in (0xFF10, 0x0660, 0x06F0, 0x0966)
)


def _transliterated(text: str, rng: random.Random) -> str:
    """``text`` with each ASCII digit, independently, swapped for the same
    digit in one of the other scripts (mixed forms included)."""
    script = rng.choice(_DIGIT_SCRIPTS)
    return "".join(
        script[int(ch)] if ch in "0123456789" and rng.random() < 0.7 else ch for ch in text
    )


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


def test_ci_anchor_offsets_survive_dotted_capital_i() -> None:
    # 'İ'.lower() is two characters; the CI haystack maps it to "i" first,
    # so lowered offsets still map 1:1 onto the text.
    detector = next(RegexDetector(rule) for rule in BUILTIN_RULES if rule.name == "generic_secret")
    text = "İİİİ password = Xk29QmPl40Vt85Zw"
    assert len(PreparedText(text).lower) == len(text)
    found = list(detector.detect_prepared(PreparedText(text)))
    assert [d.value for d in found] == ["Xk29QmPl40Vt85Zw"]
    assert text[found[0].start : found[0].end] == "Xk29QmPl40Vt85Zw"


def test_ci_anchor_falls_back_when_lowering_changes_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Should a future Unicode database give another code point a
    # multi-character lowercase, lowered offsets would no longer map 1:1:
    # the CI anchored scan must then fall back to plain finditer rather
    # than match at shifted positions.
    monkeypatch.setattr(PreparedText, "lower", property(lambda self: "xx" + self.text.lower()))
    detector = next(RegexDetector(rule) for rule in BUILTIN_RULES if rule.name == "generic_secret")
    text = "then password = Xk29QmPl40Vt85Zw"
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


# ---- Unicode: the haystacks must see what the patterns see ----
#
# `\d` in a str pattern matches every Unicode decimal digit, and
# re.IGNORECASE matches 'İ', 'ı' and 'ſ' to ASCII letters: a prefilter that
# tested ASCII literals against the raw (or plainly lowered) text skipped
# real matches the unfiltered scan finds — full-width digits in a Japanese
# My Number, Arabic-Indic digits in a phone number.


def test_fold_digits_maps_exactly_the_decimal_digits() -> None:
    # Every code point: \d matches it iff fold_digits turns it into an
    # ASCII digit of the same value; everything else is left alone.
    digit = re.compile(r"\d")
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        folded = fold_digits(ch)
        if digit.fullmatch(ch):
            assert folded in "0123456789" and len(folded) == 1, hex(cp)
            assert int(folded) == int(ch), hex(cp)
        else:
            assert folded == ch, hex(cp)


def test_ci_haystack_matches_ignorecase_semantics() -> None:
    # Every code point: lower_for_ci keeps the length (so CI anchor offsets
    # map 1:1), and a character re.IGNORECASE matches to an ASCII letter
    # lowers to exactly that letter — so a lowercase ASCII literal occurs in
    # the haystack wherever the (?i) pattern can match it.
    ascii_letter = re.compile(r"[a-z]", re.IGNORECASE)
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        lowered = lower_for_ci(ch)
        assert len(lowered) == 1, hex(cp)
        if ascii_letter.fullmatch(ch):
            assert lowered in "abcdefghijklmnopqrstuvwxyz", hex(cp)
            assert re.fullmatch(lowered, ch, re.IGNORECASE), hex(cp)


def test_ci_literals_and_anchors_are_declared_in_haystack_form() -> None:
    # Case-insensitive literals are compared as declared: they must already
    # be lowercase ASCII (lower_for_ci's fixed points).
    for rule in BUILTIN_RULES:
        declared = [lit for group in rule.required for lit in group] if rule.required_ci else []
        declared += list(rule.anchors) if rule.anchors_ci else []
        for lit in declared:
            assert lit.isascii() and lower_for_ci(lit) == lit, (rule.name, lit)


def test_user_literal_with_a_non_ascii_digit_is_folded_like_the_text() -> None:
    from llm_redact.detection.regex_rules import RegexRule

    rule = RegexRule(
        name="ticket",
        detector_type="TICKET",
        pattern=re.compile(r"TK-\d{3}"),
        required=(("TK-٣",),),
    )
    detector = RegexDetector(rule)
    for text in ("see TK-٣٤٥ now", "see TK-345 now", "see TK-３45 now"):
        assert [d.value for d in detector.detect_prepared(PreparedText(text))] == [
            d.value for d in detector.detect(text)
        ]


@pytest.mark.parametrize("rule", _FILTERED_RULES, ids=lambda r: r.name)
def test_required_literals_hold_for_non_ascii_digit_matches(rule) -> None:
    # The same necessary-condition proof over the recall corpus with its
    # digits rewritten into other scripts: every match, in its haystack
    # form, still satisfies the rule's CNF.
    rng = random.Random(17)
    for sample in generate(seed=11, samples_per_rule=20):
        text = _transliterated(sample.text, rng)
        for match in rule.pattern.finditer(text):
            matched = match.group(0)
            haystack = lower_for_ci(matched) if rule.required_ci else fold_digits(matched)
            for group in rule.required:
                assert any(lit in haystack for lit in group), (rule.name, group)


def test_differential_equivalence_non_ascii_digits() -> None:
    detectors = build_detectors(DetectionConfig())
    rng = random.Random(23)
    hits = 0
    for sample in generate(seed=29, samples_per_rule=10):
        text = _transliterated(sample.text, rng)
        fast = detect_all(detectors, text, _NO_ALLOW)
        assert fast == _naive_detect_all(detectors, text)
        hits += len(fast)
    assert hits > 100  # the transliterated corpus still detects plenty


@pytest.mark.parametrize(
    "text",
    [
        "APİ_KEY = Xk29QmPl40Vt85ZwQq",
        "apı_key: Xk29QmPl40Vt85ZwQq",
        "ſecret=Xk29QmPl40Vt85ZwQq",
        "CREDENTİAL: Xk29QmPl40Vt85ZwQq",
        "awſ_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    ],
)
def test_differential_equivalence_ignorecase_special_letters(text: str) -> None:
    detectors = build_detectors(DetectionConfig())
    fast = detect_all(detectors, text, _NO_ALLOW)
    assert fast == _naive_detect_all(detectors, text)
    assert fast  # the unfiltered scan detects each one


@pytest.mark.parametrize(
    ("text", "detector_type"),
    [
        # One full-width / Arabic-Indic digit inside an otherwise ASCII
        # value once raised KeyError / ValueError in the validator.
        ("codice RSSMRA８5T10A562S ok", "IT_CF"),
        ("codice RSSMRA٨٥T10A562S ok", "IT_CF"),
        ("curp QEBI1٣0401MDFLJSC5 ok", "MX_CURP"),
    ],
)
def test_validators_read_non_ascii_digits(text: str, detector_type: str) -> None:
    detectors = build_detectors(DetectionConfig())
    assert [d.detector_type for d in detect_all(detectors, text, _NO_ALLOW)] == [detector_type]


async def test_non_ascii_digit_in_a_checked_value_is_redacted_not_a_500() -> None:
    import httpx

    from llm_redact.config import Config
    from llm_redact.proxy import create_app

    sent: list[bytes] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return httpx.Response(200, json={"content": [], "role": "assistant"})

    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    body = {
        "model": "m",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "codice RSSMRA８5T10A562S ok"}],
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.post("/v1/messages", json=body, headers={"x-api-key": "k"})
    assert response.status_code == 200
    assert b"RSSMRA" not in sent[0] and "«IT_CF_001»".encode() in sent[0]
