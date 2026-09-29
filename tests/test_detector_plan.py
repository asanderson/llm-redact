"""DetectorPlan: gating detectors per string must be invisible — exactly the
detections of running every detector, only cheaper.

The gate is a quick-reject: a wrong one is a silent recall bug. So the
differential suite runs the gated path against every detector's full scan
(and against the unfiltered pipeline) on the recall corpus, its non-ASCII
digit transliterations, the false-positive corpus and literal fragments —
and the "teeth" tests corrupt the gate rule by rule to prove the suite
notices.
"""

import random
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

from llm_redact.bench.corpus import VALUE_GENERATORS, generate
from llm_redact.detection.base import Detection, Detector
from llm_redact.detection.deny import DenyEntry
from llm_redact.detection.engine import (
    GATED_MAX_CHARS,
    Allowlist,
    CustomRule,
    DetectionConfig,
    DetectorPlan,
    TypeFilteredDetector,
    _combinable,
    _gate_group,
    build_allowlist,
    build_detectors,
    detect_all,
    plan_for,
)
from llm_redact.detection.regex_rules import BUILTIN_RULES, PreparedText, RegexDetector
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault

NO_ALLOW = Allowlist(exact=frozenset(), patterns=())
UNGATED_LIMIT = 10**9  # a plan that gates texts of every length
ROOT = Path(__file__).resolve().parent.parent


class _Names:
    """A non-regex detector (the NER shape): always run by the plan."""

    name = "fake_ner"

    def detect(self, text: str) -> Iterable[Detection]:
        index = text.find("Jane Doe")
        if index != -1:
            yield Detection(index, index + 8, "PERSON", "Jane Doe", priority=120)


def _naive(detectors: Iterable[Detector], text: str) -> list[Detection]:
    """The unfiltered pipeline: every detector's plain scan, no prefilter."""
    detections = [d for det in detectors for d in det.detect(text)]
    detections.sort(key=lambda d: (d.start, -(d.end - d.start), d.priority))
    return detections


def _custom_config() -> DetectionConfig:
    return DetectionConfig(
        custom_rules=(
            # literal-gated (a user `required` literal)
            CustomRule(
                name="ticket", detector_type="TICKET", pattern=r"TK-\d{4}", required=("TK-",)
            ),
            # anchors only: a combined-pattern member
            CustomRule(
                name="project", detector_type="PROJ", pattern=r"proj_[a-z]{6}", anchors=("proj_",)
            ),
            # capturing group + backreference: never combined, always run
            CustomRule(name="order", detector_type="ORDER", pattern=r"ord-(\d{3})-\1"),
            # a global flag: never combined, always run
            CustomRule(name="code", detector_type="CODE", pattern=r"(?i)code[0-9]{3}"),
            # plain: a member; with a validator
            CustomRule(name="badge", detector_type="BADGE", pattern=r"\bB\d{5}\b"),
            CustomRule(
                name="card12", detector_type="CARD12", pattern=r"\b\d{12}\b", validator="luhn"
            ),
        ),
        deny_strings=(DenyEntry("aurora"), DenyEntry("Zeta", case_sensitive=True)),
    )


def _detector_sets() -> list[list[Detector]]:
    full = build_detectors(_custom_config())
    full.append(_Names())
    full.append(TypeFilteredDetector(_Names(), frozenset({"EMAIL"})))
    return [build_detectors(DetectionConfig()), full]


def _digit_script(rng: random.Random, text: str) -> str:
    zero = rng.choice((0xFF10, 0x0660, 0x06F0, 0x0966))
    return "".join(
        chr(zero + int(ch)) if ch in "0123456789" and rng.random() < 0.7 else ch for ch in text
    )


def _values(seed: int = 5, per_rule: int = 4) -> list[str]:
    rng = random.Random(seed)
    return [gen(rng) for _type, gen in VALUE_GENERATORS.values() for _ in range(per_rule)]


def _fragments() -> list[str]:
    """Short strings: every declared literal alone and glued to words, every
    corpus value alone, truncated and embedded, custom-rule shapes, and the
    Unicode traps (non-ASCII digits, İ/ı/ſ)."""
    literals = sorted({lit for rule in BUILTIN_RULES for group in rule.required for lit in group})
    out = ["", "hi", "x", "user", "assistant", "Jane Doe", "Acme", "aurora", "AURORA", "Zeta"]
    out += literals + [f"a{lit}b" for lit in literals] + [lit.upper() for lit in literals]
    rng = random.Random(9)
    for value in _values():
        out += [value, value[:-1], value[1:], f"({value})", f"x{value}y", _digit_script(rng, value)]
    out += ["TK-1234", "proj_abcdef", "ord-123-123", "CODE123", "B12345", "4111111111111111"]
    out += [
        "APİ_KEY = Xk29QmPl40Vt85ZwQq",
        "apı_key: Xk29QmPl40Vt85ZwQq",
        "ſecret=Xk29QmPl40Vt85Zw",
    ]
    return out


def _texts() -> list[str]:
    rng = random.Random(4)
    corpus = [sample.text for sample in generate(seed=42, samples_per_rule=6)]
    return _fragments() + corpus + [_digit_script(rng, text) for text in corpus]


def _fp_corpus() -> list[str]:
    root = ROOT / "bench" / "fp_corpus"
    return [
        path.read_text(encoding="utf-8")
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "MANIFEST.toml"
    ]


# ---- differential: gated == every detector == unfiltered ----


@pytest.mark.parametrize("limit", [GATED_MAX_CHARS, UNGATED_LIMIT])
@pytest.mark.parametrize("which", [0, 1], ids=["builtin", "custom+deny+ner"])
def test_gated_detection_equals_every_detector(which: int, limit: int) -> None:
    detectors = _detector_sets()[which]
    plan = DetectorPlan(detectors, gated_max_chars=limit)
    for text in _texts():
        gated = plan.detect(text, NO_ALLOW)
        assert gated == plan.detect_each(text, NO_ALLOW), text
        assert gated == _naive(detectors, text), text


@pytest.mark.parametrize("which", [0, 1], ids=["builtin", "custom+deny+ner"])
def test_gated_detection_equals_every_detector_on_the_fp_corpus(which: int) -> None:
    # Long real-world texts, gated at every length.
    detectors = _detector_sets()[which]
    plan = DetectorPlan(detectors, gated_max_chars=UNGATED_LIMIT)
    for text in _fp_corpus():
        assert plan.detect(text, NO_ALLOW) == plan.detect_each(text, NO_ALLOW)


def test_gated_detection_applies_the_allowlist_like_every_detector() -> None:
    config = DetectionConfig(
        allowlist=("10.0.0.5",), allowlist_by_type=(("EMAIL", ("a@corp.example",)),)
    )
    allowlist = build_allowlist(config)
    plan = DetectorPlan(build_detectors(config))
    for text in [
        "10.0.0.5 and 8.8.8.8",
        "a@corp.example b@corp.example",
        "127.0.0.1",
        *_fragments(),
    ]:
        assert plan.detect(text, allowlist) == plan.detect_each(text, allowlist)


# ---- teeth: a lying gate fails the differential ----


def _corpus_mismatch(plan: DetectorPlan) -> bool:
    return any(
        plan.detect(text, NO_ALLOW) != plan.detect_each(text, NO_ALLOW)
        for text in [sample.text for sample in generate(seed=42, samples_per_rule=6)]
    )


@pytest.mark.parametrize("rule", [r for r in BUILTIN_RULES if r.required], ids=lambda r: r.name)
def test_dropping_a_rules_gate_literals_is_caught(rule) -> None:
    plan = DetectorPlan(build_detectors(DetectionConfig()), gated_max_chars=UNGATED_LIMIT)
    index = next(i for i, det in enumerate(plan.detectors) if det.name == rule.name)
    for table in (plan._cs, plan._ci):
        for first in list(table):
            for lit in list(table[first]):
                table[first][lit] = [i for i in table[first][lit] if i != index]
    assert _corpus_mismatch(plan), f"{rule.name}: the differential cannot see a lying gate"


def test_dropping_a_combined_member_is_caught() -> None:
    detectors = build_detectors(DetectionConfig())
    members = [
        i for i, det in enumerate(detectors) if isinstance(det, RegexDetector) and not det.required
    ]
    assert len(members) >= 10
    for dropped in members:
        plan = DetectorPlan(detectors, gated_max_chars=UNGATED_LIMIT)
        kept = [i for i in members if i != dropped]
        plan._members = tuple(kept)
        sources = [_combinable(detectors[i].rule.pattern) for i in kept]
        plan._combined = re.compile("|".join(str(source) for source in sources))
        assert _corpus_mismatch(plan), f"{detectors[dropped].name}: a lying combined gate passed"


def test_a_gate_that_never_opens_is_caught() -> None:
    plan = DetectorPlan(build_detectors(DetectionConfig()), gated_max_chars=UNGATED_LIMIT)
    plan._combined = re.compile(r"(?!)")
    assert _corpus_mismatch(plan)


# ---- the gate's parts ----


def test_tiny_strings_run_only_the_always_on_detectors() -> None:
    detectors = _detector_sets()[1]
    plan = DetectorPlan(detectors)
    always = {
        i
        for i, det in enumerate(detectors)
        if not isinstance(det, RegexDetector) or det.name in ("order", "code")
    }
    for text in ("hi", "", "user", "assistant"):
        assert set(plan.candidates(PreparedText(text))) == always, text


def test_candidates_open_on_a_literal_or_a_combined_match() -> None:
    detectors = build_detectors(DetectionConfig())
    plan = DetectorPlan(detectors)
    names = lambda text: {detectors[i].name for i in plan.candidates(PreparedText(text))}  # noqa: E731
    assert "aws_access_key_id" in names("key AKIAIOSFODNN7EXAMPLE")
    assert "email" in names("a@b.example") and "aws_access_key_id" not in names("a@b.example")
    assert "generic_secret" in names("API_KEY = x")  # case-insensitive literal
    assert "credit_card" in names("4111 1111 1111 1111")  # combined member
    assert "credit_card" not in names("no digits at all")
    assert "us_ssn" in names("１２３-４５-６７８９")  # a digit-folded literal


def test_candidates_come_back_in_list_order() -> None:
    plan = DetectorPlan(build_detectors(DetectionConfig()))
    order = plan.candidates(
        PreparedText("a@b.example AKIAIOSFODNN7EXAMPLE 4111 1111 1111 1111 10.0.0.1")
    )
    assert list(order) == sorted(order) and len(order) > 4


def test_an_empty_user_literal_keeps_the_rule_always_on() -> None:
    detectors = build_detectors(
        DetectionConfig(
            enabled=(),
            custom_rules=(CustomRule(name="e", detector_type="E", pattern=r"z\d", required=("",)),),
        )
    )
    plan = DetectorPlan(detectors)
    assert plan.candidates(PreparedText("hi")) == (0,)
    assert [d.value for d in plan.detect("z1", NO_ALLOW)] == ["z1"]


@pytest.mark.parametrize(
    ("pattern", "combinable"),
    [
        (r"\bB\d{5}\b", True),
        (r"a(?i:b)c", True),
        (r"(?:a|b)c", True),
        (r"(a)b", False),
        (r"(?P<value>a)b", False),
        (r"(x)\1", False),
        (r"(?i)abc", False),
        (r"(?u)abc", False),  # a global flag cannot move inside the group
    ],
)
def test_combinable_members(pattern: str, combinable: bool) -> None:
    member = _combinable(re.compile(pattern))
    assert (member is not None) is combinable
    if member is not None:
        assert re.compile(member).pattern == f"(?:{pattern})"


def test_gate_group_prefers_the_group_with_the_longest_shortest_literal() -> None:
    assert _gate_group((("api", "auth"), (":", "="))) == ("api", "auth")
    assert _gate_group(((":", "="), ("aws",))) == ("aws",)
    assert _gate_group((("ab", "c"), ("de",))) == ("de",)


# ---- length threshold ----


def test_a_long_text_runs_every_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = DetectorPlan(build_detectors(DetectionConfig()))

    def refuse(prepared: PreparedText) -> tuple[int, ...]:
        raise AssertionError("a text over the threshold was gated")

    monkeypatch.setattr(plan, "candidates", refuse)
    text = "x" * GATED_MAX_CHARS + " jane@corp.example"
    assert [d.detector_type for d in plan.detect(text, NO_ALLOW)] == ["EMAIL"]


def test_a_text_at_the_threshold_is_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = DetectorPlan(build_detectors(DetectionConfig()))
    seen: list[int] = []
    real = plan.candidates

    def spy(prepared: PreparedText) -> tuple[int, ...]:
        seen.append(len(prepared.text))
        return real(prepared)

    monkeypatch.setattr(plan, "candidates", spy)
    text = ("jane@corp.example " * 100)[:GATED_MAX_CHARS]
    assert plan.detect(text, NO_ALLOW) == plan.detect_each(text, NO_ALLOW)
    assert seen == [GATED_MAX_CHARS]


# ---- plan cache and the redactor ----


def test_plan_for_caches_per_list_and_notices_a_change() -> None:
    detectors = build_detectors(DetectionConfig())
    plan = plan_for(detectors)
    assert plan_for(detectors) is plan
    assert detect_all(detectors, "a@b.example", NO_ALLOW) == plan.detect("a@b.example", NO_ALLOW)
    detectors.append(_Names())
    changed = plan_for(detectors)
    assert changed is not plan and changed.detectors[-1].name == "fake_ner"
    assert plan_for(detectors) is changed
    assert [d.detector_type for d in detect_all(detectors, "Jane Doe", NO_ALLOW)] == ["PERSON"]


def test_plan_cache_holds_no_detector_list_alive() -> None:
    # A reload's old detector list (an NER model can be hundreds of MB) must
    # go once nothing uses it: the cache holds plans weakly.
    import gc
    import weakref

    detectors = build_detectors(DetectionConfig(enabled=("email",)))
    plan = plan_for(detectors)
    assert plan_for(detectors) is plan  # live: reused
    first = weakref.ref(detectors[0])
    del plan, detectors
    gc.collect()
    assert first() is None


def test_a_live_redactor_keeps_its_plan_cached() -> None:
    detectors = build_detectors(DetectionConfig())
    redactor = Redactor(detectors, InMemoryVault(), NO_ALLOW)
    assert plan_for(detectors) is redactor._plan
    assert Redactor(detectors, InMemoryVault(), NO_ALLOW)._plan is redactor._plan


def test_redactor_copies_share_the_compiled_plan() -> None:
    redactor = Redactor(build_detectors(DetectionConfig()), InMemoryVault(), NO_ALLOW)
    raised = redactor.with_floors({"EMAIL": 5})
    assert raised._plan is redactor._plan
    assert raised.redact_text("mail a@b.example") == "mail «EMAIL_006»"
    assert redactor.redact_text("mail c@d.example") == "mail «EMAIL_007»"


def test_a_reload_frees_the_old_detector_list() -> None:
    # The proxy's shared redactor holds the live plan; replacing it on a
    # reload that changed [detection] must free the old detectors.
    import dataclasses
    import gc
    import weakref

    from llm_redact.config import Config
    from llm_redact.proxy import ProxyState

    state = ProxyState(Config(), None)
    old = weakref.ref(state.detectors[0])
    fresh = dataclasses.replace(state.config, detection=DetectionConfig(enabled=("email",)))
    assert state.apply_config(fresh) == []
    gc.collect()
    assert old() is None
    assert [det.name for det in state.redactor._plan.detectors] == ["email"]
