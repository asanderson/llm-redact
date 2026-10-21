"""User deny strings: detector, precedence, config surfaces."""

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import redactor_before_warn_rule as before
from llm_redact.config import ConfigError, parse_config
from llm_redact.detection.deny import DenyDetector, DenyEntry
from llm_redact.detection.engine import (
    DetectionConfig,
    build_allowlist,
    build_detectors,
)
from llm_redact.redactor import BlockedRequest, Redactor
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault
from warn_oracle import replaced

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


def test_a_deny_string_inside_a_redacted_value_is_redacted_with_it(vault: InMemoryVault) -> None:
    # The email is longer and starts earlier: joined with the deny string,
    # it is redacted whole, as one placeholder typed after it. The deny
    # string once won the overlap and "jane@" went upstream in the clear.
    config = DetectionConfig(deny_strings=(DenyEntry("corp.example"),))
    out = _redactor(vault, config).redact_text("mail jane@corp.example now")
    assert out == "mail «EMAIL_001» now"
    assert vault.original_for("«EMAIL_001»") == "jane@corp.example"
    # A deny string covering the value whole is the placeholder's type.
    config = DetectionConfig(deny_strings=(DenyEntry("mail jane@corp.example"),))
    out = _redactor(InMemoryVault(), config).redact_text("mail jane@corp.example now")
    assert out == "«DENY_001» now"


def test_a_deny_string_inside_a_secret_keeps_no_part_of_it_in_the_clear(
    vault: InMemoryVault,
) -> None:
    key = "AKIAIOSFODNN7EXAMPLE"
    config = DetectionConfig(deny_strings=(DenyEntry("IOSFOD"),))
    out = _redactor(vault, config).redact_text(f"key {key} x")
    assert out == "key «AWS_KEY_001» x"
    assert vault.original_for("«AWS_KEY_001»") == key


def test_deny_wins_over_warn_mode_engulfing(vault: InMemoryVault) -> None:
    # A warn-mode EMAIL match overlapping a deny span loses the overlap: the
    # deny redacts, and the rest of the email ("jane@") is sent as it was —
    # a warning, once, for the email the deny span cut.
    config = DetectionConfig(deny_strings=(DenyEntry("corp.example"),))
    redactor = _redactor(vault, config, modes={"EMAIL": "warn"})
    out = redactor.redact_text("mail jane@corp.example now")
    assert out == "mail jane@«DENY_001» now"
    assert redactor.warn_counts == {"EMAIL": 1}
    # One the deny span covers whole sends nothing: no warning.
    config = DetectionConfig(deny_strings=(DenyEntry("jane@corp.example now"),))
    redactor = _redactor(InMemoryVault(), config, modes={"EMAIL": "warn"})
    assert redactor.redact_text("mail jane@corp.example now") == "mail «DENY_001»"
    assert redactor.warn_counts == {}


@pytest.mark.parametrize("deny", ["corp.example", "mail jane@corp.example now"])
def test_a_block_mode_value_a_deny_string_overlaps_refuses(deny: str) -> None:
    # Joined with the deny string, the block-mode email refuses the request
    # — inside it or around it. The deny string once won the overlap, the
    # email never reached dispatch, and its rest ("jane@") went upstream.
    config = DetectionConfig(deny_strings=(DenyEntry(deny),))
    redactor = _redactor(InMemoryVault(), config, modes={"EMAIL": "block"})
    with pytest.raises(BlockedRequest, match="EMAIL"):
        redactor.redact_text("mail jane@corp.example now")
    assert redactor.blocked_type("mail jane@corp.example now") == "EMAIL"
    # Approved, the email goes as sent where the deny string does not cover it.
    approving = _redactor(InMemoryVault(), config, modes={"EMAIL": "block"})
    approving = approving.with_overrides(_Approving())
    out = approving.redact_text("mail jane@corp.example now")
    assert "«DENY_001»" in out and "corp.example" not in out


class _Approving:
    def allows(self, detector_type: str, value: str) -> bool:
        return True

    def unoverridable(self) -> None:
        return None


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


@pytest.mark.parametrize(
    ("entries", "placeholder"),
    [
        ((DenyEntry("acme"), DenyEntry("mecorp")), "«DENY_001»"),
        # Typed after the one starting first.
        (
            (DenyEntry("mecorp", detector_type="CORP"), DenyEntry("acme", detector_type="ORG")),
            "«ORG_001»",
        ),
    ],
)
def test_overlapping_deny_entries_are_one_placeholder(
    entries: tuple[DenyEntry, ...], placeholder: str
) -> None:
    # Two deny strings overlapping one another are redacted as one
    # placeholder over their union. One of them once won the overlap and
    # the rest of the other ("corp") went upstream.
    vault = InMemoryVault()
    out = _redactor(vault, DetectionConfig(deny_strings=entries)).redact_text("see acmecorp now")
    assert out == f"see {placeholder} now"
    assert vault.original_for(placeholder) == "acmecorp"
    assert Rehydrator(vault).rehydrate_text(out) == "see acmecorp now"


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


# ---- joined with what they overlap: never less than before ----

# Pieces of agent traffic: values the built-in rules read, and words deny
# strings are drawn from.
_PIECES = (
    "mail",
    "bob@acme.example",
    "jane.doe@corp.example",
    "key",
    "AKIAIOSFODNN7EXAMPLE",
    "card",
    "4111 1111 1111 1111",
    "call",
    "+1 415 555 0100",
    "acmecorp",
    "project aurora",
)
_RULE_MODES = ("redact", "warn", "block")


@settings(deadline=None, max_examples=400)
@given(
    pieces=st.lists(st.sampled_from(_PIECES), min_size=1, max_size=6),
    cuts=st.lists(st.tuples(st.integers(0, 120), st.integers(2, 12)), min_size=1, max_size=3),
    modes=st.dictionaries(
        st.sampled_from(("email", "aws_access_key_id", "credit_card", "phone_number")),
        st.sampled_from(_RULE_MODES),
    ),
)
def test_a_deny_string_never_redacts_or_refuses_less_than_before(
    pieces: list[str], cuts: list[tuple[int, int]], modes: dict[str, str]
) -> None:
    # Deny strings cut out of the text anywhere — inside a value, across
    # two, across one another — with the built-in rules in random modes.
    # Against the redactor that let a deny string win every overlap (the
    # frozen copy) and against the same policy without the deny strings:
    # every refusal either makes is made, every character either redacts
    # is redacted, and every deny string's characters are.
    text = " ".join(pieces)
    deny = tuple(dict.fromkeys(text[a : a + n] for a, n in cuts if text[a : a + n].strip()))
    with_deny = DetectionConfig(
        deny_strings=tuple(DenyEntry(value) for value in deny), modes=tuple(modes.items())
    )
    without = DetectionConfig(modes=tuple(modes.items()))
    now = _outcome(Redactor, with_deny, text)
    for config, redactor in ((with_deny, before.Redactor), (without, Redactor)):
        then = _outcome(redactor, config, text)
        if then is None:
            assert now is None, (text, deny, modes)
        elif now is not None:
            assert then <= now, (text, deny, modes, sorted(then - now))
    if now is not None:
        for d in DenyDetector([DenyEntry(value) for value in deny]).detect(text):
            assert set(range(d.start, d.end)) <= now, (text, d.value)


def _outcome(redactor: type, config: DetectionConfig, text: str) -> set[int] | None:
    """The positions ``text`` has replaced by placeholders, or None when the
    redaction refuses it."""
    from llm_redact.detection.engine import build_modes

    vault = InMemoryVault()
    built = redactor(
        build_detectors(config), vault, build_allowlist(config), modes=build_modes(config)
    )
    try:
        output = built.redact_text(text)
    except (BlockedRequest, before.BlockedRequest):
        return None
    return replaced(text, output, vault)


# ---- where the proxy cannot rewrite the text, a deny string is final ----


@pytest.mark.parametrize(
    "text",
    [
        "jane@corp.example",  # the block-mode email joined with the deny string
        "jane@other.example and corp.example",  # met before it, apart from it
    ],
)
def test_a_deny_string_in_unrewritable_text_mints_no_code(text: str, tmp_path: Path) -> None:
    # A scan (a verbatim field, an upload's extracted text) the deny string
    # refuses for good: the block-mode email met first refuses it too, but
    # with no code — approving the email could never let the text through
    # (it once minted one, whose every-time approval then forwarded the
    # email as sent in other requests).
    from llm_redact.detection.engine import build_modes
    from llm_redact.overrides import OverrideScope, OverrideStore

    config = DetectionConfig(deny_strings=(DenyEntry("corp.example"),), modes=(("email", "block"),))
    store = OverrideStore(tmp_path / "overrides.db")
    scope = OverrideScope(store, "")
    redactor = Redactor(
        build_detectors(config), InMemoryVault(), build_allowlist(config), modes=build_modes(config)
    ).with_overrides(scope)
    with pytest.raises(BlockedRequest, match="EMAIL"):
        redactor.scan_text(text)
    assert scope.refusal_code("block", "openai", "POST", "/v1/fine_tuning/jobs") is None
    # Where the proxy rewrites the text, the email's refusal keeps its code.
    scope = OverrideScope(store, "")
    rewriting = Redactor(
        build_detectors(config), InMemoryVault(), build_allowlist(config), modes=build_modes(config)
    ).with_overrides(scope)
    with pytest.raises(BlockedRequest, match="EMAIL"):
        rewriting.redact_text(text)
    assert scope.refusal_code("block", "openai", "POST", "/v1/chat/completions") is not None
