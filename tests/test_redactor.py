from llm_redact.detection.base import Detection
from llm_redact.redactor import BlockedRequest, Redactor, _resolve_overlaps, _sweep
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault


def _d(start: int, end: int, tier: int = 1) -> Detection:
    return Detection(start, end, "T", "x" * (end - start), tier=tier)


def test_basic_substitution(redactor: Redactor, vault: InMemoryVault) -> None:
    out = redactor.redact_text("mail jane@corp.example now")
    assert out == "mail «EMAIL_001» now"
    assert vault.original_for("«EMAIL_001»") == "jane@corp.example"


def test_deterministic_across_calls(redactor: Redactor) -> None:
    first = redactor.redact_text("jane@corp.example")
    second = redactor.redact_text("again jane@corp.example")
    assert first == "«EMAIL_001»"
    assert second == "again «EMAIL_001»"


def test_overlap_specific_beats_generic(redactor: Redactor) -> None:
    out = redactor.redact_text("key sk-ant-api03-abcdefghijklmnopqrstuv end")
    assert "«ANTHROPIC_KEY_001»" in out
    assert "OPENAI" not in out


def test_idempotent_on_redacted_text(redactor: Redactor) -> None:
    once = redactor.redact_text("mail jane@corp.example, ip 8.8.8.8")
    twice = redactor.redact_text(once)
    assert once == twice


def test_full_anthropic_body(redactor: Redactor, rehydrator: Rehydrator) -> None:
    body = {
        "model": "claude-sonnet-4-5",
        "system": "Contact admin@corp.example for help",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "my key is AKIAIOSFODNN7EXAMPLE"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "creds: password = 'X9kQ2mP7vR4wT8jL1hF3bNz6'",
                    },
                ],
            }
        ],
        "max_tokens": 100,
    }
    redacted = redactor.redact_json(body)
    flat = str(redacted)
    assert "AKIAIOSFODNN7EXAMPLE" not in flat
    assert "admin@corp.example" not in flat
    assert "X9kQ2mP7vR4wT8jL1hF3bNz6" not in flat
    assert redacted["model"] == "claude-sonnet-4-5"
    # Round trip restores everything.
    assert rehydrator.rehydrate_json(redacted) == body


def test_full_openai_body(redactor: Redactor, rehydrator: Rehydrator) -> None:
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "user email: jane@corp.example"},
            {"role": "user", "content": "token ghp_" + "a1B2" * 9},
        ],
    }
    redacted = redactor.redact_json(body)
    flat = str(redacted)
    assert "jane@corp.example" not in flat
    assert "ghp_" not in flat
    assert rehydrator.rehydrate_json(redacted) == body


def test_counts_by_type(redactor: Redactor) -> None:
    redactor.redact_text("a@b.example c@d.example AKIAIOSFODNN7EXAMPLE")
    assert redactor.counts["EMAIL"] == 2
    assert redactor.counts["AWS_KEY"] == 1


# ---- overlap resolution: precise span algebra over hand-built detections ----
# These pin the boundary arithmetic in _sweep / _resolve_overlaps directly,
# where "adjacent" (one span ends exactly where the next begins) must never be
# treated as "overlapping". A mutated comparison silently drops a real
# detection — a leak — so the boundaries carry their own tests.


def test_sweep_keeps_adjacent_nonoverlapping() -> None:
    # start == last_end is adjacency, not overlap: both spans survive.
    chosen = _sweep([_d(0, 3), _d(3, 6)])
    assert [(d.start, d.end) for d in chosen] == [(0, 3), (3, 6)]


def test_deny_and_later_rule_both_kept() -> None:
    # A deny (tier 0) and a non-overlapping later rule are both chosen; the
    # tier-0 skip in the merge loop must `continue`, not `break`.
    chosen = _resolve_overlaps([_d(0, 3, tier=0), _d(5, 8, tier=1)])
    assert len(chosen) == 2


def test_deny_adjacent_after_rule_both_kept() -> None:
    # deny ends exactly where the rule starts (3): adjacency, not overlap.
    chosen = _resolve_overlaps([_d(0, 3, tier=0), _d(3, 6, tier=1)])
    assert len(chosen) == 2


def test_rule_adjacent_before_deny_both_kept() -> None:
    # rule ends exactly where the deny starts (3): adjacency, not overlap.
    chosen = _resolve_overlaps([_d(0, 3, tier=1), _d(3, 6, tier=0)])
    assert len(chosen) == 2


def test_deny_overlap_skips_only_the_overlapping_rule() -> None:
    # deny(0,3) overlaps rule(2,5) [dropped] but not rule(10,13) [kept]: the
    # overlap skip must `continue` to later candidates, not `break`.
    chosen = _resolve_overlaps([_d(0, 3, tier=0), _d(2, 5, tier=1), _d(10, 13, tier=1)])
    assert [(d.start, d.end) for d in chosen] == [(0, 3), (10, 13)]


def test_blocked_request_carries_type_in_message() -> None:
    # The exception string carries the type (for logs); it must not be None.
    exc = BlockedRequest("EMAIL")
    assert exc.args == ("EMAIL",)
    assert str(exc) == "EMAIL"
