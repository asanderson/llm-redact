"""Price table, usage parsing and stream usage tracking (routing budgets, R-24/R-26)."""

from __future__ import annotations

import copy
import importlib.resources
import json
from pathlib import Path
from typing import Any

import pytest

from llm_redact.config import ConfigError
from llm_redact.pricing import (
    BUILTIN_PRICES_RESOURCE,
    PriceTable,
    StreamUsageTracker,
    Usage,
    inject_stream_usage,
    parse_model_price,
    parse_price_table,
    parse_usage,
)
from llm_redact.routing import ModelPrice

SONNET = ModelPrice(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)
OPUS = ModelPrice(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25)
GPT5 = ModelPrice(input=1.25, output=10.0, cache_read=0.125, cache_write=1.25)
GPT5_MINI = ModelPrice(input=0.25, output=2.0, cache_read=0.025, cache_write=0.25)
GPT5_PRO = ModelPrice(input=15.0, output=120.0, cache_read=15.0, cache_write=15.0)
O3 = ModelPrice(input=2.0, output=8.0, cache_read=0.5, cache_write=2.0)
O3_PRO = ModelPrice(input=20.0, output=80.0, cache_read=20.0, cache_write=20.0)

TABLE = {
    "claude-sonnet-5": SONNET,
    "claude-opus-5": OPUS,
    "gpt-5": GPT5,
    "gpt-5-mini": GPT5_MINI,
    "gpt-5-pro": GPT5_PRO,
    "o3": O3,
    "o3-pro": O3_PRO,
}

# (sibling, family prefix): each sibling is billed differently from the family
# key that is a boundary-aligned prefix of its id, so the vendored table MUST
# carry its own row — the longest-prefix fallback would otherwise price it at
# the family rate (gpt-5-pro at gpt-5 rates is a 12x under-count).
BUILTIN_SIBLINGS = [
    ("gpt-5-pro", "gpt-5"),
    ("o3-pro", "o3"),
    ("o3-deep-research", "o3"),
    ("o4-mini-deep-research", "o4-mini"),
    ("gpt-4o-realtime-preview", "gpt-4o"),
    ("gpt-4o-mini-realtime-preview", "gpt-4o-mini"),
    ("gemini-2.5-flash-image", "gemini-2.5-flash"),
]


@pytest.fixture
def table() -> PriceTable:
    return PriceTable(TABLE)


# --- package data -----------------------------------------------------------


def test_builtin_prices_ship_as_package_data() -> None:
    # The same importlib.resources lookup PriceTable.builtin() does — fails if
    # prices.json is ever dropped from the wheel (hatch includes non-.py files
    # under src/llm_redact; this pins that it actually did).
    resource = importlib.resources.files("llm_redact").joinpath(BUILTIN_PRICES_RESOURCE)
    data = json.loads(resource.read_text("utf-8"))
    assert data["version"]
    assert "override with [prices.override]" in data["_comment"]
    assert "USD per 1M tokens" in data["_comment"]
    assert isinstance(data["models"], dict) and data["models"]


def test_builtin_table_parses_and_covers_the_named_families() -> None:
    table = PriceTable.builtin()
    assert len(table) > 20
    for model in (
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-haiku-4-5",
        "gpt-5",
        "gpt-4o-mini",
        "o4-mini",
        "text-embedding-3-small",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "anthropic/claude-sonnet-5",
        "openai/gpt-5",
    ):
        assert model in table, model
    # Every vendored rate is a non-negative float (parse_model_price enforces it).
    assert table.lookup("claude-sonnet-5") == SONNET
    assert table.unknown_models == set()


@pytest.mark.parametrize(("sibling", "family"), BUILTIN_SIBLINGS)
def test_builtin_siblings_are_not_priced_at_the_family_rate(sibling: str, family: str) -> None:
    table = PriceTable.builtin()
    assert sibling in table, f"{sibling} needs its own row or the fallback prices it as {family}"
    assert family in table
    assert table.lookup(sibling) is not table.lookup(family)
    assert table.lookup(sibling) != table.lookup(family)
    # A dated snapshot of the sibling still resolves to the sibling, not the family.
    assert table.lookup(f"{sibling}-2026-01-01") is table.lookup(sibling)


def test_builtin_gpt_5_pro_rates_pin_the_twelve_x_gap() -> None:
    table = PriceTable.builtin()
    pro, base = table.lookup("gpt-5-pro"), table.lookup("gpt-5")
    assert pro is not None and base is not None
    assert (pro.input, pro.output) == (15.0, 120.0)
    assert pro.output / base.output == pytest.approx(12.0)


def test_builtin_dotted_successor_is_unpriced_not_the_predecessor() -> None:
    # "gpt-5.2" is a distinct, pricier model: with "." as a boundary it would
    # silently resolve to "gpt-5". Unpriced (doctor WARN) is the honest answer.
    table = PriceTable.builtin()
    assert table.lookup("gpt-5.2") is None
    assert table.lookup("gpt-5.1-codex") is None
    assert table.lookup("o3.5") is None


# --- lookup ladder ------------------------------------------------------------


def test_lookup_exact(table: PriceTable) -> None:
    assert table.lookup("claude-sonnet-5") is SONNET
    assert table.lookup("o3-pro") is O3_PRO


def test_lookup_strips_vendor_prefix(table: PriceTable) -> None:
    assert table.lookup("anthropic/claude-opus-5") is OPUS
    assert table.lookup("openai/gpt-5") is GPT5


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-sonnet-5-20260401", SONNET),  # dated snapshot
        ("claude-opus-5-latest", OPUS),  # -latest alias
        ("anthropic/claude-sonnet-5-20260401", SONNET),  # vendor + date together
    ],
)
def test_lookup_strips_snapshot_suffixes(
    table: PriceTable, model: str, expected: ModelPrice
) -> None:
    assert table.lookup(model) is expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5-codex", GPT5),  # longest prefix: "gpt-5"
        ("gpt-5-mini-2025-08-07", GPT5_MINI),  # "gpt-5-mini" beats "gpt-5"
        ("o3-mini", O3),  # no o3-mini key → "o3" at a "-" boundary
        ("o3-pro-2025-06-10", O3_PRO),  # "o3-pro" beats "o3"
        ("gpt-5-pro-2025-10-06", GPT5_PRO),  # "gpt-5-pro" beats "gpt-5"
        ("openai/gpt-5-pro", GPT5_PRO),  # vendor prefix stripped, then the exact key
        ("gpt-5:free", GPT5),  # ":" is a boundary too (OpenRouter variants)
    ],
)
def test_lookup_longest_prefix_fallback(
    table: PriceTable, model: str, expected: ModelPrice
) -> None:
    assert table.lookup(model) is expected


def test_lookup_sibling_with_its_own_key_never_inherits_the_family_rate(
    table: PriceTable,
) -> None:
    assert table.lookup("gpt-5-pro") is GPT5_PRO
    assert table.lookup("gpt-5-pro") is not GPT5
    # Without the row it WOULD fall back (this is the hazard prices.json guards).
    without = PriceTable({k: v for k, v in TABLE.items() if k != "gpt-5-pro"})
    assert without.lookup("gpt-5-pro") is GPT5


@pytest.mark.parametrize(
    "model", ["gpt-50", "o3x", "muse-64k", "", "claude", "gpt-5.2", "gpt-5.1-mini", "o3.5"]
)
def test_lookup_miss_is_none_and_boundary_aligned(table: PriceTable, model: str) -> None:
    # "gpt-50" / "o3x": a key that is a prefix but not at an id boundary never
    # prices; "gpt-5.2": "." is not a boundary — a dotted successor is a
    # different (pricier) model, never a variant of the undotted key.
    assert table.lookup(model) is None
    # lookup alone never records — only a priced request (cost_usd) does.
    assert table.unknown_models == set()


def test_cost_usd_unknown_model_is_none_and_recorded(table: PriceTable) -> None:
    assert table.cost_usd("muse-64k", Usage(10, 10)) is None
    assert table.cost_usd("muse-64k", Usage(10, 10)) is None
    assert table.unknown_models == {"muse-64k"}


# --- cost arithmetic ----------------------------------------------------------


def test_cost_usd_per_million_all_four_classes(table: PriceTable) -> None:
    million = Usage(1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert table.cost_usd("claude-sonnet-5", million) == pytest.approx(2.0 + 10.0 + 0.2 + 2.5)
    usage = Usage(input_tokens=500, output_tokens=100, cache_read=2000, cache_write=300)
    expected = (500 * 2.0 + 100 * 10.0 + 2000 * 0.2 + 300 * 2.5) / 1_000_000
    assert table.cost_usd("claude-sonnet-5", usage) == pytest.approx(expected)
    assert table.cost_usd("claude-sonnet-5", Usage()) == 0.0


def test_usage_total_tokens_is_the_sum_of_the_classes() -> None:
    assert Usage(1, 2, 3, 4).total_tokens == 10
    assert Usage().total_tokens == 0


# --- from_file / with_overrides ----------------------------------------------


def _json_table(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": "test",
                "_comment": "ignored",
                "models": {
                    "claude-sonnet-5": {
                        "input": 2.0,
                        "output": 10.0,
                        "cache_read": 0.2,
                        "cache_write": 2.5,
                    },
                    "muse-64k": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                },
            }
        )
    )
    return path


def test_from_file_json(tmp_path: Path) -> None:
    table = PriceTable.from_file(_json_table(tmp_path / "prices.json"))
    assert table.lookup("claude-sonnet-5") == SONNET
    # Integer rates are accepted and normalized to floats.
    assert table.lookup("muse-64k") == ModelPrice(0.0, 0.0, 0.0, 0.0)
    assert len(table) == 2


def test_from_file_toml_same_shape(tmp_path: Path) -> None:
    path = tmp_path / "prices.toml"
    path.write_text(
        'version = "test"\n'
        '[models."claude-sonnet-5"]\n'
        "input = 2.0\noutput = 10.0\ncache_read = 0.2\ncache_write = 2.5\n"
        '[models."gpt-5"]\n'
        "input = 1.25\noutput = 10.0\ncache_read = 0.125\ncache_write = 1.25\n"
    )
    table = PriceTable.from_file(path)
    assert table.lookup("claude-sonnet-5") == SONNET
    assert table.lookup("gpt-5") == GPT5


@pytest.mark.parametrize(
    ("name", "text", "needle"),
    [
        ("bad.json", "{not json", "not valid json"),
        ("bad.toml", "models = [\n", "not valid toml"),
        ("nomodels.json", '{"version": "x"}', '"models"'),
        ("scalar.json", "[1, 2]", '"models"'),
        ("badentry.json", '{"models": {"m": 3}}', "table of per-token rates"),
        (
            "missing.json",
            '{"models": {"m": {"input": 1, "output": 2, "cache_read": 3}}}',
            "cache_write must be a finite, non-negative number",
        ),
        (
            "negative.json",
            '{"models": {"m": {"input": -1, "output": 2, "cache_read": 3, "cache_write": 4}}}',
            "input must be a finite, non-negative number",
        ),
        (
            "bool.json",
            '{"models": {"m": {"input": true, "output": 2, "cache_read": 3, "cache_write": 4}}}',
            "input must be a finite, non-negative number",
        ),
        # json.loads accepts the NaN / Infinity literals by default; a NaN rate
        # would make the upstream's usd total NaN and the budget never trip.
        (
            "nan.json",
            '{"models": {"m": {"input": NaN, "output": 2, "cache_read": 3, "cache_write": 4}}}',
            "input must be a finite, non-negative number",
        ),
        (
            "inf.json",
            '{"models": {"m": {"input": 1, "output": Infinity, "cache_read": 3, "cache_write": 4}}}',
            "output must be a finite, non-negative number",
        ),
        (
            "neginf.json",
            '{"models": {"m": {"input": 1, "output": 2, "cache_read": -Infinity, "cache_write": 4}}}',
            "cache_read must be a finite, non-negative number",
        ),
        (
            "huge.json",
            '{"models": {"m": {"input": 1, "output": 2, "cache_read": 3, "cache_write": '
            + "1" * 400
            + "}}}",
            "cache_write must be a finite, non-negative number",
        ),
        # TOML spells them nan / inf.
        (
            "nan.toml",
            '[models."m"]\ninput = nan\noutput = 2.0\ncache_read = 3.0\ncache_write = 4.0\n',
            "input must be a finite, non-negative number",
        ),
        (
            "inf.toml",
            '[models."m"]\ninput = 1.0\noutput = inf\ncache_read = 3.0\ncache_write = 4.0\n',
            "output must be a finite, non-negative number",
        ),
        ("emptyid.json", '{"models": {"": {"input": 1}}}', "non-empty strings"),
    ],
)
def test_from_file_bad_table_is_a_config_error(
    tmp_path: Path, name: str, text: str, needle: str
) -> None:
    path = tmp_path / name
    path.write_text(text)
    with pytest.raises(ConfigError, match=needle):
        PriceTable.from_file(path)


def test_from_file_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read price table"):
        PriceTable.from_file(tmp_path / "absent.json")


def test_parse_helpers_reject_non_tables() -> None:
    with pytest.raises(ConfigError, match="expected a table"):
        parse_price_table("nope", source="s")
    with pytest.raises(ConfigError, match="per-token rates"):
        parse_model_price(None, source="s")
    assert parse_model_price(
        {"input": 1, "output": 2, "cache_read": 3, "cache_write": 4}, source="s"
    ) == ModelPrice(1.0, 2.0, 3.0, 4.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 10**400, -0.5])
def test_parse_model_price_rejects_non_finite_and_negative_rates(bad: float) -> None:
    entry = {"input": 1.0, "output": 2.0, "cache_read": bad, "cache_write": 4.0}
    with pytest.raises(ConfigError, match="cache_read must be a finite, non-negative number"):
        parse_model_price(entry, source="s")
    # Zero is a legitimate rate (local models, embeddings' output side).
    assert parse_model_price({**entry, "cache_read": 0}, source="s").cache_read == 0.0


def test_with_overrides_precedence(table: PriceTable) -> None:
    table.cost_usd("unknown-model", Usage(1, 1))
    cheaper = ModelPrice(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25)
    merged = table.with_overrides({"claude-sonnet-5": cheaper, "muse-64k": ModelPrice(0, 0, 0, 0)})
    assert merged.lookup("claude-sonnet-5") is cheaper  # override wins
    assert merged.lookup("claude-opus-5") is OPUS  # base entries kept
    assert merged.lookup("muse-64k") == ModelPrice(0, 0, 0, 0)  # new entries added
    assert merged.lookup("claude-sonnet-5-20260401") is cheaper  # ladder runs over the merge
    assert table.lookup("claude-sonnet-5") is SONNET  # the original is untouched
    assert merged.unknown_models == set()  # a fresh ledger for the new table


# --- parse_usage --------------------------------------------------------------


def test_parse_usage_anthropic() -> None:
    payload = {
        "type": "message",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 7,
        },
    }
    assert parse_usage("anthropic", payload) == Usage(10, 5, cache_read=7, cache_write=3)
    # Absent cache fields (older API versions) count as zero.
    assert parse_usage("anthropic", {"usage": {"input_tokens": 1, "output_tokens": 2}}) == Usage(
        1, 2
    )


def test_parse_usage_openai_chat_subtracts_cached_tokens() -> None:
    payload = {
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    }
    assert parse_usage("openai", payload) == Usage(60, 20, cache_read=40)
    # No details block: nothing cached.
    assert parse_usage("openai", {"usage": {"prompt_tokens": 9, "completion_tokens": 1}}) == Usage(
        9, 1
    )
    # A cached count larger than the prompt (shape drift) clamps instead of going negative.
    weird = {"usage": {"prompt_tokens": 5, "prompt_tokens_details": {"cached_tokens": 50}}}
    assert parse_usage("openai", weird) == Usage(0, 0, cache_read=5)


def test_parse_usage_openai_responses_shape() -> None:
    payload = {
        "object": "response",
        "usage": {
            "input_tokens": 200,
            "output_tokens": 30,
            "input_tokens_details": {"cached_tokens": 150},
            "output_tokens_details": {"reasoning_tokens": 10},
        },
    }
    assert parse_usage("openai", payload) == Usage(50, 30, cache_read=150)


def test_parse_usage_ollama_native_and_openai_compatible() -> None:
    native = {"model": "muse-64k", "done": True, "prompt_eval_count": 12, "eval_count": 8}
    assert parse_usage("ollama", native) == Usage(12, 8)
    compat = {"usage": {"prompt_tokens": 12, "completion_tokens": 8}}
    assert parse_usage("ollama", compat) == Usage(12, 8)


def test_parse_usage_gemini() -> None:
    payload = {
        "candidates": [],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 30,
            "cachedContentTokenCount": 25,
            "thoughtsTokenCount": 5,
            "totalTokenCount": 135,
        },
    }
    # promptTokenCount includes the cached content; thinking tokens bill as output.
    assert parse_usage("gemini", payload) == Usage(75, 35, cache_read=25)
    assert parse_usage("gemini", {"usageMetadata": {"promptTokenCount": 4}}) == Usage(4, 0)


def test_parse_usage_gemini_buffered_array_takes_the_last_usage_bearing_chunk() -> None:
    # streamGenerateContent WITHOUT alt=sse: a buffered JSON array of chunks.
    # usageMetadata grows across chunks; the last one carrying it is complete.
    chunks: list[Any] = [
        {"candidates": [{"content": {"parts": [{"text": "a"}]}}]},
        {
            "candidates": [{"content": {"parts": [{"text": "b"}]}}],
            "usageMetadata": {"promptTokenCount": 10},
        },
        "not a chunk",
        {
            "candidates": [{"content": {"parts": [{"text": "c"}]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 6,
                "cachedContentTokenCount": 4,
                "thoughtsTokenCount": 2,
            },
        },
        {"candidates": []},  # a trailing chunk without usage does not erase it
    ]
    assert parse_usage("gemini", chunks) == Usage(6, 8, cache_read=4)


@pytest.mark.parametrize(
    ("protocol", "payload"),
    [
        ("anthropic", {"type": "message", "content": []}),
        ("anthropic", {"usage": {}}),
        ("anthropic", {"usage": "n/a"}),
        ("openai", {"usage": None}),
        ("openai", {"usage": {"total_tokens": 5}}),
        ("gemini", {"candidates": []}),
        ("gemini", {"usageMetadata": {"totalTokenCount": 5}}),
        ("ollama", {"model": "muse-64k", "done": False}),
        ("cohere", {"usage": {"prompt_tokens": 1}}),
        ("anthropic", "not a dict"),
        ("anthropic", None),
        # An array body is only Gemini's buffered stream shape; for every other
        # protocol a list is not a response and nothing is guessed from it.
        ("anthropic", [{"usage": {"input_tokens": 1}}]),
        ("openai", [{"usage": {"prompt_tokens": 1}}]),
        ("ollama", [{"prompt_eval_count": 1}]),
        ("gemini", []),
        ("gemini", [{"candidates": []}, "junk", None]),
        # Buffered (non-streaming) OpenAI bodies never nest usage: a Responses
        # lifecycle event fed as a whole body is not a response body.
        ("openai", {"type": "response.completed", "response": {"usage": {"input_tokens": 1}}}),
    ],
)
def test_parse_usage_absent_is_none(protocol: str, payload: Any) -> None:
    assert parse_usage(protocol, payload) is None


def test_parse_usage_tolerates_shape_drift() -> None:
    drift = {"usage": {"input_tokens": "12", "output_tokens": True, "cache_read_input_tokens": -4}}
    assert parse_usage("anthropic", drift) == Usage(0, 0, 0, 0)
    assert parse_usage("anthropic", {"usage": {"input_tokens": 3.9, "output_tokens": 2}}) == Usage(
        3, 2
    )


# --- StreamUsageTracker -------------------------------------------------------


def _sse(events: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(event) for event in events]


def test_tracker_anthropic_message_start_then_delta_cumulative_last_wins() -> None:
    tracker = StreamUsageTracker("anthropic")
    for data in _sse(
        [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {
                        "input_tokens": 25,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 10,
                    },
                },
            },
            {"type": "ping"},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": None},
                "usage": {"output_tokens": 15},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 40},
            },
            {"type": "message_stop"},
        ]
    ):
        tracker.feed_sse_data(data)
    assert tracker.result() == Usage(input_tokens=25, output_tokens=40, cache_read=10)


def test_tracker_anthropic_delta_input_fields_override_when_present() -> None:
    tracker = StreamUsageTracker("anthropic")
    tracker.feed_sse_data(
        json.dumps(
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 25, "output_tokens": 1}},
            }
        )
    )
    # Newer API versions repeat the input side on message_delta (cumulative).
    tracker.feed_sse_data(
        json.dumps(
            {
                "type": "message_delta",
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 7,
                    "cache_creation_input_tokens": 4,
                    "cache_read_input_tokens": None,
                },
            }
        )
    )
    assert tracker.result() == Usage(input_tokens=30, output_tokens=7, cache_write=4)


def test_tracker_anthropic_delta_without_start_still_counts() -> None:
    tracker = StreamUsageTracker("anthropic")
    tracker.feed_sse_data(json.dumps({"type": "message_delta", "usage": {"output_tokens": 3}}))
    assert tracker.result() == Usage(output_tokens=3)
    # A delta with no usage (or an unrelated one) changes nothing.
    tracker.feed_sse_data(json.dumps({"type": "message_delta", "usage": {"unrelated": 1}}))
    tracker.feed_sse_data(json.dumps({"type": "message_delta"}))
    assert tracker.result() == Usage(output_tokens=3)


def test_tracker_openai_last_chunk_with_usage_wins() -> None:
    tracker = StreamUsageTracker("openai")
    chunks = [
        {
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": "a"}}],
            "usage": None,
        },
        {"object": "chat.completion.chunk", "choices": [{"delta": {"content": "b"}}]},
        {
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 60},
            },
        },
    ]
    for data in _sse(chunks):
        tracker.feed_sse_data(data)
    tracker.feed_sse_data("[DONE]")
    assert tracker.result() == Usage(40, 2, cache_read=60)


def _responses_stream(terminal: str) -> list[dict[str, Any]]:
    """The Responses API stream shape (Codex CLI, /v1/responses stream:true),
    nested like tests/test_provider_openai_responses.py's fixtures: NO event
    carries a top-level ``usage``; the lifecycle events before the terminal
    one carry ``response.usage: null``."""
    return [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {"id": "resp_1", "object": "response", "status": "in_progress", "usage": None},
        },
        {
            "type": "response.in_progress",
            "sequence_number": 1,
            "response": {"id": "resp_1", "object": "response", "status": "in_progress", "usage": None},
        },
        {"type": "response.output_item.added", "output_index": 0, "item": {"id": "item_1"}},
        {"type": "response.output_text.delta", "item_id": "item_1", "delta": "hi"},
        {"type": "response.output_text.done", "item_id": "item_1", "text": "hi"},
        {
            "type": terminal,
            "sequence_number": 9,
            "response": {
                "id": "resp_1",
                "object": "response",
                "status": terminal.rpartition(".")[2],
                "output": [
                    {
                        "id": "item_1",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hi"}],
                    }
                ],
                "usage": {
                    "input_tokens": 200,
                    "input_tokens_details": {"cached_tokens": 150},
                    "output_tokens": 30,
                    "output_tokens_details": {"reasoning_tokens": 10},
                    "total_tokens": 230,
                },
            },
        },
    ]


@pytest.mark.parametrize("terminal", ["response.completed", "response.incomplete"])
def test_tracker_openai_responses_stream_reads_nested_response_usage(terminal: str) -> None:
    tracker = StreamUsageTracker("openai")
    events = _responses_stream(terminal)
    for data in _sse(events[:-1]):
        tracker.feed_sse_data(data)
    # Nothing before the terminal event carries usage (response.usage is null).
    assert tracker.result() is None
    tracker.feed_sse_data(json.dumps(events[-1]))
    assert tracker.result() == Usage(50, 30, cache_read=150)


def test_tracker_openai_responses_null_usage_never_erases_a_count() -> None:
    tracker = StreamUsageTracker("openai")
    events = _responses_stream("response.completed")
    tracker.feed_sse_data(json.dumps(events[-1]))
    assert tracker.result() == Usage(50, 30, cache_read=150)
    # A stray lifecycle event (or a non-table response) after the count: last
    # wins only among events that CARRY usage.
    tracker.feed_sse_data(json.dumps(events[0]))
    tracker.feed_sse_data(json.dumps({"type": "response.completed", "response": "odd"}))
    tracker.feed_sse_data(json.dumps({"type": "response.completed"}))
    assert tracker.result() == Usage(50, 30, cache_read=150)


def test_tracker_nested_response_usage_is_openai_only() -> None:
    # The nesting is a Responses-API fact; other protocols never look there.
    completed = _responses_stream("response.completed")[-1]
    for protocol in ("anthropic", "gemini", "ollama", "cohere"):
        tracker = StreamUsageTracker(protocol)
        tracker.feed_sse_data(json.dumps(completed))
        assert tracker.result() is None, protocol


def test_tracker_gemini_last_usage_metadata_wins() -> None:
    tracker = StreamUsageTracker("gemini")
    for data in _sse(
        [
            {
                "candidates": [{"content": {"parts": [{"text": "a"}]}}],
                "usageMetadata": {"promptTokenCount": 10},
            },
            {
                "candidates": [{"finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 6,
                    "thoughtsTokenCount": 2,
                },
            },
        ]
    ):
        tracker.feed_sse_data(data)
    assert tracker.result() == Usage(10, 8)


def test_tracker_ollama_ndjson_done_line() -> None:
    tracker = StreamUsageTracker("ollama")
    lines = [
        b'{"model":"muse-64k","message":{"role":"assistant","content":"hi"},"done":false}\n',
        b"not json at all\n",
        b"\xff\xfe\n",
        b'{"model":"muse-64k","message":{"role":"assistant","content":""},"done":true,'
        b'"prompt_eval_count":12,"eval_count":8}\n',
    ]
    for line in lines:
        tracker.feed_ndjson_line(line)
    assert tracker.result() == Usage(12, 8)


@pytest.mark.parametrize("protocol", ["anthropic", "openai", "gemini", "ollama", "cohere"])
def test_tracker_result_none_when_nothing_seen(protocol: str) -> None:
    tracker = StreamUsageTracker(protocol)
    assert tracker.result() is None
    tracker.feed_sse_data("")
    tracker.feed_sse_data("[DONE]")
    tracker.feed_sse_data("{broken")
    tracker.feed_sse_data("[1, 2, 3]")
    tracker.feed_sse_data(json.dumps({"type": "ping"}))
    tracker.feed_ndjson_line(b"")
    tracker.feed_ndjson_line(b'{"done": false}')
    assert tracker.result() is None


# --- inject_stream_usage ------------------------------------------------------


CHAT_MESSAGES = [{"role": "user", "content": "hi"}]


def test_inject_stream_usage_sets_when_absent_and_never_mutates() -> None:
    body = {"model": "gpt-5", "stream": True, "messages": CHAT_MESSAGES}
    before = copy.deepcopy(body)
    result = inject_stream_usage(body)
    assert result == {**before, "stream_options": {"include_usage": True}}
    assert body == before  # input untouched


def test_inject_stream_usage_preserves_other_options() -> None:
    body = {
        "stream": True,
        "messages": CHAT_MESSAGES,
        "stream_options": {"include_obfuscation": False},
    }
    before = copy.deepcopy(body)
    result = inject_stream_usage(body)
    assert result == {
        "stream": True,
        "messages": CHAT_MESSAGES,
        "stream_options": {"include_obfuscation": False, "include_usage": True},
    }
    assert body == before
    assert result is not None and result["stream_options"] is not body["stream_options"]


@pytest.mark.parametrize(
    "body",
    [
        {"stream": True, "messages": CHAT_MESSAGES, "stream_options": {"include_usage": True}},
        # the client decided
        {"stream": True, "messages": CHAT_MESSAGES, "stream_options": {"include_usage": False}},
        {"stream": False, "messages": CHAT_MESSAGES},
        {"model": "gpt-5", "messages": CHAT_MESSAGES},
        {"stream": "true", "messages": CHAT_MESSAGES},  # not the boolean the API defines
        # not a table: leave it alone
        {"stream": True, "messages": CHAT_MESSAGES, "stream_options": "weird"},
        # A Responses body (input, never messages): its stream already ends with
        # response.completed carrying usage, and the Responses API rejects the
        # parameter it does not define — never inject there.
        {"model": "gpt-5", "input": "hi", "stream": True},
        {"model": "gpt-5", "input": [{"role": "user", "content": "hi"}], "stream": True},
        {"model": "gpt-5", "previous_response_id": "resp_1", "stream": True},
        {"model": "gpt-5", "input": "hi", "stream": True, "stream_options": {}},
        # Nothing that identifies the endpoint at all.
        {"stream": True},
        {"stream": True, "stream_options": None},
    ],
)
def test_inject_stream_usage_no_change_is_none(body: dict[str, Any]) -> None:
    before = copy.deepcopy(body)
    assert inject_stream_usage(body) is None
    assert body == before


def test_inject_stream_usage_null_options_treated_as_absent() -> None:
    assert inject_stream_usage(
        {"stream": True, "messages": CHAT_MESSAGES, "stream_options": None}
    ) == {
        "stream": True,
        "messages": CHAT_MESSAGES,
        "stream_options": {"include_usage": True},
    }
