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
O3 = ModelPrice(input=2.0, output=8.0, cache_read=0.5, cache_write=2.0)
O3_PRO = ModelPrice(input=20.0, output=80.0, cache_read=20.0, cache_write=20.0)

TABLE = {
    "claude-sonnet-5": SONNET,
    "claude-opus-5": OPUS,
    "gpt-5": GPT5,
    "gpt-5-mini": GPT5_MINI,
    "o3": O3,
    "o3-pro": O3_PRO,
}


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
        ("gpt-5:free", GPT5),  # ":" is a boundary too (OpenRouter variants)
    ],
)
def test_lookup_longest_prefix_fallback(
    table: PriceTable, model: str, expected: ModelPrice
) -> None:
    assert table.lookup(model) is expected


@pytest.mark.parametrize("model", ["gpt-50", "o3x", "muse-64k", "", "claude"])
def test_lookup_miss_is_none_and_boundary_aligned(table: PriceTable, model: str) -> None:
    # "gpt-50" / "o3x": a key that is a prefix but not at an id boundary never prices.
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
            "cache_write must be a non-negative number",
        ),
        (
            "negative.json",
            '{"models": {"m": {"input": -1, "output": 2, "cache_read": 3, "cache_write": 4}}}',
            "input must be a non-negative number",
        ),
        (
            "bool.json",
            '{"models": {"m": {"input": true, "output": 2, "cache_read": 3, "cache_write": 4}}}',
            "input must be a non-negative number",
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
        ("anthropic", [{"usage": {"input_tokens": 1}}]),
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


def test_inject_stream_usage_sets_when_absent_and_never_mutates() -> None:
    body = {"model": "gpt-5", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    before = copy.deepcopy(body)
    result = inject_stream_usage(body)
    assert result == {**before, "stream_options": {"include_usage": True}}
    assert body == before  # input untouched


def test_inject_stream_usage_preserves_other_options() -> None:
    body = {"stream": True, "stream_options": {"include_obfuscation": False}}
    before = copy.deepcopy(body)
    result = inject_stream_usage(body)
    assert result == {
        "stream": True,
        "stream_options": {"include_obfuscation": False, "include_usage": True},
    }
    assert body == before
    assert result is not None and result["stream_options"] is not body["stream_options"]


@pytest.mark.parametrize(
    "body",
    [
        {"stream": True, "stream_options": {"include_usage": True}},
        {"stream": True, "stream_options": {"include_usage": False}},  # the client decided
        {"stream": False},
        {"model": "gpt-5"},
        {"stream": "true"},  # not the boolean the API defines
        {"stream": True, "stream_options": "weird"},  # not a table: leave it alone
    ],
)
def test_inject_stream_usage_no_change_is_none(body: dict[str, Any]) -> None:
    before = copy.deepcopy(body)
    assert inject_stream_usage(body) is None
    assert body == before


def test_inject_stream_usage_null_options_treated_as_absent() -> None:
    assert inject_stream_usage({"stream": True, "stream_options": None}) == {
        "stream": True,
        "stream_options": {"include_usage": True},
    }
