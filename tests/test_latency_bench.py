"""Shape checks only: timing assertions on shared runners would be flaky."""

import json

from llm_redact.bench.latency import (
    CHECK_CEILINGS_MS,
    LatencyStat,
    ceiling_failures,
    run_latency,
    to_json_list,
    to_markdown,
)

EXPECTED_NAMES = {
    "redact_json_small",
    "redact_json_large",
    "redact_json_prose_large",
    "rehydrate_json_small",
    "rehydrate_json_large",
    "streaming_rehydrate",
    "baseline_json_small",
    "proxy_json_small",
    "proxy_overhead_delta_json_small",
    "baseline_stream_small",
    "proxy_stream_small",
    "proxy_overhead_delta_stream_small",
    "baseline_json_large",
    "proxy_json_large",
    "proxy_overhead_delta_json_large",
    "baseline_stream_large",
    "proxy_stream_large",
    "proxy_overhead_delta_stream_large",
}


def test_run_latency_quick_shape() -> None:
    stats = run_latency(seed=7, quick=True)
    assert {s.name for s in stats} == EXPECTED_NAMES
    for stat in stats:
        assert stat.iterations > 0
        assert stat.payload_bytes > 0
        if not stat.name.startswith("proxy_overhead_delta"):
            # Deltas may legitimately go negative on jitter; raw timings not.
            assert stat.p50_ms >= 0
            assert stat.p50_ms <= stat.p95_ms
    streaming = next(s for s in stats if s.name == "streaming_rehydrate")
    assert streaming.throughput_mb_s is not None and streaming.throughput_mb_s > 0
    # Report renderers work and the JSON form is serializable.
    assert "| redact_json_large |" in to_markdown(stats)
    json.dumps(to_json_list(stats))
    # Every gated name exists in this run's output.
    assert set(CHECK_CEILINGS_MS) <= {s.name for s in stats}


def test_ceiling_failures() -> None:
    ok = [LatencyStat(name, 1.0, 2.0, 3, 100) for name in CHECK_CEILINGS_MS]
    assert ceiling_failures(ok) == []
    slow = [LatencyStat(name, 10_000.0, 10_000.0, 3, 100) for name in CHECK_CEILINGS_MS]
    failures = ceiling_failures(slow)
    assert len(failures) == len(CHECK_CEILINGS_MS)
    assert ceiling_failures([]) != []  # missing stats fail too
