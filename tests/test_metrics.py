from collections import Counter

from llm_redact.metrics import DurationHistogram, Metrics


def _render(metrics: Metrics) -> str:
    return metrics.render(
        detections=Counter({"EMAIL": 2}),
        rehydrations=Counter({"EMAIL": 1}),
        warnings=Counter({"PHONE": 3}),
        blocked=Counter({"US_SSN": 1}),
        vault_entries=2,
        vault_sessions=1,
    )


def test_exposition_format() -> None:
    metrics = Metrics("0.3.0")
    metrics.observe_request("anthropic", 200, 0.12)
    text = _render(metrics)
    assert text.endswith("\n")
    assert "# TYPE llm_redact_requests_total counter" in text
    assert 'llm_redact_info{version="0.3.0"} 1' in text
    assert 'llm_redact_detections_total{type="EMAIL"} 2' in text
    assert 'llm_redact_rehydrations_total{type="EMAIL"} 1' in text
    assert 'llm_redact_warnings_total{type="PHONE"} 3' in text
    assert 'llm_redact_blocked_total{type="US_SSN"} 1' in text
    assert 'llm_redact_requests_total{provider="anthropic",status="200"} 1' in text
    assert "llm_redact_vault_entries 2" in text
    # Every non-comment line is "name{labels} value" or "name value".
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        assert len(line.rsplit(" ", 1)) == 2


def test_counter_monotonicity_and_passthrough_label() -> None:
    metrics = Metrics("0.3.0")
    metrics.observe_request(None, 200, 0.01)
    metrics.observe_request(None, 200, 0.01)
    metrics.observe_request("anthropic", 413, 0.001)
    text = _render(metrics)
    assert 'llm_redact_requests_total{provider="passthrough",status="200"} 2' in text
    assert 'llm_redact_requests_total{provider="anthropic",status="413"} 1' in text


def test_histogram_cumulativity() -> None:
    histogram = DurationHistogram()
    for seconds in (0.01, 0.2, 0.7, 45.0, 120.0):
        histogram.observe(seconds)
    lines = list(histogram.render("d"))
    buckets = [line for line in lines if line.startswith("d_bucket")]
    counts = [int(line.rsplit(" ", 1)[1]) for line in buckets]
    assert counts == sorted(counts)  # cumulative
    assert buckets[-1] == 'd_bucket{le="+Inf"} 5'
    assert "d_count 5" in lines
    assert any(line.startswith("d_sum") for line in lines)


def test_label_escaping() -> None:
    metrics = Metrics('ver"si\\on')
    text = _render(metrics)
    assert 'version="ver\\"si\\\\on"' in text
