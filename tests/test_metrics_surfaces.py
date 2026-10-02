"""The metrics surfaces this release added, end to end through the real app:

- ``llm_redact_proxy_overhead_seconds``: the proxy's own time, the waits on
  the upstream (and the client) excluded — buffered and streamed answers;
- the access gate's own gauges (its optional ``metrics_samples``): rendered
  under the core's value-free rules, read off the event loop and bounded,
  every fault and invalid sample counted (bookkeeping ``plugin_metrics``);
- the audit sinks' batch/drop counters, the vault map writer's gauges;
- ``[vault] map_write_wait_seconds`` (parse, emit, the proxy's bound,
  doctor, ``llm-redact status``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact import proxy
from llm_redact.config import (
    DEFAULT_MAP_WRITE_WAIT_SECONDS,
    Config,
    ConfigError,
    ProviderConfig,
    VaultConfig,
    parse_config,
)
from llm_redact.config_write import emit_config_toml as emit_config
from llm_redact.metrics import (
    CORE_METRIC_FAMILIES,
    MAX_PLUGIN_LABELS,
    MAX_PLUGIN_SAMPLES,
    Metrics,
    plugin_metric_lines,
)
from llm_redact.proxy import (
    PLUGIN_METRICS_STAGE,
    ProxyState,
    _ClientPaced,
    _RequestTiming,
    _waited,
    create_app,
)
from llm_redact.registry import Registry
from test_access_seam import FakeGate

UPSTREAM = "https://api.openai.test"
UPSTREAM_DELAY = 0.4


def _config(**overrides: Any) -> Config:
    return Config(providers={**Config().providers, "openai": ProviderConfig(UPSTREAM)}, **overrides)


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _chat(text: str = "hello", **extra: Any) -> dict[str, Any]:
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": text}], **extra}


def _render(metrics: Metrics, **kwargs: Any) -> str:
    return metrics.render(
        detections=Counter(),
        rehydrations=Counter(),
        warnings=Counter(),
        blocked=Counter(),
        vault_entries=0,
        vault_sessions=0,
        **kwargs,
    )


# --------------------------------------------------------- proxy overhead --


class _SlowBody(httpx.AsyncByteStream):
    """An upstream body that makes the proxy wait before each chunk."""

    def __init__(self, chunks: list[bytes], delay: float) -> None:
        self._chunks = chunks
        self._delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            await asyncio.sleep(self._delay)
            yield chunk


class _SlowUpstream(httpx.AsyncBaseTransport):
    """Answers after ``delay`` (the provider thinking), its body arriving in
    ``chunks`` each ``delay`` apart (a slow provider streaming)."""

    def __init__(self, chunks: list[bytes], content_type: str, delay: float) -> None:
        self.chunks = chunks
        self.content_type = content_type
        self.delay = delay

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self.delay)
        return httpx.Response(
            200,
            headers={"content-type": self.content_type},
            stream=_SlowBody(self.chunks, self.delay),
        )


def _sum(state: ProxyState, provider: str = "openai") -> tuple[float, int]:
    histogram = state.metrics._overheads[provider]
    return histogram._sum, histogram._count


def _duration(state: ProxyState, streamed: str) -> float:
    return state.metrics._durations[("openai", streamed)]._sum


async def test_a_buffered_answer_excludes_the_upstream_round_trip() -> None:
    body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
    upstream = _SlowUpstream([body[:10], body[10:]], "application/json", UPSTREAM_DELAY)
    app = create_app(_config(), upstream_transport=upstream)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat())
    assert response.status_code == 200
    state: ProxyState = app.state.proxy
    # The provider took three delays (headers, two chunks); the proxy's own
    # share is what is left: far less than one of them.
    assert _duration(state, "false") >= 3 * UPSTREAM_DELAY
    overhead, count = _sum(state)
    assert count == 1
    assert 0 <= overhead < UPSTREAM_DELAY / 2, overhead
    text = (await _client(app).get("/__llm-redact/metrics")).text
    assert "# TYPE llm_redact_proxy_overhead_seconds histogram" in text
    assert 'llm_redact_proxy_overhead_seconds_count{provider="openai"} 1' in text
    assert 'llm_redact_proxy_overhead_seconds_bucket{provider="openai",le="0.001"}' in text


async def test_a_streamed_answer_excludes_the_wait_for_every_chunk() -> None:
    events = [
        b'data: {"choices":[{"index":0,"delta":{"content":"he"}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    upstream = _SlowUpstream(events, "text/event-stream", UPSTREAM_DELAY)
    app = create_app(_config(), upstream_transport=upstream)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat(stream=True))
    assert response.status_code == 200 and b"[DONE]" in response.content
    state: ProxyState = app.state.proxy
    assert _duration(state, "true") >= 4 * UPSTREAM_DELAY
    overhead, count = _sum(state)
    assert count == 1
    assert 0 <= overhead < UPSTREAM_DELAY / 2, overhead


async def test_a_local_refusal_is_all_overhead() -> None:
    app = create_app(_config(max_body_bytes=16), upstream_transport=httpx.MockTransport(None))
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat("x" * 100))
    assert response.status_code == 413
    _overhead, count = _sum(app.state.proxy)
    assert count == 1


def test_a_realtime_row_is_never_an_overhead_sample() -> None:
    state: ProxyState = create_app(_config()).state.proxy
    token = proxy._REQUEST_TIMING.set(_RequestTiming())
    try:
        state.record_request(
            session="default",
            provider="openai",
            method="WS",
            path="/v1/realtime",
            status=101,
            started=time.perf_counter() - 30,
            streamed=True,
            detections={},
            rehydrations={},
            refusal=None,
        )
    finally:
        proxy._REQUEST_TIMING.reset(token)
    assert state.metrics._overheads == {}
    assert state.metrics.requests[("openai", "101")] == 1


async def test_waits_are_charged_to_the_request_and_only_then() -> None:
    timing = _RequestTiming()
    token = proxy._REQUEST_TIMING.set(timing)
    try:
        assert await _waited(asyncio.sleep(0.05, result=7)) == 7
        with pytest.raises(ValueError):
            await _waited(_raises())
    finally:
        proxy._REQUEST_TIMING.reset(token)
    assert timing.waited >= 0.05
    # Without a request's timing (a realtime frame, a background task)
    # nothing is charged and nothing fails.
    assert await _waited(asyncio.sleep(0, result=1)) == 1


async def _raises() -> None:
    raise ValueError("boom")


async def test_the_client_pace_of_a_stream_is_charged_once_per_chunk() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b"a"
        yield b"b"

    timing = _RequestTiming()
    token = proxy._REQUEST_TIMING.set(timing)
    try:
        paced = _ClientPaced(chunks())
    finally:
        proxy._REQUEST_TIMING.reset(token)
    assert aiter(paced) is paced
    assert await anext(paced) == b"a"
    await asyncio.sleep(0.05)  # the client takes its time with the chunk
    assert await anext(paced) == b"b"
    assert timing.waited >= 0.05
    with pytest.raises(StopAsyncIteration):
        await anext(paced)
    # Outside a request: no timing, nothing charged.
    unpaced = _ClientPaced(chunks())
    assert [chunk async for chunk in unpaced] == [b"a", b"b"]


# ------------------------------------------------------- the gate's gauges --


class MeteredGate(FakeGate):
    def __init__(self, samples: Any = (), *, delay: float = 0.0) -> None:
        super().__init__()
        self.samples = samples
        self.delay = delay
        self.calls = 0
        self.threads: set[int] = set()

    def metrics_samples(self) -> Any:
        self.calls += 1
        self.threads.add(threading.get_ident())
        if self.delay:
            time.sleep(self.delay)
        if isinstance(self.samples, Exception):
            raise self.samples
        return self.samples


@pytest.fixture
def install_gate(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(gate: Any) -> Any:
        reg = Registry()
        reg.resolve_license = lambda *args, **kwargs: resolved("team")  # type: ignore[method-assign]
        reg.build_access_gate = lambda config, license: gate  # type: ignore[method-assign]
        monkeypatch.setattr(registry_mod, "_registry", reg)
        app = create_app(_config(), upstream_transport=httpx.MockTransport(None))
        return app

    return install


async def _scrape(app: Any) -> str:
    async with _client(app) as client:
        response = await client.get("/__llm-redact/metrics")
    assert response.status_code == 200
    return response.text


async def test_the_gates_samples_are_rendered_off_the_loop(install_gate: Any) -> None:
    gate = MeteredGate(
        [
            ("llm_redact_users", {"state": "verified"}, 3),
            ("llm_redact_users", {"state": "pending"}, 1),
            ("llm_redact_seats_licensed", {}, 10),
            ("llm_redact_seats_used", {}, 4.0),
        ]
    )
    app = install_gate(gate)
    text = await _scrape(app)
    assert text.count("# TYPE llm_redact_users gauge") == 1
    assert 'llm_redact_users{state="verified"} 3' in text
    assert 'llm_redact_users{state="pending"} 1' in text
    assert "llm_redact_seats_licensed 10" in text
    assert "llm_redact_seats_used 4.0" in text
    assert gate.threads and threading.get_ident() not in gate.threads
    assert app.state.proxy.bookkeeping_errors == Counter()
    assert text.index("llm_redact_uptime_seconds") < text.index("llm_redact_users")


async def test_invalid_samples_are_dropped_and_counted_never_echoed(install_gate: Any) -> None:
    secret = "alice@corp.example"
    gate = MeteredGate(
        [
            ("llm_redact_users", {"state": "verified"}, 2),
            ("llm_redact_user", {"name": secret}, 1),  # a label value outside the charset
            ("users_total", {}, 1),  # outside the prefix
            ("llm_redact_requests_total", {}, 1),  # a core family
            ("llm_redact_request_duration_seconds_bucket", {}, 1),  # a core family's series
            ("llm_redact_x", {"le": "1"}, 1),  # a reserved label
            ("llm_redact_x", {"Bad": "a"}, 1),  # a label name outside the charset
            ("llm_redact_x", {"a": 1}, 1),  # a label value that is not a string
            ("llm_redact_x", ["state"], 1),  # labels that are not a mapping
            ("llm_redact_x", {}, True),  # a bool is not a number
            ("llm_redact_x", {}, math.nan),
            ("llm_redact_x", {}, "1"),
            ("llm_redact_x", {}),  # not a triple
            "llm_redact_x",
            ("llm_redact_users", {"state": "verified"}, 5),  # a duplicate series
        ]
    )
    app = install_gate(gate)
    text = await _scrape(app)
    assert 'llm_redact_users{state="verified"} 2' in text
    assert secret not in text and "users_total" not in text.replace("llm_redact_", "")
    assert "llm_redact_x" not in text
    assert app.state.proxy.bookkeeping_errors[PLUGIN_METRICS_STAGE] == 14
    assert f'llm_redact_bookkeeping_errors_total{{stage="{PLUGIN_METRICS_STAGE}"}} 14' in text


def test_the_sample_bounds() -> None:
    many = [("llm_redact_x", {"n": f"v{i}"}, i) for i in range(MAX_PLUGIN_SAMPLES + 3)]
    lines, dropped = plugin_metric_lines(many)
    assert dropped == 3
    assert len([line for line in lines if not line.startswith("#")]) == MAX_PLUGIN_SAMPLES
    labels = {f"l{i}": "v" for i in range(MAX_PLUGIN_LABELS + 1)}
    assert plugin_metric_lines([("llm_redact_x", labels, 1)]) == ([], 1)
    fits = dict(list(labels.items())[:MAX_PLUGIN_LABELS])
    lines, dropped = plugin_metric_lines([("llm_redact_x", fits, 1)])
    assert dropped == 0 and lines[-1] == 'llm_redact_x{l0="v",l1="v",l2="v",l3="v"} 1'
    for family in CORE_METRIC_FAMILIES:
        assert plugin_metric_lines([(family, {}, 1)]) == ([], 1)
    assert plugin_metric_lines([("llm_redact_x", {}, math.inf)]) == ([], 1)
    # An int no float can hold is not a finite number either (never a raise).
    assert plugin_metric_lines([("llm_redact_x", {}, 10**400)]) == ([], 1)
    assert plugin_metric_lines([("llm_redact_x", {}, 10**6)]) == (
        ["# HELP llm_redact_x Reported by a plugin (llm-redact-pro).", "# TYPE llm_redact_x gauge"]
        + ["llm_redact_x 1000000"],
        0,
    )


@pytest.mark.parametrize(
    ("answer", "kind"),
    [(RuntimeError("store unreadable"), "RuntimeError"), (42, "TypeError")],
)
async def test_a_failing_gate_never_fails_the_scrape(
    install_gate: Any, caplog: pytest.LogCaptureFixture, answer: Any, kind: str
) -> None:
    caplog.set_level(logging.INFO, logger="llm_redact")
    gate = MeteredGate(answer)
    app = install_gate(gate)
    for _ in range(2):
        text = await _scrape(app)
        assert "llm_redact_requests_total" in text and "llm_redact_users" not in text
    assert app.state.proxy.bookkeeping_errors[PLUGIN_METRICS_STAGE] == 2
    # Once per episode, by type only; then the recovery.
    assert caplog.text.count("metrics samples were not read") == 1
    assert f"({kind})" in caplog.text and "store unreadable" not in caplog.text
    gate.samples = [("llm_redact_seats_used", {}, 1)]
    assert "llm_redact_seats_used 1" in await _scrape(app)
    assert "metrics samples are read again" in caplog.text


async def test_an_awaitable_answer_is_refused_and_closed(install_gate: Any) -> None:
    class AsyncGate(FakeGate):
        async def metrics_samples(self) -> list[Any]:
            return [("llm_redact_users", {"state": "verified"}, 1)]

    app = install_gate(AsyncGate())
    with warnings_as_errors():
        text = await _scrape(app)
    assert "llm_redact_users" not in text
    assert app.state.proxy.bookkeeping_errors[PLUGIN_METRICS_STAGE] == 1


class warnings_as_errors:  # noqa: N801 — a context manager, used like one
    """No "coroutine was never awaited" warning may escape."""

    def __enter__(self) -> None:
        import warnings

        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        warnings.simplefilter("error", RuntimeWarning)

    def __exit__(self, *exc: Any) -> None:
        self._catch.__exit__(*exc)


async def test_a_slow_gate_is_bounded_and_never_asked_twice_at_once(
    install_gate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(proxy, "PLUGIN_METRICS_TIMEOUT_SECONDS", 0.05)
    gate = MeteredGate([("llm_redact_seats_used", {}, 1)], delay=0.5)
    app = install_gate(gate)
    started = time.monotonic()
    first = await _scrape(app)
    second = await _scrape(app)  # the first call is still running: not asked again
    assert time.monotonic() - started < 0.45
    assert "llm_redact_seats_used" not in first + second
    assert gate.calls == 1
    assert app.state.proxy.bookkeeping_errors[PLUGIN_METRICS_STAGE] == 2
    await asyncio.sleep(0.6)  # the abandoned call ends; its answer is discarded
    gate.delay = 0
    assert "llm_redact_seats_used 1" in await _scrape(app)


async def _scrapes_at_once(app: Any, n: int = 2) -> list[str]:
    async with _client(app) as client:
        answers = await asyncio.gather(*(client.get("/__llm-redact/metrics") for _ in range(n)))
    assert all(answer.status_code == 200 for answer in answers)
    return [answer.text for answer in answers]


async def test_concurrent_scrapes_share_the_call_in_flight(
    install_gate: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # Two scrapes at once (an HA pair of Prometheus servers): the second
    # once found the first's call running, rendered without the gauges and
    # counted a fault — anyone reaching /metrics could make it flap.
    caplog.set_level(logging.INFO, logger="llm_redact")
    gate = MeteredGate([("llm_redact_seats_used", {}, 1)], delay=0.1)
    app = install_gate(gate)
    texts = await _scrapes_at_once(app)
    assert all("llm_redact_seats_used 1" in text for text in texts)
    assert gate.calls == 1
    assert app.state.proxy.bookkeeping_errors == Counter()
    assert "metrics samples were not read" not in caplog.text


async def test_a_joined_scrape_waits_only_until_the_calls_own_deadline(
    install_gate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(proxy, "PLUGIN_METRICS_TIMEOUT_SECONDS", 0.05)
    gate = MeteredGate([("llm_redact_seats_used", {}, 1)], delay=0.5)
    app = install_gate(gate)
    started = time.monotonic()
    texts = await _scrapes_at_once(app)
    assert time.monotonic() - started < 0.45
    assert not any("llm_redact_seats_used" in text for text in texts)
    assert gate.calls == 1
    # Once per scrape rendered without the gauges.
    assert app.state.proxy.bookkeeping_errors[PLUGIN_METRICS_STAGE] == 2
    await asyncio.sleep(0.6)


async def test_a_gate_without_the_member_adds_nothing(install_gate: Any) -> None:
    app = install_gate(FakeGate())
    text = await _scrape(app)
    assert "Reported by a plugin" not in text
    assert app.state.proxy.bookkeeping_errors == Counter()


# --------------------------------------------- sinks and the map writer --


class _Sink:
    def __init__(self, batches: Any, dropped: Any) -> None:
        self.batches_uploaded = batches
        self.rows_dropped = dropped

    def add(self, row: Any) -> None:
        pass


async def test_the_audit_sinks_counters_are_series() -> None:
    app = create_app(_config(), upstream_transport=httpx.MockTransport(None))
    state: ProxyState = app.state.proxy
    state.audit_s3 = _Sink(7, 2)  # type: ignore[assignment]
    state.audit_azure = _Sink(True, "3")  # type: ignore[assignment] — not counts: left out
    text = await _scrape(app)
    assert 'llm_redact_audit_sink_batches_total{sink="s3"} 7' in text
    assert 'llm_redact_audit_sink_rows_dropped_total{sink="s3"} 2' in text
    assert 'sink="azure"' not in text
    state.audit_s3 = None
    text = await _scrape(app)
    assert "# TYPE llm_redact_audit_sink_batches_total counter" in text
    assert "llm_redact_audit_sink_batches_total{" not in text


async def test_the_map_writer_gauges_without_a_writer() -> None:
    app = create_app(_config(), upstream_transport=httpx.MockTransport(None))
    state: ProxyState = app.state.proxy
    text = await _scrape(app)
    assert "llm_redact_map_write_queue_depth 0" in text
    assert 'llm_redact_map_writes_mode{mode="synchronous"} 1' in text
    assert "llm_redact_map_write_wait_timeouts_total{" not in text
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["vault"]["map_writes_pending"] == 0
    assert status["vault"]["map_write_wait_timeouts_total"] == {}
    assert status["vault"]["map_write_wait_seconds"] == DEFAULT_MAP_WRITE_WAIT_SECONDS
    assert status["local_refusals_total"] == {}

    # A manager whose depth answer is not a count (or that fails) reads 0.
    for answer in (lambda: True, lambda: "3", _broken):
        state.vault_manager.map_writes_pending = answer  # type: ignore[attr-defined]
        assert state.map_writes_pending() == 0
    state.vault_manager.map_writes_pending = lambda: 4  # type: ignore[attr-defined]
    assert "llm_redact_map_write_queue_depth 4" in await _scrape(app)


def _broken() -> int:
    raise OSError("writer gone")


def test_the_map_writes_mode_gauge_is_absent_on_a_bare_metrics() -> None:
    text = _render(Metrics("0"))
    assert "# TYPE llm_redact_map_writes_mode gauge" in text
    assert "llm_redact_map_writes_mode{" not in text


# ------------------------------------------------ map_write_wait_seconds --


@pytest.mark.parametrize("value", [0.25, 1, 60])
def test_map_write_wait_seconds_parses_and_round_trips(value: float) -> None:
    config = parse_config({"vault": {"map_write_wait_seconds": value}}, "test")
    assert config.vault.map_write_wait_seconds == float(value)
    assert parse_config(_toml(emit_config(config)), "test").vault.map_write_wait_seconds == value
    state: ProxyState = create_app(config).state.proxy
    assert state.map_write_wait_seconds == float(value)


def test_the_default_wait_is_not_emitted() -> None:
    assert "map_write_wait_seconds" not in emit_config(Config())
    assert Config().vault.map_write_wait_seconds == proxy.MAP_WRITE_WAIT_SECONDS == 5.0


@pytest.mark.parametrize("value", [0, -1, 60.5, True, "5", math.nan, math.inf])
def test_map_write_wait_seconds_out_of_range_is_refused(value: Any) -> None:
    with pytest.raises(ConfigError, match="map_write_wait_seconds must be a number above 0"):
        parse_config({"vault": {"map_write_wait_seconds": value}}, "test")


def test_doctor_names_the_bound() -> None:
    from llm_redact.doctor_cli import _check_map_writes, _Report

    report = _Report(json_mode=True)
    vault = VaultConfig(backend="sqlite", map_writes="before_answer", map_write_wait_seconds=2.5)
    _check_map_writes(report, Config(vault=vault))
    assert any("at most 2.5 s" in line["message"] for line in report.rows)


def _toml(text: str) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(text)


async def test_every_core_family_is_known_and_documented() -> None:
    # CORE_METRIC_FAMILIES is what a plugin sample may never shadow: it must
    # be exactly what the core renders (a new metric added without it would
    # let a plugin emit a second family of that name). Each one is in the
    # docs' metrics reference.
    from pathlib import Path

    app = create_app(_config(), upstream_transport=httpx.MockTransport(None))
    text = await _scrape(app)
    rendered = {line.split()[2] for line in text.splitlines() if line.startswith("# TYPE ")}
    assert rendered == CORE_METRIC_FAMILIES
    doc = (Path(__file__).resolve().parent.parent / "docs" / "observability.md").read_text(
        encoding="utf-8"
    )
    reference = doc.split("## Metrics reference", 1)[1].split("\n## ", 1)[0]
    # "`llm_redact_vault_entries` / `_sessions`" documents two at once.
    shorthands = re.findall(r"/ `(_[a-z_]+)`", reference)
    for family in rendered:
        assert f"`{family}`" in reference or any(family.endswith(s) for s in shorthands), family
