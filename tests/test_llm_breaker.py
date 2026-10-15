"""The llm NER backend's circuit breaker ([detection.ner.llm] breaker_seconds,
breaker_failures).

A model server that hangs rather than refusing would cost every new string a
whole ``timeout_ms`` — strings are asked one at a time for the whole proxy —
so ``breaker_failures`` failures in a row that say the server is down (a
timeout, a transport fault, an HTTP 5xx) open the breaker: until the cooldown
has passed every string that needs a call fails at once, handled as any
failure (``on_error``), without contacting the server; then one probe call
goes out, whose success closes it and whose failure opens it again at once.

The server is ``ner_fakes.FakeModelServer`` behind an ``httpx.MockTransport``;
the breaker's clock is a hand-advanced stand-in.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_redact.cli import _print_posture
from llm_redact.config import Config
from llm_redact.config_write import emit_config_toml
from llm_redact.detection.base import DetectorUnavailable
from llm_redact.detection.engine import DetectionConfig, build_detectors, ner_backend_stats
from llm_redact.detection.llm_ner import BREAKER_OPEN, CircuitBreaker
from local_refusals import nothing_refused
from ner_fakes import FakeModelServer, install_llm, llm_ner_config
from test_llm_ner import (
    EMAIL,
    LLM_TOML,
    NAME,
    PROXY,
    _app,
    _build,
    _chat,
    _doctor,
    _llm_stats,
    _toml,
)


class _Clock:
    """A clock that moves only when a test moves it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _clocked(detector: Any) -> _Clock:
    """Give ``detector`` a breaker with its configured cooldown and
    threshold on a clock the test moves."""
    clock = _Clock()
    detector.breaker = CircuitBreaker(
        detector.breaker.seconds, threshold=detector.breaker.threshold, clock=clock
    )
    return clock


async def _get(app: Any, path: str) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PROXY) as c:
        return await c.get(path)


# --- the breaker itself -----------------------------------------------------------------


def test_the_breaker_opens_fails_fast_probes_once_and_closes() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(30, threshold=1, clock=clock)
    assert breaker.admit() and breaker.status() == {
        "state": "closed",
        "seconds": 30,
        "failures": 0,
        "threshold": 1,
        "reason": None,
        "retry_in_seconds": None,
    }
    assert breaker.failed("timeout") == "opened"
    clock.now = 12.0
    assert not breaker.admit()
    assert breaker.status() == {
        "state": "open",
        "seconds": 30,
        "failures": 1,
        "threshold": 1,
        "reason": "timeout",
        "retry_in_seconds": 18.0,
    }
    # A call admitted while it was closed failing late is no new opening.
    assert breaker.failed("timeout") is None
    clock.now = 42.0  # the cooldown restarted at 12
    assert breaker.status()["state"] == "half_open"
    assert breaker.admit()  # the probe
    assert not breaker.admit()  # one at a time
    assert breaker.status()["state"] == "half_open"
    assert breaker.failed("HTTP 503") == "reopened"
    assert not breaker.admit()
    assert breaker.status()["reason"] == "HTTP 503"
    clock.now = 72.0
    assert breaker.admit()
    breaker.succeeded()
    assert breaker.admit() and breaker.admit()
    assert breaker.status()["state"] == "closed" and breaker.status()["failures"] == 0


def test_it_opens_only_after_threshold_failures_in_a_row() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(30, threshold=3, clock=clock)
    assert breaker.failed("timeout") is None and breaker.failed("HTTP 502") is None
    assert breaker.admit()
    assert breaker.status() == {
        "state": "closed",
        "seconds": 30,
        "failures": 2,
        "threshold": 3,
        "reason": None,
        "retry_in_seconds": None,
    }
    # An answer from the server between them starts the count again.
    breaker.succeeded()
    assert breaker.status()["failures"] == 0
    assert breaker.failed("timeout") is None and breaker.failed("timeout") is None
    clock.now = 5.0
    assert breaker.failed("ConnectError") == "opened"
    assert not breaker.admit()
    status = breaker.status()
    assert (status["state"], status["failures"], status["reason"]) == ("open", 3, "ConnectError")
    assert status["retry_in_seconds"] == 30.0
    # Half open: the probe's failure opens it again at once, no new count.
    clock.now = 35.0
    assert breaker.admit()
    assert breaker.failed("timeout") == "reopened"
    assert not breaker.admit()
    assert breaker.status()["state"] == "open"
    clock.now = 65.0
    assert breaker.admit()
    breaker.succeeded()
    # Closed again: the next opening takes threshold failures in a row anew.
    assert breaker.failed("timeout") is None and breaker.failed("timeout") is None
    assert breaker.admit() and breaker.status()["state"] == "closed"


@given(st.lists(st.booleans(), max_size=40), st.integers(min_value=1, max_value=6))
def test_it_opens_exactly_at_the_first_run_of_threshold_failures(
    outcomes: list[bool], threshold: int
) -> None:
    # True: a call the server failed (down), False: one it answered. Every
    # call goes out until the run reaches the threshold; then it is open.
    breaker = CircuitBreaker(30, threshold=threshold, clock=_Clock())
    run = 0
    for down in outcomes:
        assert breaker.admit()
        if down:
            run += 1
            assert breaker.failed("timeout") == ("opened" if run == threshold else None)
            if run == threshold:
                assert not breaker.admit()
                return
        else:
            run = 0
            breaker.succeeded()
        assert breaker.status() == {
            "state": "closed",
            "seconds": 30,
            "failures": run,
            "threshold": threshold,
            "reason": None,
            "retry_in_seconds": None,
        }


def test_a_probe_that_ends_neither_way_lets_the_next_call_probe() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(5, threshold=1, clock=clock)
    breaker.failed("timeout")
    clock.now = 5.0
    assert breaker.admit() and not breaker.admit()
    breaker.release()
    assert breaker.admit()


@pytest.mark.parametrize("threshold", [1, 3])
def test_breaker_seconds_0_turns_it_off(threshold: int) -> None:
    breaker = CircuitBreaker(0, threshold=threshold, clock=_Clock())
    for _ in range(5):
        assert breaker.failed("timeout") is None
    assert breaker.admit()
    assert breaker.status() == {
        "state": "off",
        "seconds": 0,
        "failures": 0,
        "threshold": threshold,
        "reason": None,
        "retry_in_seconds": None,
    }


def test_only_one_of_many_threads_gets_the_probe() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(1, threshold=1, clock=clock)
    breaker.failed("timeout")
    clock.now = 1.0
    start = threading.Barrier(16)
    admitted: list[bool] = []

    def ask() -> None:
        start.wait()
        admitted.append(breaker.admit())

    threads = [threading.Thread(target=ask) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(admitted) == [False] * 15 + [True]


# --- the backend ------------------------------------------------------------------------


def test_a_hung_server_opens_the_breaker_and_strings_fail_without_a_call(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    server = FakeModelServer([(NAME, "person")], fail="timeout")
    redactor, detector = _build(monkeypatch, server, breaker_failures=1)
    clock = _clocked(detector)
    with caplog.at_level(logging.DEBUG, logger="llm_redact"):
        with pytest.raises(DetectorUnavailable) as raised:
            redactor.redact_text(f"write to {NAME}")
        assert raised.value.reason == "timeout"
        assert server.chats == 1 and detector.stats.breaker_opens == 1
        clock.now = 29.9
        for text in (f"call {NAME}", f"mail {EMAIL}"):
            with pytest.raises(DetectorUnavailable) as raised:
                redactor.redact_text(text)
            assert (raised.value.backend, raised.value.reason) == ("llm", BREAKER_OPEN)
        assert server.chats == 1  # the server was not asked again
        stats = detector.stats
        assert (stats.errors, stats.breaker_short_circuits, stats.breaker_opens) == (3, 2, 1)
        # A string with nothing to read needs no call: never refused.
        assert detector.detect("  ") == []
        # The cooldown has passed: one probe, which the server answers.
        server.fail = None
        clock.now = 30.0
        assert detector.breaker.status()["state"] == "half_open"
        assert redactor.redact_text(f"write to {NAME}") == "write to «PERSON_001»"
        assert server.chats == 2 and detector.breaker.status()["state"] == "closed"
        detector.detect(f"call {NAME}")
        assert server.chats == 3
    lines = [record.getMessage() for record in caplog.records]
    assert [line for line in lines if "circuit breaker open" in line] == [
        "llm NER backend: model 'pii-extractor:test' at http://127.0.0.1:11434 failed"
        " (timeout); requests are refused (on_error = block); circuit breaker open after 1"
        " failure in a row: strings fail at once, without asking it, for 30 s, then one probe"
    ]
    assert any("answers again" in line for line in lines)
    assert NAME not in caplog.text and EMAIL not in caplog.text


@pytest.mark.parametrize("threshold", [1, 3])
@pytest.mark.parametrize("fail", ["timeout", "status", "connection"])
def test_a_failed_probe_opens_it_again_at_once(
    monkeypatch: pytest.MonkeyPatch, fail: str, threshold: int
) -> None:
    server = FakeModelServer([(NAME, "person")], fail=fail)
    _redactor, detector = _build(monkeypatch, server, on_error="skip", breaker_failures=threshold)
    clock = _clocked(detector)
    for _ in range(threshold):
        assert detector.detect(f"hi {NAME}") == []
    assert detector.breaker.status()["state"] == "open"
    clock.now = 30.0
    # The probe, failing: open again at once, however many it took first.
    assert detector.detect(f"hi {NAME}") == []
    assert server.chats == threshold + 1 and detector.stats.breaker_opens == 2
    clock.now = 59.9  # the cooldown restarted with the failed probe
    assert detector.detect(f"hi {NAME}") == []
    assert server.chats == threshold + 1 and detector.stats.breaker_short_circuits == 1
    server.fail = None
    clock.now = 60.0
    assert [d.value for d in detector.detect(f"hi {NAME}")] == [NAME]
    assert server.chats == threshold + 2 and detector.stats.errors == threshold + 2
    assert detector.breaker.status()["failures"] == 0


def test_the_default_opens_it_on_the_third_failure_in_a_row(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    server = FakeModelServer([(NAME, "person")], fail="timeout")
    redactor, detector = _build(monkeypatch, server)
    assert detector.breaker.threshold == 3
    _clocked(detector)
    with caplog.at_level(logging.DEBUG, logger="llm_redact"):
        for count, text in enumerate(("one", "two"), start=1):
            # Below the threshold each string asks the server, and fails
            # with its own reason.
            with pytest.raises(DetectorUnavailable) as raised:
                redactor.redact_text(f"{text} {NAME}")
            assert raised.value.reason == "timeout"
            assert server.chats == count and detector.stats.breaker_opens == 0
            assert detector.breaker.status()["failures"] == count
        with pytest.raises(DetectorUnavailable) as raised:
            redactor.redact_text(f"three {NAME}")
        assert raised.value.reason == "timeout"
        assert server.chats == 3 and detector.stats.breaker_opens == 1
        with pytest.raises(DetectorUnavailable) as raised:
            redactor.redact_text(f"four {NAME}")
        assert raised.value.reason == BREAKER_OPEN and server.chats == 3
    stats = detector.stats
    assert (stats.errors, stats.breaker_short_circuits, stats.breaker_opens) == (4, 1, 1)
    lines = [record.getMessage() for record in caplog.records]
    # The outage is logged at its first failure, and once more as it opens.
    assert [line for line in lines if "failed (timeout)" in line] == [
        "llm NER backend: model 'pii-extractor:test' at http://127.0.0.1:11434 failed"
        " (timeout); requests are refused (on_error = block)",
        "llm NER backend: model server at http://127.0.0.1:11434 failed (timeout); circuit"
        " breaker open after 3 failures in a row: strings fail at once, without asking it,"
        " for 30 s, then one probe",
    ]


@pytest.mark.parametrize("answer", ["chat", "garbage", "redirect"])
def test_an_answer_between_failures_starts_the_count_again(
    monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    # Any answer from the server — a usable one, an unreadable one, an
    # error status below 500 — says it is up: the failures before it no
    # longer count toward an opening.
    server = FakeModelServer([(NAME, "person")], fail="timeout")
    _redactor, detector = _build(monkeypatch, server, on_error="skip")
    _clocked(detector)
    for round_ in range(3):
        for text in ("one", "two"):
            assert detector.detect(f"{text} {round_} {NAME}") == []
        assert detector.breaker.status()["failures"] == 2
        if answer == "chat":
            server.fail = None
            assert [d.value for d in detector.detect(f"ok {round_} {NAME}")] == [NAME]
        elif answer == "garbage":
            server.fail = "garbage"
            assert detector.detect(f"ok {round_} {NAME}") == []
        else:
            server.fail = "redirect"  # a 307: answered, not followed
            assert detector.detect(f"ok {round_} {NAME}") == []
        assert detector.breaker.status()["failures"] == 0
        server.fail = "timeout"
    assert detector.breaker.status()["state"] == "closed"
    assert detector.stats.breaker_opens == detector.stats.breaker_short_circuits == 0
    assert server.chats == 9


@pytest.mark.parametrize("fail", ["garbage", "redirect"])
def test_a_server_that_answers_never_opens_it(monkeypatch: pytest.MonkeyPatch, fail: str) -> None:
    # An unreadable answer or a redirect comes from a server that is up:
    # each string fails on its own, and the next is asked as before.
    server = FakeModelServer([(NAME, "person")], fail=fail)
    _redactor, detector = _build(monkeypatch, server, on_error="skip")
    for text in ("one text", "another text", "a third text"):
        assert detector.detect(text) == []
    assert server.chats == 3
    assert detector.stats.breaker_opens == detector.stats.breaker_short_circuits == 0
    assert detector.breaker.status()["state"] == "closed"


def test_a_probe_the_server_answers_unreadably_still_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeModelServer([(NAME, "person")], fail="status")
    _redactor, detector = _build(monkeypatch, server, on_error="skip", breaker_failures=1)
    clock = _clocked(detector)
    detector.detect("one text")
    server.fail = "garbage"
    clock.now = 30.0
    detector.detect("the probe")  # answered, though not usably: the server is up
    assert detector.breaker.status()["state"] == "closed"
    detector.detect("asked again")
    assert server.chats == 3 and detector.stats.breaker_opens == 1


def test_an_unexpected_fault_during_the_probe_frees_the_next_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeModelServer([(NAME, "person")], fail="timeout")
    _redactor, detector = _build(monkeypatch, server, on_error="skip", breaker_failures=1)
    clock = _clocked(detector)
    detector.detect("one text")
    clock.now = 30.0
    entities = detector._client.entities

    def interrupted(*args: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(detector._client, "entities", interrupted)
    with pytest.raises(KeyboardInterrupt):
        detector.detect("the probe")
    monkeypatch.setattr(detector._client, "entities", entities)
    server.fail = None
    assert [d.value for d in detector.detect(f"hi {NAME}")] == [NAME]


def test_a_string_read_before_is_still_answered_from_the_cache_while_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # What the server already said about a string needs no call.
    server = FakeModelServer([(NAME, "person")], fail="timeout", fail_on="secret")
    _redactor, detector = _build(monkeypatch, server, breaker_failures=1)
    _clocked(detector)
    assert [d.value for d in detector.detect(f"hi {NAME}")] == [NAME]
    with pytest.raises(DetectorUnavailable):
        detector.detect(f"secret {NAME}")
    assert [d.value for d in detector.detect(f"hi {NAME}")] == [NAME]
    assert server.chats == 2


def test_a_long_string_stops_at_the_first_window_once_it_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filler = "lorem ipsum dolor sit amet " * 200
    server = FakeModelServer([(NAME, "person")], fail="timeout")
    _redactor, detector = _build(monkeypatch, server, on_error="skip", breaker_failures=1)
    _clocked(detector)
    assert detector.detect(f"{NAME} {filler}") == []
    assert detector.detect(f"{filler} {NAME}") == []
    assert server.chats == 1


def test_a_long_string_running_out_of_its_time_on_a_server_that_answers_never_opens_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # timeout_ms bounds a string's windows together: a long string whose
    # windows the server answers, slowly, runs out of time part way. That
    # fails the string, not the server: the next string is asked as before.
    import types

    import llm_redact.detection.llm_ner as llm_ner

    clock = _Clock()
    server = FakeModelServer([(NAME, "person")])

    def slow(request: httpx.Request) -> httpx.Response:
        answer = server(request)
        if request.url.path == "/api/chat":
            clock.now += 0.6  # each window answered in 0.6 s
        return answer

    _redactor, detector = _build(
        monkeypatch, server, on_error="skip", timeout_ms=1000, breaker_failures=1
    )
    monkeypatch.setattr(llm_ner, "time", types.SimpleNamespace(monotonic=clock))
    detector._client._http = httpx.Client(transport=httpx.MockTransport(slow))
    filler = "lorem ipsum dolor sit amet " * 300
    assert detector.detect(f"{filler} {NAME}") == []  # out of time before its end
    assert server.chats == 2 and detector.stats.errors == 1
    assert detector.stats.breaker_opens == 0
    assert detector.breaker.status()["state"] == "closed"
    assert [d.value for d in detector.detect(f"hi {NAME}")] == [NAME]
    assert server.chats == 3
    # A first window timing out (the server answered nothing of the string)
    # still opens it.
    server.fail = "timeout"
    assert detector.detect(f"bye {NAME}") == []
    assert detector.stats.breaker_opens == 1
    assert detector.breaker.status()["state"] == "open"


def test_a_long_strings_own_timeout_never_counts_toward_an_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A long string whose windows the server answers, slowly, running out of
    # its own time is no failure in a row: the server answered, so the count
    # starts again, and two failures that say it is down are still needed.
    import types

    import llm_redact.detection.llm_ner as llm_ner

    clock = _Clock()
    server = FakeModelServer([(NAME, "person")])

    def slow(request: httpx.Request) -> httpx.Response:
        answer = server(request)
        if request.url.path == "/api/chat":
            clock.now += 0.6  # each window answered in 0.6 s
        return answer

    _redactor, detector = _build(
        monkeypatch, server, on_error="skip", timeout_ms=1000, breaker_failures=2
    )
    monkeypatch.setattr(llm_ner, "time", types.SimpleNamespace(monotonic=clock))
    detector._client._http = httpx.Client(transport=httpx.MockTransport(slow))
    server.fail = "timeout"
    assert detector.detect(f"hi {NAME}") == []
    assert detector.breaker.status()["failures"] == 1
    server.fail = None
    filler = "lorem ipsum dolor sit amet " * 300
    assert detector.detect(f"{filler} {NAME}") == []  # out of time before its end
    assert detector.stats.errors == 2
    assert detector.breaker.status()["failures"] == 0
    server.fail = "timeout"
    assert detector.detect(f"bye {NAME}") == []
    assert detector.breaker.status()["state"] == "closed"
    assert detector.detect(f"again {NAME}") == []
    assert detector.breaker.status()["state"] == "open"
    assert detector.stats.breaker_opens == 1


def test_breaker_seconds_0_asks_the_server_for_every_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeModelServer([(NAME, "person")], fail="timeout")
    _redactor, detector = _build(monkeypatch, server, on_error="skip", breaker_seconds=0)
    for text in ("one", "two", "three"):
        detector.detect(f"{text} {NAME}")
    assert server.chats == 3 and detector.stats.errors == 3
    assert detector.stats.breaker_opens == detector.stats.breaker_short_circuits == 0


# --- the proxy --------------------------------------------------------------------------


async def test_while_open_requests_are_refused_503_without_asking_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeModelServer([(NAME, "person")], fail="timeout")
    app, upstream = _app(monkeypatch, server, breaker_failures=1)
    ((_name, detector, _stats),) = ner_backend_stats(app.state.proxy.detectors)
    clock = _clocked(detector)
    first = await _chat(app, f"hi {NAME}")
    second = await _chat(app, f"bye {NAME}")
    assert first.status_code == second.status_code == 503
    assert "(timeout)" in first.text and f"({BREAKER_OPEN})" in second.text
    assert upstream.bodies == [] and server.chats == 1
    state = app.state.proxy
    assert state.metrics.local_refusals == Counter({("detector_unavailable", "openai"): 2})
    status = (await _get(app, "/__llm-redact/status")).json()
    entry = status["detection"]["ner"]["backends"]["llm"]
    assert entry["breaker"] == {
        "state": "open",
        "seconds": 30,
        "failures": 1,
        "threshold": 1,
        "reason": "timeout",
        "retry_in_seconds": 30.0,
    }
    server.fail = None
    clock.now = 30.0
    assert (await _chat(app, f"hi {NAME}")).status_code == 200
    assert server.chats == 2


async def test_status_shows_the_failures_in_a_row_and_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeModelServer([(NAME, "person")], fail="timeout")
    app, upstream = _app(monkeypatch, server)
    ((_name, detector, _stats),) = ner_backend_stats(app.state.proxy.detectors)
    _clocked(detector)
    for content in (f"hi {NAME}", f"bye {NAME}"):
        assert (await _chat(app, content)).status_code == 503
    status = (await _get(app, "/__llm-redact/status")).json()
    assert status["detection"]["ner"]["backends"]["llm"]["breaker"] == {
        "state": "closed",
        "seconds": 30,
        "failures": 2,
        "threshold": 3,
        "reason": None,
        "retry_in_seconds": None,
    }
    assert (await _chat(app, f"again {NAME}")).status_code == 503
    status = (await _get(app, "/__llm-redact/status")).json()
    breaker = status["detection"]["ner"]["backends"]["llm"]["breaker"]
    assert (breaker["state"], breaker["failures"], breaker["reason"]) == ("open", 3, "timeout")
    assert upstream.bodies == [] and server.chats == 3


async def test_with_skip_requests_go_on_and_the_open_breaker_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeModelServer([(NAME, "person")], fail="status")
    app, upstream = _app(monkeypatch, server, on_error="skip", breaker_failures=1)
    ((_name, detector, _stats),) = ner_backend_stats(app.state.proxy.detectors)
    clock = _clocked(detector)
    for content in (f"hi {NAME}, {EMAIL}", f"bye {NAME}, {EMAIL}"):
        assert (await _chat(app, content)).status_code == 200
    assert server.chats == 1
    for sent in upstream.bodies:
        # The other detectors still applied; the name is the coverage gap.
        assert EMAIL not in sent.decode() and NAME in sent.decode()
    nothing_refused(app.state.proxy)
    clock.now = 10.0
    status = (await _get(app, "/__llm-redact/status")).json()
    entry = status["detection"]["ner"]["backends"]["llm"]
    assert entry["breaker"]["state"] == "open" and entry["breaker"]["reason"] == "HTTP 500"
    counters = entry["counters"]
    assert (counters["errors"], counters["breaker_opens"], counters["breaker_short_circuits"]) == (
        2,
        1,
        1,
    )
    _print_posture(status)
    out = capsys.readouterr().out
    assert (
        "NER llm circuit breaker OPEN (HTTP 500): every string fails at once without asking"
        " its model server — requests go on WITHOUT its findings; a probe in 20 s (1"
        " string(s) failed at once so far)"
    ) in out
    assert 'NER llm on_error = "skip": 2 string(s) forwarded without its findings' in out
    metrics = (await _get(app, "/__llm-redact/metrics")).text
    assert 'llm_redact_ner_breaker_opens_total{backend="llm"} 1' in metrics
    assert 'llm_redact_ner_breaker_short_circuits_total{backend="llm"} 1' in metrics
    assert 'llm_redact_ner_errors_total{backend="llm"} 2' in metrics
    # Half open: the next string probes it.
    clock.now = 30.0
    status = (await _get(app, "/__llm-redact/status")).json()
    _print_posture(status)
    assert "the next string probes it" in capsys.readouterr().out


def test_the_posture_line_says_a_block_breaker_refuses(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.cli import _ner_posture

    entry = {
        "on_error": "block",
        "breaker": {"state": "open", "reason": "time\x1bout", "retry_in_seconds": 3.5},
        "counters": {"breaker_short_circuits": 4},
    }
    (line,) = _ner_posture({"backends": {"llm": entry}})
    assert line == (
        "NER llm circuit breaker OPEN (time\\x1bout): every string fails at once without"
        " asking its model server — requests that need it are refused 503; a probe in 3.5 s"
        " (4 string(s) failed at once so far)"
    )
    assert _ner_posture({"backends": {"llm": {**entry, "breaker": {"state": "closed"}}}}) == []


async def test_a_reload_that_rebuilds_the_detector_starts_a_closed_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeModelServer([(NAME, "person")], fail="timeout")
    app, _upstream = _app(monkeypatch, server, on_error="skip", breaker_failures=1)
    state = app.state.proxy
    await _chat(app, f"hi {NAME}")
    ((_n, before, _s),) = ner_backend_stats(state.detectors)
    assert before.breaker.status()["state"] == "open"
    ner = dataclasses.replace(state.config.detection.ner, entities=("PERSON", "ADDRESS"))
    detection = dataclasses.replace(state.config.detection, ner=ner)
    state.apply_config(dataclasses.replace(state.config, detection=detection))
    ((_n, after, _s),) = ner_backend_stats(state.detectors)
    assert after is not before and after.breaker.status()["state"] == "closed"
    assert _llm_stats(app).errors == 0
    await _chat(app, f"bye {NAME}")
    assert server.chats == 2  # asked again


# --- configuration ----------------------------------------------------------------------


def test_breaker_seconds_parses_round_trips_and_defaults_to_30() -> None:
    assert _toml(LLM_TOML).detection.ner.llm.breaker_seconds == 30
    for seconds in (0, 5, 3600):
        config = _toml(LLM_TOML + f"breaker_seconds = {seconds}\n")
        assert config.detection.ner.llm.breaker_seconds == seconds
        emitted = emit_config_toml(config)
        assert _toml(emitted) == config
        assert ("breaker_seconds" in emitted) == (seconds != 30)


@pytest.mark.parametrize("value", ["-1", "3601", "1.5", "true", '"30"'])
def test_a_bad_breaker_seconds_is_refused(value: str) -> None:
    from llm_redact.config import ConfigError

    with pytest.raises(ConfigError, match="breaker_seconds must be an integer from 0"):
        _toml(LLM_TOML + f"breaker_seconds = {value}\n")


def test_breaker_failures_parses_round_trips_and_defaults_to_3() -> None:
    assert _toml(LLM_TOML).detection.ner.llm.breaker_failures == 3
    for failures in (1, 2, 3, 100):
        config = _toml(LLM_TOML + f"breaker_failures = {failures}\n")
        assert config.detection.ner.llm.breaker_failures == failures
        emitted = emit_config_toml(config)
        assert _toml(emitted) == config
        assert ("breaker_failures" in emitted) == (failures != 3)
    # Accepted with the breaker off, and on an inactive table.
    _toml(LLM_TOML + "breaker_seconds = 0\nbreaker_failures = 5\n")
    _toml("[detection.ner.llm]\nbreaker_failures = 5\n")


@pytest.mark.parametrize("value", ["0", "-1", "101", "2.0", "true", '"3"'])
def test_a_bad_breaker_failures_is_refused(value: str) -> None:
    from llm_redact.config import ConfigError

    with pytest.raises(ConfigError, match="breaker_failures must be an integer from 1 to 100"):
        _toml(LLM_TOML + f"breaker_failures = {value}\n")


def test_the_detector_gets_the_configured_cooldown_and_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_llm(monkeypatch, FakeModelServer())
    config = DetectionConfig(ner=llm_ner_config(breaker_seconds=7, breaker_failures=4))
    ((_n, detector, _s),) = ner_backend_stats(build_detectors(config))
    status = detector.breaker.status()
    assert (status["seconds"], status["threshold"], status["failures"]) == (7, 4, 0)
    assert Config(detection=config).detection.ner.llm.breaker_seconds == 7


def test_doctor_warns_when_the_breaker_is_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_llm(monkeypatch, FakeModelServer())
    rows = _doctor(tmp_path, LLM_TOML + "breaker_seconds = 0\n")
    assert any(
        level == "WARN" and "breaker_seconds = 0 (circuit breaker off)" in message
        for level, message in rows["llm"]
    )
    rows = _doctor(tmp_path, LLM_TOML)
    assert not any("breaker" in message for _level, message in rows["llm"])
