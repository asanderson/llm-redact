"""The read-only scan capture seam (``plugin_api.ScanSink``,
``Registry.build_scan_sinks``, ``scan_capture``): a registered sink receives,
off the event loop, exactly the strings a request's redaction scanned, with
their detections and the detector that found each — after the redaction
succeeded and before the upstream is contacted, never for a refused request.
A slow or failing sink changes neither the answer nor its latency beyond the
enqueue, and without a sink (the Free core) nothing is recorded or queued.

Keyless: scripted sinks on a bare Registry stand in for llm-redact-pro.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import json
import logging
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import uvicorn

import llm_redact.detection.engine as engine_mod
import llm_redact.proxy as proxy_mod
import llm_redact.registry as registry_mod
import llm_redact.scan_capture as capture_mod
from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from license_fixtures import resolved
from llm_redact.cli import _print_posture
from llm_redact.config import Config, ConfigError, ProviderConfig
from llm_redact.detection.base import Detection
from llm_redact.detection.engine import (
    Allowlist,
    DetectionConfig,
    DetectorPlan,
    DetectorSource,
    build_allowlist,
    build_detectors,
    detector_source,
)
from llm_redact.ner_prefetch import collect_request_strings
from llm_redact.plugin_api import (
    Inspection,
    ScanContext,
    ScannedDetection,
    ScannedString,
    UploadPart,
)
from llm_redact.proxy import create_app
from llm_redact.redactor import BlockedRequest, Redactor, TooManyStrings
from llm_redact.registry import Registry
from llm_redact.scan_capture import ScanCapture, ScanRecorder, build_scan_capture
from llm_redact.vault import InMemoryVault
from llm_redact.vault_writer import SHUTDOWN_DRAIN_SECONDS as MAP_DRAIN_SECONDS
from ner_fakes import install_spacy
from prefetch_fixtures import (
    ANTHROPIC_HEADERS,
    EMAIL,
    EXEMPT_SERVER,
    NAME,
    OPENAI_HEADERS,
    OTHER,
    SHAPES,
    Shape,
)
from test_access_seam import FakeGate
from test_detector_plan import NO_ALLOW, UNGATED_LIMIT, _detector_sets, _fp_corpus, _texts
from test_ner_prefetch import _adapter
from test_ner_prefetch_e2e import SlowNlp, Upstream, _client, _config, _send

LIMIT = 100_000


class Sink:
    """A scripted scan sink: records what it is asked and handed (and on
    which thread), answers ``wants`` with ``want``; ``capture`` waits for
    ``release`` (set: no wait) and raises ``fail`` when given."""

    def __init__(self, *, want: Any = True, fail: Exception | None = None) -> None:
        self.want = want
        self.fail = fail
        self.wanted: list[ScanContext] = []
        self.captures: list[tuple[ScanContext, tuple[ScannedString, ...]]] = []
        self.threads: set[int] = set()
        self.events: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.closed = 0

    def wants(self, context: ScanContext) -> bool:
        self.wanted.append(context)
        if isinstance(self.want, Exception):
            raise self.want
        return self.want  # type: ignore[no-any-return]

    def capture(self, context: ScanContext, strings: tuple[ScannedString, ...]) -> None:
        self.threads.add(threading.get_ident())
        self.entered.set()
        self.release.wait(10)
        self.events.append("capture")
        if self.fail is not None:
            raise self.fail
        self.captures.append((context, strings))

    def close(self) -> None:
        self.events.append("close")
        self.closed += 1

    def texts(self) -> list[list[str]]:
        return [[scanned.text for scanned in strings] for _context, strings in self.captures]


def _register(
    monkeypatch: pytest.MonkeyPatch, *sinks: Any, reg: Registry | None = None
) -> Registry:
    reg = reg or Registry()
    reg.build_scan_sinks = lambda config, tier: sinks  # type: ignore[method-assign]
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return reg


async def _delivered(app: Any) -> None:
    """Wait for the capture's queue to empty, and for the counts its worker
    posted to the loop to land."""
    capture = app.state.proxy.scan_capture
    assert await asyncio.to_thread(capture.drain, 10) == 0
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def _expected(plan: DetectorPlan, allowlist: Allowlist, text: str) -> tuple[ScannedDetection, ...]:
    recorder = ScanRecorder(_CONTEXT, (0,))
    recorder.record(text, plan.detect_sourced(text, allowlist))
    (scanned,) = recorder.strings()
    return scanned.detections


_CONTEXT = ScanContext("http", "openai", "openai", "POST", "/v1/x", "default", None, False)


# --- the detections and their detectors ---------------------------------------


@pytest.mark.parametrize("limit", [engine_mod.GATED_MAX_CHARS, UNGATED_LIMIT])
@pytest.mark.parametrize("which", [0, 1], ids=["builtin", "custom+deny+ner"])
def test_sourced_detection_is_detection_with_each_detector_named(which: int, limit: int) -> None:
    detectors = _detector_sets()[which]
    plan = DetectorPlan(detectors, gated_max_chars=limit)
    for text in _texts():
        sourced = plan.detect_sourced(text, NO_ALLOW)
        # The same detections, in the same order (overlap resolution reads
        # the order).
        assert [d for d, _source in sourced] == plan.detect(text, NO_ALLOW), text
        # Each one from the detector named: every detector's own findings.
        own = Counter(
            (d, plan.sources[index])
            for index, det in enumerate(detectors)
            for d in det.detect(text)
        )
        assert Counter(sourced) == own, text


def test_sourced_detection_on_the_fp_corpus() -> None:
    plan = DetectorPlan(_detector_sets()[1], gated_max_chars=UNGATED_LIMIT)
    for text in _fp_corpus():
        sourced = plan.detect_sourced(text, NO_ALLOW)
        assert [d for d, _source in sourced] == plan.detect(text, NO_ALLOW)


def test_each_kind_of_detector_is_named() -> None:
    plan = DetectorPlan(_detector_sets()[1])
    kinds = {source.kind for source in plan.sources}
    assert kinds == {"rule", "custom", "deny", "other"}
    names = {source.name: source.kind for source in plan.sources}
    assert names["email"] == "rule"
    assert names["ticket"] == "custom"
    assert names["deny_strings"] == "deny"
    # Two stand-ins for a model-free plugin detector, one in a type filter
    # (named after what it wraps).
    assert names["fake_ner"] == "other"


class _Heavy:
    name = "hf"
    heavy = True

    def detect(self, text: str) -> list[Detection]:
        return []


class _Nameless:
    name = None

    def detect(self, text: str) -> list[Detection]:
        return []


def test_a_model_backed_detector_is_ner_and_a_filter_is_named_after_it() -> None:
    filtered = engine_mod.TypeFilteredDetector(_Heavy(), frozenset({"EMAIL"}))  # type: ignore[arg-type]
    assert detector_source(filtered, heavy=True) == DetectorSource("ner", "hf")
    assert detector_source(_Heavy(), heavy=True) == DetectorSource("ner", "hf")
    plan = DetectorPlan([filtered, _Nameless()])  # type: ignore[list-item]
    assert plan.sources == (DetectorSource("ner", "hf"), DetectorSource("other", "_Nameless"))


def test_an_ner_backend_is_named_as_configured_not_by_its_detector_name() -> None:
    # The spaCy backend's detector is named "ner"; a sink is told "spacy",
    # the name /status detection.ner.backends uses (its label policy's).
    from llm_redact.detection.labels import LabelPolicy

    class _Spacy(_Heavy):
        name = "ner"
        label_policy = LabelPolicy(("PERSON",), backend="spacy")

    wrapped = engine_mod.TypeFilteredDetector(_Spacy(), frozenset())  # type: ignore[arg-type]
    assert detector_source(wrapped, heavy=True) == DetectorSource("ner", "spacy")
    assert detector_source(_Spacy(), heavy=True) == DetectorSource("ner", "spacy")


def test_a_custom_rule_spelled_like_a_builtin_is_still_custom() -> None:
    (email,) = [r for r in engine_mod.BUILTIN_RULES if r.name == "email"]
    det = engine_mod.RegexDetector(dataclasses.replace(email))
    assert detector_source(det, heavy=False) == DetectorSource("custom", "email")


# --- the recording redactor: the same redaction, every text recorded -----------


class _Record:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[tuple[Detection, DetectorSource]]]] = []

    def record(self, text: str, found: Any) -> None:
        self.calls.append((text, list(found)))


def _redactors(**kwargs: Any) -> tuple[Redactor, Redactor, _Record]:
    detectors = _detector_sets()[1]
    plain = Redactor(detectors, InMemoryVault(), NO_ALLOW, **kwargs)
    other = Redactor(detectors, InMemoryVault(), NO_ALLOW, **kwargs)
    record = _Record()
    return plain, other.with_recorder(record), record


@pytest.mark.parametrize(
    "modes",
    [{}, {"EMAIL": "warn"}, {"PHONE": "block"}],
    ids=["redact", "warn", "block"],
)
def test_a_recording_redactor_redacts_exactly_as_without(modes: dict[str, str]) -> None:
    from llm_redact.detection.deny import DenyDetector, DenyEntry

    added = DetectorPlan([DenyDetector((DenyEntry("Doe at"), DenyEntry("corp")))])
    # Hand-made texts where an added deny string overlaps what the
    # configured detectors redact (their union is redacted), then corpus.
    texts = [
        f"Mail Jane Doe at {EMAIL}",
        f"to {EMAIL} or +1 415 555 0134",
        "Jane Doe at corp",
        *_texts()[:400],
    ]
    for deny in (None, added):
        plain, recording, record = _redactors(modes=modes, added_deny=deny)
        for text in texts:
            outcomes = []
            for redactor in (plain, recording):
                try:
                    outcomes.append(("text", redactor.redact_text(text)))
                except BlockedRequest as exc:
                    outcomes.append(("blocked", exc.detector_type))
                try:
                    outcomes.append(("scan", redactor.scan(text)))
                except BlockedRequest as exc:
                    outcomes.append(("blocked", exc.detector_type))
            assert outcomes[:2] == outcomes[2:], text
        assert plain.counts == recording.counts
        assert plain.warn_counts == recording.warn_counts
        # Every scanned text recorded (twice: redact_text and scan), with
        # the configured detections and the added deny strings' matches.
        assert [text for text, _found in record.calls[::2]] == texts
        for text, found in record.calls:
            configured = recording.plan.detect_sourced(text, NO_ALLOW)
            extra = added.detect_sourced(text, NO_ALLOW) if deny is not None else []
            assert found == configured + extra


class _Approving:
    """Approved overrides that count how often they are asked."""

    def __init__(self) -> None:
        self.asked = 0

    def allows(self, detector_type: str, value: str) -> bool:
        self.asked += 1
        return True

    def unoverridable(self) -> None:
        raise AssertionError("no final block here")


def test_a_recording_redactor_asks_the_overrides_as_often_as_without() -> None:
    detectors = build_detectors(DetectionConfig())
    asked = []
    for record in (None, _Record()):
        overrides = _Approving()
        redactor = Redactor(
            detectors, InMemoryVault(), Allowlist(), modes={"EMAIL": "block"}
        ).with_overrides(overrides)
        if record is not None:
            redactor = redactor.with_recorder(record)
        # The approved value goes through as sent.
        assert redactor.redact_text(f"mail {EMAIL}") == f"mail {EMAIL}"
        asked.append(overrides.asked)
    assert asked == [1, 1]


def test_a_recording_redactor_records_what_it_scans_but_not_a_block_check() -> None:
    detectors = build_detectors(DetectionConfig())
    record = _Record()
    redactor = Redactor(
        detectors, InMemoryVault(), Allowlist(), modes={"EMAIL": "block"}
    ).with_recorder(record)
    assert redactor.blocked_type(f"mail {EMAIL}") == "EMAIL"
    assert record.calls == []  # the check ahead of a redaction records nothing
    with pytest.raises(BlockedRequest):
        redactor.redact_text(f"mail {EMAIL}")
    ((text, found),) = record.calls
    assert text == f"mail {EMAIL}" and [(d.detector_type, s) for d, s in found] == [
        ("EMAIL", DetectorSource("rule", "email"))
    ]


def test_copies_keep_the_recorder() -> None:
    record = _Record()
    base = Redactor(build_detectors(DetectionConfig()), InMemoryVault(), Allowlist())
    redactor = (
        base.with_recorder(record)
        .with_budget(10)
        .with_floors({"EMAIL": 4})
        .with_overrides(object())  # type: ignore[arg-type]
        .with_precomputed({}, base.plan)
    )
    assert redactor.redact_text(f"mail {EMAIL}") == "mail «EMAIL_005»"
    assert [text for text, _found in record.calls] == [f"mail {EMAIL}"]
    # The redactor it was copied from records nothing.
    base.redact_text(f"again {EMAIL}")
    assert len(record.calls) == 1
    # A recorder keeps the floors and the budget it was attached under.
    floored = base.with_floors({"EMAIL": 8}).with_recorder(record)
    assert floored.redact_text("x b@corp.example") == "x «EMAIL_009»"
    budgeted = base.with_budget(1).with_recorder(record)
    budgeted.redact_text("one")
    with pytest.raises(TooManyStrings):
        budgeted.redact_text("two")


def test_a_recording_redactor_takes_the_heavy_detections_from_the_table() -> None:
    from test_ner_prefetch import FakeModel

    model = FakeModel()
    plan = DetectorPlan([*build_detectors(DetectionConfig()), model])
    text, other = f"{NAME} at {EMAIL}", f"{OTHER} too"
    table = {text: plan.detect_heavy(text)}
    calls = len(model.calls)
    record = _Record()
    redactor = (
        Redactor(plan, InMemoryVault(), Allowlist()).with_precomputed(table, plan)
    ).with_recorder(record)
    plain = Redactor(plan, InMemoryVault(), Allowlist())
    assert redactor.redact_text(text) == plain.redact_text(text)
    stats = model.stats
    assert stats.prefetch_misses == 0
    assert len(model.calls) == calls + 1  # only the plain redactor ran the model
    redactor.redact_text(other)  # not in the table: inline, a counted miss
    assert stats.prefetch_misses == 1
    assert {source.kind for _text, found in record.calls for _d, source in found} == {
        "rule",
        "ner",
    }


def test_a_recorder_keeps_each_string_once_with_its_detections_in_span_order() -> None:
    recorder = ScanRecorder(_CONTEXT, (0,))
    later = (Detection(10, 14, "B", "x"), DetectorSource("deny", "deny_strings"))
    longer = (Detection(2, 9, "A", "x"), DetectorSource("rule", "r"))
    shorter = (Detection(2, 5, "C", "x"), DetectorSource("ner", "hf"))
    recorder.record("one", [later, shorter, longer])
    recorder.record("two", [])
    recorder.record("one", [])  # the same text again: kept once, as first seen
    assert recorder.strings() == (
        ScannedString(
            "one",
            (
                ScannedDetection("A", 2, 9, "rule", "r"),
                ScannedDetection("C", 2, 5, "ner", "hf"),
                ScannedDetection("B", 10, 14, "deny", "deny_strings"),
            ),
        ),
        ScannedString("two", ()),
    )
    assert "one" not in repr(recorder.strings()[0])  # text is repr=False


# --- through the app: exactly the request's scanned strings --------------------


@pytest.mark.parametrize("shape", SHAPES, ids=[shape.id for shape in SHAPES])
async def test_a_sink_receives_exactly_the_strings_the_redaction_scanned(
    monkeypatch: pytest.MonkeyPatch, shape: Shape
) -> None:
    sent: list[bytes] = []
    for with_sink in (False, True):
        with monkeypatch.context() as patch:
            sink = Sink()
            if with_sink:
                _register(patch, sink)
            install_spacy(patch, SlowNlp())
            upstream = Upstream()
            app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
            async with _client(app) as client:
                response = await _send(client, shape)
            assert response.status_code == 200, response.text
            sent.extend(upstream.bodies)
            if not with_sink:
                assert app.state.proxy.scan_capture is None
                continue
            await _delivered(app)
            state = app.state.proxy
    # The upstream got the same bytes with and without the sink.
    assert sent[0] == sent[1]
    adapter = _adapter(shape)
    collected = collect_request_strings(
        adapter,
        shape.method,
        shape.path,
        copy.deepcopy(shape.body),
        limit=LIMIT,
        mcp_exempt=frozenset({EXEMPT_SERVER}),
    )
    assert collected
    ((context, strings),) = sink.captures
    # Each distinct scanned string once, in the order first scanned — the
    # client's own text, never a placeholder.
    assert [scanned.text for scanned in strings] == list(dict.fromkeys(collected))
    assert not any("«" in scanned.text for scanned in strings)
    for scanned in strings:
        assert scanned.detections == _expected(state.redactor.plan, state.allowlist, scanned.text)
    assert {d.kind for scanned in strings for d in scanned.detections} >= {"ner"}
    assert context == ScanContext(
        surface="http",
        provider=sink.wanted[0].provider,
        adapter=adapter.name,
        method=shape.method,
        path=shape.path,
        session_id=state.config.vault.session,
        subject=None,
        identity=False,
    )
    # Off the event loop.
    assert threading.get_ident() not in sink.threads
    state.scan_capture.close(1)
    assert sink.closed == 1


async def test_a_message_names_each_detection_and_its_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = Sink()
    _register(monkeypatch, sink)
    install_spacy(monkeypatch, SlowNlp())
    app = create_app(_config(), upstream_transport=httpx.MockTransport(Upstream()))
    text = f"Mail {NAME} at {EMAIL}, cc {OTHER}."
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=OPENAI_HEADERS,
            json={"model": "m", "messages": [{"role": "user", "content": text}]},
        )
    assert response.status_code == 200
    await _delivered(app)
    ((context, strings),) = sink.captures
    assert strings == (
        ScannedString(
            text,
            (
                ScannedDetection("PERSON", 5, 13, "ner", "spacy"),
                ScannedDetection("EMAIL", 17, 38, "rule", "email"),
                ScannedDetection("PERSON", 43, 52, "ner", "spacy"),
            ),
        ),
    )
    assert context.provider == "openai" and context.surface == "http"
    status = (await _client(app).get("/__llm-redact/status")).json()["scan_capture"]
    assert status == {
        "sinks": 1,
        "queued": 0,
        "delivered_total": 1,
        "dropped_total": 0,
        "faults_total": 0,
    }
    app.state.proxy.scan_capture.close(1)


async def test_the_admitted_subject_and_a_routed_hop_are_captured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = Sink()
    router = FakeRouter({"go": [Hop("a", "http://a.example/v1/messages"), Stop()]})
    reg, _calls = install(monkeypatch, router)
    reg.resolve_license = lambda *args, **kwargs: resolved("team")  # type: ignore[method-assign]
    reg.build_access_gate = lambda config, license: FakeGate()  # type: ignore[method-assign]
    _register(monkeypatch, sink, reg=reg)
    hits: list[bytes] = []

    def answer(request: httpx.Request) -> httpx.Response:
        hits.append(request.content)
        return httpx.Response(
            200, json={"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
        )

    app = create_app(routed_config(), upstream_transport=httpx.MockTransport(answer))
    async with _client(app) as client:
        response = await client.post(
            "/v1/messages",
            headers={**ANTHROPIC_HEADERS, "x-test-key": "good", ROUTE_HEADER: "go"},
            json={
                "model": "claude-x",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": f"to {EMAIL}"}],
            },
        )
    assert response.status_code == 200, response.text
    assert hits and EMAIL.encode() not in hits[0]
    await _delivered(app)
    ((context, strings),) = sink.captures
    assert context.subject == "ada via http"
    assert [scanned.text for scanned in strings] == [f"to {EMAIL}"]
    app.state.proxy.scan_capture.close(1)


# --- a refused request is never captured ---------------------------------------


async def test_a_refused_request_is_never_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = Sink()
    _register(monkeypatch, sink)
    detection = DetectionConfig(modes=(("email", "block"),))
    providers = {**Config().providers, "azure": ProviderConfig("")}
    upstream = Upstream()
    app = create_app(
        Config(detection=detection, providers=providers),
        upstream_transport=httpx.MockTransport(upstream),
    )
    async with _client(app) as client:
        # A block: refused by the redaction itself.
        blocked = await client.post(
            "/v1/chat/completions",
            headers=OPENAI_HEADERS,
            json={"model": "m", "messages": [{"role": "user", "content": f"to {EMAIL}"}]},
        )
        # Redacted, then refused before any upstream contact (no upstream).
        unconfigured = await client.post(
            "/openai/deployments/d/chat/completions?api-version=2024-10-21",
            headers={"api-key": "k"},
            json={"messages": [{"role": "user", "content": "hello there"}]},
        )
    assert blocked.status_code == 400 and unconfigured.status_code == 502
    assert len(sink.wanted) == 2  # asked before each redaction
    assert upstream.bodies == []
    capture = app.state.proxy.scan_capture
    assert capture.status()["queued"] == 0 and capture.status()["delivered_total"] == 0
    capture.close(1)
    assert sink.captures == []


async def test_a_body_forwarded_unredacted_is_never_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = Sink()
    _register(monkeypatch, sink)
    providers = {
        **Config().providers,
        "openai": ProviderConfig("https://api.openai.com", detection=False),
    }
    app = create_app(
        Config(providers=providers), upstream_transport=httpx.MockTransport(Upstream())
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=OPENAI_HEADERS,
            json={"model": "m", "messages": [{"role": "user", "content": f"to {EMAIL}"}]},
        )
        passthrough = await client.get("/v1/unknown/thing", headers=OPENAI_HEADERS)
    assert response.status_code == 200 and passthrough.status_code == 200
    assert sink.wanted == []  # never even asked
    app.state.proxy.scan_capture.close(1)


@pytest.mark.parametrize("want", [False, None, "yes", RuntimeError(EMAIL)])
async def test_only_a_sink_that_wants_a_request_gets_it(
    monkeypatch: pytest.MonkeyPatch, want: Any, caplog: pytest.LogCaptureFixture
) -> None:
    sink, other = Sink(want=want), Sink()
    _register(monkeypatch, sink, other)
    sourced: list[str] = []
    real = DetectorPlan.detect_sourced

    def spy(self: DetectorPlan, text: str, *args: Any) -> Any:
        sourced.append(text)
        return real(self, text, *args)

    monkeypatch.setattr(DetectorPlan, "detect_sourced", spy)
    app = create_app(Config(), upstream_transport=httpx.MockTransport(Upstream()))
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with _client(app) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=OPENAI_HEADERS,
                json={"model": "m", "messages": [{"role": "user", "content": f"to {EMAIL}"}]},
            )
        assert response.status_code == 200
        await _delivered(app)
    assert sink.captures == [] and len(other.captures) == 1
    state = app.state.proxy
    faults = 1 if isinstance(want, Exception) else 0
    assert state.bookkeeping_errors["scan_sink"] == faults
    assert state.scan_capture.status()["faults_total"] == faults
    assert EMAIL not in caplog.text
    assert ("RuntimeError" in caplog.text) is bool(faults)
    # Recorded once, for the sink that wanted it.
    assert sourced == [f"to {EMAIL}"]
    state.scan_capture.close(1)


async def test_no_sink_wanting_a_request_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = Sink(want=False)
    _register(monkeypatch, sink)
    monkeypatch.setattr(Redactor, "with_recorder", _never)
    app = create_app(Config(), upstream_transport=httpx.MockTransport(Upstream()))
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=OPENAI_HEADERS,
            json={"model": "m", "messages": [{"role": "user", "content": f"to {EMAIL}"}]},
        )
    assert response.status_code == 200 and len(sink.wanted) == 1
    app.state.proxy.scan_capture.close(1)


# --- a slow or failing sink costs a request nothing -----------------------------


def _never(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("must not be reached without a sink that wants the request")


async def test_a_slow_sink_never_holds_a_request_and_a_full_queue_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = Sink()
    sink.release.clear()  # every capture blocks until released
    _register(monkeypatch, sink)
    upstream = Upstream()
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    state = app.state.proxy
    state.scan_capture = ScanCapture(
        state.scan_capture.sinks, state.bookkeeping_errors, max_pending=1
    )
    body = {"model": "m", "messages": [{"role": "user", "content": f"to {EMAIL}"}]}
    answers = []
    async with _client(app) as client:
        first = await client.post("/v1/chat/completions", headers=OPENAI_HEADERS, json=body)
        assert await asyncio.to_thread(sink.entered.wait, 5)
        # The sink is stuck in the first capture: the next requests answer
        # all the same — one waits in the queue, the rest are dropped.
        started = time.monotonic()
        for _ in range(3):
            answers.append(
                await client.post("/v1/chat/completions", headers=OPENAI_HEADERS, json=body)
            )
        elapsed = time.monotonic() - started
    assert first.status_code == 200 and all(a.status_code == 200 for a in answers)
    assert elapsed < 2.0
    assert len(upstream.bodies) == 4 and len(set(upstream.bodies)) == 1
    assert state.scan_capture.status() == {
        "sinks": 1,
        "queued": 2,
        "delivered_total": 0,
        "dropped_total": 2,
        "faults_total": 0,
    }
    sink.release.set()
    await _delivered(app)
    assert len(sink.captures) == 2
    assert state.scan_capture.status()["delivered_total"] == 2
    state.scan_capture.close(1)


async def test_a_failing_sink_changes_no_answer_and_is_counted_once_per_outage(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    failing = Sink(fail=ValueError(f"cannot store {EMAIL}"))
    _register(monkeypatch, failing)
    upstream = Upstream()
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    body = {"model": "m", "messages": [{"role": "user", "content": f"to {EMAIL}"}]}
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        async with _client(app) as client:
            for _ in range(3):
                response = await client.post(
                    "/v1/chat/completions", headers=OPENAI_HEADERS, json=body
                )
                assert response.status_code == 200
                assert response.json() == {"ok": True}
            await _delivered(app)
            failing.fail = None  # it works again
            assert (
                await client.post("/v1/chat/completions", headers=OPENAI_HEADERS, json=body)
            ).status_code == 200
            await _delivered(app)
    state = app.state.proxy
    assert state.bookkeeping_errors["scan_sink"] == 3
    status = (await _client(app).get("/__llm-redact/status")).json()
    assert status["bookkeeping_errors_total"]["scan_sink"] == 3
    assert status["scan_capture"]["faults_total"] == 3
    assert status["scan_capture"]["delivered_total"] == 1
    assert caplog.text.count("scan capture: sink Sink failed (ValueError)") == 1
    assert "scan capture: sink Sink works again" in caplog.text
    assert EMAIL not in caplog.text
    state.scan_capture.close(1)


# --- uploads and realtime frames -------------------------------------------------


FORM = {"content-type": "multipart/form-data; boundary=b"}
TEXT = f"Mail {NAME} at {EMAIL}\n".encode()


def _form(purpose: str, filename: str, content: bytes, content_type: str) -> bytes:
    return (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\n'
        + purpose.encode()
        + b'\r\n--b\r\nContent-Disposition: form-data; name="file"; filename="'
        + filename.encode()
        + b'"\r\nContent-Type: '
        + content_type.encode()
        + b"\r\n\r\n"
        + content
        + b"\r\n--b--\r\n"
    )


_LINE = json.dumps(
    {
        "custom_id": "r1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "m", "messages": [{"role": "user", "content": f"hi {EMAIL}"}]},
    }
)


@pytest.mark.parametrize(
    ("purpose", "filename", "content", "content_type", "expected"),
    [
        (
            "assistants",
            f"notes of {OTHER}.txt",
            TEXT,
            "text/plain",
            ["assistants", f"notes of {OTHER}.txt", TEXT.decode()],
        ),
        (
            "batch",
            "batch.jsonl",
            (_LINE + "\n" + _LINE + "\n").encode(),
            "application/jsonl",
            ["batch", "batch.jsonl", "r1", "/v1/chat/completions", f"hi {EMAIL}"],
        ),
    ],
    ids=["text", "batch-jsonl"],
)
async def test_an_uploads_fields_names_and_text_parts_are_captured(
    monkeypatch: pytest.MonkeyPatch,
    purpose: str,
    filename: str,
    content: bytes,
    content_type: str,
    expected: list[str],
) -> None:
    sink = Sink()
    _register(monkeypatch, sink)
    upstream = Upstream()
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            headers={**OPENAI_HEADERS, **FORM},
            content=_form(purpose, filename, content, content_type),
        )
    assert response.status_code == 200, response.text
    assert EMAIL.encode() not in upstream.bodies[0]
    await _delivered(app)
    ((context, strings),) = sink.captures
    texts = [scanned.text for scanned in strings]
    assert set(expected) <= set(texts)
    assert len(texts) == len(set(texts))
    found = {d.detector_type for scanned in strings for d in scanned.detections}
    assert "EMAIL" in found
    assert context.surface == "http" and context.path == "/v1/files"
    app.state.proxy.scan_capture.close(1)


class _Inspector:
    timeout = 5.0
    max_bytes = 1 << 20

    async def inspect(self, part: UploadPart) -> Inspection:
        return Inspection(f"Report for {OTHER}", True, "fake")

    def status(self) -> dict[str, Any]:
        return {}

    async def aclose(self) -> None:
        return None


async def test_an_inspectors_extracted_text_is_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = Sink()
    reg = Registry()
    reg.build_upload_inspector = lambda config, tier: _Inspector()  # type: ignore[method-assign]
    _register(monkeypatch, sink, reg=reg)
    app = create_app(Config(), upstream_transport=httpx.MockTransport(Upstream()))
    pdf = b"%PDF-1.7\n1 0 obj << >> endobj\n%%EOF\n"
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            headers={**OPENAI_HEADERS, **FORM},
            content=_form("user_data", "report.pdf", pdf, "application/pdf"),
        )
    assert response.status_code == 200, response.text
    await _delivered(app)
    ((_context, strings),) = sink.captures
    assert f"Report for {OTHER}" in [scanned.text for scanned in strings]
    app.state.proxy.scan_capture.close(1)


class _Converting(_Inspector):
    """A complete reading holding a value, with a display text to send in
    the file's place (convert mode)."""

    async def inspect(self, part: UploadPart) -> Inspection:
        return Inspection(
            f"Report for {OTHER}, mail {EMAIL}", True, "fake", convert_text=f"Shown: {EMAIL}"
        )


async def test_a_converted_documents_text_is_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = Sink()
    reg = Registry()
    reg.build_upload_inspector = lambda config, tier: _Converting()  # type: ignore[method-assign]
    _register(monkeypatch, sink, reg=reg)
    upstream = Upstream()
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    pdf = b"%PDF-1.7\n1 0 obj << >> endobj\n%%EOF\n"
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            headers={**OPENAI_HEADERS, **FORM},
            content=_form("assistants", "report.pdf", pdf, "application/pdf"),
        )
    assert response.status_code == 200, response.text
    # Sent as its redacted display text.
    (sent,) = upstream.bodies
    assert b"Shown: \xc2\xabEMAIL_001\xc2\xbb" in sent and EMAIL.encode() not in sent
    await _delivered(app)
    ((_context, strings),) = sink.captures
    texts = [scanned.text for scanned in strings]
    # The full reading the inspection scanned and the display text the
    # conversion redacted, both as extracted (never the placeholder).
    assert f"Report for {OTHER}, mail {EMAIL}" in texts and f"Shown: {EMAIL}" in texts
    (shown,) = [scanned for scanned in strings if scanned.text == f"Shown: {EMAIL}"]
    assert [(d.detector_type, d.start, d.end) for d in shown.detections] == [
        ("EMAIL", 7, 7 + len(EMAIL))
    ]
    app.state.proxy.scan_capture.close(1)


websockets_mod = pytest.importorskip("websockets")


class FakeWsUpstream:
    """A websockets server recording the frames it receives (no echo)."""

    def __init__(self) -> None:
        self.received: list[str | bytes] = []
        self.port = 0

    async def _handler(self, connection: Any) -> None:
        async for message in connection:
            self.received.append(message)

    @contextlib.asynccontextmanager
    async def serving(self) -> AsyncIterator[FakeWsUpstream]:
        server = await websockets_mod.serve(self._handler, "127.0.0.1", 0)
        self.port = server.sockets[0].getsockname()[1]
        try:
            yield self
        finally:
            server.close()
            await server.wait_closed()


@contextlib.contextmanager
def _serve(app: Any) -> Iterator[str]:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


FRAMES = [
    json.dumps({"type": "session.update", "session": {"instructions": f"Help {OTHER}."}}),
    json.dumps(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"Mail {EMAIL}"}],
            },
        }
    ),
    "not json at all",
]


async def _relay(app: Any, upstream: FakeWsUpstream, frames: list[str]) -> None:
    with _serve(app) as host:
        async with websockets_mod.connect(f"ws://{host}/v1/realtime?model=m") as client:
            for frame in frames:
                await client.send(frame)
            for _ in range(300):
                if len(upstream.received) == len(frames):
                    break
                await asyncio.sleep(0.01)


def _ws_config(port: int, detection: DetectionConfig | None = None) -> Config:
    providers = {**Config().providers, "openai": ProviderConfig(f"http://127.0.0.1:{port}")}
    return Config(providers=providers, detection=detection or DetectionConfig())


async def test_each_realtime_client_frame_is_a_capture_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[list[str | bytes]] = []
    for with_sink in (False, True):
        with monkeypatch.context() as patch:
            sink = Sink()
            if with_sink:
                _register(patch, sink)
            upstream = FakeWsUpstream()
            async with upstream.serving():
                app = create_app(_ws_config(upstream.port))
                await _relay(app, upstream, FRAMES)
            received.append(upstream.received)
    assert received[0] == received[1] and len(received[1]) == len(FRAMES)
    assert EMAIL not in received[1][1]
    capture = app.state.proxy.scan_capture
    assert await asyncio.to_thread(capture.drain, 5) == 0
    # The JSON frames, each its own capture; the non-JSON one is relayed
    # verbatim (key auth), never scanned.
    assert len(sink.wanted) == len(FRAMES)
    assert sink.texts() == [[f"Help {OTHER}."], [f"Mail {EMAIL}"]]
    context = sink.captures[1][0]
    assert context.surface == "websocket" and context.method == "GET"
    assert context.provider == "openai" and context.path == "/v1/realtime"
    assert context.identity is False
    (email,) = sink.captures[1][1][0].detections
    assert (email.detector_type, email.kind, email.source) == ("EMAIL", "rule", "email")
    capture.close(1)


async def test_a_blocked_realtime_frame_is_never_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = Sink()
    _register(monkeypatch, sink)
    upstream = FakeWsUpstream()
    async with upstream.serving():
        app = create_app(_ws_config(upstream.port, DetectionConfig(modes=(("email", "block"),))))
        with _serve(app) as host:
            async with websockets_mod.connect(f"ws://{host}/v1/realtime?model=m") as client:
                await client.send(FRAMES[0])
                await client.send(FRAMES[1])  # blocked: the connection closes 1008
                with pytest.raises(websockets_mod.ConnectionClosed):
                    for _ in range(50):
                        await client.recv()
    capture = app.state.proxy.scan_capture
    assert await asyncio.to_thread(capture.drain, 5) == 0
    assert sink.texts() == [[f"Help {OTHER}."]]
    capture.close(1)


# --- without a sink: nothing is recorded or queued -------------------------------


async def test_without_a_sink_nothing_is_recorded_or_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Free core: no sink is registered.
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    monkeypatch.setattr(Redactor, "with_recorder", _never)
    monkeypatch.setattr(DetectorPlan, "detect_sourced", _never)
    monkeypatch.setattr(ScanRecorder, "__init__", _never)
    monkeypatch.setattr(ScanCapture, "__init__", _never)
    install_spacy(monkeypatch, SlowNlp())
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    assert app.state.proxy.scan_capture is None
    async with _client(app) as client:
        for shape in SHAPES:
            assert (await _send(client, shape)).status_code == 200
        upload = await client.post(
            "/v1/files",
            headers={**OPENAI_HEADERS, **FORM},
            content=_form("assistants", "notes.txt", TEXT, "text/plain"),
        )
        assert upload.status_code == 200
        status = (await client.get("/__llm-redact/status")).json()["scan_capture"]
    assert status == {
        "sinks": 0,
        "queued": 0,
        "delivered_total": 0,
        "dropped_total": 0,
        "faults_total": 0,
    }
    ws_upstream = FakeWsUpstream()
    async with ws_upstream.serving():
        ws_app = create_app(_ws_config(ws_upstream.port))
        await _relay(ws_app, ws_upstream, FRAMES)
    assert len(ws_upstream.received) == len(FRAMES)


def test_the_free_default_registers_no_sink() -> None:
    assert Registry().build_scan_sinks(Config(), "free") == ()
    assert build_scan_capture((), Counter()) is None


def test_a_sink_lacking_a_member_refuses_the_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    class Half:
        def wants(self, context: ScanContext) -> bool:
            return True

        def capture(self, context: ScanContext, strings: Any) -> None:
            pass

    _register(monkeypatch, Half())
    with pytest.raises(ConfigError, match=r"scan sink cannot be used: Half has no callable close"):
        create_app(Config())


def test_status_names_registered_sinks_in_the_posture_block(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_posture({"scan_capture": {"sinks": 2, "dropped_total": 3}})
    out = capsys.readouterr().out
    assert "scan sinks: 2 registered" in out and "UNREDACTED" in out and "3 dropped" in out
    _print_posture({"scan_capture": {"sinks": 0}})
    assert "scan sinks" not in capsys.readouterr().out


# --- the capture itself: bounds, faults, shutdown --------------------------------


def _recorded(context: ScanContext, *texts: str) -> ScanRecorder:
    recorder = ScanRecorder(context, (0,))
    for text in texts:
        recorder.record(text, [])
    return recorder


def test_a_capture_past_the_character_bound_is_dropped() -> None:
    sink = Sink()
    sink.release.clear()
    capture = ScanCapture([sink], Counter(), max_chars=10)
    capture.submit(_recorded(_CONTEXT, "x" * 50))  # an empty queue takes any one
    assert sink.entered.wait(5)
    capture.submit(_recorded(_CONTEXT, "y" * 8))  # queued
    capture.submit(_recorded(_CONTEXT, "z" * 8))  # 16 > 10: dropped
    capture.submit(_recorded(_CONTEXT))  # nothing recorded: not queued at all
    assert capture.status()["dropped_total"] == 1 and capture.status()["queued"] == 2
    sink.release.set()
    assert capture.drain(5) == 0
    assert sink.texts() == [["x" * 50], ["y" * 8]]
    capture.close(1)


def test_a_full_queue_logs_once_per_episode(caplog: pytest.LogCaptureFixture) -> None:
    sink = Sink()
    sink.release.clear()
    capture = ScanCapture([sink], Counter(), max_pending=1)
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        capture.submit(_recorded(_CONTEXT, "a"))
        assert sink.entered.wait(5)
        for text in "bcd":
            capture.submit(_recorded(_CONTEXT, text))
        sink.release.set()
        assert capture.drain(5) == 0
        capture.submit(_recorded(_CONTEXT, "e"))
        assert capture.drain(5) == 0
    assert caplog.text.count("the sinks' queue is full") == 1
    assert "caught up" in caplog.text
    assert capture.status()["dropped_total"] == 2
    capture.close(1)


def test_a_closed_capture_drops_what_it_is_handed_and_closes_its_sinks_once() -> None:
    sink = Sink()
    capture = ScanCapture([sink], Counter())
    capture.submit(_recorded(_CONTEXT, "a"))
    assert capture.close(1) == 0
    assert sink.events == ["capture", "close"]
    capture.submit(_recorded(_CONTEXT, "b"))
    assert capture.status()["dropped_total"] == 1
    assert capture.close(1) == 0
    assert sink.closed == 1


def test_close_drops_what_the_bound_leaves_and_never_closes_a_sink_mid_capture(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = Sink()
    sink.release.clear()
    capture = ScanCapture([sink], Counter())
    for text in "abc":
        capture.submit(_recorded(_CONTEXT, text))
    assert sink.entered.wait(5)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        started = time.monotonic()
        assert capture.close(0.3) == 2  # b and c were never delivered
        assert time.monotonic() - started < 2.0
    assert sink.closed == 0  # still inside the first capture
    assert "2 capture(s) were not delivered at shutdown" in caplog.text
    assert "still capturing at shutdown" in caplog.text
    assert capture.status()["dropped_total"] == 2
    sink.release.set()


def test_a_sink_whose_close_fails_or_hangs_is_contained(caplog: pytest.LogCaptureFixture) -> None:
    class Failing(Sink):
        def close(self) -> None:
            raise OSError(f"https://store.example/?sig={EMAIL}")

    class Hanging(Sink):
        def close(self) -> None:
            self.release.wait(10)

    hanging = Hanging()
    capture = ScanCapture([Failing(), hanging], Counter())
    hanging.release.clear()
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        capture.close(0.3)
    assert "closing sink Failing failed (OSError)" in caplog.text
    assert "a sink did not close in time" in caplog.text
    assert EMAIL not in caplog.text
    hanging.release.set()


def test_the_worker_idles_out_and_comes_back() -> None:
    sink = Sink()
    capture = ScanCapture([sink], Counter(), idle_seconds=0.05)
    capture.submit(_recorded(_CONTEXT, "a"))
    assert capture.drain(5) == 0
    deadline = time.monotonic() + 5
    while capture._thread is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert capture._thread is None
    capture.submit(_recorded(_CONTEXT, "b"))
    assert capture.drain(5) == 0
    assert sink.texts() == [["a"], ["b"]]
    capture.close(1)


def test_a_worker_dying_hands_the_queue_to_another() -> None:
    class Dying(Sink):
        def capture(self, context: ScanContext, strings: Any) -> None:
            if strings[0].text == "die":
                raise KeyboardInterrupt  # a BaseException: the thread ends
            super().capture(context, strings)

    sink = Dying()
    capture = ScanCapture([sink], Counter())
    gate = threading.Event()
    original = capture._next

    def held() -> Any:
        gate.wait(5)
        return original()

    capture._next = held  # type: ignore[method-assign]
    capture.submit(_recorded(_CONTEXT, "die"))
    capture.submit(_recorded(_CONTEXT, "after"))
    old = threading.excepthook
    threading.excepthook = lambda args: None
    try:
        gate.set()
        deadline = time.monotonic() + 5
        while not sink.captures and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        threading.excepthook = old
    assert sink.texts() == [["after"]]
    capture.close(1)


def test_a_fault_with_no_loop_is_counted_directly() -> None:
    faults: Counter[str] = Counter()
    capture = ScanCapture([Sink(fail=RuntimeError("x"))], faults)
    capture.submit(_recorded(_CONTEXT, "a"))  # outside any event loop
    assert capture.drain(5) == 0
    assert faults["scan_sink"] == 1
    capture.close(1)


async def test_a_fault_after_its_loop_closed_is_counted_directly() -> None:
    faults: Counter[str] = Counter()
    sink = Sink(fail=RuntimeError("x"))
    sink.release.clear()
    capture = ScanCapture([sink], faults)
    loop = asyncio.new_event_loop()
    capture._loop = loop
    loop.close()
    capture._post(capture._count_fault)
    assert faults["scan_sink"] == 1
    capture.close(0.1)
    sink.release.set()


# --- shutdown: drained, then closed, within the budget ---------------------------


async def test_shutdown_drains_the_queue_then_closes_the_sinks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Slow(Sink):
        def capture(self, context: ScanContext, strings: Any) -> None:
            time.sleep(0.05)
            super().capture(context, strings)

    sink = Slow()
    _register(monkeypatch, sink)
    app = create_app(Config(), upstream_transport=httpx.MockTransport(Upstream()))
    async with app.router.lifespan_context(app), _client(app) as client:
        for index in range(5):
            response = await client.post(
                "/v1/chat/completions",
                headers=OPENAI_HEADERS,
                json={"model": "m", "messages": [{"role": "user", "content": f"#{index}"}]},
            )
            assert response.status_code == 200
    # Every capture was delivered before the sinks closed, once.
    assert sink.texts() == [[f"#{index}"] for index in range(5)]
    assert sink.events == ["capture"] * 5 + ["close"]


async def test_a_stuck_sink_never_holds_the_shutdown_past_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = Sink()
    sink.release.clear()
    _register(monkeypatch, sink)
    monkeypatch.setattr(proxy_mod, "SCAN_CLOSE_SECONDS", 0.3)
    events: list[str] = []
    app = create_app(Config(), upstream_transport=httpx.MockTransport(Upstream()))
    manager = app.state.proxy.vault_manager
    original = manager.close

    def close() -> None:
        events.append("vault.close")
        original()

    manager.close = close
    started = time.monotonic()
    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            for _ in range(2):
                response = await client.post(
                    "/v1/chat/completions",
                    headers=OPENAI_HEADERS,
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                )
                assert response.status_code == 200
        assert await asyncio.to_thread(sink.entered.wait, 5)
    assert time.monotonic() - started < 5.0
    assert events == ["vault.close"] and sink.closed == 0
    assert app.state.proxy.scan_capture.status()["dropped_total"] == 1
    sink.release.set()


def test_the_scan_drain_adds_nothing_to_the_shutdown_budget() -> None:
    # It runs beside the vault's map-write drain (asyncio.gather in the
    # lifespan), never longer: the 57 s budget the chart quotes holds.
    assert capture_mod.CLOSE_SECONDS <= MAP_DRAIN_SECONDS


def test_the_capture_sees_the_allowlist_the_redaction_applies() -> None:
    config = DetectionConfig(allowlist=(EMAIL,))
    plan = DetectorPlan(build_detectors(config))
    record = _Record()
    redactor = Redactor(plan, InMemoryVault(), build_allowlist(config)).with_recorder(record)
    assert redactor.redact_text(f"to {EMAIL}") == f"to {EMAIL}"
    assert record.calls == [(f"to {EMAIL}", [])]
