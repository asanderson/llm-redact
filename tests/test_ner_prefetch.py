"""NER off the event loop: precomputed detections must change nothing but
WHERE the models run.

A model-backed (heavy) detector may be run ahead of the redaction, on a
worker thread, and the synchronous redaction then takes its results from a
table instead of calling it. The table only ever saves work: the
differential tests redact the recall corpus, the false-positive corpus and
hand-picked samples with and without a table and compare every byte, count
and warn count; a partial table, a table computed for another plan and the
copies a request makes of its redactor are pinned alongside.
"""

import asyncio
import base64
import copy
import re
import threading
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

from llm_redact.bench.corpus import generate
from llm_redact.detection.base import Detection, Detector
from llm_redact.detection.engine import (
    Allowlist,
    DetectionConfig,
    DetectorPlan,
    FairLock,
    TypeFilteredDetector,
    build_detectors,
    heavy_lock,
    plan_for,
)
from llm_redact.detection.stats import NerStats
from llm_redact.ner_prefetch import CollectingRedactor, collect, collect_request_strings, prefetch
from llm_redact.providers import ALL_ADAPTERS, ProviderAdapter, RouteKind
from llm_redact.providers.base import prepare_route_request
from llm_redact.providers.custom import build_custom_adapters
from llm_redact.redactor import BlockedRequest, Redactor, TooManyStrings, UnredactableRequest
from llm_redact.vault import InMemoryVault
from prefetch_fixtures import EXEMPT_SERVER, SHAPES, Shape

ROOT = Path(__file__).resolve().parent.parent
ALLOW = Allowlist(exact=frozenset({"Alice Smith"}), patterns=())

_NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
# A wide span around anything holding an @: it overlaps the email rule's
# match, so overlap resolution has something to decide.
_CONTACT = re.compile(r"\S*@\S*")


class FakeModel:
    """A model-backed detector: PERSON for two capitalized words, CONTACT
    around anything holding an @ (overlapping the regex email rule), with
    its own coverage counters like an NER backend."""

    name = "fake_model"
    heavy = True

    def __init__(self) -> None:
        self.stats = NerStats()
        self.calls: list[str] = []

    def detect(self, text: str) -> list[Detection]:
        self.calls.append(text)
        self.stats.scanned_whole += 1
        found = [
            Detection(m.start(), m.end(), "PERSON", m.group(), priority=120)
            for m in _NAME.finditer(text)
        ]
        found += [
            Detection(m.start(), m.end(), "CONTACT", m.group(), priority=120)
            for m in _CONTACT.finditer(text)
            if len(m.group()) > 1
        ]
        return found


class Light:
    """A plain (not model-backed) detector: never precomputed."""

    name = "light"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, text: str) -> Iterable[Detection]:
        self.calls += 1
        index = text.find("Zeta")
        if index != -1:
            yield Detection(index, index + 4, "CODENAME", "Zeta")


def _detectors() -> tuple[list[Detector], FakeModel, FakeModel]:
    """The built-in rules plus two heavy detectors (one bare, one behind
    a type filter, as build_detectors wraps NER backends) and a light
    one."""
    bare = FakeModel()
    wrapped = FakeModel()
    detectors: list[Detector] = list(build_detectors(DetectionConfig()))
    detectors += [bare, TypeFilteredDetector(wrapped, frozenset({"EMAIL"})), Light()]
    return detectors, bare, wrapped


def _redactor(detectors: list[Detector]) -> Redactor:
    return Redactor(detectors, InMemoryVault(), ALLOW, modes={"CONTACT": "warn", "IPV4": "warn"})


def _table(plan: DetectorPlan, texts: Iterable[str]) -> dict[str, dict[int, list[Detection]]]:
    return {text: plan.detect_heavy(text) for text in dict.fromkeys(texts)}


def _samples() -> list[str]:
    return [
        "",
        "hi",
        "Jane Doe wrote to jane.doe@corp.example from 10.1.2.3",
        "Alice Smith (allowlisted) and Bob Jones",
        "mail <bob@corp.example>, key AKIAIOSFODNN7EXAMPLE, Zeta",
        "x" * 2000 + " Carol King carol@corp.example " + "y" * 2000,
    ]


def _corpus() -> list[str]:
    return [sample.text for sample in generate(seed=7, samples_per_rule=3)]


def _fp_corpus() -> list[str]:
    root = ROOT / "bench" / "fp_corpus"
    return [
        path.read_text(encoding="utf-8")
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "MANIFEST.toml"
    ]


def _outcome(redactor: Redactor, texts: list[str]) -> tuple[list[str], Counter[str], Counter[str]]:
    out = [redactor.redact_text(text) for text in texts]
    return out, Counter(redactor.counts), Counter(redactor.warn_counts)


# ---- differential: a complete table changes nothing but where models run ----


@pytest.mark.parametrize("source", ["samples", "corpus", "fp_corpus"])
def test_a_complete_table_redacts_byte_identically(source: str) -> None:
    texts = {"samples": _samples, "corpus": _corpus, "fp_corpus": _fp_corpus}[source]()
    detectors, bare, wrapped = _detectors()
    inline = _redactor(detectors)
    expected = _outcome(inline, texts)
    assert bare.stats.inline_calls == wrapped.stats.inline_calls == len(texts)

    detectors, bare, wrapped = _detectors()
    redactor = _redactor(detectors)
    table = _table(redactor.plan, texts)
    calls = len(bare.calls)
    assert _outcome(redactor.with_precomputed(table, redactor.plan), texts) == expected
    # Every heavy result came from the table: no model ran again, none
    # inline, nothing missed.
    assert len(bare.calls) == calls == len(set(texts))
    for model in (bare, wrapped):
        assert model.stats.inline_calls == 0
        assert model.stats.prefetch_misses == 0


def test_the_plan_takes_precomputed_rows_and_still_applies_the_allowlist() -> None:
    detectors, bare, _wrapped = _detectors()
    plan = DetectorPlan(detectors)
    for text in _samples() + _corpus():
        row = plan.detect_heavy(text)
        assert sorted(row) == list(plan.heavy_indices) == [len(detectors) - 3, len(detectors) - 2]
        assert plan.detect(text, ALLOW, row) == plan.detect(text, ALLOW)
    # The allowlist applies to precomputed raw detections too: the model
    # reported the allowlisted name, the plan dropped it.
    text = "Alice Smith"
    row = plan.detect_heavy(text)
    assert [d.value for d in row[plan.heavy_indices[0]]] == ["Alice Smith"]
    assert plan.detect(text, ALLOW, row) == []
    # A row is taken as given (precomputed results are trusted, not re-run).
    fake = Detection(0, 5, "PERSON", "Alice", priority=120)
    calls = len(bare.calls)
    assert plan.detect(text, ALLOW, {plan.heavy_indices[0]: [fake]}) == [fake]
    assert len(bare.calls) == calls


def _verdicts(redactor: Redactor, text: str) -> tuple[str, str | None, object]:
    """What redact_text, blocked_type and scan make of ``text``."""
    try:
        redacted = redactor.redact_text(text)
    except BlockedRequest as exc:
        redacted = f"blocked {exc.detector_type}"
    try:
        scan: object = redactor.scan(text)
    except BlockedRequest as exc:
        scan = f"blocked {exc.detector_type}"
    return redacted, redactor.blocked_type(text), scan


def test_a_complete_table_gives_identical_blocks_scans_and_block_checks() -> None:
    texts = _samples()
    detectors, bare, _wrapped = _detectors()
    modes = {"PERSON": "block", "CONTACT": "warn"}
    inline = Redactor(detectors, InMemoryVault(), ALLOW, modes=modes)
    expected = [_verdicts(inline, text) for text in texts]
    assert any(verdict[1] == "PERSON" for verdict in expected)
    inline_calls = bare.stats.inline_calls
    assert inline_calls == 3 * len(texts)

    redactor = Redactor(detectors, InMemoryVault(), ALLOW, modes=modes)
    precomputed = redactor.with_precomputed(_table(redactor.plan, texts), redactor.plan)
    assert [_verdicts(precomputed, text) for text in texts] == expected
    assert bare.stats.inline_calls == inline_calls  # nothing more inline
    assert bare.stats.prefetch_misses == 0


# ---- partial and stale tables, copies ----


def test_a_partial_table_counts_its_misses_and_redacts_identically() -> None:
    texts = _samples()
    detectors, bare, wrapped = _detectors()
    expected = _outcome(_redactor(detectors), texts)

    detectors, bare, wrapped = _detectors()
    redactor = _redactor(detectors)
    # Every other text precomputed: each of the others is a miss.
    table = _table(redactor.plan, texts[::2])
    missing = [text for text in texts if text not in table]
    assert _outcome(redactor.with_precomputed(table, redactor.plan), texts) == expected
    for model in (bare, wrapped):
        assert model.stats.prefetch_misses == len(missing)
        assert model.stats.inline_calls == len(missing)
    # The miss ran the model inline, on the missing text.
    assert bare.calls[-len(missing) :] == missing


def test_a_table_for_another_plan_is_ignored() -> None:
    texts = _samples()
    detectors, bare, _wrapped = _detectors()
    expected = _outcome(_redactor(detectors), texts)

    detectors, bare, wrapped = _detectors()
    redactor = _redactor(detectors)
    # Computed for the same detectors through ANOTHER plan (a reload
    # replaced the list meanwhile), and poisoned: were it read, the names
    # would vanish from the detections.
    other = DetectorPlan(list(detectors))
    assert other is not redactor.plan
    table = {text: {index: [] for index in other.heavy_indices} for text in texts}
    assert _outcome(redactor.with_precomputed(table, other), texts) == expected
    for model in (bare, wrapped):
        assert model.stats.prefetch_misses == len(texts)
        assert model.stats.inline_calls == len(texts)


def test_copies_keep_the_table() -> None:
    detectors, bare, _wrapped = _detectors()
    redactor = _redactor(detectors)
    texts = ["mail «EMAIL_004» and Jane Doe jane@corp.example", "Bob Jones"]
    table = _table(redactor.plan, texts)
    copy = (
        redactor.with_precomputed(table, redactor.plan)
        .with_budget(10)
        .with_floors({"PERSON": 3, "EMAIL": 4})
        .with_overrides(_NoOverrides())
    )
    assert copy.plan is redactor.plan
    assert [copy.redact_text(text) for text in texts] == [
        "mail «EMAIL_004» and «PERSON_004» «EMAIL_005»",
        "«PERSON_005»",
    ]
    assert bare.stats.inline_calls == bare.stats.prefetch_misses == 0
    # The budget copy counts its strings as before.
    small = redactor.with_precomputed(table, redactor.plan).with_budget(1)
    small.redact_text(texts[1])
    with pytest.raises(TooManyStrings):
        small.redact_text(texts[1])


class _NoOverrides:
    def allows(self, detector_type: str, value: str) -> bool:
        return False

    def unoverridable(self) -> None:
        return None


def test_without_a_table_nothing_is_looked_up_or_missed() -> None:
    detectors, bare, wrapped = _detectors()
    redactor = _redactor(detectors)
    assert redactor.redact_text("Jane Doe") == "«PERSON_001»"
    assert redactor.blocked_type("Jane Doe") is None
    for model in (bare, wrapped):
        assert model.stats.prefetch_misses == 0
        assert model.stats.inline_calls == 2


def test_the_shared_redactor_is_never_given_the_table() -> None:
    detectors, bare, _wrapped = _detectors()
    redactor = _redactor(detectors)
    copy = redactor.with_precomputed(_table(redactor.plan, ["Jane Doe"]), redactor.plan)
    assert copy is not redactor
    copy.redact_text("Jane Doe")
    assert bare.stats.inline_calls == 0
    redactor.redact_text("Jane Doe")
    assert bare.stats.inline_calls == 1
    assert bare.stats.prefetch_misses == 0


def test_a_table_keeps_the_floors_and_budget_it_was_attached_under() -> None:
    detectors, bare, _wrapped = _detectors()
    redactor = _redactor(detectors).with_budget(2).with_floors({"PERSON": 7})
    table = _table(redactor.plan, ["Jane Doe", "Bob Jones", "Carol King"])
    copy = redactor.with_precomputed(table, redactor.plan)
    assert copy.redact_text("Jane Doe") == "«PERSON_008»"
    assert copy.redact_text("Bob Jones") == "«PERSON_009»"
    with pytest.raises(TooManyStrings):
        copy.redact_text("Carol King")
    assert bare.stats.inline_calls == 0


# ---- heavy detectors in the plan ----


def test_heavy_detectors_are_the_ones_that_say_so() -> None:
    model = FakeModel()
    plan = DetectorPlan([Light(), model, TypeFilteredDetector(FakeModel(), frozenset())])
    assert plan.heavy_indices == (1, 2)
    assert DetectorPlan(build_detectors(DetectionConfig())).heavy_indices == ()

    class Truthy(FakeModel):
        heavy = 1  # type: ignore[assignment]  # only True itself marks a detector

    assert DetectorPlan([Truthy()]).heavy_indices == ()
    assert TypeFilteredDetector(model, frozenset()).heavy is True
    assert TypeFilteredDetector(Light(), frozenset()).heavy is False


def test_a_type_filter_shares_its_backends_lock_and_counters() -> None:
    model = FakeModel()
    wrapped = TypeFilteredDetector(model, frozenset({"CONTACT"}))
    assert heavy_lock(wrapped) is heavy_lock(model)
    assert heavy_lock(FakeModel()) is not heavy_lock(model)
    plan = DetectorPlan([wrapped])
    # The plan's inline run goes through the filter and counts on the
    # backend's counters.
    assert [d.detector_type for d in plan.detect("Jane Doe a@b.example", ALLOW)] == ["PERSON"]
    assert model.stats.inline_calls == 1
    # Precomputed rows hold what the filter emits.
    assert [d.detector_type for d in plan.detect_heavy("Jane Doe a@b.example")[0]] == ["PERSON"]
    assert model.stats.inline_calls == 1


def test_two_plans_over_one_model_share_its_lock() -> None:
    model = FakeModel()
    first = DetectorPlan([model])
    second = DetectorPlan([TypeFilteredDetector(model, frozenset())])
    assert first._heavy[0].lock is second._heavy[0].lock


@dataclass
class _Unhashable:
    """A heavy detector that cannot key a weak map (an eq dataclass)."""

    name: str = "unhashable"
    heavy: bool = True

    def detect(self, text: str) -> list[Detection]:
        return []


def test_an_unhashable_heavy_detector_gets_a_lock_of_its_own() -> None:
    detector = _Unhashable()
    plan = DetectorPlan([detector])
    assert plan.heavy_indices == (0,)
    lock = plan._heavy[0].lock
    assert lock.acquire(blocking=False)
    lock.release()
    # No counters: its inline calls and misses count nowhere, and break nothing.
    assert plan._heavy[0].stats is None
    assert plan.detect("x", ALLOW) == []
    plan.count_prefetch_miss()


def test_inline_runs_take_the_models_lock() -> None:
    model = FakeModel()
    plan = DetectorPlan([model])
    lock = heavy_lock(model)
    seen: list[bool] = []
    original = model.detect

    def detect(text: str) -> list[Detection]:
        seen.append(lock.locked())
        return original(text)

    model.detect = detect  # type: ignore[method-assign]
    plan.detect("Jane Doe", ALLOW)
    plan.detect_heavy("Jane Doe")
    assert seen == [True, True]
    assert not lock.locked()


def test_a_failing_model_releases_its_lock() -> None:
    class Broken(FakeModel):
        def detect(self, text: str) -> list[Detection]:
            raise RuntimeError("model fault")

    model = Broken()
    plan = DetectorPlan([model])
    for run in (lambda: plan.detect("x", ALLOW), lambda: plan.detect_heavy("x")):
        with pytest.raises(RuntimeError):
            run()
        assert not heavy_lock(model).locked()


def test_a_miss_counts_on_every_heavy_detector() -> None:
    first, second = FakeModel(), FakeModel()
    plan = DetectorPlan([Light(), first, second])
    plan.count_prefetch_miss()
    plan.count_prefetch_miss()
    assert first.stats.prefetch_misses == second.stats.prefetch_misses == 2
    assert first.stats.inline_calls == 0


def test_the_redactor_names_its_plan() -> None:
    detectors, _bare, _wrapped = _detectors()
    redactor = _redactor(detectors)
    assert redactor.plan is plan_for(detectors)
    assert redactor.with_floors({"EMAIL": 2}).plan is redactor.plan


# ---- the collecting pass scans exactly what the redaction scans ----

EXEMPT = frozenset({EXEMPT_SERVER})
LIMIT = 100_000


class Recorder(FakeModel):
    """A heavy detector recording every text the real redaction hands to
    detection (and finding names, so the real pass rewrites the body)."""


def _adapter(shape: Shape) -> ProviderAdapter:
    if shape.adapter == "CustomOpenAIAdapter":
        return build_custom_adapters(["custom:lm"])[0]
    return {cls.__name__: cls for cls in ALL_ADAPTERS}[shape.adapter]()


def test_every_adapter_has_a_shape() -> None:
    assert {cls.__name__ for cls in ALL_ADAPTERS} | {"CustomOpenAIAdapter"} == {
        shape.adapter for shape in SHAPES
    }


@pytest.mark.parametrize("shape", SHAPES, ids=[shape.id for shape in SHAPES])
def test_the_collecting_pass_scans_exactly_what_the_redaction_scans(shape: Shape) -> None:
    adapter = _adapter(shape)
    kind = adapter.matches(shape.method, shape.path)
    assert kind is not RouteKind.NONE
    body = copy.deepcopy(shape.body)
    collected = collect_request_strings(
        adapter, shape.method, shape.path, body, limit=LIMIT, mcp_exempt=EXEMPT
    )
    assert body == shape.body  # the collecting pass changes nothing

    recorder = Recorder()
    redactor = Redactor(
        [*build_detectors(DetectionConfig()), recorder], InMemoryVault(), Allowlist()
    ).with_budget(LIMIT)
    prepared = prepare_route_request(
        adapter,
        shape.method,
        shape.path,
        body,
        redactor,
        inject_note=adapter.wants_system_note(kind, shape.path),
        mcp_exempt=EXEMPT,
    )
    assert prepared != shape.body  # the real pass redacted something
    assert collected == recorder.calls  # the same texts, in the same order
    assert collected
    # An exempt MCP server's block is scanned by neither.
    assert not any("is exempt" in text for text in collected)


def test_the_collected_strings_feed_a_complete_table() -> None:
    # What the collecting pass found is all the real pass needs: with a
    # table of exactly those strings nothing runs inline or misses.
    shape = next(shape for shape in SHAPES if shape.id == "openai-fine-tuning")
    adapter = _adapter(shape)
    collected = collect_request_strings(adapter, shape.method, shape.path, shape.body, limit=LIMIT)
    assert collected is not None
    model = FakeModel()
    redactor = Redactor([*build_detectors(DetectionConfig()), model], InMemoryVault(), Allowlist())
    table = _table(redactor.plan, collected)
    prepare_route_request(
        adapter,
        shape.method,
        shape.path,
        shape.body,
        redactor.with_precomputed(table, redactor.plan).with_budget(LIMIT),
        inject_note=False,
    )
    assert model.stats.inline_calls == model.stats.prefetch_misses == 0


def test_an_over_budget_body_disables_the_prefetch() -> None:
    shape = next(shape for shape in SHAPES if shape.id == "openai-chat")
    adapter = _adapter(shape)
    assert collect_request_strings(adapter, shape.method, shape.path, shape.body, limit=3) is None
    # The real pass refuses it on its own.
    redactor = Redactor(build_detectors(DetectionConfig()), InMemoryVault(), Allowlist())
    with pytest.raises(TooManyStrings):
        prepare_route_request(
            adapter,
            shape.method,
            shape.path,
            shape.body,
            redactor.with_budget(3),
            inject_note=False,
        )


def test_an_undecodable_bedrock_blob_disables_the_prefetch() -> None:
    adapter = _adapter(next(shape for shape in SHAPES if shape.adapter == "BedrockAdapter"))
    path = "/model/anthropic.claude-x/count-tokens"
    body = {"input": {"invokeModel": {"body": base64.b64encode(b"\xff not json").decode()}}}
    assert collect_request_strings(adapter, "POST", path, body, limit=LIMIT) is None
    redactor = Redactor(build_detectors(DetectionConfig()), InMemoryVault(), Allowlist())
    with pytest.raises(UnredactableRequest):
        prepare_route_request(adapter, "POST", path, body, redactor, inject_note=False)


def test_any_failure_of_a_collecting_pass_is_contained() -> None:
    def broken(collector: CollectingRedactor) -> None:
        collector.redact_text("a")
        raise RuntimeError("anything")

    assert collect(broken, limit=10) is None
    assert collect(lambda collector: collector.redact_text("a"), limit=10) == ["a"]


def test_a_collector_and_its_copies_share_one_record() -> None:
    collector = CollectingRedactor(5)
    assert collector.with_floors({"EMAIL": 3}) is collector
    assert collector.with_overrides(_NoOverrides()) is collector
    budgeted = collector.with_budget(1)
    assert budgeted.redact_text("one") == "one"
    with pytest.raises(TooManyStrings):
        budgeted.redact_text("two")
    assert collector.scan("three") == (Counter(), False)
    assert collector.scan_text("four") == Counter()
    assert collector.blocked_type("five") is None
    assert collector.redact_json({"a": ["six", {"model": "m"}]}) == {"a": ["six", {"model": "m"}]}
    # The string past the budget was never recorded (nor would it be scanned).
    assert collector.strings == ["one", "three", "four", "five", "six"]
    # Nothing detected, counted or refused.
    assert not collector.blocks
    assert collector.counts == collector.warn_counts == Counter()
    # blocked_type charges nothing (a check ahead of the redaction); the
    # rest are charged against the request's limit (three of five so far).
    collector.charge(2)
    with pytest.raises(TooManyStrings):
        collector.charge(1)


# ---- the worker-thread prefetch ----


class SlowModel(FakeModel):
    """A model taking ``delay`` seconds on a text starting "slow" (none on
    others), recording which thread ran it and how many ran at once."""

    def __init__(self, delay: float, hold: threading.Event | None = None) -> None:
        super().__init__()
        self.delay = delay
        self.hold = hold
        self.started = threading.Event()
        self.threads: set[int] = set()
        self.slow_started = 0
        self.slow_done = 0
        self.slow_done_when_fast_ran: int | None = None
        self.active = 0
        self.peak = 0
        self._guard = threading.Lock()

    def detect(self, text: str) -> list[Detection]:
        with self._guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            self.threads.add(threading.get_ident())
            if text.startswith("slow"):
                self.slow_started += 1
                self.started.set()
                if self.hold is not None:
                    self.hold.wait(5)
                time.sleep(self.delay)
                self.slow_done += 1
            else:
                self.slow_done_when_fast_ran = self.slow_done
            return super().detect(text)
        finally:
            with self._guard:
                self.active -= 1


async def test_prefetch_detects_each_unique_string_once_off_the_loop() -> None:
    model = SlowModel(0)
    plan = DetectorPlan([*build_detectors(DetectionConfig()), model, Light()])
    strings = ["Jane Doe", "x", "Jane Doe", "Bob Jones a@b.example", "x"]
    table = await prefetch(plan, strings)
    assert list(table) == ["Jane Doe", "x", "Bob Jones a@b.example"]
    assert model.calls == list(table)  # each unique string once
    assert threading.get_ident() not in model.threads  # never the loop's thread
    assert table == {text: plan.detect_heavy(text) for text in table}
    assert model.stats.inline_calls == 0
    # Nothing to do: no strings, or no heavy detector.
    assert await prefetch(plan, []) == {}
    assert await prefetch(DetectorPlan(build_detectors(DetectionConfig())), ["x"]) == {}


async def test_an_inline_call_waits_for_at_most_one_string() -> None:
    delay = 0.2
    model = SlowModel(delay)
    plan = DetectorPlan([model])
    batch = asyncio.create_task(prefetch(plan, [f"slow {i}" for i in range(10)]))
    await asyncio.to_thread(model.started.wait, 5)  # the worker is on a string
    started = model.slow_started
    began = time.perf_counter()
    plan.detect("fast", ALLOW)  # inline, on the event loop: takes the model's lock
    waited = time.perf_counter() - began
    # The inline call ran as soon as the string the worker was on ended
    # (one more at most, had the worker just taken its next turn) — not
    # after the batch.
    assert model.slow_done_when_fast_ran is not None
    assert model.slow_done_when_fast_ran <= started + 1
    assert waited < delay * 2.5
    assert not batch.done()
    assert len(await batch) == 10
    assert model.peak == 1  # never two strings at once
    assert model.stats.inline_calls == 1


async def test_a_detector_exception_reaches_the_caller() -> None:
    class Broken(FakeModel):
        def detect(self, text: str) -> list[Detection]:
            raise RuntimeError("model fault")

    model = Broken()
    with pytest.raises(RuntimeError, match="model fault"):
        await prefetch(DetectorPlan([model]), ["x"])
    assert not heavy_lock(model).locked()
    # The next prefetch is not held up by the failed one.
    assert await prefetch(DetectorPlan([FakeModel()]), ["x"]) == {"x": {0: []}}


async def test_one_prefetch_runs_at_a_time() -> None:
    model = SlowModel(0.02)
    plan = DetectorPlan([model])
    first = [f"slow a{i}" for i in range(3)]
    second = [f"slow b{i}" for i in range(3)]
    await asyncio.gather(prefetch(plan, first), prefetch(plan, second))
    # The second batch started only once the first was done.
    assert model.calls == first + second
    assert model.peak == 1


async def test_a_cancelled_prefetch_stops_its_worker_after_the_current_string() -> None:
    hold = threading.Event()
    model = SlowModel(0, hold)
    batch = asyncio.create_task(prefetch(DetectorPlan([model]), [f"slow {i}" for i in range(5)]))
    await asyncio.to_thread(model.started.wait, 5)
    batch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch
    hold.set()  # the string the worker is on ends
    await asyncio.sleep(0.2)
    assert model.calls == ["slow 0"]


def test_each_event_loop_has_its_own_prefetch_gate() -> None:
    # An asyncio primitive belongs to one loop: two loops (a test's, a
    # second app's) each contend on their own.
    model = SlowModel(0.01)
    plan = DetectorPlan([model])

    async def two() -> None:
        await asyncio.gather(prefetch(plan, ["slow 1"]), prefetch(plan, ["slow 2"]))

    asyncio.run(two())
    asyncio.run(two())
    assert len(model.calls) == 4


# ---- the fair lock ----


def test_the_fair_lock_serves_waiters_in_order() -> None:
    lock = FairLock()
    assert lock.acquire()
    assert lock.locked()
    assert not lock.acquire(blocking=False)
    order: list[str] = []

    def waiter(name: str) -> None:
        with lock:
            order.append(name)

    threads = []
    for index, name in enumerate("abc"):
        thread = threading.Thread(target=waiter, args=(name,))
        thread.start()
        threads.append(thread)
        while lock._next != index + 2:  # wait until it holds its ticket
            time.sleep(0.001)
    lock.release()
    for thread in threads:
        thread.join(5)
    assert order == ["a", "b", "c"]
    assert not lock.locked()
    with pytest.raises(RuntimeError):
        lock.release()


def test_an_interrupted_waiter_gives_its_turn_up() -> None:
    lock = FairLock()
    lock.acquire()

    def interrupted(timeout: float | None = None) -> bool:
        raise KeyboardInterrupt

    lock._cond.wait = interrupted  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        lock.acquire()
    del lock._cond.wait
    assert lock._abandoned == {1}
    lock.release()  # passes over the abandoned ticket
    assert not lock.locked()
    assert lock.acquire(blocking=False)
    lock.release()

    # Interrupted just as its turn came: the turn passes on at once.
    lock.acquire()

    def turn_came(timeout: float | None = None) -> bool:
        lock._serving += 1  # the holder released
        raise KeyboardInterrupt

    lock._cond.wait = turn_came  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        lock.acquire()
    del lock._cond.wait
    assert not lock.locked()
    assert not lock._abandoned
