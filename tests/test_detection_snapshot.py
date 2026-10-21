"""The detection snapshot and prepared builds (``DashboardHost``'s optional
``prepare_config`` / ``apply_config(prepared=)`` / ``preview_snapshot``).

The proxy's detectors, allowlist and modes are ONE immutable object
(``proxy.DetectionSnapshot``) that ``apply_config`` swaps in one assignment
(``ProxyState.detectors``/``allowlist``/``modes`` are read-only views of it:
a change is a new snapshot, never one attribute at a time): a preview on
the dashboard's worker thread reads it once, so an apply on the event loop
landing meanwhile never hands it a mix of two configurations.
``prepare_config`` runs validate_config's checks on a worker thread and KEEPS
the detection objects it built, so a save loads every NER model once — not
again, on the event loop, in the apply. A prepared build is adopted only for
an equal ``[detection]`` against the detection generation still running;
anything else is ignored (built as without it) and closed.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import re
import threading
import types
from collections.abc import Iterable, Iterator
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import llm_redact.proxy as proxy_mod
import llm_redact.vault as vault_mod
from llm_redact.config import Config, ConfigError, RoutingConfig, apply_env_overrides
from llm_redact.detection.base import Detection
from llm_redact.detection.engine import (
    CustomRule,
    DetectionConfig,
    NerConfig,
    build_modes,
)
from llm_redact.proxy import DetectionSnapshot, ProxyState, create_app

EMAIL = "jane.doe@corp.example"
TICKET = "TCK-1234"
TEXT = f"mail {EMAIL} about {TICKET} from Ada"

# Two configurations a preview could mix: the second turns EMAIL to warn and
# adds a TICKET rule. A preview reading the first's detectors with the
# second's modes would warn on the address and find no ticket.
FIRST = DetectionConfig()
SECOND = DetectionConfig(
    modes=(("email", "warn"),),
    custom_rules=(CustomRule(name="ticket", detector_type="TICKET", pattern=r"TCK-\d{4}"),),
)
SPACY = DetectionConfig(ner=NerConfig(enabled=True, backend="spacy"))
HOLD_SECONDS = 10


def _state(detection: DetectionConfig = FIRST) -> ProxyState:
    state: ProxyState = create_app(Config(detection=detection)).state.proxy
    return state


def _with(state: ProxyState, detection: DetectionConfig, **changes: Any) -> Config:
    """The running config with ``detection`` (and ``changes``): a candidate."""
    return dataclasses.replace(state.config, detection=detection, **changes)


def _spent(prepared: Any) -> bool:
    return prepared.take() is None


# --- one snapshot, one assignment ----------------------------------------------------


def test_the_attributes_read_the_snapshot() -> None:
    state = _state()
    snapshot = state.detection_snapshot
    assert state.detectors is snapshot.detectors
    assert state.allowlist is snapshot.allowlist
    assert state.modes is snapshot.modes
    assert state.overlay_builds is snapshot.overlay_builds
    # The static redactor detects through the snapshot's own plan.
    assert state.redactor.plan is snapshot.plan
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.modes = {}  # type: ignore[misc]


@pytest.mark.parametrize("name", ["detectors", "allowlist", "modes", "overlay_builds"])
def test_the_attributes_are_read_only(name: str) -> None:
    # Never assigned one at a time: a write would leave the snapshot's plan,
    # overlay builds and generation derived from objects no longer running
    # (the transitional setters that once derived a new snapshot are gone).
    state = _state()
    snapshot = state.detection_snapshot
    attribute = vars(ProxyState)[name]
    assert isinstance(attribute, property)
    assert (attribute.fset, attribute.fdel) == (None, None)
    with pytest.raises(AttributeError):
        setattr(state, name, getattr(snapshot, name))
    with pytest.raises(AttributeError):
        delattr(state, name)
    assert state.detection_snapshot is snapshot
    assert getattr(state, name) is getattr(snapshot, name)
    assert name not in vars(state)


@pytest.mark.parametrize("name", ["detectors", "allowlist", "modes"])
def test_a_new_snapshot_is_the_one_way_to_change_them(name: str) -> None:
    # What a test or a plugin that changes one of them does (llm-redact-pro's
    # tests add a detector so): a new snapshot over the running one,
    # assigned in ONE statement.
    state = _state()
    old = state.detection_snapshot
    redactor = state.redactor
    prepared = state.prepare_config(_with(state, SECOND))
    value = {
        "detectors": [*old.detectors],
        "allowlist": type(old.allowlist)(),
        "modes": {"EMAIL": "warn"},
    }[name]
    state.detection_snapshot = dataclasses.replace(old, **{name: value})
    new = state.detection_snapshot
    assert new is not old and getattr(state, name) is value
    for other in {"detectors", "allowlist", "modes"} - {name}:
        assert getattr(new, other) is getattr(old, other)
    # Derived anew from the objects it holds: plan, overlay builds, generation.
    assert new.plan.source is new.detectors and new.generation != old.generation
    assert new.overlay_builds is not old.overlay_builds and new.detection == old.detection
    assert state.overlay_builds is new.overlay_builds
    # A reader holding the old reference keeps it whole; the static redactor
    # keeps what it was built over.
    assert old.plan.source is old.detectors and state.redactor is redactor
    # A build prepared against the replaced generation is no longer adopted.
    builds: list[DetectionConfig] = []
    with pytest.MonkeyPatch.context() as patch:
        real = proxy_mod._build_detection
        patch.setattr(
            proxy_mod,
            "_build_detection",
            lambda detection, **kwargs: builds.append(detection) or real(detection, **kwargs),
        )
        state.apply_config(_with(state, SECOND), prepared=prepared)
    assert builds == [SECOND] and _spent(prepared)


def test_a_preview_reads_a_swapped_snapshot_the_static_redactor_after_an_apply() -> None:
    # A swap reaches every reader of the snapshot — the preview here — but
    # not the static session's own redactor, which static-mode requests
    # redact with: it keeps the objects it was built over until an apply
    # (any apply, one that changes nothing included) builds it anew.
    state = _state()
    snapshot = state.detection_snapshot

    def static(text: str) -> str:
        context = state.context_for(None, "POST", "/v1/chat/completions", {})
        assert context.redactor is state.redactor
        return context.redactor.redact_text(text)

    state.detection_snapshot = dataclasses.replace(
        snapshot, detectors=[*snapshot.detectors, _Holding(released=True)]
    )
    assert state.preview("from Ada")["detections"] == {"PERSON": 1}
    assert static("from Ada") == "from Ada"
    state.detection_snapshot = dataclasses.replace(
        state.detection_snapshot, modes={"PERSON": "warn"}
    )
    answer = state.preview("from Ada")
    assert answer["warnings"] == {"PERSON": 1} and answer["detections"] == {}
    state.detection_snapshot = dataclasses.replace(state.detection_snapshot, modes={})
    assert static("from Ada") == "from Ada"
    assert state.apply_config(state.config) == []
    assert state.redactor.plan is state.detection_snapshot.plan
    assert static("from Ada") == "from «PERSON_001»"


def test_an_apply_swaps_the_whole_snapshot_or_keeps_it() -> None:
    state = _state()
    first = state.detection_snapshot
    # Another section changed, [detection] did not: the same snapshot.
    state.apply_config(dataclasses.replace(state.config, inject_system_note=False))
    assert state.detection_snapshot is first
    state.apply_config(_with(state, SECOND))
    second = state.detection_snapshot
    assert second is not first and second.generation != first.generation
    assert second.detection == SECOND and state.config.detection == SECOND
    assert state.modes == {"EMAIL": "warn"} and state.redactor.plan is second.plan
    # Every cached overlay build dies with the objects it extended.
    assert second.overlay_builds is not first.overlay_builds


def test_replace_derives_a_new_plan_overlay_builds_and_generation() -> None:
    snapshot = _state().detection_snapshot
    same = dataclasses.replace(snapshot)
    assert same.detectors is snapshot.detectors and same.plan is snapshot.plan  # plan_for
    assert same.overlay_builds is not snapshot.overlay_builds
    assert same.generation != snapshot.generation
    extra = dataclasses.replace(snapshot, detectors=[*snapshot.detectors])
    assert extra.plan is not snapshot.plan and extra.plan.source is extra.detectors


# --- a preview never mixes two configurations ---------------------------------------


class _Holding:
    """A model-backed detector that holds the text until released: a
    preview reading it is past every read of the detection objects."""

    name = "holding"
    heavy = True

    def __init__(self, *, released: bool = False) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        if released:
            self.release.set()

    def detect(self, text: str) -> Iterable[Detection]:
        self.entered.set()
        assert self.release.wait(HOLD_SECONDS)
        start = text.find("Ada")
        if start >= 0:
            yield Detection(start, start + 3, "PERSON", "Ada", priority=120)


@pytest.mark.parametrize("hold", ["between_reads", "while_detecting"])
def test_a_preview_overlapped_by_an_apply_describes_one_configuration(
    monkeypatch: pytest.MonkeyPatch, hold: str
) -> None:
    state = _state()
    holding = _Holding()
    if hold == "while_detecting":
        # A model-backed detector reading the text when the apply lands.
        snapshot = state.detection_snapshot
        state.detection_snapshot = dataclasses.replace(
            snapshot, detectors=[*snapshot.detectors, holding]
        )
    else:
        # Held right where the preview used to read the detectors and then,
        # one attribute at a time, the allowlist and the modes: building its
        # throwaway vault, between those reads.
        real = vault_mod.InMemoryVault

        def held(*args: Any, **kwargs: Any) -> Any:
            holding.entered.set()
            assert holding.release.wait(HOLD_SECONDS)
            return real(*args, **kwargs)

        monkeypatch.setattr(vault_mod, "InMemoryVault", held)
        holding.release.set()
        before = state.preview(TEXT)  # the first configuration's answer
        holding.release.clear()
        holding.entered.clear()
    answers: list[dict[str, Any]] = []
    worker = threading.Thread(target=lambda: answers.append(state.preview(TEXT)))
    worker.start()
    try:
        assert holding.entered.wait(HOLD_SECONDS)
        state.apply_config(_with(state, SECOND))  # on this thread, meanwhile
    finally:
        holding.release.set()
        worker.join(HOLD_SECONDS)
    assert not worker.is_alive() and len(answers) == 1
    (answer,) = answers
    if hold == "while_detecting":
        assert answer["detections"] == {"EMAIL": 1, "PERSON": 1}
    else:
        assert answer == before
    # The first configuration whole: the address redacted (not warned on), no
    # ticket rule — never the first's detectors with the second's modes.
    assert answer["detections"].get("EMAIL") == 1 and answer["warnings"] == {}
    assert EMAIL not in answer["redacted"] and TICKET in answer["redacted"]
    # The next preview describes the second configuration whole.
    monkeypatch.undo()
    after = state.preview(TEXT)
    assert after["warnings"] == {"EMAIL": 1} and after["detections"] == {"TICKET": 1}
    assert EMAIL in after["redacted"] and TICKET not in after["redacted"]


def test_the_host_advertises_the_snapshot_preview() -> None:
    assert getattr(_state(), "preview_snapshot", False) is True


# --- prepared builds: every NER model loaded once per save -------------------------


class _CountingSpacy:
    """``spacy`` whose ``load`` counts model loads and the threads they ran on."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ner_fakes import FakeSpacy

        self.threads: list[str] = []
        nlp = FakeSpacy([("Ada", "PERSON", 0.99)])
        module = types.ModuleType("spacy")

        def load(name: str, disable: Any = ()) -> Any:
            self.threads.append(threading.current_thread().name)
            return nlp

        module.load = load  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "spacy", module)

    @property
    def loads(self) -> int:
        return len(self.threads)


async def test_prepare_off_the_loop_then_apply_loads_each_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spacy = _CountingSpacy(monkeypatch)
    state = _state()
    candidate = _with(state, SPACY)
    prepared = await asyncio.to_thread(state.prepare_config, candidate)
    assert spacy.loads == 1
    assert spacy.threads[0] != threading.current_thread().name  # off the loop
    assert state.config.detection == FIRST  # prepared: nothing applied yet
    state.apply_config(apply_env_overrides(candidate), prepared=prepared)
    assert spacy.loads == 1  # adopted: not loaded again on the loop
    assert state.config.detection == SPACY and _spent(prepared)
    assert state.preview("from Ada")["detections"] == {"PERSON": 1}
    assert state.redactor.redact_text("from Ada") == "from «PERSON_001»"
    # Without a prepared build the apply loads the model itself, as before.
    state.apply_config(_with(state, FIRST))
    state.apply_config(_with(state, SPACY))
    assert spacy.loads == 2


def test_validate_config_is_prepare_then_close(monkeypatch: pytest.MonkeyPatch) -> None:
    spacy = _CountingSpacy(monkeypatch)
    state = _state()
    closed: list[Any] = []
    real = proxy_mod._PreparedConfig.close
    monkeypatch.setattr(
        proxy_mod._PreparedConfig, "close", lambda self: (closed.append(self), real(self))
    )
    assert state.validate_config(_with(state, SPACY)) is None
    assert spacy.loads == 1 and len(closed) == 1 and _spent(closed[0])
    # An unchanged [detection]: nothing built, as before.
    state.validate_config(state.config)
    assert spacy.loads == 1 and len(closed) == 2


def _bad_candidates(state: ProxyState) -> Iterator[Config]:
    yield _with(
        state, DetectionConfig(custom_rules=(CustomRule(name="x", detector_type="X", pattern="("),))
    )
    yield _with(state, DetectionConfig(enabled=("no_such_rule",)))
    yield _with(state, DetectionConfig(modes=(("no_such_rule", "warn"),)))
    yield _with(
        state,
        DetectionConfig(
            custom_rules=(
                CustomRule(name="x", detector_type="X", pattern="x+", validator="no_such"),
            )
        ),
    )
    # The Free registry refuses an enabled [routing] naming the package.
    yield dataclasses.replace(state.config, routing=RoutingConfig(enabled=True))
    # A missing NER extra (no spacy installed in the base suite).
    yield _with(state, SPACY)


def _outcome(call: Any) -> tuple[type[BaseException], str] | None:
    try:
        call()
    except (ValueError, TypeError, re.error, ImportError) as exc:
        return type(exc), str(exc)
    return None


def test_prepare_raises_exactly_what_validate_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "spacy", None)  # import spacy -> ImportError
    state = _state()
    before = state.detection_snapshot
    for candidate in _bad_candidates(state):
        validated = _outcome(lambda c=candidate: state.validate_config(c))
        prepared = _outcome(lambda c=candidate: state.prepare_config(c))
        assert validated is not None and validated == prepared, candidate
    assert issubclass(
        _outcome(lambda: state.prepare_config(_with(state, SPACY)))[0],  # type: ignore[index]
        ConfigError,
    )
    assert state.detection_snapshot is before
    # A good candidate: validate answers None, prepare a closable build.
    assert state.validate_config(_with(state, SECOND)) is None
    prepared_ok = state.prepare_config(_with(state, SECOND))
    prepared_ok.close()
    prepared_ok.close()  # idempotent
    assert _spent(prepared_ok)


def _contents(value: Any) -> Any:
    """A comparable copy of a mutable container (identity says the rest)."""
    if isinstance(value, (dict, set, list, frozenset, tuple)) or hasattr(value, "maxlen"):
        return copy.copy(value)
    return None


def test_prepare_changes_nothing_and_runs_without_an_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spacy = _CountingSpacy(monkeypatch)
    state = _state()
    attributes = {name: (value, _contents(value)) for name, value in vars(state).items()}
    answers: list[Any] = []
    failures: list[BaseException] = []

    def run() -> None:
        # A plain thread: no running event loop, nothing to reach.
        try:
            answers.append(state.prepare_config(_with(state, SPACY)))
        except BaseException as exc:  # noqa: BLE001 - reported below
            failures.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(HOLD_SECONDS)
    assert failures == [] and len(answers) == 1 and spacy.loads == 1
    assert vars(state).keys() == attributes.keys()
    for name, (value, contents) in attributes.items():
        assert getattr(state, name) is value, name
        assert _contents(value) == contents, name
    answers[0].close()


# --- a stale prepared build is ignored and closed ----------------------------------


def test_an_apply_between_prepare_and_apply_makes_it_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spacy = _CountingSpacy(monkeypatch)
    state = _state()
    prepared = state.prepare_config(_with(state, SPACY))
    state.apply_config(_with(state, SECOND))  # a reload changing [detection]
    state.apply_config(_with(state, SPACY), prepared=prepared)
    assert spacy.loads == 2  # built again: the generation it was prepared against is gone
    assert _spent(prepared) and state.config.detection == SPACY


def test_an_apply_keeping_the_detection_keeps_a_prepared_build_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spacy = _CountingSpacy(monkeypatch)
    state = _state()
    prepared = state.prepare_config(_with(state, SPACY))
    state.apply_config(dataclasses.replace(state.config, inject_system_note=False))
    state.apply_config(_with(state, SPACY), prepared=prepared)
    assert spacy.loads == 1 and _spent(prepared)


def test_a_prepared_build_for_another_detection_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spacy = _CountingSpacy(monkeypatch)
    state = _state()
    prepared = state.prepare_config(_with(state, SPACY))
    state.apply_config(_with(state, SECOND), prepared=prepared)
    assert state.config.detection == SECOND and state.modes == {"EMAIL": "warn"}
    assert spacy.loads == 1 and _spent(prepared)
    # Prepared for the running [detection] (nothing built), applied with an
    # unchanged one: the live snapshot is kept.
    live = state.detection_snapshot
    unchanged = state.prepare_config(_with(state, SECOND))
    state.apply_config(_with(state, SECOND, inject_system_note=False), prepared=unchanged)
    assert state.detection_snapshot is live and _spent(unchanged)
    # A stale build applied with the running [detection] is dropped too.
    stale = state.prepare_config(_with(state, SPACY))
    state.apply_config(_with(state, SECOND), prepared=stale)
    assert state.detection_snapshot is live and _spent(stale)


# A [detection] key an apply keeps from the running config (a stand-in for
# the restart-only keys inside [detection], such as where the llm backend
# sends scanned text): the apply builds SECOND with the running modes.
PINNED = dataclasses.replace(SECOND, modes=FIRST.modes)


def _keeping_the_modes(monkeypatch: pytest.MonkeyPatch) -> list[DetectionConfig]:
    """Make ``apply_config`` keep the running ``[detection.modes]`` (its
    effective config's ``[detection]`` then differs from the one it was
    handed) and count the detection builds; answers the build list."""
    real_kept = proxy_mod._kept_restart_only
    real_build = proxy_mod._build_detection
    builds: list[DetectionConfig] = []

    def kept(fresh: Config, running: Config) -> Config:
        effective = real_kept(fresh, running)
        detection = dataclasses.replace(effective.detection, modes=running.detection.modes)
        return dataclasses.replace(effective, detection=detection)

    def counting(detection: DetectionConfig, **kwargs: Any) -> DetectionSnapshot:
        builds.append(detection)
        return real_build(detection, **kwargs)

    monkeypatch.setattr(proxy_mod, "_kept_restart_only", kept)
    monkeypatch.setattr(proxy_mod, "_build_detection", counting)
    return builds


def test_a_build_prepared_for_a_kept_key_is_never_run_under_the_kept_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prepared for the [detection] the apply is handed, which asks for a key
    # the apply keeps from the running config: adopting it would run that
    # key's new value under a config that says the old one is live.
    state = _state()
    builds = _keeping_the_modes(monkeypatch)
    prepared = state.prepare_config(_with(state, SECOND))
    assert builds == [SECOND]
    state.apply_config(_with(state, SECOND), prepared=prepared)
    assert builds == [SECOND, PINNED] and _spent(prepared)  # built for what runs
    assert state.config.detection == state.detection_snapshot.detection == PINNED
    assert (
        state.modes == build_modes(PINNED) and state.redactor.plan is state.detection_snapshot.plan
    )
    answer = state.preview(TEXT)  # the kept modes: the address redacted, not warned on
    assert answer["warnings"] == {} and answer["detections"] == {"EMAIL": 1, "TICKET": 1}


def test_a_build_prepared_for_what_the_apply_builds_is_adopted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prepared for the [detection] with the kept key's running value (as a
    # prepare that keeps the same keys makes it), then applied with the one
    # the editor saved: adopted — compared with what the apply builds, not
    # with what it was handed, so it is not built a second time.
    state = _state()
    builds = _keeping_the_modes(monkeypatch)
    prepared = state.prepare_config(_with(state, PINNED))
    state.apply_config(_with(state, SECOND), prepared=prepared)
    assert builds == [PINNED] and _spent(prepared)
    assert state.config.detection == state.detection_snapshot.detection == PINNED


def test_closed_foreign_and_other_proxies_builds_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spacy = _CountingSpacy(monkeypatch)
    state, other = _state(), _state()
    closed = state.prepare_config(_with(state, SPACY))
    closed.close()
    state.apply_config(_with(state, SPACY), prepared=closed)
    assert spacy.loads == 2
    # Another proxy's build (against its own running generation).
    state.apply_config(_with(state, FIRST))
    theirs = other.prepare_config(_with(other, SPACY))
    state.apply_config(_with(state, SPACY), prepared=theirs)
    assert spacy.loads == 4 and _spent(theirs)

    class Foreign:
        """Not a build this proxy made: never called into."""

        def close(self) -> None:
            raise AssertionError("closed a foreign object")

    state.apply_config(_with(state, FIRST), prepared=Foreign())
    assert state.config.detection == FIRST


def test_an_apply_that_fails_spends_the_prepared_and_changes_nothing() -> None:
    state = _state()
    before = (state.config, state.detection_snapshot)
    prepared = state.prepare_config(_with(state, SECOND))
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        # The router build refuses after the detection objects were taken.
        state.apply_config(
            _with(state, SECOND, routing=RoutingConfig(enabled=True)), prepared=prepared
        )
    assert (state.config, state.detection_snapshot) == before and _spent(prepared)


# --- the adoption rule, over random saves ------------------------------------------

DETECTIONS = (FIRST, SECOND, DetectionConfig(enabled=("email",)))

# One save as the dashboard makes it, raced: a build prepared for one
# [detection]; meanwhile nothing, an apply of another section, or a reload
# applying a [detection] (index); closed or not; then applied with a
# [detection] — the prepared one, or another.
_SAVE = st.tuples(
    st.integers(0, len(DETECTIONS) - 1),
    st.one_of(st.none(), st.just("other"), st.integers(0, len(DETECTIONS) - 1)),
    st.booleans(),
    st.integers(0, len(DETECTIONS) - 1),
)


@settings(deadline=None, max_examples=100)
@given(saves=st.lists(_SAVE, max_size=6))
def test_a_prepared_build_is_adopted_exactly_when_still_valid(
    saves: list[tuple[int, Any, bool, int]],
) -> None:
    state = _state()
    builds: list[DetectionConfig] = []
    real = proxy_mod._build_detection

    def counting(detection: DetectionConfig, **kwargs: Any) -> DetectionSnapshot:
        builds.append(detection)
        return real(detection, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(proxy_mod, "_build_detection", counting)
        for prepared_for, meanwhile, closed, applied in saves:
            candidate, running = DETECTIONS[prepared_for], state.config.detection
            builds.clear()
            prepared = state.prepare_config(_with(state, candidate))
            assert builds == ([candidate] if candidate != running else [])
            # The generation stays the one prepared against unless an apply
            # changed [detection] meanwhile.
            same_generation = True
            if meanwhile == "other":
                state.apply_config(
                    dataclasses.replace(
                        state.config, inject_system_note=not state.config.inject_system_note
                    )
                )
            elif meanwhile is not None:
                same_generation = DETECTIONS[meanwhile] == state.config.detection
                state.apply_config(_with(state, DETECTIONS[meanwhile]))
            if closed:
                prepared.close()
            detection, running = DETECTIONS[applied], state.config.detection
            # Adopted only unclosed, against the generation still running,
            # for the [detection] applied — which then differs from the
            # running one, so the prepare built it.
            adopt = (
                not closed and same_generation and candidate == detection and detection != running
            )
            builds.clear()
            state.apply_config(_with(state, detection), prepared=prepared)
            if detection == running:
                assert builds == []
            else:
                assert builds == ([] if adopt else [detection])
            assert _spent(prepared)
            # Whatever happened, the live objects are the running config's.
            snapshot = state.detection_snapshot
            assert snapshot.detection == state.config.detection == detection
            assert snapshot.modes == build_modes(detection)
            assert state.redactor.plan is snapshot.plan
