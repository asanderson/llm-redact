"""NER coverage counters (detection/stats.py): every backend counts the strings
it read and skipped and the model entities it dropped, and the proxy
publishes them as the ``/status`` block ``detection.ner``.

The fake models (ner_fakes.py) find fixed surface strings; nothing here
loads a model or reaches the network.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig
from llm_redact.detection.engine import (
    DetectionConfig,
    NerConfig,
    TypeFilteredDetector,
    build_detectors,
    ner_backend_stats,
    ner_status,
)
from llm_redact.detection.gliner_ner import GlinerDetector
from llm_redact.detection.hf_ner import HfDetector
from llm_redact.detection.labels import LabelPolicy
from llm_redact.detection.ner import NerDetector
from llm_redact.detection.presidio_ner import PresidioDetector
from llm_redact.detection.stanza_ner import StanzaDetector
from llm_redact.detection.stats import STRING_OUTCOMES, NerStats
from llm_redact.proxy import create_app
from ner_fakes import (
    FakeAnalyzer,
    FakeGliner,
    FakeHfPipe,
    FakeSpacy,
    install_gliner,
    install_presidio,
    install_spacy,
    install_stanza,
    install_transformers,
)

UPSTREAM = "https://api.openai.test"
NER_FAMILIES = (
    "llm_redact_ner_strings_total",
    "llm_redact_ner_windows_total",
    "llm_redact_ner_windows_truncated_total",
    "llm_redact_ner_labels_dropped_total",
    "llm_redact_ner_offsets_dropped_total",
)
SHORT = "hi Jane Doe"
LONG = "Jane Doe " + "x" * 60

# Each backend built directly over a fake that reports "Jane Doe" as
# `label` (scored 0.9 where the backend reads scores), max_chars = 40.
Builder = Callable[[str, tuple[str, ...]], Any]


def _spacy(label: str, entities: tuple[str, ...]) -> Any:
    return NerDetector(FakeSpacy([("Jane Doe", label, 1.0)]), frozenset(entities), 40)


def _stanza(label: str, entities: tuple[str, ...]) -> Any:
    return StanzaDetector(FakeSpacy([("Jane Doe", label, 1.0)]), frozenset(entities), 40)


def _presidio(label: str, entities: tuple[str, ...]) -> Any:
    analyzer = FakeAnalyzer([("Jane Doe", label, 0.9)], (label, "PERSON"))
    return PresidioDetector(analyzer, frozenset(entities), 40, 0.5)


def _hf(label: str, entities: tuple[str, ...]) -> Any:
    return HfDetector(FakeHfPipe([("Jane Doe", label, 0.9)]), frozenset(entities), 40, 0.5)


def _gliner(label: str, entities: tuple[str, ...]) -> Any:
    return GlinerDetector(FakeGliner([("Jane Doe", label, 0.9)]), frozenset(entities), 40, 0.5)


BUILDERS: dict[str, Builder] = {
    "spacy": _spacy,
    "stanza": _stanza,
    "presidio": _presidio,
    "hf": _hf,
    "gliner": _gliner,
}


def test_counters_start_at_zero_and_list_every_field() -> None:
    stats = NerStats()
    assert stats.as_dict() == {
        "scanned_whole": 0,
        "scanned_windowed": 0,
        "skipped_max_chars": 0,
        "windows": 0,
        "windows_truncated": 0,
        "labels_dropped": 0,
        "offsets_dropped": 0,
    }
    assert set(STRING_OUTCOMES) <= set(stats.as_dict())


@pytest.mark.parametrize("backend", sorted(BUILDERS))
def test_each_string_is_scanned_or_skipped(backend: str) -> None:
    label = "person" if backend == "gliner" else "PERSON"
    detector = BUILDERS[backend](label, ("PERSON",))
    assert [d.value for d in detector.detect(SHORT)] == ["Jane Doe"]
    assert list(detector.detect(LONG)) == []  # over max_chars: never read
    assert list(detector.detect("nothing here")) == []
    assert detector.stats.as_dict() == {
        "scanned_whole": 2,
        "scanned_windowed": 0,
        "skipped_max_chars": 1,
        "windows": 0,
        "windows_truncated": 0,
        "labels_dropped": 0,
        "offsets_dropped": 0,
    }


# Presidio is never asked for an entity whose type cannot be emitted (its
# entity list is filtered through the policy at build time), so it has no
# such drop to count.
@pytest.mark.parametrize("backend", sorted(set(BUILDERS) - {"presidio"}))
def test_a_type_outside_the_grammar_is_counted_as_dropped(backend: str, fold_raw: bool) -> None:
    # A raw request whose type cannot be a placeholder type ("3D model" ->
    # 3D_MODEL starts with a digit): the model's entity is never emitted,
    # and the drop is counted rather than silent.
    detector = BUILDERS[backend]("3D model", ("3D model",))
    assert list(detector.detect(SHORT)) == []
    assert detector.stats.labels_dropped == 1
    assert detector.stats.offsets_dropped == 0


def test_a_label_not_requested_is_not_a_drop() -> None:
    detector = _hf("ORG", ("PERSON",))
    assert list(detector.detect(SHORT)) == []
    assert detector.stats.labels_dropped == 0


def test_classify_counts_only_when_given_counters() -> None:
    policy = LabelPolicy(["3D model"], backend="hf")
    stats = NerStats()
    assert policy.classify("3D model") is None  # build-time use: never counted
    assert policy.classify("3D model", stats) is None
    assert policy.classify_gliner("3D model", stats) is None
    assert policy.classify("PER", stats) is None  # not requested: not a drop
    assert stats.labels_dropped == 2


class _Ent:
    def __init__(self, start: int, end: int, label: str) -> None:
        self.start_char = self.start = start
        self.end_char = self.end = end
        self.label_ = self.type = self.entity_type = label
        self.score = 0.9


class _Offsets:
    """Every backend's model shape, reporting fixed (start, end) pairs."""

    def __init__(self, spans: list[tuple[int | None, int | None]], label: str) -> None:
        self.spans = spans
        self.label = label

    def __call__(self, text: str) -> Any:  # spaCy, Stanza, hf
        if hasattr(self, "hf"):
            return [
                {"entity_group": self.label, "score": 0.9, "start": s, "end": e}
                for s, e in self.spans
            ]
        return type("Doc", (), {"ents": [_Ent(s, e, self.label) for s, e in self.spans]})()

    def get_supported_entities(self, language: str | None = None) -> list[str]:
        return [self.label]

    def analyze(self, text: str, **kwargs: Any) -> list[Any]:
        return [_Ent(s, e, self.label) for s, e in self.spans]  # type: ignore[arg-type]

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[Any]:
        return [{"start": s, "end": e, "label": "person", "score": 0.9} for s, e in self.spans]


@pytest.mark.parametrize("backend", ["spacy", "stanza", "presidio", "hf", "gliner"])
def test_offsets_the_text_does_not_contain_are_counted(backend: str) -> None:
    spans: list[tuple[int | None, int | None]] = [(3, 11), (5, 5), (-1, 4), (3, 99)]
    model = _Offsets(spans, "PERSON")
    entities = frozenset({"PERSON"})
    if backend == "spacy":
        detector: Any = NerDetector(model, entities, 1000)
    elif backend == "stanza":
        detector = StanzaDetector(model, entities, 1000)
    elif backend == "presidio":
        detector = PresidioDetector(model, entities, 1000, 0.5)
    elif backend == "hf":
        model.hf = True  # type: ignore[attr-defined]
        model.spans.append((None, 4))
        detector = HfDetector(model, entities, 1000, 0.5)
    else:
        detector = GlinerDetector(model, entities, 1000, 0.5)
    assert [d.value for d in detector.detect(SHORT)] == ["Jane Doe"]
    assert detector.stats.offsets_dropped == len(model.spans) - 1


def test_hf_below_threshold_is_not_an_offset_drop() -> None:
    detector = HfDetector(FakeHfPipe([("Jane Doe", "PER", 0.2)]), frozenset({"PERSON"}), 40, 0.5)
    assert list(detector.detect(SHORT)) == []
    assert detector.stats.offsets_dropped == 0


def test_gliner_without_prompts_counts_nothing() -> None:
    detector = GlinerDetector(FakeGliner([]), frozenset(), 40, 0.5)
    assert list(detector.detect(SHORT)) == []
    assert list(detector.detect(LONG)) == []
    assert detector.stats == NerStats()


def test_the_type_filter_forwards_the_backend_counters() -> None:
    inner = _hf("PER", ("PERSON",))
    wrapped = TypeFilteredDetector(inner, frozenset({"EMAIL"}))
    assert wrapped.stats is inner.stats
    assert TypeFilteredDetector(object(), frozenset()).stats is None  # type: ignore[arg-type]


def test_backend_stats_unwrap_and_skip_stand_ins() -> None:
    hf = _hf("PER", ("PERSON",))
    spacy = _spacy("PERSON", ("PERSON",))
    no_counters = _spacy("PERSON", ("PERSON",))
    del no_counters.stats
    found = ner_backend_stats([TypeFilteredDetector(hf, frozenset({"EMAIL"})), spacy, no_counters])
    assert [(name, backend) for name, backend, _stats in found] == [("hf", hf), ("spacy", spacy)]
    assert [stats for _name, _backend, stats in found] == [hf.stats, spacy.stats]


def test_status_block_names_each_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transformers(monkeypatch, FakeHfPipe([("Jane Doe", "PER", 0.9)], id2label={0: "PER"}))
    install_gliner(monkeypatch, FakeGliner([("Jane Doe", "person", 0.9)]))
    ner = NerConfig(enabled=True, backends=("hf", "gliner"), entities=("PERSON", "ZIP"))
    detectors = build_detectors(DetectionConfig(ner=ner))
    detectors[-1].detect(SHORT)
    block = ner_status(ner, detectors)
    zero = NerStats().as_dict()
    assert block == {
        "enabled": True,
        "max_chars": 20000,
        "backends": {
            "hf": {
                "model": "dslim/bert-base-NER",
                "revision": None,
                "catalog": None,
                "license": None,
                "counters": zero,
            },
            "gliner": {
                "model": "urchade/gliner_small-v2.1",
                "revision": None,
                "catalog": None,
                "license": None,
                "counters": {**zero, "scanned_whole": 1},
            },
        },
        "unmatched_entities": [],
    }
    # Without NER there is nothing to count.
    assert ner_status(NerConfig(), build_detectors(DetectionConfig())) == {
        "enabled": False,
        "max_chars": 20000,
        "backends": {},
        "unmatched_entities": [],
    }


def test_status_block_lists_unmatched_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    install_spacy(monkeypatch, FakeSpacy([], labels=("PERSON",)))
    ner = NerConfig(enabled=True, backend="spacy", entities=("PERSON", "PERSONS"))
    assert ner_status(ner, build_detectors(DetectionConfig(ner=ner)))["unmatched_entities"] == [
        "PERSONS"
    ]


# --- through the real app -----------------------------------------------------


def _upstream(seen: list[bytes]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    return httpx.MockTransport(handler)


def _config(**ner: Any) -> Config:
    return Config(
        providers={"openai": ProviderConfig(UPSTREAM)},
        inject_system_note=False,
        detection=DetectionConfig(ner=NerConfig(enabled=True, **ner)),
    )


async def _status(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/__llm-redact/status")
    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    return payload


async def test_status_counts_requests_previews_and_resets_on_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_transformers(monkeypatch, FakeHfPipe([("Jane Doe", "PER", 0.9)]))
    config = _config(backend="hf", max_chars=40)
    seen: list[bytes] = []
    app = create_app(config, upstream_transport=_upstream(seen))
    state = app.state.proxy
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
        before = await _status(client)
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1",
                "messages": [
                    {"role": "user", "content": SHORT},
                    {"role": "user", "content": LONG},
                ],
            },
        )
        assert response.status_code == 200
        after = await _status(client)
        # The short string was read (its name redacted); the long one was
        # skipped by NER (only the regex rules read it), and counted.
        sent = [message["content"] for message in json.loads(seen[0])["messages"]]
        assert sent == ["hi «PERSON_001»", LONG]
        hf = after["detection"]["ner"]["backends"]["hf"]
        assert hf["model"] == "dslim/bert-base-NER"
        assert hf["counters"]["scanned_whole"] == 1
        assert hf["counters"]["skipped_max_chars"] == 1
        assert after["detection"]["ner"]["enabled"] is True
        assert after["detection"]["ner"]["max_chars"] == 40
        assert after["detection"]["ner"]["unmatched_entities"] == []
        # The block is additive: every existing /status field is kept.
        assert after["detection"]["ner_enabled"] is True
        assert set(before["detection"]) - {"ner"} <= set(after["detection"])
        for key in ("detections_total", "warnings_total", "providers", "vault", "audit"):
            assert key in after

        # A dashboard preview runs the live detectors, so it counts too.
        state.preview(SHORT)
        counters = (await _status(client))["detection"]["ner"]["backends"]["hf"]["counters"]
        assert counters["scanned_whole"] == 2

        # A reload that keeps [detection] keeps the counters...
        state.apply_config(config)
        counters = (await _status(client))["detection"]["ner"]["backends"]["hf"]["counters"]
        assert counters["scanned_whole"] == 2
        # ...one that rebuilds the detectors starts them from zero.
        ner = dataclasses.replace(config.detection.ner, max_chars=41)
        state.apply_config(
            dataclasses.replace(config, detection=dataclasses.replace(config.detection, ner=ner))
        )
        counters = (await _status(client))["detection"]["ner"]["backends"]["hf"]["counters"]
        assert counters == NerStats().as_dict()


async def test_status_without_ner_has_an_empty_block() -> None:
    app = create_app(Config(providers={"openai": ProviderConfig(UPSTREAM)}))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
        payload = await _status(client)
    assert payload["detection"]["ner_enabled"] is False
    assert payload["detection"]["ner"] == {
        "enabled": False,
        "max_chars": 20000,
        "backends": {},
        "unmatched_entities": [],
    }


@pytest.mark.parametrize("backend", ["spacy", "stanza", "presidio", "gliner"])
def test_every_builder_hands_out_counting_detectors(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    install_spacy(monkeypatch, FakeSpacy([("Jane Doe", "PERSON", 1.0)]))
    install_stanza(monkeypatch, FakeSpacy([("Jane Doe", "PERSON", 1.0)]))
    install_presidio(monkeypatch, FakeAnalyzer([("Jane Doe", "PERSON", 0.9)], ("PERSON",)))
    install_gliner(monkeypatch, FakeGliner([("Jane Doe", "person", 0.9)]))
    ner = NerConfig(enabled=True, backend=backend)
    (detector,) = build_detectors(DetectionConfig(enabled=(), ner=ner))
    assert [d.value for d in detector.detect(SHORT)] == ["Jane Doe"]
    ((name, _backend, stats),) = ner_backend_stats([detector])
    assert name == backend
    assert stats.scanned_whole == 1


# --- /metrics -------------------------------------------------------------------


def test_metrics_render_the_ner_families_even_without_ner() -> None:
    from llm_redact.metrics import Metrics

    text = Metrics("0").render(
        detections=Counter(),
        rehydrations=Counter(),
        warnings=Counter(),
        blocked=Counter(),
        vault_entries=0,
        vault_sessions=0,
    )
    for name in NER_FAMILIES:
        assert f"# TYPE {name} counter" in text
        assert f"\n{name}{{" not in text  # no backend, no sample


def test_metrics_render_each_backend() -> None:
    from llm_redact.metrics import Metrics

    hf = NerStats(scanned_whole=4, skipped_max_chars=1, labels_dropped=2, offsets_dropped=3)
    gliner = NerStats(scanned_windowed=2, windows=9, windows_truncated=1)
    text = Metrics("0").render(
        detections=Counter(),
        rehydrations=Counter(),
        warnings=Counter(),
        blocked=Counter(),
        vault_entries=0,
        vault_sessions=0,
        ner_stats=[("hf", hf), ("gliner", gliner)],
    )
    samples = [line for line in text.splitlines() if line.startswith("llm_redact_ner_")]
    assert samples == [
        'llm_redact_ner_strings_total{backend="hf",outcome="scanned_whole"} 4',
        'llm_redact_ner_strings_total{backend="hf",outcome="scanned_windowed"} 0',
        'llm_redact_ner_strings_total{backend="hf",outcome="skipped_max_chars"} 1',
        'llm_redact_ner_strings_total{backend="gliner",outcome="scanned_whole"} 0',
        'llm_redact_ner_strings_total{backend="gliner",outcome="scanned_windowed"} 2',
        'llm_redact_ner_strings_total{backend="gliner",outcome="skipped_max_chars"} 0',
        'llm_redact_ner_windows_total{backend="hf"} 0',
        'llm_redact_ner_windows_total{backend="gliner"} 9',
        'llm_redact_ner_windows_truncated_total{backend="hf"} 0',
        'llm_redact_ner_windows_truncated_total{backend="gliner"} 1',
        'llm_redact_ner_labels_dropped_total{backend="hf"} 2',
        'llm_redact_ner_labels_dropped_total{backend="gliner"} 0',
        'llm_redact_ner_offsets_dropped_total{backend="hf"} 3',
        'llm_redact_ner_offsets_dropped_total{backend="gliner"} 0',
    ]


def test_the_ner_families_are_core_families() -> None:
    from llm_redact.metrics import CORE_METRIC_FAMILIES

    assert set(NER_FAMILIES) <= CORE_METRIC_FAMILIES


async def test_metrics_endpoint_counts_ner_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transformers(monkeypatch, FakeHfPipe([("Jane Doe", "PER", 0.9)]))
    app = create_app(_config(backend="hf", max_chars=40), upstream_transport=_upstream([]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4.1", "messages": [{"role": "user", "content": LONG}]},
        )
        assert response.status_code == 200
        text = (await client.get("/__llm-redact/metrics")).text
    assert 'llm_redact_ner_strings_total{backend="hf",outcome="skipped_max_chars"} 1' in text
    assert 'llm_redact_ner_strings_total{backend="hf",outcome="scanned_whole"} 0' in text
