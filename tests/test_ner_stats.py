"""NER coverage counters (detection/stats.py): every backend counts the strings
it read and skipped and the model entities it dropped, and the proxy
publishes them as the ``/status`` block ``detection.ner``.

The fake models (ner_fakes.py) find fixed surface strings; nothing here
loads a model or reaches the network.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
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

# The hf backend's default model since 1.13.0, at its catalog pin.
HF_DEFAULT = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
HF_DEFAULT_PIN = "a2360d3f42526fc660ac3b2b2301e1c2d94eba61"
UPSTREAM = "https://api.openai.test"
NER_FAMILIES = (
    "llm_redact_ner_strings_total",
    "llm_redact_ner_windows_total",
    "llm_redact_ner_windows_truncated_total",
    "llm_redact_ner_labels_dropped_total",
    "llm_redact_ner_offsets_dropped_total",
    "llm_redact_ner_ungrounded_total",
    "llm_redact_ner_errors_total",
    "llm_redact_ner_breaker_opens_total",
    "llm_redact_ner_breaker_short_circuits_total",
    "llm_redact_ner_inline_calls_total",
    "llm_redact_ner_prefetch_misses_total",
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
        "ungrounded": 0,
        "errors": 0,
        "breaker_opens": 0,
        "breaker_short_circuits": 0,
        "inline_calls": 0,
        "prefetch_misses": 0,
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
        "ungrounded": 0,
        "errors": 0,
        "breaker_opens": 0,
        "breaker_short_circuits": 0,
        "inline_calls": 0,
        "prefetch_misses": 0,
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
                "model": HF_DEFAULT,
                "source": "hub",
                "model_id": HF_DEFAULT,
                "revision": HF_DEFAULT_PIN,
                "pinned": True,
                "catalog": "caution",
                "license": "Apache-2.0",
                "score_threshold": 0.5,
                "score_threshold_source": "default",
                "counters": zero,
            },
            "gliner": {
                "model": "urchade/gliner_small-v2.1",
                "source": "hub",
                "model_id": "urchade/gliner_small-v2.1",
                "revision": "4e091416cf7c3481db542c2a3d26156916f3a47f",
                "pinned": True,
                "catalog": "vetted",
                "license": "Apache-2.0",
                "score_threshold": 0.5,
                "score_threshold_source": "default",
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
        assert hf["model"] == HF_DEFAULT
        assert (hf["revision"], hf["catalog"], hf["license"], hf["pinned"]) == (
            HF_DEFAULT_PIN,
            "caution",
            "Apache-2.0",
            True,
        )
        assert hf["counters"]["scanned_whole"] == 1
        assert hf["counters"]["skipped_max_chars"] == 1
        # The request's strings were detected ahead of the redaction (the NER
        # prefetch): nothing ran inline, nothing was missed.
        assert hf["counters"]["inline_calls"] == hf["counters"]["prefetch_misses"] == 0
        assert after["detection"]["ner"]["enabled"] is True
        assert after["detection"]["ner"]["max_chars"] == 40
        assert after["detection"]["ner"]["unmatched_entities"] == []
        # The block is additive: every existing /status field is kept.
        assert after["detection"]["ner_enabled"] is True
        assert set(before["detection"]) - {"ner"} <= set(after["detection"])
        for key in ("detections_total", "warnings_total", "providers", "vault", "audit"):
            assert key in after

        # A dashboard preview runs the live detectors, so it counts too —
        # run, as llm-redact-pro runs it, on a worker thread of its own: it
        # has no prefetched results, so its string is an inline call (off
        # the event loop) and no prefetch miss.
        await asyncio.to_thread(state.preview, SHORT)
        counters = (await _status(client))["detection"]["ner"]["backends"]["hf"]["counters"]
        assert counters["scanned_whole"] == 2
        assert (counters["inline_calls"], counters["prefetch_misses"]) == (1, 0)

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

    hf = NerStats(
        scanned_whole=4, skipped_max_chars=1, labels_dropped=2, offsets_dropped=3, inline_calls=5
    )
    gliner = NerStats(scanned_windowed=2, windows=9, windows_truncated=1, prefetch_misses=6)
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
        'llm_redact_ner_ungrounded_total{backend="hf"} 0',
        'llm_redact_ner_ungrounded_total{backend="gliner"} 0',
        'llm_redact_ner_errors_total{backend="hf"} 0',
        'llm_redact_ner_errors_total{backend="gliner"} 0',
        'llm_redact_ner_breaker_opens_total{backend="hf"} 0',
        'llm_redact_ner_breaker_opens_total{backend="gliner"} 0',
        'llm_redact_ner_breaker_short_circuits_total{backend="hf"} 0',
        'llm_redact_ner_breaker_short_circuits_total{backend="gliner"} 0',
        'llm_redact_ner_inline_calls_total{backend="hf"} 5',
        'llm_redact_ner_inline_calls_total{backend="gliner"} 0',
        'llm_redact_ner_prefetch_misses_total{backend="hf"} 0',
        'llm_redact_ner_prefetch_misses_total{backend="gliner"} 6',
    ]


REPO = Path(__file__).resolve().parent.parent


def _table_row(doc: str, start: str) -> str:
    """The one row of a Markdown table in ``docs/<doc>`` starting ``start``."""
    text = (REPO / "docs" / doc).read_text(encoding="utf-8")
    (row,) = [line for line in text.splitlines() if line.startswith(start)]
    return row


def _list_item(doc: str, start: str) -> str:
    """The one list item in ``docs/<doc>`` starting ``start``, its lines
    joined (up to the next item or the end of the list)."""
    lines = (REPO / "docs" / doc).read_text(encoding="utf-8").splitlines()
    (first,) = [i for i, line in enumerate(lines) if line.startswith(start)]
    item = [lines[first]]
    for line in lines[first + 1 :]:
        if not line.startswith("  "):
            break
        item.append(line)
    return " ".join(" ".join(item).split())


def test_every_definition_of_the_inline_counter_counts_a_preview() -> None:
    # A redaction preview has no prefetched results, so each string it
    # hands an NER backend is an inline call — on the event loop when the
    # host calls it there, on a worker thread of its own in llm-redact-pro's
    # dashboard (the status test above runs it so). Every place defining
    # llm_redact_ner_inline_calls_total says both: the /metrics HELP text,
    # the metrics reference, the NER counter table, the dashboard panel and
    # the metrics reference's account of that panel (the first three once
    # said every inline call held up the event loop, beside a panel saying
    # a preview counts off it). And which dashboard
    # runs it where: llm-redact-pro 0.17.1 — which its floor dependency
    # still installs beside this core — runs the preview on the event loop,
    # only its later releases on a worker thread (a definition saying "its
    # own worker thread" alone read a preview holding up every request as
    # harmless).
    from llm_redact.metrics import Metrics

    text = Metrics("0").render(
        detections=Counter(),
        rehydrations=Counter(),
        warnings=Counter(),
        blocked=Counter(),
        vault_entries=0,
        vault_sessions=0,
        ner_stats=[("hf", NerStats())],
    )
    (help_line,) = [
        line
        for line in text.splitlines()
        if line.startswith("# HELP llm_redact_ner_inline_calls_total ")
    ]
    dashboard = json.loads((REPO / "deploy" / "grafana-dashboard.json").read_text("utf-8"))
    (panel,) = [
        p
        for p in dashboard["panels"]
        if "llm_redact_ner_inline_calls_total" in json.dumps(p.get("targets", []))
    ]
    definitions = {
        "HELP": help_line,
        "observability.md": _table_row(
            "observability.md", "| `llm_redact_ner_inline_calls_total` |"
        ),
        "detection.md": _table_row("detection.md", "| `inline_calls` |"),
        "dashboard": panel["description"],
        "observability.md NER coverage row": _list_item(
            "observability.md", "- strings by outcome;"
        ),
    }
    for where, definition in definitions.items():
        for phrase in ("by the redaction itself", "request's path", "event loop", "preview"):
            assert phrase in definition, (where, phrase)
        clause = definition.split("preview", 1)[1]
        for phrase in ("worker thread", "event loop", "after 0.17.1", "0.17.1 and earlier"):
            assert phrase in clause, (where, phrase)
        # And what is counted: every call attempted inline (DetectorPlan's
        # _inline counts in a finally), one that raised — the llm backend's
        # DetectorUnavailable, any other fault — like one that returned
        # (tests/test_ner_prefetch.py). The counter once skipped every call
        # that raised, and no definition said which it counted.
        assert "every call attempted, whether it returned or raised" in definition, where


def test_every_definition_of_a_prefetch_miss_says_it_is_also_inline() -> None:
    # A miss is counted only where the backend ran on the string
    # (DetectorPlan._run, as each heavy detector starts the call), so every
    # miss is an inline call too: a backend listed after a detector that
    # raised on the string (the llm backend's on_error = "block" failure,
    # any other fault) never runs on it and counts neither
    # (tests/test_ner_prefetch.py). It once counted a miss on every heavy
    # detector before any of them ran, and every definition carried that
    # exception; none may now.
    from llm_redact.detection.engine import DetectorPlan
    from llm_redact.metrics import Metrics

    text = Metrics("0").render(
        detections=Counter(),
        rehydrations=Counter(),
        warnings=Counter(),
        blocked=Counter(),
        vault_entries=0,
        vault_sessions=0,
        ner_stats=[("hf", NerStats())],
    )
    (help_line,) = [
        line
        for line in text.splitlines()
        if line.startswith("# HELP llm_redact_ner_prefetch_misses_total ")
    ]
    dashboard = json.loads((REPO / "deploy" / "grafana-dashboard.json").read_text("utf-8"))
    (panel,) = [
        p
        for p in dashboard["panels"]
        if "llm_redact_ner_prefetch_misses_total" in json.dumps(p.get("targets", []))
    ]
    coverage_row = _list_item("observability.md", "- strings by outcome;")
    definitions = {
        "HELP": help_line,
        "observability.md": _table_row(
            "observability.md", "| `llm_redact_ner_prefetch_misses_total` |"
        ),
        "detection.md": _table_row("detection.md", "| `prefetch_misses` |"),
        "dashboard": panel["description"].split("prefetch misses =", 1)[1],
        "observability.md NER coverage row": coverage_row.split("`prefetch misses`", 1)[1],
        "DetectorPlan._run": _flat(DetectorPlan._run.__doc__),
    }
    for where, definition in definitions.items():
        assert "inline" in definition, where
        assert "only where the backend ran on that string" in definition, where
        assert "raised on that string first" not in definition, where


def _flat(text: str | None) -> str:
    return " ".join((text or "").split())


def test_every_preview_contract_says_the_live_ner_counters_count_it() -> None:
    # A preview runs the LIVE detectors (the status test above counts its
    # string as scanned and inline): the DashboardHost contract and
    # ProxyState.preview once said it wrote no metrics, beside a HELP text
    # and a panel saying it is counted, and docs/dashboard.md called the
    # dashboard's preview card the CLI's dry run, which writes no metrics
    # (it builds detectors of its own, in its own process). Each now says
    # what it leaves alone (the live vault, the audit, the request and
    # detection counters) and
    # what moves as for a request: the NER coverage counters and the llm
    # backend's circuit breaker (test_llm_ner.py runs both).
    from llm_redact import plugin_api
    from llm_redact.proxy import ProxyState

    host = _flat(plugin_api.DashboardHost.__doc__)
    contracts = {
        "DashboardHost": host[host.index("``preview`` runs") :].split(" OPTIONAL members", 1)[0],
        "ProxyState.preview": _flat(ProxyState.preview.__doc__),
        "dashboard.md": _list_item("dashboard.md", "- **Redaction preview").split(
            "preview card", 1
        )[1],
    }
    for where, contract in contracts.items():
        for phrase in ("live vault", "audit", "llm_redact_ner_", "inline call", "breaker"):
            assert phrase in contract, (where, phrase)
        assert "metrics or audit write" not in contract, where
        assert "metrics and audit are never touched" not in contract, where


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
