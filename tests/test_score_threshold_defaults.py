"""Per-model default ``score_threshold`` (owner decision 2026-10-07).

``[detection.ner] score_threshold`` is optional: unset, each confidence
backend (gliner, gliner2, presidio, hf) runs at its model's catalog default
(``CatalogEntry.score_threshold``), else the historical 0.5; a configured
value always wins, for every backend. ``NerConfig.score_threshold_for`` is
the one resolution; ``build_detectors`` hands each builder its number and
``/status`` reports it with its source. Fake models and the fake hub only:
no network.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import httpx
import pytest

from llm_redact.config import (
    CONFIDENCE_BACKENDS,
    Config,
    ConfigError,
    ProviderConfig,
    load_config,
    parse_config,
)
from llm_redact.config_write import emit_config_toml
from llm_redact.detection.engine import (
    DEFAULT_SCORE_THRESHOLD,
    DetectionConfig,
    NerConfig,
    build_detectors,
    ner_status,
)
from llm_redact.detection.gliner2_ner import build_gliner2_detector
from llm_redact.detection.model_catalog import CATALOG, DEFAULT_MODELS, SIDECAR_NAME
from llm_redact.doctor_cli import run_doctor
from llm_redact.proxy import create_app
from ner_fakes import (
    DEFAULT_REPO,
    FakeAnalyzer,
    FakeGliner,
    FakeGliner2,
    FakeHfPipe,
    FakeSpacy,
    install_gliner,
    install_gliner2,
    install_hub,
    install_presidio,
    install_spacy,
    install_transformers,
)

FASTINO_PII = "fastino/gliner2-privacy-filter-PII-multi"
FASTINO_PII_PIN = "1cb4166094dc58fa8d836429f060d6c95f62b495"
BENCH_CONFIG = Path(__file__).resolve().parents[1] / "bench" / "configs" / "gliner2-fastino.toml"


def _folder(tmp_path: Path, sidecar: str | None) -> str:
    """A local model folder (the fake GLiNER2 checkpoint's files), with a
    sidecar file holding ``sidecar`` when given."""
    for name, content in DEFAULT_REPO.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(content)
    if sidecar is not None:
        (tmp_path / SIDECAR_NAME).write_text(sidecar)
    return str(tmp_path)


def _sidecar(model_id: str, revision: str | None = None) -> str:
    return json.dumps({"model_id": model_id, "revision": revision})


# --- the catalog -------------------------------------------------------------------


def test_only_measured_models_carry_a_catalog_threshold() -> None:
    # Set only where the bench measured one or a card states one (the
    # Fastino PII model: 0.9, docs/ner-landscape.md); never guessed.
    assert {e.model_id: e.score_threshold for e in CATALOG if e.score_threshold is not None} == {
        FASTINO_PII: 0.9
    }
    # A default model never changes its threshold through the catalog.
    for backend, model in DEFAULT_MODELS.items():
        assert NerConfig(backend=backend, model=model).score_threshold_for(backend) == (
            DEFAULT_SCORE_THRESHOLD,
            "default",
        )


# --- the resolution ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ner", "backend", "expected"),
    [
        # A configured value wins over the catalog, on every backend.
        (
            NerConfig(backend="gliner2", model=FASTINO_PII, score_threshold=0.3),
            "gliner2",
            (0.3, "config"),
        ),
        (
            NerConfig(backend="gliner2", model=FASTINO_PII, score_threshold=0.5),
            "gliner2",
            (0.5, "config"),
        ),
        (NerConfig(backend="presidio", score_threshold=0.35), "presidio", (0.35, "config")),
        # Unset: the catalog's default for the model the backend loads.
        (NerConfig(backend="gliner2", model=FASTINO_PII), "gliner2", (0.9, "catalog")),
        # The Hub resolves ids case-insensitively; so does the catalog.
        (NerConfig(backend="gliner2", model=FASTINO_PII.upper()), "gliner2", (0.9, "catalog")),
        (
            NerConfig(backends=("gliner2", "hf"), models=(("gliner2", FASTINO_PII),)),
            "gliner2",
            (0.9, "catalog"),
        ),
        # ... per backend: the other backend's model has none.
        (
            NerConfig(backends=("gliner2", "hf"), models=(("gliner2", FASTINO_PII),)),
            "hf",
            (0.5, "default"),
        ),
        # The legacy `model` key applies only with one backend: with two,
        # gliner2 loads its default model.
        (NerConfig(backends=("gliner2", "hf"), model=FASTINO_PII), "gliner2", (0.5, "default")),
        # The catalog entry is for the gliner2 backend only.
        (NerConfig(backend="gliner", model=FASTINO_PII), "gliner", (0.5, "default")),
        (NerConfig(backend="hf", model=FASTINO_PII), "hf", (0.5, "default")),
        # Default models, an unknown model, and a backend with no Hub model.
        (NerConfig(backend="gliner2"), "gliner2", (0.5, "default")),
        (NerConfig(backend="hf", model="org/unknown-model"), "hf", (0.5, "default")),
        (NerConfig(backend="presidio"), "presidio", (0.5, "default")),
        # A folder path that is not there is no catalogued model.
        (NerConfig(backend="gliner2", model="/no/such/folder"), "gliner2", (0.5, "default")),
    ],
)
def test_the_effective_threshold(ner: NerConfig, backend: str, expected: tuple[float, str]) -> None:
    assert ner.score_threshold_for(backend) == expected


def test_a_local_folder_is_identified_by_its_sidecar(tmp_path: Path) -> None:
    folder = _folder(tmp_path / "pii", _sidecar(FASTINO_PII, FASTINO_PII_PIN))
    assert NerConfig(backend="gliner2", model=folder).score_threshold_for("gliner2") == (
        0.9,
        "catalog",
    )
    # A configured value still wins.
    ner = NerConfig(backend="gliner2", model=folder, score_threshold=0.6)
    assert ner.score_threshold_for("gliner2") == (0.6, "config")
    # Another model's sidecar, or none: no catalog default.
    other = _folder(tmp_path / "base", _sidecar("fastino/gliner2-base-v1"))
    assert NerConfig(backend="gliner2", model=other).score_threshold_for("gliner2") == (
        0.5,
        "default",
    )
    bare = _folder(tmp_path / "bare", None)
    assert NerConfig(backend="gliner2", model=bare).score_threshold_for("gliner2") == (
        0.5,
        "default",
    )


def test_an_unreadable_sidecar_is_refused_as_the_build_refuses_it(tmp_path: Path) -> None:
    folder = _folder(tmp_path, "{not json")
    with pytest.raises(ConfigError, match=SIDECAR_NAME):
        NerConfig(backend="gliner2", model=folder).score_threshold_for("gliner2")
    # A configured threshold needs no lookup.
    assert NerConfig(backend="gliner2", model=folder, score_threshold=0.7).score_threshold_for(
        "gliner2"
    ) == (0.7, "config")


# --- the builders read it ------------------------------------------------------------


def _gliner2_found(monkeypatch: pytest.MonkeyPatch, ner: NerConfig) -> list[str]:
    # The model scores the name 0.85: kept at 0.5, dropped at 0.9.
    install_gliner2(monkeypatch, FakeGliner2([("Jane Roe", "person", 0.85)]))
    detectors = build_detectors(DetectionConfig(enabled=(), ner=ner))
    return [d.value for detector in detectors for d in detector.detect("hi Jane Roe")]


@pytest.mark.parametrize(
    ("ner", "found"),
    [
        (NerConfig(enabled=True, backend="gliner2", model=FASTINO_PII), []),
        (
            NerConfig(enabled=True, backend="gliner2", model=FASTINO_PII, score_threshold=0.8),
            ["Jane Roe"],
        ),
        (NerConfig(enabled=True, backend="gliner2"), ["Jane Roe"]),
    ],
    ids=["catalog-0.9", "config-0.8", "default-0.5"],
)
def test_the_gliner2_builder_runs_at_the_effective_threshold(
    monkeypatch: pytest.MonkeyPatch, ner: NerConfig, found: list[str]
) -> None:
    assert _gliner2_found(monkeypatch, ner) == found


def test_a_builder_called_alone_resolves_the_same_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_gliner2(monkeypatch, FakeGliner2([("Jane Roe", "person", 0.85)]))
    unset = build_gliner2_detector(NerConfig(enabled=True, backend="gliner2", model=FASTINO_PII))
    assert list(unset.detect("hi Jane Roe")) == []
    base = build_gliner2_detector(NerConfig(enabled=True, backend="gliner2"))
    assert [d.value for d in base.detect("hi Jane Roe")] == ["Jane Roe"]


def test_each_backend_gets_its_own_effective_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    # gliner2 runs the Fastino PII model (0.9), hf its default (0.5); both
    # models score the name 0.85.
    install_gliner2(monkeypatch, FakeGliner2([("Jane Roe", "person", 0.85)]))
    install_transformers(monkeypatch, FakeHfPipe([("Jane Roe", "PER", 0.85)], id2label={0: "PER"}))
    ner = NerConfig(enabled=True, backends=("gliner2", "hf"), models=(("gliner2", FASTINO_PII),))
    detectors = build_detectors(DetectionConfig(enabled=(), ner=ner))
    status = ner_status(ner, detectors)["backends"]
    assert {
        name: (b["score_threshold"], b["score_threshold_source"]) for name, b in status.items()
    } == {
        "gliner2": (0.9, "catalog"),
        "hf": (0.5, "default"),
    }
    found = {d.value for detector in detectors for d in detector.detect("hi Jane Roe")}
    assert found == {"Jane Roe"}  # hf's, gliner2 dropped it
    # A configured value applies to both.
    both = NerConfig(
        enabled=True,
        backends=("gliner2", "hf"),
        models=(("gliner2", FASTINO_PII),),
        score_threshold=0.95,
    )
    detectors = build_detectors(DetectionConfig(enabled=(), ner=both))
    assert [d for detector in detectors for d in detector.detect("hi Jane Roe")] == []
    assert {
        name: (b["score_threshold"], b["score_threshold_source"])
        for name, b in ner_status(both, detectors)["backends"].items()
    } == {"gliner2": (0.95, "config"), "hf": (0.95, "config")}


def test_gliner_and_presidio_builders_read_it(monkeypatch: pytest.MonkeyPatch) -> None:
    install_gliner(monkeypatch, FakeGliner([("Jane Roe", "person", 0.6)]))
    install_presidio(monkeypatch, FakeAnalyzer([("Jane Roe", "PERSON", 0.6)], ("PERSON",)))
    for score_threshold, expected in ((None, 2), (0.7, 0)):
        ner = NerConfig(
            enabled=True, backends=("gliner", "presidio"), score_threshold=score_threshold
        )
        detectors = build_detectors(DetectionConfig(enabled=(), ner=ner))
        assert len([d for det in detectors for d in det.detect("hi Jane Roe")]) == expected


def test_a_local_folder_runs_at_its_catalog_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = _folder(tmp_path, _sidecar(FASTINO_PII, FASTINO_PII_PIN))
    assert (
        _gliner2_found(monkeypatch, NerConfig(enabled=True, backend="gliner2", model=folder)) == []
    )


# --- /status -------------------------------------------------------------------------


def test_status_names_the_threshold_and_its_source(monkeypatch: pytest.MonkeyPatch) -> None:
    install_gliner2(monkeypatch, FakeGliner2([]))
    install_spacy(monkeypatch, FakeSpacy([], labels=("PERSON",)))
    ner = NerConfig(enabled=True, backends=("gliner2", "spacy"), models=(("gliner2", FASTINO_PII),))
    block = ner_status(ner, build_detectors(DetectionConfig(ner=ner)))["backends"]
    assert (block["gliner2"]["score_threshold"], block["gliner2"]["score_threshold_source"]) == (
        0.9,
        "catalog",
    )
    # spaCy reports no confidences: nothing to say.
    assert (block["spacy"]["score_threshold"], block["spacy"]["score_threshold_source"]) == (
        None,
        None,
    )


async def test_status_through_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    install_gliner2(monkeypatch, FakeGliner2([]))
    install_hub(monkeypatch)
    config = Config(
        providers={"openai": ProviderConfig("https://api.openai.test")},
        detection=DetectionConfig(
            ner=NerConfig(enabled=True, backend="gliner2", model=FASTINO_PII)
        ),
    )
    transport = httpx.ASGITransport(app=create_app(config))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
        payload = (await client.get("/__llm-redact/status")).json()
    gliner2 = payload["detection"]["ner"]["backends"]["gliner2"]
    assert (gliner2["score_threshold"], gliner2["score_threshold_source"]) == (0.9, "catalog")


# --- configuration -------------------------------------------------------------------


def test_unset_stays_unset_through_parse_and_emit() -> None:
    parsed = parse_config({"detection": {"ner": {"backend": "gliner2", "model": FASTINO_PII}}}, "t")
    assert parsed.detection.ner.score_threshold is None
    emitted = emit_config_toml(parsed)
    # `config show` and an editor save never invent a value the user did not
    # set: a written 0.5 would override the model's catalog default.
    assert "score_threshold" not in emitted
    assert parse_config(tomllib.loads(emitted), "t") == parsed


@pytest.mark.parametrize("value", [0.5, 0.9, 0.3])
def test_an_explicit_value_is_kept(value: float) -> None:
    # 0.5 included: a file that names the historical default keeps it, over
    # any catalog default.
    parsed = parse_config(
        {
            "detection": {
                "ner": {"backend": "gliner2", "model": FASTINO_PII, "score_threshold": value}
            }
        },
        "t",
    )
    assert parsed.detection.ner.score_threshold == value
    emitted = emit_config_toml(parsed)
    assert f"score_threshold = {value}" in emitted
    assert parse_config(tomllib.loads(emitted), "t") == parsed
    assert parsed.detection.ner.score_threshold_for("gliner2") == (value, "config")


def test_the_parser_still_refuses_it_without_a_confidence_backend() -> None:
    assert set(CONFIDENCE_BACKENDS) == {"gliner", "gliner2", "presidio", "hf"}
    with pytest.raises(ConfigError, match="score_threshold"):
        parse_config({"detection": {"ner": {"backend": "spacy", "score_threshold": 0.5}}}, "t")


def test_the_fastino_bench_config_runs_at_the_catalog_default() -> None:
    # bench/configs/gliner2-fastino.toml relies on the catalog default; its
    # baselines (bench/ner_thresholds.toml, ner_ceilings.toml) were measured
    # at 0.9, so the default must stay 0.9 or they are no longer valid.
    ner = load_config(BENCH_CONFIG).detection.ner
    assert ner.score_threshold is None
    assert ner.score_threshold_for("gliner2") == (0.9, "catalog")


# --- doctor --------------------------------------------------------------------------


def _model_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str], ner: str) -> list[str]:
    config = tmp_path / "config.toml"
    config.write_text(f'port = 1\n[detection.ner]\nenabled = true\nbackend = "gliner2"\n{ner}\n')
    run_doctor(argparse.Namespace(config=config, json=True))
    checks = json.loads(capsys.readouterr().out)["checks"]
    return [row["message"] for row in checks if row["area"] == "models"]


def test_doctor_names_a_catalog_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    row = (
        f"gliner2: score_threshold 0.9 is the model catalog's default for {FASTINO_PII}"
        " ([detection.ner] score_threshold overrides it)"
    )
    assert row in _model_rows(tmp_path, capsys, f'model = "{FASTINO_PII}"')
    # Silent for a configured threshold and for the historical default.
    configured = _model_rows(tmp_path, capsys, f'model = "{FASTINO_PII}"\nscore_threshold = 0.5')
    assert not any("catalog's default" in message for message in configured)
    assert not any("catalog's default" in message for message in _model_rows(tmp_path, capsys, ""))
