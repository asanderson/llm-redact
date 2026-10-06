"""What the model catalog says about the models the NER backends load.

``detection/model_sources.py`` names each Hub backend's model, its revision
and catalog entry from the configuration and local files only; the detector
build records it on each backend, ``/status`` publishes it, a restricted
model logs a startup warning, and docs/detection.md's catalog tables match
the catalog. Fake models and the fake hub only: no network.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig
from llm_redact.detection.engine import (
    DetectionConfig,
    NerConfig,
    build_detectors,
    ner_status,
    ner_warnings,
)
from llm_redact.detection.model_catalog import CATALOG, SIDECAR_NAME
from llm_redact.detection.model_sources import (
    UNKNOWN_SOURCE_FIELDS,
    hub_sources,
    model_source,
)
from llm_redact.proxy import create_app
from ner_fakes import (
    FakeGliner,
    FakeHfPipe,
    FakeHub,
    FakeSpacy,
    install_gliner,
    install_hub,
    install_spacy,
    install_transformers,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "detection.md"
DSLIM = "dslim/bert-base-NER"
DSLIM_PIN = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
SHA = "0123456789abcdef0123456789abcdef01234567"
RESTRICTED = "Isotonic/distilbert_finetuned_ai4privacy_v2"


def _fields(**overrides: object) -> dict[str, object]:
    return {**UNKNOWN_SOURCE_FIELDS, **overrides}


@pytest.mark.parametrize(
    ("ner", "backend", "fields", "pinned_by"),
    [
        (
            NerConfig(backend="hf"),
            "hf",
            _fields(
                source="hub", model_id=DSLIM, revision=DSLIM_PIN, pinned=True,
                catalog="vetted", license="MIT",
            ),
            "catalog",
        ),
        (
            NerConfig(backend="hf", revisions=(("hf", SHA),)),
            "hf",
            _fields(
                source="hub", model_id=DSLIM, revision=SHA, pinned=True, catalog="vetted",
                license="MIT",
            ),
            "config",
        ),
        (
            NerConfig(backend="hf", model="org/private-ner"),
            "hf",
            _fields(source="hub", model_id="org/private-ner", pinned=False),
            None,
        ),
        (
            # Any letter case finds the entry, as the Hub resolves the id.
            NerConfig(backend="hf", model=RESTRICTED.upper()),
            "hf",
            _fields(
                source="hub", model_id=RESTRICTED.upper(), pinned=False, catalog="restricted",
                license="CC-BY-NC-4.0",
            ),
            None,
        ),
        (
            NerConfig(
                backends=("gliner", "hf"), models=(("gliner", "ai4privacy/llama-ai4privacy-x"),)
            ),
            "gliner",
            _fields(
                source="hub", model_id="ai4privacy/llama-ai4privacy-x", pinned=False,
                catalog="restricted", license="MIT",
            ),
            None,
        ),
    ],
)  # fmt: skip
def test_hub_model_sources(
    ner: NerConfig, backend: str, fields: dict[str, object], pinned_by: str | None
) -> None:
    source = model_source(ner, backend)
    assert source.status_fields() == fields
    assert source.pinned_by == pinned_by
    assert source.local is False and source.sidecar_problem is None


def test_local_folder_sources(tmp_path: Path) -> None:
    ner = NerConfig(backend="hf", model=str(tmp_path))
    assert model_source(ner, "hf").status_fields() == _fields(source="local", pinned=False)
    sidecar = tmp_path / SIDECAR_NAME
    sidecar.write_text(json.dumps({"model_id": DSLIM, "revision": DSLIM_PIN}))
    source = model_source(ner, "hf")
    assert source.status_fields() == _fields(
        source="local", model_id=DSLIM, revision=DSLIM_PIN, pinned=True, catalog="vetted",
        license="MIT",
    )  # fmt: skip
    assert source.pinned_by == "sidecar"
    sidecar.write_text(json.dumps({"model_id": DSLIM}))
    assert (model_source(ner, "hf").revision, model_source(ner, "hf").pinned_by) == (None, None)
    sidecar.write_text("[]")
    broken = model_source(ner, "hf")
    assert broken.sidecar_problem == f"{sidecar}: not a JSON object"
    assert broken.status_fields() == _fields(source="local", pinned=False)


def test_hub_sources_follow_the_active_backends() -> None:
    ner = NerConfig(backends=("spacy", "hf", "gliner"))
    assert [source.backend for source in hub_sources(ner)] == ["hf", "gliner"]
    assert hub_sources(NerConfig(backend="presidio")) == []


def test_the_restricted_warning_is_neutral_facts() -> None:
    assert model_source(NerConfig(backend="hf"), "hf").restricted_warning() is None
    warning = model_source(NerConfig(backend="hf", model=RESTRICTED), "hf").restricted_warning()
    assert warning == (
        f'[detection.ner] hf model {RESTRICTED!r} has model catalog status "restricted":'
        f" {RESTRICTED}: CC-BY-NC-4.0 (non-commercial); trained on"
        " ai4privacy/pii-masking-200k, whose license requires a company license for"
        f" organizations above 3 staff (https://huggingface.co/{RESTRICTED}, checked 2026-10-05)"
    )


# --- recorded at build time, published, warned --------------------------------------


def test_the_build_records_each_hub_backends_source(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transformers(monkeypatch, FakeHfPipe([], id2label={0: "PER"}))
    install_spacy(monkeypatch, FakeSpacy([], labels=("PERSON",)))
    ner = NerConfig(enabled=True, backends=("hf", "spacy"), model=None, revisions=(("hf", SHA),))
    detectors = build_detectors(DetectionConfig(ner=ner))
    block = ner_status(ner, detectors)["backends"]
    assert {key: block["hf"][key] for key in UNKNOWN_SOURCE_FIELDS} == _fields(
        source="hub", model_id=DSLIM, revision=SHA, pinned=True, catalog="vetted", license="MIT"
    )
    # spaCy pipelines are not Hub snapshots: nothing to say.
    assert {key: block["spacy"][key] for key in UNKNOWN_SOURCE_FIELDS} == UNKNOWN_SOURCE_FIELDS
    assert list(block["hf"]) == [
        "model", "source", "model_id", "revision", "pinned", "catalog", "license", "counters",
    ]  # fmt: skip


def test_a_restricted_model_warns_after_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transformers(monkeypatch, FakeHfPipe([], id2label={0: "PER"}))
    install_gliner(monkeypatch, FakeGliner([]), FakeHub())
    ner = NerConfig(enabled=True, backends=("gliner", "hf"), models=(("hf", RESTRICTED),))
    config = DetectionConfig(ner=ner)
    detectors = build_detectors(config)
    warnings = ner_warnings(config, detectors)
    assert [w for w in warnings if "restricted" in w] == [
        model_source(ner, "hf").restricted_warning()
    ]
    # doctor passes no detectors: it shows the line under `models` instead.
    assert not any("restricted" in w for w in ner_warnings(config))


async def test_status_and_startup_log_through_the_app(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    install_transformers(monkeypatch, FakeHfPipe([], id2label={0: "PER"}))
    install_hub(monkeypatch, FakeHub())
    config = Config(
        providers={"openai": ProviderConfig("https://api.openai.test")},
        detection=DetectionConfig(ner=NerConfig(enabled=True, backend="hf", model=RESTRICTED)),
    )
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        app = create_app(config)
    assert any(
        record.getMessage() == model_source(config.detection.ner, "hf").restricted_warning()
        for record in caplog.records
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
        payload = (await client.get("/__llm-redact/status")).json()
    hf = payload["detection"]["ner"]["backends"]["hf"]
    assert (hf["model"], hf["catalog"], hf["license"], hf["pinned"], hf["revision"]) == (
        RESTRICTED,
        "restricted",
        "CC-BY-NC-4.0",
        False,
        None,
    )


# --- docs/detection.md's catalog tables ---------------------------------------------


def _doc_rows(status: str) -> dict[str, tuple[str, str, str, str]]:
    text = DOCS.read_text(encoding="utf-8")
    match = re.search(
        rf"<!-- model-catalog:{status} -->\n(.*?)<!-- /model-catalog -->", text, re.DOTALL
    )
    assert match is not None, f"docs/detection.md has no {status} catalog table"
    rows: dict[str, tuple[str, str, str, str]] = {}
    for line in match[1].splitlines()[2:]:
        model, backends, license_, revision, base = (
            cell.strip() for cell in line.strip().strip("|").split("|")
        )
        rows[model.strip("`")] = (backends, license_, revision.strip("`"), base)
    return rows


def _base(entry: object) -> str:
    backbone = getattr(entry, "backbone", None)
    if backbone is None:
        return "—"
    pin = getattr(entry, "backbone_revision", None)
    return f"`{backbone}` (`{pin[:12]}`)" if pin else f"`{backbone}` (—)"


@pytest.mark.parametrize("status", ["vetted", "caution", "restricted"])
def test_the_docs_tables_match_the_catalog(status: str) -> None:
    rows = _doc_rows(status)
    expected = {
        f"{entry.model_id}*" if entry.prefix else entry.model_id: (
            ", ".join(entry.backends),
            entry.license,
            entry.revision[:12] if entry.revision else "—",
            _base(entry),
        )
        for entry in CATALOG
        if entry.status == status
    }
    assert rows == expected
