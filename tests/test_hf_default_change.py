"""The hf backend's default model changed in 1.13.0 (owner decision D6):
OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 instead of dslim/bert-base-NER.

A minor release must never silently narrow what is redacted, so a
configuration that loads the hf default without naming it, and asks for an
entity the previous default emitted but the new one cannot emit (a raw
``PER``, ``ORG``, ``LOC`` or ``MISC`` while raw entities keep their type), is
refused at every detector build — startup, ``serve --check``, a reload and
the editor's dry run — with both fixes named; doctor WARNs offline from the
labels the model catalog records. The default entities (``PERSON``) keep
working.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from llm_redact.cli import main
from llm_redact.config import Config, ConfigError
from llm_redact.detection import engine, labels
from llm_redact.detection.engine import (
    DetectionConfig,
    NerConfig,
    build_detectors,
    default_change_losses,
    default_change_message,
)
from llm_redact.detection.model_catalog import (
    DEFAULT_MODELS,
    PREVIOUS_HF_DEFAULT,
    PREVIOUS_HF_DEFAULT_LABELS,
    lookup,
)
from llm_redact.doctor_cli import run_doctor
from llm_redact.proxy import create_app
from ner_fakes import FakeHfPipe, FakeHub, install_transformers
from real_models import cached_snapshot, offline_hub

HF_DEFAULT = DEFAULT_MODELS["hf"]
_ENTRY = lookup(HF_DEFAULT)
assert _ENTRY is not None and _ENTRY.revision is not None
HF_DEFAULT_PIN = _ENTRY.revision
OTHER_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _id2label(names: tuple[str, ...]) -> dict[int, str]:
    """A BIO model's config.json id2label over ``names``."""
    tags = ["O", *(f"{tag}-{name}" for name in names for tag in ("B", "I"))]
    return dict(enumerate(tags))


# The new default's labels as its config.json names them (the catalog's facts).
OPENMED_ID2LABEL = _id2label(_ENTRY.model_labels)
DSLIM_ID2LABEL = dict(enumerate(PREVIOUS_HF_DEFAULT_LABELS))


def _pipe(id2label: dict[int, str] | None = None) -> FakeHfPipe:
    return FakeHfPipe(
        [("Jane Doe", "first_name", 0.9)],
        id2label=OPENMED_ID2LABEL if id2label is None else id2label,
    )


def _ner(**fields: Any) -> NerConfig:
    return NerConfig(enabled=True, backend="hf", **fields)


def _message(*entities: str) -> str:
    names = ", ".join(f'"{e}"' for e in entities)
    return (
        "[detection.ner] entities: the hf backend's default model changed in 1.13.0 from"
        f" dslim/bert-base-NER to {HF_DEFAULT}, which cannot emit {names} (refused rather"
        ' than redact less): set [detection.ner.models] hf = "dslim/bert-base-NER" to keep'
        " the previous default, or change [detection.ner] entities to types the new default"
        ' emits (for example "PERSON")'
    )


# --- the decision ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entities", "lost"),
    [
        # The default entities: the new default emits PERSON (first_name,
        # last_name ... fold into it).
        (("PERSON",), ()),
        (("PERSON", "EMAIL", "ADDRESS"), ()),
        # Raw CoNLL-2003 labels keep their own type while raw entities do not
        # fold: the previous default emitted them, the new one cannot.
        (("PER",), ("PER",)),
        (("ORG", "LOC", "MISC"), ("ORG", "LOC", "MISC")),
        (("PERSON", "per", "ORG", "EMAIL"), ("per", "ORG")),
        (("PER", "PER"), ("PER",)),
        # Neither model emits a typo: the never-match warning's case.
        (("PERSONS",), ()),
        # Only the new default emits these: nothing is lost.
        (("first_name", "ACCOUNT_NUMBER"), ()),
    ],
)
def test_the_entities_the_previous_default_emitted_and_the_new_one_cannot(
    entities: tuple[str, ...], lost: tuple[str, ...]
) -> None:
    assert default_change_losses(_ner(entities=entities)) == lost


def test_only_while_the_hf_default_loads_unnamed() -> None:
    per = ("PER",)
    # A model named for hf, by the per-backend table or the legacy key.
    assert default_change_losses(_ner(entities=per, model=PREVIOUS_HF_DEFAULT)) == ()
    assert default_change_losses(_ner(entities=per, models=(("hf", PREVIOUS_HF_DEFAULT),))) == ()
    assert default_change_losses(_ner(entities=per, model=HF_DEFAULT)) == ()
    # The legacy key applies with one backend only: with two, hf loads its default.
    two = NerConfig(enabled=True, backends=("hf", "gliner"), model="org/other", entities=per)
    assert default_change_losses(two) == per
    # No hf backend, nothing to lose.
    assert default_change_losses(NerConfig(enabled=True, backend="gliner", entities=per)) == ()


def test_label_overrides_decide_as_the_backend_decides() -> None:
    # PER = "PERSON" makes the request a PERSON one, which the new default emits.
    assert default_change_losses(_ner(entities=("PER",), labels=(("PER", "PERSON"),))) == ()
    # The 2.0 opt-out PER = "PER" keeps asking for PER, which only the
    # previous default emitted.
    assert default_change_losses(_ner(entities=("PER",), labels=(("PER", "PER"),))) == ("PER",)


def test_once_raw_entities_fold_per_is_a_person_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2.0.0 (D15 (i)): PER folds to PERSON, which the new default emits;
    # ORG folds to itself, which only the previous default emitted.
    monkeypatch.setattr(labels, "FOLD_RAW_REQUESTS", True)
    assert default_change_losses(_ner(entities=("PER",))) == ()
    assert default_change_losses(_ner(entities=("ORG",))) == ("ORG",)


def test_the_catalog_decides_only_at_its_pin() -> None:
    # Another revision of the default: the catalog's labels do not describe
    # it, so only the loaded model decides.
    ner = _ner(entities=("PER",), revisions=(("hf", OTHER_REVISION),))
    assert default_change_losses(ner) == ()
    pinned = _ner(entities=("PER",), revisions=(("hf", HF_DEFAULT_PIN),))
    assert default_change_losses(pinned) == ("PER",)


def test_the_message_names_entities_and_both_fixes_never_values() -> None:
    assert default_change_message(("PER", "ORG")) == _message("PER", "ORG")


# --- every build refuses -----------------------------------------------------------


def test_the_default_entities_build_and_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transformers(monkeypatch, _pipe())
    (detector,) = build_detectors(DetectionConfig(enabled=(), ner=_ner()))
    assert [(d.detector_type, d.value) for d in detector.detect("ask Jane Doe")] == [
        ("PERSON", "Jane Doe")
    ]


def test_a_lost_entity_is_refused_before_any_file_is_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = FakeHub()
    pipe = _pipe()
    install_transformers(monkeypatch, pipe, hub)
    with pytest.raises(ConfigError) as refused:
        build_detectors(DetectionConfig(enabled=(), ner=_ner(entities=("PERSON", "PER"))))
    assert str(refused.value) == _message("PER")
    # From the catalog's facts: no file looked up, nothing fetched or loaded.
    assert hub.calls == [] and pipe.built_with == []


def test_naming_the_previous_default_keeps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = FakeHfPipe([("Jane Doe", "PER", 0.9)], id2label=DSLIM_ID2LABEL)
    hub = FakeHub()
    install_transformers(monkeypatch, pipe, hub)
    ner = _ner(entities=("PER",), models=(("hf", PREVIOUS_HF_DEFAULT),))
    (detector,) = build_detectors(DetectionConfig(enabled=(), ner=ner))
    assert [d.detector_type for d in detector.detect("ask Jane Doe")] == ["PER"]
    # At its catalog pin, as before.
    assert [(c["repo_id"], c["revision"]) for c in hub.calls] == [
        (PREVIOUS_HF_DEFAULT, "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc")
    ]


def test_the_loaded_model_decides_where_the_catalog_cannot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pinned to another revision: refused once the model is loaded and its
    # id2label read, as the never-match check reads it.
    pipe = _pipe()
    install_transformers(monkeypatch, pipe)
    ner = _ner(entities=("PER",), revisions=(("hf", OTHER_REVISION),))
    with pytest.raises(ConfigError) as refused:
        build_detectors(DetectionConfig(enabled=(), ner=ner))
    assert str(refused.value) == _message("PER")
    assert pipe.built_with  # loaded, then refused
    # A revision whose labels still give PER passes; one that does not say
    # what it emits (no id2label) is not refused (nothing is known).
    for id2label in (DSLIM_ID2LABEL, None):
        install_transformers(monkeypatch, FakeHfPipe([], id2label=id2label))
        assert build_detectors(DetectionConfig(enabled=(), ner=ner))


def test_a_stand_in_detector_without_a_policy_decides_nothing() -> None:
    class Stub:
        name = "hf"

        def detect(self, text: str) -> list[Any]:
            return []

    assert default_change_losses(_ner(entities=("PER",)), Stub()) == ()  # type: ignore[arg-type]


def test_startup_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transformers(monkeypatch, _pipe())
    config = Config(detection=DetectionConfig(ner=_ner(entities=("PER",))))
    with pytest.raises(ConfigError, match="default model changed in 1.13.0"):
        create_app(config)


def test_serve_check_fails_with_the_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install_transformers(monkeypatch, _pipe())
    config_file = tmp_path / "config.toml"
    config_file.write_text('[detection.ner]\nenabled = true\nbackend = "hf"\nentities = ["PER"]\n')
    with pytest.raises(SystemExit) as exited:
        main(["serve", "--check", "--config", str(config_file)])
    assert exited.value.code == 1
    assert f"serve --check: FAIL: {_message('PER')}" in capsys.readouterr().err
    # The default entities pass.
    config_file.write_text('[detection.ner]\nenabled = true\nbackend = "hf"\n')
    with pytest.raises(SystemExit) as exited:
        main(["serve", "--check", "--config", str(config_file)])
    assert exited.value.code == 0


def test_a_reload_and_the_editor_dry_run_refuse_and_keep_the_running_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    install_transformers(monkeypatch, _pipe())
    config_file = tmp_path / "config.toml"
    config_file.write_text('[detection.ner]\nenabled = true\nbackend = "hf"\n')
    config = Config(detection=DetectionConfig(ner=_ner()))
    state = create_app(config, config_path=config_file).state.proxy
    detectors = state.detectors
    fresh = dataclasses.replace(config, detection=DetectionConfig(ner=_ner(entities=("PER",))))
    with pytest.raises(ConfigError, match="default model changed in 1.13.0"):
        state.validate_config(fresh)
    with pytest.raises(ConfigError, match="default model changed in 1.13.0"):
        state.apply_config(fresh)
    assert state.config.detection.ner.entities == ("PERSON",)
    assert state.detectors is detectors
    # SIGHUP: logged, the running config kept.
    config_file.write_text('[detection.ner]\nenabled = true\nbackend = "hf"\nentities = ["PER"]\n')
    with caplog.at_level(logging.ERROR, logger="llm_redact"):
        state.reload()
    assert state.config.detection.ner.entities == ("PERSON",)
    assert [r.getMessage() for r in caplog.records] == [
        f"config reload failed; keeping current config: {_message('PER')}"
    ]


# --- doctor: offline, from the catalog's facts -------------------------------------


def _doctor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], ner: str
) -> list[tuple[str, str, str]]:
    config = tmp_path / "config.toml"
    config.write_text(f"port = 1\n[detection.ner]\n{ner}\n")
    run_doctor(argparse.Namespace(config=config, json=True))
    checks = json.loads(capsys.readouterr().out)["checks"]
    return [(row["level"], row["area"], row["message"]) for row in checks]


def test_doctor_warns_without_loading_a_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ner_fakes import install_hub

    install_hub(monkeypatch)
    import sys

    monkeypatch.setitem(sys.modules, "transformers", None)  # never imported
    rows = _doctor(tmp_path, capsys, 'enabled = true\nbackend = "hf"\nentities = ["PER", "ORG"]')
    assert ("WARN", "ner", _message("PER", "ORG")) in rows
    for ner in (
        'enabled = true\nbackend = "hf"',
        'enabled = true\nbackend = "hf"\nentities = ["PER"]\nmodel = "dslim/bert-base-NER"',
        'enabled = false\nbackend = "hf"\nentities = ["PER"]',
    ):
        rows = _doctor(tmp_path, capsys, ner)
        assert not [r for r in rows if "default model changed" in r[2]], ner


# --- the facts against the real snapshots (deselected by default) ------------------


def _cached_config(repo_id: str, revision: str) -> dict[str, Any]:
    folder = cached_snapshot(repo_id, revision, ["config.json"])
    loaded: dict[str, Any] = json.loads((Path(folder) / "config.json").read_text())
    return loaded


@pytest.mark.real_model
def test_real_the_catalog_records_the_defaults_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    offline_hub(monkeypatch)
    previous = lookup(PREVIOUS_HF_DEFAULT)
    assert previous is not None and previous.revision is not None
    id2label = _cached_config(PREVIOUS_HF_DEFAULT, previous.revision)["id2label"]
    assert [id2label[str(i)] for i in range(len(id2label))] == list(PREVIOUS_HF_DEFAULT_LABELS)
    id2label = _cached_config(HF_DEFAULT, HF_DEFAULT_PIN)["id2label"]
    names = {label.split("-", 1)[1] for label in id2label.values() if label != "O"}
    assert sorted(names) == sorted(_ENTRY.model_labels)
    assert all(label == "O" or label[:2] in ("B-", "I-") for label in id2label.values())


@pytest.mark.real_model
def test_real_the_default_finds_names_and_refuses_a_lost_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    offline_hub(monkeypatch)
    cached_snapshot(HF_DEFAULT, HF_DEFAULT_PIN, ["config.json", "model.safetensors"])
    (detector,) = build_detectors(DetectionConfig(enabled=(), ner=_ner()))
    found = detector.detect("Please forward this to Jane Doe before noon.")
    assert [(d.detector_type, d.value) for d in found] == [("PERSON", "Jane Doe")]
    # With the catalog's facts out of the way, the loaded model decides alike.
    monkeypatch.setattr(engine, "_catalog_types", lambda ner, policy: None)
    with pytest.raises(ConfigError) as refused:
        build_detectors(DetectionConfig(enabled=(), ner=_ner(entities=("PERSON", "PER"))))
    assert str(refused.value) == _message("PER")
