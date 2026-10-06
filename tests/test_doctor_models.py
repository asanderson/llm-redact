"""`llm-redact doctor`'s ``models`` area: where the NER models come from.

Rows for the Hugging Face Hub backends (gliner, gliner2, hf): the download and pickle
switches, each model's pin and whether its files are in the local cache at
that pin. doctor never loads a model and never touches the network: the
fake hub (ner_fakes.FakeHub) answers its local-cache lookups.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from llm_redact.detection.model_catalog import SIDECAR_NAME
from llm_redact.doctor_cli import run_doctor
from ner_fakes import DEFAULT_REPO, FakeHub, install_hub

DSLIM = "dslim/bert-base-NER"
DSLIM_PIN = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
GLINER_SMALL = "urchade/gliner_small-v2.1"
DEBERTA = "microsoft/deberta-v3-small"
DEBERTA_PIN = "a36c739020e01763fe789b4b85e2df55d6180012"
SHA = "0123456789abcdef0123456789abcdef01234567"
# urchade's v2.1 layout (no tokenizer, no encoder_config) and its base model.
URCHADE_REPO = {
    "gliner_config.json": json.dumps({"model_name": DEBERTA}),
    "pytorch_model.bin": "",
}
DEBERTA_REPO = {
    "config.json": json.dumps({"model_type": "deberta-v2"}),
    "tokenizer_config.json": "{}",
    "spm.model": "",
}


def _rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], ner: str, *, extra: str = ""
) -> tuple[int, list[tuple[str, str]]]:
    config = tmp_path / "config.toml"
    config.write_text(f"port = 1\n[detection.ner]\nenabled = true\n{ner}\n{extra}")
    code = run_doctor(argparse.Namespace(config=config, json=True))
    checks = json.loads(capsys.readouterr().out)["checks"]
    return code, [(row["level"], row["message"]) for row in checks if row["area"] == "models"]


def _levels(rows: list[tuple[str, str]], level: str) -> list[str]:
    return [message for row_level, message in rows if row_level == level]


def test_no_models_rows_while_ner_is_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(monkeypatch)
    config = tmp_path / "config.toml"
    config.write_text('port = 1\n[detection.ner]\nbackend = "hf"\nallow_download = true\n')
    run_doctor(argparse.Namespace(config=config, json=True))
    checks = json.loads(capsys.readouterr().out)["checks"]
    assert [row for row in checks if row["area"] == "models"] == []
    assert hub.calls == []


def test_no_models_rows_for_backends_without_hub_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    _, rows = _rows(tmp_path, capsys, 'backend = "spacy"')
    assert rows == []


def test_a_cached_default_model_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(monkeypatch)
    _, rows = _rows(tmp_path, capsys, 'backend = "hf"')
    assert rows == [
        (
            "PASS",
            "downloads off (allow_download = false): models load from the local Hugging"
            " Face cache or local folders only",
        ),
        (
            "PASS",
            f"hf: {DSLIM} pinned at {DSLIM_PIN} by the model catalog (model catalog: vetted, MIT)",
        ),
        ("PASS", f"hf: {DSLIM}: every file the loader reads is in the local Hugging Face cache"),
    ]
    # A local-cache lookup at the pin, never a download.
    assert {(c["repo_id"], c["revision"], c["local_files_only"]) for c in hub.calls} == {
        (DSLIM, DSLIM_PIN, True)
    }


def test_a_cached_gliner2_model_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact.detection.model_catalog import lookup

    fastino = "fastino/gliner2-base-v1"
    entry = lookup(fastino)
    assert entry is not None and entry.revision is not None
    hub = install_hub(monkeypatch)
    _, rows = _rows(tmp_path, capsys, 'backend = "gliner2"')
    assert rows[1:] == [
        (
            "PASS",
            f"gliner2: {fastino} pinned at {entry.revision} by the model catalog (model catalog:"
            f" caution, Apache-2.0): {entry.describe()}",
        ),
        (
            "PASS",
            f"gliner2: {fastino}: every file the loader reads is in the local Hugging Face cache",
        ),
    ]
    assert {(c["repo_id"], c["revision"], c["local_files_only"]) for c in hub.calls} == {
        (fastino, entry.revision, True)
    }
    # The loader's own completeness check: a snapshot without its weights.
    hub.default = {k: v for k, v in DEFAULT_REPO.items() if k != "model.safetensors"}
    hub.root = tmp_path / "other-cache"
    _, rows = _rows(tmp_path, capsys, 'backend = "gliner2"')
    assert rows[-1][0] == "FAIL" and rows[-1][1].endswith("; missing: weights")


def test_the_three_switch_and_pin_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(monkeypatch)
    _, rows = _rows(
        tmp_path,
        capsys,
        'backend = "hf"\nmodel = "org/ner-model"\nallow_download = true\n'
        "allow_pickle_weights = true",
    )
    warnings = _levels(rows, "WARN")
    assert warnings == [
        "allow_download = true: the proxy's startup may download model weights from"
        " huggingface.co (the model id and revision, never request content; a reload never"
        " downloads)",
        "allow_pickle_weights = true: an hf model without safetensors weights loads"
        " pytorch_model.bin, a pickle (loading a pickle can run code)",
        "hf: org/ner-model has no pin (not in the model catalog): the newest cached revision"
        " of its default branch loads; pin a commit in [detection.ner.revisions] hf"
        " (`llm-redact models pull` prints the one it fetches)",
    ]
    assert all(c["local_files_only"] for c in hub.calls)


def test_the_pickle_warning_needs_the_hf_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    _, rows = _rows(tmp_path, capsys, 'backend = "gliner"\nallow_pickle_weights = true')
    assert not any("allow_pickle_weights" in message for _, message in rows)


def test_a_configured_pin_is_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(monkeypatch)
    _, rows = _rows(
        tmp_path, capsys, 'backend = "hf"', extra=f'[detection.ner.revisions]\nhf = "{SHA}"\n'
    )
    assert (
        "PASS",
        f"hf: {DSLIM} pinned at {SHA} by [detection.ner.revisions] (model catalog: vetted, MIT)",
    ) in rows
    assert {c["revision"] for c in hub.calls} == {SHA}


def test_a_model_missing_from_the_cache_fails_while_downloads_are_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch, FakeHub(uncached={DSLIM}))
    code, rows = _rows(tmp_path, capsys, 'backend = "hf"')
    assert code == 1
    assert _levels(rows, "FAIL") == [
        f"[detection.ner] hf model '{DSLIM}' at revision {DSLIM_PIN} is not (completely) in"
        " the local Hugging Face cache, and downloads are off (an older revision in the"
        " cache does not count); run `llm-redact models pull`, or set [detection.ner]"
        " allow_download = true to fetch it at startup (a reload never downloads)"
    ]


def test_a_model_missing_from_the_cache_warns_when_startup_may_fetch_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(monkeypatch, FakeHub(uncached={DSLIM}))
    _, rows = _rows(tmp_path, capsys, 'backend = "hf"\nallow_download = true')
    assert _levels(rows, "FAIL") == []
    assert (
        "WARN",
        f"hf: hf model {DSLIM} is not (completely) in the local Hugging Face cache; the"
        " proxy's startup will fetch it (allow_download = true)",
    ) in rows
    assert all(c["local_files_only"] for c in hub.calls)  # doctor itself never downloads


def test_an_incomplete_snapshot_fails_naming_what_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cached snapshot without its tokenizer (an interrupted download).
    repo = {name: text for name, text in DEFAULT_REPO.items() if "tok" not in name}
    repo.pop("vocab.txt")
    install_hub(monkeypatch, FakeHub(repos={DSLIM: repo}))
    code, rows = _rows(tmp_path, capsys, 'backend = "hf"')
    assert code == 1
    (fail,) = _levels(rows, "FAIL")
    assert fail.endswith("(a reload never downloads); missing: tokenizer files")


def test_a_missing_gliner_base_model_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(repos={GLINER_SMALL: URCHADE_REPO, DEBERTA: DEBERTA_REPO}, uncached={DEBERTA})
    install_hub(monkeypatch, hub)
    code, rows = _rows(tmp_path, capsys, 'backend = "gliner"')
    assert code == 1
    (fail,) = _levels(rows, "FAIL")
    assert fail.startswith(
        f"[detection.ner] gliner base model '{DEBERTA}' at revision {DEBERTA_PIN} is not"
        " (completely) in the local Hugging Face cache"
    )
    assert "llm-redact models pull" in fail


def test_a_cached_gliner_model_and_base_pass_without_assembling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(repos={GLINER_SMALL: URCHADE_REPO, DEBERTA: DEBERTA_REPO})
    install_hub(monkeypatch, hub)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    _, rows = _rows(tmp_path, capsys, 'backend = "gliner"')
    assert _levels(rows, "FAIL") == [] and _levels(rows, "WARN") == []
    assert (
        "PASS",
        f"gliner: {GLINER_SMALL}: every file the loader reads is in the local Hugging Face cache",
    ) in rows
    # doctor is read-only: no assembled folder is written.
    assert not (tmp_path / "data").exists()


def test_an_unpinned_gliner_base_model_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(repos={"org/gliner-x": URCHADE_REPO, DEBERTA: DEBERTA_REPO})
    install_hub(monkeypatch, hub)
    _, rows = _rows(tmp_path, capsys, 'backend = "gliner"\nmodel = "org/gliner-x"')
    assert (
        "WARN",
        f"gliner: org/gliner-x takes its tokenizer and encoder configuration from its base"
        f" model {DEBERTA}, whose revision the model catalog does not pin: the newest cached"
        " revision loads",
    ) in rows


def _folder(path: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(content)
    return path


def test_local_folders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(monkeypatch)
    folder = _folder(tmp_path / "model", DEFAULT_REPO)
    _, rows = _rows(tmp_path, capsys, f'backend = "hf"\nmodel = {json.dumps(str(folder))}')
    assert rows[1:] == [
        (
            "PASS",
            f"hf: {folder} is a local folder without {SIDECAR_NAME} (it loads as it is;"
            " nothing identifies the model)",
        ),
        ("PASS", f"hf: {folder}: every file the loader reads is in its folder"),
    ]
    (folder / SIDECAR_NAME).write_text(json.dumps({"model_id": DSLIM, "revision": DSLIM_PIN}))
    _, rows = _rows(tmp_path, capsys, f'backend = "hf"\nmodel = {json.dumps(str(folder))}')
    assert rows[1] == (
        "PASS",
        f"hf: {folder} is a local folder holding {DSLIM} at {DSLIM_PIN} (model catalog:"
        " vetted, MIT)",
    )
    (folder / SIDECAR_NAME).write_text(json.dumps({"model_id": DSLIM}))
    _, rows = _rows(tmp_path, capsys, f'backend = "hf"\nmodel = {json.dumps(str(folder))}')
    assert rows[1][1].endswith(
        f"holding {DSLIM} at an unrecorded revision (model catalog: vetted, MIT)"
    )
    assert hub.calls == []


def test_a_broken_local_folder_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    folder = _folder(tmp_path / "model", {**DEFAULT_REPO, SIDECAR_NAME: "[]"})
    (folder / "model.safetensors").unlink()
    model = json.dumps(str(folder))
    code, rows = _rows(tmp_path, capsys, f'backend = "gliner"\nmodel = {model}')
    assert code == 1
    assert _levels(rows, "FAIL") == [
        f"gliner: {folder / SIDECAR_NAME}: not a JSON object",
        f"[detection.ner] gliner model: {folder / SIDECAR_NAME}: not a JSON object",
    ]
    (folder / SIDECAR_NAME).unlink()
    _, rows = _rows(
        tmp_path, capsys, f'backend = "hf"\nmodel = {model}\nallow_pickle_weights = true'
    )
    assert _levels(rows, "FAIL") == [
        f"[detection.ner] hf model {str(folder)!r} is a local directory that lacks what the"
        " loader needs: weights"
    ]
    _, rows = _rows(
        tmp_path,
        capsys,
        f'backend = "gliner"\nmodel = {model}',
        extra=f'[detection.ner.revisions]\ngliner = "{SHA}"\n',
    )
    assert _levels(rows, "FAIL") == [
        f"[detection.ner] gliner model {str(folder)!r} is a local directory; a revision in"
        " [detection.ner.revisions] applies only to a Hugging Face model id"
    ]


def test_without_huggingface_hub_the_cache_is_not_checked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import fails
    _, rows = _rows(tmp_path, capsys, 'backends = ["gliner", "hf"]')
    assert _levels(rows, "WARN") == [
        f"gliner: {GLINER_SMALL}: the local Hugging Face cache was not checked:"
        " huggingface_hub is not installed (the gliner extra installs it)",
        f"hf: {DSLIM}: the local Hugging Face cache was not checked: huggingface_hub is not"
        " installed (the hf extra installs it)",
    ]


def test_human_output_groups_the_rows_under_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch, FakeHub(uncached={DSLIM}))
    config = tmp_path / "config.toml"
    config.write_text('port = 1\n[detection.ner]\nenabled = true\nbackend = "hf"\n')
    assert run_doctor(argparse.Namespace(config=config)) == 1
    lines: list[Any] = capsys.readouterr().out.splitlines()
    assert any(line.startswith("FAIL  models: [detection.ner] hf model") for line in lines)


# --- what the model catalog says (T14) ---------------------------------------------


def test_a_restricted_model_warns_with_the_catalog_facts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    _, rows = _rows(tmp_path, capsys, 'backend = "gliner"\nmodel = "NVIDIA/GLINER-PII"')
    warnings = _levels(rows, "WARN")
    # Unpinned (the catalog pins no restricted model), and restricted: the
    # lookup ignores letter case, as the Hub does.
    assert warnings[0].startswith(
        "gliner: NVIDIA/GLINER-PII has no pin (model catalog: restricted,"
        " LicenseRef-NVIDIA-Open-Model-License): the newest cached revision"
    )
    assert warnings[1] == (
        "[detection.ner] gliner model 'NVIDIA/GLINER-PII' has model catalog status"
        ' "restricted": nvidia/gliner-PII: NVIDIA Open Model License: not OSI-approved; its'
        " text includes termination clauses; built on urchade/gliner_large-v2.1, trained on"
        " nvidia/Nemotron-PII (CC BY 4.0) (https://huggingface.co/nvidia/gliner-PII, checked"
        " 2026-10-05)"
    )


def test_a_caution_model_shows_its_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    model = "knowledgator/gliner-pii-base-v1.0"
    _, rows = _rows(tmp_path, capsys, f'backend = "gliner"\nmodel = "{model}"')
    (pin,) = [m for level, m in rows if level == "PASS" and "pinned at" in m]
    assert pin.startswith(
        f"gliner: {model} pinned at 61726e0ad791dcab3e29339bbec3ad42ded65641 by the model"
        f" catalog (model catalog: caution, Apache-2.0): {model}: Apache-2.0;"
    )
    assert "not yet measured by the llm-redact bench" in pin
    assert _levels(rows, "WARN") == []


def test_an_older_library_than_the_model_needs_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.metadata

    install_hub(monkeypatch)
    versions = {"transformers": "4.47.1"}

    def version(name: str) -> str:
        if name in versions:
            return versions[name]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    model = "knowledgator/gliner-pii-edge-v1.0"
    code, rows = _rows(tmp_path, capsys, f'backend = "gliner"\nmodel = "{model}"')
    assert code == 1
    assert _levels(rows, "FAIL") == [
        f"gliner: {model} needs transformers >= 4.48.0 (model catalog), but transformers"
        " 4.47.1 is installed; upgrade it: uv sync --extra gliner --upgrade-package"
        " transformers"
    ]
    for newer in ("4.48.0", "4.48.0.dev0+local", "5.0", "10.1.2"):
        versions["transformers"] = newer
        _, rows = _rows(tmp_path, capsys, f'backend = "gliner"\nmodel = "{model}"')
        assert _levels(rows, "FAIL") == [], newer
    del versions["transformers"]  # not installed: the ner check's FAIL
    _, rows = _rows(tmp_path, capsys, f'backend = "gliner"\nmodel = "{model}"')
    assert _levels(rows, "FAIL") == []


def test_version_tuples() -> None:
    from llm_redact.doctor_cli import _version_tuple

    assert _version_tuple("4.48.0") == (4, 48, 0)
    assert _version_tuple("4.48.0rc1") == (4, 48, 0)
    assert _version_tuple("2.6.0+cpu") == (2, 6, 0)
    assert _version_tuple("5") == (5,)
    assert _version_tuple("dev") == ()
