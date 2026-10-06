"""`llm-redact models list|verify`: the configured NER models, offline.

``list`` and ``verify`` read the configuration and local files only (the
fake hub stands in for the local Hugging Face cache and records every
lookup: each one must be ``local_files_only``); ``verify --dir`` checks a
folder against its manifest with no configuration and no hub at all.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from llm_redact.cli import build_parser
from llm_redact.detection.model_catalog import SIDECAR_NAME
from llm_redact.detection.model_manifest import (
    MANIFEST_NAME,
    ManifestModel,
    file_records,
    folder_files,
    manifest_json,
)
from llm_redact.models_cli import run_models
from ner_fakes import DEFAULT_REPO, FakeHub, install_hub

DSLIM = "dslim/bert-base-NER"
DSLIM_PIN = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
GLINER_SMALL = "urchade/gliner_small-v2.1"
GLINER_PIN = "4e091416cf7c3481db542c2a3d26156916f3a47f"
DEBERTA = "microsoft/deberta-v3-small"
DEBERTA_PIN = "a36c739020e01763fe789b4b85e2df55d6180012"
URCHADE_REPO = {
    "gliner_config.json": json.dumps({"model_name": DEBERTA}),
    "pytorch_model.bin": "weights",
}
DEBERTA_REPO = {
    "config.json": json.dumps({"model_type": "deberta-v2"}),
    "tokenizer_config.json": "{}",
    "spm.model": "sentencepiece",
}


def _config(tmp_path: Path, ner: str, extra: str = "") -> Path:
    path = tmp_path / "config.toml"
    path.write_text(f"[detection.ner]\nenabled = true\n{ner}\n{extra}")
    return path


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    args = build_parser().parse_args(["models", *argv])
    code = run_models(args)
    return code, capsys.readouterr().out


def _offline(hub: FakeHub) -> None:
    assert hub.calls, "the cache was not looked at"
    assert all(call["local_files_only"] for call in hub.calls)


# --- list -----------------------------------------------------------------------


def test_list_shows_each_hub_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(
        monkeypatch,
        FakeHub(repos={GLINER_SMALL: URCHADE_REPO, DEBERTA: DEBERTA_REPO}, uncached={DSLIM}),
    )
    config = _config(tmp_path, 'backends = ["gliner", "hf", "spacy"]')
    code, out = _run(capsys, "list", "--config", str(config))
    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == [
        "BACKEND", "MODEL", "REVISION", "CATALOG", "LICENSE", "FILES", "BASE", "MODEL",
    ]  # fmt: skip
    assert lines[1].split() == [
        "gliner", GLINER_SMALL, GLINER_PIN[:12], "vetted", "Apache-2.0", "cached",
        f"{DEBERTA}@{DEBERTA_PIN[:12]}",
    ]  # fmt: skip
    assert lines[2].split() == [
        "hf", DSLIM, DSLIM_PIN[:12], "vetted", "MIT", "missing", "-",
    ]  # fmt: skip
    assert lines[3].startswith(f"hf: [detection.ner] hf model '{DSLIM}' at revision")
    assert lines[4] == (
        "spacy: en_core_web_sm is not a Hugging Face model; install it with:"
        " uv run python -m spacy download en_core_web_sm"
    )
    _offline(hub)


def test_list_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    config = _config(tmp_path, 'backends = ["hf", "stanza"]\nlanguage = "de"')
    code, out = _run(capsys, "list", "--config", str(config), "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["ner_enabled"] is True
    [state] = payload["models"]
    assert state == {
        "backend": "hf",
        "model": DSLIM,
        "source": "hub",
        "model_id": DSLIM,
        "revision": DSLIM_PIN,
        "pinned_by": "catalog",
        "catalog": "vetted",
        "license": "MIT",
        "facts": state["facts"],
        "min_versions": [],
        "backbone": None,
        "backbone_revision": None,
        "files": "cached",
        "file_count": 5,
        "problem": None,
    }
    assert state["facts"].startswith(f"{DSLIM}: MIT;") and "checked 2026-10-05" in state["facts"]
    assert payload["other_models"] == [
        {
            "backend": "stanza",
            "model": "stanza de",
            "install": "uv run python -c \"import stanza; stanza.download('de')\"",
        }
    ]


def test_list_an_unpinned_restricted_model_and_min_versions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    config = _config(
        tmp_path,
        'backends = ["gliner", "hf"]',
        '[detection.ner.models]\ngliner = "knowledgator/gliner-pii-edge-v1.0"\n'
        'hf = "Isotonic/distilbert_finetuned_ai4privacy_v2"\n',
    )
    code, out = _run(capsys, "list", "--config", str(config), "--json")
    gliner, hf = json.loads(out)["models"]
    assert gliner["catalog"] == "caution"
    assert gliner["min_versions"] == [["transformers", "4.48.0"]]
    # Self-contained (its own tokenizer and encoder_config): no base model,
    # although the catalog records which encoder it was built on.
    assert (gliner["backbone"], gliner["files"]) == (None, "cached")
    assert (hf["catalog"], hf["revision"], hf["pinned_by"]) == ("restricted", None, None)
    assert hf["license"] == "CC-BY-NC-4.0"


def test_list_names_the_catalogs_base_model_before_the_files_are_there(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch, FakeHub(uncached={GLINER_SMALL}))
    config = _config(tmp_path, 'backend = "gliner"')
    _, out = _run(capsys, "list", "--config", str(config), "--json")
    [state] = json.loads(out)["models"]
    assert (state["files"], state["backbone"], state["backbone_revision"]) == (
        "missing",
        DEBERTA,
        DEBERTA_PIN,
    )


def test_list_when_ner_is_off_and_no_hub_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(monkeypatch)
    config = tmp_path / "config.toml"
    config.write_text('[detection.ner]\nbackend = "presidio"\nmodel = "de_core_news_sm"\n')
    code, out = _run(capsys, "list", "--config", str(config))
    assert code == 0
    assert out.splitlines() == [
        "NER is off ([detection.ner] enabled = false); the models it would load:",
        "no Hugging Face Hub models configured (backends gliner, hf)",
        "presidio: de_core_news_sm is not a Hugging Face model; install it with:"
        " uv run python -m spacy download de_core_news_sm",
    ]
    assert hub.calls == []


def test_list_without_huggingface_hub(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    config = _config(tmp_path, 'backend = "hf"')
    code, out = _run(capsys, "list", "--config", str(config), "--json")
    [state] = json.loads(out)["models"]
    assert state["files"] == "unchecked"
    assert (
        state["problem"] == "huggingface_hub is not installed (the hf and gliner extras install it)"
    )
    code, out = _run(capsys, "verify", "--config", str(config))
    assert code == 1


def test_an_unreadable_config_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[detection.ner]\nno_such_key = 1\n")
    for command in ("list", "verify"):
        args = build_parser().parse_args(["models", command, "--config", str(config)])
        assert run_models(args) == 2
        assert "llm-redact models:" in capsys.readouterr().err


# --- verify (configuration) --------------------------------------------------------


def test_verify_passes_when_every_model_is_complete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(
        monkeypatch, FakeHub(repos={GLINER_SMALL: URCHADE_REPO, DEBERTA: DEBERTA_REPO})
    )
    config = _config(tmp_path, 'backends = ["gliner", "hf"]')
    code, out = _run(capsys, "verify", "--config", str(config))
    assert code == 0
    assert out.splitlines() == [
        f"OK    gliner: {GLINER_SMALL} at {GLINER_PIN}: 5 files complete in the local Hugging"
        " Face cache",
        f"OK    hf: {DSLIM} at {DSLIM_PIN}: 5 files complete in the local Hugging Face cache",
    ]
    _offline(hub)


@pytest.mark.parametrize(
    ("hub", "fail"),
    [
        # Not cached at its revision.
        (FakeHub(uncached={DSLIM}), "is not (completely) in the local Hugging Face cache"),
        # Cached, but the snapshot an interrupted download left lacks its
        # tokenizer: snapshot_download(local_files_only=True) returns it anyway.
        (
            FakeHub(
                repos={
                    DSLIM: {
                        k: v for k, v in DEFAULT_REPO.items() if "tok" not in k and k != "vocab.txt"
                    }
                }
            ),
            "; missing: tokenizer files",
        ),
        # A GLiNER base model that is not cached.
        (
            FakeHub(repos={GLINER_SMALL: URCHADE_REPO, DEBERTA: DEBERTA_REPO}, uncached={DEBERTA}),
            f"gliner base model '{DEBERTA}'",
        ),
    ],
)
def test_verify_fails_for_a_missing_or_incomplete_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    hub: FakeHub,
    fail: str,
) -> None:
    install_hub(monkeypatch, hub)
    config = _config(tmp_path, 'backends = ["gliner", "hf"]')
    code, out = _run(capsys, "verify", "--config", str(config))
    assert code == 1
    assert any(line.startswith("FAIL  ") and fail in line for line in out.splitlines())
    code, out = _run(capsys, "verify", "--config", str(config), "--json")
    assert code == 1 and json.loads(out)["ok"] is False


def test_verify_a_local_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(monkeypatch)
    folder = tmp_path / "model"
    folder.mkdir()
    for name, text in DEFAULT_REPO.items():
        (folder / name).write_text(text)
    config = _config(tmp_path, f'backend = "hf"\nmodel = {json.dumps(str(folder))}')
    code, out = _run(capsys, "verify", "--config", str(config))
    assert (code, out.splitlines()) == (0, [f"OK    hf: {folder}: 5 files complete in its folder"])
    (folder / "config.json").unlink()
    code, out = _run(capsys, "verify", "--config", str(config))
    assert code == 1
    assert "is a local directory that lacks what the loader needs: config.json" in out
    assert hub.calls == []


def test_verify_with_nothing_to_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.toml"
    config.write_text("port = 1\n")
    code, out = _run(capsys, "verify", "--config", str(config))
    assert code == 0
    assert out.startswith("no Hugging Face Hub models configured")


# --- verify --dir (a pulled folder, against its manifest) ---------------------------


def _pulled(root: Path) -> Path:
    """A folder as `models pull --to` writes it: one hf model folder with
    its sidecar, and the manifest."""
    folder = root / "hf-dslim--bert-base-NER"
    folder.mkdir(parents=True)
    for name, text in DEFAULT_REPO.items():
        if name != "gliner_config.json":
            (folder / name).write_text(text)
    (folder / SIDECAR_NAME).write_text(json.dumps({"model_id": DSLIM, "revision": DSLIM_PIN}))
    records = file_records(folder, folder_files(folder))
    model = ManifestModel(
        backend="hf",
        model_id=DSLIM,
        revision=DSLIM_PIN,
        folder=folder.name,
        files=tuple((r["path"], r["size"], r["sha256"]) for r in records),
    )
    (root / MANIFEST_NAME).write_text(json.dumps(manifest_json([model], "test")))
    return folder


def test_verify_dir_passes_on_an_intact_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # no hub needed at all
    _pulled(tmp_path / "models")
    code, out = _run(capsys, "verify", "--dir", str(tmp_path / "models"))
    assert code == 0
    assert out.splitlines() == [
        f"OK    hf: {DSLIM} at {DSLIM_PIN}: hf-dslim--bert-base-NER (6 files) matches"
        f" {MANIFEST_NAME}"
    ]


def _flip_one_byte(path: Path) -> None:
    data = bytearray(path.read_bytes() or b"x")
    data[0] ^= 1
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(
    ("damage", "problem"),
    [
        (lambda f: _flip_one_byte(f / "config.json"), "config.json: SHA-256 differs"),
        (lambda f: (f / "config.json").write_text("{}x"), "config.json: size differs"),
        (lambda f: (f / "vocab.txt").unlink(), "vocab.txt: missing"),
        (lambda f: (f / "pytorch_model.bin").write_text(""), "pytorch_model.bin: not listed"),
        (lambda f: shutil.rmtree(f), "hf-dslim--bert-base-NER: the folder is missing"),
    ],
)
def test_verify_dir_fails_on_a_changed_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], damage: Any, problem: str
) -> None:
    folder = _pulled(tmp_path)
    damage(folder)
    code, out = _run(capsys, "verify", "--dir", str(tmp_path))
    assert code == 1
    assert problem in out
    code, out = _run(capsys, "verify", "--dir", str(tmp_path), "--json")
    assert code == 1
    assert any(problem in line for line in json.loads(out)["models"][0]["problems"])


def test_verify_dir_checks_the_sidecar_and_what_the_loader_needs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _pulled(tmp_path)
    # A manifest naming another revision than the folder's sidecar.
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    manifest["models"][0]["revision"] = "0" * 40
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    code, out = _run(capsys, "verify", "--dir", str(tmp_path))
    assert code == 1
    assert f"{SIDECAR_NAME}: names another model or revision than the manifest" in out
    # A manifest that faithfully lists an incomplete folder.
    (folder / "vocab.txt").unlink()
    (folder / "tokenizer.json").unlink()
    _pulled(tmp_path / "again")
    again = tmp_path / "again" / folder.name
    for name in ("vocab.txt", "tokenizer.json"):
        (again / name).unlink()
    records = file_records(again, folder_files(again))
    manifest = json.loads((tmp_path / "again" / MANIFEST_NAME).read_text())
    manifest["models"][0]["files"] = records
    (tmp_path / "again" / MANIFEST_NAME).write_text(json.dumps(manifest))
    code, out = _run(capsys, "verify", "--dir", str(tmp_path / "again"))
    assert code == 1
    assert "is a local directory that lacks what the loader needs: tokenizer files" in out


@pytest.mark.parametrize(
    ("manifest", "problem"),
    [
        (None, f"no {MANIFEST_NAME} in"),
        ("not json", "is not a UTF-8 JSON document"),
        ({"manifest": "other"}, "is not an llm-redact models manifest"),
        ({"manifest": "llm-redact-models", "schema": 2, "models": []}, "schema 2 is not one"),
        ({"manifest": "llm-redact-models", "schema": 1, "models": {}}, "models must be a list"),
        (
            {"manifest": "llm-redact-models", "schema": 1, "models": [{"backend": "spacy"}]},
            "models[0].backend must be one of gliner, hf",
        ),
        (
            {
                "manifest": "llm-redact-models",
                "schema": 1,
                "models": [{"backend": "hf", "model_id": DSLIM, "folder": "../up", "files": []}],
            },
            "models[0].folder must be one plain folder name",
        ),
        (
            {
                "manifest": "llm-redact-models",
                "schema": 1,
                "models": [
                    {
                        "backend": "hf",
                        "model_id": DSLIM,
                        "folder": "f",
                        "files": [{"path": "/etc/passwd", "size": 1, "sha256": "0" * 64}],
                    }
                ],
            },
            "models[0].files[0].path must be a relative path inside the folder",
        ),
    ],
)
def test_verify_dir_refuses_a_manifest_it_cannot_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], manifest: object, problem: str
) -> None:
    if manifest is not None:
        text = manifest if isinstance(manifest, str) else json.dumps(manifest)
        (tmp_path / MANIFEST_NAME).write_text(text)
    code, out = _run(capsys, "verify", "--dir", str(tmp_path))
    assert code == 1
    assert out.startswith("FAIL  ") and problem in out
    code, out = _run(capsys, "verify", "--dir", str(tmp_path), "--json")
    payload = json.loads(out)
    assert payload["ok"] is False and problem in payload["problem"]


def test_verify_dir_notes_entries_outside_the_model_folders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _pulled(tmp_path)
    (tmp_path / "lost+found").mkdir()
    code, out = _run(capsys, "verify", "--dir", str(tmp_path))
    assert code == 0  # the loader reads only the model folders
    assert out.splitlines()[-1] == (
        f"note: {tmp_path} also holds entries the manifest does not list: lost+found"
    )
    code, out = _run(capsys, "verify", "--dir", str(tmp_path), "--json")
    assert json.loads(out)["unlisted_entries"] == ["lost+found"]


def test_the_parser_takes_the_models_command(tmp_path: Path) -> None:
    args = build_parser().parse_args(["models", "verify", "--dir", str(tmp_path)])
    assert (args.models_command, args.dir) == ("verify", tmp_path)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["models"])
    assert isinstance(args, argparse.Namespace)
