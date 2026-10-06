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


# --- pull ------------------------------------------------------------------------------

HEAD = "fedcba9876543210fedcba9876543210fedcba98"


def _pull_hub() -> FakeHub:
    return FakeHub(repos={GLINER_SMALL: URCHADE_REPO, DEBERTA: DEBERTA_REPO})


def test_pull_fetches_each_model_at_its_pin_with_the_loader_patterns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact.detection.model_files import BACKBONE_PATTERNS, GLINER_PATTERNS, HF_PATTERNS

    hub = install_hub(monkeypatch, _pull_hub())
    config = _config(tmp_path, 'backends = ["gliner", "hf"]')
    code, out = _run(capsys, "pull", "--config", str(config))
    assert code == 0
    assert [(c["repo_id"], c["revision"], c["allow_patterns"]) for c in hub.calls] == [
        (GLINER_SMALL, GLINER_PIN, list(GLINER_PATTERNS)),
        (GLINER_SMALL, GLINER_PIN, [*GLINER_PATTERNS, "pytorch_model.bin"]),
        (DEBERTA, DEBERTA_PIN, list(BACKBONE_PATTERNS)),
        (DSLIM, DSLIM_PIN, list(HF_PATTERNS)),
    ]
    assert not any(c["local_files_only"] for c in hub.calls)  # pull downloads
    assert out.splitlines() == [
        f"OK    gliner: {GLINER_SMALL} at {GLINER_PIN}: 5 files; base model {DEBERTA} at"
        f" {DEBERTA_PIN}",
        f"OK    hf: {DSLIM} at {DSLIM_PIN}: 5 files",
    ]
    # With the cache filled, the proxy's own offline check passes.
    code, _ = _run(capsys, "verify", "--config", str(config))
    assert code == 0


def test_pull_names_the_commit_of_an_unpinned_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(monkeypatch, FakeHub(heads={"org/ner": HEAD}))
    config = _config(tmp_path, 'backend = "hf"\nmodel = "org/ner"')
    code, out = _run(capsys, "pull", "--config", str(config))
    assert code == 0
    assert [c["revision"] for c in hub.calls] == [None]
    assert out.splitlines() == [
        f"note: hf: org/ner has no pin; its default branch is at {HEAD}. Pin it:"
        f' [detection.ner.revisions] hf = "{HEAD}"',
        f"OK    hf: org/ner at {HEAD}: 5 files",
    ]
    # A cache that does not name the commit: said so, never guessed.
    install_hub(monkeypatch, FakeHub())
    code, out = _run(capsys, "pull", "--config", str(config))
    assert code == 0
    assert out.splitlines()[0] == (
        "note: hf: org/ner has no pin, and the commit pulled could not be told; pin one in"
        " [detection.ner.revisions] hf"
    )


def test_pull_warns_for_a_restricted_model_and_an_unpinned_base_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(
        repos={"nvidia/gliner-PII": URCHADE_REPO, DEBERTA: DEBERTA_REPO},
        heads={"nvidia/gliner-PII": HEAD, DEBERTA: DEBERTA_PIN},
    )
    install_hub(monkeypatch, hub)
    config = _config(tmp_path, 'backend = "gliner"\nmodel = "nvidia/gliner-PII"')
    code, out = _run(capsys, "pull", "--config", str(config))
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith(
        "warning: [detection.ner] gliner model 'nvidia/gliner-PII' has model catalog status"
        ' "restricted": nvidia/gliner-PII: NVIDIA Open Model License: not OSI-approved;'
    )
    assert lines[2] == (
        f"note: gliner: nvidia/gliner-PII: the model catalog pins no revision of its base model"
        f" {DEBERTA}; pulled {DEBERTA_PIN}"
    )
    assert lines[3] == (
        f"OK    gliner: nvidia/gliner-PII at {HEAD}: 5 files; base model {DEBERTA} at {DEBERTA_PIN}"
    )


def _manifest(root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((root / MANIFEST_NAME).read_text())
    return payload


def test_pull_to_writes_portable_folders_a_manifest_and_the_snippet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from llm_redact import __version__

    install_hub(monkeypatch, _pull_hub())
    out_dir = tmp_path / "carry"
    config = _config(
        tmp_path, 'backends = ["gliner", "hf"]', f'[detection.ner.revisions]\nhf = "{DSLIM_PIN}"\n'
    )
    code, out = _run(capsys, "pull", "--config", str(config), "--to", str(out_dir))
    assert code == 0
    gliner_dir = out_dir / "gliner-urchade--gliner_small-v2.1"
    hf_dir = out_dir / "hf-dslim--bert-base-NER"
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(
        [gliner_dir.name, hf_dir.name, MANIFEST_NAME]
    )
    # The assembled GLiNER folder: weights, the base model's tokenizer and
    # configuration, a gliner_config.json naming no absolute path.
    assert folder_files(gliner_dir) == sorted(
        ["config.json", "gliner_config.json", "pytorch_model.bin", "spm.model",
         "tokenizer_config.json", SIDECAR_NAME]
    )  # fmt: skip
    written = json.loads((gliner_dir / "gliner_config.json").read_text())
    assert written["model_name"] == "." and written["encoder_config"] == {
        "model_type": "deberta-v2"
    }
    assert not any(path.is_symlink() for path in gliner_dir.iterdir())
    assert json.loads((gliner_dir / SIDECAR_NAME).read_text()) == {
        "model_id": GLINER_SMALL,
        "revision": GLINER_PIN,
    }
    manifest = _manifest(out_dir)
    assert (manifest["manifest"], manifest["schema"], manifest["llm_redact"]) == (
        "llm-redact-models",
        1,
        __version__,
    )
    gliner, hf = manifest["models"]
    assert {k: gliner[k] for k in ("backend", "model_id", "revision", "folder")} == {
        "backend": "gliner",
        "model_id": GLINER_SMALL,
        "revision": GLINER_PIN,
        "folder": gliner_dir.name,
    }
    assert (gliner["backbone"], gliner["backbone_revision"], gliner["onnx"]) == (
        DEBERTA,
        DEBERTA_PIN,
        None,
    )
    assert (hf["backbone"], hf["backbone_revision"]) == (None, None)
    for record in hf["files"]:
        data = (hf_dir / record["path"]).read_bytes()
        assert record["size"] == len(data)
        assert record["sha256"] == hashlib.sha256(data).hexdigest()
    assert [r["path"] for r in hf["files"]] == folder_files(hf_dir)
    # The snippet: local paths, and the revisions a folder no longer takes.
    lines = out.splitlines()
    snippet = lines[lines.index("[detection.ner.models]") :]
    assert snippet[1:3] == [
        f"gliner = {json.dumps(str(gliner_dir.resolve()))}",
        f"hf = {json.dumps(str(hf_dir.resolve()))}",
    ]
    assert any(
        line.startswith("and remove these backends' [detection.ner.revisions] entries")
        and line.endswith(": hf")
        for line in lines
    )
    code, verified = _run(capsys, "verify", "--dir", str(out_dir))
    assert code == 0, verified


def test_pull_to_as_prints_the_mounted_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    config = _config(tmp_path, 'backend = "hf"')
    code, out = _run(
        capsys, "pull", "--config", str(config), "--to", str(tmp_path / "out"), "--as", "/models/"
    )
    assert code == 0
    assert 'hf = "/models/hf-dslim--bert-base-NER"' in out.splitlines()
    assert (
        out.splitlines()[-1]
        == "check the folder where it is used with: llm-redact models verify --dir /models/"
    )
    assert "remove these backends" not in out


def test_a_pulled_folder_loads_with_every_network_call_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # AD11: pull on a connected machine, carry the folder, load it in an
    # enclave whose hub answers nothing (allow_download unset).
    from llm_redact.detection.engine import NerConfig
    from llm_redact.detection.gliner_ner import build_gliner_detector
    from llm_redact.detection.hf_ner import build_hf_detector
    from ner_fakes import FakeGliner, FakeHfPipe, install_gliner, install_transformers

    install_hub(monkeypatch, _pull_hub())
    out_dir = tmp_path / "carry"
    config = _config(tmp_path, 'backends = ["gliner", "hf"]', "[detection.ner.onnx]\n")
    code, _ = _run(capsys, "pull", "--config", str(config), "--to", str(out_dir))
    assert code == 0
    shutil.rmtree(sys.modules["huggingface_hub"].fake_hub.root)  # the cache is gone
    offline = FakeHub(default=None)  # and every hub call fails
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "no-assembly"))
    pipe = FakeHfPipe([])
    install_transformers(monkeypatch, pipe, offline)
    hf_folder = str(out_dir / "hf-dslim--bert-base-NER")
    build_hf_detector(NerConfig(enabled=True, backend="hf", model=hf_folder))
    assert pipe.built_with[0]["model"] == hf_folder
    model = FakeGliner([])
    install_gliner(monkeypatch, model, offline)
    gliner_folder = str(out_dir / "gliner-urchade--gliner_small-v2.1")
    build_gliner_detector(NerConfig(enabled=True, backend="gliner", model=gliner_folder))
    # Loaded from the carried folder itself: self-contained, nothing assembled.
    assert model.loaded_with[0]["model_id"] == gliner_folder
    assert offline.calls == []
    assert not (tmp_path / "no-assembly").exists()


def test_pull_to_carries_the_onnx_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = {
        "gliner_config.json": json.dumps({"model_name": DEBERTA, "encoder_config": {}}),
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "model.safetensors": "torch",
        "onnx/model_quint8.onnx": "int8",
    }
    model_id = "knowledgator/gliner-pii-base-v1.0"
    install_hub(monkeypatch, FakeHub(repos={model_id: repo}))
    config = _config(
        tmp_path,
        f'backend = "gliner"\nmodel = "{model_id}"',
        '[detection.ner.onnx]\ngliner = "onnx/model_quint8.onnx"\n',
    )
    out_dir = tmp_path / "carry"
    code, _ = _run(capsys, "pull", "--config", str(config), "--to", str(out_dir))
    assert code == 0
    folder = out_dir / "gliner-knowledgator--gliner-pii-base-v1.0"
    assert folder_files(folder) == sorted(
        [
            "gliner_config.json",
            "onnx/model_quint8.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            SIDECAR_NAME,
        ]
    )  # fmt: skip  (the ONNX file replaces the torch weights)
    assert _manifest(out_dir)["models"][0]["onnx"] == "onnx/model_quint8.onnx"
    code, _ = _run(capsys, "verify", "--dir", str(out_dir))
    assert code == 0


def test_pull_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # One model is not on the hub: nothing is written to --to.
    install_hub(monkeypatch, FakeHub(repos={DSLIM: DEFAULT_REPO}, default=None))
    out_dir = tmp_path / "carry"
    config = _config(tmp_path, 'backends = ["gliner", "hf"]')
    code, out = _run(capsys, "pull", "--config", str(config), "--to", str(out_dir))
    assert code == 1
    assert out.splitlines()[0] == (
        f"FAIL  [detection.ner] gliner model '{GLINER_SMALL}' at revision {GLINER_PIN} could"
        " not be fetched from the Hugging Face Hub: NotCached"
    )
    assert (
        out.splitlines()[-1]
        == f"FAIL  not every model was pulled; nothing was written to {out_dir}"
    )
    assert not out_dir.exists()
    code, out = _run(capsys, "pull", "--config", str(config))
    assert code == 1 and "nothing was written" not in out


def test_pull_to_never_replaces_a_folder_it_did_not_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    out_dir = tmp_path / "carry"
    config = _config(tmp_path, 'backend = "hf"')
    assert _run(capsys, "pull", "--config", str(config), "--to", str(out_dir))[0] == 0
    marker = out_dir / "hf-dslim--bert-base-NER" / "config.json"
    marker.write_text("changed")
    # A folder it wrote (it holds the sidecar) is replaced whole.
    assert _run(capsys, "pull", "--config", str(config), "--to", str(out_dir))[0] == 0
    assert marker.read_text() != "changed"
    assert not any(p.name.endswith(".partial") for p in out_dir.iterdir())
    (out_dir / "hf-dslim--bert-base-NER" / SIDECAR_NAME).unlink()
    code, out = _run(capsys, "pull", "--config", str(config), "--to", str(out_dir))
    assert code == 1
    assert out.splitlines()[-1] == (
        f"FAIL  {out_dir / 'hf-dslim--bert-base-NER'} exists and is not a folder `llm-redact"
        " models pull --to` wrote; remove it or choose another --to"
    )


def test_pull_cannot_write_its_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_hub(monkeypatch)
    blocked = tmp_path / "a-file"
    blocked.write_text("")
    config = _config(tmp_path, 'backend = "hf"')
    code, out = _run(capsys, "pull", "--config", str(config), "--to", str(blocked / "out"))
    assert code == 1
    assert out.splitlines()[-1].startswith(
        f"FAIL  cannot write the model folders to {blocked / 'out'} ("
    )


def test_pull_skips_local_folders_and_other_backends(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = install_hub(monkeypatch)
    folder = tmp_path / "model"
    folder.mkdir()
    config = _config(tmp_path, f'backends = ["hf", "spacy"]\nmodel = {json.dumps(str(folder))}')
    # The legacy single `model` key applies only with one backend active.
    config.write_text(
        '[detection.ner]\nenabled = false\nbackends = ["hf", "spacy"]\n'
        f"[detection.ner.models]\nhf = {json.dumps(str(folder))}\n"
    )
    code, out = _run(capsys, "pull", "--config", str(config))
    assert code == 0
    assert out.splitlines() == [
        "NER is off ([detection.ner] enabled = false); pulling the models it would load",
        "spacy: en_core_web_sm is not a Hugging Face model; install it with:"
        " uv run python -m spacy download en_core_web_sm",
        f"skip  hf: {folder} is a local folder; nothing to pull",
    ]
    assert hub.calls == []
    (folder / SIDECAR_NAME).write_text("{")
    code, out = _run(capsys, "pull", "--config", str(config))
    assert code == 1
    assert out.splitlines()[-1] == f"FAIL  hf: {folder / SIDECAR_NAME}: not a UTF-8 JSON document"


def test_pull_with_nothing_to_pull_or_no_hub_library(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("port = 1\n")
    code, out = _run(capsys, "pull", "--config", str(config))
    assert code == 0 and out.splitlines()[-1].startswith("no Hugging Face Hub models configured")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    hf = _config(tmp_path, 'backend = "hf"')
    code, out = _run(capsys, "pull", "--config", str(hf))
    assert code == 1
    assert out.startswith("FAIL  pulling needs huggingface_hub")


def test_pull_argument_and_config_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path, 'backend = "hf"')
    args = build_parser().parse_args(["models", "pull", "--config", str(config), "--as", "/m"])
    assert run_models(args) == 2
    assert capsys.readouterr().err.strip() == "llm-redact models pull: --as needs --to"
    config.write_text("[detection.ner]\nbogus = 1\n")
    args = build_parser().parse_args(["models", "pull", "--config", str(config)])
    assert run_models(args) == 2


def test_folder_names_and_snapshot_commits(tmp_path: Path) -> None:
    from llm_redact.detection.model_manifest import safe_path
    from llm_redact.models_cli import folder_name, snapshot_commit

    assert folder_name("hf", DSLIM) == "hf-dslim--bert-base-NER"
    assert folder_name("gliner", "a/b_c.d-e") == "gliner-a--b_c.d-e"
    assert all(
        safe_path(folder_name(b, m), folder=True) for b, m in (("hf", DSLIM), ("gliner", "x"))
    )
    assert snapshot_commit(tmp_path / DSLIM_PIN) == DSLIM_PIN
    assert snapshot_commit(tmp_path / "main") is None
    assert snapshot_commit(None) is None
