"""NER model files come from a local directory at a pinned revision.

``detection/model_files.py`` resolves a configured model to a local folder
(``huggingface_hub.snapshot_download`` with explicit top-level file names,
from the local cache unless ``allow_download``), refuses a configuration
that names code from the model repository, and — for the ``hf`` backend —
requires safetensors weights unless ``allow_pickle_weights``. The fakes
(ner_fakes.FakeHub, a fake transformers pipeline) never touch the network.
"""

from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from llm_redact.config import ConfigError
from llm_redact.detection import model_catalog
from llm_redact.detection.engine import NerConfig
from llm_redact.detection.hf_ner import build_hf_detector, catalog_window
from llm_redact.detection.model_catalog import SIDECAR_NAME, CatalogEntry
from llm_redact.detection.model_files import (
    HF_PATTERNS,
    HF_PICKLE_PATTERNS,
    check_configs,
    read_config,
    resolve_model,
)
from ner_fakes import FakeHfPipe, FakeHub, FakeTokenizer, install_hub, install_transformers

DSLIM = "dslim/bert-base-NER"
DSLIM_PIN = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
OTHER = "0123456789abcdef0123456789abcdef01234567"

# An hf repository with every kind of file a Hub model repo can hold.
FULL_REPO = {
    "config.json": json.dumps({"model_type": "bert"}),
    "model.safetensors": "",
    "pytorch_model.bin": "",
    "tf_model.h5": "",
    "flax_model.msgpack": "",
    "onnx/model.onnx": "",
    "original/pytorch_model.bin": "",
    "tokenizer_config.json": "{}",
    "vocab.txt": "",
    "README.md": "",
}
PICKLE_ONLY = {
    "config.json": json.dumps({"model_type": "bert"}),
    "pytorch_model.bin": "",
    "tokenizer.json": "{}",
}


def _build(
    monkeypatch: pytest.MonkeyPatch, hub: FakeHub | None = None, **ner: Any
) -> tuple[FakeHfPipe, FakeHub]:
    pipe = FakeHfPipe([("Jane Doe", "PER", 0.9)])
    hub = hub if hub is not None else FakeHub()
    install_transformers(monkeypatch, pipe, hub)
    build_hf_detector(NerConfig(enabled=True, backend="hf", **ner))
    return pipe, hub


def _files(folder: str | Path) -> set[str]:
    root = Path(folder)
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


# --- the Hub: pinned revision, explicit file names, local only ------------------


def test_patterns_name_top_level_files_only() -> None:
    def fetched(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pattern) for pattern in HF_PATTERNS)

    assert all(fetched(name) for name in ("config.json", "model.safetensors", "vocab.txt"))
    assert fetched("model-00001-of-00002.safetensors")
    assert fetched("model.safetensors.index.json")
    for other in (
        "pytorch_model.bin",
        "tf_model.h5",
        "flax_model.msgpack",
        "onnx/model.onnx",
        "original/pytorch_model.bin",
        "README.md",
    ):
        assert not fetched(other), other
    assert all("/" not in pattern for pattern in (*HF_PATTERNS, *HF_PICKLE_PATTERNS))


def test_the_default_model_loads_at_its_catalog_pin_from_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = FakeHub(default=FULL_REPO)
    pipe, _ = _build(monkeypatch, hub)
    assert hub.calls == [
        {
            "repo_id": DSLIM,
            "revision": DSLIM_PIN,
            "allow_patterns": list(HF_PATTERNS),
            "local_files_only": True,  # allow_download defaults to false (D2)
        }
    ]
    folder = pipe.built_with[0]["model"]
    # Only the names asked for: no pickle, TensorFlow, Flax or ONNX copy.
    assert _files(folder) == {
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.txt",
    }
    assert pipe.built_with[0]["trust_remote_code"] is False
    assert pipe.built_with[0]["model_kwargs"] == {"use_safetensors": True}
    assert pipe.built_with[0]["tokenizer"] == folder


def test_a_configured_revision_wins_and_downloads_follow_allow_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, hub = _build(
        monkeypatch, revisions=(("hf", OTHER),), allow_download=True, model="org/ner-model"
    )
    assert [(c["repo_id"], c["revision"], c["local_files_only"]) for c in hub.calls] == [
        ("org/ner-model", OTHER, False)
    ]


def test_an_uncatalogued_model_without_a_revision_is_looked_up_unpinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, hub = _build(monkeypatch, model="org/ner-model")
    assert hub.calls[0]["revision"] is None


def test_a_model_missing_from_the_cache_names_models_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub(uncached={DSLIM})
    with pytest.raises(ConfigError) as caught:
        _build(monkeypatch, hub)
    assert str(caught.value) == (
        f"[detection.ner] hf model '{DSLIM}' at revision {DSLIM_PIN} is not (completely) in"
        " the local Hugging Face cache, and downloads are off (an older revision in the"
        " cache does not count); run `llm-redact models pull`, or set [detection.ner]"
        " allow_download = true to fetch it at startup"
    )


def test_a_failed_download_names_the_exception_type(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub(default=None)  # no such repository
    with pytest.raises(ConfigError) as caught:
        _build(monkeypatch, hub, model="org/missing", allow_download=True)
    assert str(caught.value) == (
        "[detection.ner] hf model 'org/missing' (no revision pinned) could not be fetched"
        " from the Hugging Face Hub: NotCached"
    )


def test_a_value_that_is_no_directory_and_no_model_id_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = FakeHub()
    with pytest.raises(ConfigError, match="is neither a local directory nor a Hugging Face"):
        _build(monkeypatch, hub, model="not a model id")
    assert hub.calls == []


def test_without_huggingface_hub_the_extra_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import fails
    with pytest.raises(ConfigError, match="needs huggingface_hub, which the hf and gliner"):
        resolve_model(
            DSLIM, what="hf model", revision=None, allow_download=False, allow_patterns=HF_PATTERNS
        )


# --- safetensors (D3) ------------------------------------------------------------


def test_a_pickle_only_model_is_refused_without_the_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub(default=PICKLE_ONLY)
    pipe = FakeHfPipe([])
    install_transformers(monkeypatch, pipe, hub)
    with pytest.raises(ConfigError) as caught:
        build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/pickle-ner"))
    assert str(caught.value) == (
        "[detection.ner] hf model 'org/pickle-ner' has no safetensors weights;"
        " set allow_pickle_weights = true to load pytorch_model.bin"
    )
    assert pipe.built_with == []  # nothing was loaded
    assert len(hub.calls) == 1  # and the pickle was never fetched


def test_the_hatch_fetches_and_loads_the_pickle_of_a_pickle_only_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe, hub = _build(
        monkeypatch, FakeHub(default=PICKLE_ONLY), model="org/pickle-ner", allow_pickle_weights=True
    )
    assert [c["allow_patterns"] for c in hub.calls] == [
        list(HF_PATTERNS),
        [*HF_PATTERNS, *HF_PICKLE_PATTERNS],
    ]
    assert "pytorch_model.bin" in _files(pipe.built_with[0]["model"])
    # None lets transformers read the pickle (False would skip safetensors).
    assert pipe.built_with[0]["model_kwargs"] == {"use_safetensors": None}


def test_safetensors_load_even_with_the_hatch_on(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, hub = _build(monkeypatch, FakeHub(default=FULL_REPO), allow_pickle_weights=True)
    assert len(hub.calls) == 1
    assert "pytorch_model.bin" not in _files(pipe.built_with[0]["model"])
    assert pipe.built_with[0]["model_kwargs"] == {"use_safetensors": True}


# --- local directories ------------------------------------------------------------


def _folder(path: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(content)
    return path


def test_a_local_directory_is_used_as_it_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = _folder(tmp_path, FULL_REPO)
    pipe, hub = _build(monkeypatch, model=str(folder))
    assert hub.calls == []
    assert pipe.built_with[0]["model"] == str(folder)
    assert pipe.built_with[0]["model_kwargs"] == {"use_safetensors": True}


def test_a_local_pickle_only_directory_needs_the_hatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = _folder(tmp_path, PICKLE_ONLY)
    with pytest.raises(ConfigError, match="has no safetensors weights"):
        _build(monkeypatch, model=str(folder))
    pipe, hub = _build(monkeypatch, model=str(folder), allow_pickle_weights=True)
    assert hub.calls == []
    assert pipe.built_with[0]["model_kwargs"] == {"use_safetensors": None}


def test_a_revision_for_a_local_directory_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = _folder(tmp_path, FULL_REPO)
    with pytest.raises(ConfigError) as caught:
        _build(monkeypatch, model=str(folder), revisions=(("hf", OTHER),))
    assert str(caught.value) == (
        f"[detection.ner] hf model {str(folder)!r} is a local directory; a revision in"
        " [detection.ner.revisions] applies only to a Hugging Face model id"
    )


def test_a_directory_named_like_a_catalogued_id_keeps_the_catalog_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The catalog pins the id; a directory of that name is not "configured
    # with a revision" (transformers would load the directory too).
    monkeypatch.chdir(tmp_path)
    _folder(tmp_path / DSLIM, FULL_REPO)
    path = resolve_model(
        DSLIM,
        what="hf model",
        revision=DSLIM_PIN,
        allow_download=False,
        allow_patterns=HF_PATTERNS,
    )
    assert path == Path(DSLIM)


# --- no model code ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("config.json", {"model_type": "bert", "auto_map": {"AutoModel": "modeling.Model"}}),
        ("tokenizer_config.json", {"auto_map": {"AutoTokenizer": ["tok.Tokenizer", None]}}),
        ("config.json", {"model_type": "bert", "text_config": [{"auto_map": {}}]}),
    ],
)
def test_a_configuration_naming_model_code_is_refused(
    monkeypatch: pytest.MonkeyPatch, name: str, config: dict[str, Any]
) -> None:
    repo = {**FULL_REPO, name: json.dumps(config)}
    pipe = FakeHfPipe([])
    install_transformers(monkeypatch, pipe, FakeHub(default=repo))
    with pytest.raises(ConfigError) as caught:
        build_hf_detector(NerConfig(enabled=True, backend="hf"))
    assert str(caught.value) == (
        f"[detection.ner] hf model '{DSLIM}' needs code from its repository ({name} names"
        " auto_map); llm-redact never runs model code"
    )
    assert pipe.built_with == []


@pytest.mark.parametrize(
    "content",
    [b"not json", b"\xff\xfe", b"[1, 2]", b"{" + b" " * (4 * 1024 * 1024) + b"}"],
)
def test_a_configuration_that_is_no_json_object_is_refused(tmp_path: Path, content: bytes) -> None:
    (tmp_path / "config.json").write_bytes(content)
    with pytest.raises(ConfigError) as caught:
        check_configs(tmp_path, ["config.json"], what="hf model", model="org/x")
    assert str(caught.value) == (
        "[detection.ner] hf model 'org/x': config.json is not a JSON configuration"
    )


def test_an_unreadable_configuration_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").mkdir()  # opening a directory fails
    with pytest.raises(ConfigError, match=r"config.json cannot be read \("):
        read_config(tmp_path / "config.json", what="hf model", model="org/x")


# --- the catalog window -------------------------------------------------------------


def _windowed_entry(model_id: str, backends: tuple[str, ...]) -> CatalogEntry:
    return CatalogEntry(
        model_id=model_id,
        backends=backends,
        license="MIT",
        status="vetted",
        reason="test",
        window=128,
    )


def test_the_catalog_window_sizes_the_pipeline_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entry = _windowed_entry("org/windowed-ner", ("hf",))
    monkeypatch.setattr(model_catalog, "lookup", lambda model_id: entry)
    pipe = FakeHfPipe([], tokenizer=FakeTokenizer(model_max_length=512))
    install_transformers(monkeypatch, pipe)
    build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/windowed-ner"))
    assert pipe.tokenizer.model_max_length == 128
    assert pipe.built_with[1]["stride"] == 32


def test_a_local_directory_takes_the_window_of_the_model_its_sidecar_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = _folder(tmp_path, FULL_REPO)
    (folder / SIDECAR_NAME).write_text(json.dumps({"model_id": "org/windowed-ner"}))
    seen: list[str] = []

    def lookup(model_id: str) -> CatalogEntry:
        seen.append(model_id)
        return _windowed_entry(model_id, ("hf",))

    monkeypatch.setattr(model_catalog, "lookup", lookup)
    assert catalog_window(str(folder)) == 128
    assert seen == ["org/windowed-ner"]


def test_a_window_catalogued_for_another_backend_or_no_entry_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(model_catalog, "lookup", lambda m: _windowed_entry(m, ("gliner",)))
    assert catalog_window("org/gliner-model") is None
    monkeypatch.setattr(model_catalog, "lookup", lambda m: None)
    assert catalog_window("org/unknown") is None
    # A local directory without a sidecar is not identified.
    assert catalog_window(str(tmp_path)) is None


def test_a_malformed_sidecar_is_a_config_error(tmp_path: Path) -> None:
    (tmp_path / SIDECAR_NAME).write_text("[]")
    with pytest.raises(ConfigError, match=rf"\[detection.ner\] hf model: .*{SIDECAR_NAME}"):
        catalog_window(str(tmp_path))


def test_install_hub_returns_the_installed_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = install_hub(monkeypatch)
    assert sys.modules["huggingface_hub"].snapshot_download == hub.snapshot_download
