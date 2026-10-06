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
import re
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
        " allow_download = true to fetch it at startup (a reload never downloads)"
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
    # Short ids: pytest puts the id in PYTEST_CURRENT_TEST, and Windows refuses
    # an environment variable over 32,767 characters.
    ids=["prose", "utf16-bom", "array", "4mib-object"],
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


# --- GLiNER: pinned base model, self-contained folder (D13) ----------------------

GLINER_SMALL = "urchade/gliner_small-v2.1"
GLINER_PIN = "4e091416cf7c3481db542c2a3d26156916f3a47f"
DEBERTA = "microsoft/deberta-v3-small"
DEBERTA_PIN = "a36c739020e01763fe789b4b85e2df55d6180012"
DEBERTA_CONFIG = {"model_type": "deberta-v2", "hidden_size": 768, "vocab_size": 128100}
# urchade's v2.1 layout: configuration and a pickle, no tokenizer, no
# encoder_config (the base model supplies both).
URCHADE_REPO = {
    "gliner_config.json": json.dumps({"model_name": DEBERTA, "max_len": 384}),
    "pytorch_model.bin": "weights",
    "README.md": "",
}
DEBERTA_REPO = {
    "config.json": json.dumps(DEBERTA_CONFIG),
    "tokenizer_config.json": json.dumps({"do_lower_case": False}),
    "spm.model": "sentencepiece",
    "pytorch_model.bin": "backbone weights",
    "tf_model.h5": "",
}
# A self-contained checkpoint (Knowledgator's layout).
SELF_CONTAINED = {
    "gliner_config.json": json.dumps(
        {"model_name": DEBERTA, "encoder_config": {"model_type": "deberta-v2"}}
    ),
    "pytorch_model.bin": "",
    "tokenizer.json": "{}",
    "tokenizer_config.json": "{}",
    "onnx/model.onnx": "",
}


class StrictGliner:
    """GLiNER.from_pretrained as gliner 0.2.28 decides: the tokenizer comes
    from the folder only when it holds tokenizer_config.json, the encoder
    configuration only from an embedded encoder_config — otherwise GLiNER
    would ask the Hub for model_name, which this stand-in refuses."""

    def __init__(self) -> None:
        self.loaded: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, name: str, **kwargs: Any) -> Any:
        folder = Path(name)
        config = json.loads((folder / "gliner_config.json").read_text())
        if not (folder / "tokenizer_config.json").is_file():
            raise OSError(f"would fetch the tokenizer of {config['model_name']}")
        if not isinstance(config.get("encoder_config"), dict):
            raise OSError(f"would fetch the configuration of {config['model_name']}")
        weights = [w for w in ("model.safetensors", "pytorch_model.bin") if (folder / w).is_file()]
        if not weights and not kwargs.get("load_onnx_model"):
            raise FileNotFoundError("no model file")
        self.loaded.append((name, kwargs))
        from ner_fakes import FakeGliner

        return FakeGliner([])


def _gliner(monkeypatch: pytest.MonkeyPatch, hub: FakeHub, **ner: Any) -> tuple[StrictGliner, Any]:
    from llm_redact.detection.gliner_ner import build_gliner_detector
    from ner_fakes import FakeGliner, install_gliner

    install_gliner(monkeypatch, FakeGliner([]), hub)
    strict = StrictGliner()
    monkeypatch.setattr(sys.modules["gliner"], "GLiNER", strict)
    detector = build_gliner_detector(NerConfig(enabled=True, backend="gliner", **ner))
    return strict, detector


def _two_repos(**extra: dict[str, str]) -> FakeHub:
    return FakeHub(repos={GLINER_SMALL: URCHADE_REPO, DEBERTA: DEBERTA_REPO, **extra}, default=None)


def test_the_default_gliner_model_loads_from_an_assembled_local_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_redact.detection.model_files import (
        BACKBONE_PATTERNS,
        GLINER_PATTERNS,
        models_dir,
    )

    hub = _two_repos()
    strict, _ = _gliner(monkeypatch, hub)
    assert [(c["repo_id"], c["revision"], c["allow_patterns"]) for c in hub.calls] == [
        (GLINER_SMALL, GLINER_PIN, list(GLINER_PATTERNS)),
        # No safetensors in the repository: GLiNER's weights_only pickle.
        (GLINER_SMALL, GLINER_PIN, [*GLINER_PATTERNS, "pytorch_model.bin"]),
        # The base model at the catalog's pin: configuration and tokenizer.
        (DEBERTA, DEBERTA_PIN, list(BACKBONE_PATTERNS)),
    ]
    assert all(c["local_files_only"] for c in hub.calls)
    [(name, kwargs)] = strict.loaded
    assert kwargs == {"local_files_only": True, "map_location": "cpu"}
    folder = Path(name)
    assert folder.parent == models_dir() / "gliner"
    assert _files(folder) == {
        "gliner_config.json",
        "pytorch_model.bin",
        "config.json",
        "tokenizer_config.json",
        "spm.model",
    }  # never the base model's weights
    assert (folder / "pytorch_model.bin").read_text() == "weights"
    config = json.loads((folder / "gliner_config.json").read_text())
    # A local path, and no absolute one: the folder can be carried.
    assert config["model_name"] == "."
    assert config["encoder_config"] == DEBERTA_CONFIG
    assert config["max_len"] == 384
    assert str(folder) not in (folder / "gliner_config.json").read_text()


def test_a_second_load_reuses_the_assembled_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = _two_repos()
    first, _ = _gliner(monkeypatch, hub)
    marker = Path(first.loaded[0][0]) / "marker"
    marker.write_text("")
    second, _ = _gliner(monkeypatch, hub)
    assert first.loaded[0][0] == second.loaded[0][0]
    assert marker.exists()  # not rebuilt


def test_the_checkpoint_vocabulary_size_reaches_the_embedded_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = {
        **URCHADE_REPO,
        "gliner_config.json": json.dumps({"model_name": DEBERTA, "vocab_size": 128004}),
    }
    strict, _ = _gliner(monkeypatch, _two_repos(**{GLINER_SMALL: repo}))
    config = json.loads((Path(strict.loaded[0][0]) / "gliner_config.json").read_text())
    assert config["encoder_config"]["vocab_size"] == 128004


def test_a_checkpoint_with_its_own_tokenizer_keeps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = {**URCHADE_REPO, "tokenizer.json": "own", "tokenizer_config.json": "{}"}
    strict, _ = _gliner(monkeypatch, _two_repos(**{GLINER_SMALL: repo}))
    folder = Path(strict.loaded[0][0])
    assert (folder / "tokenizer.json").read_text() == "own"
    assert not (folder / "spm.model").exists()


def test_a_self_contained_checkpoint_loads_from_its_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = FakeHub(repos={"org/gliner-pii": SELF_CONTAINED}, default=None)
    strict, _ = _gliner(monkeypatch, hub, model="org/gliner-pii")
    assert [c["repo_id"] for c in hub.calls] == ["org/gliner-pii", "org/gliner-pii"]
    name = strict.loaded[0][0]
    assert Path(name) == hub.root / "org--gliner-pii" / "main"
    # Only the names asked for: no ONNX copy without [detection.ner.onnx].
    assert "onnx/model.onnx" not in _files(name)


def test_an_assembled_folder_loads_with_no_hub_at_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import shutil

    from llm_redact.detection import model_files

    # Without links (Windows without the privilege) the files are copied,
    # and the folder is then a portable copy of everything it needs.
    def refuse(*args: Any) -> None:
        raise OSError("no links here")

    monkeypatch.setattr(model_files.os, "symlink", refuse)
    monkeypatch.setattr(model_files.os, "link", refuse)
    hub = _two_repos()
    strict, _ = _gliner(monkeypatch, hub)
    folder = strict.loaded[0][0]
    shutil.rmtree(hub.root)  # the cache (base model included) is gone
    offline = FakeHub(default=None)  # and every hub call fails
    again, _ = _gliner(monkeypatch, offline, model=folder)
    assert offline.calls == []
    assert again.loaded[0][0] == folder


@pytest.mark.parametrize(
    ("repo", "base", "message"),
    [
        (
            {
                **SELF_CONTAINED,
                "gliner_config.json": json.dumps({"encoder_config": {"auto_map": {}}}),
            },
            DEBERTA_REPO,
            "gliner model 'org/gliner-x' needs code from its repository"
            " (gliner_config.json names auto_map)",
        ),
        (
            {
                **SELF_CONTAINED,
                "gliner_config.json": json.dumps({"encoder_config": {"model_type": "evil"}}),
            },
            DEBERTA_REPO,
            "gliner model 'org/gliner-x': gliner_config.json names a model type"
            " transformers does not know",
        ),
        (
            {"gliner_config.json": json.dumps({"model_name": DEBERTA}), "model.safetensors": ""},
            {
                **DEBERTA_REPO,
                "config.json": json.dumps({"model_type": "bert", "auto_map": {"x": "y"}}),
            },
            f"gliner base model '{DEBERTA}' needs code from its repository"
            " (config.json names auto_map)",
        ),
        (
            {"gliner_config.json": json.dumps({"model_name": DEBERTA}), "model.safetensors": ""},
            {**DEBERTA_REPO, "config.json": json.dumps({"model_type": "qwen-custom"})},
            f"gliner base model '{DEBERTA}': config.json names a model type"
            " transformers does not know",
        ),
        (
            {"gliner_config.json": json.dumps({"model_name": DEBERTA}), "model.safetensors": ""},
            {"tokenizer_config.json": "{}"},
            f"gliner base model '{DEBERTA}' has no config.json",
        ),
        (
            {"gliner_config.json": json.dumps({"max_len": 384}), "model.safetensors": ""},
            DEBERTA_REPO,
            "gliner model 'org/gliner-x': gliner_config.json names no base model (model_name)",
        ),
        (
            {"model.safetensors": ""},
            DEBERTA_REPO,
            "gliner model 'org/gliner-x' has no gliner_config.json",
        ),
    ],
)
def test_gliner_checkpoints_that_would_run_code_or_cannot_load_are_refused(
    monkeypatch: pytest.MonkeyPatch, repo: dict[str, str], base: dict[str, str], message: str
) -> None:
    hub = FakeHub(repos={"org/gliner-x": repo, DEBERTA: base}, default=None)
    with pytest.raises(ConfigError) as caught:
        _gliner(monkeypatch, hub, model="org/gliner-x")
    assert str(caught.value).startswith(f"[detection.ner] {message}")


def test_an_uncatalogued_base_model_loads_unpinned_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    repo = {"gliner_config.json": json.dumps({"model_name": DEBERTA}), "model.safetensors": ""}
    hub = FakeHub(repos={"org/gliner-x": repo, DEBERTA: DEBERTA_REPO}, default=None)
    with caplog.at_level("WARNING", logger="llm_redact"):
        _gliner(monkeypatch, hub, model="org/gliner-x")
    assert hub.calls[-1]["repo_id"] == DEBERTA
    assert hub.calls[-1]["revision"] is None
    assert [r.getMessage() for r in caplog.records] == [
        "[detection.ner] gliner model 'org/gliner-x' ships no tokenizer or encoder_config, and"
        f" the model catalog pins no revision of its base model '{DEBERTA}': the newest cached"
        " revision of its default branch loads"
    ]


def test_a_base_model_missing_from_the_cache_names_models_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = _two_repos()
    hub.uncached.add(DEBERTA)
    with pytest.raises(ConfigError) as caught:
        _gliner(monkeypatch, hub)
    assert str(caught.value).startswith(
        f"[detection.ner] gliner base model '{DEBERTA}' at revision {DEBERTA_PIN} is not"
        " (completely) in the local Hugging Face cache"
    )


def test_a_sidecar_identified_folder_takes_the_catalog_base_model_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = _folder(
        tmp_path / "model", {**URCHADE_REPO, SIDECAR_NAME: json.dumps({"model_id": GLINER_SMALL})}
    )
    hub = FakeHub(repos={DEBERTA: DEBERTA_REPO}, default=None)
    _gliner(monkeypatch, hub, model=str(folder))
    assert [(c["repo_id"], c["revision"]) for c in hub.calls] == [(DEBERTA, DEBERTA_PIN)]
    (folder / SIDECAR_NAME).write_text("[]")
    with pytest.raises(ConfigError, match=r"\[detection.ner\] gliner model: .*llm-redact-model"):
        _gliner(monkeypatch, hub, model=str(folder))


def test_a_failed_gliner_load_names_the_exception_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_redact.detection.gliner_ner import build_gliner_detector
    from ner_fakes import FakeGliner, install_gliner

    install_gliner(monkeypatch, FakeGliner([]), _two_repos())

    def fail(name: str, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(sys.modules["gliner"].GLiNER, "from_pretrained", fail)
    with pytest.raises(ConfigError) as caught:
        build_gliner_detector(NerConfig(enabled=True, backend="gliner"))
    assert str(caught.value) == f"failed to load GLiNER model '{GLINER_SMALL}': RuntimeError"


def test_without_transformers_the_gliner_extra_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_redact.detection.model_files import gliner_model_dir

    install_hub(monkeypatch, FakeHub(repos={"org/gliner-pii": SELF_CONTAINED}, default=None))
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ConfigError, match="needs transformers, which the gliner extra installs"):
        gliner_model_dir("org/gliner-pii", revision=None, allow_download=False)


def test_the_models_dir_follows_xdg_data_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from llm_redact.detection.model_files import models_dir

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert models_dir() == tmp_path / "llm-redact" / "models"
    monkeypatch.setenv("XDG_DATA_HOME", "")  # empty counts as unset
    assert models_dir() == Path.home() / ".local" / "share" / "llm-redact" / "models"


def test_a_concurrent_twin_of_the_folder_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_redact.detection import model_files

    real_rename = model_files.os.rename

    def rename_then_lose(source: Any, target: Any) -> None:
        real_rename(source, target)  # another process got there first
        raise FileExistsError(target)

    monkeypatch.setattr(model_files.os, "rename", rename_then_lose)
    strict, _ = _gliner(monkeypatch, _two_repos())
    assert (Path(strict.loaded[0][0]) / "gliner_config.json").is_file()


def test_a_folder_that_cannot_be_assembled_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_redact.detection import model_files

    def no_space(**kwargs: Any) -> str:
        raise OSError(28, "No space left on device")

    hub = _two_repos()
    monkeypatch.setattr(model_files.tempfile, "mkdtemp", no_space)
    with pytest.raises(ConfigError, match=r"cannot assemble its local folder under .* \(OSError\)"):
        _gliner(monkeypatch, hub)


# --- GLiNER ONNX weights ([detection.ner.onnx]) -----------------------------------

ONNX_REPO = {**SELF_CONTAINED, "onnx/model_quint8.onnx": "int8", "model.safetensors": ""}


def test_onnx_weights_load_through_gliner_and_only_they_are_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_redact.detection.model_files import GLINER_PATTERNS

    hub = FakeHub(repos={"org/gliner-pii": ONNX_REPO}, default=None)
    strict, _ = _gliner(
        monkeypatch, hub, model="org/gliner-pii", onnx=(("gliner", "onnx/model_quint8.onnx"),)
    )
    [call] = hub.calls  # no second lookup for a pickle
    assert "onnx/model_quint8.onnx" in call["allow_patterns"]
    assert "model.safetensors" not in call["allow_patterns"]
    assert set(call["allow_patterns"]) < {*GLINER_PATTERNS, "onnx/model_quint8.onnx"}
    [(name, kwargs)] = strict.loaded
    assert kwargs == {
        "local_files_only": True,
        "map_location": "cpu",
        "load_onnx_model": True,
        "onnx_model_file": "onnx/model_quint8.onnx",
    }
    assert _files(name) == {
        "gliner_config.json",
        "onnx/model_quint8.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
    }


def test_an_assembled_folder_carries_the_onnx_file(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = {**URCHADE_REPO, "onnx/model.onnx": "onnx"}
    strict, _ = _gliner(
        monkeypatch, _two_repos(**{GLINER_SMALL: repo}), onnx=(("gliner", "onnx/model.onnx"),)
    )
    folder = Path(strict.loaded[0][0])
    assert (folder / "onnx" / "model.onnx").read_text() == "onnx"
    assert not (folder / "pytorch_model.bin").exists()


def test_a_missing_onnx_file_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub(repos={"org/gliner-pii": SELF_CONTAINED}, default=None)
    with pytest.raises(ConfigError) as caught:
        _gliner(monkeypatch, hub, model="org/gliner-pii", onnx=(("gliner", "onnx/fp8.onnx"),))
    assert str(caught.value) == (
        "[detection.ner] gliner model 'org/gliner-pii' (no revision pinned) has no ONNX file"
        " 'onnx/fp8.onnx' ([detection.ner.onnx] gliner)"
    )


def test_onnx_without_onnxruntime_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "onnxruntime" else real(name, *a),
    )
    hub = FakeHub(repos={"org/gliner-pii": ONNX_REPO}, default=None)
    with pytest.raises(ConfigError, match=r"\[detection.ner.onnx\] gliner needs onnxruntime"):
        _gliner(
            monkeypatch, hub, model="org/gliner-pii", onnx=(("gliner", "onnx/model_quint8.onnx"),)
        )
    assert hub.calls == []


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ({"hf": "onnx/model.onnx"}, "hf: only the gliner backend loads ONNX weights"),
        ({"gliner": "onnx/*.onnx"}, "gliner must be a .onnx file inside the model"),
        ({"gliner": "../model.onnx"}, "gliner must be a .onnx file inside the model"),
        ({"gliner": "/abs/model.onnx"}, "gliner must be a .onnx file inside the model"),
        ({"gliner": "onnx/model.bin"}, "gliner must be a .onnx file inside the model"),
        ({"gliner": 3}, "gliner must be a .onnx file inside the model"),
        ("onnx/model.onnx", "must be a table of BACKEND"),
    ],
)
def test_the_onnx_table_is_validated(table: object, message: str) -> None:
    from llm_redact.config import parse_config

    with pytest.raises(ConfigError, match=re.escape(f"[detection.ner.onnx] {message}")):
        parse_config({"detection": {"ner": {"backend": "gliner", "onnx": table}}}, "t")
    ok = parse_config(
        {"detection": {"ner": {"backend": "gliner", "onnx": {"gliner": "onnx/m-1_q.onnx"}}}}, "t"
    )
    assert ok.detection.ner.onnx_for("gliner") == "onnx/m-1_q.onnx"
    assert ok.detection.ner.onnx_for("hf") is None
