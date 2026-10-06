"""The model-load policy seam (``plugin_api.ModelPolicy``, owner decision D4).

``Registry.build_model_policy(config, tier)`` (Free default: None) builds a
policy once at startup; every detector build of the process asks it about
each NER backend's model after the model's files are resolved and before
its weights load. A refusal — or any answer that is not None — is a
ConfigError: ``serve --check`` fails, a reload keeps the running config,
the editor's dry run refuses. The core holds no policy of its own. The
fakes never touch the network.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact.cli import main, run_preview
from llm_redact.config import Config, ConfigError
from llm_redact.detection.engine import (
    POLICY_REASON_CHARS,
    DetectionConfig,
    NerConfig,
    build_detectors,
    check_model_policy,
)
from llm_redact.detection.model_catalog import SIDECAR_NAME, ModelIdentity, sidecar_text
from llm_redact.detection.model_files import ModelNotCached
from llm_redact.detection.model_sources import model_load
from llm_redact.free_defaults import build_model_policy
from llm_redact.plugin_api import ModelLoad, ModelPolicy
from llm_redact.proxy import create_app
from llm_redact.registry import Registry
from ner_fakes import (
    DEFAULT_REPO,
    FakeGliner,
    FakeGliner2,
    FakeHfPipe,
    FakeHub,
    FakeSpacy,
    install_gliner,
    install_gliner2,
    install_spacy,
    install_transformers,
)

HF = NerConfig(enabled=True, backend="hf")
DSLIM_PIN = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"


class Policy:
    """A recording policy answering ``answer`` (or raising it)."""

    def __init__(self, answer: Any = None) -> None:
        self.answer = answer
        self.loads: list[ModelLoad] = []

    def check(self, load: ModelLoad) -> str | None:
        self.loads.append(load)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer  # type: ignore[no-any-return]


def _detection(**ner: Any) -> DetectionConfig:
    return DetectionConfig(ner=replace(HF, **ner))


@pytest.fixture
def plugged(monkeypatch: pytest.MonkeyPatch) -> tuple[Registry, list[tuple[Any, str]]]:
    """A registry whose policy factory records what it was built with."""
    reg = Registry()
    built: list[tuple[Any, str]] = []
    reg.resolve_license = lambda *args, **kwargs: resolved("team")
    reg.build_access_gate = lambda cfg, lic: None
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return reg, built


def _install(reg: Registry, built: list[tuple[Any, str]], policy: Policy) -> None:
    def factory(config: Any, tier: str) -> ModelPolicy:
        built.append((config, tier))
        return policy

    reg.build_model_policy = factory


# --- the Free default ------------------------------------------------------------


def test_the_free_default_has_no_policy() -> None:
    assert build_model_policy(Config(), "free") is None
    assert Registry().build_model_policy is build_model_policy


def test_without_a_policy_the_build_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub()
    pipe = FakeHfPipe([])
    install_transformers(monkeypatch, pipe, hub)
    build_detectors(_detection())
    # The loader's own resolve only: no extra lookup for a policy.
    assert len(hub.calls) == 1
    assert len(pipe.built_with) >= 1


# --- what the policy sees, and when ------------------------------------------------


def test_an_hf_model_is_shown_resolved_with_the_catalog_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = FakeHub()
    pipe = FakeHfPipe([])
    install_transformers(monkeypatch, pipe, hub)
    policy = Policy()
    build_detectors(_detection(), model_policy=policy)
    (load,) = policy.loads
    assert load.backend == "hf" and load.model == "dslim/bert-base-NER"
    assert load.model_id == "dslim/bert-base-NER" and load.revision == DSLIM_PIN
    assert not load.local and not load.assembled and load.onnx is None
    assert load.catalog_status == "vetted" and load.license == "MIT"
    assert load.lineage == ("conll2003",) and load.attribution
    names = [name for name, _ in load.files]
    assert names == sorted(names) and "model.safetensors" in names and "config.json" in names
    assert load.path is not None
    for name, path in load.files:
        assert Path(path) == Path(load.path) / name and Path(path).is_file()
    # The policy runs after the files are resolved and before the load.
    assert pipe.built_with


def test_the_policy_runs_before_the_weights_load(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = FakeHfPipe([])
    install_transformers(monkeypatch, pipe, FakeHub())
    with pytest.raises(ConfigError, match="refused by the model-load policy: no MIT here"):
        build_detectors(_detection(), model_policy=Policy("no MIT here"))
    assert pipe.built_with == []  # never built


def test_files_that_cannot_be_resolved_fail_before_the_policy_is_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = FakeHub(uncached={"dslim/bert-base-NER"})
    install_transformers(monkeypatch, FakeHfPipe([]), hub)
    policy = Policy()
    with pytest.raises(ModelNotCached):
        build_detectors(_detection(), model_policy=policy)
    assert policy.loads == []


def test_a_startup_that_may_download_shows_the_downloaded_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = FakeHub(uncached={"dslim/bert-base-NER"})
    install_transformers(monkeypatch, FakeHfPipe([]), hub)
    policy = Policy()
    build_detectors(_detection(allow_download=True), startup=True, model_policy=policy)
    assert hub.calls[0]["local_files_only"] is False
    assert policy.loads[0].files


def test_an_assembled_gliner_model_shows_its_base_model(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = {
        "gliner_config.json": json.dumps({"model_name": "org/base"}),
        "model.safetensors": "",
    }
    base = {"config.json": DEFAULT_REPO["config.json"], **_TOKENIZER}
    hub = FakeHub(repos={"org/checkpoint": repo, "org/base": base})
    install_gliner(monkeypatch, FakeGliner([]), hub)
    policy = Policy()
    ner = NerConfig(enabled=True, backend="gliner", model="org/checkpoint")
    detection = DetectionConfig(ner=ner)
    build_detectors(detection, model_policy=policy)
    (load,) = policy.loads
    assert load.assembled and load.backbone == "org/base" and load.backbone_revision is None
    assert load.catalog_status is None and load.license is None and load.lineage == ()
    names = dict(load.files)
    assert "tokenizer.json" in names and "config.json" in names
    assert "gliner_config.json" not in names  # the core writes that one


_TOKENIZER = {
    "tokenizer.json": DEFAULT_REPO["tokenizer.json"],
    "tokenizer_config.json": DEFAULT_REPO["tokenizer_config.json"],
}


def test_a_gliner2_model_and_a_spacy_model(monkeypatch: pytest.MonkeyPatch) -> None:
    install_gliner2(monkeypatch, FakeGliner2([]))
    install_spacy(monkeypatch, FakeSpacy([]))
    policy = Policy()
    ner = NerConfig(enabled=True, backends=("gliner2", "spacy"))
    build_detectors(DetectionConfig(ner=ner), model_policy=policy)
    gliner2, spacy = policy.loads
    assert gliner2.backend == "gliner2" and gliner2.model == "fastino/gliner2-base-v1"
    assert "encoder_config/config.json" in dict(gliner2.files)
    assert gliner2.license == "Apache-2.0" and gliner2.catalog_status == "caution"
    # Not a Hub snapshot: only the backend and the configured model.
    assert spacy == ModelLoad(backend="spacy", model="en_core_web_sm")


@pytest.mark.parametrize(
    ("ner", "expected"),
    [
        (NerConfig(enabled=True, backend="presidio", model="en_core_web_lg"), "en_core_web_lg"),
        (NerConfig(enabled=True, backend="stanza", language="de"), "de"),
        (NerConfig(enabled=True, backend="stanza", language=""), "en"),
    ],
)
def test_libraries_models_are_named_as_configured(ner: NerConfig, expected: str) -> None:
    load = model_load(ner, ner.backend)
    assert load == ModelLoad(backend=ner.backend, model=expected)


def test_a_local_folder_is_identified_by_its_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_transformers(monkeypatch, FakeHfPipe([]), FakeHub())
    folder = tmp_path / "hf-model"
    folder.mkdir()
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        (folder / name).write_text(DEFAULT_REPO[name])
    identity = ModelIdentity("dslim/bert-base-NER", DSLIM_PIN)
    (folder / SIDECAR_NAME).write_text(sidecar_text(identity))
    load = model_load(replace(HF, model=str(folder)), "hf")
    assert load.local and load.path == str(folder) and load.model == str(folder)
    assert load.model_id == "dslim/bert-base-NER" and load.revision == DSLIM_PIN
    assert load.license == "MIT"
    assert "llm-redact-model.json" not in dict(load.files)


# --- every answer but None refuses -------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (RuntimeError("secret detail"), r"policy failed \(RuntimeError\); the model is not loaded"),
        ("", r"answered neither None nor a reason \(str\)"),
        (0, r"answered neither None nor a reason \(int\)"),
        (False, r"answered neither None nor a reason \(bool\)"),
    ],
)
def test_nothing_fails_open(monkeypatch: pytest.MonkeyPatch, answer: Any, message: str) -> None:
    install_transformers(monkeypatch, FakeHfPipe([]), FakeHub())
    with pytest.raises(ConfigError, match=message) as refused:
        build_detectors(_detection(), model_policy=Policy(answer))
    assert "secret detail" not in str(refused.value)
    assert "[detection.ner] hf model 'dslim/bert-base-NER'" in str(refused.value)


def test_an_awaitable_answer_is_closed_and_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transformers(monkeypatch, FakeHfPipe([]), FakeHub())

    async def later() -> None:
        return None

    coroutine = later()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # "never awaited" would be a RuntimeWarning
        with pytest.raises(ConfigError, match=r"neither None nor a reason \(coroutine\)"):
            build_detectors(_detection(), model_policy=Policy(coroutine))
    assert coroutine.cr_frame is None  # closed
    asyncio.run(asyncio.sleep(0))


def test_a_reason_is_escaped_and_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transformers(monkeypatch, FakeHfPipe([]), FakeHub())
    reason = "\x1b[31mred" + "x" * 2000
    with pytest.raises(ConfigError) as refused:
        check_model_policy(Policy(reason), HF, "hf")
    text = str(refused.value)
    assert "\x1b" not in text and "\\x1b[31mred" in text
    shown = text.split("policy: ", 1)[1]
    assert len(shown) == POLICY_REASON_CHARS + 3 and shown.endswith("...")


# --- through the proxy and the CLI -------------------------------------------------


def test_serve_check_fails_with_the_policys_message(
    monkeypatch: pytest.MonkeyPatch,
    plugged: tuple[Registry, list[tuple[Any, str]]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install_transformers(monkeypatch, FakeHfPipe([]), FakeHub())
    reg, built = plugged
    _install(reg, built, Policy("license MIT is not on the allowed list"))
    config_file = tmp_path / "config.toml"
    config_file.write_text('[detection.ner]\nenabled = true\nbackend = "hf"\n')
    with pytest.raises(SystemExit) as exited:
        main(["serve", "--check", "--config", str(config_file)])
    assert exited.value.code == 1
    err = capsys.readouterr().err
    assert "serve --check: FAIL:" in err
    assert "refused by the model-load policy: license MIT is not on the allowed list" in err
    # Built once, with the resolved tier.
    assert [tier for _, tier in built] == ["team"]


def test_serve_check_passes_without_ner_or_with_an_allowing_policy(
    monkeypatch: pytest.MonkeyPatch,
    plugged: tuple[Registry, list[tuple[Any, str]]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install_transformers(monkeypatch, FakeHfPipe([]), FakeHub())
    reg, built = plugged
    policy = Policy()
    _install(reg, built, policy)
    config_file = tmp_path / "config.toml"
    config_file.write_text('[detection.ner]\nenabled = true\nbackend = "hf"\n')
    with pytest.raises(SystemExit) as exited:
        main(["serve", "--check", "--config", str(config_file)])
    assert exited.value.code == 0
    assert [load.backend for load in policy.loads] == ["hf"]


def test_reloads_and_the_editor_dry_run_ask_the_startup_policy(
    monkeypatch: pytest.MonkeyPatch, plugged: tuple[Registry, list[tuple[Any, str]]]
) -> None:
    install_transformers(monkeypatch, FakeHfPipe([]), FakeHub())
    reg, built = plugged
    policy = Policy()
    _install(reg, built, policy)
    config = Config(detection=_detection())
    state = create_app(config).state.proxy
    assert state.model_policy is policy and len(policy.loads) == 1
    fresh = replace(config, detection=_detection(model="org/other-ner"))
    policy.answer = "org/other-ner is not in the bundle"
    with pytest.raises(ConfigError, match="not in the bundle"):
        state.validate_config(fresh)
    with pytest.raises(ConfigError, match="not in the bundle"):
        state.apply_config(fresh)
    assert state.config.detection.ner.model is None  # the running config is kept
    assert [load.model for load in policy.loads[1:]] == ["org/other-ner"] * 2
    # An unchanged [detection] is not rebuilt (and not asked again).
    state.validate_config(config)
    assert len(policy.loads) == 3
    assert len(built) == 1  # the policy is built once per process


def test_the_preview_cli_asks_the_policy(
    monkeypatch: pytest.MonkeyPatch,
    plugged: tuple[Registry, list[tuple[Any, str]]],
    tmp_path: Path,
) -> None:
    install_transformers(monkeypatch, FakeHfPipe([]), FakeHub())
    reg, built = plugged
    _install(reg, built, Policy("refused for the preview"))
    config_file = tmp_path / "config.toml"
    config_file.write_text('[detection.ner]\nenabled = true\nbackend = "hf"\n')
    args = argparse.Namespace(config=config_file, text="hello", json=True)
    with pytest.raises(ConfigError, match="refused for the preview"):
        run_preview(args)
    assert [tier for _, tier in built] == ["team"]
