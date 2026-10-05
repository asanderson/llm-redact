"""NER model files are downloaded only at startup, and only when allowed.

``[detection.ner] allow_download`` (owner decision D2: default false) lets
the process's startup build (``serve``, ``serve --check``) fetch a model's
pinned files from the Hugging Face Hub; every other build — a SIGHUP reload,
the config editor's dry run and apply, ``llm-redact preview`` — resolves from
local files only, and a build that may not download sets the libraries'
offline switches before their first import (AD11). The fakes never touch
the network; the subprocess test watches the real import order.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from dataclasses import replace
from pathlib import Path

import pytest

from llm_redact.config import Config, ConfigError
from llm_redact.detection.engine import DetectionConfig, NerConfig, build_detectors
from llm_redact.detection.model_files import go_offline
from llm_redact.proxy import create_app
from ner_fakes import FakeGliner, FakeHfPipe, FakeHub, install_gliner, install_transformers

HF = NerConfig(enabled=True, backend="hf")


def _detection(**ner: object) -> DetectionConfig:
    return DetectionConfig(ner=replace(HF, **ner))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("allow_download", "startup", "local_only", "offline"),
    [
        (False, True, True, "1"),  # the default: the cache only, offline
        (True, True, False, None),  # a startup may download when allowed
        (True, False, True, "1"),  # any other build never downloads
        (False, False, True, "1"),
    ],
)
def test_only_an_allowed_startup_build_downloads(
    monkeypatch: pytest.MonkeyPatch,
    allow_download: bool,
    startup: bool,
    local_only: bool,
    offline: str | None,
) -> None:
    hub = FakeHub()
    install_transformers(monkeypatch, FakeHfPipe([]), hub)
    build_detectors(_detection(allow_download=allow_download), startup=startup)
    assert [call["local_files_only"] for call in hub.calls] == [local_only]
    assert os.environ.get("HF_HUB_OFFLINE") == offline
    assert os.environ.get("TRANSFORMERS_OFFLINE") == offline


def test_the_gliner_backend_and_its_base_model_follow_the_same_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = FakeHub()
    install_gliner(monkeypatch, FakeGliner([]), hub)
    config = DetectionConfig(ner=NerConfig(enabled=True, backend="gliner", allow_download=True))
    build_detectors(config)
    build_detectors(config, startup=True)
    assert [call["local_files_only"] for call in hub.calls] == [True, False]


def test_other_backends_leave_the_offline_switches_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ner_fakes import FakeSpacy, install_hub, install_spacy

    install_hub(monkeypatch)  # records and restores the switches
    install_spacy(monkeypatch, FakeSpacy([]))
    build_detectors(DetectionConfig(ner=NerConfig(enabled=True, backend="spacy")))
    assert "HF_HUB_OFFLINE" not in os.environ


def test_go_offline_reaches_an_already_imported_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HUB_OFFLINE = False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "")  # recorded, so the test restores it
        monkeypatch.delenv(name)
    go_offline()
    assert constants.HF_HUB_OFFLINE is True  # type: ignore[attr-defined]
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"


# --- through the app: startup downloads, a reload never does ---------------------


def test_a_reload_with_an_uncached_model_is_refused_without_a_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = FakeHub()
    install_transformers(monkeypatch, FakeHfPipe([]), hub)
    config = Config(detection=_detection(allow_download=True))
    state = create_app(config).state.proxy
    assert [c["local_files_only"] for c in hub.calls] == [False]  # the startup build
    hub.uncached.add("org/new-ner")
    fresh = replace(config, detection=_detection(allow_download=True, model="org/new-ner"))
    with pytest.raises(ConfigError, match=r"a reload never downloads"):
        state.validate_config(fresh)
    with pytest.raises(ConfigError, match=r"run `llm-redact models pull`"):
        state.apply_config(fresh)
    assert state.config.detection.ner.model is None  # the running config is kept
    assert [c["local_files_only"] for c in hub.calls[1:]] == [True, True]
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_a_reload_with_a_cached_model_loads_it_from_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = FakeHub()
    install_transformers(monkeypatch, FakeHfPipe([]), hub)
    config = Config(detection=_detection())
    state = create_app(config).state.proxy
    state.apply_config(replace(config, detection=_detection(model="org/new-ner")))
    assert state.config.detection.ner.model == "org/new-ner"
    assert [(c["repo_id"], c["local_files_only"]) for c in hub.calls] == [
        ("dslim/bert-base-NER", True),
        ("org/new-ner", True),
    ]


# --- a fresh process: serve --check, an empty cache, the import order ------------

# Runs `llm-redact serve --check` with fake Hugging Face libraries that
# record, when each is first imported, whether the offline switches were
# already set; the fake hub has nothing cached and records its calls.
_PROBE = textwrap.dedent(
    """
    import importlib.abc, importlib.machinery, json, os, sys, types

    seen = {}
    calls = []

    class LocalEntryNotFoundError(Exception):
        pass

    def snapshot_download(repo_id, *, revision=None, allow_patterns=None,
                          local_files_only=False, **kwargs):
        calls.append({"repo_id": repo_id, "revision": revision,
                      "local_files_only": local_files_only})
        raise LocalEntryNotFoundError("nothing is cached")

    def pipeline(*args, **kwargs):
        raise AssertionError("no model may load")

    class Fakes(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        names = ("huggingface_hub", "transformers", "gliner")

        def find_spec(self, name, path=None, target=None):
            if name in self.names:
                return importlib.machinery.ModuleSpec(name, self)
            return None

        def create_module(self, spec):
            seen[spec.name] = os.environ.get("HF_HUB_OFFLINE")
            module = types.ModuleType(spec.name)
            module.snapshot_download = snapshot_download
            module.pipeline = pipeline
            return module

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, Fakes())
    from llm_redact.cli import main
    try:
        main(["serve", "--check", "--config", sys.argv[1]])
    except SystemExit as exc:
        code = exc.code
    print(json.dumps({"code": code, "seen": seen, "calls": calls}))
    """
)


def test_serve_check_with_an_empty_cache_fails_offline_with_the_pull_hint(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[detection.ner]\nenabled = true\nbackend = "hf"\n')
    env = {
        k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, str(config)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["code"] == 1
    # Each library was first imported with the offline switch already set.
    assert report["seen"] == {"transformers": "1", "huggingface_hub": "1"}
    assert report["calls"] == [
        {
            "repo_id": "dslim/bert-base-NER",
            "revision": "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc",
            "local_files_only": True,
        }
    ]
    assert "serve --check: FAIL: [detection.ner] hf model 'dslim/bert-base-NER'" in done.stderr
    assert "run `llm-redact models pull`" in done.stderr
