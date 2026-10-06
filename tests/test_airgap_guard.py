"""Air-gapped starts open no connection (AD11).

With ``[detection.ner] allow_download`` off (the default), building every
NER backend's detector — the model lookup, the load, the first detection —
must not open a network connection: an enclave has no route out, and a
connection attempt there is at best a timeout and at worst traffic to a
host the operator never approved. A socket guard fails the test on any
attempt (``connect``, ``create_connection``, ``getaddrinfo``), whether or
not the code that made it swallowed the error.

The fakes behave like the libraries they stand for where the network is
concerned: the Hub fake connects to huggingface.co when asked to download
(as ``huggingface_hub`` does unless ``local_files_only`` or its offline
switch says otherwise), and the tldextract fake fetches the Public Suffix
List the first time Presidio's email check asks it (as tldextract 5.x does)
unless its extractor was built without list URLs. The guard catching both
is pinned here too, so a loader that started to download, or a Presidio
build that left tldextract's default in place, fails this file.

The CI ``airgap`` job (.github/workflows/ci.yml) is the same proof with the
real libraries: `serve --check` inside a network namespace with no route.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llm_redact.config import ConfigError
from llm_redact.detection.engine import DetectionConfig, NerConfig, build_detectors
from llm_redact.detection.model_files import names_a_path, resolve_model
from ner_fakes import (
    DEFAULT_REPO,
    FakeAnalyzer,
    FakeGliner,
    FakeGliner2,
    FakeHfPipe,
    FakeHub,
    FakeSpacy,
    FakeTldExtract,
    install_gliner,
    install_gliner2,
    install_presidio,
    install_spacy,
    install_stanza,
    install_transformers,
)


@dataclass
class Guard:
    """The connection attempts the socket guard refused (host, or the
    call's first argument)."""

    attempts: list[str]


@pytest.fixture
def guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[Guard]:
    attempts: list[str] = []

    def refuse(kind: str) -> Any:
        def blocked(*args: Any, **kwargs: Any) -> Any:
            target = args[1] if kind == "connect" and len(args) > 1 else (args[0] if args else "")
            attempts.append(f"{kind} {target!r}")
            raise OSError(f"network access attempted ({kind}) in an air-gapped test")

        return blocked

    monkeypatch.setattr(socket.socket, "connect", refuse("connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", refuse("connect"))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    yield Guard(attempts)


class NetworkedHub(FakeHub):
    """The fake Hub, connecting to huggingface.co whenever a call may
    download — as ``huggingface_hub.snapshot_download`` does unless
    ``local_files_only`` is passed or its offline switch is on."""

    def snapshot_download(self, repo_id: str, **kwargs: Any) -> str:
        if not kwargs.get("local_files_only") and os.environ.get("HF_HUB_OFFLINE") != "1":
            socket.create_connection(("huggingface.co", 443), timeout=5)
        return super().snapshot_download(repo_id, **kwargs)


TEXT = "write to Jane Roe at jane.roe@example.com"


def _install(monkeypatch: pytest.MonkeyPatch, backend: str, hub: FakeHub) -> Any:
    """The backend's fake library (and the hub), answering for TEXT; the
    model fake, whose load arguments the test reads."""
    if backend == "hf":
        pipe = FakeHfPipe([("Jane Roe", "PER", 0.99)])
        install_transformers(monkeypatch, pipe, hub)
        return pipe
    if backend == "gliner":
        model = FakeGliner([("Jane Roe", "person", 0.9)])
        install_gliner(monkeypatch, model, hub)
        return model
    if backend == "gliner2":
        model2 = FakeGliner2([("Jane Roe", "person", 0.9)])
        install_gliner2(monkeypatch, model2, hub)
        return model2
    if backend == "presidio":
        analyzer = FakeAnalyzer(
            [("Jane Roe", "PERSON", 0.9), ("jane.roe@example.com", "EMAIL_ADDRESS", 0.9)],
            supported=("PERSON", "EMAIL_ADDRESS"),
        )
        install_presidio(monkeypatch, analyzer)
        # A Presidio build that kept tldextract's default would now reach
        # the network on the first email it checks.
        monkeypatch.setattr(FakeTldExtract, "fetch", _reach)
        return analyzer
    nlp = FakeSpacy([("Jane Roe", "PERSON", 1.0)])
    (install_spacy if backend == "spacy" else install_stanza)(monkeypatch, nlp)
    return nlp


def _reach(host: str) -> None:
    socket.create_connection((host, 443), timeout=5)


@pytest.mark.parametrize("backend", ["hf", "gliner", "gliner2", "spacy", "presidio", "stanza"])
def test_every_backend_starts_and_detects_without_a_connection(
    monkeypatch: pytest.MonkeyPatch, guard: Guard, backend: str
) -> None:
    hub = NetworkedHub()
    model = _install(monkeypatch, backend, hub)
    entities = ("PERSON", "EMAIL") if backend == "presidio" else ("PERSON",)
    config = DetectionConfig(
        enabled=(), ner=NerConfig(enabled=True, backend=backend, entities=entities)
    )
    (detector,) = build_detectors(config, startup=True)  # serve / serve --check
    found = {d.value for d in detector.detect(TEXT)}
    assert "Jane Roe" in found
    assert guard.attempts == []
    if backend in ("hf", "gliner", "gliner2"):
        # The Hub was asked for the local cache only, and the library was
        # handed a local folder with its offline switch on: nothing it
        # does on its own may reach the network either.
        assert [call["local_files_only"] for call in hub.calls] == [True] * len(hub.calls)
        assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"
        loaded = model.built_with[0]["model"] if backend == "hf" else model.loaded_with[0]
        folder = loaded if backend == "hf" else loaded["model_id"]
        assert Path(folder).is_dir()
    if backend == "presidio":
        # Presidio's email check asked tldextract (offline) about the address.
        assert FakeTldExtract.asked == ["jane.roe@example.com"]


def test_the_guard_sees_a_startup_download(monkeypatch: pytest.MonkeyPatch, guard: Guard) -> None:
    # The fakes reach the network where the libraries would: an allowed
    # startup download of a model the cache lacks.
    hub = NetworkedHub(uncached={"dslim/bert-base-NER"})
    install_transformers(monkeypatch, FakeHfPipe([]), hub)
    config = DetectionConfig(ner=NerConfig(enabled=True, backend="hf", allow_download=True))
    with pytest.raises(ConfigError, match="could not be fetched from the Hugging Face Hub"):
        build_detectors(config, startup=True)
    assert guard.attempts == ["create_connection ('huggingface.co', 443)"]


def test_the_guard_sees_tldextracts_own_suffix_list_fetch(
    monkeypatch: pytest.MonkeyPatch, guard: Guard
) -> None:
    install_presidio(monkeypatch, FakeAnalyzer([], supported=("PERSON",)))
    monkeypatch.setattr(FakeTldExtract, "fetch", _reach)
    import tldextract  # the fake, laid out like the real package

    tldextract.extract("jane.roe@example.com")  # the default extractor
    assert guard.attempts == ["create_connection ('publicsuffix.org', 443)"]


def test_presidio_reads_the_shipped_suffix_list_and_writes_no_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_presidio(monkeypatch, FakeAnalyzer([], supported=("PERSON",)))
    import sys

    tld = sys.modules["tldextract.tldextract"]
    build_detectors(DetectionConfig(ner=NerConfig(enabled=True, backend="presidio")))
    extractor = tld.TLD_EXTRACTOR  # type: ignore[attr-defined]
    assert (extractor.cache_dir, extractor.suffix_list_urls, extractor.fallback_to_snapshot) == (
        None,
        (),
        True,
    )


@pytest.mark.parametrize("broken", ["absent", "no-extractor"])
def test_presidio_fails_closed_when_tldextract_cannot_be_kept_offline(
    monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    import sys
    import types

    install_presidio(monkeypatch, FakeAnalyzer([], supported=("PERSON",)))
    if broken == "absent":
        monkeypatch.setitem(sys.modules, "tldextract.tldextract", None)  # import fails
    else:
        monkeypatch.setitem(sys.modules, "tldextract.tldextract", types.ModuleType("x"))
    with pytest.raises(ConfigError, match="cannot keep tldextract") as caught:
        build_detectors(DetectionConfig(ner=NerConfig(enabled=True, backend="presidio")))
    assert "reinstall the presidio extra" in str(caught.value)


# --- folders written by `models pull --to`: no Hub at all ------------------------


def _folder(root: Path, name: str, files: dict[str, str]) -> Path:
    folder = root / name
    for relative, content in files.items():
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / relative).write_text(content)
    return folder


@pytest.mark.parametrize("backend", ["hf", "gliner"])
def test_a_local_folder_loads_without_the_hub_or_a_connection(
    monkeypatch: pytest.MonkeyPatch, guard: Guard, tmp_path: Path, backend: str
) -> None:
    hub = NetworkedHub()
    model = _install(monkeypatch, backend, hub)
    folder = _folder(tmp_path, f"{backend}-model", DEFAULT_REPO)
    config = DetectionConfig(
        enabled=(), ner=NerConfig(enabled=True, backend=backend, model=str(folder))
    )
    (detector,) = build_detectors(config, startup=True)
    assert [d.value for d in detector.detect(TEXT)] == ["Jane Roe"]
    assert hub.calls == []
    assert guard.attempts == []
    assert model is not None


# --- a folder that is not there names `models pull --to` -------------------------


@pytest.mark.parametrize(
    "value",
    [
        "/models/hf-dslim--bert-base-NER",
        "./models/hf-x",
        "~/models/hf-x",
        "models/hf/x",
        "C:\\models\\hf-x",
        "models\\hf-x",
    ],
)
def test_a_missing_model_folder_names_models_pull(value: str, guard: Guard) -> None:
    assert names_a_path(value)
    with pytest.raises(ConfigError) as caught:
        resolve_model(
            value, what="hf model", revision=None, allow_download=False, allow_patterns=()
        )
    message = str(caught.value)
    assert message == (
        f"[detection.ner] hf model {value!r} is not a directory: a model folder must be in"
        " place when the proxy starts (nothing mounted or copied there?); write one on a"
        " connected machine with `llm-redact models pull --to DIR`, carry it here, and check"
        " it with `llm-redact models verify --dir`"
    )
    assert guard.attempts == []


@pytest.mark.parametrize("value", ["not a model id", "org/name with space", "-x"])
def test_a_value_that_is_neither_keeps_its_message(value: str) -> None:
    assert not names_a_path(value)
    with pytest.raises(ConfigError, match="is neither a local directory nor a Hugging Face"):
        resolve_model(
            value, what="hf model", revision=None, allow_download=False, allow_patterns=()
        )


def test_the_airgap_configs_load_what_pull_prints() -> None:
    """tests/airgap: pull.toml names Hub models, config.toml loads the
    folders `models pull --to DIR --as /models` writes for them — the
    folder names it prints (models_cli.folder_name), at /models."""
    import tomllib

    from llm_redact.models_cli import folder_name

    here = Path(__file__).parent / "airgap"
    for pull_name, serve_name in (("pull.toml", "config.toml"), ("pull-edge.toml", "edge.toml")):
        pulled = tomllib.loads((here / pull_name).read_text(encoding="utf-8"))
        served = tomllib.loads((here / serve_name).read_text(encoding="utf-8"))
        ner = pulled["detection"]["ner"]
        defaults = {"hf": "dslim/bert-base-NER", "gliner": "urchade/gliner_small-v2.1"}
        models = {b: ner.get("models", {}).get(b, defaults[b]) for b in ner["backends"]}
        expected = {b: f"/models/{folder_name(b, m)}" for b, m in models.items()}
        assert served["detection"]["ner"]["models"] == expected
        assert served["detection"]["ner"]["backends"] == ner["backends"]
        assert "allow_download" not in served["detection"]["ner"]
        assert "revisions" not in served["detection"]["ner"]
