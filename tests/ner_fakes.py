"""Stand-ins for the NER libraries, so the real backend BUILDERS run without
any extra installed: each ``install_*`` puts a fake module into
``sys.modules`` that hands back a scripted model.

A fake model finds fixed surface strings and reports them under the label
the test chose — never anything derived from the text beyond offsets. The
``hf`` and ``gliner`` builders resolve model files first (model_files.py):
:class:`FakeHub` stands in for ``huggingface_hub.snapshot_download`` over a
scripted repository, writing its files into the session's throwaway home.
"""

import fnmatch
import json
import os
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# (surface text, model label, score)
Finding = tuple[str, str, float]


def _spans(text: str, findings: list[Finding]) -> list[tuple[int, int, str, float]]:
    found = []
    for surface, label, score in findings:
        start = text.find(surface)
        while start != -1:
            found.append((start, start + len(surface), label, score))
            start = text.find(surface, start + 1)
    return found


@dataclass
class FakeTokenizer:
    """A fast tokenizer: one token per whitespace-separated word, two special
    tokens per window; with overflow it windows like the Hugging Face fast
    tokenizers (``model_max_length`` tokens a window, ``stride`` shared).
    Its model marks word pieces with ``subword_prefix`` (WordPiece's "##";
    None or "" for SentencePiece and byte-level BPE models)."""

    model_max_length: int = 512
    is_fast: bool = True
    special_tokens: int = 2
    subword_prefix: str | None = "##"

    @property
    def _tokenizer(self) -> Any:
        # The `tokenizers` backend object a fast tokenizer wraps.
        return types.SimpleNamespace(
            model=types.SimpleNamespace(continuing_subword_prefix=self.subword_prefix)
        )

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        return_overflowing_tokens: bool = False,
        stride: int = 0,
        **kwargs: Any,
    ) -> dict[str, list[list[int]]]:
        tokens = list(range(len(text.split())))
        content = self.model_max_length - self.special_tokens
        if not (truncation and return_overflowing_tokens) or len(tokens) <= content:
            return {"input_ids": [tokens]}
        chunks = []
        start = 0
        while True:
            chunks.append(tokens[start : start + content])
            if start + content >= len(tokens):
                return {"input_ids": chunks}
            start += content - stride


@dataclass
class FakeHfPipe:
    """transformers token-classification pipeline, aggregation "simple"."""

    findings: list[Finding]
    id2label: dict[int, str] | None = None
    calls: list[str] = field(default_factory=list)
    tokenizer: FakeTokenizer = field(default_factory=FakeTokenizer)
    max_position_embeddings: int | None = None
    # The keyword arguments of every transformers.pipeline() call that
    # handed out this pipe (install_transformers).
    built_with: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                id2label=self.id2label, max_position_embeddings=self.max_position_embeddings
            )
        )

    def __call__(self, text: str) -> list[dict[str, Any]]:
        self.calls.append(text)
        return [
            {"entity_group": label, "score": score, "word": text[s:e], "start": s, "end": e}
            for s, e, label, score in _spans(text, self.findings)
        ]


@dataclass
class FakeGliner:
    """GLiNER: answers a finding only when its label was among the prompts."""

    findings: list[Finding]
    calls: list[list[str]] = field(default_factory=list)
    # The arguments of every GLiNER.from_pretrained() call (install_gliner).
    loaded_with: list[dict[str, Any]] = field(default_factory=list)

    def predict_entities(
        self, text: str, labels: list[str], threshold: float
    ) -> list[dict[str, Any]]:
        self.calls.append(list(labels))
        return [
            {"start": s, "end": e, "label": label, "text": text[s:e], "score": score}
            for s, e, label, score in _spans(text, self.findings)
            if label in labels and score >= threshold
        ]


@dataclass
class _Ent:
    start_char: int
    end_char: int
    label_: str
    text: str

    @property
    def type(self) -> str:  # stanza's spelling of the label
        return self.label_


class FakeSpacy:
    """spaCy Language (also stanza's Pipeline: same .ents shape)."""

    def __init__(self, findings: list[Finding], labels: tuple[str, ...] | None = None) -> None:
        self.findings = findings
        self._labels = labels

    def __call__(self, text: str) -> Any:
        ents = [_Ent(s, e, label, text[s:e]) for s, e, label, _ in _spans(text, self.findings)]
        return types.SimpleNamespace(ents=ents)

    def get_pipe(self, name: str) -> Any:
        if self._labels is None or name != "ner":
            raise KeyError(name)
        return types.SimpleNamespace(labels=self._labels)


class FakeAnalyzer:
    """presidio_analyzer.AnalyzerEngine: like the real one, it raises when
    asked only for entities no recognizer supports."""

    def __init__(self, findings: list[Finding], supported: tuple[str, ...]) -> None:
        self.findings = findings
        self.supported = supported
        self.calls: list[list[str] | None] = []

    def get_supported_entities(self, language: str | None = None) -> list[str]:
        return list(self.supported)

    def analyze(
        self,
        text: str,
        language: str,
        entities: list[str] | None = None,
        score_threshold: float = 0.0,
    ) -> list[Any]:
        self.calls.append(entities)
        if entities is not None and not set(entities) & set(self.supported):
            raise ValueError("No matching recognizers were found to serve the request.")
        return [
            types.SimpleNamespace(entity_type=label, start=s, end=e, score=score)
            for s, e, label, score in _spans(text, self.findings)
            if (entities is None or label in entities) and score >= score_threshold
        ]


# A repository every backend's fake loads from: an hf model (config.json,
# safetensors weights, a tokenizer) and a self-contained GLiNER model
# (gliner_config.json with an encoder_config, the same weights and
# tokenizer). Each backend's file patterns pick its own files.
DEFAULT_REPO: dict[str, str] = {
    "config.json": json.dumps({"model_type": "bert"}),
    "model.safetensors": "",
    "tokenizer.json": "{}",
    "tokenizer_config.json": "{}",
    "vocab.txt": "",
    "gliner_config.json": json.dumps(
        {"model_name": "microsoft/deberta-v3-small", "encoder_config": {"model_type": "deberta-v2"}}
    ),
}


class NotCached(Exception):
    """huggingface_hub's LocalEntryNotFoundError, for the fake hub."""


@dataclass
class FakeHub:
    """``huggingface_hub.snapshot_download`` over scripted repositories: a
    call writes the repository's files its ``allow_patterns`` match
    (fnmatch, as the hub matches them) into one folder per repository and
    revision, and returns it. A repository in ``uncached`` is missing from
    the cache: a ``local_files_only`` call raises. Every call is recorded.
    No network, ever."""

    repos: dict[str, dict[str, str]] = field(default_factory=dict)
    default: dict[str, str] | None = field(default_factory=lambda: dict(DEFAULT_REPO))
    uncached: set[str] = field(default_factory=set)
    calls: list[dict[str, Any]] = field(default_factory=list)
    root: Path = field(
        default_factory=lambda: Path(tempfile.mkdtemp(prefix="fake-hub-", dir=os.environ["HOME"]))
    )

    def snapshot_download(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "allow_patterns": allow_patterns,
                "local_files_only": local_files_only,
                **kwargs,
            }
        )
        files = self.repos.get(repo_id, self.default)
        if files is None:
            raise NotCached(f"no repository {repo_id}")
        if local_files_only and repo_id in self.uncached:
            raise NotCached(f"{repo_id} is not cached")
        folder = self.root / repo_id.replace("/", "--") / (revision or "main")
        for name, content in files.items():
            if allow_patterns is None or any(fnmatch.fnmatch(name, p) for p in allow_patterns):
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder)


def install_hub(monkeypatch: pytest.MonkeyPatch, hub: FakeHub | None = None) -> FakeHub:
    """Make ``huggingface_hub`` the fake ``hub`` (a default one when None)."""
    hub = hub if hub is not None else FakeHub()
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = hub.snapshot_download  # type: ignore[attr-defined]
    module.fake_hub = hub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return hub


def _ensure_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    # A fake hub a test installed itself is kept.
    if getattr(sys.modules.get("huggingface_hub"), "fake_hub", None) is None:
        install_hub(monkeypatch)


def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """The fake ``transformers`` module (one per test, shared by the fake
    hf pipeline and the GLiNER loader's model-type check)."""
    module = sys.modules.get("transformers")
    if module is None or not getattr(module, "is_fake", False):
        module = types.ModuleType("transformers")
        module.is_fake = True  # type: ignore[attr-defined]
        # The model types the GLiNER loader may see (transformers knows them).
        module.CONFIG_MAPPING = {  # type: ignore[attr-defined]
            "bert": object(),
            "deberta-v2": object(),
            "modernbert": object(),
        }
        monkeypatch.setitem(sys.modules, "transformers", module)
    return module


def install_transformers(
    monkeypatch: pytest.MonkeyPatch, pipe: FakeHfPipe, hub: FakeHub | None = None
) -> None:
    module = fake_transformers(monkeypatch)

    def pipeline(*args: Any, **kwargs: Any) -> FakeHfPipe:
        pipe.built_with.append(kwargs)
        return pipe

    module.pipeline = pipeline  # type: ignore[attr-defined]
    if hub is not None:
        install_hub(monkeypatch, hub)
    else:
        _ensure_hub(monkeypatch)


def install_gliner(
    monkeypatch: pytest.MonkeyPatch, model: FakeGliner, hub: FakeHub | None = None
) -> None:
    """A fake ``gliner`` whose ``GLiNER.from_pretrained`` records its
    arguments on ``model.loaded_with`` and hands ``model`` back; with the
    fake hub and the fake transformers model types the loader needs."""
    module = types.ModuleType("gliner")

    def from_pretrained(name: str, **kwargs: Any) -> FakeGliner:
        model.loaded_with.append({"model_id": name, **kwargs})
        return model

    module.GLiNER = types.SimpleNamespace(from_pretrained=from_pretrained)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
    fake_transformers(monkeypatch)
    if hub is not None:
        install_hub(monkeypatch, hub)
    else:
        _ensure_hub(monkeypatch)


def install_spacy(monkeypatch: pytest.MonkeyPatch, nlp: FakeSpacy) -> None:
    module = types.ModuleType("spacy")
    module.load = lambda name, disable=(): nlp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", module)


def install_stanza(monkeypatch: pytest.MonkeyPatch, nlp: FakeSpacy) -> None:
    module = types.ModuleType("stanza")
    module.Pipeline = lambda **kwargs: nlp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stanza", module)


def install_presidio(monkeypatch: pytest.MonkeyPatch, analyzer: FakeAnalyzer) -> None:
    module = types.ModuleType("presidio_analyzer")
    module.AnalyzerEngine = lambda **kwargs: analyzer  # type: ignore[attr-defined]
    engine = types.ModuleType("presidio_analyzer.nlp_engine")

    class _Provider:
        def __init__(self, nlp_configuration: dict[str, Any]) -> None:
            self.nlp_configuration = nlp_configuration

        def create_engine(self) -> object:
            return object()

    engine.NlpEngineProvider = _Provider  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "presidio_analyzer", module)
    monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", engine)
