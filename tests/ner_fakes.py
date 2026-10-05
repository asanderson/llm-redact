"""Stand-ins for the NER libraries, so the real backend BUILDERS run without
any extra installed: each ``install_*`` puts a fake module into
``sys.modules`` that hands back a scripted model.

A fake model finds fixed surface strings and reports them under the label
the test chose — never anything derived from the text beyond offsets.
"""

import sys
import types
from dataclasses import dataclass, field
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
    tokenizers (``model_max_length`` tokens a window, ``stride`` shared)."""

    model_max_length: int = 512
    is_fast: bool = True
    special_tokens: int = 2

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


def install_transformers(monkeypatch: pytest.MonkeyPatch, pipe: FakeHfPipe) -> None:
    module = types.ModuleType("transformers")

    def pipeline(*args: Any, **kwargs: Any) -> FakeHfPipe:
        pipe.built_with.append(kwargs)
        return pipe

    module.pipeline = pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)


def install_gliner(monkeypatch: pytest.MonkeyPatch, model: FakeGliner) -> None:
    module = types.ModuleType("gliner")
    module.GLiNER = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name: model
    )
    monkeypatch.setitem(sys.modules, "gliner", module)


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
