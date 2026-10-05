"""Datasets the NER bench scores (``python -m llm_redact.bench.ner --dataset``).

Each :class:`DatasetSpec` names where its rows come from, under which
license and attribution, whether it holds REAL data (real people's text:
``--dump-errors`` then needs ``--allow-real-data-dump``), the label map that
turns its labels into what the bench scores (:mod:`llm_redact.bench.ner_metrics`)
and the adapter that reads its rows into :class:`NerSample`\\ s.

Generated datasets are built at run time from a seed and never committed;
published ones are downloaded at a pinned revision into a cache outside the
repository (:mod:`llm_redact.bench.datasets.base`). Nothing in a dataset is
ever logged or printed: reports carry counts only.
"""

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from llm_redact.bench import ner_corpus
from llm_redact.bench.datasets import creddata, mapa, nemotron, openpii, privy, pupa
from llm_redact.bench.datasets.base import (
    DatasetError,
    DatasetSpec,
    LoadRequest,
    default_cache_dir,
)
from llm_redact.bench.ner_metrics import GoldSpan, NerSample
from llm_redact.detection.regex_rules import BUILTIN_RULES

if TYPE_CHECKING:
    from llm_redact.bench.corpus import Sample

__all__ = [
    "DATASETS",
    "DatasetError",
    "DatasetSpec",
    "LoadRequest",
    "dataset_key",
    "default_cache_dir",
    "resolve",
]


def _from_regex_corpus(samples: "list[Sample]", context: str) -> Iterator[NerSample]:
    for sample in samples:
        spans = tuple(GoldSpan(s.start, s.end, s.detector_type) for s in sample.spans)
        yield NerSample(sample.text, spans, context if spans else f"{context}-negative")


def _rules_adapter(spec: DatasetSpec, request: LoadRequest) -> Iterator[NerSample]:
    from llm_redact.bench.corpus import generate

    yield from _from_regex_corpus(generate(seed=request.seed, samples_per_rule=5), "rules")


RULE_TYPES = frozenset(rule.detector_type for rule in BUILTIN_RULES)

RULES = DatasetSpec(
    name="rules",
    summary="the regex bench's generated positives and decoys (every built-in rule)",
    license="generated at run time (part of llm-redact, AGPL-3.0-only)",
    attribution="llm-redact",
    real_data=False,
    label_map=MappingProxyType({type_name: type_name for type_name in RULE_TYPES}),
    splits=("generated",),
    adapter=_rules_adapter,
    notes=(
        "Structured values only: with NER on, the structured-regression check and"
        " the over-redaction rate show what a model costs the rules.",
    ),
)


def _synthetic_adapter(spec: DatasetSpec, request: LoadRequest) -> Iterator[NerSample]:
    yield from ner_corpus.generate(seed=request.seed)


SYNTHETIC = DatasetSpec(
    name="synthetic",
    summary=(
        "agent-traffic-shaped text (prose, chat, JSON tool results, code comments, log"
        " lines, git output) with names, street addresses, dates of birth, usernames and"
        " account numbers, plus hard negatives"
    ),
    license="generated at run time (part of llm-redact, AGPL-3.0-only)",
    attribution="llm-redact",
    real_data=False,
    label_map=MappingProxyType({label: label for label in ner_corpus.LABELS}),
    splits=("generated",),
    adapter=_synthetic_adapter,
    notes=(
        "Generated from small embedded name, street and handle lists; also carries"
        " EMAIL and PHONE values for the structured-regression check.",
    ),
)

DATASETS: Mapping[str, DatasetSpec] = MappingProxyType(
    {
        spec.name: spec
        for spec in (
            SYNTHETIC,
            RULES,
            openpii.SPEC,
            nemotron.SPEC,
            privy.SPEC,
            pupa.SPEC,
            mapa.SPEC,
            creddata.SPEC,
        )
    }
)


def resolve(argument: str) -> tuple[DatasetSpec, str]:
    """``NAME[:SPLIT]`` -> (dataset, split). ValueError names the problem."""
    name, _, split = argument.partition(":")
    spec = DATASETS.get(name)
    if spec is None:
        raise ValueError(f"unknown dataset {name!r}; known datasets: {', '.join(sorted(DATASETS))}")
    split = split or spec.default_split
    if split not in spec.splits:
        raise ValueError(
            f"dataset {name!r} has no split {split!r}; its splits: {', '.join(spec.splits)}"
        )
    return spec, split


def dataset_key(spec: DatasetSpec, split: str, language: str | None = None) -> str:
    """The dataset part of a thresholds key: the name, plus the split when
    it is not the default, plus ``@LANGUAGE`` when rows are filtered."""
    key = spec.name if split == spec.default_split else f"{spec.name}:{split}"
    return f"{key}@{language}" if language else key
