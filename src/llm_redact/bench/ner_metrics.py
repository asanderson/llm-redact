"""Scoring of the full detection pipeline (regex rules + NER) against
labelled samples, for the NER bench (``python -m llm_redact.bench.ner``).

Every sample carries gold spans under the DATASET's own labels; a label map
turns each label into what is scored:

* a placeholder type (``PERSON``, ``EMAIL``): scored per type and counted by
  the character-leak metric;
* :data:`LEAK`: counted only by the type-agnostic character-leak metric (a
  value the proxy should not let through, but no placeholder type matches
  it, such as a country-specific tax number);
* ``None``: not scored at all (a city, a date): a detection over it is
  neither a false positive nor over-redaction, and a miss is not a leak;
* :data:`NOT_PII`: the dataset marks the value as NOT personal data, so it
  is not gold at all (a detection over it is over-redaction).

A label missing from the map is read as ``None`` and counted in
:attr:`NerResult.unmapped`, so a dataset change surfaces in the report.

Gold parts of one name or address labelled separately ("Jane" GIVENNAME,
"Doe" SURNAME, both mapped to ``PERSON``) are merged into one span by the
rule the detectors' parts merge by (:func:`labels.merge_adjacent_parts`).

Metrics, per type: EXACT (same start, end and type) and OVERLAP-TYPED (a
gold span is found when a detection of its type overlaps it; a detection is
right when it overlaps a gold span of its type) precision, recall and F1. A
detection overlapping no typed gold span but a ``LEAK``/``None`` one is
NEUTRAL: not counted for precision. Type-agnostic: the character-leak rate
(gold characters, typed or ``LEAK``, that no detection covers, over all
such characters) and the over-redaction rate (detected characters outside
every gold span, over the characters outside every gold span). Per type:
the type's character-leak rate (characters of that type's gold spans no
detection of any type covers, over those characters), which a gate on the
requested types reads: the type-agnostic rate also counts every type the
configuration does not request. The
STRUCTURED-REGRESSION check compares against the same configuration with
NER off: a gold span of a regex rule's type that the rules alone find
exactly must still be found exactly with NER on (a model drawing a wider
span wins overlap resolution and costs the rule its match).

Nothing here keeps text: results are counts. Only :func:`score_sample`
with ``errors=`` records span text, for ``--dump-errors``.
"""

import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_redact.detection.base import Detection, Detector
from llm_redact.detection.engine import Allowlist, DetectorPlan, plan_for
from llm_redact.detection.labels import merge_adjacent_parts
from llm_redact.detection.regex_rules import RegexDetector
from llm_redact.redactor import _resolve_overlaps

# Scoring classes a label map can name besides a placeholder type.
LEAK = "@leak"
NOT_PII = "@not-pii"

# The pipeline is measured without any allowlist (the default one included),
# like the regex bench: the gate measures the detectors themselves.
_NO_ALLOW = Allowlist(exact=frozenset(), patterns=())

# Characters of context written around an error span by --dump-errors.
DUMP_CONTEXT_CHARS = 40

LabelMap = Mapping[str, str | None]


@dataclass(frozen=True)
class GoldSpan:
    start: int
    end: int
    label: str  # the dataset's own label, mapped through the dataset's LabelMap


@dataclass(frozen=True)
class NerSample:
    text: str
    spans: tuple[GoldSpan, ...]
    # Where the text comes from (a corpus context, a document type, a
    # language): reported as counts, never with the text.
    context: str = ""


@dataclass
class TypeCounts:
    gold: int = 0
    gold_hit: int = 0
    pred: int = 0
    pred_hit: int = 0

    @property
    def precision(self) -> float:
        return self.pred_hit / self.pred if self.pred else 1.0

    @property
    def recall(self) -> float:
        return self.gold_hit / self.gold if self.gold else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class StructuredCounts:
    # Gold spans of this regex type that the rules alone find exactly, and
    # how many of those the full pipeline still finds exactly.
    baseline: int = 0
    kept: int = 0

    @property
    def regressions(self) -> int:
        return self.baseline - self.kept


@dataclass
class NerResult:
    samples: int = 0
    chars: int = 0
    exact: dict[str, TypeCounts] = field(default_factory=dict)
    overlap: dict[str, TypeCounts] = field(default_factory=dict)
    gold_chars: int = 0
    leaked_chars: int = 0
    outside_chars: int = 0
    over_redacted_chars: int = 0
    # Per placeholder type: characters of its gold spans, and how many of
    # them no detection (of any type) covers.
    type_gold_chars: Counter[str] = field(default_factory=Counter)
    type_leaked_chars: Counter[str] = field(default_factory=Counter)
    structured: dict[str, StructuredCounts] = field(default_factory=dict)
    neutral_detections: int = 0
    unmapped: Counter[str] = field(default_factory=Counter)
    contexts: Counter[str] = field(default_factory=Counter)
    seconds: float = 0.0

    @property
    def leak_rate(self) -> float:
        return self.leaked_chars / self.gold_chars if self.gold_chars else 0.0

    def type_leak_rate(self, type_name: str) -> float:
        gold = self.type_gold_chars[type_name]
        return self.type_leaked_chars[type_name] / gold if gold else 0.0

    @property
    def over_redaction_rate(self) -> float:
        return self.over_redacted_chars / self.outside_chars if self.outside_chars else 0.0

    @property
    def structured_regressions(self) -> int:
        return sum(c.regressions for c in self.structured.values())


@dataclass(frozen=True)
class Pipeline:
    """The detectors a bench run scores: the full configuration and the same
    configuration with NER off (the structured-regression baseline)."""

    full: DetectorPlan
    rules: DetectorPlan
    # Placeholder types of the regex rules (built-in and custom): the types
    # the structured-regression check covers.
    regex_types: frozenset[str]

    @classmethod
    def from_detectors(cls, full: Sequence[Detector], rules: Sequence[Detector]) -> "Pipeline":
        regex_types = frozenset(d.rule.detector_type for d in rules if isinstance(d, RegexDetector))
        return cls(full=plan_for(full), rules=plan_for(rules), regex_types=regex_types)

    def detect(self, text: str) -> tuple[list[Detection], list[Detection]]:
        """(full pipeline, rules alone), each after overlap resolution."""
        return (
            _resolve_overlaps(self.full.detect(text, _NO_ALLOW)),
            _resolve_overlaps(self.rules.detect(text, _NO_ALLOW)),
        )


def _merged(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _length(merged: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merged)


def _intersection(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]]) -> int:
    """Characters two merged interval lists share (two-pointer sweep)."""
    i = j = shared = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start < end:
            shared += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return shared


def _overlaps(start: int, end: int, others: Iterable[tuple[int, int]]) -> bool:
    return any(start < o_end and o_start < end for o_start, o_end in others)


def _counts(table: dict[str, TypeCounts], type_name: str) -> TypeCounts:
    counts = table.get(type_name)
    if counts is None:
        counts = table[type_name] = TypeCounts()
    return counts


def _error(
    errors: list[dict[str, Any]],
    sample: NerSample,
    index: int,
    kind: str,
    span: tuple[int, int, str],
) -> None:
    start, end, type_name = span
    text = sample.text
    errors.append(
        {
            "sample": index,
            "context": sample.context,
            "kind": kind,
            "type": type_name,
            "start": start,
            "end": end,
            "text": text[start:end],
            "before": text[max(0, start - DUMP_CONTEXT_CHARS) : start],
            "after": text[end : end + DUMP_CONTEXT_CHARS],
        }
    )


def score_sample(
    result: NerResult,
    sample: NerSample,
    label_map: LabelMap,
    found: Sequence[Detection],
    rules_found: Sequence[Detection],
    regex_types: frozenset[str],
    *,
    index: int = 0,
    errors: list[dict[str, Any]] | None = None,
) -> None:
    """Add one sample's scores to ``result``. ``found`` is the full
    pipeline's output, ``rules_found`` the rules-alone output (both after
    overlap resolution). With ``errors`` a list, misses, leaks, false
    positives and regressions are appended to it WITH their text (for
    --dump-errors only)."""
    result.samples += 1
    result.chars += len(sample.text)
    result.contexts[sample.context] += 1

    typed: list[tuple[int, int, str]] = []
    leak: list[tuple[int, int]] = []
    ignored: list[tuple[int, int]] = []
    for span in sample.spans:
        if span.label not in label_map:
            result.unmapped[span.label] += 1
        target = label_map.get(span.label)
        if target is None:
            ignored.append((span.start, span.end))
        elif target == LEAK:
            leak.append((span.start, span.end))
        elif target != NOT_PII:
            typed.append((span.start, span.end, target))

    # Parts of one name or address labelled separately (GIVENNAME + SURNAME)
    # are one gold span, exactly as the detectors' parts merge.
    parts = [Detection(s, e, t, sample.text[s:e]) for s, e, t in typed]
    typed = [(d.start, d.end, d.detector_type) for d in merge_adjacent_parts(parts, sample.text)]

    predictions = [(d.start, d.end, d.detector_type) for d in found]
    gold_set = set(typed)
    pred_set = set(predictions)
    typed_ranges = [(s, e) for s, e, _ in typed]
    neutral_ranges = leak + ignored

    for start, end, type_name in typed:
        exact = _counts(result.exact, type_name)
        overlap = _counts(result.overlap, type_name)
        exact.gold += 1
        overlap.gold += 1
        if (start, end, type_name) in pred_set:
            exact.gold_hit += 1
        elif errors is not None:
            _error(errors, sample, index, "miss", (start, end, type_name))
        if any(t == type_name and s < end and start < e for s, e, t in predictions):
            overlap.gold_hit += 1

    for start, end, type_name in predictions:
        if not _overlaps(start, end, typed_ranges) and _overlaps(start, end, neutral_ranges):
            result.neutral_detections += 1
            continue
        exact = _counts(result.exact, type_name)
        overlap = _counts(result.overlap, type_name)
        exact.pred += 1
        overlap.pred += 1
        if (start, end, type_name) in gold_set:
            exact.pred_hit += 1
        elif errors is not None:
            _error(errors, sample, index, "false_positive", (start, end, type_name))
        if any(t == type_name and s < end and start < e for s, e, t in typed):
            overlap.pred_hit += 1

    gold_pii = _merged([*typed_ranges, *leak])
    every_gold = _merged([*typed_ranges, *leak, *ignored])
    covered = _merged((s, e) for s, e, _ in predictions)
    gold_chars = _length(gold_pii)
    result.gold_chars += gold_chars
    result.leaked_chars += gold_chars - _intersection(gold_pii, covered)
    result.outside_chars += len(sample.text) - _length(every_gold)
    result.over_redacted_chars += _length(covered) - _intersection(covered, every_gold)
    for type_name in sorted({t for _, _, t in typed}):
        of_type = _merged((s, e) for s, e, t in typed if t == type_name)
        type_chars = _length(of_type)
        result.type_gold_chars[type_name] += type_chars
        result.type_leaked_chars[type_name] += type_chars - _intersection(of_type, covered)
    if errors is not None:
        for start, end in leak:
            if _intersection([(start, end)], covered) < end - start:
                _error(errors, sample, index, "leak", (start, end, LEAK))

    rules_set = {(d.start, d.end, d.detector_type) for d in rules_found}
    for gold in typed:
        if gold[2] not in regex_types or gold not in rules_set:
            continue
        counts = result.structured.setdefault(gold[2], StructuredCounts())
        counts.baseline += 1
        if gold in pred_set:
            counts.kept += 1
        elif errors is not None:
            _error(errors, sample, index, "regression", gold)


def evaluate(
    samples: Iterable[NerSample],
    label_map: LabelMap,
    pipeline: Pipeline,
    *,
    errors: list[dict[str, Any]] | None = None,
) -> NerResult:
    """Score every sample through ``pipeline``."""
    result = NerResult()
    started = time.perf_counter()
    for index, sample in enumerate(samples):
        found, rules_found = pipeline.detect(sample.text)
        score_sample(
            result,
            sample,
            label_map,
            found,
            rules_found,
            pipeline.regex_types,
            index=index,
            errors=errors,
        )
    result.seconds = time.perf_counter() - started
    return result


def _table(title: str, table: Mapping[str, TypeCounts]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| type | gold | found | detections | right | precision | recall | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in sorted(table.items()):
        lines.append(
            f"| {name} | {c.gold} | {c.gold_hit} | {c.pred} | {c.pred_hit} |"
            f" {c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} |"
        )
    return [*lines, ""]


def _type_leaks(result: NerResult) -> list[str]:
    if not result.type_gold_chars:
        return []
    leaks = ", ".join(
        f"{name} {result.type_leak_rate(name):.4f}"
        f" ({result.type_leaked_chars[name]} of {result.type_gold_chars[name]})"
        for name in sorted(result.type_gold_chars)
    )
    return [f"Character-leak rate per type: {leaks}."]


def to_markdown(result: NerResult) -> str:
    """The metrics section of a report: counts and rates, never text."""
    lines = [
        f"Samples: {result.samples}, characters: {result.chars}, scored in {result.seconds:.1f} s.",
        "",
        f"**Character-leak rate**: {result.leak_rate:.4f}"
        f" ({result.leaked_chars} of {result.gold_chars} gold characters uncovered).",
        *_type_leaks(result),
        f"**Over-redaction rate**: {result.over_redaction_rate:.4f}"
        f" ({result.over_redacted_chars} of {result.outside_chars} non-gold characters"
        " detected).",
        f"**Structured regressions**: {result.structured_regressions}"
        " (regex-type gold spans the rules alone find exactly but the full pipeline"
        " does not).",
        f"Neutral detections (over unscored gold only): {result.neutral_detections}.",
        "",
        *_table("Overlap-typed", result.overlap),
        *_table("Exact", result.exact),
    ]
    if result.structured:
        lines += [
            "### Structured-regression check",
            "",
            "| type | found by the rules alone | still found | regressions |",
            "|---|---|---|---|",
        ]
        for name, s in sorted(result.structured.items()):
            lines.append(f"| {name} | {s.baseline} | {s.kept} | {s.regressions} |")
        lines.append("")
    if result.unmapped:
        unmapped = ", ".join(f"{label}×{n}" for label, n in sorted(result.unmapped.items()))
        lines += [f"Labels missing from the label map (not scored): {unmapped}.", ""]
    contexts = ", ".join(f"{name or '-'}×{n}" for name, n in sorted(result.contexts.items()))
    lines += [f"Contexts: {contexts}.", ""]
    return "\n".join(lines)


def _counts_json(table: Mapping[str, TypeCounts]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "gold": c.gold,
            "gold_hit": c.gold_hit,
            "pred": c.pred,
            "pred_hit": c.pred_hit,
            "precision": c.precision,
            "recall": c.recall,
            "f1": c.f1,
        }
        for name, c in sorted(table.items())
    }


def to_json_dict(result: NerResult) -> dict[str, object]:
    return {
        "samples": result.samples,
        "chars": result.chars,
        "seconds": result.seconds,
        "leak_rate": result.leak_rate,
        "gold_chars": result.gold_chars,
        "leaked_chars": result.leaked_chars,
        "type_leak": {
            name: {
                "gold_chars": result.type_gold_chars[name],
                "leaked_chars": result.type_leaked_chars[name],
                "leak_rate": result.type_leak_rate(name),
            }
            for name in sorted(result.type_gold_chars)
        },
        "over_redaction_rate": result.over_redaction_rate,
        "outside_chars": result.outside_chars,
        "over_redacted_chars": result.over_redacted_chars,
        "neutral_detections": result.neutral_detections,
        "structured_regressions": result.structured_regressions,
        "structured": {
            name: {"baseline": s.baseline, "kept": s.kept}
            for name, s in sorted(result.structured.items())
        },
        "overlap": _counts_json(result.overlap),
        "exact": _counts_json(result.exact),
        "unmapped_labels": dict(sorted(result.unmapped.items())),
        "contexts": dict(sorted(result.contexts.items())),
    }
