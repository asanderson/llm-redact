"""Exact-span scoring of the detection pipeline over a labeled corpus."""

import time
from collections import defaultdict
from dataclasses import dataclass

from llm_redact.bench.corpus import Sample
from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors, detect_all
from llm_redact.redactor import _resolve_overlaps

_NO_ALLOW = Allowlist(exact=frozenset(), patterns=())


@dataclass
class TypeScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class BenchResult:
    per_type: dict[str, TypeScore]
    overall: TypeScore
    seconds_per_mb: float
    corpus_bytes: int
    samples: int


def evaluate(corpus: list[Sample], config: DetectionConfig | None = None) -> BenchResult:
    """Score the full pipeline (detect → overlap resolution) with exact
    span+type matching."""
    detectors = build_detectors(config or DetectionConfig())
    per_type: dict[str, TypeScore] = defaultdict(TypeScore)
    overall = TypeScore()

    total_bytes = 0
    started = time.perf_counter()
    for sample in corpus:
        total_bytes += len(sample.text.encode())
        detections = _resolve_overlaps(detect_all(detectors, sample.text, _NO_ALLOW))
        found = {(d.start, d.end, d.detector_type) for d in detections}
        expected = {(s.start, s.end, s.detector_type) for s in sample.spans}
        for span in found & expected:
            per_type[span[2]].tp += 1
            overall.tp += 1
        for span in found - expected:
            per_type[span[2]].fp += 1
            overall.fp += 1
        for span in expected - found:
            per_type[span[2]].fn += 1
            overall.fn += 1
    elapsed = time.perf_counter() - started

    return BenchResult(
        per_type=dict(sorted(per_type.items())),
        overall=overall,
        seconds_per_mb=elapsed / (total_bytes / 1_000_000) if total_bytes else 0.0,
        corpus_bytes=total_bytes,
        samples=len(corpus),
    )


def to_markdown(result: BenchResult, *, seed: int) -> str:
    lines = [
        "# llm-redact detection benchmark",
        "",
        f"Synthetic corpus: {result.samples} samples, {result.corpus_bytes} bytes,"
        f" seed {seed}. Exact span+type matching over the full pipeline.",
        "",
        f"**Overall**: precision {result.overall.precision:.3f},"
        f" recall {result.overall.recall:.3f}, F1 {result.overall.f1:.3f}."
        f" Throughput: {result.seconds_per_mb:.3f} s/MB.",
        "",
        "| type | TP | FP | FN | precision | recall | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, score in result.per_type.items():
        lines.append(
            f"| {name} | {score.tp} | {score.fp} | {score.fn} |"
            f" {score.precision:.3f} | {score.recall:.3f} | {score.f1:.3f} |"
        )
    return "\n".join(lines) + "\n"


def to_json_dict(result: BenchResult, *, seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "samples": result.samples,
        "corpus_bytes": result.corpus_bytes,
        "seconds_per_mb": result.seconds_per_mb,
        "overall": {
            "tp": result.overall.tp,
            "fp": result.overall.fp,
            "fn": result.overall.fn,
            "precision": result.overall.precision,
            "recall": result.overall.recall,
            "f1": result.overall.f1,
        },
        "per_type": {
            name: {
                "tp": s.tp,
                "fp": s.fp,
                "fn": s.fn,
                "precision": s.precision,
                "recall": s.recall,
                "f1": s.f1,
            }
            for name, s in result.per_type.items()
        },
    }
