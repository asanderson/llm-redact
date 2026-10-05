"""NER latency: ``python -m llm_redact.bench.ner --config CONFIG --latency``.

Times detection with NER enabled, per string, for strings of
:data:`LENGTHS` characters — each NER backend's own ``detect`` (per backend
and model) and the full pipeline (every detector plus overlap resolution)
— and the many-small-strings body the regex latency bench uses
(:data:`llm_redact.bench.latency.MANY_SMALL_STRINGS` short chat messages)
redacted end to end by a :class:`~llm_redact.redactor.Redactor`: NER runs
once per string, which is what one such request costs the event loop.

Report only by default; ``--check`` applies the optional ceilings of
``[<config>.latency]`` in the thresholds file. The report names the CPU
model, since every number depends on it. The texts are built from the
synthetic corpus (names, addresses, dates: something for a model to find);
nothing of them is printed.
"""

import platform
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from llm_redact.bench import ner_corpus
from llm_redact.bench.latency import MANY_SMALL_STRINGS, _many_small_body, _quantiles, _time
from llm_redact.bench.ner_metrics import Pipeline
from llm_redact.detection.base import Detector
from llm_redact.detection.engine import Allowlist, ner_backends
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault

LENGTHS = (50, 500, 2_000, 10_000)
ITERATIONS = 20
MANY_SMALL_ITERATIONS = 3
PIPELINE = "pipeline"
CPUINFO = Path("/proc/cpuinfo")
_CEILING_KEYS = frozenset({"p50_ms", "p95_ms", "many_small_ms", "recorded", "note"})


@dataclass(frozen=True)
class NerLatencyStat:
    # PIPELINE (every detector + overlap resolution), "many_small" (a whole
    # body redacted) or "backend: model" (one NER backend's detect alone).
    target: str
    length: int  # characters per string (many_small: the body's string count)
    p50_ms: float
    p95_ms: float
    iterations: int


def cpu_model(cpuinfo: Path = CPUINFO) -> str:
    """The CPU's model name (Linux /proc/cpuinfo), else what the platform
    module reports."""
    try:
        for line in cpuinfo.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name" and value.strip():
                return value.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def text_of_length(length: int, seed: int = 42) -> str:
    """Exactly ``length`` characters of corpus text (positive contexts:
    names, addresses and dates a model can find)."""
    parts: list[str] = []
    size = 0
    for sample in ner_corpus.generate(seed=seed):
        if sample.context.startswith("neg-"):
            continue
        parts.append(sample.text)
        size += len(sample.text) + 1
        if size >= length:
            break
    text = " ".join(parts)
    while len(text) < length:  # a corpus shorter than the target repeats
        text = f"{text} {text}"
    return text[:length]


def _stat(target: str, length: int, func: Callable[[], object], iterations: int) -> NerLatencyStat:
    p50, p95 = _quantiles(_time(func, iterations, warmup=1))
    return NerLatencyStat(target, length, p50, p95, iterations)


def describe(backend: Detector) -> str:
    policy = getattr(backend, "label_policy", None)
    name = getattr(policy, "backend", "") or getattr(backend, "name", "ner")
    return f"{name}: {getattr(backend, 'model_name', None) or 'default model'}"


def run(
    pipeline: Pipeline,
    detectors: Sequence[Detector],
    *,
    lengths: Sequence[int] = LENGTHS,
    iterations: int = ITERATIONS,
    many_small: int = MANY_SMALL_STRINGS,
    many_small_iterations: int = MANY_SMALL_ITERATIONS,
    seed: int = 42,
) -> list[NerLatencyStat]:
    stats = []
    for length in lengths:
        text = text_of_length(length, seed)
        for backend in ner_backends(detectors):
            stats.append(
                _stat(describe(backend), length, partial(backend.detect, text), iterations)
            )
        stats.append(_stat(PIPELINE, length, partial(pipeline.detect, text), iterations))
    body = _many_small_body(random.Random(seed), many_small)
    redactor = Redactor(pipeline.full, InMemoryVault(), Allowlist())
    redact = partial(redactor.redact_json, body)
    stats.append(_stat("many_small", many_small, redact, many_small_iterations))
    return stats


def ceiling_failures(entry: Mapping[str, Any], stats: Sequence[NerLatencyStat]) -> list[str]:
    """The crossed ceilings of a ``[<config>.latency]`` entry: ``p50_ms`` and
    ``p95_ms`` tables keyed by string length (the full pipeline), and
    ``many_small_ms`` on the many-small body's p50."""
    unknown = sorted(set(entry) - _CEILING_KEYS)
    if unknown:
        raise ValueError(f"unknown latency key(s) {unknown}")
    pipeline = {s.length: s for s in stats if s.target == PIPELINE}
    failures = []
    for key, attribute in (("p50_ms", "p50_ms"), ("p95_ms", "p95_ms")):
        table = entry.get(key, {})
        if not isinstance(table, dict):
            raise ValueError(f"{key} must be a table of length = milliseconds")
        for raw_length, ceiling in sorted(table.items()):
            stat = pipeline.get(int(raw_length))
            if stat is None:
                failures.append(f"{key} ceiling for {raw_length} characters: not measured")
            elif getattr(stat, attribute) > float(ceiling):
                failures.append(
                    f"{key} at {raw_length} characters: {getattr(stat, attribute):.1f} ms"
                    f" is above the ceiling {float(ceiling):.1f} ms"
                )
    many = next((s for s in stats if s.target == "many_small"), None)
    if "many_small_ms" in entry:
        if many is None:
            failures.append("many_small_ms ceiling: not measured")
        elif many.p50_ms > float(entry["many_small_ms"]):
            failures.append(
                f"many_small_ms: {many.p50_ms:.1f} ms is above the ceiling"
                f" {float(entry['many_small_ms']):.1f} ms"
            )
    return failures


def environment() -> dict[str, str]:
    return {
        "cpu": cpu_model(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def to_markdown(stats: Sequence[NerLatencyStat], env: Mapping[str, str]) -> str:
    lines = [
        f"CPU: {env['cpu']}. Python {env['python']} on {env['platform']}.",
        "",
        "| what | characters (strings) | p50 ms | p95 ms | iterations |",
        "|---|---|---|---|---|",
    ]
    for stat in stats:
        size = f"({stat.length} strings)" if stat.target == "many_small" else str(stat.length)
        lines.append(
            f"| {stat.target} | {size} | {stat.p50_ms:.2f} | {stat.p95_ms:.2f} |"
            f" {stat.iterations} |"
        )
    return "\n".join(lines) + "\n"


def to_json(stats: Sequence[NerLatencyStat], env: Mapping[str, str]) -> dict[str, object]:
    return {
        **env,
        "stats": [
            {
                "target": s.target,
                "length": s.length,
                "p50_ms": s.p50_ms,
                "p95_ms": s.p95_ms,
                "iterations": s.iterations,
            }
            for s in stats
        ],
    }
