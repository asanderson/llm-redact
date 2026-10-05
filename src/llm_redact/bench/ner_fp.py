"""NER false positives on the vendored negatives corpus (``bench/fp_corpus``).

``python -m llm_redact.bench.ner --fp-corpus bench/fp_corpus --check``
scans every corpus file with the full pipeline and with the same
configuration's rules alone, and counts what NER ADDS: the detections of the
full pipeline that the rules alone do not make (a model span that displaced
a rule's match counts too). The rules' own counts stay gated, exactly, by
``MANIFEST.toml`` and ``python -m llm_redact.bench --check``.

Files are scanned in chunks of whole lines of at most
:data:`CHUNK_CHARS` characters — the size of a message or a tool result,
and below every backend's ``max_chars`` (a whole file in one string would
be skipped by NER and measure nothing).

The gate, ``bench/ner_ceilings.toml``, holds per config name a table per
file of per-type MAXIMUM counts (a type or file not listed allows none) and
an optional ``per_100kb_max`` on the hits per 100 KB of the whole corpus.
Results carry counts and line numbers, never text.
"""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_redact.bench.fp_scan import MANIFEST_NAME
from llm_redact.bench.ner_metrics import DUMP_CONTEXT_CHARS, Pipeline

CHUNK_CHARS = 2000
# Section keys that are not file names.
_SECTION_KEYS = frozenset({"recorded", "note", "per_100kb_max"})


@dataclass
class NerFpFile:
    name: str
    size: int = 0
    chunks: int = 0
    found: Counter[str] = field(default_factory=Counter)
    # 1-based line numbers per type, for humans chasing a failure.
    lines: dict[str, list[int]] = field(default_factory=dict)


def chunks(text: str, limit: int = CHUNK_CHARS) -> Iterator[tuple[int, str]]:
    """(offset, chunk): consecutive whole lines, each chunk at most
    ``limit`` characters unless one line alone is longer."""
    start = 0
    end = 0
    for line in text.splitlines(keepends=True):
        if end > start and end - start + len(line) > limit:
            yield start, text[start:end]
            start = end
        end += len(line)
    if end > start:
        yield start, text[start:end]


def scan(
    root: Path, pipeline: Pipeline, *, errors: list[dict[str, Any]] | None = None
) -> list[NerFpFile]:
    """Every corpus file's NER-added detections, by type. With ``errors`` a
    list, each detection is appended to it WITH its text (--dump-errors)."""
    results = []
    names = sorted(p.name for p in root.iterdir() if p.is_file() and p.name != MANIFEST_NAME)
    for name in names:
        text = (root / name).read_text(encoding="utf-8")
        result = NerFpFile(name=name, size=len(text.encode("utf-8")))
        for offset, chunk in chunks(text):
            result.chunks += 1
            found, rules_found = pipeline.detect(chunk)
            rules = {(d.start, d.end, d.detector_type) for d in rules_found}
            for d in found:
                if (d.start, d.end, d.detector_type) in rules:
                    continue
                result.found[d.detector_type] += 1
                line = text.count("\n", 0, offset + d.start) + 1
                result.lines.setdefault(d.detector_type, []).append(line)
                if errors is not None:
                    errors.append(
                        {
                            "file": name,
                            "line": line,
                            "kind": "false_positive",
                            "type": d.detector_type,
                            "text": chunk[d.start : d.end],
                            "before": chunk[max(0, d.start - DUMP_CONTEXT_CHARS) : d.start],
                            "after": chunk[d.end : d.end + DUMP_CONTEXT_CHARS],
                        }
                    )
        results.append(result)
    return results


def total_hits(files: Sequence[NerFpFile]) -> int:
    return sum(sum(f.found.values()) for f in files)


def per_100kb(files: Sequence[NerFpFile]) -> float:
    size = sum(f.size for f in files)
    return total_hits(files) * 100_000 / size if size else 0.0


def ceiling_failures(
    ceilings: Mapping[str, Any], config_name: str, files: Sequence[NerFpFile], path: Path
) -> list[str]:
    """One line per crossed ceiling (file, type, counts, line numbers). A
    config without a section fails with how to record a baseline; a ceiling
    for a file the corpus lacks is a failure too (a stale entry)."""
    section = ceilings.get(config_name)
    if not isinstance(section, dict):
        return [
            f"no NER ceilings for [{config_name}] in {path}; record a baseline from this"
            " run's report (docs/ner-bench.md, 'Recording a baseline')"
        ]
    failures: list[str] = []
    names = {f.name for f in files}
    for key in sorted(set(section) - _SECTION_KEYS - names):
        failures.append(f"{path} [{config_name}] names {key}, which is not in the corpus")
    for result in files:
        allowed = section.get(result.name, {})
        if not isinstance(allowed, dict):
            raise ValueError(f"[{config_name}] {result.name} must be a table of type = maximum")
        for type_name in sorted(result.found):
            count = result.found[type_name]
            ceiling = allowed.get(type_name, 0)
            if isinstance(ceiling, bool) or not isinstance(ceiling, int):
                raise ValueError(f"[{config_name}] {result.name} {type_name} must be an integer")
            if count > ceiling:
                numbers = ", ".join(str(n) for n in result.lines[type_name][:10])
                failures.append(
                    f"{result.name}: {type_name} found {count}, ceiling {ceiling} (lines {numbers})"
                )
    if "per_100kb_max" in section:
        ceiling_rate = section["per_100kb_max"]
        if isinstance(ceiling_rate, bool) or not isinstance(ceiling_rate, int | float):
            raise ValueError(f"[{config_name}] per_100kb_max must be a number")
        if per_100kb(files) > ceiling_rate:
            failures.append(
                f"NER hits per 100 KB {per_100kb(files):.2f} is above per_100kb_max"
                f" {float(ceiling_rate):.2f}"
            )
    return failures


def to_markdown(files: Sequence[NerFpFile]) -> str:
    lines = [
        "| file | bytes | chunks | NER detections (type×count) |",
        "|---|---|---|---|",
    ]
    for result in files:
        found = " ".join(f"{t}×{n}" for t, n in sorted(result.found.items())) or "—"
        lines.append(f"| {result.name} | {result.size} | {result.chunks} | {found} |")
    size = sum(f.size for f in files)
    lines += [
        "",
        f"Total: {total_hits(files)} NER detections in {size} bytes,"
        f" {per_100kb(files):.2f} per 100 KB.",
    ]
    return "\n".join(lines) + "\n"


def to_json_list(files: Sequence[NerFpFile]) -> list[dict[str, object]]:
    return [
        {
            "file": f.name,
            "bytes": f.size,
            "chunks": f.chunks,
            "found": dict(sorted(f.found.items())),
            "lines": {t: n for t, n in sorted(f.lines.items())},
        }
        for f in files
    ]
