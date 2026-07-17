#!/usr/bin/env python3
"""History hygiene sweep: run llm-redact's own detectors over all git history.

Read-only. Scans every commit diff (all refs) with the production detector
stack, aggregates unique (type, value) hits, and classifies each as:

- IN CURRENT TREE: the exact value exists in the working tree — a known,
  reviewable fixture (the current tree is what ships and gets reviewed).
- HISTORY-ONLY: the value appeared in a past commit and is gone now —
  exactly the set that needs eyeballs before the history ever goes public.

Values are MASKED on stdout. Pass an output path to also write the
history-only values unmasked for local review (never commit that file).

    uv run python scripts/history_sweep.py [/path/to/full_report.txt]

The findings of each run are summarized in docs/history-hygiene.md.
"""

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from llm_redact.bench.fp_scan import _NO_ALLOW
from llm_redact.detection.engine import DetectionConfig, build_detectors, detect_all
from llm_redact.redactor import _resolve_overlaps

REPO = Path(__file__).resolve().parent.parent


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def mask(value: str) -> str:
    if len(value) <= 12:
        return value[:2] + "…" + value[-2:]
    return value[:6] + f"…[{len(value)}]…" + value[-4:]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    detectors = build_detectors(DetectionConfig())
    shas = git("log", "--all", "--format=%H").split()
    print(f"scanning {len(shas)} commits", flush=True)

    hits: dict[str, str] = {}
    where: dict[str, set[str]] = defaultdict(set)
    for i, sha in enumerate(shas):
        diff = git("show", "--no-color", "--format=%h", sha)
        detections = _resolve_overlaps(detect_all(detectors, diff, _NO_ALLOW))
        if not detections:
            continue
        # map match offsets to the nearest preceding "+++ b/<file>" marker
        marks: list[tuple[int, str]] = []
        pos = 0
        for line in diff.splitlines(keepends=True):
            if line.startswith("+++ b/"):
                marks.append((pos, line[6:].strip()))
            pos += len(line)
        for d in detections:
            value = diff[d.start : d.end]
            fname = next((name for mark, name in reversed(marks) if mark <= d.start), "?")
            hits[value] = d.detector_type
            where[value].add(f"{sha[:7]}:{fname}")
        if (i + 1) % 25 == 0:
            print(f"  …{i + 1}/{len(shas)}", flush=True)

    corpus = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for f in git("ls-files").split("\n")
        if (p := REPO / f).is_file()
    )
    current: list[tuple[str, str]] = []
    history_only: list[tuple[str, str]] = []
    for value, typ in sorted(hits.items(), key=lambda kv: (kv[1], kv[0])):
        (current if value in corpus else history_only).append((typ, value))

    print(f"total unique values: {len(hits)}")
    print(f"\n== HISTORY-ONLY (review these): {len(history_only)} ==")
    for typ, value in history_only:
        ctx = "; ".join(sorted(where[value])[:3])
        print(f"  {typ:22} {mask(value):44} {ctx}")
    print(f"\n== IN CURRENT TREE (known fixtures): {len(current)} ==")
    by_type: dict[str, int] = defaultdict(int)
    for typ, _ in current:
        by_type[typ] += 1
    for typ, n in sorted(by_type.items()):
        print(f"  {typ:22} {n}")

    if out is not None:
        lines = [
            f"{typ}\t{value}\t{'; '.join(sorted(where[value])[:5])}" for typ, value in history_only
        ]
        out.write_text("== HISTORY-ONLY FULL VALUES ==\n" + "\n".join(lines) + "\n")
        print(f"\nunmasked history-only values written to {out} — do not commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
