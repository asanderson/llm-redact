"""False-positive scan over the vendored negatives corpus (bench/fp_corpus).

The recall benchmark generates its positives at runtime, so committing them
is forbidden (they are secret-shaped). Negatives contain no secrets by
construction, so a real-world corpus CAN be vendored — public-domain code,
prose, and RFC text plus authored probe files aimed at the noisy rules.

MANIFEST.toml pins the exact expected detector-type counts per file (an
absent type means zero). The gate is exact equality in both directions: an
extra detection is a precision regression, and a disappeared expected one
is behavior drift (e.g. a documented known-FP that a rule change silently
fixed — good news, but the manifest must be updated to say so).
"""

import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors, detect_all
from llm_redact.redactor import _resolve_overlaps

MANIFEST_NAME = "MANIFEST.toml"

# Same no-allowlist pipeline as the recall benchmark: the gate measures the
# detectors themselves, not a particular allowlist configuration.
_NO_ALLOW = Allowlist(exact=frozenset(), patterns=())


@dataclass
class FpFileResult:
    name: str
    found: dict[str, int] = field(default_factory=dict)
    expected: dict[str, int] = field(default_factory=dict)
    # 1-based line numbers per detector type — for humans chasing a failure;
    # matched text is never carried anywhere.
    lines: dict[str, list[int]] = field(default_factory=dict)
    missing_file: bool = False
    # False when the file exists on disk but has no manifest entry: implicit
    # all-zeros would hide a typo'd manifest key, so it is a failure.
    listed: bool = True


def load_manifest(root: Path) -> dict[str, dict[str, int]]:
    raw = tomllib.loads((root / MANIFEST_NAME).read_text())
    files = raw.get("files", {})
    return {name: {str(t): int(n) for t, n in expected.items()} for name, expected in files.items()}


def scan_fp_corpus(root: Path, config: DetectionConfig | None = None) -> list[FpFileResult]:
    """Scan every corpus file and pair it with its manifest expectation.

    The union of on-disk files and manifest entries is covered, so an
    unlisted new file and a manifest entry whose file vanished both surface
    as results (and become failures in fp_failures).
    """
    manifest = load_manifest(root)
    detectors = build_detectors(config or DetectionConfig())

    on_disk = sorted(p.name for p in root.iterdir() if p.is_file() and p.name != MANIFEST_NAME)
    results: list[FpFileResult] = []
    for name in sorted(set(on_disk) | set(manifest)):
        expected = manifest.get(name, {})
        path = root / name
        if not path.is_file():
            results.append(FpFileResult(name=name, expected=expected, missing_file=True))
            continue
        text = path.read_text(encoding="utf-8")
        detections = _resolve_overlaps(detect_all(detectors, text, _NO_ALLOW))
        found: Counter[str] = Counter(d.detector_type for d in detections)
        lines: dict[str, list[int]] = {}
        for d in detections:
            lines.setdefault(d.detector_type, []).append(text.count("\n", 0, d.start) + 1)
        results.append(
            FpFileResult(
                name=name,
                found=dict(found),
                expected=expected,
                lines=lines,
                listed=name in manifest,
            )
        )
    return results


def fp_failures(results: list[FpFileResult]) -> list[str]:
    """Human-readable gate failures; empty list = corpus is clean.

    Lines name files, detector types, counts, and line numbers — never the
    matched text.
    """
    failures: list[str] = []
    for result in results:
        if result.missing_file:
            failures.append(f"{result.name}: listed in {MANIFEST_NAME} but missing on disk")
            continue
        if not result.listed:
            failures.append(f"{result.name}: on disk but not listed in {MANIFEST_NAME}")
        for detector_type in sorted(set(result.found) | set(result.expected)):
            found = result.found.get(detector_type, 0)
            expected = result.expected.get(detector_type, 0)
            if found == expected:
                continue
            where = ""
            if result.lines.get(detector_type):
                numbers = ", ".join(str(n) for n in result.lines[detector_type][:10])
                where = f" (lines {numbers})"
            failures.append(
                f"{result.name}: {detector_type} expected {expected}, found {found}{where}"
            )
    return failures


def to_markdown_table(results: list[FpFileResult]) -> str:
    lines = [
        "| file | detections (type×count) | expected |",
        "|---|---|---|",
    ]
    for result in results:
        found = (
            " ".join(f"{t}×{n}" for t, n in sorted(result.found.items())) if result.found else "—"
        )
        expected = (
            " ".join(f"{t}×{n}" for t, n in sorted(result.expected.items()))
            if result.expected
            else "—"
        )
        lines.append(f"| {result.name} | {found} | {expected} |")
    return "\n".join(lines)
