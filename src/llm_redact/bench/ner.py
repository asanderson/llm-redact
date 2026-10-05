"""NER bench: ``python -m llm_redact.bench.ner``.

Scores the full detection pipeline (the regex rules plus the configured NER
backends, overlap resolution, no allowlist) on a labelled dataset with the
statistical metrics of :mod:`llm_redact.bench.ner_metrics`, and with
``--check`` gates the result against ``bench/ner_thresholds.toml`` (recall
floors per type, character-leak and over-redaction ceilings, keyed
``[<config name>.<dataset>]``). The deterministic gate
``python -m llm_redact.bench --check`` (recall == 1.0 per rule, exact
false-positive counts) is a different gate and stays unchanged.

Reports are metadata only: types, counts, rates, label names, model ids.
``--dump-errors PATH`` is the one way to see the text of the misses: it
writes them to a mode-0600 file outside any git work tree, and refuses a
real-data dataset unless ``--allow-real-data-dump`` is also given.
"""

import argparse
import itertools
import json
import os
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_redact.bench.datasets import DATASETS, DatasetSpec, LoadRequest, dataset_key, resolve
from llm_redact.bench.ner_metrics import NerResult, Pipeline, evaluate, to_json_dict, to_markdown
from llm_redact.config import ConfigError, load_config
from llm_redact.detection.base import Detector
from llm_redact.detection.engine import DetectionConfig, build_detectors, ner_backends

DEFAULT_THRESHOLDS = Path("bench/ner_thresholds.toml")
DEFAULT_LIMIT = 2000

# Keys a thresholds entry may hold: floors per type, ceilings, and two
# free-text fields for the baseline's provenance.
_FLOOR_KEYS = ("recall", "exact_recall")
_CEILING_KEYS = ("leak_max", "over_redaction_max")
_THRESHOLD_KEYS = frozenset(
    {*_FLOOR_KEYS, *_CEILING_KEYS, "structured_regressions_max", "recorded", "note"}
)


class BenchError(Exception):
    """A problem with the bench's inputs (config, thresholds, a dataset):
    printed and exit status 2. Messages name keys, files, datasets and
    types, never text from a dataset."""


def build_pipeline(detection: DetectionConfig) -> tuple[Pipeline, list[Detector]]:
    """The full pipeline of ``detection`` and its NER-off baseline, through
    the public detector builder (models load here). NER must be enabled."""
    if not detection.ner.enabled:
        raise BenchError(
            "the NER bench needs [detection.ner] enabled = true in the config it scores"
        )
    try:
        full = build_detectors(detection)
        rules = build_detectors(replace(detection, ner=replace(detection.ner, enabled=False)))
    except ValueError as exc:  # ConfigError included
        raise BenchError(str(exc)) from exc
    return Pipeline.from_detectors(full, rules), full


def describe_backends(detectors: Sequence[Detector]) -> list[str]:
    """``backend: model`` for each NER backend (config names and model ids)."""
    described = []
    for backend in ner_backends(detectors):
        policy = getattr(backend, "label_policy", None)
        name = getattr(policy, "backend", "") or getattr(backend, "name", "ner")
        model = getattr(backend, "model_name", None) or "default model"
        described.append(f"{name}: {model}")
    return described


def load_thresholds(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BenchError(f"thresholds file {path} not found")
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise BenchError(f"{path}: {exc}") from exc


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchError(f"{where} must be a number")
    return float(value)


def threshold_failures(
    thresholds: Mapping[str, Any], config_name: str, key: str, result: NerResult, path: Path
) -> list[str]:
    """The gate: one line per crossed floor or ceiling (names, types and
    numbers only). A missing entry is a failure that says how to record a
    baseline."""
    where = f"[{_toml_key(config_name)}.{_toml_key(key)}]"
    section = thresholds.get(config_name)
    entry = section.get(key) if isinstance(section, dict) else None
    if not isinstance(entry, dict):
        return [
            f"no thresholds for {where} in {path}; record a baseline from this run's"
            " report (docs/ner-bench.md, 'Recording a baseline')"
        ]
    unknown = sorted(set(entry) - _THRESHOLD_KEYS)
    if unknown:
        raise BenchError(f"{where} in {path}: unknown key(s) {unknown}")
    failures: list[str] = []
    for floor_key, table in (("recall", result.overlap), ("exact_recall", result.exact)):
        floors = entry.get(floor_key, {})
        if not isinstance(floors, dict):
            raise BenchError(f"{where} {floor_key} must be a table of type = floor")
        for type_name, raw in sorted(floors.items()):
            floor = _number(raw, f"{where} {floor_key}.{type_name}")
            counts = table.get(type_name)
            if counts is None or counts.gold == 0:
                failures.append(
                    f"{floor_key} floor for {type_name} cannot be checked: the run holds no"
                    f" gold spans of {type_name}"
                )
            elif counts.recall < floor:
                failures.append(
                    f"{type_name} {floor_key} {counts.recall:.3f} is below the floor {floor:.3f}"
                )
    for ceiling_key, measured in (
        ("leak_max", result.leak_rate),
        ("over_redaction_max", result.over_redaction_rate),
    ):
        if ceiling_key in entry:
            ceiling = _number(entry[ceiling_key], f"{where} {ceiling_key}")
            if measured > ceiling:
                failures.append(f"{ceiling_key}: {measured:.4f} is above the ceiling {ceiling:.4f}")
    allowed = int(_number(entry.get("structured_regressions_max", 0), f"{where} regressions"))
    if result.structured_regressions > allowed:
        failures.append(
            f"structured regressions: {result.structured_regressions} gold spans the rules"
            f" alone find exactly are lost with NER on (allowed {allowed})"
        )
    return failures


def _toml_key(key: str) -> str:
    bare = key.replace("-", "").replace("_", "").isalnum()
    return key if bare else f'"{key}"'


def dump_problem(path: Path) -> str | None:
    """Why --dump-errors may not write ``path`` (None = it may): the file
    must be outside every git work tree, so dataset text never lands where it
    could be committed."""
    directory = path.parent.resolve()
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            return (
                f"--dump-errors must name a file outside any git work tree"
                f" ({candidate} is one); dataset text must never be committed"
            )
    return None


def write_dump(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write the error records as JSON lines to a new or truncated mode-0600
    file (never through a symlink)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        os.fchmod(out.fileno(), 0o600)
        for record in records:
            out.write(json.dumps(record) + "\n")


def list_datasets(datasets: Mapping[str, DatasetSpec] = DATASETS) -> str:
    lines = []
    for spec in sorted(datasets.values(), key=lambda s: s.name):
        real = "  REAL DATA" if spec.real_data else ""
        lines += [
            f"{spec.name}: {spec.summary}{real}",
            f"  splits: {', '.join(spec.splits)} (default {spec.default_split})",
            f"  license: {spec.license}",
            f"  attribution: {spec.attribution}",
        ]
    return "\n".join(lines) + "\n"


def report_markdown(
    spec: DatasetSpec,
    split: str,
    config_name: str,
    backends: Sequence[str],
    result: NerResult,
    skipped: Mapping[str, int],
) -> str:
    lines = [
        f"# NER bench: {config_name} on {dataset_key(spec, split)}",
        "",
        f"Dataset: {spec.name} (split {split}) — {spec.summary}.",
        f"License: {spec.license}. Attribution: {spec.attribution}.",
    ]
    if spec.real_data:
        lines.append("This dataset holds real data; its text never appears in a report.")
    lines += [*spec.notes, f"NER backends: {', '.join(backends) or 'none'}.", ""]
    if skipped:
        reasons = ", ".join(f"{reason}×{n}" for reason, n in sorted(skipped.items()))
        lines += [f"Rows skipped: {reasons}.", ""]
    return "\n".join(lines) + "\n" + to_markdown(result)


def report_json(
    spec: DatasetSpec,
    split: str,
    config_name: str,
    backends: Sequence[str],
    result: NerResult,
    skipped: Mapping[str, int],
) -> dict[str, object]:
    return {
        "config": config_name,
        "dataset": spec.name,
        "split": split,
        "license": spec.license,
        "attribution": spec.attribution,
        "real_data": spec.real_data,
        "backends": list(backends),
        "skipped_rows": dict(sorted(skipped.items())),
        **to_json_dict(result),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llm_redact.bench.ner",
        description="Score regex rules + NER on a labelled dataset (docs/ner-bench.md).",
    )
    parser.add_argument("--config", type=Path, help="llm-redact config with NER enabled")
    parser.add_argument(
        "--name", help="config name for the thresholds key (default: the config file's stem)"
    )
    parser.add_argument(
        "--dataset", default="rules", help="NAME[:SPLIT] (see --list-datasets; default rules)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"score at most this many samples (default {DEFAULT_LIMIT}; 0 = all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed of generated datasets")
    parser.add_argument("--out", type=Path, help="write report.md and report.json here")
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--check", action="store_true", help="exit 1 when a floor or ceiling is crossed"
    )
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument(
        "--dump-errors",
        type=Path,
        metavar="PATH",
        help="write misses and false positives WITH their text to PATH (mode 0600,"
        " outside any git work tree)",
    )
    parser.add_argument(
        "--allow-real-data-dump",
        action="store_true",
        help="let --dump-errors write the text of a real-data dataset",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_datasets:
        print(list_datasets(), end="")
        return 0
    try:
        return _run(args)
    except BenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    if args.config is None:
        raise BenchError("--config PATH is required (a config with [detection.ner] enabled)")
    try:
        spec, split = resolve(args.dataset)
    except ValueError as exc:
        raise BenchError(str(exc)) from exc
    if args.dump_errors is not None:
        if spec.real_data and not args.allow_real_data_dump:
            raise BenchError(
                f"dataset {spec.name!r} holds real data; --dump-errors would write its text"
                " to disk: add --allow-real-data-dump to confirm"
            )
        problem = dump_problem(args.dump_errors)
        if problem is not None:
            raise BenchError(problem)
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        raise BenchError(f"config {args.config}: {exc}") from exc
    pipeline, detectors = build_pipeline(config.detection)
    config_name = args.name or args.config.stem
    backends = describe_backends(detectors)

    request = LoadRequest(split=split, seed=args.seed)
    samples = spec.adapter(spec, request)
    if args.limit > 0:
        samples = itertools.islice(samples, args.limit)
    errors: list[dict[str, Any]] | None = [] if args.dump_errors is not None else None
    result = evaluate(samples, spec.label_map, pipeline, errors=errors)

    markdown = report_markdown(spec, split, config_name, backends, result, request.skipped)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.md").write_text(markdown)
        report = report_json(spec, split, config_name, backends, result, request.skipped)
        (args.out / "report.json").write_text(json.dumps(report, indent=2))
        print(f"report written to {args.out}/report.md and report.json")
    else:
        print(markdown)
    if errors is not None:
        write_dump(args.dump_errors, errors)
        print(f"{len(errors)} error records written to {args.dump_errors} (mode 0600)")

    if not args.check:
        return 0
    thresholds = load_thresholds(args.thresholds)
    failures = threshold_failures(
        thresholds, config_name, dataset_key(spec, split), result, args.thresholds
    )
    for line in failures:
        print(f"CHECK FAILED: {line}")
    if failures:
        return 1
    print(f"check passed: [{_toml_key(config_name)}.{_toml_key(dataset_key(spec, split))}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
