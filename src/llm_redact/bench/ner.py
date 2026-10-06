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
import errno
import itertools
import json
import os
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_redact.bench import ner_fp, ner_latency
from llm_redact.bench.datasets import (
    DATASETS,
    DatasetError,
    DatasetSpec,
    LoadRequest,
    dataset_key,
    default_cache_dir,
    resolve,
)
from llm_redact.bench.datasets.base import fetch
from llm_redact.bench.latency import MANY_SMALL_STRINGS
from llm_redact.bench.ner_metrics import NerResult, Pipeline, evaluate, to_json_dict, to_markdown
from llm_redact.config import ConfigError, load_config
from llm_redact.detection.base import Detector
from llm_redact.detection.engine import DetectionConfig, build_detectors, ner_backends

DEFAULT_THRESHOLDS = Path("bench/ner_thresholds.toml")
DEFAULT_CEILINGS = Path("bench/ner_ceilings.toml")
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
    """A thresholds or ceilings TOML file."""
    if not path.is_file():
        raise BenchError(f"{path} not found")
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


def git_work_tree(path: Path) -> Path | None:
    """The git work tree ``path`` would be inside (None = none): a directory
    at or above it holding ``.git``."""
    directory = path.resolve()
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def dump_problem(path: Path) -> str | None:
    """Why --dump-errors may not write ``path`` (None = it may): the file
    must be outside every git work tree, so dataset text never lands where it
    could be committed."""
    tree = git_work_tree(path.parent)
    if tree is None:
        return None
    return (
        f"--dump-errors must name a file outside any git work tree ({tree} is one);"
        " dataset text must never be committed"
    )


def write_dump(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write the error records as JSON lines to a new or truncated mode-0600
    file (never through a symlink)."""
    if path.is_symlink():
        # O_NOFOLLOW refuses it below where the platform has the flag (POSIX);
        # Windows has none, so the link is refused here too.
        raise OSError(errno.ELOOP, "refusing to write the dump through a symlink", str(path))
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
        if spec.hub_id is not None:
            lines.append(f"  source: {spec.hub_id} at revision {spec.revision}")
        if spec.card is not None:
            lines.append(f"  card: {spec.card} (checked {spec.checked})")
        if spec.filters_language:
            lines.append("  --language filters its rows")
        if spec.needs_data_dir:
            lines.append("  reads a local checkout: --data-dir DIR")
        if spec.needs_path:
            lines.append("  reads a local file: --path FILE")
    return "\n".join(lines) + "\n"


def report_markdown(
    spec: DatasetSpec,
    request: LoadRequest,
    config_name: str,
    backends: Sequence[str],
    result: NerResult,
) -> str:
    split = request.split
    lines = [
        f"# NER bench: {config_name} on {dataset_key(spec, split, request.language)}",
        "",
        f"Dataset: {spec.name} (split {split}) — {spec.summary}.",
        f"License: {spec.license}. Attribution: {spec.attribution}.",
    ]
    if spec.hub_id is not None:
        lines.append(f"Source: {spec.hub_id} at revision {spec.revision}.")
    if request.data_dir is not None:
        lines.append("Source: a local checkout (--data-dir).")
    if request.path is not None:
        lines.append("Source: a local file (--path).")
    if request.language is not None:
        lines.append(f"Rows in language {request.language} only.")
    if spec.real_data:
        lines.append("This dataset holds real data; its text never appears in a report.")
    lines += [*spec.notes, f"NER backends: {', '.join(backends) or 'none'}.", ""]
    if request.skipped:
        reasons = ", ".join(f"{reason}×{n}" for reason, n in sorted(request.skipped.items()))
        lines += [f"Rows skipped: {reasons}.", ""]
    return "\n".join(lines) + "\n" + to_markdown(result)


def report_json(
    spec: DatasetSpec,
    request: LoadRequest,
    config_name: str,
    backends: Sequence[str],
    result: NerResult,
) -> dict[str, object]:
    return {
        "config": config_name,
        "dataset": spec.name,
        "split": request.split,
        "language": request.language,
        "license": spec.license,
        "attribution": spec.attribution,
        "source": None if spec.hub_id is None else f"{spec.hub_id}@{spec.revision}",
        "real_data": spec.real_data,
        "backends": list(backends),
        "skipped_rows": dict(sorted(request.skipped.items())),
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
        "--dataset",
        default="synthetic",
        help="NAME[:SPLIT] (see --list-datasets; default synthetic)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"score at most this many samples (default {DEFAULT_LIMIT}; 0 = all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed of generated datasets")
    parser.add_argument(
        "--language", help="keep only rows in this language (datasets that record one)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="the local checkout a dataset is read from (creddata: a CredData directory"
        " after its download_data.py ran)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        metavar="FILE",
        help="the local file a dataset is read from (agent-eval: the frozen set written by"
        " scripts/pii_corpus/review.py freeze)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="where downloaded datasets are kept (default"
        " ${XDG_CACHE_HOME:-~/.cache}/llm-redact/bench-datasets; never inside a git work tree)",
    )
    parser.add_argument("--out", type=Path, help="write report.md and report.json here")
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when a floor or ceiling is crossed (--thresholds, or --ceilings with"
        " --fp-corpus)",
    )
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument(
        "--fp-corpus",
        type=Path,
        metavar="DIR",
        help="count NER's detections on a negatives corpus (bench/fp_corpus) instead of"
        " scoring a dataset",
    )
    parser.add_argument("--ceilings", type=Path, default=DEFAULT_CEILINGS)
    parser.add_argument(
        "--latency",
        action="store_true",
        help="time NER per string (50 to 10,000 characters) and on a many-small-strings"
        " body instead of scoring a dataset",
    )
    parser.add_argument(
        "--latency-iterations",
        type=int,
        default=ner_latency.ITERATIONS,
        help=f"timed runs per string length (default {ner_latency.ITERATIONS})",
    )
    parser.add_argument(
        "--many-small-strings",
        type=int,
        default=MANY_SMALL_STRINGS,
        help=f"strings in the many-small body (default {MANY_SMALL_STRINGS})",
    )
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
    except (BenchError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    if args.config is None:
        raise BenchError("--config PATH is required (a config with [detection.ner] enabled)")
    if args.fp_corpus is not None and args.latency:
        raise BenchError("--fp-corpus and --latency are separate runs; pick one")
    spec, split = None, ""
    if args.fp_corpus is None and not args.latency:
        try:
            spec, split = resolve(args.dataset)
        except ValueError as exc:
            raise BenchError(str(exc)) from exc
    if args.language is not None and (spec is None or not spec.filters_language):
        what = f"dataset {spec.name!r}" if spec is not None else "this run"
        raise BenchError(f"--language: {what} has no language to filter on")
    if spec is not None and spec.needs_data_dir and args.data_dir is None:
        raise BenchError(
            f"dataset {spec.name!r} reads a local checkout: pass --data-dir (docs/ner-bench.md)"
        )
    if args.data_dir is not None and (spec is None or not spec.needs_data_dir):
        raise BenchError("--data-dir applies only to datasets read from a local checkout")
    if spec is not None and spec.needs_path and args.path is None:
        raise BenchError(
            f"dataset {spec.name!r} reads a local file: pass --path (docs/ner-bench.md)"
        )
    if args.path is not None and (spec is None or not spec.needs_path):
        raise BenchError("--path applies only to datasets read from a local file")
    cache_dir = args.cache_dir or default_cache_dir()
    tree = git_work_tree(cache_dir)
    if tree is not None:
        raise BenchError(
            f"--cache-dir must be outside any git work tree ({tree} is one);"
            " datasets are never committed"
        )
    if args.dump_errors is not None:
        if spec is not None and spec.real_data and not args.allow_real_data_dump:
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
    if spec is not None and spec.hub_id is not None:
        # A published dataset is downloaded BEFORE the models load: building
        # a Hub backend switches the process offline for good
        # (model_files.go_offline), after which the adapter finds the files
        # in the cache only.
        fetch(spec, split, LoadRequest(split=split, cache_dir=cache_dir))
    pipeline, detectors = build_pipeline(config.detection)
    config_name = args.name or args.config.stem
    backends = describe_backends(detectors)
    errors: list[dict[str, Any]] | None = [] if args.dump_errors is not None else None
    if args.latency:
        failures = _run_latency(args, pipeline, detectors, config_name, backends)
        passed = f"[{_toml_key(config_name)}.latency]"
    elif spec is None:
        failures = _run_fp_corpus(args, pipeline, config_name, backends, errors)
        passed = f"[{_toml_key(config_name)}] in {args.ceilings}"
    else:
        request = LoadRequest(
            split=split,
            seed=args.seed,
            language=args.language,
            cache_dir=cache_dir,
            data_dir=args.data_dir,
            path=args.path,
        )
        failures = _run_dataset(args, spec, request, pipeline, config_name, backends, errors)
        key = dataset_key(spec, split, args.language)
        passed = f"[{_toml_key(config_name)}.{_toml_key(key)}]"
    if errors is not None:
        write_dump(args.dump_errors, errors)
        print(f"{len(errors)} error records written to {args.dump_errors} (mode 0600)")
    if failures is None:
        return 0
    for line in failures:
        print(f"CHECK FAILED: {line}")
    if failures:
        return 1
    print(f"check passed: {passed}")
    return 0


def _emit(args: argparse.Namespace, markdown: str, report: Mapping[str, object]) -> None:
    if args.out is None:
        print(markdown)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(markdown)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report written to {args.out}/report.md and report.json")


def _run_dataset(
    args: argparse.Namespace,
    spec: DatasetSpec,
    request: LoadRequest,
    pipeline: Pipeline,
    config_name: str,
    backends: Sequence[str],
    errors: list[dict[str, Any]] | None,
) -> list[str] | None:
    """Score the dataset; the gate's failures with --check, else None."""
    split = request.split
    samples = spec.adapter(spec, request)
    if args.limit > 0:
        samples = itertools.islice(samples, args.limit)
    result = evaluate(samples, spec.label_map, pipeline, errors=errors)
    _emit(
        args,
        report_markdown(spec, request, config_name, backends, result),
        report_json(spec, request, config_name, backends, result),
    )
    if not args.check:
        return None
    thresholds = load_thresholds(args.thresholds)
    key = dataset_key(spec, split, request.language)
    return threshold_failures(thresholds, config_name, key, result, args.thresholds)


def _run_latency(
    args: argparse.Namespace,
    pipeline: Pipeline,
    detectors: Sequence[Detector],
    config_name: str,
    backends: Sequence[str],
) -> list[str] | None:
    """Time NER; with --check, the failures against the optional ceilings
    of [<config>.latency] (none recorded: nothing to fail), else None."""
    if args.dump_errors is not None:
        raise BenchError("--dump-errors applies to dataset and --fp-corpus runs")
    stats = ner_latency.run(
        pipeline,
        detectors,
        iterations=args.latency_iterations,
        many_small=args.many_small_strings,
        seed=args.seed,
    )
    env = ner_latency.environment()
    markdown = (
        f"# NER bench: {config_name} latency\n\n"
        f"NER backends: {', '.join(backends) or 'none'}.\n" + ner_latency.to_markdown(stats, env)
    )
    _emit(
        args,
        markdown,
        {"config": config_name, "backends": list(backends), **ner_latency.to_json(stats, env)},
    )
    if not args.check:
        return None
    section = load_thresholds(args.thresholds).get(config_name)
    entry = section.get("latency") if isinstance(section, dict) else None
    if not isinstance(entry, dict):
        print(f"no latency ceilings recorded for [{_toml_key(config_name)}.latency]; report only")
        return []
    try:
        return ner_latency.ceiling_failures(entry, stats)
    except (ValueError, TypeError) as exc:
        raise BenchError(f"{args.thresholds} [{config_name}.latency]: {exc}") from exc


def _run_fp_corpus(
    args: argparse.Namespace,
    pipeline: Pipeline,
    config_name: str,
    backends: Sequence[str],
    errors: list[dict[str, Any]] | None,
) -> list[str] | None:
    """Count NER's additions on the negatives corpus; the ceilings' failures
    with --check, else None."""
    root: Path = args.fp_corpus
    if not root.is_dir():
        raise BenchError(f"--fp-corpus {root} is not a directory")
    files = ner_fp.scan(root, pipeline, errors=errors)
    markdown = (
        f"# NER bench: {config_name} on the false-positive corpus\n\n"
        f"Corpus: {root}, scanned in chunks of whole lines of at most"
        f" {ner_fp.CHUNK_CHARS} characters; counted: detections the full pipeline"
        " makes and the rules alone do not.\n"
        f"NER backends: {', '.join(backends) or 'none'}.\n\n" + ner_fp.to_markdown(files)
    )
    report = {
        "config": config_name,
        "corpus": str(root),
        "backends": list(backends),
        "hits": ner_fp.total_hits(files),
        "per_100kb": ner_fp.per_100kb(files),
        "files": ner_fp.to_json_list(files),
    }
    _emit(args, markdown, report)
    if not args.check:
        return None
    ceilings = load_thresholds(args.ceilings)
    try:
        return ner_fp.ceiling_failures(ceilings, config_name, files, args.ceilings)
    except ValueError as exc:
        raise BenchError(f"{args.ceilings}: {exc}") from exc


if __name__ == "__main__":
    sys.exit(main())
