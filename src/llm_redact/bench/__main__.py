import argparse
import json
import sys
from pathlib import Path

from llm_redact.bench.corpus import generate
from llm_redact.bench.fp_scan import fp_failures, scan_fp_corpus, to_markdown_table
from llm_redact.bench.latency import (
    ceiling_failures,
    run_latency,
    to_json_list,
)
from llm_redact.bench.latency import (
    to_markdown as latency_to_markdown,
)
from llm_redact.bench.metrics import evaluate, to_json_dict, to_markdown

DEFAULT_FP_CORPUS = Path("bench/fp_corpus")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_redact.bench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=25, help="positives per rule")
    parser.add_argument("--out", type=Path, default=None, help="write report files here")
    parser.add_argument(
        "--fp-corpus",
        type=Path,
        default=None,
        help=f"false-positive corpus root (default: {DEFAULT_FP_CORPUS} when present)",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="also run the in-process latency benchmark (adds ~1 minute)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any rule misses its own generated positives (recall < 1.0),"
        " the fp corpus deviates from its manifest, or (with --latency) a p50"
        " smoke ceiling is crossed",
    )
    args = parser.parse_args(argv)

    corpus = generate(seed=args.seed, samples_per_rule=args.samples)
    result = evaluate(corpus)
    markdown = to_markdown(result, seed=args.seed)

    fp_root = args.fp_corpus
    if fp_root is None and DEFAULT_FP_CORPUS.is_dir():
        fp_root = DEFAULT_FP_CORPUS
    fp_results = None
    if fp_root is not None:
        fp_results = scan_fp_corpus(fp_root)
        markdown += (
            "\n## False-positive corpus\n\n"
            f"Vendored negatives at `{fp_root}`, gated on exact per-file counts.\n\n"
            + to_markdown_table(fp_results)
            + "\n"
        )

    latency_stats = None
    if args.latency:
        latency_stats = run_latency(seed=args.seed)
        markdown += "\n" + latency_to_markdown(latency_stats)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.md").write_text(markdown)
        report = to_json_dict(result, seed=args.seed)
        if latency_stats is not None:
            report["latency"] = to_json_list(latency_stats)
        (args.out / "report.json").write_text(json.dumps(report, indent=2))
        print(f"report written to {args.out}/report.md and report.json")
    else:
        print(markdown)

    if args.check:
        failed = False
        # Functional regression gate: the corpus is generated to match each
        # rule, so a missed positive is a code regression, not corpus noise.
        misses = {name: s for name, s in result.per_type.items() if s.recall < 1.0}
        for name, score in misses.items():
            print(f"CHECK FAILED: {name} recall {score.recall:.3f} ({score.fn} missed)")
            failed = True
        # Precision gate: the vendored negatives must match their manifest
        # exactly, in both directions.
        if fp_results is not None:
            for line in fp_failures(fp_results):
                print(f"CHECK FAILED (fp corpus): {line}")
                failed = True
        # Latency smoke ceilings: generous p50-only bounds against
        # accidental quadratic behavior, not perf tuning.
        if latency_stats is not None:
            for line in ceiling_failures(latency_stats):
                print(f"CHECK FAILED (latency): {line}")
                failed = True
        if failed:
            return 1
        fp_note = f", fp corpus clean ({len(fp_results)} files)" if fp_results is not None else ""
        latency_note = ", latency ceilings ok" if latency_stats is not None else ""
        print(
            f"check passed: recall 1.0 on all {len(result.per_type)} types{fp_note}{latency_note}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
