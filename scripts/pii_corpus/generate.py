"""Generate agent-traffic PII samples with a local Apache-2.0 LLM (plan T40).

    uv run python scripts/pii_corpus/generate.py --model gemma4:e4b --count 500 --seed 7

Asks a teacher served by a local Ollama (loopback by default; the teacher
policy is in teacher.py) for coding-agent artifacts with invented personal
data tagged inline (prompts.py), grounds every span (grounding.py), drops
what cannot be grounded, and writes JSONL rows

    {"id", "text", "spans": [{"start", "end", "type"}], "teacher", "prompt_id", "seed"}

to a mode-0600 file OUTSIDE every git work tree (default
``${XDG_DATA_HOME:-~/.local/share}/llm-redact/pii-corpus/``), with a
``.manifest.json`` beside it (teacher digest and license check, catalog
digest, options, counts; no text). The rows are unverified: review.py turns
them into a verified set. Output on the terminal is counts only.

Exit status: 0 done, 1 the run was cut short by a teacher error (the rows
already written are kept and the manifest says so), 2 an input problem.
"""

import argparse
import re
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import httpx

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pii_corpus import grounding, prompts  # noqa: E402
from pii_corpus.private_files import (  # noqa: E402
    CorpusError,
    default_data_dir,
    json_line,
    open_private,
    output_problem,
    write_json,
)
from pii_corpus.teacher import (  # noqa: E402
    DEFAULT_URL,
    OllamaClient,
    TeacherError,
    TeacherInfo,
    server_problem,
)

MANIFEST_FORMAT = "llm-redact-pii-corpus-generated/1"
DEFAULT_NEGATIVES = 0.25


def slug(model: str) -> str:
    """A model name as a file-name and id part (``gemma4:e4b`` -> ``gemma4-e4b``)."""
    return re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")


def manifest_path(out: Path) -> Path:
    return out.with_name(out.name + ".manifest.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/pii_corpus/generate.py",
        description="Generate tagged agent-traffic PII samples with a local Apache-2.0 LLM.",
    )
    parser.add_argument("--model", required=True, help="Ollama NAME:TAG on the teacher allowlist")
    parser.add_argument("--count", type=int, default=100, help="prompts to send (default 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negatives",
        type=float,
        default=DEFAULT_NEGATIVES,
        help=f"share of hard-negative prompts (default {DEFAULT_NEGATIVES})",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Ollama server (default {DEFAULT_URL})")
    parser.add_argument(
        "--allow-remote-server",
        action="store_true",
        help="allow a non-loopback server (https only); it receives every prompt",
    )
    parser.add_argument(
        "--out", type=Path, help="output JSONL (default: under the private data directory)"
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    return parser


def _row(
    info: TeacherInfo, prompt: prompts.Prompt, seed: int, index: int, found: grounding.Grounded
) -> dict[str, Any]:
    return {
        "id": f"{slug(info.model)}-{seed}-{index:06d}",
        "text": found.text,
        "spans": [span.as_json() for span in found.spans],
        "teacher": info.model,
        "prompt_id": prompt.prompt_id,
        "seed": seed,
    }


def generate(
    client: OllamaClient,
    info: TeacherInfo,
    out: IO[str],
    *,
    count: int,
    seed: int,
    negatives: float,
) -> tuple[Counter[str], str | None]:
    """Send ``count`` prompts and write every grounded row to ``out``. The
    counts (written, propagated, by drop reason) and the teacher error that
    cut the run short, if one did."""
    counts: Counter[str] = Counter()
    for index in range(count):
        prompt = prompts.build(seed, index, negatives)
        try:
            answer = client.chat(
                info.model, prompts.SYSTEM, prompt.user, seed=seed, json_format=False
            )
        except TeacherError as exc:
            return counts, str(exc)
        # SYSTEM asks for EVERY personal value tagged with a catalog type, so
        # a tag of a type this prompt did not ask for is kept, not dropped.
        found = grounding.ground(answer, types=prompts.TYPES, negative=prompt.negative)
        if isinstance(found, str):
            counts[f"dropped: {found}"] += 1
            continue
        if not set(prompt.types) <= {span.type for span in found.spans}:
            counts["rows missing an asked type"] += 1
        out.write(json_line(_row(info, prompt, seed, index, found)))
        out.flush()
        counts["written"] += 1
        counts["negatives written" if prompt.negative else "positives written"] += 1
        counts["spans"] += len(found.spans)
        counts["repeats made gold"] += found.propagated
    return counts, None


def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    args = _parser().parse_args(argv)
    problem = server_problem(args.url, allow_remote=args.allow_remote_server)
    if problem is None and args.count < 1:
        problem = "--count must be at least 1"
    if problem is None and not 0.0 <= args.negatives <= 1.0:
        problem = "--negatives must be between 0 and 1"
    out: Path = args.out or default_data_dir() / f"generated-{slug(args.model)}-{args.seed}.jsonl"
    if problem is None:
        problem = output_problem(out)
    if problem is not None:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    client = OllamaClient(args.url, transport=transport)
    try:
        return _run(args, client, out)
    finally:
        client.close()


def _run(args: argparse.Namespace, client: OllamaClient, out: Path) -> int:
    try:
        info = client.verify(args.model)
        handle = open_private(out, overwrite=args.force)
    except (TeacherError, CorpusError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    started = datetime.now(UTC).isoformat(timespec="seconds")
    with handle:
        counts, failure = generate(
            client, info, handle, count=args.count, seed=args.seed, negatives=args.negatives
        )
    manifest = {
        "format": MANIFEST_FORMAT,
        "rows_file": out.name,
        "teacher": {"model": info.model, "digest": info.digest, "license": info.license},
        "server": "loopback" if server_problem(args.url, allow_remote=False) is None else "remote",
        "catalog_version": prompts.CATALOG_VERSION,
        "catalog_sha256": prompts.catalog_sha256(),
        "options": {"temperature": 0, "seed": args.seed},
        "prompts_requested": args.count,
        "negatives": args.negatives,
        "started": started,
        "complete": failure is None,
        "counts": dict(sorted(counts.items())),
        "verified": False,
    }
    write_json(manifest_path(out), manifest, overwrite=True)
    summary = ", ".join(f"{key} {n}" for key, n in sorted(counts.items())) or "nothing"
    print(f"{out}: {summary}")
    if failure is not None:
        print(f"error: stopped after a teacher error: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
