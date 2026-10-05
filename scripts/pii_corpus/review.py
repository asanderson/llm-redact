"""Verify generated rows by hand and freeze the agent-traffic evaluation set (plan T42).

    uv run python scripts/pii_corpus/review.py review GENERATED.jsonl --reviewer NAME
    uv run python scripts/pii_corpus/review.py freeze GENERATED.verified.jsonl --out FROZEN.jsonl

``review`` shows each row of a generate.py file — its text with the spans
tagged inline and a list of the tagged values — ON THE REVIEWER'S TERMINAL
and reads one decision per row from it: accept, edit (the tagged text opens
in ``$VISUAL``/``$EDITOR``; the edit is grounded again like a teacher
answer), reject, skip or quit. Accepted and edited rows are appended to
``GENERATED.verified.jsonl`` with a review record (reviewer, decision,
guideline version, date, the generator run's teacher digest and catalog
digest); rejected ids go to ``GENERATED.verified.jsonl.rejected``. A row
already decided is not asked again, so a review can stop and resume. The
terminal is the only place the text is shown; nothing is logged.

``freeze`` validates every verified row and writes the frozen set — rows
sorted by id — with ``FROZEN.jsonl.manifest.json`` (format, SHA-256, counts
by type, prompt, teacher, seed and reviewer; no text), which the NER bench's
``agent-eval`` dataset checks before scoring
(``python -m llm_redact.bench.ner --dataset agent-eval --path FROZEN.jsonl``).

Every file is written outside every git work tree, mode 0600: the set is
private (plan D7). The labeling rules are in GUIDELINES.md.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_redact.bench.datasets.agent_eval import (  # noqa: E402
    FORMAT,
    file_sha256,
    manifest_path,
)
from llm_redact.detection.labels import TYPE_NAMES  # noqa: E402
from pii_corpus.grounding import Span, ground, tagged  # noqa: E402
from pii_corpus.private_files import (  # noqa: E402
    CorpusError,
    append_private,
    json_line,
    open_private,
    output_problem,
    read_jsonl,
    write_json,
)

# GUIDELINES.md's version; recorded in every review record and the manifest.
GUIDELINE_VERSION = 1
ROW_KEYS = ("id", "text", "spans", "teacher", "prompt_id", "seed")
DECISIONS = ("accepted", "edited")
PROMPT = "[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit > "

Ask = Callable[[str], str]
Show = Callable[[str], None]
Edit = Callable[[str], str]


def verified_path(generated: Path) -> Path:
    return generated.with_name(generated.stem + ".verified.jsonl")


def rejected_path(verified: Path) -> Path:
    return verified.with_name(verified.name + ".rejected")


def _spans(row: dict[str, Any]) -> list[Span] | None:
    """The row's spans when they are well formed, inside the text and do not
    overlap; else None."""
    text, raw = row.get("text"), row.get("spans")
    if not isinstance(text, str) or not isinstance(raw, list):
        return None
    spans = []
    for entry in raw:
        if not isinstance(entry, dict):
            return None
        start, end, type_name = entry.get("start"), entry.get("end"), entry.get("type")
        if type(start) is not int or type(end) is not int or not isinstance(type_name, str):
            return None
        if not 0 <= start < end <= len(text):
            return None
        spans.append(Span(start, end, type_name))
    spans.sort(key=lambda s: (s.start, s.end))
    if any(a.end > b.start for a, b in zip(spans, spans[1:], strict=False)):
        return None
    return spans


def _identified(row: dict[str, Any]) -> bool:
    """Whether the row carries its id and provenance fields."""
    strings = all(
        isinstance(row.get(key), str) and row[key] for key in ("id", "teacher", "prompt_id")
    )
    return strings and type(row.get("seed")) is int


def render(row: dict[str, Any], spans: Sequence[Span], index: int, total: int) -> str:
    text = row["text"]
    lines = [
        f"--- [{index}/{total}] {row['id']}  prompt {row['prompt_id']}  teacher {row['teacher']}",
        tagged(text, spans),
        "--- spans:" if spans else "--- no spans (a hard negative: is there really no PII?)",
    ]
    lines += [f"  {n}. {s.type}: {text[s.start : s.end]}" for n, s in enumerate(spans, 1)]
    return "\n".join(lines)


def editor_edit(command: str) -> Edit:
    """An Edit that opens the tagged text in ``command`` (a private temporary
    file, removed afterwards) and returns what was saved."""

    def edit(text: str) -> str:
        directory = tempfile.mkdtemp(prefix="llm-redact-review-")
        path = Path(directory) / "row.txt"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write(text)
            subprocess.run([*shlex.split(command), str(path)], check=True)
            edited = path.read_text(encoding="utf-8")
        finally:
            path.unlink(missing_ok=True)
            os.rmdir(directory)
        # Editors add a final newline the row did not have.
        return edited[:-1] if edited.endswith("\n") and not text.endswith("\n") else edited

    return edit


def _generator_source(generated: Path) -> dict[str, Any]:
    """Provenance of the generator run: its teacher digest and catalog digest."""
    source: dict[str, Any] = {"rows_file": generated.name}
    try:
        manifest = json.loads(
            generated.with_name(generated.name + ".manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return source
    teacher = manifest.get("teacher") if isinstance(manifest, dict) else None
    if isinstance(teacher, dict) and isinstance(teacher.get("digest"), str):
        source["teacher_digest"] = teacher["digest"]
    if isinstance(manifest, dict) and isinstance(manifest.get("catalog_sha256"), str):
        source["catalog_sha256"] = manifest["catalog_sha256"]
    return source


def _decided(verified: Path) -> set[str]:
    ids: set[str] = set()
    for path in (verified, rejected_path(verified)):
        if path.exists():
            ids |= {str(row.get("id")) for _, row in read_jsonl(path)}
    return ids


def _decide(
    row: dict[str, Any], spans: list[Span], index: int, total: int, ask: Ask, show: Show, edit: Edit
) -> tuple[str, dict[str, Any], list[Span]]:
    """The reviewer's decision on one row (accepted, edited, rejected,
    skipped or quit) and the row as decided."""
    edited = False
    while True:
        show(render(row, spans, index, total))
        try:
            answer = ask(PROMPT).strip().lower()
        except EOFError:
            return "quit", row, spans
        decisions = {"a": "edited" if edited else "accepted", "r": "rejected", "s": "skipped"}
        if answer in decisions or answer == "q":
            return decisions.get(answer, "quit"), row, spans
        if answer != "e":
            show("answer a, e, r, s or q")
            continue
        found = ground(edit(tagged(row["text"], spans)), types=TYPE_NAMES, negative=None)
        if isinstance(found, str):
            show(f"edit not applied: {found}")
            continue
        row = {**row, "text": found.text}
        spans = list(found.spans)
        edited = True


def review_file(
    generated: Path,
    verified: Path,
    *,
    reviewer: str,
    ask: Ask,
    show: Show,
    edit: Edit,
    today: str,
) -> Counter[str]:
    """Review every undecided row of ``generated``; the counts."""
    counts: Counter[str] = Counter()
    rows = [row for _, row in read_jsonl(generated)]
    done = _decided(verified)
    source = _generator_source(generated)
    with append_private(verified) as accepted, append_private(rejected_path(verified)) as rejected:
        for index, row in enumerate(rows, 1):
            spans = _spans(row)
            if spans is None or not _identified(row):
                counts["malformed"] += 1
                continue
            if str(row.get("id")) in done:
                counts["already decided"] += 1
                continue
            decision, row, spans = _decide(row, spans, index, len(rows), ask, show, edit)
            if decision == "quit":
                break
            counts[decision] += 1
            _record(decision, row, spans, reviewer, today, source, accepted, rejected)
    return counts


def _record(
    decision: str,
    row: dict[str, Any],
    spans: Sequence[Span],
    reviewer: str,
    today: str,
    source: dict[str, Any],
    accepted: IO[str],
    rejected: IO[str],
) -> None:
    if decision == "rejected":
        rejected.write(json_line({"id": row["id"], "reviewer": reviewer, "reviewed": today}))
        rejected.flush()
    elif decision in DECISIONS:
        record = {key: row[key] for key in ROW_KEYS if key != "spans"}
        record["spans"] = [s.as_json() for s in spans]
        record["review"] = {
            "reviewer": reviewer,
            "decision": decision,
            "guideline": GUIDELINE_VERSION,
            "reviewed": today,
            "source": source,
        }
        accepted.write(json_line(record))
        accepted.flush()


def row_problem(row: dict[str, Any]) -> str | None:
    """Why a verified row may not be frozen (None = it may). Names keys and
    types, never text."""
    for key in ROW_KEYS:
        if key not in row:
            return f"missing key {key!r}"
    if not isinstance(row["id"], str) or not row["id"]:
        return "the id is not a non-empty string"
    spans = _spans(row)
    if spans is None:
        return "spans are malformed, outside the text or overlapping"
    for span in spans:
        if span.type not in TYPE_NAMES:
            return f"span type {span.type!r} is not a placeholder type"
    review = row.get("review")
    if not isinstance(review, dict):
        return "no review record"
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"]:
        return "the review record names no reviewer"
    if review.get("decision") not in DECISIONS:
        return "the review decision is neither accepted nor edited"
    if review.get("guideline") != GUIDELINE_VERSION:
        return (
            f"reviewed under guideline version {review.get('guideline')!r}, not {GUIDELINE_VERSION}"
        )
    return None


def freeze(verified: Path, out: Path, *, overwrite: bool, today: str) -> dict[str, Any]:
    """Write the frozen set and its manifest; the manifest."""
    rows: dict[str, dict[str, Any]] = {}
    for number, row in read_jsonl(verified):
        problem = row_problem(row)
        if problem is None and row["id"] in rows:
            problem = "a duplicate id"
        if problem is not None:
            raise CorpusError(f"{verified} line {number}: {problem}")
        rows[row["id"]] = row
    if not rows:
        raise CorpusError(f"{verified} holds no verified row")
    with open_private(out, overwrite=overwrite) as handle:
        for row_id in sorted(rows):
            handle.write(json_line(rows[row_id]))
    spans: Counter[str] = Counter()
    for row in rows.values():
        spans.update(span["type"] for span in row["spans"])

    def tally(key: Callable[[dict[str, Any]], object]) -> dict[str, int]:
        return dict(sorted(Counter(str(key(row)) for row in rows.values()).items()))

    manifest = {
        "format": FORMAT,
        "rows_file": out.name,
        "sha256": file_sha256(out),
        "rows": len(rows),
        "negatives": sum(1 for row in rows.values() if not row["spans"]),
        "spans": dict(sorted(spans.items())),
        "prompt_ids": tally(lambda row: row["prompt_id"]),
        "teachers": tally(lambda row: row["teacher"]),
        "teacher_digests": tally(lambda row: row["review"].get("source", {}).get("teacher_digest")),
        "seeds": tally(lambda row: row["seed"]),
        "reviewers": tally(lambda row: row["review"]["reviewer"]),
        "decisions": tally(lambda row: row["review"]["decision"]),
        "guideline": GUIDELINE_VERSION,
        "frozen": today,
    }
    write_json(manifest_path(out), manifest, overwrite=overwrite)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/pii_corpus/review.py",
        description="Verify generated rows by hand and freeze the agent-traffic evaluation set.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="accept, edit or reject each generated row")
    review.add_argument("generated", type=Path, help="a generate.py JSONL file")
    review.add_argument("--reviewer", required=True, help="your name or handle (recorded)")
    review.add_argument("--out", type=Path, help="verified rows (default GENERATED.verified.jsonl)")
    review.add_argument("--editor", help="editor command (default $VISUAL, $EDITOR, vi)")
    frozen = commands.add_parser("freeze", help="write the frozen set and its manifest")
    frozen.add_argument("verified", type=Path, help="a review output file")
    frozen.add_argument("--out", type=Path, required=True, help="the frozen JSONL file")
    frozen.add_argument("--force", action="store_true", help="replace an existing frozen set")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    ask: Ask | None = None,
    edit: Edit | None = None,
    today: str | None = None,
) -> int:
    args = _parser().parse_args(argv)
    today = today or datetime.now(UTC).date().isoformat()
    try:
        if args.command == "freeze":
            manifest = freeze(args.verified, args.out, overwrite=args.force, today=today)
            print(
                f"{args.out}: {manifest['rows']} rows ({manifest['negatives']} hard negatives),"
                f" spans {manifest['spans']}, sha256 {manifest['sha256']}"
            )
            return 0
        if ask is None:
            if not sys.stdin.isatty():
                print(
                    "error: review reads each decision from a terminal (verification is by hand)",
                    file=sys.stderr,
                )
                return 2
            ask = input
        verified = args.out or verified_path(args.generated)
        problem = output_problem(verified)
        if problem is not None:
            raise CorpusError(problem)
        command = args.editor or os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
        counts = review_file(
            args.generated,
            verified,
            reviewer=args.reviewer,
            ask=ask,
            show=print,
            edit=edit or editor_edit(command),
            today=today,
        )
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = ", ".join(f"{key} {n}" for key, n in sorted(counts.items())) or "nothing to review"
    print(f"{verified}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
