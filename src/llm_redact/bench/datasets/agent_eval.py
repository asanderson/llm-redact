"""The hand-verified agent-traffic evaluation set (``--dataset agent-eval --path FILE``).

The set is PRIVATE (plan decision D7): it is never committed, never
downloaded, and this module holds no row of it. It is built out of band by
the dev-only tooling in ``scripts/pii_corpus/`` — a local Apache-2.0 teacher
generates tagged coding-agent artifacts (``generate.py``), a human accepts,
edits or rejects every row (``review.py review``, following
``scripts/pii_corpus/GUIDELINES.md``) and ``review.py freeze`` writes the
frozen JSONL with a manifest beside it (``FILE.manifest.json``). The bench
reads it by path.

Frozen rows: ``{"id", "text", "spans": [{"start", "end", "type"}],
"teacher", "prompt_id", "seed", "review": {...}}``, where every span type is a
placeholder type. The manifest (format :data:`FORMAT`) records the file's
SHA-256 and row count; a file that no longer matches it is refused (a
frozen set changes only by freezing again). The text is LLM-generated and
hand-verified, not real people's data, but the set is private and is
marked as real data, so ``--dump-errors`` needs ``--allow-real-data-dump``.
"""

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType

from llm_redact.bench.datasets.base import (
    MALFORMED,
    DatasetError,
    DatasetSpec,
    LoadRequest,
    checked_spans,
    jsonl_rows,
)
from llm_redact.bench.ner_metrics import NerSample
from llm_redact.detection.labels import TYPE_NAMES

FORMAT = "llm-redact-agent-eval/1"
# A prompt id is a catalog name (scripts/pii_corpus/prompts.py); anything
# else is reported as "other", so no row text can reach a report.
_PROMPT_ID_RE = re.compile(r"[a-z0-9-]{1,40}(?:\.negative)?")


def manifest_path(path: Path) -> Path:
    """Where a frozen set's manifest lives: beside it, ``FILE.manifest.json``."""
    return path.with_name(path.name + ".manifest.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_manifest(path: Path) -> None:
    where = manifest_path(path)
    try:
        manifest = json.loads(where.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError(
            f"agent-eval: cannot read {where.name} ({type(exc).__name__}); a frozen set is"
            " written with its manifest by scripts/pii_corpus/review.py freeze"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise DatasetError(f"agent-eval: {where.name} is not a {FORMAT} manifest")
    if manifest.get("sha256") != file_sha256(path):
        raise DatasetError(
            f"agent-eval: {path.name} does not match the SHA-256 in {where.name}; a frozen"
            " set changes only by freezing it again"
        )


def context(prompt_id: object) -> str:
    """The report context of a row: its catalog prompt id, a hard negative's
    as ``<artifact>-negative``."""
    if not isinstance(prompt_id, str) or not _PROMPT_ID_RE.fullmatch(prompt_id):
        return "other"
    artifact, _, negative = prompt_id.partition(".")
    return f"{artifact}-negative" if negative else artifact


def _spans(row: dict[str, object]) -> list[tuple[object, object, object, object]] | None:
    spans = row.get("spans")
    if not isinstance(spans, list) or not all(isinstance(s, dict) for s in spans):
        return None
    return [(s.get("start"), s.get("end"), s.get("type"), None) for s in spans]


def adapter(spec: DatasetSpec, request: LoadRequest) -> Iterator[NerSample]:
    path = request.path
    if path is None or not path.is_file():
        raise DatasetError(
            "agent-eval: --path must name the frozen set (a file written by"
            " scripts/pii_corpus/review.py freeze; docs/ner-bench.md, 'The agent-traffic"
            " evaluation set')"
        )
    _check_manifest(path)
    for row in jsonl_rows(path, request):
        text = row.get("text") if isinstance(row, dict) else None
        entries = _spans(row) if isinstance(row, dict) else None
        if not isinstance(row, dict) or not isinstance(text, str) or entries is None:
            request.skipped[MALFORMED] += 1
            continue
        gold = checked_spans(text, entries, request)
        if gold is not None:
            yield NerSample(text, gold, context(row.get("prompt_id")))


SPEC = DatasetSpec(
    name="agent-eval",
    summary=(
        "the private, hand-verified agent-traffic evaluation set: coding-agent artifacts"
        " (tool-call JSON, diffs, logs, configs, commit messages) read from --path"
    ),
    license="private, not distributed (rows written by an Apache-2.0 teacher, verified by hand)",
    attribution="llm-redact maintainers (scripts/pii_corpus)",
    real_data=True,
    label_map=MappingProxyType({type_name: type_name for type_name in TYPE_NAMES}),
    splits=("all",),
    adapter=adapter,
    notes=(
        "LLM-generated with invented values, then verified by hand"
        " (scripts/pii_corpus/GUIDELINES.md); hard negatives carry no gold. The set is"
        " private and never committed; its text never appears in a report.",
    ),
    needs_path=True,
)
