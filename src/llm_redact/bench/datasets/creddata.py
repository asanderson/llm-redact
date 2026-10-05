"""CredData (Samsung) for the NER bench: labelled secrets in REAL code.

Checked 2026-10-05 at https://github.com/Samsung/CredData commit
0b1940e171725ad8937311120b191602608a4801: the repository ships only
metadata — ``meta/<repo>.csv`` with the columns ``Id, FileID, Domain,
RepoName, FilePath, LineStart, LineEnd, GroundTruth, ValueStart, ValueEnd,
CryptographyKey, PredefinedPattern, Category`` — and its own script,
``download_data.py``, which fetches the source repositories and writes the
labelled files to ``data/<repo>/(src|test|other)/<file>`` with the
credential values obfuscated. The labels are Apache-2.0; each code file
keeps its own project's license (the ``license`` directory of a checkout).

This adapter never downloads anything: it reads a checkout the user
prepared (``--data-dir``, the CredData directory after ``download_data.py``
ran). Each distinct (file, LineStart, LineEnd) of the metadata becomes one
sample — those lines of code, joined by newlines — and every ``T``
(true credential) row's value span is gold, scored by the leak metric
only; ``F``/``X`` rows mark lines whose look-alikes are not credentials, so
a detection there is over-redaction. The code is real: the dataset is
marked as real data, and nothing from it is ever vendored.
"""

import csv
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType

from llm_redact.bench.datasets.base import DatasetError, DatasetSpec, LoadRequest
from llm_redact.bench.ner_metrics import LEAK, GoldSpan, NerSample

COMMIT = "0b1940e171725ad8937311120b191602608a4801"
TRUE_LABEL = "T"
MISSING_FILE = "a file the metadata names is missing from the checkout"
BAD_LINES = "a metadata row's lines or value offsets fall outside its file"
NO_VALUE = "a true credential row has no value offsets"
SCOPES = ("src", "test", "other")


def _int(raw: str | None) -> int:
    """A metadata integer; an empty cell is -1, as the metadata spells it."""
    return int(raw) if raw else -1


def _lines(path: Path) -> list[str] | None:
    """A file's lines (UTF-8, else Latin-1: one character per byte), with
    each line's trailing carriage return dropped; None when it is missing."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return [line.removesuffix("\r") for line in text.split("\n")]


def _scope(file_path: str) -> str:
    parts = Path(file_path).parts
    return parts[2] if len(parts) > 3 else "other"


def _groups(meta: Path) -> dict[tuple[str, int, int], list[dict[str, str]]]:
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    try:
        with meta.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["FilePath"], _int(row["LineStart"]), _int(row["LineEnd"]))
                groups.setdefault(key, []).append(row)
    except (csv.Error, KeyError, ValueError, UnicodeDecodeError) as exc:
        raise DatasetError(f"CredData: cannot read meta/{meta.name}: {type(exc).__name__}") from exc
    return groups


def _sample(
    lines: list[str], line_start: int, line_end: int, rows: list[dict[str, str]]
) -> NerSample | str:
    """One sample, or the reason the group is skipped."""
    if not 1 <= line_start <= line_end <= len(lines):
        return BAD_LINES
    chosen = lines[line_start - 1 : line_end]
    text = "\n".join(chosen)
    last_line_offset = len(text) - len(chosen[-1])
    spans = []
    for row in rows:
        if row.get("GroundTruth") != TRUE_LABEL:
            continue
        value_start, value_end = _int(row.get("ValueStart")), _int(row.get("ValueEnd"))
        if value_start < 0:
            return NO_VALUE
        end = last_line_offset + (value_end if value_end >= 0 else len(chosen[-1]))
        if not value_start < end <= len(text) or value_start > len(chosen[0]):
            return BAD_LINES
        spans.append(GoldSpan(value_start, end, TRUE_LABEL))
    suffix = Path(rows[0]["FilePath"]).suffix.lstrip(".") or "none"
    return NerSample(text, tuple(spans), suffix)


def adapter(spec: DatasetSpec, request: LoadRequest) -> Iterator[NerSample]:
    root = request.data_dir
    if root is None or not (root / "meta").is_dir():
        raise DatasetError(
            "CredData: --data-dir must name a CredData checkout with a meta/ directory"
            " (docs/ner-bench.md, 'CredData')"
        )
    for meta in sorted((root / "meta").glob("*.csv")):
        files: dict[str, list[str] | None] = {}
        for (file_path, line_start, line_end), rows in _groups(meta).items():
            if request.split != "all" and _scope(file_path) != request.split:
                continue
            if file_path not in files:
                files[file_path] = _lines(root / file_path)
            lines = files[file_path]
            if lines is None:
                request.skipped[MISSING_FILE] += 1
                continue
            sample = _sample(lines, line_start, line_end, rows)
            if isinstance(sample, str):
                request.skipped[sample] += 1
            else:
                yield sample


SPEC = DatasetSpec(
    name="creddata",
    summary="CredData: labelled secrets (obfuscated) in real code and config lines",
    license=(
        "labels Apache-2.0; each code file keeps its project's license (see the checkout's"
        " license directory)"
    ),
    attribution="CredData by Samsung, https://github.com/Samsung/CredData",
    real_data=True,
    label_map=MappingProxyType({TRUE_LABEL: LEAK}),
    splits=("all", *SCOPES),
    adapter=adapter,
    notes=(
        "Real code from a local checkout (--data-dir). True credentials are scored by the"
        " leak metric only; lines labelled not-a-credential count as over-redaction when"
        " detected.",
    ),
    card="https://github.com/Samsung/CredData",
    checked="2026-10-05",
    needs_data_dir=True,
)
