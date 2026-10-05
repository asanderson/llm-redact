"""PUPA (``Columbia-NLP/PUPA``) for the NER bench: 901 REAL user prompts.

Card checked 2026-10-05 at revision 9981b49b6ced0033988a224b6712895ebf119294
(license MIT; data of the paper "PAPILLON: Privacy Preservation from
Internet-based and Local Language Model Ensembles", Li et al., 2024,
https://huggingface.co/papers/2410.17127): two CSV files, ``PUPA_TNB.csv``
(237 rows) and ``PUPA_New.csv`` (664 rows), with columns
``conversation_hash``, ``predicted_category``, ``user_query``,
``target_response``, ``pii_units`` and ``redacted_query``. The prompts are
real WildChat user queries; ``pii_units`` holds the personal-data strings
an LLM extracted from each one — lowercased, separated by ``||``, with NO
types and NO offsets.

The adapter therefore finds every whole-word, case-insensitive occurrence
of each unit in its prompt and scores it with the type-agnostic leak metric
only; a row with a unit that does not occur in its prompt is skipped and
counted. The dataset holds real data: ``--dump-errors`` needs
``--allow-real-data-dump``.
"""

import csv
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

from llm_redact.bench.datasets.base import (
    MALFORMED,
    DatasetError,
    DatasetSpec,
    LoadRequest,
    fetch,
)
from llm_redact.bench.ner_metrics import LEAK, GoldSpan, NerSample

REVISION = "9981b49b6ced0033988a224b6712895ebf119294"
UNIT_LABEL = "PII_UNIT"
UNIT_NOT_FOUND = "a PII unit does not occur in its prompt"
FILES = MappingProxyType(
    {
        "all": ("PUPA_TNB.csv", "PUPA_New.csv"),
        "tnb": ("PUPA_TNB.csv",),
        "new": ("PUPA_New.csv",),
    }
)


def csv_rows(path: Path) -> Iterator[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
    except (csv.Error, UnicodeDecodeError) as exc:
        raise DatasetError(f"PUPA: cannot read {path.name}: {type(exc).__name__}") from exc


def _word_char(text: str, index: int) -> bool:
    return 0 <= index < len(text) and text[index].isalnum()


def unit_spans(prompt: str, unit: str) -> list[tuple[int, int]]:
    """Every whole-word, case-insensitive occurrence of ``unit`` in
    ``prompt`` (empty when lowercasing changes the prompt's length, so
    offsets could not be trusted)."""
    lowered, needle = prompt.lower(), unit.lower()
    if len(lowered) != len(prompt) or not needle:
        return []
    found = []
    start = lowered.find(needle)
    while start != -1:
        end = start + len(needle)
        if not _word_char(lowered, start - 1) and not _word_char(lowered, end):
            found.append((start, end))
        start = lowered.find(needle, start + 1)
    return found


def samples(rows: Iterable[Mapping[str, str | None]], request: LoadRequest) -> Iterator[NerSample]:
    for row in rows:
        prompt, units = row.get("user_query"), row.get("pii_units")
        if not isinstance(prompt, str) or not isinstance(units, str):
            request.skipped[MALFORMED] += 1
            continue
        spans: list[GoldSpan] = []
        for unit in dict.fromkeys(u.strip() for u in units.split("||") if u.strip()):
            found = unit_spans(prompt, unit)
            if not found:
                break
            spans.extend(GoldSpan(start, end, UNIT_LABEL) for start, end in found)
        else:
            yield NerSample(prompt, tuple(spans), row.get("predicted_category") or "")
            continue
        request.skipped[UNIT_NOT_FOUND] += 1


def adapter(spec: DatasetSpec, request: LoadRequest) -> Iterator[NerSample]:
    for path in fetch(spec, request.split, request):
        yield from samples(csv_rows(path), request)


SPEC = DatasetSpec(
    name="pupa",
    summary="PUPA: 901 real WildChat user prompts with LLM-extracted PII units",
    license="MIT",
    attribution=(
        "PUPA (PAPILLON, Li et al. 2024) by Columbia NLP,"
        " https://huggingface.co/datasets/Columbia-NLP/PUPA"
    ),
    real_data=True,
    label_map=MappingProxyType({UNIT_LABEL: LEAK}),
    splits=tuple(FILES),
    adapter=adapter,
    notes=(
        "Real prompts. The PII units were extracted by an LLM, not annotated by people,"
        " and carry no types: only the character-leak and over-redaction rates mean"
        " anything here.",
    ),
    hub_id="Columbia-NLP/PUPA",
    revision=REVISION,
    files=FILES,
    card="https://huggingface.co/datasets/Columbia-NLP/PUPA",
    checked="2026-10-05",
)
