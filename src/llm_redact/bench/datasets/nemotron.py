"""Nemotron-PII (``nvidia/Nemotron-PII``) for the NER bench.

Card checked 2026-10-05 at revision b70ffaf5ff39e079776134c5bf4381f00a9fd1ed:
parquet files ``data/train-00000-of-00001.parquet`` and
``data/test-00000-of-00001.parquet`` (50,000 English records each, US and
international locales, structured and unstructured documents), columns
``uid``, ``domain``, ``document_type``, ``document_description``,
``document_format``, ``locale``, ``text``, ``spans``, ``text_tagged``. The
``spans`` column is a STRING holding a Python-literal list of
``{'start', 'end', 'text', 'label'}`` dicts (single-quoted, so not JSON),
read here with :func:`ast.literal_eval`, which evaluates literals only.
License CC BY 4.0, by NVIDIA Corporation; synthetic data. About 7% of rows
have a span whose ``text`` differs from the text at its offsets (seen in a
1,900-row slice): those rows are skipped and counted.
"""

import ast
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from llm_redact.bench.datasets.base import (
    INSTALL_HINT,
    MALFORMED,
    DatasetError,
    DatasetSpec,
    LoadRequest,
    checked_spans,
    fetch,
)
from llm_redact.bench.ner_metrics import NerSample

REVISION = "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed"
COLUMNS = ("text", "spans", "document_format", "locale")
BATCH_ROWS = 256

_SCORED = {
    "first_name": "PERSON",
    "last_name": "PERSON",
    "street_address": "ADDRESS",
    "date_of_birth": "DATE_OF_BIRTH",
    "user_name": "USERNAME",
    "account_number": "ACCOUNT_NUMBER",
    "email": "EMAIL",
    "phone_number": "PHONE",
    "ssn": "SSN",
    "credit_debit_card": "CREDIT_CARD",
    "ipv4": "IPV4",
    "ipv6": "IPV6",
    "password": "SECRET",
    "api_key": "SECRET",
}
# Every other label seen in the dataset: not scored. Listed so a label the
# dataset adds later shows up as unmapped in the report.
_UNSCORED = (
    "age",
    "bank_routing_number",
    "biometric_identifier",
    "blood_type",
    "certificate_license_number",
    "city",
    "company_name",
    "coordinate",
    "country",
    "county",
    "customer_id",
    "cvv",
    "date",
    "date_time",
    "device_identifier",
    "education_level",
    "employee_id",
    "employment_status",
    "fax_number",
    "gender",
    "health_plan_beneficiary_number",
    "http_cookie",
    "language",
    "license_plate",
    "mac_address",
    "medical_record_number",
    "occupation",
    "pin",
    "political_view",
    "postcode",
    "race_ethnicity",
    "religious_belief",
    "sexuality",
    "state",
    "swift_bic",
    "tax_id",
    "time",
    "unique_id",
    "url",
    "vehicle_identifier",
)
LABELS = MappingProxyType({**_SCORED, **dict.fromkeys(_UNSCORED)})


def _parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetError(f"reading a parquet dataset needs pyarrow; {INSTALL_HINT}") from exc
    return pq


def parquet_rows(path: Path, columns: Sequence[str] = COLUMNS) -> Iterator[dict[str, Any]]:
    """The rows of a parquet file, a batch at a time (never the whole file)."""
    parquet_file = _parquet().ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=BATCH_ROWS, columns=list(columns)):
        yield from batch.to_pylist()


def parse_spans(raw: object) -> list[object] | None:
    """The spans of a row: a list as is, or a string holding a JSON or a
    Python-literal list; None when it is neither."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    return parsed if isinstance(parsed, list) else None


def samples(rows: Iterable[dict[str, Any]], request: LoadRequest) -> Iterator[NerSample]:
    """Nemotron rows -> samples (context: document format and locale)."""
    for row in rows:
        text, spans = row.get("text"), parse_spans(row.get("spans"))
        if not isinstance(text, str) or spans is None:
            request.skipped[MALFORMED] += 1
            continue
        if not all(isinstance(span, dict) for span in spans):
            request.skipped[MALFORMED] += 1
            continue
        entries = (
            (s.get("start"), s.get("end"), s.get("label"), s.get("text"))
            for s in spans
            if isinstance(s, dict)
        )
        gold = checked_spans(text, entries, request)
        if gold is not None:
            context = f"{row.get('document_format') or '-'}/{row.get('locale') or '-'}"
            yield NerSample(text, gold, context)


def adapter(spec: DatasetSpec, request: LoadRequest) -> Iterator[NerSample]:
    for path in fetch(spec, request.split, request):
        yield from samples(parquet_rows(path), request)


SPEC = DatasetSpec(
    name="nemotron",
    summary="Nemotron-PII: synthetic English documents (forms, emails, notes), 55+ labels",
    license="CC BY 4.0",
    attribution="Nemotron-PII by NVIDIA Corporation, https://huggingface.co/datasets/nvidia/Nemotron-PII",
    real_data=False,
    label_map=LABELS,
    splits=("test", "train"),
    adapter=adapter,
    notes=(
        "Names are labelled in parts (first_name, last_name); adjacent parts are scored"
        " as one PERSON span. Rows whose span text differs from the text at its offsets"
        " are skipped.",
    ),
    hub_id="nvidia/Nemotron-PII",
    revision=REVISION,
    files=MappingProxyType(
        {
            "test": ("data/test-00000-of-00001.parquet",),
            "train": ("data/train-00000-of-00001.parquet",),
        }
    ),
    card="https://huggingface.co/datasets/nvidia/Nemotron-PII",
    checked="2026-10-05",
)
