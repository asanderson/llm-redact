"""OpenPII 1.5M (``ai4privacy/pii-masking-openpii-1.5m``) for the NER bench.

Card checked 2026-10-05 at revision a785eb528e28be2693c3718a27e066970de5dadb:
JSONL files ``data/train.jsonl`` and ``data/validation.jsonl``; each row has
``source_text``, ``privacy_mask`` (``value``, ``start``, ``end``, ``label``),
``language`` (ISO 639-1) and ``region``; 19 labels; 30 languages; synthetic
PII only. License CC BY 4.0 (the card's metadata says ``license: other``
with ``license_name: cc-by-4.0``; its text grants CC BY 4.0), credit
"Ai4Privacy / Ai Suisse SA". The bench uses it for evaluation only.
"""

from collections.abc import Iterable, Iterator
from types import MappingProxyType

from llm_redact.bench.datasets.base import (
    MALFORMED,
    DatasetSpec,
    LoadRequest,
    checked_spans,
    fetch,
    jsonl_rows,
)
from llm_redact.bench.ner_metrics import LEAK, NerSample

REVISION = "a785eb528e28be2693c3718a27e066970de5dadb"

LABELS = MappingProxyType(
    {
        "GIVENNAME": "PERSON",
        "SURNAME": "PERSON",
        "STREET": "ADDRESS",
        "BUILDINGNUM": "ADDRESS",
        "PASSPORTNUM": "PASSPORT",
        "DRIVERLICENSENUM": "DRIVER_LICENSE",
        "EMAIL": "EMAIL",
        "TELEPHONENUM": "PHONE",
        "CREDITCARDNUMBER": "CREDIT_CARD",
        # Country-specific identifiers: no placeholder type, but they must
        # not leak.
        "SOCIALNUM": LEAK,
        "TAXNUM": LEAK,
        "IDCARDNUM": LEAK,
        "DATE": None,
        "AGE": None,
        "TITLE": None,
        "GENDER": None,
        "SEX": None,
        "CITY": None,
        "ZIPCODE": None,
    }
)


def samples(rows: Iterable[object], request: LoadRequest) -> Iterator[NerSample]:
    """OpenPII rows -> samples (context: the row's language), filtered by
    ``request.language``."""
    for row in rows:
        if not isinstance(row, dict):
            request.skipped[MALFORMED] += 1
            continue
        text, mask, language = row.get("source_text"), row.get("privacy_mask"), row.get("language")
        if request.language is not None and language != request.language:
            continue
        if not isinstance(text, str) or not isinstance(mask, list):
            request.skipped[MALFORMED] += 1
            continue
        if not all(isinstance(entry, dict) for entry in mask):
            request.skipped[MALFORMED] += 1
            continue
        entries = ((e.get("start"), e.get("end"), e.get("label"), e.get("value")) for e in mask)
        spans = checked_spans(text, entries, request)
        if spans is not None:
            yield NerSample(text, spans, language if isinstance(language, str) else "")


def adapter(spec: DatasetSpec, request: LoadRequest) -> Iterator[NerSample]:
    for path in fetch(spec, request.split, request):
        yield from samples(jsonl_rows(path, request), request)


SPEC = DatasetSpec(
    name="openpii",
    summary="OpenPII 1.5M: synthetic multilingual PII text, 30 languages, 19 labels",
    license="CC BY 4.0",
    attribution=(
        "OpenPII 1.5M by Ai4Privacy / Ai Suisse SA,"
        " https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m"
    ),
    real_data=False,
    label_map=LABELS,
    splits=("validation", "train"),
    adapter=adapter,
    notes=(
        "Used for evaluation only. Names are labelled in parts (GIVENNAME, SURNAME);"
        " adjacent parts are scored as one PERSON span.",
    ),
    hub_id="ai4privacy/pii-masking-openpii-1.5m",
    revision=REVISION,
    files=MappingProxyType(
        {"validation": ("data/validation.jsonl",), "train": ("data/train.jsonl",)}
    ),
    card="https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m",
    checked="2026-10-05",
    filters_language=True,
)
