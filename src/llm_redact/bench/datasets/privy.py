"""beki/privy for the NER bench: PII inside JSON, SQL, HTML and XML payloads,
the closest public analogue of tool-call bodies.

Card checked 2026-10-05 at revision dc137a6a976f6b5bb8768e9bb51ec58df930ccd1
(license MIT; "Privy Synthetic PII Protocol Trace Dataset", Benjamin
Kilimnik, 2022): synthetic protocol traces generated from OpenAPI
specifications. The data is one archive, ``privy-dataset.zip`` (about 300 MB),
holding ``{train,dev,test}-{small,large}.json``, each ONE JSON array of rows
with ``full_text`` and ``spans`` (``entity_type``, ``entity_value``,
``start_position``, ``end_position``). The ``small`` files label people
``PER``/``LOC``/``ORG``, the ``large`` ones ``PERSON``/``LOCATION``/
``ORGANIZATION``; both mark non-PII payload values with the label ``O``,
which the bench treats as NOT personal data. The repository's
``privy.py`` loading script is never run: the archive is read directly, a
row at a time.
"""

import io
import json
import zipfile
from collections.abc import Iterable, Iterator
from types import MappingProxyType
from typing import IO

from llm_redact.bench.datasets.base import (
    MALFORMED,
    DatasetError,
    DatasetSpec,
    LoadRequest,
    checked_spans,
    fetch,
)
from llm_redact.bench.ner_metrics import LEAK, NOT_PII, NerSample

REVISION = "dc137a6a976f6b5bb8768e9bb51ec58df930ccd1"
ARCHIVE = "privy-dataset.zip"
MEMBERS = MappingProxyType(
    {
        "test": "test-small.json",
        "dev": "dev-small.json",
        "train": "train-small.json",
        "test-large": "test-large.json",
        "dev-large": "dev-large.json",
        "train-large": "train-large.json",
    }
)
READ_CHARS = 1 << 20

_SCORED = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "US_PASSPORT": "PASSPORT",
    "US_DRIVER_LICENSE": "DRIVER_LICENSE",
    "US_BANK_NUMBER": "ACCOUNT_NUMBER",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
    "IBAN_CODE": "IBAN",
    "CREDIT_CARD": "CREDIT_CARD",
    "PASSWORD": "SECRET",
    # No placeholder type: an ITIN is a tax number, an IP_ADDRESS may be
    # either version (the fold table keeps it unfolded for that reason).
    "US_ITIN": LEAK,
    "IP_ADDRESS": LEAK,
    # Payload values the dataset marks as not personal data.
    "O": NOT_PII,
}
_UNSCORED = (
    "LOC",
    "LOCATION",
    "ORG",
    "ORGANIZATION",
    "NRP",
    "DATE_TIME",
    "URL",
    "TITLE",
    "COORDINATE",
    "IMEI",
    "LICENSE_PLATE",
    "US_LICENSE_PLATE",
    "CURRENCY",
    "FINANCIAL",
    "ROUTING_NUMBER",
    "SWIFT_CODE",
    "MAC_ADDRESS",
    "AGE",
)
LABELS = MappingProxyType({**_SCORED, **dict.fromkeys(_UNSCORED)})

_SQL_VERBS = ("SELECT", "INSERT", "UPDATE", "DELETE")


def iter_json_array(stream: IO[str], read_chars: int = READ_CHARS) -> Iterator[object]:
    """The elements of the JSON array ``stream`` holds, one at a time,
    without reading the whole array into memory."""
    decoder = json.JSONDecoder()
    buffer, eof, pos = "", False, 0

    def more(size: int) -> bool:
        nonlocal buffer, eof
        chunk = stream.read(size)
        eof = not chunk
        buffer += chunk
        return not eof

    while not eof and not buffer.lstrip():
        more(read_chars)
    buffer = buffer.lstrip()
    if not buffer.startswith("["):
        raise DatasetError("privy: a data file does not hold a JSON array")
    pos = 1
    size = read_chars
    while True:
        while pos < len(buffer) and buffer[pos] in " \t\r\n,":
            pos += 1
        if pos == len(buffer):
            if not more(read_chars):
                raise DatasetError("privy: a data file ends inside its JSON array")
            continue
        if buffer[pos] == "]":
            return
        try:
            value, end = decoder.raw_decode(buffer, pos)
        except json.JSONDecodeError:
            end = len(buffer) + 1  # incomplete: read more below
            value = None
        if end > len(buffer) or (end == len(buffer) and not eof):
            if not more(size):
                raise DatasetError("privy: a data file ends inside an element")
            size *= 2  # a long element costs O(n log n), never O(n^2)
            continue
        size = read_chars
        buffer, pos = buffer[end:], 0
        yield value


def payload_kind(text: str) -> str:
    """The payload format a row's text is in (its context in reports)."""
    head = text.lstrip()
    if head.startswith(("{", "[")):
        return "json"
    if head.startswith(("b'<", 'b"<', "<?xml")):
        return "xml"
    if head.startswith("<"):
        return "html"
    if head[:6].upper() in _SQL_VERBS:
        return "sql"
    return "other"


def samples(rows: Iterable[object], request: LoadRequest) -> Iterator[NerSample]:
    for row in rows:
        if not isinstance(row, dict):
            request.skipped[MALFORMED] += 1
            continue
        text, spans = row.get("full_text"), row.get("spans")
        if not isinstance(text, str) or not isinstance(spans, list):
            request.skipped[MALFORMED] += 1
            continue
        if not all(isinstance(span, dict) for span in spans):
            request.skipped[MALFORMED] += 1
            continue
        entries = (
            (
                s.get("start_position"),
                s.get("end_position"),
                s.get("entity_type"),
                s.get("entity_value"),
            )
            for s in spans
        )
        gold = checked_spans(text, entries, request)
        if gold is not None:
            yield NerSample(text, gold, payload_kind(text))


def adapter(spec: DatasetSpec, request: LoadRequest) -> Iterator[NerSample]:
    (archive,) = fetch(spec, request.split, request)
    try:
        with zipfile.ZipFile(archive) as zipped, zipped.open(MEMBERS[request.split]) as raw:
            yield from samples(iter_json_array(io.TextIOWrapper(raw, encoding="utf-8")), request)
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as exc:
        raise DatasetError(f"privy: cannot read {ARCHIVE}: {type(exc).__name__}") from exc


SPEC = DatasetSpec(
    name="privy",
    summary="beki/privy: synthetic PII inside JSON, SQL, HTML and XML payloads",
    license="MIT",
    attribution=(
        "Privy Synthetic PII Protocol Trace Dataset by Benjamin Kilimnik,"
        " https://huggingface.co/datasets/beki/privy"
    ),
    real_data=False,
    label_map=LABELS,
    splits=tuple(MEMBERS),
    adapter=adapter,
    notes=(
        "Payload values the dataset labels O (not personal data) count as over-redaction"
        " when detected.",
    ),
    hub_id="beki/privy",
    revision=REVISION,
    files=MappingProxyType({split: (ARCHIVE,) for split in MEMBERS}),
    card="https://huggingface.co/datasets/beki/privy",
    checked="2026-10-05",
)
