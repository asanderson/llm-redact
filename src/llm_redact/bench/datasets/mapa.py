"""MAPA (``joelniklaus/mapa``) for the NER bench: human-annotated EU legal
text in 21 languages.

Card checked 2026-10-05 at revision bbb2a0157b760465002fd12a61af81b475cd387a
(license CC BY 4.0; de Gibert Bonet et al., "Spanish Datasets for Sensitive
Entity Detection in the Legal Domain", LREC 2022; converted by Joel Niklaus
and Veton Matoshi): ``train.jsonl``, ``validation.jsonl`` and ``test.jsonl``,
one sentence per line with ``language``, ``type``, ``file_name``,
``sentence_number``, ``tokens``, ``coarse_grained`` and ``fine_grained``
(IOB tags, one per token). Twelve EUR-Lex documents (nine for Spanish),
annotated by one annotator following the MAPA guidelines.

The rows carry tokens, not text, and no whitespace information: the text
is the tokens joined with single spaces (``"Article 3 ( 1 )"``), and gold
spans are the token runs of the FINE-grained tags. The coarse ``PERSON``
tag also covers roles, professions and nationalities, so only the fine
``FAMILY NAME`` and ``INITIAL NAME`` tags are scored (as ``PERSON``); the
data carries no given-name, street, email or identifier tags at this
revision. EUR-Lex decisions name real parties, so the dataset is marked as
real data.
"""

from collections.abc import Iterable, Iterator
from types import MappingProxyType

from llm_redact.bench.datasets.base import (
    MALFORMED,
    DatasetSpec,
    LoadRequest,
    fetch,
    jsonl_rows,
)
from llm_redact.bench.ner_metrics import GoldSpan, NerSample

REVISION = "bbb2a0157b760465002fd12a61af81b475cd387a"

# Every fine-grained label in the data at this revision.
LABELS = MappingProxyType(
    {
        "FAMILY NAME": "PERSON",
        "INITIAL NAME": "PERSON",
        **dict.fromkeys(
            (
                "AGE",
                "BUILDING",
                "CITY",
                "COUNTRY",
                "DAY",
                "ETHNIC CATEGORY",
                "MARITAL STATUS",
                "MONTH",
                "NATIONALITY",
                "PLACE",
                "PROFESSION",
                "ROLE",
                "STANDARD ABBREVIATION",
                "TERRITORY",
                "TITLE",
                "TYPE",
                "UNIT",
                "URL",
                "VALUE",
                "YEAR",
            )
        ),
    }
)


def iob_spans(tokens: list[str], tags: list[str]) -> tuple[str, list[GoldSpan]]:
    """The text (tokens joined with single spaces) and the gold spans of the
    IOB tags: a ``B-X`` starts a span, an ``I-X`` continues one of the same
    label (or starts one, when it does not follow it)."""
    text_parts: list[str] = []
    spans: list[GoldSpan] = []
    offset = 0
    current: tuple[int, int, str] | None = None
    for token, tag in zip(tokens, tags, strict=True):
        start = offset
        end = start + len(token)
        prefix, _, label = tag.partition("-")
        continues = prefix == "I" and current is not None and current[2] == label
        if current is not None and not continues:
            spans.append(GoldSpan(*current))
            current = None
        if continues and current is not None:
            current = (current[0], end, label)
        elif prefix in ("B", "I") and label:
            current = (start, end, label)
        text_parts.append(token)
        offset = end + 1
    if current is not None:
        spans.append(GoldSpan(*current))
    return " ".join(text_parts), spans


def samples(rows: Iterable[object], request: LoadRequest) -> Iterator[NerSample]:
    for row in rows:
        if not isinstance(row, dict):
            request.skipped[MALFORMED] += 1
            continue
        language = row.get("language")
        if request.language is not None and language != request.language:
            continue
        tokens, tags = row.get("tokens"), row.get("fine_grained")
        if (
            not isinstance(tokens, list)
            or not isinstance(tags, list)
            or len(tokens) != len(tags)
            or not all(isinstance(t, str) for t in (*tokens, *tags))
        ):
            request.skipped[MALFORMED] += 1
            continue
        text, spans = iob_spans(tokens, tags)
        yield NerSample(text, tuple(spans), language if isinstance(language, str) else "")


def adapter(spec: DatasetSpec, request: LoadRequest) -> Iterator[NerSample]:
    for path in fetch(spec, request.split, request):
        yield from samples(jsonl_rows(path, request), request)


SPEC = DatasetSpec(
    name="mapa",
    summary="MAPA: human-annotated EUR-Lex legal text in 21 languages",
    license="CC BY 4.0",
    attribution=(
        "MAPA EUR-Lex anonymisation data by de Gibert Bonet et al. (LREC 2022), converted"
        " by Joel Niklaus and Veton Matoshi, https://huggingface.co/datasets/joelniklaus/mapa"
    ),
    real_data=True,
    label_map=LABELS,
    splits=("test", "validation", "train"),
    adapter=adapter,
    notes=(
        "Real legal documents. Text is rebuilt by joining tokens with single spaces; only"
        " family names and initials are scored (as PERSON).",
    ),
    hub_id="joelniklaus/mapa",
    revision=REVISION,
    files=MappingProxyType(
        {
            "test": ("test.jsonl",),
            "validation": ("validation.jsonl",),
            "train": ("train.jsonl",),
        }
    ),
    card="https://huggingface.co/datasets/joelniklaus/mapa",
    checked="2026-10-05",
    filters_language=True,
)
