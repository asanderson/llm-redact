"""The dataset framework of the NER bench: what a dataset is, how a run asks
for its rows, and how published datasets are fetched.

Published datasets are downloaded at run time from the Hugging Face Hub at
a PINNED revision (a commit hash) with ``huggingface_hub.hf_hub_download``
into a cache outside the repository (default
``${XDG_CACHE_HOME:-~/.cache}/llm-redact/bench-datasets``) and are never
committed. They are used for EVALUATION only. Rows whose gold spans do not
match their text are skipped and counted, never repaired.
"""

import json
import os
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from llm_redact.bench.ner_metrics import GoldSpan, LabelMap, NerSample

INSTALL_HINT = "install the bench-data extra: uv sync --extra bench-data"
SPAN_MISMATCH = "a gold span's text differs from the text at its offsets"
MALFORMED = "malformed row"


class DatasetError(Exception):
    """A dataset cannot be read (a missing extra, a failed download, a
    missing local checkout). Messages name datasets, files, revisions and
    exception types, never row content."""


# hf_hub_download's shape: keyword arguments in, local file path out.
Download = Callable[..., str]


@dataclass(frozen=True)
class LoadRequest:
    split: str
    seed: int = 42
    # Keep only rows in this language (datasets that record one).
    language: str | None = None
    cache_dir: Path | None = None
    # A local checkout the dataset is read from (datasets fetched by their
    # own tooling, not from the Hub).
    data_dir: Path | None = None
    # Stand-in for huggingface_hub.hf_hub_download (tests); None = the real one.
    download: Download | None = None
    # Rows the adapter skipped, by reason (counted, reported, never shown).
    skipped: Counter[str] = field(default_factory=Counter)


Adapter = Callable[["DatasetSpec", LoadRequest], Iterator[NerSample]]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    summary: str
    license: str
    attribution: str
    # True when the rows are real text (real prompts, real code, real
    # documents) rather than generated: --dump-errors refuses it unless
    # --allow-real-data-dump is also given.
    real_data: bool
    label_map: LabelMap
    # The first split is the default.
    splits: tuple[str, ...]
    adapter: Adapter
    # Facts the report repeats (how the labels were made, what the scores
    # mean for this dataset).
    notes: tuple[str, ...] = ()
    # Published datasets: the Hub repository, its pinned commit, the files
    # of each split, the card, and the date the card was last checked.
    hub_id: str | None = None
    revision: str | None = None
    files: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))
    card: str | None = None
    checked: str | None = None
    # Whether --language can filter the rows (the dataset records one).
    filters_language: bool = False
    # Whether the rows come from a local checkout (--data-dir).
    needs_data_dir: bool = False

    @property
    def default_split(self) -> str:
        return self.splits[0]


def default_cache_dir(environ: Mapping[str, str] = os.environ) -> Path:
    """Where downloaded datasets are kept: ``$XDG_CACHE_HOME`` (an empty
    value counts as unset) or ``~/.cache``, under
    ``llm-redact/bench-datasets``."""
    base = environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "llm-redact" / "bench-datasets"


def hf_hub_download(
    *, repo_id: str, filename: str, repo_type: str, revision: str, cache_dir: str
) -> str:
    try:
        from huggingface_hub import hf_hub_download as download
    except ImportError as exc:
        raise DatasetError(f"downloading a dataset needs huggingface_hub; {INSTALL_HINT}") from exc
    return str(
        download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
        )
    )


def fetch(spec: DatasetSpec, split: str, request: LoadRequest) -> list[Path]:
    """The local paths of ``split``'s files, downloaded at the pinned
    revision into the request's cache (reused when already there)."""
    if spec.hub_id is None or spec.revision is None:
        raise DatasetError(f"dataset {spec.name!r} names no Hub repository and revision")
    download = request.download or hf_hub_download
    cache_dir = request.cache_dir or default_cache_dir()
    paths = []
    for filename in spec.files[split]:
        try:
            path = download(
                repo_id=spec.hub_id,
                filename=filename,
                repo_type="dataset",
                revision=spec.revision,
                cache_dir=str(cache_dir),
            )
        except DatasetError:
            raise
        except Exception as exc:  # network, auth, a missing file: name what was asked
            raise DatasetError(
                f"could not download {filename} of {spec.hub_id} at revision"
                f" {spec.revision}: {type(exc).__name__}"
            ) from exc
        paths.append(Path(path))
    return paths


def checked_spans(
    text: str,
    spans: Iterable[tuple[object, object, object, object]],
    request: LoadRequest,
) -> tuple[GoldSpan, ...] | None:
    """Gold spans from ``(start, end, label, value)`` entries, or None (the
    row is skipped and counted) when an offset is not an integer inside the
    text or a value — when the dataset gives one (not None) — differs from
    the text at its offsets. Empty spans (start == end) are dropped."""
    gold = []
    for start, end, label, value in spans:
        # bool is an int subclass; an offset must be a plain int.
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(text):
            request.skipped[MALFORMED] += 1
            return None
        if value is not None and text[start:end] != str(value):
            request.skipped[SPAN_MISMATCH] += 1
            return None
        if start < end:  # an empty span (an empty payload value) covers nothing
            gold.append(GoldSpan(start, end, str(label)))
    return tuple(gold)


def jsonl_rows(path: Path, request: LoadRequest) -> Iterator[object]:
    """The JSON value of every non-blank line of a JSONL file; a line that
    does not parse is skipped and counted."""
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError:
                request.skipped[MALFORMED] += 1
