"""The NER bench's dataset framework and adapters, with inline fixture rows
and a fake hf_hub_download (no network, no extra required: pyarrow is faked
where the real one is absent)."""

import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from llm_redact.bench import ner as bench_ner
from llm_redact.bench.datasets import (
    DATASETS,
    LoadRequest,
    base,
    dataset_key,
    nemotron,
    openpii,
    resolve,
)
from llm_redact.bench.ner_metrics import LEAK, GoldSpan, NerSample, Pipeline, evaluate
from llm_redact.detection.engine import DetectionConfig, build_detectors
from ner_fakes import FakeHfPipe, install_transformers

# --- framework ----------------------------------------------------------------


def test_default_cache_dir_is_outside_the_repo(tmp_path: Path) -> None:
    assert base.default_cache_dir({"XDG_CACHE_HOME": str(tmp_path)}) == (
        tmp_path / "llm-redact" / "bench-datasets"
    )
    # An empty XDG variable counts as unset.
    assert base.default_cache_dir({"XDG_CACHE_HOME": ""}) == (
        Path.home() / ".cache" / "llm-redact" / "bench-datasets"
    )


class _Downloads:
    def __init__(self, files: dict[str, Path]) -> None:
        self.files = files
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return str(self.files[str(kwargs["filename"])])


def test_fetch_downloads_each_file_at_the_pinned_revision(tmp_path: Path) -> None:
    spec = DATASETS["openpii"]
    downloads = _Downloads({"data/validation.jsonl": tmp_path / "v.jsonl"})
    request = LoadRequest(split="validation", cache_dir=tmp_path / "cache", download=downloads)
    assert base.fetch(spec, "validation", request) == [tmp_path / "v.jsonl"]
    assert downloads.calls == [
        {
            "repo_id": "ai4privacy/pii-masking-openpii-1.5m",
            "filename": "data/validation.jsonl",
            "repo_type": "dataset",
            "revision": openpii.REVISION,
            "cache_dir": str(tmp_path / "cache"),
        }
    ]
    # Without a cache_dir, the default one.
    base.fetch(spec, "validation", LoadRequest(split="validation", download=downloads))
    assert downloads.calls[-1]["cache_dir"] == str(base.default_cache_dir())


def test_fetch_failures_name_what_was_asked() -> None:
    def offline(**kwargs: object) -> str:
        raise ConnectionError("no route to host 10.1.2.3")

    request = LoadRequest(split="test", download=offline)
    with pytest.raises(base.DatasetError) as excinfo:
        base.fetch(DATASETS["nemotron"], "test", request)
    message = str(excinfo.value)
    assert "data/test-00000-of-00001.parquet of nvidia/Nemotron-PII" in message
    assert nemotron.REVISION in message
    assert message.endswith("ConnectionError")  # the type only

    def missing_extra(**kwargs: object) -> str:
        raise base.DatasetError("needs huggingface_hub")

    with pytest.raises(base.DatasetError, match="needs huggingface_hub"):
        base.fetch(DATASETS["nemotron"], "test", LoadRequest(split="test", download=missing_extra))
    with pytest.raises(base.DatasetError, match="names no Hub repository"):
        base.fetch(DATASETS["synthetic"], "generated", LoadRequest(split="generated"))


def test_hf_hub_download_needs_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(base.DatasetError, match="uv sync --extra bench-data"):
        base.hf_hub_download(
            repo_id="r", filename="f", repo_type="dataset", revision="x", cache_dir="c"
        )
    fake = types.ModuleType("huggingface_hub")
    seen: dict[str, object] = {}

    def download(**kwargs: object) -> Path:
        seen.update(kwargs)
        return Path("/cache/f")

    fake.hf_hub_download = download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    path = base.hf_hub_download(
        repo_id="r", filename="f", repo_type="dataset", revision="x", cache_dir="c"
    )
    assert path == "/cache/f"
    assert seen == {
        "repo_id": "r",
        "filename": "f",
        "repo_type": "dataset",
        "revision": "x",
        "cache_dir": "c",
    }


def test_checked_spans_validates_offsets_and_values() -> None:
    text = "Ann Lee, 42"
    request = LoadRequest(split="x")
    assert base.checked_spans(text, [(0, 3, "GIVENNAME", "Ann"), (9, 11, "AGE", 42)], request) == (
        GoldSpan(0, 3, "GIVENNAME"),
        GoldSpan(9, 11, "AGE"),
    )
    # No value given: offsets alone.
    assert base.checked_spans(text, [(4, 7, "SURNAME", None)], request) == (
        GoldSpan(4, 7, "SURNAME"),
    )
    assert base.checked_spans(text, [], request) == ()
    assert not request.skipped
    for entry in [
        (0, 3, "X", "Bob"),  # the value differs from the text
    ]:
        assert base.checked_spans(text, [entry], request) is None
    assert request.skipped == {base.SPAN_MISMATCH: 1}
    for entry in [
        (True, 3, "X", None),
        (0, "3", "X", None),
        (5, 5, "X", None),
        (-1, 3, "X", None),
        (0, 99, "X", None),
    ]:
        assert base.checked_spans(text, [entry], request) is None
    assert request.skipped[base.MALFORMED] == 5


def test_resolve_and_keys_cover_published_datasets() -> None:
    spec, split = resolve("openpii")
    assert (spec.name, split) == ("openpii", "validation")
    assert dataset_key(spec, "train") == "openpii:train"
    assert dataset_key(spec, "validation", "de") == "openpii@de"
    assert dataset_key(spec, "train", "de") == "openpii:train@de"
    assert resolve("nemotron:train")[1] == "train"


def test_gold_parts_merge_like_detections() -> None:
    text = "Signed: Tanaka Yuki, 3 Chome"
    sample = NerSample(
        text,
        (
            GoldSpan(8, 14, "SURNAME"),
            GoldSpan(15, 19, "GIVENNAME"),
            GoldSpan(21, 22, "BUILDINGNUM"),
        ),
    )
    rules = build_detectors(DetectionConfig())
    result = evaluate([sample], openpii.LABELS, Pipeline.from_detectors(rules, rules))
    assert result.exact["PERSON"].gold == 1  # one span "Tanaka Yuki", not two
    assert result.exact["ADDRESS"].gold == 1
    assert result.gold_chars == len("Tanaka Yuki") + 1


# --- OpenPII 1.5M -------------------------------------------------------------


def _openpii_row(text: str, mask: list[tuple[str, str]], language: str = "en") -> str:
    entries = []
    for value, label in mask:
        start = text.index(value)
        entries.append({"value": value, "start": start, "end": start + len(value), "label": label})
    row = {"source_text": text, "privacy_mask": entries, "language": language, "region": "X"}
    return json.dumps(row)


def _openpii_file(tmp_path: Path) -> Path:
    good = _openpii_row(
        "Write to Ann Lee at 12 Rose Street, tax id 99-12.",
        [
            ("Ann", "GIVENNAME"),
            ("Lee", "SURNAME"),
            ("12", "BUILDINGNUM"),
            ("Rose Street", "STREET"),
            ("99-12", "TAXNUM"),
        ],
    )
    german = _openpii_row(
        "Herr Weber wohnt in Bonn.", [("Weber", "SURNAME"), ("Bonn", "CITY")], "de"
    )
    shifted = json.loads(good)
    shifted["privacy_mask"][0]["start"] += 1  # the value no longer matches
    lines = [
        good,
        german,
        json.dumps(shifted),
        "{not json",
        "",
        json.dumps(["a", "list"]),
        json.dumps({"source_text": 5, "privacy_mask": []}),
        json.dumps({"source_text": "x", "privacy_mask": ["not a dict"], "language": "en"}),
    ]
    path = tmp_path / "validation.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_openpii_adapter_reads_validates_and_counts(tmp_path: Path) -> None:
    path = _openpii_file(tmp_path)
    spec = DATASETS["openpii"]
    request = LoadRequest(split="validation", download=lambda **kw: str(path))
    samples = list(spec.adapter(spec, request))
    assert [s.context for s in samples] == ["en", "de"]
    assert {g.label for g in samples[0].spans} == {
        "GIVENNAME",
        "SURNAME",
        "BUILDINGNUM",
        "STREET",
        "TAXNUM",
    }
    assert dict(request.skipped) == {base.SPAN_MISMATCH: 1, base.MALFORMED: 4}
    german = LoadRequest(split="validation", language="de", download=lambda **kw: str(path))
    assert [s.context for s in spec.adapter(spec, german)] == ["de"]

    rules = build_detectors(DetectionConfig())
    result = evaluate(samples, spec.label_map, Pipeline.from_detectors(rules, rules))
    # "Ann" + "Lee" merged (plus "Weber"); "12" + "Rose Street" merged.
    assert result.exact["PERSON"].gold == 2
    assert result.exact["ADDRESS"].gold == 1
    assert "CITY" not in result.exact  # not scored
    assert result.gold_chars == len("Ann Lee") + len("12 Rose Street") + len("99-12") + len("Weber")
    assert result.unmapped == {}


def test_openpii_label_map_follows_the_plan() -> None:
    labels = openpii.LABELS
    assert {k for k, v in labels.items() if v == "PERSON"} == {"GIVENNAME", "SURNAME"}
    assert {k for k, v in labels.items() if v == LEAK} == {"SOCIALNUM", "TAXNUM", "IDCARDNUM"}
    assert len(labels) == 19  # the card's taxonomy
    assert DATASETS["openpii"].filters_language
    assert not DATASETS["openpii"].real_data


# --- Nemotron-PII -------------------------------------------------------------


def _nemotron_row(
    text: str, spans: list[tuple[str, str]], *, literal: bool = True
) -> dict[str, Any]:
    entries = []
    for value, label in spans:
        start = text.index(value)
        entries.append({"start": start, "end": start + len(value), "text": value, "label": label})
    raw = repr(entries) if literal else json.dumps(entries)
    return {"text": text, "spans": raw, "document_format": "unstructured", "locale": "us"}


def test_parse_spans_reads_every_shape() -> None:
    assert nemotron.parse_spans([{"a": 1}]) == [{"a": 1}]
    assert nemotron.parse_spans("[{'start': 0}]") == [{"start": 0}]
    assert nemotron.parse_spans('[{"start": 0}]') == [{"start": 0}]
    assert nemotron.parse_spans("{'start': 0}") is None  # not a list
    assert nemotron.parse_spans("__import__('os')") is None  # never evaluated
    assert nemotron.parse_spans(None) is None


def test_nemotron_samples_map_validate_and_count() -> None:
    age_row = _nemotron_row("Age 45, user dkim", [("dkim", "user_name")])
    age_row["spans"] = (
        age_row["spans"][:-1] + ", {'start': 4, 'end': 6, 'text': 45, 'label': 'age'}]"
    )
    rows = [
        _nemotron_row(
            "Dear Maya Ortiz, your account 8812 3341",
            [
                ("Maya", "first_name"),
                ("Ortiz", "last_name"),
                ("8812 3341", "account_number"),
            ],
        ),
        _nemotron_row("ssn 123-45-6789 on file", [("123-45-6789", "ssn")], literal=False),
        age_row,  # an int value compared as text
        {"text": "x", "spans": "garbage("},
        {"text": "x", "spans": "['not a dict']"},
        {"text": None, "spans": "[]"},
        {
            "text": "Dear Maya",
            "spans": "[{'start': 0, 'end': 4, 'text': 'Maya', 'label': 'first_name'}]",
        },  # shifted
    ]
    request = LoadRequest(split="test")
    samples = list(nemotron.samples(rows, request))
    assert len(samples) == 3
    assert samples[0].context == "unstructured/us"
    assert dict(request.skipped) == {base.MALFORMED: 3, base.SPAN_MISMATCH: 1}
    rules = build_detectors(DetectionConfig())
    result = evaluate(samples, nemotron.LABELS, Pipeline.from_detectors(rules, rules))
    assert result.exact["PERSON"].gold == 1
    assert result.exact["ACCOUNT_NUMBER"].gold == 1
    assert result.exact["SSN"].gold == 1
    assert result.exact["USERNAME"].gold == 1
    assert result.unmapped == {}


def test_nemotron_label_map_follows_the_plan() -> None:
    scored = {k: v for k, v in nemotron.LABELS.items() if v is not None}
    assert scored == {
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


class _Batch:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def to_pylist(self) -> list[dict[str, Any]]:
        return self.rows


def _fake_pyarrow(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> list[object]:
    calls: list[object] = []

    class ParquetFile:
        def __init__(self, path: Path) -> None:
            calls.append(path)

        def iter_batches(self, batch_size: int, columns: list[str]) -> Iterator[_Batch]:
            calls.append((batch_size, columns))
            yield _Batch(rows[:1])
            yield _Batch(rows[1:])

    parquet = types.ModuleType("pyarrow.parquet")
    parquet.ParquetFile = ParquetFile  # type: ignore[attr-defined]
    package = types.ModuleType("pyarrow")
    package.parquet = parquet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", package)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    return calls


def test_nemotron_adapter_reads_parquet_in_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        _nemotron_row("Hi Maya Ortiz", [("Maya", "first_name"), ("Ortiz", "last_name")]),
        _nemotron_row("login dkim", [("dkim", "user_name")]),
    ]
    calls = _fake_pyarrow(monkeypatch, rows)
    path = tmp_path / "test.parquet"
    spec = DATASETS["nemotron"]
    request = LoadRequest(split="test", download=lambda **kw: str(path))
    samples = list(spec.adapter(spec, request))
    assert [s.text for s in samples] == ["Hi Maya Ortiz", "login dkim"]
    assert calls == [path, (nemotron.BATCH_ROWS, list(nemotron.COLUMNS))]


def test_nemotron_needs_pyarrow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(base.DatasetError, match="uv sync --extra bench-data"):
        list(nemotron.parquet_rows(tmp_path / "x.parquet"))


def test_nemotron_reads_a_real_parquet_file(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [
        _nemotron_row("Hi Maya Ortiz", [("Maya", "first_name"), ("Ortiz", "last_name")]),
        {**_nemotron_row("login dkim", [("dkim", "user_name")]), "uid": "u2"},
    ]
    table = pa.Table.from_pylist([{k: r.get(k) for k in (*nemotron.COLUMNS, "uid")} for r in rows])
    path = tmp_path / "test.parquet"
    pq.write_table(table, path)
    read = list(nemotron.parquet_rows(path))
    assert [r["text"] for r in read] == ["Hi Maya Ortiz", "login dkim"]
    assert set(read[0]) == set(nemotron.COLUMNS)  # only the columns asked for


# --- the command line ---------------------------------------------------------


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "hf-fake.toml"
    path.write_text('[detection.ner]\nenabled = true\nbackend = "hf"\nentities = ["PERSON"]\n')
    return path


def test_cli_scores_a_downloaded_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_transformers(monkeypatch, FakeHfPipe(findings=[("Ann Lee", "PER", 0.9)]))
    path = _openpii_file(tmp_path)
    asked: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        asked.append(kwargs)
        return str(path)

    monkeypatch.setattr(base, "hf_hub_download", download)
    thresholds = tmp_path / "t.toml"
    thresholds.write_text('[hf-fake."openpii@en"]\nrecall = { PERSON = 1.0 }\n')
    cache = tmp_path / "cache"
    argv = ["--config", str(_config(tmp_path)), "--dataset", "openpii", "--language", "en"]
    argv += ["--cache-dir", str(cache), "--thresholds", str(thresholds), "--check"]
    assert bench_ner.main(argv) == 0
    printed = capsys.readouterr().out
    assert asked[0]["cache_dir"] == str(cache)
    assert "# NER bench: hf-fake on openpii@en" in printed
    assert "Attribution: OpenPII 1.5M by Ai4Privacy / Ai Suisse SA" in printed
    assert f"Source: ai4privacy/pii-masking-openpii-1.5m at revision {openpii.REVISION}." in printed
    assert "Rows in language en only." in printed
    assert "Rows skipped:" in printed
    assert 'check passed: [hf-fake."openpii@en"]' in printed
    assert "Ann" not in printed


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--dataset", "synthetic", "--language", "en"], "dataset 'synthetic' has no language"),
        (["--fp-corpus", "{tmp}", "--language", "en"], "--fp-corpus has no language"),
        (["--cache-dir", "{repo}"], "--cache-dir must be outside any git work tree"),
        (["--dataset", "nemotron"], "could not download data/test-00000-of-00001.parquet"),
    ],
)
def test_cli_dataset_input_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    install_transformers(monkeypatch, FakeHfPipe(findings=[]))

    def offline(**kwargs: object) -> str:
        raise OSError("offline")

    monkeypatch.setattr(base, "hf_hub_download", offline)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    argv = [{"{tmp}": str(tmp_path), "{repo}": str(repo / "cache")}.get(a, a) for a in extra]
    cache = [] if "--cache-dir" in extra else ["--cache-dir", str(tmp_path / "cache")]
    assert bench_ner.main(["--config", str(_config(tmp_path)), *cache, *argv]) == 2
    assert message in capsys.readouterr().err


def test_list_datasets_names_licenses_and_attributions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert bench_ner.main(["--list-datasets"]) == 0
    printed = capsys.readouterr().out
    for spec in DATASETS.values():
        assert f"{spec.name}: {spec.summary}" in printed
        assert f"  license: {spec.license}" in printed
        assert f"  attribution: {spec.attribution}" in printed
    assert f"source: nvidia/Nemotron-PII at revision {nemotron.REVISION}" in printed
    assert (
        "card: https://huggingface.co/datasets/nvidia/Nemotron-PII (checked 2026-10-05)" in printed
    )
    assert "--language filters its rows" in printed
