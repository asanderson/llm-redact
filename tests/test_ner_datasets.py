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
    creddata,
    dataset_key,
    mapa,
    nemotron,
    openpii,
    privy,
    pupa,
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
    assert base.checked_spans(text, [(5, 5, "O", "")], request) == ()  # empty: dropped
    assert not request.skipped
    for entry in [
        (0, 3, "X", "Bob"),  # the value differs from the text
    ]:
        assert base.checked_spans(text, [entry], request) is None
    assert request.skipped == {base.SPAN_MISMATCH: 1}
    for entry in [
        (True, 3, "X", None),
        (0, "3", "X", None),
        (6, 5, "X", None),
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


# --- beki/privy ---------------------------------------------------------------


def test_iter_json_array_streams_across_chunk_boundaries() -> None:
    import io

    rows = [{"full_text": "x" * n, "spans": []} for n in (1, 50, 200, 3)]
    text = " \n[\n" + ",\n ".join(json.dumps(r) for r in rows) + "\n]\n"
    for read_chars in (1, 7, 64, 1 << 20):
        assert list(privy.iter_json_array(io.StringIO(text), read_chars=read_chars)) == rows
    assert list(privy.iter_json_array(io.StringIO("[]"))) == []
    for bad, message in (
        ("", "does not hold a JSON array"),
        ('{"a": 1}', "does not hold a JSON array"),
        ('[{"a": 1}, ', "ends inside its JSON array"),
        ('[{"a": 1}, {"b": ', "ends inside an element"),
    ):
        with pytest.raises(base.DatasetError, match=message):
            list(privy.iter_json_array(io.StringIO(bad), read_chars=4))


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ('{"name": "x"}', "json"),
        ("[1]", "json"),
        ("b'<?xml version=\"1.0\"?><a/>'", "xml"),
        ("<?xml version='1.0'?>", "xml"),
        ("<TABLE><TR></TR></TABLE>", "html"),
        ("  select * from t", "sql"),
        ("INSERT INTO t VALUES (1)", "sql"),
        ("plain words", "other"),
    ],
)
def test_privy_payload_kinds(text: str, kind: str) -> None:
    assert privy.payload_kind(text) == kind


def _privy_row(text: str, spans: list[tuple[str, str]]) -> dict[str, Any]:
    entries = []
    for value, label in spans:
        start = text.index(value) if value else 0
        entries.append(
            {
                "entity_type": label,
                "entity_value": value,
                "start_position": start,
                "end_position": start + len(value),
            }
        )
    return {"full_text": text, "spans": entries, "masked": "", "tags": [], "tokens": []}


def _privy_archive(tmp_path: Path, rows: list[object]) -> Path:
    import zipfile

    path = tmp_path / "privy-dataset.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("test-small.json", json.dumps(rows, indent=2))
        archive.writestr("__MACOSX/._test-small.json", b"\x00")
    return path


def test_privy_adapter_reads_the_archive(tmp_path: Path) -> None:
    rows: list[object] = [
        _privy_row(
            '{"full_name": "Ann Lee", "status": "active", "note": ""}',
            [("Ann Lee", "PER"), ("active", "O"), ("", "O")],
        ),
        _privy_row("UPDATE t SET ssn = '123-45-6789'", [("123-45-6789", "US_SSN")]),
        ["not", "a", "row"],
        {"full_text": 3, "spans": []},
        {"full_text": "x", "spans": ["not a dict"]},
        {
            "full_text": "Ann",
            "spans": [
                {
                    "entity_type": "PER",
                    "entity_value": "Bob",
                    "start_position": 0,
                    "end_position": 3,
                }
            ],
        },
    ]
    path = _privy_archive(tmp_path, rows)
    spec = DATASETS["privy"]
    request = LoadRequest(split="test", download=lambda **kw: str(path))
    samples = list(spec.adapter(spec, request))
    assert [s.context for s in samples] == ["json", "sql"]
    assert dict(request.skipped) == {base.MALFORMED: 3, base.SPAN_MISMATCH: 1}
    rules = build_detectors(DetectionConfig())
    result = evaluate(samples, spec.label_map, Pipeline.from_detectors(rules, rules))
    assert result.exact["PERSON"].gold == 1
    assert result.exact["SSN"].gold_hit == 1  # the rules find it
    # "active" is marked not-PII: outside every gold span.
    assert result.outside_chars == sum(len(s.text) for s in samples) - len("Ann Lee") - 11


def test_privy_adapter_reports_unreadable_archives(tmp_path: Path) -> None:
    spec = DATASETS["privy"]
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip")
    request = LoadRequest(split="test", download=lambda **kw: str(broken))
    with pytest.raises(base.DatasetError, match="cannot read privy-dataset.zip: BadZipFile"):
        list(spec.adapter(spec, request))
    path = _privy_archive(tmp_path, [])
    request = LoadRequest(split="dev", download=lambda **kw: str(path))
    with pytest.raises(base.DatasetError, match="KeyError"):
        list(spec.adapter(spec, request))


def test_privy_label_map() -> None:
    labels = privy.LABELS
    assert labels["PER"] == labels["PERSON"] == "PERSON"
    assert labels["O"] == "@not-pii"
    assert labels["IP_ADDRESS"] == LEAK
    assert labels["NRP"] is None  # a sensitive attribute: never scored
    assert set(DATASETS["privy"].splits) == {
        "test",
        "dev",
        "train",
        "test-large",
        "dev-large",
        "train-large",
    }


# --- PUPA ---------------------------------------------------------------------


def test_pupa_unit_spans_are_whole_word_and_case_insensitive() -> None:
    prompt = "Email Jane Roe, cc jane roe and JANEROE; Roe's file"
    assert pupa.unit_spans(prompt, "jane roe") == [(6, 14), (19, 27)]
    assert pupa.unit_spans(prompt, "roe") == [(11, 14), (24, 27), (41, 44)]
    assert pupa.unit_spans(prompt, "ane") == []  # inside words only
    assert pupa.unit_spans(prompt, "") == []
    # Lowercasing that changes the length: offsets cannot be trusted.
    assert pupa.unit_spans("İstanbul office", "office") == []


def _pupa_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    import csv

    fields = [
        "conversation_hash",
        "predicted_category",
        "user_query",
        "target_response",
        "pii_units",
        "redacted_query",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def test_pupa_adapter_locates_units_and_counts_misses(tmp_path: Path) -> None:
    tnb = _pupa_csv(
        tmp_path / "PUPA_TNB.csv",
        [
            {
                "predicted_category": "job applications",
                "user_query": "Write a cover letter for Mara Quint at Lumen Labs.",
                "pii_units": "mara quint||lumen labs||mara quint",
            },
            {"predicted_category": "misc", "user_query": "Summarise this.", "pii_units": ""},
        ],
    )
    new = _pupa_csv(
        tmp_path / "PUPA_New.csv",
        [{"user_query": "Ask Ola.", "pii_units": "ola||nobody"}],
    )
    files = {"PUPA_TNB.csv": tnb, "PUPA_New.csv": new}
    spec = DATASETS["pupa"]
    request = LoadRequest(split="all", download=lambda **kw: str(files[str(kw["filename"])]))
    samples = list(spec.adapter(spec, request))
    assert [s.context for s in samples] == ["job applications", "misc"]
    assert [(g.start, g.end) for g in samples[0].spans] == [(25, 35), (39, 49)]
    assert samples[1].spans == ()
    assert dict(request.skipped) == {pupa.UNIT_NOT_FOUND: 1}
    assert list(pupa.samples([{"user_query": None, "pii_units": "x"}], request)) == []
    assert request.skipped[base.MALFORMED] == 1
    assert spec.real_data
    assert spec.label_map == {pupa.UNIT_LABEL: LEAK}


def test_pupa_unreadable_csv(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes(b"user_query,pii_units\n\xff\xfe,x\n")
    with pytest.raises(base.DatasetError, match="cannot read bad.csv: UnicodeDecodeError"):
        list(pupa.csv_rows(path))


def test_pupa_dump_needs_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_transformers(monkeypatch, FakeHfPipe(findings=[]))
    argv = ["--config", str(_config(tmp_path)), "--dataset", "pupa"]
    argv += ["--dump-errors", str(tmp_path / "errors.jsonl")]
    assert bench_ner.main(argv) == 2
    assert "dataset 'pupa' holds real data" in capsys.readouterr().err


# --- MAPA ---------------------------------------------------------------------


def test_iob_spans_rebuild_text_and_spans() -> None:
    tokens = ["Mr", "K.", "Muller", "v", "Rat", "der", "Stadt", "(", "1", ")"]
    tags = [
        "B-TITLE",
        "B-INITIAL NAME",
        "B-FAMILY NAME",
        "O",
        "I-ROLE",  # an I- without its B- starts a span
        "I-ROLE",
        "B-CITY",
        "O",
        "B-VALUE",
        "O",
    ]
    text, spans = mapa.iob_spans(tokens, tags)
    assert text == "Mr K. Muller v Rat der Stadt ( 1 )"
    assert [(text[s.start : s.end], s.label) for s in spans] == [
        ("Mr", "TITLE"),
        ("K.", "INITIAL NAME"),
        ("Muller", "FAMILY NAME"),
        ("Rat der", "ROLE"),
        ("Stadt", "CITY"),
        ("1", "VALUE"),
    ]
    assert mapa.iob_spans(["A", "B"], ["B-FAMILY NAME", "I-FAMILY NAME"])[1] == [
        GoldSpan(0, 3, "FAMILY NAME")
    ]


def test_mapa_adapter_filters_and_counts(tmp_path: Path) -> None:
    def row(language: str, tokens: list[str], tags: list[str]) -> str:
        return json.dumps(
            {"language": language, "type": "EUR-LEX", "tokens": tokens, "fine_grained": tags}
        )

    path = tmp_path / "test.jsonl"
    path.write_text(
        "\n".join(
            [
                row("en", ["Mrs", "Okoro", "appealed"], ["B-TITLE", "B-FAMILY NAME", "O"]),
                row("de", ["Herr", "Weber"], ["B-TITLE", "B-FAMILY NAME"]),
                row("en", ["x"], ["O", "O"]),  # length mismatch
                row("en", ["x", 3], ["O", "O"]),  # type: ignore[list-item]
                json.dumps([1]),
                "{broken",
            ]
        )
    )
    spec = DATASETS["mapa"]
    request = LoadRequest(split="test", download=lambda **kw: str(path))
    samples = list(spec.adapter(spec, request))
    assert [s.context for s in samples] == ["en", "de"]
    assert dict(request.skipped) == {base.MALFORMED: 4}
    english = LoadRequest(split="test", language="en", download=lambda **kw: str(path))
    assert [s.text for s in spec.adapter(spec, english)] == ["Mrs Okoro appealed"]
    rules = build_detectors(DetectionConfig())
    result = evaluate(samples, spec.label_map, Pipeline.from_detectors(rules, rules))
    assert result.exact["PERSON"].gold == 2
    assert result.unmapped == {}
    assert spec.real_data and spec.filters_language


# --- CredData -----------------------------------------------------------------

_CRED_HEADER = (
    "Id,FileID,Domain,RepoName,FilePath,LineStart,LineEnd,GroundTruth,ValueStart,ValueEnd,"
    "CryptographyKey,PredefinedPattern,Category\n"
)


def _cred_row(path: str, lines: tuple[int, int], truth: str, value: tuple[str, str]) -> str:
    return f"1,f,GitHub,r,{path},{lines[0]},{lines[1]},{truth},{value[0]},{value[1]},,,Password\n"


def _creddata_checkout(root: Path) -> Path:
    code = root / "data" / "r" / "src"
    code.mkdir(parents=True)
    (root / "data" / "r" / "test").mkdir()
    (code / "a.py").write_text(
        "import os\n"
        'TOKEN = "xxxx-not-a-real-value-xxxx"\n'
        "KEY = (\n"
        '    "yyyy-not-real"\n'
        ")\n"
        'password_hint = "see the vault"\n'
    )
    (root / "data" / "r" / "test" / "b.cfg").write_bytes(b"pw=\xe9t\xe9-not-real\r\n")
    meta = root / "meta"
    meta.mkdir()
    rows = [
        _cred_row("data/r/src/a.py", (2, 2), "T", ("9", "35")),
        _cred_row("data/r/src/a.py", (2, 2), "F", ("0", "5")),  # same line: no gold
        _cred_row("data/r/src/a.py", (6, 6), "F", ("", "")),  # a look-alike line
        _cred_row("data/r/src/a.py", (3, 4), "T", ("6", "19")),  # spans two lines
        _cred_row("data/r/test/b.cfg", (1, 1), "T", ("3", "")),  # to the end of the line
        _cred_row("data/r/src/a.py", (1, 1), "T", ("-1", "")),  # true but no offsets
        _cred_row("data/r/src/gone.py", (1, 1), "T", ("0", "1")),
        _cred_row("data/r/src/a.py", (5, 9), "F", ("", "")),  # past the file's end
        _cred_row("data/r/src/a.py", (5, 5), "T", ("40", "41")),  # past the line
    ]
    (meta / "r.csv").write_text(_CRED_HEADER + "".join(rows))
    return root


def test_creddata_reads_a_local_checkout(tmp_path: Path) -> None:
    root = _creddata_checkout(tmp_path / "CredData")
    spec = DATASETS["creddata"]
    request = LoadRequest(split="all", data_dir=root)
    samples = list(spec.adapter(spec, request))
    values = [[sample.text[g.start : g.end] for g in sample.spans] for sample in samples]
    assert values == [
        ["xxxx-not-a-real-value-xxxx"],
        [],  # the F-only line: a negative
        ['(\n    "yyyy-not-real"'],  # from line 3's offset to line 4's
        ["\xe9t\xe9-not-real"],  # a Latin-1 file; the value runs to the line's end
    ]
    assert [s.context for s in samples] == ["py", "py", "py", "cfg"]
    assert dict(request.skipped) == {
        creddata.NO_VALUE: 1,
        creddata.MISSING_FILE: 1,
        creddata.BAD_LINES: 2,
    }
    tests_only = LoadRequest(split="test", data_dir=root)
    assert [s.context for s in spec.adapter(spec, tests_only)] == ["cfg"]
    rules = build_detectors(DetectionConfig())
    result = evaluate(samples, spec.label_map, Pipeline.from_detectors(rules, rules))
    assert result.gold_chars == sum(len(g) for v in values for g in v)
    assert spec.real_data and spec.needs_data_dir


def test_creddata_needs_its_metadata(tmp_path: Path) -> None:
    spec = DATASETS["creddata"]
    with pytest.raises(base.DatasetError, match="meta/ directory"):
        list(spec.adapter(spec, LoadRequest(split="all", data_dir=tmp_path)))
    with pytest.raises(base.DatasetError, match="meta/ directory"):
        list(spec.adapter(spec, LoadRequest(split="all")))
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "r.csv").write_text("Id,FileID\n1,f\n")
    with pytest.raises(base.DatasetError, match="cannot read meta/r.csv: KeyError"):
        list(spec.adapter(spec, LoadRequest(split="all", data_dir=tmp_path)))


def test_cli_creddata_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_transformers(monkeypatch, FakeHfPipe(findings=[]))
    root = _creddata_checkout(tmp_path / "CredData")
    config = ["--config", str(_config(tmp_path))]
    assert bench_ner.main([*config, "--dataset", "creddata"]) == 2
    assert "reads a local checkout: pass --data-dir" in capsys.readouterr().err
    assert bench_ner.main([*config, "--data-dir", str(root)]) == 2
    assert "--data-dir applies only to" in capsys.readouterr().err
    dump = ["--dump-errors", str(tmp_path / "e.jsonl")]
    argv = [*config, "--dataset", "creddata", "--data-dir", str(root)]
    assert bench_ner.main([*argv, *dump]) == 2
    assert "holds real data" in capsys.readouterr().err
    assert bench_ner.main(argv) == 0
    printed = capsys.readouterr().out
    assert "Source: a local checkout (--data-dir)." in printed
    assert "not-a-real" not in printed
    assert bench_ner.main(["--list-datasets"]) == 0
    assert "reads a local checkout: --data-dir DIR" in capsys.readouterr().out
