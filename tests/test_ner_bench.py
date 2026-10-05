"""The NER bench (python -m llm_redact.bench.ner): metrics, the
structured-regression check, the thresholds gate and the dump guard, with
fake NER detectors and inline samples (no extra, no model, no network)."""

import json
import stat
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from llm_redact.bench import ner as bench_ner
from llm_redact.bench.datasets import DATASETS, LoadRequest, dataset_key, resolve
from llm_redact.bench.ner_metrics import (
    LEAK,
    NOT_PII,
    GoldSpan,
    NerResult,
    NerSample,
    Pipeline,
    StructuredCounts,
    TypeCounts,
    evaluate,
    to_json_dict,
    to_markdown,
)
from llm_redact.detection.base import Detection, Detector
from llm_redact.detection.engine import DetectionConfig, build_detectors
from ner_fakes import FakeHfPipe, install_transformers


@dataclass
class _FakeNer:
    """Finds fixed surface strings and reports them under a fixed type."""

    surfaces: dict[str, str]
    name: str = "fake_ner"

    def detect(self, text: str) -> Iterable[Detection]:
        found = []
        for surface, type_name in self.surfaces.items():
            start = text.find(surface)
            while start != -1:
                end = start + len(surface)
                found.append(Detection(start, end, type_name, text[start:end], priority=120))
                start = text.find(surface, start + 1)
        return found


def _pipeline(surfaces: dict[str, str]) -> Pipeline:
    rules = build_detectors(DetectionConfig())
    full: list[Detector] = [*rules, _FakeNer(surfaces)]
    return Pipeline.from_detectors(full, rules)


def _span(text: str, surface: str, label: str) -> GoldSpan:
    start = text.index(surface)
    return GoldSpan(start, start + len(surface), label)


LABELS = {
    "name": "PERSON",
    "mail": "EMAIL",
    "taxnum": LEAK,
    "city": None,
    "plain": NOT_PII,
}


def test_type_counts_rates() -> None:
    assert TypeCounts().precision == 1.0
    assert TypeCounts().recall == 1.0
    assert TypeCounts(gold=2, gold_hit=0, pred=1, pred_hit=0).f1 == 0.0
    counts = TypeCounts(gold=4, gold_hit=3, pred=6, pred_hit=3)
    assert counts.recall == 0.75
    assert counts.precision == 0.5
    assert counts.f1 == pytest.approx(0.6)


def test_exact_and_overlap_typed_scores() -> None:
    text = "Ask Jane Doe about it."
    sample = NerSample(text, (_span(text, "Jane Doe", "name"),), "prose")
    # The model finds only the first name: an overlap-typed hit, an exact miss.
    result = evaluate([sample], LABELS, _pipeline({"Jane": "PERSON"}))
    assert result.exact["PERSON"].gold == 1
    assert result.exact["PERSON"].gold_hit == 0
    assert result.exact["PERSON"].pred_hit == 0
    assert result.overlap["PERSON"].gold_hit == 1
    assert result.overlap["PERSON"].pred_hit == 1
    # "Doe" (3 of 8 gold characters) leaks.
    assert result.gold_chars == 8
    assert result.leaked_chars == 4  # " Doe"
    assert result.leak_rate == 0.5
    assert result.contexts == {"prose": 1}


def test_wrong_type_is_a_false_positive_and_a_miss() -> None:
    text = "Ask Jane Doe about it."
    sample = NerSample(text, (_span(text, "Jane Doe", "name"),))
    result = evaluate([sample], LABELS, _pipeline({"Jane Doe": "USERNAME"}))
    assert result.overlap["PERSON"].gold_hit == 0
    assert result.overlap["USERNAME"].pred == 1
    assert result.overlap["USERNAME"].pred_hit == 0
    # Type-agnostic: covered, so nothing leaks.
    assert result.leak_rate == 0.0


def test_leak_ignored_and_not_pii_classes() -> None:
    text = "tax 12-345 in Springfield, ref ABC, by Acme"
    sample = NerSample(
        text,
        (
            _span(text, "12-345", "taxnum"),  # LEAK
            _span(text, "Springfield", "city"),  # None: not scored
            _span(text, "ABC", "plain"),  # NOT_PII: not gold
            _span(text, "Acme", "orgname"),  # missing from the map
        ),
    )
    detections = {"Springfield": "ADDRESS", "ABC": "USERNAME", "12-345": "ACCOUNT_NUMBER"}
    result = evaluate([sample], LABELS, _pipeline(detections))
    # Over a LEAK or an unscored span only: neutral, never a false positive.
    assert result.neutral_detections == 2
    assert "ADDRESS" not in result.overlap
    assert "ACCOUNT_NUMBER" not in result.overlap
    # Over a NOT_PII span: a false positive and over-redaction.
    assert result.overlap["USERNAME"].pred == 1
    assert result.overlap["USERNAME"].pred_hit == 0
    assert result.over_redacted_chars == 3
    # The LEAK span is covered; gold characters are the LEAK span alone.
    assert result.gold_chars == 6
    assert result.leaked_chars == 0
    # Outside every gold span: all but the LEAK, None and unmapped spans.
    assert result.outside_chars == len(text) - 6 - 11 - 4
    assert result.unmapped == {"orgname": 1}
    assert result.over_redaction_rate == 3 / result.outside_chars


def test_structured_regression_check() -> None:
    text = "mail jane@corp.example now"
    sample = NerSample(text, (_span(text, "jane@corp.example", "mail"),))
    # A wider NER span wins overlap resolution: the EMAIL rule loses its match.
    result = evaluate([sample], LABELS, _pipeline({"mail jane@corp.example": "PERSON"}))
    assert result.structured["EMAIL"].baseline == 1
    assert result.structured["EMAIL"].kept == 0
    assert result.structured_regressions == 1
    # No NER interference: the rule keeps its exact match.
    clean = evaluate([sample], LABELS, _pipeline({}))
    assert clean.structured["EMAIL"].kept == 1
    assert clean.structured_regressions == 0
    assert clean.exact["EMAIL"].gold_hit == 1


def test_errors_carry_text_only_when_asked() -> None:
    text = "mail jane@corp.example, tax 12-345, Ask Jane Doe, or ABC"
    sample = NerSample(
        text,
        (
            _span(text, "jane@corp.example", "mail"),
            _span(text, "12-345", "taxnum"),
            _span(text, "Jane Doe", "name"),
            _span(text, "ABC", "plain"),
        ),
        "chat",
    )
    pipeline = _pipeline({"mail jane@corp.example": "PERSON", "ABC": "USERNAME"})
    errors: list[dict[str, object]] = []
    result = evaluate([sample], LABELS, pipeline, errors=errors)
    kinds = sorted({str(e["kind"]) for e in errors})
    assert kinds == ["false_positive", "leak", "miss", "regression"]
    miss = next(e for e in errors if e["kind"] == "miss" and e["type"] == "PERSON")
    assert miss["text"] == "Jane Doe"
    assert miss["context"] == "chat"
    assert str(miss["before"]).endswith("Ask ")
    # Without errors=, nothing keeps text: the result is counts only.
    assert "Jane" not in repr(evaluate([sample], LABELS, pipeline))
    assert "Jane" not in to_markdown(result)
    assert "Jane" not in json.dumps(to_json_dict(result))


def test_reports_render_counts() -> None:
    text = "Ask Jane Doe about it."
    sample = NerSample(text, (_span(text, "Jane Doe", "name"), GoldSpan(0, 3, "verb")))
    result = evaluate([sample], LABELS, _pipeline({"Jane Doe": "PERSON"}))
    markdown = to_markdown(result)
    assert "| PERSON | 1 | 1 | 1 | 1 | 1.000 | 1.000 | 1.000 |" in markdown
    assert "Labels missing from the label map (not scored): verb×1." in markdown
    data = to_json_dict(result)
    assert data["overlap"]["PERSON"]["recall"] == 1.0  # type: ignore[index]
    assert data["unmapped_labels"] == {"verb": 1}


# --- datasets -----------------------------------------------------------------


def test_resolve_and_dataset_key() -> None:
    spec, split = resolve("rules")
    assert (spec.name, split) == ("rules", "generated")
    assert dataset_key(spec, split) == "rules"
    with pytest.raises(ValueError, match="unknown dataset 'nope'"):
        resolve("nope")
    with pytest.raises(ValueError, match="has no split 'test'"):
        resolve("rules:test")


def test_rules_dataset_is_generated_and_labelled() -> None:
    spec = DATASETS["rules"]
    samples = list(spec.adapter(spec, LoadRequest(split="generated", seed=3)))
    assert samples == list(spec.adapter(spec, LoadRequest(split="generated", seed=3)))
    assert {s.context for s in samples} == {"rules", "rules-negative"}
    for sample in samples:
        for span in sample.spans:
            assert span.label in spec.label_map
            assert 0 <= span.start < span.end <= len(sample.text)


# --- the thresholds gate ------------------------------------------------------


def _result() -> NerResult:
    result = NerResult()
    result.overlap["PERSON"] = TypeCounts(gold=10, gold_hit=8, pred=9, pred_hit=8)
    result.exact["PERSON"] = TypeCounts(gold=10, gold_hit=6, pred=9, pred_hit=6)
    result.gold_chars, result.leaked_chars = 100, 20
    result.outside_chars, result.over_redacted_chars = 1000, 5
    return result


def _gate(entry: dict[str, object], result: NerResult | None = None) -> list[str]:
    thresholds = {"cfg": {"synthetic": entry}}
    return bench_ner.threshold_failures(
        thresholds, "cfg", "synthetic", result or _result(), Path("t.toml")
    )


def test_gate_passes_within_bounds() -> None:
    entry = {
        "recall": {"PERSON": 0.8},
        "exact_recall": {"PERSON": 0.6},
        "leak_max": 0.2,
        "over_redaction_max": 0.005,
        "recorded": "2026-10-05",
        "note": "fake",
    }
    assert _gate(entry) == []


def test_gate_reports_each_crossing() -> None:
    entry = {
        "recall": {"PERSON": 0.9, "ADDRESS": 0.5},
        "exact_recall": {"PERSON": 0.7},
        "leak_max": 0.1,
        "over_redaction_max": 0.001,
    }
    failures = _gate(entry)
    assert failures == [
        "recall floor for ADDRESS cannot be checked: the run holds no gold spans of ADDRESS",
        "PERSON recall 0.800 is below the floor 0.900",
        "PERSON exact_recall 0.600 is below the floor 0.700",
        "leak_max: 0.2000 is above the ceiling 0.1000",
        "over_redaction_max: 0.0050 is above the ceiling 0.0010",
    ]


def test_gate_structured_regressions() -> None:
    result = _result()
    result.structured["EMAIL"] = StructuredCounts(baseline=3, kept=1)
    assert _gate({}, result) == [
        "structured regressions: 2 gold spans the rules alone find exactly are lost with"
        " NER on (allowed 0)"
    ]
    assert _gate({"structured_regressions_max": 2}, result) == []


def test_gate_missing_entry_says_how_to_record_a_baseline() -> None:
    message = (
        "no thresholds for [cfg.synthetic] in t.toml; record a baseline from this run's"
        " report (docs/ner-bench.md, 'Recording a baseline')"
    )
    for thresholds in ({}, {"cfg": {}}, {"cfg": "x"}, {"cfg": {"synthetic": 1}}):
        failures = bench_ner.threshold_failures(
            thresholds, "cfg", "synthetic", _result(), Path("t.toml")
        )
        assert failures == [message]
    quoted = bench_ner.threshold_failures({}, "cfg", "openpii:train", _result(), Path("t.toml"))
    assert quoted[0].startswith('no thresholds for [cfg."openpii:train"]')


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"recal": {}}, r"unknown key\(s\) \['recal'\]"),
        ({"recall": 0.5}, "recall must be a table of type = floor"),
        ({"recall": {"PERSON": "high"}}, r"recall.PERSON must be a number"),
        ({"leak_max": True}, "leak_max must be a number"),
    ],
)
def test_gate_rejects_malformed_entries(entry: dict[str, object], message: str) -> None:
    with pytest.raises(bench_ner.BenchError, match=message):
        _gate(entry)


def test_load_thresholds(tmp_path: Path) -> None:
    with pytest.raises(bench_ner.BenchError, match="not found"):
        bench_ner.load_thresholds(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[x\n")
    with pytest.raises(bench_ner.BenchError, match="bad.toml"):
        bench_ner.load_thresholds(bad)
    good = tmp_path / "good.toml"
    good.write_text('[cfg.rules]\nleak_max = 0.1\n[cfg."openpii:train"]\nleak_max = 0.2\n')
    assert bench_ner.load_thresholds(good)["cfg"]["openpii:train"] == {"leak_max": 0.2}


def test_committed_thresholds_file_parses() -> None:
    root = Path(__file__).resolve().parent.parent
    assert isinstance(bench_ner.load_thresholds(root / "bench" / "ner_thresholds.toml"), dict)


# --- the dump guard -----------------------------------------------------------


def test_dump_problem_refuses_git_work_trees(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "sub").mkdir()
    assert "outside any git work tree" in str(bench_ner.dump_problem(repo / "sub" / "x.jsonl"))
    assert "outside any git work tree" in str(bench_ner.dump_problem(repo / "x.jsonl"))
    outside = tmp_path / "out"
    outside.mkdir()
    assert bench_ner.dump_problem(outside / "x.jsonl") is None


def test_write_dump_is_private(tmp_path: Path) -> None:
    target = tmp_path / "errors.jsonl"
    target.write_text("old")
    target.chmod(0o644)
    bench_ner.write_dump(target, [{"kind": "miss", "text": "Jane"}, {"kind": "leak"}])
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    lines = target.read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["miss", "leak"]
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(OSError):
        bench_ner.write_dump(link, [])


# --- the command line, end to end through build_detectors ---------------------


def _config(tmp_path: Path, *, enabled: bool = True) -> Path:
    path = tmp_path / "hf-fake.toml"
    path.write_text(
        "[detection.ner]\n"
        f"enabled = {'true' if enabled else 'false'}\n"
        'backend = "hf"\n'
        'entities = ["PERSON"]\n'
    )
    return path


def test_cli_scores_reports_and_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_transformers(monkeypatch, FakeHfPipe(findings=[], id2label={0: "O", 1: "B-PER"}))
    config = _config(tmp_path)
    thresholds = tmp_path / "thresholds.toml"
    thresholds.write_text("[hf-fake.rules]\nrecall = { EMAIL = 1.0 }\nleak_max = 0.0\n")
    out = tmp_path / "report"
    argv = ["--config", str(config), "--dataset", "rules", "--thresholds", str(thresholds)]
    assert bench_ner.main([*argv, "--limit", "0", "--out", str(out), "--check"]) == 0
    printed = capsys.readouterr().out
    assert "check passed: [hf-fake.rules]" in printed
    report = json.loads((out / "report.json").read_text())
    assert report["config"] == "hf-fake"
    spec = DATASETS["rules"]
    assert report["samples"] == len(list(spec.adapter(spec, LoadRequest(split="generated"))))
    assert report["backends"] == ["hf: dslim/bert-base-NER"]
    assert report["structured_regressions"] == 0
    assert "NER backends: hf: dslim/bert-base-NER." in (out / "report.md").read_text()

    # A floor on a type the dataset never holds fails the gate.
    thresholds.write_text("[renamed.rules]\nrecall = { PERSON = 0.5 }\n")
    assert bench_ner.main([*argv, "--limit", "50", "--name", "renamed", "--check"]) == 1
    printed = capsys.readouterr().out
    assert "# NER bench: renamed on rules" in printed
    assert "Samples: 50," in printed
    assert "CHECK FAILED: recall floor for PERSON cannot be checked" in printed


def test_cli_dumps_errors_outside_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The fake model calls "config" a person: every such span is a false
    # positive, written with its text to the dump.
    install_transformers(monkeypatch, FakeHfPipe(findings=[("config", "PER", 0.99)]))
    dump = tmp_path / "errors.jsonl"
    argv = ["--config", str(_config(tmp_path)), "--dataset", "rules", "--limit", "0"]
    argv += ["--dump-errors", str(dump)]
    assert bench_ner.main(argv) == 0
    assert "error records written to" in capsys.readouterr().out
    assert stat.S_IMODE(dump.stat().st_mode) == 0o600
    records = [json.loads(line) for line in dump.read_text().splitlines()]
    people = [r for r in records if r["type"] == "PERSON"]
    assert people and all(r["text"] == "config" for r in people)
    assert {r["kind"] for r in people} == {"false_positive"}


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "--config PATH is required"),
        (["--config", "{missing}"], "missing.toml"),
        (["--config", "{off}"], "needs [detection.ner] enabled = true"),
        (["--config", "{on}", "--dataset", "nope"], "unknown dataset 'nope'"),
        (["--config", "{on}", "--dump-errors", "{repo}"], "outside any git work tree"),
        (["--config", "{on}", "--check", "--thresholds", "{missing}"], "not found"),
    ],
)
def test_cli_input_errors_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    install_transformers(monkeypatch, FakeHfPipe(findings=[]))
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    paths = {
        "{missing}": str(tmp_path / "missing.toml"),
        "{off}": str(_config(off_dir, enabled=False)),
        "{on}": str(_config(tmp_path)),
        "{repo}": str(repo / "errors.jsonl"),
    }
    argv = [paths.get(arg, arg) for arg in extra]
    assert bench_ner.main([*argv, "--limit", "5"]) == 2
    assert message in capsys.readouterr().err


def test_cli_build_errors_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The hf extra is not installed: the builder's ConfigError is reported.
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert bench_ner.main(["--config", str(_config(tmp_path))]) == 2
    assert "hf extra is not installed" in capsys.readouterr().err


def test_cli_real_data_dump_needs_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_transformers(monkeypatch, FakeHfPipe(findings=[]))
    real = replace(DATASETS["rules"], name="real", real_data=True)
    monkeypatch.setattr("llm_redact.bench.datasets.DATASETS", {**DATASETS, "real": real})
    dump = tmp_path / "errors.jsonl"
    argv = ["--config", str(_config(tmp_path)), "--dataset", "real", "--limit", "5"]
    assert bench_ner.main([*argv, "--dump-errors", str(dump)]) == 2
    assert "holds real data" in capsys.readouterr().err
    assert not dump.exists()
    assert bench_ner.main([*argv, "--dump-errors", str(dump), "--allow-real-data-dump"]) == 0
    printed = capsys.readouterr().out
    assert "This dataset holds real data" in printed
    assert dump.exists()


def test_cli_lists_datasets(capsys: pytest.CaptureFixture[str]) -> None:
    assert bench_ner.main(["--list-datasets"]) == 0
    printed = capsys.readouterr().out
    assert "rules: the regex bench's generated positives" in printed
    assert "license: generated at run time" in printed
    real = bench_ner.list_datasets({"real": replace(DATASETS["rules"], real_data=True)})
    assert "REAL DATA" in real
