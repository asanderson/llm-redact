"""The synthetic NER corpus (bench/ner_corpus.py): deterministic, exactly
labelled, every context present, and the NER bench's default dataset."""

import json
import random
import re
from pathlib import Path

import pytest

from llm_redact.bench import ner as bench_ner
from llm_redact.bench import ner_corpus
from llm_redact.bench.datasets import DATASETS, LoadRequest, resolve
from llm_redact.bench.ner_metrics import Pipeline, evaluate
from llm_redact.detection.engine import DetectionConfig, build_detectors
from ner_fakes import FakeHfPipe, install_transformers

_SHAPES = {
    "PERSON": re.compile(
        rf"(?:{'|'.join(ner_corpus.FIRST_NAMES)}) (?:{'|'.join(ner_corpus.LAST_NAMES)})\Z"
    ),
    "ADDRESS": re.compile(
        rf"\d{{1,4}} (?:{'|'.join(ner_corpus.STREETS)})"
        rf" (?:{'|'.join(ner_corpus.STREET_SUFFIXES)})\Z"
    ),
    "DATE_OF_BIRTH": re.compile(
        r"(?:[A-Z][a-z]+ \d{1,2}, \d{4}|\d{4}-\d\d-\d\d"
        r"|\d\d/\d\d/\d{4}|\d{1,2} [A-Z][a-z]+ \d{4})\Z"
    ),
    "USERNAME": re.compile(r"[a-z_]+[a-z0-9._]*\Z"),
    "ACCOUNT_NUMBER": re.compile(r"\d{8,12}\Z"),
    "EMAIL": re.compile(r"[a-z]+\.[a-z]+@[a-z.]+\Z"),
    "PHONE": re.compile(r"[+(]?\d[\d ().-]+\Z"),
}


def test_same_seed_same_corpus() -> None:
    assert ner_corpus.generate(seed=5, samples=60) == ner_corpus.generate(seed=5, samples=60)
    assert ner_corpus.generate(seed=5, samples=60) != ner_corpus.generate(seed=6, samples=60)


def test_every_gold_span_is_its_value() -> None:
    corpus = ner_corpus.generate(seed=11)
    labels = set()
    for sample in corpus:
        for span in sample.spans:
            value = sample.text[span.start : span.end]
            assert value and value == value.strip()
            assert _SHAPES[span.label].match(value), span.label
            labels.add(span.label)
        if sample.context.startswith("neg-"):
            assert sample.spans == ()
        else:
            assert sample.spans
    assert labels == set(ner_corpus.LABELS)


def test_fill_records_offsets_while_filling() -> None:
    sample = ner_corpus.fill("a ${PERSON} b ${n} c ${ADDRESS}", random.Random(1))
    person, addr = sample.spans
    assert sample.text.startswith("a ")
    assert sample.text[person.start : person.end] == sample.text.split(" b ")[0][2:]
    assert sample.text[addr.end :] == ""
    assert (person.label, addr.label) == ("PERSON", "ADDRESS")


def test_every_context_appears() -> None:
    contexts = {s.context for s in ner_corpus.generate(seed=1, samples=len(ner_corpus.CONTEXTS))}
    assert contexts == set(ner_corpus.CONTEXTS)
    assert {"prose", "chat", "json", "code", "log", "git"} <= contexts
    negatives = {"neg-uuid", "neg-hash", "neg-identifier", "neg-path", "neg-traceback"}
    assert negatives | {"neg-timestamp"} <= contexts


def test_the_rules_find_only_the_structured_values() -> None:
    # The regex rules alone find every EMAIL and PHONE exactly and nothing
    # else: the hard negatives stay clean, and the structured-regression
    # check has a baseline to compare against.
    rules = build_detectors(DetectionConfig())
    spec = DATASETS["synthetic"]
    result = evaluate(
        ner_corpus.generate(seed=42), spec.label_map, Pipeline.from_detectors(rules, rules)
    )
    assert result.exact["EMAIL"].recall == 1.0
    assert result.exact["PHONE"].recall == 1.0
    assert result.exact["EMAIL"].precision == 1.0
    assert result.exact["PHONE"].precision == 1.0
    assert result.over_redacted_chars == 0
    assert set(result.structured) == {"EMAIL", "PHONE"}
    assert result.overlap["PERSON"].gold_hit == 0


def test_synthetic_is_the_default_dataset() -> None:
    args = bench_ner._parser().parse_args([])
    spec, split = resolve(args.dataset)
    assert spec is DATASETS["synthetic"]
    samples = list(spec.adapter(spec, LoadRequest(split=split, seed=42)))
    assert samples == ner_corpus.generate(seed=42)
    assert len(samples) == ner_corpus.SAMPLES


def test_cli_scores_a_model_on_the_synthetic_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A fake hf model that tags every first and last name of the corpus as
    # PER: the parts merge into one PERSON span (labels.merge_adjacent_parts)
    # and fold from PER through the label policy, as with a real model.
    findings = [(name, "PER", 0.9) for name in (*ner_corpus.FIRST_NAMES, *ner_corpus.LAST_NAMES)]
    install_transformers(monkeypatch, FakeHfPipe(findings=findings))
    config = tmp_path / "fake-person.toml"
    config.write_text('[detection.ner]\nenabled = true\nbackend = "hf"\nentities = ["PERSON"]\n')
    thresholds = tmp_path / "t.toml"
    thresholds.write_text(
        "[fake-person.synthetic]\n"
        "recall = { PERSON = 0.95, EMAIL = 1.0 }\n"
        "exact_recall = { PERSON = 0.95 }\n"
        "over_redaction_max = 0.0\n"
    )
    argv = ["--config", str(config), "--thresholds", str(thresholds), "--out", str(tmp_path / "r")]
    assert bench_ner.main([*argv, "--check"]) == 0
    assert "check passed: [fake-person.synthetic]" in capsys.readouterr().out
    report = json.loads((tmp_path / "r" / "report.json").read_text())
    assert report["dataset"] == "synthetic"
    assert report["exact"]["PERSON"]["recall"] == 1.0
    assert report["overlap"]["ADDRESS"]["recall"] == 0.0
    assert report["structured_regressions"] == 0
    assert report["leak_rate"] > 0  # addresses, dates, usernames, accounts
