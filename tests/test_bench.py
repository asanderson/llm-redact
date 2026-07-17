from pathlib import Path

from llm_redact.bench.corpus import LabeledSpan, Sample, generate
from llm_redact.bench.fp_scan import fp_failures, scan_fp_corpus, to_markdown_table
from llm_redact.bench.metrics import evaluate, to_json_dict, to_markdown


def test_corpus_is_deterministic() -> None:
    a = generate(seed=7, samples_per_rule=3)
    b = generate(seed=7, samples_per_rule=3)
    assert a == b
    c = generate(seed=8, samples_per_rule=3)
    assert a != c


def test_corpus_spans_are_correct() -> None:
    for sample in generate(seed=1, samples_per_rule=2):
        for span in sample.spans:
            assert 0 <= span.start < span.end <= len(sample.text)


def test_metrics_math() -> None:
    # One perfect hit, one type mismatch (counts as FP+FN), one miss.
    corpus = [
        Sample("mail a@b.example ok", (LabeledSpan(5, 16, "EMAIL"),)),
        Sample("mail c@d.example ok", (LabeledSpan(5, 16, "IPV4"),)),  # wrong label
        Sample("nothing here", (LabeledSpan(0, 7, "EMAIL"),)),  # undetectable
    ]
    result = evaluate(corpus)
    assert result.per_type["EMAIL"].tp == 1
    assert result.per_type["EMAIL"].fp == 1  # the mislabeled sample's detection
    assert result.per_type["EMAIL"].fn == 1
    assert result.per_type["IPV4"].fn == 1
    assert 0 < result.overall.f1 < 1


def test_end_to_end_bench_is_clean() -> None:
    result = evaluate(generate(seed=42, samples_per_rule=5))
    assert result.overall.recall == 1.0
    assert result.overall.precision == 1.0
    # Report renderers don't crash and carry the headline number.
    assert "recall 1.000" in to_markdown(result, seed=42)
    assert to_json_dict(result, seed=42)["overall"]["recall"] == 1.0  # type: ignore[index]


def test_fp_corpus_matches_manifest() -> None:
    # The real committed corpus: this IS the precision gate, run in-suite so
    # a rule change that breaks it fails fast, not only in the bench CI job.
    root = Path(__file__).resolve().parent.parent / "bench" / "fp_corpus"
    failures = fp_failures(scan_fp_corpus(root))
    assert failures == []


def test_fp_gate_catches_drift_both_ways(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.toml").write_text(
        '[files."has_email.txt"]\n\n[files."expects_ip.txt"]\nIPV4 = 1\n\n[files."gone.txt"]\n'
    )
    (tmp_path / "has_email.txt").write_text("contact jane@corp.example\n")  # unexpected hit
    (tmp_path / "expects_ip.txt").write_text("no address here\n")  # expected hit vanished
    (tmp_path / "unlisted.txt").write_text("plain text\n")  # not in manifest

    failures = fp_failures(scan_fp_corpus(tmp_path))
    flat = "\n".join(failures)
    assert "has_email.txt: EMAIL expected 0, found 1 (lines 1)" in flat
    assert "expects_ip.txt: IPV4 expected 1, found 0" in flat
    assert "gone.txt: listed in MANIFEST.toml but missing on disk" in flat
    assert "unlisted.txt: on disk but not listed in MANIFEST.toml" in flat
    assert "jane@corp.example" not in flat  # never the matched text


def test_fp_scan_never_carries_matched_text(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.toml").write_text('[files."f.txt"]\n')
    (tmp_path / "f.txt").write_text("key AKIAIOSFODNN7EXAMPLE\n")
    results = scan_fp_corpus(tmp_path)
    assert "AKIAIOSFODNN7EXAMPLE" not in repr(results)
    assert "AKIAIOSFODNN7EXAMPLE" not in to_markdown_table(results)
