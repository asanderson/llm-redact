import random
from pathlib import Path

import pytest

from llm_redact.bench.corpus import (
    CONTEXTUAL_GENERATORS,
    PUNCTUATION_CONTEXTS,
    VALUE_GENERATORS,
    LabeledSpan,
    Sample,
    generate,
    punctuation_samples,
    taken_after,
)
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


def test_every_rule_is_put_in_every_punctuation_context() -> None:
    # The recall gate holds every rule to a value ending a sentence, before a
    # colon, in brackets or quotes: one value of each rule per context.
    rules = len(VALUE_GENERATORS) + len(CONTEXTUAL_GENERATORS)
    samples = punctuation_samples(random.Random(3))
    assert len(samples) == rules * len(PUNCTUATION_CONTEXTS)
    for index, sample in enumerate(samples):
        before, after = PUNCTUATION_CONTEXTS[index // rules].split("{v}")
        assert sample.text.startswith(before) and sample.text.endswith(after)
    corpus = {sample.text for sample in generate(seed=3, samples_per_rule=1)}
    for context in PUNCTUATION_CONTEXTS:
        before, after = context.split("{v}")
        found = [t for t in corpus if t.startswith(before) and t.endswith(after)]
        assert len(found) >= rules, context


def test_the_corpus_reads_what_a_value_takes_in_as_its_rule_does() -> None:
    # The corpus's model of the characters right after a value that its rule
    # takes in (taken_after), pinned to the production rules: a generic
    # secret's value takes every character of its alphabet, a PlanetScale
    # token the dots after it; every other value none.
    from llm_redact.detection.regex_rules import BUILTIN_RULES, RegexDetector

    rules = {rule.name: RegexDetector(rule) for rule in BUILTIN_RULES}
    secret = "Xk29QmPl40Vt85ZwQqR7"
    afters = (
        ".", "...", '."', ".)", ". \nnext", "... and more", ", then", ".-x", ".x", "", " .",
        "\u017f ok", "\u212a ok", "...\u00e9 ok", ".**", ".\u201c", "..<br>",
    )  # fmt: skip
    for after in afters:
        (found,) = rules["generic_secret"].detect(f"password = {secret}{after}")
        assert found.end - found.start - len(secret) == taken_after("SECRET", secret, after)
    # Long runs of dots, alone or before a character that is no word's,
    # and of the token alphabet holding another token's start.
    runs = (
        "." * 40 + " ok", "." * 40 + "x", "." * 40 + "/",
        ".pscale_tkn_" + "b" * 30 + "....", "-pscale_pw_" + "c" * 40 + ".",
    )  # fmt: skip
    for after in afters + runs:
        for body in (30, 40, 60, 63, 64):
            token = "pscale_tkn_" + ("a1B2c3D4e5" * 7)[:body]
            first = [d for d in rules["planetscale_token"].detect(f"see {token}{after}")]
            if first and first[0].start == 4:
                taken = first[0].end - 4 - len(token)
                assert taken == taken_after("PLANETSCALE_TOKEN", token, after), (body, after)
            else:
                with pytest.raises(ValueError):
                    taken_after("PLANETSCALE_TOKEN", token, after)
    assert taken_after("IPV6", "2001:db8::1", "...") == 0


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


def test_fp_corpus_readme_lists_every_file() -> None:
    # Provenance is part of the corpus's contract: every vendored or
    # authored file has a row in the README's contents table.
    root = Path(__file__).resolve().parent.parent / "bench" / "fp_corpus"
    readme = (root / "README.md").read_text()
    unlisted = [
        p.name
        for p in sorted(root.iterdir())
        if p.name not in ("README.md", "MANIFEST.toml") and f"| `{p.name}` |" not in readme
    ]
    assert unlisted == []
