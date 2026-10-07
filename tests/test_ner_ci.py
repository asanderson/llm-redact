"""The CI wiring of the real NER models (docs/ner-bench.md, "CI"): the bench
configurations, the models the ner-models job pulls for the real_model
tests, the recorded baselines, the workflows' shape, and the switch that
turns a skipped real_model test into a failure in that job."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

import real_models
from llm_redact.config import load_config
from llm_redact.detection.model_sources import hub_sources

ROOT = Path(__file__).resolve().parent.parent
BENCH_CONFIGS = sorted((ROOT / "bench" / "configs").glob("*.toml"))
TEST_CONFIGS = sorted((ROOT / "tests" / "real_model_configs").glob("*.toml"))
# Measured by hand, never by CI: a model too slow for a CI runner
# (docs/ner-bench.md, "Configurations measured by hand").
MANUAL_CONFIGS = sorted((ROOT / "bench" / "configs" / "manual").glob("*.toml"))
WORKFLOWS = ROOT / ".github" / "workflows"


def _workflow(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((WORKFLOWS / name).read_text())
    return loaded


def _pulled() -> set[tuple[str, str]]:
    """(model id, revision) of every model and GLiNER base model the CI
    ner-models job pulls (`models pull --config` on each config)."""
    pulled: set[tuple[str, str]] = set()
    for path in BENCH_CONFIGS + TEST_CONFIGS:
        for source in hub_sources(load_config(path).detection.ner):
            assert source.model_id is not None and source.revision is not None, path.name
            pulled.add((source.model_id, source.revision))
            entry = source.entry
            # A pinned base model is assembled in (self-contained checkpoints
            # name a backbone they do not fetch: no pin).
            if entry is not None and entry.backbone is not None and entry.backbone_revision:
                pulled.add((entry.backbone, entry.backbone_revision))
    return pulled


# --- the configurations --------------------------------------------------------


DEFAULT_CONFIGS = ("gliner-default", "hf-default")


def _bench_config_problems(paths: list[Path]) -> list[str]:
    """The bench scores the two default models, each from a "<backend>-default"
    config naming no model ("default" means the backend's default: a config
    naming one would keep scoring it after the default moved). Any other
    config (docs/CONTRIBUTING.md, "Adding an NER model or backend") may name
    its model."""
    stems = {path.stem for path in paths}
    problems = [f"{name}.toml is missing" for name in DEFAULT_CONFIGS if name not in stems]
    for path in paths:
        if path.stem.endswith("-default"):
            ner = load_config(path).detection.ner
            if ner.model is not None or ner.models != ():
                problems.append(f"{path.name} names a model")
    return problems


def test_the_bench_scores_the_two_default_models() -> None:
    assert _bench_config_problems(BENCH_CONFIGS) == []
    assert {"gliner-pii-edge-onnx", "gliner2-default", "hf-openai-privacy-filter"} <= {
        p.stem for p in TEST_CONFIGS
    }


def test_a_bench_config_for_another_model_may_name_it(tmp_path: Path) -> None:
    # What CONTRIBUTING's checklist step 3 adds for a non-default model.
    for path in BENCH_CONFIGS:
        (tmp_path / path.name).write_text(path.read_text())
    added = tmp_path / "gliner-pii-edge.toml"
    added.write_text(
        '[detection.ner]\nenabled = true\nbackend = "gliner"\n'
        'model = "knowledgator/gliner-pii-edge-v1.0"\n'
    )
    assert _bench_config_problems(sorted(tmp_path.glob("*.toml"))) == []
    # A default config naming a model, or a default missing, is refused.
    (tmp_path / "hf-default.toml").write_text(
        '[detection.ner]\nenabled = true\nbackend = "hf"\nmodel = "dslim/bert-base-NER"\n'
    )
    (tmp_path / "gliner-default.toml").unlink()
    assert _bench_config_problems(sorted(tmp_path.glob("*.toml"))) == [
        "gliner-default.toml is missing",
        "hf-default.toml names a model",
    ]


@pytest.mark.parametrize(
    "path", BENCH_CONFIGS + TEST_CONFIGS + MANUAL_CONFIGS, ids=lambda p: p.stem
)
def test_each_config_loads_pinned_models_with_downloads_off(path: Path) -> None:
    ner = load_config(path).detection.ner
    assert ner.enabled
    assert not ner.allow_download  # the models are pulled first, then loaded offline
    sources = hub_sources(ner)
    assert sources
    for source in sources:
        assert source.pinned_by == "catalog"
        assert source.entry is not None and source.entry.revision == source.revision


def test_ci_pulls_every_model_the_real_model_tests_load() -> None:
    import test_gliner
    import test_gliner2
    import test_hf_bioes
    import test_hf_text_words
    import test_hf_windows

    needed = {
        (test_hf_windows.DSLIM, test_hf_windows.DSLIM_REVISION),
        (test_hf_bioes.DSLIM, test_hf_bioes.DSLIM_REVISION),
        (test_hf_bioes.PRIVACY, test_hf_bioes.PRIVACY_REVISION),
        (test_hf_windows.OPENMED, test_hf_windows.OPENMED_REVISION),
        (test_hf_text_words.ETTIN, test_hf_text_words.ETTIN_REVISION),
        (test_gliner.GLINER_SMALL, test_gliner.GLINER_SMALL_REVISION),
        (test_gliner.DEBERTA_SMALL, test_gliner.DEBERTA_SMALL_REVISION),
        (test_gliner.EDGE, test_gliner.EDGE_REVISION),
        (test_gliner2.BASE, test_gliner2.BASE_PIN),
        (test_gliner2.PII, test_gliner2.PII_PIN),
    }
    assert needed <= _pulled()


# --- the recorded baselines ----------------------------------------------------


@pytest.mark.parametrize("path", BENCH_CONFIGS, ids=lambda p: p.stem)
def test_each_bench_config_has_recorded_baselines(path: Path) -> None:
    thresholds = tomllib.loads((ROOT / "bench" / "ner_thresholds.toml").read_text())
    ceilings = tomllib.loads((ROOT / "bench" / "ner_ceilings.toml").read_text())
    name = path.stem
    for entry in (thresholds[name]["synthetic"], ceilings[name]):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry["recorded"])
        # The note names the model and the revision it was measured with.
        for source in hub_sources(load_config(path).detection.ner):
            assert source.model_id in entry["note"] and str(source.revision) in entry["note"]
    assert thresholds[name]["synthetic"]["recall"]["PERSON"] > 0
    # The type-agnostic leak rate also counts the corpus's types the config
    # does not request (always leaked); each requested type has its own
    # leak ceiling, or its leakage could grow severalfold unnoticed.
    requested = load_config(path).detection.ner.entities
    assert set(requested) <= set(thresholds[name]["synthetic"]["type_leak_max"])
    assert "per_100kb_max" in ceilings[name]


@pytest.mark.parametrize("path", MANUAL_CONFIGS, ids=lambda p: p.stem)
def test_a_manual_config_has_recorded_baselines_and_says_why(path: Path) -> None:
    thresholds = tomllib.loads((ROOT / "bench" / "ner_thresholds.toml").read_text())
    entry = thresholds[path.stem]["synthetic"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry["recorded"])
    for source in hub_sources(load_config(path).detection.ner):
        assert source.model_id in entry["note"] and str(source.revision) in entry["note"]
    requested = load_config(path).detection.ner.entities
    assert set(requested) <= set(entry["type_leak_max"])
    ceilings = tomllib.loads((ROOT / "bench" / "ner_ceilings.toml").read_text())
    assert "per_100kb_max" in ceilings[path.stem]
    # Its comment says why CI does not run it.
    assert "by hand" in path.read_text()


def test_a_manual_model_gets_a_smoke_test_in_ci_but_no_bench() -> None:
    # openai/privacy-filter is too slow for the bench on a runner, but the
    # ner-models job pulls it for one real_model smoke test: a config of
    # tests/real_model_configs/ (pulled, never scored) asking for what the
    # manual bench config asks for, and the test reads that config.
    import test_hf_bioes

    manual = load_config(ROOT / "bench/configs/manual/hf-openai-privacy-filter.toml")
    smoke = load_config(test_hf_bioes.PRIVACY_CONFIG)
    assert test_hf_bioes.PRIVACY_CONFIG.resolve() in TEST_CONFIGS
    assert smoke.detection.ner == manual.detection.ner
    (source,) = hub_sources(smoke.detection.ner)
    assert (source.model_id, source.revision) == (
        test_hf_bioes.PRIVACY,
        test_hf_bioes.PRIVACY_REVISION,
    )


def _step_index(steps: list[dict[str, Any]], *, run: str = "", name: str = "") -> int:
    (index,) = [
        i
        for i, step in enumerate(steps)
        if (run and step.get("run", "").strip() == run) or (name and step.get("name") == name)
    ]
    return index


def test_the_ner_models_job_never_saves_openai_privacy_filter() -> None:
    # 2.8 GB pulled on every run instead of saved with the model cache (the
    # actions cache keeps 10 GB per repository, least recently used out):
    # removed right after the real_model tests, whatever their outcome, so
    # the bench has the disk space and the cache saved at the end of the
    # job never holds it.
    steps = _workflow("ci.yml")["jobs"]["ner-models"]["steps"]
    tests = _step_index(
        steps, run='uv run --no-sync pytest -m real_model -v --deselect "$PRIVACY_FILTER_TEST"'
    )
    smoke = _step_index(steps, name="openai/privacy-filter smoke test")
    drop = _step_index(steps, name="drop openai/privacy-filter from the model cache")
    bench = _step_index(steps, name="NER bench gates")
    assert tests < smoke < drop < bench
    assert steps[drop]["if"] == "always()"
    assert "rm -rf ~/.cache/huggingface/hub/models--openai--privacy-filter\n" in steps[drop]["run"]
    (models_cache,) = [
        s for s in steps if s.get("with", {}).get("path") == "~/.cache/huggingface/hub"
    ]
    assert "actions/cache@" in models_cache["uses"]  # saved at the end of the job


def test_the_privacy_filter_smoke_test_runs_in_a_process_of_its_own() -> None:
    # In float32 the model alone takes about 6 GB: the first real_model run
    # leaves it out, and a second process runs exactly that one test.
    import test_hf_bioes

    steps = _workflow("ci.yml")["jobs"]["ner-models"]["steps"]
    tests = steps[
        _step_index(
            steps, run='uv run --no-sync pytest -m real_model -v --deselect "$PRIVACY_FILTER_TEST"'
        )
    ]
    smoke = steps[_step_index(steps, name="openai/privacy-filter smoke test")]
    node = "tests/test_hf_bioes.py::" + (
        test_hf_bioes.test_real_privacy_filter_finds_whole_spans_with_its_calibration.__name__
    )
    assert tests["env"]["PRIVACY_FILTER_TEST"] == smoke["env"]["PRIVACY_FILTER_TEST"] == node
    assert smoke["run"] == 'uv run --no-sync pytest -m real_model -v "$PRIVACY_FILTER_TEST"'


def test_the_ner_models_job_frees_disk_first() -> None:
    # The CPU-torch environment, the restored model cache and the 2.8 GB
    # openai/privacy-filter come close to a hosted runner's free disk: the
    # preinstalled toolchains the job never uses go first.
    job = _workflow("ci.yml")["jobs"]["ner-models"]
    first = job["steps"][0]
    assert first["name"] == "free disk space"
    assert "sudo rm -rf /usr/share/dotnet /usr/local/lib/android" in first["run"]
    assert first["run"].count("df -h /") == 2
    assert job["timeout-minutes"] == 180


def test_manual_configs_stay_out_of_the_ci_jobs() -> None:
    assert MANUAL_CONFIGS
    assert not {p.name for p in MANUAL_CONFIGS} & {p.name for p in BENCH_CONFIGS}
    job = _workflow("ci.yml")["jobs"]["ner-models"]
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "bench/configs/*.toml" in runs  # not recursive: manual/ is left out
    assert "manual" not in runs and "**" not in runs
    # The weekly eval lists its configurations (below): none of them manual.
    assert not set(_eval_matrix()) & {p.stem for p in MANUAL_CONFIGS}


# --- the workflows -------------------------------------------------------------


def test_the_ner_models_job() -> None:
    job = _workflow("ci.yml")["jobs"]["ner-models"]
    assert job["env"][real_models.REQUIRED_ENV] == "1"
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "bench/configs/*.toml tests/real_model_configs/*.toml" in runs
    assert "llm-redact models pull --config" in runs
    assert "pytest -m real_model" in runs
    for mode in (
        "--check --out",
        "--fp-corpus bench/fp_corpus --check",
        "--latency --many-small-strings 1000 --check",
    ):
        assert mode in runs
    # The hash-checked environment (below), never the CUDA wheel `uv sync`
    # would install.
    assert "scripts/ner_ci_env.sh" in runs
    assert "uv sync" not in runs and "uv pip install" not in runs


def test_the_docs_name_when_the_ner_models_job_runs() -> None:
    triggers = _workflow("ci.yml")[True]  # YAML 1.1 reads the key "on" as true
    assert triggers["push"] == {"branches": ["main"]}
    assert set(triggers) == {"push", "pull_request", "schedule"}
    docs = " ".join((ROOT / "docs" / "ner-bench.md").read_text().split())
    assert (
        "(a job of `.github/workflows/ci.yml`, on pull requests, pushes to `main` and the"
        " weekly CI schedule)" in docs
    )


def _eval_matrix() -> list[str]:
    configs: list[str] = _workflow("ner-eval.yml")["jobs"]["ner-eval"]["strategy"]["matrix"][
        "config"
    ]
    return configs


def test_the_weekly_eval_reports_both_datasets() -> None:
    workflow = _workflow("ner-eval.yml")
    triggers = workflow[True]  # YAML 1.1 reads the key "on" as true
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["ner-eval"]
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "for dataset in openpii nemotron" in runs
    assert "--limit 2000" in runs
    assert "--check" not in runs  # report only until baselines are recorded
    # One job per configuration, pulling and evaluating only its own.
    assert job["env"]["CONFIG"] == "bench/configs/${{ matrix.config }}.toml"
    assert 'models pull --config "$CONFIG"' in runs
    assert '--config "$CONFIG" --dataset "$dataset"' in runs
    assert "bench/configs/*.toml" not in runs
    assert job["strategy"]["fail-fast"] is False
    upload = [s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert [s["with"]["name"] for s in upload] == ["ner-eval-${{ matrix.config }}"]


def test_the_weekly_eval_runs_every_bench_config_in_its_own_job() -> None:
    # A serial loop over every configuration ran past the job's timeout
    # once the bench grew past two; each job holds one configuration (two
    # dataset slices), within a GitHub-hosted job's 360-minute cap.
    matrix = _eval_matrix()
    assert len(set(matrix)) == len(matrix)
    assert set(matrix) == {p.stem for p in BENCH_CONFIGS}
    assert 0 < _workflow("ner-eval.yml")["jobs"]["ner-eval"]["timeout-minutes"] <= 360


def test_the_weekly_eval_never_saves_the_shared_model_cache() -> None:
    # A job pulls only its own model; saving under the key ner-models
    # restores would hand that job one model and never save the full set.
    steps = _workflow("ner-eval.yml")["jobs"]["ner-eval"]["steps"]
    hub = [s for s in steps if s.get("with", {}).get("path") == "~/.cache/huggingface/hub"]
    assert [s["uses"].split("@")[0] for s in hub] == ["actions/cache/restore"]
    (models_cache,) = [
        s
        for s in _workflow("ci.yml")["jobs"]["ner-models"]["steps"]
        if s.get("with", {}).get("path") == "~/.cache/huggingface/hub"
    ]
    assert hub[0]["with"]["key"] == models_cache["with"]["key"]


def test_the_weekly_eval_installs_the_same_environment() -> None:
    steps = _workflow("ner-eval.yml")["jobs"]["ner-eval"]["steps"]
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "scripts/ner_ci_env.sh" in runs
    assert "uv sync" not in runs and "uv pip install" not in runs


# --- the environment script: every wheel hash-checked ---------------------------

SCRIPTS = ROOT / "scripts"
ENV_SCRIPT = SCRIPTS / "ner_ci_env.sh"


def _commands(text: str) -> list[str]:
    """The script's commands, comments dropped and backslash continuations
    joined."""
    joined = re.sub(r"\\\n\s*", " ", text)
    return [
        line.strip()
        for line in joined.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_the_cpu_torch_recipe_is_the_only_one() -> None:
    # One CPU-torch recipe: scripts/cpu_torch.py (its SHA-256 table pinned to
    # uv.lock's torch by tests/test_cpu_torch_script.py). No second pin file.
    assert not (SCRIPTS / "ner_ci_torch_cpu.txt").exists()
    assert "ner_ci_torch_cpu" not in ENV_SCRIPT.read_text()


def test_the_environment_script_checks_every_hash() -> None:
    commands = _commands(ENV_SCRIPT.read_text())
    # The export keeps the lock's hashes and goes through cpu_torch.py,
    # which takes torch and its GPU-only packages out whole and pins the
    # locked torch's CPU wheels by SHA-256 (never a line filter here).
    (export,) = [c for c in commands if c.startswith("uv export")]
    assert "--frozen" in export and "--no-hashes" not in export
    assert '| python3 "$here/cpu_torch.py" requirements "$work/requirements.txt"' in export
    assert export.endswith('>"$work/torch.txt"')
    assert all("grep" not in c and "sed " not in c for c in commands)
    # Both wheel installs require a hash for every requirement and resolve
    # nothing beyond the files (the export is the whole locked closure).
    installs = [c for c in commands if c.startswith("uv pip install")]
    torch, requirements, project = installs
    assert '--require-hashes --no-deps -r "$work/torch.txt"' in torch
    assert torch.endswith("--index-url https://download.pytorch.org/whl/cpu")
    assert '--require-hashes --no-deps -r "$work/requirements.txt"' in requirements
    assert "--index-url" not in requirements
    assert project.endswith("--no-deps -e .")
    # Then: no CUDA or triton distribution and a +cpu torch, and a
    # consistent environment.
    assert commands[-2] == '"$venv/bin/python" "$here/cpu_torch.py" check'
    assert commands[-1].startswith("uv pip check")
    # The venv is the Python the CI jobs run.
    assert any(c.startswith("uv venv --python 3.13") for c in commands)
    assert commands[0] == "set -euo pipefail"


def test_the_docs_scope_local_reproduction_to_what_the_script_runs_on() -> None:
    # The script runs the venv's POSIX interpreter (`$venv/bin/python`; a
    # Windows venv has Scripts/python.exe) and pipes through python3, so
    # docs/ner-bench.md must not promise a local run on Windows.
    assert '"$venv/bin/python"' in ENV_SCRIPT.read_text()
    text = (ROOT / "docs" / "ner-bench.md").read_text()
    start = text.index("To reproduce the job locally")
    paragraph = " ".join(text[start : text.index("```", start)].split())
    assert "CPython 3.13 venv; Linux, x86_64 or aarch64" in paragraph
    assert "does not run on Windows" in paragraph
    assert "Linux or Windows" not in paragraph


@pytest.mark.parametrize("name", ["ci.yml", "ner-eval.yml"])
def test_actions_are_pinned_and_checkouts_keep_no_credentials(name: str) -> None:
    for job in _workflow(name)["jobs"].values():
        for step in job["steps"]:
            uses = step.get("uses")
            if uses is None:
                continue
            assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", uses), uses
            if uses.startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False


def test_claude_md_states_the_ci_invariants() -> None:
    # The repository's agent instructions name what this file pins.
    text = (ROOT / "CLAUDE.md").read_text()
    for needle in (
        real_models.REQUIRED_ENV,
        "`ner-models`",
        "`ner-eval.yml`",
        "scripts/ner_ci_env.sh",
        "tests/real_model_configs/*.toml",
        "BEFORE `build_pipeline`",
        "`type_leak_max`",
    ):
        assert needle in text, needle


# --- a skipped real_model test fails where the models are required ---------------


def _skipped(reason: str = "Skipped: dslim/bert-base-NER is not cached") -> pytest.TestReport:
    return pytest.TestReport(
        nodeid="tests/test_x.py::test_y",
        location=("tests/test_x.py", 1, "test_y"),
        keywords={},
        outcome="skipped",
        longrepr=("tests/test_x.py", 1, reason),
        when="call",
    )


class _Item:
    def __init__(self, marked: bool) -> None:
        self.marked = marked

    def get_closest_marker(self, name: str) -> object | None:
        return object() if self.marked and name == "real_model" else None


def test_a_required_real_model_skip_fails() -> None:
    report = _skipped()
    real_models.fail_required_skips(_Item(True), report, {real_models.REQUIRED_ENV: "1"})  # type: ignore[arg-type]
    assert report.outcome == "failed"
    assert report.longrepr == (
        "LLM_REDACT_TEST_REAL_MODELS_REQUIRED=1 but this real_model test skipped:"
        " Skipped: dslim/bert-base-NER is not cached"
    )


@pytest.mark.parametrize(
    ("marked", "environ"),
    [
        (False, {real_models.REQUIRED_ENV: "1"}),  # not a real_model test
        (True, {}),  # not required (a local run)
        (True, {real_models.REQUIRED_ENV: "0"}),
    ],
)
def test_other_skips_stay_skips(marked: bool, environ: dict[str, str]) -> None:
    report = _skipped()
    real_models.fail_required_skips(_Item(marked), report, environ)  # type: ignore[arg-type]
    assert report.outcome == "skipped"


def test_a_passed_test_is_untouched() -> None:
    report = _skipped()
    report.outcome = "passed"
    report.longrepr = None
    assert real_models.required_skip_failure(report, True, {real_models.REQUIRED_ENV: "1"}) is None


def test_a_skip_without_a_location_names_its_reason() -> None:
    report = _skipped()
    report.longrepr = "plain reason"  # type: ignore[assignment]
    message = real_models.required_skip_failure(report, True, {real_models.REQUIRED_ENV: "1"})
    assert message is not None and message.endswith("skipped: plain reason")
