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


def test_the_bench_scores_the_two_default_models() -> None:
    assert [p.stem for p in BENCH_CONFIGS] == ["gliner-default", "hf-default"]
    assert [p.stem for p in TEST_CONFIGS] == ["gliner-pii-edge-onnx", "gliner2-default"]


@pytest.mark.parametrize("path", BENCH_CONFIGS + TEST_CONFIGS, ids=lambda p: p.stem)
def test_each_config_loads_pinned_models_with_downloads_off(path: Path) -> None:
    ner = load_config(path).detection.ner
    assert ner.enabled
    assert not ner.allow_download  # the models are pulled first, then loaded offline
    sources = hub_sources(ner)
    assert sources
    for source in sources:
        assert source.pinned_by == "catalog"
        assert source.entry is not None and source.entry.revision == source.revision


def test_the_default_configs_name_no_model() -> None:
    # "default" means the backend's default model: a config naming one would
    # keep scoring it after the default moved.
    for path in BENCH_CONFIGS:
        ner = load_config(path).detection.ner
        assert ner.model is None and ner.models == (), path.name


def test_ci_pulls_every_model_the_real_model_tests_load() -> None:
    import test_gliner
    import test_gliner2
    import test_hf_windows

    needed = {
        (test_hf_windows.DSLIM, test_hf_windows.DSLIM_REVISION),
        (test_gliner.GLINER_SMALL, test_gliner.GLINER_SMALL_REVISION),
        (test_gliner.DEBERTA_SMALL, test_gliner.DEBERTA_SMALL_REVISION),
        (test_gliner.EDGE, test_gliner.EDGE_REVISION),
        (test_gliner2.BASE, test_gliner2.BASE_PIN),
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
    assert "per_100kb_max" in ceilings[name]


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


def test_the_weekly_eval_reports_both_datasets() -> None:
    workflow = _workflow("ner-eval.yml")
    triggers = workflow[True]  # YAML 1.1 reads the key "on" as true
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    runs = "\n".join(step.get("run", "") for step in workflow["jobs"]["ner-eval"]["steps"])
    assert "for dataset in openpii nemotron" in runs
    assert "--limit 2000" in runs
    assert "--check" not in runs  # report only until baselines are recorded


def test_the_weekly_eval_installs_the_same_environment() -> None:
    steps = _workflow("ner-eval.yml")["jobs"]["ner-eval"]["steps"]
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "scripts/ner_ci_env.sh" in runs
    assert "uv sync" not in runs and "uv pip install" not in runs


# --- the environment script: every wheel hash-checked ---------------------------

ENV_SCRIPT = ROOT / "scripts" / "ner_ci_env.sh"
TORCH_PIN = ROOT / "scripts" / "ner_ci_torch_cpu.txt"


def _commands(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_the_cpu_torch_wheel_is_the_locked_version_pinned_by_hash() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = [p["version"] for p in lock["package"] if p["name"] == "torch"]
    (pin,) = _commands(TORCH_PIN.read_text())
    match = re.fullmatch(r"torch==([0-9.]+)\+cpu --hash=sha256:[0-9a-f]{64}", pin)
    assert match is not None, pin
    assert locked == [match.group(1)]


def test_the_environment_script_checks_every_hash() -> None:
    commands = _commands(ENV_SCRIPT.read_text())
    installs = [c for c in commands if c.startswith("uv pip install")]
    exports = [c for c in commands if c.startswith("uv export")]
    # The export that is installed keeps the lock's hashes; packages are
    # left out by name, never by filtering its lines.
    (installed_export,) = [c for c in exports if "--no-hashes" not in c]
    assert '"${omit[@]}"' in installed_export and "--frozen" in installed_export
    assert all("grep" not in c for c in commands)
    assert "--no-emit-package" in ENV_SCRIPT.read_text()
    # Both wheel installs require a hash for every requirement and resolve
    # nothing beyond the files (the export is the whole locked closure).
    requirements, torch, project = installs
    assert '--require-hashes --no-deps -r "$requirements"' in requirements
    assert '--require-hashes --no-deps -r "$pin"' in torch
    assert "--index-url https://download.pytorch.org/whl/cpu" in torch
    assert project.endswith("--no-deps -e .")
    assert commands[-1].startswith("uv pip check")
    # The venv is the platform the pinned wheel is built for.
    assert any(c.startswith("uv venv --python 3.13") for c in commands)


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
