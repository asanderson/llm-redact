"""Helpers for the ``real_model`` tests (deselected by default; run with
``pytest -m real_model``): a real NER model is loaded only from the local
Hugging Face cache, never downloaded, and the test skips when the model's
extra or the cached files are absent.

Offline is enforced twice: the environment variables (for libraries imported
later) and ``huggingface_hub.constants.HF_HUB_OFFLINE`` itself, which the hub
client reads on every request — so no request leaves even when the hub was
imported earlier in the session.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest


def offline_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every Hugging Face Hub request for the rest of the test."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    constants = pytest.importorskip("huggingface_hub.constants")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)


def cached_snapshot(repo_id: str, revision: str, files: list[str]) -> str:
    """The local folder of ``repo_id`` at ``revision`` holding ``files``
    (a partial snapshot is complete for them), or a skip."""
    hub = pytest.importorskip("huggingface_hub")
    try:
        path: str = hub.snapshot_download(
            repo_id, revision=revision, allow_patterns=files, local_files_only=True
        )
    except Exception:  # not cached here (LocalEntryNotFoundError and kin)
        pytest.skip(f"{repo_id} at {revision} is not in the Hugging Face cache")
    return path


# The CI ner-models job pulls every model these tests load and sets this
# variable, so a real_model test that skips there (a model missing from the
# cache, an extra missing from the environment) FAILS the job instead of
# turning it green without having run (as LLM_REDACT_TEST_REAL_DB_REQUIRED
# does for the RDBMS job).
REQUIRED_ENV = "LLM_REDACT_TEST_REAL_MODELS_REQUIRED"


def required_skip_failure(
    report: pytest.TestReport, marked: bool, environ: Mapping[str, str]
) -> str | None:
    """Why ``report`` must fail instead of skip: a skipped ``real_model``
    test while REQUIRED_ENV is "1"; else None."""
    if not (report.skipped and marked and environ.get(REQUIRED_ENV) == "1"):
        return None
    longrepr = report.longrepr
    reason = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)
    return f"{REQUIRED_ENV}=1 but this real_model test skipped: {reason}"


def fail_required_skips(
    item: pytest.Item, report: pytest.TestReport, environ: Mapping[str, str]
) -> None:
    """Turn ``report`` into a failure when ``required_skip_failure`` says so."""
    marked = item.get_closest_marker("real_model") is not None
    message = required_skip_failure(report, marked, environ)
    if message is not None:
        report.outcome = "failed"
        report.longrepr = message
