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
