"""Cloud-platform detection (informational placement surfacing). All probes
are faked — no test ever touches a metadata endpoint. The former k8s/cloud
placement GATES are gone: the FOSS core enforces nothing, so this suite now
also pins that a keyless proxy starts cleanly on Kubernetes."""

from __future__ import annotations

import dataclasses

import pytest

from llm_redact.cloud_detect import detect_cloud
from llm_redact.config import Config, UsersConfig


def test_env_override_wins_over_probes() -> None:
    def exploding_probe(method: str, url: str, headers: dict[str, str]) -> bool:
        raise AssertionError("probes must not run when the override is set")

    assert detect_cloud(env={"LLM_REDACT_CLOUD": "aws"}, probe=exploding_probe) == "aws"
    assert detect_cloud(env={"LLM_REDACT_CLOUD": "none"}, probe=exploding_probe) is None


def test_skip_env_disables_probes() -> None:
    def exploding_probe(method: str, url: str, headers: dict[str, str]) -> bool:
        raise AssertionError("probes must not run with the skip flag")

    assert detect_cloud(env={"LLM_REDACT_SKIP_CLOUD_DETECT": "1"}, probe=exploding_probe) is None


@pytest.mark.parametrize("platform", ["aws", "azure", "gcp"])
def test_probe_identifies_platform(platform: str) -> None:
    def probe(method: str, url: str, headers: dict[str, str]) -> bool:
        if platform == "gcp":
            return "metadata.google.internal" in url
        if platform == "aws":
            return method == "PUT"
        return "metadata/instance" in url

    assert detect_cloud(env={}, probe=probe) == platform


def test_no_probe_hits_means_on_prem() -> None:
    assert detect_cloud(env={}, probe=lambda *a: False) is None


def test_proxy_startup_on_k8s_keyless_starts(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The FOSS core has no k8s or cloud gate: a keyless proxy on Kubernetes
    # starts cleanly and never probes a metadata endpoint (the env override
    # pins that; the probe would blow up on a laptop anyway).
    from llm_redact.proxy import create_app

    config = dataclasses.replace(
        Config(),
        users=UsersConfig(path=str(tmp_path / "users.db")),  # type: ignore[operator]
    )
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("LLM_REDACT_CLOUD", "none")  # never probe in tests
    monkeypatch.delenv("LLM_REDACT_LICENSE_KEY", raising=False)
    create_app(config)
