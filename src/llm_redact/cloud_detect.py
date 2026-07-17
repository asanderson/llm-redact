"""Best-effort cloud-platform detection for license placement checks.

Team licenses carry ONE cloud entitlement and Unlimited/Managed all three
(llm-redact-pro docs/licensing.md); when the proxy runs on Kubernetes, this module asks
the instance metadata services which cloud is underneath so a Team/aws key
cannot quietly serve from GKE. It is deliberately best-effort — metadata
endpoints can be firewalled, and the deterministic gate on each cloud's
FEATURE stack (features.required_clouds) does not depend on it.

Order of authority:
1. ``LLM_REDACT_CLOUD`` env override: ``aws`` | ``azure`` | ``gcp`` |
   ``none`` — for air-gapped clusters and on-prem declarations.
2. ``LLM_REDACT_SKIP_CLOUD_DETECT=1``: no probes, detection reports None.
3. HTTP probes with sub-second timeouts against the three documented
   metadata services (never run unless the caller says so — the proxy only
   probes when it actually finds itself on Kubernetes).

The probes carry no request data and the results are a single platform
label — nothing here touches user traffic.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import httpx

logger = logging.getLogger("llm_redact.cloud")

ENV_OVERRIDE = "LLM_REDACT_CLOUD"
ENV_SKIP = "LLM_REDACT_SKIP_CLOUD_DETECT"
_TIMEOUT_S = 0.7

# One probe per platform: (label, method, url, headers). The AWS IMDSv2
# token PUT and the required Metadata headers on Azure/GCP make each probe
# specific to its platform even though AWS and Azure share the link-local IP.
_PROBES: tuple[tuple[str, str, str, dict[str, str]], ...] = (
    (
        "gcp",
        "GET",
        "http://metadata.google.internal/computeMetadata/v1/instance/id",
        {"Metadata-Flavor": "Google"},
    ),
    (
        "aws",
        "PUT",
        "http://169.254.169.254/latest/api/token",
        {"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
    ),
    (
        "azure",
        "GET",
        "http://169.254.169.254/metadata/instance/compute/azEnvironment"
        "?api-version=2021-02-01&format=text",
        {"Metadata": "true"},
    ),
)

Prober = Callable[[str, str, dict[str, str]], bool]


def _http_probe(method: str, url: str, headers: dict[str, str]) -> bool:
    try:
        response = httpx.request(method, url, headers=headers, timeout=_TIMEOUT_S)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def detect_cloud(env: dict[str, str] | None = None, probe: Prober | None = None) -> str | None:
    """The platform label ("aws"/"azure"/"gcp") or None (unknown/on-prem).

    Never raises: an unreachable metadata service is an on-prem answer,
    not an error — the deterministic feature-stack gates carry the real
    enforcement weight either way.
    """
    environ = env if env is not None else dict(os.environ)
    override = environ.get(ENV_OVERRIDE, "").strip().lower()
    if override in ("aws", "azure", "gcp"):
        return override
    if override == "none":
        return None
    if environ.get(ENV_SKIP, "") == "1":
        return None
    prober = probe if probe is not None else _http_probe
    for label, method, url, headers in _PROBES:
        if prober(method, url, headers):
            logger.info("cloud platform detected: %s", label)
            return label
    return None
