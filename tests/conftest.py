import os
import tempfile

import pytest

from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault


def pytest_configure(config: pytest.Config) -> None:
    """The public Free repo runs its suite on the FREE tier (keyless) by design
    — the paid subsystems and their tests live in the separate llm-redact-pro
    package (R4 open-core split). Tests of Free CODE that is gated behind a paid
    TIER (Bedrock/Vertex/Azure, per-conversation, audit) moved to the pro repo's
    CI, where pro is installed and the real resolver grants the tier. The gating
    tests that stay here build ResolvedLicense inputs directly (see
    tests/license_fixtures.py) — no signing, no resolver."""
    # Keep any state files (users.db etc.) out of the real home: every default
    # XDG path the suite touches lands in a throwaway dir.
    os.environ.setdefault("XDG_DATA_HOME", tempfile.mkdtemp(prefix="llm-redact-tests-xdg-"))


@pytest.fixture
def vault() -> InMemoryVault:
    return InMemoryVault()


@pytest.fixture
def redactor(vault: InMemoryVault) -> Redactor:
    config = DetectionConfig()
    return Redactor(build_detectors(config), vault, build_allowlist(config))


@pytest.fixture
def rehydrator(vault: InMemoryVault) -> Rehydrator:
    return Rehydrator(vault)
