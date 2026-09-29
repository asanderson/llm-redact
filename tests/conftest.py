import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

import isolation
from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault

# The session's throwaway home (tests/isolation.py), and the snapshot of the
# real locations taken before it was put in place.
ISOLATION_ROOT: Path | None = None
_REAL_WATCHED: list[Path] = []
_REAL_BEFORE: isolation.Snapshot = {}


def pytest_configure(config: pytest.Config) -> None:
    """The public Free repo runs its suite on the FREE tier (keyless) by design
    — the paid subsystems and their tests live in the separate llm-redact-pro
    package (R4 open-core split). Tests of Free CODE that is gated behind a paid
    TIER (Bedrock/Vertex/Azure, per-conversation, audit) moved to the pro repo's
    CI, where pro is installed and the real resolver grants the tier. The gating
    tests that stay here build ResolvedLicense inputs directly (see
    tests/license_fixtures.py) — no signing, no resolver.

    Before anything is collected, the session is isolated from the real
    machine (tests/isolation.py): HOME and the XDG config/data/state dirs
    point into one throwaway directory — unconditionally, whatever the
    environment already set — and a deployment's LLM_REDACT_* variables are
    dropped; subprocesses inherit all of it."""
    global ISOLATION_ROOT, _REAL_WATCHED, _REAL_BEFORE
    _REAL_WATCHED = isolation.watched_paths(os.environ, Path.home())
    _REAL_BEFORE = isolation.snapshot(_REAL_WATCHED)
    ISOLATION_ROOT = Path(tempfile.mkdtemp(prefix="llm-redact-tests-home-"))
    isolation.isolate(os.environ, ISOLATION_ROOT)


def pytest_unconfigure(config: pytest.Config) -> None:
    if ISOLATION_ROOT is not None:
        shutil.rmtree(ISOLATION_ROOT, ignore_errors=True)


@pytest.fixture(scope="session")
def isolation_root() -> Path:
    """This session's throwaway home (a fixture, not a module attribute: an
    in-process rerun — mutmut's — imports conftest afresh, while test modules
    stay cached)."""
    assert ISOLATION_ROOT is not None
    return ISOLATION_ROOT.resolve()


@pytest.fixture(scope="session", autouse=True)
def _real_locations_untouched() -> Iterator[None]:
    """Fails the run when anything under the REAL llm-redact config/data dirs,
    the agent plugins' command dirs or the user service unit was created,
    changed or removed while the suite ran."""
    yield
    if os.environ.get(isolation.ALLOW_REAL_DIR_CHANGES) == "1":
        return
    changed = isolation.changes(_REAL_BEFORE, isolation.snapshot(_REAL_WATCHED))
    if changed:
        pytest.fail(
            "the test run changed real locations outside its throwaway home: "
            + ", ".join(changed)
            + " (a real llm-redact running on this machine writes there too: set"
            f" {isolation.ALLOW_REAL_DIR_CHANGES}=1 to skip this check)",
            pytrace=False,
        )


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
