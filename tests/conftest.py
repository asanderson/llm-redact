import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import isolation
import mutation_limits
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
    machine (tests/isolation.py): HOME (USERPROFILE on Windows) and the XDG
    config/data/state dirs point into one throwaway directory —
    unconditionally, whatever the environment already set — and a
    deployment's LLM_REDACT_* variables are dropped; subprocesses inherit all
    of it."""
    global ISOLATION_ROOT, _REAL_WATCHED, _REAL_BEFORE
    _REAL_WATCHED = isolation.watched_paths(os.environ, Path.home())
    _REAL_BEFORE = isolation.snapshot(_REAL_WATCHED)
    ISOLATION_ROOT = Path(tempfile.mkdtemp(prefix="llm-redact-tests-home-"))
    isolation.isolate(os.environ, ISOLATION_ROOT)
    # Under mutmut, a mutant's test process gets a memory ceiling, so an
    # allocating endless loop is killed instead of exhausting the CI runner.
    mutation_limits.apply(os.environ)


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


@pytest.fixture(autouse=True)
def _local_refusals_counted_once(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every request any test sends through ``create_app`` counts at most
    ONE local refusal, and every refusal recorded with a row names a kind
    that answers its status (tests/local_refusals.py) — so a refusal site
    counted twice, or under the wrong kind, fails whichever test drives it."""
    import local_refusals
    from llm_redact import proxy
    from llm_redact.metrics import LOCAL_REFUSAL_KINDS, Metrics

    real_handle = proxy.handle
    real_count = Metrics.count_local_refusal
    real_record = proxy.ProxyState.record_request

    async def handle(request: Any) -> Any:
        counted: list[str] = []
        token = local_refusals.REQUEST_REFUSALS.set(counted)
        try:
            return await real_handle(request)
        finally:
            local_refusals.REQUEST_REFUSALS.reset(token)
            assert len(counted) <= 1, f"one request counted {counted}"

    log = os.environ.get("LLM_REDACT_TEST_REFUSAL_LOG")

    def count(self: Metrics, kind: Any, provider: str | None) -> None:
        assert kind in LOCAL_REFUSAL_KINDS, kind
        if log:
            with open(log, "a", encoding="utf-8") as out:
                out.write(f"{kind}\t{os.environ.get('PYTEST_CURRENT_TEST', '')}\n")
        counted = local_refusals.REQUEST_REFUSALS.get()
        if counted is not None:
            counted.append(kind)
        real_count(self, kind, provider)

    def record(self: Any, **row: Any) -> None:
        kind = row.get("refusal")
        if kind is not None:
            allowed = local_refusals.KIND_STATUSES[kind]
            assert row["status"] in allowed, f"{kind} recorded with status {row['status']}"
        real_record(self, **row)

    monkeypatch.setattr(proxy, "handle", handle)
    monkeypatch.setattr(Metrics, "count_local_refusal", count)
    monkeypatch.setattr(proxy.ProxyState, "record_request", record)
    yield
