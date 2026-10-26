"""Every version uv.lock pins satisfies the lock-only constraints it records.

`uv lock --check` compares the constraints the lock records ([manifest]
constraints) with pyproject.toml's [tool.uv] constraint-dependencies, not
the pinned versions with either. Git merges two lock changes line by line,
so a lock can record a constraint its own pins break and still pass that
check: once one pull request locked gliner 0.2.24 and another added
`gliner>=0.2.29`, main carried both until the next resolution moved three
packages at once. This test reads the pins themselves.
"""

import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


def _repo_root() -> Path:
    # The directory that holds uv.lock: under mutmut the suite runs from the
    # mutants/ tree copy, which has no uv.lock (tests/test_sbom_doc.py).
    p = Path(__file__).resolve().parent
    while not (p / "uv.lock").is_file():
        if p == p.parent:
            raise FileNotFoundError("no uv.lock above tests/test_lock_constraints.py")
        p = p.parent
    return p


ROOT = _repo_root()


def _violations(lock: dict[str, Any]) -> list[str]:
    """What breaks a constraint the lock records, one line each."""
    pinned: dict[str, list[str | None]] = {}
    for package in lock.get("package", []):
        pinned.setdefault(canonicalize_name(package["name"]), []).append(package.get("version"))
    problems = []
    for constraint in lock.get("manifest", {}).get("constraints", []):
        name = canonicalize_name(constraint["name"])
        if "marker" in constraint:
            # A marker scopes the constraint to some resolution forks: this
            # check would have to read each package's resolution markers.
            problems.append(f"{name}: a constraint with a marker is not checked here")
            continue
        versions = pinned.get(name)
        if not versions:
            problems.append(f"{name}: constrained but not locked")
            continue
        specifier = SpecifierSet(constraint.get("specifier", ""))
        for version in versions:
            try:
                ok = version is not None and specifier.contains(Version(version), prereleases=True)
            except InvalidVersion:
                ok = False
            if not ok:
                problems.append(f"{name} {version} breaks {name}{specifier}")
    return problems


def test_every_pin_satisfies_the_recorded_constraints() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert _violations(lock) == []


def test_the_lock_records_the_declared_constraints() -> None:
    # The check above reads the lock's record; this ties it to pyproject.toml,
    # so a lock that records none is not read as satisfying them.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        (canonicalize_name(r.name), str(r.specifier))
        for r in map(Requirement, pyproject["tool"]["uv"]["constraint-dependencies"])
    }
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    recorded = {
        (canonicalize_name(c["name"]), str(SpecifierSet(c.get("specifier", ""))))
        for c in lock["manifest"]["constraints"]
    }
    assert declared
    assert recorded == declared


def _lock(*packages: tuple[str, str | None], constraints: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "package": [
            {"name": name, **({"version": version} if version is not None else {})}
            for name, version in packages
        ],
        "manifest": {"constraints": constraints},
    }


GLINER_FLOOR = [{"name": "gliner", "specifier": ">=0.2.29"}]


@pytest.mark.parametrize(
    ("lock", "expected"),
    [
        # The merged lock: #136's pin under #137's constraint.
        (
            _lock(("gliner", "0.2.24"), ("transformers", "5.20.2"), constraints=GLINER_FLOOR),
            ["gliner 0.2.24 breaks gliner>=0.2.29"],
        ),
        (_lock(("gliner", "0.2.29"), constraints=GLINER_FLOOR), []),
        # A prerelease within the bounds satisfies them, as uv resolves it.
        (_lock(("gliner", "0.3.0rc1"), constraints=GLINER_FLOOR), []),
        # A forked resolution pins a package twice: both must satisfy it.
        (
            _lock(("GLiNER", "0.2.30"), ("gliner", "0.2.20"), constraints=GLINER_FLOOR),
            ["gliner 0.2.20 breaks gliner>=0.2.29"],
        ),
        (
            _lock(("torch", "2.14.1"), constraints=GLINER_FLOOR),
            ["gliner: constrained but not locked"],
        ),
        (_lock(("gliner", None), constraints=GLINER_FLOOR), ["gliner None breaks gliner>=0.2.29"]),
        (
            _lock(
                ("gliner", "0.2.29"),
                constraints=[{**GLINER_FLOOR[0], "marker": "sys_platform == 'linux'"}],
            ),
            ["gliner: a constraint with a marker is not checked here"],
        ),
    ],
)
def test_a_pin_breaking_a_constraint_is_reported(lock: dict[str, Any], expected: list[str]) -> None:
    assert _violations(lock) == expected
