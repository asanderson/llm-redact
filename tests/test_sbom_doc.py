"""docs/SBOM.md must name every dependency surface in pyproject.toml —
an SBOM that silently omits a package is worse than none. Both the
direct runtime deps, every extra (name and packages), and the dev group
must appear; the transitive closure is checked against uv.lock's
resolution for the project's own dependencies."""

import re
import tomllib
from pathlib import Path


def _repo_root() -> Path:
    # Walk up to the directory that holds uv.lock: under mutmut the suite
    # runs from the mutants/ tree copy, which carries pyproject/docs/tests
    # but NOT uv.lock — anchoring at parent.parent there crashed the
    # stats-collection run and took the whole mutation gate down with it.
    p = Path(__file__).resolve().parent
    while not (p / "uv.lock").is_file():
        if p == p.parent:
            raise FileNotFoundError("no uv.lock above tests/test_sbom_doc.py")
        p = p.parent
    return p


ROOT = _repo_root()
SBOM = (ROOT / "docs" / "SBOM.md").read_text()
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def _name(requirement: str) -> str:
    return re.split(r"[\[><=~;!\s]", requirement.strip(), maxsplit=1)[0]


def test_runtime_deps_named() -> None:
    for req in PYPROJECT["project"]["dependencies"]:
        assert _name(req) in SBOM, f"runtime dep {req} missing from docs/SBOM.md"


def test_every_extra_and_package_named() -> None:
    for extra, reqs in PYPROJECT["project"]["optional-dependencies"].items():
        assert f"`{extra}`" in SBOM, f"extra [{extra}] missing from docs/SBOM.md"
        for req in reqs:
            assert _name(req) in SBOM, f"extra package {req} missing from docs/SBOM.md"


def test_dev_group_named() -> None:
    for req in PYPROJECT["dependency-groups"]["dev"]:
        assert _name(req) in SBOM, f"dev tool {req} missing from docs/SBOM.md"


def test_runtime_closure_named() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    packages = {p["name"]: p for p in lock["package"]}
    closure: set[str] = set()
    frontier = [_name(r) for r in PYPROJECT["project"]["dependencies"]]
    while frontier:
        name = frontier.pop()
        if name in closure or name not in packages:
            continue
        closure.add(name)
        frontier.extend(d["name"] for d in packages[name].get("dependencies", []))
    for name in sorted(closure):
        assert name in SBOM, f"transitive runtime dep {name} missing from docs/SBOM.md"
