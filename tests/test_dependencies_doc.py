"""docs/dependencies.md must track pyproject.toml in both directions."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "dependencies.md"


def _base_name(requirement: str) -> str:
    return re.split(r"[<>=!\[; ]", requirement.strip(), maxsplit=1)[0]


def test_every_declared_package_is_documented() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    doc = DOC.read_text()
    runtime = {_base_name(r) for r in pyproject["project"]["dependencies"]}
    extras = {
        name: {_base_name(r) for r in reqs}
        for name, reqs in pyproject["project"]["optional-dependencies"].items()
    }
    for package in runtime:
        assert f"`{package}`" in doc, f"runtime dep {package} missing from docs/dependencies.md"
    for extra, packages in extras.items():
        assert f"`{extra}`" in doc, f"extra {extra} missing from docs/dependencies.md"
        for package in packages:
            assert f"`{package}`" in doc, (
                f"package {package} (extra {extra}) missing from docs/dependencies.md"
            )


def test_documented_extras_all_exist() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    declared = set(pyproject["project"]["optional-dependencies"])
    doc = DOC.read_text()
    table = re.findall(r"^\| `([a-z0-9_-]+)` \| `", doc, flags=re.M)
    documented = set(table)
    stale = documented - declared
    assert not stale, f"docs/dependencies.md documents unknown extras: {sorted(stale)}"
