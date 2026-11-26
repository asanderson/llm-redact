"""The published distribution's metadata says what is true.

CI installs the lock (``uv sync --frozen``), so it never sees the declared
floors a non-frozen ``pip install`` may resolve; and README.md is the PyPI
long description, which PyPI renders without the repository around it.
"""

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
README = (ROOT / "README.md").read_text(encoding="utf-8")
REPO = "https://github.com/asanderson/llm-redact"

# The first release of each package past ones the suite fails on:
# - uvicorn 0.50.0 hands the app a client's Sec-WebSocket-Protocol header as
#   one subprotocol, so a realtime client offering two or more fails
#   (test_realtime_relay.py, test_realtime_identity.py,
#   test_identity_hardening.py).
# - pypdf below 6.5 takes the first 0x80 byte of an inline run-length image
#   for its end-of-data marker, and the extraction worker errors on the PDF
#   (test_extract_worker.py).
FIXED_IN = {"uvicorn": Version("0.50.1"), "pypdf": Version("6.5")}


def _declared(package: str) -> list[Requirement]:
    """Every requirement on ``package``: dependencies, extras and groups."""
    project = PYPROJECT["project"]
    lines = [
        *project["dependencies"],
        *(line for extra in project["optional-dependencies"].values() for line in extra),
        *(line for group in PYPROJECT["dependency-groups"].values() for line in group),
    ]
    requirements = [Requirement(line) for line in lines if isinstance(line, str)]
    return [r for r in requirements if r.name.lower() == package]


def _floor(requirement: Requirement) -> Version | None:
    bounds = [Version(s.version) for s in requirement.specifier if s.operator in (">=", "~=", "==")]
    return max(bounds, default=None)


@pytest.mark.parametrize("package", sorted(FIXED_IN))
def test_no_declared_floor_admits_a_known_bad_release(package: str) -> None:
    requirements = _declared(package)
    assert requirements, f"{package} is not declared"
    for requirement in requirements:
        floor = _floor(requirement)
        assert floor is not None and floor >= FIXED_IN[package], str(requirement)


def test_the_pypi_page_links_the_project() -> None:
    assert PYPROJECT["project"]["urls"] == {
        "Homepage": REPO,
        "Documentation": f"{REPO}/tree/main/docs",
        "Source": REPO,
        "Issues": f"{REPO}/issues",
        "Changelog": f"{REPO}/blob/main/CHANGELOG.md",
    }


# Link and image targets: inline (](target)), reference definitions
# ([name]: target) and HTML src/href attributes.
_TARGET = re.compile(
    r"\]\(\s*<?([^\s)>]+)"
    r"|^ {0,3}\[[^\]]+\]:\s*<?([^\s>]+)"
    r"""|\b(?:src|href)\s*=\s*["']?([^\s"'>]+)""",
    re.MULTILINE | re.IGNORECASE,
)
_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
# A file (blob, raw) or folder (tree) of this repository on main.
_TREE_URL = re.compile(
    r"https://(?:github\.com/asanderson/llm-redact/(blob|tree|raw)/main"
    r"|raw\.githubusercontent\.com/asanderson/llm-redact/main)/([^\s)\"'<>#?]+)"
)


def _relative_targets(markdown: str) -> list[str]:
    """Link and image targets that are neither a URL nor an in-page anchor."""
    targets = [next(t for t in m.groups() if t) for m in _TARGET.finditer(markdown)]
    return [t for t in targets if not t.startswith("#") and not _SCHEME.match(t)]


def test_the_link_pattern_reads_each_form() -> None:
    sample = (
        '[a](docs/a.md) [b](#b) ![c](https://x.example/c.png) ![d](<docs/d.png> "t")\n'
        "[e]: docs/e.md\n[f]: https://x.example/f\n"
        "<img src=\"docs/g.png\"> <a href='docs/h.md'>h</a> <a href=i.md>i</a>\n"
    )
    assert _relative_targets(sample) == [
        "docs/a.md",
        "docs/d.png",
        "docs/e.md",
        "docs/g.png",
        "docs/h.md",
        "i.md",
    ]


def test_the_readme_links_nothing_relative() -> None:
    # PyPI renders README.md on its own: a relative link or image there is
    # dead (the architecture diagram once showed as broken alt text).
    assert _TARGET.search(README)
    assert _relative_targets(README) == []


def test_the_readme_links_into_main_name_what_is_there() -> None:
    linked = _TREE_URL.findall(README)
    assert ("blob", "LICENSE") in linked
    missing = [
        path
        for kind, path in linked
        if not ((ROOT / path).is_dir() if kind == "tree" else (ROOT / path).is_file())
    ]
    assert missing == []


# --- Windows: every extra installs from wheels ------------------------------------
#
# CI installs from the lock on Linux; a user on Windows installs from PyPI,
# where a package without a Windows wheel is built from source, which needs
# a compiler most Windows machines lack (uvloop's refuses to build there at
# all). So every extra's closure in uv.lock, as Windows resolves its
# markers, must ship a wheel for x64 Windows on every Python the package
# supports. ARM64 Windows is best effort (owner decision Q1): there no new
# gap may appear, and these are the packages known to ship no win_arm64
# wheel at their locked version (cryptography and psycopg-binary keep the
# crypto, vault-oracle and vault-postgres extras from installing there; the
# NER and bench extras need torch, pyarrow or pyyaml).
LOCK = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
ARM64_GAPS = frozenset({"cryptography", "psycopg-binary", "pyarrow", "pyyaml", "torch"})
PYTHONS = ("3.11", "3.12", "3.13", "3.14")


def _windows(python: str, machine: str) -> dict[str, str]:
    return {
        "sys_platform": "win32",
        "platform_system": "Windows",
        "os_name": "nt",
        "platform_machine": machine,
        "python_version": python,
        "python_full_version": f"{python}.0",
        "implementation_name": "cpython",
        "implementation_version": f"{python}.0",
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_version": "",
        "extra": "",
    }


def _locked(name: str, version: str | None) -> dict[str, Any]:
    found = [
        package
        for package in LOCK["package"]
        if package["name"] == name and version in (None, package.get("version"))
    ]
    assert len(found) == 1, (name, version)
    return found[0]


def _closure(extra: str, environment: dict[str, str]) -> list[dict[str, Any]]:
    """The locked packages ``pip install llm-redact-proxy[extra]`` installs
    where ``environment`` holds."""
    root = _locked("llm-redact-proxy", None)
    todo = [*root["dependencies"], *root["optional-dependencies"][extra]]
    taken: dict[tuple[str, str | None], set[str]] = {}
    while todo:
        dependency = todo.pop()
        marker = dependency.get("marker")
        if marker is not None and not Marker(marker).evaluate(environment):
            continue
        package = _locked(dependency["name"], dependency.get("version"))
        key = (package["name"], package.get("version"))
        wanted = set(dependency.get("extra", []))
        if key in taken and wanted <= taken[key]:
            continue
        taken[key] = taken.get(key, set()) | wanted
        todo.extend(package.get("dependencies", []))
        for name in wanted:
            todo.extend(package.get("optional-dependencies", {}).get(name, []))
    return [_locked(name, version) for name, version in taken]


def _has_wheel(package: dict[str, Any], python: str, machine: str) -> bool:
    platform = {"AMD64": "win_amd64", "ARM64": "win_arm64"}[machine]
    minor = int(python.split(".")[1])
    for wheel in package.get("wheels", []):
        _, _, _, tags = parse_wheel_filename(wheel["url"].rsplit("/", 1)[-1])
        for tag in tags:
            if tag.platform not in (platform, "any"):
                continue
            if tag.interpreter in ("py3", "py2.py3") and tag.abi == "none":
                return True
            if tag.interpreter == f"cp3{minor}" and tag.abi in (f"cp3{minor}", "abi3", "none"):
                return True
            stable = tag.abi == "abi3" and tag.interpreter.startswith("cp3")
            if stable and int(tag.interpreter[3:]) <= minor:
                return True
    return False


def _without_wheels(extra: str, python: str, machine: str) -> set[str]:
    return {
        package["name"]
        for package in _closure(extra, _windows(python, machine))
        if package["name"] != "llm-redact-proxy" and not _has_wheel(package, python, machine)
    }


EXTRAS = sorted(PYPROJECT["project"]["optional-dependencies"])


@pytest.mark.parametrize("python", PYTHONS)
@pytest.mark.parametrize("extra", EXTRAS)
def test_every_extra_installs_from_wheels_on_x64_windows(extra: str, python: str) -> None:
    assert _without_wheels(extra, python, "AMD64") == set()


@pytest.mark.parametrize("python", PYTHONS)
@pytest.mark.parametrize("extra", EXTRAS)
def test_arm64_windows_gains_no_new_gap(extra: str, python: str) -> None:
    assert _without_wheels(extra, python, "ARM64") <= ARM64_GAPS


def test_the_event_loop_extra_leaves_windows_out() -> None:
    # uvloop does not support Windows; uvicorn's loop="auto" uses asyncio's
    # own loop there, so `[perf]` installs and changes nothing.
    (uvloop,) = _declared("uvloop")
    assert uvloop.marker is not None
    assert not uvloop.marker.evaluate({"sys_platform": "win32", "extra": "perf"})
    assert uvloop.marker.evaluate({"sys_platform": "linux", "extra": "perf"})
    assert uvloop.marker.evaluate({"sys_platform": "darwin", "extra": "perf"})


def test_the_classifiers_name_the_operating_systems_ci_runs() -> None:
    assert {
        "Operating System :: Microsoft :: Windows :: Windows 11",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    } <= set(PYPROJECT["project"].get("classifiers", []))
