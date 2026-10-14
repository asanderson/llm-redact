"""The published distribution's metadata says what is true.

CI installs the lock (``uv sync --frozen``), so it never sees the declared
floors a non-frozen ``pip install`` may resolve; and README.md is the PyPI
long description, which PyPI renders without the repository around it.
"""

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
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
