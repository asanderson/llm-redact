"""docs/README.md must link every document in docs/ — an index that
silently misses a doc defeats its purpose."""

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"


def test_docs_index_links_every_doc() -> None:
    index = (DOCS / "README.md").read_text()
    missing = [
        p.name
        for p in sorted(DOCS.glob("*.md"))
        if p.name != "README.md" and f"({p.name})" not in index
    ]
    assert not missing, f"docs/README.md is missing links to: {missing}"


# The planning record's owner-decision and task ids (D6, T46, ...) are
# private: a public reader cannot look them up, so the docs state each
# rule and decision in plain words instead.
_PLANNING_ID = re.compile(r"\bowner decision\b|\bD[0-9]{1,2}\b|\bT[0-9]{2}[a-z]?\b")


def _planning_ids(text: str) -> list[str]:
    return [m.group(0) for m in _PLANNING_ID.finditer(text)]


def test_public_docs_cite_no_private_planning_ids() -> None:
    root = DOCS.parent
    found = {
        str(path.relative_to(root)): ids
        for path in [*sorted(DOCS.glob("*.md")), root / "README.md", root / "CHANGELOG.md"]
        if (ids := _planning_ids(path.read_text()))
    }
    assert found == {}


def test_the_planning_id_check_finds_them() -> None:
    assert _planning_ids("owner decision D6, open; ships in 2.0.0 (T46); T33b") == [
        "owner decision",
        "D6",
        "T46",
        "T33b",
    ]
    assert _planning_ids("a T4 GPU-free D-Bus run; DATE_OF_BIRTH; 2.0.0") == []
