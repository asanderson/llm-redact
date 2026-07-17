"""docs/README.md must link every document in docs/ — an index that
silently misses a doc defeats its purpose."""

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
