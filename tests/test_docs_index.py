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


# The planning record's owner-decision, architecture-decision, task and phase
# ids (D6, AD11, T46, P13.7, Phase 18, ...) are private: a public reader cannot
# look them up, so the docs state each rule and decision in plain words instead.
_PLANNING_ID = re.compile(
    r"\bowner decision\b|\bD[0-9]{1,2}\b|\bAD[0-9]{1,2}\b|\bT[0-9]{2}[a-z]?\b"
    r"|\bP[0-9]{1,2}\.[0-9]+\b|\bPhase [0-9]+\b"
)


def _planning_ids(text: str) -> list[str]:
    return [m.group(0) for m in _PLANNING_ID.finditer(text)]


def test_public_docs_cite_no_private_planning_ids() -> None:
    root = DOCS.parent
    found = {
        str(path.relative_to(root)): ids
        for path in [
            *sorted(DOCS.glob("*.md")),
            root / "README.md",
            root / "CHANGELOG.md",
            root / "config.example.toml",
            root / "src" / "llm_redact" / "user_guide.md",
        ]
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
    assert _planning_ids("air-gapped start (AD11); P13.7's config; Phase 18 proved") == [
        "AD11",
        "P13.7",
        "Phase 18",
    ]
    assert _planning_ids("a p50 of 100 ms; P95 latency; the phase 2 notes; ADD11") == []


def _fence_problems(text: str) -> list[int]:
    """Lines (1-based) where a ``` fence goes wrong: a closing fence with
    text after it (CommonMark reads it as content, so the block runs on),
    or a block still open at the end."""
    problems: list[int] = []
    open_at = 0
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("```"):
            continue
        if not open_at:
            open_at = number
        elif stripped == "```":
            open_at = 0
        else:
            problems.append(number)
    return problems + ([open_at] if open_at else [])


def test_docs_code_fences_close() -> None:
    root = DOCS.parent
    found = {
        str(path.relative_to(root)): lines
        for path in [*sorted(DOCS.glob("*.md")), root / "README.md", root / "CHANGELOG.md"]
        if (lines := _fence_problems(path.read_text()))
    }
    assert found == {}


def test_the_fence_check_finds_a_closing_fence_with_text() -> None:
    assert _fence_problems("```bash\nx\n```\n\n```toml\ny\n```\n") == []
    assert _fence_problems("```bash\nx\n``` Then prose\nmore\n```\n") == [3]
    assert _fence_problems("```bash\nx\n") == [1]
