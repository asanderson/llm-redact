"""The restart-only key lists in the docs must match ``proxy._READONLY_KEYS``.

The config editor and SIGHUP reload treat these config sections as
restart-only; README.md and docs/deployment.md each enumerate them in prose.
A key added to the code set without updating the docs (or a doc claiming a
key the code does not treat as restart-only) fails here — the
test_api_coverage.py doc-sync pattern.
"""

import re
from pathlib import Path

import pytest

from llm_redact.proxy import _READONLY_KEYS

ROOT = Path(__file__).resolve().parent.parent

# "vault, audit, host, port, log, TLS, OTel, users, and email changes …
#  "require restart" …" — capture the comma list immediately before "changes".
_LIST_RE = re.compile(r"((?:[A-Za-z]+,\s*)+and\s+[A-Za-z]+)\s+changes\b[^.]*?require restart")


@pytest.mark.parametrize("doc", ["README.md", "docs/deployment.md"])
def test_docs_enumerate_exactly_the_readonly_keys(doc: str) -> None:
    text = (ROOT / doc).read_text()
    match = _LIST_RE.search(text)
    assert match is not None, f"{doc}: restart-only sentence not found"
    names = {
        part.strip().removeprefix("and ").strip().lower()
        for part in match.group(1).split(",")
        if part.strip()
    }
    assert names == set(_READONLY_KEYS), (
        f"{doc} restart-only list {sorted(names)} != code {sorted(_READONLY_KEYS)}"
    )
