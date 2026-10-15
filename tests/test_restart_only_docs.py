"""The restart-only key lists in the docs must match ``config.RESTART_ONLY_KEYS``
and ``config.RESTART_ONLY_PATHS``.

SIGHUP reload and the (llm-redact-pro) dashboard config editor treat these config sections —
and these keys inside the hot ``[detection]`` — as restart-only; README.md,
docs/deployment.md and docs/troubleshooting.md each enumerate them in prose.
A key added to the code set without updating the docs (or a doc claiming a
key the code does not treat as restart-only) fails here — the
test_api_coverage.py doc-sync pattern.
"""

import re
from pathlib import Path

import pytest

from llm_redact.config import RESTART_ONLY_KEYS, RESTART_ONLY_PATHS

ROOT = Path(__file__).resolve().parent.parent

# "vault, audit, host, port, allowed_hosts, log, TLS, OTel, users, and email changes …
#  "require restart" …" — capture the comma list immediately before "changes".
_LIST_RE = re.compile(r"((?:[A-Za-z_]+,\s*)+and\s+[A-Za-z_]+)\s+changes\b[^.]*?require restart")


@pytest.mark.parametrize("doc", ["README.md", "docs/deployment.md", "docs/troubleshooting.md"])
def test_docs_enumerate_exactly_the_readonly_keys(doc: str) -> None:
    text = (ROOT / doc).read_text()
    match = _LIST_RE.search(text)
    assert match is not None, f"{doc}: restart-only sentence not found"
    names = {
        part.strip().removeprefix("and ").strip().lower()
        for part in match.group(1).split(",")
        if part.strip()
    }
    assert names == set(RESTART_ONLY_KEYS), (
        f"{doc} restart-only list {sorted(names)} != code {sorted(RESTART_ONLY_KEYS)}"
    )


# "`[detection.ner.llm]` `url`, `allow_remote` and `api_key_env`": a table, then
# its restart-only keys as a comma list.
_PATHS_RE = re.compile(r"`\[([a-z_.]+)\]`\s+((?:`[a-z_]+`,\s*)*`[a-z_]+`\s+and\s+`[a-z_]+`)")


@pytest.mark.parametrize("doc", ["README.md", "docs/deployment.md", "docs/troubleshooting.md"])
def test_docs_enumerate_exactly_the_restart_only_paths(doc: str) -> None:
    text = (ROOT / doc).read_text()
    expected = {(".".join(path[:-1]), path[-1]) for path in RESTART_ONLY_PATHS}
    tables = {table for table, _key in expected}
    found = [
        {(match.group(1), key) for key in re.findall(r"`([a-z_]+)`", match.group(2))}
        for match in _PATHS_RE.finditer(text)
        if match.group(1) in tables
    ]
    assert found, f"{doc}: restart-only keys sentence not found"
    assert all(names == expected for names in found), (
        f"{doc} restart-only keys {found} != code {sorted(expected)}"
    )
