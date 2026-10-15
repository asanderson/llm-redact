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


# Every shipped text that tells an operator what a reload does with the llm
# backend states the start rule pin_restart_only_paths applies: a reload
# starts it only at the values the proxy started with, ALSO on a proxy
# started without it (where they are a table the startup file held dormant,
# or the defaults). The packaged user guide (served at /__llm-redact/guide;
# its HTML is pinned to it by test_user_guide.py) and the example config are
# read as they ship.
_START_RULE_DOCS = [
    "README.md",
    "docs/deployment.md",
    "docs/troubleshooting.md",
    "docs/detection.md",
    "docs/threat-model.md",
    "docs/security-dataflows.md",
    "src/llm_redact/user_guide.md",
    "config.example.toml",
    "plugins/llm-redact/commands/config-edit.md",
]
_STARTS_ONLY = "starts the backend only at the values the proxy started with"
_ALSO_WITHOUT_IT = "also on a proxy started without it"
# The rules before this one: "never starts the backend" (no reload started
# it), "never starts the backend on a proxy started without it" and "starts
# it again only at the values …" (a reload re-activated it only on a proxy
# started running it), and that rule's refusals.
_SUPERSEDED = re.compile(
    r"never starts? (?:the |that )?(?:llm )?backend"
    r"|starts? (?:it|the backend|that backend|the llm backend) again only"
    r"|not started running it"
    r"|anywhere but where the proxy was started to send it"
)


def _prose(text: str) -> str:
    """``text`` as one line of words: comment markers at line starts and
    backticks dropped, whitespace runs folded to one space."""
    lines = (line.strip().lstrip("#").strip() for line in text.splitlines())
    return " ".join(" ".join(lines).replace("`", "").split())


@pytest.mark.parametrize("doc", _START_RULE_DOCS)
def test_docs_state_when_a_reload_starts_the_llm_backend(doc: str) -> None:
    prose = _prose((ROOT / doc).read_text())
    rule = prose.replace("that backend", "the backend").replace("the llm backend", "the backend")
    stated = f"{_STARTS_ONLY}, {_ALSO_WITHOUT_IT}"
    assert stated in rule, f"{doc}: does not say {stated!r}"
    assert not _SUPERSEDED.search(prose), f"{doc}: {_SUPERSEDED.search(prose)}"


def test_the_threat_model_states_the_residual_of_a_start_by_reload() -> None:
    """Whoever can change ``[detection]`` can switch the backend on at the
    startup values — a dormant table's or the loopback defaults — and the
    threat model says so among the residual risks, plainly."""
    prose = _prose((ROOT / "docs/threat-model.md").read_text())
    row = prose[prose.index("| The opt-in llm NER backend") :]
    residual = row[row.index("Residual:") : row.index(" |", row.index("Residual:"))]
    for phrase in (
        "whoever can change [detection] can switch the backend on at the startup values",
        "on a proxy started without it",
        "a [detection.ner.llm] table the startup file held with NER off (a dormant table)",
        "the loopback defaults",
        "every scanned string then goes to that server, unredacted",
    ):
        assert phrase in residual, phrase
