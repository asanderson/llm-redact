"""Every file a POSIX shell runs has LF line ends, in every checkout.

Git for Windows defaults ``core.autocrlf`` to true, which checks text files
out with CRLF line ends. A CRLF shell script fails in bash (``$'\\r':
command not found``) — in Git Bash, in WSL on a Windows checkout and in a
container image built from one — and a CRLF Dockerfile breaks every RUN line
continued with a backslash. ``.gitattributes`` pins these files to LF; this
test checks that each one has a rule there and, in this checkout (the
Windows CI leg's is an autocrlf one), LF line ends only."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# Never shipped or run: the fp corpus is scanned data (its line ends are
# part of it), the rest is tooling state.
_SKIPPED = {".git", ".venv", "node_modules", "fp_corpus", "__pycache__", ".mypy_cache"}
_SHELLS = (b"#!/bin/sh", b"#!/bin/bash", b"#!/usr/bin/env sh", b"#!/usr/bin/env bash")


def _files(root: Path) -> Iterator[Path]:
    for path in sorted(root.iterdir()):
        if path.name in _SKIPPED or path.name.startswith(".") and path.is_dir():
            continue
        if path.is_dir():
            yield from _files(path)
        elif path.is_file():
            yield path


def _run_by_a_shell(path: Path) -> bool:
    if path.suffix == ".sh" or path.name == "Dockerfile":
        return True
    with path.open("rb") as handle:
        return handle.read(32).startswith(_SHELLS)


def shell_files() -> list[str]:
    return [path.relative_to(ROOT).as_posix() for path in _files(ROOT) if _run_by_a_shell(path)]


def eol_rule(path: str, rules: str) -> str | None:
    """The ``eol`` attribute ``.gitattributes`` gives ``path``: the last
    matching line's (a pattern with a slash matches the path from the root,
    one without matches the file name)."""
    eol = None
    for line in rules.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        pattern, attributes = fields[0], fields[1:]
        subject = path if "/" in pattern else path.rsplit("/", 1)[-1]
        if fnmatch.fnmatchcase(subject, pattern.lstrip("/")):
            for attribute in attributes:
                if attribute.startswith("eol="):
                    eol = attribute.removeprefix("eol=")
    return eol


def test_the_scan_finds_the_shell_scripts() -> None:
    found = shell_files()
    for expected in (
        "Dockerfile",
        "scripts/install.sh",
        "tests/airgap/run_offline.sh",
        "plugins/llm-redact/bin/llm-redact-posture",
    ):
        assert expected in found


@pytest.mark.parametrize("path", shell_files())
def test_a_file_a_shell_runs_is_pinned_to_lf(path: str) -> None:
    rules = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert eol_rule(path, rules) == "lf", f"add a `text eol=lf` rule for {path} to .gitattributes"
    assert b"\r\n" not in (ROOT / path).read_bytes(), f"{path} has CRLF line ends"


def test_the_rule_reader_matches_as_git_does() -> None:
    rules = "*.sh text eol=lf\nplugins/x/bin/* text eol=lf\nREADME.md -text\n# *.md eol=crlf\n"
    assert eol_rule("scripts/a.sh", rules) == "lf"
    assert eol_rule("a.sh", rules) == "lf"
    assert eol_rule("plugins/x/bin/tool", rules) == "lf"
    assert eol_rule("bin/tool", rules) is None
    assert eol_rule("README.md", rules) is None
    assert eol_rule("x.sh", rules + "x.sh eol=crlf\n") == "crlf"
