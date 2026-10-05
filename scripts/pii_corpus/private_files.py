"""Where generated and verified rows may be written, and how.

The corpus is private (plan D7): every file this tooling writes that holds
text goes OUTSIDE every git work tree (the rule ``--dump-errors`` of the NER
bench follows), into a mode-0700 directory as a mode-0600 file, never
through a symlink. The default directory is
``${XDG_DATA_HOME:-~/.local/share}/llm-redact/pii-corpus``.
"""

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import IO, Any

from llm_redact.bench.ner import git_work_tree


class CorpusError(Exception):
    """A corpus file cannot be read or written. Messages name files, line
    numbers and keys, never row text."""


def default_data_dir(environ: Mapping[str, str] = os.environ) -> Path:
    base = environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "llm-redact" / "pii-corpus"


def output_problem(path: Path) -> str | None:
    """Why ``path`` may not hold corpus text (None = it may)."""
    tree = git_work_tree(path.parent)
    if tree is None:
        return None
    return (
        f"{path} is inside the git work tree {tree}; corpus files hold text and must"
        " never be committed: write them outside the repository"
    )


def open_private(path: Path, *, overwrite: bool) -> IO[str]:
    """A new mode-0600 text file (its directory created mode 0700). An
    existing file is refused unless ``overwrite``; a symlink always is."""
    problem = output_problem(path)
    if problem is not None:
        raise CorpusError(problem)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_TRUNC if overwrite else os.O_EXCL
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise CorpusError(f"{path} exists; pass --force to replace it") from exc
    except OSError as exc:
        raise CorpusError(f"cannot write {path}: {type(exc).__name__}") from exc
    if hasattr(os, "fchmod"):  # an overwritten file keeps its old mode otherwise
        os.fchmod(fd, 0o600)
    return os.fdopen(fd, "w", encoding="utf-8")


def write_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    with open_private(path, overwrite=overwrite) as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """(line number, object) for every non-blank line; a line that is not a
    JSON object raises, naming the line."""
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise CorpusError(f"cannot read {path}: {type(exc).__name__}") from exc
    with handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise CorpusError(f"{path} line {number}: not JSON") from exc
            if not isinstance(value, dict):
                raise CorpusError(f"{path} line {number}: not a JSON object")
            yield number, value
