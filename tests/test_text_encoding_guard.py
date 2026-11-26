"""No text I/O in src/ depends on the platform's locale or line ends.

A text-mode file or pipe that names no encoding decodes and encodes with the
locale's code page: UTF-8 on most Linux and macOS installs, cp1252 on a
Western Windows install without ``PYTHONUTF8``. And a text-mode write that
names no ``newline`` turns every ``\\n`` into ``\\r\\n`` on Windows, so a
file hashed into a manifest, compared with a digest or carried to another
system differs by platform. So every call below in src/ names its encoding,
and every text-mode write also names ``newline`` (``"\\n"`` or ``""``) —
or the code writes bytes. There are no exemptions: a new call that misses
one fails this test.

``x.open(...)`` is read as ``Path.open`` when it names no argument but a
mode (none, a ``mode=``, or a first argument such as ``"w"``): an archive's
``open(member)`` (``zipfile``) opens bytes."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "llm_redact"

# Module names whose ``.open`` is not a file opened in text mode by default.
_BINARY_OPENERS = {"os", "tarfile", "zipfile", "gzip", "bz2", "lzma", "webbrowser", "sqlite3"}
_SUBPROCESS = {"run", "Popen", "call", "check_call", "check_output"}
_NEWLINES = ("\n", "")
_MODE = re.compile(r"[rwaxbt+]+")
_LOCALE_ONLY = "encoding (these decode with the locale: use subprocess.run)"
_TEMPFILES = {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _argument(call: ast.Call, keyword: str, position: int | None) -> ast.expr | None:
    for item in call.keywords:
        if item.arg == keyword:
            return item.value
    if position is not None and len(call.args) > position:
        return call.args[position]
    return None


def _literal(node: ast.expr | None) -> object:
    return node.value if isinstance(node, ast.Constant) else ...


def _mode_problems(call: ast.Call, mode_at: int, encoding_at: int, newline_at: int) -> list[str]:
    """What an ``open``-like call lacks, given where its mode, encoding and
    newline arguments sit positionally."""
    mode = _literal(_argument(call, "mode", mode_at))
    if mode is ...:  # absent: the default "r"
        mode = "r" if _argument(call, "mode", mode_at) is None else None
    if isinstance(mode, str) and "b" in mode:
        return []
    problems = []
    if _argument(call, "encoding", encoding_at) is None:
        problems.append("encoding")
    writes = mode is None or any(flag in mode for flag in "wax+")
    if writes and _literal(_argument(call, "newline", newline_at)) not in _NEWLINES:
        problems.append('newline="\\n"')
    return problems


def _path_open(call: ast.Call) -> bool:
    """Whether ``x.open(...)`` reads as ``Path.open``: no positional
    argument, or a first one that is a mode literal."""
    if not call.args:
        return True
    first = _literal(call.args[0])
    return isinstance(first, str) and _MODE.fullmatch(first) is not None


def _problems(call: ast.Call) -> list[str]:
    name = _dotted(call.func) or ""
    method = call.func.attr if isinstance(call.func, ast.Attribute) else None
    receiver = _dotted(call.func.value) if isinstance(call.func, ast.Attribute) else None
    if name in ("open", "io.open"):
        # open(file, mode, buffering, encoding, errors, newline)
        return _mode_problems(call, 1, 3, 5)
    if name == "os.fdopen":
        return _mode_problems(call, 1, 3, 5)
    if method == "open" and receiver not in _BINARY_OPENERS and _path_open(call):
        # Path.open(mode, buffering, encoding, errors, newline)
        return _mode_problems(call, 0, 2, 4)
    if method == "read_text":
        return [] if _argument(call, "encoding", 0) is not None else ["encoding"]
    if method == "write_text":
        problems = []
        if _argument(call, "encoding", 1) is None:
            problems.append("encoding")
        if _literal(_argument(call, "newline", 3)) not in _NEWLINES:
            problems.append('newline="\\n"')
        return problems
    if name.removeprefix("tempfile.") in _TEMPFILES:
        # NamedTemporaryFile(mode="w+b", buffering, encoding, newline, ...):
        # binary unless a mode says otherwise.
        if _argument(call, "mode", 0) is None:
            return []
        return _mode_problems(call, 0, 2, 3)
    if name in ("io.TextIOWrapper", "TextIOWrapper"):
        return [] if _argument(call, "encoding", 1) is not None else ["encoding"]
    if name in ("logging.FileHandler", "FileHandler"):
        return [] if _argument(call, "encoding", 2) is not None else ["encoding"]
    if name.startswith("subprocess.") and name.split(".")[-1] in _SUBPROCESS:
        text = any(
            item.arg in ("text", "universal_newlines") and _literal(item.value) is not False
            for item in call.keywords
        )
        if text and _argument(call, "encoding", None) is None:
            return ["encoding"]
        return []
    if name in ("subprocess.getoutput", "subprocess.getstatusoutput"):
        return [_LOCALE_ONLY]
    return []


def scan(source: str) -> list[tuple[int, str]]:
    """(line, what is missing) for every text I/O call in ``source`` that
    leaves the encoding or the newline to the platform."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            for problem in _problems(node):
                found.append((node.lineno, problem))
    return sorted(found)


def _src_files() -> Iterator[Path]:
    yield from sorted(SRC.rglob("*.py"))


def test_every_text_io_call_names_its_encoding_and_newline() -> None:
    found = {}
    for path in _src_files():
        for line, problem in scan(path.read_text(encoding="utf-8")):
            found.setdefault(f"{path.relative_to(SRC).as_posix()}:{line}", []).append(problem)
    assert found == {}, (
        "text I/O that depends on the platform: pass encoding= (UTF-8 unless the format"
        ' says otherwise) and, for a text-mode write, newline="\\n" — or write bytes'
    )


@pytest.mark.parametrize(
    ("source", "missing"),
    [
        ("path.read_text()", ["encoding"]),
        ('path.write_text(text, encoding="utf-8")', ['newline="\\n"']),
        ("path.write_text(text)", ["encoding", 'newline="\\n"']),
        ('open(name, "w", encoding="utf-8")', ['newline="\\n"']),
        ("open(name)", ["encoding"]),
        ('open(name, "a+", newline="\\n")', ["encoding"]),
        ('path.open("w", encoding="utf-8")', ['newline="\\n"']),
        ("path.open()", ["encoding"]),
        ('os.fdopen(fd, "w")', ["encoding", 'newline="\\n"']),
        ("open(name, mode)", ["encoding", 'newline="\\n"']),
        ("io.TextIOWrapper(raw)", ["encoding"]),
        ("logging.FileHandler(name)", ["encoding"]),
        ("subprocess.run(command, capture_output=True, text=True)", ["encoding"]),
        ("subprocess.check_output(command, universal_newlines=True)", ["encoding"]),
        ("subprocess.getoutput(command)", [_LOCALE_ONLY]),
        ('tempfile.NamedTemporaryFile("w")', ["encoding", 'newline="\\n"']),
        ('NamedTemporaryFile(mode="w+", encoding="utf-8")', ['newline="\\n"']),
    ],
)
def test_the_scan_catches_platform_dependent_text_io(source: str, missing: list[str]) -> None:
    assert [problem for _line, problem in scan(source)] == missing


@pytest.mark.parametrize(
    "source",
    [
        'path.read_text(encoding="utf-8")',
        'path.read_text("utf-8")',
        'path.write_text(text, encoding="utf-8", newline="\\n")',
        'path.write_text(text, "utf-8", None, "\\n")',
        "path.write_bytes(data)",
        'open(name, "rb")',
        'open(name, "wb")',
        'open(name, encoding="utf-8")',
        'open(name, "w", encoding="utf-8", newline="")',
        'path.open("rb")',
        "os.open(name, flags, 0o600)",
        "tarfile.open(name)",
        'zipfile.ZipFile(name).read("x")',
        'archive.open("mimetype")',
        "archive.open(info)",
        "webbrowser.open(url)",
        'io.TextIOWrapper(raw, encoding="utf-8")',
        "subprocess.run(command, capture_output=True)",
        'subprocess.run(command, capture_output=True, encoding="utf-8")',
        "subprocess.run(command, text=False)",
        "tempfile.NamedTemporaryFile()",
        'tempfile.TemporaryFile("w+b")',
    ],
)
def test_the_scan_leaves_explicit_and_binary_io_alone(source: str) -> None:
    assert scan(source) == [], source
