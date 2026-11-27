"""No rooted path without a drive reaches a file or process call in src/
unless it is listed here, with what keeps Windows away from it and the test
that proves it.

On Windows a rooted path without a drive ("/etc/llm-redact/config.toml",
"/dev/tty", "/proc/cpuinfo") names that location at the root of whatever
drive the working directory is on: ``C:\\etc\\...``, a USB stick's root, a
network share's root. Another local account can create folders there, so
reading such a file reads one someone else chose, writing it hands them the
content, and running it runs their program. A new rooted literal that
reaches a file or process call fails this test until it is listed.

What the scan sees: a string literal starting with "/" or "\\" (an f-string
too) passed to a call below — as an argument, an element of an argument
list, or through a name bound to it in the same function or at module
level — or joined to a path with ``/``. A value that travels through a
config field (an operator's setting, or a default such as the Kubernetes
service-account token path the vault KMS login reads) is not traced: it is
configuration, not a lookup the code makes on its own."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src" / "llm_redact"

# (file under src/llm_redact, literal) -> (what keeps Windows away from it,
# the test that proves it).
LISTED: dict[tuple[str, str], tuple[str, str]] = {
    ("config.py", "/etc/llm-redact/config.toml"): (
        "config_search_candidates leaves it out on Windows",
        "test_config_search.py::test_windows_never_touches_the_drive_root_config",
    ),
    ("override_cli.py", "/dev/tty"): (
        "_open_tty reads the Windows console there (CONIN$, CONOUT$), never /dev/tty",
        "test_override_cli.py::test_windows_reads_the_console_never_a_tty_file",
    ),
    ("fips.py", "/proc/sys/crypto/fips_enabled"): (
        "collect_fips_report reads it on Linux only",
        "test_fips_posture.py::test_the_kernel_flag_is_read_on_linux_only",
    ),
    ("bench/ner_latency.py", "/proc/cpuinfo"): (
        "cpu_model reads it on Linux only",
        "test_ner_bench.py::test_cpu_model_reads_proc_on_linux_only",
    ),
    ("platform_info.py", "/proc/sys/kernel/osrelease"): (
        "is_wsl reads it on Linux only",
        "test_platform_info.py::test_windows_never_reads_the_linux_files",
    ),
    ("platform_info.py", "/proc/self/mountinfo"): (
        "mount_of reads the system's table on Linux only",
        "test_platform_info.py::test_windows_never_reads_the_linux_files",
    ),
    ("platform_info.py", "/usr/bin/wslinfo"): (
        "networking_mode looks for it on Linux only",
        "test_platform_info.py::test_windows_never_reads_the_linux_files",
    ),
}

# Calls that open, stat, create, remove or run a path given as an argument.
_PATH_TYPES = {"Path", "PurePath", "PosixPath", "WindowsPath"}
_SINKS = {
    *_PATH_TYPES,
    "open",
    "io.open",
    "io.FileIO",
    "FileIO",
    "codecs.open",
    "os.open",
    "os.stat",
    "os.lstat",
    "os.access",
    "os.listdir",
    "os.scandir",
    "os.walk",
    "os.mkdir",
    "os.makedirs",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.rename",
    "os.replace",
    "os.chmod",
    "os.chown",
    "os.utime",
    "os.readlink",
    "os.symlink",
    "os.link",
    "os.startfile",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.spawnv",
    "os.spawnve",
    "os.path.exists",
    "os.path.lexists",
    "os.path.isfile",
    "os.path.isdir",
    "os.path.islink",
    "os.path.getsize",
    "os.path.getmtime",
    "os.path.realpath",
    "os.path.abspath",
    "os.path.join",
    "shutil.copy",
    "shutil.copy2",
    "shutil.copyfile",
    "shutil.copytree",
    "shutil.move",
    "shutil.rmtree",
    "shutil.which",
    "sqlite3.connect",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
}
# Methods that join their argument onto a path, whatever the receiver.
_JOINING_METHODS = {"joinpath"}


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _sink(call: ast.Call) -> bool:
    name = _dotted(call.func)
    if name is not None:
        name = name.removeprefix("pathlib.")
        if name in _SINKS:
            return True
    return isinstance(call.func, ast.Attribute) and call.func.attr in _JOINING_METHODS


def _rooted(node: ast.expr) -> str | None:
    """The literal, when ``node`` is a string literal starting with a slash
    or a backslash (an f-string: its source)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if node.value[:1] in ("/", "\\") else None
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and str(first.value)[:1] in ("/", "\\"):
            return ast.unparse(node)
    return None


def _bindings(body: list[ast.stmt]) -> dict[str, str]:
    """Names bound to a rooted literal in ``body`` (not in nested functions
    or classes)."""
    found: dict[str, str] = {}
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            literal = _rooted(node.value)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if literal is not None and isinstance(target, ast.Name):
                    found[target.id] = literal
        stack.extend(ast.iter_child_nodes(node))
    return found


class _Scan(ast.NodeVisitor):
    def __init__(self, module: ast.Module) -> None:
        self.scopes = [_bindings(module.body)]
        self.found: list[tuple[str, int]] = []

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scopes.append(_bindings(node.body))
        self.generic_visit(node)
        self.scopes.pop()

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def _resolve(self, node: ast.expr) -> str | None:
        literal = _rooted(node)
        if literal is None and isinstance(node, ast.Name):
            for scope in (self.scopes[-1], self.scopes[0]):
                if node.id in scope:
                    return scope[node.id]
        return literal

    def _note(self, node: ast.expr, line: int) -> None:
        candidates = node.elts if isinstance(node, ast.List | ast.Tuple) else [node]
        for candidate in candidates:
            literal = self._resolve(candidate)
            if literal is not None:
                self.found.append((literal, line))

    def visit_Call(self, node: ast.Call) -> None:
        if _sink(node):
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                self._note(argument, node.lineno)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # A rooted string either side of `/` joins as a path: a str has no
        # `/` of its own, and the rooted part replaces whatever came before.
        if isinstance(node.op, ast.Div):
            for side in (node.left, node.right):
                literal = _rooted(side)
                if literal is not None:
                    self.found.append((literal, node.lineno))
        self.generic_visit(node)


def scan(source: str) -> list[tuple[str, int]]:
    """(literal, line) for every rooted literal reaching a file or process
    call in ``source``."""
    module = ast.parse(source)
    visitor = _Scan(module)
    visitor.visit(module)
    return visitor.found


def _src_files() -> Iterator[Path]:
    yield from sorted(SRC.rglob("*.py"))


def test_every_rooted_path_lookup_is_listed() -> None:
    found: dict[tuple[str, str], list[int]] = {}
    for path in _src_files():
        where = path.relative_to(SRC).as_posix()
        for literal, line in scan(path.read_text(encoding="utf-8")):
            found.setdefault((where, literal), []).append(line)
    unlisted = {key: lines for key, lines in found.items() if key not in LISTED}
    assert unlisted == {}, (
        "a rooted path without a drive reaches a file or process call: on Windows it names"
        " a file at the root of the working directory's drive. Keep Windows away from it"
        " (a sys.platform check) and list it in LISTED with that check and its test."
    )
    assert sorted(set(LISTED) - set(found)) == [], "listed but no longer found: remove it"


def _test_functions(file: str) -> set[str]:
    module = ast.parse((TESTS / file).read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


@pytest.mark.parametrize("key", sorted(LISTED))
def test_every_listing_names_its_check_and_a_test_that_exists(key: tuple[str, str]) -> None:
    check, proof = LISTED[key]
    assert check
    file, _, function = proof.partition("::")
    assert function in _test_functions(file)


@pytest.mark.parametrize(
    "source",
    [
        'Path("/etc/llm-redact/x.toml")',
        'pathlib.Path("/etc/x")',
        'PurePath("/etc/x")',
        'Path(base, "/etc/x")',
        'open("/dev/tty", "r+")',
        'io.FileIO("/dev/tty", "r+")',
        'os.path.exists("/proc/x")',
        'os.path.join(base, "/etc")',
        'sqlite3.connect("/var/lib/x.db")',
        'subprocess.run(["/bin/sh", "-c", "true"])',
        'shutil.which("/usr/bin/env")',
        'base.joinpath("/etc")',
        'base / "/etc/x"',
        '"/etc" / base',
        'Path(f"/proc/{pid}/status")',
        'Path("\\\\etc\\\\x")',
        'FLAG = "/proc/x"\nopen(FLAG)',
        'def f():\n    flag = "/proc/x"\n    return open(flag)',
        'FLAG = "/proc/x"\ndef f():\n    return Path(FLAG).read_text()',
    ],
)
def test_the_scan_catches_a_rooted_lookup(source: str) -> None:
    assert scan(source), source


@pytest.mark.parametrize(
    "source",
    [
        'Path("relative/x")',
        'Path.home() / ".config"',
        'base / "llm-redact"',
        'route = "/v1/chat/completions"',
        'client.post("/v1/chat/completions")',
        'path.endswith("/approve")',
        '"/".join(parts)',
        'def f():\n    flag = "/proc/x"\n    return flag\ndef g(flag):\n    return open(flag)',
    ],
)
def test_the_scan_leaves_routes_and_relative_paths_alone(source: str) -> None:
    assert scan(source) == [], source
