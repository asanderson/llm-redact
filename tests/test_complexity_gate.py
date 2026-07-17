"""The complexity-coverage gate's own logic, pinned.

The gate number decides CI outcomes, so the cyclomatic counter is treated
like a checksum table: every construct it counts gets an explicit case,
and the covered/uncovered join is exercised on a synthetic module with
hand-built executed-line sets.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts" / "complexity_gate.py"
_spec = importlib.util.spec_from_file_location("complexity_gate", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules["complexity_gate"] = gate
_spec.loader.exec_module(gate)


def _cc(source: str) -> int:
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
    return gate.cyclomatic_complexity(function)


def test_straight_line_is_one() -> None:
    assert _cc("def f():\n    return 1\n") == 1


def test_if_elif_else_counts_branches_not_else() -> None:
    source = """
    def f(x):
        if x == 1:
            return "a"
        elif x == 2:
            return "b"
        else:
            return "c"
    """
    assert _cc(source) == 3  # if + elif; else is not a decision


def test_boolop_counts_short_circuit_operands() -> None:
    assert _cc("def f(a, b, c):\n    return a and b or c\n") == 3  # 1 + (2-1) + (2-1)


def test_loops_asserts_ternary_and_except() -> None:
    source = """
    def f(items):
        for item in items:          # +1
            while item:             # +1
                item -= 1
        assert items                # +1
        try:
            return items[0] if items else None   # +1 (ternary)
        except IndexError:          # +1
            return None
    """
    assert _cc(source) == 6


def test_comprehension_clauses_and_ifs() -> None:
    source = """
    def f(rows):
        return [x for row in rows for x in row if x]  # +2 clauses, +1 if
    """
    assert _cc(source) == 4


def test_match_counts_cases() -> None:
    source = """
    def f(x):
        match x:
            case 1:
                return "a"
            case _:
                return "b"
    """
    assert _cc(source) == 3


def test_with_is_not_a_branch() -> None:
    source = """
    def f(path):
        with open(path) as handle:
            return handle.read()
    """
    assert _cc(source) == 1


def test_nested_functions_counted_separately() -> None:
    source = """
    def outer(x):
        def inner(y):
            if y:
                return 1
            return 0
        if x:
            return inner(x)
        return 0
    """
    assert _cc(source) == 2  # only outer's own `if`; inner's is inner's


_MODULE = """\
def straight():
    return 1


def branching(x):
    if x:
        return 1
    return 0


class Thing:
    def method(self, x):
        if x:
            return "yes"
        return "no"


def never_called(x):
    if x:
        return 1
    return 0
"""


def _collected(tmp_path: Path) -> list:
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "mod.py").write_text(_MODULE)
    return gate.collect(package)


def test_collect_names_complexities_and_body_lines(tmp_path: Path) -> None:
    functions = {f.qualified_name: f for f in _collected(tmp_path)}
    assert functions["pkg.mod.straight"].complexity == 1
    assert functions["pkg.mod.branching"].complexity == 2
    assert functions["pkg.mod.Thing.method"].complexity == 2
    # body_start is the first STATEMENT, not the def line — a def line
    # "executes" at import even for functions nothing calls.
    assert functions["pkg.mod.branching"].body_start == 6
    assert functions["pkg.mod.branching"].end_line == 8


def test_is_covered_joins_body_lines_only(tmp_path: Path) -> None:
    functions = {f.qualified_name: f for f in _collected(tmp_path)}
    branching = functions["pkg.mod.branching"]
    never = functions["pkg.mod.never_called"]
    module_path = branching.path.resolve()
    # The import executed every def line; only branching's body ran.
    executed = {module_path: {1, 5, 6, 7, 11, 12, 18}}
    assert gate.is_covered(branching, executed) is True
    # never_called: def line 18 executed at import, body 19-21 never — the
    # def-line trap the gate exists to see through.
    assert gate.is_covered(never, executed) is False


def test_gate_counts_real_tree_smoke() -> None:
    functions = gate.collect(Path("src/llm_redact"))
    branching = [f for f in functions if f.complexity > 1]
    # Coarse pins: the tree is large, and the counter finding drastically
    # fewer functions than exist would mean a broken walker, not a
    # refactored codebase.
    assert len(functions) > 500
    assert len(branching) > 300
