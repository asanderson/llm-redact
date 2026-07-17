"""Complexity-coverage gate: every branching function must be executed.

The rule (Phase 26): every function in src/llm_redact with a cyclomatic
complexity ABOVE 1 — i.e. any function containing at least one decision
point — must be executed by the test suite. Straight-line CC=1 functions
(Protocol stubs, trivial forwarders) are exempt by construction; anything
that branches is not allowed to reach a release with zero test execution.

Two deliberate design points:

- The cyclomatic counter is a ~60-line stdlib `ast` visitor rather than a
  radon dependency: the number gates CI, so it must be deterministic,
  reviewable, and pinned by tests/test_complexity_gate.py. McCabe-style:
  1 + one per `if`/`elif`/ternary, loop, `except` handler, `assert`,
  `match` case, comprehension clause and comprehension `if`, plus
  short-circuit operands (`a and b or c` = +2). `with` is not a branch.
  Nested functions are counted separately, never into their parent.
- "Executed" means at least one line of the function BODY (first
  statement onward — the `def` line itself executes at import time even
  for functions nothing ever calls) appears in the coverage data from a
  `coverage run -m pytest` pass.

Modes: report (default) prints the census and the uncovered list;
`--check` exits 1 when an uncovered CC>1 function is not in
scripts/complexity_allowlist.py, and treats STALE allowlist entries
(now covered, or no longer existing) as errors too — the same
keep-the-ledger-honest discipline as the mutation gate's allow-list.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

_BRANCH_NODES = (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.Assert)
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def cyclomatic_complexity(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """McCabe-style complexity of ONE function, nested scopes excluded."""
    count = 1

    def visit(node: ast.AST) -> None:
        nonlocal count
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPE_NODES):
                continue  # nested functions/classes are their own entries
            if isinstance(child, (*_BRANCH_NODES, ast.ExceptHandler)):
                count += 1
            elif isinstance(child, ast.BoolOp):
                count += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                count += 1 + len(child.ifs)
            elif isinstance(child, ast.Match):
                count += len(child.cases)
            visit(child)

    visit(function)
    return count


@dataclass(frozen=True)
class FunctionInfo:
    qualified_name: str  # llm_redact.vault.SqliteVault.placeholder_for
    path: Path
    complexity: int
    body_start: int  # first statement line — NOT the def line
    end_line: int


def _body_start(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return function.body[0].lineno


def functions_in_file(path: Path, module: str) -> list[FunctionInfo]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[FunctionInfo] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{child.name}"
                found.append(
                    FunctionInfo(
                        qualified_name=name,
                        path=path,
                        complexity=cyclomatic_complexity(child),
                        body_start=_body_start(child),
                        end_line=child.end_lineno or child.lineno,
                    )
                )
                visit(child, name)
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}.{child.name}")
            else:
                visit(child, prefix)

    visit(tree, module)
    return found


def collect(src_root: Path) -> list[FunctionInfo]:
    functions: list[FunctionInfo] = []
    package_root = src_root.parent
    for path in sorted(src_root.rglob("*.py")):
        relative = path.relative_to(package_root).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        functions.extend(functions_in_file(path, ".".join(parts)))
    return functions


def executed_lines(coverage_file: Path) -> dict[Path, set[int]]:
    """Resolved-path -> executed line numbers, from a .coverage database."""
    from coverage import CoverageData

    data = CoverageData(basename=str(coverage_file))
    data.read()
    lines: dict[Path, set[int]] = {}
    for measured in data.measured_files():
        file_lines = data.lines(measured)
        if file_lines:
            lines[Path(measured).resolve()] = set(file_lines)
    return lines


def is_covered(function: FunctionInfo, lines: dict[Path, set[int]]) -> bool:
    file_lines = lines.get(function.path.resolve(), set())
    return any(function.body_start <= line <= function.end_line for line in file_lines)


def load_allowlist() -> dict[str, str]:
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from complexity_allowlist import ALLOWED_UNCOVERED

        return dict(ALLOWED_UNCOVERED)
    finally:
        sys.path.pop(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("src/llm_redact"))
    parser.add_argument(
        "--coverage-file",
        type=Path,
        default=Path(".coverage"),
        help="database written by `coverage run -m pytest`",
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 on unallowed uncovered functions"
    )
    args = parser.parse_args(argv)

    functions = collect(args.src)
    branching = [f for f in functions if f.complexity > 1]
    print(f"functions: {len(functions)} total, {len(branching)} with complexity > 1")

    if not args.coverage_file.exists():
        print(f"no coverage data at {args.coverage_file} (run `coverage run -m pytest` first)")
        return 2 if args.check else 0

    lines = executed_lines(args.coverage_file)
    uncovered = sorted(f.qualified_name for f in branching if not is_covered(f, lines))
    covered_count = len(branching) - len(uncovered)
    print(f"executed by the suite: {covered_count}/{len(branching)}")

    allowlist = load_allowlist()
    unallowed = [name for name in uncovered if name not in allowlist]
    # An entry is stale the moment it stops being needed — the function is
    # covered now, or it no longer exists. Either way it must leave the
    # ledger, or the allow-list quietly rots into a bypass.
    stale = [name for name in sorted(allowlist) if name not in uncovered]
    if uncovered:
        print("\nuncovered branching functions:")
        for name in uncovered:
            marker = "  (allowlisted)" if name in allowlist else ""
            print(f"  {name}{marker}")
    if args.check:
        failed = False
        if unallowed:
            print(f"\ncomplexity gate: {len(unallowed)} uncovered function(s) not allow-listed")
            failed = True
        if stale:
            print(
                f"\ncomplexity gate: {len(stale)} STALE allow-list entr(y/ies)"
                " (now covered or gone) — remove them:"
            )
            for name in stale:
                print(f"  {name}")
            failed = True
        if failed:
            return 1
        print("complexity gate: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
