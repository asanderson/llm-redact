"""Gate mutation-testing results against the equivalent-mutants allow-list.

Reads ``mutmut results`` (the run must have happened already) and fails when
any surviving mutant is NOT recorded in scripts/mutation_equivalents.py with
a justification. This is the "nothing survives silently" rule: every survivor
is either killed by a test or explicitly, reviewably declared equivalent.

Also fails on STALE allow-list entries — an id that no longer survives (the
mutant is now killed, or renumbered by a source edit) must be removed so the
list never accumulates dead justifications. The one exception is
OSCILLATING_MUTANTS: reviewed equivalents whose survived/killed/timeout
status flips run to run under mutmut's coverage-based test selection and
timing-based per-mutant time limit — those may be absent from a given run's
survivor list without being stale.

Exit codes: 0 clean, 1 unlisted survivors or stale entries, 2 harness error.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from mutation_equivalents import EQUIVALENT_MUTANTS, OSCILLATING_MUTANTS  # noqa: E402


def main() -> int:
    # Resolve mutmut next to the running interpreter (works under `uv run`
    # and a bare venv python alike, regardless of ambient PATH).
    mutmut = Path(sys.executable).parent / "mutmut"
    proc = subprocess.run([str(mutmut), "results"], capture_output=True, text=True, cwd=ROOT)
    if proc.returncode not in (0, 1):  # mutmut exits 1 when survivors exist
        print(f"mutation gate: `mutmut results` failed:\n{proc.stderr}", file=sys.stderr)
        return 2
    survivors = {
        line.strip().split(":")[0] for line in proc.stdout.splitlines() if ": survived" in line
    }
    allowed = set(EQUIVALENT_MUTANTS)

    if not OSCILLATING_MUTANTS.issubset(allowed):
        orphans = sorted(OSCILLATING_MUTANTS - allowed)
        print(
            "mutation gate: OSCILLATING_MUTANTS entries missing a justification"
            f" in EQUIVALENT_MUTANTS: {orphans}",
            file=sys.stderr,
        )
        return 2
    unlisted = sorted(survivors - allowed)
    stale = sorted(allowed - survivors - OSCILLATING_MUTANTS)
    if unlisted:
        print(f"mutation gate: {len(unlisted)} surviving mutant(s) not in the allow-list:")
        for sid in unlisted:
            print(f"  {sid}")
        print(
            "\nEach must be KILLED with a test or added to"
            " scripts/mutation_equivalents.py with a reviewed justification."
        )
    if stale:
        print(f"mutation gate: {len(stale)} stale allow-list entr(y/ies) — no longer surviving:")
        for sid in stale:
            print(f"  {sid}")
        print("\nRemove them (killed or renumbered mutants must not keep justifications).")
    if unlisted or stale:
        return 1
    print(
        f"mutation gate: clean — {len(survivors)} survivor(s), all reviewed equivalents;"
        " every other mutant killed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
