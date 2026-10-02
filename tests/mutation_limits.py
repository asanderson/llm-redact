"""A memory ceiling for mutmut's per-mutant test processes.

mutmut forks one process per mutant and bounds only its CPU time. A mutant
that turns a parser loop into an endless one which also allocates (a
multipart ``parse`` whose boundary search restarts at offset 0 appends parts
forever) grows by tens of MB a second, so before the CPU limit fires four
such workers exhaust a CI runner's memory and the runner is shut down — the
whole mutation job dies with no result. Capped, the runaway mutant gets a
MemoryError in seconds: its tests fail and it counts as killed.
"""

from __future__ import annotations

from collections.abc import Mapping

# mutmut's own phases (clean run, stats, listing, generation) run the real
# code: they are never capped.
_NOT_A_MUTANT = frozenset({"", "fail", "stats", "list_all_tests", "mutant_generation"})

# Room above the process's address space at fork for one mutant's tests: the
# suite's own growth stays far below it.
HEADROOM_BYTES = 2 * 1024**3


def mutant_memory_limit(environ: Mapping[str, str], vm_size: int | None) -> int | None:
    """The address-space ceiling for this process, or None for no ceiling:
    only a mutmut mutant run (``MUTANT_UNDER_TEST`` names a mutant) whose
    current size is known gets one."""
    mutant = environ.get("MUTANT_UNDER_TEST")
    if mutant is None or mutant in _NOT_A_MUTANT or vm_size is None:
        return None
    return vm_size + HEADROOM_BYTES


def current_vm_size() -> int | None:
    """This process's virtual size in bytes (Linux ``/proc``), else None."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def apply(environ: Mapping[str, str]) -> int | None:
    """Cap this process when it runs a mutant (POSIX only); the ceiling set."""
    limit = mutant_memory_limit(environ, current_vm_size())
    if limit is None:
        return None
    try:
        import resource
    except ImportError:  # Windows: no rlimits (mutmut does not run there)
        return None
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return limit
