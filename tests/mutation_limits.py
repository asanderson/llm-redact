"""A memory ceiling for mutmut's per-mutant test processes.

mutmut forks one process per mutant and bounds only its CPU time. A mutant
that turns a parser loop into an endless one which also allocates (a
multipart ``parse`` whose boundary search restarts at offset 0 appends parts
forever) grows by tens of MB a second, so before the CPU limit fires four
such workers exhaust a CI runner's memory and the runner is shut down — the
whole mutation job dies with no result. Capped, the runaway mutant gets a
MemoryError in seconds: its tests fail and it counts as killed.

The ceiling bounds the process's writable memory (``RLIMIT_DATA``: private
writable mappings, the heap included), not its address space
(``RLIMIT_AS``). Address space also counts what a thread only reserves: glibc
maps each thread's malloc arena as 64 MiB of inaccessible address space, and
the 220 tests mutmut runs for a ``MapWriter.__init__`` mutant (threaded
vault-writer and proxy tests) reach about 2.3 GiB of it above their start.
Under an address-space ceiling a thread then failed to start ("can't start
new thread"), the test failed, and an equivalent mutant read as killed: the
gate saw a stale allow-list entry (the 1.13.0 release commit's run), and a
real survivor could as well have passed as killed. Reservations are not
writable memory, while everything a runaway loop allocates is.
"""

from __future__ import annotations

from collections.abc import Mapping

# mutmut's own phases (clean run, stats, listing, generation) run the real
# code: they are never capped.
_NOT_A_MUTANT = frozenset({"", "fail", "stats", "list_all_tests", "mutant_generation"})

# Room above the process's writable memory at fork for one mutant's tests:
# the suite's own growth stays far below it.
HEADROOM_BYTES = 2 * 1024**3


def mutant_memory_limit(environ: Mapping[str, str], data_size: int | None) -> int | None:
    """The writable-memory ceiling for this process, or None for no ceiling:
    only a mutmut mutant run (``MUTANT_UNDER_TEST`` names a mutant) whose
    current size is known gets one."""
    mutant = environ.get("MUTANT_UNDER_TEST")
    if mutant is None or mutant in _NOT_A_MUTANT or data_size is None:
        return None
    return data_size + HEADROOM_BYTES


def current_data_size() -> int | None:
    """This process's writable memory in bytes (``VmData``, the size
    ``RLIMIT_DATA`` bounds; Linux ``/proc``), else None."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmData:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def apply(environ: Mapping[str, str]) -> int | None:
    """Cap this process when it runs a mutant (POSIX only); the ceiling set."""
    limit = mutant_memory_limit(environ, current_data_size())
    if limit is None:
        return None
    try:
        import resource
    except ImportError:  # Windows: no rlimits (mutmut does not run there)
        return None
    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))
    return limit
