"""The mutmut memory ceiling (tests/mutation_limits.py)."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import mutation_limits

HERE = Path(__file__).resolve().parent
_MUTANT = "llm_redact.multipart.x_parse__mutmut_46"


@pytest.mark.parametrize("phase", ["", "fail", "stats", "list_all_tests", "mutant_generation"])
def test_mutmut_phases_that_run_the_real_code_are_never_capped(phase: str) -> None:
    assert mutation_limits.mutant_memory_limit({"MUTANT_UNDER_TEST": phase}, 10**9) is None


def test_a_normal_test_run_is_never_capped() -> None:
    assert mutation_limits.mutant_memory_limit({}, 10**9) is None
    assert mutation_limits.apply({}) is None


def test_a_mutant_run_gets_its_size_at_fork_plus_the_headroom() -> None:
    environ = {"MUTANT_UNDER_TEST": _MUTANT}
    limit = mutation_limits.mutant_memory_limit(environ, 10**9)
    assert limit == 10**9 + mutation_limits.HEADROOM_BYTES
    assert mutation_limits.mutant_memory_limit(environ, None) is None


def test_the_current_size_is_read_where_proc_exists() -> None:
    size = mutation_limits.current_data_size()
    assert size is None or size > 0


def _capped(body: str) -> subprocess.CompletedProcess[str]:
    """``body`` run in a fresh interpreter capped as a mutant run is."""
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(HERE)!r})
        import mutation_limits
        assert mutation_limits.apply({{"MUTANT_UNDER_TEST": {_MUTANT!r}}}) is not None
        """
    ) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False
    )


_LINUX = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RLIMIT_DATA covers mmap on Linux"
)


@_LINUX
def test_reserved_address_space_does_not_count_against_the_ceiling() -> None:
    # What threads reserve (glibc's per-thread malloc arenas, unused stack) is
    # address space, not writable memory: an address-space ceiling made a
    # thread fail to start under an equivalent MapWriter mutant, which then
    # read as killed. Three times the headroom, reserved the way glibc
    # reserves an arena (private, inaccessible, not committed), fits.
    result = _capped(
        f"""
        import mmap
        flags = mmap.MAP_PRIVATE | getattr(mmap, "MAP_NORESERVE", 0)
        size = {3 * mutation_limits.HEADROOM_BYTES}
        reserved = mmap.mmap(-1, size, flags=flags, prot=0)  # prot=0: PROT_NONE
        print("reserved")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "reserved"


@_LINUX
def test_writable_memory_past_the_headroom_is_refused() -> None:
    # What a runaway loop allocates is writable memory: an allocation past
    # the headroom is a MemoryError at once (nothing is touched first).
    result = _capped(
        f"""
        try:
            buffer = bytearray({mutation_limits.HEADROOM_BYTES + 2**28})
        except MemoryError:
            print("refused")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"
