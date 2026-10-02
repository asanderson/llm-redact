"""The mutmut memory ceiling (tests/mutation_limits.py)."""

import pytest

import mutation_limits


@pytest.mark.parametrize("phase", ["", "fail", "stats", "list_all_tests", "mutant_generation"])
def test_mutmut_phases_that_run_the_real_code_are_never_capped(phase: str) -> None:
    assert mutation_limits.mutant_memory_limit({"MUTANT_UNDER_TEST": phase}, 10**9) is None


def test_a_normal_test_run_is_never_capped() -> None:
    assert mutation_limits.mutant_memory_limit({}, 10**9) is None
    assert mutation_limits.apply({}) is None


def test_a_mutant_run_gets_its_size_at_fork_plus_the_headroom() -> None:
    environ = {"MUTANT_UNDER_TEST": "llm_redact.multipart.x_parse__mutmut_46"}
    limit = mutation_limits.mutant_memory_limit(environ, 10**9)
    assert limit == 10**9 + mutation_limits.HEADROOM_BYTES
    assert mutation_limits.mutant_memory_limit(environ, None) is None


def test_the_current_size_is_read_where_proc_exists() -> None:
    size = mutation_limits.current_vm_size()
    assert size is None or size > 0
