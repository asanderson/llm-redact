"""Hypothesis properties of the persistent vault: batching and deletion.

- A batch issues EXACTLY the tokens the per-call path issues for the same
  operations (and the in-memory reference model), whatever the batch
  boundaries — batching changes when rows are committed, never what they are.
- Across any interleaving of per-call writes, committed and rolled-back
  batches, whole-session deletes (forget, prune) by either of two proxy
  instances sharing one sqlite file, and restores through either instance at
  any clock time, a token is only ever restored to the one value it was
  issued for — never another (the never-wrong-value invariant across
  deletion). Tokens issued inside a rolled-back batch were never forwarded,
  so they carry no meaning and their numbers may go to other values.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from fake_cipher import FakeVaultCipher
from llm_redact.vault import (
    CACHE_CHECK_SECONDS,
    InMemoryVault,
    SqliteVaultManager,
    open_sqlite_vault,
    run_batched,
)

_TYPES = ("EMAIL", "PHONE")
_values = st.sampled_from([f"v{i}@corp.example" for i in range(8)])
_ops = st.lists(
    st.tuples(st.sampled_from(_TYPES), _values, st.integers(min_value=0, max_value=6)),
    min_size=1,
    max_size=25,
)


@settings(deadline=None, max_examples=60)
@given(
    ops=_ops, cuts=st.lists(st.integers(min_value=0, max_value=25), max_size=5), enc=st.booleans()
)
def test_a_batch_issues_exactly_the_per_call_tokens(
    ops: list[tuple[str, str, int]], cuts: list[int], enc: bool
) -> None:
    reference = InMemoryVault()
    expected = [reference.placeholder_for(t, v, floor=f) for t, v, f in ops]
    with tempfile.TemporaryDirectory() as directory:
        cipher = FakeVaultCipher() if enc else None
        per_call = open_sqlite_vault(Path(directory) / "a.db", "s", cipher)
        batched = open_sqlite_vault(Path(directory) / "b.db", "s", cipher)
        assert [per_call.placeholder_for(t, v, floor=f) for t, v, f in ops] == expected
        # The same operations cut into batches at arbitrary boundaries.
        bounds = sorted({0, len(ops), *(min(cut, len(ops)) for cut in cuts)})
        got: list[str] = []
        for start, end in zip(bounds, bounds[1:], strict=False):
            chunk = ops[start:end]
            got += run_batched(
                batched,
                lambda chunk=chunk: [batched.placeholder_for(t, v, floor=f) for t, v, f in chunk],
            )
        assert got == expected
        # Committed alike: every token restores through both vaults.
        for (_, value, _), token in zip(ops, expected, strict=True):
            assert per_call.original_for(token) == value
            assert batched.original_for(token) == value
        per_call.close()
        batched.close()


class _Refused(Exception):
    pass


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


_SESSIONS = ("s1", "s2")


class RetiredNumbersMachine(RuleBasedStateMachine):
    """Two instances over one sqlite file; see the module docstring."""

    @initialize(encrypted=st.booleans())
    def setup(self, encrypted: bool) -> None:
        self.directory = tempfile.mkdtemp(prefix="llm-redact-vault-machine-")
        path = Path(self.directory) / "vault.db"
        cipher = FakeVaultCipher() if encrypted else None
        self.clock = _Clock()
        self.instances = [
            SqliteVaultManager(path, cipher=cipher, clock=self.clock),
            SqliteVaultManager(path, cipher=cipher, clock=self.clock),
        ]
        # (session, token) -> the one value it was ever forwarded for.
        self.meaning: dict[tuple[str, str], str] = {}

    def teardown(self) -> None:
        for manager in getattr(self, "instances", []):
            manager.close()
        shutil.rmtree(getattr(self, "directory", ""), ignore_errors=True)

    def _issued(self, session: str, value: str, token: str) -> None:
        prior = self.meaning.setdefault((session, token), value)
        assert prior == value, f"{token} issued for two values in {session}"

    @rule(
        instance=st.integers(0, 1),
        session=st.sampled_from(_SESSIONS),
        detector_type=st.sampled_from(_TYPES),
        value=_values,
        floor=st.integers(min_value=0, max_value=4),
    )
    def write(
        self, instance: int, session: str, detector_type: str, value: str, floor: int
    ) -> None:
        view = self.instances[instance].get(session)
        self._issued(session, value, view.placeholder_for(detector_type, value, floor=floor))

    @rule(
        instance=st.integers(0, 1),
        session=st.sampled_from(_SESSIONS),
        items=st.lists(st.tuples(st.sampled_from(_TYPES), _values), min_size=1, max_size=4),
        refused=st.booleans(),
    )
    def write_batch(
        self, instance: int, session: str, items: list[tuple[str, str]], refused: bool
    ) -> None:
        view = self.instances[instance].get(session)

        def work() -> list[tuple[str, str]]:
            issued = [(value, view.placeholder_for(t, value)) for t, value in items]
            if refused:
                raise _Refused  # a blocked value: nothing is forwarded
            return issued

        if refused:
            with pytest.raises(_Refused):
                run_batched(view, work)
            return
        for value, token in run_batched(view, work):
            self._issued(session, value, token)

    @rule(instance=st.integers(0, 1), session=st.sampled_from(_SESSIONS))
    def forget(self, instance: int, session: str) -> None:
        self.instances[instance].forget_sessions([session])

    @rule(instance=st.integers(0, 1))
    def prune_everything(self, instance: int) -> None:
        manager = self.instances[instance]
        manager._conn.execute("UPDATE mappings SET created_at = '2000-01-01T00:00:00Z'")
        manager.prune_sessions(30)

    @rule(seconds=st.sampled_from([0.0, CACHE_CHECK_SECONDS / 2, CACHE_CHECK_SECONDS]))
    def tick(self, seconds: float) -> None:
        self.clock.now += seconds

    @invariant()
    def a_token_restores_to_its_own_value_or_nothing(self) -> None:
        for manager in getattr(self, "instances", []):
            for (session, token), value in self.meaning.items():
                restored = manager.get(session).original_for(token)
                assert restored in (None, value), (session, token, restored, value)


RetiredNumbersMachine.TestCase.settings = settings(
    deadline=None, max_examples=40, stateful_step_count=25
)
test_retired_numbers_machine = RetiredNumbersMachine.TestCase
