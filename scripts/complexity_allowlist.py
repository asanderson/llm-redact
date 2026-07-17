"""Reviewed exemptions from the complexity-coverage gate.

Every entry is a branching (CC>1) function the test suite deliberately
does not execute, with the justification a reviewer signed off on. The
gate (scripts/complexity_gate.py --check) fails on uncovered functions
missing from this dict AND on entries that stop being needed — the
ledger can only shrink honestly.

Keys are qualified names as the gate prints them
(llm_redact.module.Class.function).
"""

ALLOWED_UNCOVERED: dict[str, str] = {}
