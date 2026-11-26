"""Reviewed exemptions from the complexity-coverage gate.

Every entry is a branching (CC>1) function the test suite deliberately
does not execute, with the justification a reviewer signed off on. The
gate (scripts/complexity_gate.py --check) fails on uncovered functions
missing from this dict AND on entries that stop being needed — the
ledger can only shrink honestly.

Keys are qualified names as the gate prints them
(llm_redact.module.Class.function).
"""

_WINDOWS_BINDING = (
    "the kernel32 binding of `llm-redact reload`, defined only under"
    " sys.platform == 'win32' (this gate runs on Linux); the Windows CI legs run it"
    " (tests/test_reload_windows.py sets the real event of a running proxy)"
)

ALLOWED_UNCOVERED: dict[str, str] = {
    "llm_redact.winreload._NamedEvents.__init__": _WINDOWS_BINDING,
    "llm_redact.winreload._NamedEvents.wait": _WINDOWS_BINDING,
    "llm_redact.winreload._NamedEvents.close": _WINDOWS_BINDING,
}
