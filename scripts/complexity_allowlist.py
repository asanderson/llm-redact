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

_ACL_BINDING = (
    "the advapi32 binding doctor reads NTFS access control lists through, called only"
    " under sys.platform == 'win32' (this gate runs on Linux); the Windows CI legs run it"
    " (tests/test_win_acl.py reads the real lists icacls sets, tests/test_private_files.py"
    " the lists it sets)"
)

_JOB_BINDING = (
    "the kernel32 Job Object binding that confines the extraction worker, called only"
    " under sys.platform == 'win32' (this gate runs on Linux); the Windows CI legs run it"
    " (tests/test_winjob.py confines real processes)"
)

ALLOWED_UNCOVERED: dict[str, str] = {
    "llm_redact.winjob.Job.close": _JOB_BINDING,
    "llm_redact.winjob._Kernel32.create": _JOB_BINDING,
    "llm_redact.winjob._Kernel32.assign": _JOB_BINDING,
    "llm_redact.winacl._Windows.make_private": _ACL_BINDING,
    "llm_redact.winacl._Windows.read_dacl": _ACL_BINDING,
    "llm_redact.winacl._Windows._sid_text": _ACL_BINDING,
    "llm_redact.winacl._Windows._account": _ACL_BINDING,
    "llm_redact.winacl._Windows.current_user_sid": _ACL_BINDING,
    "llm_redact.winreload._NamedEvents.__init__": _WINDOWS_BINDING,
    "llm_redact.winreload._NamedEvents.wait": _WINDOWS_BINDING,
    "llm_redact.winreload._NamedEvents.close": _WINDOWS_BINDING,
    "llm_redact.winreload._create_event": _WINDOWS_BINDING,
    "llm_redact.winreload._signal": _WINDOWS_BINDING,
}
