"""Renames and deletes that Windows refuses while another program holds a file.

On Windows a file another process has open cannot be deleted, replaced or
renamed (most programs open files without FILE_SHARE_DELETE), and neither
can a folder that holds one: ``os.replace``, ``os.rename``, ``os.unlink`` and
``shutil.rmtree`` fail with PermissionError, winerror 5 (ERROR_ACCESS_DENIED)
or 32 (ERROR_SHARING_VIOLATION). A virus scanner, the search indexer, a
backup or sync client or an editor holds a file for a moment after it is
written; a running proxy holds the model files it loaded for as long as it
runs. POSIX never refuses for that reason, so nothing is retried there.

``retried`` runs such an operation again, with backoff over about two
seconds, while Windows says the file is held; the last error is then raised
for the caller to explain (``is_held``).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable
from typing import TypeVar

_T = TypeVar("_T")

# ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION
HELD_WINERRORS = frozenset({5, 32})
# The waits between attempts: about two seconds in all.
DELAYS = (0.05, 0.1, 0.2, 0.4, 0.5, 0.75)


class HeldFile(OSError):
    """Windows kept refusing: another program holds the file. Its message
    says which file and what to do."""


def is_held(error: BaseException) -> bool:
    """Whether Windows refused because another program holds the file."""
    return (
        sys.platform == "win32"
        and isinstance(error, OSError)
        and getattr(error, "winerror", None) in HELD_WINERRORS
    )


def retried(
    operation: Callable[[], _T],
    *,
    delays: Iterable[float] | None = None,
    sleep: Callable[[float], object] | None = None,
) -> _T:
    """``operation()``, run again after each of ``delays`` (``DELAYS``)
    while it fails because Windows says the file is held; any other error
    at once."""
    wait = time.sleep if sleep is None else sleep
    for delay in DELAYS if delays is None else delays:
        try:
            return operation()
        except OSError as error:
            if not is_held(error):
                raise
        wait(delay)
    return operation()
