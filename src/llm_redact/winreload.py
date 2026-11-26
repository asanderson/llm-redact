"""``llm-redact reload``: SIGHUP's reload on Windows, which has no SIGHUP.

The proxy creates an auto-reset named event, ``Local\\llm-redact-reload-<port>``,
and a thread of its own waits on it; each time it is set the thread asks the
reload scheduler for a reload (``loop.call_soon_threadsafe(state.reloads.
request)``), exactly what the SIGHUP handler does on Linux and macOS: the
file read and a changed ``[detection]`` built off the event loop, signals
arriving meanwhile coalesced into one more reload, a bad file logged with
the running configuration kept. ``llm-redact reload`` opens the event and
sets it.

Who can reach it: ``Local\\`` is the namespace of the Windows session the
proxy runs in, so a process in another session (another user signed in at
the same time, a remote desktop) cannot open the event, and the default
security of an object the proxy creates gives the right to set it to the
account that started the proxy (and to SYSTEM and administrators) — the
reach ``kill -HUP`` has on POSIX. A name some other program created first
is never used: the proxy logs a WARNING and runs without ``llm-redact
reload`` (a restart applies a new configuration).

The kernel32 binding is a thin ctypes one; the listener logic runs against
a ``ReloadEvents`` stand-in on every OS.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from collections.abc import Callable
from typing import Protocol

logger = logging.getLogger("llm_redact")

EVENT_PREFIX = "Local\\llm-redact-reload-"
# How long stop() waits for the listener's thread (it wakes at once).
STOP_SECONDS = 5.0

_POSIX_RELOAD = (
    "On Linux and macOS a running proxy reloads its configuration on SIGHUP:"
    " kill -HUP <pid of `llm-redact serve`> (e.g. `pkill -HUP -f 'llm-redact serve'`;"
    " for the systemd unit `systemctl --user kill -s HUP llm-redact.service`, for the"
    " LaunchAgent `launchctl kill SIGHUP gui/$(id -u)/com.llm-redact`)"
)


def event_name(port: int) -> str:
    """The reload event of the proxy listening on ``port``."""
    return f"{EVENT_PREFIX}{port}"


class EventTaken(OSError):
    """The event's name already exists: some other program created it."""


class ReloadEvents(Protocol):
    """The reload event and the listener's own stop event."""

    def wait(self) -> bool:
        """Block until either is set: True for a reload asked for, False
        once ``stop`` was called. Raises OSError when waiting fails."""
        ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class ReloadListener:
    """A thread that calls ``on_signal`` (from that thread) each time the
    reload event is set, until ``stop``. A failing ``on_signal`` is logged
    by type and the thread waits on; a failing wait ends it, with one
    WARNING."""

    def __init__(self, events: ReloadEvents, on_signal: Callable[[], object]) -> None:
        self._events = events
        self._on_signal = on_signal
        self._thread = threading.Thread(
            target=self._run, name="llm-redact-reload-listener", daemon=True
        )

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                asked = self._events.wait()
            except OSError as problem:
                logger.warning(
                    "llm-redact reload stopped working (%s); a restart applies a new configuration",
                    type(problem).__name__,
                )
                return
            if not asked:
                return
            try:
                self._on_signal()
            except Exception as problem:  # noqa: BLE001 — the listener lives on
                logger.warning(
                    "llm-redact reload could not ask for a reload (%s)", type(problem).__name__
                )

    def stop(self) -> None:
        """Wake the thread, wait for it (bounded), close the events."""
        self._events.stop()
        if self._thread.is_alive():
            self._thread.join(STOP_SECONDS)
        self._events.close()


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _EVENT_MODIFY_STATE = 0x0002
    _ERROR_FILE_NOT_FOUND = 2
    _ERROR_ALREADY_EXISTS = 183
    _INFINITE = 0xFFFFFFFF
    _WAIT_OBJECT_0 = 0

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateEventW.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    _kernel32.CreateEventW.restype = wintypes.HANDLE
    _kernel32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _kernel32.OpenEventW.restype = wintypes.HANDLE
    _kernel32.SetEvent.argtypes = [wintypes.HANDLE]
    _kernel32.SetEvent.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.WaitForMultipleObjects.argtypes = [
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    _kernel32.WaitForMultipleObjects.restype = wintypes.DWORD

    def _windows_error() -> OSError:
        code = ctypes.get_last_error()
        return ctypes.WinError(code)

    def _create_event(name: str | None, *, manual_reset: bool) -> int:
        ctypes.set_last_error(0)
        handle = _kernel32.CreateEventW(None, manual_reset, False, name)
        if not handle:
            raise _windows_error()
        if name is not None and ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
            _kernel32.CloseHandle(handle)
            raise EventTaken(name)
        return int(handle)

    class _NamedEvents:
        """The named auto-reset reload event and an unnamed stop event."""

        def __init__(self, port: int) -> None:
            self._reload = _create_event(event_name(port), manual_reset=False)
            try:
                self._stop = _create_event(None, manual_reset=True)
            except OSError:
                _kernel32.CloseHandle(self._reload)
                raise
            self._handles = (wintypes.HANDLE * 2)(self._reload, self._stop)
            self._closed = False

        def wait(self) -> bool:
            ended = _kernel32.WaitForMultipleObjects(2, self._handles, False, _INFINITE)
            if ended == _WAIT_OBJECT_0:
                return True
            if ended == _WAIT_OBJECT_0 + 1:
                return False
            raise _windows_error()

        def stop(self) -> None:
            _kernel32.SetEvent(self._stop)

        def close(self) -> None:
            if not self._closed:
                self._closed = True
                _kernel32.CloseHandle(self._reload)
                _kernel32.CloseHandle(self._stop)

    def _open_listening(port: int) -> ReloadEvents:
        return _NamedEvents(port)

    def _signal(port: int) -> bool:
        handle = _kernel32.OpenEventW(_EVENT_MODIFY_STATE, False, event_name(port))
        if not handle:
            if ctypes.get_last_error() == _ERROR_FILE_NOT_FOUND:
                return False
            raise _windows_error()
        try:
            if not _kernel32.SetEvent(handle):
                raise _windows_error()
        finally:
            _kernel32.CloseHandle(handle)
        return True

else:

    def _open_listening(port: int) -> ReloadEvents:
        raise OSError(f"no named events on {sys.platform}")

    def _signal(port: int) -> bool:
        raise OSError(f"no named events on {sys.platform}")


def open_listening(port: int) -> ReloadEvents:
    """The proxy's events (Windows): raises ``EventTaken`` when another
    program holds the name, OSError when Windows refuses."""
    return _open_listening(port)


def signal(port: int) -> bool:
    """Set the reload event of the proxy on ``port`` (Windows): False when
    no proxy in this session listens there; OSError when Windows refuses."""
    return _signal(port)


def start_listener(port: int, on_signal: Callable[[], object]) -> ReloadListener | None:
    """On Windows, the started listener for the proxy on ``port``; None
    elsewhere, or when the event cannot be created (a WARNING says so)."""
    if sys.platform != "win32":
        return None
    try:
        events = open_listening(port)
    except EventTaken as taken:
        logger.warning(
            "another program holds the event %s, so llm-redact reload will not reach this"
            " proxy (a restart applies a new configuration)",
            taken,
        )
        return None
    except OSError as problem:
        logger.warning(
            "llm-redact reload is unavailable (%s); a restart applies a new configuration",
            type(problem).__name__,
        )
        return None
    listener = ReloadListener(events, on_signal)
    listener.start()
    return listener


def run_reload(args: argparse.Namespace) -> int:
    """``llm-redact reload [--config PATH] [--port N]``: ask the proxy on
    the port (the config's, unless given) to reload its configuration."""
    if sys.platform != "win32":
        print(_POSIX_RELOAD)
        return 2
    port = args.port
    if port is None:
        from llm_redact.config import ConfigError, apply_env_overrides, load_config

        try:
            port = apply_env_overrides(load_config(args.config)).port
        except ConfigError as problem:
            print(f"reload: {problem}")
            return 2
    try:
        found = signal(port)
    except OSError as problem:
        reason = problem.strerror or type(problem).__name__
        print(f"reload: cannot reach the proxy on port {port} ({reason})")
        return 1
    if not found:
        print(f"reload: no proxy started in this logon session listens on port {port}")
        return 1
    print(
        f"reload: asked the proxy on port {port} to reload its configuration;"
        " `llm-redact status` shows when it applied (config_reload)"
    )
    return 0
