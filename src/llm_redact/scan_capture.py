"""Read-only capture of the strings a request's redaction scans, for the
sinks a plugin registers (``plugin_api.ScanSink``, built once at startup by
``Registry.build_scan_sinks``; the Free core registers none).

With no sink this module is never reached: ``ProxyState.scan_capture`` is
None, no redactor records anything and nothing is queued — the request path
is the one without the seam. With sinks:

1. before a request's (or realtime client frame's) redaction, each sink is
   asked whether it wants this one (``ScanSink.wants``, synchronous, on the
   event loop); when one does, the redaction runs through a recording copy
   of the request's redactor (``Redactor.with_recorder``): every DISTINCT
   string it scans is kept once, in the order first scanned, with the
   detections found in it — after the allowlists, before overlap
   resolution and modes — each naming the detector that found it
   (``ScanRecorder``). Nothing the redaction does changes;
2. once the redaction succeeded and the request is handed to its upstream
   (``_UploadFate`` settled sent; a realtime frame right before it is
   sent), the capture is QUEUED (``ScanCapture.submit``) — a request
   refused at any point is never captured;
3. ONE worker thread hands each queued capture to the sinks that wanted it
   (``ScanSink.capture``), in order, never on the event loop and never two
   at once. The queue is BOUNDED (``MAX_PENDING_CAPTURES`` captures,
   ``MAX_PENDING_CHARS`` characters, ``MAX_PENDING_ITEMS`` strings and
   detections — the objects that hold them, which cost far more memory
   than a short string's characters): a capture that does not fit is
   dropped and counted, never waited for, so a slow sink costs a request
   nothing beyond the enqueue. A sink's exception — any, a
   ``BaseException`` included: it is plugin code on the core's thread — is
   contained: counted (bookkeeping stage ``scan_sink``, posted to the event
   loop that queued the capture) and logged by its TYPE only, once per
   outage;
4. at shutdown the queue is drained for at most ``CLOSE_SECONDS`` (beside
   the vault's map-write drain, so the shutdown budget is unchanged), what
   is left is dropped and counted, and each sink is closed once, after its
   last capture returned.

A sink receives UNREDACTED text: the values the proxy exists to keep from
the provider (docs/plugins.md, docs/threat-model.md). This module never
logs one: counts, sink class names and exception types only.

The worker thread exits after ``IDLE_SECONDS`` without work and starts again
with the next capture (an idle proxy holds no extra thread).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any, NamedTuple

from llm_redact.detection.base import Detection
from llm_redact.detection.engine import DetectorSource
from llm_redact.plugin_api import ScanContext, ScannedDetection, ScannedString, ScanSink

logger = logging.getLogger("llm_redact")

# The bookkeeping stage a sink's faults count in (``bookkeeping_errors``).
STAGE = "scan_sink"
# Captures waiting for the worker, at most (each one request or frame).
MAX_PENDING_CAPTURES = 256
# Characters of text waiting for the worker, at most: a capture that would
# take the queue past it is dropped — unless the queue is empty (one capture
# is bounded by the request's own limits: max_body_bytes, max_body_strings).
MAX_PENDING_CHARS = 16 * 1024 * 1024
# Strings plus detections waiting for the worker, at most, likewise. Each is
# an object of ~100 bytes besides its text, so a body of many short strings
# would otherwise hold some 25 times the memory its characters count.
MAX_PENDING_ITEMS = 256 * 1024
# The worker thread exits after this long without work.
IDLE_SECONDS = 5.0
# How long the shutdown drains the queue and waits for the sinks to close,
# in all. Run beside the vault's map-write drain (as long), so it adds
# nothing to the shutdown budget (docs/resilience.md, "Shutdown order").
CLOSE_SECONDS = 5.0


class ScanRecorder:
    """One request's (or realtime client frame's) capture: each DISTINCT
    string its redaction scans (``Redactor.with_recorder``: the
    ``DetectionRecorder`` it hands every scanned text to), once, in the
    order first scanned, with its detections — in the order overlap
    resolution reads them (by start, longest first), each naming the
    detector that found it. ``context`` says where the request goes,
    ``sinks`` which of the capture's sinks wanted it."""

    __slots__ = ("_found", "context", "sinks")

    def __init__(self, context: ScanContext, sinks: tuple[int, ...]) -> None:
        self.context = context
        self.sinks = sinks
        self._found: dict[str, tuple[ScannedDetection, ...]] = {}

    def record(self, text: str, found: Sequence[tuple[Detection, DetectorSource]]) -> None:
        # The same text scanned again (a history resent turn by turn, a
        # block check before its redaction) is detected alike: kept once.
        if text in self._found:
            return
        ordered = sorted(found, key=lambda pair: (pair[0].start, pair[0].start - pair[0].end))
        self._found[text] = tuple(
            ScannedDetection(d.detector_type, d.start, d.end, source.kind, source.name)
            for d, source in ordered
        )

    def strings(self) -> tuple[ScannedString, ...]:
        """What was recorded, as the sinks receive it."""
        return tuple(ScannedString(text, found) for text, found in self._found.items())


class _Capture(NamedTuple):
    """A queued capture: what the sinks receive, which of them wanted it,
    and its size in characters and in strings plus detections (the queue's
    bounds)."""

    context: ScanContext
    strings: tuple[ScannedString, ...]
    sinks: tuple[int, ...]
    chars: int
    items: int


def _sink_problem(sink: object) -> str | None:
    """Why ``sink`` cannot serve as a ``ScanSink`` (a member it lacks, or
    one declared ``async def``: never awaited, its every call would capture
    nothing), else None — the type and member names only."""
    for member in ("wants", "capture", "close"):
        method = getattr(sink, member, None)
        if not callable(method):
            return f"{type(sink).__name__} has no callable {member}()"
        if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):
            return f"{type(sink).__name__}.{member}() is async def (it must be synchronous)"
    return None


def _awaitable(answer: object) -> bool:
    """Whether a sink's ``answer`` is an awaitable (never awaited: the
    members are synchronous) — a coroutine is closed unrun, so it never
    runs late nor warns that it was never awaited."""
    if not inspect.isawaitable(answer):
        return False
    if inspect.iscoroutine(answer):
        answer.close()
    return True


class ScanCapture:
    """The registered scan sinks, their bounded queue and their worker
    thread (see the module docstring). ``faults`` is the proxy's
    ``bookkeeping_errors``: incremented on the event loop only."""

    def __init__(
        self,
        sinks: Sequence[ScanSink],
        faults: Counter[str],
        *,
        max_pending: int = MAX_PENDING_CAPTURES,
        max_chars: int = MAX_PENDING_CHARS,
        max_items: int = MAX_PENDING_ITEMS,
        idle_seconds: float = IDLE_SECONDS,
    ) -> None:
        self.sinks = tuple(sinks)
        self._names = tuple(type(sink).__name__ for sink in self.sinks)
        self._faults = faults
        self._max_pending = max_pending
        self._max_chars = max_chars
        self._max_items = max_items
        self._idle_seconds = idle_seconds
        # Guards everything below; never held while a sink runs.
        self._cond = threading.Condition()
        self._queue: deque[_Capture] = deque()
        self._chars = 0
        self._items = 0
        self._in_flight: _Capture | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._delivered = 0
        self._dropped = 0
        self._fault_count = 0
        # The sinks (by index) whose last call failed: one WARNING per
        # outage, one INFO when the sink works again.
        self._failing: set[int] = set()
        # Whether the queue is full (one WARNING per episode).
        self._overflowing = False
        # The event loop the counters are posted to (the submitter's).
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- the request side (the event loop) -------------------------------------

    def recorder(self, context: ScanContext) -> ScanRecorder | None:
        """A recorder for one request (or frame) when a sink wants it
        (``ScanSink.wants``: only ``True`` counts; an exception means no
        and counts as a fault), else None — the redaction then records
        nothing."""
        wanting = tuple(
            index for index, sink in enumerate(self.sinks) if self._wants(index, sink, context)
        )
        return ScanRecorder(context, wanting) if wanting else None

    def _wants(self, index: int, sink: ScanSink, context: ScanContext) -> bool:
        try:
            answer: object = sink.wants(context)
            if _awaitable(answer):
                raise TypeError("wants() answered an awaitable")
            return answer is True
        except Exception as exc:  # noqa: BLE001 — contained: counted, logged by type
            self._failed(index, exc)
            self._count_fault()
            return False

    def submit(self, recorder: ScanRecorder) -> None:
        """Queue ``recorder``'s capture for its sinks — never waiting: past
        the queue's bounds (or once closed) it is dropped and counted. A
        capture of nothing is not queued."""
        strings = recorder.strings()
        if not strings:
            return
        with suppress(RuntimeError):
            self._loop = asyncio.get_running_loop()
        chars = sum(len(scanned.text) for scanned in strings)
        items = len(strings) + sum(len(scanned.detections) for scanned in strings)
        with self._cond:
            fits = not self._closed and (
                not self._queue
                or (
                    len(self._queue) < self._max_pending
                    and self._chars + chars <= self._max_chars
                    and self._items + items <= self._max_items
                )
            )
            if fits:
                self._queue.append(
                    _Capture(recorder.context, strings, recorder.sinks, chars, items)
                )
                self._chars += chars
                self._items += items
                self._ensure_thread()
                self._cond.notify_all()
            else:
                self._dropped += 1
        if fits:
            if self._overflowing:
                self._overflowing = False
                logger.info("scan capture: caught up; captures are queued again")
            return
        if not self._overflowing:
            self._overflowing = True
            logger.warning(
                "scan capture: the sinks' queue is full (%d captures, %d characters or %d"
                " strings and detections); further captures are dropped until it drains",
                self._max_pending,
                self._max_chars,
                self._max_items,
            )

    def status(self) -> dict[str, Any]:
        """The /status ``scan_capture`` block: counts only."""
        with self._cond:
            return {
                "sinks": len(self.sinks),
                "queued": len(self._queue) + (self._in_flight is not None),
                "delivered_total": self._delivered,
                "dropped_total": self._dropped,
                "faults_total": self._fault_count,
            }

    # -- shutdown (off the event loop: blocking) ---------------------------------

    def drain(self, timeout: float) -> int:
        """Wait up to ``timeout`` seconds for every queued capture to be
        delivered; how many have not."""
        with self._cond:
            self._cond.wait_for(self._idle, timeout)
            return len(self._queue) + (self._in_flight is not None)

    def _idle(self) -> bool:
        """Nothing left to wait for: no capture queued or in flight, or no
        worker to deliver them (under ``_cond``)."""
        return self._thread is None or (not self._queue and self._in_flight is None)

    def close(self, timeout: float = CLOSE_SECONDS) -> int:
        """Stop: drain for at most ``timeout`` seconds in all, drop (and
        count) what is still queued, then close each sink once — after its
        last capture returned: a sink still inside a capture when the time
        is up is never closed under it (logged). Returns how many captures
        were dropped."""
        with self._cond:
            if self._closed:
                return 0  # closed once: the sinks are never closed twice
        deadline = time.monotonic() + timeout
        self.drain(timeout)
        with self._cond:
            self._closed = True
            dropped = len(self._queue)
            self._queue.clear()
            self._chars = 0
            self._items = 0
            self._dropped += dropped
            thread = self._thread
            self._cond.notify_all()
        if dropped:
            logger.warning(
                "scan capture: %d capture(s) were not delivered at shutdown (dropped)", dropped
            )
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                logger.warning(
                    "scan capture: a sink was still capturing at shutdown; the sinks are"
                    " not closed under it"
                )
                return dropped
        closer = threading.Thread(target=self._close_sinks, name="llm-redact-scan-close")
        closer.daemon = True
        closer.start()
        closer.join(max(0.0, deadline - time.monotonic()))
        if closer.is_alive():
            logger.warning("scan capture: a sink did not close in time")
        return dropped

    def _close_sinks(self) -> None:
        for name, sink in zip(self._names, self.sinks, strict=True):
            try:
                # Typed as the contract allows a plugin to get it wrong.
                close: Callable[[], object] = sink.close
                if _awaitable(close()):
                    raise TypeError("close() answered an awaitable")
            except BaseException as exc:  # noqa: BLE001 — plugin code: logged by type
                logger.warning(
                    "scan capture: closing sink %s failed (%s)", name, type(exc).__name__
                )

    # -- the worker thread -------------------------------------------------------

    def _ensure_thread(self) -> None:
        """Start the worker unless one runs (under ``_cond``). A thread that
        cannot start (the process is out of threads) is never the request's
        problem: what is queued waits for the next capture to try again —
        or is counted and dropped at shutdown."""
        if self._thread is None:
            thread = threading.Thread(target=self._work, name="llm-redact-scan-sinks", daemon=True)
            self._thread = thread
            try:
                thread.start()
            except RuntimeError as exc:
                self._thread = None
                logger.warning(
                    "scan capture: the sinks' worker thread could not start (%s)",
                    type(exc).__name__,
                )

    def _work(self) -> None:
        try:
            while (capture := self._next()) is not None:
                self._deliver(capture)
        finally:
            with self._cond:
                # Only reached with the thread still registered when it died
                # (a fault of this module's own: a sink's are contained in
                # _deliver): another thread takes what is queued.
                if self._thread is threading.current_thread():
                    self._thread = None
                    self._in_flight = None
                    if self._queue and not self._closed:
                        self._ensure_thread()
                self._cond.notify_all()

    def _next(self) -> _Capture | None:
        """Block until a capture is queued and take it (in flight from now);
        None to exit (closed, or idle for ``IDLE_SECONDS``) — the thread is
        then forgotten under the lock, so the next submit starts another."""
        with self._cond:
            if not self._queue and not self._closed:
                self._cond.wait(self._idle_seconds)
            if self._queue and not self._closed:
                capture = self._queue.popleft()
                self._chars -= capture.chars
                self._items -= capture.items
                self._in_flight = capture
                return capture
            self._thread = None
            return None

    def _deliver(self, capture: _Capture) -> None:
        """Hand ``capture`` to each sink that wanted it; a fault is
        contained (counted on the loop, logged by type once per outage)."""
        for index in capture.sinks:
            try:
                # Typed as the contract allows a plugin to get it wrong.
                deliver: Callable[..., object] = self.sinks[index].capture
                if _awaitable(deliver(capture.context, capture.strings)):
                    raise TypeError("capture() answered an awaitable")
            except BaseException as exc:  # noqa: BLE001 — plugin code: counted, logged by type
                # A BaseException too (SystemExit, a CancelledError, …): on
                # this thread it would end the worker, losing the capture
                # uncounted, and threading.excepthook would print its
                # message — which may quote the text it was handed.
                self._failed(index, exc)
                self._post(self._count_fault)
            else:
                self._succeeded(index)
        with self._cond:
            self._in_flight = None
            self._cond.notify_all()

    def _failed(self, index: int, exc: BaseException) -> None:
        with self._cond:
            self._fault_count += 1
            first = index not in self._failing
            self._failing.add(index)
        if first:
            logger.warning(
                "scan capture: sink %s failed (%s); its captures are lost until it works again",
                self._names[index],
                type(exc).__name__,
            )

    def _succeeded(self, index: int) -> None:
        with self._cond:
            self._delivered += 1
            recovered = index in self._failing
            self._failing.discard(index)
        if recovered:
            logger.info("scan capture: sink %s works again", self._names[index])

    def _count_fault(self) -> None:
        """One sink fault in the proxy's bookkeeping counts (on the loop)."""
        self._faults[STAGE] += 1

    def _post(self, outcome: Callable[[], None]) -> None:
        """Run ``outcome`` on the submitting event loop — directly when there
        is none or it has closed (nothing else runs on it then)."""
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(outcome)
                return
            except RuntimeError:
                pass
        outcome()


def build_scan_capture(sinks: object, faults: Counter[str]) -> ScanCapture | None:
    """The proxy's capture for the registered ``sinks`` (the factory's
    answer, ``Registry.build_scan_sinks``) — None without any (the Free
    core: nothing is recorded or queued). An answer that is not a tuple or
    list of sinks, or a sink lacking a member of the contract, is a
    ``ConfigError`` (the startup refuses to serve)."""
    from llm_redact.config import ConfigError

    if not isinstance(sinks, tuple | list):
        raise ConfigError(
            f"the scan sink factory answered {type(sinks).__name__}, not a tuple of sinks"
        )
    if not sinks:
        return None
    for sink in sinks:
        problem = _sink_problem(sink)
        if problem is not None:
            raise ConfigError(f"a registered scan sink cannot be used: {problem}")
    return ScanCapture(sinks, faults)


def capture_status(capture: ScanCapture | None) -> dict[str, Any]:
    """The /status ``scan_capture`` block, zeros without a sink."""
    if capture is None:
        return {
            "sinks": 0,
            "queued": 0,
            "delivered_total": 0,
            "dropped_total": 0,
            "faults_total": 0,
        }
    return capture.status()
