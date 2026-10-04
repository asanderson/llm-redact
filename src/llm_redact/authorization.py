"""The access gate's authorization seams (``plugin_api.AccessGate``'s optional
``authorize_request`` and ``detection_overlay``), applied by the core.

The core holds NO user, role, group or seat logic: it hands the gate the FACTS
of a request it resolved itself (``plugin_api.AuthorizationRequest``) and
applies what the gate answers — a refusal, or a detection overlay that can
only TIGHTEN the configured policy (``plugin_api.DetectionOverlay``). Both
fail closed: a gate that raises, times out or answers anything the core
cannot apply refuses the request with the core's fixed text, counted under the
bookkeeping stage ``authorization`` and logged by exception TYPE only (never a
reason, a deny string, a path's value or a user).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Any

from llm_redact.detection.deny import DenyDetector, DenyEntry
from llm_redact.detection.engine import DetectionConfig, DetectorPlan, active_rule_names
from llm_redact.detection.regex_rules import BUILTIN_RULES
from llm_redact.plugin_api import AuthorizationRequest, DetectionOverlay

logger = logging.getLogger(__name__)

# How long an awaitable ``authorize_request`` answer may take; past it the
# request is refused and the gate's task cancelled (never awaited again).
AUTHORIZE_TIMEOUT_SECONDS = 5.0
# The bookkeeping stage a failed authorization check or overlay counts in.
AUTHORIZATION_STAGE = "authorization"
# The 403 text (and realtime close reason: it fits 123 bytes) when the gate's
# check fails or answers something other than a reason.
AUTHORIZATION_FAULT = (
    "llm-redact: the request authorization check failed; the request was not forwarded"
)
# The 403 text when the requester's detection overlay cannot be applied.
OVERLAY_FAULT = (
    "llm-redact: this requester's detection policy could not be applied;"
    " the request was not forwarded"
)
# Distinct overlays whose builds are kept (least recently used dropped).
OVERLAY_CACHE_SIZE = 64
# The modes an overlay may set, and how strict each mode is.
OVERLAY_MODES = ("redact", "block")
_STRICTNESS = {"warn": 0, "redact": 1, "block": 2}
_GUILLEMETS = ("«", "»")


class OverlayError(Exception):
    """An overlay the core cannot apply. The message is a fixed KIND (it is
    logged), never a rule's mode or a deny string."""


@dataclass(frozen=True)
class OverlayBuild:
    """One overlay applied to the configured policy it was built against: the
    type-keyed modes a request's redactor runs with instead of the configured
    ones; the overlay's deny strings — detected apart from the configured
    detectors (``Redactor(added_deny=...)``), so that they can only tighten
    what the configured policy does, never displace it; and the detector
    types whose block mode the overlay ADDED (``Redactor(final_blocks=...)``):
    their refusals take no refusal override, since the configured policy
    would have redacted or forwarded those values, never refused them."""

    modes: Mapping[str, str]
    deny: DetectorPlan | None = None
    final_blocks: frozenset[str] = frozenset()


def _require_shape(overlay: object) -> DetectionOverlay:
    """``overlay`` as a DetectionOverlay of tuples of strings (so it hashes),
    else OverlayError."""
    if type(overlay) is not DetectionOverlay:
        raise OverlayError("not a DetectionOverlay")
    if not isinstance(overlay.modes, tuple) or not all(
        isinstance(pair, tuple)
        and len(pair) == 2
        and isinstance(pair[0], str)
        and isinstance(pair[1], str)
        for pair in overlay.modes
    ):
        raise OverlayError("modes are not (rule, mode) string pairs")
    if not isinstance(overlay.deny, tuple) or not all(
        isinstance(value, str) for value in overlay.deny
    ):
        raise OverlayError("deny is not a tuple of strings")
    return overlay


class OverlayBuilds:
    """The overlays applied to ONE generation of detection objects (the
    configured detectors and type-keyed modes): built once per distinct
    overlay and kept, bounded. A reload that rebuilds the detection objects
    builds a new instance, so a cached build never outlives them."""

    def __init__(
        self,
        config: DetectionConfig,
        modes: Mapping[str, str],
        *,
        size: int = OVERLAY_CACHE_SIZE,
    ) -> None:
        self.modes = modes
        self.size = size
        self._type_by_rule = {rule.name: rule.detector_type for rule in BUILTIN_RULES}
        custom = {rule.name: rule.detector_type for rule in config.custom_rules}
        self._type_by_rule.update(custom)
        # The detector types a built rule emits: tightening a rule tightens
        # its TYPE (modes dispatch per type, as [detection.modes] does), so a
        # rule that is not built still tightens a type a built sibling emits.
        self._built_types = frozenset(
            self._type_by_rule[name] for name in active_rule_names(config)
        ) | frozenset(custom.values())
        self._cache: OrderedDict[DetectionOverlay, OverlayBuild | None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._cache)

    def build(self, overlay: object) -> OverlayBuild | None:
        """``overlay`` applied, or None when it changes nothing (the
        configured policy). Raises OverlayError for one the core cannot
        apply; only a successful build is kept."""
        checked = _require_shape(overlay)
        if not checked.modes and not checked.deny:
            return None  # empty: the configured policy, nothing to keep
        if checked in self._cache:
            self._cache.move_to_end(checked)
            return self._cache[checked]
        built = self._build(checked)
        self._cache[checked] = built
        if len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return built

    def _build(self, overlay: DetectionOverlay) -> OverlayBuild | None:
        modes = self._tightened(overlay.modes)
        for value in overlay.deny:
            if not value or any(mark in value for mark in _GUILLEMETS):
                raise OverlayError("a deny string is empty or holds a guillemet")
        final = frozenset(
            detector_type
            for detector_type, mode in modes.items()
            if mode == "block" and self.modes.get(detector_type) != "block"
        )
        if not overlay.deny:
            return None if modes is self.modes else OverlayBuild(modes, final_blocks=final)
        # One case-insensitive DenyDetector over the overlay's strings, run
        # beside the configured detectors — never among them: in their
        # overlap resolution a deny string wins every overlap, and the
        # value it cut into would leave in part (a configured block never
        # fired). The redactor takes the added matches in on top of the
        # configured winners (``Redactor._absorb``).
        entries = [DenyEntry(value) for value in dict.fromkeys(overlay.deny)]
        return OverlayBuild(modes, DetectorPlan([DenyDetector(entries)]), final)

    def _tightened(self, pairs: tuple[tuple[str, str], ...]) -> Mapping[str, str]:
        """The configured type-keyed modes with ``pairs`` applied: the same
        object when nothing tightens."""
        tightened: dict[str, str] | None = None
        for rule, mode in pairs:
            if mode not in OVERLAY_MODES:
                raise OverlayError("a mode is not redact or block")
            detector_type = self._type_by_rule.get(rule)
            if detector_type is None:
                raise OverlayError("an unknown rule name")
            if _STRICTNESS[mode] < _STRICTNESS[self.modes.get(detector_type, "redact")]:
                raise OverlayError("a relaxation of the configured mode")
            current = (tightened if tightened is not None else self.modes).get(
                detector_type, "redact"
            )
            if detector_type not in self._built_types or _STRICTNESS[mode] <= _STRICTNESS[current]:
                continue  # no built rule emits the type, or nothing tightens
            if tightened is None:
                tightened = dict(self.modes)
            if mode == "redact":
                del tightened[detector_type]  # the default: stored as absent
            else:
                tightened[detector_type] = mode
        return tightened if tightened is not None else self.modes


def _close_unrun(answer: object) -> None:
    """An awaitable a synchronous member answered: closed unrun, so nothing
    it would do happens later."""
    if inspect.iscoroutine(answer):
        answer.close()
    elif isinstance(answer, asyncio.Future):
        answer.cancel()


def _discard(task: asyncio.Future[Any]) -> None:
    """A gate task given up on: its outcome retrieved whenever it ends (never
    an unretrieved-exception warning)."""
    if not task.cancelled():
        task.exception()


class GateAuthorization:
    """The access gate's optional authorization members, read ONCE (the gate
    is restart-only). Without them every test here is one attribute read."""

    def __init__(
        self,
        gate: object,
        bookkeeping_errors: Counter[str],
        *,
        timeout: float = AUTHORIZE_TIMEOUT_SECONDS,
    ) -> None:
        authorize = getattr(gate, "authorize_request", None)
        overlay = getattr(gate, "detection_overlay", None)
        self._authorize = authorize if callable(authorize) else None
        self._overlay = overlay if callable(overlay) else None
        self.authorizes = self._authorize is not None
        self.overlays = self._overlay is not None
        self.timeout = timeout
        self._faults = bookkeeping_errors

    def _fault(self, where: str, what: str, text: str) -> str:
        self._faults[AUTHORIZATION_STAGE] += 1
        logger.warning("%s -> refused: the access gate's authorization failed (%s)", where, what)
        return text

    def refusal(
        self, request: AuthorizationRequest, where: str
    ) -> str | None | Awaitable[str | None]:
        """The gate's verdict on ``request``: None (allowed), the reason it
        is refused with, or — when the gate answers with an awaitable — an
        awaitable of either (bounded by ``timeout``). ``where`` (method and
        path, or "WS path") names the request in the log."""
        assert self._authorize is not None  # callers test ``authorizes``
        try:
            answer = self._authorize(request)
        except Exception as exc:  # noqa: BLE001 — fail closed
            return self._fault(where, type(exc).__name__, AUTHORIZATION_FAULT)
        if inspect.isawaitable(answer):
            return self._bounded(answer, where)
        return self._verdict(answer, where)

    async def _bounded(self, answer: Awaitable[Any], where: str) -> str | None:
        # A task in the request's own context (a copy: what the gate's admit
        # set is visible), given up on — cancelled, never awaited again —
        # past the bound or when the request itself is cancelled.
        task = asyncio.ensure_future(answer)
        try:
            done, _pending = await asyncio.wait({task}, timeout=self.timeout)
        finally:
            if not task.done():
                task.cancel()
                task.add_done_callback(_discard)
        if not done:
            return self._fault(where, "timed out", AUTHORIZATION_FAULT)
        if task.cancelled():
            return self._fault(where, "cancelled", AUTHORIZATION_FAULT)
        problem = task.exception()
        if problem is not None:
            return self._fault(where, type(problem).__name__, AUTHORIZATION_FAULT)
        return self._verdict(task.result(), where)

    def _verdict(self, answer: object, where: str) -> str | None:
        if answer is None:
            return None
        if isinstance(answer, str) and answer:
            return answer
        return self._fault(where, f"answered {type(answer).__name__}", AUTHORIZATION_FAULT)

    def overlay(self, builds: OverlayBuilds, where: str) -> tuple[OverlayBuild | None, str | None]:
        """The requester's detection overlay applied to ``builds``' detection
        objects: ``(build, None)`` — build None for the configured policy —
        or ``(None, refusal text)`` when it cannot be applied."""
        assert self._overlay is not None  # callers test ``overlays``
        try:
            answer = self._overlay()
        except Exception as exc:  # noqa: BLE001 — fail closed
            return None, self._fault(where, type(exc).__name__, OVERLAY_FAULT)
        if answer is None:
            return None, None
        if inspect.isawaitable(answer):
            _close_unrun(answer)
            return None, self._fault(where, "answered an awaitable", OVERLAY_FAULT)
        try:
            return builds.build(answer), None
        except OverlayError as exc:
            return None, self._fault(where, f"overlay: {exc}", OVERLAY_FAULT)
