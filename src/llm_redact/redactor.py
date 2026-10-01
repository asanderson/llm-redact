"""Outbound half: replace detected values with vault-issued placeholders."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from llm_redact.detection.base import Detection, Detector
from llm_redact.detection.engine import Allowlist, DetectorPlan, plan_for
from llm_redact.jsonwalk import transform_strings
from llm_redact.vault import PlaceholderSpaceExhausted, Vault


class BlockedRequest(Exception):
    """A block-mode rule matched: the whole request must be rejected.

    Carries only the detector type — never the matched value — so handlers
    can log and report it without violating the never-log-values rule.
    """

    def __init__(self, detector_type: str) -> None:
        super().__init__(detector_type)
        self.detector_type = detector_type


class UnredactableRequest(Exception):
    """A request field that must be redacted could not be decoded, so it
    cannot be redacted: the whole request is rejected (400), never forwarded
    unredacted. The message names the FIELD only — never its content."""


class PlaceholderLimitReached(UnredactableRequest):
    """A new value would need a placeholder number past MAX_TOKEN_NUMBER
    (only a request carrying a token numbered at the limit gets here): the
    whole request is refused, never numbered past the limit or onto a token
    the request already carries. The message names the detector type only."""


class TooManyStrings(UnredactableRequest):
    """The body carries more strings for redaction than ``max_body_strings``
    allows: refused before any upstream contact (413 over HTTP), never
    forwarded partly redacted. A subclass of UnredactableRequest, so a path
    that does not tell it apart still refuses the request."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"request body exceeds llm-redact max_body_strings ({limit})")
        self.limit = limit


class OverrideCheck(Protocol):
    """The requester's approved overrides (``overrides.OverrideScope``),
    asked only where detection has decided a refusal: whether this value
    may go through as sent."""

    def allows(self, detector_type: str, value: str) -> bool: ...


class StringBudget:
    """How many strings one request body may have redacted: its JSON string
    values, form fields, file names and uploaded JSONL lines. Redaction costs
    per string, so the count bounds the event-loop time one body can take —
    max_body_bytes alone let 10 MiB of tiny strings stall the loop for
    seconds."""

    __slots__ = ("limit", "used")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def charge(self, count: int) -> None:
        self.used += count
        if self.used > self.limit:
            raise TooManyStrings(self.limit)


def _sweep(detections: Sequence[Detection]) -> list[Detection]:
    """Greedy non-overlapping sweep over (start, -length, priority)-sorted input.

    Longer and higher-priority matches win, so «sk-ant-…» beats the generic
    «sk-…» rule and a PEM block beats anything matched inside it.
    """
    chosen: list[Detection] = []
    last_end = -1
    for d in detections:
        if d.start >= last_end:
            chosen.append(d)
            last_end = d.end
    return chosen


def _resolve_overlaps(detections: Sequence[Detection]) -> list[Detection]:
    """Two-phase sweep: tier-0 (user deny) detections win every overlap.

    A sort-key tweak cannot express "deny always wins": the sweep is
    start-ordered, so a rule match merely *starting earlier* would otherwise
    claim the span. Phase 1 sweeps tier-0 detections alone (longest deny
    wins among overlapping denies); phase 2 runs the pre-tier sweep over the
    rest, additionally skipping anything that overlaps a chosen deny span.
    With no tier-0 detections present the behavior is the old sweep,
    byte for byte.
    """
    if all(d.tier != 0 for d in detections):
        return _sweep(detections)
    deny_chosen = _sweep([d for d in detections if d.tier == 0])
    # Merge-walk: both lists are start-sorted, so one forward index suffices
    # to find each candidate's potentially-overlapping deny spans.
    others: list[Detection] = []
    i = 0
    for d in detections:
        if d.tier == 0:
            continue
        while i < len(deny_chosen) and deny_chosen[i].end <= d.start:
            i += 1
        # deny_chosen[i] is the first deny span ending after d.start (if
        # any); the spans are disjoint and start-sorted, so it is the only
        # possible overlap candidate.
        if i < len(deny_chosen) and deny_chosen[i].start < d.end:
            continue
        others.append(d)
    return sorted(deny_chosen + _sweep(others), key=lambda d: d.start)


class Redactor:
    def __init__(
        self,
        detectors: "Sequence[Detector] | DetectorPlan",
        vault: Vault,
        allowlist: Allowlist,
        counts: "Counter[str] | None" = None,
        modes: Mapping[str, str] | None = None,
        warn_counts: "Counter[str] | None" = None,
        floors: Mapping[str, int] | None = None,
        budget: StringBudget | None = None,
        overrides: OverrideCheck | None = None,
    ) -> None:
        # The detector list compiled for string-at-a-time detection (same
        # output, gated per string), taken as it is now: plan_for shares the
        # live plan of a list, and the thin copies below hand theirs on.
        self._plan = detectors if isinstance(detectors, DetectorPlan) else plan_for(detectors)
        # The strings one request body may still have redacted (with_budget);
        # None for a shared redactor, which counts nothing.
        self._budget = budget
        self._vault = vault
        self._allowlist = allowlist
        # Detection counts by type; a shared Counter may be passed in so
        # per-session redactors report into one process-wide total.
        self.counts: Counter[str] = counts if counts is not None else Counter()
        # Detector-TYPE-keyed dispatch (build_modes output); empty = all
        # redact. Modes dispatch on the overlap-resolved winner, so a
        # warn-mode long match governs a redact-mode match nested inside it —
        # the same longest-wins rule that governs substitution.
        self._modes = modes if modes is not None else {}
        self.warn_counts: Counter[str] = warn_counts if warn_counts is not None else Counter()
        # The request's token floors (placeholders.token_floors): per type,
        # the highest placeholder number the request being redacted already
        # carries. Every NEW placeholder is numbered above it. Never mutated:
        # a shared redactor (static mode) stays floor-free and each request
        # carrying tokens gets its own thin copy (with_floors).
        self._floors: Mapping[str, int] = floors if floors is not None else {}
        # The requester's approved overrides (with_overrides), asked for a
        # value only where detection refuses the request; None asks nothing.
        self._overrides = overrides

    def with_floors(self, floors: Mapping[str, int]) -> "Redactor":
        """This redactor numbering new placeholders above ``floors`` as well
        (per type, the higher of both): itself when nothing rises, else a
        thin copy sharing the detectors, vault and counters — so a floor
        never leaks into another request through a shared redactor."""
        raised = {t: n for t, n in floors.items() if n > self._floors.get(t, 0)}
        if not raised:
            return self
        return self._copy({**self._floors, **raised}, self._budget)

    def with_budget(self, limit: int) -> "Redactor":
        """A thin copy for ONE request body that refuses it (TooManyStrings)
        once it has been asked to redact more than ``limit`` strings. Its
        floor copies (with_floors) share the same count."""
        return self._copy(self._floors, StringBudget(limit))

    def with_overrides(self, overrides: OverrideCheck) -> "Redactor":
        """A thin copy that asks ``overrides`` before refusing a value (a
        block-mode winner here; ``scan_text``'s findings): an approved value
        is left in place and forwarded as sent, like a warn-mode one."""
        copy = self._copy(self._floors, self._budget)
        copy._overrides = overrides
        return copy

    def _copy(self, floors: Mapping[str, int], budget: StringBudget | None) -> "Redactor":
        return Redactor(
            self._plan,
            self._vault,
            self._allowlist,
            counts=self.counts,
            modes=self._modes,
            warn_counts=self.warn_counts,
            floors=floors,
            budget=budget,
            overrides=self._overrides,
        )

    def _overridden(self, d: Detection) -> bool:
        return self._overrides is not None and self._overrides.allows(d.detector_type, d.value)

    def charge(self, count: int) -> None:
        """Count ``count`` more pieces of the body (an uploaded file's JSONL
        lines, before they are split) against its string budget: nothing
        for a redactor without one."""
        if self._budget is not None:
            self._budget.charge(count)

    def _placeholder(self, detector_type: str, value: str) -> str:
        floor = self._floors.get(detector_type)
        try:
            if floor is None:
                # No token of this type in the request: the keyword is left
                # out, so a vault predating it keeps serving every request
                # that carries no tokens.
                return self._vault.placeholder_for(detector_type, value)
            return self._vault.placeholder_for(detector_type, value, floor=floor)
        except PlaceholderSpaceExhausted as exc:
            raise PlaceholderLimitReached(
                f"llm-redact: {exc}; the request was not forwarded"
            ) from None

    def redact_text(self, text: str) -> str:
        # Counted before any work on it: the string over the budget is
        # never scanned.
        self.charge(1)
        detections = _resolve_overlaps(self._plan.detect(text, self._allowlist))
        if not detections:
            return text
        parts: list[str] = []
        cursor = 0
        for d in detections:
            # Tier-0 (deny) always redacts: keying modes by type could let a
            # user-chosen deny type accidentally inherit a warn/block mode
            # from a rule sharing that type.
            mode = "redact" if d.tier == 0 else self._modes.get(d.detector_type, "redact")
            if mode == "block":
                if self._overridden(d):
                    # The requester approved this value: forwarded as sent,
                    # like a warn-mode one (no placeholder, no count).
                    continue
                # Fail closed immediately; any placeholders already issued
                # for earlier spans are harmless (deterministic vault,
                # nothing is forwarded).
                raise BlockedRequest(d.detector_type)
            if mode == "warn":
                # Deliberately leaves the original in place: no vault write,
                # no placeholder — the value WILL go upstream.
                self.warn_counts[d.detector_type] += 1
                continue
            parts.append(text[cursor : d.start])
            parts.append(self._placeholder(d.detector_type, d.value))
            self.counts[d.detector_type] += 1
            cursor = d.end
        parts.append(text[cursor:])
        return "".join(parts)

    def scan_text(self, text: str) -> "Counter[str]":
        """Detection only, for text the proxy cannot rewrite (a binary
        upload read through its extracted text): the detections
        ``redact_text`` would act on — the same allowlists, per-type
        allowlists, deny strings and overlap resolution — with NO
        placeholder issued and nothing written to the vault. A block-mode
        winner raises BlockedRequest; a warn-mode winner is counted in
        ``warn_counts`` (its value stays where it is); every other winner —
        a deny string always — is returned as its detector type, counted.
        Every refusing winner (block mode, or one returned) but a deny
        string is first put to the requester's overrides
        (``with_overrides``): an approved value is skipped. Charged against the string budget as one string."""
        self.charge(1)
        found: Counter[str] = Counter()
        for d in _resolve_overlaps(self._plan.detect(text, self._allowlist)):
            # Deny strings (tier 0) take no mode, exactly as in redact_text.
            mode = self._modes.get(d.detector_type) if d.tier else None
            if mode == "warn":
                self.warn_counts[d.detector_type] += 1
            elif d.tier and self._overridden(d):
                # Deny strings (tier 0) are the operator's always-redact
                # list: never put to the requester's overrides.
                continue
            elif mode == "block":
                raise BlockedRequest(d.detector_type)
            else:
                found[d.detector_type] += 1
        return found

    @property
    def blocks(self) -> bool:
        """Whether a rule is in block mode: whether ``redact_text`` can
        refuse a text (BlockedRequest) at all."""
        return "block" in self._modes.values()

    def blocked_type(self, text: str) -> str | None:
        """The detector type ``redact_text`` would refuse ``text`` for — its
        first block-mode winner, under the same allowlists, deny strings
        and overlap resolution — else None. Nothing is issued, counted or
        charged: a check AHEAD of the redaction that will scan ``text`` and
        count it (an upload's pieces, before its binary parts are handed to
        an upload inspector)."""
        for d in _resolve_overlaps(self._plan.detect(text, self._allowlist)):
            # Deny strings (tier 0) never block, exactly as in redact_text.
            if d.tier and self._modes.get(d.detector_type) == "block" and not self._overridden(d):
                return d.detector_type
        return None

    def redact_json(self, obj: Any) -> Any:
        return transform_strings(obj, self.redact_text)
