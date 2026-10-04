"""Outbound half: replace detected values with vault-issued placeholders."""

from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol

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

    def unoverridable(self) -> None:
        """The request is being refused for a finding no approval can pass
        (a deny string in text the proxy cannot rewrite): its refusal must
        carry no code."""


class TextScan(NamedTuple):
    """``Redactor.scan``'s answer: the detector types of the values found
    (counted), and whether an approved override let a refusing value
    through (the text then holds a value detection would have acted on)."""

    found: Counter[str]
    overridden: bool


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
    # The chosen deny spans are disjoint and start-sorted, so their ends are
    # sorted too: a binary search finds each candidate's only possible
    # overlap. Candidates come start-sorted, so the search never needs to
    # look left of the previous answer (``lo=i``) — the old forward walk's
    # result, with no loop a mutated index could spin forever.
    deny_ends = [d.end for d in deny_chosen]
    others: list[Detection] = []
    i = 0
    for d in detections:
        if d.tier == 0:
            continue
        i = bisect_right(deny_ends, d.start, lo=i)
        # deny_chosen[i] is the first deny span ending after d.start (if
        # any): the only possible overlap candidate.
        if i < len(deny_chosen) and deny_chosen[i].start < d.end:
            continue
        others.append(d)
    return sorted(deny_chosen + _sweep(others), key=lambda d: d.start)


def _unite(text: str, spans: list[Detection]) -> list[Detection]:
    """``spans`` in start order, every run of overlapping spans joined into
    ONE detection over their union (``_joined``); a span overlapping no other
    is kept as it is. Linear: each span is looked at a fixed number of
    times, and each union sliced from ``text`` once."""
    runs: list[list[Detection]] = []
    # Where each run ends so far.
    ends: list[int] = []
    for d in sorted(spans, key=lambda d: d.start):
        if runs and d.start < ends[-1]:
            runs[-1].append(d)
            ends[-1] = max(ends[-1], d.end)
        else:
            runs.append([d])
            ends.append(d.end)
    return [_joined(text, run) for run in runs]


def _joined(text: str, run: list[Detection]) -> Detection:
    """One run of overlapping spans as what is redacted in their place: a
    lone span itself; overlapping ones a tier-0 detection (always redacted,
    never a mode or an override) over their union, typed after the run's
    first span."""
    first = run[0]
    if len(run) == 1:
        return first
    end = max(d.end for d in run)
    return Detection(first.start, end, first.detector_type, text[first.start : end], tier=0)


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
        added_deny: DetectorPlan | None = None,
        final_blocks: frozenset[str] = frozenset(),
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
        # Deny strings ADDED on top of the configured policy (an access
        # gate's detection overlay, authorization.py), detected apart from
        # it so that they can only tighten it (``_absorb``); None for none.
        self._added_deny = added_deny
        # The detector types whose block mode that overlay ADDED, past their
        # configured mode: refusing one of those values is final — no
        # approved override passes it and its refusal mints no code — since
        # the configured policy would have redacted (or forwarded) it, an
        # approval would forward as sent a value it never would have.
        self._final_blocks = final_blocks

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
            added_deny=self._added_deny,
            final_blocks=self._final_blocks,
        )

    def _overridden(self, d: Detection) -> bool:
        """Whether the requester's approved overrides let this refusing
        value through (forwarded as sent). A block the detection overlay
        added is never put to them: the refusal is final
        (``OverrideCheck.unoverridable``) — no grant is consulted or
        consumed, no code minted."""
        if self._overrides is None:
            return False
        if d.detector_type in self._final_blocks:
            self._overrides.unoverridable()
            return False
        return self._overrides.allows(d.detector_type, d.value)

    def _winners(self, text: str) -> list[Detection]:
        """The detections redaction acts on in ``text``: the configured
        policy's (``_resolve_overlaps``), with the added deny strings'
        matches taken in (``_absorb``)."""
        winners = _resolve_overlaps(self._plan.detect(text, self._allowlist))
        if self._added_deny is None:
            return winners
        return self._absorb(text, winners, self._added_deny.detect(text, self._allowlist))

    def _absorb(
        self, text: str, winners: list[Detection], added: list[Detection]
    ) -> list[Detection]:
        """The configured winners of ``text`` with the ADDED deny strings'
        matches ``added`` taken in, so that they only ever TIGHTEN what the
        configured policy does. A winner that refuses (block mode, no
        approved override) refuses the text right here, as ``redact_text``
        would. A winner the policy forwards as sent (warn mode, an approved
        block) yields to an added match overlapping it: the match is
        redacted, the rest of the value goes as it would have. A winner
        that redacts (redact mode, a configured deny string) keeps its
        effect, and the UNION of its span and every added match overlapping
        it is redacted as one placeholder (``_unite``) — so neither the value
        the policy redacts nor the added string leaves in part."""
        redacting: list[Detection] = []
        passing: list[Detection] = []
        for d in winners:
            mode = self._modes.get(d.detector_type, "redact") if d.tier else "redact"
            if mode == "block" and not self._overridden(d):
                raise BlockedRequest(d.detector_type)
            (redacting if mode == "redact" else passing).append(d)
        # Every united span an added match is part of is tier 0, so the
        # configured resolution's own tier-0 rule drops each passing winner
        # one overlaps; the other spans are disjoint from every winner (the
        # configured winners are disjoint), so nothing else changes.
        united = _unite(text, redacting + added)
        return _resolve_overlaps(sorted(united + passing, key=lambda d: d.start))

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
        detections = self._winners(text)
        if not detections:
            return text
        parts: list[str] = []
        cursor = 0
        for d in detections:
            # Tier-0 (deny, and an added deny string's union with what it
            # overlaps) always redacts: keying modes by type could let a
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
        """``scan``'s found detector types (text the proxy cannot rewrite)."""
        return self.scan(text).found

    def scan(self, text: str, *, redactable: bool = False) -> TextScan:
        """Detection only, for text the proxy cannot rewrite (a binary
        upload read through its extracted text, a verbatim field): the
        detections ``redact_text`` would act on — the same allowlists,
        per-type allowlists, deny strings and overlap resolution — with NO
        placeholder issued and nothing written to the vault. A block-mode
        winner raises BlockedRequest; a warn-mode winner is counted in
        ``warn_counts`` (its value stays where it is); every other winner —
        a deny string always — is returned as its detector type, counted.

        Every refusing winner but a deny string is first put to the
        requester's overrides (``with_overrides``): an approved value is
        skipped (``TextScan.overridden``). ``redactable``: the caller will
        redact this text after all (convert mode sends a document's text in
        its file's place), so only a block-mode winner refuses — a value it
        returns is redacted, not refused, and is never put to the
        overrides. A deny string in text that is not ``redactable`` refuses
        it for good (``OverrideCheck.unoverridable``) — an added one too,
        with the value it overlaps (``_absorb``: the proxy could not redact
        their union). Charged against the string budget as one string."""
        self.charge(1)
        found: Counter[str] = Counter()
        overridden = False
        for d in self._winners(text):
            if not d.tier:
                # Deny strings (tier 0) take no mode, exactly as in
                # redact_text, and are the operator's always-redact list —
                # or an access gate's, added to it (with the value it
                # overlaps): never put to the requester's overrides.
                found[d.detector_type] += 1
                if not redactable and self._overrides is not None:
                    self._overrides.unoverridable()
                continue
            mode = self._modes.get(d.detector_type)
            if mode == "warn":
                self.warn_counts[d.detector_type] += 1
            elif (mode == "block" or not redactable) and self._overridden(d):
                overridden = True
            elif mode == "block":
                raise BlockedRequest(d.detector_type)
            else:
                found[d.detector_type] += 1
        return TextScan(found, overridden)

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
        an upload inspector). Added deny strings take no part: they never
        block, nor take a block-mode winner's place (``_absorb``)."""
        for d in _resolve_overlaps(self._plan.detect(text, self._allowlist)):
            # Deny strings (tier 0) never block, exactly as in redact_text.
            if d.tier and self._modes.get(d.detector_type) == "block" and not self._overridden(d):
                return d.detector_type
        return None

    def redact_json(self, obj: Any) -> Any:
        return transform_strings(obj, self.redact_text)
