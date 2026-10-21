"""Outbound half: replace detected values with vault-issued placeholders."""

import copy
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol

from llm_redact.detection.base import ADDITIVE_TIER, Detection, Detector
from llm_redact.detection.engine import (
    Allowlist,
    DetectorPlan,
    DetectorSource,
    PrecomputedTable,
    plan_for,
)
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


class DetectionRecorder(Protocol):
    """Where a redactor that records what it scans (``with_recorder``)
    hands each text it scans: with every detection found in it — the
    configured detectors' (allowlists applied) and the added deny
    strings' — each with the detector it came from, before anything acts
    on them (``scan_capture.ScanRecorder``). Read-only: it can change
    nothing the redaction does."""

    def record(self, text: str, found: Sequence[tuple[Detection, DetectorSource]]) -> None: ...


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
    """The overlap sweep's winners: tier-0 detections win every overlap
    (``_resolve_tiers``), and additive ones (tier 2, ``ADDITIVE_TIER``: the
    llm NER backend's) never displace anything. What redaction acts on when
    no value joins what it overlaps (no deny string, no ``Detection.joins``
    value: ``Redactor._resolved``), and what the benches score.

    The others are resolved exactly as without the additive detections;
    those are swept among themselves and each winner keeps only the parts
    of its span that no other winner holds (``_outside``). So an additive
    detection can only ADD redacted characters: a model the scanned text
    instructs can return a span starting just before a secret, but that
    span never takes the secret's place. With no additive detection present
    the behavior is ``_resolve_tiers``, byte for byte.
    """
    if all(d.tier != ADDITIVE_TIER for d in detections):
        return _resolve_tiers(detections)
    fixed = _resolve_tiers([d for d in detections if d.tier != ADDITIVE_TIER])
    additive = sorted(
        (d for d in detections if d.tier == ADDITIVE_TIER),
        key=lambda d: (d.start, -(d.end - d.start), d.priority),
    )
    return sorted(fixed + _outside(fixed, _sweep(additive)), key=lambda d: d.start)


def _outside(fixed: Sequence[Detection], spans: Sequence[Detection]) -> list[Detection]:
    """The parts of ``spans`` no ``fixed`` detection covers. Both are
    disjoint and start-sorted; a span overlapping none is kept as it is,
    one overlapping some is cut into the pieces between them, each a
    detection of its own (same type, priority and tier) trimmed of the
    whitespace at its edges (a piece of only whitespace is dropped). Each
    span finds the winners it overlaps by binary search, so the cost is
    linear in the spans, the winners and the pieces."""
    ends = [f.end for f in fixed]
    starts = [f.start for f in fixed]
    pieces: list[Detection] = []
    for d in spans:
        # fixed[first:last] are the winners overlapping d: those ending
        # after it starts and starting before it ends.
        first = bisect_right(ends, d.start)
        last = bisect_left(starts, d.end)
        if first >= last:
            pieces.append(d)
            continue
        cursor = d.start
        for winner in fixed[first:last]:
            pieces.extend(_piece(d, cursor, winner.start))
            cursor = winner.end
        pieces.extend(_piece(d, cursor, d.end))
    return pieces


def _piece(d: Detection, start: int, end: int) -> list[Detection]:
    """``d``'s span between ``start`` and ``end`` (clamped to it) as a
    detection of its own, its edge whitespace trimmed; none when nothing
    but whitespace is left."""
    start, end = max(start, d.start), min(end, d.end)
    raw = d.value[start - d.start : end - d.start] if start < end else ""
    kept = raw.strip()
    if not kept:
        return []
    begin = start + len(raw) - len(raw.lstrip())
    return [Detection(begin, begin + len(kept), d.detector_type, kept, d.priority, d.tier)]


def _resolve_tiers(detections: Sequence[Detection]) -> list[Detection]:
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


def _overlapping(spans: Sequence[Detection]) -> list[list[Detection]]:
    """``spans`` in start order (a stable sort), cut into runs of spans that
    overlap one another directly or through others: each run starts where
    every earlier one has ended. Linear: each span is looked at once."""
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
    return runs


def _joins(d: Detection) -> bool:
    """Whether ``d`` is joined with the values it overlaps
    (``Redactor._resolved``): a deny string (tier 0), whose rest of a value
    it cut into once went upstream — the rest of an email it occurs in, of
    another deny string it overlaps, of a block-mode value, unrefused — or
    a value that says so (``Detection.joins``)."""
    return d.joins or not d.tier


def _unite(text: str, spans: list[Detection]) -> list[Detection]:
    """``spans`` in start order, every run of overlapping spans joined into
    ONE detection over their union (``_joined``); a span overlapping no other
    is kept as it is. Linear: each span is looked at a fixed number of
    times, and each union sliced from ``text`` once."""
    return [_joined(text, run) for run in _overlapping(spans)]


def _joined(text: str, run: list[Detection]) -> Detection:
    """One run of overlapping spans as what is redacted in their place: a
    lone span itself; overlapping ones a tier-0 detection (always redacted,
    never a mode) over their union, typed after the run's first span and
    standing for the run (``Detection.members``: where the proxy cannot
    rewrite the text, ``Redactor.scan`` puts each to the overrides)."""
    first = run[0]
    if len(run) == 1:
        return first
    end = max(d.end for d in run)
    return Detection(
        first.start,
        end,
        first.detector_type,
        text[first.start : end],
        tier=0,
        members=tuple(run),
    )


def _covering(text: str, run: list[Detection]) -> Detection:
    """One run of overlapping values a joined cluster redacts
    (``Redactor._join``) as what is redacted in their place: a value
    covering the others itself (a lone value; a paper IBAN holding a card
    number), as the overlap sweep would have chosen it — unless a deny
    string is among the others, which no mode or override may then reach;
    any other run ONE tier-0 detection over their union (``_joined``)."""
    first = run[0]
    if all(d.end <= first.end and (d.tier or not first.tier) for d in run[1:]):
        return first
    return _joined(text, run)


def _clear_of(spans: list[Detection], others: list[Detection]) -> list[Detection]:
    """Those of ``others`` that overlap none of ``spans``, in their order
    (``spans`` are disjoint and start-sorted, so their ends are sorted too:
    a binary search finds the first one ending after a detection's start,
    its only possible overlap)."""
    ends = [d.end for d in spans]
    kept: list[Detection] = []
    for d in others:
        i = bisect_right(ends, d.start)
        if i == len(spans) or spans[i].start >= d.end:
            kept.append(d)
    return kept


def _not_covered(spans: list[Detection], values: Sequence[Detection]) -> list[Detection]:
    """Those of ``values`` with a character outside every one of ``spans``
    (start-sorted), in their order. Spans that overlap or touch are merged
    into the stretches they cover together, so a binary search finds the
    one stretch that may hold a value's first character (the first ending
    after its start): the value lies inside the spans exactly when it lies
    inside that stretch."""
    starts: list[int] = []
    ends: list[int] = []
    for d in spans:
        if ends and d.start <= ends[-1]:
            ends[-1] = max(ends[-1], d.end)
        else:
            starts.append(d.start)
            ends.append(d.end)
    kept: list[Detection] = []
    for d in values:
        i = bisect_right(ends, d.start)
        if i == len(ends) or d.start < starts[i] or d.end > ends[i]:
            kept.append(d)
    return kept


class _ApprovingEvery:
    """The override check of the resolution that counts the warn-mode values
    of a text refused while resolving (``Redactor._count_refused``): every
    value approved, so every block-mode value is forwarded as sent — and the
    requester's overrides are never consulted: nothing is asked, recorded or
    consumed."""

    def allows(self, detector_type: str, value: str) -> bool:
        return True

    def unoverridable(self) -> None:
        """Never called: that resolution has no final blocks."""


_APPROVING_EVERY = _ApprovingEvery()


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
        precomputed: PrecomputedTable | None = None,
        recorder: DetectionRecorder | None = None,
        encrypted_content_passthrough: bool = False,
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
        # Whether a rule is in warn mode: only then is a text's count of the
        # values it forwards as sent taken (``_settled``, ``_count_warned``),
        # so a policy without one never looks for them.
        self._warns = "warn" in self._modes.values()
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
        # The heavy (model-backed) detectors' results for the texts this
        # request will scan, computed ahead of it off the event loop
        # (with_precomputed, ner_prefetch); None runs them inline.
        self._precomputed = precomputed
        # Where every text this redactor scans is handed with its
        # detections (with_recorder: a registered scan sink's capture,
        # scan_capture); None records nothing and detects as before.
        self._recorder = recorder
        self.encrypted_content_passthrough = encrypted_content_passthrough

    @property
    def plan(self) -> DetectorPlan:
        """The detector plan this redactor detects with (shared by its
        copies): the one a precomputed table must be computed for
        (``with_precomputed``)."""
        return self._plan

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

    def with_precomputed(self, table: PrecomputedTable, plan: DetectorPlan) -> "Redactor":
        """A thin copy that takes each scanned text's heavy (model-backed)
        detections from ``table`` — computed for ``plan`` ahead of the
        redaction, off the event loop (ner_prefetch) — instead of running
        those detectors on the event loop; everything else is detected as
        before. A text missing from the table runs them inline, counted
        (``NerStats.prefetch_misses``, on each one that runs), and so does
        every text when ``plan`` is not this redactor's (a reload replaced
        the detectors meanwhile): the table only ever saves work, never
        changes a result. Its floor and budget copies keep the table."""
        copy = self._copy(self._floors, self._budget)
        copy._precomputed = table if plan is self._plan else {}
        return copy

    def with_recorder(self, recorder: DetectionRecorder) -> "Redactor":
        """A thin copy that hands every text it scans (``redact_text``,
        ``scan``) to ``recorder`` with the detections found in it, each
        with the detector it came from (``DetectionRecorder``) — the same
        detections, acted on exactly as without it. Its floor and budget
        copies keep it."""
        copy = self._copy(self._floors, self._budget)
        copy._recorder = recorder
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
            precomputed=self._precomputed,
            recorder=self._recorder,
            encrypted_content_passthrough=self.encrypted_content_passthrough,
        )

    def _detect(self, text: str) -> list[Detection]:
        """The configured detectors' detections in ``text``: the heavy
        ones' taken from the precomputed table when one is attached
        (``with_precomputed``) and it holds ``text``, else run here — a
        text missing from an attached table counted as a miss on each
        heavy detector that then runs on it (``DetectorPlan.detect``)."""
        table = self._precomputed
        if table is None:
            return self._plan.detect(text, self._allowlist)
        found = table.get(text)
        return self._plan.detect(text, self._allowlist, found, missed=found is None)

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

    def _winners(self, text: str, *, unrewritable: bool = False) -> list[Detection]:
        """The detections redaction acts on in ``text`` (``_settled``): the
        configured detectors' and the added deny strings'. ``unrewritable``:
        ``scan`` of text the proxy cannot rewrite, which a deny string
        refuses for good (``_settled``)."""
        if self._recorder is not None:
            return self._recorded_winners(text, self._recorder, unrewritable)
        found = self._detect(text)
        added = None if self._added_deny is None else self._added_deny.detect(text, self._allowlist)
        return self._settled(text, found, added, unrewritable=unrewritable)

    def _recorded_winners(
        self, text: str, recorder: DetectionRecorder, unrewritable: bool
    ) -> list[Detection]:
        """``_winners`` with a recorder attached (``with_recorder``): the
        same detections, each found with the detector it came from
        (``DetectorPlan.detect_sourced``, ``detect``'s detections in its
        order) — the heavy ones taken from the precomputed table as
        ``_detect`` takes them — handed to ``recorder`` with ``text``, then
        resolved exactly as ``_winners`` resolves them."""
        row = None
        missed = False
        table = self._precomputed
        if table is not None:
            row = table.get(text)
            missed = row is None
        found = self._plan.detect_sourced(text, self._allowlist, row, missed=missed)
        added = (
            self._added_deny.detect_sourced(text, self._allowlist)
            if self._added_deny is not None
            else []
        )
        recorder.record(text, found + added)
        return self._settled(
            text,
            [d for d, _source in found],
            None if self._added_deny is None else [d for d, _source in added],
            unrewritable=unrewritable,
        )

    def _settled(
        self,
        text: str,
        found: list[Detection],
        added: list[Detection] | None,
        *,
        unrewritable: bool = False,
    ) -> list[Detection]:
        """The detections redaction acts on in ``text`` (``_resolution``),
        the warn-mode values it forwards as sent counted once, right after
        resolution and before anything acts on them (``_count_warned``, only
        with a rule in warn mode): against what ``redact_text`` replaces —
        whatever becomes of the text, refused, redacted or (``scan``) sent
        as it is. A text refused while resolving (a block-mode value in a
        joined run, ``_join``, or among the winners an added deny string is
        taken in with, ``_absorb``) is counted all the same, as it would be
        with every block-mode value approved (``_count_refused``), before
        the refusal is raised on: where the refusing value stands changes
        nothing counted.

        ``unrewritable`` (``scan`` of text the proxy cannot rewrite): a deny
        string there, configured or added, refuses the text for good, so the
        requester's overrides are told so (``OverrideCheck.unoverridable``)
        before any value is put to them — a refusal for a block-mode value
        met first then mints no code, whose approval could never let the
        text through."""
        if (
            unrewritable
            and self._overrides is not None
            and (added or any(not d.tier for d in found))
        ):
            self._overrides.unoverridable()
        try:
            winners = self._resolution(text, found, added)
        except BlockedRequest:
            if self._warns:
                self._count_refused(text, found, added)
            raise
        if self._warns:
            self._count_warned(self._forwarded(found), self._replaced(winners))
        return winners

    def _resolution(
        self, text: str, found: list[Detection], added: list[Detection] | None
    ) -> list[Detection]:
        """The winners of the configured detections ``found``
        (``_resolved``) with the added deny strings' matches ``added`` taken
        in (``_absorb``; None: no overlay)."""
        configured = self._resolved(text, found)
        if added is None:
            return configured
        return self._absorb(text, configured, added)

    def _count_refused(
        self, text: str, found: list[Detection], added: list[Detection] | None
    ) -> None:
        """Counts the warn-mode values of a text a block-mode value refused
        while resolving (``_settled``) as the text counts with every
        block-mode value approved: against what that resolution replaces —
        resolved again with every value approved and no block final
        (``_ApprovingEvery``), so the requester's overrides are never
        consulted (nothing asked, recorded or consumed) and nothing is
        refused."""
        approving = copy.copy(self)
        approving._overrides = _APPROVING_EVERY
        approving._final_blocks = frozenset()
        winners = approving._resolution(text, found, added)
        self._count_warned(self._forwarded(found), self._replaced(winners))

    def _forwarded(self, found: list[Detection]) -> list[Detection]:
        """The values of ``found`` (the configured detections the redaction
        read, in the resolution's order) forwarded as sent, or that would
        be: those in warn or block mode — a block-mode value counts as
        forwarded whether or not an approval passes it, so a refusing value
        never changes what is counted. Never a deny string (configured; an
        added one is not among them, nor a union): it takes no mode, so no
        mode forwards it as sent. An ``llm`` finding (``ADDITIVE_TIER``) is
        one value as found, never the pieces the resolution cuts of it."""
        return [
            d for d in found if d.tier and self._modes.get(d.detector_type) in ("warn", "block")
        ]

    def _replaced(self, winners: list[Detection]) -> list[Detection]:
        """Those of ``winners`` (disjoint, start-sorted) ``redact_text``
        replaces by placeholders (``_replaces``)."""
        return [d for d in winners if self._replaces(d)]

    def _replaces(self, winner: Detection) -> bool:
        """Whether ``redact_text`` replaces ``winner`` by a placeholder: a
        deny string or a union (tier 0, never in a mode), a value in redact
        mode."""
        return not winner.tier or self._modes.get(winner.detector_type, "redact") == "redact"

    def _count_warned(self, values: list[Detection], replaced: list[Detection]) -> None:
        """Counts in ``warn_counts`` the warn-mode values a text forwards as
        sent, whole or in part — the one rule, which depends only on the
        values found and on what is replaced. ``values``: the values
        forwarded as sent, or that would be (``_forwarded``), in the
        resolution's order (by start, the longest first, then by priority);
        ``replaced``: the spans ``redact_text`` replaces by placeholders
        (disjoint, start-sorted) — also where the text is refused, or sent
        as it is by a scan. A value they cover whole sends nothing: it is not
        counted, nor does it hide another (``_not_covered``). The others
        are read as overlap resolution reads values: the overlap sweep
        (``_sweep``) takes the values of every detector but the ``llm``
        backend first, then, among themselves, the ``llm`` backend's
        findings (``ADDITIVE_TIER``) with a character outside every
        replaced span and every value taken before — so an llm finding,
        which never takes another detector's place in the resolution
        (``_resolve_overlaps``), never hides another detector's value here
        either, and counts for what it sends beyond them. Each warn-mode
        value taken counts once, under its type: two detectors finding one
        value count it once, and of values forwarded as sent that overlap
        one another the sweep counts those it takes — a warn-mode value
        overlapping a block-mode or another warn-mode value taken before it
        is not counted, however much of it is sent (whole, too). A deny
        string replacing the same characters counts alike whether
        configured or added by an access gate (the values come from the
        configured detections, the spans from the final winners). Linear,
        but for a binary search per value and, with an llm finding among
        them, a sort of the spans it is read against."""
        rules = [d for d in values if d.tier != ADDITIVE_TIER]
        taken = _sweep(_not_covered(replaced, rules))
        findings = [d for d in values if d.tier == ADDITIVE_TIER]
        if findings:
            # The replaced spans and the values taken may overlap: merged.
            held = sorted(replaced + taken, key=lambda d: d.start)
            taken += _sweep(_not_covered(held, findings))
        for d in taken:
            if self._modes.get(d.detector_type) == "warn":
                self.warn_counts[d.detector_type] += 1

    def _resolved(self, text: str, detections: list[Detection]) -> list[Detection]:
        """The configured detections' winners: ``_resolve_overlaps``'s,
        except in a run of values overlapping one another directly or
        through others that holds one that joins (``_joins``: a deny
        string, a card read inside a run of digit groups, an IBAN in the
        paper format, a value ending before a run of dots since 1.12.1),
        which is ``_join``ed. Without one, the sweep's own, byte for byte;
        with one, every character the sweep without the joining values
        would redact is redacted, and every value it refuses for still
        refuses — and, a deny string among them, every character the sweep
        with it would redact too (it won every overlap there, and the rest
        of what it cut into went upstream). Save for an additive detection
        a value forwarded as sent covers: the join may keep a different one
        of two overlapping warn-mode values than the sweep does, and the
        additive detection is cut around the one kept. Additive detections
        (``ADDITIVE_TIER``: the llm NER backend's) are never joined: they
        keep only what lies outside the winners, exactly as
        ``_resolve_overlaps`` treats them, so one never takes the place of
        a card or an IBAN, or of the type its placeholder carries."""
        if not any(_joins(d) for d in detections):
            return _resolve_overlaps(detections)
        joined: list[Detection] = []
        swept: list[Detection] = []
        for run in _overlapping([d for d in detections if d.tier != ADDITIVE_TIER]):
            if any(_joins(d) for d in run):
                joined += self._join(text, run)
            else:
                swept += run
        winners = sorted(joined + _resolve_tiers(swept), key=lambda d: d.start)
        additive = sorted(
            (d for d in detections if d.tier == ADDITIVE_TIER),
            key=lambda d: (d.start, -(d.end - d.start), d.priority),
        )
        if not additive:
            return winners
        return sorted(winners + _outside(winners, _sweep(additive)), key=lambda d: d.start)

    def _join(self, text: str, run: list[Detection]) -> list[Detection]:
        """One run of overlapping values holding one that joins, redacted so
        that no value loses characters to another: a value in block mode
        refuses the text (``BlockedRequest``) unless an approved override
        passes it; every value that redacts (redact mode, a deny string) is
        redacted — values overlapping one another as ONE placeholder over
        their union typed after the first (``_covering``); a value forwarded
        as sent (warn mode, an approved block) only where no redacted value
        covers it: one overlapping such a value yields to it (a warn-mode
        one counted by ``_settled``, ``_count_warned``, when part of it is
        still sent)."""
        redacting, passing = self._by_mode(run)
        covered = [_covering(text, values) for values in _overlapping(redacting)]
        kept = _clear_of(covered, passing)
        # Disjoint, so in start order once ``_resolved`` sorts them.
        return covered + _sweep(kept)

    def _by_mode(self, detections: list[Detection]) -> tuple[list[Detection], list[Detection]]:
        """``detections`` as the ones that redact (redact mode, a deny
        string) and the ones forwarded as sent (warn mode, a block an
        approved override passes); a block no approval passes refuses the
        text right here (``BlockedRequest``), as ``redact_text`` would."""
        redacting: list[Detection] = []
        passing: list[Detection] = []
        for d in detections:
            mode = self._modes.get(d.detector_type, "redact") if d.tier else "redact"
            if mode == "block" and not self._overridden(d):
                raise BlockedRequest(d.detector_type)
            (redacting if mode == "redact" else passing).append(d)
        return redacting, passing

    def _absorb(
        self, text: str, winners: list[Detection], added: list[Detection]
    ) -> list[Detection]:
        """The configured winners of ``text`` with the ADDED deny strings'
        matches ``added`` taken in, so that they only ever TIGHTEN what the
        configured policy does. A winner that refuses (block mode, no
        approved override) refuses the text right here, as ``redact_text``
        would. A winner the policy forwards as sent (warn mode, an approved
        block) yields to an added match overlapping it: the match is
        redacted, the rest of the value goes as it would have (a warn-mode
        one counted by ``_settled``, ``_count_warned``, when part of it is
        still sent). A winner
        that redacts (redact mode, a configured deny string) keeps its
        effect, and the UNION of its span and every added match overlapping
        it is redacted as one placeholder (``_unite``) — so neither the
        value the policy redacts nor the added string leaves in part."""
        redacting, passing = self._by_mode(winners)
        # Every united span an added match is part of is tier 0, so the
        # tier-0 rule (``_resolve_tiers``) drops each passing winner one
        # overlaps; the other spans are disjoint from every winner (the
        # configured winners are disjoint), so nothing else changes. Not
        # ``_resolve_overlaps``: it would cut an additive (llm) passing
        # winner around the union into new values, each put to the
        # overrides again and refused as a fragment.
        united = _unite(text, redacting + added)
        return _resolve_tiers(sorted(united + passing, key=lambda d: d.start))

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
            # Tier-0 (deny, an added deny string's union with what it
            # overlaps, joined values' union) always redacts: keying modes
            # by type could let a user-chosen deny type accidentally inherit
            # a warn/block mode from a rule sharing that type.
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
                # no placeholder — the value WILL go upstream (counted
                # before this loop, ``_settled``).
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
        winner raises BlockedRequest; a warn-mode winner stays where it is;
        every other winner — a deny string always — is returned as its
        detector type, counted.

        The warn-mode values are counted in ``warn_counts`` exactly as
        ``redact_text`` counts them (``_settled``), against every span it
        would replace, whatever becomes of the text: refused — here (a
        block, wherever it stands) or by the caller (a value found) —,
        redacted after all (``redactable``), or sent as it is (not
        ``redactable``, every value that would refuse it approved: nothing
        found). So the overrides an approval passes never change what is
        counted: a warn-mode value an approved redact-mode value covers
        whole goes upstream uncounted as a warning, the approved value
        counted as an override used.

        Every refusing winner but a deny string is first put to the
        requester's overrides (``with_overrides``): an approved value is
        skipped (``TextScan.overridden``). ``redactable``: the caller will
        redact this text after all (convert mode sends a document's text in
        its file's place), so only a block-mode winner refuses — a value it
        returns is redacted, not refused, and is never put to the
        overrides. A deny string in text that is not ``redactable`` refuses
        it for good (``OverrideCheck.unoverridable``, said before any value
        is put to the overrides, ``_settled``) — configured or added, with
        the value it is joined with or overlaps (``_join``, ``_absorb``: the
        proxy could not redact their union). A span joining values with no
        deny string among them (``_join``) is put to the overrides value by
        value (``_joined_passes``): skipped only when each is approved.
        Charged against the string budget as one string."""
        self.charge(1)
        found: Counter[str] = Counter()
        overridden = False
        overrides = self._overrides
        winners = self._winners(text, unrewritable=not redactable)
        for d in winners:
            if not d.tier:
                # Deny strings (tier 0) take no mode, exactly as in
                # redact_text, and are the operator's always-redact list —
                # or an access gate's, added to it — never put to the
                # requester's overrides, nor the span joining one with the
                # values it overlaps. A span joining values is, each of
                # them.
                if not redactable and overrides is not None and self._joined_passes(d, overrides):
                    overridden = True
                else:
                    found[d.detector_type] += 1
                continue
            mode = self._modes.get(d.detector_type)
            if mode == "warn":
                # Forwarded as sent: counted already (_settled).
                continue
            if (mode == "block" or not redactable) and self._overridden(d):
                overridden = True
            elif mode == "block":
                raise BlockedRequest(d.detector_type)
            else:
                found[d.detector_type] += 1
        return TextScan(found, overridden)

    def _joined_passes(self, d: Detection, overrides: OverrideCheck) -> bool:
        """Whether ``overrides`` let a tier-0 winner of text the proxy
        cannot rewrite (``scan``) through as sent: a span joining values
        (``Detection.members``) when every one of them is approved. Each is
        put to them as a refusing winner of its own would be — all of them,
        none skipped, so a refusal records them all for ONE code, whose
        approval then passes them all. A deny string is never put to them,
        nor a span with one among its values (a configured one's join,
        ``_join``; an added one's union, ``_absorb``), nor one holding a
        value whose block the detection overlay added (``_final_blocks``;
        its block mode refuses first, in ``_by_mode``): the refusal is final
        (``unoverridable``), no grant consulted."""
        members = d.members
        if not members or any(not m.tier or m.detector_type in self._final_blocks for m in members):
            overrides.unoverridable()
            return False
        # Every value asked before any answer counts (no short cut).
        approved = [overrides.allows(m.detector_type, m.value) for m in members]
        return all(approved)

    @property
    def blocks(self) -> bool:
        """Whether a rule is in block mode: whether ``redact_text`` can
        refuse a text (BlockedRequest) at all."""
        return "block" in self._modes.values()

    def blocked_type(self, text: str) -> str | None:
        """The detector type ``redact_text`` would refuse ``text`` for — a
        block-mode value joined with a joining one (``_join``), else its
        first block-mode winner, under the same allowlists, deny strings
        and overlap resolution — else None. Nothing is issued, counted or
        charged: a check AHEAD of the redaction that will scan ``text`` and
        count it (an upload's pieces, before its binary parts are handed to
        an upload inspector). Added deny strings take no part: they never
        block, nor take a block-mode winner's place (``_absorb``)."""
        try:
            winners = self._resolved(text, self._detect(text))
        except BlockedRequest as blocked:
            # A block-mode value joined with a joining one (``_join``).
            return blocked.detector_type
        for d in winners:
            # Deny strings (tier 0) never block, exactly as in redact_text.
            if d.tier and self._modes.get(d.detector_type) == "block" and not self._overridden(d):
                return d.detector_type
        return None

    def redact_json(self, obj: Any) -> Any:
        return transform_strings(
            obj,
            self.redact_text,
            encrypted_content_passthrough=self.encrypted_content_passthrough,
        )
