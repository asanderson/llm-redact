"""Outbound half: replace detected values with vault-issued placeholders."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from llm_redact.detection.base import Detection, Detector
from llm_redact.detection.engine import Allowlist, detect_all
from llm_redact.jsonwalk import transform_strings
from llm_redact.vault import Vault


class BlockedRequest(Exception):
    """A block-mode rule matched: the whole request must be rejected.

    Carries only the detector type — never the matched value — so handlers
    can log and report it without violating the never-log-values rule.
    """

    def __init__(self, detector_type: str) -> None:
        super().__init__(detector_type)
        self.detector_type = detector_type


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
        detectors: Sequence[Detector],
        vault: Vault,
        allowlist: Allowlist,
        counts: "Counter[str] | None" = None,
        modes: Mapping[str, str] | None = None,
        warn_counts: "Counter[str] | None" = None,
    ) -> None:
        self._detectors = detectors
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

    def redact_text(self, text: str) -> str:
        detections = _resolve_overlaps(detect_all(self._detectors, text, self._allowlist))
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
            parts.append(self._vault.placeholder_for(d.detector_type, d.value))
            self.counts[d.detector_type] += 1
            cursor = d.end
        parts.append(text[cursor:])
        return "".join(parts)

    def redact_json(self, obj: Any) -> Any:
        return transform_strings(obj, self.redact_text)
