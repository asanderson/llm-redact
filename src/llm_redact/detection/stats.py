"""NER coverage counters: what each NER backend scanned, skipped and dropped.

A model reads a bounded amount of text, and a string the NER layer never
read is a coverage gap the regex rules alone cover. Every gap is counted
here, per backend, so ``/status`` (``detection.ner``), ``/metrics``
(``llm_redact_ner_*``) and ``llm-redact status`` can show it instead of
hiding it.

One :class:`NerStats` is owned by each NER backend detector and lives
exactly as long as it does: a reload that rebuilds the detectors starts
fresh counters (Prometheus reads the drop as a counter reset). Counts only —
never a value, a window's text or a label the model returned.

Each string a backend is handed lands in exactly one of
:data:`STRING_OUTCOMES`: read in one model call (``scanned_whole``), read in
overlapping windows (``scanned_windowed``, each window counted in
``windows``), or not read at all because it is longer than
``[detection.ner] max_chars`` (``skipped_max_chars``). The other counters are
per model entity or window, and two count where the model ran: on the
event loop (``inline_calls``) and how many of those a request's NER prefetch
should have covered (``prefetch_misses``). To add a counter, add a field:
the ``/status`` block lists every field.
"""

from dataclasses import dataclass, fields

# The mutually exclusive outcomes of one string handed to a backend, in
# the order /metrics labels them.
STRING_OUTCOMES = ("scanned_whole", "scanned_windowed", "skipped_max_chars")


@dataclass
class NerStats:
    """One NER backend's coverage counters (integers only)."""

    # Strings read in one model call.
    scanned_whole: int = 0
    # Strings longer than one model window, read in overlapping windows.
    scanned_windowed: int = 0
    # Strings longer than [detection.ner] max_chars: not read by the model
    # (the regex rules still scan them).
    skipped_max_chars: int = 0
    # Windows the windowed strings were read in (model calls for them).
    windows: int = 0
    # Windows (a string read whole counts as one) longer than the model's
    # token limit because one word alone exceeds it: the model may read
    # only part of them.
    windows_truncated: int = 0
    # Model entities whose placeholder type does not fit the placeholder
    # grammar (labels.LabelPolicy.classify): never emitted.
    labels_dropped: int = 0
    # Model entities of a requested type whose offsets the text does not
    # contain (or that came without offsets): never redacted, since only
    # the exact sent text can be restored.
    offsets_dropped: int = 0
    # Strings this backend was run on BY THE EVENT LOOP (DetectorPlan._run)
    # instead of ahead of the redaction on the NER worker thread
    # (ner_prefetch): every call that held up every other request on the
    # proxy for one string's inference. Zero while every request shape is
    # prefetched.
    inline_calls: int = 0
    # Strings a request's redaction looked up in its precomputed NER results
    # and did not find (or found computed for detectors a reload replaced):
    # run inline instead, and counted in inline_calls too.
    prefetch_misses: int = 0

    def as_dict(self) -> dict[str, int]:
        """Every counter by name, in field order."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
