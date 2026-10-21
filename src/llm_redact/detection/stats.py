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
per model entity or window (``ungrounded``, ``errors``, ``breaker_opens``
and ``breaker_short_circuits`` only the llm backend's, which asks a model
server), and two count where the model ran: the strings the redaction itself
ran a backend on (``inline_calls``; on a request's path, the event loop) and
how many of those a request's NER prefetch should have covered
(``prefetch_misses``). To add a counter, add a field: the ``/status`` block
lists every field.
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
    # Entities a model server returned (the llm backend) whose text does
    # not occur in the text it was given, is shorter than two characters
    # or holds a placeholder's guillemet (an earlier turn's token the text
    # carries), and entities that came without a text or a type: never
    # redacted.
    ungrounded: int = 0
    # Strings a model server failed on (the llm backend: a timeout, an
    # error status, a connection fault, an answer it could not read). With
    # on_error = "block" each refused its request; with "skip" each went on
    # without the backend's findings past the failure. Strings failed at
    # once by its open circuit breaker are counted here too.
    errors: int = 0
    # Times the llm backend's circuit breaker opened (a timeout, a
    # transport fault or an HTTP 5xx; a failed probe opens it again).
    breaker_opens: int = 0
    # Strings the llm backend failed at once, without asking its server,
    # because its circuit breaker was open (each also counted in errors).
    breaker_short_circuits: int = 0
    # Strings this backend was run on BY THE REDACTION ITSELF
    # (DetectorPlan._run) instead of ahead of it on the NER worker thread
    # (ner_prefetch). On a request's path that is the event loop: every such
    # call held up every other request on the proxy for one string's
    # inference. A redaction preview (ProxyState.preview, which the
    # llm-redact-pro dashboard runs on a worker thread of its own in its
    # releases after 0.17.1, on the event loop in 0.17.1 and earlier) has no
    # prefetched results and is counted too. Zero while every request shape
    # is prefetched and nobody runs a preview.
    inline_calls: int = 0
    # Strings a request's redaction looked up in its precomputed NER results
    # and did not find (or found computed for detectors a reload replaced):
    # run inline instead, and counted in inline_calls too.
    prefetch_misses: int = 0

    def as_dict(self) -> dict[str, int]:
        """Every counter by name, in field order."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
