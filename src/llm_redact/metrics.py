"""Prometheus text exposition, hand-rolled (stdlib only, no client library).

Always-on and in-memory, independent of the opt-in audit log. Metadata only —
metric names, detector types, providers, status codes — consistent with the
proxy's never-log-values posture.
"""

import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from llm_redact.detection.stats import NerStats

_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
# The proxy's own share of a request is milliseconds on healthy hardware:
# finer buckets than the end-to-end duration's, which include the provider.
_OVERHEAD_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

# Every response the PROXY ITSELF generates instead of forwarding the request
# (or, for a realtime connection, instead of relaying it or a frame), one kind
# each — ``llm_redact_local_refusals_total{kind,provider}``. An upstream's own
# answer, whatever its status, is never one. docs/observability.md documents
# each (``LOCAL_REFUSAL_KINDS`` is pinned against it); a refusal site passes
# its kind to ``ProxyState.record_request(refusal=)`` — or, for a refusal that
# is never recorded (its path may hold a key), counts it alone
# (``ProxyState.count_local_refusal``).
LocalRefusal = Literal[
    "access_gate",
    "audit_unavailable",
    "authorization",
    "binary_values",
    "blocked_value",
    "budget",
    "credential_protocol",
    "delivery_fault",
    "disabled_provider",
    "identity_path",
    "identity_route",
    "method_override",
    "misaddressed",
    "no_route",
    "no_upstream",
    "object_access",
    "override_fault",
    "override_raced",
    "placeholder_limit",
    "realtime_unavailable",
    "redirect_refused",
    "reload",
    "request_origin",
    "request_target",
    "route_unsupported",
    "scanned_body",
    "sealed_session",
    "too_large",
    "too_many_strings",
    "unattributed",
    "unchecked_body",
    "unredactable",
    "unscanned_upload",
    "unsupported_encoding",
    "upstream_auth",
    "upstream_fault",
    "vault_fault",
    "verbatim_field",
]
LOCAL_REFUSAL_KINDS: tuple[str, ...] = get_args(LocalRefusal)

# A plugin's own gauges (``plugin_metric_lines``): a name under the core's
# prefix, label names and values from a small fixed charset — so neither can
# carry a path, an e-mail address or a key in its usual spelling — and at
# most a bounded number of samples and labels. The charset is a SHAPE check
# only: a user name such as ``alice_smith`` fits it, so keeping label values
# to small fixed sets is the plugin's obligation. The core also drops a label
# value holding a run of ``_ID_LIKE_RUN`` hexadecimal characters with a digit
# among them (a key, a hash, an id, a long number) — a backstop, not a proof.
PLUGIN_METRIC_PREFIX = "llm_redact_"
_PLUGIN_METRIC_NAME = re.compile(r"llm_redact_[a-z][a-z0-9_]{0,62}")
_PLUGIN_LABEL_NAME = re.compile(r"[a-z][a-z0-9_]{0,31}")
_PLUGIN_LABEL_VALUE = re.compile(r"[a-z0-9_]{1,32}")
_ID_LIKE_RUN = 8
_HEX_RUN = re.compile(rf"[0-9a-f]{{{_ID_LIKE_RUN},}}")
# Label names Prometheus itself gives meaning to (the histogram bound, the
# target labels a scrape attaches): never a plugin's.
_RESERVED_LABELS = frozenset({"le", "quantile", "job", "instance"})
MAX_PLUGIN_SAMPLES = 256
MAX_PLUGIN_LABELS = 4
# Metric families the core renders itself: a plugin sample never shadows one.
CORE_METRIC_FAMILIES = frozenset(
    {
        "llm_redact_info",
        "llm_redact_detections_total",
        "llm_redact_rehydrations_total",
        "llm_redact_warnings_total",
        "llm_redact_blocked_total",
        "llm_redact_requests_total",
        "llm_redact_request_duration_seconds",
        "llm_redact_proxy_overhead_seconds",
        "llm_redact_local_refusals_total",
        "llm_redact_compaction_forks_total",
        "llm_redact_upstream_errors_total",
        "llm_redact_bookkeeping_errors_total",
        "llm_redact_connections_closed_total",
        "llm_redact_overrides_used_total",
        "llm_redact_unscanned_uploads_total",
        "llm_redact_inspected_uploads_total",
        "llm_redact_ner_strings_total",
        "llm_redact_ner_windows_total",
        "llm_redact_ner_windows_truncated_total",
        "llm_redact_ner_labels_dropped_total",
        "llm_redact_ner_offsets_dropped_total",
        "llm_redact_routed_requests_total",
        "llm_redact_reissues_total",
        "llm_redact_audit_sink_batches_total",
        "llm_redact_audit_sink_rows_dropped_total",
        "llm_redact_map_write_queue_depth",
        "llm_redact_map_write_wait_timeouts_total",
        "llm_redact_map_writes_mode",
        "llm_redact_vault_entries",
        "llm_redact_vault_sessions",
        "llm_redact_start_time_seconds",
        "llm_redact_uptime_seconds",
    }
)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class DurationHistogram:
    """Cumulative histogram in Prometheus semantics."""

    def __init__(self, buckets: tuple[float, ...] = _BUCKETS) -> None:
        self._buckets = buckets
        self._counts = [0] * (len(buckets) + 1)  # trailing slot = +Inf
        self._sum = 0.0
        self._count = 0

    def observe(self, seconds: float) -> None:
        self._sum += seconds
        self._count += 1
        for i, upper in enumerate(self._buckets):
            if seconds <= upper:
                self._counts[i] += 1
        self._counts[-1] += 1

    def render(self, name: str) -> Iterable[str]:
        yield f"# HELP {name} Proxy request duration in seconds."
        yield f"# TYPE {name} histogram"
        yield from self.series(name, "")

    def series(self, name: str, labels: str) -> Iterable[str]:
        """The bucket/sum/count lines only (no HELP/TYPE), with an optional
        label set like 'provider="anthropic",streamed="true"' so several
        labeled histograms can share one metric name."""
        # observe() increments every bucket whose bound covers the value, so
        # the stored counts are already cumulative (Prometheus semantics).
        prefix = f"{labels}," if labels else ""
        suffix = f"{{{labels}}}" if labels else ""
        for i, upper in enumerate(self._buckets):
            yield f'{name}_bucket{{{prefix}le="{upper}"}} {self._counts[i]}'
        yield f'{name}_bucket{{{prefix}le="+Inf"}} {self._count}'
        yield f"{name}_sum{suffix} {self._sum}"
        yield f"{name}_count{suffix} {self._count}"


class Metrics:
    def __init__(self, version: str) -> None:
        self._version = version
        self._started = time.time()
        # (provider, status) -> count; provider is anthropic/openai/passthrough.
        self.requests: Counter[tuple[str, str]] = Counter()
        # (provider, streamed) -> histogram: per-provider p95 plus a streamed
        # dimension. The provider set is bounded, so label cardinality is safe.
        self._durations: dict[tuple[str, str], DurationHistogram] = {}
        # Routing layer (llm-redact-pro; the core counts from the plugin_api
        # contract's facts): (upstream, rule) -> requests the routing layer
        # delivered, and (from_upstream, to_upstream) -> fallback re-issues.
        # Both label sets are config-bounded (upstream names, rule ids),
        # never request-derived.
        self.routed: Counter[tuple[str, str]] = Counter()
        self.reissues: Counter[tuple[str, str]] = Counter()
        # (kind, provider) -> responses the proxy generated itself instead
        # of forwarding (LOCAL_REFUSAL_KINDS): both label sets are bounded —
        # a fixed enum and the configured provider names.
        self.local_refusals: Counter[tuple[str, str]] = Counter()
        # provider -> the proxy's own share of each HTTP request's duration
        # (``observe_overhead``): the end-to-end duration minus the time it
        # waited on the upstream and on the client.
        self._overheads: dict[str, DurationHistogram] = {}
        # The effective [vault] map_writes (restart-only, set once by the
        # proxy at startup): rendered as an info-style gauge. None: not set
        # (a bare Metrics, as in tests) — nothing rendered.
        self.map_writes_mode: str | None = None

    def observe_request(
        self, provider: str | None, status: int | None, seconds: float, streamed: bool = False
    ) -> None:
        prov = provider or "passthrough"
        self.requests[(prov, str(status or 0))] += 1
        key = (prov, "true" if streamed else "false")
        self._durations.setdefault(key, DurationHistogram()).observe(seconds)

    def observe_overhead(self, provider: str | None, seconds: float) -> None:
        """The proxy's own time for one HTTP request (never negative)."""
        prov = provider or "passthrough"
        histogram = self._overheads.get(prov)
        if histogram is None:
            histogram = self._overheads[prov] = DurationHistogram(_OVERHEAD_BUCKETS)
        histogram.observe(max(seconds, 0.0))

    def count_local_refusal(self, kind: LocalRefusal, provider: str | None) -> None:
        """One response the proxy generated itself (``kind``), by the
        provider the request was attributed to ("passthrough" when none:
        the ``llm_redact_requests_total`` label)."""
        self.local_refusals[(kind, provider or "passthrough")] += 1

    def render(
        self,
        *,
        detections: Counter[str],
        rehydrations: Counter[str],
        warnings: Counter[str],
        blocked: Counter[str],
        vault_entries: int,
        vault_sessions: int,
        compaction_forks: int = 0,
        upstream_errors: "Counter[str] | None" = None,
        bookkeeping_errors: "Counter[str] | None" = None,
        connections_closed: "Counter[str] | None" = None,
        unscanned_uploads: "Counter[str] | None" = None,
        inspected_uploads: "Counter[tuple[str, str]] | None" = None,
        overrides_used: "Counter[str] | None" = None,
        audit_sink_batches: "Counter[str] | None" = None,
        audit_sink_rows_dropped: "Counter[str] | None" = None,
        map_write_queue_depth: int = 0,
        map_write_wait_timeouts: "Counter[str] | None" = None,
        ner_stats: "Iterable[tuple[str, NerStats]]" = (),
    ) -> str:
        lines: list[str] = []
        lines.append("# HELP llm_redact_info Build information.")
        lines.append("# TYPE llm_redact_info gauge")
        lines.append(f'llm_redact_info{{version="{_escape_label(self._version)}"}} 1')

        for name, help_text, counter in (
            ("llm_redact_detections_total", "Values redacted, by detector type.", detections),
            ("llm_redact_rehydrations_total", "Values restored, by detector type.", rehydrations),
            (
                "llm_redact_warnings_total",
                "Warn-mode detections (value forwarded unredacted), by detector type.",
                warnings,
            ),
            (
                "llm_redact_blocked_total",
                "Requests rejected by block-mode rules, by detector type.",
                blocked,
            ),
        ):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            for label, count in sorted(counter.items()):
                lines.append(f'{name}{{type="{_escape_label(label)}"}} {count}')

        lines.append("# HELP llm_redact_requests_total Requests proxied, by provider and status.")
        lines.append("# TYPE llm_redact_requests_total counter")
        for (provider, status), count in sorted(self.requests.items()):
            lines.append(
                f'llm_redact_requests_total{{provider="{_escape_label(provider)}",'
                f'status="{_escape_label(status)}"}} {count}'
            )

        duration_name = "llm_redact_request_duration_seconds"
        lines.append(
            f"# HELP {duration_name} Proxy request duration in seconds, by provider and streamed."
        )
        lines.append(f"# TYPE {duration_name} histogram")
        for (provider, streamed), histogram in sorted(self._durations.items()):
            labels = f'provider="{_escape_label(provider)}",streamed="{streamed}"'
            lines.extend(histogram.series(duration_name, labels))

        overhead_name = "llm_redact_proxy_overhead_seconds"
        lines.append(
            f"# HELP {overhead_name} The proxy's own time per HTTP request, by provider: the"
            " request duration minus the time spent waiting on the upstream (sending, its"
            " answer's headers and body) and on the client (its request body, a stream's"
            " consumer). Realtime connections are not observed."
        )
        lines.append(f"# TYPE {overhead_name} histogram")
        for provider, histogram in sorted(self._overheads.items()):
            lines.extend(histogram.series(overhead_name, f'provider="{_escape_label(provider)}"'))

        lines.append(
            "# HELP llm_redact_local_refusals_total Responses the proxy generated itself instead"
            " of forwarding the request (realtime: instead of relaying the connection or a"
            " frame), by kind and provider. An upstream's own answer is never counted here."
        )
        lines.append("# TYPE llm_redact_local_refusals_total counter")
        for (kind, provider), count in sorted(self.local_refusals.items()):
            lines.append(
                f'llm_redact_local_refusals_total{{kind="{_escape_label(kind)}",'
                f'provider="{_escape_label(provider)}"}} {count}'
            )

        lines.append(
            "# HELP llm_redact_compaction_forks_total New per-conversation sessions whose"
            " first message already carried placeholders (history compaction signature)."
        )
        lines.append("# TYPE llm_redact_compaction_forks_total counter")
        lines.append(f"llm_redact_compaction_forks_total {compaction_forks}")

        lines.append(
            "# HELP llm_redact_upstream_errors_total Upstream transport faults"
            " (connect/read/timeout/mid-body drop), failed closed with a 502, by provider."
        )
        lines.append("# TYPE llm_redact_upstream_errors_total counter")
        for provider, count in sorted((upstream_errors or Counter()).items()):
            lines.append(
                f'llm_redact_upstream_errors_total{{provider="{_escape_label(provider)}"}} {count}'
            )

        lines.append(
            "# HELP llm_redact_bookkeeping_errors_total Faults in the proxy's own bookkeeping,"
            " by stage: after the upstream answered, session bookkeeping (response_id,"
            " object_ids, listing, response_observer — contained, the answer still delivered)"
            " and delivery"
            " (restoring the answer — a recorded 502); before any upstream contact, vault"
            " (issuing a request's placeholders failed — a recorded 503, a realtime frame"
            " closes 1011); vault_check (a vault view's staleness check could not read its"
            " database — contained, the cache kept); recheck (an open connection's access"
            " re-check failed — the connection closed); map_write_wait (an answer sent"
            " before its map writes landed); plugin_metrics (a plugin's metrics samples"
            " unreadable or dropped as invalid)."
        )
        lines.append("# TYPE llm_redact_bookkeeping_errors_total counter")
        for stage, count in sorted((bookkeeping_errors or Counter()).items()):
            lines.append(
                f'llm_redact_bookkeeping_errors_total{{stage="{_escape_label(stage)}"}} {count}'
            )

        lines.append(
            "# HELP llm_redact_connections_closed_total Open long-lived connections (realtime"
            " relays, live-events streams) closed because their admission ended, by cause:"
            " revoked (the access gate revoked the user or credential), recheck (a periodic"
            " re-check refused it), recheck_error (a re-check failed or timed out — fail closed)."
        )
        lines.append("# TYPE llm_redact_connections_closed_total counter")
        for cause, count in sorted((connections_closed or Counter()).items()):
            lines.append(
                f'llm_redact_connections_closed_total{{cause="{_escape_label(cause)}"}} {count}'
            )

        lines.append(
            "# HELP llm_redact_overrides_used_total Refusals passed on the requester's approved"
            " override (the value, body or binary part FORWARDED as sent), by kind: once"
            " (a one-time grant) or always (an every-time rule)."
        )
        lines.append("# TYPE llm_redact_overrides_used_total counter")
        for kind, count in sorted((overrides_used or Counter()).items()):
            lines.append(f'llm_redact_overrides_used_total{{kind="{_escape_label(kind)}"}} {count}')

        lines.append(
            "# HELP llm_redact_unscanned_uploads_total Binary file parts of uploads forwarded"
            " UNSCANNED with the client's own credential ([detection] binary_uploads ="
            ' "forward"), by provider.'
        )
        lines.append("# TYPE llm_redact_unscanned_uploads_total counter")
        for provider, count in sorted((unscanned_uploads or Counter()).items()):
            lines.append(
                f'llm_redact_unscanned_uploads_total{{provider="{_escape_label(provider)}"}}'
                f" {count}"
            )

        lines.append(
            "# HELP llm_redact_inspected_uploads_total Binary file parts of uploads read as text"
            " by an upload inspector, by provider and outcome (clean: forwarded after a clean"
            " scan of the EXTRACTED text only; clean_refused: scanned clean, the upload"
            " refused; converted: the file replaced by its redacted extracted text;"
            " overridden: sent as is because an approved refusal override let its values"
            " through)."
        )
        lines.append("# TYPE llm_redact_inspected_uploads_total counter")
        for (provider, outcome), count in sorted((inspected_uploads or Counter()).items()):
            lines.append(
                "llm_redact_inspected_uploads_total{"
                f'provider="{_escape_label(provider)}",outcome="{_escape_label(outcome)}"}}'
                f" {count}"
            )

        lines.extend(_ner_lines(list(ner_stats)))

        lines.append(
            "# HELP llm_redact_routed_requests_total Requests delivered through the routing"
            " layer, by the upstream that produced the response and the rule that chose it."
        )
        lines.append("# TYPE llm_redact_routed_requests_total counter")
        for (upstream, rule), count in sorted(self.routed.items()):
            lines.append(
                f'llm_redact_routed_requests_total{{upstream="{_escape_label(upstream)}",'
                f'rule="{_escape_label(rule)}"}} {count}'
            )

        lines.append(
            "# HELP llm_redact_reissues_total Fallback re-issues to the next chain member,"
            " by the upstream that failed and the one that took over."
        )
        lines.append("# TYPE llm_redact_reissues_total counter")
        for (from_upstream, to_upstream), count in sorted(self.reissues.items()):
            lines.append(
                f'llm_redact_reissues_total{{from_upstream="{_escape_label(from_upstream)}",'
                f'to_upstream="{_escape_label(to_upstream)}"}} {count}'
            )

        for name, help_text, sink_counter in (
            (
                "llm_redact_audit_sink_batches_total",
                "Audit row batches an off-machine audit sink uploaded, by sink (s3, azure).",
                audit_sink_batches,
            ),
            (
                "llm_redact_audit_sink_rows_dropped_total",
                "Audit rows an off-machine audit sink dropped (an upload failed, its buffer"
                " was full, credentials or the batch key were missing), by sink.",
                audit_sink_rows_dropped,
            ),
        ):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            for sink, count in sorted((sink_counter or Counter()).items()):
                lines.append(f'{name}{{sink="{_escape_label(sink)}"}} {count}')

        lines.append(
            "# HELP llm_redact_map_write_queue_depth Durable vault map writes (Responses"
            " chains, stored-object owners, Live resumption handles) queued or in flight on"
            " the background writer; 0 without one."
        )
        lines.append("# TYPE llm_redact_map_write_queue_depth gauge")
        lines.append(f"llm_redact_map_write_queue_depth {map_write_queue_depth}")
        lines.append(
            '# HELP llm_redact_map_write_wait_timeouts_total [vault] map_writes = "before_answer"'
            " answers sent before their map writes landed, by cause: bound (the wait reached"
            " map_write_wait_seconds) or stuck (not waited for: an earlier write the writer"
            " is stuck on has not landed)."
        )
        lines.append("# TYPE llm_redact_map_write_wait_timeouts_total counter")
        for cause, count in sorted((map_write_wait_timeouts or Counter()).items()):
            lines.append(
                f'llm_redact_map_write_wait_timeouts_total{{cause="{_escape_label(cause)}"}}'
                f" {count}"
            )
        lines.append(
            "# HELP llm_redact_map_writes_mode The effective [vault] map_writes (value 1):"
            " before_answer, background or synchronous."
        )
        lines.append("# TYPE llm_redact_map_writes_mode gauge")
        if self.map_writes_mode is not None:
            lines.append(
                f'llm_redact_map_writes_mode{{mode="{_escape_label(self.map_writes_mode)}"}} 1'
            )

        for name, help_text, value in (
            ("llm_redact_vault_entries", "Placeholder mappings held.", vault_entries),
            ("llm_redact_vault_sessions", "Vault sessions in use.", vault_sessions),
            ("llm_redact_start_time_seconds", "Unix start time.", self._started),
            ("llm_redact_uptime_seconds", "Seconds since start.", time.time() - self._started),
        ):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        return "\n".join(lines) + "\n"


def _ner_lines(backends: list[tuple[str, "NerStats"]]) -> list[str]:
    """The NER coverage counters (detection/stats.py), by backend name. The
    HELP/TYPE lines are written even with NER off, so dashboards and alerts
    reading them always find the family."""
    from llm_redact.detection.stats import STRING_OUTCOMES

    lines = [
        "# HELP llm_redact_ner_strings_total Strings handed to an NER backend, by backend and"
        " outcome: scanned_whole (the model read it in one call), scanned_windowed (in"
        " overlapping windows), skipped_max_chars (longer than [detection.ner] max_chars: the"
        " model never read it, the regex rules still did). Restarts from zero when a reload"
        " rebuilds the detectors.",
        "# TYPE llm_redact_ner_strings_total counter",
    ]
    for backend, stats in backends:
        counts = stats.as_dict()
        for outcome in STRING_OUTCOMES:
            lines.append(
                f'llm_redact_ner_strings_total{{backend="{_escape_label(backend)}",'
                f'outcome="{outcome}"}} {counts[outcome]}'
            )
    for name, field, help_text in (
        (
            "llm_redact_ner_windows_total",
            "windows",
            "Windows the windowed strings were read in, by NER backend.",
        ),
        (
            "llm_redact_ner_windows_truncated_total",
            "windows_truncated",
            "NER windows (a string read whole counts as one) longer than the model's token"
            " limit because one word alone exceeds it: the model may read only part of"
            " them, by backend.",
        ),
        (
            "llm_redact_ner_labels_dropped_total",
            "labels_dropped",
            "NER model entities never emitted because their type cannot be a placeholder"
            " type, by backend.",
        ),
        (
            "llm_redact_ner_offsets_dropped_total",
            "offsets_dropped",
            "NER model entities of a requested type never redacted because the scanned"
            " string does not contain their span, by backend.",
        ),
    ):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        for backend, stats in backends:
            value = stats.as_dict()[field]
            lines.append(f'{name}{{backend="{_escape_label(backend)}"}} {value}')
    return lines


def plugin_metric_lines(samples: Iterable[object]) -> tuple[list[str], int]:
    """A plugin's gauge samples — ``(name, labels, value)`` triples — as
    exposition lines (one HELP/TYPE per name, in first-seen order), and how
    many were dropped as invalid: a name outside ``llm_redact_[a-z0-9_]``
    or one the core renders itself, labels that are not a mapping of names
    and values from the fixed charsets (more than ``MAX_PLUGIN_LABELS``, a
    reserved name, an id-like value — ``_id_like``), a value that is not a
    finite number (a bool is not one), a duplicate series, or anything past
    ``MAX_PLUGIN_SAMPLES``. Nothing of a dropped sample is ever echoed."""
    families: dict[str, list[str]] = {}
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    dropped = 0
    taken = 0
    for sample in samples:
        if taken >= MAX_PLUGIN_SAMPLES:
            dropped += 1
            continue
        line = _plugin_sample_line(sample, seen)
        if line is None:
            dropped += 1
            continue
        taken += 1
        families.setdefault(line[0], []).append(line[1])
    lines: list[str] = []
    for name, series in families.items():
        lines.append(f"# HELP {name} Reported by a plugin (llm-redact-pro).")
        lines.append(f"# TYPE {name} gauge")
        lines.extend(series)
    return lines, dropped


def _finite(value: float) -> bool:
    """A finite number a float can hold (an int past it is not one)."""
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _id_like(label_value: str) -> bool:
    """Whether a label value holds a run of at least ``_ID_LIKE_RUN``
    hexadecimal characters with a digit among them: the shape of a key, a
    hash, an id or a long number, never of a fixed state name."""
    return any(any(char.isdigit() for char in run) for run in _HEX_RUN.findall(label_value))


def _shadows_core(name: str) -> bool:
    """Whether ``name`` is a core family or one of its series (a
    histogram's ``_bucket``/``_sum``/``_count``)."""
    return any(name == family or name.startswith(family + "_") for family in CORE_METRIC_FAMILIES)


def _plugin_sample_line(
    sample: object, seen: set[tuple[str, tuple[tuple[str, str], ...]]]
) -> tuple[str, str] | None:
    """(family, exposition line) for one valid sample, else None."""
    if not isinstance(sample, tuple) or len(sample) != 3:
        return None
    name, labels, value = sample
    if (
        not isinstance(name, str)
        or not _PLUGIN_METRIC_NAME.fullmatch(name)
        or _shadows_core(name)
        or not isinstance(labels, Mapping)
        or len(labels) > MAX_PLUGIN_LABELS
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _finite(value)
    ):
        return None
    pairs: list[tuple[str, str]] = []
    for label, label_value in labels.items():
        if (
            not isinstance(label, str)
            or not isinstance(label_value, str)
            or not _PLUGIN_LABEL_NAME.fullmatch(label)
            or label in _RESERVED_LABELS
            or not _PLUGIN_LABEL_VALUE.fullmatch(label_value)
            or _id_like(label_value)
        ):
            return None
        pairs.append((label, label_value))
    key = (name, tuple(sorted(pairs)))
    if key in seen:
        return None
    seen.add(key)
    rendered = ",".join(f'{label}="{label_value}"' for label, label_value in key[1])
    return name, f"{name}{{{rendered}}} {value}" if rendered else f"{name} {value}"
