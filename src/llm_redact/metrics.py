"""Prometheus text exposition, hand-rolled (stdlib only, no client library).

Always-on and in-memory, independent of the opt-in audit log. Metadata only —
metric names, detector types, providers, status codes — consistent with the
proxy's never-log-values posture.
"""

import time
from collections import Counter
from collections.abc import Iterable

_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


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

    def observe_request(
        self, provider: str | None, status: int | None, seconds: float, streamed: bool = False
    ) -> None:
        prov = provider or "passthrough"
        self.requests[(prov, str(status or 0))] += 1
        key = (prov, "true" if streamed else "false")
        self._durations.setdefault(key, DurationHistogram()).observe(seconds)

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
