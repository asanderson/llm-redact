"""Spend accounting and monthly budgets for routed upstreams (spec R-24 … R-28).

Every successful routed response becomes one ``SpendRow``: tokens by class, the
hop that produced it (1 = the primary, >= 2 = a re-issue) and a USD figure when
the price table knows the model. Rows go to a ``SpendStore`` (the sqlite vault
file's ``spend`` table over its OWN connection, or memory) and into the
``BudgetLedger``'s in-memory totals for the current period.

Load-bearing constraints:

- A store write NEVER raises into the request path: ``BudgetLedger.record``
  swallows the exception, logs its TYPE only, and still updates the in-memory
  totals so budget enforcement keeps working on a wedged disk (R-25).
- Budget periods are UTC calendar months anchored at ``reset_day`` (1..28, so
  every month has the day); the ledger rolls over LAZILY on the first call past
  the boundary and re-reads the new period's rows from the store.
- A store READ fault (at open or at rollover) starts the period from zero in
  memory rather than refusing requests — but it is RETRIED, at most once per
  ``_LOAD_RETRY_SECONDS``, until the store answers: without the retry a
  transient lock at startup would un-exhaust an over-budget upstream for the
  rest of the month. Rows whose store write failed are kept aside
  (``_unpersisted``) and folded back in on a successful reload, so nothing
  recorded while the store was wedged is lost when it recovers. The retry is
  rate-limited because a locked sqlite read blocks for its busy_timeout and
  ``exhausted`` sits on the request path.
- Passthrough upstreams have no budget (their usage is the subscription's to
  meter) and zero-cost upstreams ignore theirs: ``exhausted`` is False for both
  and for any name the ledger does not know.
- Rows carry counts and money, never content: no body text, no header values.
  Timestamps are the vault's ``%Y-%m-%dT%H:%M:%SZ`` form so lexical order is
  chronological and both stores filter periods with plain string bounds.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from llm_redact.pricing import PriceTable, Usage

logger = logging.getLogger("llm_redact")

_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
# The memory store is in-process only (documented); bound it like the recent
# ring so a long-lived proxy on the memory vault cannot grow without limit.
_MEMORY_ROWS = 100_000
# Minimum spacing between reload attempts after a store read fault: a locked
# sqlite read blocks for its busy_timeout (5 s), and the ledger is consulted on
# every routed request, so an unbounded retry would stall the hot path.
_LOAD_RETRY_SECONDS = 60.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  upstream TEXT NOT NULL,
  model TEXT NOT NULL,
  hop INTEGER NOT NULL,
  in_tokens INTEGER NOT NULL,
  out_tokens INTEGER NOT NULL,
  cache_read INTEGER NOT NULL,
  cache_write INTEGER NOT NULL,
  usd REAL
);
CREATE INDEX IF NOT EXISTS spend_ts ON spend (ts);
"""


@dataclass(frozen=True)
class Budget:
    """One upstream's monthly limits; ``zero_cost`` upstreams ignore both."""

    usd: float | None = None
    tokens: int | None = None
    zero_cost: bool = False


@dataclass(frozen=True)
class SpendRow:
    ts: str
    upstream: str
    model: str
    hop: int
    usage: Usage
    usd: float | None


@dataclass
class PeriodTotals:
    """One upstream's aggregate for one period; ``reissue_*`` = rows with hop >= 2."""

    in_tokens: int = 0
    out_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    usd: float = 0.0
    rows: int = 0
    unpriced_rows: int = 0
    reissue_rows: int = 0
    reissue_usd: float = 0.0
    reissue_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.in_tokens + self.out_tokens + self.cache_read + self.cache_write

    def add(self, *, hop: int, usage: Usage, usd: float | None) -> None:
        """The single aggregation path — both stores and the ledger use it, so a
        period total means the same thing wherever it was computed."""
        self.in_tokens += usage.input_tokens
        self.out_tokens += usage.output_tokens
        self.cache_read += usage.cache_read
        self.cache_write += usage.cache_write
        self.rows += 1
        if usd is None:
            self.unpriced_rows += 1
        else:
            self.usd += usd
        if hop >= 2:
            self.reissue_rows += 1
            self.reissue_tokens += usage.total_tokens
            self.reissue_usd += usd or 0.0

    def absorb(self, other: PeriodTotals) -> None:
        self.in_tokens += other.in_tokens
        self.out_tokens += other.out_tokens
        self.cache_read += other.cache_read
        self.cache_write += other.cache_write
        self.usd += other.usd
        self.rows += other.rows
        self.unpriced_rows += other.unpriced_rows
        self.reissue_rows += other.reissue_rows
        self.reissue_usd += other.reissue_usd
        self.reissue_tokens += other.reissue_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "total_tokens": self.total_tokens,
            "usd": round(self.usd, 6),
            "rows": self.rows,
            "unpriced_rows": self.unpriced_rows,
            "reissue_rows": self.reissue_rows,
            "reissue_usd": round(self.reissue_usd, 6),
            "reissue_tokens": self.reissue_tokens,
        }


def _as_utc(when: datetime) -> datetime:
    # A naive datetime is taken as UTC (the ledger's own clock is always aware).
    return when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def format_ts(when: datetime) -> str:
    """The row timestamp: second precision, UTC, lexically ordered."""
    return _as_utc(when).strftime(_TS_FORMAT)


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _previous_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def period_bounds(now: datetime, reset_day: int) -> tuple[datetime, datetime]:
    """``[start, end)`` of the budget period containing ``now``: the most recent
    ``reset_day`` at 00:00 UTC up to (excluding) the next month's. ``reset_day``
    is 1..28 so the day exists in every month (February included)."""
    if not 1 <= reset_day <= 28:
        raise ValueError("budget_reset_day must be between 1 and 28")
    now = _as_utc(now)
    year, month = now.year, now.month
    if now.day < reset_day:
        year, month = _previous_month(year, month)
    next_year, next_month = _next_month(year, month)
    return (
        datetime(year, month, reset_day, tzinfo=UTC),
        datetime(next_year, next_month, reset_day, tzinfo=UTC),
    )


def period_label(start: datetime) -> str:
    """``YYYY-MM`` of the period's start (the month the reset day falls in)."""
    return _as_utc(start).strftime("%Y-%m")


def period_for_label(label: str, reset_day: int) -> tuple[datetime, datetime]:
    """Inverse of ``period_label`` for ``spend --month YYYY-MM``."""
    try:
        first = datetime.strptime(label, "%Y-%m").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"month must be YYYY-MM, got {label!r}") from exc
    # The label names the month the period STARTS in; any day on/after the
    # reset day resolves to that period.
    return period_bounds(first.replace(day=reset_day), reset_day)


def over_budget(budget: Budget | None, totals: PeriodTotals) -> bool:
    """Decision 11: usd spent >= budget usd OR total tokens >= budget tokens,
    whichever limits are set; never for zero-cost or budget-less upstreams."""
    if budget is None or budget.zero_cost:
        return False
    if budget.usd is not None and totals.usd >= budget.usd:
        return True
    return budget.tokens is not None and totals.total_tokens >= budget.tokens


def _remaining_usd(limit: float | None, spent: float) -> float | None:
    return None if limit is None else round(max(limit - spent, 0.0), 6)


def _remaining_tokens(limit: int | None, spent: int) -> int | None:
    return None if limit is None else max(limit - spent, 0)


def _budget_view(budget: Budget | None, totals: PeriodTotals) -> dict[str, Any]:
    """The budget half of a snapshot / report entry (zero-cost: limits shown but
    remaining_* None, since they are ignored)."""
    budget = budget or Budget()
    ignored = budget.zero_cost
    return {
        "budget_usd": budget.usd,
        "budget_tokens": budget.tokens,
        "zero_cost": budget.zero_cost,
        "remaining_usd": None if ignored else _remaining_usd(budget.usd, totals.usd),
        "remaining_tokens": None
        if ignored
        else _remaining_tokens(budget.tokens, totals.total_tokens),
        "exhausted": over_budget(budget, totals),
    }


class SpendStore(Protocol):
    def record(self, row: SpendRow) -> None: ...

    def totals(self, start: datetime, end: datetime) -> dict[str, PeriodTotals]: ...

    def close(self) -> None: ...


class InMemorySpendStore:
    """In-process rows (the memory vault backend); dies with the process."""

    def __init__(self) -> None:
        self._rows: deque[SpendRow] = deque(maxlen=_MEMORY_ROWS)

    def record(self, row: SpendRow) -> None:
        self._rows.append(row)

    def totals(self, start: datetime, end: datetime) -> dict[str, PeriodTotals]:
        low, high = format_ts(start), format_ts(end)
        out: dict[str, PeriodTotals] = {}
        for row in self._rows:
            if low <= row.ts < high:
                out.setdefault(row.upstream, PeriodTotals()).add(
                    hop=row.hop, usage=row.usage, usd=row.usd
                )
        return out

    def close(self) -> None:
        return None


def _open_connection(path: Path) -> sqlite3.Connection:
    # The vault's discipline (vault.py _open_connection): private directory,
    # file pre-created 0600 BEFORE SQLite touches it. The spend table normally
    # lives in the vault file itself, which already exists with these modes.
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    # NORMAL rather than the vault's FULL: losing a spend row on power loss is
    # acceptable (the audit log makes the same call); losing a vault row is not.
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    return conn


class SqliteSpendStore:
    """The ``spend`` table in a sqlite file (normally the vault DB), own connection."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn = _open_connection(path)

    @property
    def path(self) -> Path:
        return self._path

    def record(self, row: SpendRow) -> None:
        usage = row.usage
        self._conn.execute(
            "INSERT INTO spend (ts, upstream, model, hop, in_tokens, out_tokens, cache_read,"
            " cache_write, usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.ts,
                row.upstream,
                row.model,
                row.hop,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read,
                usage.cache_write,
                row.usd,
            ),
        )

    def totals(self, start: datetime, end: datetime) -> dict[str, PeriodTotals]:
        cursor = self._conn.execute(
            "SELECT upstream, hop, in_tokens, out_tokens, cache_read, cache_write, usd"
            " FROM spend WHERE ts >= ? AND ts < ?",
            (format_ts(start), format_ts(end)),
        )
        out: dict[str, PeriodTotals] = {}
        for upstream, hop, in_tokens, out_tokens, cache_read, cache_write, usd in cursor:
            out.setdefault(upstream, PeriodTotals()).add(
                hop=hop,
                usage=Usage(in_tokens, out_tokens, cache_read, cache_write),
                usd=usd,
            )
        return out

    def close(self) -> None:
        self._conn.close()


class BudgetLedger:
    """Per-upstream period totals + budget decisions, backed by a SpendStore.

    The in-memory totals are the source of truth for ``exhausted`` (the hot
    path never touches the store); the store is the durable record the
    ``spend`` CLI reads and the source the ledger reloads from at rollover —
    and, after a read fault, on a rate-limited retry until it answers
    (``loaded`` says whether the totals currently reflect the store)."""

    def __init__(
        self,
        store: SpendStore,
        budgets: Mapping[str, Budget],
        *,
        reset_day: int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._budgets: dict[str, Budget] = dict(budgets)
        self._reset_day = reset_day
        self._clock: Callable[[], datetime] = clock if clock is not None else _utcnow
        now = _as_utc(self._clock())
        self._period = period_bounds(now, reset_day)
        self._totals: dict[str, PeriodTotals] = {}
        # Rows recorded this period whose store write failed: they exist only
        # here, so a later reload from the store must add them back.
        self._unpersisted: dict[str, PeriodTotals] = {}
        self._loaded = False
        self._retry_at = now
        self._reload(now)

    @property
    def period(self) -> tuple[datetime, datetime]:
        return self._period

    @property
    def loaded(self) -> bool:
        """Whether the in-memory totals reflect the store for this period
        (False after a read fault, until a retry succeeds)."""
        return self._loaded

    def _load(self) -> dict[str, PeriodTotals] | None:
        try:
            return dict(self._store.totals(*self._period))
        except Exception as exc:
            # Count the period from zero in memory rather than refuse every
            # request; the retry (not this warning alone) is what keeps a
            # transient fault from un-exhausting a budget for the whole period.
            logger.warning(
                "spend store read failed (%s); period %s is counted from zero in memory"
                " until the store answers (next attempt in %ds)",
                type(exc).__name__,
                period_label(self._period[0]),
                int(_LOAD_RETRY_SECONDS),
            )
            return None

    def _reload(self, now: datetime) -> None:
        """Replace the in-memory totals with the store's for the current period
        (plus the rows the store never accepted); on a read fault keep what is
        in memory and schedule the next attempt."""
        loaded = self._load()
        if loaded is None:
            self._loaded = False
            self._retry_at = now + timedelta(seconds=_LOAD_RETRY_SECONDS)
            return
        for name, extra in self._unpersisted.items():
            loaded.setdefault(name, PeriodTotals()).absorb(extra)
        self._totals = loaded
        self._loaded = True

    def _roll(self) -> None:
        now = _as_utc(self._clock())
        start, end = self._period
        if not start <= now < end:
            self._period = period_bounds(now, self._reset_day)
            # A new period starts from zero whatever the store says next; the
            # old period's memory-only rows are its own and are dropped here.
            self._totals = {}
            self._unpersisted = {}
            self._reload(now)
        elif not self._loaded and now >= self._retry_at:
            self._reload(now)
            if self._loaded:
                logger.info(
                    "spend store read recovered; period %s totals reloaded",
                    period_label(self._period[0]),
                )

    def record(
        self, *, upstream: str, model: str, hop: int, usage: Usage, price_table: PriceTable
    ) -> SpendRow:
        """Price and persist one response; the store failing is logged by
        exception TYPE and never surfaces to the request path."""
        self._roll()
        usd = price_table.cost_usd(model, usage)
        row = SpendRow(
            ts=format_ts(self._clock()),
            upstream=upstream,
            model=model,
            hop=hop,
            usage=usage,
            usd=usd,
        )
        try:
            self._store.record(row)
        except Exception as exc:
            logger.warning(
                "spend store write failed for upstream %s (%s); the row is counted in memory only",
                upstream,
                type(exc).__name__,
            )
            self._unpersisted.setdefault(upstream, PeriodTotals()).add(
                hop=hop, usage=usage, usd=usd
            )
        self._totals.setdefault(upstream, PeriodTotals()).add(hop=hop, usage=usage, usd=usd)
        return row

    def exhausted(self, upstream: str) -> bool:
        self._roll()
        return over_budget(self._budgets.get(upstream), self._totals.get(upstream, PeriodTotals()))

    def totals(self, upstream: str) -> PeriodTotals:
        self._roll()
        return self._totals.get(upstream, PeriodTotals())

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Every upstream with a budget entry or any spend this period."""
        self._roll()
        label = period_label(self._period[0])
        names = sorted(set(self._budgets) | set(self._totals))
        out: dict[str, dict[str, Any]] = {}
        for name in names:
            totals = self._totals.get(name, PeriodTotals())
            out[name] = {
                "period": label,
                **totals.as_dict(),
                **_budget_view(self._budgets.get(name), totals),
            }
        return out

    def rebudget(self, budgets: Mapping[str, Budget]) -> None:
        """SIGHUP: new limits, same totals."""
        self._budgets = dict(budgets)


def report(
    store: SpendStore,
    *,
    month: str | None,
    budgets: Mapping[str, Budget],
    reset_day: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The JSON ``llm-redact spend`` prints: per-upstream tokens/USD, the hop-origin
    split (primary vs re-issue), remaining budget, and grand totals. ``month``
    selects a past period by its label; ``None`` is the current one."""
    current = _as_utc(now) if now is not None else _utcnow()
    current_start, _ = period_bounds(current, reset_day)
    if month is None:
        start, end = current_start, period_bounds(current, reset_day)[1]
    else:
        start, end = period_for_label(month, reset_day)
    totals = store.totals(start, end)
    grand = PeriodTotals()
    upstreams: dict[str, dict[str, Any]] = {}
    for name in sorted(set(totals) | set(budgets)):
        entry = totals.get(name, PeriodTotals())
        grand.absorb(entry)
        primary_usd = entry.usd - entry.reissue_usd
        upstreams[name] = {
            **entry.as_dict(),
            "hops": {
                "primary": {
                    "rows": entry.rows - entry.reissue_rows,
                    "tokens": entry.total_tokens - entry.reissue_tokens,
                    "usd": round(primary_usd, 6),
                },
                "reissue": {
                    "rows": entry.reissue_rows,
                    "tokens": entry.reissue_tokens,
                    "usd": round(entry.reissue_usd, 6),
                },
                "reissue_share": _share(entry),
            },
            **_budget_view(budgets.get(name), entry),
        }
    return {
        "period": period_label(start),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "reset_day": reset_day,
        "current": start == current_start,
        "upstreams": upstreams,
        "totals": grand.as_dict(),
    }


def _share(entry: PeriodTotals) -> float:
    """Fraction of this upstream's spend that came from re-issues: by USD when
    priced, by tokens when nothing was priced, 0.0 when there was nothing."""
    if entry.usd > 0:
        return round(entry.reissue_usd / entry.usd, 4)
    if entry.total_tokens > 0:
        return round(entry.reissue_tokens / entry.total_tokens, 4)
    return 0.0
