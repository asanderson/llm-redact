"""Spend stores, budget periods, the BudgetLedger and the spend report (R-25/R-27/R-28)."""

from __future__ import annotations

import logging
import sqlite3
import stat
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from llm_redact.pricing import PriceTable, Usage
from llm_redact.routing import ModelPrice
from llm_redact.spend import (
    Budget,
    BudgetLedger,
    InMemorySpendStore,
    PeriodTotals,
    SpendRow,
    SpendStore,
    SqliteSpendStore,
    format_ts,
    over_budget,
    period_bounds,
    period_for_label,
    period_label,
    report,
)
from llm_redact.vault import SqliteVaultManager

PRICES = PriceTable(
    {
        # $2 / $10 / $0.2 / $2.5 per 1M: Usage(100_000, 50_000) costs exactly $0.70.
        "claude-sonnet-5": ModelPrice(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5),
    }
)
SEPT = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)


def _dt(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _row(
    when: datetime, upstream: str, *, hop: int = 1, usd: float | None = 0.5, model: str = "m"
) -> SpendRow:
    return SpendRow(
        ts=format_ts(when),
        upstream=upstream,
        model=model,
        hop=hop,
        usage=Usage(input_tokens=100, output_tokens=10, cache_read=5, cache_write=1),
        usd=usd,
    )


# --- periods ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "reset_day", "start", "end"),
    [
        # reset day 28 in February: the day exists, the period is Jan 28 -> Feb 28 ...
        (_dt(2026, 2, 27, 23), 28, _dt(2026, 1, 28), _dt(2026, 2, 28)),
        # ... and rolls at 00:00 on the 28th.
        (_dt(2026, 2, 28), 28, _dt(2026, 2, 28), _dt(2026, 3, 28)),
        # year rollover, both directions
        (_dt(2026, 12, 15), 1, _dt(2026, 12, 1), _dt(2027, 1, 1)),
        (_dt(2027, 1, 1), 1, _dt(2027, 1, 1), _dt(2027, 2, 1)),
        (_dt(2026, 1, 5), 10, _dt(2025, 12, 10), _dt(2026, 1, 10)),
        # "now" before / on the reset day
        (_dt(2026, 9, 14, 23), 15, _dt(2026, 8, 15), _dt(2026, 9, 15)),
        (_dt(2026, 9, 15), 15, _dt(2026, 9, 15), _dt(2026, 10, 15)),
        # plain calendar month
        (SEPT, 1, _dt(2026, 9, 1), _dt(2026, 10, 1)),
    ],
)
def test_period_bounds(now: datetime, reset_day: int, start: datetime, end: datetime) -> None:
    assert period_bounds(now, reset_day) == (start, end)


def test_period_bounds_normalizes_timezones() -> None:
    naive = datetime(2026, 2, 27, 23, 0)  # taken as UTC
    assert period_bounds(naive, 28) == (_dt(2026, 1, 28), _dt(2026, 2, 28))
    # 2026-02-28 01:00 at UTC+3 is still 2026-02-27 22:00 UTC → the earlier period.
    east = datetime(2026, 2, 28, 1, 0, tzinfo=timezone(timedelta(hours=3)))
    assert period_bounds(east, 28) == (_dt(2026, 1, 28), _dt(2026, 2, 28))


@pytest.mark.parametrize("reset_day", [0, 29, 31, -1])
def test_period_bounds_rejects_days_outside_1_28(reset_day: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 28"):
        period_bounds(SEPT, reset_day)


def test_period_label_and_inverse() -> None:
    start, end = period_bounds(_dt(2026, 2, 27), 28)
    assert period_label(start) == "2026-01"
    assert period_for_label("2026-01", 28) == (start, end)
    assert period_for_label("2026-12", 1) == (_dt(2026, 12, 1), _dt(2027, 1, 1))
    with pytest.raises(ValueError, match="YYYY-MM"):
        period_for_label("September", 1)
    with pytest.raises(ValueError, match="YYYY-MM"):
        period_for_label("2026-13", 1)


def test_format_ts_is_second_precision_utc() -> None:
    assert format_ts(datetime(2026, 9, 21, 10, 0, 5, 999, tzinfo=UTC)) == "2026-09-21T10:00:05Z"
    assert format_ts(datetime(2026, 9, 21, 13, 0, tzinfo=timezone(timedelta(hours=3)))) == (
        "2026-09-21T10:00:00Z"
    )


# --- stores -------------------------------------------------------------------


StoreFactory = Callable[[], SpendStore]


@pytest.fixture(params=["memory", "sqlite"])
def store_factory(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StoreFactory]:
    """Each call opens a store over the SAME backing (a fresh sqlite connection
    for the file; the same object for memory), so reopen tests work for both."""
    opened: list[SpendStore] = []
    memory = InMemorySpendStore()

    def factory() -> SpendStore:
        store: SpendStore = (
            memory if request.param == "memory" else SqliteSpendStore(tmp_path / "vault.db")
        )
        opened.append(store)
        return store

    yield factory
    for store in opened:
        store.close()


def _seed(store: SpendStore) -> None:
    aug, sep = _dt(2026, 8, 20), _dt(2026, 9, 5)
    store.record(_row(aug, "anthropic_key", usd=0.25))
    store.record(_row(sep, "anthropic_key", hop=1, usd=0.5))
    store.record(_row(sep, "anthropic_key", hop=2, usd=0.75))  # a re-issue
    store.record(_row(sep, "anthropic_key", hop=3, usd=None))  # re-issue, unpriced model
    store.record(_row(sep, "ollama", hop=1, usd=None))
    store.record(_row(_dt(2026, 10, 1), "anthropic_key", usd=9.0))  # next period


def test_store_totals_partition_by_period_and_upstream_with_hop_breakdown(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    _seed(store)
    sept = store.totals(_dt(2026, 9, 1), _dt(2026, 10, 1))
    assert set(sept) == {"anthropic_key", "ollama"}
    key = sept["anthropic_key"]
    assert (key.rows, key.unpriced_rows) == (3, 1)
    assert key.usd == pytest.approx(1.25)
    assert (key.in_tokens, key.out_tokens, key.cache_read, key.cache_write) == (300, 30, 15, 3)
    assert key.total_tokens == 348
    # hop >= 2 rows are the re-issue share: two rows, one priced at 0.75.
    assert (key.reissue_rows, key.reissue_tokens) == (2, 232)
    assert key.reissue_usd == pytest.approx(0.75)
    ollama = sept["ollama"]
    assert (ollama.rows, ollama.unpriced_rows, ollama.usd, ollama.reissue_rows) == (1, 1, 0.0, 0)

    aug = store.totals(_dt(2026, 8, 1), _dt(2026, 9, 1))
    assert set(aug) == {"anthropic_key"}
    assert aug["anthropic_key"].usd == pytest.approx(0.25)
    assert aug["anthropic_key"].rows == 1
    # The end bound is exclusive: the Oct 1 row is not September's.
    assert store.totals(_dt(2026, 11, 1), _dt(2026, 12, 1)) == {}


def test_store_rows_survive_reopen(store_factory: StoreFactory) -> None:
    first = store_factory()
    first.record(_row(SEPT, "anthropic_key", usd=0.5))
    second = store_factory()
    second.record(_row(SEPT, "anthropic_key", usd=0.5))
    totals = second.totals(_dt(2026, 9, 1), _dt(2026, 10, 1))
    assert totals["anthropic_key"].rows == 2


def test_sqlite_store_creates_private_file_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "data" / "vault.db"
    store = SqliteSpendStore(path)
    assert store.path == path
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    store.record(_row(SEPT, "anthropic_key"))
    store.close()
    # Reopening runs CREATE TABLE IF NOT EXISTS again: no error, rows kept, one table.
    again = SqliteSpendStore(path)
    assert again.totals(_dt(2026, 9, 1), _dt(2026, 10, 1))["anthropic_key"].rows == 1
    again.close()
    with sqlite3.connect(path) as conn:
        names = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "spend" in names
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()


def test_sqlite_store_coexists_with_the_vault_in_one_file(tmp_path: Path) -> None:
    # The spend table lives in the vault DB (decision 11) over its own
    # connection: opening either side first must leave the other intact.
    path = tmp_path / "vault.db"
    store = SqliteSpendStore(path)
    store.record(_row(SEPT, "anthropic_key"))
    manager = SqliteVaultManager(path)
    assert manager.sessions_summary() == []
    with sqlite3.connect(path) as conn:
        names = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    conn.close()
    assert {"spend", "mappings"} <= names
    assert store.totals(_dt(2026, 9, 1), _dt(2026, 10, 1))["anthropic_key"].rows == 1
    store.close()


def test_memory_store_is_bounded() -> None:
    store = InMemorySpendStore()
    assert store._rows.maxlen == 100_000


# --- BudgetLedger -------------------------------------------------------------


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _ledger(
    budgets: dict[str, Budget], store: SpendStore | None = None, now: datetime = SEPT
) -> tuple[BudgetLedger, _Clock]:
    clock = _Clock(now)
    return BudgetLedger(store or InMemorySpendStore(), budgets, reset_day=1, clock=clock), clock


def _spend(ledger: BudgetLedger, upstream: str = "anthropic_key", hop: int = 1) -> SpendRow:
    # $0.70 per call at the fixture price.
    return ledger.record(
        upstream=upstream,
        model="claude-sonnet-5",
        hop=hop,
        usage=Usage(100_000, 50_000),
        price_table=PRICES,
    )


def test_ledger_record_prices_and_persists() -> None:
    store = InMemorySpendStore()
    ledger, _ = _ledger({"anthropic_key": Budget(usd=10.0)}, store)
    row = _spend(ledger)
    assert row == SpendRow(
        ts="2026-09-21T10:00:00Z",
        upstream="anthropic_key",
        model="claude-sonnet-5",
        hop=1,
        usage=Usage(100_000, 50_000),
        usd=pytest.approx(0.7),  # type: ignore[arg-type]
    )
    assert store.totals(*ledger.period)["anthropic_key"].usd == pytest.approx(0.7)
    assert ledger.totals("anthropic_key").usd == pytest.approx(0.7)
    assert ledger.period == (_dt(2026, 9, 1), _dt(2026, 10, 1))


def test_ledger_exhausted_usd() -> None:
    ledger, _ = _ledger({"anthropic_key": Budget(usd=1.0)})
    _spend(ledger)
    assert not ledger.exhausted("anthropic_key")  # 0.70 < 1.00
    _spend(ledger)
    assert ledger.exhausted("anthropic_key")  # 1.40 >= 1.00


def test_ledger_exhausted_tokens() -> None:
    ledger, _ = _ledger({"anthropic_key": Budget(tokens=200_000)})
    _spend(ledger)
    assert not ledger.exhausted("anthropic_key")  # 150k < 200k
    _spend(ledger)
    assert ledger.exhausted("anthropic_key")  # 300k >= 200k


def test_ledger_exhausted_both_limits_either_trips() -> None:
    ledger, _ = _ledger({"anthropic_key": Budget(usd=100.0, tokens=150_000)})
    _spend(ledger)
    assert ledger.exhausted("anthropic_key")  # tokens trip long before usd
    other, _ = _ledger({"anthropic_key": Budget(usd=0.5, tokens=10_000_000)})
    _spend(other)
    assert other.exhausted("anthropic_key")  # usd trips long before tokens


def test_ledger_zero_cost_and_unknown_names_are_never_exhausted() -> None:
    ledger, _ = _ledger({"ollama": Budget(usd=0.0, tokens=1, zero_cost=True), "free": Budget()})
    _spend(ledger, "ollama")
    _spend(ledger, "free")
    _spend(ledger, "anthropic_oauth")  # passthrough: no Budget entry at all
    assert not ledger.exhausted("ollama")
    assert not ledger.exhausted("free")
    assert not ledger.exhausted("anthropic_oauth")
    assert not ledger.exhausted("never-seen")
    assert not over_budget(None, PeriodTotals(usd=1e9))


def test_ledger_lazy_rollover_at_the_period_boundary() -> None:
    store = InMemorySpendStore()
    ledger, clock = _ledger({"anthropic_key": Budget(usd=1.0)}, store)
    _spend(ledger)
    _spend(ledger)
    assert ledger.exhausted("anthropic_key")
    assert ledger.snapshot()["anthropic_key"]["period"] == "2026-09"
    # Rows landing in October before the ledger notices (another process, or
    # the store already held them) are picked up by the reload at rollover.
    store.record(_row(_dt(2026, 10, 2), "anthropic_key", usd=0.1))
    clock.now = _dt(2026, 9, 30, 23)
    assert ledger.exhausted("anthropic_key")  # still September
    clock.now = _dt(2026, 10, 3)
    assert not ledger.exhausted("anthropic_key")  # new period, budget reset
    assert ledger.period == (_dt(2026, 10, 1), _dt(2026, 11, 1))
    assert ledger.totals("anthropic_key").usd == pytest.approx(0.1)
    assert ledger.snapshot()["anthropic_key"]["period"] == "2026-10"
    _spend(ledger)
    assert ledger.totals("anthropic_key").rows == 2
    # September's rows are still in the store, untouched.
    assert store.totals(_dt(2026, 9, 1), _dt(2026, 10, 1))["anthropic_key"].rows == 2


def test_ledger_rebudget_keeps_totals() -> None:
    ledger, _ = _ledger({"anthropic_key": Budget(usd=1.0)})
    _spend(ledger)
    _spend(ledger)
    assert ledger.exhausted("anthropic_key")
    ledger.rebudget({"anthropic_key": Budget(usd=5.0)})
    assert not ledger.exhausted("anthropic_key")
    assert ledger.totals("anthropic_key").usd == pytest.approx(1.4)  # spend kept
    ledger.rebudget({})  # budget removed on reload: nothing to exhaust
    assert not ledger.exhausted("anthropic_key")
    assert "anthropic_key" in ledger.snapshot()  # still reported (it has spend)


def test_ledger_snapshot_shape() -> None:
    ledger, _ = _ledger(
        {
            "anthropic_key": Budget(usd=2.0, tokens=1_000_000),
            "ollama": Budget(zero_cost=True),
            "idle": Budget(usd=3.0),
        }
    )
    _spend(ledger)
    _spend(ledger, hop=2)
    _spend(ledger, "ollama")
    _spend(ledger, "unbudgeted")
    ledger.record(
        upstream="anthropic_key", model="mystery", hop=2, usage=Usage(10, 10), price_table=PRICES
    )
    snap = ledger.snapshot()
    assert set(snap) == {"anthropic_key", "ollama", "idle", "unbudgeted"}
    expected_keys = {
        "period",
        "in_tokens",
        "out_tokens",
        "cache_read",
        "cache_write",
        "total_tokens",
        "usd",
        "rows",
        "unpriced_rows",
        "reissue_rows",
        "reissue_usd",
        "reissue_tokens",
        "budget_usd",
        "budget_tokens",
        "zero_cost",
        "remaining_usd",
        "remaining_tokens",
        "exhausted",
    }
    for entry in snap.values():
        assert set(entry) == expected_keys
    key = snap["anthropic_key"]
    assert key["period"] == "2026-09"
    assert (key["rows"], key["unpriced_rows"], key["reissue_rows"]) == (3, 1, 2)
    assert key["usd"] == pytest.approx(1.4)
    assert key["reissue_usd"] == pytest.approx(0.7)
    assert key["reissue_tokens"] == 150_020
    assert key["total_tokens"] == 300_020
    assert (key["budget_usd"], key["budget_tokens"]) == (2.0, 1_000_000)
    assert key["remaining_usd"] == pytest.approx(0.6)
    assert key["remaining_tokens"] == 699_980
    assert key["exhausted"] is False
    assert key["zero_cost"] is False
    ollama = snap["ollama"]
    assert ollama["zero_cost"] is True
    assert (ollama["remaining_usd"], ollama["remaining_tokens"], ollama["exhausted"]) == (
        None,
        None,
        False,
    )
    idle = snap["idle"]
    assert (idle["rows"], idle["usd"], idle["remaining_usd"], idle["remaining_tokens"]) == (
        0,
        0.0,
        3.0,
        None,
    )
    unbudgeted = snap["unbudgeted"]
    assert (unbudgeted["budget_usd"], unbudgeted["remaining_usd"], unbudgeted["exhausted"]) == (
        None,
        None,
        False,
    )
    assert PRICES.unknown_models >= {"mystery"}


def test_ledger_remaining_never_negative() -> None:
    ledger, _ = _ledger({"anthropic_key": Budget(usd=0.5, tokens=1_000)})
    _spend(ledger)
    entry = ledger.snapshot()["anthropic_key"]
    assert (entry["remaining_usd"], entry["remaining_tokens"], entry["exhausted"]) == (0.0, 0, True)


class _FaultyStore:
    """A store whose writes (and optionally reads) raise, like a wedged disk."""

    def __init__(self, *, read_fails: bool = False) -> None:
        self.read_fails = read_fails
        self.attempts = 0

    def record(self, row: SpendRow) -> None:
        self.attempts += 1
        raise OSError("disk full at /var/lib/secret-path")

    def totals(self, start: datetime, end: datetime) -> dict[str, PeriodTotals]:
        if self.read_fails:
            raise sqlite3.OperationalError("database is locked")
        return {}

    def close(self) -> None:
        return None


def test_ledger_record_swallows_store_faults_and_keeps_counting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _FaultyStore()
    ledger, _ = _ledger({"anthropic_key": Budget(usd=1.0)}, store)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        row = _spend(ledger)
        _spend(ledger)
    assert row.usd == pytest.approx(0.7)
    assert store.attempts == 2
    # In-memory totals still advance, so enforcement works without the store.
    assert ledger.totals("anthropic_key").rows == 2
    assert ledger.exhausted("anthropic_key")
    # Logged by exception TYPE only — the message could carry paths or values.
    assert "OSError" in caplog.text
    assert "secret-path" not in caplog.text


def test_ledger_read_fault_at_open_starts_from_zero(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        ledger, _ = _ledger({"anthropic_key": Budget(usd=1.0)}, _FaultyStore(read_fails=True))
    assert "OperationalError" in caplog.text
    assert "database is locked" not in caplog.text
    assert not ledger.exhausted("anthropic_key")
    assert ledger.totals("anthropic_key") == PeriodTotals()


def test_ledger_reloads_persisted_totals_from_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "vault.db"
    store = SqliteSpendStore(path)
    ledger, _ = _ledger({"anthropic_key": Budget(usd=1.0)}, store)
    _spend(ledger)
    _spend(ledger)
    assert ledger.exhausted("anthropic_key")
    store.close()
    # A restart: a fresh ledger over a fresh connection sees the period's spend.
    reopened = SqliteSpendStore(path)
    restarted, _ = _ledger({"anthropic_key": Budget(usd=1.0)}, reopened)
    assert restarted.exhausted("anthropic_key")
    assert restarted.totals("anthropic_key").usd == pytest.approx(1.4)
    reopened.close()


def test_ledger_default_clock_is_utc_now() -> None:
    ledger = BudgetLedger(InMemorySpendStore(), {}, reset_day=1)
    start, end = ledger.period
    assert start <= datetime.now(UTC) < end
    assert start.tzinfo is UTC


# --- report -------------------------------------------------------------------


def _reported(month: str | None, **budgets: Budget) -> dict[str, Any]:
    store = InMemorySpendStore()
    _seed(store)
    return report(store, month=month, budgets=budgets, reset_day=1, now=SEPT)


def test_report_current_period_shape_with_hop_origin_breakdown() -> None:
    out = _reported(
        None, anthropic_key=Budget(usd=2.0), ollama=Budget(zero_cost=True), idle=Budget(usd=1.0)
    )
    assert out["period"] == "2026-09"
    assert out["current"] is True
    assert (out["start"], out["end"]) == ("2026-09-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00")
    assert out["reset_day"] == 1
    assert set(out["upstreams"]) == {"anthropic_key", "ollama", "idle"}
    key = out["upstreams"]["anthropic_key"]
    assert key["rows"] == 3
    assert key["usd"] == pytest.approx(1.25)
    assert key["hops"]["primary"] == {"rows": 1, "tokens": 116, "usd": pytest.approx(0.5)}
    assert key["hops"]["reissue"] == {"rows": 2, "tokens": 232, "usd": pytest.approx(0.75)}
    assert key["hops"]["reissue_share"] == pytest.approx(0.6)
    assert key["remaining_usd"] == pytest.approx(0.75)
    assert key["exhausted"] is False
    ollama = out["upstreams"]["ollama"]
    assert (ollama["rows"], ollama["unpriced_rows"], ollama["usd"]) == (1, 1, 0.0)
    assert ollama["hops"]["reissue_share"] == 0.0
    assert ollama["zero_cost"] is True
    idle = out["upstreams"]["idle"]
    assert (idle["rows"], idle["remaining_usd"]) == (0, 1.0)
    totals = out["totals"]
    assert (totals["rows"], totals["unpriced_rows"], totals["reissue_rows"]) == (4, 2, 2)
    assert totals["usd"] == pytest.approx(1.25)
    assert totals["total_tokens"] == 464


def test_report_month_selects_a_past_period() -> None:
    out = _reported("2026-08", anthropic_key=Budget(usd=2.0))
    assert out["period"] == "2026-08"
    assert out["current"] is False
    assert set(out["upstreams"]) == {"anthropic_key"}
    key = out["upstreams"]["anthropic_key"]
    assert (key["rows"], key["reissue_rows"]) == (1, 0)
    assert key["usd"] == pytest.approx(0.25)
    assert key["remaining_usd"] == pytest.approx(1.75)
    empty = _reported("2026-05")
    assert empty["upstreams"] == {} and empty["totals"]["rows"] == 0
    with pytest.raises(ValueError, match="YYYY-MM"):
        _reported("08/2026")


def test_report_reissue_share_by_tokens_when_nothing_is_priced() -> None:
    store = InMemorySpendStore()
    store.record(_row(SEPT, "ollama", hop=1, usd=None))
    store.record(_row(SEPT, "ollama", hop=2, usd=None))
    store.record(_row(SEPT, "ollama", hop=2, usd=None))
    out = report(store, month=None, budgets={}, reset_day=1, now=SEPT)
    hops = out["upstreams"]["ollama"]["hops"]
    assert hops["reissue_share"] == pytest.approx(2 / 3, abs=5e-5)  # rounded to 4 places
    assert hops["primary"]["usd"] == 0.0 and hops["reissue"]["usd"] == 0.0


def test_report_default_now_is_the_current_period() -> None:
    out = report(InMemorySpendStore(), month=None, budgets={}, reset_day=1)
    assert out["period"] == datetime.now(UTC).strftime("%Y-%m")
    assert out["current"] is True
