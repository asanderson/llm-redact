"""[audit] required — the zero-loss seam (4.2.0), Free-code side.

The concrete write-ahead audit log is a paid subsystem; these tests drive
the FREE code around it — parse validation, the startup capability check,
`begin_audit`'s no-upstream-contact-on-failure rule, and the token
threading through `record_request` — with fakes registered through the
plugin registry (the test_open_core_free_coverage.py pattern: Free code
whose only end-to-end path is a paid feature, driven without a license
key via registry seams, never a signed key).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Coroutine
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact.audit import AuditRecord, AuditWriteError
from llm_redact.config import AuditConfig, Config, ConfigError, ProviderConfig, parse_config
from llm_redact.proxy import create_app
from llm_redact.registry import Registry

UPSTREAM = "http://upstream.test"


class FakeAudit:
    """A registry-built audit log recording every call, faultable per phase."""

    def __init__(self, *, fail_begin: bool = False, none_token: bool = False) -> None:
        self.fail_begin = fail_begin
        self.none_token = none_token
        self.begun: list[AuditRecord] = []
        self.finalized: list[tuple[object, AuditRecord]] = []
        self.recorded: list[AuditRecord] = []

    def record(self, entry: AuditRecord) -> None:
        self.recorded.append(entry)

    def begin(self, entry: AuditRecord) -> object | None:
        if self.fail_begin:
            raise AuditWriteError("injected write fault")
        if self.none_token:
            return None  # a broken write-ahead impl
        self.begun.append(entry)
        return len(self.begun)  # a row-id-shaped token

    def finalize(self, token: object, entry: AuditRecord) -> None:
        self.finalized.append((token, entry))

    def recent(self, limit: int) -> list[dict[str, object]]:
        return []

    def count(self) -> int:
        return len(self.begun) + len(self.recorded)

    def close(self) -> None:
        pass


class LegacyAudit:
    """A pro AuditLog predating the write-ahead pair (no begin/finalize)."""

    def record(self, entry: AuditRecord) -> None:
        pass

    def recent(self, limit: int) -> list[dict[str, object]]:
        return []

    def count(self) -> int:
        return 0

    def close(self) -> None:
        pass


@pytest.fixture
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> Registry:
    """A registry resolving a Pro license and building a fake audit log."""
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("pro")
    reg.build_access_gate = lambda cfg, lic: None  # single-user deployment
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return reg


def _config(*, required: bool = True) -> Config:
    return Config(
        providers={"anthropic": ProviderConfig(upstream_base_url=UPSTREAM)},
        audit=AuditConfig(enabled=True, required=required),
    )


def _upstream_transport(calls: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            200,
            json={"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            headers={"content-type": "application/json"},
        )

    return httpx.MockTransport(handler)


async def _post_messages(app: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        return await client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "email jane.doe@corp.example please"}],
            },
        )


# ------------------------------------------------------------------- config


def test_required_defaults_off() -> None:
    assert parse_config({}, "test").audit.required is False


def test_required_without_enabled_is_a_parse_error() -> None:
    with pytest.raises(ConfigError, match=r"\[audit\] required = true needs"):
        parse_config({"audit": {"required": True}}, "test")


# The emitter round-trip for `required` lives with the rest of the emitter
# coverage: test_config_write.py's every-field-nondefault case.

# ------------------------------------------------------- startup capability


def test_required_with_legacy_pro_audit_fails_closed(fake_registry: Registry) -> None:
    fake_registry.build_audit = lambda cfg: LegacyAudit() if cfg.enabled else None
    with pytest.raises(ConfigError, match="begin/finalize"):
        create_app(_config(required=True))


def test_required_with_no_audit_log_fails_closed(fake_registry: Registry) -> None:
    # Unreachable through parse_config (required needs enabled, and an
    # enabled config either builds a log or raises), but a buggy plugin
    # returning None must refuse — never silently run fail-open under a
    # config that promises zero loss.
    fake_registry.build_audit = lambda cfg: None
    with pytest.raises(ConfigError, match="begin/finalize"):
        create_app(_config(required=True))


def test_required_with_write_ahead_audit_starts(fake_registry: Registry) -> None:
    fake_registry.build_audit = lambda cfg: FakeAudit() if cfg.enabled else None
    create_app(_config(required=True))


# ------------------------------------------------------------ request path


async def test_begin_before_upstream_and_finalize_after(fake_registry: Registry) -> None:
    fake = FakeAudit()
    fake_registry.build_audit = lambda cfg: fake if cfg.enabled else None
    upstream_calls: list[str] = []
    app = create_app(_config(required=True))
    app.state.proxy.client = httpx.AsyncClient(transport=_upstream_transport(upstream_calls))

    response = await _post_messages(app)

    assert response.status_code == 200
    # START committed, exactly once, before the upstream call could return.
    assert len(fake.begun) == 1 and len(upstream_calls) == 1
    start = fake.begun[0]
    assert start.status is None and start.detections == {"EMAIL": 1}
    # END finalized with the START token; record() never used for this row.
    assert len(fake.finalized) == 1 and fake.finalized[0][0] == 1
    end = fake.finalized[0][1]
    assert end.status == 200 and end.rehydrations == {}
    assert fake.recorded == []


async def test_begin_fault_refuses_503_without_upstream_contact(
    fake_registry: Registry,
) -> None:
    fake = FakeAudit(fail_begin=True)
    fake_registry.build_audit = lambda cfg: fake if cfg.enabled else None
    upstream_calls: list[str] = []
    app = create_app(_config(required=True))
    state = app.state.proxy
    state.client = httpx.AsyncClient(transport=_upstream_transport(upstream_calls))

    response = await _post_messages(app)

    assert response.status_code == 503
    assert upstream_calls == []  # the load-bearing guarantee
    body = response.json()
    assert "audit log unavailable" in str(body)
    # Metrics/recent still saw the refusal even though the audit row failed.
    assert state.recent[-1]["status"] == 503


async def test_begin_returning_no_token_is_a_write_fault(fake_registry: Registry) -> None:
    # None is the "required mode off" sentinel record_request dispatches on;
    # a write-ahead log minting it would get its END row silently routed
    # through record(), orphaning the START. begin_audit fails closed instead.
    fake = FakeAudit(none_token=True)
    fake_registry.build_audit = lambda cfg: fake if cfg.enabled else None
    upstream_calls: list[str] = []
    app = create_app(_config(required=True))
    app.state.proxy.client = httpx.AsyncClient(transport=_upstream_transport(upstream_calls))

    response = await _post_messages(app)

    assert response.status_code == 503
    assert upstream_calls == []


async def test_fail_open_mode_never_calls_begin(fake_registry: Registry) -> None:
    fake = FakeAudit()
    fake_registry.build_audit = lambda cfg: fake if cfg.enabled else None
    upstream_calls: list[str] = []
    app = create_app(_config(required=False))
    app.state.proxy.client = httpx.AsyncClient(transport=_upstream_transport(upstream_calls))

    response = await _post_messages(app)

    assert response.status_code == 200
    # The historical single-row path, byte-for-byte: record() only.
    assert fake.begun == [] and fake.finalized == []
    assert len(fake.recorded) == 1 and fake.recorded[0].status == 200


async def test_status_surfaces_required(fake_registry: Registry) -> None:
    fake_registry.build_audit = lambda cfg: FakeAudit() if cfg.enabled else None
    app = create_app(_config(required=True))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["audit"]["required"] is True


# ------------------------------------------------- synchronous write-ahead


class Unawaited:
    """The coroutines a SYNC audit member answered, kept so a test asserts
    each was CLOSED unrun (``inspect.getcoroutinestate``): one dropped
    unclosed only warns "never awaited" from its finalizer, where the
    warning is unraisable and fails no test."""

    def __init__(self) -> None:
        self.answers: list[Coroutine[Any, Any, Any]] = []

    def __call__(self, answer: Coroutine[Any, Any, Any]) -> Coroutine[Any, Any, Any]:
        self.answers.append(answer)
        return answer

    def closed(self) -> bool:
        states = [inspect.getcoroutinestate(answer) for answer in self.answers]
        return bool(states) and set(states) == {inspect.CORO_CLOSED}


class AsyncBeginAudit(FakeAudit):
    """A write-ahead log whose ``begin`` answers a coroutine (what an
    ``async def begin`` returns), which once counted as a valid token: no
    START row written."""

    def __init__(self) -> None:
        super().__init__()
        self.begin_ran = False
        self.unawaited = Unawaited()

    async def _write_start(self, entry: AuditRecord) -> object:
        self.begin_ran = True
        return 1

    def begin(self, entry: AuditRecord) -> object | None:
        return self.unawaited(self._write_start(entry))


class AsyncFinalizeAudit(FakeAudit):
    """A write-ahead log whose ``finalize`` answers a coroutine."""

    def __init__(self) -> None:
        super().__init__()
        self.finalize_ran = False
        self.unawaited = Unawaited()

    async def _write_end(self, token: object, entry: AuditRecord) -> None:
        self.finalize_ran = True

    def finalize(self, token: object, entry: AuditRecord) -> Any:
        return self.unawaited(self._write_end(token, entry))


async def test_an_async_begin_refuses_503_without_upstream_contact(
    fake_registry: Registry, caplog: pytest.LogCaptureFixture
) -> None:
    # begin is synchronous: an awaitable answer has committed nothing. It is
    # closed unrun (no "never awaited" warning) and the request refused
    # like a START row that cannot commit — before any upstream contact.
    fake = AsyncBeginAudit()
    fake_registry.build_audit = lambda cfg: fake if cfg.enabled else None
    upstream_calls: list[str] = []
    app = create_app(_config(required=True))
    state = app.state.proxy
    state.client = httpx.AsyncClient(transport=_upstream_transport(upstream_calls))
    caplog.set_level(logging.CRITICAL, logger="llm_redact")
    response = await _post_messages(app)
    assert response.status_code == 503 and upstream_calls == []
    assert response.json()["error"]["message"] == (
        "llm-redact: audit log unavailable and [audit] required is enabled"
    )
    assert fake.begin_ran is False and fake.unawaited.closed() and fake.finalized == []
    # The refusal is recorded (metrics, /recent; its own row best-effort).
    assert [row["status"] for row in state.recent] == [503]
    assert "audit write failed with [audit] required (AuditWriteError)" in caplog.text
    assert "jane.doe" not in caplog.text


async def test_an_async_finalize_is_an_end_row_fault(
    fake_registry: Registry, caplog: pytest.LogCaptureFixture
) -> None:
    # The answer is already committed: an awaitable END answer is closed
    # unrun and logged CRITICAL by type, like any other END-row fault.
    fake = AsyncFinalizeAudit()
    fake_registry.build_audit = lambda cfg: fake if cfg.enabled else None
    upstream_calls: list[str] = []
    app = create_app(_config(required=True))
    app.state.proxy.client = httpx.AsyncClient(transport=_upstream_transport(upstream_calls))
    caplog.set_level(logging.CRITICAL, logger="llm_redact")
    response = await _post_messages(app)
    assert response.status_code == 200 and len(upstream_calls) == 1
    assert len(fake.begun) == 1 and fake.recorded == []
    assert fake.finalize_ran is False and fake.unawaited.closed()
    assert "audit write failed AFTER response (POST /v1/messages): AuditWriteError" in (caplog.text)
    assert "jane.doe" not in caplog.text


class TaskAudit(FakeAudit):
    """``begin``/``finalize`` answer an asyncio Task (an awaitable that is no
    coroutine): its write would commit the row after the answer was refused
    — an orphan START row adopted as an interrupted request that never left,
    an END row written after CRITICAL said it failed."""

    def __init__(self, member: str) -> None:
        super().__init__()
        self.member = member
        self.tasks: list[asyncio.Task[Any]] = []
        self.late_rows: list[AuditRecord] = []

    def _task(self, entry: AuditRecord) -> asyncio.Task[Any]:
        async def write() -> int:
            self.late_rows.append(entry)
            return 1

        task = asyncio.ensure_future(write())
        self.tasks.append(task)
        return task

    def begin(self, entry: AuditRecord) -> object | None:
        return self._task(entry) if self.member == "begin" else super().begin(entry)

    def finalize(self, token: object, entry: AuditRecord) -> Any:
        if self.member == "finalize":
            return self._task(entry)
        return super().finalize(token, entry)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_a_task_from_begin_is_cancelled_and_refused(fake_registry: Registry) -> None:
    fake = TaskAudit("begin")
    fake_registry.build_audit = lambda cfg: fake if cfg.enabled else None
    upstream_calls: list[str] = []
    app = create_app(_config(required=True))
    app.state.proxy.client = httpx.AsyncClient(transport=_upstream_transport(upstream_calls))
    response = await _post_messages(app)
    await _settle()
    assert response.status_code == 503 and upstream_calls == []
    assert [task.cancelled() for task in fake.tasks] == [True]
    assert fake.late_rows == []  # no START row for a request that never left


async def test_a_task_from_finalize_is_cancelled_and_logged(
    fake_registry: Registry, caplog: pytest.LogCaptureFixture
) -> None:
    fake = TaskAudit("finalize")
    fake_registry.build_audit = lambda cfg: fake if cfg.enabled else None
    upstream_calls: list[str] = []
    app = create_app(_config(required=True))
    app.state.proxy.client = httpx.AsyncClient(transport=_upstream_transport(upstream_calls))
    caplog.set_level(logging.CRITICAL, logger="llm_redact")
    response = await _post_messages(app)
    await _settle()
    assert response.status_code == 200 and len(upstream_calls) == 1
    assert [task.cancelled() for task in fake.tasks] == [True]
    assert fake.late_rows == []  # never an END row after CRITICAL said it failed
    assert "audit write failed AFTER response (POST /v1/messages): AuditWriteError" in caplog.text
