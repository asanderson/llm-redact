"""A realtime connection refused before the upstream dial leaves a row.

The HTTP twins of these refusals are recorded (metrics, /recent, /events,
OTel, audit sinks); the relay's used to close 1011 without a trace:

- the access gate's refusal (HTTP: a recorded 403),
- a disabled provider, or one whose upstream is not configured (HTTP: a
  recorded 502),
- an ``[audit] required`` START row that cannot be committed (HTTP: the
  recorded 503 twin of the upstream-fault 502).

Real sockets end to end (the test_realtime_relay harness).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import websockets

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact.config import AuditConfig, Config, ProviderConfig
from llm_redact.plugin_api import Admission
from llm_redact.registry import Registry
from test_access_seam import FakeGate
from test_audit_required import AsyncBeginAudit, AsyncFinalizeAudit, FakeAudit
from test_realtime_identity import _closed, _recent
from test_realtime_relay import FakeUpstream, _config, _proxy

pytestmark = pytest.mark.asyncio


class RefusingGate(FakeGate):
    def admit(self, conn: Any, surface: str) -> Admission:
        return Admission(refusal="a test key is required")


async def _refused_row(config: Config, path: str = "/v1/realtime") -> tuple[Any, dict[str, Any]]:
    with _proxy(config) as proxy_host:
        async with websockets.connect(f"ws://{proxy_host}{path}") as client:
            closed = await _closed(client)
        row = await _recent(proxy_host, lambda r: r["method"] == "WS")
    return closed, row


async def test_access_gate_refusal_is_recorded_403(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("team")
    reg.build_access_gate = lambda config, license: RefusingGate()
    monkeypatch.setattr(registry_mod, "_registry", reg)
    async with FakeUpstream() as fake:
        closed, row = await _refused_row(_config(fake.port))
    assert fake.paths == []
    assert (closed.code, closed.reason) == (1011, "a test key is required")
    assert (row["status"], row["provider"], row["path"]) == (403, "openai", "/v1/realtime")


@pytest.mark.parametrize(
    ("provider_config", "reason"),
    [
        (ProviderConfig("http://127.0.0.1:1", enabled=False), "disabled"),
        (ProviderConfig(""), "upstream not configured"),
    ],
    ids=["disabled", "unconfigured"],
)
async def test_unusable_provider_refusal_is_recorded_502(
    monkeypatch: pytest.MonkeyPatch, provider_config: ProviderConfig, reason: str
) -> None:
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    config = Config(providers={**Config().providers, "openai": provider_config})
    closed, row = await _refused_row(config)
    assert closed.code == 1011 and reason in closed.reason
    assert (row["status"], row["provider"]) == (502, "openai")


async def test_audit_start_fault_is_recorded_503(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("pro")
    reg.build_access_gate = lambda cfg, lic: None
    audit = FakeAudit(fail_begin=True)
    reg.build_audit = lambda cfg: audit if cfg.enabled else None
    monkeypatch.setattr(registry_mod, "_registry", reg)
    async with FakeUpstream() as fake:
        config = Config(
            providers={
                **Config().providers,
                "openai": ProviderConfig(f"http://127.0.0.1:{fake.port}"),
            },
            audit=AuditConfig(enabled=True, required=True),
        )
        closed, row = await _refused_row(config)
    assert fake.paths == []  # no upstream contact without a committed START
    assert closed.code == 1011 and "[audit] required" in closed.reason
    assert (row["status"], row["provider"]) == (503, "openai")


def _audited(monkeypatch: pytest.MonkeyPatch, audit: FakeAudit, port: int) -> Config:
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("pro")
    reg.build_access_gate = lambda cfg, lic: None
    reg.build_audit = lambda cfg: audit if cfg.enabled else None
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return Config(
        providers={**Config().providers, "openai": ProviderConfig(f"http://127.0.0.1:{port}")},
        audit=AuditConfig(enabled=True, required=True),
    )


async def test_an_async_begin_is_recorded_503_and_never_dialled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # begin is synchronous: a coroutine answer committed nothing. Closed
    # unrun and refused like a START row that cannot commit — never dialled.
    audit = AsyncBeginAudit()
    caplog.set_level(logging.CRITICAL, logger="llm_redact")
    async with FakeUpstream() as fake:
        closed, row = await _refused_row(_audited(monkeypatch, audit, fake.port))
    assert fake.paths == []
    assert closed.code == 1011 and "[audit] required" in closed.reason
    assert (row["status"], row["provider"]) == (503, "openai")
    assert audit.begin_ran is False and audit.unawaited.closed() and audit.finalized == []
    assert "audit write failed with [audit] required (AuditWriteError)" in caplog.text


async def test_an_async_finalize_at_close_is_logged_critical(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    audit = AsyncFinalizeAudit()
    caplog.set_level(logging.CRITICAL, logger="llm_redact")
    async with FakeUpstream() as fake:
        with _proxy(_audited(monkeypatch, audit, fake.port)) as proxy_host:
            async with websockets.connect(f"ws://{proxy_host}/v1/realtime") as client:
                await client.send('{"type":"noop"}')
                await client.recv()
            row = await _recent(proxy_host, lambda r: r["method"] == "WS")
    assert fake.paths == ["/v1/realtime"] and row["status"] == 101
    assert len(audit.begun) == 1
    assert audit.finalize_ran is False and audit.unawaited.closed()
    assert "audit write failed AFTER response (WS /v1/realtime): AuditWriteError" in caplog.text
