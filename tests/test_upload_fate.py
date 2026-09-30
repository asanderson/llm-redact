"""An upload's honesty counts settle once its fate is known.

``inspected_uploads_total{outcome="clean"}`` ("sent byte-identical after a
clean scan") and ``unscanned_uploads_total`` count only parts of an upload
the proxy HANDED TO THE UPSTREAM. A refusal after redaction — no upstream
configured, the upstream authorizer failing, the ``[audit] required``
START row failing, a routed budget refusal — sends nothing: an inspected
clean part is then ``clean_refused`` and a binary part is not counted as
forwarded unscanned (they once were, as soon as redaction returned). A send
that fails in transit still counts as sent (bytes may have left).

Driven end to end through the real app with a scripted inspector
(tests/test_upload_inspection.py) and fake upstreams, authorizers, audit
logs and routers — keyless.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from fake_router import ROUTE_HEADER, FakeRouter, Hop, Refuse402, Stop, install, routed_config
from license_fixtures import resolved
from llm_redact.config import AuditConfig, Config
from llm_redact.detection.engine import DetectionConfig
from llm_redact.proxy import create_app
from test_audit_required import FakeAudit
from test_upload_inspection import (
    FORM,
    FakeInspector,
    Upstream,
    _client,
    _form,
    _pdf,
    _registry,
    reads,
)
from test_upstream_auth import AZURE, FakeAuth, _identity
from test_upstream_auth import _install as install_auth

AZURE_FILES = "/openai/files?api-version=1"
AZURE_FORM = {"api-key": "client", "content-type": FORM["content-type"]}


async def _post(app: Any, path: str, headers: dict[str, str], body: bytes) -> httpx.Response:
    async with _client(app) as client:
        return await client.post(path, content=body, headers=headers)


async def test_no_upstream_configured_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Azure has no default upstream: the request is answered 502 after
    # redaction. One part is read clean, the other cannot be (no text).
    inspector = FakeInspector(reads("clean"), scripts={_pdf("b"): reads(None)})
    upstream = Upstream()
    _registry(monkeypatch, inspector)
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    caplog.set_level(logging.INFO, logger="llm_redact")
    reply = await _post(app, AZURE_FILES, AZURE_FORM, _form(_pdf("a"), _pdf("b")))
    assert reply.status_code == 502 and upstream.requests == []
    assert app.state.proxy.inspected_uploads == {
        ("azure", "clean_refused"): 1,
        ("azure", "incomplete"): 1,
    }
    # The unread part was never forwarded: not counted, not logged as sent.
    assert app.state.proxy.unscanned_uploads == {}
    assert "unscanned" not in caplog.text


async def test_the_same_upload_sent_counts_both(monkeypatch: pytest.MonkeyPatch) -> None:
    inspector = FakeInspector(reads("clean"), scripts={_pdf("b"): reads(None)})
    upstream = Upstream()
    _registry(monkeypatch, inspector)
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    reply = await _post(app, "/v1/files", FORM, _form(_pdf("a"), _pdf("b")))
    assert reply.status_code == 200 and len(upstream.requests) == 1
    assert app.state.proxy.inspected_uploads == {
        ("openai", "clean"): 1,
        ("openai", "incomplete"): 1,
    }
    assert app.state.proxy.unscanned_uploads == {"openai": 1}


async def test_an_authorizer_failure_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    reg, _ = install_auth(
        monkeypatch, lambda name, p: FakeAuth(name, error=RuntimeError("token fetch failed"))
    )
    inspector = FakeInspector(reads("clean", proxy_credential=True))
    upstream = Upstream()
    _registry(monkeypatch, inspector, reg)
    providers = {**Config().providers, "azure": _identity(AZURE)}
    app = create_app(Config(providers=providers), upstream_transport=httpx.MockTransport(upstream))
    headers = {"content-type": FORM["content-type"]}
    reply = await _post(app, AZURE_FILES, headers, _form(_pdf("a")))
    assert reply.status_code == 502 and upstream.requests == []
    assert app.state.proxy.inspected_uploads == {("azure", "clean_refused"): 1}


async def test_a_failed_audit_start_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    inspector = FakeInspector(reads("clean"))
    upstream = Upstream()
    reg = _registry_with_audit(monkeypatch, inspector, FakeAudit(fail_begin=True))
    assert reg is not None
    config = Config(audit=AuditConfig(enabled=True, required=True))
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    reply = await _post(app, "/v1/files", FORM, _form(_pdf("a")))
    assert reply.status_code == 503 and upstream.requests == []
    assert app.state.proxy.inspected_uploads == {("openai", "clean_refused"): 1}


def _registry_with_audit(monkeypatch: pytest.MonkeyPatch, inspector: Any, audit: Any) -> Any:
    from llm_redact.registry import Registry

    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("pro")
    reg.build_access_gate = lambda cfg, lic: None
    reg.build_audit = lambda cfg: audit if cfg.enabled else None
    _registry(monkeypatch, inspector, reg)
    return reg


async def test_a_send_that_fails_in_transit_counts_as_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Handed to the upstream: the proxy cannot always tell how much left.
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _registry(monkeypatch, FakeInspector(reads("clean")))
    app = create_app(Config(), upstream_transport=httpx.MockTransport(down))
    reply = await _post(app, "/v1/files", FORM, _form(_pdf("a")))
    assert reply.status_code == 502
    assert app.state.proxy.inspected_uploads == {("openai", "clean"): 1}


@pytest.mark.parametrize(
    ("script", "status", "outcome"),
    [
        pytest.param([Refuse402()], 402, "clean_refused", id="budget-refusal"),
        pytest.param(
            [Hop("gone", "https://api.openai.com/v1/files", unavailable="KEY_VAR"), Stop()],
            502,
            "clean_refused",
            id="no-hop-sent",
        ),
        pytest.param(
            [Hop("a", "https://api.openai.com/v1/files"), Stop()], 200, "clean", id="hop-sent"
        ),
    ],
)
async def test_a_routed_upload_counts_as_sent_at_its_first_hop(
    monkeypatch: pytest.MonkeyPatch, script: list[Any], status: int, outcome: str
) -> None:
    upstream = Upstream()
    reg, _ = install(
        monkeypatch,
        FakeRouter({"x": script}, plan_kwargs={"x": {"proxy_credential": False}}),
    )
    _registry(monkeypatch, FakeInspector(reads("clean")), reg)
    app = create_app(
        routed_config(detection=DetectionConfig(binary_uploads="refuse")),
        upstream_transport=httpx.MockTransport(upstream),
    )
    reply = await _post(app, "/v1/files", {**FORM, ROUTE_HEADER: "x"}, _form(_pdf("a")))
    assert reply.status_code == status, reply.text
    assert len(upstream.requests) == (status == 200)
    assert app.state.proxy.inspected_uploads == {("openai", outcome): 1}
