"""An upload's honesty counts settle once its fate is known.

``inspected_uploads_total{outcome="clean"}`` ("sent byte-identical after a
clean scan") and ``unscanned_uploads_total`` count only parts of an upload
the proxy HANDED TO THE UPSTREAM. A refusal after redaction — the upstream
authorizer failing, the counted ``[audit] required`` START row failing, a
routed budget refusal — sends nothing: an inspected clean part is then
``clean_refused`` and a binary part is not counted as forwarded unscanned
(they once were, as soon as redaction returned). What refuses the request
whatever the redaction finds — no upstream configured, a routed local
refusal, the early START row failing — comes before the inspection, so
nothing is inspected or counted. A send
that fails in transit still counts as sent (bytes may have left). The
text parts an upload redacted as one text — remembered so their download
is restored byte-exact (``openai.RAW_TEXT_FILES``, bounded) — settle the
same way: remembered only once handed to the upstream.

Driven end to end through the real app with a scripted inspector
(tests/test_upload_inspection.py) and fake upstreams, authorizers, audit
logs and routers — keyless.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from fake_router import (
    ROUTE_HEADER,
    FakeRouter,
    Hop,
    Refuse402,
    Refuse404,
    Stop,
    install,
    routed_config,
)
from license_fixtures import resolved
from llm_redact.audit import AuditWriteError
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


async def test_no_upstream_configured_never_reaches_the_inspector(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Azure has no default upstream: the request is answered 502 BEFORE its
    # binary parts are handed to the inspector (which may send a file off
    # the machine) — a request refused anyway never gets that far.
    inspector = FakeInspector(reads("clean"), scripts={_pdf("b"): reads(None)})
    upstream = Upstream()
    _registry(monkeypatch, inspector)
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    caplog.set_level(logging.INFO, logger="llm_redact")
    reply = await _post(app, AZURE_FILES, AZURE_FORM, _form(_pdf("a"), _pdf("b")))
    assert reply.status_code == 502 and upstream.requests == []
    assert "upstream_base_url" in reply.text
    assert inspector.parts == []
    assert app.state.proxy.inspected_uploads == {}
    # The unread part was never forwarded: not counted, not logged as sent.
    assert app.state.proxy.unscanned_uploads == {}
    assert "unscanned" not in caplog.text
    (row,) = app.state.proxy.recent
    assert (row["status"], row["provider"]) == (502, "azure")


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


async def test_a_failed_audit_start_never_reaches_the_inspector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [audit] required: the START row is written before the inspection, so
    # one that cannot commit refuses the upload before any part is read.
    inspector = FakeInspector(reads("clean"))
    upstream = Upstream()
    audit = FakeAudit(fail_begin=True)
    _registry_with_audit(monkeypatch, inspector, audit)
    config = Config(audit=AuditConfig(enabled=True, required=True))
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    reply = await _post(app, "/v1/files", FORM, _form(_pdf("a")))
    assert reply.status_code == 503 and upstream.requests == []
    assert inspector.parts == []
    assert app.state.proxy.inspected_uploads == {}
    # No START row, so no END row: the refusal is a classic best-effort row.
    assert audit.begun == [] and audit.finalized == []
    assert [entry.status for entry in audit.recorded] == [503]


def _registry_with_audit(
    monkeypatch: pytest.MonkeyPatch, inspector: Any, audit: Any, reg: Any = None
) -> Any:
    from llm_redact.registry import Registry

    reg = reg or Registry()
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


# --- the remembered raw-text uploads settle the same way --------------------------------

TEXT_FILE = b"mail jane.doe@corp.example\n"


@pytest.fixture
def remembered(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A fresh remembered-upload set (``openai.RAW_TEXT_FILES``, bounded:
    what a refused request records evicts what sent ones did)."""
    from llm_redact.providers import openai

    fresh = openai._RawTextFiles()
    monkeypatch.setattr(openai, "RAW_TEXT_FILES", fresh)
    return fresh


def _text_form() -> bytes:
    return _form(TEXT_FILE, filename="notes.txt", content_type="text/plain")


async def test_a_text_upload_refused_after_redaction_is_not_remembered(
    remembered: Any,
) -> None:
    # Redacted as one text, then answered 502 (no Azure upstream): nothing
    # was sent, so nothing a download would trust is remembered.
    upstream = Upstream()
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    reply = await _post(app, AZURE_FILES, AZURE_FORM, _text_form())
    assert reply.status_code == 502 and upstream.requests == []
    assert len(remembered._digests) == 0
    # The same upload handed to the upstream is.
    sent = await _post(app, "/v1/files", FORM, _text_form())
    assert sent.status_code == 200 and len(upstream.requests) == 1
    assert len(remembered._digests) == 1
    assert _sent_text(upstream.requests[0]) in remembered


@pytest.mark.parametrize(
    ("script", "status", "count"),
    [
        pytest.param([Refuse402()], 402, 0, id="budget-refusal"),
        pytest.param([Hop("a", "https://api.openai.com/v1/files"), Stop()], 200, 1, id="hop-sent"),
    ],
)
async def test_a_routed_text_upload_is_remembered_at_its_first_hop(
    monkeypatch: pytest.MonkeyPatch, remembered: Any, script: list[Any], status: int, count: int
) -> None:
    upstream = Upstream()
    install(monkeypatch, FakeRouter({"x": script}, plan_kwargs={"x": {"proxy_credential": False}}))
    app = create_app(routed_config(), upstream_transport=httpx.MockTransport(upstream))
    reply = await _post(app, "/v1/files", {**FORM, ROUTE_HEADER: "x"}, _text_form())
    assert reply.status_code == status, reply.text
    assert len(remembered._digests) == count


def _sent_text(request: httpx.Request) -> bytes:
    return request.content.split(b"\r\n\r\n")[-1].rsplit(b"\r\n--b--", 1)[0]


@pytest.mark.parametrize(
    ("adapter", "path", "body"),
    [
        pytest.param("openai", "/v1/files", None, id="openai"),
        pytest.param("anthropic", "/v1/files", None, id="anthropic"),
        pytest.param(
            "gemini",
            "/upload/v1beta/files",
            b'--b\r\nContent-Type: application/json\r\n\r\n{"file": {}}\r\n'
            b"--b\r\nContent-Type: text/plain\r\n\r\n" + TEXT_FILE + b"\r\n--b--\r\n",
            id="gemini",
        ),
    ],
)
def test_every_upload_adapter_tells_the_proxy_instead_of_remembering(
    remembered: Any, adapter: str, path: str, body: bytes | None
) -> None:
    from llm_redact.detection.engine import Allowlist, build_detectors
    from llm_redact.providers.anthropic import AnthropicAdapter
    from llm_redact.providers.gemini import GeminiAdapter
    from llm_redact.providers.openai import OpenAIAdapter
    from llm_redact.redactor import Redactor
    from llm_redact.vault import InMemoryVault

    adapters = {"openai": OpenAIAdapter, "anthropic": AnthropicAdapter, "gemini": GeminiAdapter}
    chosen = adapters[adapter]()
    assert chosen.redacts_multipart(path)
    redactor = Redactor(
        build_detectors(DetectionConfig(enabled=("email",))),
        InMemoryVault(),
        Allowlist(exact=frozenset(), patterns=()),
    )
    told: list[bytes] = []
    out = chosen.redact_multipart(
        path,
        body or _text_form(),
        b"b",
        redactor,
        inject_note=False,
        require_scanned=True,
        remember_text=told.append,
    )
    assert out is not None and b"@corp.example" not in out
    assert told == [b"mail \xc2\xabEMAIL_001\xc2\xbb\n"]
    assert len(remembered._digests) == 0  # the proxy's to remember
    # Without it (a direct caller), remembered once redacted, as before.
    chosen.redact_multipart(
        path, body or _text_form(), b"b", redactor, inject_note=False, require_scanned=True
    )
    assert told[0] in remembered


# --- [audit] required: the START row precedes an upload's inspection ----------------------
#
# The write-ahead START row commits BEFORE any byte leaves the machine — for
# an upload whose binary parts go to the inspector, before that inspection
# (the inspector may send a file to a service). Every START row gets exactly
# one END row: the refusals after the inspection record without a token and
# finalize it; a way out that records nothing is closed by handle().

EMAIL = "jane.doe@corp.example"


class OrderedAudit(FakeAudit):
    """A FakeAudit logging its calls in order into a shared event list."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def begin(self, entry: Any) -> object | None:
        self.events.append("start")
        return super().begin(entry)

    def finalize(self, token: object, entry: Any) -> None:
        self.events.append(f"end:{entry.status}")
        super().finalize(token, entry)

    def record(self, entry: Any) -> None:
        self.events.append(f"row:{entry.status}")
        super().record(entry)


class OrderedInspector(FakeInspector):
    def __init__(self, events: list[str], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.events = events

    async def inspect(self, part: Any) -> Any:
        self.events.append("inspect")
        return await super().inspect(part)


def _balanced(audit: FakeAudit) -> None:
    """One START row, finalized once by its own token; no classic row."""
    assert len(audit.begun) == 1
    assert [token for token, _ in audit.finalized] == [1]
    assert audit.recorded == []


def _required(**overrides: Any) -> Config:
    return Config(audit=AuditConfig(enabled=True, required=True), **overrides)


async def test_the_start_row_precedes_the_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    audit, upstream = OrderedAudit(events), Upstream()
    _registry_with_audit(monkeypatch, OrderedInspector(events, reads("clean")), audit)
    app = create_app(_required(), upstream_transport=httpx.MockTransport(upstream))
    reply = await _post(app, "/v1/files", FORM, _form(_pdf("a")))
    assert reply.status_code == 200 and len(upstream.requests) == 1
    assert events == ["start", "inspect", "end:200"]
    _balanced(audit)
    # Written before redaction: the START row carries no detections; the END
    # row carries the request's own.
    assert audit.begun[0].detections == {}
    assert app.state.proxy.inspected_uploads == {("openai", "clean"): 1}


async def test_the_end_row_carries_what_the_redaction_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    audit, upstream = OrderedAudit(events), Upstream()
    _registry_with_audit(monkeypatch, OrderedInspector(events, reads("clean")), audit)
    app = create_app(_required(), upstream_transport=httpx.MockTransport(upstream))
    body = _form(_pdf("a"), filename=f"{EMAIL}.pdf")
    reply = await _post(app, "/v1/files", FORM, body)
    assert reply.status_code == 200 and EMAIL.encode() not in upstream.requests[0].content
    # The redaction found a value: a second START row carrying it commits
    # before the send and supersedes the early one (ended, no status).
    assert events == ["start", "inspect", "start", "end:None", "end:200"]
    _superseded(audit)
    assert audit.begun[1].detections == {"EMAIL": 1}
    assert audit.finalized[1][1].detections == {"EMAIL": 1}
    # The request is counted once: one recent row, no classic audit row.
    assert [row["status"] for row in app.state.proxy.recent] == [200]


def _superseded(audit: FakeAudit) -> None:
    """Two START rows, each finalized once by its own token, the early one
    first (before the send); no classic row."""
    assert len(audit.begun) == 2
    assert [token for token, _ in audit.finalized] == [1, 2]
    assert audit.finalized[0][1].status is None
    assert audit.finalized[0][1].detections == {}
    assert audit.recorded == []


class EndFails(OrderedAudit):
    """Every END row fails (a full disk after the START rows committed)."""

    def finalize(self, token: object, entry: Any) -> None:
        self.events.append(f"end-failed:{entry.status}")
        raise AuditWriteError("disk full")


async def test_a_warned_value_is_durable_before_the_send(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A warn-mode value is FORWARDED: what is committed before upstream
    # contact must say so even when no END row is ever written (the early
    # START row, written before the redaction, cannot).
    events: list[str] = []
    audit, upstream = EndFails(events), Upstream()
    _registry_with_audit(monkeypatch, OrderedInspector(events, reads("clean")), audit)
    config = _required(detection=DetectionConfig(modes=(("email", "warn"),)))
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    caplog.set_level(logging.CRITICAL, logger="llm_redact")
    body = _form(_pdf("a"), filename=f"{EMAIL}.pdf")
    reply = await _post(app, "/v1/files", FORM, body)
    assert reply.status_code == 200 and EMAIL.encode() in upstream.requests[0].content
    assert events == ["start", "inspect", "start", "end-failed:None", "end-failed:200"]
    assert audit.begun[1].warned == {"EMAIL": 1}
    # Both END faults are loud, by type only.
    assert "superseded START row" in caplog.text and "disk full" not in caplog.text
    assert EMAIL not in caplog.text


class SecondStartFails(OrderedAudit):
    def begin(self, entry: Any) -> object | None:
        if self.begun:
            self.events.append("start-failed")
            raise AuditWriteError("injected write fault")
        return super().begin(entry)


@pytest.mark.parametrize("routed", [False, True], ids=["legacy", "routed"])
async def test_a_counted_start_row_that_cannot_commit_refuses(
    monkeypatch: pytest.MonkeyPatch, routed: bool
) -> None:
    # The second START row fails: 503 before any upstream contact, and its
    # refusal row ends the early START row — the pair stays balanced.
    events: list[str] = []
    audit, upstream = SecondStartFails(events), Upstream()
    reg = None
    headers = FORM
    config = _required()
    if routed:
        reg, _ = install(
            monkeypatch,
            FakeRouter(
                {"x": [Hop("a", "https://api.openai.com/v1/files"), Stop()]},
                plan_kwargs={"x": {"proxy_credential": False}},
            ),
            audit=audit,
        )
        headers = {**FORM, ROUTE_HEADER: "x"}
        config = routed_config(audit=AuditConfig(enabled=True, required=True))
    _registry_with_audit(monkeypatch, OrderedInspector(events, reads("clean")), audit, reg)
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    body = _form(_pdf("a"), filename=f"{EMAIL}.pdf")
    reply = await _post(app, "/v1/files", headers, body)
    assert reply.status_code == 503 and upstream.requests == []
    assert events == ["start", "inspect", "start-failed", "end:503"]
    _balanced(audit)
    assert audit.finalized[0][1].detections == {"EMAIL": 1}
    assert app.state.proxy.inspected_uploads == {("openai", "clean_refused"): 1}


# --- a log with the optional amend member keeps ONE START row ----------------------------
#
# WriteAheadAudit's optional ``amend(token, entry)``: the early START row is
# amended with the request's counts before the send instead of followed by a
# second START row; the END row finalizes the same token.


class AmendingAudit(OrderedAudit):
    """An OrderedAudit with the optional ``amend`` member."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.amended: list[tuple[object, Any]] = []

    def amend(self, token: object, entry: Any) -> None:
        self.events.append("amend")
        self.amended.append((token, entry))


class AmendFails(AmendingAudit):
    def amend(self, token: object, entry: Any) -> None:
        self.events.append("amend-failed")
        raise AuditWriteError("injected write fault")


def test_the_amend_member_is_read_once_and_only_when_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amending = AmendingAudit([])
    _registry_with_audit(monkeypatch, None, amending)
    state = create_app(_required()).state.proxy
    assert state.write_ahead_amend == amending.amend
    # Not required: no write-ahead log, nothing to amend.
    assert create_app(Config(audit=AuditConfig(enabled=True))).state.proxy.write_ahead_amend is None
    # A non-callable attribute of that name is no member: two START rows.
    odd = OrderedAudit([])
    odd.amend = "not a method"  # type: ignore[attr-defined]
    _registry_with_audit(monkeypatch, None, odd)
    assert create_app(_required()).state.proxy.write_ahead_amend is None


@pytest.mark.parametrize("routed", [False, True], ids=["legacy", "routed"])
async def test_an_amending_log_keeps_one_start_row(
    monkeypatch: pytest.MonkeyPatch, routed: bool
) -> None:
    events: list[str] = []
    audit, upstream = AmendingAudit(events), Upstream()
    reg, headers, config = _maybe_routed(monkeypatch, audit, routed)
    _registry_with_audit(monkeypatch, OrderedInspector(events, reads("clean")), audit, reg)
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    body = _form(_pdf("a"), filename=f"{EMAIL}.pdf")
    reply = await _post(app, "/v1/files", headers, body)
    assert reply.status_code == 200 and EMAIL.encode() not in upstream.requests[0].content
    # The early START row is amended with what the redaction found, before
    # the send; its END row is the request's own. No second START row.
    assert events == ["start", "inspect", "amend", "end:200"]
    _balanced(audit)
    ((token, amendment),) = audit.amended
    assert token == 1 and amendment.detections == {"EMAIL": 1}
    assert (amendment.status, amendment.method, amendment.path) == (None, "POST", "/v1/files")
    assert amendment.session == audit.begun[0].session
    assert audit.begun[0].detections == {}
    assert audit.finalized[0][1].detections == {"EMAIL": 1}
    # The request is counted once.
    assert [row["status"] for row in app.state.proxy.recent] == [200]
    assert app.state.proxy.inspected_uploads == {("openai", "clean"): 1}


def _maybe_routed(
    monkeypatch: pytest.MonkeyPatch, audit: FakeAudit, routed: bool
) -> tuple[Any, dict[str, str], Config]:
    if not routed:
        return None, FORM, _required()
    reg, _ = install(
        monkeypatch,
        FakeRouter(
            {"x": [Hop("a", "https://api.openai.com/v1/files"), Stop()]},
            plan_kwargs={"x": {"proxy_credential": False}},
        ),
        audit=audit,
    )
    config = routed_config(audit=AuditConfig(enabled=True, required=True))
    return reg, {**FORM, ROUTE_HEADER: "x"}, config


async def test_an_amendment_carries_a_warned_value_before_the_send(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A warn-mode value is FORWARDED: the amendment, committed before the
    # send, says so even when no END row is ever written.
    class AmendThenEndFails(AmendingAudit):
        def finalize(self, token: object, entry: Any) -> None:
            self.events.append(f"end-failed:{entry.status}")
            raise AuditWriteError("disk full")

    events: list[str] = []
    audit, upstream = AmendThenEndFails(events), Upstream()
    _registry_with_audit(monkeypatch, OrderedInspector(events, reads("clean")), audit)
    config = _required(detection=DetectionConfig(modes=(("email", "warn"),)))
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    caplog.set_level(logging.CRITICAL, logger="llm_redact")
    body = _form(_pdf("a"), filename=f"{EMAIL}.pdf")
    reply = await _post(app, "/v1/files", FORM, body)
    assert reply.status_code == 200 and EMAIL.encode() in upstream.requests[0].content
    assert events == ["start", "inspect", "amend", "end-failed:200"]
    assert len(audit.begun) == 1 and audit.amended[0][1].warned == {"EMAIL": 1}
    assert "superseded" not in caplog.text and EMAIL not in caplog.text


async def test_nothing_found_amends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    audit, upstream = AmendingAudit(events), Upstream()
    _registry_with_audit(monkeypatch, OrderedInspector(events, reads("clean")), audit)
    app = create_app(_required(), upstream_transport=httpx.MockTransport(upstream))
    reply = await _post(app, "/v1/files", FORM, _form(_pdf("a")))
    assert reply.status_code == 200
    assert events == ["start", "inspect", "end:200"] and audit.amended == []


@pytest.mark.parametrize("routed", [False, True], ids=["legacy", "routed"])
async def test_an_amendment_that_cannot_commit_refuses(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, routed: bool
) -> None:
    # The amendment fails: the same provider-shaped 503 as a START row that
    # cannot commit, before any upstream contact; the refusal's row ends the
    # one START row, and the inspector-cleared part never counts as clean.
    events: list[str] = []
    audit, upstream = AmendFails(events), Upstream()
    reg, headers, config = _maybe_routed(monkeypatch, audit, routed)
    _registry_with_audit(monkeypatch, OrderedInspector(events, reads("clean")), audit, reg)
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    caplog.set_level(logging.CRITICAL, logger="llm_redact")
    body = _form(_pdf("a"), filename=f"{EMAIL}.pdf")
    reply = await _post(app, "/v1/files", headers, body)
    assert reply.status_code == 503 and upstream.requests == []
    assert reply.json()["error"]["message"] == (
        "llm-redact: audit log unavailable and [audit] required is enabled"
    )
    assert events == ["start", "inspect", "amend-failed", "end:503"]
    _balanced(audit)
    assert audit.finalized[0][1].detections == {"EMAIL": 1}
    assert app.state.proxy.inspected_uploads == {("openai", "clean_refused"): 1}
    assert app.state.proxy.unscanned_uploads == {}
    assert [row["status"] for row in app.state.proxy.recent] == [503]
    assert "audit write failed" in caplog.text and "injected" not in caplog.text
    assert EMAIL not in caplog.text


def _down(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


@pytest.mark.parametrize(
    ("script", "detection", "transport", "status"),
    [
        pytest.param(reads(f"mail {EMAIL}"), None, None, 400, id="value-in-the-file"),
        pytest.param(
            reads(f"mail {EMAIL}"),
            DetectionConfig(modes=(("email", "block"),)),
            None,
            400,
            id="block-mode-value-in-the-file",
        ),
        pytest.param(reads("clean"), None, _down, 502, id="send-fails-in-transit"),
    ],
)
async def test_a_refusal_after_the_inspection_ends_the_start_row(
    monkeypatch: pytest.MonkeyPatch,
    script: Any,
    detection: DetectionConfig | None,
    transport: Any,
    status: int,
) -> None:
    events: list[str] = []
    audit, upstream = OrderedAudit(events), Upstream()
    _registry_with_audit(monkeypatch, OrderedInspector(events, script), audit)
    extra = {"detection": detection} if detection is not None else {}
    app = create_app(
        _required(**extra), upstream_transport=httpx.MockTransport(transport or upstream)
    )
    reply = await _post(app, "/v1/files", FORM, _form(_pdf("a")))
    assert reply.status_code == status and upstream.requests == []
    assert events == ["start", "inspect", f"end:{status}"]
    _balanced(audit)


async def test_an_authorizer_failure_ends_the_start_row(monkeypatch: pytest.MonkeyPatch) -> None:
    # The authorizer signs the FINAL bytes, so it stays after the inspection;
    # its refusal is the START row's END row.
    events: list[str] = []
    audit = OrderedAudit(events)
    reg, _ = install_auth(
        monkeypatch, lambda name, p: FakeAuth(name, error=RuntimeError("token fetch failed"))
    )
    inspector = OrderedInspector(events, reads("clean", proxy_credential=True))
    _registry_with_audit(monkeypatch, inspector, audit, reg)
    upstream = Upstream()
    providers = {**Config().providers, "azure": _identity(AZURE)}
    app = create_app(
        _required(providers=providers), upstream_transport=httpx.MockTransport(upstream)
    )
    headers = {"content-type": FORM["content-type"]}
    reply = await _post(app, AZURE_FILES, headers, _form(_pdf("a")))
    assert reply.status_code == 502 and upstream.requests == []
    assert events == ["start", "inspect", "end:502"]
    _balanced(audit)


@pytest.mark.parametrize(
    ("script", "status", "sent"),
    [
        pytest.param([Refuse402()], 402, False, id="budget-refusal"),
        pytest.param(
            [Hop("gone", "https://api.openai.com/v1/files", unavailable="KEY_VAR"), Stop()],
            502,
            False,
            id="no-hop-sent",
        ),
        pytest.param(
            [Hop("a", "https://api.openai.com/v1/files"), Stop()], 200, True, id="hop-sent"
        ),
    ],
)
async def test_a_routed_upload_writes_its_start_row_before_the_inspection(
    monkeypatch: pytest.MonkeyPatch, script: list[Any], status: int, sent: bool
) -> None:
    events: list[str] = []
    audit, upstream = OrderedAudit(events), Upstream()
    reg, _ = install(
        monkeypatch,
        FakeRouter({"x": script}, plan_kwargs={"x": {"proxy_credential": False}}),
        audit=audit,
    )
    _registry_with_audit(monkeypatch, OrderedInspector(events, reads("clean")), audit, reg)
    app = create_app(
        routed_config(audit=AuditConfig(enabled=True, required=True)),
        upstream_transport=httpx.MockTransport(upstream),
    )
    reply = await _post(app, "/v1/files", {**FORM, ROUTE_HEADER: "x"}, _form(_pdf("a")))
    assert reply.status_code == status, reply.text
    assert len(upstream.requests) == sent
    assert events == ["start", "inspect", f"end:{status}"]
    _balanced(audit)


async def test_a_routed_local_refusal_never_reaches_the_inspector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The router's local refusal (the count_tokens 404 shape) precedes the
    # START row and, for an upload, the inspection.
    events: list[str] = []
    audit, upstream = OrderedAudit(events), Upstream()
    inspector = OrderedInspector(events, reads("clean"))
    reg, _ = install(
        monkeypatch,
        FakeRouter({"x": [Refuse404()]}, plan_kwargs={"x": {"proxy_credential": False}}),
        audit=audit,
    )
    _registry_with_audit(monkeypatch, inspector, audit, reg)
    app = create_app(
        routed_config(audit=AuditConfig(enabled=True, required=True)),
        upstream_transport=httpx.MockTransport(upstream),
    )
    reply = await _post(app, "/v1/files", {**FORM, ROUTE_HEADER: "x"}, _form(_pdf("a")))
    assert reply.status_code == 404 and upstream.requests == []
    assert inspector.parts == [] and events == ["row:404"]
    assert app.state.proxy.inspected_uploads == {}


async def test_a_way_out_that_records_nothing_still_ends_the_start_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An exception after the START row (nothing recorded, no answer): the
    # row is closed by handle() with no status, never left without its END.
    from llm_redact.providers.openai import OpenAIAdapter

    def broken(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("an unexpected fault")

    events: list[str] = []
    audit, upstream = OrderedAudit(events), Upstream()
    _registry_with_audit(monkeypatch, OrderedInspector(events, reads("clean")), audit)
    monkeypatch.setattr(OpenAIAdapter, "redact_multipart", broken)
    app = create_app(_required(), upstream_transport=httpx.MockTransport(upstream))
    with pytest.raises(RuntimeError, match="an unexpected fault"):
        await _post(app, "/v1/files", FORM, _form(_pdf("a")))
    assert upstream.requests == []
    assert events == ["start", "inspect", "end:None"]
    _balanced(audit)


@pytest.mark.parametrize("inspector", [True, False], ids=["inspector", "no-inspector"])
async def test_without_a_part_to_inspect_the_start_row_stays_after_redaction(
    monkeypatch: pytest.MonkeyPatch, inspector: bool
) -> None:
    # A text upload (no binary part) — and any upload without an inspector —
    # keeps the old order: START after redaction, carrying its detections.
    events: list[str] = []
    audit, upstream = OrderedAudit(events), Upstream()
    _registry_with_audit(
        monkeypatch, OrderedInspector(events, reads("clean")) if inspector else None, audit
    )
    app = create_app(_required(), upstream_transport=httpx.MockTransport(upstream))
    body = _form(f"mail {EMAIL}\n".encode(), filename="notes.txt", content_type="text/plain")
    reply = await _post(app, "/v1/files", FORM, body)
    assert reply.status_code == 200 and len(upstream.requests) == 1
    assert events == ["start", "end:200"]
    _balanced(audit)
    assert audit.begun[0].detections == {"EMAIL": 1}


async def test_without_an_inspector_no_upstream_still_answers_after_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unchanged without an inspector: a block-mode value is found by the
    # redaction before the missing Azure upstream is noticed.
    upstream = Upstream()
    config = Config(detection=DetectionConfig(modes=(("email", "block"),)))
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    body = _form(b"x", filename=f"{EMAIL}.pdf")
    reply = await _post(app, AZURE_FILES, AZURE_FORM, body)
    assert reply.status_code == 400
    assert 'mode = "block"' in reply.json()["error"]["message"]
    assert upstream.requests == []
