"""The upload-inspection seam: a plugin reads binary upload parts as text,
the core scans that text and decides (``upload_inspection``).

Driven end to end through the real app with a scripted inspector on every
upload route the core redacts: OpenAI/custom/Azure ``…/files``, OpenAI
container files, Anthropic's Files API and the Gemini API's single-request
upload. What is pinned:

- a complete reading that scans clean lets the part go out BYTE-IDENTICAL
  with the client's own key (counted ``clean``, not unscanned — even under
  ``binary_uploads = "refuse"``) and, under a credential the proxy holds,
  only when the inspection allows it;
- a value that would be redacted refuses the request (400, types only),
  complete reading or not; block mode blocks; warn mode counts and
  forwards; allowlists, per-type allowlists and deny strings apply — and no
  placeholder is ever issued for an extracted text;
- no text, an incomplete clean reading, a timeout, an exception, a
  malformed answer, a part over the inspector's size limit: the unscanned
  binary rules exactly as without an inspector;
- the core's bounds (concurrency, part count, deadline, text and string
  budgets), the counters on /status and /metrics, the posture line, the
  plugin_api validation of the declared bounds and the shutdown close.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from fake_router import install, routed_config
from lent_routes import LentRouter
from llm_redact.config import Config, ConfigError, ProviderConfig, parse_config
from llm_redact.detection.engine import DetectionConfig, build_allowlist
from llm_redact.multipart import MultipartPart
from llm_redact.plugin_api import Inspection, UploadPart
from llm_redact.providers.base import InspectedUpload, ProviderAdapter
from llm_redact.providers.documents import read_files_upload, redact_files_upload
from llm_redact.providers.openai import PartsReading, reading_of
from llm_redact.proxy import create_app
from llm_redact.redactor import Redactor
from llm_redact.registry import Registry
from llm_redact.upload_inspection import (
    INSPECT_CONCURRENCY,
    MAX_INSPECT_SECONDS,
    MAX_INSPECTED_PARTS,
    Limits,
    declared_type,
    inspect_parts,
)
from llm_redact.vault import InMemoryVault
from test_upstream_auth import AZURE, _identity
from test_upstream_auth import _install as install_auth

EMAIL = "jane.doe@corp.example"
PHONE = "+1 415 555 0132"
PDF = b"%PDF-1.7\n1 0 obj << >> endobj\n%%EOF\n"
KEY = {"authorization": "Bearer sk-client"}
FORM = {**KEY, "content-type": "multipart/form-data; boundary=b"}


def _form(*files: bytes, filename: str = "report.pdf", content_type: str = "application/pdf"):
    out = b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
    for content in files:
        out += (
            b'--b\r\nContent-Disposition: form-data; name="file"; filename="'
            + filename.encode()
            + b'"\r\nContent-Type: '
            + content_type.encode()
            + b"\r\n\r\n"
            + content
            + b"\r\n"
        )
    return out + b"--b--\r\n"


def _pdf(tag: str) -> bytes:
    """A distinct binary file per tag (a PDF by signature)."""
    return PDF + tag.encode()


Script = Callable[[UploadPart], Awaitable[Any]]


def reads(text: str | None, *, complete: bool = True, proxy_credential: bool = False) -> Script:
    async def script(part: UploadPart) -> Inspection:
        return Inspection(text, complete, "fake", proxy_credential=proxy_credential)

    return script


class FakeInspector:
    """A scripted ``plugin_api.UploadInspector``: each part's reading comes
    from ``scripts`` (by content), else ``default``."""

    def __init__(
        self,
        default: Script | None = None,
        *,
        timeout: Any = 5.0,
        max_bytes: Any = 1 << 20,
        scripts: dict[bytes, Script] | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.default = default or reads("nothing to see", complete=True)
        self.scripts = scripts or {}
        self.parts: list[UploadPart] = []
        self.active = 0
        self.peak = 0
        self.cancelled = 0
        self.closed = 0
        self.status_error = False

    async def inspect(self, part: UploadPart) -> Inspection:
        self.parts.append(part)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0)
            return await self.scripts.get(part.content, self.default)(part)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1

    def status(self) -> dict[str, Any]:
        if self.status_error:
            raise RuntimeError("boom")
        return {"formats": ["pdf"]}

    async def aclose(self) -> None:
        self.closed += 1


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"id": "file-1", "object": "file"})


def _registry(monkeypatch: pytest.MonkeyPatch, inspector: Any, reg: Registry | None = None):
    reg = reg or Registry()
    calls: list[tuple[Config, str]] = []

    def build(config: Config, tier: str) -> Any:
        calls.append((config, tier))
        return inspector

    reg.build_upload_inspector = build
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return calls


def _app(monkeypatch: pytest.MonkeyPatch, inspector: Any, upstream: Upstream, **config: Any):
    _registry(monkeypatch, inspector)
    providers = {
        **Config().providers,
        "custom:lm": ProviderConfig("http://127.0.0.1:1234/v1"),
    }
    return create_app(
        Config(providers=providers, **config), upstream_transport=httpx.MockTransport(upstream)
    )


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _sent_file(request: httpx.Request) -> bytes:
    """The (last) file part's bytes as the upstream received them."""
    return request.content.split(b"\r\n\r\n")[-1].rsplit(b"\r\n--b--", 1)[0]


# --- the seam on the OpenAI Files upload (own key) --------------------------------------


async def test_a_complete_clean_reading_forwards_the_file_byte_identical(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    inspector, upstream = FakeInspector(reads("quarterly figures, nothing personal")), Upstream()
    app = _app(monkeypatch, inspector, upstream)
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
        status = (await client.get("/__llm-redact/status")).json()
        metrics = (await client.get("/__llm-redact/metrics")).text
    assert reply.status_code == 200
    (sent,) = upstream.requests
    assert _sent_file(sent) == _pdf("a")
    (part,) = inspector.parts
    assert part == UploadPart(_pdf("a"), "application/pdf", "openai", False, extension="pdf")
    assert status["inspected_uploads_total"] == {"openai": {"clean": 1}}
    assert status["unscanned_uploads_total"] == {}
    assert status["upload_inspector"] == {
        "enabled": True,
        "timeout_seconds": 5.0,
        "max_part_bytes": 1 << 20,
        "inspector": {"formats": ["pdf"]},
    }
    assert 'llm_redact_inspected_uploads_total{provider="openai",outcome="clean"} 1' in metrics
    assert "inspected 1 binary upload file part(s): clean=1" in caplog.text
    assert "report.pdf" not in caplog.text


async def test_a_clean_file_goes_out_even_when_unscanned_binaries_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = Upstream()
    app = _app(
        monkeypatch,
        FakeInspector(reads("clean")),
        upstream,
        detection=DetectionConfig(binary_uploads="refuse"),
    )
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
    assert reply.status_code == 200 and len(upstream.requests) == 1


@pytest.mark.parametrize("complete", [True, False], ids=["complete", "incomplete"])
async def test_a_value_to_redact_refuses_the_request_naming_types_only(
    monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    upstream = Upstream()
    inspector = FakeInspector(reads(f"call {PHONE} or mail {EMAIL}", complete=complete))
    app = _app(monkeypatch, inspector, upstream)
    state = app.state.proxy
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
        recent = (await client.get("/__llm-redact/recent")).json()["entries"]
    assert reply.status_code == 400
    message = reply.json()["error"]["message"]
    assert "EMAIL" in message and "PHONE" in message and "extracted text" in message
    assert EMAIL not in reply.text and PHONE not in reply.text
    assert upstream.requests == []
    # Nothing was redacted, so nothing was issued: the vault stays empty.
    assert len(state.vault) == 0 and dict(state.detection_counts) == {}
    assert state.inspected_uploads == {("openai", "detected"): 1}
    assert recent[0]["status"] == 400


async def test_block_mode_blocks_and_warn_mode_counts_and_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = Upstream()
    blocked = FakeInspector(reads(f"mail {EMAIL}"))
    block = DetectionConfig(modes=(("email", "block"),))
    app = _app(monkeypatch, blocked, upstream, detection=block)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
    assert reply.status_code == 400 and 'mode = "block"' in reply.json()["error"]["message"]
    assert app.state.proxy.blocked_counts == {"EMAIL": 1}
    assert app.state.proxy.inspected_uploads == {("openai", "blocked"): 1}

    warned = FakeInspector(reads(f"mail {EMAIL}"))
    app = _app(monkeypatch, warned, upstream, detection=DetectionConfig(modes=(("email", "warn"),)))
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
    assert reply.status_code == 200 and _sent_file(upstream.requests[-1]) == _pdf("a")
    assert app.state.proxy.warn_counts == {"EMAIL": 1}
    assert app.state.proxy.inspected_uploads == {("openai", "clean"): 1}


async def test_each_row_counts_only_its_own_request_across_the_inspection(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The inspection is awaited inside the upload's count window: a request
    # redacting (and warn-forwarding) meanwhile was once counted on the
    # upload's row too — its log line, recent/events and audit rows.
    started, release = asyncio.Event(), asyncio.Event()

    async def slow(part: UploadPart) -> Inspection:
        started.set()
        await release.wait()
        return Inspection(f"call {PHONE}", True, "fake")  # the file's own warn value

    upstream = Upstream()
    app = _app(
        monkeypatch,
        FakeInspector(slow),
        upstream,
        detection=DetectionConfig(modes=(("phone_number", "warn"),)),
    )
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        upload = asyncio.create_task(
            client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
        )
        await started.wait()
        chat = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": f"{EMAIL} {PHONE}"}]},
            headers=KEY,
        )
        release.set()
        reply = await upload
    assert chat.status_code == 200 and reply.status_code == 200
    rows = {row["path"]: row for row in app.state.proxy.recent}
    assert rows["/v1/chat/completions"]["detections"] == {"EMAIL": 1}
    assert rows["/v1/chat/completions"]["warned"] == {"PHONE": 1}
    assert rows["/v1/files"]["detections"] == {}
    assert rows["/v1/files"]["warned"] == {"PHONE": 1}  # its extracted text's own
    (line,) = [r.getMessage() for r in caplog.records if "/v1/files -> 200" in r.getMessage()]
    assert "redacted" not in line
    assert app.state.proxy.warn_counts == {"PHONE": 2}


async def test_allowlists_and_deny_strings_apply_to_the_extracted_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = Upstream()
    text = f"mail {EMAIL}, call {PHONE}"
    for config, status in (
        (DetectionConfig(allowlist=(EMAIL,)), 400),  # the phone is still found
        (DetectionConfig(allowlist_by_type=(("PHONE", (PHONE,)),)), 400),  # the email
        (DetectionConfig(allowlist=(EMAIL,), allowlist_by_type=(("PHONE", (PHONE,)),)), 200),
    ):
        app = _app(monkeypatch, FakeInspector(reads(text)), upstream, detection=config)
        async with _client(app) as client:
            reply = await client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
        assert reply.status_code == status
    assert len(upstream.requests) == 1

    denied = FakeInspector(reads("the Project Nightjar budget"))
    config = parse_config({"detection": {"deny": ["project nightjar"]}}, "t").detection
    app = _app(monkeypatch, denied, upstream, detection=config)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
    assert reply.status_code == 400 and "Nightjar" not in reply.text
    assert len(upstream.requests) == 1


@pytest.mark.parametrize(
    ("script", "outcome"),
    [
        (reads(None), "incomplete"),
        (reads("clean but partial", complete=False), "incomplete"),
        (reads("clean", complete=1), "incomplete"),  # type: ignore[arg-type]  # never truthy
    ],
    ids=["no-text", "partial", "sloppy-complete"],
)
@pytest.mark.parametrize("mode", ["forward", "refuse"])
async def test_an_unread_part_keeps_the_unscanned_binary_rules(
    monkeypatch: pytest.MonkeyPatch, script: Script, outcome: str, mode: str
) -> None:
    upstream = Upstream()
    app = _app(
        monkeypatch, FakeInspector(script), upstream, detection=DetectionConfig(binary_uploads=mode)
    )
    state = app.state.proxy
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
    assert state.inspected_uploads == {("openai", outcome): 1}
    if mode == "forward":
        assert reply.status_code == 200 and _sent_file(upstream.requests[0]) == _pdf("a")
        assert state.unscanned_uploads == {"openai": 1}
    else:
        assert reply.status_code == 400 and "binary" in reply.json()["error"]["message"]
        assert upstream.requests == [] and state.unscanned_uploads == {}


async def _raises(part: UploadPart) -> Inspection:
    raise RuntimeError(f"cannot parse {part.content!r}")


async def _not_an_inspection(part: UploadPart) -> Any:
    return {"text": "clean", "complete": True}


async def _hangs(part: UploadPart) -> Inspection:
    await asyncio.sleep(60)
    raise AssertionError("never reached")


async def _cancels_itself(part: UploadPart) -> Inspection:
    raise asyncio.CancelledError


@pytest.mark.parametrize(
    ("script", "outcome"),
    [
        (_raises, "error"),
        (_not_an_inspection, "error"),
        (_cancels_itself, "error"),
        (_hangs, "timeout"),
    ],
)
async def test_faults_and_timeouts_fail_closed_to_the_unscanned_rules(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    script: Script,
    outcome: str,
) -> None:
    upstream, inspector = Upstream(), FakeInspector(script, timeout=0.2)
    app = _app(monkeypatch, inspector, upstream)
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
    assert reply.status_code == 200 and _sent_file(upstream.requests[0]) == _pdf("a")
    assert app.state.proxy.inspected_uploads == {("openai", outcome): 1}
    assert app.state.proxy.unscanned_uploads == {"openai": 1}
    assert "%PDF" not in caplog.text  # an exception's message never reaches the log
    if outcome == "timeout":
        for _ in range(3):
            await asyncio.sleep(0)  # the cancellation lands
        assert inspector.cancelled == 1


async def test_parts_over_the_size_limit_are_never_handed_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream, inspector = Upstream(), FakeInspector(reads("clean"), max_bytes=len(_pdf("a")))
    app = _app(monkeypatch, inspector, upstream)
    async with _client(app) as client:
        reply = await client.post(
            "/v1/files", content=_form(_pdf("a"), _pdf("bigger")), headers=FORM
        )
    assert reply.status_code == 200
    assert [p.content for p in inspector.parts] == [_pdf("a")]
    state = app.state.proxy
    assert state.inspected_uploads == {("openai", "clean"): 1, ("openai", "not_inspected"): 1}
    assert state.unscanned_uploads == {"openai": 1}


async def test_bounded_concurrency_and_part_count(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream, inspector = Upstream(), FakeInspector(reads("clean"))
    app = _app(monkeypatch, inspector, upstream)
    files = [_pdf(str(n)) for n in range(MAX_INSPECTED_PARTS + 2)]
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(*files), headers=FORM)
    assert reply.status_code == 200
    assert len(inspector.parts) == MAX_INSPECTED_PARTS
    assert inspector.peak == INSPECT_CONCURRENCY
    assert app.state.proxy.inspected_uploads == {
        ("openai", "clean"): MAX_INSPECTED_PARTS,
        ("openai", "not_inspected"): 2,
    }
    assert app.state.proxy.unscanned_uploads == {"openai": 2}


async def test_one_dirty_part_refuses_the_whole_upload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    upstream = Upstream()
    inspector = FakeInspector(reads("clean"), scripts={_pdf("b"): reads(f"to {EMAIL}")})
    app = _app(monkeypatch, inspector, upstream)
    async with _client(app) as client:
        reply = await client.post(
            "/v1/files", content=_form(_pdf("a"), _pdf("b"), _pdf("c")), headers=FORM
        )
    assert reply.status_code == 400 and upstream.requests == []
    # The clean parts were not sent: never counted (nor reported) as
    # forwarded after a clean scan.
    assert app.state.proxy.inspected_uploads == {
        ("openai", "clean_refused"): 2,
        ("openai", "detected"): 1,
    }
    from llm_redact.cli import _print_posture

    async with _client(app) as client:
        _print_posture((await client.get("/__llm-redact/status")).json())
    assert "forwarded after a clean scan" not in capsys.readouterr().out


async def test_the_extracted_text_is_bounded_by_the_body_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = Upstream()
    body_parts = (_pdf("a"), _pdf("b"), _pdf("c"))
    body = _form(*body_parts[:2])
    # More extracted text than max_body_bytes allows over the request: the
    # part that would exceed it is not scanned (incomplete).
    inspector = FakeInspector(reads("x" * 3000))
    app = _app(monkeypatch, inspector, upstream, max_body_bytes=5000)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=body, headers=FORM)
    assert reply.status_code == 200
    assert app.state.proxy.inspected_uploads == {
        ("openai", "clean"): 1,
        ("openai", "incomplete"): 1,
    }
    # Each extracted text is one string against max_body_strings: the
    # request is refused (413), and every inspected part still counts once
    # — the one that ran the budget out unscanned. Five parts (the parts
    # cap holds) and a three-line JSONL file (three strings, charged as the
    # upload is read): the third extracted text is the sixth string.
    app = _app(monkeypatch, FakeInspector(reads("clean")), upstream, max_body_strings=5)
    jsonl = b'{"a": 1}\n{"b": 2}\n{"c": 3}'
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(jsonl, *body_parts), headers=FORM)
    assert reply.status_code == 413 and len(upstream.requests) == 1
    assert app.state.proxy.inspected_uploads == {
        ("openai", "clean_refused"): 2,
        ("openai", "incomplete"): 1,
    }


async def test_a_token_inside_the_file_bounds_the_new_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The file carries «EMAIL_007» where its raw bytes do not show it (a
    # compressed stream): the new value in the file NAME is numbered above.
    upstream = Upstream()
    inspector = FakeInspector(reads("as agreed with «EMAIL_007»"))
    app = _app(monkeypatch, inspector, upstream)
    body = _form(_pdf("a"), filename=f"from {EMAIL}")
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=body, headers=FORM)
    assert reply.status_code == 200
    (sent,) = upstream.requests
    assert "from «EMAIL_008»".encode() in sent.content
    assert b"EMAIL_001" not in sent.content


async def test_no_binary_part_no_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream, inspector = Upstream(), FakeInspector()
    app = _app(monkeypatch, inspector, upstream)
    async with _client(app) as client:
        reply = await client.post(
            "/v1/files", content=_form(f"notes {EMAIL}".encode(), filename="n.txt"), headers=FORM
        )
    assert reply.status_code == 200 and inspector.parts == []
    assert EMAIL.encode() not in upstream.requests[0].content
    assert app.state.proxy.inspected_uploads == {}


async def test_detection_off_never_inspects(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream, inspector = Upstream(), FakeInspector(reads(f"mail {EMAIL}"))
    _registry(monkeypatch, inspector)
    openai = dataclasses.replace(Config().providers["openai"], detection=False)
    providers = {**Config().providers, "openai": openai}
    app = create_app(Config(providers=providers), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
    assert reply.status_code == 200 and inspector.parts == []


async def test_an_unreadable_upload_is_refused_before_any_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream, inspector = Upstream(), FakeInspector()
    app = _app(monkeypatch, inspector, upstream)
    body = b"stray preamble\r\n" + _form(_pdf("a"))
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=body, headers=FORM)
    assert reply.status_code == 400 and "preamble" in reply.json()["error"]["message"]
    assert inspector.parts == [] and upstream.requests == []


_CTE = b"Content-Type: application/pdf\r\nContent-Transfer-Encoding: base64\r\n"


@pytest.mark.parametrize(
    "body",
    [
        # A file part in an encoding the upstream would decode, unread.
        _form(_pdf("a")).replace(b"Content-Type: application/pdf\r\n", _CTE),
        # A scanned field declaring a charset the proxy does not decode.
        _form(_pdf("a")).replace(
            b'name="purpose"\r\n', b'name="purpose"\r\nContent-Type: text/plain; charset=koi8-r\r\n'
        ),
        # A file name without a single reading.
        _form(_pdf("a")).replace(b'filename="report.pdf"', b"filename*=iso-8859-1''r%E9port.pdf"),
        _form(_pdf("a")).replace(
            b"Content-Type: application/pdf\r\n",
            b'Content-Type: application/pdf\r\nContent-Disposition: form-data; name="x"\r\n',
        ),
        # A part without a header/body separator: a reader accepting a bare
        # LF finds a transfer encoding in it the proxy never read.
        _form(_pdf("a")).replace(
            b"--b--",
            b"--b\r\nContent-Type: text/plain\nContent-Transfer-Encoding: base64\n\nx\r\n--b--",
        ),
        # A form field the part loop cannot read as UTF-8 text, and a JSONL
        # line nesting deeper than the walk reads: refused for their format
        # before any binary part is handed over.
        _form(_pdf("a")).replace(b"user_data", b"\xff\xfe"),
        _form(b'{"a": ' * 200 + b"1" + b"}" * 200, _pdf("a")),
    ],
    ids=[
        "transfer-encoding",
        "charset",
        "filename-charset",
        "repeated-disposition",
        "headerless",
        "non-utf8-field",
        "too-deep-jsonl-line",
    ],
)
async def test_a_part_the_redaction_refuses_is_never_inspected(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    # Checked before any part is handed over: an inspector may send a file
    # to a service, and a request refused anyway must send nothing anywhere.
    upstream, inspector = Upstream(), FakeInspector(reads("clean"))
    app = _app(monkeypatch, inspector, upstream)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=body, headers=FORM)
    assert reply.status_code == 400, reply.text
    assert inspector.parts == [] and upstream.requests == []
    assert app.state.proxy.inspected_uploads == {}


async def test_readable_parts_beside_a_binary_one_are_inspected_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream, inspector = Upstream(), FakeInspector(reads("clean"))
    app = _app(monkeypatch, inspector, upstream)
    jsonl = b'{"note": "' + EMAIL.encode() + b'"}\n\n{"a": 1}'
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(jsonl, _pdf("a")), headers=FORM)
    assert reply.status_code == 200, reply.text
    assert len(inspector.parts) == 1
    (sent,) = upstream.requests
    assert EMAIL.encode() not in sent.content and _pdf("a") in sent.content


BLOCK_EMAIL = DetectionConfig(modes=(("email", "block"),))


def _field(name: str, value: bytes) -> bytes:
    """A plain form field, to put ahead of a ``_form`` body."""
    return (
        b'--b\r\nContent-Disposition: form-data; name="'
        + name.encode()
        + b'"\r\n\r\n'
        + value
        + b"\r\n"
    )


def _batch_line(model: str) -> bytes:
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    return json.dumps({"custom_id": "1", "method": "POST", "url": "/v1/x", "body": body}).encode()


@pytest.mark.parametrize(
    "body",
    [
        _form(_pdf("a"), filename=f"from {EMAIL}"),
        _form(f"contact {EMAIL}".encode(), _pdf("a")),
        _field("user", EMAIL.encode()) + _form(_pdf("a")),
        _form(b'{"note": "' + EMAIL.encode() + b'"}\n\n{"a": 1}', _pdf("a")),
        # purpose user_data: a line is DATA, every value of it scanned.
        _form(_batch_line(EMAIL), _pdf("a")),
    ],
    ids=["file-name", "text-file", "form-field", "jsonl-line", "data-line-model"],
)
async def test_a_block_mode_value_anywhere_refuses_before_inspection(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    # The inspector may send a file to a service: an upload a block-mode
    # rule refuses — for a value in another part or in a file name — is
    # refused before any binary part is handed over (it once was after).
    upstream, inspector = Upstream(), FakeInspector(reads("clean"))
    app = _app(monkeypatch, inspector, upstream, detection=BLOCK_EMAIL)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=body, headers=FORM)
    assert reply.status_code == 400 and 'mode = "block"' in reply.json()["error"]["message"]
    assert inspector.parts == [] and upstream.requests == []
    state = app.state.proxy
    assert state.inspected_uploads == {} and state.blocked_counts == {"EMAIL": 1}
    assert state.detection_counts == {}  # nothing redacted, nothing issued


async def test_the_block_check_reads_a_request_line_as_the_redaction_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # purpose batch: a line's body is a REQUEST, walked with the request
    # walk's structural skips — a value the redaction never scans (the
    # model name) refuses nothing before the inspection either.
    upstream, inspector = Upstream(), FakeInspector(reads("clean"))
    app = _app(monkeypatch, inspector, upstream, detection=BLOCK_EMAIL)
    body = _form(_batch_line(EMAIL), _pdf("a")).replace(b"user_data", b"batch")
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=body, headers=FORM)
    assert reply.status_code == 200, reply.text
    assert len(inspector.parts) == 1
    (sent,) = upstream.requests
    assert EMAIL.encode() in sent.content  # the structural model, as sent


async def test_the_block_check_counts_nothing_the_redaction_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A rule in block mode makes the check run; a warn-mode value it reads
    # is counted once — by the redaction — and nothing is issued twice.
    upstream, inspector = Upstream(), FakeInspector(reads("clean"))
    modes = (("email", "warn"), ("phone_number", "block"))
    app = _app(monkeypatch, inspector, upstream, detection=DetectionConfig(modes=modes))
    body = _form(f"mail {EMAIL} ssn 123-45-6789".encode(), _pdf("a"), filename=f"from {EMAIL}")
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=body, headers=FORM)
    assert reply.status_code == 200, reply.text
    state = app.state.proxy
    assert state.warn_counts == {"EMAIL": 3}  # two file names and the text, once each
    assert state.recent[0]["warned"] == {"EMAIL": 3}
    assert state.detection_counts == {"SSN": 1} and state.recent[0]["detections"] == {"SSN": 1}
    assert b"\xc2\xabSSN_001\xc2\xbb" in upstream.requests[0].content


async def test_the_block_check_walks_no_more_strings_than_the_redaction_may(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Its own budget of max_body_strings: over it, 413 before inspection
    # (the redaction, charging the same strings, would refuse it too).
    upstream, inspector = Upstream(), FakeInspector(reads("clean"))
    many = json.dumps({"v": ["x"] * 20}).encode()
    app = _app(monkeypatch, inspector, upstream, detection=BLOCK_EMAIL, max_body_strings=10)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(many, _pdf("a")), headers=FORM)
    assert reply.status_code == 413 and inspector.parts == [] and upstream.requests == []
    # Without a rule in block mode there is no check: the redaction refuses,
    # after the inspection.
    app = _app(monkeypatch, inspector, upstream, max_body_strings=10)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_form(many, _pdf("a")), headers=FORM)
    assert reply.status_code == 413 and len(inspector.parts) == 1 and upstream.requests == []


# --- every other upload route ----------------------------------------------------------


def _anthropic_form(content: bytes) -> bytes:
    return (
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="r.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n" + content + b"\r\n--b--\r\n"
    )


def _related(content: bytes) -> bytes:
    metadata = json.dumps({"file": {"display_name": f"notes of {EMAIL}"}}).encode()
    return (
        b"--b\r\nContent-Type: application/json\r\n\r\n"
        + metadata
        + b"\r\n--b\r\nContent-Type: application/pdf\r\n\r\n"
        + content
        + b"\r\n--b--\r\n"
    )


ROUTES: dict[str, tuple[str, dict[str, str], Callable[[bytes], bytes], str]] = {
    "openai": ("/v1/files", FORM, _form, "openai"),
    "custom": ("/custom/lm/v1/files", FORM, _form, "custom:lm"),
    "container": ("/v1/containers/cntr_1/files", FORM, _form, "openai"),
    "anthropic": (
        "/v1/files",
        {
            "anthropic-version": "2023-06-01",
            "x-api-key": "sk-ant-client",
            "content-type": "multipart/form-data; boundary=b",
        },
        _anthropic_form,
        "anthropic",
    ),
    "gemini": (
        "/upload/v1beta/files",
        {
            "x-goog-api-key": "client-key",
            "x-goog-upload-protocol": "multipart",
            "content-type": "multipart/related; boundary=b",
        },
        _related,
        "gemini",
    ),
}


@pytest.mark.parametrize("route", sorted(ROUTES))
async def test_every_upload_route_inspects_its_binary_parts(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    path, headers, body, provider = ROUTES[route]
    upstream = Upstream()
    inspector = FakeInspector(reads("clean"), scripts={_pdf("dirty"): reads(f"to {EMAIL}")})
    app = _app(monkeypatch, inspector, upstream, detection=DetectionConfig(binary_uploads="refuse"))
    async with _client(app) as client:
        clean = await client.post(path, content=body(_pdf("clean")), headers=headers)
        dirty = await client.post(path, content=body(_pdf("dirty")), headers=headers)
    assert clean.status_code == 200, clean.text
    assert dirty.status_code == 400 and "EMAIL" in dirty.json()["error"]["message"]
    (sent,) = upstream.requests
    assert _pdf("clean") in sent.content and EMAIL.encode() not in sent.content
    assert [p.provider for p in inspector.parts] == [provider, provider]
    assert app.state.proxy.inspected_uploads == {(provider, "clean"): 1, (provider, "detected"): 1}


@pytest.mark.parametrize("route", sorted(ROUTES))
async def test_every_upload_route_checks_part_headers_before_inspection(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    path, headers, body, _ = ROUTES[route]
    upstream, inspector = Upstream(), FakeInspector(reads("clean"))
    app = _app(monkeypatch, inspector, upstream)
    encoded = body(_pdf("a")).replace(b"Content-Type: application/pdf\r\n", _CTE)
    async with _client(app) as client:
        reply = await client.post(path, content=encoded, headers=headers)
    assert reply.status_code == 400 and "Content-Transfer-Encoding" in reply.text
    assert inspector.parts == [] and upstream.requests == []


async def test_gemini_metadata_redaction_cannot_read_is_never_inspected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The metadata part is read (as the create's JSON body) with the upload,
    # before any binary part is handed over.
    path, headers, body, _ = ROUTES["gemini"]
    upstream, inspector = Upstream(), FakeInspector(reads("clean"))
    app = _app(monkeypatch, inspector, upstream)
    declared_text = body(_pdf("a")).replace(b"application/json", b"text/plain", 1)
    async with _client(app) as client:
        reply = await client.post(path, content=declared_text, headers=headers)
    assert reply.status_code == 400 and "application/json" in reply.text
    assert inspector.parts == [] and upstream.requests == []


@pytest.mark.parametrize("route", sorted(ROUTES))
async def test_every_upload_route_checks_block_mode_before_inspection(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    # The address in the PDF's file name, or the Gemini upload's metadata.
    path, headers, body, _ = ROUTES[route]
    upstream, inspector = Upstream(), FakeInspector(reads("clean"))
    app = _app(monkeypatch, inspector, upstream, detection=BLOCK_EMAIL)
    named = body(_pdf("a")).replace(b'.pdf"', b" from " + EMAIL.encode() + b'"')
    assert EMAIL.encode() in named
    async with _client(app) as client:
        reply = await client.post(path, content=named, headers=headers)
    assert reply.status_code == 400 and "configured with mode" in reply.text
    assert inspector.parts == [] and upstream.requests == []


# --- under a credential the proxy holds ---------------------------------------------------


AZURE_FILES = "/openai/files?api-version=1"


def _azure_app(monkeypatch: pytest.MonkeyPatch, inspector: Any, upstream: Upstream):
    reg, built = install_auth(monkeypatch)
    _registry(monkeypatch, inspector, reg)
    config = Config(providers={**Config().providers, "azure": _identity(AZURE)})
    return create_app(config, upstream_transport=httpx.MockTransport(upstream)), built


@pytest.mark.parametrize("allowed", [True, False], ids=["allowed", "not-allowed"])
async def test_identity_forwards_a_clean_file_only_when_the_inspection_allows_it(
    monkeypatch: pytest.MonkeyPatch, allowed: bool
) -> None:
    upstream = Upstream()
    inspector = FakeInspector(reads("clean", proxy_credential=allowed))
    app, built = _azure_app(monkeypatch, inspector, upstream)
    async with _client(app) as client:
        reply = await client.post(
            AZURE_FILES, content=_form(_pdf("a")), headers={"content-type": FORM["content-type"]}
        )
    (part,) = inspector.parts
    assert part.identity is True and part.provider == "azure"
    outcome = "clean" if allowed else "clean_refused"
    assert app.state.proxy.inspected_uploads == {("azure", outcome): 1}
    if allowed:
        assert reply.status_code == 200
        (sent,) = upstream.requests
        assert _sent_file(sent) == _pdf("a")
        # The authorizer signed exactly the bytes sent.
        assert built[0].calls[0][3] == sent.content
    else:
        assert reply.status_code == 400 and "binary" in reply.json()["error"]["message"]
        assert upstream.requests == [] and built[0].calls == []
    assert app.state.proxy.unscanned_uploads == {}


@pytest.mark.parametrize(
    "script", [reads("clean", complete=False, proxy_credential=True), _raises, reads(None)]
)
async def test_identity_refuses_every_part_not_read_completely(
    monkeypatch: pytest.MonkeyPatch, script: Script
) -> None:
    upstream = Upstream()
    app, built = _azure_app(monkeypatch, FakeInspector(script), upstream)
    async with _client(app) as client:
        reply = await client.post(
            AZURE_FILES, content=_form(_pdf("a")), headers={"content-type": FORM["content-type"]}
        )
    assert reply.status_code == 400 and upstream.requests == [] and built[0].calls == []


async def test_a_routed_operator_key_is_a_proxy_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = Upstream()
    inspector = FakeInspector(
        reads("clean"), scripts={_pdf("ok"): reads("clean", proxy_credential=True)}
    )
    reg, _ = install(monkeypatch, LentRouter("https://api.openai.com"))
    _registry(monkeypatch, inspector, reg)
    app = create_app(routed_config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        refused = await client.post("/v1/files", content=_form(_pdf("a")), headers=FORM)
        allowed = await client.post("/v1/files", content=_form(_pdf("ok")), headers=FORM)
    assert refused.status_code == 400 and allowed.status_code == 200
    assert [p.identity for p in inspector.parts] == [True, True]
    (sent,) = upstream.requests
    assert _sent_file(sent) == _pdf("ok")


# --- bounds, surfaces, lifecycle -----------------------------------------------------------


@pytest.mark.parametrize(
    ("timeout", "max_bytes"),
    [
        (0, 1),
        (-1.0, 1),
        (float("nan"), 1),
        (float("inf"), 1),
        ("5", 1),
        (True, 1),
        (None, 1),
        (5, 0),
        (5, 1.5),
        (5, True),
        (5, None),
    ],
)
def test_undeclared_or_nonsense_bounds_refuse_to_start(
    monkeypatch: pytest.MonkeyPatch, timeout: Any, max_bytes: Any
) -> None:
    _registry(monkeypatch, FakeInspector(timeout=timeout, max_bytes=max_bytes))
    with pytest.raises(ConfigError, match="upload inspector declares no positive"):
        create_app(Config())


def test_the_declared_timeout_is_capped_by_the_core(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _registry(monkeypatch, FakeInspector(timeout=10_000, max_bytes=7))
    state = create_app(Config()).state.proxy
    assert state.inspection_limits == Limits(MAX_INSPECT_SECONDS, 7)
    assert calls[-1][1] == "free"  # built with the resolved tier


def test_without_an_inspector_nothing_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    assert Registry().build_upload_inspector(Config(), "free") is None
    state = create_app(Config()).state.proxy
    assert state.upload_inspector is None and state.inspection_limits is None


async def test_status_without_an_inspector_and_with_a_failing_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    async with _client(create_app(Config())) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["upload_inspector"] == {"enabled": False}
    assert status["inspected_uploads_total"] == {}
    inspector = FakeInspector()
    inspector.status_error = True
    app = _app(monkeypatch, inspector, Upstream())
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["upload_inspector"]["inspector"] == {"status_error": "RuntimeError"}


async def test_shutdown_closes_the_inspector(monkeypatch: pytest.MonkeyPatch) -> None:
    inspector = FakeInspector()
    app = _app(monkeypatch, inspector, Upstream())
    async with app.router.lifespan_context(app):
        pass
    assert inspector.closed == 1

    class Failing(FakeInspector):
        async def aclose(self) -> None:
            raise RuntimeError("worker gone")

    failing = Failing()
    app = _app(monkeypatch, failing, Upstream())
    async with app.router.lifespan_context(app):
        pass  # the fault is logged, the rest of the shutdown still runs


def test_the_posture_line_names_clean_forwards(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.cli import _print_posture

    base: dict[str, Any] = {"warnings_total": {}, "detection": {}, "audit": {}}
    _print_posture({**base, "inspected_uploads_total": {"openai": {"incomplete": 2}}})
    assert "all traffic redacted" in capsys.readouterr().out
    _print_posture(
        {**base, "inspected_uploads_total": {"openai": {"clean": 3}, "gemini": {"clean": 1}}}
    )
    out = capsys.readouterr().out
    assert "binary uploads: gemini×1 openai×3 file part(s) forwarded after a clean scan" in out
    assert "EXTRACTED text" in out


# --- the pieces ------------------------------------------------------------------------------


def _part(headers: bytes | None, content: bytes = PDF) -> MultipartPart:
    return MultipartPart(headers, content)


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (b"Content-Type: Application/PDF; name=x", "application/pdf"),
        (
            b"Content-Type: application/vnd.oasis.opendocument.text",
            "application/vnd.oasis.opendocument.text",
        ),
        (b"Content-Type: text/plain\r\nContent-Type: text/html", None),  # ambiguous
        (b"Content-Type: not a type", None),
        (b"Content-Type: pdf", None),
        (b'Content-Disposition: form-data; name="file"', None),
        (None, None),
    ],
)
def test_declared_type(headers: bytes | None, expected: str | None) -> None:
    assert declared_type(_part(headers)) == expected


async def test_a_cancelled_request_cancels_its_inspections() -> None:
    inspector = FakeInspector(_hangs)
    parts = [(0, _part(None)), (1, _part(None, PDF + b"2"))]
    task = asyncio.ensure_future(
        inspect_parts(inspector, parts, provider="openai", identity=False, limits=Limits(30, 99))
    )
    while inspector.active < 2:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert inspector.cancelled == 2


async def test_an_inspection_that_ignores_cancellation_is_abandoned_quietly() -> None:
    released = asyncio.Event()

    async def stubborn(part: UploadPart) -> Inspection:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await released.wait()
            raise RuntimeError("late failure") from None
        raise AssertionError("never reached")

    inspector = FakeInspector(stubborn)
    results = await inspect_parts(
        inspector, [(3, _part(None))], provider="p", identity=False, limits=Limits(0.05, 99)
    )
    assert results == {3: "timeout"}
    released.set()
    for _ in range(5):
        await asyncio.sleep(0)  # the abandoned task ends; its exception is retrieved


def test_the_reading_is_parsed_once_and_reused() -> None:
    body = _anthropic_form(_pdf("a"))
    reading = read_files_upload(body, b"b", lambda n: None)
    assert isinstance(reading, PartsReading)
    assert [index for index, _ in reading.binary_parts()] == [0]
    assert reading_of(None) is None
    assert reading_of(InspectedUpload(reading)) is reading
    redactor = Redactor([], InMemoryVault(), build_allowlist(DetectionConfig()))
    # Cleared: the binary part goes out although nothing forwards binaries.
    assert (
        redact_files_upload(
            body,
            b"b",
            redactor,
            require_scanned=True,
            forward_binary=None,
            inspected=InspectedUpload(reading, frozenset({0})),
        )
        is None
    )

    class Foreign:
        def binary_parts(self) -> list[tuple[int, MultipartPart]]:
            return []

    # A reading of another kind is never trusted: the body is read again,
    # the clearance still applying to the same positions.
    assert reading_of(InspectedUpload(Foreign())) is None
    counted: list[int] = []
    redact_files_upload(
        body + b"",
        b"b",
        redactor,
        require_scanned=True,
        forward_binary=counted.append,
        inspected=InspectedUpload(Foreign(), frozenset()),
    )
    assert counted == [1]


def test_the_base_adapter_reads_no_upload() -> None:
    class Bare(ProviderAdapter):
        name = "bare"

        def matches(self, method: str, path: str) -> Any:
            raise NotImplementedError

        def rehydrate_event(self, event: Any, pool: Any) -> Any:
            raise NotImplementedError

        def inject_system_note(self, body: dict[str, Any]) -> dict[str, Any]:
            raise NotImplementedError

    assert Bare().read_multipart("/x", b"", b"b", lambda n: None) is None
