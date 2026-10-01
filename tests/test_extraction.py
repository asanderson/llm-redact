"""``[extraction]``: the config section, the factory's refusals, the
inspector's local worker (real processes: killed at the deadline, capped in
memory and output) and the self-hosted services (httpx.MockTransport:
Tika, docling-serve, unstructured). No network. The cloud OCR services are
tests/test_extraction_cloud.py; the core's inspection seam end to end is
tests/test_extraction_e2e.py."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import httpx
import pytest

from document_fixtures import docx, package, pdf
from llm_redact import extraction
from llm_redact.config import (
    EXTRACTION_CLASSES,
    ConfigError,
    ExtractionConfig,
    ExtractionService,
    parse_config,
    parse_extraction,
)
from llm_redact.config_write import emit_config_toml, emit_extraction
from llm_redact.extraction import (
    ExtractionInspector,
    FileReading,
    build_inspector,
    declared_class,
    extraction_problems,
    polyglot,
    sniff,
    upload_name,
)
from llm_redact.free_defaults import build_upload_inspector
from llm_redact.plugin_api import UploadPart

EMAIL = "jane.doe@corp.example"
# A generic binary no local extractor reads.
BLOB = b"\x00\x01\x02 binary blob"


# --- the section ------------------------------------------------------------------------


def test_defaults_and_round_trip() -> None:
    assert parse_extraction({}) == ExtractionConfig()
    raw = {
        "enabled": True,
        "formats": ["ooxml", "pdf"],
        "max_file_bytes": 1000,
        "timeout_seconds": 5,
        "request_timeout_seconds": 125.5,
        "max_text_chars": 50,
        "max_inflated_bytes": 4096,
        "worker_memory_mb": 256,
        "max_workers": 3,
        "proxy_credential": True,
        "services": [
            {
                "kind": "tika",
                "url": "https://tika.corp.example/",
                "trusted": True,
                "complete": True,
                "token_env": "TIKA_TOKEN",
                "formats": ["image", "pdf"],
                "timeout_seconds": 90,
            },
            {"kind": "docling", "url": "http://127.0.0.1:5001"},
        ],
    }
    config = parse_extraction(raw)
    assert config.formats == ("pdf", "ooxml")  # canonical order
    assert config.services[0].url == "https://tika.corp.example"
    assert config.services[0].formats == ("pdf", "image")
    assert config.services[1] == ExtractionService("docling", "http://127.0.0.1:5001")
    emitted = emit_extraction(config)
    assert parse_extraction(emitted) == config
    # Through the core's config parser and TOML emitter.
    whole = parse_config({"extraction": raw}, "config.toml")
    assert whole.extraction == config
    assert parse_config(tomllib.loads(emit_config_toml(whole)), "again").extraction == config
    assert emit_extraction(ExtractionConfig()) == {"enabled": False}
    assert "[extraction]" not in emit_config_toml(parse_config({}, "t"))


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "must be a table"),
        ({"nope": 1}, "unknown key"),
        ({"enabled": "yes"}, "enabled must be a boolean"),
        ({"formats": ["pdf", "doc"]}, "unknown ['doc']"),
        ({"formats": "pdf"}, "must be a list of strings"),
        ({"max_file_bytes": 0}, "max_file_bytes must be an integer"),
        ({"max_workers": True}, "max_workers must be an integer"),
        ({"worker_memory_mb": 64}, "worker_memory_mb must be an integer from 128"),
        ({"timeout_seconds": 0}, "timeout_seconds must be a number of seconds"),
        ({"timeout_seconds": float("nan")}, "timeout_seconds must be"),
        ({"timeout_seconds": 301}, "at most 300"),
        ({"timeout_seconds": 30, "request_timeout_seconds": 10}, "at least timeout_seconds"),
        ({"services": {}}, "services must be an array of tables"),
        ({"services": ["tika"]}, "services #1 must be a table"),
        ({"services": [{"kind": "tika", "url": "http://127.0.0.1", "x": 1}]}, "unknown key"),
        ({"services": [{"kind": "ocr", "url": "https://a.example"}]}, "kind must be"),
        ({"convert": "yes"}, "convert must be true, false or a list"),
        ({"convert": ["pdf", "video"]}, "unknown ['video']"),
        ({"services": [{"kind": "tika"}]}, "needs url"),
        ({"services": [{"kind": "tika", "url": "ftp://a.example"}]}, "http(s) URL"),
        ({"services": [{"kind": "tika", "url": "https://u:p@a.example"}]}, "credentials"),
        ({"services": [{"kind": "tika", "url": "https://a.example/?k=1"}]}, "query"),
        ({"services": [{"kind": "tika", "url": "http://tika.corp.example"}]}, "https off"),
        ({"services": [{"kind": "tika", "url": "https://tika.corp.example"}]}, "trusted = true"),
        (
            {"services": [{"kind": "tika", "url": "http://127.0.0.1", "token_env": "no-t"}]},
            "token_env must name",
        ),
        (
            {"services": [{"kind": "tika", "url": "http://localhost", "formats": ["video"]}]},
            "unknown ['video']",
        ),
    ],
)
def test_bad_values_are_refused_by_setting(raw: Any, message: str) -> None:
    with pytest.raises(ConfigError) as caught:
        parse_extraction(raw, "[extraction] in config.toml")
    assert message in str(caught.value)
    assert "u:p" not in str(caught.value)


def test_the_default_timeouts_fit_the_request_deadline() -> None:
    config = parse_extraction({"services": [{"kind": "tika", "url": "http://127.0.0.1:9998"}]})
    local, (service,) = config.timeout_seconds, config.services
    assert local + service.timeout_seconds <= config.request_timeout_seconds
    # Services asked for different classes never run for the same file.
    parse_extraction(
        {
            "services": [
                {"kind": "tika", "url": "http://127.0.0.1:1", "formats": ["pdf"]},
                {"kind": "docling", "url": "http://127.0.0.1:2", "formats": ["image"]},
            ]
        }
    )


@pytest.mark.parametrize(
    ("raw", "coarse", "needed"),
    [
        # A service's own timeout past what the request leaves it.
        (
            {
                "timeout_seconds": 5,
                "request_timeout_seconds": 10,
                "services": [{"kind": "tika", "url": "http://127.0.0.1", "timeout_seconds": 300}],
            },
            "pdf",
            "305",
        ),
        # Two services asked for the same class run one after another.
        (
            {
                "services": [
                    {"kind": "tika", "url": "http://127.0.0.1:1"},
                    {"kind": "docling", "url": "http://127.0.0.1:2"},
                ]
            },
            "pdf",
            "90",
        ),
        # A class no local extractor reads counts the services only.
        (
            {
                "formats": [],
                "timeout_seconds": 10,
                "request_timeout_seconds": 20,
                "services": [{"kind": "tika", "url": "http://127.0.0.1", "formats": ["image"]}],
            },
            "image",
            "30",
        ),
    ],
)
def test_service_timeouts_must_fit_the_request_deadline(
    raw: dict[str, Any], coarse: str, needed: str
) -> None:
    # The core cancels a request's inspections at request_timeout_seconds:
    # a service it would always cut off first is refused at startup.
    with pytest.raises(ConfigError) as caught:
        parse_extraction(raw, "[extraction] in config.toml")
    message = str(caught.value)
    assert f"shorter than one {coarse} file may take" in message
    assert f"add up to {needed};" in message


def test_the_deadline_check_is_per_file_and_documented_so() -> None:
    # One file's worst case fits (0.5 + 1.5 <= 2); five slow files of one
    # request do not, four at a time — the fifth is `timeout`, never sent on
    # a reading that did not happen. docs/extraction.md says the check is
    # per file rather than promising every request's files fit.
    config = parse_extraction(
        {
            "timeout_seconds": 0.5,
            "request_timeout_seconds": 2,
            "services": [{"kind": "tika", "url": "http://127.0.0.1", "timeout_seconds": 1.5}],
        }
    )
    assert config.request_timeout_seconds == 2
    docs = (Path(__file__).parent.parent / "docs" / "extraction.md").read_text()
    assert "That check is per file." in docs and "four at a time" in docs


def test_loopback_services_need_no_trust_declaration() -> None:
    for url in ("http://127.0.0.1:9998", "http://[::1]:9998", "http://localhost:9998"):
        service = parse_extraction({"services": [{"kind": "tika", "url": url}]}).services[0]
        assert service.loopback and not service.trusted


def test_convert_takes_a_bool_or_classes() -> None:
    assert parse_extraction({}).convert == ()
    assert parse_extraction({"convert": False}).convert == ()
    assert parse_extraction({"convert": True}).convert == EXTRACTION_CLASSES
    assert parse_extraction({"convert": ["office", "pdf"]}).convert == ("pdf", "office")
    assert emit_extraction(parse_extraction({"convert": True}))["convert"] is True


def test_the_section_is_the_cores_and_restart_only() -> None:
    from llm_redact.config import CORE_SECTION_KEYS, RESTART_ONLY_KEYS

    assert "extraction" in CORE_SECTION_KEYS and "extraction" in RESTART_ONLY_KEYS


def test_the_worker_formats_match_the_config() -> None:
    from llm_redact.config import EXTRACTION_FORMATS
    from llm_redact.extract_worker import FORMATS

    assert FORMATS == EXTRACTION_FORMATS


# --- the factory ----------------------------------------------------------------------------


def _config(**raw: Any) -> Any:
    return parse_config({"extraction": raw}, "config.toml")


def test_the_factory_builds_only_when_enabled_on_every_tier() -> None:
    assert build_upload_inspector(parse_config({}, "t"), "free") is None
    assert build_upload_inspector(_config(enabled=False), "pro") is None
    for tier in ("free", "pro"):  # the core gates nothing
        inspector = build_upload_inspector(_config(enabled=True, max_file_bytes=77), tier)
        assert isinstance(inspector, ExtractionInspector)
        assert inspector.max_bytes == 77 and inspector.timeout == 60.0


def test_the_factory_refuses_what_cannot_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TIKA_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="TIKA_TOKEN is not set"):
        build_upload_inspector(
            _config(
                enabled=True,
                services=[{"kind": "tika", "url": "http://127.0.0.1", "token_env": "TIKA_TOKEN"}],
            ),
            "pro",
        )
    with pytest.raises(ConfigError, match="nothing reads"):
        build_inspector(_config(enabled=True, formats=[]).extraction)
    monkeypatch.setattr(extraction.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigError, match=r"llm-redact-proxy\[extract\]"):
        build_upload_inspector(_config(enabled=True), "free")
    assert extraction_problems(ExtractionConfig(formats=("ooxml",)), {}) == []


def test_an_enabled_section_needs_an_inspector(monkeypatch: pytest.MonkeyPatch) -> None:
    # A plugin factory predating the core's [extraction] builds none: the
    # proxy refuses to start rather than silently read nothing.
    from llm_redact.proxy import ProxyState
    from llm_redact.registry import Registry

    registry = Registry()
    registry.build_upload_inspector = lambda config, tier: None
    monkeypatch.setattr("llm_redact.proxy.get_registry", lambda: registry)
    with pytest.raises(ConfigError, match="leaves \\[extraction\\] to the core"):
        ProxyState(_config(enabled=True), None)


# --- sniffing ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "coarse", "name"),
    [
        (pdf(["x"]), "pdf", "upload.pdf"),
        (docx(["x"]), "office", "upload.docx"),
        (package({"xl/workbook.xml": b"<w/>"}), "office", "upload.xlsx"),
        (package({"notes.txt": b"x"}), "office", "upload.bin"),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1 legacy", "office", "upload.bin"),
        (b"<html>\xe9</html>", "markup", "upload.html"),
        (b"{\\rtf1 x}", "rtf", "upload.rtf"),
        (b"\x89PNG\r\n\x1a\n", "image", "upload.bin"),
        (b"\x00\x00\x00\x1cftypheic", "image", "upload.bin"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image", "upload.bin"),
        (BLOB, "other", "upload.bin"),
    ],
)
def test_sniff_and_upload_name(data: bytes, coarse: str, name: str) -> None:
    assert sniff(data) == coarse
    assert upload_name(data, coarse) == name


# --- the local worker (real processes) -----------------------------------------------------


def _inspector(**config: Any) -> ExtractionInspector:
    return ExtractionInspector(ExtractionConfig(enabled=True, **config))


async def test_the_worker_reads_a_text_pdf_completely() -> None:
    inspector = _inspector()
    reading = await inspector.read(pdf([f"mail {EMAIL}"]))
    assert reading.complete is True and reading.extractor == "local:pdf"
    assert reading.text is not None and EMAIL in reading.text
    assert inspector.status()["readings_total"] == {"local:pdf": {"complete": 1}}
    assert inspector.running == 0 and inspector.killed == 0


async def test_an_image_only_pdf_is_incomplete() -> None:
    reading = await _inspector().read(pdf([None], image_page=True))
    assert reading == FileReading("", False, "local:pdf")
    assert reading.display is None


async def test_a_pdf_that_keeps_the_parser_busy_is_killed_at_the_deadline() -> None:
    # Millions of text operators: pypdf would spend far longer than allowed.
    content = b"BT /F1 12 Tf 72 712 Td " + b"(a) Tj " * 2_000_000 + b"ET"
    inspector = _inspector(timeout_seconds=0.5)
    reading = await inspector.read(pdf(["x"], raw_content=content, compress=True))
    assert reading == FileReading(None, False, "none")
    assert inspector.killed == 1 and inspector.running == 0
    assert inspector.status()["readings_total"] == {"local": {"timeout": 1}}


async def test_a_worker_over_its_memory_limit_fails_closed() -> None:
    # 200 MiB through a worker allowed 128: it dies of its own memory
    # limit, and nothing is read.
    inspector = _inspector(worker_memory_mb=128, max_file_bytes=1 << 30)
    reading = await inspector.read(pdf(["x"]) + b"\n" * (200 << 20))
    assert reading.complete is False and reading.text is None
    # Linux kills the worker at its limit; macOS ignores RLIMIT_AS, so the
    # file is never handed over (larger than a quarter of the limit).
    outcome = "failed" if extraction.MEMORY_LIMIT_ENFORCED else "memory_unenforced"
    assert inspector.status()["readings_total"] == {"local": {outcome: 1}}


async def test_without_an_enforced_memory_limit_a_large_file_is_not_handed_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Where the kernel ignores the address-space limit (macOS), a file over
    # a quarter of worker_memory_mb never reaches the worker: nothing read.
    monkeypatch.setattr(extraction, "MEMORY_LIMIT_ENFORCED", False)
    inspector = _inspector(worker_memory_mb=256, max_file_bytes=1 << 30)
    big = pdf(["x"]) + b"\n" * (65 << 20)
    assert await inspector.read(big) == FileReading(None, False, "none")
    assert inspector.status()["readings_total"] == {"local": {"memory_unenforced": 1}}
    # A file within the bound is still read.
    small = await inspector.read(pdf(["x"]))
    assert small.complete is True and small.text == "x"


async def test_a_zip_bomb_is_refused_by_the_worker() -> None:
    from document_fixtures import zip_bomb

    inspector = _inspector()
    assert await inspector.read(zip_bomb(8 << 20)) == FileReading(None, False, "none")
    assert inspector.status()["readings_total"] == {"local:ooxml": {"limit": 1}}


async def test_output_over_the_cap_kills_the_worker() -> None:
    inspector = _inspector(max_text_chars=10)
    inspector._command = (
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('x' * 100000); sys.stdout.flush()",
    )
    assert await inspector.read(pdf(["x"])) == FileReading(None, False, "none")
    assert inspector.status()["readings_total"] == {"local": {"failed": 1}}


@pytest.mark.parametrize(
    "command",
    [
        ("/nonexistent/python",),
        (sys.executable, "-c", "print('not json')"),
        (sys.executable, "-c", "print('[1, 2]')"),
    ],
)
async def test_a_worker_that_gives_no_reading_fails_closed(command: tuple[str, ...]) -> None:
    inspector = _inspector()
    inspector._command = command
    assert await inspector.read(pdf(["x"])) == FileReading(None, False, "none")
    assert inspector.status()["readings_total"] == {"local": {"failed": 1}}


async def test_an_unexpected_worker_fault_is_no_reading(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("secret-looking detail")

    monkeypatch.setattr(extraction.asyncio, "create_subprocess_exec", broken)
    inspector = _inspector()
    assert await inspector.read(pdf(["x"])) == FileReading(None, False, "none")
    assert "ValueError" in caplog.text and "secret-looking" not in caplog.text


async def test_a_worker_that_stops_reading_its_input() -> None:
    inspector = _inspector()
    inspector._command = (
        sys.executable,
        "-c",
        "import sys, json; sys.stdin.close(); print(json.dumps({'format': 'pdf',"
        " 'text': 'partial', 'complete': False, 'reason': 'ok'}))",
    )
    reading = await inspector.read(pdf(["x"]) + b"\n" * (4 << 20))
    assert reading == FileReading("partial", False, "local:pdf")


async def test_the_worker_inherits_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_REDACT_TEST_SECRET_CANARY", "s3cret")
    inspector = _inspector()
    inspector._command = (
        sys.executable,
        "-c",
        "import os, json; print(json.dumps({'format': 'pdf', 'text': json.dumps(dict(os.environ)),"
        " 'complete': True, 'reason': 'ok'}))",
    )
    reading = await inspector.read(pdf(["x"]))
    assert reading.text is not None and "s3cret" not in reading.text


async def test_unsupported_formats_skip_the_worker() -> None:
    inspector = _inspector(formats=("pdf",))
    assert await inspector.read(docx(["x"])) == FileReading(None, False, "none")
    assert inspector.status()["readings_total"] == {}


async def test_workers_are_bounded_per_event_loop() -> None:
    import asyncio

    inspector = _inspector(max_workers=1)
    readings = await asyncio.gather(*(inspector.read(pdf([f"p{n}"])) for n in range(3)))
    assert [r.complete for r in readings] == [True, True, True]
    gate = inspector._gate()
    assert inspector._gate() is gate


# --- services (MockTransport) ------------------------------------------------------------------


class Service:
    def __init__(self, answer: Any, status: int = 200) -> None:
        self.answer = answer
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = self.answer if isinstance(self.answer, bytes) else json.dumps(self.answer).encode()
        return httpx.Response(self.status, content=body)


def _with_service(
    service: Service, *, kind: str = "tika", complete: bool = True, **extra: Any
) -> ExtractionInspector:
    config = ExtractionConfig(
        enabled=True,
        formats=extra.pop("formats", ()),
        services=(ExtractionService(kind, "http://127.0.0.1:9998", complete=complete, **extra),),
    )
    return ExtractionInspector(
        config, transport=httpx.MockTransport(service), environ={"TOKEN": "tok-123"}
    )


async def test_tika_reads_every_document_recursively() -> None:
    service = Service(
        [
            {"X-TIKA:content": f"scan text {EMAIL}", "dc:creator": ["Ada", "Grace"]},
            {"X-TIKA:content": "embedded", "X-TIKA:embedded_depth": "1"},
        ]
    )
    inspector = _with_service(service, token_env="TOKEN")
    reading = await inspector.read(BLOB)
    assert reading.complete is True and reading.extractor == "tika"
    assert reading.text is not None and EMAIL in reading.text and "Grace" in reading.text
    (request,) = service.requests
    assert request.method == "PUT" and request.url.path == "/rmeta/text"
    assert request.content == BLOB
    assert request.headers["authorization"] == "Bearer tok-123"
    await inspector.aclose()
    await inspector.aclose()


async def test_tika_reporting_an_exception_is_incomplete() -> None:
    service = Service([{"X-TIKA:content": "part", "X-TIKA:EXCEPTION:embedded_exception": "x"}])
    reading = await _with_service(service).read(BLOB)
    assert reading == FileReading("part", False, "tika")


async def test_a_service_is_complete_only_when_declared() -> None:
    reading = await _with_service(Service([{"X-TIKA:content": "t"}]), complete=False).read(BLOB)
    assert reading.complete is False


async def test_docling() -> None:
    service = Service(
        {"document": {"text_content": f"converted {EMAIL}"}, "status": "success", "errors": []}
    )
    inspector = _with_service(service, kind="docling", token_env="TOKEN")
    reading = await inspector.read(pdf(["x"]))
    assert reading.complete is True and reading.extractor == "docling"
    (request,) = service.requests
    assert request.url.path == "/v1/convert/file"
    assert request.headers["x-api-key"] == "tok-123"
    assert b'filename="upload.pdf"' in request.content and b"to_formats" in request.content
    partial = Service(
        {"document": {"text_content": "t"}, "status": "partial_success", "errors": ["page 2"]}
    )
    assert (await _with_service(partial, kind="docling").read(BLOB)).complete is False


async def test_unstructured() -> None:
    service = Service(
        [
            {"type": "Title", "text": "Report"},
            {"type": "NarrativeText", "text": EMAIL, "metadata": {"link_urls": ["https://x"]}},
        ]
    )
    inspector = _with_service(service, kind="unstructured", token_env="TOKEN")
    reading = await inspector.read(BLOB)
    assert reading.text is not None and "https://x" in reading.text and EMAIL in reading.text
    assert service.requests[0].headers["unstructured-api-key"] == "tok-123"
    assert service.requests[0].url.path == "/general/v0/general"


@pytest.mark.parametrize(
    ("kind", "service"),
    [
        ("tika", Service({"not": "a list"})),
        ("tika", Service([1, 2])),
        ("tika", Service(b"not json")),
        ("tika", Service([], status=500)),
        ("tika", Service([], status=302)),
        ("docling", Service({"document": {}})),
        ("docling", Service([])),
        ("unstructured", Service({"x": 1})),
    ],
)
async def test_a_failing_service_is_no_reading(
    kind: str, service: Service, caplog: pytest.LogCaptureFixture
) -> None:
    inspector = _with_service(service, kind=kind)
    assert await inspector.read(BLOB) == FileReading(None, False, "none")
    assert inspector.status()["readings_total"] == {kind: {"failed": 1}}
    assert "127.0.0.1" in caplog.text and "binary blob" not in caplog.text


async def test_an_oversized_answer_is_no_reading() -> None:
    service = Service([{"X-TIKA:content": "x" * (3 << 20)}])
    inspector = _with_service(service)
    inspector.config = ExtractionConfig(
        enabled=True, formats=(), services=inspector.config.services, max_text_chars=10
    )
    assert await inspector.read(BLOB) == FileReading(None, False, "none")


async def test_a_missing_credential_at_call_time_is_no_reading(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = Service([{"X-TIKA:content": "t"}])
    inspector = _with_service(service, token_env="GONE")
    assert await inspector.read(BLOB) == FileReading(None, False, "none")
    assert service.requests == [] and "GONE is not set" in caplog.text


async def test_services_are_asked_only_for_their_formats_and_in_order() -> None:
    first = Service([{"X-TIKA:content": "first"}])
    second = Service({"document": {"text_content": "second"}, "status": "success"})
    config = ExtractionConfig(
        enabled=True,
        formats=(),
        services=(
            ExtractionService("tika", "http://127.0.0.1:1", formats=("image",)),
            ExtractionService("tika", "http://127.0.0.1:2", complete=False),
            ExtractionService("docling", "http://127.0.0.1:3", complete=True),
            ExtractionService("tika", "http://127.0.0.1:4", complete=True),
        ),
    )

    def route(request: httpx.Request) -> httpx.Response:
        return (first if request.url.port in (1, 2, 4) else second)(request)

    inspector = ExtractionInspector(config, transport=httpx.MockTransport(route))
    reading = await inspector.read(BLOB)
    assert reading == FileReading("first\nsecond", True, "tika+docling", "second")
    assert [r.url.port for r in first.requests + second.requests] == [2, 3]


async def test_a_complete_local_reading_skips_the_services() -> None:
    service = Service([{"X-TIKA:content": "never"}])
    inspector = _with_service(service, formats=("pdf",))
    reading = await inspector.read(pdf(["local"]))
    assert reading.extractor == "local:pdf" and service.requests == []
    # An incomplete one asks them, and both texts are scanned.
    reading = await inspector.read(pdf(["local"], image_page=True))
    assert reading.extractor == "local:pdf+tika" and reading.complete is True
    assert reading.text is not None and "never" in reading.text


_PNG = b"\x89PNG\r\n\x1a\n" + bytes(64)
_OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(64)
_SECRET_DOCX = docx([f"contact {EMAIL}"])


@pytest.mark.parametrize(
    ("data", "formats"),
    [
        # A clean PDF with a Word file appended: the local PDF reading is not
        # vouched for, and Tika — which opens it by its leading bytes — reads
        # the PDF completely; python-docx would open the Word file.
        (pdf(["quarterly figures only"]) + _SECRET_DOCX, ("pdf",)),
        # Classes no local extractor reads: an image or a legacy Office file
        # carrying a zip, an image carrying a PDF header.
        (_PNG + _SECRET_DOCX, ()),
        (_OLE2 + _SECRET_DOCX, ()),
        (_PNG + pdf(["x"]), ()),
    ],
    ids=["pdf-docx", "image-docx", "ole2-docx", "image-pdf"],
)
async def test_a_service_does_not_vouch_for_a_polyglot(
    data: bytes, formats: tuple[str, ...]
) -> None:
    service = Service([{"X-TIKA:content": "quarterly figures only"}])
    inspector = _with_service(service, formats=formats)
    reading = await inspector.read(data)
    assert reading.complete is False and len(service.requests) == 1
    assert reading.text is not None and "quarterly figures only" in reading.text
    assert polyglot(data) is True
    # The file it carries alone is vouched for.
    carrier = data[: len(data) - len(_SECRET_DOCX)] if data.endswith(_SECRET_DOCX) else _PNG
    assert (await inspector.read(carrier)).complete is True
    assert polyglot(carrier) is False


async def test_status_names_services_by_host_only() -> None:
    inspector = _with_service(Service([]), token_env="TOKEN")
    status = inspector.status()
    assert status["services"] == [
        {"kind": "tika", "host": "127.0.0.1", "trusted": False, "complete": True}
    ]
    assert "tok-123" not in json.dumps(status)
    assert EXTRACTION_CLASSES == ("pdf", "office", "markup", "rtf", "image", "other")


@pytest.mark.parametrize(
    ("kind", "answer"),
    [
        ("tika", [{"X-TIKA:content": "t"}]),
        ("docling", {"document": {"text_content": "t"}, "status": "success"}),
        ("unstructured", [{"text": "t"}]),
    ],
)
async def test_a_service_request_carries_the_service_timeout(kind: str, answer: Any) -> None:
    # httpx's own default (5 s per read) once cut every slower service
    # (OCR) short of its configured timeout_seconds.
    service = Service(answer)
    inspector = _with_service(service, kind=kind, timeout_seconds=42.0)
    assert (await inspector.read(BLOB)).text == "t"
    (request,) = service.requests
    phases = ("connect", "read", "write", "pool")
    assert request.extensions["timeout"] == dict.fromkeys(phases, 42.0)


@pytest.mark.parametrize(
    ("identity", "proxy_credential", "asked"),
    [(False, False, True), (True, False, False), (True, True, True), (False, True, True)],
)
async def test_services_see_a_file_only_when_its_reading_can_clear_it(
    identity: bool, proxy_credential: bool, asked: bool
) -> None:
    service = Service([{"X-TIKA:content": "service text"}])
    config = ExtractionConfig(
        enabled=True,
        proxy_credential=proxy_credential,
        services=(ExtractionService("tika", "http://127.0.0.1:9998", complete=True),),
    )
    inspector = ExtractionInspector(config, transport=httpx.MockTransport(service))
    scan = pdf([f"cover {EMAIL}"], image_page=True)  # incomplete locally
    inspection = await inspector.inspect(UploadPart(scan, "application/pdf", "openai", identity))
    assert bool(service.requests) is asked
    # The local reading still names what the file holds (the refusal's types).
    assert inspection.text is not None and EMAIL in inspection.text
    assert inspection.complete is asked


@pytest.mark.parametrize(
    ("declared", "claimed"),
    [
        (None, None),
        ("application/octet-stream", None),
        ("application/pdf", "pdf"),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "office"),
        ("application/vnd.oasis.opendocument.text", "office"),
        ("application/vnd.ms-excel.sheet.macroenabled.12", "office"),
        ("application/msword", "office"),
        ("text/html", "markup"),
        ("image/svg+xml", "markup"),
        ("text/rtf", "rtf"),
        ("image/png", "image"),
        ("application/zip", "other"),
        ("text/plain", "other"),
    ],
)
def test_declared_class(declared: str | None, claimed: str | None) -> None:
    assert declared_class(declared) == claimed


@pytest.mark.parametrize(
    ("declared", "complete"),
    [
        ("application/pdf", True),
        (None, True),
        ("application/octet-stream", True),
        # Declared as a Word file: the provider may open it with a docx
        # reader (a PDF with a docx inside), or as a zip, or as text.
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", False),
        ("application/zip", False),
        ("text/plain", False),
    ],
)
async def test_a_file_declared_as_another_format_is_not_vouched_for(
    declared: str | None, complete: bool
) -> None:
    inspection = await _inspector().inspect(
        UploadPart(pdf([f"mail {EMAIL}"]), declared, "openai", False)
    )
    assert inspection.complete is complete
    assert inspection.text is not None and EMAIL in inspection.text  # still scanned
