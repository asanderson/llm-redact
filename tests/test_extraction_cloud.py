"""The cloud OCR extraction services — AWS Textract, Google Document AI,
Azure AI Document Intelligence — over ``httpx.MockTransport`` fakes answering
in the services' documented shapes. No network. Each needs ``trusted =
true`` and https; its credential comes only from the environment variables
(or the key file) the config names; any failure is no reading."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from document_fixtures import jpeg, pdf, pdf_objects, pdf_stream, png
from llm_redact import extraction
from llm_redact.config import ConfigError, ExtractionConfig, ExtractionService, parse_extraction
from llm_redact.extraction import (
    GOOGLE_TOKEN_URI,
    ExtractionInspector,
    FileReading,
    _retry_after,
    _same_origin,
    load_google_key,
    media_type,
)

EMAIL = "jane.doe@corp.example"
PNG = png()
AWS_ENV = {"AWS_ACCESS_KEY_ID": "AKIDTEST", "AWS_SECRET_ACCESS_KEY": "aws-secret-value"}
PROCESSOR = "projects/p1/locations/eu/processors/abcdef12"
DI = "https://di.cognitiveservices.azure.com"


def _service(**raw: Any) -> ExtractionService:
    return parse_extraction(
        {"services": [{"trusted": True, **raw}], "request_timeout_seconds": 300}
    ).services[0]


class Recorder:
    """A scripted service: each answer in turn (the last one repeats)."""

    def __init__(self, *answers: httpx.Response) -> None:
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def _inspector(
    service: ExtractionService, handler: Any, environ: dict[str, str], **kwargs: Any
) -> ExtractionInspector:
    config = ExtractionConfig(enabled=True, formats=(), services=(service,))
    return ExtractionInspector(
        config, transport=httpx.MockTransport(handler), environ=environ, **kwargs
    )


# --- configuration ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"kind": "textract", "region": "us-east-1"}, "trusted = true"),
        ({"kind": "textract", "trusted": True}, "needs region"),
        ({"kind": "textract", "trusted": True, "region": "mars-1"}, "needs region"),
        (
            {"kind": "textract", "trusted": True, "region": "us-east-1", "url": "http://127.0.0.1"},
            "https",
        ),
        (
            {"kind": "textract", "trusted": True, "region": "us-east-1", "token_env": "X"},
            "unknown key",
        ),
        (
            {
                "kind": "textract",
                "trusted": True,
                "region": "us-east-1",
                "url": "https://vpce.example/extra",
            },
            "no path",
        ),
        ({"kind": "documentai", "trusted": True, "token_env": "T"}, "needs processor"),
        ({"kind": "documentai", "trusted": True, "processor": PROCESSOR}, "exactly one of"),
        (
            {
                "kind": "documentai",
                "trusted": True,
                "processor": PROCESSOR,
                "token_env": "T",
                "credentials_file": "/k.json",
            },
            "exactly one of",
        ),
        (
            {"kind": "documentai", "trusted": True, "processor": PROCESSOR, "credentials_file": 1},
            "credentials_file must be a file path",
        ),
        ({"kind": "azure_docintel", "trusted": True, "url": DI}, "needs token_env"),
        ({"kind": "azure_docintel", "trusted": True, "token_env": "K"}, "needs url"),
        (
            {
                "kind": "azure_docintel",
                "trusted": True,
                "token_env": "K",
                "url": DI,
                "model": "a b",
            },
            "model must be",
        ),
        (
            {"kind": "tika", "url": "http://127.0.0.1", "region": "us-east-1"},
            "unknown key",
        ),
    ],
)
def test_cloud_services_are_refused_by_setting(raw: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message.replace("(", r"\(")):
        parse_extraction({"services": [raw], "request_timeout_seconds": 300})


def test_cloud_defaults() -> None:
    textract = _service(kind="textract", region="eu-west-1")
    assert textract.url == "https://textract.eu-west-1.amazonaws.com"
    assert textract.formats == ("pdf", "image") and textract.cloud
    assert textract.credential_env() == ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    docai = _service(kind="documentai", processor=PROCESSOR, token_env="GTOKEN")
    assert docai.url == "https://eu-documentai.googleapis.com" and docai.credential_env() == (
        "GTOKEN",
    )
    azure = _service(kind="azure_docintel", url=DI + "/", token_env="DI_KEY")
    assert azure.url == DI and azure.model == "prebuilt-read"
    assert azure.formats == ("pdf", "office", "markup", "image")


# --- AWS Textract ------------------------------------------------------------------------------


def _blocks(*lines: str) -> httpx.Response:
    blocks = [{"BlockType": "PAGE"}]
    for line in lines:
        blocks.append({"BlockType": "LINE", "Text": line})
        blocks.extend({"BlockType": "WORD", "Text": word} for word in line.split())
    return httpx.Response(200, json={"Blocks": blocks, "DocumentMetadata": {"Pages": 1}})


async def test_textract_reads_every_line_signed_with_sigv4() -> None:
    service = Recorder(_blocks("Invoice", f"contact {EMAIL}"))
    environ = {**AWS_ENV, "AWS_SESSION_TOKEN": "session-tok"}
    inspector = _inspector(
        _service(kind="textract", region="eu-west-1", complete=True), service, environ
    )
    reading = await inspector.read(PNG)
    assert reading == FileReading(f"Invoice\ncontact {EMAIL}", True, "textract", reading.display)
    assert reading.display == f"Invoice\ncontact {EMAIL}"
    (request,) = service.requests
    assert str(request.url) == "https://textract.eu-west-1.amazonaws.com/"
    assert request.headers["x-amz-target"] == "Textract.DetectDocumentText"
    assert request.headers["content-type"] == "application/x-amz-json-1.1"
    assert request.headers["x-amz-security-token"] == "session-tok"
    auth = request.headers["authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIDTEST/")
    assert "/eu-west-1/textract/aws4_request" in auth and "aws-secret-value" not in auth
    assert json.loads(request.content) == {"Document": {"Bytes": base64.b64encode(PNG).decode()}}


async def test_textract_is_complete_only_when_declared() -> None:
    inspector = _inspector(
        _service(kind="textract", region="us-east-1"), Recorder(_blocks("x")), AWS_ENV
    )
    assert (await inspector.read(PNG)).complete is False


async def test_textract_without_its_credential_reads_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = Recorder(_blocks("x"))
    inspector = _inspector(
        _service(kind="textract", region="us-east-1"), service, {"AWS_ACCESS_KEY_ID": "AKIDTEST"}
    )
    assert await inspector.read(PNG) == FileReading(None, False, "none")
    assert service.requests == [] and "AWS_SECRET_ACCESS_KEY is not set" in caplog.text
    assert "AKIDTEST" not in caplog.text


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(400, json={"__type": "UnsupportedDocumentException"}),
        httpx.Response(200, json={"Blocks": "x"}),
        httpx.Response(200, json=[1]),
        httpx.Response(307, headers={"location": "https://elsewhere.example/"}),
    ],
)
async def test_a_textract_fault_is_no_reading(answer: httpx.Response) -> None:
    service = Recorder(answer)
    inspector = _inspector(_service(kind="textract", region="us-east-1"), service, AWS_ENV)
    assert await inspector.read(PNG) == FileReading(None, False, "none")
    assert len(service.requests) == 1  # a redirect is never followed
    assert inspector.status()["readings_total"] == {"textract": {"failed": 1}}


# --- Google Document AI ------------------------------------------------------------------------


def _document(text: str, **extra: Any) -> httpx.Response:
    return httpx.Response(200, json={"document": {"text": text, "pages": [{}], **extra}})


async def test_documentai_with_an_access_token() -> None:
    service = Recorder(_document(f"scan {EMAIL}"))
    inspector = _inspector(
        _service(kind="documentai", processor=PROCESSOR, token_env="GTOKEN", complete=True),
        service,
        {"GTOKEN": "ya29.token"},
    )
    reading = await inspector.read(PNG)
    assert reading.complete is True and reading.display == f"scan {EMAIL}"
    (request,) = service.requests
    assert str(request.url) == f"https://eu-documentai.googleapis.com/v1/{PROCESSOR}:process"
    assert request.headers["authorization"] == "Bearer ya29.token"
    body = json.loads(request.content)
    assert body["rawDocument"] == {
        "content": base64.b64encode(PNG).decode(),
        "mimeType": "image/png",
    }
    assert body["skipHumanReview"] is True


async def test_documentai_reporting_an_error_is_incomplete() -> None:
    service = Recorder(_document("part", error={"code": 3}))
    inspector = _inspector(
        _service(kind="documentai", processor=PROCESSOR, token_env="GTOKEN", complete=True),
        service,
        {"GTOKEN": "t"},
    )
    reading = await inspector.read(PNG)
    assert reading.complete is False and reading.text == "part"


async def test_an_expired_documentai_access_token_fails_closed_until_rotated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """docs/extraction.md: ``token_env`` is a static bearer token, never
    refreshed. Once Google answers 401 every call fails closed — no reading
    (the file stays incomplete), counted ``failed`` — the same token is sent
    again, no token endpoint is ever asked, and the log names the service
    and status only, never the token."""
    service = Recorder(httpx.Response(401))
    inspector = _inspector(
        _service(kind="documentai", processor=PROCESSOR, token_env="GTOKEN", complete=True),
        service,
        {"GTOKEN": "ya29.expired"},
    )
    for _ in range(2):
        assert await inspector.read(PNG) == FileReading(None, False, "none")
    assert [r.headers["authorization"] for r in service.requests] == ["Bearer ya29.expired"] * 2
    assert {r.url.host for r in service.requests} == {"eu-documentai.googleapis.com"}
    assert inspector.status()["readings_total"] == {"documentai": {"failed": 2}}
    assert "HTTP 401" in caplog.text and "ya29" not in caplog.text


def _key_file(tmp_path: Path, **overrides: Any) -> tuple[Path, Any]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    info = {
        "type": "service_account",
        "client_email": "reader@p1.iam.gserviceaccount.com",
        "private_key_id": "kid-1",
        "private_key": pem,
        "token_uri": GOOGLE_TOKEN_URI,
        **overrides,
    }
    path = tmp_path / "sa.json"
    path.write_text(json.dumps(info))
    return path, key.public_key()


def _b64(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


async def test_documentai_with_a_service_account_key_file(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    path, public = _key_file(tmp_path)
    requests: list[httpx.Request] = []

    def google(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url == GOOGLE_TOKEN_URI:
            return httpx.Response(
                200, json={"access_token": f"tok{len(requests)}", "expires_in": 3600}
            )
        return _document("scanned")

    now = [1_000_000.0]
    service = _service(kind="documentai", processor=PROCESSOR, credentials_file=str(path))
    inspector = _inspector(service, google, {}, clock=lambda: now[0])
    await inspector.read(PNG)
    await inspector.read(PNG)  # the cached token
    token_call, first, second = requests
    assert first.headers["authorization"] == second.headers["authorization"] == "Bearer tok1"
    form = dict(httpx.QueryParams(token_call.content.decode()))
    assert form["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
    header, claims, signature = form["assertion"].split(".")
    public.verify(
        _b64(signature), f"{header}.{claims}".encode(), padding.PKCS1v15(), hashes.SHA256()
    )
    assert json.loads(_b64(header)) == {"alg": "RS256", "typ": "JWT", "kid": "kid-1"}
    decoded = json.loads(_b64(claims))
    assert decoded["iss"] == "reader@p1.iam.gserviceaccount.com"
    assert decoded["aud"] == GOOGLE_TOKEN_URI and decoded["exp"] - decoded["iat"] == 3600
    # A minute before it expires, a fresh one.
    now[0] += 3600 - 59
    await inspector.read(PNG)
    assert requests[-2].url == GOOGLE_TOKEN_URI
    assert requests[-1].headers["authorization"] == "Bearer tok4"


async def test_a_token_endpoint_without_a_token_is_no_reading(tmp_path: Path) -> None:
    path, _ = _key_file(tmp_path)
    service = Recorder(httpx.Response(200, json={"error": "invalid_grant"}))
    inspector = _inspector(
        _service(kind="documentai", processor=PROCESSOR, credentials_file=str(path)), service, {}
    )
    assert await inspector.read(PNG) == FileReading(None, False, "none")
    assert len(service.requests) == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"type": "authorized_user"}, "not a service-account key file"),
        ({"client_email": ""}, "not a service-account key file"),
        ({"token_uri": "https://evil.example/token"}, "other than"),
        (
            {"private_key": "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n"},
            "no usable RSA",
        ),
    ],
)
def test_a_bad_key_file_refuses_to_start(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    path, _ = _key_file(tmp_path, **overrides)
    with pytest.raises(ConfigError, match=message) as caught:
        load_google_key(str(path))
    assert "PRIVATE KEY" not in str(caught.value)


def test_an_unreadable_key_file_and_a_missing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ConfigError, match="cannot be read as JSON"):
        load_google_key(str(tmp_path / "absent.json"))
    config = parse_extraction(
        {
            "enabled": True,
            "request_timeout_seconds": 90,
            "services": [
                {
                    "kind": "documentai",
                    "trusted": True,
                    "processor": PROCESSOR,
                    "credentials_file": str(tmp_path / "absent.json"),
                }
            ],
        }
    )
    assert any("cannot be read" in p for p in extraction.extraction_problems(config, {}))
    monkeypatch.setattr(extraction.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigError, match=r"llm-redact-proxy\[crypto\]"):
        load_google_key(str(tmp_path / "absent.json"))


def test_the_ec_key_of_a_key_file_is_refused(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    pem = (
        ec.generate_private_key(ec.SECP256R1())
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    path, _ = _key_file(tmp_path, private_key=pem)
    with pytest.raises(ConfigError, match="no usable RSA"):
        load_google_key(str(path))


# --- Azure AI Document Intelligence ----------------------------------------------------------


OPERATION = f"{DI}/documentintelligence/documentModels/prebuilt-read/analyzeResults/op-1"


def _accepted(location: str = OPERATION) -> httpx.Response:
    return httpx.Response(202, headers={"operation-location": location})


def _status(status: str, **extra: Any) -> httpx.Response:
    return httpx.Response(200, json={"status": status, **extra}, headers={"retry-after": "0.1"})


def _azure(handler: Any, sleeps: list[float] | None = None, **raw: Any) -> ExtractionInspector:
    async def sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)
        await asyncio.sleep(0)

    service = _service(kind="azure_docintel", url=DI, token_env="DI_KEY", **raw)
    return _inspector(service, handler, {"DI_KEY": "di-secret-key"}, sleep=sleep)


async def test_document_intelligence_analyzes_then_polls_its_operation() -> None:
    service = Recorder(
        _accepted(),
        _status("notStarted"),
        _status("running"),
        _status("succeeded", analyzeResult={"content": f"page one\n{EMAIL}", "pages": [{}]}),
    )
    sleeps: list[float] = []
    reading = await _azure(service, sleeps, complete=True).read(PNG)
    assert reading.complete is True and reading.text == f"page one\n{EMAIL}"
    analyze, *polls = service.requests
    assert analyze.method == "POST" and analyze.content == PNG
    assert analyze.url.path == "/documentintelligence/documentModels/prebuilt-read:analyze"
    assert analyze.url.params["api-version"] == "2024-11-30"
    assert [str(p.url) for p in polls] == [OPERATION] * 3
    assert {r.headers["ocp-apim-subscription-key"] for r in service.requests} == {"di-secret-key"}
    assert sleeps == [0.5, 0.5]  # Retry-After within the poll bounds


@pytest.mark.parametrize(
    "answers",
    [
        # The operation on another host: the key is never sent there.
        (_accepted("https://evil.example/op"),),
        (_accepted("http://di.cognitiveservices.azure.com/op"),),
        (_accepted("https://user@di.cognitiveservices.azure.com/op"),),
        (httpx.Response(202),),
        (httpx.Response(401),),
        (_accepted(), _status("failed")),
        (_accepted(), httpx.Response(200, json=[])),
        (_accepted(), _status("succeeded", analyzeResult={"pages": []})),
    ],
)
async def test_a_document_intelligence_fault_is_no_reading(
    answers: tuple[httpx.Response, ...],
) -> None:
    service = Recorder(*answers)
    assert await _azure(service).read(PNG) == FileReading(None, False, "none")
    assert all(r.url.host == "di.cognitiveservices.azure.com" for r in service.requests)


async def test_document_intelligence_warnings_are_incomplete() -> None:
    service = Recorder(
        _accepted(),
        _status("succeeded", analyzeResult={"content": "t", "warnings": [{"code": "x"}]}),
    )
    assert (await _azure(service, complete=True).read(PNG)).complete is False


async def test_an_operation_that_never_finishes_ends_at_the_service_timeout() -> None:
    service = Recorder(_accepted(), _status("running"))
    inspector = _azure(service, timeout_seconds=0.2)

    async def slow(seconds: float) -> None:
        await asyncio.sleep(0.02)

    inspector._sleep = slow
    assert await inspector.read(PNG) == FileReading(None, False, "none")
    assert inspector.status()["readings_total"] == {"azure_docintel": {"failed": 1}}


@pytest.mark.parametrize(
    ("value", "seconds"),
    [(None, 1.0), ("0", 0.5), ("2", 2.0), ("60", 5.0), ("soon", 1.0), ("nan", 1.0), ("inf", 1.0)],
)
def test_retry_after(value: str | None, seconds: float) -> None:
    assert _retry_after(value) == seconds


def test_same_origin() -> None:
    assert _same_origin(f"{DI}/x", DI)
    assert not _same_origin("https://di.cognitiveservices.azure.com:444/x", DI)
    assert not _same_origin("https://[::1/x", DI)


def test_media_type_comes_from_the_bytes() -> None:
    assert media_type(PNG, "image") == "image/png"
    assert media_type(b"\xff\xd8\xff\xe0", "image") == "image/jpeg"
    assert media_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image") == "image/webp"
    assert media_type(b"%PDF-1.7", "pdf") == "application/pdf"
    assert media_type(b"\x00\x01", "other") == "application/octet-stream"


async def test_status_never_carries_a_credential() -> None:
    inspector = _azure(Recorder(_accepted()))
    assert "di-secret-key" not in json.dumps(inspector.status())
    assert inspector.status()["services"][0]["host"] == "di.cognitiveservices.azure.com"


# --- page coverage: a cloud reading vouches only for every page ---------------------------------


def _azure_pages(pages: int) -> Recorder:
    """Azure's answer covering ``pages`` pages and reporting no warning —
    the free tier's shape for a longer file (it reads the first two)."""
    analyzed = {"content": "page one\npage two", "pages": [{}] * pages}
    return Recorder(_accepted(), _status("succeeded", analyzeResult=analyzed))


@pytest.mark.parametrize(("analyzed", "complete"), [(2, False), (3, True)])
async def test_a_cloud_reading_of_fewer_pages_than_the_pdf_is_incomplete(
    analyzed: int, complete: bool
) -> None:
    service = _service(kind="azure_docintel", url=DI, token_env="DI_KEY", complete=True)
    config = ExtractionConfig(enabled=True, formats=("pdf",), services=(service,))

    async def sleep(seconds: float) -> None:
        await asyncio.sleep(0)

    inspector = ExtractionInspector(
        config,
        transport=httpx.MockTransport(_azure_pages(analyzed)),
        environ={"DI_KEY": "k"},
        sleep=sleep,
    )
    # Three pages, one only an image: the local reading is incomplete.
    reading = await inspector.read(pdf([None, None], image_page=True))
    assert reading.complete is complete
    assert "page two" in (reading.text or "")  # read either way: still scanned
    counted = inspector.status()["readings_total"]["azure_docintel"]
    assert ("pages_unverified" in counted) is not complete


@pytest.mark.parametrize(
    "data",
    [
        b"II*\x00" + bytes(64),  # a TIFF may hold several pages
        b"GIF89a" + bytes(64),
        b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x08acTL" + bytes(64),  # an animated PNG
        b"%PDF-1.7\n",  # a PDF the local extractors do not read: its pages unknown
    ],
    ids=["tiff", "gif", "apng", "pdf-unread"],
)
async def test_a_cloud_reading_of_a_file_of_unknown_pages_is_incomplete(data: bytes) -> None:
    service = Recorder(
        _accepted(), _status("succeeded", analyzeResult={"content": "t", "pages": [{}]})
    )
    reading = await _azure(service, complete=True).read(data)
    assert reading.complete is False and reading.text == "t"


@pytest.mark.parametrize(
    ("answer", "complete"),
    [
        ({"Blocks": [], "DocumentMetadata": {"Pages": 1}}, True),
        ({"Blocks": [], "DocumentMetadata": {"Pages": 2}}, False),
        ({"Blocks": []}, False),
        ({"Blocks": [], "DocumentMetadata": {"Pages": True}}, False),
    ],
)
async def test_textract_must_report_the_images_one_page(
    answer: dict[str, Any], complete: bool
) -> None:
    service = Recorder(httpx.Response(200, json=answer))
    inspector = _inspector(
        _service(kind="textract", region="us-east-1", complete=True), service, AWS_ENV
    )
    assert (await inspector.read(jpeg())).complete is complete


async def test_documentai_must_report_the_images_one_page() -> None:
    service = Recorder(httpx.Response(200, json={"document": {"text": "t"}}))
    inspector = _inspector(
        _service(kind="documentai", processor=PROCESSOR, token_env="G", complete=True),
        service,
        {"G": "t"},
    )
    assert (await inspector.read(b"BM" + bytes(16))).complete is False


# --- what page OCR cannot vouch for (EXT-1) ----------------------------------------------

SSN = "123-45-6789"


def _textract_inspector(formats: tuple[str, ...] = ("pdf",)) -> tuple[ExtractionInspector, Any]:
    service = Recorder(
        *(
            httpx.Response(
                200,
                json={
                    "Blocks": [{"BlockType": "LINE", "Text": "Quarterly report"}],
                    "DocumentMetadata": {"Pages": 1},
                },
            )
            for _ in range(4)
        )
    )
    config = ExtractionConfig(
        enabled=True,
        formats=formats,
        services=(_service(kind="textract", region="us-east-1", complete=True),),
    )
    inspector = ExtractionInspector(config, transport=httpx.MockTransport(service), environ=AWS_ENV)
    return inspector, service


def _with_attachment(data: bytes) -> bytes:
    import io

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(data)))
    writer.add_attachment("secret.txt", f"employee ssn {SSN}".encode())
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


async def test_an_ocr_reading_completes_only_what_page_ocr_reads() -> None:
    inspector, service = _textract_inspector()
    # A scanned page: the worker left only the page's image unread.
    scanned = await inspector.read(pdf([], image_page=True))
    assert scanned.complete is True and scanned.extractor == "local:pdf+textract"
    # The same page with an attached file (a Factur-X invoice's XML, say):
    # Textract reads the page, never the attachment — not complete.
    attached = await inspector.read(_with_attachment(pdf([], image_page=True)))
    blank = await inspector.read(_with_attachment(pdf(["Quarterly report"])))
    for reading in (attached, blank):
        assert reading.complete is False and reading.text is not None
        assert "Quarterly report" in reading.text  # still scanned
    readings = inspector.status()["readings_total"]["textract"]
    assert readings["unseen_content"] == 2
    assert len(service.requests) == 3


def _scan_pdf(image: bytes, data: bytes, *extra: bytes) -> bytes:
    """A page showing a line of text and drawing one image (its dictionary
    ``image``, its ``data``) — a scan, as a phone or a copier makes it;
    ``extra`` objects follow from 7."""
    return pdf_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
            b" /Resources << /Font << /F1 5 0 R >> /XObject << /Im1 6 0 R >> >> >>",
            pdf_stream(
                b"", b"BT /F1 12 Tf 72 712 Td (Quarterly report) Tj ET q 9 0 0 9 0 0 cm /Im1 Do Q"
            ),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            pdf_stream(
                b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray"
                b" /BitsPerComponent 8 " + image,
                data,
            ),
            *extra,
        ]
    )


_XMP = pdf_stream(b"/Type /Metadata /Subtype /XML", b"<x>ssn " + SSN.encode() + b"</x>")


@pytest.mark.parametrize(
    ("data", "complete"),
    [
        (_scan_pdf(b"/Filter /DCTDecode", jpeg()), True),
        (_scan_pdf(b"/Filter /DCTDecode", jpeg((0xFE, b"owner ssn " + SSN.encode()))), False),
        (_scan_pdf(b"/Filter /DCTDecode", jpeg((0xE1, b"Exif\x00\x00" + SSN.encode()))), False),
        (_scan_pdf(b"/Metadata 7 0 R", b"\x80", _XMP), False),
    ],
    ids=["jpeg", "jpeg-comment", "jpeg-exif", "image-xmp"],
)
async def test_an_ocr_reading_never_completes_a_pdf_whose_image_holds_more(
    data: bytes, complete: bool
) -> None:
    # The PDF twin of the picture rule below: Textract reads the scan's
    # pixels, never its JPEG comment, Exif or the image's XMP.
    inspector, _ = _textract_inspector()
    reading = await inspector.read(data)
    assert reading.complete is complete
    counted = inspector.status()["readings_total"]["textract"]
    assert ("unseen_content" in counted) is not complete


@pytest.mark.parametrize(
    ("data", "complete"),
    [
        (png(), True),
        (jpeg(), True),
        (png((b"tEXt", b"Comment\x00ssn " + SSN.encode())), False),
        (png((b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + SSN.encode())), False),
        (jpeg((0xE1, b"Exif\x00\x00" + SSN.encode())), False),
        (jpeg((0xFE, SSN.encode())), False),
        (jpeg(thumbnail=True), False),
        (png(trailer=SSN.encode()), False),
        (b"BM" + bytes(64), False),  # no metadata walk for BMP: never vouched for
    ],
    ids=[
        "png",
        "jpeg",
        "png-text",
        "png-xmp",
        "jpeg-exif",
        "jpeg-comment",
        "thumb",
        "trail",
        "bmp",
    ],
)
async def test_an_ocr_reading_never_completes_a_picture_with_metadata(
    data: bytes, complete: bool
) -> None:
    inspector, _ = _textract_inspector(formats=())
    reading = await inspector.read(data)
    assert reading.complete is complete
    counted = inspector.status()["readings_total"].get("textract", {})
    assert ("unseen_content" in counted) is not complete
