"""Reading binary uploads as text: the built-in upload inspector (docs/extraction.md).

``[extraction] enabled = true`` makes ``Registry.build_upload_inspector``
(the ``plugin_api.UploadInspector`` seam; ``free_defaults``) return an
``ExtractionInspector``. The core hands it each BINARY file part of an
upload (a PDF, an Office document, an image) before redaction; this module
only READS it — ``upload_inspection`` scans the text with the live
detectors and decides: a value found refuses the upload (or, in convert
mode, sends the redacted text instead of the file), a COMPLETE clean
reading lets the file go out byte-identical (under a credential the proxy
holds only with ``[extraction] proxy_credential = true``), anything else
keeps the core's unscanned-binary rules.

Two ways to read a file, in this order:

- LOCAL extractors (``extract_worker.py``: PDF text layers through pypdf,
  OOXML/ODF packages, markup, RTF) in a fresh, resource-limited process per
  file, killed at ``timeout_seconds`` — a parser wedged by a hostile file
  costs one process, never the proxy's event loop or memory;
- SERVICES (``[[extraction.services]]``) for what the local extractors
  cannot read completely (a scan needing OCR, a legacy Office file, an
  image): self-hosted Apache Tika, docling-serve or unstructured, or a
  cloud OCR service — AWS Textract, Google Document AI, Azure AI Document
  Intelligence — over HTTPS with bounded time and answer size. A service
  that sees files off this machine must be declared ``trusted`` (a cloud
  one always); a credential is only ever read from the environment
  variables the config NAMES (or, for Document AI, a service-account key
  file it names); a service's reading counts as complete only when the
  operator declares ``complete = true`` and the service reports no error.

A file that also holds another format than its leading bytes show (a
polyglot) is never vouched for, whoever read it: the worker checks its own
readings, and a service's complete reading is checked here. Nor is a file
whose declared media type or file-name extension (``UploadPart``) names
another class than its bytes: the provider may open it as that one.

Any failure — a killed worker, an unparsable answer, a service fault — is a
reading that did not happen (fail closed: the core's unscanned rules). Logs
name the extractor, the service host and a fixed failure kind or the
exception TYPE only: never a file name, content, a token or a service's
answer.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib.util
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from llm_redact.config import (
    EXTRACTION_LOCAL_READERS,
    ConfigError,
    ExtractionConfig,
    ExtractionService,
)
from llm_redact.plugin_api import Inspection, UploadPart
from llm_redact.sigv4 import AwsCredentials, sign_request

logger = logging.getLogger("llm_redact")

INSTALL_HINT = 'pip install "llm-redact-proxy[extract]"'
CRYPTO_HINT = 'pip install "llm-redact-proxy[crypto]"'
# The worker's environment: nothing inherited (no credential of the proxy's
# reaches the process that parses hostile files); isolated mode (-I)
# ignores PYTHON* variables anyway.
_WORKER_ENV = {"PATH": os.defpath}
_WORKER = (sys.executable, "-I", "-m", "llm_redact.extract_worker")
_IMAGE_SIGNATURES = (
    b"\x89PNG",
    b"\xff\xd8\xff",
    b"GIF8",
    b"II*\x00",
    b"MM\x00*",
    b"BM",
)
_OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
# Zip packages by the part that names their kind (a file name for services
# that detect formats by extension; the worker decides by the package).
_ZIP_HINTS = (
    (b"word/document.xml", "docx"),
    (b"xl/workbook.xml", "xlsx"),
    (b"ppt/presentation.xml", "pptx"),
    (b"application/vnd.oasis.opendocument.text", "odt"),
    (b"application/vnd.oasis.opendocument.spreadsheet", "ods"),
    (b"application/vnd.oasis.opendocument.presentation", "odp"),
)
_AUTH_HEADERS = {
    "tika": "authorization",
    "docling": "x-api-key",
    "unstructured": "unstructured-api-key",
    "documentai": "authorization",
    "azure_docintel": "ocp-apim-subscription-key",
}
# Worst-case JSON escaping of the worker's text (\\uXXXX per character).
_JSON_BYTES_PER_CHAR = 6
# Document Intelligence: the API version the analyze call names, and how
# long to wait between polls of its operation (a Retry-After within these).
DOCINTEL_API_VERSION = "2024-11-30"
_POLL_MIN_SECONDS = 0.5
_POLL_MAX_SECONDS = 5.0
# Google service-account tokens: the one token endpoint a key file may
# name, the scope asked for, and how early a cached token is renewed.
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
_GOOGLE_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_TOKEN_EARLY_SECONDS = 60.0
_JWT_LIFETIME = 3600


class _Failed(Exception):
    """A reading that did not happen (a fixed kind, never content)."""


@dataclass(frozen=True)
class FileReading:
    """What the extractors made of one file: its text (None: nothing
    read), whether EVERY text-bearing element was read, by whom, and its
    DISPLAY text (the file's text once, as a reader sees it — what convert
    mode sends; None unless the reading is complete and has one)."""

    text: str | None = field(repr=False)
    complete: bool
    extractor: str
    display: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class _Read:
    """One reader's answer: its text, whether it counts as complete, its
    display text (the document's text once) and how many pages it covers
    (the worker: the PDF's page count; a cloud OCR service: the pages it
    analyzed; None when not known)."""

    text: str = field(repr=False)
    complete: bool
    display: str | None = field(default=None, repr=False)
    pages: int | None = None


def polyglot(data: bytes) -> bool:
    """Whether ``data`` also holds a file of another format than its
    leading bytes show (``extract_worker.carries_another``) — for every
    class, those the local extractors do not read included."""
    from llm_redact.extract_worker import carries_another, detect

    return carries_another(data, detect(data) or "other")


def sniff(data: bytes) -> str:
    """The coarse class of a file by its bytes, as a service's ``formats``
    names it: pdf, office (a zip package or a legacy OLE2 file), markup,
    rtf, image or other."""
    from llm_redact.extract_worker import detect

    kind = detect(data)
    if kind == "zip" or data.startswith(_OLE2):
        return "office"
    if kind == "html":
        return "markup"
    if kind in ("pdf", "rtf"):
        return kind
    if data.startswith(_IMAGE_SIGNATURES) or data[4:12] in (b"ftypheic", b"ftypavif"):
        return "image"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image"
    return "other"


# Declared media types by the class their readers open (``sniff``); a
# type naming no particular format (``_GENERIC``) is no claim at all, and
# any other is a format none of these classes is.
_DECLARED = {
    "application/pdf": "pdf",
    "application/x-pdf": "pdf",
    "application/rtf": "rtf",
    "application/x-rtf": "rtf",
    "text/rtf": "rtf",
    "text/html": "markup",
    "application/xhtml+xml": "markup",
    "text/xml": "markup",
    "application/xml": "markup",
    "image/svg+xml": "markup",
    "application/msword": "office",
    "application/vnd.ms-excel": "office",
    "application/vnd.ms-powerpoint": "office",
}
_DECLARED_PREFIXES = (
    ("application/vnd.openxmlformats-officedocument.", "office"),
    ("application/vnd.oasis.opendocument.", "office"),
    ("application/vnd.ms-word.", "office"),
    ("application/vnd.ms-excel.", "office"),
    ("application/vnd.ms-powerpoint.", "office"),
    ("image/", "image"),
)
_GENERIC = frozenset(
    {
        "application/octet-stream",
        "binary/octet-stream",
        "application/binary",
        "application/x-binary",
        "application/unknown",
        "application/download",
        "application/x-download",
        "application/force-download",
    }
)
# File-name extensions by the class their readers open; an extension naming
# no particular format (``_GENERIC_EXTENSIONS``) is no claim, any other is a
# format none of these classes is (``other``).
_OFFICE_EXTENSIONS = [
    "docx",
    "docm",
    "dotx",
    "dotm",
    "xlsx",
    "xlsm",
    "xltx",
    "xltm",
    "pptx",
    "pptm",
    "potx",
    "potm",
    "ppsx",
    "ppsm",
    "odt",
    "ods",
    "odp",
    "odg",
    "ott",
    "ots",
    "otp",
    "doc",
    "dot",
    "xls",
    "xlt",
    "ppt",
    "pot",
    "pps",
]
_IMAGE_EXTENSIONS = [
    "png",
    "jpg",
    "jpeg",
    "jpe",
    "gif",
    "tif",
    "tiff",
    "bmp",
    "webp",
    "heic",
    "heif",
    "avif",
]
_EXTENSIONS = {
    "pdf": "pdf",
    "rtf": "rtf",
    **dict.fromkeys(("html", "htm", "xhtml", "xht", "xml", "svg"), "markup"),
    **dict.fromkeys(_OFFICE_EXTENSIONS, "office"),
    **dict.fromkeys(_IMAGE_EXTENSIONS, "image"),
}
_GENERIC_EXTENSIONS = frozenset({"bin", "dat", "tmp"})
# The media type a service is told a file of each image signature is.
_IMAGE_TYPES = (
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"BM", "image/bmp"),
)


def declared_class(content_type: str | None) -> str | None:
    """The class a part's DECLARED media type claims (as ``sniff`` names
    classes; ``other`` for a specific type of none of them), or None when
    it claims nothing (absent or generic)."""
    if content_type is None or content_type in _GENERIC:
        return None
    if content_type in _DECLARED:
        return _DECLARED[content_type]
    return next(
        (coarse for prefix, coarse in _DECLARED_PREFIXES if content_type.startswith(prefix)),
        "other",
    )


def extension_class(extension: str | None) -> str | None:
    """The class a part's file-name EXTENSION claims (``UploadPart.extension``):
    None when it claims nothing (no extension, a generic one); ``other``
    for an extension of none of the classes — and for names that disagree
    (``""``): the provider may go by either."""
    if extension is None or extension in _GENERIC_EXTENSIONS:
        return None
    return _EXTENSIONS.get(extension, "other")


def upload_name(data: bytes, coarse: str) -> str:
    """A file name for services that pick a converter by extension —
    derived from the bytes, never the client's name."""
    if coarse == "pdf":
        return "upload.pdf"
    if coarse == "office" and not data.startswith(_OLE2):
        for marker, extension in _ZIP_HINTS:
            if marker in data:
                return f"upload.{extension}"
    return {"markup": "upload.html", "rtf": "upload.rtf"}.get(coarse, "upload.bin")


def media_type(data: bytes, coarse: str) -> str:
    """The media type a service is told the file is — derived from the
    bytes (a PDF, an image by its signature), never the client's claim."""
    if coarse == "pdf":
        return "application/pdf"
    for signature, kind in _IMAGE_TYPES:
        if data.startswith(signature):
            return kind
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _local_class(coarse: str, formats: tuple[str, ...]) -> bool:
    """Whether a local extractor reads this class."""
    return any(f in formats for f in EXTRACTION_LOCAL_READERS.get(coarse, ()))


class ExtractionInspector:
    """The ``plugin_api.UploadInspector`` (see the module). ``transport``,
    ``environ``, ``clock`` and ``sleep`` are injectable for tests."""

    def __init__(
        self,
        config: ExtractionConfig,
        *,
        environ: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        command: Sequence[str] = _WORKER,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.config = config
        # The core's bounds: how long it waits for one request's files, and
        # the largest part it hands over.
        self.timeout = config.request_timeout_seconds
        self.max_bytes = config.max_file_bytes
        self._environ = os.environ if environ is None else environ
        self._transport = transport
        self._command = tuple(command)
        self._clock = clock
        self._sleep = sleep
        self._client: httpx.AsyncClient | None = None
        self._slots: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None
        # Google service-account keys (read once, here) and their tokens.
        self._google_keys = {
            service.credentials_file: load_google_key(service.credentials_file)
            for service in config.services
            if service.credentials_file is not None
        }
        self._google_tokens: dict[str, tuple[str, float]] = {}
        self.running = 0
        self.killed = 0
        self.readings: Counter[tuple[str, str]] = Counter()

    # --- the seam -------------------------------------------------------------------

    async def inspect(self, part: UploadPart) -> Inspection:
        # Under a credential the proxy holds, a clean reading clears the
        # file only with proxy_credential: without it the upload is refused
        # whatever is read, so no service is sent a copy of the file (the
        # local reading still names what it holds in the refusal).
        services = not part.identity or self.config.proxy_credential
        reading = await self.read(part.content, services=services)
        # A file declared (by media type or file-name extension) as one
        # format and read as another may be opened by the provider as the
        # declared one (a polyglot, or just a parser this reading is not):
        # whatever was read is scanned, but it does not vouch for the file.
        coarse = sniff(part.content)
        claims = (
            declared_class(part.content_type),
            extension_class(getattr(part, "extension", None)),
        )
        complete = reading.complete and all(claim in (None, coarse) for claim in claims)
        convert = complete and coarse in self.config.convert
        return Inspection(
            reading.text,
            complete,
            reading.extractor,
            proxy_credential=self.config.proxy_credential,
            convert_text=reading.display if convert else None,
        )

    async def read(self, data: bytes, *, services: bool = True) -> FileReading:
        """``data`` read by the local extractor for its format, then — until
        a reading is complete, and unless ``services`` is False — by each
        service asked for its class."""
        coarse = sniff(data)
        texts: list[str] = []
        names: list[str] = []
        complete = False
        display: str | None = None
        pages = 1 if _one_image(data) else None
        if _local_class(coarse, self.config.formats):
            local = await self._local(data)
            if local is not None:
                name, read = local
                texts.append(read.text)
                names.append(name)
                complete, display, pages = read.complete, read.display, read.pages
        if not complete and services:
            for service in self.config.services:
                if coarse not in service.formats:
                    continue
                answer = await self._service(service, data, coarse)
                if answer is None:
                    continue
                texts.append(answer.text)
                names.append(service.kind)
                if service.kind in _PAGED and (pages is None or answer.pages != pages):
                    # A cloud OCR service analyzes up to its tier's page
                    # limit and says nothing of the pages past it (Azure's
                    # free tier reads two): complete only when it covered
                    # every page the file has, which must be known.
                    self.readings[(service.kind, "pages_unverified")] += 1
                    continue
                if answer.complete:
                    # A service vouches for the format it opened; whether the
                    # file also holds another is a property of its bytes (the
                    # worker checks its own readings the same way).
                    complete = not await asyncio.to_thread(polyglot, data)
                    display = answer.display
                    break
        return FileReading(
            "\n".join(texts) if texts else None,
            complete,
            "+".join(names) or "none",
            display if complete else None,
        )

    def status(self) -> dict[str, Any]:
        readings: dict[str, dict[str, int]] = {}
        for (extractor, result), count in sorted(self.readings.items()):
            readings.setdefault(extractor, {})[result] = count
        return {
            "formats": list(self.config.formats),
            "services": [
                {
                    "kind": service.kind,
                    "host": service.host,
                    "trusted": service.trusted,
                    "complete": service.complete,
                }
                for service in self.config.services
            ],
            "proxy_credential": self.config.proxy_credential,
            "convert": list(self.config.convert),
            "max_file_bytes": self.config.max_file_bytes,
            "timeout_seconds": self.config.timeout_seconds,
            "max_workers": self.config.max_workers,
            "workers_running": self.running,
            "workers_killed_total": self.killed,
            "readings_total": readings,
        }

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # --- local extraction -------------------------------------------------------------

    def _settings(self) -> str:
        config = self.config
        return json.dumps(
            {
                "formats": list(config.formats),
                "max_bytes": config.max_file_bytes,
                "max_chars": config.max_text_chars,
                "max_inflated": config.max_inflated_bytes,
                "memory_bytes": config.worker_memory_mb << 20,
                "cpu_seconds": math.ceil(config.timeout_seconds) + 1,
            }
        )

    def _gate(self) -> asyncio.Semaphore:
        """``max_workers`` extraction processes at a time (per event loop)."""
        loop = asyncio.get_running_loop()
        if self._slots is None or self._slots[0] is not loop:
            self._slots = (loop, asyncio.Semaphore(self.config.max_workers))
        return self._slots[1]

    async def _local(self, data: bytes) -> tuple[str, _Read] | None:
        """The worker's reading: (extractor name, reading), or None when it
        read nothing (unsupported, killed, failed)."""
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                async with self._gate():
                    result = await self._run_worker(data)
        except TimeoutError:
            self.readings[("local", "timeout")] += 1
            logger.warning("document extraction worker killed at its deadline")
            return None
        except Exception as exc:  # noqa: BLE001 — any fault: nothing read
            self.readings[("local", "failed")] += 1
            reason = str(exc) if isinstance(exc, _Failed) else type(exc).__name__
            logger.warning("document extraction worker failed (%s)", reason)
            return None
        name = f"local:{result.get('format')}"
        text, complete = result.get("text"), result.get("complete") is True
        pages = result.get("pages")
        pages = pages if type(pages) is int else None
        if not isinstance(text, str):
            self.readings[(name, str(result.get("reason")))] += 1
            return None
        self.readings[(name, "complete" if complete else "incomplete")] += 1
        display = result.get("display")
        return name, _Read(text, complete, display if isinstance(display, str) else None, pages)

    async def _run_worker(self, data: bytes) -> dict[str, Any]:
        """One worker process for ``data``, killed on any way out but its
        own exit (the deadline, a cancelled request, too much output)."""
        # The full reading and the display reading, each JSON-escaped.
        cap = 2 * self.config.max_text_chars * _JSON_BYTES_PER_CHAR + 4096
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command,
                self._settings(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=_WORKER_ENV,
                start_new_session=True,
            )
        except OSError as exc:
            raise _Failed(f"start: {type(exc).__name__}") from None
        self.running += 1
        try:
            output = await _exchange(proc, data, cap)
        finally:
            self.running -= 1
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                self.killed += 1
                # Reaped (and its pipes closed) even when the request is
                # cancelled meanwhile: SIGKILL cannot be ignored.
                await asyncio.shield(proc.wait())
        try:
            result = json.loads(output)
        except ValueError:
            raise _Failed(f"exit {proc.returncode}, no reading") from None
        if not isinstance(result, dict):
            raise _Failed("malformed reading")
        return result

    # --- services ---------------------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            # No redirects, no proxy or netrc from the environment: a file
            # goes exactly to the service configured for it. Each request
            # carries its service's own timeout_seconds (httpx's 5 s default
            # would cut every OCR service short of it).
            self._client = httpx.AsyncClient(
                transport=self._transport, follow_redirects=False, trust_env=False
            )
        return self._client

    async def _service(self, service: ExtractionService, data: bytes, coarse: str) -> _Read | None:
        """One service's reading, or None when it failed (logged by kind,
        host and a fixed failure kind or exception type only)."""
        try:
            async with asyncio.timeout(service.timeout_seconds):
                read = await _SERVICES[service.kind](self, service, data, coarse)
        except Exception as exc:  # noqa: BLE001 — any fault: not read
            self.readings[(service.kind, "failed")] += 1
            reason = str(exc) if isinstance(exc, _Failed) else type(exc).__name__
            logger.warning(
                "extraction service %s (%s) failed (%s)", service.kind, service.host, reason
            )
            return None
        self.readings[(service.kind, "complete" if read.complete else "incomplete")] += 1
        return read

    def _env(self, name: str) -> str:
        """A credential variable's value; ``_Failed`` (naming the variable,
        never a value) when it is unset."""
        value = self._environ.get(name)
        if not value:
            raise _Failed(f"{name} is not set")
        return value

    async def _auth(self, service: ExtractionService) -> dict[str, str]:
        """The credential header of a token-authorized service (none when
        it names no credential)."""
        if service.credentials_file is not None:
            token = await self._google_token(service)
        elif service.token_env is not None:
            token = self._env(service.token_env)
        else:
            return {}
        name = _AUTH_HEADERS[service.kind]
        return {name: f"Bearer {token}" if name == "authorization" else token}

    async def _send(
        self, request: httpx.Request, accept: tuple[int, ...] = (200,)
    ) -> tuple[httpx.Response, Any]:
        """A service's answer and its JSON body (None when empty), read no
        further than the text cap allows; a status outside ``accept`` (a
        redirect included: none is followed) fails."""
        cap = self.config.max_text_chars * _JSON_BYTES_PER_CHAR + (1 << 20)
        response = await self._http().send(request, stream=True)
        try:
            if response.status_code not in accept:
                raise _Failed(f"HTTP {response.status_code}")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body += chunk
                if len(body) > cap:
                    raise _Failed("answer too large")
        finally:
            await response.aclose()
        return response, (json.loads(bytes(body)) if body else None)

    async def _tika(self, service: ExtractionService, data: bytes, coarse: str) -> _Read:
        """Apache Tika's recursive reading (``PUT /rmeta/text``): every
        document and embedded document's text and metadata; complete only
        when declared so and Tika reports no exception."""
        request = self._http().build_request(
            "PUT",
            f"{service.url}/rmeta/text",
            content=data,
            headers={**await self._auth(service), "accept": "application/json"},
            timeout=service.timeout_seconds,
        )
        _, answer = await self._send(request)
        if not isinstance(answer, list) or not all(isinstance(d, dict) for d in answer):
            raise _Failed("malformed answer")
        pieces: list[str] = []
        shown: list[str] = []
        faulted = False
        for document in answer:
            for key, value in document.items():
                if key.startswith("X-TIKA:EXCEPTION"):
                    faulted = True
                else:
                    pieces.extend(_strings(value))
            content = document.get("X-TIKA:content")
            if isinstance(content, str):
                shown.append(content.strip())
        return _Read("\n".join(pieces), service.complete and not faulted, "\n\n".join(shown))

    async def _docling(self, service: ExtractionService, data: bytes, coarse: str) -> _Read:
        """docling-serve (``POST /v1/convert/file``, text output); complete
        only when declared so and the conversion fully succeeded."""
        request = self._http().build_request(
            "POST",
            f"{service.url}/v1/convert/file",
            files={"files": (upload_name(data, coarse), data, "application/octet-stream")},
            data={"to_formats": "text"},
            headers=await self._auth(service),
            timeout=service.timeout_seconds,
        )
        _, answer = await self._send(request)
        document = answer.get("document") if isinstance(answer, dict) else None
        text = document.get("text_content") if isinstance(document, dict) else None
        if not isinstance(text, str):
            raise _Failed("malformed answer")
        clean = answer.get("status") == "success" and not answer.get("errors")
        return _Read(text, service.complete and clean, text)

    async def _unstructured(self, service: ExtractionService, data: bytes, coarse: str) -> _Read:
        """unstructured (``POST /general/v0/general``): every element's text
        and string metadata (link targets, HTML renderings)."""
        request = self._http().build_request(
            "POST",
            f"{service.url}/general/v0/general",
            files={"files": (upload_name(data, coarse), data, "application/octet-stream")},
            data={"output_format": "application/json"},
            headers=await self._auth(service),
            timeout=service.timeout_seconds,
        )
        _, answer = await self._send(request)
        if not isinstance(answer, list) or not all(isinstance(e, dict) for e in answer):
            raise _Failed("malformed answer")
        pieces: list[str] = []
        shown: list[str] = []
        for element in answer:
            pieces.extend(_strings(element.get("text")))
            pieces.extend(_strings(element.get("metadata")))
            shown.extend(_strings(element.get("text")))
        return _Read("\n".join(pieces), service.complete, "\n".join(shown))

    async def _textract(self, service: ExtractionService, data: bytes, coarse: str) -> _Read:
        """AWS Textract ``DetectDocumentText`` (synchronous: an image or a
        one-page PDF — Textract refuses a longer PDF, which is then not
        read), signed with SigV4 from the credential variables the config
        names. Every LINE block's text; complete only when declared so."""
        credentials = AwsCredentials(
            self._env(service.access_key_env),
            self._env(service.secret_key_env),
            self._environ.get(service.session_token_env) or None,
        )
        region = service.region or ""
        # Encoding and hashing a large file is CPU work: off the event loop.
        body, headers = await asyncio.to_thread(
            _textract_request, service.url, region, credentials, data
        )
        request = self._http().build_request(
            "POST",
            f"{service.url}/",
            content=body,
            headers=headers,
            timeout=service.timeout_seconds,
        )
        _, answer = await self._send(request)
        blocks = answer.get("Blocks") if isinstance(answer, dict) else None
        if not isinstance(blocks, list) or not all(isinstance(b, dict) for b in blocks):
            raise _Failed("malformed answer")
        lines = [
            block["Text"]
            for block in blocks
            if block.get("BlockType") == "LINE" and isinstance(block.get("Text"), str)
        ]
        text = "\n".join(lines)
        return _Read(text, service.complete, text, _count(answer.get("DocumentMetadata"), "Pages"))

    async def _documentai(self, service: ExtractionService, data: bytes, coarse: str) -> _Read:
        """Google Document AI (``POST /v1/{processor}:process``) with an
        OAuth access token (``token_env``, or minted from a service-account
        key file); complete only when declared so and the document carries
        no error."""
        payload = await asyncio.to_thread(_documentai_body, data, media_type(data, coarse))
        request = self._http().build_request(
            "POST",
            f"{service.url}/v1/{service.processor}:process",
            content=payload,
            headers={**await self._auth(service), "content-type": "application/json"},
            timeout=service.timeout_seconds,
        )
        _, answer = await self._send(request)
        document = answer.get("document") if isinstance(answer, dict) else None
        if not isinstance(document, dict) or not isinstance(document.get("text"), str):
            raise _Failed("malformed answer")
        text = document["text"]
        complete = service.complete and not document.get("error")
        return _Read(text, complete, text, _count(document, "pages"))

    async def _azure_docintel(self, service: ExtractionService, data: bytes, coarse: str) -> _Read:
        """Azure AI Document Intelligence: ``:analyze`` (asynchronous — 202
        and an ``Operation-Location``), then its operation polled until it
        succeeds, all inside the service's ``timeout_seconds``. The
        operation must be on the service's own origin (the key is never
        sent anywhere else). Complete only when declared so and the result
        carries no warning."""
        headers = await self._auth(service)
        request = self._http().build_request(
            "POST",
            f"{service.url}/documentintelligence/documentModels/{service.model}:analyze",
            params={"api-version": DOCINTEL_API_VERSION},
            content=data,
            headers={**headers, "content-type": "application/octet-stream"},
            timeout=service.timeout_seconds,
        )
        response, _ = await self._send(request, (202,))
        operation = response.headers.get("operation-location", "")
        if not _same_origin(operation, service.url):
            raise _Failed("operation off the service's origin")
        while True:
            poll = self._http().build_request(
                "GET", operation, headers=headers, timeout=service.timeout_seconds
            )
            response, answer = await self._send(poll)
            status = answer.get("status") if isinstance(answer, dict) else None
            if status == "succeeded":
                break
            if status not in ("notStarted", "running"):
                raise _Failed("operation did not succeed")
            await self._sleep(_retry_after(response.headers.get("retry-after")))
        result = answer.get("analyzeResult")
        if not isinstance(result, dict) or not isinstance(result.get("content"), str):
            raise _Failed("malformed answer")
        text = result["content"]
        complete = service.complete and not result.get("warnings")
        return _Read(text, complete, text, _count(result, "pages"))

    # --- Google service-account tokens ---------------------------------------------------

    async def _google_token(self, service: ExtractionService) -> str:
        """An access token minted from the service-account key file (a
        signed JWT exchanged at Google's token endpoint), cached until a
        minute before it expires."""
        path = service.credentials_file or ""
        cached = self._google_tokens.get(path)
        now = self._clock()
        if cached is not None and cached[1] > now:
            return cached[0]
        assertion = await asyncio.to_thread(_google_assertion, self._google_keys[path], int(now))
        request = self._http().build_request(
            "POST",
            GOOGLE_TOKEN_URI,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": assertion,
            },
            timeout=service.timeout_seconds,
        )
        _, answer = await self._send(request)
        token = answer.get("access_token") if isinstance(answer, dict) else None
        if not isinstance(token, str) or not token:
            raise _Failed("token endpoint gave no access token")
        assert isinstance(answer, dict)
        lifetime = answer.get("expires_in")
        seconds = float(lifetime) if isinstance(lifetime, (int, float)) else 0.0
        self._google_tokens[path] = (token, now + max(seconds - _TOKEN_EARLY_SECONDS, 0.0))
        return token


_SERVICES: dict[str, Callable[..., Awaitable[_Read]]] = {
    "tika": ExtractionInspector._tika,
    "docling": ExtractionInspector._docling,
    "unstructured": ExtractionInspector._unstructured,
    "textract": ExtractionInspector._textract,
    "documentai": ExtractionInspector._documentai,
    "azure_docintel": ExtractionInspector._azure_docintel,
}


# The cloud OCR services: a reading of theirs vouches for a file only when
# it covers as many pages as the file has (``ExtractionInspector.read``).
_PAGED = frozenset({"textract", "documentai", "azure_docintel"})


def _count(value: Any, key: str) -> int | None:
    """The page count a service's answer reports under ``key`` (a number,
    or a list of pages), or None when it reports none."""
    count = value.get(key) if isinstance(value, dict) else None
    if isinstance(count, list):
        return len(count)
    return count if type(count) is int else None


def _one_image(data: bytes) -> bool:
    """Whether ``data`` is an image format that holds exactly one picture:
    a PNG that is not animated (no ``acTL`` chunk), a JPEG or a BMP. A
    TIFF, GIF, WebP or HEIF may hold several: its page count is unknown."""
    if data.startswith(b"\x89PNG"):
        return b"acTL" not in data
    return data.startswith((b"\xff\xd8\xff", b"BM"))


def _documentai_body(data: bytes, mime: str) -> bytes:
    document = {"content": base64.b64encode(data).decode("ascii"), "mimeType": mime}
    return json.dumps({"rawDocument": document, "skipHumanReview": True}).encode("ascii")


def _textract_request(
    url: str, region: str, credentials: AwsCredentials, data: bytes
) -> tuple[bytes, dict[str, str]]:
    """The signed DetectDocumentText request: (body, headers)."""
    document = {"Bytes": base64.b64encode(data).decode("ascii")}
    body = json.dumps({"Document": document}).encode("ascii")
    headers = [
        ("content-type", "application/x-amz-json-1.1"),
        ("x-amz-target", "Textract.DetectDocumentText"),
    ]
    signed = sign_request(
        "POST", f"{url}/", headers, body, region=region, service="textract", credentials=credentials
    )
    return body, dict(signed)


def _same_origin(url: str, base: str) -> bool:
    """Whether ``url`` is https on exactly ``base``'s host and port, with
    no user information."""
    try:
        target, origin = urlsplit(url), urlsplit(base)
        return (
            target.scheme == origin.scheme == "https"
            and target.hostname == origin.hostname
            and target.port == origin.port
            and "@" not in target.netloc
        )
    except ValueError:
        return False


def _retry_after(value: str | None) -> float:
    """Seconds to wait before the next poll: a numeric Retry-After within
    ``_POLL_MIN_SECONDS``..``_POLL_MAX_SECONDS``, else one second."""
    try:
        seconds = float(value) if value is not None else 1.0
    except ValueError:
        seconds = 1.0
    if not math.isfinite(seconds):
        seconds = 1.0
    return min(max(seconds, _POLL_MIN_SECONDS), _POLL_MAX_SECONDS)


def load_google_key(path: str) -> dict[str, str]:
    """A service-account key file's signing material — a ConfigError naming
    the FILE (never its content) when it cannot be read, is not a service
    account key, names another token endpoint than Google's, holds no RSA
    key, or the ``crypto`` extra (RS256 signing) is missing."""
    source = f"[extraction] documentai credentials_file {path}"
    if importlib.util.find_spec("cryptography") is None:
        raise ConfigError(f"{source} needs the crypto extra to sign tokens: {CRYPTO_HINT}")
    try:
        info = json.loads(Path(path).expanduser().read_text("utf-8"))
    except (OSError, ValueError):
        raise ConfigError(f"{source} cannot be read as JSON") from None
    fields = ("client_email", "private_key", "token_uri")
    if (
        not isinstance(info, dict)
        or info.get("type") != "service_account"
        or not all(isinstance(info.get(key), str) and info[key] for key in fields)
    ):
        raise ConfigError(f"{source} is not a service-account key file")
    if info["token_uri"] != GOOGLE_TOKEN_URI:
        raise ConfigError(f"{source} names a token endpoint other than {GOOGLE_TOKEN_URI}")
    key = {name: info[name] for name in fields}
    if isinstance(info.get("private_key_id"), str):
        key["private_key_id"] = info["private_key_id"]
    try:
        _rsa_key(key["private_key"])
    except Exception:  # noqa: BLE001 — any fault: a key nothing can sign with
        raise ConfigError(f"{source} holds no usable RSA private key") from None
    return key


def _rsa_key(pem: str) -> Any:
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    key = load_pem_private_key(pem.encode("utf-8"), password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise TypeError("not an RSA key")
    return key


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _google_assertion(key: Mapping[str, str], now: int) -> str:
    """The RS256-signed JWT a service account exchanges for a token."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    header = {"alg": "RS256", "typ": "JWT"}
    if "private_key_id" in key:
        header["kid"] = key["private_key_id"]
    claims = {
        "iss": key["client_email"],
        "scope": _GOOGLE_SCOPE,
        "aud": key["token_uri"],
        "iat": now,
        "exp": now + _JWT_LIFETIME,
    }
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(claims).encode())}"
    signature = _rsa_key(key["private_key"]).sign(
        signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256()
    )
    return f"{signing_input}.{_b64url(signature)}"


def _strings(value: Any, depth: int = 0) -> list[str]:
    """Every string in a JSON value (a few levels deep)."""
    if isinstance(value, str):
        return [value]
    if depth > 4:
        return []
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item, depth + 1)]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item, depth + 1)]
    return []


async def _exchange(proc: asyncio.subprocess.Process, data: bytes, cap: int) -> bytes:
    """Feed ``data`` to the worker and read its answer, at most ``cap``
    bytes, until it exits."""
    assert proc.stdin is not None and proc.stdout is not None

    async def feed() -> None:
        assert proc.stdin is not None
        try:
            proc.stdin.write(data)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the worker stopped reading: its answer says why
        finally:
            proc.stdin.close()

    feeder = asyncio.ensure_future(feed())
    try:
        chunks: list[bytes] = []
        size = 0
        while chunk := await proc.stdout.read(1 << 16):
            size += len(chunk)
            if size > cap:
                raise _Failed("output over its cap")
            chunks.append(chunk)
        await proc.wait()
    finally:
        feeder.cancel()
    return b"".join(chunks)


def build_inspector(config: ExtractionConfig) -> ExtractionInspector:
    """The inspector for an enabled ``[extraction]`` — or a ConfigError
    naming what keeps it from working (``extraction_problems``: pypdf for
    PDFs, a credential variable by NAME, a key file, nothing to read)."""
    problems = extraction_problems(config, os.environ)
    if problems:
        raise ConfigError(problems[0])
    return ExtractionInspector(config)


def extraction_problems(config: ExtractionConfig, environ: Mapping[str, str]) -> list[str]:
    """What keeps an enabled ``[extraction]`` from starting (values and
    variable NAMES only)."""
    problems: list[str] = []
    if "pdf" in config.formats and importlib.util.find_spec("pypdf") is None:
        problems.append('[extraction] formats includes "pdf", which needs pypdf: ' + INSTALL_HINT)
    for service in config.services:
        problems.extend(
            f"[extraction] {service.kind} service at {service.host}: {name}"
            " is not set (the credential comes from the environment, never the file)"
            for name in service.credential_env()
            if not environ.get(name)
        )
        if service.credentials_file is not None:
            try:
                load_google_key(service.credentials_file)
            except ConfigError as problem:
                problems.append(str(problem))
    if not config.formats and not config.services:
        problems.append("[extraction] enabled with no formats and no services: nothing reads")
    return problems


def extraction_checks(
    config: ExtractionConfig, environ: Mapping[str, str]
) -> list[tuple[str, str]]:
    """``llm-redact doctor``'s ``extraction`` rows — offline and value-free:
    what keeps it from starting (the ``extract`` extra, a credential
    variable by NAME, a key file), each service and whether it sees files
    off this machine, convert mode, and what a clean scan lets through."""
    if not config.enabled:
        return [("PASS", "disabled: binary uploads keep the [detection] binary_uploads rules")]
    rows: list[tuple[str, str]] = [
        ("FAIL", f"{problem} — the proxy will refuse to start")
        for problem in extraction_problems(config, environ)
    ]
    for service in config.services:
        local = service.loopback and not service.cloud
        where = "on this machine" if local else "OFF this machine (trusted = true)"
        claim = "counts as complete" if service.complete else "never counts as complete"
        rows.append(
            (
                "PASS" if local else "WARN",
                f"{service.kind} service at {service.host} sees every file it is asked to"
                f" read, {where}; its reading {claim}",
            )
        )
    if config.convert:
        rows.append(
            (
                "WARN",
                f"convert on for {', '.join(config.convert)}: a document holding values is"
                " sent as its REDACTED EXTRACTED TEXT instead of the file (the model sees"
                " text, not the original)",
            )
        )
    formats = ", ".join(config.formats) or "none"
    credential = (
        "also under a credential the proxy holds"
        if config.proxy_credential
        else "with the client's own key only"
    )
    rows.append(
        (
            "PASS",
            f"binary uploads read as text (local: {formats}; services:"
            f" {len(config.services)}); a complete clean reading sends the file as is,"
            f" {credential} — the scan covers the extracted text only",
        )
    )
    return rows
