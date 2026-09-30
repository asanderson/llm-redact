"""A text upload that is ONE JSON document is redacted escape-aware.

A pretty-printed JSON file (a GCP service-account key) used to be redacted
as ONE raw text: the vault held each value as its source spells it (a PEM
key with literal ``\\n`` escapes), and only a process that remembered the
upload (``RAW_TEXT_FILES``, per process, newest 1024 — evicted even by
uploads the proxy then refused) restored it raw. Every other process (a
second replica over a shared vault, another ``llm-redact run`` proxy, a
restart) restored it JSON-escaped once more: a silently broken PEM inside
valid JSON. Each string literal is now decoded, redacted and written back,
so the vault holds decoded values and the download's JSON-source
restoration is the exact inverse in any process; what the escape-aware
reading cannot stand in for is redacted raw as before.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_redact.bench.corpus import generate
from llm_redact.config import parse_config
from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.multipart import parse
from llm_redact.providers import openai
from llm_redact.providers.openai import redact_document
from llm_redact.proxy import create_app
from llm_redact.redactor import Redactor
from llm_redact.upload_content import FileContent
from llm_redact.vault import InMemoryVault

PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\n"
    "ABCDEF\n"
    "-----END PRIVATE KEY-----\n"
)
SERVICE_ACCOUNT = {
    "type": "service_account",
    "project_id": "demo",
    "private_key_id": "0123456789abcdef0123456789abcdef01234567",
    "private_key": PEM,
    "client_email": "svc@demo.iam.gserviceaccount.com",
}
ORIGINAL = json.dumps(SERVICE_ACCOUNT, indent=2).encode() + b"\n"

_OPENAI = {"authorization": "Bearer sk-own"}
_ANTHROPIC = {"anthropic-version": "2023-06-01", "x-api-key": "sk-ant-own"}
_GEMINI = {"x-goog-api-key": "AIza-own"}


def _form(content: bytes, extra: bytes = b"") -> tuple[bytes, str]:
    body = (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="sa.json"\r\n'
        b"Content-Type: application/json\r\n\r\n" + content + b"\r\n" + extra + b"--b--\r\n"
    )
    return body, "multipart/form-data; boundary=b"


def _related(content: bytes) -> tuple[bytes, str]:
    body = (
        b"--x\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
        b'{"file": {"display_name": "sa"}}\r\n'
        b"--x\r\nContent-Type: application/json\r\n\r\n" + content + b"\r\n--x--\r\n"
    )
    return body, "multipart/related; boundary=x"


# route -> (upload path, builder, extra upload headers, download path, headers)
_ROUTES: dict[
    str, tuple[str, Callable[[bytes], tuple[bytes, str]], dict[str, str], str, dict[str, str]]
] = {
    "openai": ("/v1/files", _form, {}, "/v1/files/file-1/content", _OPENAI),
    "anthropic": ("/v1/files", _form, {}, "/v1/files/file-1/content", _ANTHROPIC),
    "gemini": (
        "/upload/v1beta/files",
        _related,
        {"x-goog-upload-protocol": "multipart"},
        "/download/v1beta/files/f1:download",
        _GEMINI,
    ),
}


class _Store:
    """A fake upstream keeping the last file part it received."""

    def __init__(self) -> None:
        self.file = b""
        self.uploads = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            boundary = request.headers["content-type"].split("boundary=")[1].encode()
            parsed = parse(request.content, boundary)
            assert parsed is not None
            self.file = parsed.parts[-1].content
            self.uploads += 1
            return httpx.Response(200, json={"id": "file-1", "file": {"name": "files/f1"}})
        return httpx.Response(
            200, content=self.file, headers={"content-type": "application/octet-stream"}
        )


async def _send(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        return await client.request(method, path, **kwargs)


@pytest.mark.parametrize("route", sorted(_ROUTES))
async def test_a_service_account_key_round_trips_through_another_process(
    route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_path, build, upload_headers, download_path, headers = _ROUTES[route]
    store = _Store()
    config = parse_config({"vault": {"backend": "sqlite", "path": str(tmp_path / "v.db")}}, "t")
    monkeypatch.setattr(openai, "RAW_TEXT_FILES", openai._RawTextFiles())
    body, content_type = build(ORIGINAL)
    first = create_app(config, upstream_transport=httpx.MockTransport(store))
    async with first.router.lifespan_context(first):
        upload = await _send(
            first,
            "POST",
            upload_path,
            content=body,
            headers={**headers, **upload_headers, "content-type": content_type},
        )
        assert upload.status_code == 200, upload.text
    # Every secret left, and the file is still one JSON document.
    for secret in (b"PRIVATE KEY", b"0123456789abcdef", b"svc@demo"):
        assert secret not in store.file
    sent = json.loads(store.file)
    assert sent["private_key"].endswith("\n") and sent["type"] == "service_account"
    uploaded = openai.RAW_TEXT_FILES
    # A second process over the same vault (a replica, a restart, another
    # `llm-redact run` proxy) restores the file exactly.
    monkeypatch.setattr(openai, "RAW_TEXT_FILES", openai._RawTextFiles())
    second = create_app(config, upstream_transport=httpx.MockTransport(store))
    async with second.router.lifespan_context(second):
        download = await _send(second, "GET", download_path, headers=headers)
    assert download.status_code == 200
    assert download.content == ORIGINAL
    assert json.loads(download.content)["private_key"] == PEM
    # The vault holds the key as it decodes: nothing needed remembering.
    assert not uploaded._digests


async def test_refused_uploads_never_evict_a_remembered_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Text that is not JSON (an invalid escape) and reads as JSON Lines only
    # once redacted is remembered for its raw restoration. 1024 uploads the
    # proxy REFUSES (a base64 part after such a text) record nothing: they
    # were never forwarded.
    monkeypatch.setattr(openai, "RAW_TEXT_FILES", openai._RawTextFiles())
    store = _Store()
    config = parse_config({"detection": {"deny": ["CORP\\jdoe"]}}, "t")
    app = create_app(config, upstream_transport=httpx.MockTransport(store))
    original = b'{"user":"CORP\\jdoe"}\n'
    multipart = {**_OPENAI, "content-type": "multipart/form-data; boundary=b"}
    refused_part = (
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="g"\r\n'
        b"Content-Transfer-Encoding: base64\r\n\r\nzzz\r\n"
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        upload = await client.post("/v1/files", content=_form(original)[0], headers=multipart)
        assert upload.status_code == 200, upload.text
        remembered = store.file
        statuses = set()
        for index in range(openai._RawTextFiles._MAX):
            other = b'{"user":"CORP\\jdoe", "n": %d}\n' % index
            refused = await client.post(
                "/v1/files", content=_form(other, refused_part)[0], headers=multipart
            )
            statuses.add(refused.status_code)
        store.file = remembered
        download = await client.get("/v1/files/file-1/content", headers=_OPENAI)
    assert statuses == {400} and store.uploads == 1
    assert download.content == original


class _RecordingVault(InMemoryVault):
    """An in-memory vault remembering every value it was asked to map."""

    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def placeholder_for(self, detector_type: str, original: str, *, floor: int = 0) -> str:
        self.values.append(original)
        return super().placeholder_for(detector_type, original, floor=floor)


def _documents() -> list[tuple[str, bool]]:
    """JSON documents (pretty-printed: text files, not JSON Lines) around
    every recall-corpus sample: as a string value, as a key, a contextual
    sample written straight into the source, a digit value as a number —
    each with whether a value lies outside every string (the raw reading
    must take over)."""
    documents: list[tuple[str, bool]] = []
    for sample in generate(seed=11, samples_per_rule=3):
        documents.append((json.dumps({"note": sample.text, "n": 1.5}, indent=2), False))
        documents.append((json.dumps({sample.text: [1, "x"]}, indent=2), False))
        inline = "{\n  " + sample.text + "\n}"
        try:
            json.loads(inline)
        except ValueError:
            pass
        else:
            documents.append((inline, False))  # "private_key_id": "…" as the source spells it
        for span in sample.spans:
            value = sample.text[span.start : span.end]
            if re.fullmatch(r"[1-9][0-9]*", value):
                documents.append(('{\n  "id": ' + value + "\n}", True))
    documents.append(('{\n  "card": 4111111111111111,\n  "note": "PEM"\n}', True))
    documents.append((json.dumps({"key": PEM, "who": "a@b.example"}, indent=2), False))
    return documents


def _surfaces(source: str) -> list[str]:
    """Where a value could still be read: the source, and every string of
    it (keys included) as it decodes when it is JSON."""
    surfaces = [source]

    def strings(value: Any) -> None:
        if isinstance(value, str):
            surfaces.append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                surfaces.append(key)
                strings(item)
        elif isinstance(value, list):
            for item in value:
                strings(item)

    with contextlib.suppress(ValueError):
        strings(json.loads(source))
    return surfaces


def _decoded(fragment: str) -> str:
    try:
        value = json.loads('"' + fragment + '"')
    except ValueError:
        return fragment
    return value if isinstance(value, str) else fragment


def test_the_escape_aware_reading_misses_nothing_the_raw_reading_redacts() -> None:
    config = DetectionConfig()
    detectors, allowlist = build_detectors(config), build_allowlist(config)
    documents = _documents()
    assert sum(outside for _, outside in documents) > 1
    for document, outside in documents:
        raw_vault, vault = _RecordingVault(), _RecordingVault()
        Redactor(detectors, raw_vault, allowlist).redact_text(document)
        content = FileContent("text", document, "utf-8")
        redacted, raw = redact_document(content, Redactor(detectors, vault, allowlist))
        # The raw reading takes over only where it must (a value outside
        # every string: a card number written as a JSON number).
        assert raw == outside, document
        if not raw:
            json.loads(redacted)  # still one JSON document
        surfaces = _surfaces(redacted)
        for value in raw_vault.values:
            for form in {value, _decoded(value)}:
                assert not any(form in surface for surface in surfaces), (document, value)


def test_a_value_only_the_source_form_shows_falls_back_to_the_raw_reading() -> None:
    # A deny string spelled as the JSON source spells it (two backslashes)
    # never matches the decoded value: the raw reading redacts the file
    # instead, nothing counted twice.
    config = parse_config({"detection": {"deny": ["CORP\\\\jdoe"]}}, "t").detection
    detectors, allowlist = build_detectors(config), build_allowlist(config)
    redactor = Redactor(detectors, InMemoryVault(), allowlist)
    document = '{\n  "user": "CORP\\\\jdoe",\n  "who": "a@b.example"\n}'
    redacted, raw = redact_document(FileContent("text", document, "utf-8"), redactor)
    assert raw and "CORP" not in redacted and "a@b.example" not in redacted
    assert redactor.counts == {"DENY": 1, "EMAIL": 1}
    # A card number written as a JSON number: the same, and a warn-mode
    # value inside a string is counted once.
    redactor = Redactor(detectors, InMemoryVault(), allowlist, modes={"EMAIL": "warn"})
    document = '{\n  "card": 4111111111111111,\n  "who": "a@b.example"\n}'
    redacted, raw = redact_document(FileContent("text", document, "utf-8"), redactor)
    assert raw and "4111111111111111" not in redacted and "a@b.example" in redacted
    assert redactor.counts == {"CREDIT_CARD": 1} and redactor.warn_counts == {"EMAIL": 1}


def test_a_context_found_value_is_redacted_in_place_counted_once() -> None:
    # The service-account key id is found by its key's context (a raw
    # reading of the source): redacted inside its escape-free literal, the
    # document stays JSON and a warn-mode value is counted once.
    config = DetectionConfig()
    redactor = Redactor(
        build_detectors(config), InMemoryVault(), build_allowlist(config), modes={"EMAIL": "warn"}
    )
    redacted, raw = redact_document(FileContent("text", ORIGINAL.decode(), "utf-8"), redactor)
    assert not raw
    sent = json.loads(redacted)
    assert sent["private_key_id"] == "«GCP_KEY_ID_001»"
    assert sent["private_key"] == "«PRIVATE_KEY_001»\n"
    assert sent["client_email"] == SERVICE_ACCOUNT["client_email"]
    assert redactor.counts == {"GCP_KEY_ID": 1, "PRIVATE_KEY": 1}
    assert redactor.warn_counts == {"EMAIL": 1}
