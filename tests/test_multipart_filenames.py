"""Upload filenames never reach the provider, and come back restored.

A part's Content-Disposition ``filename`` / ``filename*`` is user content
(``jane.doe@corp.example.jsonl``): redacted on every multipart route in BOTH
auth modes, the rest of the header block byte-identical. The provider echoes
the name in the file object (the upload response, the list, one file), all
CHAT, so the tool sees its original name again (static session).

A filename without a single reading is refused, and so is a part the proxy
could not read as plain bytes — a Content-Transfer-Encoding, a declared
charset other than UTF-8/US-ASCII: under identity auth and, since the
scanned-body rule, under the client's own key wherever redaction applies
(both pinned). ``detection = false`` with the client's own key forwards
such an upload as sent (pinned).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes

import httpx
import pytest

from llm_redact.config import Config, DetectionConfig, ProviderConfig, VaultConfig
from llm_redact.providers.azure_openai import AzureOpenAIAdapter
from llm_redact.providers.base import RouteKind
from llm_redact.providers.custom import CustomOpenAIAdapter
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.proxy import create_app
from test_session_ownership_seams import OwnershipRouter, _registry
from test_upstream_auth import AZURE, EMAIL, _client, _config, _identity, _install

TOKEN = "«EMAIL_001»"
# Space-delimited: "jane.doe@corp.example.jsonl" would itself be one address.
FILENAME = f"{EMAIL} batch.jsonl"
LINE = b'{"custom_id": "a", "body": {"messages": [{"role": "user", "content": "hi"}]}}'
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe" + b"\x00" * 8
AUTH = {"authorization": "Bearer sk-test"}


def _part(headers: bytes, content: bytes) -> bytes:
    return b"--b\r\n" + headers + b"\r\n\r\n" + content + b"\r\n"


def _form(*parts: bytes) -> bytes:
    return b"".join(parts) + b"--b--\r\n"


def _headers() -> dict[str, str]:
    return {"content-type": "multipart/form-data; boundary=b"}


PURPOSE = _part(b'Content-Disposition: form-data; name="purpose"', b"batch")


def _jsonl_part(disposition_tail: bytes, content: bytes = LINE, extra: bytes = b"") -> bytes:
    return _part(
        b'Content-Disposition: form-data; name="file"; '
        + disposition_tail
        + b"\r\nContent-Type: application/jsonl"
        + extra,
        content,
    )


def _wire_filename(body: bytes) -> str:
    """How the Files API reads the name: filename* (RFC 8187) first."""
    ext = re.search(rb"filename\*=([^;\r\n]+)", body)
    if ext is not None:
        _, _, encoded = ext.group(1).split(b"'", 2)
        return unquote_to_bytes(encoded).decode()
    quoted = re.search(rb'filename="((?:[^"\\]|\\.)*)"', body)
    if quoted is None:
        return ""  # a form without a file part
    return re.sub(rb"\\(.)", rb"\1", quoted.group(1)).decode()


class FilesUpstream:
    """A Files API: stores each upload's name as received and echoes it in
    the file object (upload response, list, one file)."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.filename = ""

    def _object(self) -> dict[str, Any]:
        return {
            "id": "file-1",
            "object": "file",
            "bytes": 3,
            "created_at": 1,
            "filename": self.filename,
            "purpose": "batch",
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST":
            self.filename = _wire_filename(request.content)
            return httpx.Response(200, json=self._object())
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"object": "list", "data": [self._object()]})
        return httpx.Response(200, json=self._object())


def _app(config: Config, upstream: FilesUpstream) -> Any:
    return create_app(config, upstream_transport=httpx.MockTransport(upstream))


# (upload path, list path, one-file path, identity auth?) per surface.
SURFACES = {
    "openai-key": ("/v1/files", "/v1/files", "/v1/files/file-1", False),
    "azure-identity": (
        "/openai/files?api-version=1",
        "/openai/files?api-version=1",
        "/openai/v1/files/file-1?api-version=1",
        True,
    ),
    "custom-key": (
        "/custom/vllm/v1/files",
        "/custom/vllm/v1/files",
        "/custom/vllm/v1/files/file-1",
        False,
    ),
}


def _surface_config(surface: str) -> Config:
    if surface == "azure-identity":
        return _config(azure=_identity(AZURE))
    if surface == "custom-key":
        return _config(**{"custom:vllm": ProviderConfig("http://up-vllm")})
    return Config()


@pytest.mark.parametrize("surface", sorted(SURFACES))
@pytest.mark.parametrize(
    "tail",
    [
        b'filename="' + FILENAME.encode() + b'"',
        b"filename*=UTF-8''" + FILENAME.replace("@", "%40").replace(" ", "%20").encode(),
    ],
    ids=["filename", "filename-star"],
)
async def test_filename_redacted_upstream_and_restored_on_every_echo(
    monkeypatch: pytest.MonkeyPatch, surface: str, tail: bytes
) -> None:
    _, built = _install(monkeypatch)
    upload, listing, item, identity = SURFACES[surface]
    upstream = FilesUpstream()
    app = _app(_surface_config(surface), upstream)
    body = _form(PURPOSE, _jsonl_part(tail))
    async with _client(app) as client:
        uploaded = await client.post(upload, content=body, headers={**_headers(), **AUTH})
        listed = await client.get(listing, headers=AUTH)
        one = await client.get(item, headers=AUTH)
    sent = upstream.requests[0].content
    assert EMAIL.encode() not in sent and EMAIL.replace("@", "%40").encode() not in sent
    # Only the filename value changed: every other byte is the original's.
    if b"filename*" in tail:
        new_tail = b"filename*=UTF-8''%C2%ABEMAIL_001%C2%BB%20batch.jsonl"
    else:
        new_tail = b'filename="' + TOKEN.encode() + b' batch.jsonl"'
    assert sent == body.replace(tail, new_tail)
    assert upstream.filename == f"{TOKEN} batch.jsonl"
    # Every echo of the file object hands the tool its own name back.
    assert uploaded.json()["filename"] == FILENAME
    assert listed.json()["data"][0]["filename"] == FILENAME
    assert one.json()["filename"] == FILENAME
    if identity:
        assert [call[3] for call in built[0].calls][0] == sent


@pytest.mark.parametrize("identity", [False, True])
async def test_image_part_filename_redacted_media_bytes_kept(
    monkeypatch: pytest.MonkeyPatch, identity: bool
) -> None:
    _install(monkeypatch)
    upstream = FilesUpstream()
    config = _config(azure=_identity(AZURE)) if identity else Config()
    path = "/openai/deployments/img/images/edits?api-version=1" if identity else "/v1/images/edits"
    image = _part(
        b'Content-Disposition: form-data; name="image"; filename="'
        + EMAIL.encode()
        + b' photo.png"\r\nContent-Type: image/png',
        PNG,
    )
    body = _form(_part(b'Content-Disposition: form-data; name="prompt"', b"brighter"), image)
    async with _client(_app(config, upstream)) as client:
        response = await client.post(path, content=body, headers={**_headers(), **AUTH})
    assert response.status_code == 200
    assert upstream.requests[0].content == body.replace(EMAIL.encode(), TOKEN.encode())


def test_file_routes_keep_no_system_note_and_object_tracking() -> None:
    openai, azure = OpenAIAdapter(), AzureOpenAIAdapter()
    custom = CustomOpenAIAdapter("vllm")
    for adapter, path in (
        (openai, "/v1/files/file-1"),
        (openai, "/v1/files/file-1/content"),
        (azure, "/openai/files/file-1"),
        (azure, "/openai/v1/files/file-1"),
        (custom, "/custom/vllm/v1/files/file-1"),
    ):
        assert not adapter.wants_system_note(RouteKind.CHAT, path), path
    # The upload is still the tracked create; reads never are.
    assert openai.tracks_object_ids("POST", "/v1/files")
    assert azure.tracks_object_ids("POST", "/openai/files")
    assert not openai.tracks_object_ids("GET", "/v1/files")
    assert not openai.tracks_object_ids("GET", "/v1/files/file-1")


async def test_upload_reports_the_created_file_and_restores_in_its_own_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A user-scoping session router (llm-redact-pro's named users): the
    # upload's filename is redacted AND restored in the session it resolves,
    # and the created file id still reaches the router with that session.
    router = OwnershipRouter()
    _registry(monkeypatch, build_session_router=lambda config, **kw: router)
    upstream = FilesUpstream()
    config = dataclasses.replace(
        Config(), vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db"))
    )
    body = _form(PURPOSE, _jsonl_part(b'filename="' + FILENAME.encode() + b'"'))
    async with _client(_app(config, upstream)) as client:
        uploaded = await client.post("/v1/files", content=body, headers={**_headers(), **AUTH})
    assert EMAIL.encode() not in upstream.requests[0].content
    assert uploaded.json()["filename"] == FILENAME
    assert router.objects == [("file-1", "user:n1:main")]


async def test_filename_star_alone_now_routes_the_part_as_a_file() -> None:
    # A filename*-only part (urllib3 < 2 sends one for a non-ASCII name) was
    # read as a plain field, so its JSONL lines went out unredacted.
    upstream = FilesUpstream()
    line = b'{"custom_id": "a", "body": {"q": "' + EMAIL.encode() + b'"}}'
    body = _form(PURPOSE, _jsonl_part(b"filename*=UTF-8''donn%C3%A9es.jsonl", content=line))
    async with _client(_app(Config(), upstream)) as client:
        await client.post("/v1/files", content=body, headers={**_headers(), **AUTH})
    sent = upstream.requests[0].content
    assert EMAIL.encode() not in sent and TOKEN.encode() in sent


async def test_block_mode_on_a_filename_rejects_the_upload() -> None:
    upstream = FilesUpstream()
    config = dataclasses.replace(Config(), detection=DetectionConfig(modes=(("email", "block"),)))
    body = _form(PURPOSE, _jsonl_part(b'filename="' + EMAIL.encode() + b'.jsonl"'))
    async with _client(_app(config, upstream)) as client:
        response = await client.post("/v1/files", content=body, headers={**_headers(), **AUTH})
    assert response.status_code == 400 and upstream.requests == []


# --- identity refusals, each pinned verbatim under key auth -------------------------------

AMBIGUOUS = "a multipart part header cannot be parsed unambiguously"
NOT_UTF8 = "a multipart filename* is not UTF-8"
CTE = "a multipart part declares a Content-Transfer-Encoding llm-redact does not decode"
CHARSET = "a multipart part declares a charset llm-redact does not decode"

FIELD_HEAD = b'Content-Disposition: form-data; name="purpose"'
REFUSED: dict[str, tuple[bytes, str]] = {
    # Filenames without a single reading.
    "bare-backslash": (
        _form(PURPOSE, _jsonl_part(b'filename="C:\\data\\' + EMAIL.encode() + b'.jsonl"')),
        AMBIGUOUS,
    ),
    "folded": (
        _form(
            PURPOSE,
            _part(
                b'Content-Disposition: form-data; name="file";\r\n filename="'
                + EMAIL.encode()
                + b'.jsonl"',
                LINE,
            ),
        ),
        AMBIGUOUS,
    ),
    "repeated-filename": (
        _form(PURPOSE, _jsonl_part(b'filename="a.jsonl"; filename="' + EMAIL.encode() + b'"')),
        AMBIGUOUS,
    ),
    "repeated-disposition": (
        _form(
            PURPOSE,
            _part(
                b'Content-Disposition: form-data; name="file"; filename="a.jsonl"\r\n'
                b'Content-Disposition: form-data; name="file"; filename="' + EMAIL.encode() + b'"',
                LINE,
            ),
        ),
        AMBIGUOUS,
    ),
    "latin1-filename-star": (
        _form(PURPOSE, _jsonl_part(b"filename*=iso-8859-1''" + EMAIL.replace("@", "%40").encode())),
        NOT_UTF8,
    ),
    # Encodings the proxy does not decode.
    "cte-jsonl": (
        _form(
            PURPOSE,
            _jsonl_part(b'filename="a.jsonl"', extra=b"\r\nContent-Transfer-Encoding: base64"),
        ),
        CTE,
    ),
    "cte-field": (
        _form(_part(FIELD_HEAD + b"\r\nContent-Transfer-Encoding: quoted-printable", b"batch")),
        CTE,
    ),
    "cte-media": (
        _form(
            _part(b'Content-Disposition: form-data; name="prompt"', b"brighter"),
            _part(
                b'Content-Disposition: form-data; name="image"; filename="i.png"\r\n'
                b"Content-Transfer-Encoding: base64",
                b"iVBORw0KGgo=",
            ),
        ),
        CTE,
    ),
    "charset-field": (
        _form(_part(FIELD_HEAD + b"\r\nContent-Type: text/plain; charset=utf-16", b"batch")),
        CHARSET,
    ),
    "charset-jsonl": (
        _form(
            PURPOSE,
            _jsonl_part(b'filename="a.jsonl"', extra=b'; charset="ISO-8859-1"'),
        ),
        CHARSET,
    ),
    "charset-field-default": (
        _form(PURPOSE, _part(b'Content-Disposition: form-data; name="_charset_"', b"iso-8859-1")),
        CHARSET,
    ),
    "malformed-content-type": (
        _form(_part(FIELD_HEAD + b"\r\nContent-Type: text/plain; charset", b"batch")),
        AMBIGUOUS,
    ),
}


def _route(case: str, identity: bool) -> str:
    media = case == "cte-media"
    if identity:
        return "/openai/deployments/img/images/edits?api-version=1" if media else "/openai/files"
    return "/v1/images/edits" if media else "/v1/files"


@pytest.mark.parametrize("case", sorted(REFUSED))
async def test_identity_refuses_what_it_cannot_read(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    _, built = _install(monkeypatch)
    upstream = FilesUpstream()
    body, kind = REFUSED[case]
    async with _client(_app(_config(azure=_identity(AZURE)), upstream)) as client:
        response = await client.post(_route(case, True), content=body, headers=_headers())
    assert response.status_code == 400
    assert upstream.requests == [] and built[0].calls == []
    message = response.json()["error"]["message"]
    assert message == (
        f"llm-redact: {kind}, and this provider is authorized with the proxy's own identity;"
        " the request was not forwarded"
    )


@pytest.mark.parametrize("case", sorted(REFUSED))
async def test_key_auth_refuses_the_same_uploads(case: str) -> None:
    upstream = FilesUpstream()
    body, kind = REFUSED[case]
    async with _client(_app(Config(), upstream)) as client:
        response = await client.post(
            _route(case, False), content=body, headers={**_headers(), **AUTH}
        )
    assert response.status_code == 400 and upstream.requests == []
    assert response.json()["error"]["message"] == (
        f"llm-redact: {kind}, and on this route llm-redact forwards only bodies it has"
        " redacted; the request was not forwarded"
    )


@pytest.mark.parametrize("case", sorted(REFUSED))
async def test_detection_off_forwards_the_same_uploads_as_sent(case: str) -> None:
    upstream = FilesUpstream()
    body, _ = REFUSED[case]
    off = ProviderConfig("https://api.openai.com", detection=False)
    config = Config(providers={**Config().providers, "openai": off})
    async with _client(_app(config, upstream)) as client:
        response = await client.post(
            _route(case, False), content=body, headers={**_headers(), **AUTH}
        )
    assert response.status_code == 200
    assert upstream.requests[0].content == body


async def test_an_ambiguous_filename_refuses_the_whole_upload_under_key_auth() -> None:
    # The lines would be scanned, but the part header has no single
    # reading: the upstream could file the content under another name.
    upstream = FilesUpstream()
    line = b'{"custom_id": "a", "body": {"q": "' + EMAIL.encode() + b'"}}'
    tail = b'filename="C:\\data\\' + EMAIL.encode() + b'.jsonl"'
    body = _form(PURPOSE, _jsonl_part(tail, content=line))
    async with _client(_app(Config(), upstream)) as client:
        response = await client.post("/v1/files", content=body, headers={**_headers(), **AUTH})
    assert response.status_code == 400 and upstream.requests == []
    assert EMAIL not in response.text


@pytest.mark.parametrize(
    ("extra", "content_type"),
    [
        (b"\r\nContent-Transfer-Encoding: 7bit", b"application/jsonl"),
        (b"\r\nContent-Transfer-Encoding: 8BIT", b"application/jsonl; charset=UTF-8"),
        (b"\r\nContent-Transfer-Encoding: binary", b'application/jsonl; charset="us-ascii"'),
    ],
)
async def test_identity_accepts_plain_declarations(
    monkeypatch: pytest.MonkeyPatch, extra: bytes, content_type: bytes
) -> None:
    _install(monkeypatch)
    upstream = FilesUpstream()
    part = _part(
        b'Content-Disposition: form-data; name="file"; filename="a.jsonl"\r\nContent-Type: '
        + content_type
        + extra,
        LINE,
    )
    body = _form(_part(b'Content-Disposition: form-data; name="_charset_"', b" UTF-8 "), part)
    async with _client(_app(_config(azure=_identity(AZURE)), upstream)) as client:
        response = await client.post("/openai/files", content=body, headers=_headers())
    assert response.status_code == 200
    assert upstream.requests[0].content == body


async def test_identity_ignores_a_charset_on_media_it_never_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    upstream = FilesUpstream()
    body = _form(
        _part(b'Content-Disposition: form-data; name="prompt"', b"brighter"),
        _part(
            b'Content-Disposition: form-data; name="image"; filename="i.png"\r\n'
            b"Content-Type: image/png; charset=utf-16",
            PNG,
        ),
    )
    async with _client(_app(_config(azure=_identity(AZURE)), upstream)) as client:
        response = await client.post(
            "/openai/deployments/img/images/edits?api-version=1", content=body, headers=_headers()
        )
    assert response.status_code == 200 and upstream.requests[0].content == body
