"""The Gemini API's Files API: redacted, restored and tracked, so a
credential the proxy holds (a routed operator key) may reach it.

- a file's metadata (``GET /v1beta/files/<id>``) and the list echo each
  file's ``displayName``: restored (a listed file, by its ``name``, in its
  creator's session);
- delete carries the name only;
- the download (``…/files/<id>:download``, and the ``/download/v1beta/``
  media form a batch's output file is fetched from) serves the file back:
  a text file restored (a JSON or JSONL file as JSON source), a binary one
  never read;
- the metadata-only create and the SINGLE-REQUEST upload (the multipart
  protocol: ``multipart/related``, the JSON metadata then the media) are
  redacted part by part — a text file as text (JSON as JSON), a binary one
  forwarded as sent only with the client's own key and refused under a
  credential the proxy holds — and the File they answer with is restored
  and reported to the session router as its creator's;
- the RESUMABLE upload's start answers with an upload URL on Google's host,
  minted for the request's credential, to which the client sends the data
  directly — never through the proxy. With the client's own key that is
  the client's session (the start's metadata is still redacted); under a
  credential the proxy holds the start is refused (403, before anything is
  sent) and an upload-session header is never relayed. Data chunks and the
  raw protocol are no route llm-redact recognizes: pass-through with the
  client's own key, refused under the proxy's.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from lent_routes import SESSION, Owners, client, lent_app
from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors
from llm_redact.providers.base import RouteKind
from llm_redact.providers.documents import redact_related_upload, rehydrate_download
from llm_redact.providers.gemini import GeminiAdapter
from llm_redact.redactor import Redactor, UnredactableRequest
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"
OTHER = "sam.roe@corp.example"
TOKEN = "«EMAIL_001»"
GEMINI = "https://gemini.test"
KEY = {"x-goog-api-key": "AIza-client"}
UPLOAD = "/upload/v1beta/files"
MULTIPART = {"x-goog-upload-protocol": "multipart", "content-type": "multipart/related; boundary=b"}
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(range(256))


def _redactor(vault: InMemoryVault) -> Redactor:
    return Redactor(
        build_detectors(DetectionConfig(enabled=("email",))),
        vault,
        Allowlist(exact=frozenset(), patterns=()),
    )


def _related(*parts: tuple[str, bytes]) -> bytes:
    """A multipart/related body (boundary ``b``), assembled longhand."""
    out = b""
    for content_type, content in parts:
        out += b"--b\r\nContent-Type: " + content_type.encode() + b"\r\n\r\n" + content + b"\r\n"
    return out + b"--b--\r\n"


def _metadata(email: str) -> bytes:
    return json.dumps({"file": {"display_name": f"notes of {email}"}}).encode()


# The first part is the file's metadata (the create's body): a part-loop
# case goes after one.
META = ("application/json", b'{"file": {}}')


# --- routing ---------------------------------------------------------------------------


def test_files_routes_are_recognized() -> None:
    adapter = GeminiAdapter()
    for method, path, kind in (
        ("GET", "/v1beta/files", RouteKind.CHAT),
        ("GET", "/v1beta/files/abc", RouteKind.CHAT),
        ("GET", "/v1beta/files/abc:download", RouteKind.CHAT),
        ("GET", "/download/v1beta/files/abc:download", RouteKind.CHAT),
        ("DELETE", "/v1beta/files/abc", RouteKind.REDACT_ONLY),
        ("POST", "/v1beta/files", RouteKind.CHAT),
        ("POST", UPLOAD, RouteKind.CHAT),
        ("POST", "/v1beta/files:register", RouteKind.NONE),
        ("GET", "/download/v1beta/files/abc", RouteKind.NONE),
        ("DELETE", "/v1beta/files", RouteKind.NONE),
        ("GET", "/v1beta/files/abc/x", RouteKind.NONE),
        ("PATCH", "/v1beta/files/abc", RouteKind.NONE),
        ("GET", UPLOAD, RouteKind.NONE),
    ):
        assert adapter.matches(method, path) is kind, (method, path)
        assert not adapter.wants_system_note(kind, path), path


@pytest.mark.parametrize(
    ("headers", "query", "kind", "single"),
    [
        ({"x-goog-upload-protocol": "multipart"}, "", RouteKind.CHAT, True),
        ({}, "uploadType=multipart", RouteKind.CHAT, True),
        ({}, "", RouteKind.CHAT, True),
        ({}, "alt=json", RouteKind.CHAT, True),
        (
            {"x-goog-upload-protocol": "resumable", "x-goog-upload-command": "start"},
            "",
            RouteKind.CHAT,
            False,
        ),
        ({"x-goog-upload-protocol": "Resumable"}, "", RouteKind.CHAT, False),
        ({}, "uploadType=resumable", RouteKind.CHAT, False),
        ({"x-goog-upload-protocol": "multipart, resumable"}, "", RouteKind.CHAT, False),
        ({"x-goog-upload-command": "upload, finalize"}, "", RouteKind.NONE, False),
        ({"x-goog-upload-command": "query"}, "", RouteKind.NONE, False),
        ({}, "upload_id=AB12&upload_protocol=resumable", RouteKind.NONE, False),
        ({}, "Upload%5Fid=AB12", RouteKind.NONE, False),
        ({"x-goog-upload-protocol": "raw"}, "", RouteKind.NONE, False),
        ({}, "uploadType=media", RouteKind.NONE, False),
        ({}, "upload_session=x", RouteKind.CHAT, False),
    ],
)
def test_upload_protocols(
    headers: dict[str, str], query: str, kind: RouteKind, single: bool
) -> None:
    adapter = GeminiAdapter()
    assert adapter.matches_request("POST", UPLOAD, httpx.Headers(headers), query) is kind
    refusal = adapter.proxy_credential_refusal("POST", UPLOAD, httpx.Headers(headers), query)
    assert (refusal is None) is single
    # A plain mapping reads the same (no getlist).
    assert adapter.matches_request("POST", UPLOAD, headers, query) is kind


def test_only_the_upload_route_is_ever_refused_or_read_as_related() -> None:
    adapter = GeminiAdapter()
    resumable = {"x-goog-upload-protocol": "resumable"}
    assert adapter.proxy_credential_refusal("POST", "/v1beta/files", resumable, "") is None
    assert adapter.proxy_credential_refusal("GET", UPLOAD, resumable, "") is None
    assert adapter.matches_request("POST", "/v1beta/files", resumable, "upload_id=x") is (
        RouteKind.CHAT
    )
    related = 'multipart/related; boundary="b"'
    assert adapter.multipart_boundary(UPLOAD, related) == b"b"
    assert adapter.multipart_boundary(UPLOAD, "multipart/related; type=x") is None
    assert adapter.multipart_boundary(UPLOAD, "multipart/form-data; boundary=b") is None
    assert adapter.multipart_boundary("/v1beta/files", related) is None
    assert adapter.multipart_boundary("/v1beta/files", "multipart/form-data; boundary=b") == b"b"
    assert adapter.redacts_multipart(UPLOAD) and not adapter.redacts_multipart("/v1beta/files")


def test_file_listing_hooks() -> None:
    adapter = GeminiAdapter()
    assert adapter.lists_objects("GET", "/v1beta/files")
    assert not adapter.lists_objects("GET", "/v1beta/files/abc")
    listing = {"files": [{"name": "files/a"}, {"displayName": "x"}], "nextPageToken": "n"}
    items = adapter.listing_items(listing)
    assert items is listing["files"]
    assert [adapter.listing_item_id(item) for item in items] == ["files/a", None]
    assert adapter.listing_items({"files": {}}) is None


# --- the upload policy (the metadata a JSON value, every other part a file) ----------


def _related_upload(body: bytes, **kwargs: object) -> bytes | None:
    options: dict = {"require_scanned": True, "forward_binary": None, **kwargs}
    return redact_related_upload(body, b"b", _redactor(InMemoryVault()), **options)


def test_the_upload_is_redacted_part_by_part_the_media_a_file() -> None:
    vault = InMemoryVault()
    body = _related(
        ("application/json; charset=UTF-8", _metadata(EMAIL)),
        ("text/plain", f"call {OTHER}".encode()),
    )
    out = redact_related_upload(
        body, b"b", _redactor(vault), require_scanned=True, forward_binary=None
    )
    assert out is not None and b"@corp.example" not in out
    # The metadata part is the create's JSON body: redacted as that value.
    assert b'"display_name": "notes of \xc2\xabEMAIL_001\xc2\xbb"' in out
    assert out.endswith(b"call \xc2\xabEMAIL_002\xc2\xbb\r\n--b--\r\n")
    assert _related_upload(_related(META, ("text/plain", b"nothing"))) is None
    # A UTF-16 text file (byte-order mark) is redacted and re-encoded as it came.
    utf16 = "\ufeffmail " + EMAIL
    out = _related_upload(_related(META, ("text/plain", utf16.encode("utf-16-le"))))
    assert out is not None and "mail «EMAIL_001»".encode("utf-16-le") in out


@pytest.mark.parametrize("indent", [None, 2, "\t"], ids=["one-line", "pretty", "tabs"])
def test_the_metadata_part_is_redacted_as_the_json_value_the_check_reads(
    indent: int | str | None,
) -> None:
    # One reading: escapes resolved whatever the whitespace — a
    # pretty-printed metadata part was once redacted as raw TEXT, so an
    # escaped address ("\u0040") went out and the provider decoded it.
    text = json.dumps({"file": {"display_name": f"notes of {EMAIL}"}}, indent=indent)
    escaped = text.replace("@", "\\u0040").encode()
    out = _related_upload(_related(("application/json", escaped), ("text/plain", b"hi")))
    assert out is not None
    metadata = out.split(b"\r\n\r\n", 1)[1].split(b"\r\n--b", 1)[0]
    assert json.loads(metadata) == {"file": {"display_name": f"notes of {TOKEN}"}}
    # Only the JSON text is rewritten: the part keeps its header block.
    assert out.startswith(b"--b\r\nContent-Type: application/json\r\n\r\n{")


def test_the_metadata_part_changes_only_when_a_value_or_a_repeated_key_does() -> None:
    pretty = b'\xef\xbb\xbf {\n  "file": {"display_name": "notes"}\n}\r\n'
    assert _related_upload(_related(("application/json", pretty))) is None
    # A repeated key: exactly the reading (its LAST occurrence) goes out,
    # the bytes around the JSON text kept.
    repeated = b' {"file": {"name": "files/a"}, "file": {"display_name": "x"}} '
    out = _related_upload(_related(("application/json", repeated)))
    assert out == _related(("application/json", b' {"file": {"display_name": "x"}} '))
    # Keys are never redacted; every string value is, "name" included.
    keyed = json.dumps({"file": {"name": f"files/{EMAIL}", EMAIL: 1}}).encode()
    out = _related_upload(_related(("application/json", keyed)))
    assert out is not None
    metadata = out.split(b"\r\n\r\n", 1)[1].split(b"\r\n--b", 1)[0]
    assert json.loads(metadata) == {"file": {"name": f"files/{TOKEN}", EMAIL: 1}}


@pytest.mark.parametrize(
    ("first", "message"),
    [
        (("text/plain", _metadata(EMAIL)), "is not declared application/json"),
        (("application/json", b"{file: {display_name: 'x'}}"), "not a JSON object"),
        (("application/json", b'{"file": "\xff"}'), "is not UTF-8 text"),
        (("image/png", PNG), "is not declared application/json"),
        (("application/json; charset=latin-1", b"{}"), "charset"),
    ],
    ids=["declared-text", "lenient-json", "not-utf8", "media-first", "charset"],
)
def test_metadata_the_check_cannot_read_is_unredactable(
    first: tuple[str, bytes], message: str
) -> None:
    # Where every piece must be scanned, the reading the stored-object
    # check refuses is refused by redaction too (a JSON body the proxy
    # cannot read is); leniently, the part is read as a file.
    body = _related(first, ("text/plain", b"hi"))
    with pytest.raises(UnredactableRequest, match=message):
        _related_upload(body)
    lenient = _related_upload(body, require_scanned=False)  # read as a file instead
    assert lenient is None or EMAIL.encode() not in lenient


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (_related(META, ("image/png", PNG)), "is binary"),
        (_related(META, ("application/pdf", b"%PDF-1.7 all ascii")), "is binary"),
        (b"pre\r\n" + _related(("text/plain", b"x")), "preamble or epilogue"),
        (_related(("text/plain", b"x")) + b"tail", "preamble or epilogue"),
        (b"--b\r\nno closing", "outside the canonical form"),
        (
            _related(META)[:-7]
            + b"--b\r\nContent-Type: text/plain\r\nContent-Transfer-Encoding: base64\r\n\r\n"
            b"eA==\r\n--b--\r\n",
            "Content-Transfer-Encoding",
        ),
        (_related(META, ("text/plain; charset=latin-1", b"x")), "charset"),
        (
            _related(META)[:-7]
            + b'--b\r\nContent-Disposition: form-data; name="f"; filename="a\\b"\r\n\r\n'
            b"x\r\n--b--\r\n",
            "cannot be parsed unambiguously",
        ),
    ],
)
def test_an_upload_it_cannot_read_is_refused(body: bytes, message: str) -> None:
    with pytest.raises(UnredactableRequest, match=message):
        _related_upload(body)


def test_a_binary_file_goes_as_sent_only_when_allowed() -> None:
    body = _related(("application/json", _metadata(EMAIL)), ("image/png", PNG), ("x/y", PNG))
    counted: list[int] = []
    out = _related_upload(body, forward_binary=counted.append)
    assert out is not None and PNG in out and b"@corp.example" not in out
    assert counted == [2]  # told once, with the count
    plain = _related(META, ("text/plain", b"hi"))
    assert _related_upload(plain, forward_binary=counted.append) is None
    assert counted == [2]  # no binary part: never told
    # Lenient (never the proxy): nothing unreadable refuses.
    assert _related_upload(_related(("image/png", PNG)), require_scanned=False) is None
    assert _related_upload(b"junk", require_scanned=False) is None


def _named_metadata(file_name: str, metadata: bytes) -> bytes:
    """A metadata part naming a file (Content-Disposition), then a text part."""
    return (
        b"--b\r\nContent-Type: application/json\r\n"
        b'Content-Disposition: form-data; name="metadata"; filename="'
        + file_name.encode()
        + b'"\r\n\r\n'
        + metadata
        + b"\r\n"
        + _related(("text/plain", b"hi"))
    )


@pytest.mark.parametrize("metadata", [b'{"file": {}}', _metadata(EMAIL)], ids=["clean", "email"])
def test_the_metadata_parts_file_name_is_redacted(metadata: bytes) -> None:
    # The metadata part is redacted as its JSON value, not by the file
    # part loop — its file name still is, like any part's, whether or not
    # a value of the metadata changed.
    body = _named_metadata(f"from {EMAIL}", metadata)
    out = _related_upload(body)
    assert out is not None and EMAIL.encode() not in out
    assert b'filename="from \xc2\xabEMAIL_001\xc2\xbb"' in out
    assert out.endswith(_related(("text/plain", b"hi")))


def test_the_upload_floors_every_token_it_carries() -> None:
    body = _related(
        ("application/json", _metadata(EMAIL)),
        ("text/plain", "keep «EMAIL_007» as is".encode()),
    )
    out = _related_upload(body)
    assert out is not None and "«EMAIL_008»".encode() in out  # never onto 007


def test_a_download_is_restored_like_an_openai_file_download() -> None:
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", 'jane "j" doe@corp.example')
    rehydrator = Rehydrator(vault)
    jsonl = ('{"text": "to ' + token + '"}\n{"args": {"arguments": "' + token + '"}}\n').encode()
    restored = rehydrate_download(jsonl, rehydrator)
    assert restored is not None
    first, second, _ = restored.decode().split("\n")
    assert json.loads(first) == {"text": 'to jane "j" doe@corp.example'}
    # The plain JSON walk: no OpenAI ``arguments`` source override.
    assert json.loads(second) == {"args": {"arguments": 'jane "j" doe@corp.example'}}
    assert rehydrate_download(f"to {token}".encode(), rehydrator) == (
        b'to jane "j" doe@corp.example'
    )
    assert rehydrate_download(b"nothing here", rehydrator) is None
    assert rehydrate_download(PNG + token.encode(), rehydrator) is None


# --- end to end ------------------------------------------------------------------------


class FilesAPI:
    """A Gemini API Files surface storing each file as uploaded."""

    def __init__(self, *, session_url: bool = False) -> None:
        self.files: dict[str, dict[str, Any]] = {}
        self.content: dict[str, tuple[bytes, str]] = {}
        self.received: list[httpx.Request] = []
        self.session_url = session_url

    def _create(self, metadata: dict[str, Any], data: bytes, mime: str) -> httpx.Response:
        name = f"files/f{len(self.files) + 1}"
        self.files[name] = {
            "name": name,
            "displayName": metadata.get("display_name", ""),
            "mimeType": mime,
            "uri": f"{GEMINI}/v1beta/{name}",
            "state": "ACTIVE",
        }
        self.content[name] = (data, mime)
        session = {"x-goog-upload-url": f"{GEMINI}/upload?upload_id=SECRET"}
        headers = session if self.session_url else {}
        return httpx.Response(200, json={"file": self.files[name]}, headers=headers)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.received.append(request)
        path = request.url.path
        if request.method == "POST" and path == UPLOAD:
            if request.headers.get("x-goog-upload-command") == "start":
                return httpx.Response(
                    200,
                    headers={
                        "x-goog-upload-url": f"{GEMINI}{UPLOAD}?upload_id=SESSION1",
                        "x-goog-upload-status": "active",
                    },
                )
            if "upload_id" in request.url.params:
                return self._create({}, request.content, "application/octet-stream")
            boundary = request.headers["content-type"].split("boundary=")[1].encode()
            parts = request.content.split(b"--" + boundary)[1:-1]
            metadata = json.loads(parts[0].split(b"\r\n\r\n", 1)[1][:-2])["file"]
            head, data = parts[1].split(b"\r\n\r\n", 1)
            mime = head.split(b"Content-Type: ")[1].decode()
            return self._create(metadata, data[:-2], mime)
        if request.method == "GET" and path == "/v1beta/files":
            return httpx.Response(200, json={"files": list(self.files.values())})
        name = path.removeprefix("/download").removeprefix("/v1beta/").removesuffix(":download")
        if name not in self.files:
            return httpx.Response(404, json={"error": {"code": 404, "message": "not found"}})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        if path.endswith(":download"):
            data, mime = self.content[name]
            return httpx.Response(200, content=data, headers={"content-type": mime})
        return httpx.Response(200, json=self.files[name])


async def _upload(http: httpx.AsyncClient, email: str, media: tuple[str, bytes]) -> httpx.Response:
    body = _related(("application/json; charset=UTF-8", _metadata(email)), media)
    return await http.post(UPLOAD, content=body, headers={**KEY, **MULTIPART})


@pytest.mark.parametrize("proxy_credential", [True, False], ids=["operator-key", "own-key"])
async def test_a_text_file_is_served_end_to_end_under_a_routed_credential(
    proxy_credential: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = FilesAPI()
    owners = Owners(listings=["/v1beta/files"])
    app, router = lent_app(
        monkeypatch, "gemini", GEMINI, files, owners=owners, proxy_credential=proxy_credential
    )
    jsonl = json.dumps({"key": "1", "request": {"contents": [{"parts": [{"text": OTHER}]}]}})
    async with client(app) as http:
        uploaded = await _upload(http, EMAIL, ("text/plain", f"call {OTHER}".encode()))
        batch_input = await _upload(http, EMAIL, ("application/jsonl", jsonl.encode()))
        assert uploaded.status_code == 200, uploaded.text
        assert batch_input.status_code == 200, batch_input.text
        name = uploaded.json()["file"]["name"]
        metadata = await http.get(f"/v1beta/{name}", headers=KEY)
        listed = await http.get("/v1beta/files", headers=KEY)
        downloaded = await http.get(
            f"/v1beta/{name}:download", params={"alt": "media"}, headers=KEY
        )
        jsonl_back = await http.get(
            "/download/v1beta/files/f2:download", params={"alt": "media"}, headers=KEY
        )
        deleted = await http.delete(f"/v1beta/{name}", headers=KEY)

    assert all(plan.begun for plan in router.plans)  # routed, never refused
    assert all(b"@corp.example" not in r.content for r in files.received)
    assert files.content["files/f1"][0] == b"call \xc2\xabEMAIL_002\xc2\xbb"
    assert files.files["files/f1"]["displayName"] == f"notes of {TOKEN}"
    # The upload's answer, the metadata, the list and the download: restored.
    assert uploaded.json()["file"]["displayName"] == f"notes of {EMAIL}"
    assert metadata.json()["displayName"] == f"notes of {EMAIL}"
    assert [f["displayName"] for f in listed.json()["files"]] == [f"notes of {EMAIL}"] * 2
    assert downloaded.content == f"call {OTHER}".encode()
    back = json.loads(jsonl_back.content)
    back.pop("observed", None)  # the fake routing delivery marks what it observes
    assert back == json.loads(jsonl)
    assert deleted.status_code == 200
    # Each upload reported as its creator's; reads never are.
    assert owners.objects == [("files/f1", SESSION), ("files/f2", SESSION)]


async def test_a_binary_file_needs_the_clients_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for proxy_credential in (True, False):
        files = FilesAPI()
        app, _ = lent_app(monkeypatch, "gemini", GEMINI, files, proxy_credential=proxy_credential)
        async with client(app) as http:
            uploaded = await _upload(http, EMAIL, ("image/png", PNG))
            if proxy_credential:
                assert uploaded.status_code == 400, uploaded.text
                assert "is binary" in uploaded.text and files.received == []
                assert app.state.proxy.recent[0]["status"] == 400
                continue
            assert uploaded.status_code == 200, uploaded.text
            downloaded = await http.get("/v1beta/files/f1:download", headers=KEY)
        assert files.content["files/f1"][0] == PNG  # as sent
        assert b"@corp.example" not in files.received[0].content  # metadata redacted
        assert downloaded.content == PNG  # never read
        assert app.state.proxy.unscanned_uploads == {"gemini": 1}  # counted


@pytest.mark.parametrize("proxy_credential", [True, False], ids=["operator-key", "own-key"])
async def test_the_metadata_parts_file_name_never_reaches_the_provider(
    proxy_credential: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = FilesAPI()
    app, _ = lent_app(monkeypatch, "gemini", GEMINI, files, proxy_credential=proxy_credential)
    body = _named_metadata(f"from {EMAIL}", _metadata(OTHER))
    async with client(app) as http:
        uploaded = await http.post(UPLOAD, content=body, headers={**KEY, **MULTIPART})
    assert uploaded.status_code == 200, uploaded.text
    (sent,) = files.received
    assert b"@corp.example" not in sent.content
    # The file name first (EMAIL_001), then the metadata's value.
    assert b'filename="from \xc2\xabEMAIL_001\xc2\xbb"' in sent.content
    assert b'"notes of \xc2\xabEMAIL_002\xc2\xbb"' in sent.content
    assert uploaded.json()["file"]["displayName"] == f"notes of {OTHER}"


async def test_binary_uploads_refuse_holds_for_the_clients_own_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = FilesAPI()
    app, _ = lent_app(
        monkeypatch,
        "gemini",
        GEMINI,
        files,
        proxy_credential=False,
        detection=DetectionConfig(binary_uploads="refuse"),
    )
    async with client(app) as http:
        uploaded = await _upload(http, EMAIL, ("image/png", PNG))
    assert uploaded.status_code == 400 and "is binary" in uploaded.text
    assert files.received == [] and not app.state.proxy.unscanned_uploads


async def test_a_resumable_upload_is_never_started_with_the_proxys_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = {
        **KEY,
        "x-goog-upload-protocol": "resumable",
        "x-goog-upload-command": "start",
        "x-goog-upload-header-content-length": "5",
    }
    chunk = {**KEY, "x-goog-upload-command": "upload, finalize", "x-goog-upload-offset": "0"}
    files = FilesAPI()
    app, router = lent_app(monkeypatch, "gemini", GEMINI, files)
    async with client(app) as http:
        started = await http.post(UPLOAD, json={"file": {"display_name": EMAIL}}, headers=start)
        chunked = await http.post(f"{UPLOAD}?upload_id=X", content=b"hello", headers=chunk)
    assert started.status_code == 403, started.text
    assert "resumable upload" in started.json()["error"]["message"]
    assert EMAIL not in started.text
    assert chunked.status_code == 403 and "credential the proxy holds" in chunked.text
    assert files.received == [] and not any(plan.begun for plan in router.plans)
    assert [row["status"] for row in app.state.proxy.recent] == [403, 403]

    # The client's own key: the start's metadata redacted, its session URL
    # (the client's own) relayed, a data chunk passed through as sent.
    files = FilesAPI()
    app, _ = lent_app(monkeypatch, "gemini", GEMINI, files, proxy_credential=False)
    async with client(app) as http:
        started = await http.post(UPLOAD, json={"file": {"display_name": EMAIL}}, headers=start)
        chunked = await http.post(f"{UPLOAD}?upload_id=X", content=b"hello", headers=chunk)
    assert started.status_code == 200
    assert started.headers["x-goog-upload-url"].endswith("upload_id=SESSION1")
    assert json.loads(files.received[0].content) == {"file": {"display_name": TOKEN}}
    assert chunked.status_code == 200 and files.received[1].content == b"hello"


async def test_an_upload_session_header_is_never_relayed_under_the_proxys_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for proxy_credential in (True, False):
        files = FilesAPI(session_url=True)
        app, _ = lent_app(monkeypatch, "gemini", GEMINI, files, proxy_credential=proxy_credential)
        async with client(app) as http:
            uploaded = await _upload(http, EMAIL, ("text/plain", b"hello"))
        assert uploaded.status_code == 200
        assert ("x-goog-upload-url" in uploaded.headers) is not proxy_credential


async def test_a_listed_file_nobody_is_recorded_creating_keeps_its_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = FilesAPI()
    owners = Owners(listings=["/v1beta/files"])
    app, _ = lent_app(monkeypatch, "gemini", GEMINI, files, owners=owners)
    async with client(app) as http:
        assert (await _upload(http, EMAIL, ("text/plain", b"x"))).status_code == 200
        owners.objects.clear()
        listed = await http.get("/v1beta/files", headers=KEY)
    assert listed.json()["files"][0]["displayName"] == f"notes of {TOKEN}"


async def test_a_jsonl_file_served_as_json_is_restored_line_by_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``application/jsonl`` contains ``application/json``: a body that is
    not ONE JSON value falls back to the download restoration."""
    files = FilesAPI()
    app, _ = lent_app(monkeypatch, "gemini", GEMINI, files)
    jsonl = f'{{"a": "{EMAIL}"}}\n{{"b": "{EMAIL}"}}\n'
    async with client(app) as http:
        assert (await _upload(http, "x", ("application/jsonl", jsonl.encode()))).status_code == 200
        back = await http.get("/download/v1beta/files/f1:download", headers=KEY)
    assert b"\xc2\xab" in files.content["files/f1"][0]
    assert back.content.decode() == jsonl
