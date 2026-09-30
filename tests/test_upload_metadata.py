"""The Gemini API's single-request upload shows the stored-object check the
file's METADATA — what the provider reads as the create's body.

``POST /upload/v1beta/files`` with the multipart protocol is a
``multipart/related`` body: the file's JSON metadata
(``{"file": {"name"?, "displayName"?, …}}``), then its media. The metadata
can CHOOSE the file's name, and a Gemini API file name can be issued twice
(a deleted or expired file's name is free again), so a session router's
ownership check (``object_access_refusal``, llm-redact-pro's named users)
must see it exactly as it sees the metadata-only JSON create's body — before
anything is sent, under the client's own key and a credential the proxy
holds alike:

- the check is handed the metadata OBJECT (``upload_view
  .read_upload_metadata``: the first part, read like a JSON body — strict
  UTF-8, ``loads_request``, ``MAX_JSON_DEPTH``); the media is not read;
- a repeated key is read as its LAST occurrence and the part is sent
  re-serialized — exactly what was checked, never an earlier occurrence a
  first-wins provider would act on (with ``detection = false`` too); a
  re-serialization that would change how the part is redacted is refused;
- metadata the check cannot read is refused under a credential the proxy
  holds and, as a JSON body the proxy cannot read is, wherever redaction
  applies; only with the client's own key and ``detection = false`` does it
  go out unchecked (as such a JSON body does).

Keyless: scripted fakes on a bare Registry stand in for llm-redact-pro.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import pytest

from fake_router import install, routed_config
from lent_routes import LentRouter
from llm_redact.config import Config, ProviderConfig
from llm_redact.jsonwalk import MAX_JSON_DEPTH
from llm_redact.providers.gemini import GeminiAdapter
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.proxy import create_app
from llm_redact.registry import Registry
from llm_redact.upload_view import (
    AMBIGUOUS,
    CHARSET,
    METADATA_NOT_OBJECT,
    METADATA_NOT_TEXT,
    METADATA_TYPE,
    OUTSIDE_GRAMMAR,
    TOO_DEEP,
    TRANSFER_ENCODED,
    UploadView,
    read_upload_metadata,
)
from test_object_access_seams import _registry

GEMINI = "https://gemini.test"
UPLOAD = "/upload/v1beta/files"
KEY = {"x-goog-api-key": "AIza-client"}
MULTIPART = {"x-goog-upload-protocol": "multipart", "content-type": "multipart/related; boundary=b"}
EMAIL = "jane.doe@corp.example"
TAKEN = "files/ada-notes"
SESSION = "user:n1:main"
REFUSAL = "llm-redact: that file name belongs to another user"
MEDIA = ("text/plain", b"hello")


def _part(head: bytes | None, content: bytes) -> bytes:
    """One body part, assembled longhand (``head`` None: no header block)."""
    if head is None:
        return b"--b\r\n" + content + b"\r\n"
    return b"--b\r\n" + head + b"\r\n\r\n" + content + b"\r\n"


def _related(*parts: bytes) -> bytes:
    return b"".join(parts) + b"--b--\r\n"


def _upload(metadata: bytes, head: bytes = b"Content-Type: application/json") -> bytes:
    return _related(_part(head, metadata), _part(b"Content-Type: text/plain", b"hello"))


def _meta(**file: Any) -> bytes:
    return json.dumps({"file": file}).encode()


def _read(body: bytes) -> UploadView:
    return read_upload_metadata(body, b"b")


# --- the adapter hook ----------------------------------------------------------------


def test_only_the_gemini_upload_route_reads_a_metadata_part() -> None:
    gemini = GeminiAdapter()
    related = 'multipart/related; boundary="b"'
    assert gemini.upload_metadata_boundary(UPLOAD, related) == b"b"
    typed = 'multipart/related; type="application/json"; boundary=b'
    assert gemini.upload_metadata_boundary(UPLOAD, typed) == b"b"
    assert gemini.upload_metadata_boundary(UPLOAD, "multipart/form-data; boundary=b") is None
    assert gemini.upload_metadata_boundary("/v1beta/files", related) is None
    assert OpenAIAdapter().upload_metadata_boundary("/v1/files", related) is None


# --- the reader ----------------------------------------------------------------------


def test_the_first_part_is_read_as_the_metadata_object() -> None:
    metadata = {"file": {"name": TAKEN, "displayName": f"notes of {EMAIL}"}}
    # A JSON second part is media, never the metadata.
    body = _related(
        _part(b"Content-Type: application/json; charset=UTF-8", json.dumps(metadata).encode()),
        _part(b"Content-Type: application/json", _meta(name="files/other")),
    )
    assert _read(body) == UploadView(metadata)


@pytest.mark.parametrize(
    ("body", "metadata"),
    [
        pytest.param(b"--b--\r\n", {}, id="no-part"),
        pytest.param(_upload(b""), {}, id="empty"),
        pytest.param(_upload(b" \r\n\t "), {}, id="blank"),
        pytest.param(_upload(b"\xef\xbb\xbf \n"), {}, id="byte-order-mark-only"),
        pytest.param(_upload(b'\n  {"file": {}}  \r\n'), {"file": {}}, id="whitespace"),
        pytest.param(
            _upload(b'\xef\xbb\xbf {"file": {"name": "files/a"}}'),
            {"file": {"name": "files/a"}},
            id="byte-order-mark",
        ),
        pytest.param(
            _upload(b' \xef\xbb\xbf{"file": {}}\n'),
            {"file": {}},
            id="whitespace-then-byte-order-mark",
        ),
        pytest.param(
            # An EMPTY header block (the part opens with CRLF): every reader
            # finds it there.
            _related(
                _part(None, b"\r\n" + _meta(name=TAKEN)), _part(b"Content-Type: text/plain", b"x")
            ),
            {"file": {"name": TAKEN}},
            id="empty-header-block",
        ),
        pytest.param(
            _related(_part(None, b""), _part(b"Content-Type: text/plain", b"x")),
            {},
            id="empty-part",
        ),
        pytest.param(
            _upload(_meta(name=TAKEN), b"X-Other: 1"),
            {"file": {"name": TAKEN}},
            id="no-declared-type",
        ),
        pytest.param(
            _upload(_meta(name=TAKEN), b"Content-Type: Application/JSON ; charset=UTF-8; x=y"),
            {"file": {"name": TAKEN}},
            id="declared-json-with-parameters",
        ),
        pytest.param(
            _upload(
                b'{"file": {}}',
                b"Content-Type: application/json; charset=US-ASCII\r\n"
                b"Content-Transfer-Encoding: 8BIT",
            ),
            {"file": {}},
            id="plain-encodings",
        ),
    ],
)
def test_what_the_metadata_part_holds(body: bytes, metadata: dict[str, Any]) -> None:
    # Declared JSON, or no type at all: the provider finds it by position.
    assert _read(body) == UploadView(metadata)


def test_a_repeated_key_is_read_as_its_last_occurrence_and_sent_so() -> None:
    repeated = b'{"file": {"name": "' + TAKEN.encode() + b'"}, "file": {"displayName": "\xc2\xab"}}'
    body = _upload(repeated)
    view = _read(body)
    assert view.cited == {"file": {"displayName": "«"}}
    # Only the metadata part changed: the media is the upload's own bytes.
    assert view.normalized == body.replace(repeated, '{"file": {"displayName": "«"}}'.encode())


REPEATED = b'{"file": {"displayName": "x"}, "file": {"displayName": "y"}}'
JSON_HEAD = b"Content-Type: application/json"
LAST = b'{"file": {"displayName": "y"}}'


@pytest.mark.parametrize(
    ("head", "before", "after"),
    [
        # An EMPTY header block: the part opens with CRLF, which must stay —
        # without it a reader would take the JSON for a header line.
        pytest.param(None, b"\r\n", b"", id="empty-header-block"),
        pytest.param(JSON_HEAD, b"", b"", id="bare"),
        pytest.param(JSON_HEAD, b" \r\n\xef\xbb\xbf", b"\r\n \t", id="whitespace-and-bom"),
        pytest.param(JSON_HEAD, b"\t", b"\n", id="whitespace"),
    ],
)
def test_only_the_json_text_of_a_repeated_key_part_is_rewritten(
    head: bytes | None, before: bytes, after: bytes
) -> None:
    media = _part(b"Content-Type: text/plain", b"hello")
    view = _read(_related(_part(head, before + REPEATED + after), media))
    assert view.cited == {"file": {"displayName": "y"}}
    assert view.normalized == _related(_part(head, before + LAST + after), media)


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        pytest.param(b"--b\nlf only\n--b--", OUTSIDE_GRAMMAR, id="outside-the-grammar"),
        pytest.param(
            _upload(b"{}", b"Content-Type: application/json\r\nContent-Type: text/plain"),
            AMBIGUOUS,
            id="two-content-types",
        ),
        pytest.param(
            _upload(b"e30=", b"Content-Transfer-Encoding: base64"),
            TRANSFER_ENCODED,
            id="transfer-encoded",
        ),
        pytest.param(
            _upload(
                '{"file": {}}'.encode("utf-16"),
                b"Content-Type: application/json; charset=utf-16",
            ),
            CHARSET,
            id="charset",
        ),
        pytest.param(_upload(b'{"file": {"name": "\xff"}}'), METADATA_NOT_TEXT, id="not-utf8"),
        pytest.param(
            _upload(b"\xff\xfe" + '{"file": {}}'.encode("utf-16-le")),
            METADATA_NOT_TEXT,
            id="utf-16",
        ),
        pytest.param(
            _upload(b"{file: {name: 'files/ada-notes'}}"), METADATA_NOT_OBJECT, id="lenient-json"
        ),
        pytest.param(_upload(b"hello", b"X-Other: 1"), METADATA_NOT_OBJECT, id="not-json"),
        pytest.param(
            _upload(_meta(name=TAKEN), b"Content-Type: text/plain"),
            METADATA_TYPE,
            id="declared-as-text",
        ),
        pytest.param(
            # Strict JSON AND a form-encoded body naming file.name: a server
            # parsing by the declared type reads the name.
            _upload(
                _meta(displayName=f"x&file.name={TAKEN}&y"),
                b"Content-Type: application/x-www-form-urlencoded",
            ),
            METADATA_TYPE,
            id="declared-form-encoded",
        ),
        pytest.param(
            _upload(b'{"file": {}}', b"Content-Type: application/x-protobuf"),
            METADATA_TYPE,
            id="declared-protobuf",
        ),
        pytest.param(
            _upload(b'{"file": {}}', b"Content-Type: application/jsonl"),
            METADATA_TYPE,
            id="declared-jsonl",
        ),
        pytest.param(_upload(b'[{"file": {}}]'), METADATA_NOT_OBJECT, id="array"),
        pytest.param(_upload(b'{"file": {}} {}'), METADATA_NOT_OBJECT, id="trailing-bytes"),
        pytest.param(
            _upload(b'{"a": ' * (MAX_JSON_DEPTH + 1) + b"1" + b"}" * (MAX_JSON_DEPTH + 1)),
            TOO_DEEP,
            id="too-deep",
        ),
    ],
)
def test_what_the_check_cannot_read_is_named(body: bytes, problem: str) -> None:
    assert _read(body) == UploadView(None, problem=problem)


# A header block a reader accepting a bare LF (or CR) as a line break ends
# EARLIER than the check does: what the check reads as a header value (the
# part's content blank) would there be the metadata — a name nobody checked.
HIDDEN = _meta(name=TAKEN, displayName=f"of {EMAIL}")
HIDDEN_HEADS = {
    "lf-blank-line": b"Content-Type: application/json\n\n" + HIDDEN,
    "cr-blank-line": b"Content-Type: application/json\r\r" + HIDDEN,
    "crlf-then-lf": b"Content-Type: application/json\r\n\n" + HIDDEN,
    "nul": b"Content-Type: application/json\x00" + HIDDEN,
}


@pytest.mark.parametrize("head", HIDDEN_HEADS.values(), ids=HIDDEN_HEADS.keys())
def test_a_control_in_the_metadata_header_block_is_unreadable(head: bytes) -> None:
    assert _read(_upload(b"", head)) == UploadView(None, problem=AMBIGUOUS)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(_meta(name=TAKEN), id="json"),
        pytest.param(b"Content-Type: application/json\n\n" + _meta(name=TAKEN), id="lf-headers"),
        pytest.param(b"\n" + _meta(name=TAKEN), id="lf"),
    ],
)
def test_a_part_without_a_header_block_or_an_empty_one_is_unreadable(content: bytes) -> None:
    # No header/body separator and no leading CRLF: a strict reader takes
    # its first lines for headers, a lenient one may end them at a bare-LF
    # blank line — neither reads the part where the check would.
    body = _related(_part(None, content), _part(b"Content-Type: text/plain", b"x"))
    assert _read(body) == UploadView(None, problem=AMBIGUOUS)
    # The MEDIA too (every part): its content is not read, but a reader
    # finding a transfer encoding in it decodes bytes nobody read.
    media = _related(_part(JSON_HEAD, _meta(name="files/a")), _part(None, content))
    assert _read(media) == UploadView(None, problem=AMBIGUOUS)


@pytest.mark.parametrize("media", [b"", b"\r\nhello"], ids=["empty", "empty-header-block"])
def test_a_media_part_every_reader_finds_empty_headers_in_is_read_past(media: bytes) -> None:
    body = _related(_part(JSON_HEAD, _meta(name="files/a")), _part(None, media))
    assert _read(body) == UploadView({"file": {"name": "files/a"}})


# --- through the real app -------------------------------------------------------------


class NameRouter:
    """A per-user session router whose ownership check refuses a file
    create choosing ``TAKEN`` (llm-redact-pro's shape: the body's
    ``file.name``), recording what it was shown."""

    mode = "per-user"

    def __init__(self) -> None:
        self.checks: list[tuple[str | None, str, str, Any, bool]] = []

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return SESSION

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def record_object_id(self, object_id: str, session_id: str) -> bool:
        return True

    def object_access_refusal(
        self, adapter_name: str | None, method: str, path: str, body: Any, *, identity: bool
    ) -> str | None:
        self.checks.append((adapter_name, method, path, copy.deepcopy(body), identity))
        file = body.get("file") if isinstance(body, dict) else None
        return REFUSAL if isinstance(file, dict) and file.get("name") == TAKEN else None


class FilesAPI:
    """The Gemini upload, answering with the file its metadata part names."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        metadata = request.content.split(b"\r\n\r\n", 1)[1].split(b"\r\n--b")[0]
        try:
            file = json.loads(metadata)["file"]
        except ValueError:  # a lenient reading's metadata: not the fake's
            file = {}
        return httpx.Response(200, json={"file": {"name": "files/new", **file}})

    def metadata(self) -> bytes:
        (sent,) = self.requests
        return sent.content.split(b"\r\n\r\n", 1)[1].split(b"\r\n--b")[0]


CREDENTIALS = [
    pytest.param(False, id="own-key"),
    pytest.param(True, id="operator-key"),
]
DETECTION = [pytest.param(True, id="redacted"), pytest.param(False, id="detection-off")]


def _app(
    monkeypatch: pytest.MonkeyPatch,
    router: Any,
    upstream: FilesAPI,
    *,
    operator: bool,
    detection: bool = True,
) -> Any:
    """The real app, Gemini configured: with the client's own key
    (unrouted), or routed with a plan lending the proxy's credential."""
    gemini = ProviderConfig(GEMINI, detection=detection)
    if operator:
        reg, _ = install(monkeypatch, LentRouter(GEMINI))
        _registry(monkeypatch, router, reg)
        config = routed_config(providers={**Config().providers, "gemini": gemini})
    else:
        _registry(monkeypatch, router, Registry())
        config = Config(providers={**Config().providers, "gemini": gemini})
    return create_app(config, upstream_transport=httpx.MockTransport(upstream))


async def _post(app: Any, body: bytes) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        return await client.post(UPLOAD, content=body, headers={**KEY, **MULTIPART})


@pytest.mark.parametrize("detection", DETECTION)
@pytest.mark.parametrize("operator", CREDENTIALS)
async def test_the_check_is_shown_the_metadata_before_anything_is_sent(
    operator: bool, detection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = NameRouter()
    upstream = FilesAPI()
    app = _app(monkeypatch, router, upstream, operator=operator, detection=detection)
    response = await _post(app, _upload(_meta(displayName=f"notes of {EMAIL}")))
    assert response.status_code == 200, response.text
    # The metadata object, exactly as sent (before redaction), once.
    assert router.checks == [
        ("gemini", "POST", UPLOAD, {"file": {"displayName": f"notes of {EMAIL}"}}, operator)
    ]
    sent = json.loads(upstream.metadata())
    assert (sent["file"]["displayName"] == f"notes of {EMAIL}") is not detection
    assert response.json()["file"]["displayName"] == f"notes of {EMAIL}"


@pytest.mark.parametrize("detection", DETECTION)
@pytest.mark.parametrize("operator", CREDENTIALS)
async def test_a_chosen_name_the_check_refuses_is_never_sent(
    operator: bool, detection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = NameRouter()
    upstream = FilesAPI()
    app = _app(monkeypatch, router, upstream, operator=operator, detection=detection)
    response = await _post(app, _upload(_meta(name=TAKEN, displayName=f"of {EMAIL}")))
    assert response.status_code == 403
    assert REFUSAL in response.json()["error"]["message"]
    assert upstream.requests == []
    assert app.state.proxy.vault_manager.total_entries() == 0  # nothing redacted
    (row,) = app.state.proxy.recent
    assert row["status"] == 403 and row["provider"] == "gemini"


@pytest.mark.parametrize("detection", DETECTION)
@pytest.mark.parametrize("operator", CREDENTIALS)
@pytest.mark.parametrize(
    ("repeated", "checked", "status"),
    [
        pytest.param(
            b'{"file": {"name": "' + TAKEN.encode() + b'"}, "file": {"displayName": "x"}}',
            {"file": {"displayName": "x"}},
            200,
            id="earlier-name",
        ),
        pytest.param(
            b'{"file": {"displayName": "x"}, "file": {"name": "' + TAKEN.encode() + b'"}}',
            {"file": {"name": TAKEN}},
            403,
            id="later-name",
        ),
    ],
)
async def test_a_repeated_key_is_sent_exactly_as_it_was_checked(
    repeated: bytes,
    checked: dict[str, Any],
    status: int,
    operator: bool,
    detection: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = NameRouter()
    upstream = FilesAPI()
    app = _app(monkeypatch, router, upstream, operator=operator, detection=detection)
    response = await _post(app, _upload(repeated))
    assert response.status_code == status, response.text
    assert [check[3] for check in router.checks] == [checked]
    if status == 200:
        # The LAST occurrence, re-serialized: the name the check never saw
        # is not sent — redacted or not.
        assert json.loads(upstream.metadata()) == checked
        assert TAKEN.encode() not in upstream.requests[0].content
    else:
        assert upstream.requests == []


@pytest.mark.parametrize("detection", DETECTION)
@pytest.mark.parametrize("operator", CREDENTIALS)
async def test_a_repeated_key_part_with_an_empty_header_block_keeps_it(
    operator: bool, detection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sent exactly as checked: the part still opens with its empty header
    # block, so every reader finds the metadata as its CONTENT (without the
    # CRLF a reader takes the JSON for a malformed header line).
    router = NameRouter()
    upstream = FilesAPI()
    app = _app(monkeypatch, router, upstream, operator=operator, detection=detection)
    body = _related(_part(None, b"\r\n" + REPEATED), _part(b"Content-Type: text/plain", b"hello"))
    response = await _post(app, body)
    assert response.status_code == 200, response.text
    assert [check[3] for check in router.checks] == [{"file": {"displayName": "y"}}]
    (sent,) = upstream.requests
    assert sent.content.startswith(b"--b\r\n\r\n" + LAST + b"\r\n--b\r\n")


@pytest.mark.parametrize("detection", DETECTION)
@pytest.mark.parametrize("operator", CREDENTIALS)
async def test_a_repeated_key_over_several_lines_is_refused(
    operator: bool, detection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Re-serialized, a pretty-printed metadata part becomes one JSON line:
    # redaction would read it as another kind of part than the one sent, so
    # it is refused whatever the credential — never sent with the earlier
    # occurrence the check did not read.
    router = NameRouter()
    upstream = FilesAPI()
    app = _app(monkeypatch, router, upstream, operator=operator, detection=detection)
    pretty = b'{\n  "file": {"name": "' + TAKEN.encode() + b'"},\n  "file": {}\n}\n'
    response = await _post(app, _upload(pretty))
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "changed what a file part is" in message and "credential" not in message
    assert upstream.requests == [] and router.checks == []


JSON = b"Content-Type: application/json"
UNREADABLE = [
    pytest.param(
        b"{file: {name: 'files/ada-notes'}}", JSON, METADATA_NOT_OBJECT, id="lenient-json"
    ),
    pytest.param(b"\xff\xfe{\x00}\x00", JSON, METADATA_NOT_TEXT, id="utf-16"),
    pytest.param(
        _meta(name=TAKEN).decode().encode("utf-16"),
        JSON + b"; charset=utf-16",
        CHARSET,
        id="utf-16-declared",
    ),
    pytest.param(
        b'{"a": ' * (MAX_JSON_DEPTH + 1) + b"1" + b"}" * (MAX_JSON_DEPTH + 1),
        JSON,
        TOO_DEEP,
        id="too-deep",
    ),
    # The media first, no metadata part: the provider reads the first part
    # as the metadata.
    pytest.param(b"hello", b"Content-Type: text/plain", METADATA_TYPE, id="media-first"),
    pytest.param(b"hello", b"X-Other: 1", METADATA_NOT_OBJECT, id="untyped-media-first"),
    pytest.param(
        _meta(displayName=f"x&file.name={TAKEN}&y"),
        b"Content-Type: application/x-www-form-urlencoded",
        METADATA_TYPE,
        id="form-encoded",
    ),
]


@pytest.mark.parametrize("detection", DETECTION)
@pytest.mark.parametrize(("metadata", "head", "problem"), UNREADABLE)
async def test_metadata_the_check_cannot_read_is_never_sent_with_the_proxys_credential(
    metadata: bytes, head: bytes, problem: str, detection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = NameRouter()
    upstream = FilesAPI()
    app = _app(monkeypatch, router, upstream, operator=True, detection=detection)
    response = await _post(app, _upload(metadata, head))
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert problem in message and "the proxy's own provider credential" in message
    assert "ada-notes" not in response.text  # the construct only, never content
    assert upstream.requests == [] and router.checks == []
    (row,) = app.state.proxy.recent
    assert row["status"] == 400


@pytest.mark.parametrize(("metadata", "head", "problem"), UNREADABLE)
async def test_with_the_clients_own_key_unreadable_metadata_is_refused_where_redaction_applies(
    metadata: bytes, head: bytes, problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As a JSON body the proxy cannot read is (the scanned-body rule): a
    provider reading it leniently could choose a file name nobody
    checked."""
    router = NameRouter()
    upstream = FilesAPI()
    app = _app(monkeypatch, router, upstream, operator=False)
    response = await _post(app, _upload(metadata, head))
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert problem in message and "a file create names must be checked" in message
    assert "credential" not in message and "ada-notes" not in response.text
    assert upstream.requests == [] and router.checks == []
    (row,) = app.state.proxy.recent
    assert row["status"] == 400


HEADLESS = _related(
    _part(None, b"Content-Type: application/json\n\n" + HIDDEN),
    _part(b"Content-Type: text/plain", b"hello"),
)
HIDDEN_BODIES = [
    *(pytest.param(_upload(b"", head), id=name) for name, head in HIDDEN_HEADS.items()),
    pytest.param(HEADLESS, id="no-header-block"),
]


@pytest.mark.parametrize("body", HIDDEN_BODIES)
@pytest.mark.parametrize(
    ("operator", "detection"),
    [
        pytest.param(False, True, id="own-key"),
        pytest.param(True, True, id="operator-key"),
        pytest.param(True, False, id="operator-key-detection-off"),
    ],
)
async def test_metadata_a_lenient_reader_finds_elsewhere_is_never_sent(
    body: bytes, operator: bool, detection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The check would be shown a blank part while an upstream accepting a
    # bare LF reads a chosen name (and an unredacted display name): refused
    # before anything is sent, the check never asked.
    router = NameRouter()
    upstream = FilesAPI()
    app = _app(monkeypatch, router, upstream, operator=operator, detection=detection)
    response = await _post(app, body)
    assert response.status_code == 400, response.text
    assert AMBIGUOUS in response.json()["error"]["message"]
    assert "ada-notes" not in response.text and EMAIL not in response.text
    assert upstream.requests == [] and router.checks == []


@pytest.mark.parametrize("head", HIDDEN_HEADS.values(), ids=HIDDEN_HEADS.keys())
async def test_without_an_ownership_check_such_a_header_block_is_refused_where_redaction_applies(
    head: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Free core: redaction reads the part's headers strictly too, so a
    # display name hidden in a header value is never sent unredacted.
    upstream = FilesAPI()
    app = _app(monkeypatch, OlderRouter(), upstream, operator=False)
    response = await _post(app, _upload(b"", head))
    assert response.status_code == 400, response.text
    assert AMBIGUOUS in response.json()["error"]["message"]
    assert upstream.requests == []


# A MEDIA part without a header block: a reader accepting a bare LF finds
# a quoted-printable transfer encoding hiding an address nobody scanned.
HEADLESS_MEDIA = _related(
    _part(JSON, _meta(displayName="notes")),
    _part(
        None,
        b"Content-Type: text/plain\nContent-Transfer-Encoding: quoted-printable\n\ncontact "
        + EMAIL.replace("@", "=40").encode(),
    ),
)


@pytest.mark.parametrize("body", [HEADLESS, HEADLESS_MEDIA], ids=["metadata", "media"])
@pytest.mark.parametrize("check", [True, False], ids=["checked", "free-core"])
async def test_a_part_without_a_header_block_is_refused_where_redaction_applies(
    body: bytes, check: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every part of the upload, whether or not a stored-object check reads
    # it: redaction once read such a part as a whole file (the metadata's
    # headers included), whatever a server finds in it.
    upstream = FilesAPI()
    app = _app(monkeypatch, NameRouter() if check else OlderRouter(), upstream, operator=False)
    response = await _post(app, body)
    assert response.status_code == 400, response.text
    assert AMBIGUOUS in response.json()["error"]["message"]
    assert upstream.requests == [] and EMAIL not in response.text


@pytest.mark.parametrize("detection", DETECTION)
async def test_a_media_part_without_a_header_block_is_never_sent_with_the_proxys_credential(
    detection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = NameRouter()
    upstream = FilesAPI()
    app = _app(monkeypatch, router, upstream, operator=True, detection=detection)
    response = await _post(app, HEADLESS_MEDIA)
    assert response.status_code == 400, response.text
    assert AMBIGUOUS in response.json()["error"]["message"]
    assert upstream.requests == [] and router.checks == []


# A multipart/related content type with more than one reading: canonical
# under "a" (benign metadata, then a text "media" part), while a reader
# taking the LAST boundary parameter — or splitting quoted parameters
# naively — parses it with "b" and finds metadata choosing TAKEN inside
# that media.
_INNER = (
    b"\r\n--b\r\nContent-Type: application/json\r\n\r\n"
    + _meta(name=TAKEN)
    + b"\r\n--b\r\nContent-Type: text/plain\r\n\r\nhi\r\n--b--"
)
TWO_BOUNDARIES = (
    b"--a\r\nContent-Type: application/json\r\n\r\n"
    + _meta(displayName="x")
    + b"\r\n--a\r\nContent-Type: text/plain\r\n\r\nmedia"
    + _INNER
    + b"\r\n--a--\r\n"
)


@pytest.mark.parametrize(
    "content_type",
    [
        pytest.param("multipart/related; boundary=a; boundary=b", id="repeated-boundary"),
        pytest.param('multipart/related; x="; boundary=b"; boundary=a', id="naive-split"),
        pytest.param('multipart/related; boundary="a;b"', id="quoted-semicolon"),
        # RFC 2387 "start" names the root part: the metadata need not be the
        # first part, the one the check reads.
        pytest.param('multipart/related; boundary=a; start="<meta@x>"', id="start"),
        pytest.param('multipart/related; START="<meta@x>"; boundary=a', id="start-upper"),
    ],
)
@pytest.mark.parametrize(
    ("operator", "detection"),
    [
        pytest.param(False, True, id="own-key"),
        pytest.param(True, True, id="operator-key"),
        pytest.param(True, False, id="operator-key-detection-off"),
    ],
)
async def test_a_content_type_with_two_readings_is_never_sent(
    content_type: str, operator: bool, detection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = NameRouter()
    upstream = FilesAPI()
    app = _app(monkeypatch, router, upstream, operator=operator, detection=detection)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        response = await client.post(
            UPLOAD,
            content=TWO_BOUNDARIES,
            headers={**KEY, **MULTIPART, "content-type": content_type},
        )
    assert response.status_code == 400, response.text
    assert "ada-notes" not in response.text
    assert upstream.requests == [] and router.checks == []


async def test_with_the_clients_own_key_and_detection_off_it_goes_out_unchecked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The explicit opt-out: nothing is redacted, and — as with a JSON body
    # the proxy cannot parse — the check is shown no body.
    router = NameRouter()
    upstream = FilesAPI()
    app = _app(monkeypatch, router, upstream, operator=False, detection=False)
    body = _upload(b"{file: {}}")
    response = await _post(app, body)
    assert response.status_code == 200
    assert upstream.requests[0].content == body
    assert router.checks == [("gemini", "POST", UPLOAD, None, False)]


class OlderRouter:
    """A router from before ``object_access_refusal``: nothing reads the
    upload for a check."""

    mode = "per-user"

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return SESSION

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None


@pytest.mark.parametrize(("metadata", "head", "problem"), UNREADABLE)
async def test_without_an_ownership_check_unreadable_metadata_is_refused_where_redaction_applies(
    metadata: bytes, head: bytes, problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No check to show it to (the Free core, an older router): redaction
    # reads the metadata as the same JSON value, so what that reading
    # refuses is a body the proxy cannot redact — refused as the scanned-body
    # rule refuses a JSON body it cannot read (it was once redacted as a
    # text file and forwarded).
    upstream = FilesAPI()
    app = _app(monkeypatch, OlderRouter(), upstream, operator=False)
    response = await _post(app, _upload(metadata, head))
    assert response.status_code == 400, response.text
    message = response.json()["error"]["message"]
    assert problem in message and "forwards only bodies it has redacted" in message
    assert "ada-notes" not in response.text and upstream.requests == []


@pytest.mark.parametrize("check", [True, False], ids=["checked", "free-core"])
async def test_pretty_printed_metadata_is_redacted_with_its_escapes_resolved(
    check: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = FilesAPI()
    app = _app(monkeypatch, NameRouter() if check else OlderRouter(), upstream, operator=False)
    pretty = json.dumps({"file": {"displayName": f"notes of {EMAIL}"}}, indent=2)
    response = await _post(app, _upload(pretty.replace("@", "\\u0040").encode()))
    assert response.status_code == 200, response.text
    assert json.loads(upstream.metadata()) == {"file": {"displayName": "notes of «EMAIL_001»"}}
    # The answer's echo is restored in the request's session.
    assert response.json()["file"]["displayName"] == f"notes of {EMAIL}"
