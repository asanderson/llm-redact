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
            # No header/body separator: a server may read it as a part with
            # no headers, so it is read too.
            _related(_part(None, _meta(name=TAKEN)), _part(b"Content-Type: text/plain", b"x")),
            {"file": {"name": TAKEN}},
            id="no-header-block",
        ),
        pytest.param(
            _upload(_meta(name=TAKEN), b"Content-Type: text/plain"),
            {"file": {"name": TAKEN}},
            id="declared-as-text",
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
    # Whatever type the part declares: the provider finds it by position.
    assert _read(body) == UploadView(metadata)


def test_a_repeated_key_is_read_as_its_last_occurrence_and_sent_so() -> None:
    repeated = b'{"file": {"name": "' + TAKEN.encode() + b'"}, "file": {"displayName": "\xc2\xab"}}'
    body = _upload(repeated)
    view = _read(body)
    assert view.cited == {"file": {"displayName": "«"}}
    # Only the metadata part changed: the media is the upload's own bytes.
    assert view.normalized == body.replace(repeated, '{"file": {"displayName": "«"}}'.encode())


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
        pytest.param(_upload(b"hello"), METADATA_NOT_OBJECT, id="media-first"),
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
    # as the metadata whatever it declares.
    pytest.param(b"hello", b"Content-Type: text/plain", METADATA_NOT_OBJECT, id="media-first"),
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


async def test_without_an_ownership_check_the_upload_is_redacted_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No check to show it to: a metadata part is redacted like any other
    # text part of the upload and forwarded (the Free core, an older router).
    upstream = FilesAPI()
    app = _app(monkeypatch, OlderRouter(), upstream, operator=False)
    lenient = f"{{file: {{displayName: '{EMAIL}'}}}}".encode()
    response = await _post(app, _upload(lenient))
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert EMAIL.encode() not in sent.content and "«EMAIL_001»".encode() in sent.content
