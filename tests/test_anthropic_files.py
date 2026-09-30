"""Anthropic's Files API (beta): redacted, restored and tracked, so a
credential the proxy holds (a routed operator key) may reach it.

Its paths are OpenAI's too (``/v1/files…``): a request carrying the
``anthropic-version`` marker alone is Anthropic's (the OpenAI adapter
declines it), one carrying another provider's marker as well is neither's.

- the upload (``POST /v1/files``, multipart/form-data: the document part)
  follows the OpenAI Files upload's content policy: a TEXT file is redacted
  as text (a JSON or JSONL file as JSON), a BINARY file (a PDF, an image) is
  forwarded as sent only with the client's own key and refuses the upload
  (400) under a credential the proxy holds; every part's file name is
  redacted;
- every file object echoes the uploaded ``filename``: the upload's answer,
  the list (``{"data": [...]}``, each item attributed to its creator's
  session by a session router that does) and a file's metadata restore it;
- a file's content (``GET /v1/files/{id}/content``: downloadable for files a
  tool created) is restored when it is text, binary never read;
- delete carries the id only.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from lent_routes import SESSION, Owners, client, lent_app
from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.base import RouteKind
from llm_redact.providers.openai import OpenAIAdapter

EMAIL = "jane.doe@corp.example"
OTHER = "sam.roe@corp.example"
TOKEN = "«EMAIL_001»"
ANTHROPIC = "https://anthropic.test"
MARKER = {"anthropic-version": "2023-06-01", "anthropic-beta": "files-api-2025-04-14"}
KEY = {**MARKER, "x-api-key": "sk-ant-client"}
PDF = b"%PDF-1.7\n\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"


@pytest.mark.parametrize(
    ("method", "path", "kind"),
    [
        ("POST", "/v1/files", RouteKind.CHAT),
        ("GET", "/v1/files", RouteKind.CHAT),
        ("GET", "/v1/files/file_1", RouteKind.CHAT),
        ("GET", "/v1/files/file_1/content", RouteKind.CHAT),
        ("DELETE", "/v1/files/file_1", RouteKind.REDACT_ONLY),
        ("DELETE", "/v1/files", RouteKind.NONE),
        ("POST", "/v1/files/file_1", RouteKind.NONE),
        ("GET", "/v1/files/file_1/content/x", RouteKind.NONE),
    ],
)
def test_files_routes_with_the_anthropic_marker(method: str, path: str, kind: RouteKind) -> None:
    anthropic = AnthropicAdapter()
    assert anthropic.matches_request(method, path, MARKER) is kind
    assert not anthropic.wants_system_note(kind, path)
    # The OpenAI adapter never claims a request carrying the marker; the
    # Anthropic one never claims a path without it, nor with a second
    # provider's marker.
    assert OpenAIAdapter().matches_request(method, path, MARKER) is RouteKind.NONE
    assert anthropic.matches_request(method, path, {}) is RouteKind.NONE
    both = {**MARKER, "openai-beta": "x"}
    assert anthropic.matches_request(method, path, both) is RouteKind.NONE
    assert anthropic.matches(method, path) is RouteKind.NONE


def test_files_hooks() -> None:
    anthropic = AnthropicAdapter()
    assert anthropic.redacts_multipart("/v1/files")
    assert not anthropic.redacts_multipart("/v1/messages")
    assert anthropic.lists_objects("GET", "/v1/files")
    assert not anthropic.lists_objects("GET", "/v1/files/file_1")
    listing = {"data": [{"id": "file_1"}, {"type": "file"}], "has_more": False}
    items = anthropic.listing_items(listing)
    assert items is listing["data"]
    assert [anthropic.listing_item_id(item) for item in items] == ["file_1", None]
    assert anthropic.listing_items({"data": "x"}) is None
    assert anthropic.rehydrate_raw_body("/v1/messages", b"x", None) is None  # type: ignore[arg-type]


class FilesAPI:
    """Anthropic's Files API, storing each document as uploaded."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, Any]] = {}
        self.content: dict[str, bytes] = {}
        self.received: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.received.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/v1/files":
            boundary = request.headers["content-type"].split("boundary=")[1].encode()
            part = request.content.split(b"--" + boundary)[1]
            head, data = part.split(b"\r\n\r\n", 1)
            filename = head.split(b'filename="')[1].split(b'"')[0].decode()
            file_id = f"file_{len(self.files) + 1}"
            self.files[file_id] = {
                "id": file_id,
                "type": "file",
                "filename": filename,
                "mime_type": "text/plain",
                "size_bytes": len(data) - 2,
                "created_at": "2026-09-30T00:00:00Z",
                "downloadable": True,
            }
            self.content[file_id] = data[:-2]
            return httpx.Response(200, json=self.files[file_id])
        if request.method == "GET" and path == "/v1/files":
            return httpx.Response(200, json={"data": list(self.files.values()), "has_more": False})
        file_id = path.split("/")[3]
        if file_id not in self.files:
            return httpx.Response(404, json={"type": "error", "error": {"type": "not_found_error"}})
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": file_id, "type": "file_deleted"})
        if path.endswith("/content"):
            return httpx.Response(
                200, content=self.content[file_id], headers={"content-type": "text/plain"}
            )
        return httpx.Response(200, json=self.files[file_id])


async def _upload(
    http: httpx.AsyncClient, filename: str, data: bytes, mime: str = "text/plain"
) -> httpx.Response:
    return await http.post("/v1/files", files={"file": (filename, data, mime)}, headers=KEY)


@pytest.mark.parametrize("proxy_credential", [True, False], ids=["operator-key", "own-key"])
async def test_a_text_file_is_served_end_to_end_under_a_routed_credential(
    proxy_credential: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = FilesAPI()
    owners = Owners(listings=["/v1/files"])
    app, router = lent_app(
        monkeypatch, "anthropic", ANTHROPIC, files, owners=owners, proxy_credential=proxy_credential
    )
    rows = "\n".join(json.dumps({"to": email}) for email in (EMAIL, OTHER))
    async with client(app) as http:
        uploaded = await _upload(http, f"{EMAIL} notes.txt", f"call {OTHER}".encode())
        jsonl = await _upload(http, "rows.jsonl", rows.encode(), "application/jsonl")
        assert uploaded.status_code == 200, uploaded.text
        assert jsonl.status_code == 200, jsonl.text
        file_id = uploaded.json()["id"]
        metadata = await http.get(f"/v1/files/{file_id}", headers=KEY)
        listed = await http.get("/v1/files", headers=KEY)
        content = await http.get(f"/v1/files/{file_id}/content", headers=KEY)
        rows_back = await http.get("/v1/files/file_2/content", headers=KEY)
        deleted = await http.delete(f"/v1/files/{file_id}", headers=KEY)

    assert all(plan.begun for plan in router.plans)  # routed, never refused
    assert all(b"@corp.example" not in r.content for r in files.received)
    assert files.files["file_1"]["filename"] == f"{TOKEN} notes.txt"
    assert files.content["file_1"] == "call «EMAIL_002»".encode()
    assert [json.loads(line) for line in files.content["file_2"].split(b"\n")] == [
        {"to": TOKEN},
        {"to": "«EMAIL_002»"},
    ]
    # The upload's answer, the metadata, the list and the content: restored.
    for echoed in (uploaded.json(), metadata.json(), listed.json()["data"][0]):
        assert echoed["filename"] == f"{EMAIL} notes.txt"
    assert content.content == f"call {OTHER}".encode()
    assert rows_back.content == rows.encode()
    assert deleted.status_code == 200
    # Each upload reported as its creator's; reads never are.
    assert owners.objects == [("file_1", SESSION), ("file_2", SESSION)]


async def test_a_binary_document_needs_the_clients_own_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for proxy_credential in (True, False):
        files = FilesAPI()
        app, _ = lent_app(
            monkeypatch, "anthropic", ANTHROPIC, files, proxy_credential=proxy_credential
        )
        async with client(app) as http:
            uploaded = await _upload(http, f"{EMAIL} scan.pdf", PDF, "application/pdf")
        if proxy_credential:
            assert uploaded.status_code == 400, uploaded.text
            assert uploaded.json()["error"]["type"] == "invalid_request_error"
            assert "not UTF-8 text" in uploaded.text and EMAIL not in uploaded.text
            assert files.received == []
            continue
        assert uploaded.status_code == 200, uploaded.text
        assert files.content["file_1"] == PDF  # as sent
        assert files.files["file_1"]["filename"] == f"{TOKEN} scan.pdf"  # its name redacted
        assert uploaded.json()["filename"] == f"{EMAIL} scan.pdf"


async def test_a_listed_file_nobody_is_recorded_creating_keeps_its_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = FilesAPI()
    owners = Owners(listings=["/v1/files"])
    app, _ = lent_app(monkeypatch, "anthropic", ANTHROPIC, files, owners=owners)
    async with client(app) as http:
        assert (await _upload(http, f"{EMAIL} a.txt", b"x")).status_code == 200
        owners.objects.clear()
        listed = await http.get("/v1/files", headers=KEY)
    assert listed.json()["data"][0]["filename"] == f"{TOKEN} a.txt"
