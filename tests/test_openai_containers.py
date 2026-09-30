"""OpenAI code interpreter containers, recognized so a credential the proxy
holds may reach them (a routed operator key, Azure's identity auth on the
v1 API).

A container file upload rides the ``/v1/files`` multipart machinery
(``redact_multipart``: the file part, its filename, every form field — what
it cannot scan refuses the upload); a JSON create copies a stored file in
(its ``file_id`` verbatim). The container file object echoes the filename
in ``path`` (restored), and a download is restored like a Files API
download: JSON-object lines only, every other byte as sent. A container's
``name`` and starting ``file_ids`` are verbatim. Created containers and
container files are reported to the session router.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llm_redact.providers.azure_openai import AzureOpenAIAdapter
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault
from stored_objects import EMAIL, TOKEN, Routed

FILE = "file-XjGxS3KTG0uNmNOK362iJua3"
BOUNDARY = "llmredactboundary"


def _multipart(filename: str, content: bytes) -> bytes:
    return (
        (
            f"--{BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: application/jsonl\r\n\r\n"
        ).encode()
        + content
        + f"\r\n--{BOUNDARY}--\r\n".encode()
    )


class ContainerHost:
    """A provider that keeps containers and their files as it receives them."""

    def __init__(self) -> None:
        self.containers: dict[str, dict[str, Any]] = {}
        self.files: dict[str, dict[str, Any]] = {}
        self.contents: dict[str, bytes] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        segments = request.url.path.rstrip("/").split("/")[2:]  # after /v1
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": segments[-1], "deleted": True})
        if segments == ["containers"] and request.method == "POST":
            body = json.loads(request.content)
            container = {"id": "cntr_1", "object": "container", "status": "running", **body}
            self.containers["cntr_1"] = container
            return httpx.Response(200, json=container)
        if segments == ["containers"]:
            data = list(self.containers.values())
            return httpx.Response(200, json={"object": "list", "data": data, "has_more": False})
        if segments[2:] == ["files"] and request.method == "POST":
            return httpx.Response(200, json=self._create_file(request))
        if segments[2:] == ["files"]:
            data = list(self.files.values())
            return httpx.Response(200, json={"object": "list", "data": data, "has_more": False})
        if segments[4:] == ["content"]:
            return httpx.Response(
                200,
                content=self.contents[segments[3]],
                headers={"content-type": "application/octet-stream"},
            )
        if len(segments) == 4:
            return httpx.Response(200, json=self.files[segments[3]])
        return httpx.Response(200, json=self.containers[segments[1]])

    def _create_file(self, request: httpx.Request) -> dict[str, Any]:
        file_id = f"cfile_{len(self.files) + 1}"
        if request.headers.get("content-type", "").startswith("multipart/"):
            head, _, rest = request.content.partition(b"\r\n\r\n")
            name = head.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
            self.contents[file_id] = rest.rsplit(f"\r\n--{BOUNDARY}--".encode(), 1)[0]
            source, path = "user", f"/mnt/data/{name}"
        else:
            source, path = "user", f"/mnt/data/{json.loads(request.content)['file_id']}"
        self.files[file_id] = {
            "id": file_id,
            "object": "container.file",
            "container_id": "cntr_1",
            "path": path,
            "source": source,
        }
        return self.files[file_id]


async def test_containers_are_served_under_an_operator_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = ContainerHost()
    paths = [
        "/v1/containers",
        "/v1/containers/cntr_1",
        "/v1/containers/cntr_1/files",
        "/v1/containers/cntr_1/files/cfile_1",
        "/v1/containers/cntr_1/files/cfile_1/content",
    ]
    routed = Routed(monkeypatch, host, paths)
    create = {
        "name": "sandbox",
        "file_ids": [FILE],
        "expires_after": {"anchor": "last_active_at", "minutes": 20},
        "memory_limit": "4g",
    }
    created = await routed.send("POST", "/v1/containers", create)
    assert created.status_code == 200, created.text  # forwarded, not the 403
    assert routed.provider.last_json() == create  # verbatim and structural as sent
    assert routed.sessions.objects == [("cntr_1", "default")]

    line = json.dumps({"note": f"call {EMAIL}"}).encode()
    upload = await routed.send(
        "POST",
        "/v1/containers/cntr_1/files",
        content=_multipart(f"{EMAIL}.jsonl", line + b"\n"),
        headers={"content-type": f"multipart/form-data; boundary={BOUNDARY}"},
    )
    assert upload.status_code == 200, upload.text
    sent = routed.provider.requests[-1].content
    assert EMAIL.encode() not in sent and TOKEN.encode() in sent
    assert b"character for character" not in sent  # never a system note
    assert upload.json()["path"] == f"/mnt/data/{EMAIL}.jsonl"  # the echo restored
    assert routed.sessions.objects[-1] == ("cfile_1", "default")

    copied = await routed.send("POST", "/v1/containers/cntr_1/files", {"file_id": FILE})
    assert copied.status_code == 200
    # The container file is new (cfile_2); the stored file it copied is not.
    assert routed.sessions.objects[-1] == ("cfile_2", "default")

    listing = await routed.send("GET", "/v1/containers/cntr_1/files")
    assert f"/mnt/data/{EMAIL}.jsonl" in {item["path"] for item in listing.json()["data"]}
    assert routed.sessions.listed == []  # a container's own files: not attributed
    one = await routed.send("GET", "/v1/containers/cntr_1/files/cfile_1")
    assert one.json()["path"] == f"/mnt/data/{EMAIL}.jsonl"

    host.contents["cfile_1"] += b"\nplain text " + TOKEN.encode() + b"\n\x89PNG"
    content = await routed.send("GET", "/v1/containers/cntr_1/files/cfile_1/content")
    lines = content.content.split(b"\n")
    assert json.loads(lines[0]) == {"note": f"call {EMAIL}"}  # a JSON line restored
    # Every other byte as sent: text rehydration of non-JSON content comes
    # with the uploads classifier; a placeholder left in place is safe.
    assert lines[2] == b"plain text " + TOKEN.encode() and lines[3] == b"\x89PNG"

    containers = await routed.send("GET", "/v1/containers")
    assert containers.json()["data"][0]["name"] == "sandbox"
    assert routed.sessions.listed == ["cntr_1"]  # the container list is attributed
    assert (await routed.send("GET", "/v1/containers/cntr_1")).status_code == 200
    assert (await routed.send("DELETE", "/v1/containers/cntr_1/files/cfile_1")).status_code == 200
    assert (await routed.send("DELETE", "/v1/containers/cntr_1")).status_code == 200
    assert all(identity for *_, identity in routed.sessions.checks)


@pytest.mark.parametrize(
    ("path", "body", "field"),
    [
        ("/v1/containers", {"name": f"box of {EMAIL}"}, "name"),
        ("/v1/containers", {"name": "box", "file_ids": [EMAIL]}, "file_ids"),
        ("/v1/containers/cntr_1/files", {"file_id": EMAIL}, "file_id"),
    ],
)
async def test_a_value_in_a_verbatim_field_refuses_the_request(
    path: str, body: dict[str, Any], field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    routed = Routed(monkeypatch, ContainerHost(), [path])
    response = await routed.send("POST", path, body)
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert f"`{field}`" in message and EMAIL not in message
    assert routed.provider.requests == [] and routed.sessions.objects == []


async def test_a_container_upload_llm_redact_cannot_scan_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "/v1/containers/cntr_1/files"
    routed = Routed(monkeypatch, ContainerHost(), [path])
    response = await routed.send(
        "POST",
        path,
        content=_multipart("notes.txt", f"plain {EMAIL}\n".encode()),
        headers={"content-type": f"multipart/form-data; boundary={BOUNDARY}"},
    )
    assert response.status_code == 400
    assert routed.provider.requests == [] and EMAIL not in response.text


@pytest.mark.parametrize(
    ("adapter", "path"),
    [
        (OpenAIAdapter(), "/v1/containers/cntr_1/files/cfile_1/content"),
        (AzureOpenAIAdapter(), "/openai/v1/containers/cntr_1/files/cfile_1/content"),
    ],
)
def test_a_container_file_download_restores_json_lines(adapter: Any, path: str) -> None:
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    raw = json.dumps({"to": token}).encode() + b"\nraw " + token.encode()
    out = adapter.rehydrate_raw_body(path, raw, Rehydrator(vault))
    assert out is not None
    first, second = out.split(b"\n")
    assert json.loads(first) == {"to": EMAIL} and second == b"raw " + token.encode()
    assert adapter.rehydrate_raw_body(path.replace("/content", ""), raw, Rehydrator(vault)) is None
    assert (
        AzureOpenAIAdapter().rehydrate_raw_body("/openai/v1/containers", raw, Rehydrator(vault))
        is None
    )
