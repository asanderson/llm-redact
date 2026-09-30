"""OpenAI code interpreter containers, recognized so a credential the proxy
holds may reach them (a routed operator key, Azure's identity auth on the
v1 API).

A container file upload rides the ``/v1/files`` multipart machinery
(``redact_multipart`` with ``upload_content.classify_file``: a JSONL or
text file redacted, a binary one refused under a credential the proxy holds
and forwarded unscanned with the client's own key); a JSON create copies a
stored file in (its ``file_id`` verbatim). The container file object echoes
the filename in ``path`` (restored), and a download is restored like a
Files API download: a text file line by line, a binary file untouched. A
container's ``name`` is a label, redacted and restored on every echo; its
starting ``file_ids`` are verbatim. Created containers and container files
are reported to the session router.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig
from llm_redact.providers.azure_openai import AzureOpenAIAdapter
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.proxy import create_app
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
        "/v1/containers/cntr_1/files/cfile_9/content",
    ]
    routed = Routed(monkeypatch, host, paths)
    create = {
        "name": f"box of {EMAIL}",
        "file_ids": [FILE],
        "expires_after": {"anchor": "last_active_at", "minutes": 20},
        "memory_limit": "4g",
    }
    created = await routed.send("POST", "/v1/containers", create)
    assert created.status_code == 200, created.text  # forwarded, not the 403
    # The name is a label: redacted (and restored below); the rest as sent.
    assert routed.provider.last_json() == {**create, "name": f"box of {TOKEN}"}
    assert created.json()["name"] == create["name"]
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

    # A download is restored as a /v1/files download is: a text file line by
    # line (a JSON line as JSON, any other line as text) ...
    host.contents["cfile_1"] += b"\nplain text " + TOKEN.encode()
    content = await routed.send("GET", "/v1/containers/cntr_1/files/cfile_1/content")
    lines = content.content.split(b"\n")
    assert json.loads(lines[0]) == {"note": f"call {EMAIL}"}
    assert lines[2] == b"plain text " + EMAIL.encode()
    # ... and a binary file (one the code wrote: a PNG) left untouched.
    png = b"\x89PNG\r\n\x1a\n" + TOKEN.encode() + b"\x00"
    host.contents["cfile_9"] = png
    image = await routed.send("GET", "/v1/containers/cntr_1/files/cfile_9/content")
    assert image.content == png

    containers = await routed.send("GET", "/v1/containers")
    assert containers.json()["data"][0]["name"] == create["name"]
    assert routed.sessions.listed == ["cntr_1"]  # the container list is attributed
    assert (await routed.send("GET", "/v1/containers/cntr_1")).status_code == 200
    assert (await routed.send("DELETE", "/v1/containers/cntr_1/files/cfile_1")).status_code == 200
    assert (await routed.send("DELETE", "/v1/containers/cntr_1")).status_code == 200
    assert all(identity for *_, identity in routed.sessions.checks)


@pytest.mark.parametrize(
    ("path", "body", "field"),
    [
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


PNG = b"\x89PNG\r\n\x1a\n" + EMAIL.encode() + b"\x00\x01"


async def test_a_container_upload_follows_the_upload_content_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A container file upload is read like a /v1/files upload
    (``upload_content.classify_file``): a text file is redacted as text,
    and a binary file — never read — is refused under a credential the
    proxy holds."""
    path = "/v1/containers/cntr_1/files"
    routed = Routed(monkeypatch, ContainerHost(), [path])
    headers = {"content-type": f"multipart/form-data; boundary={BOUNDARY}"}
    text = await routed.send(
        "POST", path, content=_multipart("notes.txt", f"plain {EMAIL}\n".encode()), headers=headers
    )
    assert text.status_code == 200, text.text
    sent = routed.provider.requests[-1].content
    assert EMAIL.encode() not in sent and b"plain " + TOKEN.encode() in sent
    binary = await routed.send("POST", path, content=_multipart("plot.png", PNG), headers=headers)
    assert binary.status_code == 400 and EMAIL not in binary.text
    assert len(routed.provider.requests) == 1  # the binary upload never left


async def test_a_binary_container_upload_with_the_clients_own_key_goes_unscanned() -> None:
    """With the client's own key ([detection] binary_uploads = "forward",
    the default) a binary container file goes out as sent, counted as an
    unscanned upload."""
    host = ContainerHost()
    upstream = "https://api.openai.test"
    config = Config(providers={**Config().providers, "openai": ProviderConfig(upstream)})
    app = create_app(config, upstream_transport=httpx.MockTransport(host))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/containers/cntr_1/files",
            content=_multipart("plot.png", PNG),
            headers={
                "authorization": "Bearer sk-client",
                "content-type": f"multipart/form-data; boundary={BOUNDARY}",
            },
        )
    assert response.status_code == 200, response.text
    assert host.contents["cfile_1"] == PNG
    assert app.state.proxy.unscanned_uploads["openai"] == 1


@pytest.mark.parametrize(
    ("adapter", "path"),
    [
        (OpenAIAdapter(), "/v1/containers/cntr_1/files/cfile_1/content"),
        (AzureOpenAIAdapter(), "/openai/v1/containers/cntr_1/files/cfile_1/content"),
    ],
)
def test_a_container_file_download_is_restored_like_a_files_download(
    adapter: Any, path: str
) -> None:
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    raw = json.dumps({"to": token}).encode() + b"\n" + json.dumps({"cc": token}).encode()
    out = adapter.rehydrate_raw_body(path, raw, Rehydrator(vault))
    assert out is not None
    first, second = out.split(b"\n")
    assert json.loads(first) == {"to": EMAIL} and json.loads(second) == {"cc": EMAIL}
    text = b"raw " + token.encode()
    assert adapter.rehydrate_raw_body(path, text, Rehydrator(vault)) == b"raw " + EMAIL.encode()
    binary = b"\x89PNG\r\n\x1a\n" + token.encode()
    assert adapter.rehydrate_raw_body(path, binary, Rehydrator(vault)) is None
    assert adapter.rehydrate_raw_body(path.replace("/content", ""), raw, Rehydrator(vault)) is None
    assert (
        AzureOpenAIAdapter().rehydrate_raw_body("/openai/v1/containers", raw, Rehydrator(vault))
        is None
    )
