"""An uploaded JSONL DATA file is redacted with no skip set.

The request-body structural keys (``jsonwalk.STRUCTURAL_KEYS``: ``id``,
``name``, ``type``, ``data`` …) guard protocol fields of a request the
provider runs. In the caller's own data file they are ordinary content, so
only the lines a provider RUNS keep that reading: an OpenAI Files upload of
purpose ``batch`` (each line's ``body``) or ``fine-tune`` (the example's
conversation). Every other JSONL line — another purpose, a container's file,
Anthropic's Files, the Gemini upload — has every string value redacted,
whatever its key. End to end through the real app, per upload route.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from llm_redact.config import Config
from llm_redact.jsonwalk import STRUCTURAL_KEYS
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"
# One secret under every structural key, plus an ordinary one.
DATA_LINE = json.dumps({**{key: f"of {EMAIL}" for key in sorted(STRUCTURAL_KEYS)}, "memo": EMAIL})
_OPENAI = {"authorization": "Bearer sk-own"}
_ANTHROPIC = {"anthropic-version": "2023-06-01", "x-api-key": "sk-ant-own"}


def _form(*fields: tuple[str, bytes], content: bytes) -> bytes:
    out = b""
    for name, value in fields:
        out += f'--b\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += value + b"\r\n"
    out += b'--b\r\nContent-Disposition: form-data; name="file"; filename="rows.jsonl"\r\n'
    out += b"Content-Type: application/jsonl\r\n\r\n" + content + b"\r\n--b--\r\n"
    return out


def _related(content: bytes) -> bytes:
    return (
        b"--x\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
        b'{"file": {"display_name": "rows"}}\r\n'
        b"--x\r\nContent-Type: application/jsonl\r\n\r\n" + content + b"\r\n--x--\r\n"
    )


async def _upload(
    path: str, body: bytes, headers: dict[str, str], content_type: str
) -> httpx.Request:
    seen: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "file-1", "file": {"name": "files/f1"}})

    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            path, content=body, headers={**headers, "content-type": content_type}
        )
    assert response.status_code == 200, response.text
    assert len(seen) == 1
    return seen[0]


def _file_line(sent: httpx.Request) -> dict[str, object]:
    """The first JSONL line of the forwarded file part."""
    # The file is the last part: after the last header block.
    parsed = json.loads(sent.content.rsplit(b"\r\n\r\n", 1)[1].split(b"\n", 1)[0])
    assert isinstance(parsed, dict)
    return parsed


_FORM = "multipart/form-data; boundary=b"
_ROUTES: dict[str, tuple[str, Callable[[bytes], bytes], dict[str, str], str]] = {
    "openai-user-data": (
        "/v1/files",
        lambda c: _form(("purpose", b"user_data"), content=c),
        _OPENAI,
        _FORM,
    ),
    "openai-no-purpose": ("/v1/files", lambda c: _form(content=c), _OPENAI, _FORM),
    "openai-two-purposes": (
        "/v1/files",
        lambda c: _form(("purpose", b"batch"), ("purpose", b"fine-tune"), content=c),
        _OPENAI,
        _FORM,
    ),
    # A purpose field on Anthropic's upload names nothing: still data.
    "anthropic-files": (
        "/v1/files",
        lambda c: _form(("purpose", b"batch"), content=c),
        _ANTHROPIC,
        _FORM,
    ),
    "openai-container-file": (
        "/v1/containers/cntr_1/files",
        lambda c: _form(("purpose", b"batch"), content=c),
        _OPENAI,
        _FORM,
    ),
    "gemini-upload": (
        "/upload/v1beta/files",
        _related,
        {"x-goog-api-key": "AIza-own", "x-goog-upload-protocol": "multipart"},
        "multipart/related; boundary=x",
    ),
}


@pytest.mark.parametrize("route", sorted(_ROUTES))
async def test_every_value_of_a_data_line_is_redacted(route: str) -> None:
    path, build, headers, content_type = _ROUTES[route]
    sent = await _upload(path, build(DATA_LINE.encode() + b"\n"), headers, content_type)
    assert EMAIL.encode() not in sent.content
    line = _file_line(sent)
    # Keys stay; every value under them — structural names included — is
    # redacted.
    assert set(line) == {*STRUCTURAL_KEYS, "memo"}
    assert all("«EMAIL_001»" in str(value) for value in line.values())


async def test_a_batch_line_reads_only_its_body_as_a_request() -> None:
    batch_line = json.dumps(
        {
            "custom_id": "req-1",
            "id": EMAIL,  # the envelope is data
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": f"to {EMAIL}"}],
            },
        }
    )
    sent = await _upload(
        "/v1/files",
        _form(("purpose", b"batch"), content=batch_line.encode() + b"\n"),
        _OPENAI,
        _FORM,
    )
    assert EMAIL.encode() not in sent.content
    line = _file_line(sent)
    assert line["id"] == "«EMAIL_001»"
    body = line["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-4o"
    messages = body["messages"]
    assert isinstance(messages, list)
    # The request's structure as sent, the system note added.
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[-1]["content"] == "to «EMAIL_001»"


async def test_a_fine_tune_line_reads_only_its_conversation_as_a_request() -> None:
    example = json.dumps(
        {
            "messages": [
                {"role": "user", "content": f"mail {EMAIL}"},
                {"role": "assistant", "content": "ok"},
            ],
            # A reinforcement example's grader fields are data.
            "id": EMAIL,
            "reference": {"name": EMAIL},
        }
    )
    sent = await _upload(
        "/v1/files",
        _form(("purpose", b"fine-tune"), content=example.encode() + b"\n"),
        _OPENAI,
        _FORM,
    )
    assert EMAIL.encode() not in sent.content
    line = _file_line(sent)
    assert line["id"] == "«EMAIL_001»"
    assert line["reference"] == {"name": "«EMAIL_001»"}
    messages = line["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]


# route -> the download path serving that upload's file back.
_DOWNLOADS = {
    "openai-user-data": "/v1/files/file-1/content",
    "openai-no-purpose": "/v1/files/file-1/content",
    "openai-two-purposes": "/v1/files/file-1/content",
    "anthropic-files": "/v1/files/file-1/content",
    "openai-container-file": "/v1/containers/cntr_1/files/cfile_1/content",
    "gemini-upload": "/download/v1beta/files/f1:download",
}


@pytest.mark.parametrize("media_type", ["application/octet-stream", "application/jsonl"])
@pytest.mark.parametrize("route", sorted(_ROUTES))
async def test_a_data_file_downloads_with_every_value_restored(route: str, media_type: str) -> None:
    # The download restores what the upload redacted: every value of a data
    # line, under a structural name too (the request-body skip set once left
    # «EMAIL_001» under id/name/type/data in the user's own file).
    path, build, headers, content_type = _ROUTES[route]
    original = DATA_LINE.encode() + b"\n"
    stored: dict[str, bytes] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            stored["file"] = request.content.rsplit(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--", 1)[0]
            return httpx.Response(200, json={"id": "file-1", "file": {"name": "files/f1"}})
        return httpx.Response(200, content=stored["file"], headers={"content-type": media_type})

    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        upload = await client.post(
            path, content=build(original), headers={**headers, "content-type": content_type}
        )
        assert upload.status_code == 200, upload.text
        assert EMAIL.encode() not in stored["file"]
        download = await client.get(_DOWNLOADS[route], headers=headers)
    assert download.status_code == 200
    assert download.content == original


def test_a_batch_output_tool_call_is_still_restored_as_json_source() -> None:
    # A batch output's tool call `arguments` is raw JSON source: a restored
    # value is JSON-escaped there (and a structural name is restored too).
    from llm_redact.providers.openai import OpenAIAdapter
    from llm_redact.rehydrate import Rehydrator
    from llm_redact.vault import InMemoryVault

    vault = InMemoryVault()
    quoted = 'say "hi" to jane.doe@corp.example'
    token = vault.placeholder_for("DENY", quoted)
    arguments = json.dumps({"to": token}, ensure_ascii=False)
    line = json.dumps(
        {
            "id": token,
            "response": {
                "body": {
                    "choices": [
                        {"message": {"tool_calls": [{"function": {"arguments": arguments}}]}}
                    ]
                }
            },
        },
        ensure_ascii=False,
    )
    restored = OpenAIAdapter().rehydrate_raw_body(
        "/v1/files/file-1/content", line.encode() + b"\n", Rehydrator(vault)
    )
    assert restored is not None
    value = json.loads(restored)
    assert value["id"] == quoted
    call = value["response"]["body"]["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"to": quoted}
