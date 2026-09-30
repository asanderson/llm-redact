"""A downloaded FILE is restored per file whatever its Content-Type.

Providers serve a file's content with the file's own media type. A text
file uploaded redacted as ONE text — its keys and every value included —
that happens to be one JSON document came back as ``application/json`` and
took the whole-body JSON walk: tokens under keys and structural names
(``id``, ``data``) stayed in the file, and a restored value re-serialized
the whole file (indentation, ``1.50``, ``\\/`` lost). JSON Lines and
event-stream media types took the streaming readings. Every file-download
path now restores through the adapter's per-file reading
(``restores_file_download`` → ``rehydrate_raw_body``), end to end through
the real app, per provider path.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig, parse_config
from llm_redact.multipart import parse
from llm_redact.proxy import create_app

# Pretty-printed JSON: uploaded as text (not JSON Lines), keys redacted too.
ORIGINAL = (
    b"{\n"
    b'  "id": "jane.doe@corp.example",\n'
    b'  "sam.roe@corp.example": 1,\n'
    b'  "data": "x jane.doe@corp.example",\n'
    b'  "ratio": 1.50,\n'
    b'  "path": "C:\\/data"\n'
    b"}\n"
)

_OPENAI = {"authorization": "Bearer sk-own"}
_ANTHROPIC = {"anthropic-version": "2023-06-01", "x-api-key": "sk-ant-own"}
_GEMINI = {"x-goog-api-key": "AIza-own"}
_AZURE = {"api-key": "azure-own"}


def _form(content: bytes) -> tuple[bytes, str]:
    body = (
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="doc.json"\r\n'
        b"Content-Type: application/json\r\n\r\n" + content + b"\r\n--b--\r\n"
    )
    return body, "multipart/form-data; boundary=b"


def _related(content: bytes) -> tuple[bytes, str]:
    body = (
        b"--x\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
        b'{"file": {"display_name": "doc"}}\r\n'
        b"--x\r\nContent-Type: text/plain\r\n\r\n" + content + b"\r\n--x--\r\n"
    )
    return body, "multipart/related; boundary=x"


# route -> (upload path, upload builder, extra upload headers, download path, headers)
_ROUTES: dict[
    str, tuple[str, Callable[[bytes], tuple[bytes, str]], dict[str, str], str, dict[str, str]]
] = {
    "openai-file": ("/v1/files", _form, {}, "/v1/files/file-1/content", _OPENAI),
    "openai-container-file": (
        "/v1/containers/cntr_1/files",
        _form,
        {},
        "/v1/containers/cntr_1/files/cfile_1/content",
        _OPENAI,
    ),
    "anthropic-file": ("/v1/files", _form, {}, "/v1/files/file-1/content", _ANTHROPIC),
    "gemini-download": (
        "/upload/v1beta/files",
        _related,
        {"x-goog-upload-protocol": "multipart"},
        "/download/v1beta/files/f1:download",
        _GEMINI,
    ),
    "azure-file": (
        "/openai/files",
        _form,
        {},
        "/openai/files/file-1/content",
        _AZURE,
    ),
    "azure-container-file": (
        "/openai/v1/containers/cntr_1/files",
        _form,
        {},
        "/openai/v1/containers/cntr_1/files/cfile_1/content",
        _AZURE,
    ),
    "custom-file": (
        "/custom/vllm/v1/files",
        _form,
        {},
        "/custom/vllm/v1/files/file-1/content",
        _OPENAI,
    ),
}


def _config() -> Config:
    config = parse_config(
        {"providers": {"custom": {"vllm": {"upstream_base_url": "http://vllm.example"}}}}, "t"
    )
    return Config(providers={**config.providers, "azure": ProviderConfig("http://azure.example")})


@pytest.mark.parametrize(
    "media_type",
    ["application/json", "application/x-ndjson", "text/event-stream", "application/octet-stream"],
)
@pytest.mark.parametrize("route", sorted(_ROUTES))
async def test_a_downloaded_file_round_trips_whatever_its_content_type(
    route: str, media_type: str
) -> None:
    upload_path, build, upload_headers, download_path, headers = _ROUTES[route]
    stored: dict[str, bytes] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            boundary = request.headers["content-type"].split("boundary=")[1].encode()
            parsed = parse(request.content, boundary)
            assert parsed is not None
            stored["file"] = parsed.parts[-1].content
            return httpx.Response(200, json={"id": "file-1", "file": {"name": "files/f1"}})
        return httpx.Response(200, content=stored["file"], headers={"content-type": media_type})

    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    body, content_type = build(ORIGINAL)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        upload = await client.post(
            upload_path,
            content=body,
            headers={**headers, **upload_headers, "content-type": content_type},
        )
        assert upload.status_code == 200, upload.text
        # Keys and structural names were redacted as text on the way out.
        assert b"corp.example" not in stored["file"]
        assert b'"id": "\xc2\xabEMAIL_001\xc2\xbb"' in stored["file"]
        download = await client.get(download_path, headers=headers)
    assert download.status_code == 200
    assert download.content == ORIGINAL


PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\n"
    "-----END PRIVATE KEY-----"
)


@pytest.mark.parametrize("media_type", ["application/json", "application/octet-stream"])
async def test_a_json_file_the_model_wrote_stays_valid_json(media_type: str) -> None:
    # A pretty-printed JSON file a model wrote around a placeholder (not an
    # upload this proxy redacted): a restored value that needs escaping (a
    # PEM key's newlines) is JSON-escaped where it lands, formatting kept.
    state: dict[str, str] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/messages":
            state["token"] = json.loads(request.content)["messages"][0]["content"]
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": "m",
                    "stop_reason": "end_turn",
                },
            )
        written = {"service": "x", state["token"]: "key", "private_key": state["token"]}
        content = json.dumps(written, indent=2, ensure_ascii=False).encode()
        return httpx.Response(200, content=content, headers={"content-type": media_type})

    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        sent = await client.post(
            "/v1/messages",
            json={"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": PEM}]},
            headers=_ANTHROPIC,
        )
        assert sent.status_code == 200 and state["token"] != PEM
        download = await client.get("/v1/files/file_1/content", headers=_ANTHROPIC)
    assert download.status_code == 200
    restored = json.loads(download.content)
    # Values and keys restored exactly; the indentation kept.
    assert restored == {"service": "x", PEM: "key", "private_key": PEM}
    assert download.content.startswith(b'{\n  "service": "x",\n')


async def test_a_json_text_upload_is_byte_exact_while_remembered() -> None:
    # A user's pretty-printed JSON text file, redacted as ONE text (the
    # value as it appears in the source, escapes included): restored raw,
    # byte for byte, while this process remembers the upload. Once
    # forgotten (a restart), its download reads as JSON a model could have
    # written: restored JSON-escaped — still valid JSON, the documented
    # residual.
    from llm_redact.providers import openai

    original = b'{\n  "user": "CORP\\\\jdoe",\n  "n": 1.50\n}\n'
    stored: dict[str, bytes] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            boundary = request.headers["content-type"].split("boundary=")[1].encode()
            parsed = parse(request.content, boundary)
            assert parsed is not None
            stored["file"] = parsed.parts[-1].content
            return httpx.Response(200, json={"id": "file-1"})
        return httpx.Response(
            200, content=stored["file"], headers={"content-type": "application/json"}
        )

    config = parse_config({"detection": {"deny": ["CORP\\\\jdoe"]}}, "t")
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    body, content_type = _form(original)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        upload = await client.post(
            "/v1/files", content=body, headers={**_OPENAI, "content-type": content_type}
        )
        assert upload.status_code == 200, upload.text
        assert b"CORP" not in stored["file"]
        remembered = await client.get("/v1/files/file-1/content", headers=_OPENAI)
        saved = openai.RAW_TEXT_FILES
        openai.RAW_TEXT_FILES = openai._RawTextFiles()
        try:
            forgotten = await client.get("/v1/files/file-1/content", headers=_OPENAI)
        finally:
            openai.RAW_TEXT_FILES = saved
    assert remembered.content == original
    assert json.loads(forgotten.content) == {"user": "CORP\\\\jdoe", "n": 1.5}
    assert forgotten.content.endswith(b'\n  "n": 1.50\n}\n')
