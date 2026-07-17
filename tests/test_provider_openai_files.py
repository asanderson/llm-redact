"""OpenAI Files + Batches: multipart upload redaction, output rehydration.

The upload's JSONL file part is redacted line by line (batch lines get the
system note inside body; fine-tune lines directly); everything else in the
multipart body — form fields, binary parts, unparseable lines — must be
byte-identical. /v1/batches and file metadata stay pass-through.
"""

import json
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig
from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors
from llm_redact.multipart import parse, parse_boundary
from llm_redact.providers.base import SYSTEM_NOTE, RouteKind
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.proxy import create_app
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"
BOUNDARY = b"testboundary123"


def _redactor(vault: InMemoryVault) -> Redactor:
    return Redactor(
        detectors=build_detectors(DetectionConfig()),
        vault=vault,
        allowlist=Allowlist(exact=frozenset(), patterns=()),
    )


def _upload_body(*lines: str) -> bytes:
    jsonl = "".join(f"{line}\n" for line in lines).encode()
    return (
        b"--testboundary123\r\n"
        b'Content-Disposition: form-data; name="purpose"\r\n'
        b"\r\n"
        b"batch\r\n"
        b"--testboundary123\r\n"
        b'Content-Disposition: form-data; name="file"; filename="input.jsonl"\r\n'
        b"Content-Type: application/jsonl\r\n"
        b"\r\n" + jsonl + b"\r\n"
        b"--testboundary123--\r\n"
    )


def test_files_routing() -> None:
    adapter = OpenAIAdapter()
    assert adapter.matches("POST", "/v1/files") is RouteKind.REDACT_ONLY
    assert adapter.matches("GET", "/v1/files/file_abc/content") is RouteKind.CHAT
    # Metadata surfaces stay pass-through: ids and processing state only.
    assert adapter.matches("GET", "/v1/files") is RouteKind.NONE
    assert adapter.matches("GET", "/v1/files/file_abc") is RouteKind.NONE
    assert adapter.matches("DELETE", "/v1/files/file_abc") is RouteKind.NONE
    assert adapter.matches("POST", "/v1/batches") is RouteKind.NONE
    assert adapter.matches("GET", "/v1/batches/batch_1") is RouteKind.NONE
    assert adapter.matches("POST", "/v1/batches/batch_1/cancel") is RouteKind.NONE


def test_upload_batch_lines_redacted_and_noted() -> None:
    adapter = OpenAIAdapter()
    vault = InMemoryVault()
    batch_line = json.dumps(
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": f"to {EMAIL}"}]},
        }
    )
    clean_line = json.dumps({"custom_id": "req-2", "body": {"messages": []}})
    body = _upload_body(batch_line, "not json at all", clean_line)

    out = adapter.redact_multipart("/v1/files", body, BOUNDARY, _redactor(vault), inject_note=True)
    assert out is not None
    parsed = parse(out, BOUNDARY)
    assert parsed is not None
    assert parsed.parts[0].content == b"batch"  # form field untouched

    lines = parsed.parts[1].content.split(b"\n")
    rewritten = json.loads(lines[0])
    assert EMAIL not in lines[0].decode()
    assert "«EMAIL_001»" in rewritten["body"]["messages"][-1]["content"]
    assert rewritten["body"]["messages"][0] == {"role": "system", "content": SYSTEM_NOTE}
    assert rewritten["custom_id"] == "req-1"
    assert lines[1] == b"not json at all"  # unparseable line byte-identical
    assert json.loads(lines[2]) == json.loads(clean_line)  # clean line unchanged bytes
    assert lines[2] == clean_line.encode()


def test_upload_fine_tune_lines() -> None:
    adapter = OpenAIAdapter()
    vault = InMemoryVault()
    ft_line = json.dumps({"messages": [{"role": "user", "content": f"contact {EMAIL}"}]})
    out = adapter.redact_multipart(
        "/v1/files", _upload_body(ft_line), BOUNDARY, _redactor(vault), inject_note=True
    )
    assert out is not None
    parsed = parse(out, BOUNDARY)
    assert parsed is not None
    rewritten = json.loads(parsed.parts[1].content.split(b"\n")[0])
    assert "«EMAIL_001»" in rewritten["messages"][-1]["content"]
    assert rewritten["messages"][0]["content"] == SYSTEM_NOTE


def test_upload_without_secrets_forwards_verbatim() -> None:
    adapter = OpenAIAdapter()
    body = _upload_body(json.dumps({"custom_id": "req-1", "body": {"messages": []}}))
    out = adapter.redact_multipart(
        "/v1/files", body, BOUNDARY, _redactor(InMemoryVault()), inject_note=True
    )
    assert out is None  # nothing changed: proxy forwards the ORIGINAL bytes


def test_upload_binary_file_untouched() -> None:
    adapter = OpenAIAdapter()
    binary = (
        b"--testboundary123\r\n"
        b'Content-Disposition: form-data; name="file"; filename="doc.pdf"\r\n'
        b"\r\n"
        b"%PDF-1.7 \x00\x01\x02 not jsonl\r\n"
        b"--testboundary123--\r\n"
    )
    assert (
        adapter.redact_multipart(
            "/v1/files", binary, BOUNDARY, _redactor(InMemoryVault()), inject_note=True
        )
        is None
    )


def test_output_file_rehydrated() -> None:
    adapter = OpenAIAdapter()
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    rehydrator = Rehydrator(vault)
    output_line = json.dumps(
        {
            "id": "batch_req_1",
            "custom_id": "req-1",
            "response": {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": f"sent to {token}"}}]},
            },
        }
    ).encode()
    raw = output_line + b"\nnot json\n"
    out = adapter.rehydrate_raw_body("/v1/files/file_abc/content", raw, rehydrator)
    assert out is not None
    restored = json.loads(out.split(b"\n")[0])
    assert restored["response"]["body"]["choices"][0]["message"]["content"] == f"sent to {EMAIL}"
    assert out.split(b"\n")[1] == b"not json"
    # Non-file-content paths and token-free bodies stay untouched.
    assert adapter.rehydrate_raw_body("/v1/other", raw, rehydrator) is None
    assert (
        adapter.rehydrate_raw_body("/v1/files/file_abc/content", b"plain text\n", rehydrator)
        is None
    )


# --- integration: real app, fake files/batches upstream ---------------------

received: dict[str, Any] = {}


def _fake_upstream() -> Starlette:
    async def upload(request: Request) -> Response:
        received["upload_raw"] = await request.body()
        received["upload_content_type"] = request.headers.get("content-type", "")
        return JSONResponse({"id": "file_abc", "object": "file", "purpose": "batch"})

    async def create_batch(request: Request) -> Response:
        received["batch_create"] = await request.json()
        return JSONResponse({"id": "batch_1", "status": "in_progress"})

    async def content(request: Request) -> Response:
        # Echo every placeholder seen in the upload as a batch output file.
        boundary = parse_boundary(received["upload_content_type"])
        assert boundary is not None
        parsed = parse(received["upload_raw"], boundary)
        assert parsed is not None
        import re as _re

        # The injected system note itself contains example tokens
        # («TYPE_NNN», «EMAIL_001») — echo only vault-issued EMAIL tokens
        # from the actual message content.
        tokens = _re.findall("«EMAIL_[0-9]+»", parsed.parts[1].content.decode())
        lines = b"".join(
            json.dumps(
                {
                    "custom_id": f"req-{i}",
                    "response": {"status_code": 200, "body": {"content": f"echo {token}"}},
                }
            ).encode()
            + b"\n"
            for i, token in enumerate(dict.fromkeys(tokens))
        )
        return Response(content=lines, media_type="application/octet-stream")

    return Starlette(
        routes=[
            Route("/v1/files", upload, methods=["POST"]),
            Route("/v1/batches", create_batch, methods=["POST"]),
            Route("/v1/files/file_abc/content", content, methods=["GET"]),
        ]
    )


@pytest.fixture
def client() -> httpx.AsyncClient:
    received.clear()
    config = Config(providers={"openai": ProviderConfig(upstream_base_url="http://upstream")})
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


@pytest.mark.anyio
async def test_files_batch_round_trip(client: httpx.AsyncClient) -> None:
    line = json.dumps(
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": f"to {EMAIL}"}]},
        }
    )
    uploaded = await client.post(
        "/v1/files",
        data={"purpose": "batch"},
        files={"file": ("input.jsonl", f"{line}\n".encode(), "application/jsonl")},
    )
    assert uploaded.status_code == 200

    # The upstream never saw the real value — and its multipart framing
    # still parses with a standard parser (starlette would have failed the
    # request otherwise; assert on the recorded raw body too).
    assert EMAIL.encode() not in received["upload_raw"]
    assert "«EMAIL_001»".encode() in received["upload_raw"]

    # Batch creation is metadata: passes through untouched.
    batch = await client.post("/v1/batches", json={"input_file_id": "file_abc"})
    assert batch.json()["id"] == "batch_1"
    assert received["batch_create"] == {"input_file_id": "file_abc"}

    # Output download restores the original value.
    output = await client.get("/v1/files/file_abc/content")
    assert output.status_code == 200
    first = json.loads(output.content.splitlines()[0])
    assert first["response"]["body"]["content"] == f"echo {EMAIL}"


def test_pass_through_provider_inference_covers_uploads(tmp_path):
    # /v1/uploads (the multipart Uploads API) has NO adapter — its content
    # is a documented non-goal — but pass-through must still reach the
    # OPENAI upstream. It was missing from the inference prefixes, so
    # Uploads traffic fell through to the anthropic default and hit the
    # wrong provider entirely.
    from llm_redact.config import load_config
    from llm_redact.proxy import ProxyState

    config_path = tmp_path / "config.toml"
    config_path.write_text("")
    state = ProxyState(load_config(config_path), None, config_path=config_path)
    for path in ("/v1/uploads", "/v1/uploads/upload_abc/parts", "/v1/uploads/upload_abc/complete"):
        assert state.provider_for(None, path) == "openai"
    # The siblings and the default stay as they were.
    assert state.provider_for(None, "/v1/batches") == "openai"
    assert state.provider_for(None, "/v1/messages/something-unknown") == "anthropic"
    # Anthropic's beta Files API (same paths, anthropic-version header)
    # still wins over the OpenAI inference.
    assert state.provider_for(None, "/v1/files", {"anthropic-version": "2023-06-01"}) == "anthropic"
