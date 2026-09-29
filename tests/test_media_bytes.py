"""Bedrock base64 media (``source.bytes``) never reaches the detectors.

Bedrock carries image, document and video blocks as base64 under
``{"source": {"bytes": …}}`` — Converse, ApplyGuardrail, CountTokens and
native Nova invoke bodies alike. Scanning base64 can never find the plaintext
secrets the detectors look for (plaintext inside base64 is the documented
media non-goal), so it only cost event-loop CPU (~0.5 s per MB) and now and
then found a token-shaped run inside the blob and rewrote it, corrupting the
image. ``source.bytes`` is skipped like the base64 ``data`` of the other
providers — only as a string at exactly that position.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig
from llm_redact.jsonwalk import transform_strings
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"
BEDROCK = "https://bedrock-runtime.us-east-1.amazonaws.com"
# A PNG-looking blob whose base64 holds a Bitbucket-app-password-shaped run.
BLOB = "iVBORw0KGgo////ATBB" + "A1b2C3d4E5f6G7h8I9j0K1l2" + "////AAAASUVORK5CYII="


def _mark(s: str) -> str:
    return "<" + s + ">"


def _media(kind: str) -> dict[str, Any]:
    return {kind: {"format": "png", "name": "n", "source": {"bytes": BLOB}}}


@pytest.mark.parametrize("kind", ["image", "document", "video"])
def test_source_bytes_is_never_walked(kind: str) -> None:
    body = {"messages": [{"role": "user", "content": [_media(kind), {"text": "t"}]}]}
    out = transform_strings(body, _mark)
    assert out["messages"][0]["content"] == [
        {kind: {"format": "<png>", "name": "n", "source": {"bytes": BLOB}}},
        {"text": "<t>"},
    ]


def test_bytes_is_media_only_at_its_position_and_only_as_a_string() -> None:
    # Elsewhere a `bytes` key is ordinary; an object under source.bytes is
    # walked like any object under a skipped name.
    assert transform_strings({"bytes": "a"}, _mark) == {"bytes": "<a>"}
    assert transform_strings({"x": {"bytes": "a"}}, _mark) == {"x": {"bytes": "<a>"}}
    assert transform_strings({"source": {"bytes": {"k": "a"}}}, _mark) == {
        "source": {"bytes": {"k": "<a>"}}
    }
    # Inside caller JSON (an opaque position) nothing is skipped.
    tool = {"toolUse": {"input": {"source": {"bytes": "a"}}}}
    assert transform_strings(tool, _mark) == {"toolUse": {"input": {"source": {"bytes": "<a>"}}}}


def _converse(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": [*blocks, {"text": f"mail {EMAIL}"}]}]}


def _count_tokens_invoke() -> dict[str, Any]:
    nova = _converse(_media("image"))
    encoded = base64.b64encode(json.dumps(nova).encode()).decode()
    return {"input": {"invokeModel": {"body": encoded}}}


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/model/amazon.nova-pro-v1%3A0/converse", _converse(_media("image"))),
        (
            "/guardrail/gr1/version/1/apply",
            {
                "source": "INPUT",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": BLOB}}},
                    {"text": {"text": EMAIL}},
                ],
            },
        ),
        (
            "/model/amazon.nova-pro-v1%3A0/count-tokens",
            {"input": {"converse": _converse(_media("image"), _media("document"))}},
        ),
        ("/model/amazon.nova-pro-v1%3A0/count-tokens", _count_tokens_invoke()),
        (
            "/async-invoke",
            {
                "modelId": "amazon.nova-reel-v1:0",
                "modelInput": {
                    "taskType": "TEXT_VIDEO",
                    "textToVideoParams": {
                        "text": f"a video for {EMAIL}",
                        "images": [{"format": "png", "source": {"bytes": BLOB}}],
                    },
                },
                "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://b/out"}},
            },
        ),
    ],
    ids=[
        "converse",
        "apply-guardrail",
        "count-tokens-converse",
        "count-tokens-invoke",
        "async-invoke",
    ],
)
async def test_media_bytes_reach_bedrock_byte_identical(path: str, body: dict[str, Any]) -> None:
    sent: list[bytes] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return httpx.Response(200, json={})

    config = Config(providers={**Config().providers, "bedrock": ProviderConfig(BEDROCK)})
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(path, json=body, headers={"authorization": "Bearer k"})
    assert response.status_code == 200
    (forwarded,) = sent
    text = forwarded.decode()
    if "invokeModel" in text:
        inner = json.loads(text)["input"]["invokeModel"]["body"]
        text = base64.b64decode(inner).decode()
    assert BLOB in text  # the media, untouched
    assert EMAIL not in text and "«EMAIL_" in text  # the text beside it, redacted
    assert "«BITBUCKET" not in text
