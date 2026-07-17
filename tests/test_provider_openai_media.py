"""OpenAI media endpoints whose REQUESTS carry text: images, speech, videos.

The media non-goal applies to media BYTES — prompts are plain text and
must not reach the provider in the clear. /v1/images/generations is a
JSON body (prompt walked); /v1/images/edits is multipart whose prompt is
a plain form FIELD next to image/mask file parts (only named text fields
are scanned; the file parts stay byte-identical); /v1/images/variations
carries no text at all and stays pass-through.
"""

from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig
from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors
from llm_redact.multipart import parse
from llm_redact.providers.base import RouteKind
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.proxy import create_app
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"
BOUNDARY = b"mediaboundary42"


def _redactor(vault: InMemoryVault) -> Redactor:
    return Redactor(
        detectors=build_detectors(DetectionConfig()),
        vault=vault,
        allowlist=Allowlist(exact=frozenset(), patterns=()),
    )


def _edits_body(prompt: bytes) -> bytes:
    return (
        b"--mediaboundary42\r\n"
        b'Content-Disposition: form-data; name="model"\r\n'
        b"\r\n"
        b"gpt-image-1\r\n"
        b"--mediaboundary42\r\n"
        b'Content-Disposition: form-data; name="prompt"\r\n'
        b"\r\n" + prompt + b"\r\n"
        b"--mediaboundary42\r\n"
        b'Content-Disposition: form-data; name="image"; filename="in.png"\r\n'
        b"Content-Type: image/png\r\n"
        b"\r\n"
        b"\x89PNG\r\n\x1a\n fake image bytes jane.doe@corp.example\r\n"
        b"--mediaboundary42--\r\n"
    )


def test_images_routing() -> None:
    adapter = OpenAIAdapter()
    assert adapter.matches("POST", "/v1/images/generations") is RouteKind.REDACT_ONLY
    assert adapter.matches("POST", "/v1/images/edits") is RouteKind.REDACT_ONLY
    # variations uploads an image and nothing else: no text to protect.
    assert adapter.matches("POST", "/v1/images/variations") is RouteKind.NONE
    # No note injection on media bodies — there is no messages field.
    assert not adapter.wants_system_note(RouteKind.REDACT_ONLY, "/v1/images/generations")
    assert not adapter.wants_system_note(RouteKind.REDACT_ONLY, "/v1/images/edits")


def test_edits_prompt_field_redacted_file_part_untouched() -> None:
    adapter = OpenAIAdapter()
    body = _edits_body(f"a birthday card for {EMAIL}".encode())
    out = adapter.redact_multipart(
        "/v1/images/edits", body, BOUNDARY, _redactor(InMemoryVault()), inject_note=False
    )
    assert out is not None
    parsed = parse(out, BOUNDARY)
    assert parsed is not None
    assert parsed.parts[0].content == b"gpt-image-1"  # non-prompt field untouched
    assert parsed.parts[1].content == "a birthday card for «EMAIL_001»".encode()
    # The image FILE part is media: byte-identical even though its bytes
    # happen to contain an email-shaped string.
    assert parsed.parts[2].content.startswith(b"\x89PNG")
    assert EMAIL.encode() in parsed.parts[2].content


def test_edits_prompt_with_filename_attribute_still_scanned() -> None:
    # Fail-closed: a prompt field dressed up as a file upload (filename
    # attribute set) is matched by NAME and scanned all the same.
    adapter = OpenAIAdapter()
    body = (
        b"--mediaboundary42\r\n"
        b'Content-Disposition: form-data; name="prompt"; filename="p.txt"\r\n'
        b"Content-Type: text/plain\r\n"
        b"\r\n"
        b"contact jane.doe@corp.example\r\n"
        b"--mediaboundary42--\r\n"
    )
    out = adapter.redact_multipart(
        "/v1/images/edits", body, BOUNDARY, _redactor(InMemoryVault()), inject_note=False
    )
    assert out is not None
    parsed = parse(out, BOUNDARY)
    assert parsed is not None
    assert parsed.parts[0].content == "contact «EMAIL_001»".encode()


def test_edits_without_secrets_forwards_verbatim() -> None:
    adapter = OpenAIAdapter()
    body = _edits_body(b"a plain landscape")
    out = adapter.redact_multipart(
        "/v1/images/edits", body, BOUNDARY, _redactor(InMemoryVault()), inject_note=False
    )
    assert out is None  # nothing changed: the proxy forwards the ORIGINAL bytes


def test_edits_non_utf8_prompt_left_alone() -> None:
    adapter = OpenAIAdapter()
    body = _edits_body(b"\xff\xfe not text")
    out = adapter.redact_multipart(
        "/v1/images/edits", body, BOUNDARY, _redactor(InMemoryVault()), inject_note=False
    )
    assert out is None


# --- integration: real app, fake images upstream -----------------------------

received: dict[str, Any] = {}


def _fake_upstream() -> Starlette:
    async def generations(request: Request) -> Response:
        received["generations"] = await request.json()
        return JSONResponse({"created": 1, "data": [{"b64_json": "aWNvbg=="}]})

    async def edits(request: Request) -> Response:
        received["edits_raw"] = await request.body()
        return JSONResponse({"created": 1, "data": [{"b64_json": "aWNvbg=="}]})

    async def variations(request: Request) -> Response:
        received["variations_raw"] = await request.body()
        return JSONResponse({"created": 1, "data": [{"url": "https://img.example/1"}]})

    return Starlette(
        routes=[
            Route("/v1/images/generations", generations, methods=["POST"]),
            Route("/v1/images/edits", edits, methods=["POST"]),
            Route("/v1/images/variations", variations, methods=["POST"]),
        ]
    )


@pytest.fixture
def client() -> httpx.AsyncClient:
    received.clear()
    config = Config(providers={"openai": ProviderConfig(upstream_base_url="http://upstream")})
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


@pytest.mark.anyio
async def test_generations_prompt_redacted(client: httpx.AsyncClient) -> None:
    reply = await client.post(
        "/v1/images/generations",
        json={"model": "gpt-image-1", "prompt": f"a card for {EMAIL}", "size": "1024x1024"},
    )
    assert reply.status_code == 200
    assert received["generations"]["prompt"] == "a card for «EMAIL_001»"
    assert received["generations"]["model"] == "gpt-image-1"  # structural key untouched
    # The media response comes back verbatim.
    assert reply.json()["data"][0]["b64_json"] == "aWNvbg=="


@pytest.mark.anyio
async def test_edits_round_trip_through_proxy(client: httpx.AsyncClient) -> None:
    reply = await client.post(
        "/v1/images/edits",
        data={"model": "gpt-image-1", "prompt": f"add a note to {EMAIL}"},
        files={"image": ("in.png", b"\x89PNG fake", "image/png")},
    )
    assert reply.status_code == 200
    assert EMAIL.encode() not in received["edits_raw"]
    assert "«EMAIL_001»".encode() in received["edits_raw"]
    assert b"\x89PNG fake" in received["edits_raw"]  # image part byte-identical


@pytest.mark.anyio
async def test_variations_pass_through_verbatim(client: httpx.AsyncClient) -> None:
    reply = await client.post(
        "/v1/images/variations",
        files={"image": ("in.png", b"\x89PNG fake", "image/png")},
    )
    assert reply.status_code == 200
    assert b"\x89PNG fake" in received["variations_raw"]


def test_deny_string_in_prompt_blocks_nothing_but_redacts() -> None:
    # Deny strings are tier 0 and ALWAYS redact — including inside a
    # multipart prompt form field.
    from llm_redact.detection.deny import DenyDetector, DenyEntry

    vault = InMemoryVault()
    redactor = Redactor(
        detectors=(
            *build_detectors(DetectionConfig()),
            DenyDetector([DenyEntry("Project Nightingale")]),
        ),
        vault=vault,
        allowlist=Allowlist(exact=frozenset(), patterns=()),
    )
    body = _edits_body(b"logo for Project Nightingale")
    out = OpenAIAdapter().redact_multipart(
        "/v1/images/edits", body, BOUNDARY, redactor, inject_note=False
    )
    assert out is not None
    parsed = parse(out, BOUNDARY)
    assert parsed is not None
    assert b"Project Nightingale" not in parsed.parts[1].content


# --- audio/speech + videos (P22.3) -------------------------------------------


def test_speech_and_videos_routing() -> None:
    adapter = OpenAIAdapter()
    assert adapter.matches("POST", "/v1/audio/speech") is RouteKind.REDACT_ONLY
    # Audio UPLOADS are the media non-goal: never scanned, never rerouted.
    assert adapter.matches("POST", "/v1/audio/transcriptions") is RouteKind.NONE
    assert adapter.matches("POST", "/v1/audio/translations") is RouteKind.NONE
    # Video jobs echo their prompt, so every JSON surface is CHAT ...
    assert adapter.matches("POST", "/v1/videos") is RouteKind.CHAT
    assert adapter.matches("GET", "/v1/videos") is RouteKind.CHAT
    assert adapter.matches("GET", "/v1/videos/video_1") is RouteKind.CHAT
    assert adapter.matches("POST", "/v1/videos/video_1/remix") is RouteKind.CHAT
    # ... while the binary download and delete stay pass-through.
    assert adapter.matches("GET", "/v1/videos/video_1/content") is RouteKind.NONE
    assert adapter.matches("DELETE", "/v1/videos/video_1") is RouteKind.NONE
    # No note injection anywhere near these bodies: no messages field.
    assert not adapter.wants_system_note(RouteKind.REDACT_ONLY, "/v1/audio/speech")
    assert not adapter.wants_system_note(RouteKind.CHAT, "/v1/videos")
    assert not adapter.wants_system_note(RouteKind.CHAT, "/v1/videos/video_1/remix")


def _media_upstream() -> Starlette:
    async def speech(request: Request) -> Response:
        received["speech"] = await request.json()
        return Response(content=b"ID3 fake mp3 bytes", media_type="audio/mpeg")

    async def create_video(request: Request) -> Response:
        body = await request.json()
        received["video_create"] = body
        # The job object echoes the (redacted) prompt — the real API does.
        return JSONResponse(
            {"id": "video_1", "object": "video", "status": "queued", "prompt": body["prompt"]}
        )

    async def get_video(request: Request) -> Response:
        return JSONResponse(
            {
                "id": "video_1",
                "object": "video",
                "status": "completed",
                "prompt": received["video_create"]["prompt"],
            }
        )

    async def list_videos(request: Request) -> Response:
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": "video_1", "prompt": received["video_create"]["prompt"]}],
            }
        )

    async def content(request: Request) -> Response:
        return Response(content=b"\x00\x00ftypmp42 fake video", media_type="video/mp4")

    return Starlette(
        routes=[
            Route("/v1/audio/speech", speech, methods=["POST"]),
            Route("/v1/videos", create_video, methods=["POST"]),
            Route("/v1/videos", list_videos, methods=["GET"]),
            Route("/v1/videos/video_1", get_video, methods=["GET"]),
            Route("/v1/videos/video_1/content", content, methods=["GET"]),
        ]
    )


@pytest.fixture
def media_client() -> httpx.AsyncClient:
    received.clear()
    config = Config(providers={"openai": ProviderConfig(upstream_base_url="http://upstream")})
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_media_upstream()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


@pytest.mark.anyio
async def test_speech_input_redacted_audio_verbatim(media_client: httpx.AsyncClient) -> None:
    reply = await media_client.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "input": f"call {EMAIL} tomorrow", "voice": "alloy"},
    )
    assert reply.status_code == 200
    assert received["speech"]["input"] == "call «EMAIL_001» tomorrow"
    assert "messages" not in received["speech"]  # no note grafted
    assert reply.content == b"ID3 fake mp3 bytes"  # audio bytes verbatim


@pytest.mark.anyio
async def test_video_prompt_redacted_and_echo_restored(media_client: httpx.AsyncClient) -> None:
    created = await media_client.post(
        "/v1/videos",
        json={"model": "sora-2", "prompt": f"a note to {EMAIL}", "seconds": "8"},
    )
    assert created.status_code == 200
    assert received["video_create"]["prompt"] == "a note to «EMAIL_001»"
    assert "messages" not in received["video_create"]
    # The upstream echoed the placeholder; the tool sees the original.
    assert created.json()["prompt"] == f"a note to {EMAIL}"

    fetched = await media_client.get("/v1/videos/video_1")
    assert fetched.json()["prompt"] == f"a note to {EMAIL}"

    # The list envelope ({object: list, data: [...]}) is walked too.
    listed = await media_client.get("/v1/videos")
    assert listed.json()["data"][0]["prompt"] == f"a note to {EMAIL}"

    # The binary download is media: byte-identical pass-through.
    video = await media_client.get("/v1/videos/video_1/content")
    assert video.content == b"\x00\x00ftypmp42 fake video"
