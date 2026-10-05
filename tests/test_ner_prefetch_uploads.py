"""NER off the event loop for multipart uploads and realtime client frames:
what reaches the upstream is byte-for-byte what inline detection sends, and
no model runs on the event loop.

Uploads ride ASGITransport; realtime needs real sockets (uvicorn on port 0
and a fake ``websockets`` upstream, the test_realtime_relay pattern).
"""

import asyncio
import contextlib
import dataclasses
import json
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import uvicorn

import llm_redact.proxy as proxy_module
import llm_redact.realtime as realtime_module
import llm_redact.registry as registry_mod
from llm_redact.config import Config, ProviderConfig
from llm_redact.plugin_api import Inspection, UploadPart
from llm_redact.proxy import create_app
from llm_redact.registry import Registry
from ner_fakes import install_spacy
from prefetch_fixtures import ANTHROPIC_HEADERS, EMAIL, NAME, OPENAI_HEADERS, OTHER
from test_ner_prefetch_e2e import SlowNlp, Upstream, _client, _config, _stats

GOOGLE = {"x-goog-api-key": "AIzaFAKE"}
PDF = b"%PDF-1.7\n1 0 obj << >> endobj\n%%EOF\n"


def _form(purpose: str, filename: str, content: bytes, content_type: str) -> bytes:
    return (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\n'
        + purpose.encode()
        + b'\r\n--b\r\nContent-Disposition: form-data; name="file"; filename="'
        + filename.encode()
        + b'"\r\nContent-Type: '
        + content_type.encode()
        + b"\r\n\r\n"
        + content
        + b"\r\n--b--\r\n"
    )


def _related(*parts: tuple[str, bytes]) -> bytes:
    out = b""
    for content_type, content in parts:
        out += b"--b\r\nContent-Type: " + content_type.encode() + b"\r\n\r\n" + content + b"\r\n"
    return out + b"--b--\r\n"


_LINE = json.dumps(
    {
        "custom_id": "r1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "m", "messages": [{"role": "user", "content": f"{NAME} {EMAIL}"}]},
    }
)
FORM = {"content-type": "multipart/form-data; boundary=b"}
TEXT = f"Mail {NAME} at {EMAIL}; cc {OTHER}.\n{NAME} again\n".encode()

# (id, path, headers, body)
UPLOADS: list[tuple[str, str, dict[str, str], bytes]] = [
    (
        "openai-text",
        "/v1/files",
        {**OPENAI_HEADERS, **FORM},
        _form("assistants", f"notes of {OTHER}.txt", TEXT, "text/plain"),
    ),
    (
        "openai-batch-jsonl",
        "/v1/files",
        {**OPENAI_HEADERS, **FORM},
        _form("batch", "batch.jsonl", (_LINE + "\n" + _LINE + "\n").encode(), "application/jsonl"),
    ),
    (
        "openai-binary",
        "/v1/files",
        {**OPENAI_HEADERS, **FORM},
        _form("user_data", f"{NAME}.pdf", PDF, "application/pdf"),
    ),
    (
        "anthropic-files",
        "/v1/files",
        {**ANTHROPIC_HEADERS, **FORM},
        _form("assistants", "notes.txt", TEXT, "text/plain"),
    ),
    (
        "gemini-related",
        "/upload/v1beta/files",
        {
            **GOOGLE,
            "x-goog-upload-protocol": "multipart",
            "content-type": "multipart/related; boundary=b",
        },
        _related(
            (
                "application/json; charset=UTF-8",
                json.dumps({"file": {"display_name": f"notes of {NAME}"}}).encode(),
            ),
            ("text/plain", TEXT),
        ),
    ),
]


async def _prefetch_off(*args: Any, **kwargs: Any) -> None:
    return None


async def _upload(
    monkeypatch: pytest.MonkeyPatch, path: str, headers: dict[str, str], body: bytes, *, on: bool
) -> tuple[bytes, Any]:
    with monkeypatch.context() as patch:
        if not on:
            patch.setattr(proxy_module, "_prefetch_upload", _prefetch_off)
        nlp = SlowNlp()
        install_spacy(patch, nlp)
        upstream = Upstream()
        app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
        async with _client(app) as client:
            response = await client.post(path, headers=headers, content=body)
        assert response.status_code == 200, response.text
        (sent,) = upstream.bodies
        assert (threading.get_ident() in nlp.threads) is not on
        return sent, _stats(app)


@pytest.mark.parametrize(
    ("path", "headers", "body"),
    [row[1:] for row in UPLOADS],
    ids=[row[0] for row in UPLOADS],
)
async def test_an_upload_sends_the_same_bytes_with_and_without_prefetch(
    monkeypatch: pytest.MonkeyPatch, path: str, headers: dict[str, str], body: bytes
) -> None:
    inline, inline_stats = await _upload(monkeypatch, path, headers, body, on=False)
    prefetched, stats = await _upload(monkeypatch, path, headers, body, on=True)
    assert prefetched == inline
    assert "«PERSON_001»".encode() in prefetched
    assert inline_stats.inline_calls > 0
    assert stats.inline_calls == stats.prefetch_misses == 0


# --- an upload inspector's extracted texts (convert mode) ---------------------


class Inspector:
    """A scripted upload inspector: every binary part reads as a text
    holding names, with a display text to convert it to."""

    timeout = 5.0
    max_bytes = 1 << 20

    async def inspect(self, part: UploadPart) -> Inspection:
        return Inspection(
            f"Report by {NAME} for {OTHER}",
            True,
            "fake",
            convert_text=f"Report by {NAME}",
        )

    def status(self) -> dict[str, Any]:
        return {}

    async def aclose(self) -> None:
        return None


@pytest.mark.parametrize("modes", [{}, {"email": "block"}], ids=["redact", "block-check"])
async def test_an_inspected_upload_detects_its_extracted_texts_off_the_loop(
    monkeypatch: pytest.MonkeyPatch, modes: dict[str, str]
) -> None:
    reg = Registry()
    reg.build_upload_inspector = lambda config, tier: Inspector()  # type: ignore[method-assign]
    monkeypatch.setattr(registry_mod, "_registry", reg)
    config = _config()
    detection = dataclasses.replace(config.detection, modes=tuple(modes.items()))
    body = _form("user_data", f"{NAME}.pdf", PDF, "application/pdf")

    async def send(on: bool) -> tuple[bytes, Any]:
        with monkeypatch.context() as patch:
            if not on:
                patch.setattr(proxy_module, "_prefetch_upload", _prefetch_off)
            install_spacy(patch, SlowNlp())
            upstream = Upstream()
            app = create_app(
                dataclasses.replace(config, detection=detection),
                upstream_transport=httpx.MockTransport(upstream),
            )
            async with _client(app) as client:
                response = await client.post(
                    "/v1/files", headers={**OPENAI_HEADERS, **FORM}, content=body
                )
            assert response.status_code == 200, response.text
            return upstream.bodies[0], _stats(app)

    inline, inline_stats = await send(False)
    prefetched, stats = await send(True)
    assert prefetched == inline
    # Converted: the display text, redacted, in the file's place.
    assert b"Report by \xc2\xabPERSON_001\xc2\xbb" in prefetched and PDF not in prefetched
    assert inline_stats.inline_calls > 0
    assert stats.inline_calls == stats.prefetch_misses == 0


# --- realtime client frames ---------------------------------------------------


websockets = pytest.importorskip("websockets")

FRAMES = [
    json.dumps({"type": "session.update", "session": {"instructions": f"Help {NAME}."}}),
    json.dumps(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"Mail {OTHER} at {EMAIL}"}],
            },
        }
    ),
    "not json at all",
]


class FakeUpstream:
    """A websockets server recording the frames it receives (no echo)."""

    def __init__(self) -> None:
        self.received: list[str | bytes] = []
        self.port = 0

    async def _handler(self, connection: Any) -> None:
        async for message in connection:
            self.received.append(message)

    @contextlib.asynccontextmanager
    async def serving(self) -> AsyncIterator["FakeUpstream"]:
        server = await websockets.serve(self._handler, "127.0.0.1", 0)
        self.port = server.sockets[0].getsockname()[1]
        try:
            yield self
        finally:
            server.close()
            await server.wait_closed()


@contextlib.contextmanager
def _serve(app: Any) -> Iterator[str]:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _ws_config(port: int) -> Config:
    config = _config()
    return dataclasses.replace(
        config,
        providers={**config.providers, "openai": ProviderConfig(f"http://127.0.0.1:{port}")},
    )


async def _relay_frames(monkeypatch: pytest.MonkeyPatch, *, on: bool) -> tuple[list[Any], Any]:
    with monkeypatch.context() as patch:
        if not on:
            patch.setattr(realtime_module, "_prefetch_frame", _prefetch_off)
        nlp = SlowNlp()
        install_spacy(patch, nlp)
        upstream = FakeUpstream()
        async with upstream.serving():
            app = create_app(_ws_config(upstream.port))
            with _serve(app) as host:
                async with websockets.connect(f"ws://{host}/v1/realtime?model=m") as client:
                    for frame in FRAMES:
                        await client.send(frame)
                    for _ in range(200):
                        if len(upstream.received) == len(FRAMES):
                            break
                        await asyncio.sleep(0.01)
        return upstream.received, _stats(app)


async def test_realtime_frames_are_sent_the_same_with_and_without_prefetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inline, inline_stats = await _relay_frames(monkeypatch, on=False)
    prefetched, stats = await _relay_frames(monkeypatch, on=True)
    assert len(prefetched) == len(FRAMES)
    assert prefetched == inline
    assert "«PERSON_001»" in prefetched[0] and "«PERSON_002»" in prefetched[1]
    assert prefetched[2] == "not json at all"
    assert inline_stats.inline_calls > 0
    assert stats.inline_calls == stats.prefetch_misses == 0


async def test_a_reload_during_a_frames_prefetch_never_sends_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_spacy(monkeypatch, SlowNlp())
    upstream = FakeUpstream()
    async with upstream.serving():
        app = create_app(_ws_config(upstream.port))
        state = app.state.proxy
        real = realtime_module.prefetch
        reloaded = dataclasses.replace(
            state.config,
            detection=dataclasses.replace(
                state.config.detection,
                enabled=tuple(n for n in state.config.detection.enabled if n != "email"),
            ),
        )

        async def prefetch_then_reload(plan: Any, strings: Any) -> Any:
            table = await real(plan, strings)
            state.apply_config(reloaded)  # on the server's loop
            return table

        monkeypatch.setattr(realtime_module, "prefetch", prefetch_then_reload)
        with _serve(app) as host:
            async with websockets.connect(f"ws://{host}/v1/realtime?model=m") as client:
                await client.send(FRAMES[1])
                with pytest.raises(websockets.ConnectionClosed) as closed:
                    await asyncio.wait_for(client.recv(), 5)
        assert closed.value.rcvd is not None and closed.value.rcvd.code == 1012
    assert upstream.received == []
