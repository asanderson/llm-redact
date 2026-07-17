"""The /__llm-redact/events live SSE feed: fan-out, stream shape, guards.

The stream is infinite, so it must never be driven through ASGITransport to
completion (the transport buffers the whole body — the test would hang).
The stream-shape test speaks raw ASGI instead and cancels the app task once
it has the frames it needs.
"""

import asyncio
import contextlib
import json
import time
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig
from llm_redact.proxy import ProxyState, create_app

EMAIL = "jane.doe@corp.example"


def _fake_upstream() -> Starlette:
    async def messages(request: Request) -> JSONResponse:
        return JSONResponse({"content": [{"type": "text", "text": "ok"}], "role": "assistant"})

    return Starlette(routes=[Route("/v1/messages", messages, methods=["POST"])])


def _make_app() -> Any:
    config = Config(
        providers={**Config().providers, "anthropic": ProviderConfig("http://upstream")}
    )
    return create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))


def _record(state: ProxyState, **overrides: Any) -> None:
    row: dict[str, Any] = {
        "session": "default",
        "provider": "anthropic",
        "method": "POST",
        "path": "/v1/messages",
        "status": 200,
        "started": time.perf_counter(),
        "streamed": False,
        "detections": {"EMAIL": 1},
        "rehydrations": {},
    }
    row.update(overrides)
    state.record_request(**row)


def test_record_request_fans_out_to_subscribers() -> None:
    state: ProxyState = _make_app().state.proxy
    first: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
    full: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
    full.put_nowait({"stuffed": True})
    state.event_subscribers.update({first, full})

    _record(state)

    row = first.get_nowait()
    assert row["path"] == "/v1/messages"
    assert row["detections"] == {"EMAIL": 1}
    assert row == state.recent[-1]
    # Metadata only — the same row shape as /recent, never values.
    assert EMAIL not in json.dumps(row)
    # The full queue dropped the event (slow consumer) without raising and
    # without disturbing anyone else.
    assert full.get_nowait() == {"stuffed": True}
    assert full.empty()


async def test_events_stream_shape_and_cleanup() -> None:
    app = _make_app()
    state: ProxyState = app.state.proxy
    frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def receive() -> dict[str, Any]:
        await asyncio.Event().wait()  # client never disconnects
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        await frames.put(message)

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/__llm-redact/events",
        "raw_path": b"/__llm-redact/events",
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1:8787")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8787),
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        start = await asyncio.wait_for(frames.get(), timeout=5)
        assert start["type"] == "http.response.start"
        assert start["status"] == 200
        headers = {k.decode(): v.decode() for k, v in start["headers"]}
        assert headers["content-type"].startswith("text/event-stream")
        assert headers["cache-control"] == "no-store"

        connected = await asyncio.wait_for(frames.get(), timeout=5)
        assert connected["body"] == b": connected\n\n"
        assert len(state.event_subscribers) == 1

        # A request lands while the stream is open: its row arrives as one
        # SSE data event, identical to what /recent serves.
        _record(state)
        event = await asyncio.wait_for(frames.get(), timeout=5)
        assert event["body"].startswith(b"data: ")
        assert event["body"].endswith(b"\n\n")
        row = json.loads(event["body"][len(b"data: ") : -2])
        assert row == state.recent[-1]
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    # Disconnect (here: cancellation) unregisters the subscriber.
    assert state.event_subscribers == set()


async def test_events_reject_foreign_host() -> None:
    # DNS-rebinding defense applies to the feed like the other local reads.
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_make_app()), base_url="http://evil.example"
    )
    assert (await client.get("/__llm-redact/events")).status_code == 403


async def test_events_post_rejected() -> None:
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_make_app()), base_url="http://127.0.0.1:8787"
    )
    assert (await client.post("/__llm-redact/events", content=b"x")).status_code == 405
