"""The session router's observation of realtime UPSTREAM frames (the optional
``SessionRouter.realtime_server_frame``).

The server-side twin of the per-frame client check: a router with the member
is handed EVERY upstream frame of a realtime connection that parses as JSON
(OpenAI Realtime, Azure, Gemini Live and Vertex Live; text or binary), as its
own parse of the provider's bytes — placeholders, never a restored value —
synchronously BEFORE the frame is restored or sent, in the connection's own
context. It cannot change or refuse the frame: its return value is ignored
and what it does to its parse never reaches the client. A router that raises
is contained (bookkeeping stage ``realtime_server_frame``, type-only log) and
the frame is delivered as usual. Frames that are not JSON, or nest deeper
than the proxy's bound, are not observed; a router without the member costs
no parse at all.

Real sockets end to end: uvicorn on port 0 in a thread, a fake ``websockets``
upstream in the test's loop (the test_realtime_reload harness), a scripted
router on a bare Registry.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
import websockets

from llm_redact import realtime
from llm_redact.jsonwalk import MAX_JSON_DEPTH
from llm_redact.proxy import REALTIME_SERVER_FRAME_STAGE
from test_realtime_frame_check import _ADMITTED, ADAPTERS, IDENTITY, FrameRouter, _config, _install
from test_realtime_identity import EMAIL, _recent
from test_realtime_reload import PATHS, Upstream, _frame, _serve, _until

pytestmark = pytest.mark.asyncio

PLACEHOLDER = "«EMAIL_001»"


class ObservingRouter:
    """A static-mode session router with only the server-frame observer:
    records every call (with what the gate set for the connection), then
    tampers with its parse (which must never reach the client) — or raises."""

    mode = "static"

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def resolve(self, adapter_name: Any, method: str, path: str, body: Any) -> str:
        return "default"

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def realtime_server_frame(
        self, adapter_name: str, path: str, frame: Any, *, identity: bool, session_id: str
    ) -> Any:
        self.calls.append(
            {
                "adapter": adapter_name,
                "path": path,
                "frame": json.dumps(frame, ensure_ascii=False),
                "identity": identity,
                "session": session_id,
                "user": _ADMITTED.get(),
            }
        )
        if self.error is not None:
            raise self.error
        if isinstance(frame, dict):
            frame.clear()  # tampering with its own parse changes nothing
        return "ignored"


class Replying(Upstream):
    """The echoing upstream that first sends ``greetings`` (frames the proxy
    must relay as they came) on every connection."""

    def __init__(self, greetings: list[str | bytes]) -> None:
        super().__init__()
        self.greetings = greetings

    async def _handler(self, connection: Any) -> None:
        for greeting in self.greetings:
            await connection.send(greeting)
        await super()._handler(connection)


def _text(frame: str | bytes) -> str:
    return frame.decode() if isinstance(frame, bytes) else frame


@pytest.mark.parametrize("provider", sorted(ADAPTERS))
async def test_every_upstream_frame_is_observed_as_the_provider_sent_it(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    router = ObservingRouter()
    _install(monkeypatch, router, gate=True)  # type: ignore[arg-type]
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                await client.send(_frame(provider, f"mail {EMAIL}"))
                echo = _text(await client.recv())
            await _until(lambda: len(fake.close_codes) == 1)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            bookkeeping = dict(proxy.state.bookkeeping_errors)
    # The upstream got the placeholder; its echo was observed AS SENT
    # (placeholders, never the restored value), in the connection's context.
    assert PLACEHOLDER in fake.texts()[0]
    [call] = router.calls
    assert (call["adapter"], call["identity"], call["session"], call["user"]) == (
        ADAPTERS[provider],
        provider in IDENTITY,
        "default",
        "ada",
    )
    assert call["path"] == PATHS[provider].split("?")[0]
    assert PLACEHOLDER in call["frame"] and EMAIL not in call["frame"]
    # What the router did to its parse never reached the client.
    assert echo != "{}"
    if provider in ("openai", "azure"):
        assert EMAIL in echo  # the item echo is restored as usual
    else:
        # A client message echoed back is no server message shape: relayed
        # as it came (the observation changed nothing either).
        assert PLACEHOLDER in echo
    assert row["status"] == 101
    assert REALTIME_SERVER_FRAME_STAGE not in bookkeeping


@pytest.mark.parametrize("provider", ["gemini", "vertex"])
async def test_a_live_server_message_is_observed_before_it_is_restored(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    # A Gemini Live server message (the resumption update pro records) and
    # model output naming a placeholder issued on this connection.
    router = ObservingRouter()
    _install(monkeypatch, router)  # type: ignore[arg-type]

    class Answering(Upstream):
        async def _handler(self, connection: Any) -> None:
            async for message in connection:
                self.received.append(message)
                update = {"sessionResumptionUpdate": {"newHandle": "h-1", "resumable": True}}
                await connection.send(json.dumps(update).encode())
                parts = [{"text": f"wrote to {PLACEHOLDER}"}]
                answer = {"serverContent": {"modelTurn": {"parts": parts}, "turnComplete": True}}
                await connection.send(json.dumps(answer).encode())

    async with Answering() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                await client.send(_frame(provider, f"mail {EMAIL}"))
                update = _text(await client.recv())
                answer = _text(await client.recv())
    assert json.loads(update) == {
        "sessionResumptionUpdate": {"newHandle": "h-1", "resumable": True}
    }
    assert EMAIL in answer and PLACEHOLDER not in answer
    assert [json.loads(call["frame"]) for call in router.calls] == [
        {"sessionResumptionUpdate": {"newHandle": "h-1", "resumable": True}},
        {
            "serverContent": {
                "modelTurn": {"parts": [{"text": f"wrote to {PLACEHOLDER}"}]},
                "turnComplete": True,
            }
        },
    ]


@pytest.mark.parametrize("provider", sorted(ADAPTERS))
async def test_an_observer_that_raises_is_contained_and_the_frame_delivered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, provider: str
) -> None:
    router = ObservingRouter(error=RuntimeError(f"cannot record {EMAIL} {PLACEHOLDER}"))
    _install(monkeypatch, router)  # type: ignore[arg-type]
    caplog.set_level(logging.WARNING, logger="llm_redact")
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                await client.send(_frame(provider, f"mail {EMAIL}"))
                first = _text(await client.recv())
                await client.send(_frame(provider, "second"))
                second = _text(await client.recv())
            await _until(lambda: len(fake.close_codes) == 1)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            counted = proxy.state.bookkeeping_errors[REALTIME_SERVER_FRAME_STAGE]
    # Both frames delivered, the connection never closed by the fault.
    assert len(router.calls) == 2
    assert "second" in second
    if provider in ("openai", "azure"):
        assert EMAIL in first
    assert row["status"] == 101
    assert counted == 2
    logged = [r.getMessage() for r in caplog.records if "realtime_server_frame" in r.getMessage()]
    assert len(logged) == 2 and all("RuntimeError" in line for line in logged)
    assert not any(EMAIL in line or PLACEHOLDER in line for line in caplog.messages)


async def test_frames_that_are_not_json_or_nest_too_deep_are_not_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ObservingRouter()
    _install(monkeypatch, router)  # type: ignore[arg-type]
    deep = "[" * (MAX_JSON_DEPTH + 5) + "]" * (MAX_JSON_DEPTH + 5)
    greetings: list[str | bytes] = ["not json", deep, b"\xff\xfe binary", '{"type": "ok"}']
    async with Replying(greetings) as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS['openai']}") as client:
                relayed = [await asyncio.wait_for(client.recv(), 5) for _ in greetings]
    # Every frame relayed as it came; only the JSON one observed.
    assert relayed == greetings
    assert [json.loads(call["frame"]) for call in router.calls] == [{"type": "ok"}]


async def test_a_router_without_the_member_costs_no_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = FrameRouter()
    _install(monkeypatch, router)
    observed: list[Any] = []
    monkeypatch.setattr(realtime, "_observe_server_frame", lambda *a, **k: observed.append(a))
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS['openai']}") as client:
                await client.send(_frame("openai", f"mail {EMAIL}"))
                assert EMAIL in _text(await client.recv())
            assert proxy.state.observes_realtime_server_frames is False
    assert observed == []
