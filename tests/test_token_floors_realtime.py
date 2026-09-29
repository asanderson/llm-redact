"""Token floors on a realtime WebSocket connection (real sockets end to end).

A realtime conversation lives on the provider: the client sends each turn
once and never resends history. So the floor is a RUNNING one — every
client frame's tokens raise it for the rest of the connection — or a token
restored into the conversation by an early frame could be issued to a new
value by a later one. Like test_realtime_relay: uvicorn on port 0 in a
thread, a fake ``websockets`` upstream in the test's loop.
"""

from __future__ import annotations

import json
from typing import Any

import websockets

from llm_redact.realtime import frame_floors
from test_realtime_relay import FakeUpstream, _config, _proxy

BOB = "bob@corp.example"


class ReplyingUpstream(FakeUpstream):
    """Answers every conversation item with a ``response.output_text.done``
    repeating the item's text exactly as the upstream received it."""

    async def _handler(self, connection: Any) -> None:
        self.paths.append(connection.request.path)
        async for message in connection:
            self.received.append(message)
            event = json.loads(message)
            if event.get("type") != "conversation.item.create":
                continue
            reply = {
                "type": "response.output_text.done",
                "item_id": f"item_{len(self.received)}",
                "output_index": 0,
                "content_index": 0,
                "text": event["item"]["content"][0]["text"],
            }
            await connection.send(json.dumps(reply, ensure_ascii=False))


def _item(text: str) -> str:
    return json.dumps(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        },
        ensure_ascii=False,
    )


def _sent_text(frame: str | bytes) -> str:
    return str(json.loads(frame)["item"]["content"][0]["text"])


async def _say(client: Any, text: str) -> str:
    await client.send(_item(text))
    return str(json.loads(await client.recv())["text"])


async def test_an_earlier_frames_tokens_bound_a_later_frames_values() -> None:
    async with ReplyingUpstream() as fake:
        with _proxy(_config(fake.port)) as proxy_host:
            async with websockets.connect(f"ws://{proxy_host}/v1/realtime") as client:
                # A client restoring an earlier conversation: tokens its
                # (fresh, static) session never issued, no new value yet.
                history = "Earlier: «EMAIL_001» wrote to «EMAIL_003» («email_2»)."
                restored = await _say(client, history)
                new_value = await _say(client, f"mail {BOB}")
                echo_foreign = await _say(client, "remember «EMAIL_001»?")
                both = await _say(client, f"«EMAIL_007» and {BOB} and carol@corp.example")

    sent = [_sent_text(frame) for frame in fake.received]
    # The second frame carries no token itself, yet its value is numbered
    # above the FIRST frame's: the conversation on the provider holds both.
    assert sent[1] == "mail «EMAIL_004»"
    # One frame carrying a token and new values: numbered above it too, and
    # a value already mapped keeps its token.
    assert sent[3] == "«EMAIL_007» and «EMAIL_004» and «EMAIL_008»"
    # The foreign tokens were never issued here: their echoes stay
    # placeholders, never the new value; the connection's own restore.
    assert restored == "Earlier: «EMAIL_001» wrote to «EMAIL_003» («email_2»)."
    assert new_value == f"mail {BOB}"
    assert echo_foreign == "remember «EMAIL_001»?"
    assert both == f"«EMAIL_007» and {BOB} and carol@corp.example"


async def test_a_binary_json_frame_raises_the_floor_too() -> None:
    # Gemini Live sends JSON in binary frames; the floor reads them decoded.
    async with ReplyingUpstream() as fake:
        with _proxy(_config(fake.port)) as proxy_host:
            async with websockets.connect(f"ws://{proxy_host}/v1/realtime") as client:
                await client.send(_item("history «EMAIL_005»").encode())
                await client.recv()
                await client.send(_item(f"mail {BOB}"))
                await client.recv()
    assert _sent_text(fake.received[1]) == "mail «EMAIL_006»"


def test_frame_floors() -> None:
    assert frame_floors(_item("a «EMAIL_002» b")) == {"EMAIL": 2}
    assert frame_floors(_item("«PHONE_3»").encode()) == {"PHONE": 3}
    # A JSON frame escaping its guillemets (ensure_ascii) reads decoded.
    assert frame_floors(json.dumps({"text": "«EMAIL_004»"})) == {"EMAIL": 4}
    # Non-JSON frames are forwarded verbatim; their text still counts.
    assert frame_floors("not json «EMAIL_005»") == {"EMAIL": 5}
    assert frame_floors("not json «EMAIL_006»".encode() + b"\xff") == {"EMAIL": 6}
    # No guillemet in any encoding: nothing parsed at all.
    assert frame_floors('{"type": "input_audio_buffer.append", "audio": "UklGRg=="}') == {}
    assert frame_floors(b"\x00\x01binary-opaque") == {}
