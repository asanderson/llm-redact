"""The eventstream codec must round-trip byte-identically and fail loudly.

The golden fixture is assembled longhand in the test — independent of the
codec's own serializer — so the implementation is not tested against
itself.
"""

import struct
import uuid
import zlib

import pytest

from llm_redact.eventstream import (
    BOOL_FALSE,
    BOOL_TRUE,
    BYTE,
    BYTE_ARRAY,
    INT,
    LONG,
    SHORT,
    STRING,
    TIMESTAMP,
    UUID,
    EventStreamError,
    EventStreamMessage,
    EventStreamParser,
    serialize,
    string_header,
)


def _golden_frame() -> bytes:
    """A converse-stream-shaped event frame built by hand."""
    headers = b""
    for name, value in (
        (b":message-type", b"event"),
        (b":event-type", b"contentBlockDelta"),
        (b":content-type", b"application/json"),
    ):
        headers += bytes((len(name),)) + name + bytes((7,)) + struct.pack(">H", len(value)) + value
    payload = b'{"contentBlockIndex":0,"delta":{"text":"hello"}}'
    total = 16 + len(headers) + len(payload)
    prelude = struct.pack(">II", total, len(headers))
    body = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + payload
    return body + struct.pack(">I", zlib.crc32(body))


def test_parse_golden_frame() -> None:
    parser = EventStreamParser()
    messages = parser.feed(_golden_frame())
    parser.close()
    assert len(messages) == 1
    message = messages[0]
    assert message.message_type == "event"
    assert message.event_type == "contentBlockDelta"
    assert message.header(":content-type") == "application/json"
    assert message.payload == b'{"contentBlockIndex":0,"delta":{"text":"hello"}}'


def test_serialize_matches_golden_bytes() -> None:
    message = EventStreamMessage(
        headers=[
            string_header(":message-type", "event"),
            string_header(":event-type", "contentBlockDelta"),
            string_header(":content-type", "application/json"),
        ],
        payload=b'{"contentBlockIndex":0,"delta":{"text":"hello"}}',
    )
    assert serialize(message) == _golden_frame()


def test_every_header_type_round_trips() -> None:
    message = EventStreamMessage(
        headers=[
            ("t", BOOL_TRUE, True),
            ("f", BOOL_FALSE, False),
            ("b", BYTE, -3),
            ("s", SHORT, -1234),
            ("i", INT, -123456),
            ("l", LONG, -(2**40)),
            ("ba", BYTE_ARRAY, b"\x00\x01\xff"),
            ("st", STRING, "héllo «token»"),
            ("ts", TIMESTAMP, 1717228819123),
            ("u", UUID, uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")),
        ],
        payload=b"payload bytes",
    )
    raw = serialize(message)
    parser = EventStreamParser()
    (back,) = parser.feed(raw)
    parser.close()
    assert back == message
    # And a second serialization is byte-identical (type codes preserved).
    assert serialize(back) == raw


def test_split_at_every_offset() -> None:
    stream = b"".join(
        serialize(
            EventStreamMessage(
                headers=[string_header(":event-type", f"e{n}")],
                payload=f'{{"n":{n},"text":"chunk «EMAIL_00{n}» end"}}'.encode(),
            )
        )
        for n in range(3)
    )
    for split in range(len(stream) + 1):
        parser = EventStreamParser()
        messages = parser.feed(stream[:split]) + parser.feed(stream[split:])
        parser.close()
        assert [m.event_type for m in messages] == ["e0", "e1", "e2"]


def test_prelude_crc_corruption_rejected() -> None:
    frame = bytearray(_golden_frame())
    frame[9] ^= 0xFF  # inside the prelude CRC field
    with pytest.raises(EventStreamError, match="prelude CRC"):
        EventStreamParser().feed(bytes(frame))


def test_message_crc_corruption_rejected() -> None:
    frame = bytearray(_golden_frame())
    frame[-10] ^= 0xFF  # inside the payload: message CRC no longer matches
    with pytest.raises(EventStreamError, match="message CRC"):
        EventStreamParser().feed(bytes(frame))


def test_frame_size_bound() -> None:
    frame = serialize(EventStreamMessage(payload=b"x" * 1000))
    with pytest.raises(EventStreamError, match="implausible"):
        EventStreamParser(max_frame_bytes=100).feed(frame)


def test_trailing_garbage_on_close() -> None:
    parser = EventStreamParser()
    parser.feed(_golden_frame()[:20])
    with pytest.raises(EventStreamError, match="trailing"):
        parser.close()


def test_residual_exposes_unconsumed_bytes() -> None:
    frame = _golden_frame()
    parser = EventStreamParser()
    parser.feed(frame[:10])
    assert parser.residual == frame[:10]


def test_error_mid_feed_loses_no_bytes() -> None:
    # A valid frame followed by a corrupted one in the SAME feed call: the
    # raise must leave the valid-but-unreturned frame in the buffer, so a
    # degrading caller forwarding `residual` drops nothing.
    good = _golden_frame()
    bad = bytearray(_golden_frame())
    bad[-10] ^= 0xFF
    parser = EventStreamParser()
    with pytest.raises(EventStreamError):
        parser.feed(good + bytes(bad))
    assert parser.residual == good + bytes(bad)


def test_returned_frames_never_reappear_in_residual() -> None:
    good = _golden_frame()
    bad = bytearray(_golden_frame())
    bad[-10] ^= 0xFF
    parser = EventStreamParser()
    assert len(parser.feed(good)) == 1  # returned: consumed for good
    with pytest.raises(EventStreamError):
        parser.feed(bytes(bad))
    assert parser.residual == bytes(bad)


def test_unknown_header_type_rejected() -> None:
    headers = bytes((1,)) + b"x" + bytes((42,))
    total = 16 + len(headers)
    prelude = struct.pack(">II", total, len(headers))
    body = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers
    frame = body + struct.pack(">I", zlib.crc32(body))
    with pytest.raises(EventStreamError, match="unknown header value type"):
        EventStreamParser().feed(frame)
