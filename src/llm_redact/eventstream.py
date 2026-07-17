"""AWS event stream (application/vnd.amazon.eventstream): binary framing.

The sse.py of the Bedrock path — an incremental, byte-level parser and
serializer. Same discipline: bytes in, bytes out, never assume typed
objects, bounded buffering. The framing (per the Smithy event stream spec):

    prelude:  total_length (4B BE, whole message incl. final CRC)
              headers_length (4B BE)
              prelude_crc (4B BE, CRC32 of the previous 8 bytes)
    headers:  repeated {1B name_len, name, 1B type_code, value}
    payload:  total_length - headers_length - 16 bytes
    message_crc (4B BE, CRC32 of everything before it)

CRC32 is the GZIP polynomial — exactly ``zlib.crc32``. Header value types
(all big-endian): 0/1 bool true/false (no value bytes), 2 byte, 3 short,
4 int, 5 long, 6 byte-array (2B len), 7 string (2B len, UTF-8),
8 timestamp (8B epoch millis), 9 UUID (16B).

Rewriting a message means recomputing both lengths and both CRCs —
``serialize`` always does. Headers preserve their exact type codes through
a parse/serialize round trip so untouched messages re-emit byte-identical.
"""

import struct
import uuid as _uuid
import zlib
from dataclasses import dataclass, field

_PRELUDE = struct.Struct(">III")
_UINT32 = struct.Struct(">I")

# Header value type codes.
BOOL_TRUE = 0
BOOL_FALSE = 1
BYTE = 2
SHORT = 3
INT = 4
LONG = 5
BYTE_ARRAY = 6
STRING = 7
TIMESTAMP = 8
UUID = 9

_INT_FORMATS = {BYTE: ">b", SHORT: ">h", INT: ">i", LONG: ">q", TIMESTAMP: ">q"}


class EventStreamError(ValueError):
    """Framing can no longer be trusted (bad CRC, bound violation, garbage).

    Callers on the proxy path degrade to verbatim pass-through: forwarding
    unrestored placeholders is safe; guessing at corrupted frames is not.
    """


@dataclass
class EventStreamMessage:
    # (name, type_code, value) — the code is kept so re-serialization is
    # byte-identical (an int re-encoded with a different width would change
    # the frame even when the value is equal).
    headers: list[tuple[str, int, object]] = field(default_factory=list)
    payload: bytes = b""

    def header(self, name: str) -> object | None:
        for header_name, _code, value in self.headers:
            if header_name == name:
                return value
        return None

    @property
    def message_type(self) -> str | None:
        value = self.header(":message-type")
        return value if isinstance(value, str) else None

    @property
    def event_type(self) -> str | None:
        value = self.header(":event-type")
        return value if isinstance(value, str) else None

    @property
    def exception_type(self) -> str | None:
        value = self.header(":exception-type")
        return value if isinstance(value, str) else None


def _parse_headers(data: bytes) -> list[tuple[str, int, object]]:
    headers: list[tuple[str, int, object]] = []
    pos = 0
    end = len(data)
    while pos < end:
        name_len = data[pos]
        pos += 1
        if pos + name_len + 1 > end:
            raise EventStreamError("truncated header name")
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        code = data[pos]
        pos += 1
        value: object
        if code == BOOL_TRUE:
            value = True
        elif code == BOOL_FALSE:
            value = False
        elif code in _INT_FORMATS:
            fmt = struct.Struct(_INT_FORMATS[code])
            if pos + fmt.size > end:
                raise EventStreamError("truncated header value")
            value = fmt.unpack_from(data, pos)[0]
            pos += fmt.size
        elif code in (BYTE_ARRAY, STRING):
            if pos + 2 > end:
                raise EventStreamError("truncated header value length")
            (length,) = struct.unpack_from(">H", data, pos)
            pos += 2
            if pos + length > end:
                raise EventStreamError("truncated header value")
            raw = data[pos : pos + length]
            pos += length
            value = raw.decode("utf-8") if code == STRING else raw
        elif code == UUID:
            if pos + 16 > end:
                raise EventStreamError("truncated header value")
            value = _uuid.UUID(bytes=data[pos : pos + 16])
            pos += 16
        else:
            raise EventStreamError(f"unknown header value type {code}")
        headers.append((name, code, value))
    return headers


def _serialize_headers(headers: list[tuple[str, int, object]]) -> bytes:
    parts: list[bytes] = []
    for name, code, value in headers:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 255:
            raise EventStreamError("header name over 255 bytes")
        parts.append(bytes((len(name_bytes),)) + name_bytes + bytes((code,)))
        if code in (BOOL_TRUE, BOOL_FALSE):
            continue
        if code in _INT_FORMATS:
            parts.append(struct.pack(_INT_FORMATS[code], value))
        elif code in (BYTE_ARRAY, STRING):
            raw = value.encode("utf-8") if isinstance(value, str) else value
            assert isinstance(raw, bytes)
            parts.append(struct.pack(">H", len(raw)) + raw)
        elif code == UUID:
            assert isinstance(value, _uuid.UUID)
            parts.append(value.bytes)
        else:
            raise EventStreamError(f"unknown header value type {code}")
    return b"".join(parts)


def serialize(message: EventStreamMessage) -> bytes:
    """Frame a message, recomputing lengths and both CRCs."""
    headers = _serialize_headers(message.headers)
    total = 16 + len(headers) + len(message.payload)
    prelude = struct.pack(">II", total, len(headers))
    prelude_crc = zlib.crc32(prelude)
    body = prelude + _UINT32.pack(prelude_crc) + headers + message.payload
    return body + _UINT32.pack(zlib.crc32(body))


class EventStreamParser:
    """Incremental parser: feed bytes, get complete validated messages.

    Both CRCs are checked and a per-frame size bound enforced; any
    violation raises EventStreamError and the parser must be abandoned
    (the caller switches to verbatim pass-through).
    """

    def __init__(self, max_frame_bytes: int = 10 * 1024 * 1024) -> None:
        self._buffer = bytearray()
        self._max_frame_bytes = max_frame_bytes

    @property
    def residual(self) -> bytes:
        """Every byte fed but never returned as a message — after an
        EventStreamError this is exactly what a degrading caller must
        forward verbatim (no byte is lost or duplicated)."""
        return bytes(self._buffer)

    def feed(self, chunk: bytes) -> list[EventStreamMessage]:
        self._buffer.extend(chunk)
        messages: list[EventStreamMessage] = []
        pos = 0
        # Consumed bytes are deleted only after the whole round succeeds: a
        # raise mid-round keeps already-parsed (but unreturned) frames in
        # the buffer, so `residual` never drops them.
        while len(self._buffer) - pos >= 12:
            total, headers_len, prelude_crc = _PRELUDE.unpack_from(self._buffer, pos)
            if total < 16 or total > self._max_frame_bytes or headers_len > total - 16:
                raise EventStreamError(f"implausible frame lengths ({total}, {headers_len})")
            if zlib.crc32(bytes(self._buffer[pos : pos + 8])) != prelude_crc:
                raise EventStreamError("prelude CRC mismatch")
            if len(self._buffer) - pos < total:
                break
            frame = bytes(self._buffer[pos : pos + total])
            (message_crc,) = _UINT32.unpack_from(frame, total - 4)
            if zlib.crc32(frame[: total - 4]) != message_crc:
                raise EventStreamError("message CRC mismatch")
            headers = _parse_headers(frame[12 : 12 + headers_len])
            payload = frame[12 + headers_len : total - 4]
            messages.append(EventStreamMessage(headers=headers, payload=payload))
            pos += total
        del self._buffer[:pos]
        return messages

    def close(self) -> None:
        """End of stream: any buffered leftover means a truncated frame."""
        if self._buffer:
            raise EventStreamError(f"{len(self._buffer)} trailing bytes at stream end")


def string_header(name: str, value: str) -> tuple[str, int, object]:
    """Convenience for constructing the common all-string event headers."""
    return (name, STRING, value)
