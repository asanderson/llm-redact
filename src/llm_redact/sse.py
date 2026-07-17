"""Incremental Server-Sent Events parsing and serialization.

The proxy operates on the actual byte stream — never on assumed typed
objects — which is the fix for the transport-mismatch failure class.
Bytes are buffered until complete lines exist, so multi-byte UTF-8
sequences split across network chunks are decoded safely.
"""

from dataclasses import dataclass, field


@dataclass
class SSEEvent:
    event: str | None = None
    data: str = ""
    id: str | None = None
    comments: list[str] = field(default_factory=list)

    @property
    def is_comment_only(self) -> bool:
        return self.event is None and not self.data and self.id is None and bool(self.comments)


class SSEParser:
    """Feed raw bytes, receive complete events.

    An event is complete at a blank line. Handles LF and CRLF, multi-line
    ``data:`` fields, and comment lines (``: ping`` keep-alives), which are
    preserved so they can be re-emitted verbatim.
    """

    def __init__(self) -> None:
        self._byte_buffer = b""
        self._event: SSEEvent | None = None

    def _current(self) -> SSEEvent:
        if self._event is None:
            self._event = SSEEvent()
        return self._event

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._byte_buffer += chunk
        events: list[SSEEvent] = []
        while True:
            newline = self._byte_buffer.find(b"\n")
            if newline == -1:
                break
            raw_line = self._byte_buffer[:newline]
            self._byte_buffer = self._byte_buffer[newline + 1 :]
            # Lines are only decoded once complete (up to the LF), so a valid
            # multi-byte char is never split mid-decode; `replace` therefore
            # only ever fires on genuinely-malformed upstream bytes, where
            # degrading beats crashing the stream (the never-break-the-tool
            # rule) and leaves valid streams byte-identical.
            line = raw_line.rstrip(b"\r").decode("utf-8", "replace")
            if line == "":
                if self._event is not None:
                    events.append(self._event)
                    self._event = None
                continue
            if line.startswith(":"):
                self._current().comments.append(line[1:].lstrip(" "))
                continue
            name, _, value = line.partition(":")
            value = value.removeprefix(" ")
            event = self._current()
            if name == "event":
                event.event = value
            elif name == "data":
                event.data = value if not event.data else event.data + "\n" + value
            elif name == "id":
                event.id = value
        return events

    def close(self) -> list[SSEEvent]:
        """Flush any unterminated trailing event at end of stream."""
        events: list[SSEEvent] = []
        if self._byte_buffer:
            events.extend(self.feed(b"\n"))
            if self._byte_buffer:
                # Data with no trailing newline at all: force line completion.
                self._byte_buffer += b"\n"
                events.extend(self.feed(b"\n"))
        if self._event is not None:
            events.append(self._event)
            self._event = None
        return events


def serialize(event: SSEEvent) -> bytes:
    lines: list[str] = []
    for comment in event.comments:
        lines.append(f": {comment}" if comment else ":")
    if event.event is not None:
        lines.append(f"event: {event.event}")
    if event.id is not None:
        lines.append(f"id: {event.id}")
    if event.data:
        lines.extend(f"data: {part}" for part in event.data.split("\n"))
    return ("\n".join(lines) + "\n\n").encode("utf-8")
