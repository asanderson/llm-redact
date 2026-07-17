"""NDJSON streaming (application/x-ndjson): line framing.

The third streaming transport next to SSE and the AWS event stream, used
by Ollama's native API. Same discipline: incremental, byte-level, bounded
buffering. Framing is one JSON object per LF-terminated line; the parser
buffers at most one partial trailing line across chunk boundaries and
returns raw line bytes (no trailing newline) so untouched lines can be
re-emitted byte-identically.
"""


class NDJSONParser:
    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, chunk: bytes) -> list[bytes]:
        """Complete lines in order, each without its trailing newline."""
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        return lines

    def close(self) -> bytes:
        """The unterminated final line, if the stream ended without LF."""
        leftover, self._buffer = self._buffer, b""
        return leftover
