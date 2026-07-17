"""multipart/form-data codec: byte-faithful parse and re-serialize.

Used for OpenAI ``/v1/files`` uploads, whose file part is JSONL carrying
user content. The parser preserves every byte it does not deliberately
rewrite: the preamble, each part's raw header block, part contents, and
the epilogue re-serialize byte-identically, so a body the adapter leaves
alone round-trips exactly. Anything outside the canonical delimiter
grammar (``\\r\\n--boundary\\r\\n`` separators, ``--boundary--`` close)
makes ``parse`` return None and the caller forwards the original bytes
verbatim — the never-break-the-tool default.

Golden fixtures in tests are assembled longhand, never with this module's
own serializer (the eventstream discipline).
"""

from dataclasses import dataclass


@dataclass
class MultipartPart:
    """One body part. ``headers`` is the raw header block (no trailing
    blank line); None means the part had no header/body separator and is
    carried opaquely in ``content``."""

    headers: bytes | None
    content: bytes

    def _header_value(self, name: bytes) -> bytes | None:
        if self.headers is None:
            return None
        for line in self.headers.split(b"\r\n"):
            key, _, value = line.partition(b":")
            if key.strip().lower() == name:
                return value.strip()
        return None

    def _disposition_param(self, param: bytes) -> str | None:
        disposition = self._header_value(b"content-disposition")
        if disposition is None:
            return None
        for piece in disposition.split(b";"):
            key, _, value = piece.strip().partition(b"=")
            if key.strip().lower() == param:
                return value.strip().strip(b'"').decode("utf-8", "replace")
        return None

    @property
    def name(self) -> str | None:
        return self._disposition_param(b"name")

    @property
    def filename(self) -> str | None:
        return self._disposition_param(b"filename")


@dataclass
class Multipart:
    boundary: bytes
    preamble: bytes  # bytes before the first delimiter, verbatim (incl. CRLF)
    parts: list[MultipartPart]
    epilogue: bytes  # bytes after the closing delimiter, verbatim

    def serialize(self) -> bytes:
        delim = b"--" + self.boundary
        out = bytearray(self.preamble)
        for part in self.parts:
            out += delim + b"\r\n"
            if part.headers is not None:
                out += part.headers + b"\r\n\r\n"
            out += part.content + b"\r\n"
        out += delim + b"--" + self.epilogue
        return bytes(out)


def parse_boundary(content_type: str) -> bytes | None:
    """The boundary parameter of a multipart/form-data content type."""
    media, _, params = content_type.partition(";")
    if media.strip().lower() != "multipart/form-data":
        return None
    for piece in params.split(";"):
        key, _, value = piece.strip().partition("=")
        if key.strip().lower() == "boundary":
            boundary = value.strip().strip('"')
            return boundary.encode("ascii", "ignore") or None
    return None


def parse(body: bytes, boundary: bytes) -> Multipart | None:
    """Parse the canonical delimiter grammar; None on anything else."""
    delim = b"--" + boundary
    if body.startswith(delim):
        preamble = b""
        rest = body[len(delim) :]
    else:
        idx = body.find(b"\r\n" + delim)
        if idx < 0:
            return None
        preamble = body[: idx + 2]
        rest = body[idx + 2 + len(delim) :]

    parts: list[MultipartPart] = []
    while True:
        if rest.startswith(b"--"):
            return Multipart(boundary, preamble, parts, rest[2:])
        if not rest.startswith(b"\r\n"):
            return None  # transport padding and LF-only bodies: verbatim
        end = rest.find(b"\r\n" + delim, 2)
        if end < 0:
            return None  # no closing delimiter
        raw = rest[2:end]
        rest = rest[end + 2 + len(delim) :]
        head, sep, content = raw.partition(b"\r\n\r\n")
        if sep:
            parts.append(MultipartPart(headers=head, content=content))
        else:
            parts.append(MultipartPart(headers=None, content=raw))
