"""multipart/form-data codec: byte-faithful parse and re-serialize.

Used for OpenAI ``/v1/files`` uploads, whose file part is JSONL carrying
user content. The parser preserves every byte it does not deliberately
rewrite: the preamble, each part's raw header block, part contents, and
the epilogue re-serialize byte-identically, so a body the adapter leaves
alone round-trips exactly. Anything outside the canonical delimiter
grammar (``\\r\\n--boundary\\r\\n`` separators, ``--boundary--`` close)
makes ``parse`` return None and the caller forwards the original bytes
verbatim — the never-break-the-tool default.

Header parameters (``Content-Disposition: form-data; name="f";
filename="a.jsonl"``) are read with one strict grammar: RFC 9110
quoted-strings where only ``\\"`` and ``\\\\`` are escapes (the two every
parser — RFC, Go's mime, python-multipart — reads alike; a browser never
escapes a backslash), unquoted values up to the next ``;``, and RFC 8187
``filename*`` ext-values. Anything with more than one reading raises
AmbiguousHeaders; callers decide whether that refuses (identity auth) or
leaves the bytes alone.

Golden fixtures in tests are assembled longhand, never with this module's
own serializer (the eventstream discipline).
"""

from collections.abc import Callable
from dataclasses import dataclass

_ALNUM = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# RFC 9110 token characters: header field names and parameter names.
_TCHAR = frozenset(_ALNUM + b"!#$%&'*+-.^_`|~")
# RFC 8187 attr-char: what a filename* value carries without percent-encoding.
_ATTR_CHAR = frozenset(_ALNUM + b"!#$&+-.^_`|~")
_HEX = frozenset(b"0123456789ABCDEFabcdef")
_OWS = frozenset(b" \t")
# Inside a quoted-string: controls other than HTAB have no single reading.
_QUOTED_BAD = frozenset({*range(0x09), *range(0x0A, 0x20), 0x7F})
# What ends an unquoted value: controls, space, DQUOTE, ";", backslash, DEL.
_VALUE_STOP = frozenset({*range(0x21), 0x22, 0x3B, 0x5C, 0x7F})
_AMBIGUOUS = "a multipart part header cannot be parsed unambiguously"


class AmbiguousHeaders(ValueError):
    """A part header, or one of its parameters, without a single reading.
    The message names the construct only, never a value."""


@dataclass(frozen=True)
class Param:
    """One ``key=value`` header parameter: ``value`` decoded (quoted-string
    escapes resolved), ``start``/``end`` the span of its RAW bytes in the
    part's header block (a quoted value's quotes excluded)."""

    value: str
    quoted: bool
    start: int
    end: int


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

    def _field(self, name: bytes) -> tuple[bytes, int, int] | None:
        """(header block, start, end) of header ``name``'s value, whitespace
        trimmed; None when absent. AmbiguousHeaders when any line of the
        block is not a ``token: value`` field (an obs-fold continuation
        included) or ``name`` repeats."""
        block = self.headers
        if block is None:
            return None
        found = None
        pos = 0
        for line in block.split(b"\r\n"):
            key, colon, _ = line.partition(b":")
            if not colon or not key or not _TCHAR.issuperset(key):
                raise AmbiguousHeaders(_AMBIGUOUS)
            if key.lower() == name:
                if found is not None:
                    raise AmbiguousHeaders(_AMBIGUOUS)
                start = _skip_ows(block, pos + len(key) + 1, pos + len(line))
                end = pos + len(line)
                while end > start and block[end - 1] in _OWS:
                    end -= 1
                found = (block, start, end)
            pos += len(line) + 2
        return found

    def header(self, name: str) -> bytes | None:
        """The value of header ``name`` (case-insensitive), None when the
        part has none. Raises AmbiguousHeaders (see ``_field``)."""
        found = self._field(name.lower().encode())
        return None if found is None else found[0][found[1] : found[2]]

    def params(self, name: str) -> dict[str, Param] | None:
        """The parameters of header ``name`` (``type; k=v; k="v"``), keys
        lowercased; None when the part has no such header. Raises
        AmbiguousHeaders on anything without a single reading."""
        found = self._field(name.lower().encode())
        return None if found is None else _parse_params(*found)

    def _disposition(self, key: str) -> str | None:
        try:
            params = self.params("content-disposition")
        except AmbiguousHeaders:
            # No single reading: the lenient one still ROUTES the part
            # (identity auth refuses such a part before routing reads it).
            return self._disposition_param(key.encode())
        param = None if params is None else params.get(key)
        return None if param is None else param.value

    @property
    def name(self) -> str | None:
        return self._disposition("name")

    @property
    def filename(self) -> str | None:
        """The upload's file name: ``filename``, else the raw RFC 8187
        ``filename*`` value; None for a plain form field."""
        plain = self._disposition("filename")
        return plain if plain is not None else self._disposition("filename*")

    def redact_filenames(self, redact: Callable[[str], str], *, strict: bool) -> bool:
        """Pass the Content-Disposition ``filename`` and ``filename*`` values
        through ``redact``, in place. Only those value bytes change: the
        parameter order, quoting style, ``name`` (structural, like a JSON
        key) and every other header stay byte-identical. ``filename`` keeps
        its quoting with the result as raw UTF-8 (as browsers send it);
        ``filename*`` is percent-encoded again under its own charset tag.
        True when the header block changed.

        A disposition without a single reading raises AmbiguousHeaders when
        ``strict`` (identity auth), and so does a ``filename*`` in a charset
        other than UTF-8; otherwise the disposition — or just that
        ``filename*`` — is left byte-identical."""
        block = self.headers
        if block is None:
            return False
        try:
            params = self.params("content-disposition") or {}
            ext = params.get("filename*")
            ext_text = None
            if ext is not None:
                charset, language, data = _ext_value(ext)
                if charset.lower() == "utf-8":
                    ext_text = (f"{charset}'{language}'", _utf8(data))
                elif strict:
                    raise AmbiguousHeaders("a multipart filename* is not UTF-8")
        except AmbiguousHeaders:
            if strict:
                raise
            return False
        edits: list[tuple[int, int, bytes]] = []
        plain = params.get("filename")
        if plain is not None:
            new = redact(plain.value)
            if new != plain.value:
                edits.append(
                    (plain.start, plain.end, (_quote(new) if plain.quoted else new).encode())
                )
        if ext is not None and ext_text is not None:
            prefix, text = ext_text
            new = redact(text)
            if new != text:
                edits.append((ext.start, ext.end, (prefix + _ext_encode(new.encode())).encode()))
        for start, end, raw in sorted(edits, reverse=True):
            block = block[:start] + raw + block[end:]
        self.headers = block
        return bool(edits)


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


def _skip_ows(block: bytes, pos: int, end: int) -> int:
    while pos < end and block[pos] in _OWS:
        pos += 1
    return pos


def _utf8(raw: bytes) -> str:
    try:
        return raw.decode()
    except UnicodeDecodeError:
        raise AmbiguousHeaders(_AMBIGUOUS) from None


def _parse_params(block: bytes, pos: int, end: int) -> dict[str, Param]:
    """``type *( OWS ";" OWS key OWS "=" OWS value )`` with an optional
    trailing ``;``; the type (``form-data``, ``text/plain``) is skipped."""
    kind, _, _ = block[pos:end].partition(b";")
    pos += len(kind)  # at the first ";", or at end: no parameters
    params: dict[str, Param] = {}
    while pos < end:  # here block[pos] is ";"
        pos = _skip_ows(block, pos + 1, end)
        if pos == end:
            break  # a trailing ";"
        key_end = pos
        while key_end < end and block[key_end] in _TCHAR:
            key_end += 1
        key = block[pos:key_end].decode().lower()
        pos = _skip_ows(block, key_end, end)
        if not key or pos == end or block[pos] != 0x3D:  # "="
            raise AmbiguousHeaders(_AMBIGUOUS)
        param, pos = _parse_value(block, _skip_ows(block, pos + 1, end), end)
        if key in params:
            raise AmbiguousHeaders(_AMBIGUOUS)
        params[key] = param
        pos = _skip_ows(block, pos, end)
        if pos < end and block[pos] != 0x3B:  # ";"
            raise AmbiguousHeaders(_AMBIGUOUS)
    return params


def _parse_value(block: bytes, pos: int, end: int) -> tuple[Param, int]:
    """A quoted-string or an unquoted value at ``pos``: (Param, next pos)."""
    if pos < end and block[pos] == 0x22:  # DQUOTE
        out = bytearray()
        i = pos + 1
        while i < end and block[i] != 0x22:
            byte = block[i]
            if byte == 0x5C:  # backslash: only \" and \\ have one reading
                i += 1
                if i == end or block[i] not in b'"\\':
                    raise AmbiguousHeaders(_AMBIGUOUS)
                byte = block[i]
            elif byte in _QUOTED_BAD:
                raise AmbiguousHeaders(_AMBIGUOUS)
            out.append(byte)
            i += 1
        if i == end:
            raise AmbiguousHeaders(_AMBIGUOUS)  # unterminated
        return Param(_utf8(bytes(out)), True, pos + 1, i), i + 1
    stop = pos
    while stop < end and block[stop] not in _VALUE_STOP:
        stop += 1
    if stop == pos:
        raise AmbiguousHeaders(_AMBIGUOUS)
    return Param(_utf8(block[pos:stop]), False, pos, stop), stop


def _quote(text: str) -> str:
    """Quoted-string content for ``text`` (the surrounding quotes excluded)."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _ext_value(param: Param) -> tuple[str, str, bytes]:
    """RFC 8187 ``charset'language'value-chars``: (charset, language, the
    percent-decoded value bytes). A quoted or malformed value raises."""
    charset, _, rest = param.value.partition("'")
    language, second, encoded = rest.partition("'")
    if param.quoted or not charset or not second:
        raise AmbiguousHeaders(_AMBIGUOUS)
    raw = encoded.encode()
    data = bytearray()
    i = 0
    while i < len(raw):
        pair = raw[i + 1 : i + 3]
        if raw[i] == 0x25 and len(pair) == 2 and _HEX.issuperset(pair):  # "%"
            data.append(int(pair, 16))
            i += 3
        elif raw[i] in _ATTR_CHAR:
            data.append(raw[i])
            i += 1
        else:
            raise AmbiguousHeaders(_AMBIGUOUS)
    return charset, language, bytes(data)


def _ext_encode(data: bytes) -> str:
    """RFC 8187 value-chars for ``data``: attr-chars as-is, the rest %XX."""
    return "".join(chr(byte) if byte in _ATTR_CHAR else f"%{byte:02X}" for byte in data)
