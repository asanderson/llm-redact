"""multipart/form-data codec: byte-faithful round trips on longhand fixtures.

Golden bodies are assembled by hand (never with the codec's own
serializer), mirroring the eventstream test convention.
"""

import httpx

from llm_redact.multipart import Multipart, MultipartPart, parse, parse_boundary

BOUNDARY = b"9a8b7c6d5e4f"

# The canonical two-part upload an SDK produces: a purpose field + a file.
GOLDEN = (
    b"--9a8b7c6d5e4f\r\n"
    b'Content-Disposition: form-data; name="purpose"\r\n'
    b"\r\n"
    b"batch\r\n"
    b"--9a8b7c6d5e4f\r\n"
    b'Content-Disposition: form-data; name="file"; filename="input.jsonl"\r\n'
    b"Content-Type: application/jsonl\r\n"
    b"\r\n"
    b'{"custom_id": "a", "body": {"messages": []}}\n'
    b'{"custom_id": "b", "body": {"messages": []}}\r\n'
    b"--9a8b7c6d5e4f--\r\n"
)


def test_golden_parse_and_round_trip() -> None:
    parsed = parse(GOLDEN, BOUNDARY)
    assert parsed is not None
    assert len(parsed.parts) == 2
    assert parsed.parts[0].name == "purpose"
    assert parsed.parts[0].filename is None
    assert parsed.parts[0].content == b"batch"
    assert parsed.parts[1].name == "file"
    assert parsed.parts[1].filename == "input.jsonl"
    assert parsed.parts[1].content.startswith(b'{"custom_id": "a"')
    assert parsed.serialize() == GOLDEN


def test_preamble_and_epilogue_preserved() -> None:
    body = b"ignored preamble\r\n" + GOLDEN[:-2] + b"\r\ntrailing epilogue\r\n"
    parsed = parse(body, BOUNDARY)
    assert parsed is not None
    assert parsed.preamble == b"ignored preamble\r\n"
    assert parsed.epilogue == b"\r\ntrailing epilogue\r\n"
    assert parsed.serialize() == body


def test_content_rewrite_keeps_framing() -> None:
    parsed = parse(GOLDEN, BOUNDARY)
    assert parsed is not None
    parsed.parts[1].content = b'{"custom_id": "a"}\n'
    out = parse(parsed.serialize(), BOUNDARY)
    assert out is not None
    assert out.parts[0].content == b"batch"
    assert out.parts[1].content == b'{"custom_id": "a"}\n'


def test_part_without_header_separator_is_opaque() -> None:
    body = b"--9a8b7c6d5e4f\r\nno blank line here\r\n--9a8b7c6d5e4f--"
    parsed = parse(body, BOUNDARY)
    assert parsed is not None
    assert parsed.parts[0].headers is None
    assert parsed.parts[0].content == b"no blank line here"
    assert parsed.serialize() == body


def test_non_canonical_bodies_return_none() -> None:
    assert parse(b"no delimiter at all", BOUNDARY) is None
    # Missing closing delimiter.
    assert parse(b"--9a8b7c6d5e4f\r\nx: y\r\n\r\ndata", BOUNDARY) is None
    # LF-only framing (not the canonical CRLF grammar).
    assert parse(b"--9a8b7c6d5e4f\nx: y\n\ndata\n--9a8b7c6d5e4f--", BOUNDARY) is None


def test_parse_boundary_header() -> None:
    assert parse_boundary("multipart/form-data; boundary=9a8b7c6d5e4f") == b"9a8b7c6d5e4f"
    assert parse_boundary('multipart/form-data; boundary="quoted-b"') == b"quoted-b"
    assert parse_boundary("multipart/form-data") is None
    assert parse_boundary("application/json") is None
    assert parse_boundary("") is None


def test_httpx_generated_body_round_trips() -> None:
    """A real client library's multipart encoding parses and round-trips."""
    request = httpx.Request(
        "POST",
        "http://example.invalid/v1/files",
        data={"purpose": "batch"},
        files={"file": ("input.jsonl", b'{"custom_id": "a"}\n', "application/jsonl")},
    )
    body = request.read()
    boundary = parse_boundary(request.headers["content-type"])
    assert boundary is not None
    parsed = parse(body, boundary)
    assert parsed is not None
    assert parsed.serialize() == body
    names = {part.name for part in parsed.parts}
    assert names == {"purpose", "file"}


def test_serialize_is_inverse_on_constructed_value() -> None:
    document = Multipart(
        boundary=BOUNDARY,
        preamble=b"",
        parts=[MultipartPart(headers=b'Content-Disposition: form-data; name="x"', content=b"1")],
        epilogue=b"\r\n",
    )
    assert parse(document.serialize(), BOUNDARY) == document


class _CountingBytes(bytes):
    """Bytes whose slices stay counted: every byte a slice copies (of this
    body, or of a slice of it) is added to ``copied``."""

    copied = 0

    def __getitem__(self, key):  # type: ignore[no-untyped-def,override]
        out = bytes.__getitem__(self, key)
        if isinstance(key, slice):
            _CountingBytes.copied += len(out)
            return _CountingBytes(out)
        return out


def test_parse_copies_each_byte_a_bounded_number_of_times() -> None:
    # Many tiny parts: re-slicing the remainder per part copied parts x size
    # (minutes of a frozen event loop at max_body_bytes); locating parts by
    # offset copies each byte of the body about once.
    parts = 3000
    body = _CountingBytes(b"--b" + b"\r\nx: y\r\n\r\nv\r\n--b" * parts + b"--\r\nend")
    _CountingBytes.copied = 0
    parsed = parse(body, b"b")
    assert parsed is not None and len(parsed.parts) == parts
    assert parsed.epilogue == b"\r\nend"
    assert _CountingBytes.copied <= 2 * len(body)
    assert parsed.serialize() == bytes(body)


def test_preamble_and_empty_parts_located_by_offset() -> None:
    body = b"pre\r\n--b\r\n\r\n\r\n--b\r\nh: 1\r\n\r\n\r\n--b--"
    parsed = parse(body, b"b")
    assert parsed is not None
    assert parsed.preamble == b"pre\r\n" and parsed.epilogue == b""
    # A part with no blank line is carried opaquely; one with it splits.
    assert [(p.headers, p.content) for p in parsed.parts] == [(None, b"\r\n"), (b"h: 1", b"")]
    assert parsed.serialize() == body
    # A delimiter that is only a prefix of a longer boundary is not one.
    assert parse(b"--bb\r\n\r\n\r\n--b--", b"b") is None
