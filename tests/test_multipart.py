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
