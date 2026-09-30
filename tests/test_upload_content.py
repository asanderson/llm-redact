"""upload_content.classify_file: what an uploaded file is, by its content.

JSONL (every non-blank line a JSON object) keeps the per-line path; other
text — strict UTF-8, or UTF-16/32 behind a byte-order mark — is redacted
as one text and re-encoded exactly as it came; everything else (bytes that
do not decode, a NUL character, one of the few binary signatures real text
cannot open with even when the bytes would decode) is binary and never
rewritten.
"""

from __future__ import annotations

import json

import pytest

from llm_redact.upload_content import BINARY, FileContent, classify_file, decode_text

ASCII_TAIL = b"1.7\n1 0 obj << /Type /Catalog >> endobj\nxref\n0 1\ntrailer\n%%EOF\n"


# Every signature a strict decode would ACCEPT, followed by all-ASCII text:
# only the signature makes these binary.
@pytest.mark.parametrize(
    "prefix",
    [
        b"%PDF-",
        b"PK\x03\x04",
        b"PK\x05\x06",
        b"Rar!\x1a\x07",
        b"\x1a\x45\xdf\xa3",
        b"\x7fELF",
    ],
)
def test_a_signature_makes_decodable_bytes_binary(prefix: bytes) -> None:
    content = prefix + ASCII_TAIL
    content.decode("utf-8")  # it WOULD decode
    assert classify_file(content) == BINARY
    assert decode_text(content) is None
    # Not a signature anywhere but at the start.
    assert classify_file(b"x" + content).kind == "text"


def test_an_all_ascii_pdf_is_binary() -> None:
    pdf = b"%PDF-" + ASCII_TAIL + b"(jane.doe@corp.example) Tj\n"
    assert classify_file(pdf).kind == "binary"


# Printable-word signatures are NOT signatures: text opening with one of them
# (a CSV row "ID3,name,email") is text and is redacted. Real files of these
# formats carry a NUL or a non-UTF-8 byte in their fixed header, which the
# strict decode already reads as binary.
@pytest.mark.parametrize(
    ("word", "real_header"),
    [
        (b"GIF87a", b"GIF87a\x10\x00\x10\x00\x80\x00\x00"),
        (b"GIF89a", b"GIF89a\x01\x00\x01\x00\x00\xff\x00"),
        (b"RIFF", b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00"),
        (b"ID3", b"ID3\x03\x00\x00\x00\x00\x0f\x76"),
        (b"OggS", b"OggS\x00\x02\x00\x00"),
        (b"fLaC", b"fLaC\x00\x00\x00\x22"),
        (b"wOFF", b"wOFF\x00\x01\x00\x00\x00\x00\x02\x50"),
        (b"wOF2", b"wOF2\x00\x01\x00\x00\x00\x00\x02\x50"),
    ],
)
def test_a_printable_word_opening_text_is_text(word: bytes, real_header: bytes) -> None:
    csv = word + b",name,email\n" + word + b"001,Jane,jane.doe@corp.example\n"
    assert classify_file(csv).kind == "text"
    assert classify_file(word + b" notes: " + ASCII_TAIL).kind == "text"
    # The real format's fixed header still classifies as binary.
    assert classify_file(real_header + ASCII_TAIL).kind == "binary"


def test_ftyp_at_offset_4_is_no_signature() -> None:
    assert classify_file(b"The ftyp box notes: jane.doe@corp.example\n").kind == "text"
    assert classify_file(b"    ftypisom" + ASCII_TAIL).kind == "text"
    # A real ISO media file opens with a big-endian box size: NUL bytes.
    assert classify_file(b"\x00\x00\x00\x20ftypisom" + ASCII_TAIL).kind == "binary"


# Signatures that are invalid UTF-8 themselves: the strict decode refuses.
@pytest.mark.parametrize(
    "prefix",
    [
        b"\x89PNG\r\n\x1a\n",  # PNG
        b"\xff\xd8\xff\xe0",  # JPEG
        b"\x1f\x8b\x08",  # gzip
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",  # OLE2 (doc/xls/ppt/msg)
        b"\xff\xfb\x90",  # MPEG audio frame (MP3 without an ID3 tag)
        b"7z\xbc\xaf\x27\x1c",
        b"\xfd7zXZ\x00",
        b"\x28\xb5\x2f\xfd",  # zstd
    ],
)
def test_signatures_that_do_not_decode_are_binary(prefix: bytes) -> None:
    assert classify_file(prefix + ASCII_TAIL).kind == "binary"


def test_nul_makes_text_binary() -> None:
    assert classify_file(b"plain text\x00more").kind == "binary"
    assert classify_file(b"\x00").kind == "binary"
    assert classify_file(b'{"a": 1}\n\x00\n').kind == "binary"
    # UTF-16 carries NUL BYTES for ASCII, never a NUL character: text.
    assert classify_file(b"\xff\xfe" + "hi".encode("utf-16-le")).kind == "text"
    # ... unless it decodes to one.
    assert classify_file(b"\xff\xfe" + "h\x00i".encode("utf-16-le")).kind == "binary"


def test_undecodable_is_binary() -> None:
    assert classify_file(b"caf\xe9 au lait").kind == "binary"  # Latin-1
    assert classify_file(b"\xc3").kind == "binary"  # truncated sequence
    assert classify_file(b"\xed\xa0\x80").kind == "binary"  # UTF-8 surrogate


def test_utf8_text_and_bom() -> None:
    plain = classify_file("naïve jane.doe@corp.example\n".encode())
    assert plain == FileContent("text", "naïve jane.doe@corp.example\n", "utf-8", b"")
    bom = classify_file(b"\xef\xbb\xbfhello")
    assert bom == FileContent("text", "hello", "utf-8", b"\xef\xbb\xbf")
    assert bom.encode("bye") == b"\xef\xbb\xbfbye"
    assert plain.encode("é") == "é".encode()


@pytest.mark.parametrize(
    ("bom", "codec"),
    [
        (b"\xff\xfe", "utf-16-le"),
        (b"\xfe\xff", "utf-16-be"),
        (b"\xff\xfe\x00\x00", "utf-32-le"),
        (b"\x00\x00\xfe\xff", "utf-32-be"),
    ],
)
def test_utf16_and_utf32_by_their_bom(bom: bytes, codec: str) -> None:
    text = "mail jane.doe@corp.example «»\r\n"
    content = classify_file(bom + text.encode(codec))
    assert content == FileContent("text", text, codec, bom)
    # Re-encoded exactly as it came: same codec, same mark.
    assert content.encode(text) == bom + text.encode(codec)
    assert content.encode("x") == bom + "x".encode(codec)
    # JSON lines in UTF-16/32 are text: the JSONL path reads UTF-8 bytes.
    assert classify_file(bom + '{"a": 1}\n'.encode(codec)).kind == "text"


def test_a_bom_with_a_broken_body_is_binary() -> None:
    assert classify_file(b"\xff\xfe" + b"a").kind == "binary"  # odd length
    assert classify_file(b"\xff\xfe" + "\ud800".encode("utf-16-le", "surrogatepass")).kind == (
        "binary"
    )
    assert classify_file(b"\x00\x00\xfe\xff" + b"\x00\x11\x00\x00").kind == "binary"
    # A mark alone is an empty text.
    assert classify_file(b"\xfe\xff") == FileContent("text", "", "utf-16-be", b"\xfe\xff")


def test_jsonl() -> None:
    lines = [json.dumps({"custom_id": str(i), "body": {"q": "x"}}) for i in range(3)]
    content = "\n".join(lines).encode()
    classified = classify_file(content)
    assert classified.kind == "jsonl" and classified.text == content.decode()
    # Blank lines, CRLF, surrounding whitespace, a trailing newline, a BOM.
    padded = b"\n  \r\n" + b"\r\n".join(line.encode() for line in lines) + b"\r\n\n"
    assert classify_file(padded).kind == "jsonl"
    assert classify_file(b"\xef\xbb\xbf" + content).kind == "jsonl"
    assert classify_file(b'{"a": 1}').kind == "jsonl"  # no newline at all


@pytest.mark.parametrize(
    "content",
    [
        b'{"a": 1}\nnot json\n{"b": 2}\n',  # mixed: prose between objects
        b'{"a": 1}\n[1, 2]\n',  # an array line
        b'{"a": 1}\n"str"\n',
        b'{"a": 1}\n{"b": \n',  # a broken object
        b'not json\n{"a": 1}\n',
        b"name,email\njane,jane.doe@corp.example\n",  # CSV
        b"[1, 2]\n",
        b"",
        b"   \n\t\r\n",
        b"\x0b\x0c",
    ],
)
def test_everything_else_that_decodes_is_text(content: bytes) -> None:
    assert classify_file(content).kind == "text"


def test_a_line_nesting_past_the_parser_stays_jsonl() -> None:
    # json.loads cannot read it (RecursionError): the JSONL path's bounded
    # reading refuses it rather than a text redaction missing its escapes.
    deep = b'{"a": ' * 5000 + b"1" + b"}" * 5000
    assert classify_file(b'{"ok": 1}\n' + deep + b"\n").kind == "jsonl"
    assert classify_file(deep).kind == "jsonl"


def test_charge_counts_lines_only_for_a_jsonl_candidate() -> None:
    charged: list[int] = []
    assert classify_file(b'\n\n{"a": 1}\n{"b": 2}', charge=charged.append).kind == "jsonl"
    assert charged == [4]
    charged.clear()
    # Its first non-blank line is an object: the file is walked (charged)
    # even though a later line turns it into text.
    assert classify_file(b'{"a": 1}\nprose\n', charge=charged.append).kind == "text"
    assert charged == [3]
    charged.clear()
    # Prose first: one parse decides it, nothing charged.
    many = b"prose\n" + b'{"a": 1}\n' * 10
    assert classify_file(many, charge=charged.append).kind == "text"
    assert classify_file(b"\n" * 1000 + b"prose", charge=charged.append).kind == "text"
    assert classify_file(b"\n" * 1000, charge=charged.append).kind == "text"
    assert classify_file(b"%PDF-" + b"\n" * 1000, charge=charged.append).kind == "binary"
    assert charged == []


def test_the_first_line_is_found_between_its_newlines() -> None:
    # The first non-blank line runs from the previous newline to the next.
    assert classify_file(b'  \n  {"a": 1}  \nprose').kind == "text"
    assert classify_file(b'  \n  {"a": 1}  \n{"b": 2}').kind == "jsonl"
    assert classify_file(b'\n{"a": 1}').kind == "jsonl"
    assert classify_file(b"\n  x{}").kind == "text"
    # Blank is what bytes.strip reads as blank: vertical tab and form feed
    # lines are skipped like the JSONL redaction skips them...
    assert classify_file(b'\x0b\n\x0c\r\n{"a": 1}\n\x0b').kind == "jsonl"
    # ... but inside a line they are not JSON whitespace.
    assert classify_file(b'\x0b{"a": 1}').kind == "text"
    assert classify_file(b'{"a": 1}\n\x0c{"b": 1}').kind == "text"
