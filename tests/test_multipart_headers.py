"""multipart part headers: the strict parameter grammar and filename rewrite.

Fixtures are assembled longhand (the eventstream discipline). Every
refusal is asserted with its exact message and every rewrite with the
exact resulting header block: the rewrite must change the filename value
bytes and nothing else.
"""

from collections.abc import Callable

import pytest

from llm_redact.multipart import (
    AmbiguousHeaders,
    MultipartPart,
    Param,
    _ext_encode,
    _ext_value,
    _quote,
)

AMBIGUOUS = "a multipart part header cannot be parsed unambiguously"
EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"


def _part(headers: bytes | None) -> MultipartPart:
    return MultipartPart(headers=headers, content=b"body")


def _redact(text: str) -> str:
    return text.replace(EMAIL, TOKEN)


def _ambiguous(call: Callable[[], object], message: str = AMBIGUOUS) -> None:
    with pytest.raises(AmbiguousHeaders) as raised:
        call()
    assert str(raised.value) == message


# --- header() --------------------------------------------------------------------------


def test_header_is_case_insensitive_and_trimmed() -> None:
    part = _part(
        b'Content-Disposition: form-data; name="f"\r\n'
        b"X-Other:1\r\n"
        b"Content-Type: \t application/jsonl \t"
    )
    assert part.header("CONTENT-TYPE") == b"application/jsonl"
    assert part.header("x-other") == b"1"
    assert part.header("content-disposition") == b'form-data; name="f"'
    assert part.header("content-transfer-encoding") is None


def test_header_absent_block_and_empty_values() -> None:
    assert _part(None).header("content-type") is None
    assert _part(None).params("content-type") is None
    # An empty value on the LAST line (nothing after it to stop a scan).
    assert _part(b"X-A: 1\r\nContent-Type:").header("content-type") == b""
    assert _part(b"Content-Type:   ").header("content-type") == b""
    assert _part(b"Content-Type:   ").params("content-type") == {}


@pytest.mark.parametrize(
    "block",
    [
        b"no colon here",
        b": empty name",
        b"Content Type: spaced name",
        b"Content-Type : space before colon",
        b'Content-Disposition: form-data; name="f";\r\n filename="folded.jsonl"',
        b'Content-Disposition: form-data; name="f"\r\n\tfolded: tab',
        b"",
        b"Content-Type: a\r\ncontent-type: b",
    ],
    ids=[
        "no-colon",
        "empty-name",
        "space-in-name",
        "space-before-colon",
        "obs-fold",
        "obs-fold-tab",
        "empty-block",
        "repeated",
    ],
)
def test_header_blocks_without_one_reading(block: bytes) -> None:
    _ambiguous(lambda: _part(block).header("content-type"))


@pytest.mark.parametrize(
    "block",
    [
        b'Content-Type: application/json\n\n{"file": {"name": "files/a"}}',
        b"Content-Type: text/plain\nX-Other: 1",
        b"Content-Type: text/plain\rX-Other: 1",
        b"Content-Type: text/plain\x00",
        b"Content-Type: text/\x0bplain",
        b"Content-Type: text/plain\x7f",
        b'Content-Type: text/plain; charset="utf-8\n"',
        b"X-Other: \x1f\r\nContent-Type: text/plain",
    ],
    ids=["lf-blank-line", "bare-lf", "bare-cr", "nul", "vt", "del", "lf-quoted", "other-line"],
)
def test_a_control_in_any_header_line_has_no_single_reading(block: bytes) -> None:
    # A reader accepting a bare LF (or CR) as a line break ends the header
    # block where this one does not: what follows would be the part's
    # content there, never read as such here.
    part = _part(block)
    _ambiguous(lambda: part.header("content-type"))
    _ambiguous(lambda: part.params("content-type"))


def test_a_tab_is_the_one_control_a_header_line_carries() -> None:
    part = _part(b'Content-Type:\ttext/plain;\tcharset="utf-8\t"')
    assert part.header("content-type") == b'text/plain;\tcharset="utf-8\t"'
    params = part.params("content-type")
    assert params is not None and params["charset"].value == "utf-8\t"


def test_a_malformed_line_anywhere_poisons_every_lookup() -> None:
    part = _part(b"Content-Type: text/plain\r\ngarbage")
    _ambiguous(lambda: part.header("content-type"))


# --- params() --------------------------------------------------------------------------


def test_params_values_quoting_and_raw_spans() -> None:
    block = b'Content-Disposition: form-data; NAME="file"; filename=in.jsonl ;x = "y" ;'
    params = _part(block).params("content-disposition")
    assert params is not None
    assert set(params) == {"name", "filename", "x"}
    name, filename, extra = params["name"], params["filename"], params["x"]
    assert (name.value, name.quoted) == ("file", True)
    assert block[name.start : name.end] == b"file"
    assert block[name.start - 1] == 0x22  # the span excludes the quotes
    assert (filename.value, filename.quoted) == ("in.jsonl", False)
    assert block[filename.start : filename.end] == b"in.jsonl"
    assert (extra.value, extra.quoted) == ("y", True)
    assert block[extra.start : extra.end] == b"y"


def test_params_without_parameters() -> None:
    assert _part(b"Content-Type: application/octet-stream").params("content-type") == {}
    assert _part(b"Content-Type: text/plain;  ").params("content-type") == {}


def test_quoted_escapes_resolve_and_span_the_raw_bytes() -> None:
    block = b'Content-Disposition: form-data; filename="a\\"b\\\\c\td\xc3\xa9"'
    param = (_part(block).params("content-disposition") or {})["filename"]
    assert param.value == 'a"b\\c\tdé'
    assert block[param.start : param.end] == b'a\\"b\\\\c\td\xc3\xa9'
    assert param.end == len(block) - 1


@pytest.mark.parametrize(
    "value",
    [
        b"form-data; name",  # no "="
        b"form-data; name ",  # key then end
        b'form-data; ="x"',  # empty key
        b'form-data; (x)="y"',  # not a token
        b'form-data; name="a"; NAME="b"',  # repeated (case-insensitive)
        b'form-data; name="a"x',  # garbage after a quoted value
        b"form-data; name=a b",  # space inside an unquoted value
        b'form-data; name="a" filename="b"',  # missing ";"
        b"form-data; name=",  # empty unquoted value
        b"form-data; name=;",  # empty unquoted value before ";"
        b'form-data; name="abc',  # unterminated
        b'form-data; name="abc\\"',  # an escaped quote, then unterminated
        b'form-data; name="abc\\',  # a backslash ends the block
        b'form-data; filename="C:\\data\\x.jsonl"',  # a bare backslash
        b'form-data; filename="a\\Xb"',  # a bare backslash before any letter
        b'form-data; name="a\x01b"',  # a control character
        b'form-data; name="a\x7fb"',  # DEL
        b'form-data; name="\xff"',  # not UTF-8 (quoted)
        b"form-data; name=\xff",  # not UTF-8 (unquoted)
        b"form-data; name=a\\b",  # a backslash in an unquoted value
    ],
)
def test_parameters_without_one_reading(value: bytes) -> None:
    part = _part(b"Content-Disposition: " + value)
    _ambiguous(lambda: part.params("content-disposition"))


def test_whitespace_around_the_equals_sign_is_accepted() -> None:
    params = _part(b"Content-Type: text/plain ; charset = utf-8").params("content-type")
    assert params is not None and params["charset"].value == "utf-8"


# --- RFC 8187 ext-values ----------------------------------------------------------------


def _ext(value: str, *, quoted: bool = False) -> Param:
    return Param(value=value, quoted=quoted, start=0, end=len(value))


def test_ext_value_decodes_percent_and_attr_chars() -> None:
    assert _ext_value(_ext("UTF-8''jane.doe%40corp.example.jsonl")) == (
        "UTF-8",
        "",
        b"jane.doe@corp.example.jsonl",
    )
    assert _ext_value(_ext("utf-8'en'%c3%a9t%C3%A9~_!")) == ("utf-8", "en", b"\xc3\xa9t\xc3\xa9~_!")
    assert _ext_value(_ext("UTF-8''")) == ("UTF-8", "", b"")


@pytest.mark.parametrize(
    ("value", "quoted"),
    [
        ("UTF-8''name.jsonl", True),  # RFC 8187 values are never quoted
        ("''name.jsonl", False),  # no charset
        ("UTF-8'name.jsonl", False),  # one quote only
        ("UTF-8name.jsonl", False),  # no quotes
        ("UTF-8''%4", False),  # truncated escape
        ("UTF-8''%", False),
        ("UTF-8''%zz", False),  # not hex
        ("UTF-8''a'b", False),  # a quote in the value
        ("UTF-8''a(b", False),  # not an attr-char
        ("UTF-8''caf\u00e9", False),  # raw non-ASCII
    ],
)
def test_malformed_ext_values(value: str, quoted: bool) -> None:
    _ambiguous(lambda: _ext_value(_ext(value, quoted=quoted)))


def test_ext_encode_and_quote() -> None:
    assert _ext_encode(b"a b%\xc2\xab.jsonl~") == "a%20b%25%C2%AB.jsonl~"
    assert _ext_encode(b"") == ""
    assert _quote('a"b\\c') == 'a\\"b\\\\c'


# --- routing reads: name / filename ------------------------------------------------------


def test_name_and_filename_read_the_strict_grammar() -> None:
    part = _part(b'Content-Disposition: form-data; name="a\\"b"; filename=""')
    assert part.name == 'a"b'
    assert part.filename == ""  # an empty file input is still a file part
    assert _part(b"Content-Type: text/plain").name is None
    assert _part(b"Content-Type: text/plain").filename is None
    assert _part(b"Content-Disposition: form-data").filename is None


def test_filename_star_alone_makes_a_file_part() -> None:
    part = _part(b"Content-Disposition: form-data; name=f; filename*=UTF-8''a%20b.jsonl")
    assert part.filename == "UTF-8''a%20b.jsonl"


def test_ambiguous_disposition_falls_back_to_the_lenient_reading() -> None:
    part = _part(b'Content-Disposition: form-data; name="f"; filename="C:\\data\\x.jsonl"')
    assert part.name == "f"
    assert part.filename == "C:\\data\\x.jsonl"


# --- redact_filenames -------------------------------------------------------------------


def test_plain_quoted_filename_rewritten_in_place() -> None:
    part = _part(
        b'Content-Disposition: form-data; name="file"; filename="' + EMAIL.encode() + b'.jsonl"\r\n'
        b"Content-Type: application/jsonl"
    )
    assert part.redact_filenames(_redact, strict=True) is True
    assert part.headers == (
        b'Content-Disposition: form-data; name="file"; filename="\xc2\xabEMAIL_001\xc2\xbb.jsonl"'
        b"\r\nContent-Type: application/jsonl"
    )


def test_plain_unquoted_filename_stays_unquoted() -> None:
    part = _part(b"Content-Disposition: form-data; filename=" + EMAIL.encode() + b".jsonl; name=x")
    assert part.redact_filenames(_redact, strict=False) is True
    assert part.headers == (
        b"Content-Disposition: form-data; filename=\xc2\xabEMAIL_001\xc2\xbb.jsonl; name=x"
    )


def test_quoted_filename_keeps_its_escapes() -> None:
    part = _part(b'Content-Disposition: form-data; filename="a\\"b\\\\ ' + EMAIL.encode() + b'"')
    assert part.redact_filenames(_redact, strict=True) is True
    assert part.headers == (
        b'Content-Disposition: form-data; filename="a\\"b\\\\ \xc2\xabEMAIL_001\xc2\xbb"'
    )


@pytest.mark.parametrize("ext_first", [True, False])
def test_both_filename_forms_rewritten_in_either_order(ext_first: bool) -> None:
    ext = b"filename*=utf-8'en'" + EMAIL.replace("@", "%40").encode() + b".jsonl"
    plain = b'filename="' + EMAIL.encode() + b'.jsonl"'
    first, second = (ext, plain) if ext_first else (plain, ext)
    part = _part(b'Content-Disposition: form-data; name="file"; ' + first + b"; " + second)
    assert part.redact_filenames(_redact, strict=True) is True
    new_ext = b"filename*=utf-8'en'%C2%ABEMAIL_001%C2%BB.jsonl"
    new_plain = b'filename="\xc2\xabEMAIL_001\xc2\xbb.jsonl"'
    expected = (new_ext, new_plain) if ext_first else (new_plain, new_ext)
    assert part.headers == (
        b'Content-Disposition: form-data; name="file"; ' + expected[0] + b"; " + expected[1]
    )


def test_nothing_to_redact_changes_nothing() -> None:
    block = b"Content-Disposition: form-data; name=f; filename*=UTF-8''clean.jsonl; filename=c"
    part = _part(block)
    assert part.redact_filenames(_redact, strict=True) is False
    assert part.headers == block
    assert _part(None).redact_filenames(_redact, strict=True) is False
    assert _part(b"Content-Type: text/plain").redact_filenames(_redact, strict=True) is False


def test_foreign_charset_filename_star() -> None:
    block = (
        b"Content-Disposition: form-data; filename*=iso-8859-1''caf%E9.jsonl; filename=\""
        + EMAIL.encode()
        + b'.jsonl"'
    )
    _ambiguous(
        lambda: _part(block).redact_filenames(_redact, strict=True),
        "a multipart filename* is not UTF-8",
    )
    # Lenient: that filename* stays byte-identical, the plain one is redacted.
    part = _part(block)
    assert part.redact_filenames(_redact, strict=False) is True
    assert part.headers == (
        b"Content-Disposition: form-data; filename*=iso-8859-1''caf%E9.jsonl; filename=\""
        b'\xc2\xabEMAIL_001\xc2\xbb.jsonl"'
    )


def _never(text: str) -> str:
    raise AssertionError("redact must not run on a disposition without one reading")


@pytest.mark.parametrize(
    "block",
    [
        b'Content-Disposition: form-data; filename="C:\\x\\' + EMAIL.encode() + b'"',
        b'Content-Disposition: form-data; filename="a"; filename*="UTF-8\'\'b"',
        b"Content-Disposition: form-data; filename=\"a\"; filename*=UTF-8''%ff",
        b'Content-Disposition: form-data; name="f"\r\n filename="' + EMAIL.encode() + b'"',
    ],
    ids=["bare-backslash", "quoted-ext", "ext-not-utf8-bytes", "folded"],
)
def test_ambiguous_dispositions(block: bytes) -> None:
    _ambiguous(lambda: _part(block).redact_filenames(_never, strict=True))
    part = _part(block)
    assert part.redact_filenames(_never, strict=False) is False
    assert part.headers == block
