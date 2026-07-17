"""Targeted kills for mutation round 2 (1.16.0) over the byte-level codecs.

Each test names the survivor(s) it kills. These are the real test gaps the
mutation gate surfaced across jsonwalk / placeholders / multipart /
eventstream / sse — behaviors the existing sweeps did not reach (a custom
skip_keys, an exact-boundary length, a crafted malformed frame, a trailing
whitespace byte). The justified equivalents live in
scripts/mutation_equivalents.py.
"""

import struct
import zlib

import pytest

from llm_redact.eventstream import (
    INT,
    STRING,
    UUID,
    EventStreamError,
    EventStreamMessage,
    EventStreamParser,
    _parse_headers,
    serialize,
    string_header,
)
from llm_redact.jsonwalk import transform_strings
from llm_redact.multipart import parse, parse_boundary
from llm_redact.placeholders import canonicalize, viable_prefix_start
from llm_redact.sse import SSEEvent, SSEParser
from llm_redact.sse import serialize as sse_serialize

# --- K1: jsonwalk must forward skip_keys / key_overrides through recursion ---


def test_transform_strings_forwards_skip_keys_into_dict_recursion() -> None:
    # mutmut_58: dropping skip_keys in the dict branch would transform a
    # protected key nested one level down — under realtime's custom skip
    # sets that redacts base64 audio / enum values.
    out = transform_strings({"outer": {"k": "v"}}, str.upper, skip_keys=frozenset({"k"}))
    assert out == {"outer": {"k": "v"}}  # k skipped at depth
    # Sanity: a non-skipped nested value IS transformed (keys are never touched).
    assert transform_strings({"outer": {"j": "v"}}, str.upper) == {"outer": {"j": "V"}}


def test_transform_strings_forwards_skip_keys_into_list_recursion() -> None:
    # mutmut_8: the list branch must forward skip_keys too.
    out = transform_strings([{"k": "v"}], str.upper, skip_keys=frozenset({"k"}))
    assert out == [{"k": "v"}]


def test_transform_strings_forwards_kwargs_through_object_list_envelope() -> None:
    # mutmut_44/47/48: the {"object":"list","data":[...]} envelope walk must
    # forward BOTH skip_keys and key_overrides — the OpenAI Conversations
    # list-items rehydration rides this path with the `arguments` override.
    envelope = {"object": "list", "data": [{"k": "v", "arguments": "raw"}]}
    out = transform_strings(
        envelope,
        str.upper,
        skip_keys=frozenset({"k"}),
        key_overrides={"arguments": lambda s: s + "!"},
    )
    assert out["data"][0]["k"] == "v"  # skip_keys forwarded
    assert out["data"][0]["arguments"] == "raw!"  # override forwarded, not str.upper


# --- K9/K10: placeholder length + prefix boundaries ---------------------------


def test_canonicalize_accepts_exactly_max_length() -> None:
    # canonicalize_29 (> -> >=): a 31-char type + 6-digit counter canonicalizes
    # to exactly MAX_PLACEHOLDER_LEN (40); the mutant would drop it, silently
    # failing a fuzzy restore so the token leaks through unrestored.
    token = "«" + "A" * 31 + "_100000»"
    result = canonicalize(token)
    assert result is not None and len(result) == 40


def test_viable_prefix_start_releases_at_exact_max_length() -> None:
    # viable_prefix_start_12 (>= -> >): a tail of exactly MAX_PLACEHOLDER_LEN
    # must RELEASE (return None), not hold back forever.
    assert viable_prefix_start("«" + "A" * 39) is None  # tail length == 40


def test_viable_prefix_start_drops_only_the_leading_guillemet() -> None:
    # viable_prefix_start_14 (tail[1:] -> tail[2:]): "«/EMAIL" has an invalid
    # body first char (/), so the run can be emitted (None). The tail[2:]
    # mutant would inspect "EMAIL", judge it viable, and hold back a run that
    # can never close.
    assert viable_prefix_start("«/EMAIL") is None


# --- K16/K17: multipart parse framing -----------------------------------------

_BOUNDARY = b"bnd"
_GOLDEN = b'--bnd\r\nContent-Disposition: form-data; name="a"\r\n\r\nvalue\r\n--bnd--\r\n'


def test_parse_accepts_leading_crlf_preamble() -> None:
    # parse_13/14 (idx < 0 -> <=0 / <1): a body whose delimiter is preceded by
    # CRLF must parse (idx==0), not be rejected — a rejection forwards the
    # whole multipart UNREDACTED.
    parsed = parse(b"\r\n" + _GOLDEN, _BOUNDARY)
    assert parsed is not None
    assert parsed.preamble == b"\r\n"
    assert parsed.parts[0].content == b"value"


def test_parse_delimiter_search_skips_part_leading_boundary_text() -> None:
    # parse_40/42 (find(..., 2) -> find(..., 0)): a part whose content begins
    # (right after the opening delimiter's CRLF) with the boundary text must
    # parse — the offset-2 skip of the leading CRLF is what prevents a
    # premature match at position 0 (the mutant returns None → forwards the
    # whole body unredacted).
    body = b"--bnd\r\n--bndXX\r\n--bnd--\r\n"
    parsed = parse(body, _BOUNDARY)
    assert parsed is not None
    assert parsed.parts[0].content == b"--bndXX"


def test_parse_splits_headers_at_first_blank_line() -> None:
    # parse_57 (partition -> rpartition): a part whose CONTENT contains an
    # embedded blank line must split headers at the FIRST blank, not the last.
    body = b"--bnd\r\nX-H: 1\r\n\r\nline1\r\n\r\nline2\r\n--bnd--\r\n"
    parsed = parse(body, _BOUNDARY)
    assert parsed is not None
    assert parsed.parts[0].headers == b"X-H: 1"
    assert parsed.parts[0].content == b"line1\r\n\r\nline2"


# --- K18: parse_boundary parameter parsing ------------------------------------


def test_parse_boundary_handles_multi_param_and_edge_values() -> None:
    # parse_boundary_3/9/10 (partition/split edge cases with a trailing param).
    assert parse_boundary("multipart/form-data; boundary=abc; charset=utf-8") == b"abc"
    # parse_boundary_13 (partition("=") -> rpartition): a boundary value with =.
    assert parse_boundary("multipart/form-data; boundary=a=b") == b"a=b"
    # parse_boundary_21 (strip('"') -> strip includes extra char): a value
    # starting with the quote-adjacent letter must not be stripped.
    assert parse_boundary("multipart/form-data; boundary=Xabc") == b"Xabc"


def test_parse_boundary_ignores_non_ascii_bytes() -> None:
    # parse_boundary_26/29/30 (encode errors arg dropped / case-flipped): a
    # non-ASCII boundary must degrade via "ignore", never raise.
    assert parse_boundary("multipart/form-data; boundary=abcé") == b"abc"


# --- K11/K12/K13/K14/K15: eventstream framing bounds --------------------------


def _frame_with_headers(headers: bytes, payload: bytes = b"") -> bytes:
    total = 16 + len(headers) + len(payload)
    prelude = struct.pack(">II", total, len(headers))
    body = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + payload
    return body + struct.pack(">I", zlib.crc32(body))


@pytest.mark.parametrize(
    "block",
    [
        bytes((2,)) + b":x",  # 10: name fills the block, no type byte follows
        bytes((9,)) + b":x" + bytes((STRING,)) + struct.pack(">H", 1) + b"v",  # 11: name overrun
        bytes((2,)) + b":x" + bytes((INT,)) + b"\x00",  # 37: INT value truncated (<4 bytes)
        bytes((2,)) + b":x" + bytes((STRING,)) + b"\x00",  # 51: 2-byte length prefix truncated
        # 69 (pos + length -> pos - length): VALID length prefix declaring more
        # value bytes than the block holds — the mutant slices short and
        # returns a mis-parsed header instead of raising.
        bytes((2,)) + b":x" + bytes((STRING,)) + struct.pack(">H", 5) + b"ab",
        bytes((2,)) + b":x" + bytes((UUID,)) + b"\x00\x01",  # 84: UUID value truncated (<16 bytes)
    ],
)
def test_parse_headers_rejects_internally_truncated_block(block: bytes) -> None:
    # K11: a header block whose declared name/value length overruns its own
    # bytes must raise EventStreamError specifically — the length-bound
    # mutants each turn that into an IndexError / struct.error / ValueError,
    # which the proxy's eventstream branch would NOT treat as a framing
    # violation. Test _parse_headers directly (the frame CRCs are moot here).
    with pytest.raises(EventStreamError):
        _parse_headers(block)


def test_large_header_value_round_trips() -> None:
    # K12: parse_65 / serialize_headers_36 (>H -> >h): a header value >= 32768
    # bytes must round-trip; a signed-short length would go negative.
    big = "z" * 40000
    message = EventStreamMessage(headers=[string_header("big", big)], payload=b"p")
    out = EventStreamParser().feed(serialize(message))
    assert len(out) == 1
    assert out[0].header("big") == big


def test_header_name_length_boundary_is_255() -> None:
    # K13: serialize_headers_6/7 (> 255 -> >=255 / >256). 255 bytes serializes;
    # 256 raises EventStreamError (not a ValueError from bytes((256,))).
    ok = EventStreamMessage(headers=[string_header("n" * 255, "v")])
    assert EventStreamParser().feed(serialize(ok))[0].header("n" * 255) == "v"
    with pytest.raises(EventStreamError):
        serialize(EventStreamMessage(headers=[string_header("n" * 256, "v")]))


def test_frame_exactly_at_max_size_is_accepted() -> None:
    # K14: feed_17 (total > max -> >=): a frame whose total equals
    # max_frame_bytes must be accepted, not rejected as implausible.
    message = EventStreamMessage(headers=[string_header("k", "v")], payload=b"xyz")
    frame = serialize(message)
    parser = EventStreamParser(max_frame_bytes=len(frame))
    assert parser.feed(frame)[0].header("k") == "v"


def test_frame_with_headers_len_overrunning_total_is_rejected() -> None:
    # K14: feed_19 (headers_len > total-16 -> total+16): a frame whose declared
    # headers_len exceeds the frame with valid CRCs must raise.
    message = EventStreamMessage(headers=[string_header("k", "v")], payload=b"body")
    frame = bytearray(serialize(message))
    total = struct.unpack_from(">I", frame, 0)[0]
    struct.pack_into(">I", frame, 4, total - 8)  # headers_len in (total-16, total]
    struct.pack_into(">I", frame, 8, zlib.crc32(bytes(frame[:8])))  # fix prelude CRC
    struct.pack_into(">I", frame, total - 4, zlib.crc32(bytes(frame[: total - 4])))  # fix msg CRC
    with pytest.raises(EventStreamError):
        EventStreamParser().feed(bytes(frame))


def test_twelve_byte_bad_prelude_raises_immediately() -> None:
    # K15: feed_6/7 (>= 12 -> > 12 / >= 13): a 12-byte implausible prelude must
    # be validated at feed time, not deferred to close.
    bad_prelude = struct.pack(">II", 4, 0) + struct.pack(">I", 12345)  # total<16, wrong CRC
    parser = EventStreamParser()
    with pytest.raises(EventStreamError):
        parser.feed(bad_prelude)


# --- K19/K20/K21/K22/K23: SSE serialize + parse -------------------------------


def test_sse_serialize_emits_id_and_comment() -> None:
    # serialize_7 (append(id) -> append(None)): an event with an id must
    # serialize the id line (a None would crash the join).
    assert b"id: 1" in sse_serialize(SSEEvent(id="1", data="x"))
    # serialize_3 (else ":" -> "XX:XX"): a bare comment starts with ":".
    assert sse_serialize(SSEEvent(comments=[""])).startswith(b":")


def test_sse_feed_parses_id_field() -> None:
    # feed_64/65/66/67 (== "id" flips / event.id = None): the id field must be
    # captured.
    events = SSEParser().feed(b"id: abc\ndata: x\n\n")
    assert events[0].id == "abc"


def test_sse_feed_preserves_trailing_whitespace_in_data() -> None:
    # feed_22 (rstrip(b"\r") -> rstrip(None)): trailing spaces in a data value
    # must survive; only the CR is stripped.
    events = SSEParser().feed(b"data: hi \n\n")
    assert events[0].data == "hi "
    # feed_24 (rstrip(b"\r") -> rstrip(b"XX\rXX")): a trailing 'X' before the
    # newline must survive.
    events2 = SSEParser().feed(b"data: abX\n\n")
    assert events2[0].data == "abX"


def test_sse_feed_comment_stripping() -> None:
    # feed_40 (line[1:] -> line[2:]): a comment with no space after the colon.
    assert SSEParser().feed(b":ping\n\n")[0].comments == ["ping"]
    # feed_38 (lstrip(" ") -> lstrip(None)): a tab-led comment keeps the tab.
    assert SSEParser().feed(b":\tping\n\n")[0].comments == ["\tping"]
    # feed_41 (lstrip(" ") -> lstrip("XX XX")): an X-led comment keeps the X.
    assert SSEParser().feed(b":Xabc\n\n")[0].comments == ["Xabc"]


def test_sse_feed_multiple_comments_then_data_in_one_chunk() -> None:
    # feed_42 (continue -> break): 3+ comment events then a data event, all in
    # one chunk. A `break` would defer the buffer and lose the data event
    # (close() only re-feeds a bounded amount).
    events = SSEParser().feed(b": a\n\n: b\n\n: c\n\ndata: X\n\n")
    assert [e.comments for e in events if e.comments] == [["a"], ["b"], ["c"]]
    assert any(e.data == "X" for e in events)


def test_sse_parser_reusable_after_close() -> None:
    # close_13 (self._event = None -> ""): close() flushing an UNTERMINATED
    # event must reset _event so a reused parser starts clean; a "" leftover
    # (a str, not an SSEEvent) crashes the next comment append. The event must
    # be unterminated (no blank line) so close() reaches the reset at all.
    parser = SSEParser()
    parser.feed(b"data: unterminated")  # no trailing blank line
    flushed = parser.close()
    assert flushed and flushed[-1].data == "unterminated"
    events = parser.feed(b":ping\n\n")
    assert events[0].comments == ["ping"]
