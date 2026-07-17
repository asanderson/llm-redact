"""Adversarial / differential fuzzing of the streaming byte codecs.

The codecs parse UNTRUSTED upstream bytes; each carries a fail-safe invariant
the chunk-split sweeps assert over golden fixtures. Hypothesis throws random and
mutated bytes at them and asserts those invariants hold on inputs no fixture
covers: parsers never raise an unexpected exception, never lose bytes, and the
byte-faithful multipart codec round-trips. Extends (never replaces) the sweeps.
"""

from hypothesis import given
from hypothesis import strategies as st

from llm_redact import multipart
from llm_redact.eventstream import EventStreamError, EventStreamParser
from llm_redact.multipart import Multipart, MultipartPart
from llm_redact.ndjson import NDJSONParser
from llm_redact.sse import SSEParser


def _chunks(data: bytes, offsets: list[int]) -> list[bytes]:
    cuts = sorted({o % (len(data) + 1) for o in offsets} | {0, len(data)})
    return [data[a:b] for a, b in zip(cuts, cuts[1:], strict=False) if a != b] or [data]


# --- NDJSON: no byte is ever lost, regardless of chunking --------------------


@given(data=st.binary(max_size=800), offsets=st.lists(st.integers(0, 800), max_size=8))
def test_ndjson_reconstructs_every_byte(data: bytes, offsets: list[int]) -> None:
    parser = NDJSONParser()
    lines: list[bytes] = []
    for chunk in _chunks(data, offsets):
        lines.extend(parser.feed(chunk))
    # Returned lines + the unterminated tail, rejoined on LF, IS the input.
    assert b"\n".join([*lines, parser.close()]) == data


# --- SSE: never crashes; chunking does not change the parse ------------------


@given(data=st.binary(max_size=800), offsets=st.lists(st.integers(0, 800), max_size=8))
def test_sse_split_invariant_and_crash_free(data: bytes, offsets: list[int]) -> None:
    whole = SSEParser()
    one_shot = [(e.event, e.data) for e in whole.feed(data)]
    split = SSEParser()
    chunked: list[tuple[str | None, str]] = []
    for chunk in _chunks(data, offsets):
        chunked.extend((e.event, e.data) for e in split.feed(chunk))
    assert one_shot == chunked


# --- AWS eventstream: only EventStreamError, and residual loses no bytes -----


@given(data=st.binary(max_size=2000))
def test_eventstream_only_raises_its_own_error(data: bytes) -> None:
    parser = EventStreamParser()
    try:
        messages = parser.feed(data)
    except EventStreamError:
        # A framing violation is the documented degrade path — residual holds
        # every unreturned byte so the caller can forward verbatim.
        assert isinstance(parser.residual, bytes)
        return
    # On success the residual is a genuine suffix of what was fed (a partial
    # trailing frame), never fabricated bytes.
    assert data.endswith(parser.residual)
    assert isinstance(messages, list)


# --- multipart: never crashes; parse is byte-faithful when it succeeds -------


@given(data=st.binary(max_size=1500), boundary=st.binary(min_size=1, max_size=16))
def test_multipart_parse_never_crashes_and_round_trips(data: bytes, boundary: bytes) -> None:
    result = multipart.parse(data, boundary)  # must never raise
    if result is not None:
        assert result.serialize() == data  # byte-faithful


_HEADER = st.sampled_from(
    [
        None,
        b'content-disposition: form-data; name="f"',
        b'content-disposition: form-data; name="f"; filename="a.jsonl"',
    ]
)


@st.composite
def _valid_multipart(draw: st.DrawFn) -> Multipart:
    boundary = draw(st.text("abcdef0123456789", min_size=4, max_size=16)).encode()
    n = draw(st.integers(1, 3))
    parts = [
        MultipartPart(headers=draw(_HEADER), content=draw(st.binary(max_size=60))) for _ in range(n)
    ]
    return Multipart(boundary=boundary, preamble=b"", parts=parts, epilogue=b"")


@given(m=_valid_multipart())
def test_multipart_serialize_round_trips_through_parse(m: Multipart) -> None:
    body = m.serialize()
    reparsed = multipart.parse(body, m.boundary)
    assert reparsed is not None
    assert reparsed.serialize() == body
