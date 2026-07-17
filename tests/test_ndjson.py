"""The NDJSON line framing must reassemble under any chunking."""

from llm_redact.ndjson import NDJSONParser

STREAM = b'{"response":"hel"}\n{"response":"lo"}\n\n{"done":true}\n'
LINES = [b'{"response":"hel"}', b'{"response":"lo"}', b"", b'{"done":true}']


def test_split_at_every_offset() -> None:
    for split in range(len(STREAM) + 1):
        parser = NDJSONParser()
        lines = parser.feed(STREAM[:split]) + parser.feed(STREAM[split:])
        assert lines == LINES, split
        assert parser.close() == b""


def test_unterminated_tail_surfaces_on_close() -> None:
    parser = NDJSONParser()
    assert parser.feed(b'{"a":1}\n{"partial"') == [b'{"a":1}']
    assert parser.close() == b'{"partial"'
    assert parser.close() == b""  # drained


def test_lines_are_raw_bytes() -> None:
    # CRLF bytes stay inside the line: byte fidelity is the caller's to keep.
    parser = NDJSONParser()
    assert parser.feed(b'{"a":1}\r\n') == [b'{"a":1}\r']
