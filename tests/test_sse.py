from llm_redact.sse import SSEEvent, SSEParser, serialize

TRANSCRIPT = (
    b"event: message_start\n"
    b'data: {"type": "message_start"}\n'
    b"\n"
    b": ping keep-alive\n"
    b"\n"
    b"event: content_block_delta\n"
    b'data: {"type": "content_block_delta",\n'
    b'data:  "index": 0}\n'
    b"\n"
    b"data: [DONE]\n"
    b"\n"
)


def _parse_all(chunks: list[bytes]) -> list[SSEEvent]:
    parser = SSEParser()
    events: list[SSEEvent] = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.close())
    return events


def test_whole_transcript() -> None:
    events = _parse_all([TRANSCRIPT])
    assert len(events) == 4
    assert events[0].event == "message_start"
    assert events[1].is_comment_only
    assert events[1].comments == ["ping keep-alive"]
    assert events[2].data == '{"type": "content_block_delta",\n "index": 0}'
    assert events[3].data == "[DONE]"


def test_split_at_every_byte_offset() -> None:
    expected = _parse_all([TRANSCRIPT])
    for cut in range(len(TRANSCRIPT) + 1):
        events = _parse_all([TRANSCRIPT[:cut], TRANSCRIPT[cut:]])
        assert events == expected, f"failed at byte cut={cut}"


def test_crlf_line_endings() -> None:
    raw = b"event: ping\r\ndata: {}\r\n\r\n"
    events = _parse_all([raw])
    assert len(events) == 1
    assert events[0].event == "ping"
    assert events[0].data == "{}"


def test_multibyte_utf8_split_mid_character() -> None:
    text = "data: héllo «EMAIL_001»\n\n".encode()
    # Split inside the é (2 bytes) and inside « (2 bytes).
    for cut in range(len(text) + 1):
        events = _parse_all([text[:cut], text[cut:]])
        assert len(events) == 1
        assert events[0].data == "héllo «EMAIL_001»"


def test_serialize_round_trip() -> None:
    events = _parse_all([TRANSCRIPT])
    replayed = b"".join(serialize(e) for e in events)
    assert _parse_all([replayed]) == events


def test_unterminated_trailing_event_flushed_on_close() -> None:
    events = _parse_all([b"data: tail-no-newline"])
    assert len(events) == 1
    assert events[0].data == "tail-no-newline"
