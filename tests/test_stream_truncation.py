"""Early-termination invariants: a stream that ends mid-token must still be
safe.

The chunk-split sweeps (test_rehydrate.py / test_sse.py) prove that a COMPLETE
stream reassembles no matter how it is cut. This suite proves the other fault:
the stream ENDS early — the upstream closed cleanly after delivering only a
prefix (or the connection dropped and the generator flushed what it had). A
partial placeholder left in the buffer must be emitted verbatim, never guessed
into a wrong value and never silently dropped.

The invariant is exact and reuses the non-streaming reference: for every
truncation point, `feed(prefix) + flush()` equals `rehydrate_text(prefix)` —
which restores exactly the complete tokens present in the prefix and passes
everything else, including a trailing partial token, through unchanged.

All streaming codecs (SSE / eventstream / ndjson / realtime WS) funnel their
text through this one StreamingRehydrator, so the core sweep below covers them;
the SSE end-to-end test ties the property to the real parser+adapter path.
"""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_redact.providers import OpenAIAdapter
from llm_redact.rehydrate import Rehydrator, RehydratorPool, StreamingRehydrator
from llm_redact.sse import SSEParser
from llm_redact.vault import InMemoryVault

EMAIL = "jane@corp.example"
KEY = "AKIAIOSFODNN7EXAMPLE"


@pytest.fixture
def loaded_vault() -> InMemoryVault:
    vault = InMemoryVault()
    vault.placeholder_for("EMAIL", EMAIL)  # «EMAIL_001»
    vault.placeholder_for("AWS_KEY", KEY)  # «AWS_KEY_001»
    return vault


def _expected(vault: InMemoryVault, text: str, fuzzy: bool) -> str:
    return Rehydrator(vault, fuzzy=fuzzy).rehydrate_text(text)


TRUNC_CORPUS = [
    "a «EMAIL_001» b",
    "«EMAIL_001»«AWS_KEY_001»",
    "x «EMAIL_001» y «AWS_KEY_001» z",
    "unknown «MYSTERY_042» here",
    "false «EMAIL_ start",
    "mangled «email_001» case",
    "« EMAIL_001 » padded",
    "trailing lone «",
]


@pytest.mark.parametrize("fuzzy", [False, True])
@pytest.mark.parametrize("text", TRUNC_CORPUS)
def test_truncation_at_every_offset_is_safe(
    loaded_vault: InMemoryVault, text: str, fuzzy: bool
) -> None:
    """The stream is cut short at `trunc`; the delivered prefix is itself cut
    at `s`. However both fall, the streamed output equals the non-streaming
    rehydration of exactly what arrived — a partial token flushes verbatim,
    never a wrong value, never dropped."""
    for trunc in range(len(text) + 1):
        prefix = text[:trunc]
        expected = _expected(loaded_vault, prefix, fuzzy)
        for s in range(trunc + 1):
            r = StreamingRehydrator(loaded_vault, fuzzy=fuzzy)
            out = r.feed(prefix[:s]) + r.feed(prefix[s:]) + r.flush()
            assert out == expected, f"trunc={trunc} s={s} fuzzy={fuzzy}"


@settings(max_examples=300, deadline=None)
@given(
    fragments=st.lists(
        st.sampled_from(
            ["hello ", "«EMAIL_001»", " world ", "«AWS_KEY_001»", "«PARTIAL_", "» ", "«", "x"]
        ),
        min_size=0,
        max_size=8,
    ),
    trunc=st.integers(min_value=0, max_value=200),
    split=st.integers(min_value=0, max_value=200),
    fuzzy=st.booleans(),
)
def test_truncation_property(fragments: list[str], trunc: int, split: int, fuzzy: bool) -> None:
    vault = InMemoryVault()
    vault.placeholder_for("EMAIL", EMAIL)
    vault.placeholder_for("AWS_KEY", KEY)
    text = "".join(fragments)
    prefix = text[: trunc % (len(text) + 1)] if text else ""
    s = split % (len(prefix) + 1) if prefix else 0
    r = StreamingRehydrator(vault, fuzzy=fuzzy)
    out = r.feed(prefix[:s]) + r.feed(prefix[s:]) + r.flush()
    assert out == _expected(vault, prefix, fuzzy)


def test_json_source_partial_token_flushes_verbatim(loaded_vault: InMemoryVault) -> None:
    """A tool-call argument stream truncated mid-token flushes the partial
    guillemet text verbatim — the value is never spliced in on a partial."""
    r = StreamingRehydrator(loaded_vault, json_source=True)
    out = r.feed('{"k": "«EMAIL_0') + r.flush()
    assert EMAIL not in out  # never guessed from a partial token
    assert "«EMAIL_0" in out  # never dropped
    # And the complete stream still restores (re-escaped into valid JSON).
    r2 = StreamingRehydrator(loaded_vault, json_source=True)
    full = r2.feed('{"k": "«EMAIL_001»"}') + r2.flush()
    assert json.loads(full) == {"k": EMAIL}


def test_sse_truncated_stream_never_leaks_a_wrong_value(loaded_vault: InMemoryVault) -> None:
    """End-to-end through the real SSE parser + adapter: an OpenAI stream whose
    token is split across two deltas, truncated at every byte offset. The
    restored value only ever appears once its whole token was delivered, and a
    value whose token never arrived never leaks."""
    adapter = OpenAIAdapter()
    token = "«EMAIL_001»"

    def _delta(part: str) -> bytes:
        payload = {"choices": [{"index": 0, "delta": {"content": part}}]}
        return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()

    first = _delta("mail " + token[:4])
    second = _delta(token[4:] + " sent")
    stream = first + second + b"data: [DONE]\n\n"
    # The token is reassembled across two content deltas; its value can only
    # surface once every byte of the token has been delivered (the closing
    # guillemet lives in the second delta). Framing after that point is
    # irrelevant to safety.
    token_delivered_at = len(first) + second.rindex(b"\xbb") + 1

    def _content(data: str) -> str:
        data = data.strip()
        if not data or data == "[DONE]":
            return ""
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return data  # a partial trailing line the adapter forwarded verbatim
        choices = payload.get("choices") or [{}]
        return choices[0].get("delta", {}).get("content", "") or ""

    for n in range(len(stream) + 1):
        delivered = stream[:n]
        parser = SSEParser()
        pool = RehydratorPool(loaded_vault)
        out = ""
        for event in [*parser.feed(delivered), *parser.close()]:
            for rewritten in adapter.rehydrate_event(event, pool):
                out += _content(rewritten.data)
        out += "".join(pool.flush_all().values())
        # never a wrong value: the AWS key was never in this stream at all.
        assert KEY not in out
        # never a guessed value: the email only appears once every byte of
        # its token (split across the two deltas) was delivered.
        if EMAIL in out:
            assert n >= token_delivered_at
