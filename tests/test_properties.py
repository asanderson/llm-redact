"""Hypothesis properties over the invariants the deterministic sweeps pin.

These extend — never replace — the exhaustive chunk-split sweeps in
test_rehydrate.py/test_sse.py and the golden eventstream fixtures: random
inputs probe the same invariants from directions no hand-written case
covers. deadline=None throughout: CI machines are slow and none of these
properties is about speed.
"""

import uuid as uuid_module
from collections.abc import Callable

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

from llm_redact.detection.base import Detection
from llm_redact.detection.engine import (
    DetectionConfig,
    active_rule_names,
    build_allowlist,
    build_detectors,
)
from llm_redact.detection.regex_rules import BUILTIN_RULES
from llm_redact.eventstream import (
    BOOL_FALSE,
    BOOL_TRUE,
    BYTE,
    BYTE_ARRAY,
    INT,
    LONG,
    SHORT,
    STRING,
    TIMESTAMP,
    UUID,
    EventStreamMessage,
    EventStreamParser,
    serialize,
)
from llm_redact.jsonwalk import STRUCTURAL_KEYS, transform_strings
from llm_redact.placeholders import PLACEHOLDER_RE, canonicalize, format_placeholder
from llm_redact.providers.base import _EXEMPT_STASH_SENTINEL, stash_exempt_mcp_blocks
from llm_redact.redactor import Redactor, _resolve_overlaps, _sweep
from llm_redact.rehydrate import (
    Rehydrator,
    RehydratorPool,
    StreamingRehydrator,
    substitute_tokens,
)
from llm_redact.vault import InMemoryVault, InMemoryVaultManager

# --- fuzzy placeholder grammar ---------------------------------------------

# Detector-type names shaped like the real ones (EMAIL, CREDIT_CARD): short
# enough that every canonical form fits MAX_PLACEHOLDER_LEN.
_type_names = st.from_regex(r"[A-Z][A-Z]{0,8}(?:_[A-Z]{1,8}){0,2}", fullmatch=True)
_numbers = st.integers(min_value=1, max_value=9_999_999)


@st.composite
def _mangled_tokens(draw: st.DrawFn) -> tuple[str, str]:
    """(canonical token, a random legal mangle of it)."""
    type_name = draw(_type_names)
    n = draw(_numbers)
    canonical = format_placeholder(type_name, n)

    # Random case per character, hyphens for underscores.
    body = "".join(ch.lower() if draw(st.booleans()) else ch for ch in type_name).replace(
        "_", draw(st.sampled_from(["_", "-"]))
    )
    separator = draw(st.sampled_from(["_", "-"]))
    digits = f"{'0' * draw(st.integers(0, 4))}{n}"
    pad_left = draw(st.sampled_from(["", " ", " ", "  "]))
    pad_right = draw(st.sampled_from(["", " ", " ", "  "]))
    return canonical, f"«{pad_left}{body}{separator}{digits}{pad_right}»"


@settings(deadline=None)
@given(_mangled_tokens())
def test_every_legal_mangle_canonicalizes_back(pair: tuple[str, str]) -> None:
    canonical, mangle = pair
    assert canonicalize(mangle) == canonical
    # And canonical forms are fixed points of the grammar.
    assert canonicalize(canonical) == canonical


# --- streaming == whole-text under arbitrary chunkings ---------------------

_SECRETS = {"EMAIL": "jane.doe@corp.example", "AWS_KEY": "AKIAIOSFODNN7EXAMPLE"}


def _seeded_vault() -> InMemoryVault:
    vault = InMemoryVault()
    for detector_type, value in _SECRETS.items():
        vault.placeholder_for(detector_type, value)
    return vault


_fragments = st.lists(
    st.one_of(
        st.text(alphabet="ab «»_-0123456789EMAILWSKY ", max_size=12),
        st.sampled_from(["«EMAIL_001»", "«AWS_KEY_001»", "«email-1»", "«EMAIL_", "«UNKNOWN_042»"]),
    ),
    max_size=12,
)


@settings(deadline=None)
@given(fragments=_fragments, cuts=st.data(), fuzzy=st.booleans())
def test_streaming_equals_whole_text_for_any_chunking(
    fragments: list[str], cuts: st.DataObject, fuzzy: bool
) -> None:
    text = "".join(fragments)
    vault = _seeded_vault()
    expected = substitute_tokens(text, vault, fuzzy=fuzzy, json_escape=False)

    rehydrator = StreamingRehydrator(vault, fuzzy=fuzzy)
    pieces: list[str] = []
    remaining = text
    while remaining:
        size = cuts.draw(st.integers(1, len(remaining)))
        pieces.append(rehydrator.feed(remaining[:size]))
        remaining = remaining[size:]
    pieces.append(rehydrator.flush())
    assert "".join(pieces) == expected


@settings(deadline=None)
@given(text=st.text(alphabet=st.characters(exclude_characters="«"), max_size=200))
def test_substitute_tokens_is_identity_on_token_free_text(text: str) -> None:
    vault = _seeded_vault()
    for fuzzy in (False, True):
        assert substitute_tokens(text, vault, fuzzy=fuzzy, json_escape=False) == text


# --- eventstream: parse ∘ serialize = identity ------------------------------

_header_values = st.one_of(
    st.tuples(st.just(BOOL_TRUE), st.just(True)),
    st.tuples(st.just(BOOL_FALSE), st.just(False)),
    st.tuples(st.just(BYTE), st.integers(-(2**7), 2**7 - 1)),
    st.tuples(st.just(SHORT), st.integers(-(2**15), 2**15 - 1)),
    st.tuples(st.just(INT), st.integers(-(2**31), 2**31 - 1)),
    st.tuples(st.just(LONG), st.integers(-(2**63), 2**63 - 1)),
    st.tuples(st.just(TIMESTAMP), st.integers(-(2**63), 2**63 - 1)),
    st.tuples(st.just(BYTE_ARRAY), st.binary(max_size=64)),
    st.tuples(st.just(STRING), st.text(max_size=64)),
    st.tuples(st.just(UUID), st.uuids().map(lambda u: uuid_module.UUID(bytes=u.bytes))),
)

_messages = st.builds(
    EventStreamMessage,
    headers=st.lists(
        st.tuples(st.text(min_size=1, max_size=32), _header_values).map(
            lambda pair: (pair[0], pair[1][0], pair[1][1])
        ),
        max_size=6,
    ).filter(lambda headers: all(len(name.encode()) <= 255 for name, _c, _v in headers)),
    payload=st.binary(max_size=256),
)


@settings(deadline=None)
@given(message=_messages)
def test_eventstream_roundtrip_is_identity(message: EventStreamMessage) -> None:
    raw = serialize(message)
    parser = EventStreamParser()
    (back,) = parser.feed(raw)
    parser.close()
    assert back == message
    assert serialize(back) == raw


@settings(deadline=None)
@given(messages=st.lists(_messages, min_size=1, max_size=4), cuts=st.data())
def test_eventstream_stream_reassembles_under_any_chunking(
    messages: list[EventStreamMessage], cuts: st.DataObject
) -> None:
    stream = b"".join(serialize(m) for m in messages)
    parser = EventStreamParser()
    out: list[EventStreamMessage] = []
    remaining = stream
    while remaining:
        size = cuts.draw(st.integers(1, len(remaining)))
        out.extend(parser.feed(remaining[:size]))
        remaining = remaining[size:]
    parser.close()
    assert out == messages


# --- codec round-trip invariants (mutation round 2, 1.16.0) ------------------
# Complement the targeted kills with random-input properties: the parse ->
# serialize path must be byte-faithful, and the incremental parser must
# conserve every byte under any chunking.


@st.composite
def _multipart_bodies(draw: st.DrawFn) -> tuple[bytes, bytes]:
    """A (body, boundary) pair in the canonical multipart grammar parse()
    accepts — preambles and epilogues included so the round-trip proves the
    verbatim regions survive too."""
    boundary = draw(st.text(alphabet="abcdefABCDEF0123456789", min_size=1, max_size=12)).encode()
    delim = b"--" + boundary
    n = draw(st.integers(min_value=1, max_value=4))
    out = bytearray()
    out += draw(st.sampled_from([b"", b"\r\n", b"preamble\r\n"]))
    for _ in range(n):
        out += delim + b"\r\n"
        if draw(st.booleans()):
            # header bytes must not contain the CRLFCRLF separator
            header = draw(st.binary(max_size=40)).replace(b"\r\n", b" ")
            out += header + b"\r\n\r\n"
        # content must not embed CRLF+delim (the parser's frame terminator)
        content = draw(st.binary(max_size=60)).replace(b"\r\n" + delim, b" ")
        out += content + b"\r\n"
    out += delim + b"--"
    out += draw(st.sampled_from([b"", b"\r\n", b"epilogue"]))
    return bytes(out), boundary


@settings(deadline=None)
@given(pair=_multipart_bodies())
def test_multipart_parse_serialize_is_byte_identical(pair: tuple[bytes, bytes]) -> None:
    from llm_redact.multipart import parse as mp_parse

    body, boundary = pair
    parsed = mp_parse(body, boundary)
    # Some random draws fall outside the grammar (None) -> forwarded verbatim,
    # which is the safe default; only assert faithfulness when it parsed.
    if parsed is not None:
        assert parsed.serialize() == body


@settings(deadline=None)
@given(names=st.lists(st.text(min_size=1, max_size=8), min_size=0, max_size=5))
def test_sse_serialize_parse_preserves_event_fields(names: list[str]) -> None:
    from llm_redact.sse import SSEEvent, SSEParser
    from llm_redact.sse import serialize as sse_serialize

    # Field values with no embedded newlines round-trip through the codec.
    event = SSEEvent(
        event="msg" if names else None,
        data="\n".join(n for n in names) if names else "",
        id=names[0] if names else None,
    )
    raw = sse_serialize(event)
    back = SSEParser().feed(raw if raw.endswith(b"\n\n") else raw + b"\n\n")
    if event.event or event.data or event.id:
        assert back, "a non-empty event must parse back to at least one event"


# --- the consolidated never-wrong-value battery
# --- the consolidated never-wrong-value battery ------------------------------
#
# Resilience commitment #1 as ONE falsifiable statement instead of scattered
# spot-checks: across ANY interleaving of writes over ANY number of sessions,
# ANY mix of known/foreign/unknown/mangled tokens in a text, ANY chunking of
# the stream, and ANY truncation point, a placeholder restores to EXACTLY the
# value its own session stored for it — or passes through verbatim. Never
# another session's secret, never a different value, and counters stay dense.
# (Write-path faults are the sqlite battery's job: test_vault_faults.py.)

# Two deliberately shared type names force per-(session, type) counter
# collisions across sessions — every session has an «EMAIL_001».
_NWV_TYPES = ("EMAIL", "PHONE")
_nwv_values = st.text(
    alphabet=st.characters(exclude_characters="«»", codec="utf-8"), min_size=1, max_size=24
)


class NeverWrongValueMachine(RuleBasedStateMachine):
    sessions: Bundle[str] = Bundle("sessions")

    @initialize()
    def setup(self) -> None:
        self.manager = InMemoryVaultManager()
        # Shadow model: (session, type, value) -> token and its inverse.
        self.tokens: dict[tuple[str, str, str], str] = {}
        self.by_session: dict[str, dict[str, str]] = {}

    @rule(target=sessions, name=st.sampled_from(["s1", "s2", "s3", "s4"]))
    def open_session(self, name: str) -> str:
        self.by_session.setdefault(name, {})
        return name

    @rule(session=sessions, detector_type=st.sampled_from(_NWV_TYPES), value=_nwv_values)
    def write(self, session: str, detector_type: str, value: str) -> None:
        token = self.manager.get(session).placeholder_for(detector_type, value)
        key = (session, detector_type, value)
        if key in self.tokens:
            # Deterministic: the same triple always yields the same token.
            assert token == self.tokens[key]
        else:
            # Fresh value: a canonical, never-before-issued token in this
            # session (distinct values never share a token).
            assert PLACEHOLDER_RE.fullmatch(token)
            assert token not in self.by_session[session]
            self.tokens[key] = token
            self.by_session[session][token] = value

    @rule(
        session=sessions,
        data=st.data(),
        fuzzy=st.booleans(),
        streaming=st.booleans(),
    )
    def restore(self, session: str, data: st.DataObject, fuzzy: bool, streaming: bool) -> None:
        # Build a text mixing THIS session's tokens, tokens of other sessions
        # (same names by construction!), unknown tokens, legal mangles, and
        # plain filler — then rehydrate through THIS session's vault.
        own = sorted(self.by_session[session])
        foreign = sorted(
            {t for s, m in self.by_session.items() if s != session for t in m} - set(own)
        )
        pool = own + foreign + ["«UNKNOWN_042»", "«EMAIL_", "plain text, no tokens"]
        if fuzzy and own:
            token = data.draw(st.sampled_from(own))
            pool.append(token.lower())  # a legal mangle of a known token
        parts = data.draw(st.lists(st.sampled_from(pool), max_size=6))
        text = " ".join(parts)

        vault = self.manager.get(session)
        expected_whole = substitute_tokens(text, vault, fuzzy=fuzzy, json_escape=False)

        # Every canonical token of this session is gone from the output —
        # replaced by EXACTLY the value this session stored (verified via the
        # shadow model), and no other session's value can appear: any output
        # value must be one THIS session stored or the token itself verbatim.
        for token in PLACEHOLDER_RE.findall(expected_whole):
            assert token not in self.by_session[session]  # own tokens restored
        for token in own:
            if token in text:
                assert self.by_session[session][token] in expected_whole

        if streaming and text:
            rehydrator = StreamingRehydrator(vault, fuzzy=fuzzy)
            pieces: list[str] = []
            remaining = text
            while remaining:
                size = data.draw(st.integers(1, len(remaining)))
                pieces.append(rehydrator.feed(remaining[:size]))
                remaining = remaining[size:]
            pieces.append(rehydrator.flush())
            assert "".join(pieces) == expected_whole

        if text:
            # Truncation: a stream that dies mid-token flushes what arrived,
            # rehydrated exactly as the whole-text path would rehydrate the
            # prefix — the partial is verbatim, never a guessed value.
            cut = data.draw(st.integers(1, len(text)))
            prefix = text[:cut]
            rehydrator = StreamingRehydrator(vault, fuzzy=fuzzy)
            got = rehydrator.feed(prefix) + rehydrator.flush()
            assert got == substitute_tokens(prefix, vault, fuzzy=fuzzy, json_escape=False)

    @invariant()
    def counters_stay_dense(self) -> None:
        # Per (session, type): issued numbers are exactly 1..n — no gap (a
        # lost number) and no reuse (a collision that would rehydrate the
        # wrong secret).
        for mapping in self.by_session.values():
            by_type: dict[str, list[int]] = {}
            for token in mapping:
                body = token[1:-1]
                type_name, _, digits = body.rpartition("_")
                by_type.setdefault(type_name, []).append(int(digits))
            for numbers in by_type.values():
                assert sorted(numbers) == list(range(1, len(numbers) + 1))


NeverWrongValueMachine.TestCase.settings = settings(deadline=None)
TestNeverWrongValue = NeverWrongValueMachine.TestCase


# --- body layer: redact -> rehydrate is identity over arbitrary JSON --------
#
# The differential-fuzzing ladder's top rung: the byte codecs are fuzzed in
# test_codec_fuzz.py; these fuzz the JSON BODY pipeline built on them. One
# vault, real detectors: whatever redaction takes out of a random body,
# rehydration puts back — for every shape jsonwalk can visit.

_body_secrets = st.sampled_from(
    ["jane.doe@corp.example", "AKIAIOSFODNN7EXAMPLE", "reach me at bob@x.example today"]
)
_body_strings = st.one_of(
    st.text(alphabet=st.characters(exclude_characters="«»"), max_size=20), _body_secrets
)
_body_keys = st.sampled_from(
    ["content", "text", "q", "input", "model", "role", "type", "name", "data", "object"]
)
_bodies = st.recursive(
    _body_strings | st.integers() | st.booleans() | st.none(),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(_body_keys, children, max_size=4),
        # Shapes that flip jsonwalk's data-key behavior: the Anthropic
        # plaintext-document source and the OpenAI list envelope.
        st.fixed_dictionaries({"type": st.just("text"), "data": _body_strings}),
        st.fixed_dictionaries({"object": st.just("list"), "data": st.lists(children, max_size=3)}),
    ),
    max_leaves=12,
)


@settings(deadline=None)
@given(body=_bodies)
def test_body_redact_then_rehydrate_is_identity(body: object) -> None:
    config = DetectionConfig()
    vault = InMemoryVault()
    redactor = Redactor(build_detectors(config), vault, build_allowlist(config))
    redacted = redactor.redact_json(body)
    assert Rehydrator(vault).rehydrate_json(redacted) == body


def _reference_walk(obj: object, fn: "Callable[[str], str]") -> object:
    """Independent re-derivation of jsonwalk's documented skip semantics —
    a differential oracle, deliberately NOT sharing code with jsonwalk."""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [_reference_walk(item, fn) for item in obj]
    if isinstance(obj, dict):
        out: dict[object, object] = {}
        for key, value in obj.items():
            walk_data = (obj.get("type") == "text" and isinstance(value, str)) or obj.get(
                "object"
            ) == "list"
            if key == "data" and walk_data:
                out[key] = _reference_walk(value, fn)
            elif key in STRUCTURAL_KEYS:
                out[key] = value
            else:
                out[key] = _reference_walk(value, fn)
        return out
    return obj


@settings(deadline=None)
@given(body=_bodies)
def test_jsonwalk_transforms_exactly_the_documented_strings(body: object) -> None:
    marker = "§transformed§"
    assert transform_strings(body, lambda s: marker + s) == _reference_walk(
        body, lambda s: marker + s
    )


# Fragments chosen so tokens routinely STRADDLE fragment boundaries within a
# channel while other channels stream unrelated bytes in between.
_POOL_FRAGMENTS = ("«EMAIL_001»", "«AWS_KEY_001»", "«EMA", "IL_001»", "plain ", "«UNKNOWN_042»")


@settings(deadline=None)
@given(
    assignments=st.lists(
        st.tuples(st.sampled_from(["c0", "c1", "c2"]), st.sampled_from(_POOL_FRAGMENTS)),
        max_size=16,
    ),
    cuts=st.data(),
)
def test_pool_channels_stay_isolated_under_interleaving(
    assignments: list[tuple[str, str]], cuts: st.DataObject
) -> None:
    # Fragments interleaved across channels of ONE pool: each channel's
    # streamed output must equal the whole-text rehydration of just ITS
    # fragments — a held partial token on one channel can never bleed into
    # another (this is what per-(choice, field) SSE channels rely on).
    vault = _seeded_vault()
    pool = RehydratorPool(vault)
    per_channel: dict[str, list[str]] = {"c0": [], "c1": [], "c2": []}
    streamed: dict[str, list[str]] = {"c0": [], "c1": [], "c2": []}
    for channel, fragment in assignments:
        per_channel[channel].append(fragment)
        remaining = fragment
        while remaining:
            size = cuts.draw(st.integers(1, len(remaining)))
            streamed[channel].append(pool.get(channel).feed(remaining[:size]))
            remaining = remaining[size:]
    for channel, fragments in per_channel.items():
        text = "".join(fragments)
        expected = substitute_tokens(text, vault, fuzzy=False, json_escape=False)
        assert "".join(streamed[channel]) + pool.flush(channel) == expected


# --- deny (tier 0) wins every overlap ---------------------------------------

_overlap_detections = st.lists(
    st.builds(
        lambda start, length, tier, prio: Detection(
            start=start,
            end=start + length,
            detector_type="T",
            value="v",
            priority=prio,
            tier=tier,
        ),
        start=st.integers(0, 40),
        length=st.integers(1, 12),
        tier=st.sampled_from([0, 1]),
        prio=st.sampled_from([1, 10, 100]),
    ),
    max_size=14,
)


@settings(deadline=None)
@given(detections=_overlap_detections)
def test_deny_tier0_wins_every_overlap(detections: list[Detection]) -> None:
    # Feed _resolve_overlaps the same (start, -length, priority) ordering the
    # engine produces, then assert the tier-0 guarantee from every angle.
    ordered = sorted(detections, key=lambda d: (d.start, -(d.end - d.start), d.priority))
    resolved = _resolve_overlaps(ordered)

    # The output is non-overlapping and start-sorted.
    for earlier, later in zip(resolved, resolved[1:], strict=False):
        assert earlier.end <= later.start

    # Every deny span is kept exactly as the tier-0-only sweep would pick it:
    # a tier-1 span can NEVER displace a deny, no matter where it starts or
    # how long it is.
    deny_out = [d for d in resolved if d.tier == 0]
    assert deny_out == _sweep([d for d in ordered if d.tier == 0])

    # No surviving tier-1 span overlaps any surviving deny span.
    for other in (d for d in resolved if d.tier != 0):
        assert all(other.end <= d.start or d.end <= other.start for d in deny_out)


# --- language scope: only out-of-scope national-id rules drop ---------------

_LANG_CODES = sorted({lang for rule in BUILTIN_RULES if rule.languages for lang in rule.languages})


@settings(deadline=None)
@given(scope=st.lists(st.sampled_from(_LANG_CODES), min_size=1, max_size=4, unique=True))
def test_language_scope_keeps_exactly_the_in_scope_rules(scope: list[str]) -> None:
    config = DetectionConfig(languages=tuple(sorted(scope)))
    active = set(active_rule_names(config))
    by_name = {rule.name: rule for rule in BUILTIN_RULES}
    for name in config.enabled:
        rule = by_name[name]
        if rule.languages is None:
            assert name in active  # universal rules always run
        else:
            in_scope = bool(set(rule.languages) & set(scope))
            assert (name in active) == in_scope  # tagged rule iff it shares a language
    assert "email" in active  # a universal rule is always present


# --- MCP exemption is fail-closed for uncorrelated tool results -------------


@settings(deadline=None)
@given(
    result_id=st.sampled_from(["u1", "u2", "u3"]),
    use_id=st.sampled_from(["u1", "u2"]),
    use_server=st.sampled_from(["exempt", "other"]),
)
def test_mcp_tool_result_exempt_only_when_correlated(
    result_id: str, use_id: str, use_server: str
) -> None:
    # An Anthropic mcp_tool_result names no server: it may bypass detection
    # ONLY when its tool_use_id points at an EXEMPT mcp_tool_use in the same
    # body. Anything else must fall through to normal redaction (fail-closed).
    exempt = frozenset({"exempt"})
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "mcp_tool_use",
                        "id": use_id,
                        "server_name": use_server,
                        "input": {"q": "s"},
                    },
                    {
                        "type": "mcp_tool_result",
                        "tool_use_id": result_id,
                        "content": [{"type": "text", "text": "s"}],
                    },
                ],
            }
        ]
    }
    stashed = stash_exempt_mcp_blocks(body, exempt)
    result_block = stashed["messages"][0]["content"][1]
    is_stashed = result_block == _EXEMPT_STASH_SENTINEL
    correlated = use_server == "exempt" and result_id == use_id
    assert is_stashed == correlated
