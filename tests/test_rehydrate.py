"""The exhaustive chunk-split suite: streaming output must equal
non-streaming rehydration of the same text, no matter how the stream is cut."""

import json

import pytest

from llm_redact.placeholders import MAX_PLACEHOLDER_LEN
from llm_redact.rehydrate import Rehydrator, RehydratorPool, StreamingRehydrator
from llm_redact.vault import InMemoryVault


@pytest.fixture
def loaded_vault() -> InMemoryVault:
    vault = InMemoryVault()
    vault.placeholder_for("EMAIL", "jane@corp.example")  # «EMAIL_001»
    vault.placeholder_for("AWS_KEY", "AKIAIOSFODNN7EXAMPLE")  # «AWS_KEY_001»
    vault.placeholder_for("SECRET", 'pa"ss\nwor\\d')  # «SECRET_001», nasty chars
    return vault


CORPUS = [
    "plain text without tokens",
    "«EMAIL_001»",
    "prefix «EMAIL_001»",
    "«EMAIL_001» suffix",
    "a «EMAIL_001» b «AWS_KEY_001» c",
    "adjacent «EMAIL_001»«AWS_KEY_001» tokens",
    "unknown «MYSTERY_042» passes through",
    "il a dit «bonjour» et «EMAIL_001»",
    "lone « guillemet",
    "trailing «",
    "false prefix «EMAIL_ that never closes",
    "«EMAIL_001",  # incomplete token at end of stream
    # Mangled forms: restored when fuzzy, byte-identical when exact.
    "mangled «email_001» case",
    "mangled «Email_001» case",
    "mangled «EMAIL_1» padding",
    "mangled «EMAIL_0001» padding",
    "mangled «EMAIL-001» hyphen",
    "mangled « EMAIL_001 » pads",
    "combo «email-1 » everything",
    "adjacent mangles «email_001»«aws-key-1»",
    "il a dit « bonjour » et «email_001»",
    "les «top-10» mondiaux",  # canonicalizes to «TOP_010»: vault miss, verbatim
    "«peut-être» accentué",
    "unknown mangle «email_999» passes through",
    "unclosed mangle «email_00",
]


def _expected(vault: InMemoryVault, text: str, fuzzy: bool = False) -> str:
    return Rehydrator(vault, fuzzy=fuzzy).rehydrate_text(text)


@pytest.mark.parametrize("fuzzy", [False, True])
@pytest.mark.parametrize("text", CORPUS)
def test_two_way_split_at_every_offset(loaded_vault: InMemoryVault, text: str, fuzzy: bool) -> None:
    expected = _expected(loaded_vault, text, fuzzy)
    for cut in range(len(text) + 1):
        r = StreamingRehydrator(loaded_vault, fuzzy=fuzzy)
        out = r.feed(text[:cut]) + r.feed(text[cut:]) + r.flush()
        assert out == expected, f"failed at cut={cut} fuzzy={fuzzy}"


@pytest.mark.parametrize("fuzzy", [False, True])
@pytest.mark.parametrize(
    "text",
    ["x «EMAIL_001» y", "«AWS_KEY_001»«EMAIL_001»", "x « email-1 » y", "«email_001»«aws-key-1»"],
)
def test_three_way_split_sweep(loaded_vault: InMemoryVault, text: str, fuzzy: bool) -> None:
    expected = _expected(loaded_vault, text, fuzzy)
    for a in range(len(text) + 1):
        for b in range(a, len(text) + 1):
            r = StreamingRehydrator(loaded_vault, fuzzy=fuzzy)
            out = r.feed(text[:a]) + r.feed(text[a:b]) + r.feed(text[b:]) + r.flush()
            assert out == expected, f"failed at cuts=({a},{b}) fuzzy={fuzzy}"


def test_fuzzy_restores_mangles_and_exact_does_not(loaded_vault: InMemoryVault) -> None:
    text = "mail «email-1» and key « AWS_KEY_001 »"
    assert (
        _expected(loaded_vault, text, fuzzy=True)
        == "mail jane@corp.example and key AKIAIOSFODNN7EXAMPLE"
    )
    assert _expected(loaded_vault, text, fuzzy=False) == text  # byte-identical


def test_fuzzy_vault_gate_never_restores_unissued(loaded_vault: InMemoryVault) -> None:
    # «top-10» canonicalizes to «TOP_010» but no TOP token was ever issued.
    text = "les «top-10» mondiaux et «email_999»"
    assert _expected(loaded_vault, text, fuzzy=True) == text


def test_fuzzy_lowercase_holdback_bounded(loaded_vault: InMemoryVault) -> None:
    r = StreamingRehydrator(loaded_vault, fuzzy=True)
    emitted = r.feed("«" + "a" * MAX_PLACEHOLDER_LEN)
    assert emitted.startswith("«aaa")
    assert r.flush() == ""


def test_char_by_char_feed(loaded_vault: InMemoryVault) -> None:
    text = "start «EMAIL_001» middle «AWS_KEY_001» end"
    r = StreamingRehydrator(loaded_vault)
    out = "".join(r.feed(ch) for ch in text) + r.flush()
    assert out == _expected(loaded_vault, text)


def test_holdback_is_bounded(loaded_vault: InMemoryVault) -> None:
    r = StreamingRehydrator(loaded_vault)
    # « followed by token-body chars beyond the max length must be released.
    emitted = r.feed("«" + "A" * MAX_PLACEHOLDER_LEN)
    assert emitted.startswith("«AAA")
    assert r.flush() == ""


def test_ordinary_text_not_delayed(loaded_vault: InMemoryVault) -> None:
    r = StreamingRehydrator(loaded_vault)
    assert r.feed("hello world") == "hello world"


def test_json_source_mode_escapes_correctly(loaded_vault: InMemoryVault) -> None:
    # Simulate streamed tool-call arguments containing a placeholder whose
    # original has quotes, newlines, and backslashes.
    argument_stream = ['{"secret": "«SECR', 'ET_001»", "n": 1}']
    r = StreamingRehydrator(loaded_vault, json_source=True)
    reassembled = "".join(r.feed(part) for part in argument_stream) + r.flush()
    parsed = json.loads(reassembled)
    assert parsed == {"secret": 'pa"ss\nwor\\d', "n": 1}


def test_non_json_mode_splices_raw(loaded_vault: InMemoryVault) -> None:
    r = StreamingRehydrator(loaded_vault)
    out = r.feed("«SECRET_001»") + r.flush()
    assert out == 'pa"ss\nwor\\d'


def test_pool_channels_are_independent(loaded_vault: InMemoryVault) -> None:
    pool = RehydratorPool(loaded_vault)
    a = pool.get(("text", 0))
    b = pool.get(("text", 1))
    assert a is not b
    assert pool.get(("text", 0)) is a
    a.feed("«EMA")
    assert pool.flush(("text", 0)) == "«EMA"
    assert pool.flush(("text", 0)) == ""  # flushing again is empty


def test_pool_flush_matching(loaded_vault: InMemoryVault) -> None:
    pool = RehydratorPool(loaded_vault)
    assert pool.get((0, "content")).feed("held «EM") == "held "
    assert pool.get((1, "content")).feed("other «AW") == "other "
    flushed = pool.flush_matching(lambda k: isinstance(k, tuple) and k[0] == 0)
    assert flushed == {(0, "content"): "«EM"}
    remaining = pool.flush_all()
    assert remaining == {(1, "content"): "«AW"}


# ---- \uXXXX-escaped guillemets in json_source streams ----

# Each entry is raw JSON source as it appears on the wire inside tool-call
# argument deltas. Escaped guillemets («=«, »=») must be recognized
# across any chunk split; literal-backslash decoys must never be rewritten.
JSON_SOURCE_CORPUS = [
    '{"to": "\\u00abEMAIL_001\\u00bb"}',  # fully escaped token
    '{"to": "\\u00ABEMAIL_001\\u00BB"}',  # uppercase hex
    '{"to": "\\u00abEMAIL_001»"}',  # mixed escaped/raw
    '{"to": "«EMAIL_001\\u00bb"}',  # raw open, escaped close
    '{"m": "\\u00abemail_001\\u00bb"}',  # escaped + mangled body
    '{"secret": "\\u00abSECRET_001\\u00bb"}',  # nasty original re-escaped
    '{"path": "C:\\\\u00ab"}',  # literal \\ decoy: not an escape
    '{"x": "\\u00abMYSTERY_042\\u00bb"}',  # unknown token: verbatim body
    '{"p": "C:\\\\temp\\\\new"}',  # ordinary backslashes
    '{"n": "\\u00b5m"}',  # µ (micro): not a guillemet
    '{"end": "trailing backslash \\\\"}',
]


@pytest.mark.parametrize("fuzzy", [False, True])
@pytest.mark.parametrize("source", JSON_SOURCE_CORPUS)
def test_json_source_escape_split_sweep(
    loaded_vault: InMemoryVault, source: str, fuzzy: bool
) -> None:
    expected = Rehydrator(loaded_vault, fuzzy=fuzzy).rehydrate_json_source_text(source)
    # The reassembled stream must remain valid JSON source.
    json.loads(expected)
    for cut in range(len(source) + 1):
        r = StreamingRehydrator(loaded_vault, json_source=True, fuzzy=fuzzy)
        out = r.feed(source[:cut]) + r.feed(source[cut:]) + r.flush()
        assert out == expected, f"failed at cut={cut} fuzzy={fuzzy}"


def test_json_source_escaped_token_restores(loaded_vault: InMemoryVault) -> None:
    r = Rehydrator(loaded_vault)
    out = r.rehydrate_json_source_text('{"to": "\\u00abEMAIL_001\\u00bb"}')
    assert json.loads(out) == {"to": "jane@corp.example"}


def test_json_source_literal_backslash_decoy_untouched(loaded_vault: InMemoryVault) -> None:
    source = '{"path": "C:\\\\u00abEMAIL_001\\u00bb"}'
    out = Rehydrator(loaded_vault).rehydrate_json_source_text(source)
    # The \\u00ab is literal text (escaped backslash), not a guillemet: the
    # opening escape must not be rewritten, so no token can form before it...
    assert json.loads(out)["path"].startswith("C:\\u00ab")


def test_json_source_char_by_char(loaded_vault: InMemoryVault) -> None:
    source = '{"a": "\\u00abEMAIL_001\\u00bb", "p": "C:\\\\u00ab", "s": "\\u00abSECRET_001\\u00bb"}'
    expected = Rehydrator(loaded_vault).rehydrate_json_source_text(source)
    r = StreamingRehydrator(loaded_vault, json_source=True)
    out = "".join(r.feed(ch) for ch in source) + r.flush()
    assert out == expected
    assert json.loads(out)["a"] == "jane@corp.example"
    assert json.loads(out)["s"] == 'pa"ss\nwor\\d'
