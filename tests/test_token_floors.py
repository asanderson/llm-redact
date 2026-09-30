"""Token floors: the per-type highest placeholder number a request carries.

A request can carry tokens its session never issued (a compacted history
forked into a fresh session, an answer pasted from another conversation).
The redactor numbers every NEW value above those, so one token name never
means two values upstream. These pin the scan (every form the rehydrator
could restore — canonical, fuzzy-mangled, JSON-escaped), the byte gate in
front of it (never False when a decoded body could hold a guillemet), the
Redactor's per-request copy, and the adapter-level floor sources the proxy
cannot see into (multipart parts, the Bedrock count-tokens blob).
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from fake_cipher import FakeVaultCipher
from llm_redact import multipart
from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.placeholders import (
    MAX_TOKEN_NUMBER,
    json_floors,
    may_carry_tokens,
    merge_floors,
    token_floors,
)
from llm_redact.providers.bedrock import BedrockAdapter
from llm_redact.providers.openai import OpenAIAdapter, _multipart_floors, _part_kind, _read_part
from llm_redact.redactor import PlaceholderLimitReached, Redactor, UnredactableRequest
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import EncryptedInMemoryVault, InMemoryVault

NBSP = "\xa0"
ESC_OPEN = "\\u00ab"  # a JSON escape of «, as literal text
ESC_CLOSE = "\\u00bb"


def _redactor(vault: Any = None) -> Redactor:
    config = DetectionConfig()
    return Redactor(
        build_detectors(config),
        vault if vault is not None else InMemoryVault(),
        build_allowlist(config),
    )


# --- the scan ---------------------------------------------------------------


def test_no_token_no_floor() -> None:
    assert token_floors("") == {}
    assert token_floors("plain prose, no guillemets") == {}
    assert token_floors("French «bonjour» and «EMAIL» without a number") == {}
    assert token_floors("an unclosed «EMAIL_004 token") == {}


def test_canonical_tokens_per_type_maximum() -> None:
    text = "«EMAIL_001» then «EMAIL_003» and «PERSON_002», again «EMAIL_002»"
    assert token_floors(text) == {"EMAIL": 3, "PERSON": 2}


@pytest.mark.parametrize(
    ("mangle", "expected"),
    [
        ("«email_2»", {"EMAIL": 2}),  # case shift, altered padding
        ("«EMAIL-7»", {"EMAIL": 7}),  # hyphen for underscore
        ("«Credit-Card-0004»", {"CREDIT_CARD": 4}),  # hyphens throughout
        ("«EMAIL_00012»", {"EMAIL": 12}),  # extra zero-padding
        (f"« EMAIL_5{NBSP}»", {"EMAIL": 5}),  # pad space and NBSP
        ("«  EMAIL_6  »", {"EMAIL": 6}),  # two pads each side
        ("«EMAIL__008»", {"EMAIL_": 8}),  # the canonical split: last separator
        ("«A_1_2»", {"A_1": 2}),
    ],
)
def test_fuzzy_mangles_count(mangle: str, expected: dict[str, int]) -> None:
    assert token_floors(f"echo {mangle} back") == expected


def test_three_pads_are_not_a_token() -> None:
    # Beyond the fuzzy grammar: the rehydrator never restores it either.
    assert token_floors("«   EMAIL_5»") == {}


def test_json_escaped_guillemets_count_whatever_the_parity() -> None:
    # A tool call's JSON-source arguments restore «EMAIL_009» written with
    # escaped guillemets; either hex case counts.
    assert token_floors(f"{ESC_OPEN}EMAIL_009{ESC_CLOSE}") == {"EMAIL": 9}
    assert token_floors("\\u00ABEMAIL_010\\u00BB") == {"EMAIL": 10}
    assert token_floors("\\u00aBEMAIL_011\\u00Bb") == {"EMAIL": 11}
    # Mixed: a raw opening guillemet, an escaped closing one.
    assert token_floors(f"«EMAIL_012{ESC_CLOSE}") == {"EMAIL": 12}
    # An escaped backslash before the escape (literal text): still counted
    # — a superset only ever raises a floor.
    assert token_floors(f"\\{ESC_OPEN}EMAIL_013{ESC_CLOSE}") == {"EMAIL": 13}
    # Other escapes are left alone.
    assert token_floors("\\u00a0EMAIL_014\\u00a0") == {}


def test_number_zero_and_numbers_past_the_limit_raise_nothing() -> None:
    # 0 is never issued; a tenth significant digit is past MAX_TOKEN_NUMBER,
    # which no vault issues, so it can never collide.
    assert token_floors("«EMAIL_000»") == {}
    assert token_floors("«EMAIL_1234567890»") == {}
    assert token_floors(f"«EMAIL_{MAX_TOKEN_NUMBER}»") == {"EMAIL": MAX_TOKEN_NUMBER}
    # Leading zeros are not significant digits.
    assert token_floors("«EMAIL_0000000000005»") == {"EMAIL": 5}


def test_long_type_names_count() -> None:
    # PLACEHOLDER_RE has no type-length bound (a long custom rule type is
    # restored in non-fuzzy mode), so the scan has none either.
    long_type = "A_VERY_LONG_CUSTOM_DETECTOR_TYPE_NAME_PAST_THIRTY"
    assert token_floors(f"«{long_type}_001»") == {long_type: 1}


def test_adjacent_and_nested_guillemets() -> None:
    assert token_floors("««EMAIL_004»»«PHONE_2»") == {"EMAIL": 4, "PHONE": 2}
    assert token_floors("«EMAIL_004 «PHONE_2»") == {"PHONE": 2}


def test_json_floors_reads_every_key_and_string() -> None:
    body = {
        "model": "«MODEL_007»",  # a structural scalar the redactor skips
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi «EMAIL_003»"}]},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "send",
                            # JSON-source arguments, guillemets escaped inside
                            "arguments": json.dumps({"to": "«EMAIL_005»"}),
                        }
                    }
                ],
            },
        ],
        "«KEY_002»": [1, 2.5, None, True, "«email-4»"],
    }
    assert json_floors(body) == {"MODEL": 7, "EMAIL": 5, "KEY": 2}
    assert json_floors("«EMAIL_001»") == {"EMAIL": 1}
    assert json_floors(12) == {}
    assert json_floors(None) == {}


def test_merge_floors_only_raises() -> None:
    into = {"EMAIL": 4, "PHONE": 2}
    merge_floors(into, {"EMAIL": 3, "PHONE": 5, "IBAN": 1, "PERSON": 0})
    assert into == {"EMAIL": 4, "PHONE": 5, "IBAN": 1}


# --- the gate -----------------------------------------------------------------


def test_gate_bytes() -> None:
    body = {"content": "mail «EMAIL_001»"}
    assert may_carry_tokens(json.dumps(body, ensure_ascii=False).encode())  # raw UTF-8
    assert may_carry_tokens(json.dumps(body).encode())  # « escapes
    assert may_carry_tokens(json.dumps(body, ensure_ascii=False).encode("utf-16"))
    assert may_carry_tokens(json.dumps(body, ensure_ascii=False).encode("utf-32-le"))
    assert may_carry_tokens(b"filename*=UTF-8''%C2%ABEMAIL_001%C2%BB")
    assert may_carry_tokens(b"filename*=UTF-8''%c2%abEMAIL_001%c2%bb")
    assert not may_carry_tokens(b'{"content": "plain\\n", "note": "41.5% of traffic"}')
    assert not may_carry_tokens(b"")
    # Conservative on purpose: ANY \u00XX escape (an ANSI \u001b in a tool
    # output) sends the body to the decoded walk, which decides exactly.
    assert may_carry_tokens(b'{"content": "\\u001b[0m done"}')
    # UTF-16/32 text hides an ASCII escape from a byte search: a NUL byte
    # (never in UTF-8 JSON) always passes. Here a JSON-source argument
    # string escapes its guillemets, and no 0xAB byte exists anywhere.
    source = {"arguments": json.dumps({"to": "«EMAIL_001»"})}
    utf16 = json.dumps(source, ensure_ascii=False).encode("utf-16")
    assert b"\xab" not in utf16 and b"\\u00" not in utf16
    assert may_carry_tokens(utf16)
    assert json_floors(json.loads(utf16)) == {"EMAIL": 1}


def test_gate_text() -> None:
    assert may_carry_tokens('{"type": "x", "text": "«EMAIL_001»"}')
    assert may_carry_tokens('{"text": "\\u00abEMAIL_001\\u00bb"}')
    assert may_carry_tokens("%ABEMAIL")
    assert not may_carry_tokens('{"type": "response.audio.delta", "delta": "UklGRg=="}')


# --- the redactor ---------------------------------------------------------------


def test_with_floors_is_a_copy_only_when_a_floor_rises() -> None:
    shared = _redactor()
    assert shared.with_floors({}) is shared
    assert shared.with_floors({"EMAIL": 0}) is shared
    floored = shared.with_floors({"EMAIL": 3})
    assert floored is not shared
    # Counters are shared (per-request attribution diffs the process total).
    assert floored.counts is shared.counts
    assert floored.warn_counts is shared.warn_counts
    # A floor at or below the copy's own adds nothing.
    assert floored.with_floors({"EMAIL": 2}) is floored
    assert floored.with_floors({"EMAIL": 3}) is floored
    raised = floored.with_floors({"EMAIL": 2, "PHONE": 1})
    assert raised is not floored
    assert raised.redact_text("a@corp.example +1 415 555 0100") == "«EMAIL_004» «PHONE_002»"


def test_the_shared_redactor_is_never_floored_by_a_request() -> None:
    vault = InMemoryVault()
    shared = _redactor(vault)
    floored = shared.with_floors({"EMAIL": 6})
    assert floored.redact_text("a@corp.example") == "«EMAIL_007»"
    # The next request with nothing to raise it continues from MAX(n) —
    # the floor did not stick to the shared redactor.
    assert shared.redact_text("b@corp.example") == "«EMAIL_008»"
    assert shared.with_floors({"EMAIL": 2}).redact_text("c@corp.example") == "«EMAIL_009»"


def test_a_mapped_value_keeps_its_token_under_a_floor() -> None:
    redactor = _redactor()
    assert redactor.redact_text("a@corp.example") == "«EMAIL_001»"
    assert redactor.with_floors({"EMAIL": 5}).redact_text("a@corp.example") == "«EMAIL_001»"


def test_warn_and_block_modes_issue_nothing_whatever_the_floor() -> None:
    config = DetectionConfig()
    vault = InMemoryVault()
    warn = Redactor(
        build_detectors(config), vault, build_allowlist(config), modes={"EMAIL": "warn"}
    ).with_floors({"EMAIL": 4})
    assert warn.redact_text("a@corp.example") == "a@corp.example"
    assert len(vault) == 0


def test_the_limit_refuses_the_request_naming_the_type_only() -> None:
    redactor = _redactor().with_floors({"EMAIL": MAX_TOKEN_NUMBER})
    with pytest.raises(PlaceholderLimitReached) as refused:
        redactor.redact_text("mail bob@corp.example")
    assert isinstance(refused.value, UnredactableRequest)  # every refusal path handles it
    message = str(refused.value)
    assert message.startswith("llm-redact: no EMAIL placeholder number")
    assert "the request was not forwarded" in message
    assert "bob" not in message
    # Other types are unaffected; a floor at the limit with nothing new passes.
    assert redactor.redact_text("no values here") == "no values here"


def test_redact_json_uses_the_floor_everywhere_in_the_walk() -> None:
    redactor = _redactor(EncryptedInMemoryVault(FakeVaultCipher(), "s")).with_floors({"EMAIL": 2})
    body = {"system": "a@corp.example", "messages": [{"content": "b@corp.example"}]}
    assert redactor.redact_json(body) == {
        "system": "«EMAIL_003»",
        "messages": [{"content": "«EMAIL_004»"}],
    }


# --- the property the floor exists for -----------------------------------------------


def test_a_fork_never_gives_a_foreign_token_a_second_meaning() -> None:
    # The compaction fork: a fresh session whose first message is a summary
    # quoting tokens of the ORIGINAL session. New values are numbered above
    # them; the foreign tokens keep resolving to nothing, so an echo of one
    # passes through verbatim instead of restoring the new value.
    vault = InMemoryVault()
    body = {
        "messages": [
            {"role": "user", "content": "Summary: «EMAIL_001» wrote to «EMAIL_003» («email_2»)."},
            {"role": "user", "content": "now also mail bob@corp.example"},
        ]
    }
    redacted = _redactor(vault).with_floors(json_floors(body)).redact_json(body)
    assert redacted["messages"][1]["content"] == "now also mail «EMAIL_004»"
    rehydrator = Rehydrator(vault, fuzzy=True)
    echo = "«EMAIL_001», «email_2», «EMAIL_003» and «EMAIL_004»"
    assert rehydrator.rehydrate_text(echo) == (
        "«EMAIL_001», «email_2», «EMAIL_003» and bob@corp.example"
    )


# --- adapter-level floor sources ------------------------------------------------------

BOUNDARY = b"floorboundary"


def _upload(*parts: bytes) -> bytes:
    body = b"".join(b"--floorboundary\r\n" + part + b"\r\n" for part in parts)
    return body + b"--floorboundary--\r\n"


def _file_part(content: bytes, disposition: bytes = b'filename="in.jsonl"') -> bytes:
    return (
        b'Content-Disposition: form-data; name="file"; '
        + disposition
        + b"\r\nContent-Type: application/jsonl\r\n\r\n"
        + content
    )


def _jsonl(*objs: dict[str, Any], ensure_ascii: bool = False) -> bytes:
    return b"\n".join(json.dumps(obj, ensure_ascii=ensure_ascii).encode() for obj in objs) + b"\n"


def _line(text: str) -> dict[str, Any]:
    return {"custom_id": "c", "body": {"messages": [{"role": "user", "content": text}]}}


def _redacted_lines(out: bytes | None) -> list[str]:
    assert out is not None
    parsed = multipart.parse(out, BOUNDARY)
    assert parsed is not None
    return [
        json.loads(line)["body"]["messages"][-1]["content"]
        for part in parsed.parts
        if part.filename is not None
        for line in part.content.split(b"\n")
        if line.strip()
    ]


def test_a_later_jsonl_line_bounds_an_earlier_lines_number() -> None:
    body = _upload(_file_part(_jsonl(_line("mail bob@corp.example"), _line("history «EMAIL_007»"))))
    out = OpenAIAdapter().redact_multipart(
        "/v1/files", body, BOUNDARY, _redactor(), inject_note=False
    )
    assert _redacted_lines(out) == ["mail «EMAIL_008»", "history «EMAIL_007»"]


def test_jsonl_floors_read_the_decoded_line() -> None:
    # A Python-default (ensure_ascii) JSONL line: escaped guillemets and an
    # escaped NBSP pad — only the decoded string shows the fuzzy token.
    history = _line(f"history «{NBSP}EMAIL_011{NBSP}»")
    body = _upload(_file_part(_jsonl(_line("mail bob@corp.example"), history, ensure_ascii=True)))
    assert b"\\u00a0" in body and b"\xc2\xab" not in body
    out = OpenAIAdapter().redact_multipart(
        "/v1/files", body, BOUNDARY, _redactor(), inject_note=False
    )
    assert _redacted_lines(out)[0] == "mail «EMAIL_012»"


def test_a_percent_encoded_file_name_token_bounds_the_upload() -> None:
    disposition = b"filename*=UTF-8''%C2%ABEMAIL_009%C2%BB.jsonl"
    body = _upload(_file_part(_jsonl(_line("mail bob@corp.example")), disposition))
    out = OpenAIAdapter().redact_multipart(
        "/v1/files", body, BOUNDARY, _redactor(), inject_note=False
    )
    assert _redacted_lines(out) == ["mail «EMAIL_010»"]


def test_multipart_floors_read_text_fields_and_skip_media() -> None:
    parsed = multipart.parse(
        _upload(
            'Content-Disposition: form-data; name="prompt"\r\n\r\nedit «EMAIL_003» please'.encode(),
            'Content-Disposition: form-data; name="purpose"\r\n\r\n«PURPOSE_2»'.encode(),
            b'Content-Disposition: form-data; name="image"; filename="a.png"\r\n'
            b"Content-Type: image/png\r\n\r\n\x89PNG " + "«EMAIL_050»".encode() + b" \xab\xff",
            _file_part("not json «PHONE_4»\n\n".encode() + _jsonl(_line("«EMAIL_006»"))),
        ),
        BOUNDARY,
    )
    assert parsed is not None

    def floors(media: bool) -> dict[str, int]:
        readings = [
            _read_part(part, media=media, require_scanned=True, charge=lambda n: None)
            for part in parsed.parts
        ]
        return _multipart_floors(parsed, readings)

    # Media route: the image file part is never read; the prompt field is.
    assert floors(media=True) == {"EMAIL": 3, "PURPOSE": 2}
    # Upload route: a binary file part is read as UTF-8 (the upstream may
    # read it as text), a text file (here: prose, then a JSON line) as the
    # text it decodes to.
    assert floors(media=False) == {"EMAIL": 50, "PURPOSE": 2, "PHONE": 4}


def test_multipart_floors_read_each_file_as_the_text_it_is() -> None:
    utf16 = "history «EMAIL_021»\n".encode("utf-16-le")
    escaped = json.dumps(_line("«PHONE_033»"), ensure_ascii=True).encode()
    parsed = multipart.parse(
        _upload(
            _file_part(b"\xff\xfe" + utf16),  # a UTF-16 text file
            _file_part(escaped + b"\n"),  # JSONL: the line's decoded JSON
            _file_part(b"prose\n" + escaped),  # text: read as text
        ),
        BOUNDARY,
    )
    assert parsed is not None
    readings = [
        _read_part(part, media=False, require_scanned=True, charge=lambda n: None)
        for part in parsed.parts
    ]
    assert [reading.kind for reading in readings] == ["document", "jsonl", "document"]
    assert _multipart_floors(parsed, readings) == {"EMAIL": 21, "PHONE": 33}


def test_part_kind() -> None:
    parsed = multipart.parse(
        _upload(
            b'Content-Disposition: form-data; name="prompt"; filename="p.txt"\r\n\r\nx',
            b'Content-Disposition: form-data; name="image"; filename="a.png"\r\n\r\nx',
            b'Content-Disposition: form-data; name="purpose"\r\n\r\nbatch',
        ),
        BOUNDARY,
    )
    assert parsed is not None
    prompt, image, purpose = parsed.parts
    assert _part_kind(prompt, media=True, require_scanned=False) == "text"
    assert _part_kind(prompt, media=False, require_scanned=False) == "file"
    assert _part_kind(image, media=True, require_scanned=False) == "media"
    assert _part_kind(purpose, media=False, require_scanned=False) == "field"
    assert _part_kind(purpose, media=False, require_scanned=True) == "text"


def test_an_upload_without_any_guillemet_skips_the_floor_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_redact.providers.openai as openai_mod

    def never(*args: Any, **kwargs: Any) -> dict[str, int]:
        raise AssertionError("the gate must skip the scan")

    monkeypatch.setattr(openai_mod, "_multipart_floors", never)
    body = _upload(_file_part(_jsonl(_line("mail bob@corp.example"))))
    out = OpenAIAdapter().redact_multipart(
        "/v1/files", body, BOUNDARY, _redactor(), inject_note=False
    )
    assert _redacted_lines(out) == ["mail «EMAIL_001»"]


def test_bedrock_count_tokens_floors_include_the_decoded_blob() -> None:
    # CountTokens carries the model body base64-encoded: the proxy's floor
    # (over the envelope) cannot see the tokens inside it.
    inner = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role": "user", "content": "mail bob@corp.example"},
            {"role": "assistant", "content": "noted «EMAIL_005»"},
        ],
    }
    envelope = {
        "input": {"invokeModel": {"body": base64.b64encode(json.dumps(inner).encode()).decode()}}
    }
    assert not may_carry_tokens(json.dumps(envelope).encode())
    out = BedrockAdapter().prepare_request(envelope, _redactor(), inject_note=False)
    decoded = json.loads(base64.b64decode(out["input"]["invokeModel"]["body"]))
    assert decoded["messages"][0]["content"] == "mail «EMAIL_006»"
    assert decoded["messages"][1]["content"] == "noted «EMAIL_005»"
