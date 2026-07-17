from typing import Any

from llm_redact.jsonwalk import transform_strings


def upper(s: str) -> str:
    return s.upper()


def test_nested_structures_transformed() -> None:
    obj: Any = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "result body"}
                ],
            },
        ],
        "system": "be nice",
    }
    out = transform_strings(obj, upper)
    assert out["system"] == "BE NICE"
    assert out["messages"][0]["content"][0]["text"] == "HELLO"
    assert out["messages"][1]["content"][0]["content"] == "RESULT BODY"


def test_structural_keys_untouched() -> None:
    obj = {
        "model": "claude-sonnet-4-5",
        "role": "user",
        "type": "image",
        "name": "get_weather",
        "data": "aGVsbG8=",
        "text": "real content",
    }
    out = transform_strings(obj, upper)
    assert out["model"] == "claude-sonnet-4-5"
    assert out["role"] == "user"
    assert out["name"] == "get_weather"
    assert out["data"] == "aGVsbG8="
    assert out["text"] == "REAL CONTENT"


def test_non_string_scalars_pass_through() -> None:
    obj = {"max_tokens": 100, "stream": True, "temperature": 0.5, "stop": None}
    assert transform_strings(obj, upper) == obj


def test_top_level_list_and_string() -> None:
    assert transform_strings(["a", {"text": "b"}], upper) == ["A", {"text": "B"}]
    assert transform_strings("plain", upper) == "PLAIN"


def test_plaintext_document_data_is_transformed() -> None:
    # Anthropic plaintext documents: source.type == "text" carries the full
    # document in `data` — it must NOT be skipped like base64 media.
    block = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": "call jane"},
    }
    out = transform_strings(block, upper)
    assert out["source"]["data"] == "CALL JANE"


def test_base64_document_data_still_skipped() -> None:
    block = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "aGVsbG8="},
    }
    out = transform_strings(block, upper)
    assert out["source"]["data"] == "aGVsbG8="
