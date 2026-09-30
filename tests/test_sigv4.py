"""The core's SigV4 signer (``sigv4.py``, the Textract extraction service),
pinned by the official AWS documentation example (IAM ListUsers)."""

from __future__ import annotations

from datetime import UTC, datetime

from llm_redact.sigv4 import AwsCredentials, canonical_query, sign_request

DOC_CREDS = AwsCredentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
DOC_NOW = datetime(2015, 8, 30, 12, 36, 0, tzinfo=UTC)


def _auth(headers: list[tuple[str, str]]) -> str:
    (value,) = [value for name, value in headers if name == "authorization"]
    return value


def test_iam_list_users_documentation_vector() -> None:
    headers = sign_request(
        "GET",
        "https://iam.amazonaws.com/?Action=ListUsers&Version=2010-05-08",
        [("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")],
        b"",
        region="us-east-1",
        service="iam",
        credentials=DOC_CREDS,
        now=DOC_NOW,
    )
    assert _auth(headers) == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request, "
        "SignedHeaders=content-type;host;x-amz-date, "
        "Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7"
    )
    assert ("x-amz-date", "20150830T123600Z") in headers
    assert ("host", "iam.amazonaws.com") in headers


def test_a_session_token_is_signed_and_caller_signing_headers_are_replaced() -> None:
    creds = AwsCredentials("AKID", "secret", "session-token")
    headers = sign_request(
        "POST",
        "https://textract.us-east-1.amazonaws.com/",
        [("Authorization", "stale"), ("X-Amz-Date", "old"), ("x-amz-target", "T.Op")],
        b"{}",
        region="us-east-1",
        service="textract",
        credentials=creds,
        now=DOC_NOW,
    )
    names = [name.lower() for name, _ in headers]
    assert names.count("authorization") == 1 and names.count("x-amz-date") == 1
    assert ("x-amz-security-token", "session-token") in headers
    assert "SignedHeaders=host;x-amz-date;x-amz-security-token;x-amz-target," in _auth(headers)
    assert "secret" not in _auth(headers) and "secret" not in repr(creds)


def test_canonical_query_sorts_and_encodes() -> None:
    assert canonical_query("b=2&a=1&&c=%7E x") == "a=1&b=2&c=~%20x"
