"""AWS Signature Version 4 for the core's own AWS calls (the Textract
extraction service). Stdlib ``hmac``/``hashlib`` only; pinned by the
official AWS documentation vector (tests/test_sigv4.py).

Canonicalization follows the published SigV4 rules: the query's pairs
decoded, re-encoded per RFC 3986 and sorted; header names lower-cased,
values trimmed with runs of spaces collapsed; the path (wire form)
URI-encoded once more (every service but S3, which this module does not
sign for). Nothing here logs; a key or token never leaves the returned
headers.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

ALGORITHM = "AWS4-HMAC-SHA256"
_UNRESERVED = "-_.~"
_SPACES = re.compile(r" +")
# Signing headers this module owns: a caller copy is dropped, never doubled.
_OWNED_HEADERS = frozenset({"authorization", "x-amz-date", "x-amz-security-token"})


@dataclass(frozen=True)
class AwsCredentials:
    """An access key pair and an optional session token (never printed)."""

    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    session_token: str | None = field(default=None, repr=False)


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _encode(text: str) -> str:
    return urllib.parse.quote(text, safe=_UNRESERVED)


def canonical_query(query: str) -> str:
    """The canonical query string of a raw (wire-form) query."""
    pairs = []
    for part in query.split("&"):
        if part:
            key, _, value = part.partition("=")
            pairs.append((_encode(urllib.parse.unquote(key)), _encode(urllib.parse.unquote(value))))
    return "&".join(f"{key}={value}" for key, value in sorted(pairs))


def sign_request(
    method: str,
    url: str,
    headers: Sequence[tuple[str, str]],
    body: bytes,
    *,
    region: str,
    service: str,
    credentials: AwsCredentials,
    now: datetime | None = None,
) -> list[tuple[str, str]]:
    """``headers`` plus ``host`` (when absent), ``x-amz-date``,
    ``x-amz-security-token`` (temporary credentials) and ``authorization``;
    every header in the result is signed."""
    split = urllib.parse.urlsplit(url)
    moment = (now or datetime.now(tz=UTC)).astimezone(UTC)
    amz_date = moment.strftime("%Y%m%dT%H%M%SZ")
    date = amz_date[:8]
    out = [(name, value) for name, value in headers if name.strip().lower() not in _OWNED_HEADERS]
    if not any(name.strip().lower() == "host" for name, _ in out):
        out.append(("host", split.netloc.rpartition("@")[2]))
    out.append(("x-amz-date", amz_date))
    if credentials.session_token:
        out.append(("x-amz-security-token", credentials.session_token))
    merged: dict[str, list[str]] = {}
    for name, value in out:
        merged.setdefault(name.strip().lower(), []).append(_SPACES.sub(" ", value.strip()))
    names = sorted(merged)
    signed_headers = ";".join(names)
    canonical_request = "\n".join(
        [
            method.upper(),
            urllib.parse.quote(split.path or "/", safe="/" + _UNRESERVED),
            canonical_query(split.query),
            "".join(f"{name}:{','.join(merged[name])}\n" for name in names),
            signed_headers,
            hashlib.sha256(body).hexdigest(),
        ]
    )
    scope = f"{date}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [ALGORITHM, amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest()]
    )
    key = _hmac(("AWS4" + credentials.secret_access_key).encode("utf-8"), date)
    for part in (region, service, "aws4_request"):
        key = _hmac(key, part)
    signature = hmac.new(key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    out.append(
        (
            "authorization",
            f"{ALGORITHM} Credential={credentials.access_key_id}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}",
        )
    )
    return out
