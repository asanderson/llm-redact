"""Named validators for user custom rules.

A `[[detection.custom_rules]]` entry may set `validator = "..."` to gate its
regex with a checksum/format check: the rule fires only when the pattern
matches AND the named validator passes on the whole match. This is the recipe
the built-in card/national-id rules use — a loose pattern (few false
negatives) backed by a checksum (few false positives) — made available to
users without writing Python.

Each validator maps `re.Match` → bool over `match.group(0)`. Unknown names are
a build-time error (see `build_detectors`), listing the valid set.
"""

import base64
import binascii
import json
import math
import re
from collections import Counter
from collections.abc import Callable

from llm_redact.detection.regex_rules import _VERHOEFF_D, _VERHOEFF_P, _luhn_checksum


def _digits(match: re.Match[str]) -> list[int]:
    return [int(c) for c in match.group(0) if c.isdigit()]


def _luhn(match: re.Match[str]) -> bool:
    digits = _digits(match)
    # Reject the all-zero string: it satisfies the arithmetic but is the
    # canonical placeholder shape, not a real number.
    return len(digits) >= 2 and any(digits) and _luhn_checksum(digits) == 0


def _verhoeff(match: re.Match[str]) -> bool:
    digits = _digits(match)
    if not digits:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[i % 8][d]]
    return checksum == 0


def _to_mod97_number(alnum: list[str]) -> int | None:
    try:
        return int("".join(str(int(c, 36)) if c.isalpha() else c for c in alnum))
    except ValueError:
        return None


def _mod97(match: re.Match[str]) -> bool:
    """ISO 7064 MOD-97-10, remainder 1 (letters A=10 … Z=35). Accepts both
    conventions: check digits appended at the END (plain MOD-97-10) and the
    IBAN layout where the leading 4 chars move to the end before the check."""
    alnum = [c for c in match.group(0).upper() if c.isalnum()]
    if len(alnum) < 4:
        return False
    as_is = _to_mod97_number(alnum)
    rearranged = _to_mod97_number(alnum[4:] + alnum[:4])  # IBAN move-first-4
    return (as_is is not None and as_is % 97 == 1) or (
        rearranged is not None and rearranged % 97 == 1
    )


_JWT_RE = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*\Z")


def _b64url_json_object(segment: str) -> bool:
    try:
        decoded = base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
        obj = json.loads(decoded)
    except (binascii.Error, ValueError):
        return False
    return isinstance(obj, dict)


def _jwt(match: re.Match[str]) -> bool:
    """A well-formed JWT: three base64url segments whose header and payload
    each decode to a JSON object (the signature is opaque)."""
    value = match.group(0)
    if not _JWT_RE.fullmatch(value):
        return False
    header, payload, _signature = value.split(".", 2)
    return _b64url_json_object(header) and _b64url_json_object(payload)


# Shannon bits per character: random secrets sit well above 3.5; ordinary
# words and identifiers sit below it, so this filters prose out of a loose
# high-entropy-token pattern.
_ENTROPY_MIN_BITS = 3.5


def _entropy(match: re.Match[str]) -> bool:
    value = match.group(0)
    # Real high-entropy secrets are a single contiguous token; whitespace is
    # the cheapest, strongest discriminator against high-entropy PROSE (which
    # can reach the bit threshold across many distinct letters + spaces).
    if len(value) < 16 or any(c.isspace() for c in value):
        return False
    counts = Counter(value)
    n = len(value)
    bits = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return bits >= _ENTROPY_MIN_BITS


VALIDATORS: dict[str, Callable[[re.Match[str]], bool]] = {
    "luhn": _luhn,
    "mod97": _mod97,
    "verhoeff": _verhoeff,
    "jwt": _jwt,
    "entropy": _entropy,
}
