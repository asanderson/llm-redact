"""Placeholder token format: «TYPE_NNN».

Guillemet delimiters are chosen because they cannot collide with brackets,
braces, or template syntax that occurs naturally in code and markdown.
"""

import re
from typing import Any

# Longest realistic token, e.g. «CREDIT_CARD_0000001». Streaming holdback is
# bounded by this length: text is never delayed by more than one token.
MAX_PLACEHOLDER_LEN = 40

# The highest number a vault issues for one (session, type). Nine digits is
# the fuzzy grammar's reach (below), so every issued token stays restorable
# in either rehydration mode, and it fits a 32-bit signed INTEGER column
# (PostgreSQL/MySQL). A counter only gets here when a request carries a
# token numbered 999999999; the vault then refuses rather than issue past it.
MAX_TOKEN_NUMBER = 999_999_999

PLACEHOLDER_RE = re.compile(r"«[A-Z][A-Z0-9_]*_\d{3,}»")

# Fuzzy grammar: what an LLM-mangled token can look like. Every recognized
# mangle canonicalizes to a vault lookup, so accepting a shape here never
# causes a false restore on its own — the vault gate decides.
#   - case shifts («email_001»)
#   - hyphens for underscores («EMAIL-001»)
#   - altered zero-padding («EMAIL_1», «EMAIL_0001»)
#   - up to 2 pad spaces/NBSP inside the guillemets (French typography)
# Bracket swaps ([EMAIL_001]) and bare EMAIL_001 are deliberately excluded:
# generated code legitimately contains such identifiers, and restoring one
# would inject a secret into program text.
FUZZY_PLACEHOLDER_RE = re.compile(
    r"«[  ]{0,2}"
    r"(?P<body>[A-Za-z][A-Za-z0-9_-]{0,30}[_-]0*(?P<n>\d{1,9}))"
    r"[  ]{0,2}»"
)

# Characters permitted between « and » in a valid (possibly partial) token.
_BODY_CHAR_RE = re.compile(r"[A-Z0-9_]")

# Prefix language of the fuzzy grammar: what an incomplete mangled token can
# look like before its closing ». A trailing pad is only viable after a digit
# (canonical tokens end in digits), so «bonjour et releases at the space.
_FUZZY_INTERIOR_PREFIX_RE = re.compile(
    r"[  ]{0,2}(?:[A-Za-z][A-Za-z0-9_-]*(?:(?<=\d)[  ]{0,2})?)?\Z"
)


def format_placeholder(type_name: str, n: int) -> str:
    """Render the n-th placeholder for a detector type, e.g. («EMAIL_001»)."""
    return f"«{type_name}_{n:03d}»"


# The longest placeholder TYPE a token can carry: every token the vault can
# issue for it (up to «TYPE_999999999») must fit MAX_PLACEHOLDER_LEN, the
# bound on streaming holdback and the fuzzy grammar's reach. 28 characters.
MAX_TYPE_NAME_LEN = MAX_PLACEHOLDER_LEN - len(format_placeholder("", MAX_TOKEN_NUMBER))

_PLACEHOLDER_TYPE_RE = re.compile(r"[A-Z][A-Z0-9_]*\Z")


def is_placeholder_type(name: str) -> bool:
    """Whether ``name`` fits the placeholder grammar as a token's TYPE: an
    uppercase letter, then uppercase letters, digits and ``_``, at most
    MAX_TYPE_NAME_LEN characters — so every issued token matches
    PLACEHOLDER_RE and stays restorable. A type that does not fit (a model
    label such as ``3D_MODEL``) must never be issued: its tokens could not
    be rehydrated."""
    return len(name) <= MAX_TYPE_NAME_LEN and _PLACEHOLDER_TYPE_RE.match(name) is not None


def canonicalize(matched: str) -> str | None:
    """Reduce a fuzzy-grammar match to canonical «TYPE_NNN» form.

    Returns None when the match cannot be normalized (paranoia guard; the
    grammar should not produce such matches). The caller must still gate on
    a vault lookup — canonicalization alone never authorizes a restore.
    """
    match = FUZZY_PLACEHOLDER_RE.fullmatch(matched)
    if match is None:
        return None
    body = match.group("body").upper().replace("-", "_")
    type_name, _, _digits = body.rpartition("_")
    if not type_name:
        return None
    canonical = format_placeholder(type_name, int(match.group("n")))
    if len(canonical) > MAX_PLACEHOLDER_LEN:
        return None
    return canonical


def viable_prefix_start(text: str, *, fuzzy: bool = False) -> int | None:
    """Return the index of a trailing partial placeholder in ``text``, if any.

    A viable prefix is a final «-initiated run of token-body characters that
    has not yet been closed by » and is still short enough to become a real
    token. Returns None when the tail of ``text`` cannot be a token prefix,
    meaning every character can be emitted safely.

    With ``fuzzy`` the prefix language widens to the mangle grammar
    (lowercase, hyphens, interior pads). Release rules keep the holdback
    bounded: a closing » resolves it, an out-of-language character releases
    the whole run, and MAX_PLACEHOLDER_LEN caps the wait — so French prose
    like «bonjour» is delayed only until its own closing guillemet.
    """
    start = text.rfind("«")
    if start == -1:
        return None
    tail = text[start:]
    if "»" in tail:
        return None
    if len(tail) >= MAX_PLACEHOLDER_LEN:
        return None
    body = tail[1:]
    if fuzzy:
        if _FUZZY_INTERIOR_PREFIX_RE.fullmatch(body) is None:
            return None
        return start
    if body and not all(_BODY_CHAR_RE.fullmatch(ch) for ch in body):
        return None
    return start


# --- token floors -------------------------------------------------------------
#
# Every session numbers its own tokens from 001, so a request can carry a
# token its session never issued (a compacted history forked into a fresh
# session, an answer pasted from another conversation). If the session then
# issued that very name for a NEW value, the upstream would see one token
# with two meanings and an echo of it would restore the new value where the
# model meant the other. The floor is the per-type highest token number a
# request already carries; the vault numbers every new value above it.

# Every token either rehydration mode could restore: a superset of
# PLACEHOLDER_RE and FUZZY_PLACEHOLDER_RE (any type length, any case,
# hyphens, zero-padding, pads). The number keeps the fuzzy grammar's nine
# significant digits: a longer one exceeds MAX_TOKEN_NUMBER, which no vault
# issues, so it can never collide. Over-matching only raises a floor, which
# leaves a gap in the numbering — never a collision.
_FLOOR_TOKEN_RE = re.compile(
    r"«[ \u00a0]{0,2}(?P<type>[A-Za-z][A-Za-z0-9_-]*)[_-]0*(?P<n>\d{1,9})[ \u00a0]{0,2}»"
)

# A guillemet written as a JSON escape. JSON-source channels (tool-call
# arguments) restore «EMAIL_001» spelled \u00abEMAIL_001\u00bb, so the scan
# reads such escapes as guillemets too — whatever the backslash parity,
# which can only add tokens (a conservative superset of the rehydrator's
# parity-aware normalization).
_GUILLEMET_ESCAPE_RE = re.compile(r"\\u00([aAbB])[bB]")

# The byte every encoding of « holds (UTF-8 C2 AB, UTF-16 AB 00 / 00 AB,
# UTF-32 likewise).
_GUILLEMET_BYTE = bytes([ord("«")])

# What an RFC 8187 percent-encoded guillemet leaves in raw bytes: its
# UTF-8 is C2 AB, so a multipart filename* carrying one holds %AB.
_PERCENT_GUILLEMET_BYTES = re.compile(rb"%[aA][bB]")
_PERCENT_GUILLEMET_TEXT = re.compile(r"%[aA][bB]")


def _scan_into(text: str, floors: dict[str, int]) -> None:
    """Raise ``floors`` to every token ``text`` carries (see token_floors)."""
    if "\\u00" in text:
        text = _GUILLEMET_ESCAPE_RE.sub(lambda m: "«" if m.group(1).lower() == "a" else "»", text)
    if "«" not in text:
        return
    for match in _FLOOR_TOKEN_RE.finditer(text):
        type_name = match.group("type").upper().replace("-", "_")
        n = int(match.group("n"))
        if n > floors.get(type_name, 0):
            floors[type_name] = n


def token_floors(text: str) -> dict[str, int]:
    """Per canonical type name (``EMAIL``), the highest placeholder number
    ``text`` carries in any form the rehydrator could restore: canonical,
    fuzzy-mangled (``«email-3»`` counts as EMAIL 3), or with JSON-escaped
    guillemets. Types without a token are absent; linear in ``text``."""
    floors: dict[str, int] = {}
    _scan_into(text, floors)
    return floors


def json_floors(obj: Any) -> dict[str, int]:
    """``token_floors`` over EVERY key and string of a parsed JSON value —
    decoded, so escapes the client used are already resolved — including
    fields the redactor never rewrites (structural scalars, MCP blocks): the
    upstream reads those too."""
    floors: dict[str, int] = {}
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            _scan_into(node, floors)
        elif isinstance(node, dict):
            stack.extend(node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return floors


def merge_floors(into: dict[str, int], found: dict[str, int]) -> None:
    """Raise ``into`` to ``found``, per type (in place)."""
    for type_name, n in found.items():
        if n > into.get(type_name, 0):
            into[type_name] = n


def may_carry_tokens(raw: str | bytes) -> bool:
    """False only when no decoding the proxy applies to request content can
    produce a guillemet: a raw one (its UTF-8, UTF-16 and UTF-32 encodings
    all hold the byte 0xAB), a JSON ``\\u00ab`` escape (JSON-source text's
    own escapes included), or an RFC 8187 ``%AB`` percent escape. Bytes
    holding a NUL are not ASCII-compatible text (UTF-16/32 JSON, a binary
    part) — the byte search cannot see an escape in them, so they always
    pass. The cheap gate in front of every floor scan: a body without any
    of them costs a few substring searches."""
    if isinstance(raw, str):
        return "«" in raw or "\\u00" in raw or _PERCENT_GUILLEMET_TEXT.search(raw) is not None
    return (
        _GUILLEMET_BYTE in raw
        or b"\\u00" in raw
        or b"\x00" in raw
        or _PERCENT_GUILLEMET_BYTES.search(raw) is not None
    )
