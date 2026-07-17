"""Placeholder token format: «TYPE_NNN».

Guillemet delimiters are chosen because they cannot collide with brackets,
braces, or template syntax that occurs naturally in code and markdown.
"""

import re

# Longest realistic token, e.g. «CREDIT_CARD_0000001». Streaming holdback is
# bounded by this length: text is never delayed by more than one token.
MAX_PLACEHOLDER_LEN = 40

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
