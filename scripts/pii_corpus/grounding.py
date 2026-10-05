"""Turning a teacher's tagged answer into a row with grounded gold spans.

The teacher writes each invented value inline as ``<pii type="TYPE">value</pii>``.
:func:`ground` removes the tags and keeps the row only when every span is
grounded:

* the tags are well formed (no nesting, no stray ``<pii`` or ``</pii>``)
  and name a type the run asked for;
* each value is a non-empty exact substring of the clean text, with no
  surrounding whitespace, on one line (an address may span lines);
* one value has one type;
* every OTHER exact occurrence of a tagged value, as a whole token (not
  inside a longer word), is gold too: an untagged repeat would otherwise be
  scored as a miss the label set itself caused. An occurrence already inside
  a span of its type ("Ann" in "Ann Lee") stays as it is; one that overlaps
  any other span is a conflict and drops the row;
* a hard-negative row (asked to hold no personal data) carries no tag, and a
  positive row carries at least one.

A dropped row is counted by its reason (:data:`REASONS`), never shown.
"""

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass

TAG_RE = re.compile(r'<pii type="([A-Z][A-Z0-9_]{0,27})">(.*?)</pii>', re.DOTALL)
_OPEN_OR_CLOSE = re.compile(r"</?pii\b", re.IGNORECASE)
_FENCE_RE = re.compile(r"\A```[^\n]*\n(.*)\n```\Z", re.DOTALL)
MAX_VALUE_CHARS = 200
# Types whose values may span lines.
MULTILINE_TYPES = frozenset({"ADDRESS"})

MALFORMED = "malformed tags"
UNKNOWN_TYPE = "a type the run did not ask for"
BAD_VALUE = "an empty, padded, multi-line or over-long value"
TWO_TYPES = "one value tagged with two types"
CONFLICT = "a repeat of a value overlaps another span"
NEGATIVE_TAGGED = "a hard negative carries tags"
NO_SPANS = "a positive row carries no tag"
PLACEHOLDER = "the text holds a placeholder guillemet"
EMPTY = "an empty answer"
REASONS = (
    MALFORMED,
    UNKNOWN_TYPE,
    BAD_VALUE,
    TWO_TYPES,
    CONFLICT,
    NEGATIVE_TAGGED,
    NO_SPANS,
    PLACEHOLDER,
    EMPTY,
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str

    def as_json(self) -> dict[str, object]:
        return {"start": self.start, "end": self.end, "type": self.type}


@dataclass(frozen=True)
class Grounded:
    text: str
    spans: tuple[Span, ...]
    # Untagged repeats of a tagged value that became gold.
    propagated: int = 0


def strip_fence(answer: str) -> str:
    """The answer without one code fence wrapping all of it."""
    stripped = answer.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1) if match else stripped


def _parse(answer: str) -> tuple[str, list[Span]] | str:
    """(clean text, tagged spans), or the reason the tags are unusable."""
    pieces: list[str] = []
    spans: list[Span] = []
    length = 0
    position = 0
    for match in TAG_RE.finditer(answer):
        before = answer[position : match.start()]
        value = match.group(2)
        if _OPEN_OR_CLOSE.search(before) or _OPEN_OR_CLOSE.search(value):
            return MALFORMED
        pieces += [before, value]
        length += len(before)
        spans.append(Span(length, length + len(value), match.group(1)))
        length += len(value)
        position = match.end()
    rest = answer[position:]
    if _OPEN_OR_CLOSE.search(rest):
        return MALFORMED
    pieces.append(rest)
    return "".join(pieces), spans


def _value_ok(value: str, type_name: str) -> bool:
    if not value or value != value.strip() or len(value) > MAX_VALUE_CHARS:
        return False
    return type_name in MULTILINE_TYPES or "\n" not in value


def whole_token(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (before.isalnum() or before == "_" or after.isalnum() or after == "_")


def _overlaps(start: int, end: int, spans: Sequence[Span]) -> bool:
    return any(start < s.end and s.start < end for s in spans)


def _inside(start: int, end: int, type_name: str, spans: Sequence[Span]) -> bool:
    """Whether a span of the same type already covers [start, end) (the
    value itself, or "Ann" inside "Ann Lee")."""
    return any(s.start <= start and end <= s.end and s.type == type_name for s in spans)


def _propagate(text: str, spans: list[Span]) -> tuple[list[Span], int] | str:
    """Every whole-token repeat of a tagged value added as gold; the reason
    when a repeat overlaps another span."""
    types: dict[str, str] = {}
    for span in spans:
        value = text[span.start : span.end]
        if types.setdefault(value, span.type) != span.type:
            return TWO_TYPES
    added: list[Span] = []
    for value, type_name in types.items():
        start = text.find(value)
        while start != -1:
            end = start + len(value)
            if whole_token(text, start, end) and not _inside(start, end, type_name, spans):
                if _overlaps(start, end, spans) or _overlaps(start, end, added):
                    return CONFLICT
                added.append(Span(start, end, type_name))
            start = text.find(value, end)
    return sorted([*spans, *added], key=lambda s: (s.start, s.end)), len(added)


def ground(answer: str, *, types: Collection[str], negative: bool | None) -> Grounded | str:
    """The grounded row of a tagged answer, or the reason it is dropped.
    ``negative`` True = a hard negative (no tag allowed), False = a positive
    (at least one), None = either (a reviewer's edit)."""
    parsed = _parse(strip_fence(answer))
    if isinstance(parsed, str):
        return parsed
    text, spans = parsed
    if not text.strip():
        return EMPTY
    if "«" in text or "»" in text:
        return PLACEHOLDER
    if negative is True and spans:
        return NEGATIVE_TAGGED
    if negative is False and not spans:
        return NO_SPANS
    for span in spans:
        if span.type not in types:
            return UNKNOWN_TYPE
        if not _value_ok(text[span.start : span.end], span.type):
            return BAD_VALUE
    propagated = _propagate(text, spans)
    if isinstance(propagated, str):
        return propagated
    final, added = propagated
    return Grounded(text=text, spans=tuple(final), propagated=added)


def tagged(text: str, spans: Sequence[Span]) -> str:
    """The inline-tagged form of a row (what a reviewer edits); spans must
    not overlap."""
    pieces = []
    position = 0
    for span in sorted(spans, key=lambda s: s.start):
        pieces += [
            text[position : span.start],
            f'<pii type="{span.type}">{text[span.start : span.end]}</pii>',
        ]
        position = span.end
    pieces.append(text[position:])
    return "".join(pieces)
