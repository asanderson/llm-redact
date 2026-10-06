"""Turning a teacher's tagged answer into a row with grounded gold spans.

The teacher writes each invented value inline as ``<pii type="TYPE">value</pii>``
(inside a JSON string it may escape the quotes: ``<pii type=\\"TYPE\\">``).
:func:`ground` removes the tags and keeps the row only when every span is
grounded:

* the tags are well formed (no nesting, no stray ``<pii`` or ``</pii>``)
  and name one of the allowed types (the generator's catalog types; any
  placeholder type for a reviewer's edit);
* each value is a non-empty exact substring of the clean text, with no
  surrounding whitespace, on one line (an address may span lines);
* one value has one type;
* every OTHER exact occurrence of a tagged value, as a whole token (not
  inside a longer word), is gold too: an untagged repeat would otherwise be
  scored as a miss the label set itself caused. An occurrence that lies
  entirely inside another span, of any type ("Ann" in "Ann Lee", "jdoe" in
  "jdoe@acme.com"), stays as it is (spans never overlap; the outer span
  covers it); one that only partly overlaps another span is a conflict and
  drops the row. A reviewer's edit is taken as written (``propagate=False``):
  what the reviewer leaves untagged stays untagged;
* a hard-negative row (asked to hold no personal data) carries no tag, and a
  positive row carries at least one.

A dropped row is counted by its reason (:data:`REASONS`), never shown.
"""

import re
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass

# The quotes around the type may be JSON-escaped (a tag inside a JSON string
# the teacher kept valid), both alike: group 1 is the escape, 2 the type, 3
# the value.
TAG_RE = re.compile(r'<pii type=(\\?)"([A-Z][A-Z0-9_]{0,27})\1">(.*?)</pii>', re.DOTALL)
_OPEN_OR_CLOSE = re.compile(r"</?pii\b", re.IGNORECASE)
_FENCE_RE = re.compile(r"\A```[^\n]*\n(.*)\n```\Z", re.DOTALL)
MAX_VALUE_CHARS = 200
# Types whose values may span lines.
MULTILINE_TYPES = frozenset({"ADDRESS"})

MALFORMED = "malformed tags"
UNKNOWN_TYPE = "a type outside the allowed types"
BAD_VALUE = "an empty, padded, multi-line or over-long value"
TWO_TYPES = "one value tagged with two types"
CONFLICT = "a repeat of a value partly overlaps another span"
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
        value = match.group(3)
        if _OPEN_OR_CLOSE.search(before) or _OPEN_OR_CLOSE.search(value):
            return MALFORMED
        pieces += [before, value]
        length += len(before)
        spans.append(Span(length, length + len(value), match.group(2)))
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


def _inside(start: int, end: int, spans: Sequence[Span]) -> bool:
    """Whether a span of any type already covers [start, end): the value
    itself, "Ann" inside "Ann Lee", or "jdoe" inside "jdoe@acme.com"."""
    return any(s.start <= start and end <= s.end for s in spans)


def _occurrences(text: str, value: str) -> Iterator[tuple[int, int]]:
    """Every non-overlapping whole-token occurrence of ``value``."""
    start = text.find(value)
    while start != -1:
        end = start + len(value)
        if whole_token(text, start, end):
            yield start, end
        start = text.find(value, end)


def _types(text: str, spans: Sequence[Span]) -> dict[str, str] | str:
    """Each tagged value's one type, or TWO_TYPES."""
    types: dict[str, str] = {}
    for span in spans:
        value = text[span.start : span.end]
        if types.setdefault(value, span.type) != span.type:
            return TWO_TYPES
    return types


def _propagate(text: str, spans: list[Span], types: dict[str, str]) -> tuple[list[Span], int] | str:
    """Every whole-token repeat of a tagged value added as gold; the reason
    when a repeat partly overlaps another span. Longer values go first, so a
    short value repeated inside a repeat of a longer one ("Lee" in a second
    "12 Lee Street") lies inside the span that repeat became."""
    added: list[Span] = []
    for value, type_name in sorted(types.items(), key=lambda item: -len(item[0])):
        for start, end in _occurrences(text, value):
            covered = [*spans, *added]
            if _inside(start, end, covered):
                continue
            if _overlaps(start, end, covered):
                return CONFLICT
            added.append(Span(start, end, type_name))
    return sorted([*spans, *added], key=lambda s: (s.start, s.end)), len(added)


def untagged_repeats(text: str, spans: Sequence[Span]) -> int:
    """How many whole-token occurrences of a tagged value no span covers
    (shown to a reviewer, whose edit leaves them untagged)."""
    values = {text[s.start : s.end] for s in spans}
    return sum(
        1
        for value in values
        for start, end in _occurrences(text, value)
        if not _inside(start, end, spans)
    )


def ground(
    answer: str, *, types: Collection[str], negative: bool | None, propagate: bool = True
) -> Grounded | str:
    """The grounded row of a tagged answer, or the reason it is dropped.
    ``negative`` True = a hard negative (no tag allowed), False = a positive
    (at least one), None = either. ``propagate`` False takes the tags as
    written (a reviewer's edit): untagged repeats stay untagged."""
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
    tagged_types = _types(text, spans)
    if isinstance(tagged_types, str):
        return tagged_types
    if not propagate:
        return Grounded(text=text, spans=tuple(spans))
    propagated = _propagate(text, spans, tagged_types)
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
