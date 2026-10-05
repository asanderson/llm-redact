"""Reading long strings with fixed-window NER models.

Encoder models read a bounded number of tokens; past it they truncate,
silently. The ``hf`` and ``gliner`` backends therefore read a long string in
overlapping windows — the token-classification pipeline's own ``stride``, and
GLiNER word windows (:func:`word_windows`) — so a value anywhere in the string
can be found. An entity inside an overlap can be reported by both windows:
:func:`drop_exact_duplicates` keeps one copy of each (start, end, type), and
partial overlaps (a name cut at one window's edge, whole in the next) are left
to the redactor's overlap resolution, where the longest span wins.

Nothing here sees more than offsets and types; nothing logs.
"""

import re
from collections.abc import Iterable, Iterator, Sequence

from llm_redact.detection.base import Detection

# gliner 0.2.28 WhitespaceTokenSplitter (data_processing/tokenizer.py), the
# default words splitter: every run of word characters (joined by single "-"
# or "_") is one word, and every other non-space character is a word of its
# own — so each JSON brace, quote, colon and comma counts.
GLINER_WORD_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")


def drop_exact_duplicates(detections: Iterable[Detection]) -> list[Detection]:
    """``detections`` without repeats of one (start, end, type): what two
    overlapping windows both report. First occurrence kept, order kept."""
    seen: set[tuple[int, int, str]] = set()
    kept: list[Detection] = []
    for detection in detections:
        key = (detection.start, detection.end, detection.detector_type)
        if key not in seen:
            seen.add(key)
            kept.append(detection)
    return kept


def gliner_words(text: str) -> list[tuple[int, int]]:
    """The (start, end) offsets of the words GLiNER's default whitespace
    splitter makes of ``text``."""
    return [match.span() for match in GLINER_WORD_RE.finditer(text)]


def word_windows(
    costs: Sequence[int], max_words: int, budget: int | None
) -> Iterator[tuple[int, int, bool]]:
    """Overlapping windows over a text's words, as ``(first, end, over)``
    word-index ranges (``end`` exclusive).

    ``costs`` holds each word's size in model tokens. A window holds at most
    ``max_words`` words and, when ``budget`` is given, words whose costs sum
    to at most ``budget`` — except a single word that alone exceeds it,
    which gets a window of its own with ``over`` True (the model can read
    only part of it). Each later window starts a fifth of the previous
    window's length before that window's end, so a value up to that many
    words long near a seam lies whole inside one window. A text that fits
    is one window; a text without words is one empty window.
    """
    count = len(costs)
    first = 0
    while True:
        end = first
        spent = 0
        while end < count and end - first < max_words:
            if budget is not None and end > first and spent + costs[end] > budget:
                break
            spent += costs[end]
            end += 1
        yield first, end, budget is not None and spent > budget
        if end >= count:
            return
        first = max(end - (end - first) // 5, first + 1)
