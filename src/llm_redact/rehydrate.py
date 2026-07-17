"""Inbound half: restore original values for placeholders in responses.

The streaming path is the known-hard part. Three failure classes are attested
in prior tools and handled here:
- transport mismatch is avoided upstream (the proxy feeds raw SSE bytes);
- placeholders split across chunk boundaries are held back via a bounded
  prefix buffer («EM → «EMAIL_ → «EMAIL_001»);
- LLM-mangled placeholders («email_001», «EMAIL-1») are recognized by the
  fuzzy grammar in ``placeholders.py`` when ``fuzzy`` is on; every candidate
  is gated on a vault lookup of its canonical form, so a miss always passes
  through verbatim and a false restore requires prose that canonicalizes to
  an actually-issued token.
"""

import json
import re
from collections import Counter
from collections.abc import Callable, Hashable
from typing import Any

from llm_redact.jsonwalk import transform_strings
from llm_redact.placeholders import (
    FUZZY_PLACEHOLDER_RE,
    PLACEHOLDER_RE,
    canonicalize,
    viable_prefix_start,
)
from llm_redact.vault import Vault

# A « / » escape is only real when the backslash that starts it is
# preceded by an even number of backslashes ("\\u00ab" is a literal backslash
# followed by plain text). JSON requires a lowercase u; hex case is free.
_GUILLEMET_ESCAPE_RE = re.compile(r"(?<!\\)((?:\\\\)*)\\u00([aA][bB]|[bB][bB])")


def normalize_guillemet_escapes(text: str) -> str:
    """Rewrite \\u00ab / \\u00bb escapes in JSON source to raw guillemets.

    Raw guillemets are legal JSON string characters, so the source stays
    valid while becoming matchable by the placeholder patterns.
    """
    return _GUILLEMET_ESCAPE_RE.sub(
        lambda m: m.group(1) + ("«" if m.group(2).lower() == "ab" else "»"), text
    )


def _escape_body_is_guillemet_prefix(body: str) -> bool:
    """True when ``body`` (text after a backslash) could grow into u00ab/u00bb."""
    if not body:
        return True  # lone trailing backslash: could become anything, hold it
    if len(body) > 4 or body[0] != "u":
        return False
    hexpart = body[1:]
    for i, ch in enumerate(hexpart):
        if i < 2:
            if ch != "0":
                return False
        elif ch.lower() not in ("a", "b"):
            return False
    return True


def escape_prefix_start(text: str) -> int | None:
    """Index of a trailing partial guillemet escape to hold back, if any.

    Bounded at 5 characters (a complete 6-char escape is normalized before
    this runs). Backslash parity is checked so the second half of a literal
    ``\\\\`` pair is never held.
    """
    i = text.rfind("\\")
    if i == -1:
        return None
    if not _escape_body_is_guillemet_prefix(text[i + 1 :]):
        return None
    j = i
    while j > 0 and text[j - 1] == "\\":
        j -= 1
    if (i - j) % 2 == 1:  # completes a literal backslash pair
        return None
    return i


def substitute_tokens(
    text: str,
    vault: Vault,
    *,
    fuzzy: bool,
    json_escape: bool,
    counts: "Counter[str] | None" = None,
) -> str:
    """Replace every recognized placeholder in ``text`` with its original.

    The single substitution code path shared by streaming and non-streaming
    rehydration — this is what makes the "streaming output equals whole-text
    output" sweep invariant hold for either matching mode.

    ``json_escape`` re-escapes restored values as JSON string source, for
    channels that carry raw JSON text (tool-call arguments). ``counts``
    accumulates successful restores by detector type (for audit/status —
    types and counts only, never values).
    """

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        canonical = canonicalize(token) if fuzzy else token
        original = vault.original_for(canonical) if canonical is not None else None
        if original is None:
            # Unknown placeholders pass through verbatim: never corrupt output.
            return token
        if counts is not None and canonical is not None:
            counts[canonical[1:-1].rpartition("_")[0]] += 1
        if json_escape:
            return json.dumps(original)[1:-1]
        return original

    pattern = FUZZY_PLACEHOLDER_RE if fuzzy else PLACEHOLDER_RE
    return pattern.sub(replace, text)


class Rehydrator:
    """Whole-text restoration for non-streaming bodies."""

    def __init__(
        self, vault: Vault, *, fuzzy: bool = False, counts: "Counter[str] | None" = None
    ) -> None:
        self._vault = vault
        self._fuzzy = fuzzy
        self._counts = counts

    def rehydrate_text(self, text: str) -> str:
        return substitute_tokens(
            text, self._vault, fuzzy=self._fuzzy, json_escape=False, counts=self._counts
        )

    def rehydrate_json(self, obj: Any) -> Any:
        return transform_strings(obj, self.rehydrate_text)

    def rehydrate_json_source_text(self, text: str) -> str:
        """Restore placeholders inside a string that is raw JSON source
        (a complete tool-call ``arguments`` value): guillemet escapes are
        normalized and restored originals are re-escaped, so the result
        remains valid JSON source."""
        return substitute_tokens(
            normalize_guillemet_escapes(text),
            self._vault,
            fuzzy=self._fuzzy,
            json_escape=True,
            counts=self._counts,
        )

    def streaming_channel(self, *, json_source: bool = False) -> "StreamingRehydrator":
        """A streaming channel sharing this rehydrator's vault, fuzzy flag and
        counter — for *buffered* body shapes that still split tokens across
        elements (Gemini's non-SSE ``streamGenerateContent`` chunk array)."""
        return StreamingRehydrator(
            self._vault, json_source=json_source, fuzzy=self._fuzzy, counts=self._counts
        )


class StreamingRehydrator:
    """Incremental restoration for one stream channel (one content block).

    ``feed`` emits everything that is provably safe and retains at most one
    partial-placeholder prefix (bounded by MAX_PLACEHOLDER_LEN), so ordinary
    text is never delayed by more than one token length.

    ``json_source=True`` marks channels whose fragments are raw JSON source
    text (tool-call arguments). Restored originals are spliced in re-escaped
    so the reassembled argument string stays valid JSON.
    """

    def __init__(
        self,
        vault: Vault,
        *,
        json_source: bool = False,
        fuzzy: bool = False,
        counts: "Counter[str] | None" = None,
    ) -> None:
        self._vault = vault
        self._json_source = json_source
        self._fuzzy = fuzzy
        self._counts = counts
        self._buffer = ""
        # Raw tail that might be a split « / » escape (json_source
        # only); held until the next chunk resolves it, at most 5 chars.
        self._escape_tail = ""

    def _match(self, text: str) -> str:
        return substitute_tokens(
            text,
            self._vault,
            fuzzy=self._fuzzy,
            json_escape=self._json_source,
            counts=self._counts,
        )

    def feed(self, text: str) -> str:
        if self._json_source:
            data = self._escape_tail + text
            self._escape_tail = ""
            cut = escape_prefix_start(data)
            if cut is not None:
                self._escape_tail = data[cut:]
                data = data[:cut]
            data = normalize_guillemet_escapes(data)
        else:
            data = text
        self._buffer += data
        substituted = self._match(self._buffer)
        holdback = viable_prefix_start(substituted, fuzzy=self._fuzzy)
        if holdback is None:
            self._buffer = ""
            return substituted
        self._buffer = substituted[holdback:]
        return substituted[:holdback]

    def flush(self) -> str:
        if self._escape_tail:
            # An escape that never completed is plain text after all.
            self._buffer += normalize_guillemet_escapes(self._escape_tail)
            self._escape_tail = ""
        out = self._match(self._buffer)
        self._buffer = ""
        return out


class RehydratorPool:
    """Lazily creates one StreamingRehydrator per stream channel."""

    def __init__(self, vault: Vault, *, fuzzy: bool = False) -> None:
        self._vault = vault
        self._fuzzy = fuzzy
        self._channels: dict[Hashable, StreamingRehydrator] = {}
        # Successful restores by type across all channels of this request.
        self.counts: Counter[str] = Counter()

    def get(self, key: Hashable, *, json_source: bool = False) -> StreamingRehydrator:
        rehydrator = self._channels.get(key)
        if rehydrator is None:
            rehydrator = StreamingRehydrator(
                self._vault, json_source=json_source, fuzzy=self._fuzzy, counts=self.counts
            )
            self._channels[key] = rehydrator
        return rehydrator

    def flush(self, key: Hashable) -> str:
        rehydrator = self._channels.pop(key, None)
        return rehydrator.flush() if rehydrator is not None else ""

    def flush_all(self) -> dict[Hashable, str]:
        leftovers = {key: r.flush() for key, r in self._channels.items()}
        self._channels.clear()
        return {key: text for key, text in leftovers.items() if text}

    def flush_matching(self, predicate: Callable[[Hashable], bool]) -> dict[Hashable, str]:
        keys = [key for key in self._channels if predicate(key)]
        leftovers = {key: self._channels.pop(key).flush() for key in keys}
        return {key: text for key, text in leftovers.items() if text}

    def rehydrate_whole(self, text: str, *, json_source: bool = False) -> str:
        """One-shot rehydration of a complete string with this pool's config.

        For events that repeat the full accumulated text (e.g. Responses API
        ``*.done`` events) rather than a delta.
        """
        rehydrator = StreamingRehydrator(
            self._vault, json_source=json_source, fuzzy=self._fuzzy, counts=self.counts
        )
        return rehydrator.feed(text) + rehydrator.flush()
