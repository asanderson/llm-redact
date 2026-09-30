"""Binary upload parts read as text by a plugin, judged by the core.

A BINARY file part of an upload (``upload_content.classify_file``: a PDF,
an Office document, an image) cannot be redacted. An ``UploadInspector``
(``plugin_api``; llm-redact-pro's document extractors) may read it as text
— the core never trusts it with more than that. Per request, before
redaction and before any upstream contact:

1. ``inspect_parts`` hands each binary file part (at most
   ``MAX_INSPECTED_PARTS``, none larger than the inspector's ``max_bytes``)
   to the inspector, ``INSPECT_CONCURRENCY`` at a time, and waits at most
   the inspector's ``timeout`` (capped at ``MAX_INSPECT_SECONDS``) for all
   of them; what is still running is cancelled. A part not handed over, a
   timeout, an exception or an answer that is not an ``Inspection`` is a
   part NOT read.
2. ``judge`` scans every extracted text with the request's live detectors
   (``Redactor.scan_text``: allowlists, modes and deny strings apply; no
   placeholder is issued, nothing is written to the vault). A block-mode
   value refuses the request (``blocked``), and so does any value that
   would be redacted (``detected``: the proxy cannot redact inside the
   file) — whether or not the reading was complete. A warn-mode value is
   counted and stays in the file. A COMPLETE reading that scans clean
   CLEARS the part: it goes out byte-identical — with the client's own
   key always, under a credential the proxy holds only when the
   inspection allows it (``Inspection.proxy_credential``). Every other
   part keeps the rules of an unscanned binary part: forwarded unscanned
   with the client's own key under ``[detection] binary_uploads =
   "forward"`` (counted), refused otherwise.

What a clean scan covers is the EXTRACTED text only: the file itself is
forwarded as sent, and whatever the extractor did not read (it says so
through ``complete``) was never scanned.

Outcomes, one per binary part, counted per provider
(``/status`` ``inspected_uploads_total``,
``llm_redact_inspected_uploads_total{provider,outcome}``): ``clean``,
``detected``, ``blocked``, ``incomplete`` (no text, a partial reading, or
more text than the request's scan budget), ``not_inspected`` (larger than
``max_bytes`` or past ``MAX_INSPECTED_PARTS``), ``timeout``, ``error``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from llm_redact.config import ConfigError
from llm_redact.multipart import AmbiguousHeaders, MultipartPart
from llm_redact.placeholders import merge_floors, token_floors
from llm_redact.plugin_api import Inspection, UploadInspector, UploadPart
from llm_redact.redactor import BlockedRequest, Redactor, UnredactableRequest

logger = logging.getLogger("llm_redact")

# The longest the core waits for one request's inspections, whatever the
# inspector declares: the request is held open meanwhile.
MAX_INSPECT_SECONDS = 300.0
# How many of one request's parts are inspected at a time, and at most.
INSPECT_CONCURRENCY = 4
MAX_INSPECTED_PARTS = 16

OUTCOMES = ("clean", "detected", "blocked", "incomplete", "not_inspected", "timeout", "error")

# A declared media type handed to the inspector: ``type/subtype`` tokens
# only (RFC 6838 restricted names), lower-cased; anything else is None.
_MEDIA_TYPE = re.compile(r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*")


class Limits(NamedTuple):
    """The inspector's declared bounds, as the core applies them."""

    timeout: float
    max_bytes: int


def inspector_limits(inspector: UploadInspector) -> Limits:
    """``timeout`` (seconds, capped at ``MAX_INSPECT_SECONDS``) and
    ``max_bytes`` as the inspector declares them, read once at startup.
    A value that is not a positive number (an int for ``max_bytes``) is a
    ConfigError: a plugin must not leave the bounds to chance."""
    timeout = getattr(inspector, "timeout", None)
    max_bytes = getattr(inspector, "max_bytes", None)
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ConfigError("the upload inspector declares no positive timeout (seconds)")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ConfigError("the upload inspector declares no positive max_bytes")
    return Limits(min(float(timeout), MAX_INSPECT_SECONDS), max_bytes)


def declared_type(part: MultipartPart) -> str | None:
    """The part's declared media type (``type/subtype``, lower-cased), or
    None when absent, ambiguous or not a plain media type."""
    try:
        value = part.header("content-type")
    except AmbiguousHeaders:
        return None
    if value is None:
        return None
    media = value.split(b";", 1)[0].strip().decode("latin-1").lower()
    return media if _MEDIA_TYPE.fullmatch(media) else None


async def inspect_parts(
    inspector: UploadInspector,
    parts: Sequence[tuple[int, MultipartPart]],
    *,
    provider: str,
    identity: bool,
    limits: Limits,
) -> dict[int, Inspection | str]:
    """Each binary part's inspection, by position — or the outcome that
    stands for no reading (``not_inspected``, ``timeout``, ``error``). At
    most ``INSPECT_CONCURRENCY`` at a time; everything still running when
    ``limits.timeout`` passes (or when the request itself is cancelled) is
    cancelled and never awaited again."""
    results: dict[int, Inspection | str] = {}
    handed: list[tuple[int, MultipartPart]] = []
    for index, part in parts:
        if len(handed) >= MAX_INSPECTED_PARTS or len(part.content) > limits.max_bytes:
            results[index] = "not_inspected"
        else:
            handed.append((index, part))
    if not handed:
        return results
    gate = asyncio.Semaphore(INSPECT_CONCURRENCY)

    async def one(part: MultipartPart) -> Inspection:
        async with gate:
            return await inspector.inspect(
                UploadPart(part.content, declared_type(part), provider, identity)
            )

    tasks = {asyncio.ensure_future(one(part)): index for index, part in handed}
    try:
        done, _ = await asyncio.wait(tasks, timeout=limits.timeout)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
                task.add_done_callback(_discard)
    for task, index in tasks.items():
        results[index] = _outcome(task) if task in done else "timeout"
    return results


def _discard(task: asyncio.Future[Inspection]) -> None:
    """A cancelled inspection's result, dropped whenever it ends (never an
    unretrieved exception)."""
    if not task.cancelled():
        task.exception()


def _outcome(task: asyncio.Future[Inspection]) -> Inspection | str:
    """A finished inspection: its ``Inspection``, or ``error`` (logged by
    exception type only — never content)."""
    if task.cancelled():
        return "error"
    exc = task.exception()
    if exc is not None:
        logger.warning(
            "upload inspector failed (%s); the part counts as not read", type(exc).__name__
        )
        return "error"
    result = task.result()
    if not isinstance(result, Inspection):
        logger.warning("upload inspector answered %s, not an Inspection", type(result).__name__)
        return "error"
    return result


class Judgement(NamedTuple):
    """The core's verdict on one request's inspected binary parts."""

    cleared: frozenset[int]  # positions that go out byte-identical
    outcomes: Counter[str]  # per part, see OUTCOMES
    detected: Counter[str]  # detector types of values that would be redacted
    blocked: str | None  # the first block-mode type found
    # Token floors of the scanned texts: placeholders a file carries where
    # the raw bytes do not show them (a compressed PDF stream), which a new
    # value of this request must never be numbered onto.
    floors: dict[str, int]


def judge(
    results: Mapping[int, Inspection | str],
    redactor: Redactor,
    *,
    identity: bool,
    text_budget: int,
) -> Judgement:
    """Scan every extracted text (at most ``text_budget`` characters over
    the request: a text that would exceed it is not scanned, its part
    ``incomplete``) and decide each part (see the module). Raises
    TooManyStrings when the texts exceed the request's string budget."""
    cleared: set[int] = set()
    outcomes: Counter[str] = Counter()
    detected: Counter[str] = Counter()
    blocked: str | None = None
    floors: dict[str, int] = {}
    remaining = text_budget
    for index in sorted(results):
        result = results[index]
        if isinstance(result, str):
            outcomes[result] += 1
            continue
        text = result.text
        if not isinstance(text, str) or len(text) > remaining:
            outcomes["incomplete"] += 1
            continue
        remaining -= len(text)
        merge_floors(floors, token_floors(text))
        try:
            found = redactor.scan_text(text)
        except BlockedRequest as exc:
            blocked = blocked or exc.detector_type
            outcomes["blocked"] += 1
            continue
        if found:
            detected.update(found)
            outcomes["detected"] += 1
        elif result.complete is not True:
            outcomes["incomplete"] += 1
        else:
            outcomes["clean"] += 1
            if not identity or result.proxy_credential is True:
                cleared.add(index)
    return Judgement(frozenset(cleared), outcomes, detected, blocked, floors)


class BinaryValuesDetected(UnredactableRequest):
    """An upload's binary file holds values that would be redacted (found
    in its extracted text): the file cannot be rewritten, so the request is
    refused (400). The message names the detector TYPES only."""

    def __init__(self, types: Counter[str]) -> None:
        super().__init__(
            "llm-redact: an uploaded binary file holds values it must redact"
            f" ({', '.join(sorted(types))}, found in the file's extracted text) and"
            " cannot redact inside the file; the request was not forwarded"
        )
        self.types = types
