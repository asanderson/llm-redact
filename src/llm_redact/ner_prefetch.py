"""NER off the event loop: the strings a request's redaction will scan,
detected by the model-backed detectors ahead of it.

Redaction is synchronous by design: every new value of a request is written
in ONE vault transaction (``vault.run_batched``), and nothing may await
inside it. A model-backed (heavy) detector — every NER backend — costs
milliseconds to seconds per string, and running it there held up every
other request on the proxy. So only DETECTION moves:

1. a COLLECTING pass (``CollectingRedactor``) runs the request's own
   redaction code — the adapter's walk, the verbatim and label fields, a
   Bedrock blob, an upload's parts — with a stand-in redactor that records
   every string the real pass will scan and changes nothing;
2. those strings are detected by the heavy detectors on a worker thread
   (``prefetch``), giving a table of their raw detections per string;
3. the unchanged synchronous redaction runs with the table attached
   (``Redactor.with_precomputed``): each heavy detector's result comes from
   the table, a string missing from it is detected inline and counted
   (``NerStats.prefetch_misses``), so the outcome never depends on the
   table — only where the models ran.

A collecting pass that fails in any way (a body over its string budget, a
field it cannot decode, anything else) disables the prefetch for that
request and never changes its outcome: the real pass meets the same body
and refuses it on its own terms.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from llm_redact.providers.base import ProviderAdapter, prepare_route_request
from llm_redact.redactor import Redactor, StringBudget, TextScan


class CollectingRedactor(Redactor):
    """A stand-in for a request's ``Redactor`` in a collecting pass: every
    string handed to it for detection (``redact_text``, ``scan``,
    ``blocked_type``) is recorded in ``strings``, in order, and comes back
    unchanged; nothing is detected, issued, counted or refused — no vault,
    no detectors. Strings are charged against a budget of its own at the
    request's limit (TooManyStrings past it, as the real pass would refuse),
    so a collecting pass never walks more than the redaction may. Its
    copies (floors, a budget, overrides) share one record."""

    # Deliberately not Redactor.__init__: a collector holds no plan, vault
    # or counters of the proxy's. Every Redactor member an adapter uses is
    # overridden below or reads only the attributes set here.
    def __init__(self, limit: int, strings: list[str] | None = None) -> None:
        self.strings: list[str] = strings if strings is not None else []
        self._limit = limit
        self._budget = StringBudget(limit)
        self.counts: Counter[str] = Counter()
        self.warn_counts: Counter[str] = Counter()

    def redact_text(self, text: str) -> str:
        self.charge(1)
        self.strings.append(text)
        return text

    def scan(self, text: str, *, redactable: bool = False) -> TextScan:
        self.charge(1)
        self.strings.append(text)
        return TextScan(Counter(), False)

    def blocked_type(self, text: str) -> str | None:
        # Charged nothing, like the real one: a check ahead of a redaction
        # that will charge the same text.
        self.strings.append(text)
        return None

    @property
    def blocks(self) -> bool:
        return False

    def with_floors(self, floors: Mapping[str, int]) -> "CollectingRedactor":
        # Floors number new placeholders; they change no string scanned.
        return self

    def with_budget(self, limit: int) -> "CollectingRedactor":
        return CollectingRedactor(limit, self.strings)

    def with_overrides(self, overrides: Any) -> "CollectingRedactor":
        # Overrides decide refusals; a collector refuses nothing.
        return self


def collect(run: Callable[[CollectingRedactor], object], *, limit: int) -> list[str] | None:
    """The strings ``run`` — one redaction pass, given a collector in its
    redactor's place — hands to detection, in order (duplicates kept), or
    None when it raises anything at all: prefetching is then off for the
    request, whose real pass refuses the body (or not) on its own."""
    collector = CollectingRedactor(limit)
    try:
        run(collector)
    except Exception:  # noqa: BLE001 — contained by design: the real pass decides
        return None
    return collector.strings


def collect_request_strings(
    adapter: ProviderAdapter,
    method: str,
    path: str,
    body: dict[str, Any],
    *,
    limit: int,
    mcp_exempt: frozenset[str] = frozenset(),
) -> list[str] | None:
    """The strings the redaction of this JSON request body will scan
    (``providers.base.prepare_route_request``, the proxy's entry point: the
    adapter's walk with its verbatim and label fields), or None when the
    collecting pass fails (``collect``). ``body`` is never changed; the
    system note is left out (it is added after the redaction, and never
    scanned)."""
    return collect(
        lambda collector: prepare_route_request(
            adapter,
            method,
            path,
            body,
            collector,
            inject_note=False,
            mcp_exempt=mcp_exempt,
            # Its own count of exempt MCP blocks: the request's is the
            # real pass's.
            exempt_blocks=[0],
        ),
        limit=limit,
    )
