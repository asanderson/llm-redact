"""Price table, usage parsing and stream usage tracking for routing budgets.

Feeds ``spend.py`` (spec R-24 … R-26): every successful routed response yields a
``Usage`` (four token classes), the ``PriceTable`` turns it into USD, and the
ledger stores both. Three rules hold throughout:

- Token classes are DISJOINT: ``input_tokens`` is the uncached input only.
  Anthropic reports it that way natively; OpenAI and Gemini fold cached tokens
  into ``prompt_tokens`` / ``promptTokenCount``, so those are subtracted here.
  ``total_tokens`` is therefore always the plain sum of the four classes.
- An unknown model is never guessed at: ``cost_usd`` returns ``None`` and
  records the id in ``unknown_models`` (doctor WARNs per id); the row is still
  counted in tokens. That ledger is BOUNDED (``UnknownModels``): the id is the
  one request-body field detection never scans and it is surfaced verbatim on
  /status, the dashboard and ``llm-redact status``, so at most
  ``UNKNOWN_MODELS_CAP`` distinct ids are kept and only ids that LOOK like a
  model id (short, printable, no whitespace) are stored — everything else is
  counted in ``unknown_models.dropped``, never listed.
- Nothing here mutates its input: ``inject_stream_usage`` returns a new body or
  ``None`` so the proxy's forward-the-original-bytes short-circuit stays intact
  when there is nothing to change.
- Every documented usage shape is read, streamed or buffered: the Responses
  API nests its only usage block under ``response`` in the terminal
  ``response.completed`` event (the tracker reads it there — no Responses
  event carries a top-level ``usage``), and Gemini's ``streamGenerateContent``
  without ``alt=sse`` answers a buffered JSON ARRAY whose last usage-bearing
  element is the complete count (``parse_usage`` accepts the list). A shape
  this module does not know yields ``None`` — never a guess — so the
  integrator sees an unbilled row, not a wrong one.

``ModelPrice`` has a single definition in ``routing.py``; this module imports it.
"""

from __future__ import annotations

import importlib.resources
import json
import math
import re
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_redact.routing import ModelPrice

BUILTIN_PRICES_RESOURCE = "prices.json"  # package data next to dashboard.html

_PRICE_FIELDS = ("input", "output", "cache_read", "cache_write")
# A dated snapshot ("claude-sonnet-4-5-20250929") or "-latest" alias suffix.
_SUFFIX_RE = re.compile(r"-(\d{8}|latest)$")
# A prefix key must end at an id boundary so "gpt-5" never prices "gpt-50".
# "." is deliberately NOT a boundary: "gpt-5.2" is a distinct (pricier) model,
# not a variant of "gpt-5", and every dotted key ("gpt-4.1", "gemini-2.5-*")
# carries its dot inside the key itself. ":" and "@" bound OpenRouter-style
# variant / pin suffixes ("gpt-5:free").
_BOUNDARY_CHARS = "-:@"

_ANTHROPIC_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
_GEMINI_USAGE_KEYS = (
    "promptTokenCount",
    "candidatesTokenCount",
    "cachedContentTokenCount",
    "thoughtsTokenCount",
)


@dataclass(frozen=True)
class Usage:
    """Token counts of one response, by billing class (all disjoint)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read + self.cache_write


def parse_model_price(entry: Any, *, source: str) -> ModelPrice:
    """One ``{input, output, cache_read, cache_write}`` table → ModelPrice.

    All four rates are required (a missing one is a config error naming the
    field, never defaulted — a silently zero rate would under-report spend).
    Rates must be FINITE: ``json.loads`` accepts the ``NaN``/``Infinity``
    literals and TOML has ``nan``/``inf``, and a NaN rate would poison an
    upstream's period total (``nan >= budget`` is always False, so the budget
    could never trip) and leak ``NaN`` into /status JSON."""
    from llm_redact.config import ConfigError

    if not isinstance(entry, Mapping):
        raise ConfigError(f"{source}: expected a table of per-token rates")
    values: dict[str, float] = {}
    for name in _PRICE_FIELDS:
        value = entry.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(
                f"{source}: {name} must be a finite, non-negative number (USD per 1M tokens)"
            )
        try:
            rate = float(value)
        except OverflowError:  # an int too large for a float is as unusable as inf
            rate = math.inf
        if not math.isfinite(rate) or rate < 0:
            raise ConfigError(
                f"{source}: {name} must be a finite, non-negative number (USD per 1M tokens)"
            )
        values[name] = rate
    return ModelPrice(
        input=values["input"],
        output=values["output"],
        cache_read=values["cache_read"],
        cache_write=values["cache_write"],
    )


def parse_price_table(data: Any, *, source: str) -> dict[str, ModelPrice]:
    """The ``{"models": {id: {...}}}`` shape shared by prices.json and user files.

    Top-level keys other than ``models`` (``version``, ``_comment``) are
    ignored so the vendored file can carry provenance."""
    from llm_redact.config import ConfigError

    if not isinstance(data, Mapping):
        raise ConfigError(f'price table {source}: expected a table with a "models" key')
    models = data.get("models")
    if not isinstance(models, Mapping):
        raise ConfigError(f'price table {source}: "models" must be a table of model id -> rates')
    prices: dict[str, ModelPrice] = {}
    for model, entry in models.items():
        if not isinstance(model, str) or not model:
            raise ConfigError(f"price table {source}: model ids must be non-empty strings")
        prices[model] = parse_model_price(entry, source=f"price table {source}, model {model}")
    return prices


def _candidates(model: str) -> list[str]:
    """The normalization ladder, most specific first: the id as given, the id
    without its vendor prefix (OpenRouter's ``anthropic/claude-sonnet-5``), and
    each of those without a dated-snapshot / ``-latest`` suffix."""
    stripped = model.rpartition("/")[2]
    raw = [model, stripped, _SUFFIX_RE.sub("", model), _SUFFIX_RE.sub("", stripped)]
    return [candidate for candidate in dict.fromkeys(raw) if candidate]


def _is_prefix(key: str, candidate: str) -> bool:
    if not candidate.startswith(key):
        return False
    return len(candidate) == len(key) or candidate[len(key)] in _BOUNDARY_CHARS


# Distinct unpriced ids kept for the ops surfaces; the rest are counted.
UNKNOWN_MODELS_CAP = 100
# Longest id shape worth listing: a Bedrock inference-profile ARN is ~110
# chars; anything longer is not a model id, whatever the client called it.
MAX_UNKNOWN_MODEL_ID_LEN = 200


def looks_like_model_id(model: str) -> bool:
    """A non-empty, bounded, printable string with no whitespace or control
    characters — the only shape that is ever LISTED as an unpriced model. The
    ``model`` body field is client text that detection never scans; a pasted
    prompt, a multi-line blob or a control-character payload is counted as
    unpriced but never carried onto /status or the dashboard."""
    return (
        0 < len(model) <= MAX_UNKNOWN_MODEL_ID_LEN
        and model.isprintable()
        and not any(ch.isspace() for ch in model)
    )


class UnknownModels(AbstractSet[str]):
    """A bounded set of unpriced model ids (set-comparable and iterable like
    the plain set it replaces). ``add`` keeps at most ``UNKNOWN_MODELS_CAP``
    ids that pass ``looks_like_model_id`` and counts every other addition in
    ``dropped``; an id already kept is a no-op either way, so the ledger is
    idempotent per model. Nothing here is ever removed at runtime — the
    proxy rebuilds the table on reload and copies over only the ids the new
    table still cannot price."""

    __slots__ = ("_ids", "dropped")

    def __init__(self, ids: Iterable[str] = ()) -> None:
        self._ids: set[str] = set()
        self.dropped = 0
        self.update(ids)

    def add(self, model: str) -> None:
        if model in self._ids:
            return
        if len(self._ids) >= UNKNOWN_MODELS_CAP or not looks_like_model_id(model):
            self.dropped += 1
            return
        self._ids.add(model)

    def update(self, ids: Iterable[str]) -> None:
        for model in ids:
            self.add(model)

    def __contains__(self, model: object) -> bool:
        return model in self._ids

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return f"UnknownModels({sorted(self._ids)!r}, dropped={self.dropped})"


class PriceTable:
    """Model id → ModelPrice with a forgiving lookup and a bounded unknown-id
    ledger (``unknown_models``, see ``UnknownModels``)."""

    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._prices: dict[str, ModelPrice] = dict(prices)
        self.unknown_models = UnknownModels()

    @classmethod
    def builtin(cls) -> PriceTable:
        """The vendored table, read through the same importlib.resources lookup
        the proxy uses for dashboard.html (a packaging slip fails loudly)."""
        text = (
            importlib.resources.files("llm_redact")
            .joinpath(BUILTIN_PRICES_RESOURCE)
            .read_text("utf-8")
        )
        return cls(parse_price_table(json.loads(text), source=BUILTIN_PRICES_RESOURCE))

    @classmethod
    def from_file(cls, path: Path) -> PriceTable:
        """A ``.json`` or ``.toml`` file in the prices.json shape (``[prices] table = PATH``)."""
        from llm_redact.config import ConfigError

        fmt = "toml" if path.suffix.lower() == ".toml" else "json"
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ConfigError(f"cannot read price table {path}: {type(exc).__name__}") from exc
        try:
            data: Any = tomllib.loads(raw.decode("utf-8")) if fmt == "toml" else json.loads(raw)
        except ValueError as exc:  # TOMLDecodeError, JSONDecodeError and UnicodeDecodeError
            raise ConfigError(
                f"price table {path} is not valid {fmt}: {type(exc).__name__}"
            ) from exc
        return cls(parse_price_table(data, source=str(path)))

    def with_overrides(self, overrides: Mapping[str, ModelPrice]) -> PriceTable:
        """A new table where ``overrides`` win over this table's entries."""
        merged = dict(self._prices)
        merged.update(overrides)
        return PriceTable(merged)

    def __contains__(self, model: str) -> bool:
        return model in self._prices

    def __len__(self) -> int:
        return len(self._prices)

    def lookup(self, model: str) -> ModelPrice | None:
        """Exact id; then without the vendor prefix; then without a trailing
        ``-YYYYMMDD`` / ``-latest``; then the LONGEST table key that is a
        boundary-aligned prefix of any of those (``claude-sonnet-5-20260401``
        → ``claude-sonnet-5``; ``o3-pro`` never falls back to ``o3`` because
        ``o3-pro`` is its own key and longer). That fallback is exactly what
        would mis-price a differently-billed sibling WITHOUT its own row
        (``gpt-5-pro`` at ``gpt-5`` rates), so prices.json lists every such
        sibling explicitly. ``None`` when nothing applies."""
        candidates = _candidates(model)
        for candidate in candidates:
            hit = self._prices.get(candidate)
            if hit is not None:
                return hit
        best: ModelPrice | None = None
        best_len = 0
        for candidate in candidates:
            for key, price in self._prices.items():
                if len(key) > best_len and _is_prefix(key, candidate):
                    best, best_len = price, len(key)
        return best

    def cost_usd(self, model: str, usage: Usage) -> float | None:
        """USD for ``usage`` at ``model``'s per-1M rates; ``None`` (and the id
        recorded in the bounded ``unknown_models`` ledger) when the table has
        no entry."""
        price = self.lookup(model)
        if price is None:
            self.unknown_models.add(model)
            return None
        return (
            usage.input_tokens * price.input
            + usage.output_tokens * price.output
            + usage.cache_read * price.cache_read
            + usage.cache_write * price.cache_write
        ) / 1_000_000


def _count(value: Any) -> int:
    """A token count field: non-numeric / negative / bool → 0 (never raise on a
    provider's shape drift — a miscounted row beats a failed request)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(int(value), 0)


def _has_any(usage: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in usage for key in keys)


def _anthropic_usage(usage: Any) -> Usage | None:
    """Messages ``usage``: ``input_tokens`` already EXCLUDES the cache classes."""
    if not isinstance(usage, Mapping) or not _has_any(usage, _ANTHROPIC_USAGE_KEYS):
        return None
    return Usage(
        input_tokens=_count(usage.get("input_tokens")),
        output_tokens=_count(usage.get("output_tokens")),
        cache_read=_count(usage.get("cache_read_input_tokens")),
        cache_write=_count(usage.get("cache_creation_input_tokens")),
    )


def _openai_usage(usage: Any) -> Usage | None:
    """Chat Completions (``prompt_tokens`` / ``completion_tokens`` /
    ``prompt_tokens_details.cached_tokens``) and Responses (``input_tokens`` /
    ``output_tokens`` / ``input_tokens_details.cached_tokens``). Cached tokens
    are part of the prompt count upstream, so they are subtracted out."""
    if not isinstance(usage, Mapping):
        return None
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        prompt = _count(usage.get("prompt_tokens"))
        completion = _count(usage.get("completion_tokens"))
        details = usage.get("prompt_tokens_details")
    elif "input_tokens" in usage or "output_tokens" in usage:
        prompt = _count(usage.get("input_tokens"))
        completion = _count(usage.get("output_tokens"))
        details = usage.get("input_tokens_details")
    else:
        return None
    cached = _count(details.get("cached_tokens")) if isinstance(details, Mapping) else 0
    cached = min(cached, prompt)
    return Usage(input_tokens=prompt - cached, output_tokens=completion, cache_read=cached)


def _gemini_usage(usage: Any) -> Usage | None:
    """``usageMetadata``: ``promptTokenCount`` INCLUDES ``cachedContentTokenCount``
    (subtracted); thinking tokens (``thoughtsTokenCount``) bill as output."""
    if not isinstance(usage, Mapping) or not _has_any(usage, _GEMINI_USAGE_KEYS):
        return None
    prompt = _count(usage.get("promptTokenCount"))
    cached = min(_count(usage.get("cachedContentTokenCount")), prompt)
    output = _count(usage.get("candidatesTokenCount")) + _count(usage.get("thoughtsTokenCount"))
    return Usage(input_tokens=prompt - cached, output_tokens=output, cache_read=cached)


def _ollama_native_usage(payload: Mapping[str, Any]) -> Usage | None:
    """Native ``/api/chat`` + ``/api/generate`` counts live on the payload itself."""
    if "prompt_eval_count" not in payload and "eval_count" not in payload:
        return None
    return Usage(
        input_tokens=_count(payload.get("prompt_eval_count")),
        output_tokens=_count(payload.get("eval_count")),
    )


def parse_usage(protocol: str, payload: Any) -> Usage | None:
    """The usage block of one complete (non-streaming) response body, or ``None``
    when the body carries none. ``protocol`` is the routing protocol of the
    request (``anthropic`` / ``openai`` / ``gemini`` / ``ollama``); an unknown
    protocol yields ``None`` rather than a guess.

    A JSON ARRAY body is Gemini's ``streamGenerateContent`` without ``alt=sse``
    (the buffered form): its chunks carry ``usageMetadata`` progressively and
    the last one bearing it is the complete count — the same last-wins rule the
    SSE tracker applies. No other protocol has an array body shape."""
    if isinstance(payload, list):
        if protocol != "gemini":
            return None
        usage: Usage | None = None
        for chunk in payload:
            parsed = parse_usage(protocol, chunk)
            if parsed is not None:
                usage = parsed
        return usage
    if not isinstance(payload, Mapping):
        return None
    if protocol == "anthropic":
        return _anthropic_usage(payload.get("usage"))
    if protocol == "openai":
        return _openai_usage(payload.get("usage"))
    if protocol == "gemini":
        return _gemini_usage(payload.get("usageMetadata"))
    if protocol == "ollama":
        native = _ollama_native_usage(payload)
        # Ollama's own OpenAI-compatible responses carry a "usage" table instead.
        return native if native is not None else _openai_usage(payload.get("usage"))
    return None


def _responses_event_usage(event: Mapping[str, Any]) -> Usage | None:
    """The Responses API's streamed usage: nested under ``response`` (the
    fixture-pinned ``{"type": "response.completed", "response": {..., "usage":
    {...}}}`` shape). Any ``response.*`` lifecycle event is accepted — the
    non-terminal ones carry ``usage: null`` and yield nothing, so last-wins
    lands on the terminal event's complete count."""
    response = event.get("response")
    if not isinstance(response, Mapping):
        return None
    return _openai_usage(response.get("usage"))


def _pick(delta: Mapping[str, Any], key: str, previous: int) -> int:
    value = delta.get(key)
    return previous if value is None else _count(value)


class StreamUsageTracker:
    """Accumulates the usage of ONE streamed response from its delivered events.

    Feed every SSE ``data:`` payload (or NDJSON line) as it is forwarded; the
    proxy reads ``result()`` in the finalizer. Anthropic splits usage across
    ``message_start`` (input side) and ``message_delta`` (cumulative output side,
    plus any input fields the API repeats); every other protocol's last usage
    block wins outright. For ``openai`` that block is a Chat Completions
    chunk's top-level ``usage`` (the final chunk, when
    ``stream_options.include_usage`` is set) OR the Responses API's
    ``response.usage`` inside its terminal ``response.completed`` /
    ``response.incomplete`` event — Responses streams never carry a top-level
    ``usage`` on any event, and the lifecycle events before the terminal one
    carry ``usage: null``. Unparsable or usage-free events are ignored."""

    def __init__(self, protocol: str) -> None:
        self._protocol = protocol
        self._usage: Usage | None = None

    def feed_sse_data(self, data: str) -> None:
        # "[DONE]", a ping's empty data, or a non-JSON payload: nothing to learn.
        try:
            payload = json.loads(data)
        except ValueError:
            return
        self._feed(payload)

    def feed_ndjson_line(self, line: bytes) -> None:
        try:
            payload = json.loads(line)
        except ValueError:  # includes UnicodeDecodeError
            return
        self._feed(payload)

    def _feed(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        if self._protocol == "anthropic":
            self._feed_anthropic(payload)
            return
        usage = parse_usage(self._protocol, payload)
        if usage is None and self._protocol == "openai":
            usage = _responses_event_usage(payload)
        if usage is not None:
            self._usage = usage  # last wins

    def _feed_anthropic(self, event: Mapping[str, Any]) -> None:
        kind = event.get("type")
        if kind == "message_start":
            usage = parse_usage("anthropic", event.get("message"))
            if usage is not None:
                self._usage = usage
        elif kind == "message_delta":
            delta = event.get("usage")
            if not isinstance(delta, Mapping) or not _has_any(delta, _ANTHROPIC_USAGE_KEYS):
                return
            previous = self._usage or Usage()
            # output_tokens is cumulative (last wins); input-side fields are
            # repeated by newer API versions and override message_start's when
            # present, otherwise message_start's values stand.
            self._usage = Usage(
                input_tokens=_pick(delta, "input_tokens", previous.input_tokens),
                output_tokens=_pick(delta, "output_tokens", previous.output_tokens),
                cache_read=_pick(delta, "cache_read_input_tokens", previous.cache_read),
                cache_write=_pick(delta, "cache_creation_input_tokens", previous.cache_write),
            )

    def result(self) -> Usage | None:
        return self._usage


def inject_stream_usage(body: Mapping[str, Any]) -> dict[str, Any] | None:
    """OpenAI protocol, Chat Completions, ``stream: true``: ask for the final
    usage chunk via ``stream_options.include_usage`` when the client did not
    decide either way. The body must carry ``messages`` — the Chat Completions
    signature. A Responses body (``input`` / ``previous_response_id``, never
    ``messages``) is left alone: its stream always ends with
    ``response.completed`` carrying ``response.usage`` (nothing to ask for),
    and the Responses API does not define ``include_usage`` — OpenAI rejects
    parameters it does not know with a 400, so injecting there would break the
    request outright. Returns the rewritten copy, or ``None`` when nothing
    changes (stream off, not a Chat Completions body, the key already present
    with any value, or ``stream_options`` not a table — the proxy then forwards
    the original bytes). Never mutates ``body``."""
    if body.get("stream") is not True or "messages" not in body:
        return None
    options = body.get("stream_options")
    if options is None:
        new_options: dict[str, Any] = {"include_usage": True}
    elif isinstance(options, Mapping):
        if "include_usage" in options:
            return None
        new_options = {**options, "include_usage": True}
    else:
        return None
    result = dict(body)
    result["stream_options"] = new_options
    return result
