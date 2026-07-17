"""In-process latency benchmark: what the proxy costs, not what sockets cost.

Micro benches time the redact/rehydrate primitives on Anthropic-shaped
bodies built from the recall corpus's value generators. Macro benches drive
the real ASGI app via httpx.ASGITransport against an in-process fake
upstream and subtract a direct-to-upstream baseline, isolating the overhead
the proxy adds (parse, redact, forward, rehydrate) from transport costs the
proxy does not control.

Numbers are report-only except two deliberately generous p50 smoke ceilings
enforced by ``--check --latency`` (see __main__): shared CI runners make
p95 noise, but a 10x regression on the medians is a real bug.
"""

import asyncio
import functools
import json
import random
import statistics
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from llm_redact.bench.corpus import VALUE_GENERATORS
from llm_redact.config import Config, ProviderConfig
from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.proxy import create_app
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator, StreamingRehydrator
from llm_redact.vault import InMemoryVault

SMALL_BYTES = 2_000
LARGE_BYTES = 100_000
STREAM_BYTES = 1_000_000
CHUNK_CHARS = 40


@dataclass
class LatencyStat:
    name: str
    p50_ms: float
    p95_ms: float
    iterations: int
    payload_bytes: int
    throughput_mb_s: float | None = None


def _quantiles(seconds: list[float]) -> tuple[float, float]:
    if len(seconds) == 1:
        return seconds[0] * 1000.0, seconds[0] * 1000.0
    cuts = statistics.quantiles(seconds, n=20, method="inclusive")
    return statistics.median(seconds) * 1000.0, cuts[18] * 1000.0


def _time(func: Callable[[], object], iterations: int, warmup: int = 2) -> list[float]:
    for _ in range(warmup):
        func()
    seconds: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        func()
        seconds.append(time.perf_counter() - started)
    return seconds


def _secret_text(rng: random.Random, target_bytes: int) -> str:
    """Prose interleaved with secret-shaped values, ~1 value per 200 bytes —
    a body that is mostly ordinary text but keeps the detectors busy."""
    generators = list(VALUE_GENERATORS.values())
    parts: list[str] = []
    size = 0
    while size < target_bytes:
        _type, gen = rng.choice(generators)
        sentence = (
            f"Deploy note {rng.randrange(10_000)}: rotate {gen(rng)} before the next "
            f"maintenance window and file the change under ticket OPS-{rng.randrange(9_999)}. "
        )
        parts.append(sentence)
        size += len(sentence)
    return "".join(parts)


def _prose_text(rng: random.Random, target_bytes: int) -> str:
    """Secret-free prose that still teases the prefilter: keyword roots,
    colons, digits and dots appear, but nothing forms a full match — the
    realistic shape of most real traffic and the fast path's worst case."""
    # fmt: off
    words = [
        "deploy", "the", "service", "at", "noon", "and", "watch", "metrics", "for",
        "regressions", "config", "value", "10.5", "looks", "fine", "token", "bucket",
        "refill", "secret:", "none", "call", "support", "if", "latency", "p95",
        "exceeds", "200", "ms", "access", "review", "=", "done", "release", "2.14.0",
        "rolled", "to", "41.5%", "of", "traffic", "without", "incident",
    ]
    # fmt: on
    parts: list[str] = []
    size = 0
    while size < target_bytes:
        word = rng.choice(words)
        parts.append(word)
        size += len(word) + 1
    return " ".join(parts)


def _anthropic_body(text: str) -> dict[str, object]:
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "system": "You are a deployment assistant.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
    }


def _micro_stats(rng: random.Random, *, quick: bool) -> list[LatencyStat]:
    config = DetectionConfig()
    detectors = build_detectors(config)
    allowlist = build_allowlist(config)
    vault = InMemoryVault()
    redactor = Redactor(detectors, vault, allowlist)
    rehydrator = Rehydrator(vault)

    stats: list[LatencyStat] = []
    sizes = (("small", SMALL_BYTES), ("large", LARGE_BYTES))
    for label, target in sizes:
        body = _anthropic_body(_secret_text(rng, target))
        payload = len(json.dumps(body).encode())
        iterations = 3 if quick else (200 if label == "small" else 30)

        seconds = _time(functools.partial(redactor.redact_json, body), iterations)
        p50, p95 = _quantiles(seconds)
        stats.append(LatencyStat(f"redact_json_{label}", p50, p95, iterations, payload))

        redacted = redactor.redact_json(body)
        seconds = _time(functools.partial(rehydrator.rehydrate_json, redacted), iterations)
        p50, p95 = _quantiles(seconds)
        stats.append(LatencyStat(f"rehydrate_json_{label}", p50, p95, iterations, payload))

    # Prose-heavy large body: most real traffic detects nothing, so this is
    # the number a typical large request actually pays.
    prose_body = _anthropic_body(_prose_text(rng, LARGE_BYTES))
    prose_payload = len(json.dumps(prose_body).encode())
    iterations = 3 if quick else 30
    seconds = _time(functools.partial(redactor.redact_json, prose_body), iterations)
    p50, p95 = _quantiles(seconds)
    stats.append(LatencyStat("redact_json_prose_large", p50, p95, iterations, prose_payload))

    # Streaming throughput: a token-dense stream fed in tiny chunks, the
    # worst case for the partial-placeholder holdback logic.
    stream_target = 50_000 if quick else STREAM_BYTES
    redacted_stream = redactor.redact_text(_secret_text(rng, stream_target))
    stream_bytes = len(redacted_stream.encode())
    iterations = 2 if quick else 10

    def run_stream() -> None:
        channel = StreamingRehydrator(vault, fuzzy=True)
        for i in range(0, len(redacted_stream), CHUNK_CHARS):
            channel.feed(redacted_stream[i : i + CHUNK_CHARS])
        channel.flush()

    seconds = _time(run_stream, iterations)
    p50, p95 = _quantiles(seconds)
    stats.append(
        LatencyStat(
            "streaming_rehydrate",
            p50,
            p95,
            iterations,
            stream_bytes,
            throughput_mb_s=(stream_bytes / 1_000_000) / statistics.median(seconds),
        )
    )
    return stats


def _fake_upstream() -> Starlette:
    """Anthropic-shaped fake: echoes any placeholders it received so the
    proxy path pays real rehydration costs on the way back."""

    async def messages(request: Request) -> Response:
        raw = await request.body()
        tokens = [w for w in raw.decode().split() if w.startswith("«")][:20]
        reply = "Acknowledged: " + " ".join(tokens)
        body = json.loads(raw)
        if body.get("stream"):

            async def stream() -> AsyncIterator[bytes]:
                yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
                for i in range(0, len(reply), CHUNK_CHARS):
                    payload = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": reply[i : i + CHUNK_CHARS]},
                    }
                    yield (
                        b"event: content_block_delta\ndata: "
                        + json.dumps(payload, ensure_ascii=False).encode()
                        + b"\n\n"
                    )
                yield (
                    b"event: content_block_stop\n"
                    b'data: {"type": "content_block_stop", "index": 0}\n\n'
                )
                yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'

            return StreamingResponse(stream(), media_type="text/event-stream")
        return JSONResponse({"content": [{"type": "text", "text": reply}], "role": "assistant"})

    return Starlette(routes=[Route("/v1/messages", messages, methods=["POST"])])


async def _timed_requests(
    client: httpx.AsyncClient,
    body: dict[str, object],
    *,
    stream: bool,
    iterations: int,
    warmup: int = 2,
) -> list[float]:
    async def once() -> None:
        if stream:
            async with client.stream(
                "POST", "/v1/messages", json={**body, "stream": True}
            ) as response:
                async for _chunk in response.aiter_bytes():
                    pass
        else:
            (await client.post("/v1/messages", json=body)).raise_for_status()

    for _ in range(warmup):
        await once()
    seconds: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        await once()
        seconds.append(time.perf_counter() - started)
    return seconds


async def _macro_stats(rng: random.Random, *, quick: bool) -> list[LatencyStat]:
    upstream = _fake_upstream()
    config = Config(
        providers={
            "anthropic": ProviderConfig(upstream_base_url="http://upstream"),
            "openai": ProviderConfig(upstream_base_url="http://upstream"),
            "gemini": ProviderConfig(upstream_base_url="http://upstream"),
            "azure": ProviderConfig(upstream_base_url=""),
        }
    )
    proxy_app = create_app(config, upstream_transport=httpx.ASGITransport(app=upstream))

    stats: list[LatencyStat] = []
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream), base_url="http://upstream"
        ) as baseline_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_app), base_url="http://proxy"
        ) as proxy_client,
    ):
        for label, target in (("small", SMALL_BYTES), ("large", LARGE_BYTES)):
            body = _anthropic_body(_secret_text(rng, target))
            payload = len(json.dumps(body).encode())
            iterations = 3 if quick else (40 if label == "small" else 15)
            for mode, stream in (("json", False), ("stream", True)):
                base = await _timed_requests(
                    baseline_client, body, stream=stream, iterations=iterations
                )
                via = await _timed_requests(
                    proxy_client, body, stream=stream, iterations=iterations
                )
                base_p50, base_p95 = _quantiles(base)
                via_p50, via_p95 = _quantiles(via)
                stats.append(
                    LatencyStat(f"baseline_{mode}_{label}", base_p50, base_p95, iterations, payload)
                )
                stats.append(
                    LatencyStat(f"proxy_{mode}_{label}", via_p50, via_p95, iterations, payload)
                )
                stats.append(
                    LatencyStat(
                        f"proxy_overhead_delta_{mode}_{label}",
                        via_p50 - base_p50,
                        via_p95 - base_p95,
                        iterations,
                        payload,
                    )
                )
    return stats


def run_latency(seed: int = 42, *, quick: bool = False) -> list[LatencyStat]:
    rng = random.Random(seed)
    stats = _micro_stats(rng, quick=quick)
    stats.extend(asyncio.run(_macro_stats(rng, quick=quick)))
    return stats


def to_markdown(stats: list[LatencyStat]) -> str:
    lines = [
        "## Latency",
        "",
        "In-process overhead (redaction, rehydration, proxy round trip via"
        " ASGITransport minus a direct-to-upstream baseline). Socket and TLS"
        " costs are out of scope — they exist with or without the proxy.",
        "",
        "| bench | p50 ms | p95 ms | iters | payload bytes | MB/s |",
        "|---|---|---|---|---|---|",
    ]
    for stat in stats:
        mbs = f"{stat.throughput_mb_s:.1f}" if stat.throughput_mb_s is not None else "—"
        lines.append(
            f"| {stat.name} | {stat.p50_ms:.2f} | {stat.p95_ms:.2f} |"
            f" {stat.iterations} | {stat.payload_bytes} | {mbs} |"
        )
    return "\n".join(lines) + "\n"


def to_json_list(stats: list[LatencyStat]) -> list[dict[str, object]]:
    return [
        {
            "name": s.name,
            "p50_ms": s.p50_ms,
            "p95_ms": s.p95_ms,
            "iterations": s.iterations,
            "payload_bytes": s.payload_bytes,
            "throughput_mb_s": s.throughput_mb_s,
        }
        for s in stats
    ]


# Smoke ceilings folded into --check when --latency is given. Deliberately
# ~10x above healthy numbers: shared CI runners are slow and jittery, and
# the point is catching accidental quadratic behavior, not perf tuning.
# proxy_overhead_delta_json_large is a DELTA of two independently-noisy
# medians (proxy p50 minus baseline p50), so its jitter is the sum of both
# runs' noise — the loosest ceiling on purpose (healthy is ~10 ms; a real
# quadratic blowup on a 100 KB body would be seconds, not ~150 ms). Bumped
# from 100 after a shared runner landed at 103.7 ms on an unchanged hot path.
# redact_json_prose_large (dense prose, 100 KB) bumped 60 -> 100 for the same
# reason: a loaded runner lands at ~62 ms on a hot path unchanged since it was
# set (a quadratic blowup would be seconds), so 60 was too tight to be jitter-proof.
CHECK_CEILINGS_MS = {
    "redact_json_large": 150.0,
    "redact_json_prose_large": 100.0,
    "proxy_overhead_delta_json_large": 150.0,
}


def ceiling_failures(stats: list[LatencyStat]) -> list[str]:
    by_name = {s.name: s for s in stats}
    failures = []
    for name, ceiling in CHECK_CEILINGS_MS.items():
        stat = by_name.get(name)
        if stat is None:
            failures.append(f"{name}: expected in latency stats but missing")
        elif stat.p50_ms >= ceiling:
            failures.append(f"{name}: p50 {stat.p50_ms:.1f} ms >= ceiling {ceiling:.0f} ms")
    return failures
