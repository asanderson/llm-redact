"""NER off the event loop, through the real app: a request's model-backed
detection runs on a worker thread while the event loop keeps answering, and
what reaches the upstream is byte-for-byte what inline detection sends.

The NER backend is the spaCy one over a fake pipeline (ner_fakes) that can
be slowed down, so the suite needs no model and no extra.
"""

import asyncio
import base64
import dataclasses
import json
import threading
import time
from typing import Any

import httpx
import pytest

import llm_redact.proxy as proxy_module
from llm_redact.config import Config
from llm_redact.detection.engine import DetectionConfig, NerConfig, ner_backend_stats
from llm_redact.detection.stats import NerStats
from llm_redact.proxy import create_app
from ner_fakes import FakeSpacy, install_spacy
from prefetch_fixtures import EXEMPT_SERVER, NAME, OPENAI_HEADERS, OTHER, SHAPES, Shape, config

PROXY = "http://127.0.0.1:8787"


class SlowNlp(FakeSpacy):
    """A spaCy pipeline finding the fixtures' names, taking ``delay``
    seconds on a text holding "slow", raising on one holding "boom", and
    recording the threads it ran on."""

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__([(NAME, "PERSON", 1.0), (OTHER, "PERSON", 1.0)], labels=("PERSON",))
        self.delay = delay
        self.started = threading.Event()
        self.threads: set[int] = set()
        self.calls: list[str] = []

    def __call__(self, text: str) -> Any:
        self.calls.append(text)
        self.threads.add(threading.get_ident())
        if "boom" in text:
            raise RuntimeError("model fault")
        if "slow" in text:
            self.started.set()
            time.sleep(self.delay)
        return super().__call__(text)


class Upstream:
    def __init__(self) -> None:
        self.bodies: list[bytes] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content)
        return httpx.Response(200, json={"ok": True})


def _config(detection: DetectionConfig | None = None) -> Config:
    return config(
        detection=detection
        or DetectionConfig(
            ner=NerConfig(enabled=True, backend="spacy"), mcp_exempt_servers=(EXEMPT_SERVER,)
        )
    )


def _stats(app: Any) -> NerStats:
    ((_name, _backend, stats),) = ner_backend_stats(app.state.proxy.detectors)
    return stats


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PROXY)


async def _send(client: httpx.AsyncClient, shape: Shape) -> httpx.Response:
    target = shape.path + (f"?{shape.query}" if shape.query else "")
    return await client.request(
        shape.method,
        target,
        headers={**shape.headers, "content-type": "application/json"},
        content=json.dumps(shape.body).encode(),
    )


def _prefetch_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # The collecting pass "fails": the request redacts inline, as before.
    monkeypatch.setattr(proxy_module, "collect_request_strings", lambda *a, **k: None)


async def _upstream_bytes(
    monkeypatch: pytest.MonkeyPatch, shape: Shape, *, prefetching: bool
) -> tuple[bytes, NerStats]:
    with monkeypatch.context() as patch:
        if not prefetching:
            _prefetch_off(patch)
        install_spacy(patch, SlowNlp())
        upstream = Upstream()
        app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
        async with _client(app) as client:
            response = await _send(client, shape)
        assert response.status_code == 200, response.text
        (body,) = upstream.bodies
        return body, _stats(app)


@pytest.mark.parametrize("shape", SHAPES, ids=[shape.id for shape in SHAPES])
async def test_the_upstream_gets_the_same_bytes_with_and_without_prefetch(
    monkeypatch: pytest.MonkeyPatch, shape: Shape
) -> None:
    inline, inline_stats = await _upstream_bytes(monkeypatch, shape, prefetching=False)
    prefetched, stats = await _upstream_bytes(monkeypatch, shape, prefetching=True)
    assert prefetched == inline
    # NER found and redacted names (count-tokens: inside the base64 blob).
    sent = prefetched.decode()
    if "count-tokens" in shape.path:
        sent = base64.b64decode(json.loads(sent)["input"]["invokeModel"]["body"]).decode()
    assert "«PERSON_001»" in sent
    assert inline_stats.inline_calls > 0
    # Every NER call ran on the worker thread: none inline, none missed.
    assert stats.inline_calls == stats.prefetch_misses == 0
    assert stats.scanned_whole > 0


async def test_the_event_loop_answers_while_a_request_runs_ner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delay = 0.2
    nlp = SlowNlp(delay)
    install_spacy(monkeypatch, nlp)
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    messages = [{"role": "user", "content": f"slow {i}: {NAME}"} for i in range(10)]
    async with _client(app) as client:
        ner = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": messages},
                headers=OPENAI_HEADERS,
            )
        )
        await asyncio.to_thread(nlp.started.wait, 5)
        began = time.perf_counter()
        health = await client.get("/__llm-redact/healthz")
        waited = time.perf_counter() - began
        assert health.status_code == 200
        assert not ner.done()  # ten strings of 200 ms still running
        # Answered without waiting for any string's inference.
        assert waited < delay / 2
        response = await ner
    assert response.status_code == 200
    sent = json.loads(upstream.bodies[0])["messages"]
    # (after the system note: a redaction happened)
    assert [m["content"] for m in sent[1:]] == [f"slow {i}: «PERSON_001»" for i in range(10)]
    assert threading.get_ident() not in nlp.threads
    assert _stats(app).inline_calls == 0


async def test_an_inline_upload_waits_for_one_string_not_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A multipart upload still runs NER inline on the event loop: while a
    # JSON request's batch runs on the worker, its call waits for the
    # string the worker is on, then runs — the loop is held up for at most
    # one string's inference (plus its own), never the batch.
    delay = 0.2
    nlp = SlowNlp(delay)
    install_spacy(monkeypatch, nlp)
    app = create_app(_config(), upstream_transport=httpx.MockTransport(Upstream()))
    messages = [{"role": "user", "content": f"slow {i}"} for i in range(10)]
    async with _client(app) as client:
        ner = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": messages},
                headers=OPENAI_HEADERS,
            )
        )
        await asyncio.to_thread(nlp.started.wait, 5)
        began = time.perf_counter()
        upload = asyncio.create_task(
            client.post(
                "/v1/files",
                headers=OPENAI_HEADERS,
                data={"purpose": "assistants"},
                files={"file": ("notes.txt", f"fast {NAME}".encode(), "text/plain")},
            )
        )
        health = await client.get("/__llm-redact/healthz")
        health_waited = time.perf_counter() - began
        uploaded = await upload
        upload_waited = time.perf_counter() - began
        assert not ner.done()
        assert health.status_code == 200 and uploaded.status_code == 200
        # The upload's strings (its purpose field, file name and text) ran
        # inline, each after at most the one string the worker was on.
        inline = _stats(app).inline_calls
        assert inline == 3
        assert upload_waited < (inline + 1) * delay
        assert health_waited < (inline + 1) * delay
        await ner


async def test_a_reload_between_prefetch_and_redaction_falls_back_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_spacy(monkeypatch, SlowNlp())
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    state = app.state.proxy
    old = _stats(app)
    # The reload drops the email rule: the NER detector's index in the plan
    # moves, so a table read for the wrong plan would hand it another
    # detector's results.
    detection = dataclasses.replace(
        state.config.detection,
        enabled=tuple(name for name in state.config.detection.enabled if name != "email"),
    )
    reloaded = dataclasses.replace(state.config, detection=detection)
    real = proxy_module.prefetch

    async def prefetch_then_reload(plan: Any, strings: Any) -> Any:
        table = await real(plan, strings)
        state.apply_config(reloaded)
        return table

    monkeypatch.setattr(proxy_module, "prefetch", prefetch_then_reload)
    shape = next(shape for shape in SHAPES if shape.id == "openai-chat")
    async with _client(app) as client:
        response = await _send(client, shape)
    assert response.status_code == 200
    new = _stats(app)
    assert new is not old
    # Every string the redaction scanned missed (the table was for the old
    # plan) and ran inline on the new detectors; the old ones ran none.
    assert new.prefetch_misses == new.inline_calls > 0
    assert old.inline_calls == 0
    # Exactly what the reloaded configuration sends.
    monkeypatch.setattr(proxy_module, "prefetch", real)
    fresh_upstream = Upstream()
    fresh = create_app(reloaded, upstream_transport=httpx.MockTransport(fresh_upstream))
    async with _client(fresh) as client:
        assert (await _send(client, shape)).status_code == 200
    assert upstream.bodies == fresh_upstream.bodies
    assert b"jane.doe@corp.example" in upstream.bodies[0]  # email no longer redacted


@pytest.mark.parametrize("prefetching", [True, False], ids=["prefetch", "inline"])
async def test_a_model_fault_refuses_the_request_either_way(
    monkeypatch: pytest.MonkeyPatch, prefetching: bool
) -> None:
    if not prefetching:
        _prefetch_off(monkeypatch)
    nlp = SlowNlp()
    install_spacy(monkeypatch, nlp)
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        with pytest.raises(RuntimeError, match="model fault"):
            await client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "boom"}]},
                headers=OPENAI_HEADERS,
            )
    assert upstream.bodies == []  # never forwarded
    on_loop = threading.get_ident() in nlp.threads
    assert on_loop is not prefetching


async def test_no_prefetch_without_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    # detection = false and pass-through redact nothing: nothing to detect
    # ahead, and no model runs at all.
    nlp = SlowNlp()
    install_spacy(monkeypatch, nlp)
    base = _config()
    providers = dict(base.providers)
    providers["openai"] = dataclasses.replace(providers["openai"], detection=False)
    app = create_app(
        dataclasses.replace(base, providers=providers),
        upstream_transport=httpx.MockTransport(Upstream()),
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": NAME}]},
            headers=OPENAI_HEADERS,
        )
        assert response.status_code == 200
        response = await client.post(
            "/v1/some_new_resource", json={"q": NAME}, headers=OPENAI_HEADERS
        )
        assert response.status_code == 200
    assert nlp.calls == []
