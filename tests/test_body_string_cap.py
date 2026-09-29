"""max_body_strings: one request body may not cost the event loop more strings
than the cap allows.

Redaction runs string by string, synchronously, on the proxy's event loop; a
10 MiB body of ~300k tiny strings once held it for over half a minute while
every other request waited. The cap is enforced where every redaction
passes — the per-request Redactor copy (``with_budget``) counts each string
it is asked to redact (JSON string values, form fields, file names), the
multipart adapter charges an uploaded file's JSONL lines before splitting
it, and a multipart body's parts are counted before any parse. Over the cap:
a recorded, provider-shaped 413 before any upstream contact (a realtime
client frame: close 1009).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from typing import Any

import httpx
import pytest
import websockets

from llm_redact import multipart
from llm_redact.config import (
    DEFAULT_MAX_BODY_STRINGS,
    Config,
    ConfigError,
    ProviderConfig,
    parse_config,
)
from llm_redact.config_write import emit_config_toml
from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.proxy import ProxyState, create_app
from llm_redact.redactor import (
    Redactor,
    StringBudget,
    TooManyStrings,
    UnredactableRequest,
)
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"


def _redactor(**kwargs: Any) -> Redactor:
    config = DetectionConfig()
    return Redactor(build_detectors(config), InMemoryVault(), build_allowlist(config), **kwargs)


# ---- the budget -------------------------------------------------------------


def test_budget_allows_exactly_its_limit() -> None:
    budget = StringBudget(3)
    budget.charge(2)
    budget.charge(1)
    with pytest.raises(TooManyStrings) as refused:
        budget.charge(1)
    assert refused.value.limit == 3
    assert str(refused.value) == "request body exceeds llm-redact max_body_strings (3)"
    # A path that does not tell it apart still refuses the request.
    assert isinstance(refused.value, UnredactableRequest)


def test_one_charge_over_the_limit_refuses() -> None:
    with pytest.raises(TooManyStrings):
        StringBudget(3).charge(4)


def test_budgeted_redactor_counts_every_string_and_refuses_past_the_limit() -> None:
    redactor = _redactor().with_budget(3)
    assert redactor.redact_text("one") == "one"
    assert redactor.redact_text("two") == "two"
    assert redactor.redact_text(f"mail {EMAIL}") == "mail «EMAIL_001»"
    with pytest.raises(TooManyStrings):
        redactor.redact_text("four")


def test_the_string_over_the_limit_is_never_scanned() -> None:
    shared = _redactor()
    redactor = shared.with_budget(1)
    redactor.redact_text("first")
    with pytest.raises(TooManyStrings):
        redactor.redact_text(f"mail {EMAIL}")
    assert shared.counts == Counter()  # nothing detected, nothing issued
    assert shared.redact_text(f"mail {EMAIL}") == "mail «EMAIL_001»"


def test_a_shared_redactor_counts_nothing() -> None:
    redactor = _redactor()
    redactor.charge(10**9)  # no budget: nothing to exceed
    for _ in range(50):
        redactor.redact_text("x")


def test_charge_counts_uploaded_lines_against_the_same_budget() -> None:
    redactor = _redactor().with_budget(5)
    redactor.charge(4)
    redactor.redact_text("one string left")
    with pytest.raises(TooManyStrings):
        redactor.charge(1)


def test_redact_json_counts_the_strings_the_walk_redacts() -> None:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "a"}, {"role": "x", "content": "b"}],
    }
    _redactor().with_budget(2).redact_json(body)  # model/role are structural: 2 strings
    with pytest.raises(TooManyStrings):
        _redactor().with_budget(1).redact_json(body)


def test_budget_copy_keeps_floors_counters_and_modes() -> None:
    counts: Counter[str] = Counter()
    warn_counts: Counter[str] = Counter()
    base = _redactor(counts=counts, warn_counts=warn_counts, modes={"IPV4": "warn"})
    budgeted = base.with_floors({"EMAIL": 4}).with_budget(10)
    assert budgeted.redact_text(f"mail {EMAIL} from 8.8.8.8") == "mail «EMAIL_005» from 8.8.8.8"
    assert counts == Counter({"EMAIL": 1})
    assert warn_counts == Counter({"IPV4": 1})


def test_floor_copies_share_the_budget() -> None:
    budgeted = _redactor().with_budget(2)
    budgeted.redact_text("one")
    raised = budgeted.with_floors({"EMAIL": 3})
    assert raised is not budgeted
    raised.redact_text("two")
    with pytest.raises(TooManyStrings):
        raised.redact_text("three")
    with pytest.raises(TooManyStrings):
        budgeted.redact_text("three")


def test_each_budget_copy_counts_on_its_own() -> None:
    shared = _redactor()
    first = shared.with_budget(1)
    second = shared.with_budget(1)
    first.redact_text("a")
    second.redact_text("b")  # a separate request body: its own count


# ---- config -----------------------------------------------------------------


def test_default_and_explicit_values() -> None:
    assert Config().max_body_strings == DEFAULT_MAX_BODY_STRINGS == 100_000
    assert parse_config({}, "test").max_body_strings == 100_000
    assert parse_config({"max_body_strings": 7}, "test").max_body_strings == 7


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_value_refused(value: int) -> None:
    with pytest.raises(ConfigError, match="max_body_strings must be a positive integer"):
        parse_config({"max_body_strings": value}, "test")


def test_non_positive_byte_cap_refused_too() -> None:
    with pytest.raises(ConfigError, match="max_body_bytes must be a positive integer"):
        parse_config({"max_body_bytes": 0}, "test")


def test_emitted_and_parsed_back() -> None:
    import tomllib

    text = emit_config_toml(Config(max_body_strings=1234))
    assert "max_body_strings = 1234" in text
    assert parse_config(tomllib.loads(text), "test").max_body_strings == 1234


# ---- end to end: JSON bodies -------------------------------------------------


class _Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"ok": True})


def _app(upstream: _Upstream, **config: Any) -> Any:
    providers = {**Config().providers, **config.pop("providers", {})}
    return create_app(
        Config(providers=providers, **config), upstream_transport=httpx.MockTransport(upstream)
    )


def _client(app: Any) -> httpx.AsyncClient:
    # A loopback name: a request that spends the proxy's own credential
    # (identity auth) must name a host the proxy answers to.
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _chat(strings: int) -> dict[str, Any]:
    # model/role are structural: exactly `strings` redactable strings.
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": f"line {i} {EMAIL}"} for i in range(strings)],
    }


async def test_json_body_at_the_cap_is_forwarded() -> None:
    upstream = _Upstream()
    app = _app(upstream, max_body_strings=5, inject_system_note=False)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat(5))
    assert response.status_code == 200
    assert EMAIL.encode() not in upstream.requests[0].content


async def test_json_body_over_the_cap_is_refused_413_before_upstream(
    caplog: pytest.LogCaptureFixture,
) -> None:
    upstream = _Upstream()
    app = _app(upstream, max_body_strings=5, inject_system_note=False)
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat(6))
    assert response.status_code == 413
    error = response.json()["error"]
    assert error["message"] == "request body exceeds llm-redact max_body_strings (5)"
    assert error["code"] == "request_too_large"  # the provider's own 413 shape
    assert upstream.requests == []
    row = app.state.proxy.recent[-1]
    assert (row["status"], row["path"], row["detections"]) == (413, "/v1/chat/completions", {})
    assert EMAIL not in caplog.text and EMAIL not in response.text


async def test_anthropic_shaped_413() -> None:
    upstream = _Upstream()
    app = _app(upstream, max_body_strings=2)
    body = {
        "model": "m",
        "max_tokens": 5,
        "system": "s",
        "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
    }
    async with _client(app) as client:
        response = await client.post("/v1/messages", json=body, headers={"x-api-key": "k"})
    assert response.status_code == 413
    assert response.json()["type"] == "error"
    assert "max_body_strings (2)" in response.json()["error"]["message"]
    assert upstream.requests == []


async def test_budget_carries_over_the_floor_copy() -> None:
    # A body carrying a token takes the with_floors copy: still counted.
    upstream = _Upstream()
    app = _app(upstream, max_body_strings=2, inject_system_note=False)
    body = _chat(3)
    body["messages"][0]["content"] = "earlier «EMAIL_004»"
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 413 and upstream.requests == []


async def test_cap_is_hot_and_read_once_per_request(tmp_path: Any) -> None:
    upstream = _Upstream()
    app = _app(upstream, max_body_strings=2, inject_system_note=False)
    state: ProxyState = app.state.proxy
    async with _client(app) as client:
        assert (await client.post("/v1/chat/completions", json=_chat(3))).status_code == 413
        assert (
            state.apply_config(Config(providers=state.config.providers, max_body_strings=3)) == []
        )
        assert (await client.post("/v1/chat/completions", json=_chat(3))).status_code == 200


async def test_pass_through_and_detection_off_are_not_counted() -> None:
    # Nothing is redacted on either path, so nothing costs per string.
    upstream = _Upstream()
    app = _app(
        upstream,
        max_body_strings=1,
        providers={"openai": ProviderConfig("http://up", detection=False)},
    )
    async with _client(app) as client:
        off = await client.post("/v1/chat/completions", json=_chat(3))
        through = await client.post("/v1/assistants", json=_chat(3))  # unrecognized route
    assert (off.status_code, through.status_code) == (200, 200)
    assert len(upstream.requests) == 2


async def test_status_and_doctor_show_the_cap() -> None:
    from llm_redact.doctor_cli import _check_body_cap, _Report

    app = _app(_Upstream(), max_body_strings=4321)
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["max_body_strings"] == 4321
    report = _Report(json_mode=True)
    _check_body_cap(report, Config(max_body_strings=4321))
    assert "max_body_strings 4321" in report.rows[0]["message"]


# ---- end to end: multipart ----------------------------------------------------

_MP = {"content-type": "multipart/form-data; boundary=b", "authorization": "Bearer sk-t"}


def _form(*parts: tuple[bytes, bytes]) -> bytes:
    out = b""
    for headers, value in parts:
        out += b"--b\r\n" + headers + b"\r\n\r\n" + value + b"\r\n"
    return out + b"--b--\r\n"


def _field(name: str, value: bytes) -> tuple[bytes, bytes]:
    return (f'Content-Disposition: form-data; name="{name}"'.encode(), value)


def _jsonl(lines: list[bytes]) -> tuple[bytes, bytes]:
    return (
        b'Content-Disposition: form-data; name="file"; filename="in.jsonl"\r\n'
        b"Content-Type: application/octet-stream",
        b"\n".join(lines),
    )


def _line(text: str) -> bytes:
    return json.dumps(
        {"custom_id": "a", "body": {"messages": [{"role": "user", "content": text}]}}
    ).encode()


async def test_multipart_parts_over_the_cap_refused_before_any_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _Upstream()
    app = _app(upstream, max_body_strings=3)

    def no_parse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("parsed a multipart body over the part cap")

    monkeypatch.setattr(multipart, "parse", no_parse)
    body = _form(*[_field("purpose", b"batch")] * 4)
    async with _client(app) as client:
        response = await client.post("/v1/files", content=body, headers=_MP)
    assert response.status_code == 413
    assert "max_body_strings (3)" in response.json()["error"]["message"]
    assert upstream.requests == []
    assert app.state.proxy.recent[-1]["status"] == 413


async def test_parts_are_counted_on_every_matched_route() -> None:
    # Like max_body_bytes: a JSON route sent a multipart body is capped too
    # (the OpenAI-family hook scans multipart on any matched route).
    upstream = _Upstream()
    app = _app(upstream, max_body_strings=3)
    body = _form(*[_field("x", b"")] * 4)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", content=body, headers=_MP)
    assert response.status_code == 413 and upstream.requests == []


async def test_multipart_parts_at_the_cap_are_parsed_and_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _Upstream()
    app = _app(upstream, max_body_strings=3)
    parsed: list[int] = []
    real_parse = multipart.parse

    def spy(body: bytes, boundary: bytes) -> Any:
        found = real_parse(body, boundary)
        parsed.append(len(found.parts) if found is not None else -1)
        return found

    monkeypatch.setattr(multipart, "parse", spy)
    body = _form(*[_field("purpose", b"batch")] * 3)
    async with _client(app) as client:
        response = await client.post("/v1/files", content=body, headers=_MP)
    assert response.status_code == 200 and len(upstream.requests) == 1
    assert parsed and set(parsed) == {3}


def test_the_delimiter_count_bounds_the_parts() -> None:
    from llm_redact.proxy import _multipart_parts_over

    headers = httpx.Headers({"content-type": "multipart/form-data; boundary=b"})
    body = _form(*[_field("f", b"")] * 3)
    parts = multipart.parse(body, b"b")
    assert parts is not None and len(parts.parts) == 3
    assert not _multipart_parts_over(headers, body, 3)
    assert _multipart_parts_over(headers, body, 2)
    # A preamble adds one delimiter: the bound errs toward refusing.
    assert _multipart_parts_over(headers, b"preamble\r\n" + body, 3)
    assert not _multipart_parts_over(httpx.Headers({"content-type": "text/plain"}), body, 0)


async def test_uploaded_jsonl_lines_count_before_they_are_split() -> None:
    # One part, but far more lines (blank ones too) than the cap allows.
    upstream = _Upstream()
    app = _app(upstream, max_body_strings=50)
    body = _form(_field("purpose", b"batch"), _jsonl([b""] * 60))
    async with _client(app) as client:
        response = await client.post("/v1/files", content=body, headers=_MP)
    assert response.status_code == 413 and upstream.requests == []


async def test_jsonl_lines_and_their_strings_share_the_budget() -> None:
    upstream = _Upstream()
    # The file name + 3 lines + (custom_id + content) x 3 = 10 strings.
    body = _form(_jsonl([_line(f"to {EMAIL}"), _line("hi"), _line("yo")]))
    async with _client(_app(upstream, max_body_strings=10)) as client:
        assert (await client.post("/v1/files", content=body, headers=_MP)).status_code == 200
    async with _client(_app(upstream, max_body_strings=9)) as client:
        assert (await client.post("/v1/files", content=body, headers=_MP)).status_code == 413
    assert len(upstream.requests) == 1
    assert EMAIL.encode() not in upstream.requests[0].content


async def test_multipart_on_a_detection_off_provider_is_not_counted() -> None:
    upstream = _Upstream()
    app = _app(
        upstream,
        max_body_strings=1,
        providers={"openai": ProviderConfig("http://up", detection=False)},
    )
    body = _form(*[_field("purpose", b"batch")] * 5)
    async with _client(app) as client:
        response = await client.post("/v1/files", content=body, headers=_MP)
    assert response.status_code == 200
    assert upstream.requests[0].content == body  # forwarded byte-identical


async def test_identity_multipart_parts_counted_before_the_vouching_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_upstream_auth import AZURE, _identity, _install

    _, built = _install(monkeypatch)
    upstream = _Upstream()
    app = _app(upstream, max_body_strings=2, providers={"azure": _identity(AZURE)})

    def no_parse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("parsed a multipart body over the part cap")

    monkeypatch.setattr(multipart, "parse", no_parse)
    body = _form(*[_field("purpose", b"batch")] * 3)
    async with _client(app) as client:
        response = await client.post(
            "/openai/files?api-version=1",
            content=body,
            headers={"content-type": "multipart/form-data; boundary=b"},
        )
    assert response.status_code == 413
    assert upstream.requests == [] and built[0].calls == []


# ---- end to end: realtime frames ---------------------------------------------


async def test_realtime_frame_over_the_cap_closes_1009_unsent() -> None:
    from test_realtime_relay import FakeUpstream, _proxy

    frame = json.dumps(
        {
            "type": "conversation.item.create",
            "item": {"content": [{"text": f"{i}"} for i in range(4)]},
        }
    )
    small = json.dumps({"type": "conversation.item.create", "item": {"content": [{"text": EMAIL}]}})
    async with FakeUpstream() as fake:
        config = Config(
            max_body_strings=3,
            providers={
                **Config().providers,
                "openai": ProviderConfig(f"http://127.0.0.1:{fake.port}"),
            },
        )
        with _proxy(config) as proxy_host:
            async with websockets.connect(f"ws://{proxy_host}/v1/realtime") as client:
                await client.send(small)  # within the cap: relayed, redacted
                await client.recv()
                await client.send(frame)
                with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                    await client.recv()
                assert closed.value.rcvd is not None
                assert closed.value.rcvd.code == 1009
                assert "max_body_strings (3)" in closed.value.rcvd.reason
            rows: list[dict[str, Any]] = []
            async with httpx.AsyncClient(base_url=f"http://{proxy_host}") as http:
                for _ in range(50):
                    rows = (await http.get("/__llm-redact/recent")).json()["entries"]
                    if rows:
                        break
                    await asyncio.sleep(0.05)
    assert len(fake.received) == 1  # the over-cap frame never reached the upstream
    assert EMAIL not in str(fake.received[0]) and "«EMAIL_001»" in str(fake.received[0])
    assert rows[0]["method"] == "WS" and rows[0]["status"] == 413
