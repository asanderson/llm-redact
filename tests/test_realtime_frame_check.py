"""The session router's per-frame realtime check (the optional
``SessionRouter.realtime_frame_refusal``).

A realtime conversation is one long request: its client frames can cite the
same stored objects and cloud storage an HTTP body can, which the HTTP path
puts to the router's ``object_access_refusal``. A router with the frame check
is asked for EVERY client frame that parses as JSON — text or binary, OpenAI
Realtime, Azure, Gemini Live and Vertex Live alike — synchronously, before the
frame is redacted, numbered or sent, in the connection's own context (what
the access gate set for it is visible). A refusal closes the connection 1008
with the router's fixed reason and records a 403; nothing of the frame
reaches the upstream or the vault. A check that raises or answers nonsense
closes it the same way with the core's own reason, counted as the
``realtime_frame`` bookkeeping stage (type-only log).

Real sockets end to end: uvicorn on port 0 in a thread, a fake
``websockets`` upstream in the test's loop (the test_realtime_relay /
test_realtime_reload harness), a scripted router on a bare Registry.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any

import pytest
import websockets

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact import realtime
from llm_redact.config import Config
from llm_redact.plugin_api import Admission
from llm_redact.proxy import REALTIME_FRAME_FAULT, REALTIME_FRAME_STAGE, ProxyState
from llm_redact.registry import Registry
from test_access_seam import FakeGate
from test_realtime_identity import EMAIL, FakeAuth, _recent
from test_realtime_reload import PATHS, Proxy, Upstream, _cfg, _closed, _frame, _serve, _until

pytestmark = pytest.mark.asyncio

MARK = "cited-object-of-another-user"
REASON = "llm-redact: frame cites another user's object"
OTHER_EMAIL = "second.person@corp.example"
IDENTITY = ("azure", "vertex")
ADAPTERS = {
    "openai": "openai-realtime",
    "azure": "azure-realtime",
    "gemini": "gemini-live",
    "vertex": "vertex-live",
}
# What the access gate below sets for the connection it admits (a stand-in
# for llm-redact-pro's per-user namespace).
_ADMITTED: ContextVar[str | None] = ContextVar("test_admitted", default=None)


class FrameRouter:
    """A static-mode session router with only the frame check: records every
    call (with what the gate set for the connection) and refuses a frame
    carrying ``MARK`` — or raises / answers ``answer`` when scripted."""

    mode = "static"

    def __init__(self, *, error: Exception | None = None, answer: Any = None) -> None:
        self.error = error
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def resolve(self, adapter_name: Any, method: str, path: str, body: Any) -> str:
        return "default"

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def realtime_frame_refusal(
        self, adapter_name: str, path: str, frame: Any, *, identity: bool, session_id: str
    ) -> Any:
        self.calls.append(
            {
                "adapter": adapter_name,
                "path": path,
                "frame": frame,
                "identity": identity,
                "session": session_id,
                "user": _ADMITTED.get(),
            }
        )
        if self.error is not None:
            raise self.error
        if MARK in json.dumps(frame):
            return REASON
        return self.answer


class SettingGate(FakeGate):
    """Admits every connection, setting ``_ADMITTED`` in the handler's own
    context (asynchronously, like llm-redact-pro's gate)."""

    async def admit(self, conn: Any, surface: str) -> Admission:  # type: ignore[override]
        _ADMITTED.set("ada")
        return Admission(subject="ada")


def _install(
    monkeypatch: pytest.MonkeyPatch, router: FrameRouter, *, gate: bool = False
) -> FakeAuth:
    auth = FakeAuth()
    reg = Registry()
    reg.build_session_router = lambda config, **kw: router
    reg.build_upstream_auth = lambda name, provider: auth if provider.auth == "identity" else None
    if gate:
        reg.resolve_license = lambda *args, **kwargs: resolved("team")
        reg.build_access_gate = lambda config, license: SettingGate()
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return auth


def _config(provider: str, base: str, **settings: Any) -> Config:
    auth = "identity" if provider in IDENTITY else "passthrough"
    return _cfg({provider: {"upstream_base_url": base, "auth": auth, **settings}})


@contextlib.contextmanager
def _counting(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    calls = [0]
    parse = realtime.parse_client_frame

    def counted(data: str | bytes) -> Any:
        calls[0] += 1
        return parse(data)

    monkeypatch.setattr(realtime, "parse_client_frame", counted)
    yield calls


def _texts(fake: Upstream) -> list[str]:
    return fake.texts()


def _vault_values(proxy: Proxy) -> int:
    state: ProxyState = proxy.state
    return len(state.vault)


@pytest.mark.parametrize("provider", sorted(ADAPTERS))
async def test_every_frame_is_checked_before_it_is_redacted_and_a_refusal_closes(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    router = FrameRouter()
    _install(monkeypatch, router, gate=True)
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy, _counting(monkeypatch) as parses:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                await client.send(_frame(provider, f"mail {EMAIL}"))
                echo = await client.recv()
                # A second frame citing another user's object: refused.
                await client.send(_frame(provider, f"{MARK} for {OTHER_EMAIL}"))
                closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            await _until(lambda: len(fake.close_codes) == 1)
            stored = _vault_values(proxy)
    # The allowed frame was checked AS SENT (the email, before redaction),
    # then redacted: the upstream got a placeholder, the client its value.
    first = router.calls[0]
    assert (first["adapter"], first["identity"], first["session"]) == (
        ADAPTERS[provider],
        provider in IDENTITY,
        "default",
    )
    assert first["path"] == PATHS[provider].split("?")[0]
    assert EMAIL in json.dumps(first["frame"])
    # In the connection's own context: what the gate set is visible.
    assert [call["user"] for call in router.calls] == ["ada", "ada"]
    assert len(fake.received) == 1
    assert EMAIL not in _texts(fake)[0] and "«EMAIL_001»" in _texts(fake)[0]
    if provider in ("openai", "azure"):
        # The item echo is restored (a Gemini Live client message echoed
        # back is no server message shape: forwarded as it came).
        assert EMAIL in (echo.decode() if isinstance(echo, bytes) else echo)
    # The refused frame: closed 1008 with the router's reason, nothing of it
    # sent upstream or numbered into the vault, recorded as a 403.
    assert closed is not None and (closed.code, closed.reason) == (1008, REASON)
    assert stored == 1
    assert (row["status"], row["user"]) == (403, "ada")
    assert fake.close_codes == [1000]
    # One parse per frame: the check's parse is the one redaction walks.
    assert parses[0] == 2


@pytest.mark.parametrize(
    ("error", "answer", "logged"),
    [
        (RuntimeError(f"cannot read {EMAIL}"), None, "RuntimeError"),
        (None, True, "answered bool"),
        (None, "", "answered str"),
        (None, 0, "answered int"),
        (None, b"refused", "answered bytes"),
    ],
    ids=["raises", "true", "empty", "zero", "bytes"],
)
async def test_a_failing_check_closes_the_connection_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception | None,
    answer: Any,
    logged: str,
) -> None:
    caplog.set_level(logging.INFO, logger="llm_redact")
    router = FrameRouter(error=error, answer=answer)
    _install(monkeypatch, router)
    async with Upstream() as fake:
        with _serve(_config("gemini", fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS['gemini']}") as client:
                await client.send(_frame("gemini", f"hello {EMAIL}"))
                closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            faults = dict(proxy.state.bookkeeping_errors)
            stored = _vault_values(proxy)
    assert closed is not None and (closed.code, closed.reason) == (1008, REALTIME_FRAME_FAULT)
    assert fake.received == [] and stored == 0
    assert row["status"] == 403
    assert faults == {REALTIME_FRAME_STAGE: 1}
    assert f"realtime_frame_refusal failed ({logged}); closing" in caplog.text
    assert EMAIL not in caplog.text


async def test_detection_off_still_checks_and_sends_what_was_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = FrameRouter()
    _install(monkeypatch, router)
    # A repeated key: the check reads the LAST occurrence (Python's rule); a
    # first-wins upstream must never see the earlier one it did not read.
    repeated = (
        '{"type": "conversation.item.create",'
        f' "item": {{"note": "{MARK}"}}, "item": {{"note": "{EMAIL}"}}}}'
    )
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url(), detection=False)) as proxy:
            async with websockets.connect(f"ws://{proxy.host}/v1/realtime") as client:
                await client.send(repeated)
                await client.recv()
                await client.send(json.dumps({"type": "session.update", "note": MARK}))
                closed = await _closed(client)
    assert [call["frame"] for call in router.calls][0] == {
        "type": "conversation.item.create",
        "item": {"note": EMAIL},
    }
    # Sent unredacted (detection off) — and exactly as checked.
    assert [json.loads(frame) for frame in fake.received] == [
        {"type": "conversation.item.create", "item": {"note": EMAIL}}
    ]
    assert MARK not in _texts(fake)[0]
    assert closed is not None and (closed.code, closed.reason) == (1008, REASON)


@pytest.mark.parametrize("detection", [True, False], ids=["detection", "detection-off"])
async def test_under_identity_a_frame_the_check_cannot_read_is_refused(
    monkeypatch: pytest.MonkeyPatch, detection: bool
) -> None:
    router = FrameRouter()
    auth = _install(monkeypatch, router)
    async with Upstream() as fake:
        with _serve(_config("azure", fake.url(), detection=detection)) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS['azure']}") as client:
                await client.send("not json at all")
                closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
    assert len(auth.calls) == 1  # the upgrade was authorized and dialled
    assert closed is not None and closed.code == 1008 and "not JSON" in closed.reason
    assert fake.received == [] and router.calls == []
    assert row["status"] == 400


@pytest.mark.parametrize("detection", [True, False], ids=["detection", "detection-off"])
async def test_under_the_clients_key_a_non_json_frame_is_relayed_unchecked(
    monkeypatch: pytest.MonkeyPatch, detection: bool
) -> None:
    router = FrameRouter()
    _install(monkeypatch, router)
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url(), detection=detection)) as proxy:
            async with websockets.connect(f"ws://{proxy.host}/v1/realtime") as client:
                await client.send(f"not json {MARK}")
                assert await client.recv() == f"not json {MARK}"
                await client.send(b"\x00\x01opaque")
                assert await client.recv() == b"\x00\x01opaque"
    assert fake.received == [f"not json {MARK}", b"\x00\x01opaque"]
    assert router.calls == []


@pytest.mark.parametrize("detection", [True, False], ids=["detection", "detection-off"])
async def test_a_frame_nesting_too_deep_is_refused_whatever_the_credential(
    monkeypatch: pytest.MonkeyPatch, detection: bool
) -> None:
    router = FrameRouter()
    _install(monkeypatch, router)
    deep = '{"type": "x", "a": ' + "[" * 300 + "]" * 300 + "}"
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url(), detection=detection)) as proxy:
            async with websockets.connect(f"ws://{proxy.host}/v1/realtime") as client:
                await client.send(deep)
                closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
    assert closed is not None and closed.code == 1008 and "deeper" in closed.reason
    assert fake.received == [] and router.calls == []
    assert row["status"] == 400


async def test_a_long_reason_is_cut_to_fit_the_close_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_reason = "llm-redact: " + "é" * 200  # 412 bytes of UTF-8
    router = FrameRouter(answer=long_reason)
    _install(monkeypatch, router)
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}/v1/realtime") as client:
                await client.send(json.dumps({"type": "session.update"}))
                closed = await _closed(client)
    assert closed is not None and closed.code == 1008
    assert long_reason.startswith(closed.reason) and len(closed.reason.encode()) <= 123
    assert fake.received == []


async def test_a_router_without_the_member_is_never_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlainRouter:
        mode = "static"

        def resolve(self, adapter_name: Any, method: str, path: str, body: Any) -> str:
            return "default"

        def record_response_id(self, response_id: str, session_id: str) -> None:
            return None

    reg = Registry()
    reg.build_session_router = lambda config, **kw: PlainRouter()
    monkeypatch.setattr(registry_mod, "_registry", reg)
    state = ProxyState(Config(), None)
    assert not state.checks_realtime_frames
    assert (
        state.realtime_frame_refusal(
            "openai-realtime", "/v1/realtime", {}, identity=False, session_id="default"
        )
        is None
    )


# JSON the parser refuses although it is JSON: an integer past Python's
# int-digit limit (4300) raises a ValueError that is no JSONDecodeError.
_BIG_INT = "1" * 4301


@pytest.mark.parametrize("checked", [True, False], ids=["frame-check", "no-frame-check"])
@pytest.mark.parametrize("provider", ["openai", "gemini"])
async def test_a_frame_the_parser_refuses_is_refused_not_relayed_unread(
    monkeypatch: pytest.MonkeyPatch, checked: bool, provider: str
) -> None:
    # Under the client's own key a frame that is NOT JSON is relayed as it
    # came; this one is JSON no walk can read, so — like the HTTP body (400)
    # and a frame nesting too deep — it is refused: it once went upstream
    # neither checked nor redacted.
    router = FrameRouter()
    if checked:
        _install(monkeypatch, router)
    text = f'{{"type": "session.update", "note": "mail {EMAIL} {MARK}", "n": {_BIG_INT}}}'
    frame = text.encode() if provider == "gemini" else text
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                await client.send(frame)
                closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
    assert closed is not None and closed.code == 1008 and "cannot read" in closed.reason
    assert fake.received == [] and router.calls == []
    assert row["status"] == 400


async def test_an_upstream_frame_the_parser_refuses_is_forwarded_as_it_came() -> None:
    # The upstream side keeps its rule: what the proxy cannot read goes to
    # the client as it came (placeholders left in place), never an error.
    text = f'{{"type": "response.done", "n": {_BIG_INT}, "note": "«EMAIL_001»"}}'
    assert realtime.parse_json_text(text) is None
    assert realtime.parse_json_text(text.encode()) is None
    assert realtime.frame_floors(text) == {"EMAIL": 1}
    # Not JSON at all is still "not JSON" (None), not a refusal.
    assert realtime.parse_client_frame("not json") is None
    assert realtime.parse_client_frame(b"\xff\xfe") is None
