"""Refusal overrides on a realtime relay: a block-mode value closes the
connection 1008 with the refusal's code in the close reason (it fits the
123-byte limit, whatever the detector type), and an approval lets the value
through on the NEXT connection — once, or every time. Real sockets, like the
relay suite."""

import json
import re
from pathlib import Path

import pytest
import websockets

from llm_redact.config import Config, OverridesConfig, ProviderConfig
from llm_redact.detection.engine import DetectionConfig
from llm_redact.overrides import CODE_LENGTH, OverrideStore
from llm_redact.realtime import blocked_reason
from test_realtime_relay import FakeUpstream, _proxy

EMAIL = "jane.doe@corp.example"
CODE_RE = re.compile(r"llm-redact override ([0-9A-Z]{12}) --once\|--always")


def _config(port: int, store: Path) -> Config:
    return Config(
        providers={**Config().providers, "openai": ProviderConfig(f"http://127.0.0.1:{port}")},
        detection=DetectionConfig(modes=(("email", "block"),)),
        overrides=OverridesConfig(path=str(store)),
    )


def _frame() -> str:
    return json.dumps(
        {
            "type": "conversation.item.create",
            "item": {"type": "message", "content": [{"type": "input_text", "text": EMAIL}]},
        }
    )


async def _refused(url: str) -> str:
    async with websockets.connect(url) as client:
        await client.send(_frame())
        with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
            await client.recv()
    assert closed.value.rcvd is not None and closed.value.rcvd.code == 1008
    reason = closed.value.rcvd.reason
    assert EMAIL not in reason
    match = CODE_RE.search(reason)
    assert match is not None, reason
    return match.group(1)


async def test_a_blocked_frame_carries_a_code_and_the_next_connection_passes(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "overrides.db"
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, store_path)) as host:
            url = f"ws://{host}/v1/realtime?model=gpt-realtime"
            code = await _refused(url)
            assert fake.received == []
            OverrideStore(store_path).approve("once", approver=None, code=code)
            async with websockets.connect(url) as client:
                await client.send(_frame())
                echoed = json.loads(await client.recv())
                assert echoed["item"]["content"][0]["text"] == EMAIL
                # The grant is used: the next frame is refused again.
                await client.send(_frame())
                with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                    await client.recv()
                assert closed.value.rcvd is not None and closed.value.rcvd.code == 1008
            again = await _refused(url)
            OverrideStore(store_path).approve("always", approver=None, code=again)
            async with websockets.connect(url) as client:
                for _ in range(2):
                    await client.send(_frame())
                    await client.recv()
    assert sum(EMAIL in str(frame) for frame in fake.received) == 3
    entries = OverrideStore(store_path).entries()
    # The refused frame of the second connection minted a code of its own.
    assert [(e.state, e.uses) for e in entries] == [("pending", 0), ("always", 2)]


async def test_a_raced_grant_closes_the_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = tmp_path / "overrides.db"
    monkeypatch.setattr(OverrideStore, "consume", lambda self, once, always: False)
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, store_path)) as host:
            url = f"ws://{host}/v1/realtime?model=gpt-realtime"
            code = await _refused(url)
            store = OverrideStore(store_path)
            store.approve("once", approver=None, code=code)
            async with websockets.connect(url) as client:
                await client.send(_frame())
                with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                    await client.recv()
    assert closed.value.rcvd is not None and closed.value.rcvd.code == 1008
    assert "already used" in closed.value.rcvd.reason
    assert fake.received == []


def test_the_close_reason_always_fits_with_its_code() -> None:
    code = "0" * CODE_LENGTH
    for detector_type in ("EMAIL", "GOOGLE_OAUTH_CLIENT_SECRET", "X" * 40, "Y" * 200):
        reason = blocked_reason(detector_type, code)
        assert len(reason.encode()) <= 123 and code in reason
    assert blocked_reason("EMAIL", None) == "blocked by llm-redact policy (EMAIL)"
    assert "(EMAIL); to allow" in blocked_reason("EMAIL", code)
    assert blocked_reason("X" * 40, code).startswith("blocked (XXX")
    assert blocked_reason("Y" * 200, code).startswith("blocked by llm-redact policy; to allow")
