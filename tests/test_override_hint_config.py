"""The refusal hint names the proxy's config file when the CLI would not
find it.

A proxy started with an explicit config file (``serve --config PATH``, or
``LLM_REDACT_CONFIG``) that is not what ``llm-redact override`` finds by its
default search prints ``to allow: llm-redact override --config PATH CODE
--once | --always``: run as printed, the CLI reads the same config — and so
the same override store — as the proxy. Otherwise the plain hint. A
realtime close reason carries the path only when it fits the 123 bytes,
else the plain hint."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets

from llm_redact import override_cli
from llm_redact.cli import main
from llm_redact.config import Config, OverridesConfig, ProviderConfig, default_config_path
from llm_redact.detection.engine import DetectionConfig
from llm_redact.overrides import CODE_LENGTH, allow_hint, hint_config
from llm_redact.proxy import create_app
from llm_redact.realtime import blocked_reason
from test_override_cli import FakeTty
from test_realtime_relay import FakeUpstream, _proxy

EMAIL = "jane.doe@corp.example"
KEY = {"authorization": "Bearer sk-client"}
HINT_RE = re.compile(r"to allow: (llm-redact override .*) --once \| --always")


def _write_config(path: Path, store: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[overrides]\nenabled = true\npath = "{store}"\n')
    return path


def _config(store: Path, upstream: str = "http://upstream") -> Config:
    return Config(
        providers={**Config().providers, "openai": ProviderConfig(upstream)},
        detection=DetectionConfig(modes=(("email", "block"),)),
        overrides=OverridesConfig(enabled=True, path=str(store)),
    )


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )


async def _chat(app: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        return await client.post(
            "/v1/chat/completions",
            headers=KEY,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": EMAIL}]},
        )


def _command(response: httpx.Response) -> list[str]:
    """The CLI command the refusal prints, as a shell would split it."""
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    match = HINT_RE.search(message)
    assert match is not None, message
    return shlex.split(match.group(1))


# --------------------------------------------------------------- hint_config


def test_no_explicit_config_means_the_plain_hint(tmp_path: Path) -> None:
    assert hint_config(None, {}) is None
    assert hint_config(None, {"LLM_REDACT_CONFIG": ""}) is None
    assert allow_hint("C0DE") == "to allow: llm-redact override C0DE --once | --always"


def test_an_explicit_config_is_named_absolute_and_quoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spaced = tmp_path / "my configs" / "proxy.toml"
    assert hint_config(spaced, {}) == shlex.quote(str(spaced))
    assert shlex.split(hint_config(spaced, {}) or "") == [str(spaced)]
    # A relative path is named as the proxy resolved it (its working dir).
    monkeypatch.chdir(tmp_path)
    assert hint_config(Path("rel.toml"), {}) == str(tmp_path / "rel.toml")
    # LLM_REDACT_CONFIG counts as explicit: the operator's shell may not set it.
    env_path = tmp_path / "env.toml"
    assert hint_config(None, {"LLM_REDACT_CONFIG": str(env_path)}) == str(env_path)
    # serve --config wins over the variable, as it does when loading.
    other = tmp_path / "flag.toml"
    assert hint_config(other, {"LLM_REDACT_CONFIG": str(env_path)}) == str(other)
    assert allow_hint("C0DE", "/x/c.toml") == (
        "to allow: llm-redact override --config /x/c.toml C0DE --once | --always"
    )


def test_the_default_search_result_needs_no_config_argument(tmp_path: Path) -> None:
    # The CLI's default search finds the XDG file (the isolated test home):
    # naming it explicitly to serve changes nothing for the CLI.
    searched = default_config_path()
    searched.parent.mkdir(parents=True, exist_ok=True)
    searched.write_text("")
    try:
        assert hint_config(searched, {}) is None
        assert hint_config(None, {"LLM_REDACT_CONFIG": str(searched)}) is None
        # The same file reached through a symlink is still that file.
        link = tmp_path / "link.toml"
        link.symlink_to(searched)
        assert hint_config(link, {}) is None
        # Another file is not.
        assert hint_config(tmp_path / "other.toml", {}) == str(tmp_path / "other.toml")
    finally:
        searched.unlink()


def test_a_path_the_hint_cannot_carry_falls_back_to_the_plain_hint() -> None:
    undecodable = Path(os.fsdecode(b"/tmp/\xff-config.toml"))
    assert hint_config(undecodable, {}) is None


# ------------------------------------------------------------------- HTTP


async def test_the_refusal_names_serve_config_and_the_cli_accepts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "proxy-store" / "overrides.db"
    config_file = _write_config(tmp_path / "etc dir" / "proxy.toml", store)
    upstream = Upstream()
    app = create_app(
        _config(store), upstream_transport=httpx.MockTransport(upstream), config_path=config_file
    )
    command = _command(await _chat(app))
    code = command[-1]
    assert command[:4] == ["llm-redact", "override", "--config", str(config_file)]
    assert len(code) == CODE_LENGTH and upstream.requests == []
    # Run as printed (with --once): the CLI reads the proxy's config, so it
    # finds the store holding the code — no LLM_REDACT_CONFIG, no --db.
    monkeypatch.setattr(override_cli, "_open_tty", lambda: FakeTty("allow\n"))
    with pytest.raises(SystemExit) as exited:
        main([*command[1:], "--once"])
    assert exited.value.code in (0, None)
    assert "approved once" in capsys.readouterr().out
    passed = await _chat(app)
    assert passed.status_code == 200 and EMAIL in upstream.requests[0].content.decode()


async def test_llm_redact_config_counts_as_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "overrides.db"
    config_file = _write_config(tmp_path / "env.toml", store)
    monkeypatch.setenv("LLM_REDACT_CONFIG", str(config_file))
    app = create_app(_config(store), upstream_transport=httpx.MockTransport(Upstream()))
    command = _command(await _chat(app))
    assert command[:4] == ["llm-redact", "override", "--config", str(config_file)]


async def test_without_an_explicit_config_the_hint_is_plain(tmp_path: Path) -> None:
    app = create_app(
        _config(tmp_path / "overrides.db"), upstream_transport=httpx.MockTransport(Upstream())
    )
    command = _command(await _chat(app))
    assert command[:2] == ["llm-redact", "override"] and len(command) == 3
    assert "--config" not in (await _chat(app)).text


async def test_the_default_search_file_gives_the_plain_hint(tmp_path: Path) -> None:
    store = tmp_path / "overrides.db"
    searched = _write_config(default_config_path(), store)
    try:
        app = create_app(
            _config(store), upstream_transport=httpx.MockTransport(Upstream()), config_path=searched
        )
        assert "--config" not in (await _chat(app)).text
    finally:
        searched.unlink()


# --------------------------------------------------------------- realtime


def test_the_close_reason_carries_the_path_only_when_it_fits() -> None:
    code = "0" * CODE_LENGTH
    short = "/etc/lr.toml"
    reason = blocked_reason("EMAIL", code, config_arg=short)
    assert reason == (
        f"blocked by llm-redact policy (EMAIL); to allow: llm-redact override --config"
        f" {short} {code} --once|--always"
    )
    # The type's wording shrinks before the path is dropped.
    mid = "/srv/" + "c" * 39 + ".toml"  # 49 characters: the longest that fits
    assert len(mid) == 49
    for detector_type in ("GOOGLE_OAUTH_CLIENT_SECRET", "Y" * 200):
        reason = blocked_reason(detector_type, code, config_arg=mid)
        assert len(reason.encode()) <= 123 and f"--config {mid} {code}" in reason
    # Too long for any wording: the plain hint, never a cut-off path.
    long = "/srv/" + "c" * 40 + ".toml"
    for detector_type in ("EMAIL", "Y" * 200):
        reason = blocked_reason(detector_type, code, config_arg=long)
        assert len(reason.encode()) <= 123 and code in reason
        assert "--config" not in reason and reason.endswith(f"override {code} --once|--always")
    # A named user's reason never names a path.
    named = blocked_reason("EMAIL", code, named=True, config_arg=short)
    assert short not in named and code not in named


@pytest.mark.asyncio
async def test_a_realtime_close_reason_names_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    store = tmp_path / "overrides.db"
    # A short absolute path (a pytest tmp_path is too long for a close
    # reason, which then carries the plain hint: the test above).
    short_dir = Path(tempfile.mkdtemp(prefix="lr", dir="/tmp"))
    request.addfinalizer(lambda: shutil.rmtree(short_dir, ignore_errors=True))
    config_file = _write_config(short_dir / "rt.toml", store)
    monkeypatch.setenv("LLM_REDACT_CONFIG", str(config_file))
    frame = json.dumps(
        {
            "type": "conversation.item.create",
            "item": {"type": "message", "content": [{"type": "input_text", "text": EMAIL}]},
        }
    )
    async with FakeUpstream() as fake:
        with _proxy(_config(store, f"http://127.0.0.1:{fake.port}")) as host:
            async with websockets.connect(f"ws://{host}/v1/realtime?model=m") as client:
                await client.send(frame)
                with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                    await client.recv()
    assert fake.received == []
    assert closed.value.rcvd is not None and closed.value.rcvd.code == 1008
    reason = closed.value.rcvd.reason
    assert len(reason.encode()) <= 123 and EMAIL not in reason
    assert f"llm-redact override --config {config_file} " in reason


# ------------------------------------------------------- unreadable search


def _unreadable_default_search(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stat of the XDG candidate fails with EACCES, as for a proxy whose
    HOME it may not read (``sudo -u svc`` keeping HOME=/root)."""
    searched = default_config_path()
    real_stat = Path.stat

    def stat(self: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if self == searched:
            raise PermissionError(13, "Permission denied", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    return searched


def test_an_unreadable_default_search_names_the_explicit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The CLI's default search cannot be checked: the explicit file is named
    # (a redundant --config is harmless; a missing one points elsewhere).
    _unreadable_default_search(monkeypatch)
    explicit = tmp_path / "proxy.toml"
    assert hint_config(explicit, {}) == str(explicit)
    assert hint_config(None, {"LLM_REDACT_CONFIG": str(explicit)}) == str(explicit)


def test_serve_check_starts_with_an_unreadable_default_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `serve --config X` never needed the default search to start; the hint
    # must not make an unreadable HOME a startup traceback.
    config_file = _write_config(tmp_path / "proxy.toml", tmp_path / "overrides.db")
    _unreadable_default_search(monkeypatch)
    with pytest.raises(SystemExit) as exited:
        main(["serve", "--check", "--config", str(config_file)])
    assert exited.value.code in (0, None)
    assert "serve --check: OK" in capsys.readouterr().out


def test_a_lost_working_directory_gives_the_plain_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A relative explicit path with no working directory to resolve it
    # against has no absolute form to print: the plain hint, no traceback.
    def gone() -> str:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(os, "getcwd", gone)
    assert hint_config(Path("rel.toml"), {}) is None
    assert hint_config(tmp_path / "abs.toml", {}) == str(tmp_path / "abs.toml")
