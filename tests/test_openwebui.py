"""The optional WebUI deployment and its real-proxy OpenAI demo contract."""

import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType

import httpx
import pytest
import yaml

from llm_redact.config import Config, ProviderConfig, VaultConfig
from llm_redact.proxy import create_app

ROOT = Path(__file__).resolve().parents[1]
EMAIL = "jane.doe@corp.example"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


def _smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "webui_smoke", ROOT / "scripts/openwebui_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_reads_sse_text_and_requires_completion_marker() -> None:
    smoke = _smoke()
    body = (
        b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
        b'data: {"choices":[],"usage":{"total_tokens":2}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert smoke.read_sse(io.BytesIO(body)) == "hello world"
    with pytest.raises(RuntimeError, match="completion marker"):
        smoke.read_sse(io.BytesIO(body.replace(b"data: [DONE]\n\n", b"")))


def test_smoke_rejects_missing_restoration_and_upstream_leaks() -> None:
    smoke = _smoke()
    smoke.check_reply(f"{EMAIL} {AWS_KEY}")
    with pytest.raises(RuntimeError, match="restored"):
        smoke.check_reply("«EMAIL_001» «AWS_KEY_001»")
    smoke.check_upstream("«EMAIL_001» «AWS_KEY_001»")
    with pytest.raises(RuntimeError, match="upstream"):
        smoke.check_upstream(f"«EMAIL_001» «AWS_KEY_001» {EMAIL}")


@pytest.fixture
def fake() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "fake_upstream", ROOT / "scripts/fake_upstream.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_demo_model_discovery(fake: ModuleType) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.create_app()), base_url="http://upstream"
    ) as client:
        response = await client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "fake-openwebui"


def _text(response: httpx.Response, stream: bool) -> str:
    if not stream:
        return response.json()["choices"][0]["message"]["content"]
    events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    return "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mangle", [False, True])
async def test_webui_protocol_redacts_and_restores(
    fake: ModuleType, capsys: pytest.CaptureFixture[str], stream: bool, mangle: bool
) -> None:
    fake.MANGLE = mangle
    app = create_app(
        Config(providers={"openai": ProviderConfig(upstream_base_url="http://upstream")}),
        upstream_transport=httpx.ASGITransport(app=fake.create_app()),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):
        assert (await client.get("/v1/models")).json()["data"][0]["id"] == "fake-openwebui"
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-openwebui",
                "stream": stream,
                "messages": [{"role": "user", "content": f"Email {EMAIL}; rotate {AWS_KEY}"}],
            },
        )
    assert response.status_code == 200
    reply = _text(response, stream)
    assert EMAIL in reply and AWS_KEY in reply
    assert "«" not in reply
    upstream_log = capsys.readouterr().out
    assert EMAIL not in upstream_log and AWS_KEY not in upstream_log
    assert "«EMAIL_001»" in upstream_log and "«AWS_KEY_001»" in upstream_log


async def test_webui_vault_restores_after_restart(fake: ModuleType, tmp_path: Path) -> None:
    config = Config(
        providers={"openai": ProviderConfig(upstream_base_url="http://upstream")},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "vault.db"), session="webui"),
    )
    for message in [EMAIL, "Repeat «EMAIL_001»"]:
        app = create_app(config, upstream_transport=httpx.ASGITransport(app=fake.create_app()))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
            ) as client,
        ):
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake-openwebui",
                    "messages": [{"role": "user", "content": message}],
                },
            )
        assert EMAIL in _text(response, False)


def test_optional_stack_confines_backend_and_provider_configuration() -> None:
    stack = yaml.safe_load((ROOT / "compose.openwebui.yaml").read_text())
    services = stack["services"]
    webui = services["open-webui"]
    proxy = services["proxy"]
    assert webui["profiles"] == proxy["profiles"] == ["webui"]
    assert webui["ports"] == ["127.0.0.1:${OPENWEBUI_PORT:-3000}:8080"]
    assert proxy["ports"] == ["127.0.0.1:${OPENWEBUI_PROXY_PORT:-8787}:8787"]
    assert webui["networks"] == ["webui-internal"]
    assert stack["networks"]["webui-internal"]["internal"] is True
    assert "provider-egress" in proxy["networks"]
    env = webui["environment"]
    assert env["OPENAI_API_BASE_URLS"] == "http://proxy:8787/v1"
    for name in [
        "ENABLE_PERSISTENT_CONFIG",
        "ENABLE_DIRECT_CONNECTIONS",
        "ENABLE_OLLAMA_API",
        "ENABLE_FORWARD_USER_INFO_HEADERS",
        "ENABLE_PLUGINS",
        "ENABLE_WEB_SEARCH",
    ]:
        assert env[name] == "false", name
    assert env["WEBUI_AUTH"] == "true"
    assert env["ENABLE_SIGNUP"] == "false"
    assert "WEBUI_ADMIN_PASSWORD:?" in env["WEBUI_ADMIN_PASSWORD"]
    # This is a separate opt-in stack; default startup still runs the original demo.
    default = yaml.safe_load((ROOT / "compose.yaml").read_text())
    assert "open-webui" not in default["services"]
