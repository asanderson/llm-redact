"""POST /__llm-redact/preview: a config dry-run over caller-supplied text
that runs the live detection pipeline with no upstream/vault/metrics/audit
side effects, behind the same Host/Origin/CSRF guard as the config editor."""

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig
from llm_redact.detection.engine import DetectionConfig
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
KEY_BLOCK = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAB3NzaC1yc2E\n-----END OPENSSH PRIVATE KEY-----"
)

forwarded: list[str] = []


def _fake_upstream() -> Starlette:
    async def catch(request: Request) -> JSONResponse:
        forwarded.append(request.url.path)
        return JSONResponse({"ok": True})

    return Starlette(routes=[Route("/{path:path}", catch, methods=["GET", "POST"])])


def _app(detection: DetectionConfig | None = None) -> Starlette:
    config = Config(
        providers={"anthropic": ProviderConfig(upstream_base_url="http://upstream")},
        detection=detection or DetectionConfig(),
    )
    return create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))


@pytest.fixture
def client() -> httpx.AsyncClient:
    forwarded.clear()
    app = _app()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


async def _csrf(client: httpx.AsyncClient) -> str:
    return (await client.get("/__llm-redact/config")).json()["csrf_token"]


async def _preview(client: httpx.AsyncClient, text: str) -> httpx.Response:
    token = await _csrf(client)
    return await client.post(
        "/__llm-redact/preview",
        json={"text": text},
        headers={"x-llm-redact-csrf": token},
    )


@pytest.mark.anyio
async def test_preview_reports_redactions_by_type(client: httpx.AsyncClient) -> None:
    response = await _preview(client, f"mail {EMAIL} key {AWS_KEY}")
    assert response.status_code == 200
    body = response.json()
    assert body["detections"] == {"EMAIL": 1, "AWS_KEY": 1}
    assert body["blocked"] is None
    # The redacted text shows placeholders, never the originals.
    assert EMAIL not in body["redacted"] and AWS_KEY not in body["redacted"]
    assert "«EMAIL_001»" in body["redacted"] and "«AWS_KEY_001»" in body["redacted"]
    await client.aclose()


@pytest.mark.anyio
async def test_preview_has_no_side_effects(client: httpx.AsyncClient) -> None:
    # Run a preview, then confirm nothing persisted: no vault entries, no
    # detection totals, no forwarded upstream request.
    await _preview(client, f"mail {EMAIL}")
    status = (await client.get("/__llm-redact/status")).json()
    assert status["vault"]["entries"] == 0
    assert status["detections_total"] == {}
    assert forwarded == []
    await client.aclose()


@pytest.mark.anyio
async def test_preview_surfaces_warn_and_block_modes() -> None:
    detection = DetectionConfig(
        modes=(("phone_number", "warn"), ("private_key", "block")),
    )
    app = _app(detection)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")

    # Warn mode: the value stays in the redacted text (it WOULD be forwarded).
    warn = (await _preview(client, "call +14155550132 now")).json()
    assert warn["warnings"] == {"PHONE": 1}
    assert "+14155550132" in warn["redacted"]  # honest: warn forwards the value

    # Block mode: a real request would be a 400; preview reports the type only.
    blocked = (await _preview(client, f"key {KEY_BLOCK}")).json()
    assert blocked["blocked"] == {"type": "PRIVATE_KEY"}
    assert blocked["redacted"] is None
    await client.aclose()


@pytest.mark.anyio
async def test_preview_guard_chain(client: httpx.AsyncClient) -> None:
    # No CSRF header → 403; wrong method → 405; bad body → 400.
    no_csrf = await client.post("/__llm-redact/preview", json={"text": "x"})
    assert no_csrf.status_code == 403
    assert (await client.get("/__llm-redact/preview")).status_code == 405
    token = await _csrf(client)
    bad = await client.post(
        "/__llm-redact/preview", json={"nottext": 1}, headers={"x-llm-redact-csrf": token}
    )
    assert bad.status_code == 400
    # Never forwarded upstream regardless.
    assert forwarded == []
    await client.aclose()
