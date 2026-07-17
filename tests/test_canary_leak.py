"""Self-output canary leak harness.

The "the map never leaves the machine / values are never logged" promise spans
many code paths (access logging off, httpx logger raised, JsonFormatter drops
record extras, audit/metrics/recent are metadata-only, error bodies name
positions not values, `?key=` query auth is never logged). Each is tested
piecemeal elsewhere; this drives real traffic carrying planted CANARY secrets
and asserts none of them surface in ANYTHING the proxy emits about itself —
captured logs, Prometheus text, /recent, /audit, or /status. A new logging or
telemetry call site that echoes a value fails here.
"""

import json
import logging

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig
from llm_redact.proxy import create_app

# Secret-shaped fakes (never real secrets).
CANARY_EMAIL = "wilbur.canary@secret.example"
CANARY_AWS = "AKIACANARYKEY0000000"  # AKIA + 16 chars
CANARY_OPENAI = "sk-canaryKEY1234567890abcdefghij"
CANARY_QUERY = "canaryquerysecret123"  # ?key= auth value (passes through, never logged)
CANARY_DENY = "ProjectCanary"  # a deny-string value -> always redacted

ALL_CANARIES = [CANARY_EMAIL, CANARY_AWS, CANARY_OPENAI, CANARY_QUERY, CANARY_DENY]


def _echo_upstream() -> Starlette:
    async def chat(request: Request) -> Response:
        body = await request.json()
        # Echo back whatever placeholders the upstream saw, so the response
        # path (rehydration) is exercised too.
        flat = json.dumps(body, ensure_ascii=False)
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": flat}}]})

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


@pytest.mark.anyio
async def test_no_canary_secret_appears_in_any_self_output(tmp_path) -> None:
    # The audit log is a Pro feature, so the audit-surface variant of this
    # harness lives in the pro repo (tests/test_canary_leak_pro.py). Here we
    # cover the tier-independent self-output surfaces: captured logs, Prometheus
    # text, /recent, and /status.
    # Force the deny string so a value with no built-in rule is still redacted.
    from llm_redact.detection.deny import DenyEntry
    from llm_redact.detection.engine import DetectionConfig

    config = Config(
        providers={"openai": ProviderConfig(upstream_base_url="http://up")},
        detection=DetectionConfig(deny_strings=(DenyEntry(value=CANARY_DENY),)),
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_echo_upstream()))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")

    # Capture everything the proxy logs.
    buffer = logging.StreamHandler()
    import io

    stream = io.StringIO()
    buffer.setStream(stream)
    buffer.setLevel(logging.DEBUG)
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(buffer)
    # Production silences httpx's request-URL logging (query strings carry
    # `?key=` auth) via log.py:setup_logging; replicate that so the harness
    # reflects the deployed posture, not the bare test client's chatter.
    httpx_logger = logging.getLogger("httpx")
    httpx_prev = httpx_logger.level
    httpx_logger.setLevel(logging.WARNING)
    try:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"mail {CANARY_EMAIL} aws {CANARY_AWS} key {CANARY_OPENAI} "
                        f"codename {CANARY_DENY}"
                    ),
                }
            ],
        }
        # Include the ?key= auth secret in the query string.
        resp = await client.post(f"/v1/chat/completions?key={CANARY_QUERY}", json=body)
        assert resp.status_code == 200
        # The response came back rehydrated (client sees originals) — that is
        # allowed; we only forbid canaries in the proxy's OWN metadata surfaces.

        metrics = (await client.get("/__llm-redact/metrics")).text
        recent = json.dumps((await client.get("/__llm-redact/recent")).json(), ensure_ascii=False)
        status = json.dumps((await client.get("/__llm-redact/status")).json(), ensure_ascii=False)
    finally:
        root.removeHandler(buffer)
        root.setLevel(prev_level)
        httpx_logger.setLevel(httpx_prev)
    logs = stream.getvalue()

    surfaces = {
        "logs": logs,
        "metrics": metrics,
        "recent": recent,
        "status": status,
    }
    for name, text in surfaces.items():
        for canary in ALL_CANARIES:
            assert canary not in text, f"canary {canary!r} leaked into {name}: {text[:400]}"
        # Placeholder tokens (which encode nothing about the value) are fine and
        # expected in metadata; the guarantee is about VALUES, so we assert the
        # detected TYPES did get recorded (proving traffic actually flowed).
    assert "EMAIL" in status  # detection counts recorded (metadata, not values)


@pytest.mark.anyio
async def test_blocked_request_400_names_type_not_value(tmp_path) -> None:
    from llm_redact.detection.engine import DetectionConfig

    config = Config(
        providers={"openai": ProviderConfig(upstream_base_url="http://up")},
        detection=DetectionConfig(modes=(("email", "block"),)),
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_echo_upstream()))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": f"mail {CANARY_EMAIL}"}]}
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400
    # The 400 body names the blocked TYPE, never the value.
    assert CANARY_EMAIL not in resp.text
    assert "EMAIL" in resp.text or "block" in resp.text.lower()
