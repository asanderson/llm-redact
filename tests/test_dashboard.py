"""The /__llm-redact/ dashboard: serving, packaging, and non-forwarding."""

import importlib.resources

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig
from llm_redact.proxy import create_app

MARKER = "data-llm-redact-dashboard"

forwarded: list[str] = []


def _fake_upstream() -> Starlette:
    async def catch_all(request: Request) -> JSONResponse:
        forwarded.append(request.url.path)
        return JSONResponse({"ok": True})

    return Starlette(routes=[Route("/{path:path}", catch_all, methods=["GET", "POST"])])


@pytest.fixture
def client() -> httpx.AsyncClient:
    forwarded.clear()
    config = Config(
        providers={
            "anthropic": ProviderConfig(upstream_base_url="http://upstream"),
            "openai": ProviderConfig(upstream_base_url="http://upstream"),
        }
    )
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


@pytest.mark.parametrize("path", ["/__llm-redact", "/__llm-redact/"])
async def test_dashboard_served_on_both_paths(client: httpx.AsyncClient, path: str) -> None:
    response = await client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    assert MARKER in response.text
    assert "<title>llm-redact dashboard</title>" in response.text


async def test_dashboard_post_rejected(client: httpx.AsyncClient) -> None:
    assert (await client.post("/__llm-redact/", content=b"x")).status_code == 405
    assert (await client.post("/__llm-redact", content=b"x")).status_code == 405


async def test_dashboard_never_forwarded(client: httpx.AsyncClient) -> None:
    await client.get("/__llm-redact")
    await client.get("/__llm-redact/")
    await client.post("/__llm-redact/", content=b"x")
    assert forwarded == []


@pytest.mark.parametrize(
    "path",
    ["/__llm-redact/", "/__llm-redact/status", "/__llm-redact/metrics", "/__llm-redact/recent"],
)
async def test_security_headers_on_reserved_endpoints(client: httpx.AsyncClient, path: str) -> None:
    response = await client.get(path)
    assert response.status_code == 200
    csp = response.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    # The dashboard is self-contained: its inline <script>/<style> and its
    # same-origin fetch/EventSource must stay allowed by the CSP.
    assert "script-src 'unsafe-inline'" in csp
    assert "style-src 'unsafe-inline'" in csp
    assert "connect-src 'self'" in csp
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


async def test_security_headers_not_on_forwarded_traffic(client: httpx.AsyncClient) -> None:
    # Only reserved endpoints are hardened. Upstream responses pass through
    # verbatim — stamping headers there could clash with what a tool expects.
    response = await client.get("/v1/models")
    assert "content-security-policy" not in response.headers
    assert "x-frame-options" not in response.headers


@pytest.mark.parametrize("path", ["/__llm-redact/recent", "/__llm-redact/audit"])
async def test_recent_and_audit_reject_foreign_host(path: str) -> None:
    # Security review 3.1.1: /recent and /audit expose the same request
    # metadata as the already-gated /events, so a DNS-rebinding page must not
    # reach them either. A foreign Host is refused; loopback passes.
    forwarded.clear()
    config = Config(providers={"anthropic": ProviderConfig(upstream_base_url="http://upstream")})
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://attacker.example"
    ) as evil:
        assert (await evil.get(path)).status_code == 403
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as ok:
        # /audit is 404 when disabled (default) but NOT 403 — the host passed.
        assert (await ok.get(path)).status_code in (200, 404)
    assert forwarded == []


async def test_healthz_and_readyz(client: httpx.AsyncClient) -> None:
    health = await client.get("/__llm-redact/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    ready = await client.get("/__llm-redact/readyz")
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready" and "version" in body and "realtime" in body
    # DB-free liveness probes are still answered locally, never forwarded.
    assert forwarded == []


async def test_dashboard_carries_preview(client: httpx.AsyncClient) -> None:
    text = (await client.get("/__llm-redact/")).text
    assert "data-preview" in text
    assert 'id="preview-input"' in text
    assert "/__llm-redact/preview" in text


async def test_dashboard_carries_config_editor(client: httpx.AsyncClient) -> None:
    text = (await client.get("/__llm-redact/")).text
    assert "data-config-editor" in text
    assert 'id="config-form"' in text
    assert "x-llm-redact-csrf" in text  # POSTs carry the CSRF header
    assert "comments are not preserved" in text
    assert "Restart-required settings" in text
    # The NER card: backend picker plus the threshold field the JS hides
    # for spacy (parse_config rejects score_threshold there).
    # Per-backend toggles + models, and the per-type single-source note.
    assert 'id="cfg-ner-backend-spacy"' in text
    assert 'id="cfg-ner-backend-gliner"' in text
    assert 'id="cfg-ner-backend-presidio"' in text
    assert 'data-ner-model="presidio"' in text
    assert 'id="cfg-ner-folded"' in text
    assert 'id="cfg-ner-threshold"' in text
    assert 'data-ner-backend="presidio"' in text
    # Per-provider detection toggle + MCP exempt-servers editor, and the
    # status-page warning marker for providers with detection off.
    assert "data-provider-detection" in text
    assert 'id="cfg-mcp-exempt"' in text
    assert "DETECTION OFF" in text
    # Language scope input + the live effective-rule display.
    assert 'id="cfg-languages"' in text
    assert 'id="cfg-languages-effect"' in text
    # Object-store audit sinks and OTel are surfaced in the readonly card.
    assert "audit azure sink" in text
    assert 'id="otel"' in text


async def test_dashboard_carries_ops_surfacing(client: httpx.AsyncClient) -> None:
    text = (await client.get("/__llm-redact/")).text
    # Runtime pills for OTel and object-store sink drop counts.
    assert 'id="otel"' in text
    assert 'id="audit-sinks"' in text
    assert "rows_dropped" in text  # the drop count is read and shown loudly


async def test_dashboard_reads_recent_and_sessions(client: httpx.AsyncClient) -> None:
    # The recent table reads the in-memory ring buffer (so it works with
    # audit disabled — the default config), and the sessions card ships
    # with its guarded prune wiring.
    text = (await client.get("/__llm-redact/")).text
    assert "/__llm-redact/recent" in text
    assert "/__llm-redact/sessions" in text
    assert "/__llm-redact/sessions/prune" in text


async def test_dashboard_live_feed(client: httpx.AsyncClient) -> None:
    # The live SSE feed, with the 3 s poll kept as the authoritative
    # fallback (EventSource closes on a never-connected error).
    text = (await client.get("/__llm-redact/")).text
    assert "/__llm-redact/events" in text
    assert "EventSource" in text
    assert "setInterval(refresh, 3000)" in text


def test_dashboard_ships_as_package_data() -> None:
    # The same importlib.resources lookup the proxy does at startup — fails
    # if dashboard.html is ever dropped from the package.
    resource = importlib.resources.files("llm_redact").joinpath("dashboard.html")
    text = resource.read_text("utf-8")
    assert MARKER in text
    # Self-contained by design: no CDN scripts, fonts, or remote fetches.
    assert "http://" not in text
    assert "https://" not in text


def test_dashboard_element_ids_are_unique() -> None:
    # 3.2.1: a duplicate id="sessions" (status pill + table tbody) meant
    # getElementById fed session rows into the PILL while the table sat on
    # "loading…" forever — visible in the project's own screenshot. Pin
    # uniqueness for every id so the class of bug is dead.
    import re

    html = importlib.resources.files("llm_redact").joinpath("dashboard.html").read_text("utf-8")
    ids = re.findall(r'\bid="([^"]+)"', html)
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate element id(s) in dashboard.html: {sorted(duplicates)}"


def test_dashboard_fetches_render_errors_not_fake_empty_states() -> None:
    # 3.2.1: a host-gated 403 on /recent rendered as "no requests recorded
    # yet" — an operator checking for leaks would draw exactly the wrong
    # conclusion. Both gated tables must branch on response.ok.
    html = importlib.resources.files("llm_redact").joinpath("dashboard.html").read_text("utf-8")
    assert "function unavailableRow" in html
    assert html.count("unavailableRow(body,") >= 2  # recent + sessions
    assert "access restricted from this host" in html
    assert 'getElementById("session-rows")' in html


def test_dashboard_has_posture_banner_and_answerable_history() -> None:
    # Phase 28: the protection-posture line is the page's most important fact,
    # totals are labeled with the start time (they reset on restart), warn-mode
    # hits are marked as FORWARDED per request, and the recent card can read
    # the persistent audit history.
    html = importlib.resources.files("llm_redact").joinpath("dashboard.html").read_text("utf-8")
    assert 'id="posture"' in html
    assert "coverage opt-outs" in html
    assert "totals since" in html
    assert "recent-source-audit" in html


def test_dashboard_surfaces_licensed_features_package() -> None:
    # R5 open-core honesty (llm-redact-pro docs/licensing.md): the license pill reflects the
    # paid-package state, and an installed-but-unregistered package is a loud
    # posture opt-out (paid features silently OFF).
    html = importlib.resources.files("llm_redact").joinpath("dashboard.html").read_text("utf-8")
    assert "package_installed" in html
    assert "pro pkg" in html
    assert "no plugin registered" in html
    assert "FORWARDED upstream unredacted" in html  # warned column tooltip
    assert "configFingerprint" in html  # stale-form guard wired
