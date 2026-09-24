"""The browser-dashboard seam: the dashboard page, config editor and
redaction preview are an llm-redact-pro surface; the core keeps the
``DashboardHost`` methods they run on and dispatches ONLY its fixed
dashboard paths to a registered ``Dashboard``.

Without one, those paths answer a 404 naming the package (never forwarded,
never a silent blank). With one (a fake here — the real one lives in the pro
package), exactly those paths reach it, its replies get the reserved-path
security headers, and a plugin can never shadow a core endpoint. The host
methods (``validate_config``, ``preview``, the guards) are pinned directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

import llm_redact.proxy as proxy_mod
import llm_redact.registry as registry_mod
from llm_redact.config import Config, ConfigError, ProviderConfig, load_config
from llm_redact.detection.engine import DetectionConfig
from llm_redact.plugin_api import DashboardHost
from llm_redact.proxy import CSRF_HEADER, DASHBOARD_PATHS, ProxyState, create_app
from llm_redact.registry import Registry

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


def _config(detection: DetectionConfig | None = None) -> Config:
    return Config(
        providers={"anthropic": ProviderConfig(upstream_base_url="http://upstream")},
        detection=detection or DetectionConfig(),
    )


def _client(app: Starlette) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _state(app: Starlette) -> ProxyState:
    state: ProxyState = app.state.proxy
    return state


class FakeDashboard:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    async def handle(self, request: Request, host: DashboardHost) -> Response:
        self.seen.append((request.method, request.url.path))
        return JSONResponse({"served": request.url.path, "csrf": host.csrf_token})


def _install(monkeypatch: pytest.MonkeyPatch, tiers: list[str]) -> list[FakeDashboard]:
    """A bare Registry whose build_dashboard hands out a fresh fake on any
    non-free tier (the pro builder's shape) and records every call."""
    reg = Registry()
    built: list[FakeDashboard] = []

    def build_dashboard(tier: str) -> FakeDashboard | None:
        tiers.append(tier)
        if tier == "free":
            return None
        built.append(FakeDashboard())
        return built[-1]

    reg.build_dashboard = build_dashboard
    # A fake pro tier must not make the Free users factory demand the package.
    reg.build_users_store = lambda config, tier: None
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return built


# --- without the pro package ------------------------------------------------


@pytest.mark.parametrize("path", sorted(DASHBOARD_PATHS))
@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_dashboard_paths_404_naming_the_package(path: str, method: str) -> None:
    forwarded.clear()
    app = create_app(_config(), upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    assert _state(app).dashboard is None
    async with _client(app) as client:
        response = await client.request(method, path)
    assert response.status_code == 404
    error = response.json()["error"]
    assert "llm-redact-pro" in error
    assert "/__llm-redact/status" in error and "/__llm-redact/metrics" in error
    # Still a reserved reply: hardened, and never forwarded upstream.
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert forwarded == []


def test_dashboard_404_reason_names_the_actual_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(_config(), upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    state = _state(app)
    monkeypatch.setattr(proxy_mod, "pro_package_installed", lambda: False)
    assert "which is not installed" in proxy_mod._dashboard_unavailable(state)
    monkeypatch.setattr(proxy_mod, "pro_package_installed", lambda: True)
    monkeypatch.setattr(proxy_mod, "loaded_plugins", lambda: [])
    assert "did not register" in proxy_mod._dashboard_unavailable(state)
    monkeypatch.setattr(proxy_mod, "loaded_plugins", lambda: ["pro"])
    reason = proxy_mod._dashboard_unavailable(state)
    assert "Pro license key" in reason and "current tier: free" in reason


async def test_core_endpoints_still_serve_without_the_dashboard() -> None:
    # The machine APIs the CLI and monitoring read stay in the free core.
    app = create_app(_config(), upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    async with _client(app) as client:
        for path in ("status", "metrics", "healthz", "readyz", "recent", "sessions", "guide"):
            response = await client.get(f"/__llm-redact/{path}")
            assert response.status_code == 200, path


# --- with a registered dashboard -------------------------------------------


async def test_only_the_dashboard_paths_reach_the_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiers: list[str] = []
    built = _install(monkeypatch, tiers)
    monkeypatch.setattr(
        proxy_mod,
        "_resolve_license_info",
        lambda config: proxy_mod.ResolvedLicense(
            tier="pro", license=None, source="env", warnings=()
        ),
    )
    forwarded.clear()
    app = create_app(_config(), upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    assert tiers == ["pro"] and len(built) == 1
    dashboard = built[0]
    async with _client(app) as client:
        for path in sorted(DASHBOARD_PATHS):
            response = await client.post(path)
            assert response.status_code == 200
            assert response.json() == {"served": path, "csrf": _state(app).csrf_token}
            # The core stamps the reserved-path security headers on the
            # plugin's reply (setdefault: a handler's own header would win).
            assert response.headers["referrer-policy"] == "no-referrer"
        # A plugin can never shadow a core endpoint.
        status = await client.get("/__llm-redact/status")
        assert status.status_code == 200 and "version" in status.json()
    assert sorted(path for _method, path in dashboard.seen) == sorted(DASHBOARD_PATHS)
    assert forwarded == []


def test_reload_rebuilds_the_dashboard_only_on_a_tier_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiers: list[str] = []
    built = _install(monkeypatch, tiers)
    tier = {"value": "free"}
    monkeypatch.setattr(
        proxy_mod,
        "_resolve_license_info",
        lambda config: proxy_mod.ResolvedLicense(
            tier=tier["value"], license=None, source="absent", warnings=()
        ),
    )
    app = create_app(_config(), upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    state = _state(app)
    assert state.dashboard is None and tiers == ["free"]

    state.apply_config(state.config)  # same tier: no rebuild
    assert tiers == ["free"]

    tier["value"] = "pro"  # a key arrived (file + SIGHUP)
    state.apply_config(state.config)
    assert tiers == ["free", "pro"] and state.dashboard is built[0]

    state.apply_config(state.config)
    assert state.dashboard is built[0] and len(built) == 1

    tier["value"] = "free"  # the key was removed
    state.apply_config(state.config)
    assert state.dashboard is None


# --- the DashboardHost methods ----------------------------------------------


def test_proxy_state_satisfies_the_host_protocol() -> None:
    app = create_app(_config(), upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    host: DashboardHost = _state(app)  # structural: mypy checks the assignment
    assert host.csrf_token


def test_config_file_path_prefers_the_explicit_path(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("")
    app = create_app(
        load_config(config_file),
        upstream_transport=httpx.ASGITransport(app=_fake_upstream()),
        config_path=config_file,
    )
    assert _state(app).config_file_path() == config_file
    bare = create_app(_config(), upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    assert _state(bare).config_file_path().name == "config.toml"


async def test_guards_via_the_host_methods() -> None:
    app = create_app(_config(), upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    state = _state(app)
    seen: dict[str, Any] = {}

    async def probe(request: Request) -> Response:
        seen["host"] = state.host_allowed(request)
        seen["origin"] = state.origin_allowed(request)
        payload, refusal = await state.guarded_post_json(request)
        seen["payload"] = payload
        return refusal if refusal is not None else JSONResponse({"ok": True})

    probe_app = Starlette(routes=[Route("/p", probe, methods=["POST"])])
    transport = httpx.ASGITransport(app=probe_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        ok = await client.post("/p", json={"a": 1}, headers={CSRF_HEADER: state.csrf_token})
        assert ok.status_code == 200 and seen == {"host": True, "origin": True, "payload": {"a": 1}}
        missing = await client.post("/p", json={"a": 1})
        assert missing.status_code == 403
        foreign = await client.post(
            "/p",
            json={},
            headers={CSRF_HEADER: state.csrf_token, "origin": "http://evil.example"},
        )
        assert seen["origin"] is False and foreign.status_code == 200
    rebinding = httpx.AsyncClient(transport=transport, base_url="http://evil.example")
    async with rebinding as client:
        await client.post("/p", json={}, headers={CSRF_HEADER: state.csrf_token})
        assert seen["host"] is False


def test_preview_reports_redactions_by_type_without_side_effects() -> None:
    forwarded.clear()
    app = create_app(_config(), upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    state = _state(app)
    body = state.preview(f"mail {EMAIL} key {AWS_KEY}")
    assert body["detections"] == {"EMAIL": 1, "AWS_KEY": 1}
    assert body["warnings"] == {} and body["blocked"] is None
    assert EMAIL not in body["redacted"] and AWS_KEY not in body["redacted"]
    assert "«EMAIL_001»" in body["redacted"] and "«AWS_KEY_001»" in body["redacted"]
    # A throwaway vault and fresh counters: nothing live moved.
    assert state.vault_manager.total_entries() == 0
    assert dict(state.detection_counts) == {}
    assert forwarded == []


def test_preview_surfaces_warn_and_block_modes() -> None:
    detection = DetectionConfig(modes=(("phone_number", "warn"), ("private_key", "block")))
    app = create_app(
        _config(detection), upstream_transport=httpx.ASGITransport(app=_fake_upstream())
    )
    state = _state(app)
    warn = state.preview("call +14155550132 now")
    assert warn["warnings"] == {"PHONE": 1}
    assert "+14155550132" in warn["redacted"]  # honest: warn forwards the value
    blocked = state.preview(f"key {KEY_BLOCK}")
    assert blocked == {
        "redacted": None,
        "detections": {},
        "warnings": {},
        "blocked": {"type": "PRIVATE_KEY"},
    }


ROUTING_TOML = (
    '[providers.anthropic]\nupstream_base_url = "http://upstream"\n'
    '\n[upstreams.x]\nprotocol = "anthropic"\nbase_url = "http://x.example"\n'
    '\n[routing]\nenabled = false\ndefault_upstream = "x"\n'
)


def _load(tmp_path: Path, text: str) -> Config:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return load_config(path)


def test_validate_config_rejects_what_a_hot_apply_would_fail_on(tmp_path: Path) -> None:
    app = create_app(_config(), upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    state = _state(app)
    state.validate_config(_config())  # a good candidate passes silently
    bad_regex = _load(
        tmp_path,
        '[[detection.custom_rules]]\nname = "x"\ntype = "X"\npattern = "("\n',
    )
    with pytest.raises((ValueError, re.error)):
        state.validate_config(bad_regex)
    unknown_rule = _load(tmp_path, '[detection]\nenabled = ["no_such_rule"]\n')
    with pytest.raises(ValueError):
        state.validate_config(unknown_rule)
    # Validation never swaps anything.
    assert state.config == _config() or state.config.detection == _config().detection


def test_validate_config_hands_the_candidate_to_the_running_router(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fake_router import FakeRouter, install

    router = FakeRouter()
    install(monkeypatch, router)
    enabled = _load(tmp_path, ROUTING_TOML.replace("enabled = false", "enabled = true"))
    app = create_app(enabled, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    state = _state(app)
    candidate = _load(
        tmp_path,
        ROUTING_TOML.replace("enabled = false", "enabled = true")
        + "\n[rehydration]\nfuzzy = false\n",
    )
    state.validate_config(candidate)
    assert [c.rehydration.fuzzy for c in router.validated] == [False]
    assert router.reconfigured == []  # a dry run: nothing applied
    router.validate_error = ConfigError("price table: nope")
    with pytest.raises(ConfigError, match="price table: nope"):
        state.validate_config(candidate)


def test_validate_config_probes_and_closes_when_the_file_enabled_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fake_router import FakeRouter, install

    created: list[FakeRouter] = []

    def make() -> FakeRouter:
        created.append(FakeRouter())
        return created[-1]

    _reg, calls = install(monkeypatch, make)
    app = create_app(
        _load(tmp_path, ROUTING_TOML), upstream_transport=httpx.ASGITransport(app=_fake_upstream())
    )
    state = _state(app)
    assert state.router is None and len(calls) == 1
    state.validate_config(
        _load(tmp_path, ROUTING_TOML.replace("enabled = false", "enabled = true"))
    )
    # The dry run PROBED the build and closed the probe; nothing was held.
    assert len(created) == 1 and created[0].closed == 1
    assert state.router is None


def test_validate_config_probe_refusal_names_the_package(tmp_path: Path) -> None:
    # Default registry: the Free factory refuses an enabled [routing] naming
    # llm-redact-pro — raised by the dry run, before any write.
    app = create_app(
        _load(tmp_path, ROUTING_TOML), upstream_transport=httpx.ASGITransport(app=_fake_upstream())
    )
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        _state(app).validate_config(
            _load(tmp_path, ROUTING_TOML.replace("enabled = false", "enabled = true"))
        )
