"""Consolidated red-team boundary suite.

`docs/threat-model.md` names a handful of trust boundaries and, at each, the
gate that holds it. Those gates are enforced in a few lines of `proxy.py` and
`config.py`; a refactor can silently weaken one (drop a Host check, forget to
stamp a header, widen a cap) and every OTHER test still passes because they
exercise the happy path. This file is the adversary's view: one test per
threat-model boundary, each named for the attack it repels and cross-referenced
to the doc section it guards, so a weakened gate fails HERE even when the
feature it protects still works.

Individual endpoints have their own deep suites (`test_sessions_endpoint.py`,
`test_tls.py`, `test_proxy_integration.py`; the llm-redact-pro dashboard's
config editor and preview carry the same map in that package's
`test_security_boundaries_dashboard.py`); this file deliberately overlaps
them — the value is the single, complete map from boundary to guard, and
parametrization over EVERY guarded endpoint so a newly added one that forgets a
layer is caught.

Boundary map (threat-model.md § / guard):
  B1  Local ops surface / reserved paths answered before routing (never forwarded)
  B2  Local ops surface / Host validation (DNS rebinding) — 403
  B3  Local ops surface / Origin validation — 403
  B4  Local ops surface / per-process CSRF token — 403
  B5  Local ops surface / CORS preflight dies (OPTIONS -> 405, no CORS headers)
  B6  Local ops surface / JSON content-type required — 415
  B7  Local ops surface / 1 MiB guarded-POST body cap — 413
  B8  Outbound requests / max_body_bytes + max_body_strings fail-closed — 413, never forwarded
  B9  Local ops surface / browser-hardening headers on every reserved reply
  B10 Trust boundaries / fail-closed bind policy (validate_bind_security)
  B11 Logging posture / ?key= query auth never logged (cross-ref canary harness)
  B12 Local ops surface / metadata-only (status/metrics never carry values)
  B17 Requests from web pages / API routes refuse cross-site, rebound and foreign-Host
      browser requests (and foreign-Host requests spending a proxy-held credential) — 403
"""

import io
import json
import logging

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from llm_redact.config import (
    Config,
    ConfigError,
    ProviderConfig,
    TlsConfig,
    validate_bind_security,
)
from llm_redact.detection.deny import DenyEntry
from llm_redact.detection.engine import DetectionConfig
from llm_redact.proxy import (
    _SECURITY_HEADERS,
    CSRF_HEADER,
    DASHBOARD_PATHS,
    RESERVED_PREFIX,
    create_app,
)

LOOPBACK = "http://127.0.0.1:8787"

# The guarded mutating endpoints share ONE guard chain in the code
# (`_guarded_post_json`, wrapped by per-endpoint Host/Origin checks). A minimal
# valid body per endpoint lets us prove each layer independently of the
# endpoint's own payload validation, which runs strictly after the guards.
# (The core's user invite/revoke POSTs answer 403 before the chain without
# the llm-redact-pro users registry; the dashboard's /config and /preview
# POSTs are pinned by the pro package's own boundary suite.)
GUARDED_POSTS = {
    f"{RESERVED_PREFIX}/sessions/prune": {"older_than_days": 30},
}

# Every local endpoint that consults Host before doing anything — the GET
# reads plus the guarded POSTs.
HOST_GATED = [
    f"{RESERVED_PREFIX}/sessions",
    f"{RESERVED_PREFIX}/recent",
    f"{RESERVED_PREFIX}/events",
    f"{RESERVED_PREFIX}/audit",
    f"{RESERVED_PREFIX}/users",
    *GUARDED_POSTS,
]


def _echo_upstream() -> Starlette:
    """A fake provider that reflects the redacted body back, so we can prove a
    reserved path was NEVER forwarded (its bytes would show up if it had been)."""

    async def chat(request: Request) -> Response:
        text = (await request.body()).decode("utf-8", "replace")
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": text}}]})

    async def catch_all(request: Request) -> Response:
        return JSONResponse({"seen_path": request.url.path})

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/{path:path}", catch_all, methods=["GET", "POST"]),
        ]
    )


def _client(config: Config, *, base_url: str = LOOPBACK) -> httpx.AsyncClient:
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_echo_upstream()))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)
    _tokens[id(client)] = app.state.proxy.csrf_token
    return client


# The per-process CSRF token. The llm-redact-pro dashboard hands it to
# same-origin pages; without that package there is no page, so the positive
# controls read it off the live ProxyState.
_tokens: dict[int, str] = {}


def _base_config(**overrides) -> Config:
    return Config(providers={"openai": ProviderConfig(upstream_base_url="http://up")}, **overrides)


async def _csrf(client: httpx.AsyncClient) -> str:
    return _tokens[id(client)]


# --- B1: reserved paths answered before routing, provably never forwarded ----


@pytest.mark.anyio
async def test_b1_reserved_paths_never_reach_upstream() -> None:
    """Threat-model § Local ops surface: reserved replies are produced before
    any routing/upstream code. The echo upstream tags every path it sees; a
    reserved path must never carry that tag."""
    client = _client(_base_config())
    for path in ("/status", "/metrics", "/recent", "/sessions"):
        resp = await client.get(f"{RESERVED_PREFIX}{path}")
        assert resp.status_code == 200, path
        assert "seen_path" not in resp.text, f"{path} was forwarded upstream"
    # The dashboard paths without llm-redact-pro: a local 404 naming the
    # package — still answered before routing, never forwarded.
    for path in sorted(DASHBOARD_PATHS):
        for method in ("GET", "POST"):
            resp = await client.request(method, path)
            assert resp.status_code == 404, path
            assert "llm-redact-pro" in resp.json()["error"], path
            assert "seen_path" not in resp.text, f"{path} was forwarded upstream"


# --- B2: Host validation (DNS rebinding) -------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("path", HOST_GATED)
async def test_b2_hostile_host_rejected(path: str) -> None:
    """Threat-model § Local ops surface (DNS rebinding): a rebinding page
    reaches 127.0.0.1 but its requests carry the attacker's domain in Host.
    Every host-gated endpoint returns 403 before doing anything — including
    before handing out the CSRF token."""
    client = _client(_base_config(), base_url="http://evil.example")
    method = "post" if path in GUARDED_POSTS else "get"
    resp = await getattr(client, method)(
        path, **({"headers": {CSRF_HEADER: "x"}, "json": {}} if method == "post" else {})
    )
    assert resp.status_code == 403, path


# --- B3: Origin validation ---------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("origin", ["https://evil.example", "http://evil.example", "null"])
async def test_b3_hostile_origin_rejected(origin: str) -> None:
    """Threat-model § Local ops surface: a present Origin must be a local
    origin. `null` (sandboxed iframe / file://) and any remote origin are
    refused. Without TLS, even an https loopback origin is refused."""
    client = _client(_base_config())
    resp = await client.get(f"{RESERVED_PREFIX}/sessions", headers={"origin": origin})
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_b3_https_origin_refused_without_tls() -> None:
    """The scheme is pinned to the proxy's own: an https Origin is only
    acceptable when the proxy itself serves TLS."""
    client = _client(_base_config())
    resp = await client.get(
        f"{RESERVED_PREFIX}/sessions", headers={"origin": "https://127.0.0.1:8787"}
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_b3_local_origin_accepted() -> None:
    client = _client(_base_config())
    resp = await client.get(f"{RESERVED_PREFIX}/sessions", headers={"origin": LOOPBACK})
    assert resp.status_code == 200


# --- B4: per-process CSRF token ----------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("path", list(GUARDED_POSTS))
async def test_b4_missing_or_wrong_csrf_rejected(path: str) -> None:
    """Threat-model § Local ops surface: every guarded POST requires the
    per-process CSRF token in a custom header — readable only via a
    same-origin GET. Missing and wrong both 403."""
    client = _client(_base_config())
    body = GUARDED_POSTS[path]
    missing = await client.post(path, json=body)
    assert missing.status_code == 403, f"{path}: missing token not rejected"
    wrong = await client.post(path, headers={CSRF_HEADER: "not-the-token"}, json=body)
    assert wrong.status_code == 403, f"{path}: wrong token not rejected"


@pytest.mark.anyio
@pytest.mark.parametrize("path", list(GUARDED_POSTS))
async def test_b4_valid_csrf_passes_the_gate(path: str) -> None:
    """The positive control: with the real token the request clears the guard
    chain (it may then 200 or fail on its own payload rules — never 403/415)."""
    client = _client(_base_config())
    token = await _csrf(client)
    resp = await client.post(path, headers={CSRF_HEADER: token}, json=GUARDED_POSTS[path])
    assert resp.status_code not in (403, 415), f"{path}: valid token blocked ({resp.status_code})"


# --- B5: CORS preflight dies -------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("path", list(GUARDED_POSTS))
async def test_b5_options_preflight_405_no_cors(path: str) -> None:
    """Threat-model § Local ops surface: the custom CSRF header forces a CORS
    preflight for any cross-origin fetch. The proxy answers OPTIONS with 405
    and emits NO `access-control-*` headers, so the preflight fails and the
    real request is never sent."""
    client = _client(_base_config())
    resp = await client.options(
        path,
        headers={
            "origin": LOOPBACK,
            "access-control-request-method": "POST",
            "access-control-request-headers": CSRF_HEADER,
        },
    )
    assert resp.status_code == 405, path
    assert not any(name.lower().startswith("access-control-") for name in resp.headers)


# --- B6: JSON content-type required ------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("path", list(GUARDED_POSTS))
async def test_b6_non_json_content_type_415(path: str) -> None:
    """Threat-model § Local ops surface: a JSON content-type is required, so a
    simple-request `text/plain` form POST (which needs no preflight) cannot
    reach the handler."""
    client = _client(_base_config())
    token = await _csrf(client)
    resp = await client.post(
        path,
        headers={CSRF_HEADER: token, "content-type": "text/plain"},
        content=json.dumps(GUARDED_POSTS[path]).encode(),
    )
    assert resp.status_code == 415, path


# --- B7: 1 MiB guarded-POST body cap -----------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("path", list(GUARDED_POSTS))
async def test_b7_guarded_post_body_cap_413(path: str) -> None:
    """Threat-model § Local ops surface: guarded POSTs are capped at 1 MiB,
    read incrementally so a lying Content-Length cannot exhaust memory."""
    client = _client(_base_config())
    token = await _csrf(client)
    huge = {"text": "x" * (1024 * 1024 + 16), "config": {}, "older_than_days": 1}
    resp = await client.post(path, headers={CSRF_HEADER: token}, json=huge)
    assert resp.status_code == 413, path


# --- B8: max_body_bytes fail-closed on the redaction path --------------------


@pytest.mark.anyio
async def test_b8_oversized_redactable_body_413_not_forwarded() -> None:
    """Threat-model § Outbound requests: a redactable body too large to buffer
    is rejected 413 rather than forwarded unredacted. The tiny cap here makes a
    normal body oversized; the echo upstream would reflect it if it leaked."""
    client = _client(_base_config(max_body_bytes=64))
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "x" * 500}]}
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 413
    assert "x" * 500 not in resp.text  # never round-tripped through the upstream


@pytest.mark.anyio
async def test_b8_too_many_strings_413_not_forwarded() -> None:
    """Threat-model § Outbound requests: redaction costs per string on the
    event loop, so a body of more strings than max_body_strings is refused
    413 — never forwarded, partly redacted or not."""
    client = _client(_base_config(max_body_strings=10))
    messages = [{"role": "user", "content": f"line {i} x@corp.example"} for i in range(11)]
    resp = await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": messages})
    assert resp.status_code == 413
    assert "max_body_strings (10)" in resp.text
    assert "x@corp.example" not in resp.text  # never round-tripped through the upstream


@pytest.mark.anyio
async def test_b8_normal_body_under_cap_forwarded() -> None:
    """Positive control: a body under the cap flows (and comes back redacted)."""
    client = _client(
        _base_config(detection=DetectionConfig(deny_strings=(DenyEntry(value="ProjectX"),)))
    )
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "codename ProjectX"}]}
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200


# --- B9: browser-hardening headers on every reserved reply -------------------


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["/", "/status", "/metrics", "/recent", "/config"])
async def test_b9_security_headers_on_every_reserved_reply(path: str) -> None:
    """Threat-model § Local ops surface: every reserved reply carries the full
    browser-hardening header set (strict CSP, X-Frame-Options DENY, nosniff,
    no-referrer), stamped in ONE place so it cannot drift per handler."""
    client = _client(_base_config())
    resp = await client.get(f"{RESERVED_PREFIX}{path}" if path != "/" else f"{RESERVED_PREFIX}/")
    for header, expected in _SECURITY_HEADERS.items():
        assert resp.headers.get(header) == expected, f"{path} missing {header}"
    # The CSP must actually forbid remote code and framing.
    csp = resp.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp


@pytest.mark.anyio
async def test_b9_forwarded_traffic_is_never_stamped() -> None:
    """The hardening headers belong to the proxy's OWN pages; genuine provider
    responses pass through untouched (stamping them could break a client)."""
    client = _client(_base_config())
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert "content-security-policy" not in resp.headers


# --- B10: fail-closed bind policy --------------------------------------------


def _tls_full() -> TlsConfig:
    return TlsConfig(certfile="/c", keyfile="/k", client_ca="/ca")


def test_b10_loopback_binds_freely() -> None:
    """Threat-model § Trust boundaries: loopback is the default and safe."""
    for host in ("127.0.0.1", "localhost", "::1"):
        validate_bind_security(host, TlsConfig(), {})  # no raise


def test_b10_non_loopback_without_mtls_refused() -> None:
    """A non-loopback bind exposes rehydrated values and the config editor to
    the network, so it demands FULL mutual TLS."""
    with pytest.raises(ConfigError) as exc:
        validate_bind_security("0.0.0.0", TlsConfig(), {})
    assert "mutual TLS" in str(exc.value)


def test_b10_non_loopback_server_only_tls_still_refused() -> None:
    """Server-only TLS (no client_ca) is NOT enough off loopback — a network
    client could still read secrets over the encrypted channel."""
    with pytest.raises(ConfigError):
        validate_bind_security("0.0.0.0", TlsConfig(certfile="/c", keyfile="/k"), {})


def test_b10_non_loopback_with_full_mtls_allowed() -> None:
    validate_bind_security("0.0.0.0", _tls_full(), {})  # no raise


def test_b10_insecure_bind_hatch() -> None:
    """The container's confined-bind escape hatch, honored only when set."""
    validate_bind_security("0.0.0.0", TlsConfig(), {"LLM_REDACT_INSECURE_BIND": "1"})
    with pytest.raises(ConfigError):
        validate_bind_security("0.0.0.0", TlsConfig(), {"LLM_REDACT_INSECURE_BIND": "0"})


def test_b10_unresolvable_hostname_is_non_loopback() -> None:
    """A hostname we cannot PROVE is loopback fails closed (treated as a wider
    bind), so it demands mutual TLS."""
    with pytest.raises(ConfigError):
        validate_bind_security("not-a-loopback.example", TlsConfig(), {})
    validate_bind_security("not-a-loopback.example", _tls_full(), {})  # no raise


# --- B11: ?key= query auth never logged (cross-ref canary harness) -----------


@pytest.mark.anyio
async def test_b11_query_auth_never_logged() -> None:
    """Threat-model § Logging posture: Gemini and others carry `?key=` auth in
    the query string. The proxy's own log lines carry path + status + counts,
    never the query. (The canary harness proves this across every self-output
    surface; this is the focused request-path assertion.)"""
    secret = "querysecret_ABC123"
    client = _client(_base_config())

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    prev = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    # Mirror production: httpx's request-URL INFO line is silenced (log.py).
    httpx_logger = logging.getLogger("httpx")
    httpx_prev = httpx_logger.level
    httpx_logger.setLevel(logging.WARNING)
    try:
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        resp = await client.post(f"/v1/chat/completions?key={secret}", json=body)
        assert resp.status_code == 200
    finally:
        root.removeHandler(handler)
        root.setLevel(prev)
        httpx_logger.setLevel(httpx_prev)
    assert secret not in stream.getvalue()


# --- B12: metadata-only ops surfaces -----------------------------------------


@pytest.mark.anyio
async def test_b12_status_and_metrics_carry_no_values() -> None:
    """Threat-model § Local ops surface: status/metrics expose types and counts
    only. Even the allowlist (a configured value list) must not surface there —
    the (llm-redact-pro) config-editor GET is the single documented exception
    for allowlists, pinned in that package."""
    secret_allow = "allowlisted.person@corp.example"
    client = _client(_base_config(detection=DetectionConfig(allowlist=(secret_allow,))))
    # Drive traffic so counters populate.
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "mail a@b.example"}]}
    await client.post("/v1/chat/completions", json=body)

    status = (await client.get(f"{RESERVED_PREFIX}/status")).text
    metrics = (await client.get(f"{RESERVED_PREFIX}/metrics")).text
    assert secret_allow not in status
    assert secret_allow not in metrics


# --- B13 (1.16.0): disabled provider fails closed on inferred MEDIA paths ----


@pytest.mark.anyio
@pytest.mark.parametrize(
    "path",
    ["/v1/images/generations", "/v1/images/variations", "/v1/audio/speech", "/v1/videos"],
)
async def test_b13_disabled_provider_covers_media_paths(path: str) -> None:
    """Threat-model § Fail-closed provider disable: the 1.15.0 media routes
    infer as openai traffic, so [providers.openai] enabled = false must 502
    them BEFORE any body is read — matched AND pass-through shapes alike."""
    config = Config(
        providers={"openai": ProviderConfig(upstream_base_url="http://up", enabled=False)}
    )
    client = _client(config)
    resp = await client.post(path, json={"prompt": "secret jane.doe@corp.example"})
    assert resp.status_code == 502
    assert "disabled" in resp.text
    # The echo upstream would have reported seen_path had it been forwarded.
    assert "seen_path" not in resp.text


# --- B14 (1.16.0): multipart media prompts cannot dodge the scan -------------


@pytest.mark.anyio
async def test_b14_multipart_prompt_dressed_as_file_still_redacted() -> None:
    """A `prompt` part carrying a filename attribute must still be scanned:
    matching by NAME (not by part kind) is what keeps the media multipart
    branch fail-closed against dressed-up fields."""
    received: dict[str, bytes] = {}

    async def edits(request: Request) -> Response:
        received["raw"] = await request.body()
        return JSONResponse({"created": 1, "data": []})

    upstream = Starlette(routes=[Route("/v1/images/edits", edits, methods=["POST"])])
    app_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=create_app(_base_config(), upstream_transport=httpx.ASGITransport(app=upstream))
        ),
        base_url=LOOPBACK,
    )
    resp = await app_client.post(
        "/v1/images/edits",
        files={
            "prompt": ("innocent.txt", b"a card for jane.doe@corp.example", "text/plain"),
            "image": ("in.png", b"\x89PNG fake", "image/png"),
        },
    )
    assert resp.status_code == 200
    assert b"jane.doe@corp.example" not in received["raw"]
    assert "«EMAIL_001»".encode() in received["raw"]
    assert b"\x89PNG fake" in received["raw"]  # media part untouched


# --- B15 (1.16.0): block mode rejects a media multipart WHOLE request --------


@pytest.mark.anyio
async def test_b15_block_mode_rejects_media_multipart_before_upstream() -> None:
    """One blocked value anywhere in a media upload rejects the WHOLE request
    with a provider-shaped 400 before any upstream contact."""
    reached = {"upstream": False}

    async def edits(request: Request) -> Response:
        reached["upstream"] = True
        return JSONResponse({"created": 1})

    upstream = Starlette(routes=[Route("/v1/images/edits", edits, methods=["POST"])])
    config = _base_config(detection=DetectionConfig(modes=(("email", "block"),)))
    app_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=create_app(config, upstream_transport=httpx.ASGITransport(app=upstream))
        ),
        base_url=LOOPBACK,
    )
    resp = await app_client.post(
        "/v1/images/edits",
        data={"prompt": "mail jane.doe@corp.example"},
        files={"image": ("in.png", b"\x89PNG", "image/png")},
    )
    assert resp.status_code == 400
    assert "jane.doe@corp.example" not in resp.text  # type only, never the value
    assert reached["upstream"] is False


# --- B16: the request target is a path, and the upstream is the configured one


async def _raw_asgi(app, raw_path: bytes, path: str) -> tuple[int, bytes]:
    """Drive the app with a hand-built scope: an HTTP client can't send a
    non-origin-form target through httpx, but a raw socket client can."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": raw_path,
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1:8787"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8787),
    }
    messages = [{"type": "http.request", "body": b"{}", "more_body": False}]
    sent: list[dict] = []

    async def receive() -> dict:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    await app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("raw_path", "path", "expected"),
    [
        # userinfo@host smuggled through a percent-encoded target that is not
        # a path: the decoded path routes, but joined onto the base URL the
        # raw target would name a different host. handle() refuses it.
        (b"%2Fv1%2Fprojects%2Fx@evil.example:1/steal", "/v1/projects/x@evil.example:1/steal", 400),
        # Targets whose decoded path is not a path either never match the
        # catch-all route: Starlette answers them (404, or a 307 to the
        # proxy's OWN root) before handle() runs.
        (b"@evil.example/v1/chat/completions", "@evil.example/v1/chat/completions", None),
        (b"", "", None),
    ],
)
async def test_b16_non_path_request_target_is_never_forwarded(
    raw_path: bytes, path: str, expected: int | None
) -> None:
    reached: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        reached.append(str(request.url))
        return httpx.Response(200, json={})

    app = create_app(_base_config(), upstream_transport=httpx.MockTransport(handler))
    status, body = await _raw_asgi(app, raw_path, path)
    if expected is not None:
        assert status == expected
        assert b"must be a path" in body
    else:
        assert status in (307, 404)
    assert reached == []


def test_b16_built_url_must_stay_on_the_configured_upstream() -> None:
    from llm_redact.proxy import _same_upstream

    assert _same_upstream("http://up/v1/chat/completions?x=1", "http://up")
    assert _same_upstream("https://h.example:8443/base/a", "https://h.example:8443/base")
    # The base URL's path is part of the upstream: a sibling path on the same
    # host (another API behind one gateway) is a different upstream.
    assert not _same_upstream("https://h.example:8443/a", "https://h.example:8443/base")
    assert not _same_upstream("https://h.example:8443/basement", "https://h.example:8443/base")
    assert not _same_upstream("http://up@evil.example/v1", "http://up")
    assert not _same_upstream("http://up.evil.example/v1", "http://up")
    assert not _same_upstream("http://up:1/v1", "http://up")
    assert not _same_upstream("https://up/v1", "http://up")
    assert not _same_upstream("http://[::1/v1", "http://up")  # unparseable


def test_b16_origin_form_target_reads_the_raw_target() -> None:
    from llm_redact.proxy import origin_form_target

    assert origin_form_target({"raw_path": b"/v1/x", "path": "/v1/x"})
    assert not origin_form_target({"raw_path": b"%2Fv1", "path": "/v1"})
    assert origin_form_target({"path": "/v1"})  # servers that omit raw_path
    assert not origin_form_target({"path": "v1"})
    assert not origin_form_target({})


# --- B17: a web page never drives an API route (threat-model.md § "Requests from
# web pages") -------------------------------------------------------------------
#
# Every forwarded request borrows the vault (a rehydrating route restores the
# operator's values into whatever the upstream echoes — a page with its OWN
# key could read them back), and some borrow a credential the proxy holds. A
# page reaches 127.0.0.1 with a "simple" POST, a CORS request whose preflight
# used to be forwarded, a WebSocket handshake (no CORS at all), or — after DNS
# rebinding — same-origin reads. The WebSocket twin lives in
# test_realtime_request_origin.py; the proxy-held-credential cases (identity
# auth, routed operator keys) in test_request_origin.py.

B17_API = "/v1/chat/completions"
B17_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "https://evil.example"},  # CSRF / CORS
        {"origin": "null"},  # a sandboxed frame
        {"origin": "http://127.0.0.1:3000"},  # another port of this machine
        {"sec-fetch-site": "cross-site"},
        {"sec-fetch-site": "same-site"},
    ],
)
async def test_b17_cross_origin_page_never_reaches_an_upstream(headers: dict[str, str]) -> None:
    async with _client(_base_config()) as client:
        post = await client.post(B17_API, json=B17_BODY, headers=headers)
        preflight = await client.options(
            B17_API, headers={**headers, "access-control-request-method": "POST"}
        )
    for response in (post, preflight):
        assert response.status_code == 403
        assert "choices" not in response.text and "seen_path" not in response.text
        assert not any(h.startswith("access-control-") for h in response.headers)


async def test_b17_rebound_page_never_reaches_an_upstream() -> None:
    # Same-origin to the browser after DNS rebinding: only the Host is foreign.
    async with _client(_base_config(), base_url="http://rebind.example:8787") as client:
        response = await client.get("/v1/models", headers={"sec-fetch-site": "same-origin"})
    assert response.status_code == 403
    assert "seen_path" not in response.text


async def test_b17_tools_and_the_proxys_own_origin_are_served() -> None:
    # CLI tools and SDKs send no browser markers: an alias host (a compose
    # service) is served on the client's own credential, as is a page the
    # proxy itself served.
    async with _client(_base_config(), base_url="http://llm-redact:8787") as client:
        tool = await client.post(B17_API, json=B17_BODY)
    async with _client(_base_config()) as client:
        own = await client.post(
            B17_API,
            json=B17_BODY,
            headers={"origin": LOOPBACK, "sec-fetch-site": "same-origin"},
        )
    assert (tool.status_code, own.status_code) == (200, 200)
    assert "choices" in tool.text and "choices" in own.text


async def test_b17_only_the_operators_listed_origin_is_served() -> None:
    # allowed_origins is the operator's explicit grant: the exact listed
    # origin reaches the API routes; a look-alike does not, a rebound Host
    # does not, and the listing never opens the reserved endpoints.
    listed = "https://chat.example.com"
    config = _base_config(allowed_origins=(listed,))
    cross = {"origin": listed, "sec-fetch-site": "cross-site"}
    async with _client(config) as client:
        api = await client.post(B17_API, json=B17_BODY, headers=cross)
        look_alike = await client.post(
            B17_API,
            json=B17_BODY,
            headers={**cross, "origin": "https://chat.example.com.evil.example"},
        )
        prune = await client.post(
            f"{RESERVED_PREFIX}/sessions/prune",
            json={"older_than_days": 1},
            headers={**cross, CSRF_HEADER: await _csrf(client)},
        )
    async with _client(config, base_url="http://rebind.example:8787") as client:
        rebound = await client.post(B17_API, json=B17_BODY, headers=cross)
    assert api.status_code == 200 and "choices" in api.text
    assert (look_alike.status_code, prune.status_code, rebound.status_code) == (403, 403, 403)
