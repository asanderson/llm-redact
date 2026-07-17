"""Consolidated red-team boundary suite.

`docs/threat-model.md` names a handful of trust boundaries and, at each, the
gate that holds it. Those gates are enforced in a few lines of `proxy.py` and
`config.py`; a refactor can silently weaken one (drop a Host check, forget to
stamp a header, widen a cap) and every OTHER test still passes because they
exercise the happy path. This file is the adversary's view: one test per
threat-model boundary, each named for the attack it repels and cross-referenced
to the doc section it guards, so a weakened gate fails HERE even when the
feature it protects still works.

Individual endpoints have their own deep suites (`test_config_endpoint.py`,
`test_tls.py`, `test_proxy_integration.py`); this file deliberately overlaps
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
  B8  Outbound requests / max_body_bytes fail-closed — 413, never forwarded
  B9  Local ops surface / browser-hardening headers on every reserved reply
  B10 Trust boundaries / fail-closed bind policy (validate_bind_security)
  B11 Logging posture / ?key= query auth never logged (cross-ref canary harness)
  B12 Local ops surface / metadata-only (status/metrics never carry values)
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
    RESERVED_PREFIX,
    create_app,
)

LOOPBACK = "http://127.0.0.1:8787"

# The three guarded mutating endpoints share ONE guard chain in the code
# (`_guarded_post_json`, wrapped by per-endpoint Host/Origin checks). A minimal
# valid body per endpoint lets us prove each layer independently of the
# endpoint's own payload validation, which runs strictly after the guards.
GUARDED_POSTS = {
    f"{RESERVED_PREFIX}/config": {"config": {}},
    f"{RESERVED_PREFIX}/sessions/prune": {"older_than_days": 30},
    f"{RESERVED_PREFIX}/preview": {"text": "hello"},
}

# Every local endpoint that consults Host/Origin before doing anything —
# the GET reads plus the guarded POSTs.
HOST_GATED = [
    f"{RESERVED_PREFIX}/config",
    f"{RESERVED_PREFIX}/sessions",
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
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


def _base_config(**overrides) -> Config:
    return Config(providers={"openai": ProviderConfig(upstream_base_url="http://up")}, **overrides)


async def _csrf(client: httpx.AsyncClient) -> str:
    resp = await client.get(f"{RESERVED_PREFIX}/config")
    assert resp.status_code == 200
    return str(resp.json()["csrf_token"])


# --- B1: reserved paths answered before routing, provably never forwarded ----


@pytest.mark.anyio
async def test_b1_reserved_paths_never_reach_upstream() -> None:
    """Threat-model § Local ops surface: reserved replies are produced before
    any routing/upstream code. The echo upstream tags every path it sees; a
    reserved path must never carry that tag."""
    client = _client(_base_config())
    for path in ("/status", "/metrics", "/recent", "/config"):
        resp = await client.get(f"{RESERVED_PREFIX}{path}")
        assert resp.status_code == 200, path
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
    resp = await client.get(f"{RESERVED_PREFIX}/config", headers={"origin": origin})
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_b3_https_origin_refused_without_tls() -> None:
    """The scheme is pinned to the proxy's own: an https Origin is only
    acceptable when the proxy itself serves TLS."""
    client = _client(_base_config())
    resp = await client.get(
        f"{RESERVED_PREFIX}/config", headers={"origin": "https://127.0.0.1:8787"}
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_b3_local_origin_accepted() -> None:
    client = _client(_base_config())
    resp = await client.get(f"{RESERVED_PREFIX}/config", headers={"origin": LOOPBACK})
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
    the config-editor GET is the single documented exception for allowlists."""
    secret_allow = "allowlisted.person@corp.example"
    client = _client(_base_config(detection=DetectionConfig(allowlist=(secret_allow,))))
    # Drive traffic so counters populate.
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "mail a@b.example"}]}
    await client.post("/v1/chat/completions", json=body)

    status = (await client.get(f"{RESERVED_PREFIX}/status")).text
    metrics = (await client.get(f"{RESERVED_PREFIX}/metrics")).text
    assert secret_allow not in status
    assert secret_allow not in metrics
    # The config-editor GET IS allowed to echo the allowlist (documented).
    editor = (await client.get(f"{RESERVED_PREFIX}/config")).json()
    assert secret_allow in editor["editable"]["detection"]["allowlist"]


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
