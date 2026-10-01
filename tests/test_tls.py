"""[tls] config, the fail-closed bind policy, and a real-socket mTLS run."""

import ipaddress
import ssl
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from llm_redact.config import (
    Config,
    ConfigError,
    TlsConfig,
    parse_config,
    validate_bind_security,
)
from llm_redact.config_write import emit_config_toml
from llm_redact.proxy import create_app

FULL = TlsConfig(certfile="/c.crt", keyfile="/c.key", client_ca="/ca.crt")
SERVER_ONLY = TlsConfig(certfile="/c.crt", keyfile="/c.key")
NO_TLS = TlsConfig()


def test_parse_tls_section() -> None:
    config = parse_config(
        {"tls": {"certfile": "/a.crt", "keyfile": "/a.key", "client_ca": "/ca.crt"}}, "test"
    )
    assert config.tls == TlsConfig(certfile="/a.crt", keyfile="/a.key", client_ca="/ca.crt")
    assert config.tls.enabled and config.tls.mutual
    assert parse_config({}, "test").tls == NO_TLS

    with pytest.raises(ConfigError, match="together"):
        parse_config({"tls": {"certfile": "/a.crt"}}, "test")
    with pytest.raises(ConfigError, match="client_ca requires"):
        parse_config({"tls": {"client_ca": "/ca.crt"}}, "test")
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config({"tls": {"cert": "/a.crt"}}, "test")


def test_tls_round_trips_through_emitter() -> None:
    for tls in (FULL, SERVER_ONLY, NO_TLS):
        config = Config(tls=tls)
        assert (
            parse_config(__import__("tomllib").loads(emit_config_toml(config)), "round-trip")
            == config
        )


def test_bind_security_matrix() -> None:
    # Loopback: fine with or without TLS.
    for host in ("127.0.0.1", "127.0.0.2", "::1", "localhost", "LOCALHOST"):
        validate_bind_security(host, NO_TLS, {})
        validate_bind_security(host, SERVER_ONLY, {})

    # Non-loopback without full mutual TLS: refused, whatever the shape.
    for host in ("0.0.0.0", "::", "192.168.1.5", "myhost.internal"):
        with pytest.raises(ConfigError, match="mutual TLS"):
            validate_bind_security(host, NO_TLS, {})
        with pytest.raises(ConfigError, match="mutual TLS"):
            validate_bind_security(host, SERVER_ONLY, {})

    # Full mutual TLS unlocks a wider bind.
    validate_bind_security("0.0.0.0", FULL, {})

    # The container hatch: only the documented exact value counts.
    validate_bind_security("0.0.0.0", NO_TLS, {"LLM_REDACT_INSECURE_BIND": "1"})
    with pytest.raises(ConfigError):
        validate_bind_security("0.0.0.0", NO_TLS, {"LLM_REDACT_INSECURE_BIND": "true"})


def _write_pem(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _make_test_pki(tmp_path: Path) -> dict[str, Path]:
    """A throwaway CA plus server (SAN 127.0.0.1) and client certs.

    RFC 5280-complete (SKI/AKI, KeyUsage, EKU): Python 3.13's
    create_default_context() enables VERIFY_X509_STRICT, which rejects
    minimal certificates ("Missing Authority Key Identifier")."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    def make(
        common_name: str,
        *,
        issuer: tuple[x509.Certificate, ec.EllipticCurvePrivateKey] | None = None,
        is_ca: bool = False,
        san_ip: str | None = None,
        eku: object | None = None,
    ) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        issuer_key = issuer[1] if issuer else key
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer[0].subject if issuer else subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(UTC) - timedelta(days=1))
            .not_valid_after(datetime.now(UTC) + timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=is_ca, path_length=None), critical=True)
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()),
                critical=False,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=is_ca,
                    crl_sign=is_ca,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
        )
        if eku is not None:
            builder = builder.add_extension(x509.ExtendedKeyUsage([eku]), critical=False)
        if san_ip is not None:
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(san_ip))]),
                critical=False,
            )
        cert = builder.sign(issuer_key, hashes.SHA256())
        return cert, key

    ca_cert, ca_key = make("llm-redact test CA", is_ca=True)
    server_cert, server_key = make(
        "localhost",
        issuer=(ca_cert, ca_key),
        san_ip="127.0.0.1",
        eku=ExtendedKeyUsageOID.SERVER_AUTH,
    )
    client_cert, client_key = make(
        "llm-redact test client",
        issuer=(ca_cert, ca_key),
        eku=ExtendedKeyUsageOID.CLIENT_AUTH,
    )

    def pem_cert(cert: x509.Certificate) -> bytes:
        return cert.public_bytes(serialization.Encoding.PEM)

    def pem_key(key: ec.EllipticCurvePrivateKey) -> bytes:
        return key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    return {
        "ca": _write_pem(tmp_path / "ca.crt", pem_cert(ca_cert)),
        "server_crt": _write_pem(tmp_path / "server.crt", pem_cert(server_cert)),
        "server_key": _write_pem(tmp_path / "server.key", pem_key(server_key)),
        "client_crt": _write_pem(tmp_path / "client.crt", pem_cert(client_cert)),
        "client_key": _write_pem(tmp_path / "client.key", pem_key(client_key)),
    }


def test_mutual_tls_over_a_real_socket(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    import uvicorn

    pki = _make_test_pki(tmp_path)
    upstream = httpx.MockTransport(lambda request: httpx.Response(502))
    app = create_app(Config(), upstream_transport=upstream)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            log_level="warning",
            ssl_certfile=str(pki["server_crt"]),
            ssl_keyfile=str(pki["server_key"]),
            ssl_ca_certs=str(pki["ca"]),
            ssl_cert_reqs=ssl.CERT_REQUIRED,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 15
        while not server.started:
            if time.time() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        base = f"https://127.0.0.1:{port}"

        # With a CA-signed client certificate: full round trip.
        mtls_ctx = ssl.create_default_context(cafile=str(pki["ca"]))
        mtls_ctx.load_cert_chain(str(pki["client_crt"]), str(pki["client_key"]))
        response = httpx.get(f"{base}/__llm-redact/status", verify=mtls_ctx, timeout=10.0)
        assert response.status_code == 200
        assert "version" in response.json()

        # Without a client certificate: the handshake (or first read) fails.
        no_cert_ctx = ssl.create_default_context(cafile=str(pki["ca"]))
        with pytest.raises(httpx.HTTPError):
            httpx.get(f"{base}/__llm-redact/status", verify=no_cert_ctx, timeout=10.0)

        # Plain http against the TLS socket: refused.
        with pytest.raises(httpx.HTTPError):
            httpx.get(f"http://127.0.0.1:{port}/__llm-redact/status", timeout=10.0)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# --- the ASGI TLS extension (tls_scope.py) -------------------------------------


def test_rfc4514_name() -> None:
    from llm_redact.tls_scope import rfc4514_name

    subject = (
        (("countryName", "US"),),
        (("organizationName", "Acme, Inc."),),
        (("commonName", " ada+lovelace "), ("userId", "7")),
    )
    assert rfc4514_name(subject) == r"CN=\ ada\+lovelace\ +UID=7,O=Acme\, Inc.,C=US"
    assert rfc4514_name(()) is None
    assert rfc4514_name(((("emailAddress", "a@b.example"),),)) == "emailAddress=a@b.example"


def test_tls_extension_skips_plain_connections() -> None:
    from llm_redact.tls_scope import tls_extension

    class Plain:
        def get_extra_info(self, name: str) -> None:
            return None

    assert tls_extension(Plain()) is None  # type: ignore[arg-type]


def test_serve_wires_the_extension_for_mutual_tls_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from llm_redact import serving
    from llm_redact.cli import main

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(serving, "run_server", lambda app, **kwargs: captured.append(kwargs))
    mutual = tmp_path / "mutual.toml"
    mutual.write_text('[tls]\ncertfile = "/c.crt"\nkeyfile = "/c.key"\nclient_ca = "/ca.crt"\n')
    server_only = tmp_path / "server.toml"
    server_only.write_text('[tls]\ncertfile = "/c.crt"\nkeyfile = "/c.key"\n')
    main(["serve", "--config", str(mutual)])
    main(["serve", "--config", str(server_only)])
    http_cls = captured[0]["http"]
    assert isinstance(http_cls, type) and http_cls.__name__.startswith("TlsExtension")
    assert "http" not in captured[1] and "ws" not in captured[1]


def _echo_tls_app() -> object:
    import json

    async def app(scope: dict, receive: object, send: object) -> None:  # type: ignore[type-arg]
        tls = (scope.get("extensions") or {}).get("tls")
        if scope["type"] == "lifespan":
            while True:
                message = await receive()  # type: ignore[operator]
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})  # type: ignore[operator]
                else:
                    await send({"type": "lifespan.shutdown.complete"})  # type: ignore[operator]
                    return
        body = json.dumps(tls).encode()
        if scope["type"] == "websocket":
            await receive()  # type: ignore[operator]  # websocket.connect
            await send({"type": "websocket.accept"})  # type: ignore[operator]
            await send({"type": "websocket.send", "text": body.decode()})  # type: ignore[operator]
            await send({"type": "websocket.close", "code": 1000})  # type: ignore[operator]
            return
        await send(  # type: ignore[operator]
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})  # type: ignore[operator]

    return app


def test_client_certificate_reaches_the_scope_over_a_real_socket(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    import asyncio
    import json

    import uvicorn

    from llm_redact.tls_scope import uvicorn_protocol_kwargs

    pki = _make_test_pki(tmp_path)
    server = uvicorn.Server(
        uvicorn.Config(
            _echo_tls_app(),  # type: ignore[arg-type]
            host="127.0.0.1",
            port=0,
            log_level="warning",
            ssl_certfile=str(pki["server_crt"]),
            ssl_keyfile=str(pki["server_key"]),
            ssl_ca_certs=str(pki["ca"]),
            ssl_cert_reqs=ssl.CERT_REQUIRED,
            **uvicorn_protocol_kwargs(),  # type: ignore[arg-type]
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 15
        while not server.started:
            if time.time() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        ctx = ssl.create_default_context(cafile=str(pki["ca"]))
        ctx.load_cert_chain(str(pki["client_crt"]), str(pki["client_key"]))

        tls = httpx.get(f"https://127.0.0.1:{port}/", verify=ctx, timeout=10.0).json()
        assert tls["client_cert_name"] == "CN=llm-redact test client"
        assert tls["client_cert_chain"][0].startswith("-----BEGIN CERTIFICATE-----")
        assert tls["client_cert_chain"][0] == ssl.DER_cert_to_PEM_cert(
            ssl.PEM_cert_to_DER_cert(pki["client_crt"].read_text())
        )
        assert tls["tls_version"] in (0x0303, 0x0304)

        websockets = pytest.importorskip("websockets")

        async def over_websocket() -> dict:  # type: ignore[type-arg]
            async with websockets.connect(f"wss://127.0.0.1:{port}/ws", ssl=ctx) as ws:
                return json.loads(await ws.recv())  # type: ignore[no-any-return]

        ws_tls = asyncio.run(over_websocket())
        assert ws_tls["client_cert_name"] == "CN=llm-redact test client"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
