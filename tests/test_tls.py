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
