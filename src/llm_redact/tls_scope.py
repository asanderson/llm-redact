"""Expose the verified TLS client certificate to the app.

uvicorn uses the TLS connection only to decide http vs https; the client
certificate a mutual-TLS handshake verified never reaches the ASGI scope.
``serve`` therefore wraps uvicorn's HTTP and WebSocket protocol classes so
every request on a TLS connection carries the standard ASGI TLS extension
(``scope["extensions"]["tls"]``, https://asgi.readthedocs.io/en/latest/
specs/tls.html): the PEM chain the handshake verified, leaf first, and its
RFC 4514 subject name.

Transport information only: nothing here identifies a user or decides
anything. An installed access gate (llm-redact-pro) may map the certificate
to one; without a gate the extension is simply unread. Never logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import ssl
from collections.abc import Callable
from typing import Any

# ssl.SSLObject.version() -> the ASGI extension's integer protocol version.
_TLS_VERSIONS = {"TLSv1": 0x0301, "TLSv1.1": 0x0302, "TLSv1.2": 0x0303, "TLSv1.3": 0x0304}

# getpeercert() long attribute names -> RFC 4514 short names. Anything else
# keeps its long name (the RFC allows an OID instead; stdlib only gives names).
_RFC4514_NAMES = {
    "commonName": "CN",
    "localityName": "L",
    "stateOrProvinceName": "ST",
    "organizationName": "O",
    "organizationalUnitName": "OU",
    "countryName": "C",
    "streetAddress": "STREET",
    "domainComponent": "DC",
    "userId": "UID",
}


def _escape_rfc4514(value: str) -> str:
    out = []
    for index, char in enumerate(value):
        if (
            char in ',+"\\<>;'
            or (index == 0 and char in "# ")
            or (index == len(value) - 1 and char == " ")
        ):
            out.append("\\" + char)
        elif char == "\x00":
            out.append("\\00")
        else:
            out.append(char)
    return "".join(out)


def rfc4514_name(subject: Any) -> str | None:
    """``getpeercert()["subject"]`` (RDNs most-significant first) as an
    RFC 4514 string (least-significant first, multi-valued RDNs joined
    with ``+``)."""
    if not subject:
        return None
    rdns = []
    for rdn in reversed(subject):
        attributes = (f"{_RFC4514_NAMES.get(k, k)}={_escape_rfc4514(v)}" for k, v in rdn)
        rdns.append("+".join(attributes))
    return ",".join(rdns)


def tls_extension(transport: asyncio.BaseTransport) -> dict[str, Any] | None:
    """The ASGI TLS extension for one connection, or None when it is not
    TLS. ``client_cert_chain`` is empty when no client certificate was
    presented (server-only TLS)."""
    ssl_object = transport.get_extra_info("ssl_object")
    if ssl_object is None:
        return None
    chain: list[str] = []
    name: str | None = None
    leaf = ssl_object.getpeercert(binary_form=True)
    if leaf:
        ders: list[bytes] = [leaf]
        verified_chain = getattr(ssl_object, "get_verified_chain", None)  # Python 3.13+
        if verified_chain is not None:
            with contextlib.suppress(ssl.SSLError, ValueError):
                ders = [bytes(der) for der in verified_chain()] or ders
        chain = [ssl.DER_cert_to_PEM_cert(der) for der in ders]
        name = rfc4514_name((ssl_object.getpeercert() or {}).get("subject"))
    return {
        "server_cert": None,
        "client_cert_chain": chain,
        "client_cert_name": name,
        "client_cert_error": None,
        "tls_version": _TLS_VERSIONS.get(ssl_object.version() or ""),
        "cipher_suite": None,  # stdlib exposes only the cipher's name
    }


def with_tls_extension(base: type[asyncio.Protocol]) -> type[asyncio.Protocol]:
    """A subclass of a uvicorn HTTP or WebSocket protocol that adds the TLS
    extension to every scope on its connection.

    uvicorn builds one protocol instance per connection and hands each
    request's scope to ``self.app`` (a WebSocket upgrade builds its own
    protocol and calls ``connection_made`` with the same transport), so
    wrapping ``self.app`` once per connection covers both.
    """

    class TlsExtensionProtocol(base):  # type: ignore[valid-type, misc]
        def connection_made(self, transport: asyncio.BaseTransport) -> None:
            super().connection_made(transport)
            extension = tls_extension(transport)
            if extension is None:
                return
            inner: Callable[..., Any] = self.app  # type: ignore[has-type]

            async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
                extensions = dict(scope.get("extensions") or {})
                extensions["tls"] = extension
                await inner({**scope, "extensions": extensions}, receive, send)

            self.app = app

    TlsExtensionProtocol.__name__ = f"TlsExtension{base.__name__}"
    TlsExtensionProtocol.__qualname__ = TlsExtensionProtocol.__name__
    return TlsExtensionProtocol


def uvicorn_protocol_kwargs() -> dict[str, object]:
    """``http=``/``ws=`` for ``uvicorn.run``: the protocols uvicorn's "auto"
    setting would pick, each wrapped with the TLS extension. Only used for
    mutual TLS; every other ``serve`` keeps uvicorn's own defaults."""
    from uvicorn.protocols.http.auto import AutoHTTPProtocol
    from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol

    kwargs: dict[str, object] = {"http": with_tls_extension(AutoHTTPProtocol)}
    if isinstance(AutoWebSocketsProtocol, type):
        kwargs["ws"] = with_tls_extension(AutoWebSocketsProtocol)
    return kwargs
