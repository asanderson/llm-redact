"""Binding the proxy's port exclusively on Windows (SO_EXCLUSIVEADDRUSE).

uvicorn binds through asyncio, which sets no socket option on Windows. There,
another program may bind the port on a specific address beside a wildcard
listener (``0.0.0.0``, ``::``) and receive the connections sent to that
address, 127.0.0.1 included: a program running as the same user while the
listener runs, any account's before it starts (a plain wildcard bind then
succeeds anyway). A socket bound with SO_EXCLUSIVEADDRUSE shares its port
with no other socket, and its own bind fails when another program holds the
port on any address. For a specific address Microsoft's bind tables give
another socket the same answers with the option as without it ("Using
SO_REUSEADDR and SO_EXCLUSIVEADDRUSE").

Standard library only: doctor probes the port with ``bind_exclusive`` and
must not import the server (uvicorn) to do so.
"""

from __future__ import annotations

import errno
import ipaddress
import socket
from typing import Any

AddressInfo = tuple[Any, ...]


def exclusive_option() -> int | None:
    """SO_EXCLUSIVEADDRUSE where the system has it (Windows), else None."""
    option: int | None = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
    return option


def addresses(host: str, port: int) -> list[AddressInfo]:
    """The addresses asyncio's ``create_server`` binds for uvicorn, in the
    resolver's order: every address of the name (``localhost`` is 127.0.0.1
    and ::1), all interfaces for an empty host. Raises ``socket.gaierror``
    for a name that does not resolve."""
    infos = socket.getaddrinfo(host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    return list(dict.fromkeys(infos))


def is_wildcard(info: AddressInfo) -> bool:
    """Whether an address (a ``getaddrinfo`` entry) is all interfaces."""
    try:
        return ipaddress.ip_address(str(info[4][0]).partition("%")[0]).is_unspecified
    except ValueError:
        return False


def where(address: tuple[Any, ...]) -> str:
    """``host:port``, an IPv6 address in brackets, all interfaces ``*``."""
    host, port = str(address[0]) or "*", address[1]
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def bind_exclusive(
    host: str, port: int, infos: list[AddressInfo] | None = None
) -> list[socket.socket]:
    """One bound, not yet listening, socket per address of ``host``
    (``addresses``, or ``infos`` already resolved), each with
    SO_EXCLUSIVEADDRUSE; IPv6 sockets IPv6 only and an address family the
    machine lacks skipped, as asyncio binds them. Raises OSError, with every
    socket closed, when an address cannot be bound, and where the system
    has no SO_EXCLUSIVEADDRUSE."""
    option = exclusive_option()
    if option is None:
        raise OSError("SO_EXCLUSIVEADDRUSE is a Windows socket option")
    if infos is None:
        infos = addresses(host, port)
    sockets: list[socket.socket] = []
    try:
        for family, kind, proto, _name, address in infos:
            try:
                sock = socket.socket(family, kind, proto)
            except OSError:
                continue  # an address family this machine lacks
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, option, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                sock.bind(address)
            except OSError as problem:
                if problem.errno != errno.EADDRNOTAVAIL:
                    raise OSError(problem.errno, f"{where(address)}: {problem.strerror}") from None
                sockets.pop().close()  # the address family is not enabled
        if not sockets:
            raise OSError(f"no address of {host or 'all interfaces'} can be bound")
    except BaseException:
        for sock in sockets:
            sock.close()
        raise
    return sockets
