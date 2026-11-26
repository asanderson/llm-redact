"""The proxy's port is its alone on Windows, and doctor tells why a port
cannot be bound.

uvicorn binds through asyncio, which sets no socket option on Windows. There,
another program may bind the port on a specific address beside a WILDCARD
listener (``host = "0.0.0.0"``) and receive the connections sent to that
address, 127.0.0.1 included: a program running as the same user while the
proxy runs, any account's before it starts (the proxy's bind then succeeds
anyway). serve binds a wildcard host's addresses with SO_EXCLUSIVEADDRUSE and
hands the sockets to uvicorn (``serving.listening_sockets``); a specific
address (the default 127.0.0.1) gets the same answers either way
(Microsoft's bind tables), so uvicorn binds it as everywhere, and Linux and
macOS always keep uvicorn's bind. doctor probes the port exclusively on
Windows and tells a port in use (WSAEADDRINUSE, 10048) from one Windows will
not bind (WSAEACCES, 10013: a range Hyper-V, WSL or Docker reserves, or a
program holding it exclusively).

The stand-in tests run everywhere; the real-socket tests run on the Windows
CI legs and pin Windows' own behaviour: what a plain listener lets another
socket do (the reason for the change), what the proxy's listener does not,
and that the proxy binds its port again right after it stops."""

from __future__ import annotations

import errno
import select
import socket
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from llm_redact import ports, serving
from llm_redact.config import Config

SO_EXCLUSIVEADDRUSE = getattr(socket, "SO_EXCLUSIVEADDRUSE", -5)
EXCLUSIVE = (socket.SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1)
IPV6_ONLY = (socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
V4, V6 = socket.AF_INET, socket.AF_INET6

# What the stand-in resolver answers: (family, address) per host, in order.
RESOLVES: dict[str | None, list[tuple[int, tuple[Any, ...]]]] = {
    "127.0.0.1": [(V4, ("127.0.0.1", 8787))],
    "::1": [(V6, ("::1", 8787, 0, 0))],
    "localhost": [(V6, ("::1", 8787, 0, 0)), (V4, ("127.0.0.1", 8787))],
    "0.0.0.0": [(V4, ("0.0.0.0", 8787))],
    "::": [(V6, ("::", 8787, 0, 0))],
    None: [(V6, ("::", 8787, 0, 0)), (V4, ("0.0.0.0", 8787))],
}


class _Socket:
    """A socket stand-in that records its options, bind and close."""

    made: list[_Socket] = []
    # bind errors by address host; families the machine "lacks"
    refuse: dict[str, OSError] = {}
    lacking: set[int] = set()

    def __init__(self, family: int = V4, kind: int = socket.SOCK_STREAM, proto: int = 0) -> None:
        if family in _Socket.lacking:
            raise OSError(errno.EAFNOSUPPORT, "address family not supported")
        self.family = family
        self.options: list[tuple[int, int, int]] = []
        self.bound: tuple[Any, ...] | None = None
        self.inheritable = False
        self.closed = False
        _Socket.made.append(self)

    def setsockopt(self, level: int, option: int, value: int) -> None:
        self.options.append((level, option, value))

    def bind(self, address: tuple[Any, ...]) -> None:
        error = _Socket.refuse.get(str(address[0]))
        if error is not None:
            raise error
        self.bound = address

    def set_inheritable(self, flag: bool) -> None:
        self.inheritable = flag

    def close(self) -> None:
        self.closed = True


def _getaddrinfo(
    host: str | None, port: int, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0
) -> list[tuple[Any, ...]]:
    assert type == socket.SOCK_STREAM and flags == socket.AI_PASSIVE
    if host not in RESOLVES:
        raise socket.gaierror(11001, "getaddrinfo failed")
    return [(fam, socket.SOCK_STREAM, 6, "", address) for fam, address in RESOLVES[host]]


@pytest.fixture(autouse=True)
def _fresh_stand_ins() -> None:
    _Socket.made, _Socket.refuse, _Socket.lacking = [], {}, set()


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ports.socket, "socket", _Socket)
    monkeypatch.setattr(ports.socket, "getaddrinfo", _getaddrinfo)
    monkeypatch.setattr(ports.socket, "SO_EXCLUSIVEADDRUSE", SO_EXCLUSIVEADDRUSE, raising=False)
    yield


@pytest.mark.parametrize(
    ("host", "bound"),
    [
        ("0.0.0.0", [("0.0.0.0", 8787)]),
        ("::", [("::", 8787, 0, 0)]),
        ("", [("::", 8787, 0, 0), ("0.0.0.0", 8787)]),  # all interfaces, both families
    ],
)
@pytest.mark.usefixtures("windows")
def test_windows_binds_a_wildcard_host_exclusively(host: str, bound: list[tuple[Any, ...]]) -> None:
    sockets = serving.listening_sockets(host, 8787)
    assert sockets == _Socket.made
    assert [sock.bound for sock in _Socket.made] == bound
    for sock in _Socket.made:
        expected = [EXCLUSIVE, IPV6_ONLY] if sock.family == V6 else [EXCLUSIVE]
        assert sock.options == expected  # IPv6 sockets IPv6 only, as asyncio binds them
        assert not sock.inheritable and not sock.closed  # a child process never gets the port


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
@pytest.mark.usefixtures("windows")
def test_windows_leaves_a_specific_address_to_uvicorn(host: str) -> None:
    assert serving.listening_sockets(host, 8787) is None
    assert _Socket.made == []


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_leaves_the_bind_to_uvicorn(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(serving.socket, "socket", _Socket)
    assert serving.listening_sockets("0.0.0.0", 8787) is None
    assert _Socket.made == []


def test_binding_exclusively_needs_the_windows_option(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(ports.socket, "SO_EXCLUSIVEADDRUSE", raising=False)
    monkeypatch.setattr(ports.socket, "socket", _Socket)
    assert ports.exclusive_option() is None
    with pytest.raises(OSError, match="Windows socket option"):
        ports.bind_exclusive("0.0.0.0", 8787)
    assert _Socket.made == []
    # serve leaves the bind to uvicorn there, even on a platform named win32
    monkeypatch.setattr(sys, "platform", "win32")
    assert serving.listening_sockets("0.0.0.0", 8787) is None


@pytest.mark.usefixtures("windows")
def test_bind_exclusive_binds_every_address_of_the_name() -> None:
    sockets = ports.bind_exclusive("localhost", 8787)
    assert [sock.bound for sock in sockets] == [("::1", 8787, 0, 0), ("127.0.0.1", 8787)]
    assert sockets[0].options == [EXCLUSIVE, IPV6_ONLY] and sockets[1].options == [EXCLUSIVE]


@pytest.mark.usefixtures("windows")
def test_an_address_family_the_machine_lacks_is_skipped() -> None:
    _Socket.lacking = {V6}
    (sock,) = serving.listening_sockets("", 8787) or []
    assert sock.bound == ("0.0.0.0", 8787)


@pytest.mark.usefixtures("windows")
def test_an_address_family_not_enabled_is_skipped() -> None:
    _Socket.refuse = {"::": OSError(errno.EADDRNOTAVAIL, "address not available")}
    (sock,) = serving.listening_sockets("", 8787) or []
    assert sock.bound == ("0.0.0.0", 8787)
    assert _Socket.made[0].closed  # the IPv6 one, closed when it was dropped


@pytest.mark.usefixtures("windows")
def test_a_held_address_stops_serve_and_closes_the_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _Socket.refuse = {"0.0.0.0": OSError(10048, "Only one usage of each socket address")}
    with pytest.raises(SystemExit) as exit_:
        serving.listening_sockets("", 8787)
    assert exit_.value.code == serving.BIND_FAILURE == 1  # as uvicorn's own bind failure
    assert all(sock.closed for sock in _Socket.made)
    assert (
        "cannot listen on *:8787 (0.0.0.0:8787: Only one usage of each socket address)"
        in caplog.text
    )
    assert "excludedportrange" not in caplog.text


@pytest.mark.usefixtures("windows")
def test_a_port_windows_will_not_bind_names_the_reserved_ranges(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _Socket.refuse = {"::": OSError(10013, "An attempt was made to access a socket")}
    with pytest.raises(SystemExit):
        serving.listening_sockets("::", 8787)
    assert "cannot listen on [::]:8787 ([::]:8787: An attempt was made" in caplog.text
    assert "netsh int ipv4 show excludedportrange protocol=tcp" in caplog.text


@pytest.mark.usefixtures("windows")
def test_no_address_to_bind_stops_serve(caplog: pytest.LogCaptureFixture) -> None:
    _Socket.lacking = {V4, V6}
    with pytest.raises(SystemExit):
        serving.listening_sockets("", 8787)
    assert "cannot listen on *:8787 (no address of all interfaces can be bound)" in caplog.text


@pytest.mark.usefixtures("windows")
def test_an_unresolvable_host_stops_serve(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exit_:
        serving.listening_sockets("proxy.invalid", 8787)
    assert exit_.value.code == serving.BIND_FAILURE
    assert "cannot listen on proxy.invalid:8787 (getaddrinfo failed)" in caplog.text


def test_the_server_runs_on_the_sockets_and_closes_them(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_redact.proxy import create_app

    made = [_Socket(V4)]
    asked: list[tuple[str, int]] = []
    ran: list[object] = []

    def bound(host: str, port: int) -> list[_Socket]:
        asked.append((host, port))
        return made

    def run(self: serving.ProxyServer, sockets: object = None) -> None:
        ran.append(sockets)
        self.started = True

    monkeypatch.setattr(serving, "listening_sockets", bound)
    monkeypatch.setattr(serving.ProxyServer, "run", run)
    serving.run_server(create_app(Config()), host="0.0.0.0", port=8787)
    assert asked == [("0.0.0.0", 8787)] and ran == [made]
    assert made[0].closed  # closed when the server ends, as uvicorn's shutdown does


def test_a_port_that_cannot_be_bound_stops_the_server_before_it_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_redact.proxy import create_app

    def refused(host: str, port: int) -> None:
        raise SystemExit(serving.BIND_FAILURE)

    monkeypatch.setattr(serving, "listening_sockets", refused)
    monkeypatch.setattr(serving.ProxyServer, "run", lambda self, sockets=None: pytest.fail("ran"))
    with pytest.raises(SystemExit) as exit_:
        serving.run_server(create_app(Config()), host="0.0.0.0", port=8787)
    assert exit_.value.code == serving.BIND_FAILURE


def _doctor_port(monkeypatch: pytest.MonkeyPatch, error: OSError | None) -> list[dict[str, str]]:
    import llm_redact.doctor_cli as doctor

    _Socket.made = []
    _Socket.refuse = {} if error is None else {"127.0.0.1": error}
    _Socket.lacking = set()
    monkeypatch.setattr(doctor.socket, "socket", _Socket)
    report = doctor._Report(json_mode=True)
    doctor._check_port_free(report, Config(port=8787))
    return report.rows


@pytest.mark.usefixtures("windows")
def test_doctor_tells_a_reserved_port_from_one_in_use_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (reserved,) = _doctor_port(monkeypatch, OSError(10013, "forbidden"))
    assert reserved["level"] == "WARN"
    assert "Windows will not bind port 8787" in reserved["message"]
    assert "netsh int ipv4 show excludedportrange protocol=tcp" in reserved["message"]
    (in_use,) = _doctor_port(monkeypatch, OSError(10048, "in use"))
    assert "in use by something that does not answer" in in_use["message"]
    assert "excludedportrange" not in in_use["message"]
    (other,) = _doctor_port(monkeypatch, OSError(10022, "An invalid argument was supplied"))
    assert other["message"] == (
        "cannot bind port 8787 on 127.0.0.1 (127.0.0.1:8787: An invalid argument was supplied)"
    )
    (free,) = _doctor_port(monkeypatch, None)
    assert free["level"] == "PASS"
    assert all(sock.closed for sock in _Socket.made)
    assert all(sock.options == [EXCLUSIVE] for sock in _Socket.made)  # as serve would bind


def test_doctor_on_posix_keeps_its_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    (row,) = _doctor_port(monkeypatch, OSError(13, "Permission denied"))
    assert "in use by something that does not answer" in row["message"]


# Real sockets, Windows only: Windows' bind rules themselves.

on_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows' own bind rules")


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _listening(address: str, port: int, *options: int) -> socket.socket:
    sock = socket.socket()
    try:
        for option in options:
            sock.setsockopt(socket.SOL_SOCKET, option, 1)
        sock.bind((address, port))
        sock.listen()
    except BaseException:
        sock.close()
        raise
    return sock


def _receiver(port: int, *servers: socket.socket) -> socket.socket:
    """Which of ``servers`` a connection to 127.0.0.1:port reaches."""
    with socket.create_connection(("127.0.0.1", port), timeout=10):
        ready, _, _ = select.select(list(servers), [], [], 10)
    assert len(ready) == 1, f"{len(ready)} of the listeners got the connection"
    return ready[0]


@on_windows
def test_windows_lets_a_socket_shadow_a_plain_wildcard_listener() -> None:
    # Why: asyncio's listener has no option; the same user binds 127.0.0.1
    # beside it and gets what local clients send there.
    port = _free_port()
    listener = _listening("0.0.0.0", port)
    try:
        thief = _listening("127.0.0.1", port)
        try:
            assert _receiver(port, listener, thief) is thief
        finally:
            thief.close()
    finally:
        listener.close()


@on_windows
def test_no_socket_binds_beside_the_proxys_wildcard_listener() -> None:
    port = _free_port()
    sockets = ports.bind_exclusive("0.0.0.0", port)
    try:
        for sock in sockets:
            sock.listen()
        for address, options in [
            ("127.0.0.1", ()),
            ("127.0.0.1", (socket.SO_REUSEADDR,)),
            ("0.0.0.0", (socket.SO_REUSEADDR,)),
            ("0.0.0.0", ()),
        ]:
            with pytest.raises(OSError):
                _listening(address, port, *options).close()
    finally:
        for sock in sockets:
            sock.close()


@on_windows
def test_a_listener_bound_first_stops_the_proxys_wildcard_bind() -> None:
    port = _free_port()
    thief = _listening("127.0.0.1", port)
    try:
        # Why: asyncio's plain wildcard bind succeeds beside it ...
        _listening("0.0.0.0", port).close()
        # ... the proxy's does not.
        with pytest.raises(OSError):
            ports.bind_exclusive("0.0.0.0", port)
    finally:
        thief.close()


@on_windows
def test_a_plain_loopback_listener_keeps_its_address() -> None:
    # Why uvicorn keeps binding a specific address: no socket takes it.
    port = _free_port()
    listener = _listening("127.0.0.1", port)
    try:
        for options in [(), (socket.SO_REUSEADDR,)]:
            with pytest.raises(OSError):
                _listening("127.0.0.1", port, *options).close()
    finally:
        listener.close()


@on_windows
def test_serve_binds_its_wildcard_port_again_right_after_it_stops() -> None:
    import httpx
    import uvicorn

    from llm_redact.proxy import create_app

    port = _free_port()
    # A client keeps its idle connection across the restart, as an agent's
    # connection pool does.
    with httpx.Client() as client:
        for attempt in ("start", "restart"):
            try:
                sockets = serving.listening_sockets("0.0.0.0", port)
            except SystemExit:
                pytest.fail(f"serve could not bind port {port} on its {attempt}")
            assert sockets is not None
            server = serving.ProxyServer(
                uvicorn.Config(
                    create_app(Config(host="0.0.0.0", port=port)),
                    host="0.0.0.0",
                    port=port,
                    log_level="warning",
                )
            )
            thread = threading.Thread(target=server.run, kwargs={"sockets": sockets}, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic() + 30
                while not server.started:
                    assert time.monotonic() < deadline, "the proxy did not start"
                    time.sleep(0.05)
                answer = client.get(f"http://127.0.0.1:{port}/__llm-redact/healthz")
                assert answer.status_code == 200
            finally:
                server.should_exit = True
                thread.join(30)
