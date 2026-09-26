"""Client authentication: who is making this request.

One ``AuthChain`` turns whatever credential a client presented into an
``Identity``. Every ``Authenticator`` in the chain has two jobs, kept apart
on purpose:

- ``extract`` finds the method's credential on the connection and SCRUBS it
  from the ASGI scope (path, raw path, headers). The chain calls it on EVERY
  authenticator, before any verification, so no credential channel can
  survive to routing, logging, recording or forwarding just because an
  earlier method already decided the request. Credentials are OUR secrets;
  a provider or a log line must never see one.
- ``verify`` resolves an extracted credential to an identity, or None.

Decision rule: the FIRST credential presented (in chain order) decides. A
presented credential that fails verification never falls through to a later
method: a bad key must not be rescued by some other channel the client also
filled in. What a missing or failed identity MEANS (refuse or serve
unattributed) is the caller's policy, not the chain's — today the proxy
refuses only while named-user enforcement is on.

Open-core split: authentication is a PAID feature (llm-redact-pro, Pro
tier and above). The core holds only this seam and the credential
scrubbing, which is unconditional: a credential sent to a Free proxy must
still never reach a provider or a log line. The one method today,
``user_key`` (the named-user key via the ``/u/<key>/`` base-path prefix or
the ``x-llm-redact-user`` header), verifies only against the named-user
registry, which exists only when llm-redact-pro builds it (never on Free).
Further methods (client-certificate identity, OIDC, Basic/LDAP, brokered
provider credentials) are implemented in llm-redact-pro, never here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from starlette.requests import HTTPConnection

    from .users import UsersStore

# Which kind of connection a credential arrives on. Channels differ per
# surface: the WebSocket surface has never accepted the path prefix.
Surface = Literal["http", "websocket"]

USER_KEY_HEADER = "x-llm-redact-user"
USER_PATH_PREFIX = "/u/"


@dataclass(frozen=True)
class Credential:
    """A credential one authenticator extracted (and already scrubbed).

    ``method`` and ``channel`` are metadata — safe to log or record. The
    ``secret`` is never logged, recorded or forwarded; ``repr`` omits it so
    an accidental ``%r`` cannot leak it either.
    """

    method: str
    channel: str
    secret: str = field(repr=False)


@dataclass(frozen=True)
class Identity:
    """Who a request belongs to. ``subject`` is the name attributed in the
    recent/events/audit rows; ``method`` names the authenticator that
    vouched for it."""

    subject: str
    method: str


@dataclass(frozen=True)
class AuthResult:
    """The chain's answer for one connection.

    ``presented`` is the credential that decided (None when the client sent
    none); ``identity`` is None when nothing was presented or when the
    presented credential did not verify.
    """

    identity: Identity | None
    presented: Credential | None

    @property
    def rejected(self) -> bool:
        """A credential was presented and did not verify."""
        return self.presented is not None and self.identity is None


class Authenticator(Protocol):
    """One authentication method in the chain."""

    method: str

    def extract(self, conn: HTTPConnection, surface: Surface) -> Credential | None: ...

    def verify(self, credential: Credential) -> Identity | None: ...


class AuthChain:
    """Runs every authenticator's ``extract`` (scrubbing all channels), then
    lets the first presented credential decide."""

    def __init__(self, authenticators: Sequence[Authenticator]) -> None:
        self._authenticators = tuple(authenticators)

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(a.method for a in self._authenticators)

    def authenticate(self, conn: HTTPConnection, surface: Surface) -> AuthResult:
        decider: tuple[Authenticator, Credential] | None = None
        for authenticator in self._authenticators:
            credential = authenticator.extract(conn, surface)
            if credential is not None and decider is None:
                decider = (authenticator, credential)
        if decider is None:
            return AuthResult(identity=None, presented=None)
        authenticator, credential = decider
        return AuthResult(identity=authenticator.verify(credential), presented=credential)


# --- scrubbing ----------------------------------------------------------------


def _drop_cached(conn: HTTPConnection) -> None:
    # Starlette caches Headers and URL on first access; drop both so every
    # later reader (incl. forwarding) sees the scrubbed scope.
    for attr in ("_headers", "_url"):
        if hasattr(conn, attr):
            delattr(conn, attr)


def pop_header(conn: HTTPConnection, name: str) -> str | None:
    """Remove every ``name`` header from the scope and return the last value
    (stripped), or None when absent."""
    wanted = name.lower().encode("ascii")
    found: str | None = None
    remaining: list[tuple[bytes, bytes]] = []
    for key, value in conn.scope["headers"]:
        if key.lower() == wanted:
            found = value.decode("latin-1").strip()
        else:
            remaining.append((key, value))
    if found is not None:
        conn.scope["headers"] = remaining
        _drop_cached(conn)
    return found


def pop_path_segment(conn: HTTPConnection, prefix: str) -> str | None:
    """Strip ``prefix`` + one segment (``/u/<segment>/rest`` -> ``/rest``)
    from the scope's path AND raw path, returning the segment.

    Forwarding builds the upstream URL from ``raw_path``, so it is scrubbed
    too. A byte-prefix match on the DECODED segment silently fails when the
    segment is percent-encoded (``/u/lrk_%41BC/...``) and would leave the
    credential in the forwarded URL, so the first RAW segment after the
    prefix is stripped instead — encoding-agnostic, and the remainder keeps
    its own encoding (Bedrock ARN model ids need it verbatim). When the raw
    path does not start with the prefix at all (an encoded prefix), it fails
    closed by rebuilding the raw path from the already-scrubbed path.
    """
    scope = conn.scope
    path: str = scope["path"]
    if not path.startswith(prefix):
        return None
    segment, _, remainder = path[len(prefix) :].partition("/")
    if not segment:
        return None
    scope["path"] = "/" + remainder
    raw: bytes | None = scope.get("raw_path")
    if raw is not None:
        prefix_bytes = prefix.encode("latin-1")
        if raw.startswith(prefix_bytes):
            _, _, remainder_raw = raw[len(prefix_bytes) :].partition(b"/")
            scope["raw_path"] = (b"/" + remainder_raw) if remainder_raw else b"/"
        else:
            scope["raw_path"] = scope["path"].encode("latin-1")
    _drop_cached(conn)
    return segment


# --- the named-user key -------------------------------------------------------


class UserKeyAuthenticator:
    """The named-user key (llm-redact-pro docs/users.md).

    Two channels on HTTP: the universal ``/u/<key>/`` base-path prefix (the
    one knob every tool has is its base URL) and the ``x-llm-redact-user``
    header; the path wins when both are present, and both are scrubbed.
    WebSocket upgrades carry the header only. Keys resolve against the live
    named-user registry (``store`` is read per call, so a registry built or
    dropped later is honored); with no registry nothing verifies.
    """

    method = "user_key"

    def __init__(self, store: Callable[[], UsersStore | None]) -> None:
        self._store = store

    def extract(self, conn: HTTPConnection, surface: Surface) -> Credential | None:
        from_path = pop_path_segment(conn, USER_PATH_PREFIX) if surface == "http" else None
        from_header = pop_header(conn, USER_KEY_HEADER)
        if from_path is not None:
            return Credential(self.method, "path", from_path)
        if from_header:
            return Credential(self.method, "header", from_header)
        return None

    def verify(self, credential: Credential) -> Identity | None:
        store = self._store()
        if store is None:
            return None
        name = store.lookup_key(credential.secret)
        return Identity(subject=name, method=self.method) if name is not None else None
