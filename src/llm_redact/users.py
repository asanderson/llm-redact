"""Named-user registry contract (the Free side of the open-core split).

The Pro, Team, and Unlimited tiers bound the number of NAMED USERS
(llm-redact-pro docs/licensing.md). The concrete registry — the SQLite store
that invites, verifies, counts, and
resolves per-user keys — is a paid subsystem and lives in
``llm_redact_pro.users``. This module holds only what the Free core needs at
the seam:

- the ``UsersStore`` **Protocol** the proxy talks to (so ``proxy.py`` stays
  type-checked without importing the pro package),
- the contract types the proxy reads (``UserRow``) or raises/catches
  (``UsersError``),
- the identity constants (``USER_KEY_PREFIX``) and default path, and
- ``send_verification_email`` — generic stdlib SMTP glue with no paid secrecy
  value, called by the dashboard's invite endpoint; inert until a store exists.

The fail-closed ``build_users_store`` default lives here too: the Free tier is
the implicit single local user (no registry), and a Pro+ tier without the pro
package installed is a ``ConfigError`` — never a silent downgrade.
"""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .config import ConfigError

if TYPE_CHECKING:
    from .config import UsersConfig

USER_KEY_PREFIX = "lrk_"
CODE_TTL_HOURS = 24

_PRO_HINT = "install the llm-redact-pro package to enable it"


def default_users_path() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    return Path(xdg) / "llm-redact" / "users.db"


class UsersError(ValueError):
    """A registry operation was refused; the message says why and never
    contains codes or keys."""


@dataclass(frozen=True)
class UserRow:
    name: str
    email: str
    invited_at: str
    verified_at: str | None
    revoked_at: str | None

    @property
    def status(self) -> str:
        if self.revoked_at is not None:
            return "revoked"
        return "verified" if self.verified_at is not None else "invited"


class UsersStore(Protocol):
    """The named-user registry surface the proxy depends on.

    The concrete SQLite implementation is ``llm_redact_pro.users.UsersStore``;
    the Free core holds only this structural contract. Methods mirror what the
    request path (``lookup_key``) and the dashboard endpoints
    (invite/revoke/list, the counts) call.
    """

    def lookup_key(self, presented_key: str) -> str | None: ...

    def verified_count(self) -> int: ...

    def active_count(self) -> int: ...

    def list_users(self) -> list[UserRow]: ...

    def invite(self, name: str, email: str, *, max_users: int | None) -> str: ...

    def revoke(self, email: str, *, purge: bool = ...) -> None: ...

    def close(self) -> None: ...


# -- verification email -------------------------------------------------------


class _SmtpLike(Protocol):  # the slice of smtplib.SMTP the sender uses
    def starttls(self) -> object: ...
    def login(self, user: str, password: str) -> object: ...
    def send_message(self, msg: EmailMessage) -> object: ...
    def quit(self) -> object: ...


def send_verification_email(
    *,
    smtp_host: str,
    smtp_port: int,
    starttls: bool,
    username: str | None,
    password_env: str,
    from_address: str,
    to_address: str,
    display_name: str,
    code: str,
    smtp_factory: type[smtplib.SMTP] | None = None,
) -> None:
    """Send the verification code over operator-configured SMTP (stdlib —
    no dependency, no vendor service, nothing leaves the operator's own
    infrastructure). The SMTP password comes from the environment variable
    named by ``password_env``, never from the config file."""
    message = EmailMessage()
    message["Subject"] = "llm-redact: verify your user account"
    message["From"] = from_address
    message["To"] = to_address
    message.set_content(
        f"Hello {display_name},\n\n"
        "An llm-redact administrator invited you as a named user.\n"
        f"Your verification code is: {code}\n\n"
        "Complete verification on the proxy machine with:\n"
        f"    llm-redact users verify {to_address} {code}\n\n"
        "Verifying prints your personal proxy key exactly ONCE — have a"
        " password manager ready to store it.\n\n"
        f"The code expires in {CODE_TTL_HOURS} hours. If you did not expect"
        " this email, ignore it.\n"
    )
    factory = smtp_factory if smtp_factory is not None else smtplib.SMTP
    client: _SmtpLike = factory(smtp_host, smtp_port, timeout=30)
    try:
        if starttls:
            client.starttls()
        if username:
            password = os.environ.get(password_env, "")
            if not password:
                raise UsersError(
                    f"[email] username is set but {password_env} is not in the"
                    " environment (the SMTP password never lives in the config file)"
                )
            client.login(username, password)
        client.send_message(message)
    finally:
        client.quit()


def build_users_store(config: UsersConfig, tier: str) -> UsersStore | None:
    """Fail-closed Free default for the named-user registry.

    The Free tier is the implicit single local user — no registry, no users.db.
    On a Pro+ tier the registry is a paid subsystem whose implementation lives
    in llm-redact-pro; without that package this fails closed rather than
    silently running without enforcement.
    """
    if tier == "free":
        return None
    raise ConfigError(f"named users require the llm-redact-pro package; {_PRO_HINT}")
