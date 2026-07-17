"""Vault-key resolution helpers (the Free side of the open-core split).

The at-rest encryption itself — the Fernet/HKDF ``VaultCipher`` — is a paid
subsystem (``llm_redact_pro.vault_crypto``). This module keeps only the generic
glue the Free CLI and doctor use: the env-var / key-command / keyring
resolution order, key generation, and master-key decode/validation. None of it
encrypts anything; it just locates and validates a key, then hands off.

Resolution order (used by the paid cipher's ``from_env``): env var
``LLM_REDACT_VAULT_KEY``, then the command in ``LLM_REDACT_VAULT_KEY_CMD``,
then the OS keychain (``keyring`` extra), then fail closed. Every non-env source
that errors is treated as key-absent — never a silent downgrade, never a
traceback that could echo the command or its output.
"""

import base64
import binascii
import logging
import os
import subprocess

from llm_redact.vault import VaultKeyError

logger = logging.getLogger("llm_redact")

ENV_KEY = "LLM_REDACT_VAULT_KEY"
CMD_ENV_KEY = "LLM_REDACT_VAULT_KEY_CMD"  # a command whose stdout is the key
NEW_ENV_KEY = "LLM_REDACT_NEW_VAULT_KEY"  # the target key for `vault rotate-key`
KEYRING_SERVICE = "llm-redact"
KEYRING_ITEM = "vault-key"
_CMD_TIMEOUT_S = 15


def generate_key() -> str:
    """A fresh Fernet-format master key (44-char urlsafe base64, 32 bytes)."""
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


def key_from_command() -> str | None:
    """The key printed by ``LLM_REDACT_VAULT_KEY_CMD``, or None.

    Runs the operator-configured command (via the shell, since it is their
    own command and often carries pipes/args) and returns its stdout,
    stripped. Any failure — command unset, non-zero exit, missing binary,
    timeout — returns None so the caller falls through to the keyring and
    ultimately fails closed. We log the failure by exception TYPE only:
    CalledProcessError carries captured stdout/stderr that could contain the
    key, so it is never formatted into the message.
    """
    cmd = os.environ.get(CMD_ENV_KEY, "").strip()
    if not cmd:
        return None
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_CMD_TIMEOUT_S,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(
            "%s failed (%s); treating the vault key as absent", CMD_ENV_KEY, type(exc).__name__
        )
        return None
    key = result.stdout.strip()
    return key or None


def key_from_keyring() -> str | None:
    """The stored key from the OS keychain, or None.

    None covers every no-key case alike: extra not installed, no stored
    entry, or a backend error (headless Linux without a Secret Service, a
    locked keychain) — the caller fails closed on None, so a broken
    backend can never silently downgrade to no encryption.
    """
    try:
        import keyring
        from keyring.errors import KeyringError
    except ImportError:
        return None
    try:
        stored = keyring.get_password(KEYRING_SERVICE, KEYRING_ITEM)
    except KeyringError:
        return None
    return str(stored) if stored is not None else None


def decode_master_key(raw: str, source: str) -> bytes:
    """Validate and decode a master key (shared by from_env and set-key)."""
    try:
        master = base64.urlsafe_b64decode(raw.encode("ascii"))
    except (ValueError, binascii.Error) as exc:
        raise VaultKeyError(f"{source} is not valid urlsafe base64") from exc
    if len(master) != 32:
        raise VaultKeyError(
            f"{source} must decode to 32 bytes (a 44-char Fernet key); "
            "generate one with `llm-redact vault gen-key`"
        )
    return master
