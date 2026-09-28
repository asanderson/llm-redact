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

``[vault.kms]`` replaces that order entirely: the key is stored wrapped by a
cloud KMS and llm-redact-pro unwraps it through ``Registry.resolve_vault_key``.
``resolve_cipher`` is the ONE cipher resolution every consumer shares — the
proxy (llm-redact-pro's ``build_vault_manager`` calls it) and every vault CLI
command — so a KMS-wrapped key works identically everywhere.
"""

import base64
import binascii
import logging
import os
import subprocess
from collections.abc import Mapping
from typing import TYPE_CHECKING

from llm_redact.vault import VaultKeyError

if TYPE_CHECKING:
    from llm_redact.config import VaultConfig
    from llm_redact.plugin_api import VaultCipher
    from llm_redact.registry import Registry

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


# --- [vault.kms]: the KMS-wrapped key seam ------------------------------------


def resolve_vault_key(config: "VaultConfig") -> bytes | None:
    """Free default of ``Registry.resolve_vault_key``.

    None without ``[vault.kms]`` (the cipher's own env / key command /
    keychain order applies). With it, unwrapping needs the paid package:
    fail closed naming the feature AND the package — never a fall back to a
    local key source."""
    if config.kms is None:
        return None
    from llm_redact.config import ConfigError

    raise ConfigError(
        f"[vault.kms] (a {config.kms.provider} KMS-wrapped vault key) requires the"
        " llm-redact-pro package (pip install llm-redact-pro), a version that supports"
        " [vault.kms]; the key is never read from another source instead"
    )


def key_source(config: "VaultConfig") -> str | None:
    """Where the vault key comes from, for /status and doctor: None when the
    vault is not encrypted, ``kms:<provider>`` with [vault.kms], else
    ``local`` (LLM_REDACT_VAULT_KEY, the key command or the OS keychain)."""
    if config.encryption != "fernet":
        return None
    if config.kms is not None:
        return f"kms:{config.kms.provider}"
    return "local"


def ambiguous_local_key(
    config: "VaultConfig", environ: Mapping[str, str] | None = None
) -> str | None:
    """The local key variable set alongside [vault.kms], else None."""
    if config.kms is None:
        return None
    env = os.environ if environ is None else environ
    for name in (ENV_KEY, CMD_ENV_KEY):
        if env.get(name, "").strip():
            return name
    return None


def require_vault_key_source(
    config: "VaultConfig", registry: "Registry", environ: Mapping[str, str] | None = None
) -> None:
    """Startup guard for [vault.kms], run before any cipher or vault is built.

    Refuses (ConfigError) a local key variable set alongside it — two sources
    are ambiguous, and quietly preferring either hides a misconfiguration —
    and a registry whose ``resolve_vault_key`` is still this Free default:
    no llm-redact-pro, or one that predates the seam and would otherwise
    build its cipher from LLM_REDACT_VAULT_KEY, ignoring [vault.kms]."""
    if config.kms is None:
        return
    name = ambiguous_local_key(config, environ)
    if name is not None:
        from llm_redact.config import ConfigError

        raise ConfigError(
            f"[vault.kms] is configured and {name} is also set: the vault key would have two"
            f" sources; unset {name} (with [vault.kms] the key comes only from the KMS)"
        )
    if registry.resolve_vault_key is resolve_vault_key:
        resolve_vault_key(config)  # raises: nothing registered can unwrap it


def resolve_cipher(
    config: "VaultConfig", registry: "Registry | None" = None
) -> "VaultCipher | None":
    """The vault cipher for ``config`` — the one resolution path.

    Without [vault.kms]: ``registry.build_cipher`` (None when unencrypted;
    the local key order otherwise). With it: the guard above, then
    ``registry.resolve_vault_key`` (the KMS unwrap) and
    ``registry.cipher_from_key``; a resolver that yields no 32-byte key is
    refused, never answered from another source."""
    if registry is None:
        from llm_redact.registry import get_registry

        registry = get_registry()
    if config.kms is None:
        return registry.build_cipher(config)
    require_vault_key_source(config, registry)
    key = registry.resolve_vault_key(config)
    if not isinstance(key, bytes) or len(key) != 32:
        raise VaultKeyError(
            f"[vault.kms] ({config.kms.provider}) yielded no usable 32-byte vault key;"
            " refusing to fall back to another key source"
        )
    return registry.cipher_from_key(key)
