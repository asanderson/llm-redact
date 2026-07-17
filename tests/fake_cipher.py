"""A ``VaultCipher``-Protocol stand-in with no cryptography / llm-redact-pro
dependency.

The Free vault classes (``SqliteVault``, ``EncryptedInMemoryVault``, the
migration/verify helpers) accept any object satisfying the ``VaultCipher``
Protocol and never reach for the concrete Fernet cipher themselves — that is
the paid subsystem (``llm_redact_pro.vault_crypto``). So the Free-side tests
that exercise those classes' encrypted arms (preload, insert, cold-cache
lookup, key_check verification, v2->v3 migration, dense counters) bind to
this fake and stay importable without the paid package installed.

It is deliberately NOT real encryption: ``encrypt`` prefixes the plaintext,
so tests that must prove real at-rest protection (no plaintext on disk / in
RAM) or the env/command/keyring key-resolution order use the concrete cipher
under ``tests/pro`` instead. ``key_check`` depends on the seed, so two fakes
with different seeds model a key mismatch at open.
"""

import hashlib
import hmac

_MARKER = b"fakect:"


class FakeVaultCipher:
    def __init__(self, seed: bytes = b"k" * 32) -> None:
        self._seed = bytes(seed)

    def key_check(self) -> str:
        return hashlib.sha256(b"llm-redact/fake-key-check" + self._seed).hexdigest()

    def mac(self, session_id: str, detector_type: str, original: str) -> str:
        message = b"\x00".join(p.encode("utf-8") for p in (session_id, detector_type, original))
        return hmac.new(self._seed, message, hashlib.sha256).hexdigest()

    def encrypt(self, original: str) -> bytes:
        return _MARKER + original.encode("utf-8")

    def decrypt(self, token: bytes) -> str:
        raw = bytes(token)
        if not raw.startswith(_MARKER):
            from llm_redact.vault import VaultKeyError

            raise VaultKeyError("fake ciphertext failed authentication")
        return raw[len(_MARKER) :].decode("utf-8")
