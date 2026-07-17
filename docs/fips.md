# FIPS 140-3 posture

**Claim, precisely:** llm-redact uses FIPS-approved algorithm selections
everywhere it uses cryptography, and is deployable on FIPS 140-3 validated
hosts. llm-redact itself is not, and cannot be, FIPS-validated — validation
applies to cryptographic *modules* (the host kernel's crypto API and the
OpenSSL/provider stack), not to applications that call them. Any stronger
wording is an overclaim; the banned word is spelled out in
`tests/test_fips_posture.py`, which fails any doc or source file that
uses it near a FIPS claim.

## Cryptographic inventory

| Use | Primitive | FIPS status |
|---|---|---|
| Vault encryption (`[vault] encryption = "fernet"` — the sqlite at-rest store and the encrypted in-memory backend share one cipher) | Fernet = AES-128-CBC + HMAC-SHA256 | Approved algorithms (AES, SHA-2 HMAC) |
| Key split of `LLM_REDACT_VAULT_KEY` | HKDF-SHA256 (two domain-separated subkeys) | Approved (SP 800-56C / 800-108 family) |
| Deterministic vault value index | HMAC-SHA256, domain-separated over (session, type, value) | Approved |
| Per-conversation session anchors | SHA-256 (domain-separated; only the hash is ever stored) | Approved |
| CSRF token, key generation | `secrets` / `os.urandom` | Host kernel DRBG — approved on a FIPS host |
| TLS / mutual TLS (`[tls]`) | Python `ssl` over the system OpenSSL | Uses the host's TLS provider |
| AWS event-stream framing | `zlib.crc32` | **Not cryptography** — transport framing integrity only, mirrored from AWS's wire format |
| Placeholder tokens («EMAIL_001») | None (counters) | n/a — tokens are names, not ciphertext |

Non-approved primitives (MD5, SHA-1, RC4, 3DES, Blowfish) appear nowhere in
the package; `tests/test_fips_posture.py` scans the source and fails if one
is ever introduced, and also fails any doc that claims certification.

## Deploying on a FIPS 140-3 validated host

1. Use an OS with a validated crypto stack in FIPS mode (e.g. RHEL/UBI or
   Ubuntu Pro with `fips=1`): the kernel exposes
   `/proc/sys/crypto/fips_enabled = 1` and the system OpenSSL runs its
   validated provider.
2. Make Python's `hashlib`/`ssl` link the *system* OpenSSL. Distribution
   Python packages already do. If you install the `crypto` extra, build
   `cryptography` against the system OpenSSL instead of using the bundled
   wheel:

   ```bash
   pip install --no-binary cryptography cryptography
   ```

3. Verify:

   ```bash
   llm-redact fips-check
   ```

   The command reports Python's `_hashlib.get_fips_mode()`, the OpenSSL
   the `ssl` module linked, the OpenSSL `cryptography` linked (when the
   extra is installed), the kernel flag, and the vault encryption
   algorithm selections (static — the same inventory as the table above).
   Exit code 0 means FIPS mode was detected on the host; 1 means it was
   not.

On a FIPS host, non-approved algorithms are unavailable at the OpenSSL
layer, so a regression that introduced one would fail loudly rather than
silently degrade.

## Scope notes

- The vault's write-through caches hold plaintext in process memory; FIPS
  validation does not change the at-rest-only scope of vault encryption
  (see the threat model).
- `zlib.crc32` in `eventstream.py` is deliberately outside the inventory's
  cryptographic claims: it is AWS's transport framing checksum, carried
  and recomputed because the wire format requires it.
