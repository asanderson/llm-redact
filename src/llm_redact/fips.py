"""FIPS 140-3 posture reporting (`llm-redact fips-check`).

llm-redact uses only FIPS-approved algorithm selections where cryptography
is involved — AES-128-CBC + HMAC-SHA256 (Fernet), HKDF-SHA256, SHA-256
session anchors, os.urandom-backed `secrets` tokens; zlib.crc32 in the
event-stream codec is framing integrity, not cryptography. Whether a
deployment is *FIPS-validated* is a property of the HOST's crypto modules
(kernel + OpenSSL), never of this package — see docs/fips.md. Wording rule
everywhere: "FIPS-approved algorithm selection, deployable on FIPS 140-3
validated hosts"; never "certified".
"""

import ssl
from pathlib import Path

_KERNEL_FLAG = Path("/proc/sys/crypto/fips_enabled")


def collect_fips_report() -> list[tuple[str, str]]:
    """Best-effort, label/value pairs; every probe degrades to 'unknown'."""
    report: list[tuple[str, str]] = []

    try:
        import _hashlib

        report.append(("python hashlib FIPS mode", str(_hashlib.get_fips_mode())))
    except (ImportError, AttributeError):
        report.append(("python hashlib FIPS mode", "unknown (no _hashlib.get_fips_mode)"))

    report.append(("linked OpenSSL (ssl module)", ssl.OPENSSL_VERSION))

    try:
        from cryptography.hazmat.backends.openssl.backend import backend

        report.append(("cryptography's OpenSSL", backend.openssl_version_text()))
    except ImportError:
        report.append(("cryptography's OpenSSL", "crypto extra not installed"))

    try:
        flag = _KERNEL_FLAG.read_text().strip()
        report.append(("kernel fips_enabled", flag))
    except OSError:
        report.append(("kernel fips_enabled", "unknown (no /proc/sys/crypto/fips_enabled)"))

    # Static, not probed: the algorithm selections behind [vault] encryption
    # (sqlite at-rest and the encrypted in-memory backend share VaultCipher).
    # Note the PyPI cryptography wheel bundles its own OpenSSL — FIPS hosts
    # should build it against the validated system provider (docs/fips.md).
    report.append(
        (
            "vault crypto algorithms",
            "Fernet (AES-128-CBC + HMAC-SHA256), HKDF-SHA256 subkey split,"
            " HMAC-SHA256 index — approved algorithm selections",
        )
    )

    return report


def fips_mode_detected(report: list[tuple[str, str]] | None = None) -> bool:
    values = dict(report if report is not None else collect_fips_report())
    return values.get("python hashlib FIPS mode") == "1" or values.get("kernel fips_enabled") == "1"


def run_fips_check() -> int:
    report = collect_fips_report()
    width = max(len(label) for label, _value in report)
    for label, value in report:
        print(f"{label:<{width}}  {value}")
    if fips_mode_detected(report):
        print("FIPS mode: detected on this host")
        return 0
    print(
        "FIPS mode: not detected. llm-redact selects FIPS-approved algorithms,"
        " but validation comes from the host's crypto modules — see docs/fips.md."
    )
    return 1
