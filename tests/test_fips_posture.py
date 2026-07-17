"""FIPS posture: no banned primitives in src/, and the report degrades
gracefully. The wording rule (never "certified") is enforced on docs."""

import re
from pathlib import Path

from llm_redact.fips import collect_fips_report, fips_mode_detected, run_fips_check

SRC = Path(__file__).resolve().parent.parent / "src" / "llm_redact"
# The concrete vault cipher lives in the paid llm-redact-pro package now
# (R4 open-core split); its banned-primitive sweep + approved-primitive pin
# ride with it in that repo's FIPS test.
DOCS = Path(__file__).resolve().parent.parent / "docs"

# Non-approved primitives that must never appear in the package. CRC32 is
# allowed: the event-stream codec uses it for framing integrity, which is
# not a cryptographic use.
_BANNED = re.compile(r"\b(md5|sha1|rc4|des3|3des|blowfish|arcfour)\b", re.IGNORECASE)


def test_no_banned_primitives_in_source() -> None:
    hits: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        for number, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
            if _BANNED.search(line):
                hits.append(f"{path.name}:{number}: {line.strip()}")
    assert hits == [], "non-FIPS-approved primitive referenced:\n" + "\n".join(hits)


def test_docs_never_claim_certification() -> None:
    # "FIPS-approved algorithm selection, deployable on FIPS 140-3
    # validated hosts" is the ceiling; "certified" would overclaim.
    offenders: list[str] = []
    for path in [*DOCS.rglob("*.md"), *SRC.rglob("*.py")]:
        text = path.read_text("utf-8")
        for match in re.finditer(r"FIPS[- ][^.\n]{0,80}", text, re.IGNORECASE):
            if "certifi" in match.group(0).lower():
                offenders.append(f"{path.name}: {match.group(0)!r}")
    assert offenders == []


def test_report_has_all_probes_and_degrades_to_strings() -> None:
    report = collect_fips_report()
    labels = [label for label, _value in report]
    assert labels == [
        "python hashlib FIPS mode",
        "linked OpenSSL (ssl module)",
        "cryptography's OpenSSL",
        "kernel fips_enabled",
        "vault crypto algorithms",
    ]
    assert all(isinstance(value, str) and value for _label, value in report)


def test_vault_crypto_algorithms_reported() -> None:
    # The concrete cipher's source-level primitive pin (SHA256/Fernet/HKDF)
    # moved to the pro repo with vault_crypto.py; the Free FIPS report still
    # names the algorithm selection it documents.
    report = dict(collect_fips_report())
    assert "AES-128-CBC" in report["vault crypto algorithms"]


def test_detection_logic() -> None:
    assert fips_mode_detected([("python hashlib FIPS mode", "1")])
    assert fips_mode_detected([("kernel fips_enabled", "1")])
    assert not fips_mode_detected(
        [("python hashlib FIPS mode", "0"), ("kernel fips_enabled", "unknown")]
    )


def test_run_fips_check_exit_code_matches_detection(capsys: object) -> None:
    code = run_fips_check()
    assert code == (0 if fips_mode_detected() else 1)
