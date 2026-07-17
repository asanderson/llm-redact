"""Built-in regex detection rules.

Rule design notes:
- Secrets rules follow the Gitleaks style: anchored vendor prefixes where
  they exist, keyword-context assignment plus an entropy gate for the
  generic case. Research on secrets scanners shows no single ruleset gets
  both precision and recall, so the set is deliberately user-extensible.
- Validators (Luhn, entropy) run after the regex match and can veto it.
"""

import datetime
import ipaddress
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from llm_redact.detection.base import Detection
from llm_redact.detection.wallet_checksums import (
    base58check_ok,
    bech32_address_ok,
    eth_address_ok,
)


@dataclass(frozen=True)
class RegexRule:
    name: str
    detector_type: str
    pattern: re.Pattern[str]
    validator: Callable[[re.Match[str]], bool] | None = None
    priority: int = 100
    # Which regex group holds the sensitive value; 0 = whole match.
    group: int = 0
    # Prefilter, in CNF: the regex CANNOT match unless, for EVERY inner
    # group, at least one of its literals appears in the text. str-in is
    # memchr-fast, so a body that lacks "AKIA" never pays for the AWS scan.
    # Empty = always scan. A wrong declaration is a silent recall bug, so
    # the soundness test asserts every recall-corpus match contains a
    # literal from each group.
    required: tuple[tuple[str, ...], ...] = ()
    # Test the literals against the lowercased haystack (for (?i) rules;
    # declare the literals lowercase).
    required_ci: bool = False
    # Stronger premise than `required`: EVERY match of the pattern starts
    # with one of these literals. When declared, the scan becomes
    # find-then-match: str.find each anchor occurrence and attempt
    # pattern.match(text, pos) there instead of running finditer over the
    # whole body. pattern.match at pos keeps \b and lookbehinds correct —
    # they evaluate against the real neighboring characters. The soundness
    # test asserts every recall-corpus match starts with a declared anchor.
    anchors: tuple[str, ...] = ()
    # Find anchors in the lowered haystack ((?i) rules; declare lowercase).
    anchors_ci: bool = False
    # Languages this rule's identifier belongs to (ISO 639-1 codes). None =
    # universal (emails, IPs, vendor tokens, credit cards, IBANs, phones):
    # such rules run regardless of [detection] languages. A tagged rule is
    # NOT BUILT when the configured language list shares no entry with it.
    languages: tuple[str, ...] | None = None


class PreparedText:
    """A haystack shared across all rules of one detect_all call, with the
    lowercase form computed at most once (for case-insensitive prefilters)."""

    __slots__ = ("text", "_lower")

    def __init__(self, text: str) -> None:
        self.text = text
        self._lower: str | None = None

    @property
    def lower(self) -> str:
        if self._lower is None:
            self._lower = self.text.lower()
        return self._lower


def _luhn_checksum(digits: list[int]) -> int:
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10


def _luhn_ok(match: re.Match[str]) -> bool:
    digits = [int(c) for c in re.sub(r"[ -]", "", match.group(0))]
    if not 13 <= len(digits) <= 19:
        return False
    if not any(digits):
        # All zeros satisfies the checksum arithmetic but no card network
        # issues it — it's a placeholder shape common in docs and configs.
        return False
    return _luhn_checksum(digits) == 0


def _sin_ok(match: re.Match[str]) -> bool:
    """Canadian SIN: Luhn plus the one never-assigned first digit.

    Area 8 has never been issued (pure false-positive guard). Area 0 is
    fictitious-only — the government's own example 046 454 286 — and IS
    matched: redacting a doc example round-trips invisibly, while a real
    number mistaken for one would leak."""
    digits = [int(c) for c in re.sub(r"[ -]", "", match.group(0))]
    if digits[0] == 8:
        return False
    return _luhn_checksum(digits) == 0


# HMRC-published administrative pairs that are never issued.
_NINO_INVALID_PREFIXES = frozenset({"BG", "GB", "NK", "KN", "TN", "NT", "ZZ"})


def _nino_ok(match: re.Match[str]) -> bool:
    return match.group(0).replace(" ", "")[:2] not in _NINO_INVALID_PREFIXES


# Verhoeff dihedral-group tables (Aadhaar's checksum). The corpus generator
# deliberately derives these from the D5 group operation instead of copying
# them, so a transcription typo in either place fails the recall gate.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def _aadhaar_ok(match: re.Match[str]) -> bool:
    """Aadhaar: Verhoeff over all 12 digits (first digit 2-9 is in the
    pattern — UIDAI never issues 0/1-leading numbers)."""
    digits = [int(c) for c in re.sub(r"[ -]", "", match.group(0))]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[i % 8][d]]
    return checksum == 0


_TFN_WEIGHTS = (1, 4, 3, 7, 5, 8, 6, 9, 10)


def _tfn_ok(match: re.Match[str]) -> bool:
    """Australian TFN: ATO weighted checksum, whole number mod 11 == 0."""
    digits = [int(c) for c in re.sub(r"[ -]", "", match.group(0))]
    return sum(w * d for w, d in zip(_TFN_WEIGHTS, digits, strict=True)) % 11 == 0


_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


def _dni_ok(match: re.Match[str]) -> bool:
    """Spanish DNI/NIE: control letter = table[number mod 23]; NIE prefixes
    X/Y/Z stand for 0/1/2 in the computation."""
    value = match.group(0).replace("-", "")
    body, letter = value[:-1], value[-1]
    if body[0] in "XYZ":
        body = str("XYZ".index(body[0])) + body[1:]
    return _DNI_LETTERS[int(body) % 23] == letter


def _nir_ok(match: re.Match[str]) -> bool:
    """French NIR: two-digit key == 97 - (13-digit number mod 97).

    Corsican departments write 2A/2B in the department slot; the official
    key computation substitutes 0 for the letter and subtracts 1,000,000
    (2A) or 2,000,000 (2B) before the modulus."""
    value = match.group(0).replace(" ", "")
    body, key = value[:-2], value[-2:]
    if "A" in body:
        number = int(body.replace("A", "0")) - 1_000_000
    elif "B" in body:
        number = int(body.replace("B", "0")) - 2_000_000
    else:
        number = int(body)
    return int(key) == 97 - number % 97


def _cpf_ok(match: re.Match[str]) -> bool:
    """Brazilian CPF: two weighted mod-11 check digits over the dotted
    display form. All-same-digit numbers pass the arithmetic but are never
    issued, so they are rejected."""
    digits = [int(c) for c in re.sub(r"[.-]", "", match.group(0))]
    if len(set(digits[:9])) == 1:
        return False
    for n in (9, 10):
        total = sum(d * w for d, w in zip(digits[:n], range(n + 1, 1, -1), strict=True))
        if (total * 10) % 11 % 10 != digits[n]:
            return False
    return True


# Check-character table for the ODD (1-indexed) positions of a codice
# fiscale; even positions use the plain value (digit, or letter index).
# The corpus generator transcribes this table INDEPENDENTLY — a typo on
# either side fails the recall gate — and the pinned real-world example
# in the tests (RSSMRA85T10A562S) checks out against both.
_CF_ODD = {
    "0": 1, "1": 0, "2": 5, "3": 7, "4": 9, "5": 13, "6": 15, "7": 17, "8": 19, "9": 21,
    "A": 1, "B": 0, "C": 5, "D": 7, "E": 9, "F": 13, "G": 15, "H": 17, "I": 19, "J": 21,
    "K": 2, "L": 4, "M": 18, "N": 20, "O": 11, "P": 3, "Q": 6, "R": 8, "S": 12, "T": 14,
    "U": 16, "V": 10, "W": 22, "X": 25, "Y": 24, "Z": 23,
}  # fmt: skip


def _codice_fiscale_ok(match: re.Match[str]) -> bool:
    """Italian codice fiscale: day-of-birth range (women add 40) plus the
    mod-26 check letter over odd/even position tables. The omocodia
    letter-substitution variants are deliberately out of scope — their
    grammar collides with random uppercase identifiers."""
    value = match.group(0)
    day = int(value[9:11])
    if not (1 <= day <= 31 or 41 <= day <= 71):
        return False
    total = 0
    for i, ch in enumerate(value[:15]):
        if i % 2 == 0:  # 0-indexed even = spec's odd positions
            total += _CF_ODD[ch]
        else:
            total += int(ch) if ch.isdigit() else ord(ch) - 65
    return chr(total % 26 + 65) == value[15]


def _ahv_ok(match: re.Match[str]) -> bool:
    """Swiss AHV/AVS number: EAN-13 check digit over the dotted display
    form (the 756 country prefix is in the pattern). Verified against the
    official example 756.9217.0769.85."""
    digits = [int(c) for c in match.group(0).replace(".", "")]
    total = sum(d * (3 if i % 2 else 1) for i, d in enumerate(digits[:12]))
    return (10 - total % 10) % 10 == digits[12]


def _personnummer_ok(match: re.Match[str]) -> bool:
    """Swedish personnummer: month/day ranges (coordination numbers add 60
    to the day and are ACCEPTED — they are real identifiers for real
    people) plus the Luhn variant over all 10 digits (alternating 2,1
    weights starting at the first digit)."""
    value = match.group(0)
    digits = [int(c) for c in value if c.isdigit()]
    month = digits[2] * 10 + digits[3]
    day = digits[4] * 10 + digits[5]
    if not (1 <= month <= 12 and (1 <= day <= 31 or 61 <= day <= 91)):
        return False
    total = 0
    for i, d in enumerate(digits):
        doubled = d * 2 if i % 2 == 0 else d
        total += doubled - 9 if doubled > 9 else doubled
    return total % 10 == 0


def _belgian_nn_ok(match: re.Match[str]) -> bool:
    """Belgian Rijksregisternummer (YY.MM.DD-NNN.CC): the trailing two
    check digits equal 97 - (9-digit body mod 97). Births from 2000 on
    prepend a '2' to the 9-digit body before the modulus; the birth
    century is not knowable from YY alone, so both forms are accepted.
    Month/day may be 0 (unknown birth date is a legitimate Belgian NN)."""
    value = re.sub(r"[.-]", "", match.group(0))
    body, check = value[:9], int(value[9:])
    month, day = int(body[2:4]), int(body[4:6])
    if not (0 <= month <= 12 and 0 <= day <= 31):
        return False
    return any(check == 97 - int(prefix + body) % 97 for prefix in ("", "2"))


# Finnish HETU check-character table (index = 9-digit number mod 31) and the
# accepted century signs (1800s '+', 1900s '-YXWVU', 2000s 'ABCDEF'). The
# corpus generator transcribes this table INDEPENDENTLY, so a typo on either
# side fails the recall gate; 131052-308T (a published example) pins it.
_HETU_CHECK = "0123456789ABCDEFHJKLMNPRSTUVWXY"
_HETU_CENTURY = frozenset("+-YXWVUABCDEF")


def _hetu_ok(match: re.Match[str]) -> bool:
    """Finnish henkilötunnus: DDMMYY + century sign + 3-digit individual
    number + a mod-31 check character over the 9 digits."""
    value = match.group(0)
    ddmmyy, sign, zzz, check = value[:6], value[6], value[7:10], value[10]
    if sign not in _HETU_CENTURY:
        return False
    day, month = int(ddmmyy[:2]), int(ddmmyy[2:4])
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return False
    return _HETU_CHECK[int(ddmmyy + zzz) % 31] == check


def _nhs_ok(match: re.Match[str]) -> bool:
    """UK NHS Number: 10 digits, mod-11 over the first nine (weights 10..2),
    check = 11 - (sum mod 11) with 11 -> 0 and 10 -> invalid. Matched in the
    3-3-4 spaced display form only; a bare 10-digit run never fires."""
    digits = match.group(0).replace(" ", "")
    total = sum(int(d) * (10 - i) for i, d in enumerate(digits[:9]))
    check = 11 - total % 11
    if check == 11:
        check = 0
    return check != 10 and check == int(digits[9])


# Norwegian fødselsnummer control-digit weights (Skatteetaten). Transcribed
# independently in the corpus generator so a typo fails the recall gate.
_FNR_K1 = (3, 7, 6, 1, 8, 9, 4, 5, 2)
_FNR_K2 = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)


def _norwegian_fnr_ok(match: re.Match[str]) -> bool:
    """Norwegian fødselsnummer: 11 digits, TWO mod-11 control digits (k1 over
    the first nine, k2 over the first ten); either landing on 10 makes the
    number invalid. Spaced/hyphenated DDMMYY-NNNNN display form only."""
    digits = re.sub(r"\D", "", match.group(0))
    k1 = 11 - sum(int(digits[i]) * _FNR_K1[i] for i in range(9)) % 11
    if k1 == 11:
        k1 = 0
    if k1 == 10 or k1 != int(digits[9]):
        return False
    k2 = 11 - sum(int(digits[i]) * _FNR_K2[i] for i in range(10)) % 11
    if k2 == 11:
        k2 = 0
    return k2 != 10 and k2 == int(digits[10])


# Korean RRN control-digit weights; transcribed independently in the corpus.
_RRN_W = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)


def _rrn_ok(match: re.Match[str]) -> bool:
    """Korean resident registration number YYMMDD-SBBBBNC: 13 digits in the
    hyphenated 6-7 display form only (a bare run never fires). mod-11 check
    over the first twelve, check = (11 - sum mod 11) mod 10; month/day and a
    1-8 gender-century digit gate the birthdate half."""
    digits = match.group(0)[:6] + match.group(0)[7:]
    month, day = int(digits[2:4]), int(digits[4:6])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return False
    if digits[6] not in "12345678":
        return False
    check = (11 - sum(int(digits[i]) * _RRN_W[i] for i in range(12)) % 11) % 10
    return check == int(digits[12])


# Singapore NRIC/FIN checksum tables (S/T share one, F/G the other).
_NRIC_W = (2, 7, 6, 5, 4, 3, 2)
_NRIC_ST = "JZIHGFEDCBA"
_NRIC_FG = "XWUTRQPNMLK"


def _nric_ok(match: re.Match[str]) -> bool:
    """Singapore NRIC/FIN: prefix S/T/F/G + 7 digits + a checksum letter. The
    weighted sum (T/G add 4) mod 11 indexes the prefix-specific letter table;
    the bracketing letters make the solid form safe (like the DNI letter)."""
    value = match.group(0)
    prefix, body, check = value[0], value[1:8], value[8]
    total = sum(int(body[i]) * _NRIC_W[i] for i in range(7))
    if prefix in "TG":
        total += 4
    table = _NRIC_ST if prefix in "ST" else _NRIC_FG
    return table[total % 11] == check


# Chinese Resident Identity Card: ISO 7064 MOD 11-2 weights are 2^(17-i)
# mod 11 (the corpus transcribes the PUBLISHED table independently, so a
# derivation bug here fails the recall gate); check map per GB 11643.
_CN_ID_W = tuple(pow(2, 17 - i, 11) for i in range(17))
_CN_ID_CHECK = "10X98765432"
# Valid province/region prefixes (GB/T 2260 top-level codes incl. Taiwan,
# Hong Kong, and Macao as issued on mainland cards).
_CN_ID_PROVINCES = frozenset(
    {11, 12, 13, 14, 15, 21, 22, 23, 31, 32, 33, 34, 35, 36, 37}
    | {41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54}
    | {61, 62, 63, 64, 65, 71, 81, 82}
)


def _cn_resident_ok(match: re.Match[str]) -> bool:
    """Chinese Resident ID (GB 11643), solid 18-char form: 6-digit region +
    YYYYMMDD + 3-digit sequence + MOD 11-2 check char. The province gate,
    a REAL calendar date (Feb 30 rejected, not just shape), and the check
    character together carry the precision the separator-less form lacks."""
    value = match.group(0).upper()
    if int(value[:2]) not in _CN_ID_PROVINCES:
        return False
    try:
        datetime.date(int(value[6:10]), int(value[10:12]), int(value[12:14]))
    except ValueError:
        return False
    total = sum(int(value[i]) * _CN_ID_W[i] for i in range(17))
    return _CN_ID_CHECK[total % 11] == value[17]


def _my_number_ok(match: re.Match[str]) -> bool:
    """Japan Individual Number (My Number), 4-4-4 display form: 11 body
    digits + a check digit. The Cabinet Ordinance defines weights on digit
    positions counted from the RIGHT of the body (q = m+1 for m <= 6, else
    m-5); the corpus transcribes the folded left-to-right weight tuple
    independently, so a derivation bug here fails the recall gate."""
    digits = [int(c) for c in match.group(0).replace(" ", "").replace("-", "")]
    body, check = digits[:11], digits[11]
    total = sum(
        d * ((m + 1) if m <= 6 else (m - 5)) for m, d in zip(range(11, 0, -1), body, strict=True)
    )
    remainder = total % 11
    return check == (0 if remainder <= 1 else 11 - remainder)


def _thai_id_ok(match: re.Match[str]) -> bool:
    """Thai Citizen ID, dashed 1-4-5-2-1 display form: the first 12 digits
    weighted 13..2, mod 11, folded to one digit by (11 - r) mod 10."""
    digits = [int(c) for c in match.group(0).replace("-", "")]
    total = sum(d * (13 - i) for i, d in enumerate(digits[:12]))
    return digits[12] == (11 - total % 11) % 10


_PPS_ALPHABET = "WABCDEFGHIJKLMNOPQRSTUV"


def _pps_ok(match: re.Match[str]) -> bool:
    """Irish PPS number: 7 digits weighted 8..2, plus an optional ninth
    character (legacy W contributes nothing; post-2013 A/H contributes its
    alphabet value times 9); the sum mod 23 indexes the check letter
    (0 = W, 1 = A, ..., 22 = V)."""
    value = match.group(0)
    total = sum(int(value[i]) * (8 - i) for i in range(7))
    if len(value) == 9 and value[8] != "W":
        total += (ord(value[8]) - 64) * 9
    return _PPS_ALPHABET[total % 23] == value[7]


# RENAPO's check charset includes Ñ (value 24), which shifts every letter
# from O upward — an ASCII-only table would validate nothing past N.
_CURP_CHARSET = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
_CURP_STATES = frozenset(
    {
        "AS",
        "BC",
        "BS",
        "CC",
        "CL",
        "CM",
        "CS",
        "CH",
        "DF",
        "DG",
        "GT",
        "GR",
        "HG",
        "JC",
        "MC",
        "MN",
        "MS",
        "NT",
        "NL",
        "OC",
        "PL",
        "QT",
        "QR",
        "SP",
        "SL",
        "SR",
        "TC",
        "TS",
        "TL",
        "VZ",
        "YN",
        "ZS",
        "NE",
    }
)


def _curp_ok(match: re.Match[str]) -> bool:
    """Mexican CURP: state-code gate, a REAL calendar date (the century
    comes from the homoclave — digit means 1900s, letter means 2000s, per
    RENAPO), and the mod-10 check digit over the Ñ-bearing charset."""
    value = match.group(0)
    if value[11:13] not in _CURP_STATES:
        return False
    century = 1900 if value[16].isdigit() else 2000
    try:
        datetime.date(century + int(value[4:6]), int(value[6:8]), int(value[8:10]))
    except ValueError:
        return False
    total = sum(_CURP_CHARSET.index(value[i]) * (18 - i) for i in range(17))
    return (10 - total % 10) % 10 == int(value[17])


def _steuer_id_ok(match: re.Match[str]) -> bool:
    """German Steuer-ID: structural repeat rule + ISO 7064 MOD 11,10.

    Among the first ten digits exactly one value appears twice or (since
    2016) three times and the rest are distinct; the eleventh digit is the
    ISO 7064 MOD 11,10 check over the first ten."""
    digits = [int(c) for c in match.group(0).replace(" ", "")]
    first10, check = digits[:10], digits[10]
    counts: dict[int, int] = {}
    for d in first10:
        counts[d] = counts.get(d, 0) + 1
    repeats = sorted(c for c in counts.values() if c > 1)
    if repeats != [2] and repeats != [3]:
        return False
    product = 10
    for d in first10:
        s = (d + product) % 10
        if s == 0:
            s = 10
        product = (2 * s) % 11
    expected = 11 - product
    if expected == 10:
        expected = 0
    return check == expected


def _valid_ipv4(match: re.Match[str]) -> bool:
    try:
        ipaddress.IPv4Address(match.group(0))
    except ValueError:
        return False
    return True


def _valid_ipv6(match: re.Match[str]) -> bool:
    """ipaddress.IPv6Address plus two gates the grammar can't express.

    Plain parsing accepts shapes that are overwhelmingly NOT addresses in
    real text: Python slices ("x[::2]" contains the valid address ::2) and
    colon-separated hex pairs (a certificate serial like
    04:9f:86:d0:81:88:4c:7d IS a valid IPv6 spelling). Gate 1 requires a
    hex letter or at least four non-empty groups — real addresses in prose
    essentially always have one; slice offsets have neither. Gate 2 vetoes
    exactly-eight two-hex-char groups — the serial/MAC-dump shape that
    nobody uses to write a real address (they zero-trim).
    """
    value = match.group(0)
    try:
        ipaddress.IPv6Address(value)
    except ValueError:
        return False
    non_empty = [g for g in value.split(":") if g]
    has_letter = any(c in "abcdefABCDEF" for c in value)
    if not has_letter and len(non_empty) < 4:
        return False
    return not (len(non_empty) == 8 and all(len(g) == 2 for g in non_empty))


def _phone_ok(match: re.Match[str]) -> bool:
    # >= 8 digits also rejects a truncated match the trailing boundary can
    # produce (e.g. "+1 415 555" left over from "+1 415 555 0100-").
    digits = sum(ch.isdigit() for ch in match.group(0))
    return 8 <= digits <= 15


def _ssn_ok(match: re.Match[str]) -> bool:
    area, group, serial = match.group(0).split("-")
    if area in ("000", "666") or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


# Official IBAN lengths for common countries; other country codes fall back
# to the checksum alone (the grammar already bounds 15-34 chars).
# fmt: off
_IBAN_LENGTHS = {
    "AD": 24, "AT": 20, "BE": 16, "CH": 21, "CZ": 24, "DE": 22, "DK": 18,
    "ES": 24, "FI": 18, "FR": 27, "GB": 22, "IE": 22, "IT": 27, "LU": 20,
    "NL": 18, "NO": 15, "PL": 28, "PT": 25, "SE": 24,
}
# fmt: on


def _iban_ok(match: re.Match[str]) -> bool:
    value = match.group(0)
    expected = _IBAN_LENGTHS.get(value[:2])
    if expected is not None and len(value) != expected:
        return False
    # ISO 13616 mod-97: move the first four chars to the end, map letters to
    # 10..35, and the whole number must be ≡ 1 (mod 97). This eliminates
    # random uppercase-alphanumeric identifiers 96 times out of 97.
    rearranged = value[4:] + value[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _entropy_gate(match: re.Match[str]) -> bool:
    return shannon_entropy(match.group("value")) >= 3.5


def _whole_entropy_gate(match: re.Match[str]) -> bool:
    return shannon_entropy(match.group(0)) >= 3.5


def _base64_gate(match: re.Match[str]) -> bool:
    value = match.group("value")
    return len(value) % 4 == 0 and shannon_entropy(value) >= 3.5


def _jwt_header_ok(match: re.Match[str]) -> bool:
    """The first segment of a real JWT base64url-decodes to a JSON object."""
    import base64
    import binascii
    import json as _json

    header = match.group(0).split(".", 1)[0]
    padded = header + "=" * (-len(header) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded)
        return isinstance(_json.loads(decoded), dict)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return False


# Prefilter literal groups shared by the digit-punctuation rules: any ipv4
# match contains a digit-then-dot pair, any hyphenated SSN a digit-then-dash.
_DIGIT_DOT = tuple(f"{d}." for d in "0123456789")
_DIGIT_DASH = tuple(f"{d}-" for d in "0123456789")
_DIGIT_SPACE = tuple(f"{d} " for d in "0123456789")
_DIGIT_DOT = tuple(f"{d}." for d in "0123456789")
_ANY_DIGIT = tuple("0123456789")

BUILTIN_RULES: tuple[RegexRule, ...] = (
    RegexRule(
        name="email",
        detector_type="EMAIL",
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        required=(("@",),),
    ),
    RegexRule(
        name="ipv4",
        detector_type="IPV4",
        pattern=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        validator=_valid_ipv4,
        required=(_DIGIT_DOT,),
    ),
    RegexRule(
        # Loose colon-hex candidate; the validator does the real work (see
        # _valid_ipv6). The repetition bound keeps long hex-pair dumps from
        # ever forming a candidate.
        name="ipv6",
        detector_type="IPV6",
        pattern=re.compile(r"(?<![\w:.])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,8}(?![\w:.])"),
        validator=_valid_ipv6,
        required=((":",),),
    ),
    RegexRule(
        # scheme://user:password@host — redacts ONLY the password. No
        # entropy gate: real connection-string passwords are often
        # low-entropy, and a false redaction round-trips invisibly.
        name="url_credentials",
        detector_type="URL_PASSWORD",
        pattern=re.compile(r"\b[a-z][a-z0-9+.-]{1,15}://[^\s/@:]{1,64}:(?P<value>[^\s@/]{1,256})@"),
        group=1,
        priority=10,
        required=(("://",),),
    ),
    RegexRule(
        # Anchored to end on a digit so a trailing separator is never
        # swallowed into the detected span.
        name="credit_card",
        detector_type="CREDIT_CARD",
        pattern=re.compile(r"\b\d(?:[ -]?\d){12,18}\b"),
        validator=_luhn_ok,
    ),
    RegexRule(
        # Precision-first grammar: E.164 (+country, separator-tolerant) or
        # national formats WITH separators. Bare digit runs ("4155550100")
        # are deliberately never matched — they collide with IDs, versions,
        # and timestamps far too often.
        name="phone_number",
        detector_type="PHONE",
        pattern=re.compile(
            r"""(?x)
            (?<![\w.+-])
            (?:
                \+[1-9]\d{0,2}                  # +country code
                (?:[ .-]?\(\d{1,4}\))?          # optional (area)
                (?:[ .-]?\d{2,4}){2,5}          # grouped digits
              | \(\d{3}\)[ .-]?\d{3}[ .-]\d{4}  # (212) 555-0100
              | \d{3}[.-]\d{3}[.-]\d{4}         # 212-555-0100 / 212.555.0100
            )
            (?![\w-])
            """
        ),
        validator=_phone_ok,
        # One alternative needs "+", one "(", one a digit-punct pair.
        required=(("+", "(") + _DIGIT_DASH + _DIGIT_DOT,),
    ),
    RegexRule(
        # Hyphenated form only: a bare 9-digit run is too collision-prone.
        name="us_ssn",
        detector_type="SSN",
        pattern=re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])"),
        validator=_ssn_ok,
        languages=("en",),
        required=(_DIGIT_DASH,),
    ),
    RegexRule(
        name="iban",
        detector_type="IBAN",
        pattern=re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
        validator=_iban_ok,
    ),
    RegexRule(
        # Ethereum (and every EVM chain): 0x + 40 hex. A mixed-case address
        # must satisfy the EIP-55 keccak-256 checksum; an all-lower/all-upper
        # address carries no checksum and is accepted on shape. The single-
        # nibble null/burn address is rejected as a placeholder.
        name="eth_address",
        detector_type="ETH_ADDRESS",
        pattern=re.compile(r"(?<![0-9A-Za-z])0x[0-9a-fA-F]{40}(?![0-9A-Za-z])"),
        validator=eth_address_ok,
        priority=20,
        required=(("0x",),),
        anchors=("0x",),
    ),
    RegexRule(
        # Bitcoin legacy P2PKH (1…) and P2SH (3…): base58check, version byte
        # 0x00/0x05, double-SHA256 checksum. No anchor literal (like iban).
        name="btc_address",
        detector_type="BTC_ADDRESS",
        pattern=re.compile(r"(?<![A-Za-z0-9])[13][1-9A-HJ-NP-Za-km-z]{25,34}(?![A-Za-z0-9])"),
        validator=base58check_ok,
        priority=20,
    ),
    RegexRule(
        # Bitcoin native segwit (bc1…): bech32 for witness v0, bech32m for v1+
        # (taproot). The data charset excludes 1/b/i/o. Same BTC_ADDRESS type.
        name="btc_bech32",
        detector_type="BTC_ADDRESS",
        pattern=re.compile(r"(?<![A-Za-z0-9])bc1[ac-hj-np-z02-9]{8,87}(?![A-Za-z0-9])"),
        validator=bech32_address_ok,
        priority=20,
        required=(("bc1",),),
        anchors=("bc1",),
    ),
    RegexRule(
        # Separated triads only ("046 454 286" / "046-454-286", one
        # consistent separator): a bare 9-digit run is too collision-prone —
        # the same stance as us_ssn.
        name="canadian_sin",
        detector_type="CA_SIN",
        pattern=re.compile(r"(?<![\w-])\d{3}([ -])\d{3}\1\d{3}(?![\w-])"),
        validator=_sin_ok,
        languages=("en", "fr"),
        # Every match contains a digit followed by its separator.
        required=(_DIGIT_SPACE + _DIGIT_DASH,),
    ),
    RegexRule(
        # HMRC grammar: no D/F/I/Q/U/V in either prefix letter, no O second,
        # suffix A-D; solid ("QQ123456C") or space-grouped ("QQ 12 34 56 C").
        # Uppercase only — the lowercase form barely occurs in real text and
        # the case restriction keeps identifiers from firing.
        name="uk_nino",
        detector_type="UK_NINO",
        pattern=re.compile(r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z](?: ?\d{2}){3} ?[A-D]\b"),
        validator=_nino_ok,
        languages=("en",),
        required=(_ANY_DIGIT,),
    ),
    RegexRule(
        # Displayed 4-4-4 groups only (bare 12-digit runs are banned, the
        # us_ssn stance); UIDAI numbers never start 0/1; Verhoeff-checked.
        name="aadhaar",
        detector_type="AADHAAR",
        pattern=re.compile(r"(?<![\w-])[2-9]\d{3}([ -])\d{4}\1\d{4}(?![\w-])"),
        validator=_aadhaar_ok,
        languages=("en", "hi"),
        required=(_DIGIT_SPACE + _DIGIT_DASH,),
    ),
    RegexRule(
        # Separated triads, the same shape as canadian_sin (which sits
        # earlier in this list and wins exact-span ties); the ATO weighted
        # checksum keeps random digit groups out, and the corpus generators
        # keep the two rules' values disjoint (TFNs generated with leading
        # 8, which _sin_ok rejects).
        name="australian_tfn",
        detector_type="AU_TFN",
        pattern=re.compile(r"(?<![\w-])\d{3}([ -])\d{3}\1\d{3}(?![\w-])"),
        validator=_tfn_ok,
        languages=("en",),
        required=(_DIGIT_SPACE + _DIGIT_DASH,),
    ),
    RegexRule(
        # DNI (8 digits) / NIE (X|Y|Z + 7 digits) with the mod-23 control
        # letter; optional hyphen only — allowing a space before the letter
        # would pull in "12345678 A"-style prose. Like credit_card, declares
        # no prefilter literals (solid digits have no stable substring).
        name="spanish_dni",
        detector_type="ES_DNI",
        pattern=re.compile(r"(?<![\w.-])(?:\d{8}|[XYZ]\d{7})-?[A-Z](?![\w-])"),
        validator=_dni_ok,
        languages=("es",),
    ),
    RegexRule(
        # Space-grouped display form only (1 85 03 75 116 384 69) — the
        # solid 15-digit run is banned like every bare digit run. The
        # department slot admits Corsican 2A/2B.
        name="french_nir",
        detector_type="FR_NIR",
        pattern=re.compile(
            r"(?<![\w-])[12] \d{2} \d{2} (?:\d{2}|2[AB]) \d{3} \d{3} \d{2}(?![\w-])"
        ),
        validator=_nir_ok,
        languages=("fr",),
        # Beats credit_card on exact-span ties: 1/10 of NIRs are also
        # Luhn-valid and the loose card grammar (13-19 digits, any [ -]
        # grouping) covers the whole spaced NIR. The 1-2-2-2-3-3-2 display
        # grouping is NIR-specific — no real card is written that way, and
        # a 16-digit card can never match this pattern in reverse.
        priority=90,
        required=(_DIGIT_SPACE,),
    ),
    RegexRule(
        # Space-grouped display form only (12 345 678 901); leading digit
        # nonzero per the spec; repeat-structure + ISO 7064 checked.
        name="german_steuer_id",
        detector_type="DE_STEUER_ID",
        pattern=re.compile(r"(?<![\w-])[1-9]\d \d{3} \d{3} \d{3}(?![\w-])"),
        validator=_steuer_id_ok,
        languages=("de",),
        required=(_DIGIT_SPACE,),
    ),
    RegexRule(
        # Dotted-dashed display form only (000.000.000-00) — the solid
        # 11-digit run is banned like every bare digit run.
        name="brazilian_cpf",
        detector_type="BR_CPF",
        pattern=re.compile(r"(?<![\w.-])\d{3}\.\d{3}\.\d{3}-\d{2}(?![\w-])"),
        validator=_cpf_ok,
        languages=("pt",),
        required=(_DIGIT_DOT,),
    ),
    RegexRule(
        # Fixed 16-char grammar: 6 name letters, 2-digit year, month
        # letter, day (women +40, checked in the validator), place code,
        # check letter. Uppercase display form only; no stable prefilter
        # literal exists (like credit_card, declares none).
        name="italian_codice_fiscale",
        detector_type="IT_CF",
        pattern=re.compile(r"(?<![\w-])[A-Z]{6}\d{2}[ABCDEHLMPRST]\d{2}[A-Z]\d{3}[A-Z](?![\w-])"),
        validator=_codice_fiscale_ok,
        languages=("it",),
    ),
    RegexRule(
        # Dotted display form only (756.XXXX.XXXX.XX) — the 756 country
        # prefix is both the grammar anchor and the prefilter literal.
        name="swiss_ahv",
        detector_type="CH_AHV",
        pattern=re.compile(r"(?<![\w.-])756\.\d{4}\.\d{4}\.\d{2}(?![\w.-])"),
        validator=_ahv_ok,
        languages=("de", "fr", "it"),
        required=(("756.",),),
        anchors=("756.",),
    ),
    RegexRule(
        # Separator display form only (YYMMDD-NNNN; + marks 100-year-olds).
        # The bare 10/12-digit machine forms never fire (digit-run ban);
        # coordination numbers (day+60) are accepted in the validator.
        name="swedish_personnummer",
        detector_type="SE_PNR",
        pattern=re.compile(r"(?<![\w.+-])\d{6}[+-]\d{4}(?![\w-])"),
        validator=_personnummer_ok,
        languages=("sv",),
        required=(_DIGIT_DASH + tuple(f"{d}+" for d in "0123456789"),),
    ),
    RegexRule(
        # Dotted-dashed display form only (YY.MM.DD-NNN.CC) — the bare
        # 11-digit run never fires. mod-97 check; the century-prefix rule
        # is in the validator. 93.05.18-223.61 pins it.
        name="belgian_nn",
        detector_type="BE_NN",
        pattern=re.compile(r"(?<![\w.-])\d{2}\.\d{2}\.\d{2}-\d{3}\.\d{2}(?![\w.-])"),
        validator=_belgian_nn_ok,
        languages=("nl", "fr", "de"),
        required=(_DIGIT_DOT,),
    ),
    RegexRule(
        # DDMMYY + century sign + NNN + mod-31 check char. The century sign
        # (a letter or +/-) makes the solid form safe — no separator needed,
        # like italian_codice_fiscale. 131052-308T pins it.
        name="finnish_hetu",
        detector_type="FI_HETU",
        pattern=re.compile(r"(?<![\w-])\d{6}[-+A-FU-Y]\d{3}[0-9A-Y](?![\w-])"),
        validator=_hetu_ok,
        languages=("fi",),
    ),
    RegexRule(
        # 3-3-4 spaced display form only (the bare 10-digit run never fires).
        # mod-11 check; the spaced form is disjoint from the phone grammar,
        # which does not claim plain-space 3-3-4. 943 476 5919 pins it.
        name="nhs_number",
        detector_type="NHS_NUMBER",
        pattern=re.compile(r"(?<![\w-])\d{3} \d{3} \d{4}(?![\w-])"),
        validator=_nhs_ok,
        languages=("en",),
    ),
    RegexRule(
        # DDMMYY-NNNNN spaced/hyphenated display form; double mod-11 (1/121
        # random pass), so the loose date is left to the checksum like the
        # other Nordics. 15038550060 shape pins it.
        name="norwegian_fnr",
        detector_type="NO_FNR",
        pattern=re.compile(r"(?<![\w-])\d{6}[ -]\d{5}(?![\w-])"),
        validator=_norwegian_fnr_ok,
        languages=("no", "nb", "nn"),
    ),
    RegexRule(
        # Korean RRN, hyphenated 6-7 display form only; mod-11 check plus
        # month/day/gender-digit gates. A bare 13-digit run never fires.
        # priority 90 (like french_nir): the 13-digit hyphenated form is also a
        # valid credit_card grammar span, and ~1/10 pass Luhn — the RRN-specific
        # grouping must win that exact-span tie.
        name="korean_rrn",
        detector_type="KR_RRN",
        pattern=re.compile(r"(?<![\w-])\d{6}-\d{7}(?![\w-])"),
        validator=_rrn_ok,
        priority=90,
        languages=("ko",),
    ),
    RegexRule(
        # Singapore NRIC/FIN: [STFG] + 7 digits + checksum letter. The letter
        # bracketing makes the solid form safe; 1/11 random letter pass rate.
        name="singapore_nric",
        detector_type="SG_NRIC",
        pattern=re.compile(r"(?<![A-Za-z0-9])[STFG]\d{7}[A-Z](?![A-Za-z0-9])"),
        validator=_nric_ok,
        languages=("en",),
    ),
    RegexRule(
        # Chinese Resident Identity Card (GB 11643): the customary display
        # form is the SOLID 18 chars — no separators to lean on, so
        # precision rests on the province gate, the embedded calendar date
        # (the grammar already pins 19xx/20xx and month/day shapes), and
        # the MOD 11-2 check char. Priority 90 because the loose
        # 13-19-digit card grammar covers an all-digit ID and ~1/10 are
        # Luhn-valid — the ID-specific structure must win the exact-span
        # tie (the korean_rrn/french_nir rule).
        name="chinese_resident_id",
        detector_type="CN_RESIDENT_ID",
        pattern=re.compile(
            r"(?<![\w-])\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
            r"\d{3}[0-9Xx](?![\w-])"
        ),
        validator=_cn_resident_ok,
        priority=90,
        languages=("zh",),
    ),
    RegexRule(
        # Japan Individual Number, 4-4-4 grouped display form only (bare
        # 12-digit runs never fire, the aadhaar stance). The grammar
        # overlaps aadhaar's when the lead digit is 2-9: a value passing
        # BOTH checksums (~1/11 of those) is claimed by aadhaar on
        # registration order — still redacted, only typed differently.
        # The corpus keeps its values disjoint by generating lead 0/1,
        # which aadhaar's grammar rejects outright.
        name="japanese_my_number",
        detector_type="JP_MY_NUMBER",
        # The extra digit-group lookarounds keep the rule off 4-4-4
        # SUBSPANS of longer separated runs (a 16-digit card is 4-4-4-4,
        # and ~1/11 of its 12-digit windows pass the mod-11 check).
        pattern=re.compile(r"(?<!\d[ -])(?<![\w-])\d{4}([ -])\d{4}\1\d{4}(?![\w-])(?![ -]\d)"),
        validator=_my_number_ok,
        languages=("ja",),
        required=(_DIGIT_SPACE + _DIGIT_DASH,),
    ),
    RegexRule(
        # Thai Citizen ID, dashed 1-4-5-2-1 display form only; lead digit
        # 1-8 (person-type). Priority 90 because the loose 13-19-digit
        # card grammar also covers a dashed 13-digit run and ~1/10 pass
        # Luhn — the ID-specific grouping must win the exact-span tie
        # (the korean_rrn/french_nir rule).
        name="thai_id",
        detector_type="TH_ID",
        pattern=re.compile(r"(?<![\w-])[1-8]-\d{4}-\d{5}-\d{2}-\d(?![\w-])"),
        validator=_thai_id_ok,
        priority=90,
        languages=("th",),
        required=(_DIGIT_DASH,),
    ),
    RegexRule(
        # Irish PPS number: 7 digits + mod-23 check letter, optionally a
        # ninth letter (legacy W / post-2013 A or H). The bracketing check
        # letter makes the solid form safe (the DNI/NRIC pattern).
        name="irish_pps",
        detector_type="IE_PPS",
        pattern=re.compile(r"(?<![\w-])\d{7}[A-W][AHW]?(?![\w-])"),
        validator=_pps_ok,
        languages=("en",),
    ),
    RegexRule(
        # Mexican CURP: 18-char grammar (initials with a vowel slot, date,
        # sex, state, internal consonants, homoclave, check digit). The
        # grammar + state gate + real-date check + mod-10 checksum make
        # the solid form safe (the codice_fiscale pattern).
        name="mexican_curp",
        detector_type="MX_CURP",
        pattern=re.compile(
            r"(?<![\w-])[A-Z][AEIOUX][A-Z]{2}\d{2}(?:0[1-9]|1[0-2])"
            r"(?:0[1-9]|[12]\d|3[01])[HM][A-Z]{2}[B-DF-HJ-NP-TV-Z]{3}[0-9A-Z]\d(?![\w-])"
        ),
        validator=_curp_ok,
        languages=("es",),
    ),
    RegexRule(
        # AKIA = long-term access keys; ASIA = STS temporary access keys.
        # Both are 20-char AWS access-key IDs and equally sensitive.
        name="aws_access_key_id",
        detector_type="AWS_KEY",
        pattern=re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        priority=10,
        required=(("AKIA", "ASIA"),),
        anchors=("AKIA", "ASIA"),
    ),
    RegexRule(
        name="aws_secret_key",
        detector_type="AWS_SECRET",
        pattern=re.compile(
            r"""(?ix)
            aws[_\-\. ]{0,10}(?:secret[_\-\. ]{0,10})?(?:access[_\-\. ]{0,10})?key
            [^\S\n]{0,10}[:=][^\S\n]{0,10}["']?
            (?P<value>[A-Za-z0-9/+=]{40})["']?
            """
        ),
        group=1,
        priority=10,
        required=(("aws",), (":", "=")),
        required_ci=True,
        anchors=("aws",),
        anchors_ci=True,
    ),
    RegexRule(
        name="github_token",
        detector_type="GITHUB_TOKEN",
        pattern=re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),
        priority=10,
        required=(("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),),
        anchors=("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
    ),
    RegexRule(
        name="anthropic_api_key",
        detector_type="ANTHROPIC_KEY",
        pattern=re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
        priority=5,
        required=(("sk-ant-",),),
        anchors=("sk-ant-",),
    ),
    RegexRule(
        name="openai_api_key",
        detector_type="OPENAI_KEY",
        pattern=re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
        priority=20,
        required=(("sk-",),),
        anchors=("sk-",),
    ),
    RegexRule(
        name="slack_token",
        detector_type="SLACK_TOKEN",
        pattern=re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
        priority=10,
        required=(("xoxb-", "xoxa-", "xoxp-", "xoxr-", "xoxs-"),),
        anchors=("xoxb-", "xoxa-", "xoxp-", "xoxr-", "xoxs-"),
    ),
    RegexRule(
        name="private_key",
        detector_type="PRIVATE_KEY",
        # PEM armor for RSA/EC/DSA/OPENSSH/ENCRYPTED/unqualified keys, PLUS the
        # PGP/GPG form `-----BEGIN PGP PRIVATE KEY BLOCK-----` whose ` BLOCK`
        # suffix sits between "PRIVATE KEY" and the dashes. The literal
        # "PRIVATE KEY" keeps PGP PUBLIC KEY blocks out (they say "PUBLIC KEY").
        pattern=re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY( BLOCK)?-----"
            r".*?-----END [A-Z ]*PRIVATE KEY( BLOCK)?-----",
            re.DOTALL,
        ),
        priority=1,
        required=(("-----BEGIN",),),
        anchors=("-----BEGIN ",),
    ),
    RegexRule(
        name="google_api_key",
        detector_type="GOOGLE_API_KEY",
        pattern=re.compile(r"\bAIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])"),
        priority=10,
        required=(("AIza",),),
        anchors=("AIza",),
    ),
    RegexRule(
        name="gcp_private_key_id",
        detector_type="GCP_KEY_ID",
        pattern=re.compile(r'"private_key_id"\s*:\s*"(?P<value>[a-f0-9]{40})"'),
        group=1,
        priority=10,
        required=(('"private_key_id"',),),
        anchors=('"private_key_id"',),
    ),
    RegexRule(
        name="azure_storage_key",
        detector_type="AZURE_STORAGE_KEY",
        pattern=re.compile(r"AccountKey=(?P<value>[A-Za-z0-9+/]{60,}={0,2})"),
        validator=_base64_gate,
        group=1,
        priority=10,
        required=(("AccountKey=",),),
        anchors=("AccountKey=",),
    ),
    RegexRule(
        name="jwt",
        detector_type="JWT",
        pattern=re.compile(
            r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}(?![A-Za-z0-9_-])"
        ),
        validator=_jwt_header_ok,
        priority=10,
        required=(("eyJ",),),
        anchors=("eyJ",),
    ),
    RegexRule(
        # Live-mode keys only: sk_test_/rk_test_ are deliberately excluded
        # (harmless by design, and common in docs and examples).
        name="stripe_key",
        detector_type="STRIPE_KEY",
        pattern=re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{16,}\b"),
        priority=10,
        required=(("sk_live_", "rk_live_"),),
        anchors=("sk_live_", "rk_live_"),
    ),
    RegexRule(
        name="sendgrid_key",
        detector_type="SENDGRID_KEY",
        pattern=re.compile(r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])"),
        priority=10,
        required=(("SG.",),),
        anchors=("SG.",),
    ),
    RegexRule(
        # Case-sensitive AC/SK prefix + lowercase hex; a git SHA fragment
        # would need exactly this shape to collide.
        name="twilio_id",
        detector_type="TWILIO_ID",
        pattern=re.compile(r"\b(?:AC|SK)[0-9a-f]{32}\b"),
        priority=15,
        required=(("AC", "SK"),),
        anchors=("AC", "SK"),
    ),
    RegexRule(
        name="npm_token",
        detector_type="NPM_TOKEN",
        pattern=re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
        priority=10,
        required=(("npm_",),),
        anchors=("npm_",),
    ),
    RegexRule(
        name="pypi_token",
        detector_type="PYPI_TOKEN",
        pattern=re.compile(r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{50,}(?![A-Za-z0-9_-])"),
        priority=10,
        required=(("pypi-AgEIcHlwaS5vcmc",),),
        anchors=("pypi-AgEIcHlwaS5vcmc",),
    ),
    RegexRule(
        name="github_fine_grained_pat",
        detector_type="GITHUB_TOKEN",
        pattern=re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b"),
        priority=10,
        required=(("github_pat_",),),
        anchors=("github_pat_",),
    ),
    RegexRule(
        name="huggingface_token",
        detector_type="HF_TOKEN",
        pattern=re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
        validator=_whole_entropy_gate,
        priority=15,
        required=(("hf_",),),
        anchors=("hf_",),
    ),
    RegexRule(
        # Must beat the generic sk- rule at equal start (same precedent as
        # sk-ant-): lower priority number wins the overlap tie.
        name="openrouter_key",
        detector_type="OPENROUTER_KEY",
        pattern=re.compile(r"\bsk-or-v1-[a-f0-9]{40,}\b"),
        priority=5,
        required=(("sk-or-v1-",),),
        anchors=("sk-or-v1-",),
    ),
    RegexRule(
        name="groq_key",
        detector_type="GROQ_KEY",
        pattern=re.compile(r"\bgsk_[A-Za-z0-9]{40,}\b"),
        priority=10,
        required=(("gsk_",),),
        anchors=("gsk_",),
    ),
    RegexRule(
        name="gitlab_pat",
        detector_type="GITLAB_TOKEN",
        pattern=re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
        priority=10,
        required=(("glpat-",),),
        anchors=("glpat-",),
    ),
    RegexRule(
        # The GitLab 16+ routable-token family beyond glpat- (runner, CI
        # build/job, deploy, pipeline-trigger, agent, incoming-mail). Same
        # GITLAB_TOKEN type — they all agree on mode. The legacy unprefixed
        # CI_JOB_TOKEN has no anchor and stays undetectable (documented).
        name="gitlab_token",
        detector_type="GITLAB_TOKEN",
        pattern=re.compile(
            r"\bgl(?:rt|cbt|dt|ptt|agent|imt|soat)-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"
        ),
        priority=10,
        required=(("glrt-", "glcbt-", "gldt-", "glptt-", "glagent-", "glimt-", "glsoat-"),),
        anchors=("glrt-", "glcbt-", "gldt-", "glptt-", "glagent-", "glimt-", "glsoat-"),
    ),
    RegexRule(
        # Google Cloud OAuth client secret. The service-account PEM
        # (private_key) and private_key_id are already covered; this is the
        # OAuth client secret, a distinct prefix-anchored form.
        name="google_oauth_client_secret",
        detector_type="GOOGLE_OAUTH_SECRET",
        pattern=re.compile(r"\bGOCSPX-[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_-])"),
        priority=10,
        required=(("GOCSPX-",),),
        anchors=("GOCSPX-",),
    ),
    RegexRule(
        # Sentry org (sntrys_) / user (sntryu_) auth tokens over a base64
        # payload. The legacy 64-hex Sentry token has no prefix and stays
        # out (same stance as Datadog); the public DSN key is not a secret.
        name="sentry_token",
        detector_type="SENTRY_TOKEN",
        pattern=re.compile(r"\bsntry[su]_[A-Za-z0-9+/=_-]{40,}(?![A-Za-z0-9+/=_-])"),
        priority=10,
        required=(("sntrys_", "sntryu_"),),
        anchors=("sntrys_", "sntryu_"),
    ),
    RegexRule(
        # xAI (Grok) API key. Distinct prefix; does not collide with the
        # sk- family. DeepSeek uses sk- and is covered by openai_api_key.
        name="xai_key",
        detector_type="XAI_KEY",
        pattern=re.compile(r"\bxai-[A-Za-z0-9]{40,}(?![A-Za-z0-9])"),
        priority=10,
        required=(("xai-",),),
        anchors=("xai-",),
    ),
    RegexRule(
        # Perplexity API key.
        name="perplexity_key",
        detector_type="PERPLEXITY_KEY",
        pattern=re.compile(r"\bpplx-[A-Za-z0-9]{40,}(?![A-Za-z0-9])"),
        priority=10,
        required=(("pplx-",),),
        anchors=("pplx-",),
    ),
    RegexRule(
        # HashiCorp Vault service (hvs.) / batch (hvb.) tokens.
        name="hashicorp_vault_token",
        detector_type="VAULT_TOKEN",
        pattern=re.compile(r"\bhv[sb]\.[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
        priority=10,
        required=(("hvs.", "hvb."),),
        anchors=("hvs.", "hvb."),
    ),
    RegexRule(
        # LangSmith API key (personal token lsv2_pt_ / service key lsv2_sk_).
        name="langsmith_key",
        detector_type="LANGSMITH_KEY",
        pattern=re.compile(r"\blsv2_(?:pt|sk)_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
        priority=10,
        required=(("lsv2_pt_", "lsv2_sk_"),),
        anchors=("lsv2_pt_", "lsv2_sk_"),
    ),
    RegexRule(
        # Replicate API token.
        name="replicate_token",
        detector_type="REPLICATE_TOKEN",
        pattern=re.compile(r"\br8_[A-Za-z0-9]{37,}(?![A-Za-z0-9])"),
        priority=10,
        required=(("r8_",),),
        anchors=("r8_",),
    ),
    RegexRule(
        # Pinecone API key.
        name="pinecone_key",
        detector_type="PINECONE_KEY",
        pattern=re.compile(r"\bpcsk_[A-Za-z0-9_]{20,}(?![A-Za-z0-9_])"),
        priority=10,
        required=(("pcsk_",),),
        anchors=("pcsk_",),
    ),
    RegexRule(
        name="databricks_token",
        detector_type="DATABRICKS_TOKEN",
        pattern=re.compile(r"\bdapi[0-9a-f]{32,}\b"),
        priority=10,
        required=(("dapi",),),
        anchors=("dapi",),
    ),
    RegexRule(
        name="bitbucket_app_password",
        detector_type="BITBUCKET_TOKEN",
        pattern=re.compile(r"\bATBB[A-Za-z0-9]{24,}\b"),
        priority=10,
        required=(("ATBB",),),
        anchors=("ATBB",),
    ),
    RegexRule(
        # Jira/Confluence API tokens — common in agent workflows.
        name="atlassian_api_token",
        detector_type="ATLASSIAN_TOKEN",
        pattern=re.compile(r"\bATATT[A-Za-z0-9_=-]{20,}(?![A-Za-z0-9_=-])"),
        priority=10,
        required=(("ATATT",),),
        anchors=("ATATT",),
    ),
    RegexRule(
        # tskey-api-… / tskey-auth-… / tskey-client-…: kind segment plus
        # the key body, hyphens allowed inside but never terminal.
        name="tailscale_key",
        detector_type="TAILSCALE_KEY",
        pattern=re.compile(r"\btskey-[A-Za-z0-9][A-Za-z0-9-]{14,80}[A-Za-z0-9](?![\w-])"),
        priority=10,
        required=(("tskey-",),),
        anchors=("tskey-",),
    ),
    RegexRule(
        # dop_v1_ personal / doo_v1_ OAuth / dor_v1_ refresh, 64 hex.
        name="digitalocean_token",
        detector_type="DO_TOKEN",
        pattern=re.compile(r"\bdo[por]_v1_[0-9a-f]{64}\b"),
        priority=10,
        required=(("dop_v1_", "doo_v1_", "dor_v1_"),),
        anchors=("dop_v1_", "doo_v1_", "dor_v1_"),
    ),
    RegexRule(
        # ntn_ (current) and the legacy secret_ integration tokens; the
        # legacy prefix is a common word, so the exact 43-char body does
        # the disambiguation (the Gitleaks-proven form).
        name="notion_token",
        detector_type="NOTION_TOKEN",
        pattern=re.compile(r"\b(?:ntn_[A-Za-z0-9]{40,60}|secret_[A-Za-z0-9]{43})\b"),
        priority=10,
        required=(("ntn_", "secret_"),),
        anchors=("ntn_", "secret_"),
    ),
    RegexRule(
        name="linear_api_key",
        detector_type="LINEAR_KEY",
        pattern=re.compile(r"\blin_(?:api|oauth)_[A-Za-z0-9]{32,}\b"),
        priority=10,
        required=(("lin_api_", "lin_oauth_"),),
        anchors=("lin_api_", "lin_oauth_"),
    ),
    RegexRule(
        # sbp_ personal access tokens and sb_secret_ API keys; the
        # sb_publishable_ form is deliberately unmatched (not a secret).
        name="supabase_key",
        detector_type="SUPABASE_KEY",
        pattern=re.compile(r"\b(?:sbp_[0-9a-f]{40}|sb_secret_[A-Za-z0-9_]{20,})\b"),
        priority=10,
        required=(("sbp_", "sb_secret_"),),
        anchors=("sbp_", "sb_secret_"),
    ),
    RegexRule(
        name="planetscale_token",
        detector_type="PLANETSCALE_TOKEN",
        pattern=re.compile(r"\bpscale_(?:tkn|oauth|pw)_[A-Za-z0-9=._-]{30,64}(?![\w=.-])"),
        priority=10,
        required=(("pscale_tkn_", "pscale_oauth_", "pscale_pw_"),),
        anchors=("pscale_tkn_", "pscale_oauth_", "pscale_pw_"),
    ),
    RegexRule(
        # dp.pt. personal / dp.st. service / dp.ct. CLI / dp.sa. service
        # account.
        name="doppler_token",
        detector_type="DOPPLER_TOKEN",
        pattern=re.compile(r"\bdp\.(?:pt|st|ct|sa)\.[A-Za-z0-9]{40,}\b"),
        priority=10,
        required=(("dp.pt.", "dp.st.", "dp.ct.", "dp.sa."),),
        anchors=("dp.pt.", "dp.st.", "dp.ct.", "dp.sa."),
    ),
    RegexRule(
        name="postman_key",
        detector_type="POSTMAN_KEY",
        pattern=re.compile(r"\bPMAK-[0-9a-f]{24}-[0-9a-f]{34}\b"),
        priority=10,
        required=(("PMAK-",),),
        anchors=("PMAK-",),
    ),
    RegexRule(
        # pat + 14-char id + dot + 64-hex secret half. "pat" is a common
        # substring, so the anchor mostly serves the find-then-match scan;
        # the dot-hex tail is what keeps prose out.
        name="airtable_pat",
        detector_type="AIRTABLE_PAT",
        pattern=re.compile(r"\bpat[A-Za-z0-9]{14}\.[0-9a-f]{64}\b"),
        priority=10,
        required=(("pat",),),
        anchors=("pat",),
    ),
    RegexRule(
        # shpat_ admin / shpss_ shared secret / shpca_ custom app /
        # shppa_ private app, all 32 hex.
        name="shopify_token",
        detector_type="SHOPIFY_TOKEN",
        pattern=re.compile(r"\bshp(?:at|ss|ca|pa)_[0-9a-fA-F]{32}\b"),
        priority=10,
        required=(("shpat_", "shpss_", "shpca_", "shppa_"),),
        anchors=("shpat_", "shpss_", "shpca_", "shppa_"),
    ),
    RegexRule(
        # New Relic user API key: NRAK- + 27 upper-alphanumeric.
        name="new_relic_key",
        detector_type="NEW_RELIC_KEY",
        pattern=re.compile(r"\bNRAK-[A-Z0-9]{27}\b"),
        priority=10,
        required=(("NRAK-",),),
        anchors=("NRAK-",),
    ),
    RegexRule(
        # Grafana service-account token: glsa_ + 32 alnum + _ + 8 hex checksum.
        name="grafana_service_account",
        detector_type="GRAFANA_TOKEN",
        pattern=re.compile(r"\bglsa_[A-Za-z0-9]{32}_[0-9a-f]{8}\b"),
        priority=10,
        required=(("glsa_",),),
        anchors=("glsa_",),
    ),
    RegexRule(
        # Jina AI key: jina_ + 40+ alphanumeric.
        name="jina_key",
        detector_type="JINA_KEY",
        pattern=re.compile(r"\bjina_[A-Za-z0-9]{40,}\b"),
        priority=10,
        required=(("jina_",),),
        anchors=("jina_",),
    ),
    RegexRule(
        # Telegram bot token: <8-10 digit bot id>:<35-char base64url secret>.
        # The colon + fixed-width secret half is what disambiguates it from a
        # bare ratio/timestamp; no single literal anchor exists.
        name="telegram_bot_token",
        detector_type="TELEGRAM_BOT_TOKEN",
        pattern=re.compile(r"(?<![\w:])\d{8,10}:[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])"),
        priority=10,
    ),
    RegexRule(
        # Agent-stack round (r10): the keys that sit in agentic tools' env
        # files and MCP configs. Tavily ships tvly- with optional dev/prod
        # environment segments.
        name="tavily_key",
        detector_type="TAVILY_KEY",
        pattern=re.compile(r"\btvly-(?:dev-|prod-)?[A-Za-z0-9]{20,}\b"),
        priority=10,
        required=(("tvly-",),),
        anchors=("tvly-",),
    ),
    RegexRule(
        # Firecrawl: 32 lowercase hex behind a SHORT prefix — the exact
        # fixed-length hex body keeps "fc-..." prose ids out, and the
        # dash-excluding lookarounds keep it out of resource/host names
        # like grid-fc-<hash> (caught by the fp gate).
        name="firecrawl_key",
        detector_type="FIRECRAWL_KEY",
        pattern=re.compile(r"(?<![\w-])fc-[0-9a-f]{32}(?![\w-])"),
        priority=10,
        required=(("fc-",),),
        anchors=("fc-",),
    ),
    RegexRule(
        # NVIDIA NIM / API Catalog personal keys.
        name="nvidia_key",
        detector_type="NVIDIA_KEY",
        pattern=re.compile(r"\bnvapi-[A-Za-z0-9_-]{40,}(?![A-Za-z0-9_-])"),
        priority=10,
        required=(("nvapi-",),),
        anchors=("nvapi-",),
    ),
    RegexRule(
        name="cerebras_key",
        detector_type="CEREBRAS_KEY",
        pattern=re.compile(r"\bcsk-[A-Za-z0-9]{30,}\b"),
        priority=10,
        required=(("csk-",),),
        anchors=("csk-",),
    ),
    RegexRule(
        # Langfuse SECRET keys (sk-lf- + UUID). Priority 5 beats the
        # generic sk- OpenAI rule on the shared span (the sk-ant-/sk-or-
        # pattern); the paired pk-lf- public key is deliberately unmatched
        # (the sb_publishable_ stance).
        name="langfuse_secret_key",
        detector_type="LANGFUSE_KEY",
        pattern=re.compile(
            r"\bsk-lf-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
        ),
        priority=5,
        required=(("sk-lf-",),),
        anchors=("sk-lf-",),
    ),
    RegexRule(
        # Figma personal access tokens.
        name="figma_pat",
        detector_type="FIGMA_PAT",
        pattern=re.compile(r"\bfigd_[A-Za-z0-9_-]{30,}(?![A-Za-z0-9_-])"),
        priority=10,
        required=(("figd_",),),
        anchors=("figd_",),
    ),
    RegexRule(
        name="generic_secret",
        detector_type="SECRET",
        pattern=re.compile(
            r"""(?ix)
            (?:api[_\-]?key|auth[_\-]?token|access[_\-]?token|secret|password|passwd
               |credential[s]?|token)
            [^\S\n]{0,10}[:=][^\S\n]{0,10}["']?
            (?P<value>[A-Za-z0-9/+=_\-\.]{16,})["']?
            """
        ),
        group=1,
        priority=50,
        # Every keyword alternative contains one of these roots ("passw"
        # covers password AND passwd), and the grammar always requires a
        # colon or equals sign.
        required=(
            ("api", "auth", "access", "secret", "passw", "credential", "token"),
            (":", "="),
        ),
        required_ci=True,
        # Every keyword alternative also STARTS with one of these roots.
        anchors=("api", "auth", "access", "secret", "passw", "credential", "token"),
        anchors_ci=True,
    ),
)

# generic_secret additionally requires high entropy so that placeholder-ish
# values ("changeme", "your-password-here") don't get redacted.
_GENERIC_VALIDATORS: dict[str, Callable[[re.Match[str]], bool]] = {
    "generic_secret": _entropy_gate,
    "aws_secret_key": _entropy_gate,
}


class RegexDetector:
    def __init__(self, rule: RegexRule) -> None:
        self.name = rule.name
        self._rule = rule
        self._validator = rule.validator or _GENERIC_VALIDATORS.get(rule.name)

    def detect_prepared(self, prepared: PreparedText) -> Iterable[Detection]:
        """detect(), skipped or narrowed via the rule's declared literals —
        identical output by construction (required literals are necessary
        conditions; anchors are guaranteed match prefixes)."""
        rule = self._rule
        if rule.required:
            haystack = prepared.lower if rule.required_ci else prepared.text
            if not all(any(lit in haystack for lit in group) for group in rule.required):
                return ()
        if rule.anchors:
            return self._detections_from(self._anchored_matches(prepared))
        return self.detect(prepared.text)

    def _anchored_matches(self, prepared: PreparedText) -> Iterable["re.Match[str]"]:
        """finditer, restricted to positions where a declared anchor occurs.

        Reproduces finditer's leftmost-non-overlapping semantics exactly
        under the anchor premise: positions are visited in ascending order
        and anything before the previous match's end is skipped (a vetoed
        validator does not un-consume a span — the validator runs later, as
        in the plain path).
        """
        rule = self._rule
        text = prepared.text
        if rule.anchors_ci:
            haystack = prepared.lower
            if len(haystack) != len(text):
                # str.lower changed the string's length ('İ' → 2 chars), so
                # lowered offsets no longer map 1:1; fall back to the full
                # scan rather than guess.
                yield from rule.pattern.finditer(text)
                return
        else:
            haystack = text
        positions: set[int] = set()
        for anchor in rule.anchors:
            idx = haystack.find(anchor)
            while idx != -1:
                positions.add(idx)
                idx = haystack.find(anchor, idx + 1)
        last_end = -1
        for pos in sorted(positions):
            if pos < last_end:
                continue
            match = rule.pattern.match(text, pos)
            if match is not None:
                yield match
                last_end = match.end()

    def detect(self, text: str) -> Iterable[Detection]:
        return self._detections_from(self._rule.pattern.finditer(text))

    def _detections_from(self, matches: Iterable["re.Match[str]"]) -> Iterable[Detection]:
        rule = self._rule
        for match in matches:
            if self._validator is not None and not self._validator(match):
                continue
            if rule.group and match.group("value") is not None:
                start, end = match.span("value")
                value = match.group("value")
            else:
                start, end = match.span(0)
                value = match.group(0)
            yield Detection(
                start=start,
                end=end,
                detector_type=rule.detector_type,
                value=value,
                priority=rule.priority,
            )
