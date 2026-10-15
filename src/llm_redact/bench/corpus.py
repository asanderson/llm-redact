"""Deterministic synthetic corpus for the detection benchmark.

Positive samples are generated to match each rule's grammar and validators
and are embedded in realistic prose/JSON/code contexts with labeled spans.
Negative decoys are shapes that legitimately occur in code and logs and must
never fire. Generated fresh from a seed at every run.
"""

import base64
import json
import random
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from llm_redact.detection.wallet_checksums import (
    base58check_encode,
    bech32_encode,
    eth_checksum,
)

_ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_HEX = "0123456789abcdef"
_B64 = _ALNUM + "+/"


@dataclass(frozen=True)
class LabeledSpan:
    start: int
    end: int
    detector_type: str


@dataclass(frozen=True)
class Sample:
    text: str
    spans: tuple[LabeledSpan, ...]


def _pick(rng: random.Random, alphabet: str, n: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _luhn_complete(rng: random.Random) -> str:
    digits = [rng.randrange(10) for _ in range(15)]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:  # positions counted with the check digit appended
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    digits.append((10 - checksum % 10) % 10)
    return "".join(str(d) for d in digits)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def _jwt(rng: random.Random) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256"}).encode()).decode().rstrip("=")
    payload = (
        base64.urlsafe_b64encode(json.dumps({"sub": _pick(rng, "0123456789", 10)}).encode())
        .decode()
        .rstrip("=")
    )
    return f"{header}.{payload}.{_pick(rng, _ALNUM + '-_', 32)}"


def _phone(rng: random.Random) -> str:
    a = rng.randrange(200, 999)
    b = rng.randrange(200, 999)
    c = rng.randrange(1000, 9999)
    return rng.choice(
        [
            f"+1 {a} {b} {c}",
            f"+44 20 {rng.randrange(1000, 9999)} {c}",
            f"+{rng.randrange(2, 9)}{a}{b}{c}",
            f"({a}) {b}-{c}",
            f"{a}-{b}-{c}",
            f"{a}.{b}.{c}",
        ]
    )


def _ssn(rng: random.Random) -> str:
    area = rng.choice([n for n in range(1, 900) if n != 666])
    return f"{area:03d}-{rng.randrange(1, 100):02d}-{rng.randrange(1, 10000):04d}"


# Country -> (IBAN length, BBAN starts with 4 letters), transcribed from the
# registry independently of the rule's length table; the checksum below is
# computed, so every generated IBAN is genuinely valid. BE and PL end on a
# full group of four in the paper format, MT is the longest here.
_IBAN_COUNTRIES = {
    "BE": (16, False),
    "DE": (22, False),
    "FR": (27, False),
    "GB": (22, True),
    "MT": (31, True),
    "NL": (18, True),
    "PL": (28, False),
}
# The ISO 13616 paper format's separator, and the no-break spaces the rule
# also reads as one.
_IBAN_PAPER_SEPARATORS = (" ", " ", " ", "\u00a0", "\u202f")


def _iban_solid(rng: random.Random, country: str | None = None) -> str:
    """A valid IBAN in the solid electronic form."""
    country = country or rng.choice(sorted(_IBAN_COUNTRIES))
    length, alpha_prefix = _IBAN_COUNTRIES[country]
    bban_len = length - 4
    if alpha_prefix:
        bban = _pick(rng, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 4) + _pick(rng, "0123456789", bban_len - 4)
    else:
        bban = _pick(rng, "0123456789", bban_len)
    return _with_check_digits(country, bban)


def _with_check_digits(country: str, bban: str) -> str:
    """``country`` + the mod-97 check digits that make it valid + ``bban``."""
    numeric = "".join(str(int(ch, 36)) for ch in bban + country + "00")
    check = 98 - int(numeric) % 97
    return f"{country}{check:02d}{bban}"


def _paper(iban: str, separator: str = " ") -> str:
    """``iban`` in the paper format: groups of four, the last one shorter
    or not."""
    return separator.join(iban[i : i + 4] for i in range(0, len(iban), 4))


def _iban(rng: random.Random) -> str:
    """A valid IBAN, half in the solid form, half in the paper format."""
    solid = _iban_solid(rng)
    if rng.random() < 0.5:
        return solid
    return _paper(solid, rng.choice(_IBAN_PAPER_SEPARATORS))


def _canadian_sin(rng: random.Random) -> str:
    """Luhn-valid, assigned-area SIN in either separated-triads form."""
    digits = [rng.choice([1, 2, 3, 4, 5, 6, 7, 9])]
    digits += [rng.randrange(10) for _ in range(7)]
    # Check digit: with nine digits the last one sits at reversed index 0
    # (never doubled), so it directly completes the checksum.
    partial = 0
    for i, d in enumerate(reversed(digits)):
        d = d * 2 if i % 2 == 0 else d  # reversed positions shift by one
        partial += d - 9 if d > 9 else d
    digits.append((10 - partial % 10) % 10)
    sep = rng.choice([" ", "-"])
    joined = "".join(str(d) for d in digits)
    return f"{joined[:3]}{sep}{joined[3:6]}{sep}{joined[6:]}"


_NINO_FIRST = "ABCEGHJKLMNOPRSTWXYZ"  # no D F I Q U V
_NINO_SECOND = "ABCEGHJKLMNPRSTWXYZ"  # additionally no O
_NINO_BAD_PAIRS = {"BG", "GB", "NK", "KN", "TN", "NT", "ZZ"}


def _uk_nino(rng: random.Random) -> str:
    while True:
        prefix = rng.choice(_NINO_FIRST) + rng.choice(_NINO_SECOND)
        if prefix not in _NINO_BAD_PAIRS:
            break
    digits = f"{rng.randrange(1_000_000):06d}"
    suffix = rng.choice("ABCD")
    if rng.random() < 0.5:
        return f"{prefix}{digits}{suffix}"
    return f"{prefix} {digits[:2]} {digits[2:4]} {digits[4:]} {suffix}"


def _verhoeff_d(j: int, k: int) -> int:
    """D5 dihedral multiplication, derived from the group law rather than
    copying the rule's lookup table — a transcription typo in either place
    fails the recall gate instead of passing silently."""
    if j < 5 and k < 5:
        return (j + k) % 5
    if j < 5 <= k:
        return (j + k) % 5 + 5
    if k < 5 <= j:
        return (j - k) % 5 + 5
    return (j - k) % 5


_VERHOEFF_BASE_P = (1, 5, 7, 6, 2, 8, 3, 0, 9, 4)


def _aadhaar(rng: random.Random) -> str:
    """Verhoeff-valid Aadhaar in the displayed 4-4-4 grouping."""
    digits = [rng.randrange(2, 10)] + [rng.randrange(10) for _ in range(10)]
    perms = [tuple(range(10))]
    for _ in range(7):
        perms.append(tuple(_VERHOEFF_BASE_P[x] for x in perms[-1]))
    check = 0
    for check in range(10):
        c = 0
        for i, d in enumerate(reversed([*digits, check])):
            c = _verhoeff_d(c, perms[i % 8][d])
        if c == 0:
            break
    joined = "".join(str(d) for d in digits) + str(check)
    sep = rng.choice([" ", "-"])
    return f"{joined[:4]}{sep}{joined[4:8]}{sep}{joined[8:]}"


def _australian_tfn(rng: random.Random) -> str:
    """ATO-checksum-valid TFN. Always leading 8: the shape is identical to
    canadian_sin, which sits earlier in the rule list and wins exact-span
    ties — a leading 8 is rejected by _sin_ok, keeping the two disjoint."""
    while True:
        digits = [8] + [rng.randrange(10) for _ in range(7)]
        partial = sum(w * d for w, d in zip((1, 4, 3, 7, 5, 8, 6, 9), digits, strict=True))
        last = (-partial) * 10 % 11  # 10 is 10⁻¹ mod 11; digit 9's weight is 10
        if last < 10:
            break
    digits.append(last)
    sep = rng.choice([" ", "-"])
    joined = "".join(str(d) for d in digits)
    return f"{joined[:3]}{sep}{joined[3:6]}{sep}{joined[6:]}"


_DNI_CONTROL = "TRWAGMYFPDXBNJZSQVHLCKE"


def _spanish_dni(rng: random.Random) -> str:
    if rng.random() < 0.3:  # NIE: X/Y/Z stand for 0/1/2 in the computation
        prefix = rng.choice("XYZ")
        tail = rng.randrange(10_000_000)
        number = int(str("XYZ".index(prefix)) + f"{tail:07d}")
        body = f"{prefix}{tail:07d}"
    else:
        number = rng.randrange(10_000_000, 100_000_000)
        body = f"{number:08d}"
    sep = rng.choice(["", "-"])
    return f"{body}{sep}{_DNI_CONTROL[number % 23]}"


def _french_nir(rng: random.Random) -> str:
    """Key-valid NIR in the spaced display form; sometimes Corsican."""
    roll = rng.random()
    dept = "2A" if roll < 0.1 else ("2B" if roll < 0.2 else f"{rng.randrange(1, 96):02d}")
    body = (
        rng.choice("12")
        + f"{rng.randrange(100):02d}"
        + f"{rng.randrange(1, 13):02d}"
        + dept
        + f"{rng.randrange(1, 1000):03d}"
        + f"{rng.randrange(1, 1000):03d}"
    )
    if "A" in body:
        number = int(body.replace("A", "0")) - 1_000_000
    elif "B" in body:
        number = int(body.replace("B", "0")) - 2_000_000
    else:
        number = int(body)
    key = 97 - number % 97
    return f"{body[0]} {body[1:3]} {body[3:5]} {body[5:7]} {body[7:10]} {body[10:13]} {key:02d}"


def _swiss_ahv(rng: random.Random) -> str:
    """AHV in the dotted display form; EAN-13 check digit transcribed
    independently of the rule's validator."""
    digits = [7, 5, 6] + [rng.randrange(10) for _ in range(9)]
    weighted = 0
    for index, digit in enumerate(digits):
        weighted += digit if index % 2 == 0 else digit * 3
    check = (10 - weighted % 10) % 10
    joined = "".join(str(d) for d in digits) + str(check)
    return f"{joined[:3]}.{joined[3:7]}.{joined[7:11]}.{joined[11:]}"


def _swedish_personnummer(rng: random.Random) -> str:
    """Personnummer with the Swedish Luhn check; sometimes a coordination
    number (day+60), sometimes the + century separator."""
    month = rng.randrange(1, 13)
    day = rng.randrange(1, 29) + (60 if rng.random() < 0.2 else 0)
    body = [
        rng.randrange(10),
        rng.randrange(10),
        month // 10,
        month % 10,
        day // 10,
        day % 10,
        rng.randrange(10),
        rng.randrange(10),
        rng.randrange(10),
    ]
    total = 0
    for index, digit in enumerate(body):
        product = digit * 2 if index % 2 == 0 else digit
        total += product // 10 + product % 10
    check = (10 - total % 10) % 10
    joined = "".join(str(d) for d in body) + str(check)
    separator = "+" if rng.random() < 0.1 else "-"
    return f"{joined[:6]}{separator}{joined[6:]}"


def _belgian_nn(rng: random.Random) -> str:
    """Rijksregisternummer in the dotted display form (YY.MM.DD-NNN.CC).
    The two check digits are 97 - (body mod 97), the body optionally
    '2'-prefixed for post-2000 births. Pure modular arithmetic, computed
    here independently of the rule's validator. 93.05.18-223.61 pins it."""
    yy = rng.randrange(100)
    mm = rng.randrange(1, 13)
    dd = rng.randrange(1, 32)
    nnn = rng.randrange(1000)
    body = f"{yy:02d}{mm:02d}{dd:02d}{nnn:03d}"
    prefix = "2" if rng.random() < 0.5 else ""
    check = 97 - int(prefix + body) % 97
    return f"{yy:02d}.{mm:02d}.{dd:02d}-{nnn:03d}.{check:02d}"


# Finnish HETU check-character alphabet, built by FILTERING rather than
# copying regex_rules' literal (G/I/O/Q are the omitted letters) so a typo on
# either side fails the recall gate. Index = 9-digit number mod 31; the century
# signs are transcribed the same way. 131052-308T pins both tables.
_HETU_ALPHABET = "0123456789" + "".join(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXY" if c not in "GIOQ")
_HETU_SIGNS = "+-YXWVUABCDEF"


def _reads_as_personnummer(value: str) -> bool:
    """Whether a DDMMYY±NNNC value is also a valid Swedish personnummer
    (YYMMDD±NNNC: month 1-12, day 1-31 or 61-91, Luhn over the ten digits),
    which the swedish_personnummer rule, earlier in the list, would claim."""
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 10 or value[6] not in "+-":
        return False
    month, day = digits[2] * 10 + digits[3], digits[4] * 10 + digits[5]
    if not (1 <= month <= 12 and (1 <= day <= 31 or 61 <= day <= 91)):
        return False
    total = sum(sum(divmod(d * (2 - i % 2), 10)) for i, d in enumerate(digits))
    return total % 10 == 0


def _finnish_hetu(rng: random.Random) -> str:
    """Henkilötunnus: DDMMYY + century sign + 3-digit individual number + a
    mod-31 check character over the nine digits. A value that also reads as
    a Swedish personnummer is drawn again (the TFN/SIN stance)."""
    while True:
        day = rng.randrange(1, 29)
        month = rng.randrange(1, 13)
        year = rng.randrange(100)
        ddmmyy = f"{day:02d}{month:02d}{year:02d}"
        sign = rng.choice(_HETU_SIGNS)
        zzz = f"{rng.randrange(1000):03d}"
        check = _HETU_ALPHABET[int(ddmmyy + zzz) % 31]
        value = f"{ddmmyy}{sign}{zzz}{check}"
        if not _reads_as_personnummer(value):
            return value


def _nhs_number(rng: random.Random) -> str:
    """NHS Number in the 3-3-4 spaced display form; mod-11 check digit
    (10 is invalid and re-rolled). Independent arithmetic transcription."""
    while True:
        digits = [rng.randrange(10) for _ in range(9)]
        check = 11 - sum(d * (10 - i) for i, d in enumerate(digits)) % 11
        if check == 11:
            check = 0
        if check != 10:
            break
    joined = "".join(str(d) for d in digits) + str(check)
    return f"{joined[:3]} {joined[3:6]} {joined[6:]}"


# INDEPENDENT transcription of the fødselsnummer control-digit weights (the
# rule has its own copy): a typo on either side fails the recall gate.
_NO_FNR_K1 = (3, 7, 6, 1, 8, 9, 4, 5, 2)
_NO_FNR_K2 = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)


def _norwegian_fnr(rng: random.Random) -> str:
    """Norwegian fødselsnummer, DDMMYY-NNNNN spaced/hyphenated display form;
    two mod-11 control digits (either landing on 10 is re-rolled)."""
    while True:
        day, month = rng.randrange(1, 29), rng.randrange(1, 13)
        body = [
            int(c) for c in f"{day:02d}{month:02d}{rng.randrange(100):02d}{rng.randrange(1000):03d}"
        ]
        k1 = 11 - sum(body[i] * _NO_FNR_K1[i] for i in range(9)) % 11
        if k1 == 11:
            k1 = 0
        if k1 == 10:
            continue
        with_k1 = [*body, k1]
        k2 = 11 - sum(with_k1[i] * _NO_FNR_K2[i] for i in range(10)) % 11
        if k2 == 11:
            k2 = 0
        if k2 == 10:
            continue
        joined = "".join(str(d) for d in body) + str(k1) + str(k2)
        sep = rng.choice([" ", "-"])
        return f"{joined[:6]}{sep}{joined[6:]}"


# INDEPENDENT transcription of the Korean RRN weights (the rule has its copy).
_KR_RRN_W = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)


def _korean_rrn(rng: random.Random) -> str:
    """Korean RRN YYMMDD-SBBBBNC, hyphenated 6-7 display form; mod-11 check
    digit (11 - sum mod 11) mod 10, gender/century digit 1-8."""
    month, day = rng.randrange(1, 13), rng.randrange(1, 29)
    body = [int(c) for c in f"{rng.randrange(100):02d}{month:02d}{day:02d}"]
    body.append(rng.randrange(1, 9))  # gender/century digit
    body += [rng.randrange(10) for _ in range(5)]  # 12 digits total
    check = (11 - sum(body[i] * _KR_RRN_W[i] for i in range(12)) % 11) % 10
    joined = "".join(str(d) for d in body) + str(check)
    return f"{joined[:6]}-{joined[6:]}"


# INDEPENDENT transcription of the GB 11643 MOD 11-2 weight table (the
# rule derives it as 2^(17-i) mod 11 — a derivation bug there fails here).
_CN_ID_W = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_CN_ID_CHECK = "10X98765432"
_CN_ID_PROVINCES = (11, 13, 21, 31, 33, 42, 44, 51, 61, 65)


def _chinese_resident_id(rng: random.Random) -> str:
    """CN Resident ID: region + YYYYMMDD + seq + MOD 11-2 check char, solid
    18-char display form."""
    region = f"{rng.choice(_CN_ID_PROVINCES)}{rng.randrange(100):02d}{rng.randrange(100):02d}"
    date = f"{rng.randrange(1940, 2020)}{rng.randrange(1, 13):02d}{rng.randrange(1, 29):02d}"
    body = f"{region}{date}{rng.randrange(1000):03d}"
    total = sum(int(body[i]) * _CN_ID_W[i] for i in range(17))
    return body + _CN_ID_CHECK[total % 11]


# INDEPENDENT transcription of the Singapore NRIC checksum tables.
_SG_NRIC_W = (2, 7, 6, 5, 4, 3, 2)
_SG_NRIC_ST = "JZIHGFEDCBA"
_SG_NRIC_FG = "XWUTRQPNMLK"


def _singapore_nric(rng: random.Random) -> str:
    """Singapore NRIC/FIN [STFG] + 7 digits + prefix-specific checksum letter."""
    prefix = rng.choice("STFG")
    digits = [rng.randrange(10) for _ in range(7)]
    total = sum(digits[i] * _SG_NRIC_W[i] for i in range(7))
    if prefix in "TG":
        total += 4
    table = _SG_NRIC_ST if prefix in "ST" else _SG_NRIC_FG
    return prefix + "".join(str(d) for d in digits) + table[total % 11]


# INDEPENDENT transcription: the ordinance's right-to-left weights (m+1
# for m <= 6, m-5 above) folded into a left-to-right tuple.
_JP_MY_NUMBER_W = (6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)


def _japanese_my_number(rng: random.Random) -> str:
    """Japan My Number, 4-4-4 grouped. Lead 0/1 keeps generated values
    outside aadhaar's lead-2-9 grammar — a value valid under BOTH
    checksums is claimed by aadhaar (registration order) and would fail
    this rule's recall row."""
    body = [rng.randrange(2)] + [rng.randrange(10) for _ in range(10)]
    total = sum(d * w for d, w in zip(body, _JP_MY_NUMBER_W, strict=True))
    remainder = total % 11
    check = 0 if remainder <= 1 else 11 - remainder
    digits = "".join(str(d) for d in body) + str(check)
    sep = rng.choice(" -")
    return f"{digits[:4]}{sep}{digits[4:8]}{sep}{digits[8:]}"


# INDEPENDENT transcription of the descending 13..2 weights.
_TH_ID_W = (13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2)


def _thai_id(rng: random.Random) -> str:
    """Thai Citizen ID in the dashed 1-4-5-2-1 display form."""
    body = [rng.randrange(1, 9)] + [rng.randrange(10) for _ in range(11)]
    total = sum(d * w for d, w in zip(body, _TH_ID_W, strict=True))
    s = "".join(str(d) for d in body) + str((11 - total % 11) % 10)
    return f"{s[0]}-{s[1:5]}-{s[5:10]}-{s[10:12]}-{s[12]}"


# INDEPENDENT transcription of the mod-23 alphabet and 8..2 weights.
_IE_PPS_ALPHABET = "WABCDEFGHIJKLMNOPQRSTUV"
_IE_PPS_W = (8, 7, 6, 5, 4, 3, 2)


def _irish_pps(rng: random.Random) -> str:
    """Irish PPSN: 7 digits + check letter, sometimes a second letter
    (legacy W contributes 0; post-2013 A/H contributes value x 9)."""
    digits = [rng.randrange(10) for _ in range(7)]
    total = sum(d * w for d, w in zip(digits, _IE_PPS_W, strict=True))
    second = rng.choice(("", "W", "A", "H"))
    if second and second != "W":
        total += (ord(second) - ord("A") + 1) * 9
    return "".join(str(d) for d in digits) + _IE_PPS_ALPHABET[total % 23] + second


# INDEPENDENT transcription of RENAPO's charset (Ñ at value 24) and the
# descending 18..2 weights.
_MX_CURP_CHARSET = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
_MX_CURP_W = tuple(range(18, 1, -1))
_MX_CURP_STATES = (
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
)
_MX_CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ"


def _mexican_curp(rng: random.Random) -> str:
    """Mexican CURP: initials + date + sex + state + consonants +
    homoclave + check digit. Homoclave digit <=> 1900s birth year, letter
    <=> 2000s (the rule the validator uses to pick the century)."""
    upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    initials = rng.choice(upper) + rng.choice("AEIOUX") + rng.choice(upper) + rng.choice(upper)
    if rng.randrange(2):
        year, homoclave = rng.randrange(1940, 2000), str(rng.randrange(10))
    else:
        year, homoclave = rng.randrange(2000, 2016), rng.choice(upper)
    date = f"{year % 100:02d}{rng.randrange(1, 13):02d}{rng.randrange(1, 29):02d}"
    middle = (
        rng.choice("HM")
        + rng.choice(_MX_CURP_STATES)
        + "".join(rng.choice(_MX_CONSONANTS) for _ in range(3))
    )
    first17 = initials + date + middle + homoclave
    total = sum(_MX_CURP_CHARSET.index(c) * w for c, w in zip(first17, _MX_CURP_W, strict=True))
    return first17 + str((10 - total % 10) % 10)


def _brazilian_cpf(rng: random.Random) -> str:
    """CPF in the dotted display form with both weighted mod-11 check
    digits (independent transcription of the arithmetic — a mistake on
    either side fails the recall gate)."""
    while True:
        digits = [rng.randrange(10) for _ in range(9)]
        if len(set(digits)) > 1:
            break
    for n in (9, 10):
        total = sum(d * w for d, w in zip(digits[:n], range(n + 1, 1, -1), strict=True))
        remainder = total % 11
        digits.append(0 if remainder < 2 else 11 - remainder)
    joined = "".join(str(d) for d in digits)
    return f"{joined[:3]}.{joined[3:6]}.{joined[6:9]}-{joined[9:]}"


# INDEPENDENT transcription of the codice-fiscale odd-position table (the
# rule has its own copy in regex_rules.py): a typo in either fails recall.
_CF_ODD_POSITION = {
    "0": 1,
    "1": 0,
    "2": 5,
    "3": 7,
    "4": 9,
    "5": 13,
    "6": 15,
    "7": 17,
    "8": 19,
    "9": 21,
    "A": 1,
    "B": 0,
    "C": 5,
    "D": 7,
    "E": 9,
    "F": 13,
    "G": 15,
    "H": 17,
    "I": 19,
    "J": 21,
    "K": 2,
    "L": 4,
    "M": 18,
    "N": 20,
    "O": 11,
    "P": 3,
    "Q": 6,
    "R": 8,
    "S": 12,
    "T": 14,
    "U": 16,
    "V": 10,
    "W": 22,
    "X": 25,
    "Y": 24,
    "Z": 23,
}


def _italian_codice_fiscale(rng: random.Random) -> str:
    """Codice fiscale with the mod-26 check letter; day slot covers both
    the male (01-31) and female (+40) encodings."""
    letters = "".join(rng.choice("BCDFGHJKLMNPQRSTVWXYZAEIOU") for _ in range(6))
    year = f"{rng.randrange(100):02d}"
    month = rng.choice("ABCDEHLMPRST")
    day = rng.randrange(1, 32) + (40 if rng.random() < 0.5 else 0)
    place = rng.choice("ABDEFGHILMZ") + f"{rng.randrange(1000):03d}"
    body = f"{letters}{year}{month}{day:02d}{place}"
    total = 0
    for i, ch in enumerate(body):
        if i % 2 == 0:
            total += _CF_ODD_POSITION[ch]
        elif ch.isdigit():
            total += int(ch)
        else:
            total += ord(ch) - ord("A")
    return body + chr(total % 26 + ord("A"))


def _german_steuer_id(rng: random.Random) -> str:
    """Steuer-ID in the spaced display form: nine distinct digits plus one
    doubled, nonzero lead, ISO 7064 MOD 11,10 check digit."""
    while True:
        pool = list(range(10))
        rng.shuffle(pool)
        first10 = pool[:9] + [rng.choice(pool[:9])]
        rng.shuffle(first10)
        if first10[0] != 0:
            break
    product = 10
    for d in first10:
        s = (d + product) % 10
        if s == 0:
            s = 10
        product = (2 * s) % 11
    check = 11 - product
    if check == 10:
        check = 0
    joined = "".join(str(d) for d in first10) + str(check)
    return f"{joined[:2]} {joined[2:5]} {joined[5:8]} {joined[8:]}"


def _corrupt_iban(rng: random.Random, country: str | None = None) -> str:
    """A valid-shaped solid IBAN whose checksum is off by one digit — must
    not fire."""
    valid = _iban_solid(rng, country)
    last = str((int(valid[-1]) + 1) % 10)
    return valid[:-1] + last


def _ipv6(rng: random.Random) -> str:
    # Shapes that pass all validator gates: hex letters guaranteed by the
    # 2001:db8 documentation prefix, never eight two-char groups. One in
    # five ends in "::", which the punctuation contexts follow with "." or
    # ":" (it must still be found whole).
    roll = rng.random()
    if roll < 0.4:
        tail = ":".join(f"{rng.randrange(16**4):x}" for _ in range(6))
        return f"2001:db8:{tail}"
    if roll < 0.8:
        return f"2001:db8::{rng.randrange(16**4):x}:{rng.randrange(1, 16**4):x}"
    return f"2001:db8:{rng.randrange(1, 16**4):x}::"


# Armor labels the private_key rule must catch. PGP carries a ` BLOCK` suffix
# after "PRIVATE KEY"; the others do not. All are PRIVATE (never PUBLIC) — a
# PGP PUBLIC KEY BLOCK must stay unmatched, which the fp corpus pins.
_PEM_LABELS = (
    "PRIVATE KEY",
    "RSA PRIVATE KEY",
    "EC PRIVATE KEY",
    "OPENSSH PRIVATE KEY",
    "ENCRYPTED PRIVATE KEY",
    "PGP PRIVATE KEY BLOCK",
)


def _pem(rng: random.Random) -> str:
    label = rng.choice(_PEM_LABELS)
    return (
        f"-----BEGIN {label}-----\n"
        + _pick(rng, _B64, 48)
        + "\n"
        + _pick(rng, _B64, 48)
        + f"\n-----END {label}-----"
    )


def _eth_address(rng: random.Random) -> str:
    lower = _pick(rng, _HEX, 40)
    while len(set(lower)) == 1:  # avoid the single-nibble placeholder
        lower = _pick(rng, _HEX, 40)
    return "0x" + eth_checksum(lower)  # EIP-55 mixed-case, exercises the checksum


def _btc_address(rng: random.Random) -> str:
    return base58check_encode(rng.choice((0x00, 0x05)), rng.randbytes(20))


def _btc_bech32(rng: random.Random) -> str:
    witver = rng.choice((0, 1))
    size = 20 if (witver == 0 and rng.random() < 0.5) else 32
    return bech32_encode("bc", witver, rng.randbytes(size))


# Value generators for rules whose match is context-free.
VALUE_GENERATORS: dict[str, tuple[str, Callable[[random.Random], str]]] = {
    "email": ("EMAIL", lambda r: f"user{_pick(r, '0123456789', 4)}@corp{r.randrange(9)}.example"),
    "ipv4": (
        "IPV4",
        lambda r: f"172.{r.randrange(16, 32)}.{r.randrange(256)}.{r.randrange(1, 255)}",
    ),
    "credit_card": ("CREDIT_CARD", _luhn_complete),
    "phone_number": ("PHONE", _phone),
    "us_ssn": ("SSN", _ssn),
    "iban": ("IBAN", _iban),
    "canadian_sin": ("CA_SIN", _canadian_sin),
    "uk_nino": ("UK_NINO", _uk_nino),
    "aadhaar": ("AADHAAR", _aadhaar),
    "australian_tfn": ("AU_TFN", _australian_tfn),
    "spanish_dni": ("ES_DNI", _spanish_dni),
    "french_nir": ("FR_NIR", _french_nir),
    "german_steuer_id": ("DE_STEUER_ID", _german_steuer_id),
    "swiss_ahv": ("CH_AHV", _swiss_ahv),
    "swedish_personnummer": ("SE_PNR", _swedish_personnummer),
    "brazilian_cpf": ("BR_CPF", _brazilian_cpf),
    "italian_codice_fiscale": ("IT_CF", _italian_codice_fiscale),
    "belgian_nn": ("BE_NN", _belgian_nn),
    "finnish_hetu": ("FI_HETU", _finnish_hetu),
    "nhs_number": ("NHS_NUMBER", _nhs_number),
    "norwegian_fnr": ("NO_FNR", _norwegian_fnr),
    "korean_rrn": ("KR_RRN", _korean_rrn),
    "singapore_nric": ("SG_NRIC", _singapore_nric),
    "chinese_resident_id": ("CN_RESIDENT_ID", _chinese_resident_id),
    "japanese_my_number": ("JP_MY_NUMBER", _japanese_my_number),
    "thai_id": ("TH_ID", _thai_id),
    "irish_pps": ("IE_PPS", _irish_pps),
    "mexican_curp": ("MX_CURP", _mexican_curp),
    "eth_address": ("ETH_ADDRESS", _eth_address),
    "btc_address": ("BTC_ADDRESS", _btc_address),
    "btc_bech32": ("BTC_ADDRESS", _btc_bech32),
    "aws_access_key_id": (
        "AWS_KEY",
        lambda r: r.choice(("AKIA", "ASIA")) + _pick(r, "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", 16),
    ),
    "github_token": ("GITHUB_TOKEN", lambda r: "ghp_" + _pick(r, _ALNUM, 36)),
    "github_fine_grained_pat": (
        "GITHUB_TOKEN",
        lambda r: "github_pat_" + _pick(r, _ALNUM + "_", 40),
    ),
    "anthropic_api_key": ("ANTHROPIC_KEY", lambda r: "sk-ant-api03-" + _pick(r, _ALNUM, 24)),
    "openai_api_key": (
        "OPENAI_KEY",
        # Half classic sk-, half project-scoped sk-proj- (same rule): pins
        # that the generic prefix keeps covering the newer format.
        lambda r: "sk-" + ("proj-" if r.random() < 0.5 else "") + _pick(r, _ALNUM, 32),
    ),
    "openrouter_key": ("OPENROUTER_KEY", lambda r: "sk-or-v1-" + _pick(r, _HEX, 48)),
    "groq_key": ("GROQ_KEY", lambda r: "gsk_" + _pick(r, _ALNUM, 44)),
    "gitlab_pat": ("GITLAB_TOKEN", lambda r: "glpat-" + _pick(r, _ALNUM + "_-", 22)),
    "gitlab_token": (
        "GITLAB_TOKEN",
        lambda r: (
            r.choice(["glrt-", "glcbt-", "gldt-", "glptt-", "glagent-", "glimt-", "glsoat-"])
            + _pick(r, _ALNUM + "_-", 22)
        ),
    ),
    "google_oauth_client_secret": (
        "GOOGLE_OAUTH_SECRET",
        lambda r: "GOCSPX-" + _pick(r, _ALNUM + "_-", 28),
    ),
    "sentry_token": (
        "SENTRY_TOKEN",
        lambda r: r.choice(["sntrys_", "sntryu_"]) + _pick(r, _ALNUM + "_-", 44),
    ),
    "xai_key": ("XAI_KEY", lambda r: "xai-" + _pick(r, _ALNUM, 80)),
    "perplexity_key": ("PERPLEXITY_KEY", lambda r: "pplx-" + _pick(r, _ALNUM, 48)),
    "hashicorp_vault_token": (
        "VAULT_TOKEN",
        lambda r: r.choice(("hvs.", "hvb.")) + _pick(r, _ALNUM + "_-", 26),
    ),
    "langsmith_key": (
        "LANGSMITH_KEY",
        lambda r: r.choice(("lsv2_pt_", "lsv2_sk_")) + _pick(r, _ALNUM, 36),
    ),
    "replicate_token": ("REPLICATE_TOKEN", lambda r: "r8_" + _pick(r, _ALNUM, 37)),
    "pinecone_key": ("PINECONE_KEY", lambda r: "pcsk_" + _pick(r, _ALNUM + "_", 30)),
    "tavily_key": (
        "TAVILY_KEY",
        lambda r: "tvly-" + r.choice(("", "dev-", "prod-")) + _pick(r, _ALNUM, 28),
    ),
    "firecrawl_key": ("FIRECRAWL_KEY", lambda r: "fc-" + _pick(r, _HEX, 32)),
    "nvidia_key": ("NVIDIA_KEY", lambda r: "nvapi-" + _pick(r, _ALNUM + "_-", 56)),
    "cerebras_key": ("CEREBRAS_KEY", lambda r: "csk-" + _pick(r, _ALNUM, 40)),
    "langfuse_secret_key": (
        "LANGFUSE_KEY",
        lambda r: "sk-lf-" + "-".join(_pick(r, _HEX, n) for n in (8, 4, 4, 4, 12)),
    ),
    "figma_pat": ("FIGMA_PAT", lambda r: "figd_" + _pick(r, _ALNUM + "_-", 40)),
    "new_relic_key": (
        "NEW_RELIC_KEY",
        lambda r: "NRAK-" + _pick(r, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 27),
    ),
    "grafana_service_account": (
        "GRAFANA_TOKEN",
        lambda r: "glsa_" + _pick(r, _ALNUM, 32) + "_" + _pick(r, _HEX, 8),
    ),
    "jina_key": ("JINA_KEY", lambda r: "jina_" + _pick(r, _ALNUM, 64)),
    "telegram_bot_token": (
        "TELEGRAM_BOT_TOKEN",
        lambda r: _pick(r, "0123456789", 9) + ":" + _pick(r, _ALNUM + "_-", 35),
    ),
    "databricks_token": ("DATABRICKS_TOKEN", lambda r: "dapi" + _pick(r, _HEX, 32)),
    "bitbucket_app_password": ("BITBUCKET_TOKEN", lambda r: "ATBB" + _pick(r, _ALNUM, 26)),
    "atlassian_api_token": (
        "ATLASSIAN_TOKEN",
        lambda r: "ATATT" + _pick(r, _ALNUM + "_-", 24) + ("=" if r.random() < 0.5 else ""),
    ),
    "huggingface_token": ("HF_TOKEN", lambda r: "hf_" + _pick(r, _ALNUM, 32)),
    "slack_token": (
        "SLACK_TOKEN",
        lambda r: "xoxb-" + _pick(r, "0123456789", 12) + "-" + _pick(r, _ALNUM, 20),
    ),
    "google_api_key": ("GOOGLE_API_KEY", lambda r: "AIza" + _pick(r, _ALNUM + "_-", 35)),
    "jwt": ("JWT", _jwt),
    "stripe_key": (
        "STRIPE_KEY",
        lambda r: ("sk" if r.random() < 0.5 else "rk") + "_live_" + _pick(r, _ALNUM, 24),
    ),
    "sendgrid_key": (
        "SENDGRID_KEY",
        lambda r: "SG." + _pick(r, _ALNUM + "_-", 22) + "." + _pick(r, _ALNUM + "_-", 43),
    ),
    "twilio_id": ("TWILIO_ID", lambda r: ("AC" if r.random() < 0.5 else "SK") + _pick(r, _HEX, 32)),
    "npm_token": ("NPM_TOKEN", lambda r: "npm_" + _pick(r, _ALNUM, 36)),
    "pypi_token": ("PYPI_TOKEN", lambda r: "pypi-AgEIcHlwaS5vcmc" + _pick(r, _ALNUM + "_-", 56)),
    "private_key": ("PRIVATE_KEY", _pem),
    "ipv6": ("IPV6", _ipv6),
    "tailscale_key": (
        "TAILSCALE_KEY",
        lambda r: (
            "tskey-"
            + r.choice(["api", "auth", "client"])
            + "-k"
            + _pick(r, _ALNUM, 10)
            + "-"
            + _pick(r, _ALNUM, 24)
        ),
    ),
    "digitalocean_token": (
        "DO_TOKEN",
        lambda r: r.choice(["dop_v1_", "doo_v1_", "dor_v1_"]) + _pick(r, _HEX, 64),
    ),
    "notion_token": (
        "NOTION_TOKEN",
        lambda r: (
            "ntn_" + _pick(r, _ALNUM, 46) if r.random() < 0.5 else "secret_" + _pick(r, _ALNUM, 43)
        ),
    ),
    "linear_api_key": (
        "LINEAR_KEY",
        lambda r: "lin_" + r.choice(["api", "oauth"]) + "_" + _pick(r, _ALNUM, 40),
    ),
    "supabase_key": (
        "SUPABASE_KEY",
        lambda r: (
            "sbp_" + _pick(r, _HEX, 40)
            if r.random() < 0.5
            else "sb_secret_" + _pick(r, _ALNUM + "_", 30)
        ),
    ),
    "planetscale_token": (
        "PLANETSCALE_TOKEN",
        lambda r: (
            "pscale_"
            + r.choice(["tkn", "oauth", "pw"])
            + "_"
            # 40 or the most, 64 characters: the punctuation contexts put
            # dots after the longest too.
            + _pick(r, _ALNUM + "._-", r.choice((38, 62)))
            + _pick(r, _ALNUM, 2)
        ),
    ),
    "doppler_token": (
        "DOPPLER_TOKEN",
        lambda r: "dp." + r.choice(["pt", "st", "ct", "sa"]) + "." + _pick(r, _ALNUM, 43),
    ),
    "postman_key": (
        "POSTMAN_KEY",
        lambda r: "PMAK-" + _pick(r, _HEX, 24) + "-" + _pick(r, _HEX, 34),
    ),
    "airtable_pat": (
        "AIRTABLE_PAT",
        lambda r: "pat" + _pick(r, _ALNUM, 14) + "." + _pick(r, _HEX, 64),
    ),
    "shopify_token": (
        "SHOPIFY_TOKEN",
        lambda r: "shp" + r.choice(["at", "ss", "ca", "pa"]) + "_" + _pick(r, _HEX, 32),
    ),
}


def _url_credentials(rng: random.Random) -> tuple[str, str]:
    # Passwords carry at least one non-email character: an all-alnum
    # password followed by @host forms an email shape, and the longer EMAIL
    # match deliberately wins that overlap (the value is still redacted,
    # typed EMAIL — pinned in test_detectors). The recall gate covers the
    # spans this rule is meant to own.
    password = list(_pick(rng, _ALNUM, rng.randrange(8, 20)))
    password.insert(rng.randrange(len(password) + 1), rng.choice("!$*!~"))
    scheme = rng.choice(["postgres", "mysql", "redis", "amqp", "mongodb"])
    user = _pick(rng, "abcdefghijklmnopqrstuvwxyz", 6)
    joined = "".join(password)
    return joined, f"{scheme}://{user}:{joined}@db{rng.randrange(9)}.internal:5432/app"


def _aws_secret(rng: random.Random) -> tuple[str, str]:
    value = _pick(rng, _B64, 40)
    if rng.random() < 0.5:  # bare: the punctuation contexts touch the value
        return value, f"aws_secret_access_key = {value}"
    return value, f'aws_secret_access_key = "{value}"'


def _gcp_key_id(rng: random.Random) -> tuple[str, str]:
    value = _pick(rng, _HEX, 40)
    return value, f'"private_key_id": "{value}"'


def _azure_key(rng: random.Random) -> tuple[str, str]:
    value = base64.b64encode(rng.randbytes(48)).decode()
    if rng.random() < 0.5:  # the key last: the punctuation contexts touch it
        return value, f"DefaultEndpointsProtocol=https;AccountKey={value}"
    return value, f"DefaultEndpointsProtocol=https;AccountKey={value};EndpointSuffix=example"


def _generic_secret(rng: random.Random) -> tuple[str, str]:
    value = _pick(rng, _ALNUM, 28)
    if rng.random() < 0.5:  # bare: the punctuation contexts touch the value
        return value, f"password = {value}"
    return value, f'password = "{value}"'


# Rules whose grammar includes surrounding context: (value, full text).
CONTEXTUAL_GENERATORS: dict[str, tuple[str, Callable[[random.Random], tuple[str, str]]]] = {
    "aws_secret_key": ("AWS_SECRET", _aws_secret),
    "gcp_private_key_id": ("GCP_KEY_ID", _gcp_key_id),
    "azure_storage_key": ("AZURE_STORAGE_KEY", _azure_key),
    "generic_secret": ("SECRET", _generic_secret),
    "url_credentials": ("URL_PASSWORD", _url_credentials),
}

# Neutral contexts for context-free values. The placeholder word "contact"
# is deliberately not a secret keyword, so generic_secret never competes.
_CONTEXTS: tuple[str, ...] = (
    "Please use {v} when you configure the integration.",
    '{{"field": "{v}", "note": "from config"}}',
    'contact = "{v}"  # loaded at startup',
    "The value {v} appeared in the log output.",
)

# Every rule's values (the contextual rules' whole text) are also put in each
# of these, once: a value that ends a sentence or a clause, or sits in
# brackets or quotes, must be found exactly as in plain prose. A
# sentence-final "." once hid four rules' values (planetscale_token's when
# its body is 64 characters); the recall gate holds all rules to it now. No
# word here is a secret keyword.
PUNCTUATION_CONTEXTS: tuple[str, ...] = (
    "It is {v}.",
    "It is {v}. Next line.",
    "It is {v}...",
    "First {v}, then the rest.",
    "First {v}; then the rest.",
    "Reached {v}: no reply.",
    "Found {v}! Done.",
    "Is it {v}? Yes.",
    "(see {v}) for details",
    "[see {v}] for details",
    "see ({v} for details",
    "see [{v} for details",
    "({v})",
    "[{v}]",
    'quoted "{v}" here',
    "quoted '{v}' here",
    '"{v}."',
    "„{v}.“",
    "It is **{v}.**",
    # In a tool call's arguments, redacted as JSON source: an escaped line
    # break follows the sentence's ".".
    '{"content": "It is {v}.\\nNext line."}',
    "It ends with {v}",
)


# A card written next to another digit group (a quantity, a year, a CVV,
# an expiry month): the candidate the credit_card regex reads takes the
# group in and fails the Luhn check, and the card is read inside it
# (regex_rules._card_inside). {g} and {h} are digit groups of the given
# lengths, drawn so that every candidate holding one fails the check.
CARD_RUN_CONTEXTS: tuple[tuple[str, int, int], ...] = (
    ("Order {g} {v} shipped.", 2, 0),
    ("Item {g}-{v}", 1, 0),
    ("ref {g} {v}", 4, 0),
    ("card {v} {h}", 0, 3),
    ("card {v} {h}/27", 0, 2),
    ('{{"card": "{v} {h}"}}', 0, 3),
    ("Order {g} {v} {h}, thanks", 2, 3),
)


def _grouped_card(rng: random.Random) -> str:
    card = _luhn_complete(rng)
    separator = rng.choice(("", " ", "-"))
    return separator.join(card[i : i + 4] for i in range(0, 16, 4)) if separator else card


def _holds_another_card(groups: list[str], card: range) -> bool:
    """Whether a run of consecutive ``groups`` that takes in one outside
    ``card`` (the card's own groups) is 13 to 19 digits and passes the
    Luhn check: a card the rule could read instead of the labeled one."""
    for i in range(len(groups)):
        for j in range(i + 1, len(groups) + 1):
            run = "".join(groups[i:j])
            if (i < card.start or j > card.stop) and 13 <= len(run) <= 19 and _luhn_valid(run):
                return True
    return False


def card_run_samples(rng: random.Random) -> list[Sample]:
    """One card (solid, or in groups of four) in each CARD_RUN_CONTEXTS
    context, labeled at the card."""
    samples: list[Sample] = []
    for context, before, after in CARD_RUN_CONTEXTS:
        card = _grouped_card(rng)
        own = re.split("[ -]", card)
        while True:
            g, h = _pick(rng, "0123456789", before), _pick(rng, "0123456789", after)
            if before == 4:
                # A group of four before the card leads with 0 or 1: with the
                # card's first two it would be an Aadhaar number's 4-4-4 when
                # it led with 2-9, and that starts first and wins.
                g = rng.choice("01") + g[1:]
            groups = [g] * bool(g) + own + [h] * bool(h)
            if not _holds_another_card(groups, range(bool(g), bool(g) + len(own))):
                break
        text = context.format(g=g, h=h, v=card)
        start = text.index(card)
        samples.append(Sample(text, (LabeledSpan(start, start + len(card), "CREDIT_CARD"),)))
    return samples


def _decoys(rng: random.Random) -> list[str]:
    return [
        f"request id {uuid.UUID(int=rng.getrandbits(128))}",
        f"commit {_pick(rng, _HEX, 40)} on main",
        f"sha256:{_pick(rng, _HEX, 64)}",
        f"data blob {_pick(rng, _B64, 80)}== inline",  # base64 without AccountKey context
        f"function {_pick(rng, 'abcdefghijklmnopqrstuvwxyz', 8)}_handler_v2()",
        f"version {rng.randrange(9)}.{rng.randrange(20)}.{rng.randrange(20)}",
        "timestamp 2026-07-07T12:00:00Z recorded",
        "/usr/local/lib/python3.13/site-packages/httpx/_client.py",
        'password = "changeme"',  # low entropy: must not fire
        "card 4111 1111 1111 1112 rejected",  # fails Luhn
        "sk-tooshort",
        "eyJnotarealheader.zzzzzzzzzzzzzz.zzzzz",  # fake JWT
        "hf_" + "a" * 30,  # low entropy
        "AIza short",
        "identifier USER_001 in scope",
        # Phone near-misses: bare digit runs and truncated forms never fire.
        f"invoice {_pick(rng, '0123456789', 10)} paid",
        "call 212-555-01 back",  # too short for the grammar
        "ticket 123-45-6789012 escalated",  # SSN shape inside a longer run
        # Invalid SSN areas/groups/serials must be vetoed by the validator.
        "000-12-3456 on file",
        "666-12-3456 on file",
        f"9{rng.randrange(10)}{rng.randrange(10)}-12-3456 on file",
        "123-00-4567 on file",
        "123-45-0000 on file",
        # Valid-shaped IBAN with a corrupted checksum digit, solid and in the
        # paper format (NL: its 10-digit account number forms no other
        # rule's digit grouping, a card's 13+ or a 4-4-4 id).
        f"ref {_corrupt_iban(rng)} rejected",
        f"ref {_paper(_corrupt_iban(rng, 'NL'))} rejected",
        # A value glued into a longer dotted or colon token is not the value:
        # a "." or ":" ends one only before a non-word character.
        f"build {_swiss_ahv(rng)}.{rng.randrange(10)} done",
        f"id {_belgian_nn(rng)}.{rng.randrange(10)} done",
        f"addr {_ipv6(rng)}:x done",
        f"addr {_ipv6(rng)}.{rng.randrange(10)} done",
        "ratio 1:2:3. done",  # letterless, three groups: vetoed
        "slice x[::2]. done",
        "at 08:00:01: done",
    ]


# What a value takes in of the characters right after it, as its rule's
# regex reads them: a generic secret's value runs on over every character
# of its alphabet ("." included, as many as follow; the rule is
# case-insensitive, so "ſ", "K" and "ı" are of its alphabet too), and a
# PlanetScale token over the token alphabet: up to 64 characters before a
# character that is not a word character, or with only dots past the 64th
# when whitespace, a closing quote or bracket, "*", "<", a backslash or the
# end of the text follows them (the rules' _CLOSE) and no other token starts
# in the run. Every other rule's value ends where its generated value does.
_SECRET_VALUE = re.compile(r"[A-Za-z0-9/+=_\-\.]*", re.IGNORECASE)
_PLANETSCALE_PREFIX = re.compile(r"pscale_(?:tkn|oauth|pw)_")
_PLANETSCALE_RUN = re.compile(r"[A-Za-z0-9=._-]*")
_PLANETSCALE_NESTED = re.compile(r"[=.-]pscale_(?:tkn|oauth|pw)_")
_WORD = re.compile(r"\w")
_CLOSING = re.compile(r"""[\s"'`)\]}>»”’“‘«‹›*<\\]|\Z""")


def taken_after(detector_type: str, value: str, after: str) -> int:
    """How many of the characters that open ``after`` a ``value`` of
    ``detector_type`` right before them takes in. ValueError when the
    value's rule finds no value there at all (a PlanetScale token whose
    run goes on)."""
    if detector_type == "SECRET":
        return _SECRET_VALUE.match(after).end()  # type: ignore[union-attr]
    if detector_type == "PLANETSCALE_TOKEN":
        prefix = _PLANETSCALE_PREFIX.match(value).end()  # type: ignore[union-attr]
        body = len(value) - prefix
        run = _PLANETSCALE_RUN.match(after).end()  # type: ignore[union-attr]
        if body + run <= 64 and not _WORD.match(after, run):
            return run
        if (
            body + run > 64
            and not after[64 - body : run].strip(".")
            and _CLOSING.match(after, run)
            and not _PLANETSCALE_NESTED.search(value[prefix:] + after[:run])
        ):
            return run
        raise ValueError("the token's run goes on")
    return 0


def _embedded(context: str, value: str, text: str, detector_type: str) -> Sample:
    """``text`` (holding ``value``) put in ``context``, labeled at ``value``
    (with what it takes in of the characters after it, when it ends
    ``text``)."""
    before, after = context.split("{v}")
    start = len(before) + text.find(value)
    end = start + len(value)
    if text.endswith(value):
        end += taken_after(detector_type, value, after)
    return Sample(before + text + after, (LabeledSpan(start, end, detector_type),))


def punctuation_samples(rng: random.Random) -> list[Sample]:
    """One value of every rule in each punctuation context."""
    samples: list[Sample] = []
    for context in PUNCTUATION_CONTEXTS:
        for detector_type, gen in VALUE_GENERATORS.values():
            value = gen(rng)
            samples.append(_embedded(context, value, value, detector_type))
        for detector_type, gen_ctx in CONTEXTUAL_GENERATORS.values():
            value, text = gen_ctx(rng)
            samples.append(_embedded(context, value, text, detector_type))
    return samples


def generate(seed: int = 42, samples_per_rule: int = 25) -> list[Sample]:
    """Build the labeled corpus: positives for every rule plus decoys."""
    rng = random.Random(seed)
    corpus: list[Sample] = []

    for _name, (detector_type, gen) in VALUE_GENERATORS.items():
        for _ in range(samples_per_rule):
            value = gen(rng)
            context = rng.choice(_CONTEXTS)
            text = context.replace("{v}", value)
            start = text.find(value)
            corpus.append(Sample(text, (LabeledSpan(start, start + len(value), detector_type),)))

    for _name, (detector_type, gen_ctx) in CONTEXTUAL_GENERATORS.items():
        for _ in range(samples_per_rule):
            value, text = gen_ctx(rng)
            start = text.find(value)
            corpus.append(Sample(text, (LabeledSpan(start, start + len(value), detector_type),)))

    for _ in range(samples_per_rule):
        for decoy in _decoys(rng):
            corpus.append(Sample(decoy, ()))
        corpus.extend(card_run_samples(rng))

    corpus.extend(punctuation_samples(rng))

    rng.shuffle(corpus)
    return corpus
