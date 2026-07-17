"""Checksum validators for cryptocurrency wallet addresses.

Pure stdlib, no new dependency. Three families, each veto-able like the
national-id validators:

- **Ethereum** (`0x` + 40 hex): EIP-55 mixed-case checksum. The checksum needs
  Keccak-256 — the ORIGINAL Keccak, not NIST SHA3 (they differ only in the
  padding byte), which `hashlib` does not provide — so it is vendored here and
  pinned against published Keccak/EIP-55 test vectors. An all-lowercase or
  all-uppercase address carries no checksum and is accepted on shape; a
  mixed-case address must satisfy EIP-55.
- **Bitcoin base58check** (P2PKH `1…`, P2SH `3…`): double-SHA256 checksum over
  the 21-byte version+payload, version byte 0x00 or 0x05. SHA-256 is stdlib.
- **Bitcoin bech32/bech32m** (`bc1…` segwit): the BIP-173/BIP-350 polymod, with
  witness version 0 using bech32 (const 1) and v1+ using bech32m
  (const 0x2bc830a3).

The corpus generators reuse the encode side of these primitives, so recall is
gated on the SAME checksum the validators enforce; correctness rests on the
independent published vectors pinned in tests/test_wallet_checksums.py.
"""

import hashlib
import re

# --- Keccak-256 (Ethereum flavour) ------------------------------------------

_KECCAK_RC = (
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
)

# Rho rotation offsets r[x][y] (Keccak reference table).
_KECCAK_ROT = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)

_MASK64 = (1 << 64) - 1


def _rotl64(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (64 - shift))) & _MASK64


def _keccak_f1600(a: list[list[int]]) -> None:
    for rc in _KECCAK_RC:
        # theta
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rotl64(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                a[x][y] ^= d[x]
        # rho + pi
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rotl64(a[x][y], _KECCAK_ROT[x][y])
        # chi
        for x in range(5):
            for y in range(5):
                a[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y])
        # iota
        a[0][0] ^= rc


def keccak256(data: bytes) -> bytes:
    """Ethereum's Keccak-256 (pad byte 0x01, not SHA3's 0x06)."""
    rate = 136  # 1088-bit rate, 512-bit capacity
    padded = bytearray(data)
    padded.append(0x01)
    while len(padded) % rate != 0:
        padded.append(0x00)
    padded[-1] ^= 0x80
    state = [[0] * 5 for _ in range(5)]
    for off in range(0, len(padded), rate):
        block = padded[off : off + rate]
        for i in range(rate // 8):
            lane = int.from_bytes(block[i * 8 : i * 8 + 8], "little")
            state[i % 5][i // 5] ^= lane
        _keccak_f1600(state)
    out = bytearray()
    while len(out) < 32:
        for i in range(rate // 8):
            if len(out) >= 32:
                break
            out += state[i % 5][i // 5].to_bytes(8, "little")
    return bytes(out[:32])


def eth_checksum(addr_lower: str) -> str:
    """EIP-55 mixed-case encoding of a 40-hex address (no 0x prefix)."""
    digest = keccak256(addr_lower.encode()).hex()
    return "".join(
        ch.upper() if ch.isalpha() and int(digest[i], 16) >= 8 else ch
        for i, ch in enumerate(addr_lower)
    )


def eth_address_ok(match: "re.Match[str]") -> bool:
    hexpart = match.group(0)[2:]
    # Reject placeholders like the null/burn address (a single repeated nibble).
    if len(set(hexpart.lower())) == 1:
        return False
    lower = hexpart.lower()
    # All-lower or all-upper carries no EIP-55 checksum: accept on shape.
    if hexpart in (lower, hexpart.upper()):
        return True
    return hexpart == eth_checksum(lower)


# --- Bitcoin base58check (P2PKH / P2SH) -------------------------------------

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {ch: i for i, ch in enumerate(_B58_ALPHABET)}


def base58_encode(raw: bytes) -> str:
    num = int.from_bytes(raw, "big")
    chars: list[str] = []
    while num > 0:
        num, rem = divmod(num, 58)
        chars.append(_B58_ALPHABET[rem])
    # Leading zero bytes map to leading '1's.
    pad = len(raw) - len(raw.lstrip(b"\x00"))
    return "1" * pad + "".join(reversed(chars))


def base58check_encode(version: int, payload20: bytes) -> str:
    body = bytes((version,)) + payload20
    checksum = hashlib.sha256(hashlib.sha256(body).digest()).digest()[:4]
    return base58_encode(body + checksum)


def base58check_ok(match: "re.Match[str]") -> bool:
    addr = match.group(0)
    num = 0
    for ch in addr:
        idx = _B58_INDEX.get(ch)
        if idx is None:
            return False
        num = num * 58 + idx
    raw = num.to_bytes((num.bit_length() + 7) // 8, "big")
    pad = len(addr) - len(addr.lstrip("1"))
    raw = b"\x00" * pad + raw
    if len(raw) != 25:
        return False
    body, checksum = raw[:21], raw[21:]
    if hashlib.sha256(hashlib.sha256(body).digest()).digest()[:4] != checksum:
        return False
    return body[0] in (0x00, 0x05)  # mainnet P2PKH / P2SH


# --- Bitcoin bech32 / bech32m (segwit) --------------------------------------

_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_BECH32_CONST = 1
_BECH32M_CONST = 0x2BC830A3


def _bech32_polymod(values: list[int]) -> int:
    gen = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
    chk = 1
    for value in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ value
        for i in range(5):
            chk ^= gen[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp: str) -> list[int]:
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def _convertbits(data: bytes, frombits: int, tobits: int, pad: bool) -> list[int] | None:
    acc = 0
    bits = 0
    ret: list[int] = []
    maxv = (1 << tobits) - 1
    for value in data:
        if value < 0 or (value >> frombits):
            return None
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret


def bech32_encode(hrp: str, witver: int, program: bytes) -> str:
    """Encode a segwit address (bech32 for v0, bech32m for v1+)."""
    data = [witver] + (_convertbits(program, 8, 5, True) or [])
    const = _BECH32_CONST if witver == 0 else _BECH32M_CONST
    values = _bech32_hrp_expand(hrp) + data
    polymod = _bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ const
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(_BECH32_CHARSET[d] for d in data + checksum)


def bech32_address_ok(match: "re.Match[str]") -> bool:
    addr = match.group(0)
    # bech32 is case-insensitive but must not be mixed-case; real addresses are
    # lowercase and the regex only matches lowercase, so this is already so.
    pos = addr.rfind("1")
    hrp = addr[:pos]
    if hrp != "bc":
        return False
    data: list[int] = []
    for ch in addr[pos + 1 :]:
        idx = _BECH32_CHARSET.find(ch)
        if idx == -1:
            return False
        data.append(idx)
    if len(data) < 7:  # witver + at least one data char + 6 checksum
        return False
    witver = data[0]
    if witver > 16:
        return False
    const = _BECH32_CONST if witver == 0 else _BECH32M_CONST
    return _bech32_polymod(_bech32_hrp_expand(hrp) + data) == const


__all__ = [
    "base58_encode",
    "base58check_encode",
    "base58check_ok",
    "bech32_address_ok",
    "bech32_encode",
    "eth_address_ok",
    "eth_checksum",
    "keccak256",
]
