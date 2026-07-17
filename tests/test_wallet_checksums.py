"""Wallet checksum primitives pinned against PUBLISHED vectors.

The bench corpus generates addresses with the same primitives the validators
enforce, so a self-consistent bug (e.g. a transposed Keccak constant) would
pass the recall gate. These vectors come from the Keccak/EIP-55/BIP-173/BIP-350
specs — independent of this implementation — and are the real correctness net.
"""

import re

from llm_redact.detection.wallet_checksums import (
    base58check_encode,
    base58check_ok,
    bech32_address_ok,
    bech32_encode,
    eth_address_ok,
    eth_checksum,
    keccak256,
)


def _m(s: str) -> "re.Match[str]":
    match = re.match(r".*", s, re.DOTALL)
    assert match is not None
    return match


def test_keccak256_published_vectors() -> None:
    # Ethereum Keccak-256 (NOT NIST SHA3) of the empty string and "abc".
    assert keccak256(b"").hex() == (
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    )
    assert keccak256(b"abc").hex() == (
        "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
    )


# The four EIP-55 example addresses from the spec (checksum bodies only).
EIP55 = (
    "5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
    "fB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
    "dbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB",
    "D1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb",
)


def test_eip55_spec_addresses_round_trip() -> None:
    for body in EIP55:
        assert eth_checksum(body.lower()) == body
        assert eth_address_ok(_m("0x" + body))


def test_eth_mixed_case_bad_checksum_rejected() -> None:
    # Flip one letter's case: the EIP-55 checksum must now fail.
    bad = "5AAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
    assert not eth_address_ok(_m("0x" + bad))


def test_eth_all_lower_or_upper_accepted_on_shape() -> None:
    body = EIP55[0]
    assert eth_address_ok(_m("0x" + body.lower()))
    assert eth_address_ok(_m("0x" + body.upper()))


def test_eth_null_address_rejected() -> None:
    assert not eth_address_ok(_m("0x" + "0" * 40))
    assert not eth_address_ok(_m("0x" + "f" * 40))


def test_base58check_known_addresses() -> None:
    # Bitcoin genesis P2PKH and a documented P2SH address.
    assert base58check_ok(_m("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"))
    assert base58check_ok(_m("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy"))


def test_base58check_tamper_rejected() -> None:
    assert not base58check_ok(_m("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNb"))


def test_base58check_encode_round_trips() -> None:
    enc = base58check_encode(0x00, bytes(range(20)))
    assert base58check_ok(_m(enc))


def test_bech32_bip_vectors() -> None:
    # BIP-173 witness v0 (P2WPKH) and BIP-350 witness v1 (taproot).
    assert bech32_address_ok(_m("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"))
    assert bech32_address_ok(_m("bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0"))


def test_bech32_tamper_rejected() -> None:
    assert not bech32_address_ok(_m("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5"))


def test_bech32_encode_round_trips() -> None:
    assert bech32_address_ok(_m(bech32_encode("bc", 0, bytes(range(20)))))
    assert bech32_address_ok(_m(bech32_encode("bc", 1, bytes(range(32)))))
