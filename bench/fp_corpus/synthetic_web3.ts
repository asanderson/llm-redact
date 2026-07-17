// Web3 / EVM client code. Every crypto-shaped value here is a NEGATIVE:
// hashes, selectors, slots, and near-miss addresses that must NOT be
// redacted. Real all-lowercase / EIP-55 addresses are deliberately absent
// (they would legitimately fire) — this file pins that the near-misses do not.

import { keccak256, getAddress } from "ethers";

// 32-byte hashes are 0x + 64 hex — NOT the 40-hex address shape.
const TX_HASH = "0x9f2c4d7e1a3b5c6d8e0f1a2b3c4d5e6f708192a3b4c5d6e7f80912a3b4c5d6e7";
const BLOCK_ROOT = "0xdeadbeefcafebabefeedface0123456789abcdef0123456789abcdef01234567abcd";

// A 4-byte function selector and a storage slot: too short / too long.
const SELECTOR = "0xa9059cbb"; // transfer(address,uint256)
const SLOT = "0x0000000000000000000000000000000000000000000000000000000000000003";

// A mixed-case address whose EIP-55 checksum is DELIBERATELY broken (one
// letter's case flipped). ethers' getAddress() would throw; our detector
// must reject it as a typo, not redact it.
const BAD_CHECKSUM = "0x5AAeb6053F3E94C9b9A09f33669435E7Ef1BeAed";

// An IPFS CIDv0 is base58 but starts with "Qm" (not 1/3), so it is not a
// Bitcoin address; and a base58 string that starts with 1 but fails the
// double-SHA256 checksum (last char bumped) is a typo, not an address.
const IPFS_CID = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG";
const BAD_BTC = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNZ";

// A bech32-shaped string with a corrupted checksum is likewise not valid.
const BAD_SEGWIT = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3zz";

export async function balances(provider: any) {
  const slot = BigInt(SLOT);
  return { TX_HASH, BLOCK_ROOT, SELECTOR, slot, BAD_CHECKSUM, IPFS_CID, BAD_BTC, BAD_SEGWIT };
}
