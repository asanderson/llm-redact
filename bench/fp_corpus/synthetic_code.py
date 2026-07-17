"""FP-corpus probe: ordinary application code full of secret-shaped values.

Every literal below is fake and chosen to sit just outside a detection
rule's grammar or to be vetoed by its validator. None of this file should
ever be redacted; if a rule fires here, precision regressed.
"""

import hashlib
import uuid

# Bare hex digests: 40-hex (git/SHA-1) and 64-hex (SHA-256) with no
# "private_key_id" JSON context and no AC/SK prefix.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
PINNED_COMMITS = [
    "da39a3ee5e6b4b0d3255bfef95601890afd80709",
    "356a192b7913b04c54574d18c28d46e6395428ab",
    "77de68daecd823babbb58edb1c8e14d7106e83bb",
]

# UUIDs: hyphenated segments keep the SSN lookarounds from matching.
NAMESPACE_DNS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
REQUEST_IDS = (
    "550e8400-e29b-41d4-a716-446655440000",
    "f47ac10b-58cc-4372-a567-0e02b2c3d479",
)

# Version tuples and dotted versions: never enough octets for the IPv4 rule.
KNOWN_GOOD = {"httpx": "0.28.1", "starlette": "0.41.3", "uvicorn": "0.34.0"}
KERNEL = "6.12.9"
USER_AGENT = "order-processor/2.14.0 (build 20260601.4)"

# Ten-digit ids and separator-free digit runs: the phone rule requires
# separators or a + prefix by design.
INVOICE_NUMBERS = [9876543210, 4155550100, 2125550100]
EPOCH_MS = 1717228819123

# Hex colors and escape-ish tokens.
PALETTE = ["#deadbe", "#c0ffee", "#0f172a", "#f97316"]

# Query fragments with '=' but no secret keyword on the left.
CANNED_QUERIES = [
    "page=3&per_page=100&sort=created_at",
    "filter=status:open&assignee=none",
]


def cache_key(namespace: str, item_id: int) -> str:
    # A derived identifier that is long and alphanumeric but has no
    # keyword context: b64-ish output must not fire the generic rule.
    return f"{namespace}:v2:{item_id:010d}"


def content_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class RetryPolicy:
    """Docstring with number soup: 3 retries, backoff 2.0s/4.0s/8.0s,
    deadline 2026-06-01T08:00:00Z, jitter 0.415."""

    max_attempts = 5
    backoff_seconds = (2.0, 4.0, 8.0, 16.0)


# Prefixes one character away from real vendor grammars.
ALMOST_KEYS = [
    "sk-proj-abc123",  # sk- but only 11 chars of body (min is 20)
    "ghp_tooshort123",  # ghp_ needs 36+
    "AKIAABC123",  # AKIA needs 16 more uppercase/digits
    "xoxq-not-a-slack-prefix-1234567890",  # xox[baprs] only
    "npm_abcdef",  # npm_ needs exactly 36
    "hf_short",  # hf_ needs 30+
    "gsk_abcdefghijklmnop",  # gsk_ needs 40+
]

if __name__ == "__main__":
    for commit in PINNED_COMMITS:
        print(cache_key("commits", len(commit)), content_digest(commit.encode()))

# ipv6-rule probes: valid-to-the-parser shapes that are not addresses in
# context. Slices are letterless with <4 groups; the serial is 8 hex pairs;
# times/MACs fail the parser outright.
SLICED = [1, 2, 3, 4, 5, 6][::2]
REVERSED_EVENS = [[1, 2], [3, 4]][1::2]
CERT_SERIAL = "04:9f:86:d0:81:88:4c:7d"
BOOT_TIME = "08:00:01"
DEVICE_MAC = "00:1b:44:11:3a:b7"

# url_credentials probes: URLs with ports or users but NO password.
ENDPOINTS = [
    "https://api.example.com:8443/v2/items",
    "ssh://git@build.internal/repo.git",
    "http://readonly@metrics.internal/dash",
]
