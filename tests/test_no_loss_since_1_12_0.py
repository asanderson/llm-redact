"""Nothing 1.12.0 redacted goes upstream in 1.12.1.

1.12.1 changes five built-in rules: belgian_nn, swiss_ahv, ipv6, iban and
planetscale_token. Their 1.12.0 definitions are frozen below. The 1.12.0
pipeline is the production rule list with those five swapped back, through
the same overlap resolution; every character it redacts must be redacted by
this release's production Redactor. The inputs: the bench corpus at three
seeds (every rule's values in every punctuation context); every value
glued to punctuation (a JSON string's escapes too), to another value and
to a PlanetScale run (dots past the 64-character cap, another token
inside); the start of every value grafted onto the end of a
checksum-valid spaced IBAN; and generic secrets of low and border
entropy. There are no exceptions: a loss is fixed in the rules.

This holds under the default [detection.modes]: a newly read value wins
an overlap like any longer match, so the value it takes in follows its
rule's mode (docs/detection.md says so).

A digest pins every other rule to its 1.12.0 definition (its fields, and
its validator with every helper and table it uses), so changing one later
fails here until its 1.12.0 definition is frozen too.
"""

import dataclasses
import hashlib
import importlib
import inspect
import ipaddress
import random
import re
import types
from collections.abc import Iterable

import pytest

from llm_redact.bench.corpus import CONTEXTUAL_GENERATORS, VALUE_GENERATORS, generate
from llm_redact.detection import regex_rules
from llm_redact.detection.base import Detection
from llm_redact.detection.engine import (
    DetectionConfig,
    DetectorPlan,
    build_allowlist,
    build_detectors,
)
from llm_redact.detection.regex_rules import (
    _GENERIC_VALIDATORS,
    _IBAN_PAPER_LENGTHS,
    BUILTIN_RULES,
    RegexRule,
)
from llm_redact.placeholders import PLACEHOLDER_RE
from llm_redact.redactor import Redactor, _resolve_overlaps
from llm_redact.vault import InMemoryVault

# --- 1.12.0, frozen: copied from src/llm_redact/detection/regex_rules.py at v1.12.0 ---


def _valid_ipv6(match: re.Match[str]) -> bool:
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


_IBAN_LENGTHS = {
    "AD": 24, "AT": 20, "BE": 16, "CH": 21, "CZ": 24, "DE": 22, "DK": 18,
    "ES": 24, "FI": 18, "FR": 27, "GB": 22, "IE": 22, "IT": 27, "LU": 20,
    "NL": 18, "NO": 15, "PL": 28, "PT": 25, "SE": 24,
}  # fmt: skip


def _iban_ok(match: re.Match[str]) -> bool:
    value = match.group(0)
    expected = _IBAN_LENGTHS.get(value[:2])
    if expected is not None and len(value) != expected:
        return False
    rearranged = value[4:] + value[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


def _ahv_ok(match: re.Match[str]) -> bool:
    digits = [int(c) for c in match.group(0).replace(".", "")]
    total = sum(d * (3 if i % 2 else 1) for i, d in enumerate(digits[:12]))
    return (10 - total % 10) % 10 == digits[12]


def _belgian_nn_ok(match: re.Match[str]) -> bool:
    value = re.sub(r"[.-]", "", match.group(0))
    body, check = value[:9], int(value[9:])
    month, day = int(body[2:4]), int(body[4:6])
    if not (0 <= month <= 12 and 0 <= day <= 31):
        return False
    return any(check == 97 - int(prefix + body) % 97 for prefix in ("", "2"))


# name -> (detector type, priority, pattern, validator)
FROZEN_1_12_0 = {
    "ipv6": (
        "IPV6",
        100,
        r"(?<![\w:.])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,8}(?![\w:.])",
        _valid_ipv6,
    ),
    "iban": ("IBAN", 100, r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", _iban_ok),
    "swiss_ahv": ("CH_AHV", 100, r"(?<![\w.-])756\.\d{4}\.\d{4}\.\d{2}(?![\w.-])", _ahv_ok),
    "belgian_nn": (
        "BE_NN",
        100,
        r"(?<![\w.-])\d{2}\.\d{2}\.\d{2}-\d{3}\.\d{2}(?![\w.-])",
        _belgian_nn_ok,
    ),
    "planetscale_token": (
        "PLANETSCALE_TOKEN",
        10,
        r"\bpscale_(?:tkn|oauth|pw)_[A-Za-z0-9=._-]{30,64}(?![\w=.-])",
        None,
    ),
}

# Every other built-in rule as 1.12.0 defined it: _digest of its 82 rules,
# computed from the v1.12.0 source.
UNCHANGED_1_12_0 = (82, "810b4a010f221874fff0be32a087dc343432845606d0c8e14392767aae45e31e")


class Frozen:
    """A frozen 1.12.0 rule, scanned as 1.12.0 scanned it."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.detector_type, self.priority, pattern, self.validator = FROZEN_1_12_0[name]
        self.pattern = re.compile(pattern)

    def detect(self, text: str) -> Iterable[Detection]:
        for match in self.pattern.finditer(text):
            if self.validator is None or self.validator(match):
                yield Detection(
                    match.start(), match.end(), self.detector_type, match.group(), self.priority
                )


CONFIG = DetectionConfig()
ALLOWLIST = build_allowlist(CONFIG)
OLD_PLAN = DetectorPlan(
    [Frozen(det.name) if det.name in FROZEN_1_12_0 else det for det in build_detectors(CONFIG)]
)
VAULT = InMemoryVault()
RELEASE = Redactor(build_detectors(CONFIG), VAULT, ALLOWLIST)


def redacted_by_1_12_0(text: str) -> set[int]:
    winners = _resolve_overlaps(OLD_PLAN.detect(text, ALLOWLIST))
    return {i for d in winners for i in range(d.start, d.end)}


def redacted_by_release(text: str) -> set[int]:
    """The offsets the release's Redactor replaces, read back from its
    output: each placeholder stands for its vault value at that place."""
    redacted = RELEASE.redact_text(text)
    offsets: set[int] = set()
    pos = cursor = 0
    for token in PLACEHOLDER_RE.finditer(redacted):
        pos += token.start() - cursor
        value = VAULT.original_for(token.group())
        assert value is not None and text.startswith(value, pos)
        offsets.update(range(pos, pos + len(value)))
        pos, cursor = pos + len(value), token.end()
    return offsets


# --- the inputs -----------------------------------------------------------------------

GENERATORS = [gen for _type, gen in VALUE_GENERATORS.values()] + [
    lambda rng, gen=gen: gen(rng)[1] for _type, gen in CONTEXTUAL_GENERATORS.values()
]
# "\\n" and "\\t": a JSON string's escapes, as a tool call's arguments hold them.
JOINERS = (
    "", " ", ".", "..", ". ", ":", ": ", "-", "_", "=", "/", "+", "@", ",", "\u00a0", "\n",
    "\\n", ".\\n", ":\\t",
)  # fmt: skip
ENDINGS = (
    "", ".", "...", ". x", ".x", ".5", ":", ": x", ":x", "::", "-1", " 1", "@a.example",
    "..@a.example", "+x@a.example", "%x@a.example", ".**", ".</p>", "..“", " c", " C",
    ".\\n", '.\\"', "..\\t", ":\\n", ".\\x", ".\\",
)  # fmt: skip
SEPARATORS = (" ", " ", "\u00a0", "\u202f")
ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# What a spaced IBAN can end in: groups of four after single spaces, the
# last of one to four.
IBAN_TAIL = re.compile(r"(?:[A-Z0-9]{4} )*[A-Z0-9]{1,4}")


def with_check_digits(country: str, bban: str) -> str:
    digits = "".join(str(int(ch, 36)) for ch in bban + country + "00")
    return f"{country}{98 - int(digits) % 97:02d}{bban}"


def iban_ending_in(rng: random.Random, tail: str) -> str | None:
    """A checksum-valid spaced IBAN of a registry country whose last
    groups are ``tail``'s groups, or None when no registry length fits."""
    groups = tail.split(" ")
    compact = "".join(groups)
    fits = [
        (code, length)
        for code, length in _IBAN_PAPER_LENGTHS.items()
        if length - 4 >= len(compact) and (length - len(groups[-1])) % 4 == 0
    ]
    if not fits:
        return None
    code, length = rng.choice(fits)
    filler = "".join(rng.choice(ALNUM) for _ in range(length - 4 - len(compact)))
    value = with_check_digits(code, filler + compact)
    return rng.choice(SEPARATORS).join(value[i : i + 4] for i in range(0, len(value), 4))


def grafted(rng: random.Random, value: str) -> Iterable[str]:
    """``value`` with each stretch at a word start that a spaced IBAN can
    end in replaced by a checksum-valid one ending in it: what 1.12.0 found
    starting there runs on past the IBAN's end."""
    for start in range(len(value)):
        if start and (value[start - 1].isalnum() or value[start - 1] == "_"):
            continue
        for end in range(start + 1, min(len(value), start + 40) + 1):
            if end < len(value) and value[end] in ALNUM:
                continue
            if IBAN_TAIL.fullmatch(value, start, end):
                iban = iban_ending_in(rng, value[start:end])
                if iban is not None:
                    yield value[:start] + iban + value[end:]


def planetscale_run(rng: random.Random) -> str:
    """A PlanetScale token's run around the 64-character cap, dots after
    it, at times another token or a secret keyword inside it."""
    body = "".join(rng.choice(ALNUM + "abc=._-") for _ in range(rng.randrange(0, 60)))
    cut = rng.randrange(len(body) + 1)
    inner = rng.choice(("", ".pscale_tkn_", "-pscale_pw_", "=pscale_oauth_", "=token="))
    kind = rng.choice(("tkn", "oauth", "pw"))
    return f"pscale_{kind}_{body[:cut]}{inner}{body[cut:]}" + "." * rng.randrange(0, 70)


def texts(seed: int) -> Iterable[str]:
    rng = random.Random(seed)
    yield from (sample.text for sample in generate(seed=seed))
    for _ in range(3):
        values = [gen(rng) for gen in GENERATORS]
        for value in values:
            yield from (value + ending for ending in ENDINGS)
            yield from (joiner + value for joiner in JOINERS)
            other = rng.choice(values)
            for joiner in JOINERS:
                yield value + joiner + other
                yield value + joiner + planetscale_run(rng) + rng.choice(ENDINGS)
            # Its start, also regrouped in fours ("QQ12 3456 C"), grafted.
            compact = value.replace(" ", "")
            yield from grafted(rng, value)
            yield from grafted(rng, " ".join(compact[i : i + 4] for i in range(0, len(compact), 4)))
            # A generic secret around the entropy gate, holding the value.
            alphabet = "".join(rng.sample(ALNUM, rng.randrange(2, 14)))
            low = "".join(rng.choice(alphabet) for _ in range(rng.randrange(12, 26)))
            keyword = rng.choice(("password", "token", "secret", "api_key", "PASSWORD"))
            assign = rng.choice((" = ", ": ", "=", ' = "', ":'"))
            secret = rng.choice((value, low, low + value, low + "." + value))
            yield keyword + assign + secret + "." * rng.randrange(0, 4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nothing_1_12_0_redacted_is_forwarded(seed: int) -> None:
    lost, gained, count = [], 0, 0
    for text in texts(seed):
        count += 1
        release, old = redacted_by_release(text), redacted_by_1_12_0(text)
        if old - release:
            lost.append((text, sorted(old - release)))
        gained += bool(release - old)
    assert lost == []
    # Not vacuous: the inputs hold values 1.12.0 forwarded.
    assert count > 20_000 and gained > 1_000


def _as_written(func: types.FunctionType) -> tuple[types.FunctionType, str]:
    """``func`` and its source as written. In mutmut's mutants/ copy a
    function of a module it mutates (validators.py) is a trampoline wrapping
    the function as written (``__wrapped__``) under one added decorator
    line; both are read through, so the digest is the same there, where no
    mutant can change a source anyway."""
    func = inspect.unwrap(func)
    lines = inspect.getsource(func).splitlines(keepends=True)
    added = "@_mutmut_mutated("
    return func, "".join(line for line in lines if not line.lstrip().startswith(added))


def _source(func: types.FunctionType) -> list[str]:
    """The source of ``func`` and of every function it names (of its module,
    or imported in its body), and the value of every other name of theirs
    (a table), transitively; sorted."""
    found: set[str] = set()
    todo = [func]
    while todo:
        func, source = _as_written(todo.pop())
        found.add(source)
        codes = [func.__code__]
        while codes:
            code = codes.pop()
            codes += [const for const in code.co_consts if inspect.iscode(const)]
            scopes = [func.__globals__] + [
                vars(importlib.import_module(name))
                for name in code.co_names
                if name.startswith("llm_redact.")
            ]
            names = [scope[name] for scope in scopes for name in code.co_names if name in scope]
            for value in names:
                if inspect.isfunction(value):
                    if _as_written(value)[1] not in found:
                        todo.append(value)
                elif not inspect.ismodule(value):
                    found.add(repr(sorted(value) if isinstance(value, frozenset) else value))
    return sorted(found)


def _digest(rules: Iterable[RegexRule]) -> tuple[int, str]:
    """How many rules, and a SHA-256 over every field of each (a field at
    its default skipped, so a field added since 1.12.0 leaves a rule that
    does not set it as it was), a validator by its _source."""
    digest = hashlib.sha256()
    count = 0
    for rule in rules:
        count += 1
        for field in dataclasses.fields(rule):
            value = getattr(rule, field.name)
            if value == field.default:
                continue
            if isinstance(value, re.Pattern):
                value = (value.pattern, value.flags)
            elif callable(value):
                value = _source(value)
            digest.update(repr((field.name, value)).encode())
        generic = _GENERIC_VALIDATORS.get(rule.name)
        digest.update(repr(generic and _source(generic)).encode())
    return count, digest.hexdigest()


def test_every_other_rule_is_as_1_12_0_defined_it(monkeypatch: pytest.MonkeyPatch) -> None:
    unchanged = [rule for rule in BUILTIN_RULES if rule.name not in FROZEN_1_12_0]
    assert _digest(unchanged) == UNCHANGED_1_12_0
    # A helper a validator calls counts as part of it (generic_secret's).
    monkeypatch.setattr(regex_rules, "shannon_entropy", len)
    assert _digest(unchanged) != UNCHANGED_1_12_0


@pytest.mark.parametrize(
    ("text", "forwarded"),
    [
        ("NN 93.05.18-223.61.", "93.05.18-223.61"),
        ("AHV 756.9217.0769.85. Done", "756.9217.0769.85"),
        ("Ship to 2001:db8::1.", "2001:db8::1"),
        ("2001:db8::1: unreachable", "2001:db8::1"),
        ("pay GB82 WEST 1234 5698 7654 32 now", "GB82 WEST 1234 5698 7654 32"),
        ("token pscale_tkn_" + "a1B2c3D4" * 8 + ". Done", "pscale_tkn_" + "a1B2c3D4" * 8),
    ],
)
def test_the_reference_is_1_12_0(text: str, forwarded: str) -> None:
    # The frozen rules are a real reference: each reported value went
    # upstream through them, and the release redacts it.
    start = text.index(forwarded)
    span = set(range(start, start + len(forwarded)))
    assert not span & redacted_by_1_12_0(text)
    assert span <= redacted_by_release(text)
