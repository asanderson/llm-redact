"""Synthetic NER corpus for the NER bench, generated from a seed at run time.

Samples look like coding-agent traffic — prose, chat turns, JSON tool
results carried as text, code comments, log lines and git output — with
gold spans for the contextual types the NER models add (``PERSON``,
``ADDRESS``, ``DATE_OF_BIRTH``, ``USERNAME``, ``ACCOUNT_NUMBER``) and, for
the structured-regression check, a few ``EMAIL`` and ``PHONE`` values the
regex rules own. Hard negatives carry no gold at all: UUIDs, commit hashes,
CamelCase identifiers (many built from tool names that are also surnames —
Jenkins, Jackson), file paths, stack traces and timestamps; any detection
there is over-redaction.

The names, streets and handles are drawn from small embedded lists and
combined at random: none of it describes a real person. The corpus is never
committed (CLAUDE.md, Bench: positives are generated, never vendored).
"""

import random
import re
from collections.abc import Callable

from llm_redact.bench.corpus import VALUE_GENERATORS
from llm_redact.bench.ner_metrics import GoldSpan, NerSample

# fmt: off
FIRST_NAMES = (
    "Amara", "Bjorn", "Carmen", "Dmitri", "Elena", "Farid", "Grace", "Hiroshi",
    "Ingrid", "Jamal", "Keiko", "Liam", "Mei", "Nikolai", "Olivia", "Priya",
    "Quentin", "Rosa", "Samuel", "Tanvi", "Umar", "Valeria", "Wei", "Ximena",
    "Yusuf", "Zofia", "Aiden", "Beatriz", "Chloe", "Diego", "Esther", "Femi",
)
LAST_NAMES = (
    "Okafor", "Lindqvist", "Delgado", "Volkov", "Rossi", "Haddad", "Whitfield",
    "Tanaka", "Berg", "Abdullah", "Nakamura", "Gallagher", "Chen", "Petrov",
    "Moreau", "Raman", "Duval", "Ferreira", "Osei", "Kapoor", "Siddiqui",
    "Castillo", "Zhang", "Navarro", "Demir", "Kowalski", "Brennan", "Santos",
    "Fischer", "Mensah", "Novak", "Lindgren",
)
STREETS = (
    "Maple", "Harbor", "Juniper", "Willow", "Cedar Ridge", "Lakeview", "Orchard",
    "Kingsley", "Riverside", "Elmwood", "Granite", "Sycamore", "Beacon", "Holloway",
)
STREET_SUFFIXES = ("Street", "Avenue", "Road", "Lane", "Drive", "Boulevard", "Court", "Way")
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)
# Tool and library names that are also surnames or given names: agent
# traffic is full of them, and they are not people.
TOOL_NAMES = (
    "Jenkins", "Jackson", "Hudson", "Kafka", "Sphinx", "Ansible", "Django",
    "Gradle", "Celery", "Pandas", "Hugo", "Jasmine", "Mocha", "Karma", "Helm",
)
IDENTIFIER_PARTS = (
    "Account", "Address", "User", "Name", "Profile", "Session", "Billing",
    "Handler", "Resolver", "Factory", "Service", "Manager", "Serializer",
    "Validator", "Repository", "Controller", "Mapper", "Builder", "Client",
)
# fmt: on

# The types this corpus labels; the label map is the identity on them.
NER_TYPES = ("PERSON", "ADDRESS", "DATE_OF_BIRTH", "USERNAME", "ACCOUNT_NUMBER")
STRUCTURED_TYPES = ("EMAIL", "PHONE")
LABELS = (*NER_TYPES, *STRUCTURED_TYPES)

# A slot ${NAME} in a template: a gold value of that label, or (lowercase
# slot names) filler that carries no gold.
_SLOT_RE = re.compile(r"\$\{([A-Za-z_]+)\}")

SAMPLES = 1200

_HEX = "0123456789abcdef"


def person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def address(rng: random.Random) -> str:
    return f"{rng.randrange(2, 9999)} {rng.choice(STREETS)} {rng.choice(STREET_SUFFIXES)}"


def date_of_birth(rng: random.Random) -> str:
    year, month, day = rng.randrange(1940, 2006), rng.randrange(1, 13), rng.randrange(1, 29)
    return rng.choice(
        (
            f"{MONTHS[month - 1]} {day}, {year}",
            f"{year}-{month:02d}-{day:02d}",
            f"{month:02d}/{day:02d}/{year}",
            f"{day} {MONTHS[month - 1]} {year}",
        )
    )


def username(rng: random.Random) -> str:
    first, last = rng.choice(FIRST_NAMES).lower(), rng.choice(LAST_NAMES).lower()
    return rng.choice(
        (
            f"{first[0]}{last}{rng.randrange(1, 99)}",
            f"{first}.{last[0]}",
            f"{first}_{last}",
            f"dev_{first}{rng.randrange(10, 99)}",
        )
    )


def account_number(rng: random.Random) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(rng.randrange(8, 13)))


def email(rng: random.Random) -> str:
    first, last = rng.choice(FIRST_NAMES).lower(), rng.choice(LAST_NAMES).lower()
    return f"{first}.{last}@{rng.choice(('example.com', 'example.org', 'mail.example'))}"


def phone(rng: random.Random) -> str:
    return VALUE_GENERATORS["phone_number"][1](rng)


def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_HEX) for _ in range(n))


def _uuid(rng: random.Random) -> str:
    h = _hex(rng, 32)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _timestamp(rng: random.Random) -> str:
    return (
        f"2025-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}T"
        f"{rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}Z"
    )


def _identifier(rng: random.Random) -> str:
    parts = rng.sample(IDENTIFIER_PARTS, 2)
    return rng.choice((rng.choice(TOOL_NAMES), "")) + "".join(parts)


GOLD: dict[str, Callable[[random.Random], str]] = {
    "PERSON": person,
    "ADDRESS": address,
    "DATE_OF_BIRTH": date_of_birth,
    "USERNAME": username,
    "ACCOUNT_NUMBER": account_number,
    "EMAIL": email,
    "PHONE": phone,
}
FILLER: dict[str, Callable[[random.Random], str]] = {
    "n": lambda rng: str(rng.randrange(1, 5000)),
    "hash": lambda rng: _hex(rng, 40),
    "short_hash": lambda rng: _hex(rng, 7),
    "uuid": _uuid,
    "ts": _timestamp,
    "ident": _identifier,
    "tool": lambda rng: rng.choice(TOOL_NAMES),
    "ms": lambda rng: str(rng.randrange(3, 900)),
}

TEMPLATES: dict[str, tuple[str, ...]] = {
    "prose": (
        "Please send the signed lease to ${PERSON} at ${ADDRESS} before Friday.",
        "${PERSON} was born on ${DATE_OF_BIRTH} and moved to ${ADDRESS} last spring.",
        "The refund goes to account ${ACCOUNT_NUMBER}, held by ${PERSON}.",
        "Our new contractor, ${PERSON}, will use the login ${USERNAME} from Monday.",
        "If ${PERSON} cannot be reached on ${PHONE}, write to ${EMAIL} instead.",
    ),
    "chat": (
        "user: hi, I'm ${PERSON}, my date of birth is ${DATE_OF_BIRTH}. can you fill"
        " in the insurance form?\nassistant: Sure, which policy number?",
        "user: update the shipping address for order ${n} to ${ADDRESS}\n"
        "assistant: Done. Anything else?",
        "user: my username is ${USERNAME} and I can't log in since the ${tool} upgrade",
        "user: wire it to account ${ACCOUNT_NUMBER} please, reference ${n}",
        "user: draft a reply to ${PERSON} (${EMAIL}) declining the meeting",
    ),
    "json": (
        '{"customer": {"name": "${PERSON}", "address": "${ADDRESS}", "dob":'
        ' "${DATE_OF_BIRTH}"}, "id": "${uuid}"}',
        '{"user": {"login": "${USERNAME}", "display_name": "${PERSON}", "email":'
        ' "${EMAIL}"}, "created_at": "${ts}"}',
        '{"payee": "${PERSON}", "account_number": "${ACCOUNT_NUMBER}", "amount": ${n}}',
        '{"results": [{"name": "${PERSON}", "phone": "${PHONE}", "score": 0.${n}}]}',
    ),
    "code": (
        "# Reported by ${PERSON}; reproduces only for account ${ACCOUNT_NUMBER}.\n"
        "def test_${ident}():\n    assert parse(raw) is not None\n",
        "// TODO: ask ${PERSON} why ${ident} rejects ${ADDRESS}\nconst handler = new ${ident}();\n",
        '    # fixture user, see ticket ${n}: login "${USERNAME}"\n'
        "    user = make_user(birth_date=DOB)  # ${DATE_OF_BIRTH}\n",
        "/* maintainer: ${PERSON} <${EMAIL}> */\nstatic int retries = ${n};\n",
    ),
    "log": (
        "${ts} INFO auth: login succeeded user=${USERNAME} session=${uuid}",
        '${ts} WARN billing: charge declined for "${PERSON}" account=${ACCOUNT_NUMBER}',
        "${ts} INFO shipping: label printed for ${ADDRESS} in ${ms}ms",
        "${ts} DEBUG kyc: verified dob=${DATE_OF_BIRTH} for ${PERSON}",
    ),
    "git": (
        "commit ${hash}\nAuthor: ${PERSON} <${EMAIL}>\nDate:   Tue Mar 4 10:12:01"
        " 2025 +0100\n\n    Fix address parsing for ${ADDRESS}\n",
        "${short_hash} Merge pull request #${n} from ${USERNAME}/fix-${ident}\n"
        "${short_hash} Bump ${tool} to 2.${n}.0\n",
        "    Co-authored-by: ${PERSON} <${EMAIL}>\n    Reviewed-by: ${PERSON}\n",
    ),
    "neg-uuid": (
        "request ${uuid} retried; parent span ${uuid}",
        '{"trace_id": "${uuid}", "span_id": "${short_hash}", "sampled": true}',
    ),
    "neg-hash": (
        "commit ${hash}\nMerge: ${short_hash} ${short_hash}\n\n    Bump ${tool} plugin\n",
        "${short_hash} (HEAD -> main, origin/main) Update ${ident} tests",
    ),
    "neg-identifier": (
        "class ${ident}(${ident}):\n    def resolve(self, ctx): return self.${tool}",
        "error: cannot find symbol ${ident} in ${ident}.java (${tool} build ${n})",
        "${tool} pipeline #${n} passed: ${ident}, ${ident}, ${ident}",
    ),
    "neg-path": (
        "M src/billing/${ident}.py\nA tests/unit/test_${ident}.py\nD docs/${tool}.md",
        "/home/runner/work/app/app/node_modules/${tool}/lib/${ident}.js",
    ),
    "neg-traceback": (
        "Traceback (most recent call last):\n"
        '  File "/srv/app/handlers.py", line ${n}, in handle\n'
        "    return ${ident}().run(payload)\n"
        "KeyError: 'account_number'\n",
        "at com.example.${ident}.process(${ident}.java:${n})\n"
        "at org.${tool}.core.Runner.run(Runner.java:${n})",
    ),
    "neg-timestamp": (
        "${ts} ${ts} build finished in ${ms}ms",
        "[${ts}] ${tool} worker ${n} heartbeat ok",
    ),
}
CONTEXTS = tuple(TEMPLATES)


def fill(template: str, rng: random.Random) -> NerSample:
    """One sample from a template: every ``${LABEL}`` slot gets a gold
    value of that label, every lowercase slot filler without gold."""
    parts: list[str] = []
    spans: list[GoldSpan] = []
    size = 0
    last = 0
    for match in _SLOT_RE.finditer(template):
        literal = template[last : match.start()]
        parts.append(literal)
        size += len(literal)
        slot = match.group(1)
        gold = GOLD.get(slot)
        value = gold(rng) if gold is not None else FILLER[slot](rng)
        if gold is not None:
            spans.append(GoldSpan(size, size + len(value), slot))
        parts.append(value)
        size += len(value)
        last = match.end()
    parts.append(template[last:])
    return NerSample("".join(parts), tuple(spans))


def generate(seed: int = 42, samples: int = SAMPLES) -> list[NerSample]:
    """``samples`` samples; contexts take turns, so each appears once
    ``samples`` reaches their number. Same seed, same corpus."""
    rng = random.Random(seed)
    corpus = []
    for index in range(samples):
        context = CONTEXTS[index % len(CONTEXTS)]
        sample = fill(rng.choice(TEMPLATES[context]), rng)
        corpus.append(NerSample(sample.text, sample.spans, context))
    return corpus
