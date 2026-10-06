"""The prompt catalog of the generator (README.md lists it).

Each prompt asks the teacher for one coding-agent artifact — tool-call JSON
arguments and results, diffs, logs, config files, commit messages, test
fixtures, SQL, chat turns — with INVENTED personal data tagged inline, or,
for a hard negative, the same kind of artifact with none at all but with
name-like identifiers that are not people. Everything a prompt varies (the
artifact, scenario, format, types) is drawn from a ``random.Random`` seeded
by the run seed and the row index, so a run is reproducible from
(teacher, seed, index) given a deterministic server (temperature 0 and the
same seed on every request).
"""

import hashlib
import json
import random
from dataclasses import dataclass

# Bump when a prompt's wording changes: rows record the catalog through the
# run manifest's catalog_version and catalog_sha256.
CATALOG_VERSION = 1

# The types the generator asks for: the canonical NER types and, for the
# bench's structured-regression check, two the regex rules own.
TYPE_HINTS: dict[str, str] = {
    "PERSON": "a person's name (a full, first or last name)",
    "ADDRESS": "a street address (house number and street, optionally unit, city and"
    " postcode, as one value)",
    "DATE_OF_BIRTH": "a person's date of birth (any common date format)",
    "PASSPORT": "a passport number",
    "DRIVER_LICENSE": "a driver's license number",
    "USERNAME": "a person's login name or user handle",
    "ACCOUNT_NUMBER": "a bank or customer account number",
    "EMAIL": "an email address",
    "PHONE": "a phone number",
}
TYPES = tuple(TYPE_HINTS)


@dataclass(frozen=True)
class Artifact:
    id: str
    description: str
    formats: tuple[str, ...]


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        "tool-args",
        "the JSON arguments of one tool call a coding agent makes (for example creating a"
        " contact, filing a ticket, updating a database row or sending a message)",
        ("JSON",),
    ),
    Artifact(
        "tool-result",
        "the JSON result a tool returns to a coding agent (a database query, an API"
        " response, a search result or a file listing)",
        ("JSON",),
    ),
    Artifact(
        "diff",
        "a unified git diff that changes a source file, a test fixture or a seed-data file",
        ("Python", "TypeScript", "Go", "Java", "Ruby", "YAML", "SQL"),
    ),
    Artifact(
        "log",
        "application or CI log lines with timestamps, levels and request ids",
        ("plain text logs", "JSON lines", "nginx access log", "GitHub Actions log"),
    ),
    Artifact(
        "config",
        "a configuration file of a service",
        ("YAML", "TOML", ".env", "JSON", "INI"),
    ),
    Artifact(
        "commit",
        "a git commit message with author and trailer lines (Signed-off-by, Co-authored-by)",
        ("git log output", "a commit message body"),
    ),
    Artifact(
        "test-fixture",
        "a unit test with fixture data",
        ("Python pytest", "TypeScript jest", "Go testing", "Java JUnit", "Ruby RSpec"),
    ),
    Artifact(
        "sql",
        "SQL statements or the text output of a query",
        ("PostgreSQL", "MySQL", "SQLite"),
    ),
    Artifact(
        "chat",
        "a user's message to a coding assistant asking for help and pasting the data involved",
        ("plain prose", "prose with a pasted code block"),
    ),
    Artifact(
        "traceback",
        "an error report: an exception traceback with the local variables or the request"
        " that failed",
        ("Python", "Java", "Node.js"),
    ),
)

SCENARIOS = (
    "a CRM migration script",
    "an HR onboarding service",
    "a support-ticket triage bot",
    "a billing reconciliation job",
    "a shipping-label service",
    "a clinic appointment scheduler",
    "a school enrolment system",
    "a bank statement importer",
    "a car-rental booking API",
    "a mailing-list export",
    "a hotel check-in kiosk",
    "a payroll report generator",
)

# Identifiers that look like names but are not people (hard negatives).
CONFUSERS = (
    "tool names that are also surnames (Jenkins, Hudson, Jackson, Kafka)",
    "CamelCase class names (UserAccountService, CustomerAddressMapper)",
    "UUIDs, commit hashes and request ids",
    "file paths and module names",
    "dummy field names (first_name, billing_address, user_id) with no values",
    "timestamps, version numbers and port numbers",
)

SYSTEM = (
    "You write realistic artifacts that a coding agent sends to or receives from a"
    " language model. Every personal value you write must be INVENTED: never a real"
    " person, a real address or a real account. Use names from many cultures.\n"
    'Wrap EVERY personal value, every time it occurs, in <pii type="TYPE">value</pii>,'
    " with one of these types:\n"
    + "".join(f"- {name}: {hint}\n" for name, hint in TYPE_HINTS.items())
    + "Never tag anything else: not company or product names, a city or country alone,"
    " tool names, code identifiers, hashes, ids or timestamps. Tags never nest.\n"
    "Answer with the artifact only: no explanation and no code fence."
)


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    user: str
    negative: bool
    # The types a positive prompt asked for (a hard negative: none).
    types: tuple[str, ...]


def build(seed: int, index: int, negatives: float) -> Prompt:
    """Prompt number ``index`` of a run: a hard negative with probability
    ``negatives``, else a positive asking for two to four types."""
    rng = random.Random(f"{seed}:{index}")
    negative = rng.random() < negatives
    artifact = rng.choice(ARTIFACTS)
    scenario = rng.choice(SCENARIOS)
    fmt = rng.choice(artifact.formats)
    head = f"Write {artifact.description}.\nContext: {scenario}.\nFormat: {fmt}.\n"
    if negative:
        confusers = rng.sample(CONFUSERS, 2)
        user = (
            head + "It must contain NO personal data at all, so it carries no tags. Make it"
            f" realistic with {confusers[0]} and {confusers[1]}.\nKeep it under 40 lines."
        )
        return Prompt(f"{artifact.id}.negative", user, True, ())
    types = tuple(sorted(rng.sample(TYPES, rng.randint(2, 4))))
    user = (
        head + f"Include each of these kinds of personal data at least once: {', '.join(types)}."
        "\nKeep it under 40 lines."
    )
    return Prompt(artifact.id, user, False, types)


def catalog_sha256() -> str:
    """A digest of everything the prompts are built from (recorded in the
    run manifest, so a changed catalog is visible)."""
    catalog = {
        "version": CATALOG_VERSION,
        "system": SYSTEM,
        "artifacts": [[a.id, a.description, list(a.formats)] for a in ARTIFACTS],
        "scenarios": list(SCENARIOS),
        "confusers": list(CONFUSERS),
    }
    encoded = json.dumps(catalog, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
