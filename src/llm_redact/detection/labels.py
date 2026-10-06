"""NER label identity: one placeholder type per value, whichever backend found it.

The vault is keyed on (session, type, value), so two names for one type issue
two tokens for one value: a spaCy ``PERSON`` and an hf ``PER`` on the same name
would become «PERSON_001» and «PER_001». Every NER backend therefore turns its
model's labels into placeholder types through one :class:`LabelPolicy`:

* :func:`normalize_label` gives every label one spelling (``"B-first_name"`` ->
  ``FIRST_NAME``, ``"street address"`` -> ``STREET_ADDRESS``);
* :data:`DEFAULT_FOLDS` folds synonyms into one placeholder type (``PER``,
  ``FIRST_NAME``, ``SURNAME`` ... -> ``PERSON``; ``EMAIL_ADDRESS`` -> ``EMAIL``);
* each configured entity is a TYPE REQUEST (its normalized form is a
  placeholder type: :data:`TYPE_NAMES`) or a RAW REQUEST (anything else, such as
  ``PER`` or ``"job title"``), and a model label is kept only when its type was
  requested.

Release path of raw requests (owner decisions D1 (a), D15 (i)): with
:data:`FOLD_RAW_REQUESTS` false (1.12.x) a raw request still requests and
emits its own normalized label, exactly as before (``entities = ["PER"]``
emits ``PER``; Presidio's five legacy folds still apply on the presidio
backend); with it true (2.0.0) raw requests fold too (``PER`` -> ``PERSON``).
Both modes are implemented and tested.

A type that does not fit the placeholder grammar is never emitted
(:func:`llm_redact.placeholders.is_placeholder_type`). Labels describing a
sensitive attribute (D10, :data:`SENSITIVE_LABELS`) are never folded: they are
detected only when listed raw. Nothing here logs; callers log types, labels
and counts only, never detected text.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from llm_redact.detection.base import Detection
from llm_redact.detection.regex_rules import BUILTIN_RULES
from llm_redact.placeholders import is_placeholder_type

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig
    from llm_redact.detection.stats import NerStats

# D15 (i): 1.12.x ships the transitional mode (raw requests keep their own
# type, with a startup deprecation warning); 2.0.0 flips this to True.
FOLD_RAW_REQUESTS = False

# Contextual PII types the NER models add beside the built-in rule types.
CANONICAL_NER_TYPES = (
    "PERSON",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "PASSPORT",
    "DRIVER_LICENSE",
    "USERNAME",
    "ACCOUNT_NUMBER",
)

# Every placeholder type an entity can request by name: the canonical NER
# types and every built-in rule's type.
TYPE_NAMES: frozenset[str] = frozenset(CANONICAL_NER_TYPES) | frozenset(
    rule.detector_type for rule in BUILTIN_RULES
)

# Placeholder type -> the normalized model labels that fold into it.
_FOLD_TABLE: dict[str, tuple[str, ...]] = {
    "PERSON": (
        "PER",
        "PERSON",
        "NAME",
        "FULL_NAME",
        "FIRST_NAME",
        "FIRSTNAME",
        "GIVENNAME",
        "GIVEN_NAME",
        "MIDDLE_NAME",
        "MIDDLENAME",
        "LAST_NAME",
        "LASTNAME",
        "SURNAME",
        "FAMILY_NAME",
        "PRIVATE_PERSON",
    ),
    "ADDRESS": (
        "ADDRESS",
        "STREET_ADDRESS",
        "STREETADDRESS",
        "STREET",
        "LOCATION_STREET",
        "LOCATION_ADDRESS",
        "BUILDINGNUM",
        "BUILDING_NUMBER",
        "BUILDINGNUMBER",
        "PRIVATE_ADDRESS",
    ),
    "DATE_OF_BIRTH": ("DATE_OF_BIRTH", "DATEOFBIRTH", "DOB", "BIRTH_DATE", "BIRTHDATE"),
    "PASSPORT": ("PASSPORT", "PASSPORT_NUMBER", "PASSPORTNUM", "PASSPORTNUMBER"),
    "DRIVER_LICENSE": (
        "DRIVER_LICENSE",
        "DRIVERS_LICENSE",
        "DRIVER_LICENSE_NUMBER",
        "DRIVERS_LICENSE_NUMBER",
        "DRIVERLICENSENUM",
        "DRIVER_LICENCE",
    ),
    "USERNAME": ("USERNAME", "USER_NAME"),
    "ACCOUNT_NUMBER": ("ACCOUNT_NUMBER", "ACCOUNTNUM", "BANK_ACCOUNT", "BANK_ACCOUNT_NUMBER"),
    "EMAIL": ("EMAIL", "EMAIL_ADDRESS", "PRIVATE_EMAIL"),
    "PHONE": ("PHONE", "PHONE_NUMBER", "TELEPHONE", "TELEPHONENUM", "PRIVATE_PHONE"),
    "SSN": ("SSN", "US_SSN"),
    "IBAN": ("IBAN", "IBAN_CODE"),
    "CREDIT_CARD": (
        "CREDIT_CARD",
        "CREDIT_CARD_NUMBER",
        "CREDITCARDNUMBER",
        "CREDIT_DEBIT_CARD",
        "CARD_NUMBER",
        "PAYMENT_CARD",
    ),
    "IPV4": ("IPV4",),
    "IPV6": ("IPV6",),
    # The type of the built-in generic_secret rule: disabling that rule also
    # suppresses model PASSWORD detections (rule toggles are per TYPE).
    "SECRET": ("PASSWORD", "SECRET", "API_KEY", "ACCESS_TOKEN"),
}

# Normalized model label -> placeholder type.
DEFAULT_FOLDS: Mapping[str, str] = MappingProxyType(
    {label: type_name for type_name, labels in _FOLD_TABLE.items() for label in labels}
)

# The five folds the presidio backend has always applied (a subset of
# DEFAULT_FOLDS). On their own they matter only while raw requests do not
# fold: there they still apply on the presidio backend, and on no other.
LEGACY_FOLDS: Mapping[str, str] = MappingProxyType(
    {
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "US_SSN": "SSN",
        "IBAN_CODE": "IBAN",
        "CREDIT_CARD": "CREDIT_CARD",
    }
)

# Deliberately NOT folded: kept only when listed raw in `entities`. IP_ADDRESS
# covers v4 and v6 (either built-in name would mislabel the other); place
# names, plain dates and organisations are far broader than the PII types;
# the national-id labels are country-ambiguous (the national-id rules own
# those types).
UNFOLDED_LABELS = frozenset(
    {
        "IP_ADDRESS",
        "LOC",
        "LOCATION",
        "GPE",
        "CITY",
        "STATE",
        "COUNTRY",
        "ZIPCODE",
        "POSTCODE",
        "DATE",
        "PRIVATE_DATE",
        "TIME",
        "ORG",
        "COMPANY_NAME",
        "PRIVATE_URL",
        "URL",
        "SOCIALNUM",
        "TAXNUM",
        "IDCARDNUM",
    }
)

# D10: labels describing a sensitive attribute are never folded and never
# part of a recommended entity list; listing one raw is the only way in.
SENSITIVE_LABELS = frozenset(
    {
        "RACE",
        "ETHNICITY",
        "RACE_ETHNICITY",
        "RACIAL_OR_ETHNIC_ORIGIN",
        "RELIGION",
        "RELIGIOUS_BELIEF",
        "RELIGIOUS_BELIEFS",
        "POLITICAL_VIEW",
        "POLITICAL_VIEWS",
        "POLITICAL_OPINION",
        "POLITICAL_AFFILIATION",
        "SEXUALITY",
        "SEXUAL_ORIENTATION",
        "GENDER",
        "SEX",
        "NRP",
    }
)

# The zero-shot backends: prompted with label TEXT, they return the prompts
# as labels (gliner_prompt, LabelPolicy.prompts and classify_gliner).
ZERO_SHOT_BACKENDS = frozenset({"gliner", "gliner2"})

# GLiNER is zero-shot: it is prompted with label TEXT. A type request sends
# a natural-language prompt (a model reads "street address" better than
# "ADDRESS"); built-in types without an entry send their name lowercased
# with "_" -> " ".
GLINER_PROMPTS: Mapping[str, str] = MappingProxyType(
    {
        "PERSON": "person",
        "ADDRESS": "street address",
        "DATE_OF_BIRTH": "date of birth",
        "PASSPORT": "passport number",
        "DRIVER_LICENSE": "driver license number",
        "USERNAME": "username",
        "ACCOUNT_NUMBER": "account number",
        "EMAIL": "email address",
        "PHONE": "phone number",
    }
)

# One leading BIO/BIOES/BILOU tag ("B-", "I-", "E-", "S-", "L-", "U-"),
# dropped only when a letter follows ("I-PER" -> PER; "B-2" stays).
_TAG_PREFIX_RE = re.compile(r"[BIESLUbieslu]-(?=[^\W\d_])")
_NOT_TYPE_CHAR_RE = re.compile(r"[^A-Z0-9]+")


def normalize_label(raw: str) -> str:
    """One spelling for a model label or a configured entity.

    Strip whitespace, drop one leading tagging prefix when a letter follows,
    uppercase, replace each run of characters outside ``[A-Z0-9]`` with
    ``_`` and strip leading and trailing ``_``.
    """
    label = raw.strip()
    if _TAG_PREFIX_RE.match(label):
        label = label[2:]
    return _NOT_TYPE_CHAR_RE.sub("_", label.upper()).strip("_")


def gliner_prompt(type_name: str) -> str:
    """The prompt a type request sends to a GLiNER model."""
    prompt = GLINER_PROMPTS.get(type_name)
    return prompt if prompt is not None else type_name.lower().replace("_", " ")


class LabelPolicy:
    """How one backend turns model labels into placeholder types.

    Built once per backend from the configured ``entities`` (and, from the
    ``[detection.ner.labels]`` table, ``overrides``: normalized label ->
    placeholder type, ``""`` dropping the label). ``classify`` is the whole
    decision for one model label; it never sees the detected text.
    """

    __slots__ = (
        "backend",
        "entities",
        "fold_raw",
        "overrides",
        "prompt_types",
        "prompts",
        "raw_requested",
        "requested",
    )

    def __init__(
        self,
        entities: Iterable[str],
        *,
        backend: str = "",
        overrides: Mapping[str, str] | Iterable[tuple[str, str]] = (),
        fold_raw: bool | None = None,
    ) -> None:
        self.backend = backend
        self.entities = tuple(entities)
        self.fold_raw = FOLD_RAW_REQUESTS if fold_raw is None else fold_raw
        pairs = overrides.items() if isinstance(overrides, Mapping) else overrides
        self.overrides: Mapping[str, str] = MappingProxyType(
            {normalize_label(label): type_name for label, type_name in pairs}
        )
        requested: set[str] = set()
        raw_requested: set[str] = set()
        # GLiNER prompts: a type request's natural-language prompt first,
        # then each raw request verbatim — each prompt once, and a raw
        # request equal to a type request's prompt is not sent (the type
        # request wins: "phone number" beside PHONE comes back as PHONE).
        prompt_types: dict[str, str] = {}
        raw_prompts: dict[str, None] = {}
        for entity in self.entities:
            label = normalize_label(entity)
            if label in TYPE_NAMES:
                type_name = self.fold(label)
                if type_name:
                    requested.add(type_name)
                    prompt_types.setdefault(gliner_prompt(label), type_name)
                continue
            raw_prompts.setdefault(entity, None)
            if not self.fold_raw and label not in self.overrides:
                raw_requested.add(label)
                continue
            type_name = self.fold(label)
            if type_name:
                requested.add(type_name)
        sent_types = {normalize_label(prompt) for prompt in prompt_types}
        unsent = {raw for raw in raw_prompts if normalize_label(raw) in sent_types}
        if backend in ZERO_SHOT_BACKENDS:
            # What an unsent raw request's label comes back as is the type
            # request's type, so classify must not keep it raw either.
            raw_requested -= {normalize_label(raw) for raw in unsent}
        self.requested = frozenset(requested)
        self.raw_requested = frozenset(raw_requested)
        self.prompt_types: Mapping[str, str] = MappingProxyType(prompt_types)
        self.prompts = tuple(prompt_types) + tuple(raw for raw in raw_prompts if raw not in unsent)

    def fold(self, label: str) -> str:
        """Normalized label -> placeholder type ("" = dropped): the user's
        override, else the default fold, else the label itself."""
        override = self.overrides.get(label)
        if override is not None:
            return override
        return DEFAULT_FOLDS.get(label, label)

    def classify(self, raw_label: str, stats: "NerStats | None" = None) -> str | None:
        """The placeholder type a model label is emitted as, or None when
        it was not requested, is dropped, or does not fit the placeholder
        grammar — the last counted in ``stats.labels_dropped`` when a
        backend passes its counters."""
        label = normalize_label(raw_label)
        if label in self.raw_requested:
            # Transitional mode: a raw request keeps its own label (the
            # presidio backend's five legacy folds still apply).
            presidio = self.backend == "presidio"
            type_name = LEGACY_FOLDS.get(label, label) if presidio else label
        else:
            type_name = self.fold(label)
            if type_name not in self.requested:
                return None
        return _fitting(type_name, stats)

    def entry_type(self, name: str) -> str | None:
        """The placeholder type a configured NAME stands for (an entity, an
        `[detection.allowlist_by_type]` key): its override, else — once raw
        names fold — its fold, else its normalized self. None when the name
        is dropped or cannot be a placeholder type."""
        label = normalize_label(name)
        type_name = self.fold(label) if self.fold_raw else self.overrides.get(label, label)
        return type_name if type_name and is_placeholder_type(type_name) else None

    def classify_gliner(self, returned_label: str, stats: "NerStats | None" = None) -> str | None:
        """A GLiNER label: one of this policy's type-request prompts is
        emitted as its type directly; anything else goes through classify."""
        type_name = self.prompt_types.get(returned_label)
        if type_name is not None:
            return _fitting(type_name, stats)
        return self.classify(returned_label, stats)


def _fitting(type_name: str, stats: "NerStats | None") -> str | None:
    """``type_name`` when it fits the placeholder grammar; otherwise None,
    counted as a dropped label when ``stats`` is given."""
    if is_placeholder_type(type_name):
        return type_name
    if stats is not None:
        stats.labels_dropped += 1
    return None


# Types whose parts a model may report separately ("Jane" FIRST_NAME, "Doe"
# LAST_NAME -> two PERSON spans) and that merge_adjacent_parts joins.
MERGED_TYPES = frozenset({"PERSON", "ADDRESS"})
# What may separate two parts of one value: one or two of these characters.
# Never a newline, punctuation, a quote, a comma or a JSON delimiter.
_PART_GAP_CHARS = frozenset(" \t\u00a0")


def merge_adjacent_parts(detections: Iterable[Detection], text: str) -> list[Detection]:
    """One backend's detections with adjacent parts of one name or address
    joined: consecutive detections of the same type in MERGED_TYPES whose
    gap is 1-2 characters, each a space, tab or no-break space, become one
    span whose value is ``text[start:end]`` — so "Jane Doe" is one PERSON
    token, not two. Every other detection is kept as it is."""
    merged: list[Detection] = []
    last_of_type: dict[str, int] = {}
    for detection in sorted(detections, key=lambda d: (d.start, d.end)):
        index = last_of_type.get(detection.detector_type)
        if index is not None:
            previous = merged[index]
            gap = text[previous.end : detection.start]
            if 1 <= len(gap) <= 2 and set(gap) <= _PART_GAP_CHARS:
                merged[index] = replace(
                    previous, end=detection.end, value=text[previous.start : detection.end]
                )
                continue
        if detection.detector_type in MERGED_TYPES:
            last_of_type[detection.detector_type] = len(merged)
        merged.append(detection)
    return merged


def raw_entity_deprecations(ner: "NerConfig") -> list[str]:
    """One deprecation message per configured raw entity whose emitted type
    changes when raw entities start folding in 2.0.0 (D15 (i)), on any
    active backend: ``PER`` (emitted as PER now, PERSON then), but not
    Presidio's ``EMAIL_ADDRESS`` (already emitted as EMAIL there), a type
    request, or an entity with a [detection.ner.labels] override. Empty
    once raw entities fold. Config names only, never detected text."""
    if FOLD_RAW_REQUESTS:
        return []
    messages: list[str] = []
    for backend in ner.active_backends():
        now = LabelPolicy(ner.entities, backend=backend, overrides=ner.labels, fold_raw=False)
        then = LabelPolicy(ner.entities, backend=backend, overrides=ner.labels, fold_raw=True)
        for entity in ner.entities:
            current, future = now.classify(entity), then.classify(entity)
            if current == future:  # a type request, an override, a legacy fold
                continue
            message = (
                f'[detection.ner] entities: "{entity}" is emitted as {current} now and as'
                f' {future} from 2.0.0; write "{future}" to switch now, or set'
                f' [detection.ner.labels] {normalize_label(entity)} = "{current}" to keep'
                f" {current}"
            )
            if message not in messages:
                messages.append(message)
    return messages
