"""The NER label policy (detection/labels.py): one placeholder type per value
across every backend, in both raw-entity modes (FOLD_RAW_REQUESTS False =
1.12.x, True = 2.0.0; the ``fold_raw`` fixture runs a test in each)."""

from pathlib import Path

import pytest

from llm_redact.detection import labels
from llm_redact.detection.engine import Allowlist, DetectionConfig, NerConfig, build_detectors
from llm_redact.detection.gliner_ner import GlinerDetector
from llm_redact.detection.hf_ner import HfDetector
from llm_redact.detection.labels import (
    CANONICAL_NER_TYPES,
    DEFAULT_FOLDS,
    GLINER_PROMPTS,
    LEGACY_FOLDS,
    SENSITIVE_LABELS,
    TYPE_NAMES,
    UNFOLDED_LABELS,
    LabelPolicy,
    gliner_prompt,
    normalize_label,
)
from llm_redact.detection.ner import NerDetector
from llm_redact.detection.regex_rules import BUILTIN_RULES
from llm_redact.placeholders import MAX_TYPE_NAME_LEN
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault
from ner_fakes import (
    FakeAnalyzer,
    FakeGliner,
    FakeHfPipe,
    FakeSpacy,
    install_gliner,
    install_presidio,
    install_spacy,
    install_stanza,
    install_transformers,
)

# --- normalization ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("street address", "STREET_ADDRESS"),
        ("B-first_name", "FIRST_NAME"),
        ("private_person", "PRIVATE_PERSON"),
        ("  I-PER ", "PER"),
        ("i-per", "PER"),
        ("E-LOC", "LOC"),
        ("S-ORG", "ORG"),
        ("L-PER", "PER"),
        ("U-PER", "PER"),
        ("B-I-PER", "I_PER"),  # one prefix only
        ("B-2nd", "B_2ND"),  # kept: no letter after the dash
        ("X-PER", "X_PER"),  # not a tagging prefix
        ("PER", "PER"),
        ("phone  number", "PHONE_NUMBER"),
        ("__job.title!!", "JOB_TITLE"),
        ("Straße", "STRASSE"),
        ("", ""),
        ("---", ""),
    ],
)
def test_normalize_label(raw: str, normalized: str) -> None:
    assert normalize_label(raw) == normalized


# --- the fold table -------------------------------------------------------------


def test_type_names_are_canonical_plus_builtin_types() -> None:
    assert frozenset(CANONICAL_NER_TYPES) <= TYPE_NAMES
    assert {rule.detector_type for rule in BUILTIN_RULES} <= TYPE_NAMES
    assert len(TYPE_NAMES) == len(CANONICAL_NER_TYPES) + len(
        {rule.detector_type for rule in BUILTIN_RULES}
    )


def test_every_type_name_requests_itself() -> None:
    # A type request asks for exactly its own type: no placeholder type
    # folds into another one.
    policy = LabelPolicy(())
    assert all(policy.fold(name) == name for name in TYPE_NAMES)


def test_folds_target_placeholder_types_and_are_normalized() -> None:
    assert set(DEFAULT_FOLDS.values()) <= TYPE_NAMES
    assert all(normalize_label(label) == label for label in DEFAULT_FOLDS)


def test_legacy_folds_are_the_five_presidio_folds_inside_the_defaults() -> None:
    assert dict(LEGACY_FOLDS) == {
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "US_SSN": "SSN",
        "IBAN_CODE": "IBAN",
        "CREDIT_CARD": "CREDIT_CARD",
    }
    assert all(DEFAULT_FOLDS[label] == type_name for label, type_name in LEGACY_FOLDS.items())


def test_broad_and_sensitive_labels_are_never_folded() -> None:
    # D10: a sensitive attribute is detected only when listed raw.
    assert not (UNFOLDED_LABELS | SENSITIVE_LABELS) & set(DEFAULT_FOLDS)
    assert not SENSITIVE_LABELS & TYPE_NAMES
    assert all(normalize_label(label) == label for label in UNFOLDED_LABELS | SENSITIVE_LABELS)


def test_canonical_types_fit_the_placeholder_grammar() -> None:
    from llm_redact.placeholders import is_placeholder_type

    assert all(is_placeholder_type(name) for name in TYPE_NAMES)


def test_gliner_prompts() -> None:
    assert set(GLINER_PROMPTS) <= TYPE_NAMES
    assert gliner_prompt("ADDRESS") == "street address"
    assert gliner_prompt("CREDIT_CARD") == "credit card"
    assert gliner_prompt("GITHUB_TOKEN") == "github token"
    # Each prompt reads back (normalized) as a label that folds into its type.
    policy = LabelPolicy(())
    assert all(policy.fold(normalize_label(gliner_prompt(t))) == t for t in TYPE_NAMES)


# --- classify, in both modes ----------------------------------------------------


@pytest.mark.parametrize(
    ("entities", "backend", "label", "kept", "folded"),
    [
        # Type requests fold every synonym in both modes.
        (("PERSON",), "hf", "PER", "PERSON", "PERSON"),
        (("PERSON",), "hf", "B-PER", "PERSON", "PERSON"),
        (("PERSON",), "hf", "first_name", "PERSON", "PERSON"),
        (("PERSON",), "hf", "ORG", None, None),
        (("person",), "spacy", "PERSON", "PERSON", "PERSON"),
        (("EMAIL",), "hf", "EMAIL_ADDRESS", "EMAIL", "EMAIL"),
        (("ADDRESS",), "hf", "street_address", "ADDRESS", "ADDRESS"),
        (("SECRET",), "hf", "password", "SECRET", "SECRET"),
        # Raw requests keep their own type until they fold.
        (("PER",), "hf", "PER", "PER", "PERSON"),
        (("PER",), "hf", "I-PER", "PER", "PERSON"),
        (("PER",), "spacy", "PERSON", None, "PERSON"),
        (("phone number",), "gliner", "phone number", "PHONE_NUMBER", "PHONE"),
        (("EMAIL_ADDRESS",), "hf", "EMAIL_ADDRESS", "EMAIL_ADDRESS", "EMAIL"),
        # ...except Presidio's five legacy folds, on the presidio backend.
        (("EMAIL_ADDRESS",), "presidio", "EMAIL_ADDRESS", "EMAIL", "EMAIL"),
        (("US_SSN",), "presidio", "US_SSN", "SSN", "SSN"),
        (("IP_ADDRESS",), "presidio", "IP_ADDRESS", "IP_ADDRESS", "IP_ADDRESS"),
        # Unfolded labels stay themselves.
        (("ORG",), "spacy", "ORG", "ORG", "ORG"),
        (("job title",), "gliner", "job title", "JOB_TITLE", "JOB_TITLE"),
        (("RELIGION",), "hf", "religion", "RELIGION", "RELIGION"),
        (("PERSON",), "hf", "religion", None, None),
        # Not requested at all.
        ((), "hf", "PER", None, None),
    ],
)
def test_classify(
    entities: tuple[str, ...],
    backend: str,
    label: str,
    kept: str | None,
    folded: str | None,
    fold_raw: bool,
) -> None:
    policy = LabelPolicy(entities, backend=backend)
    assert policy.fold_raw is fold_raw
    assert policy.classify(label) == (folded if fold_raw else kept)


def test_explicit_mode_beats_the_module_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(labels, "FOLD_RAW_REQUESTS", False)
    assert LabelPolicy(("PER",), fold_raw=True).classify("PER") == "PERSON"
    monkeypatch.setattr(labels, "FOLD_RAW_REQUESTS", True)
    assert LabelPolicy(("PER",), fold_raw=False).classify("PER") == "PER"


def test_the_shipped_mode_keeps_raw_entities() -> None:
    # D15 (i): 1.12.x warns; 2.0.0 (T46) flips this constant.
    assert labels.FOLD_RAW_REQUESTS is False


def test_requested_types(fold_raw: bool) -> None:
    policy = LabelPolicy(("PERSON", "PER", "job title", "email"))
    if fold_raw:
        assert policy.requested == {"PERSON", "JOB_TITLE", "EMAIL"}
        assert policy.raw_requested == frozenset()
    else:
        assert policy.requested == {"PERSON", "EMAIL"}
        assert policy.raw_requested == {"PER", "JOB_TITLE"}


@pytest.mark.parametrize(
    ("entity", "label"),
    [
        ("3d model", "3d model"),  # starts with a digit
        ("x" * (MAX_TYPE_NAME_LEN + 1), "x" * (MAX_TYPE_NAME_LEN + 1)),
        ("---", "---"),  # normalizes to nothing
    ],
)
def test_a_type_outside_the_placeholder_grammar_is_never_emitted(
    entity: str, label: str, fold_raw: bool
) -> None:
    assert LabelPolicy((entity,)).classify(label) is None


def test_the_longest_placeholder_type_is_emitted(fold_raw: bool) -> None:
    entity = "x" * MAX_TYPE_NAME_LEN
    assert LabelPolicy((entity,)).classify(entity) == entity.upper()


def test_overrides_are_keyed_by_normalized_label() -> None:
    # Overrides arrive normalized or not; both spellings land on one key.
    policy = LabelPolicy(("PERSON",), overrides={"first name": "PERSON"})
    assert dict(policy.overrides) == {"FIRST_NAME": "PERSON"}


# --- hf -------------------------------------------------------------------------


def test_default_hf_config_detects_person(monkeypatch: pytest.MonkeyPatch, fold_raw: bool) -> None:
    # The shipped default (`entities = ["PERSON"]`) against a PER-emitting
    # model (dslim/bert-base-NER) once detected nothing.
    install_transformers(monkeypatch, FakeHfPipe([("Jane Doe", "PER", 0.99)]))
    (detector,) = build_detectors(
        DetectionConfig(enabled=(), ner=NerConfig(enabled=True, backend="hf"))
    )
    assert [(d.detector_type, d.value) for d in detector.detect("ask Jane Doe")] == [
        ("PERSON", "Jane Doe")
    ]


def test_hf_raw_entity_per(monkeypatch: pytest.MonkeyPatch, fold_raw: bool) -> None:
    install_transformers(monkeypatch, FakeHfPipe([("Jane Doe", "PER", 0.99)]))
    (detector,) = build_detectors(
        DetectionConfig(enabled=(), ner=NerConfig(enabled=True, backend="hf", entities=("PER",)))
    )
    assert [d.detector_type for d in detector.detect("ask Jane Doe")] == [
        "PERSON" if fold_raw else "PER"
    ]


def test_spacy_person_and_hf_per_share_one_token(fold_raw: bool) -> None:
    # Two backends, two label names, one PERSON request: the vault sees one
    # (PERSON, value) identity, so one name gets one token.
    spacy_like = NerDetector(FakeSpacy([("Jane Doe", "PERSON", 1.0)]), frozenset({"PERSON"}), 1000)
    hf = HfDetector(
        FakeHfPipe([("Jane Doe", "PER", 0.9), ("Bob Roe", "PER", 0.9)]),
        frozenset({"PERSON"}),
        max_chars=1000,
        threshold=0.5,
    )
    vault = InMemoryVault()
    redactor = Redactor([spacy_like, hf], vault, Allowlist(exact=frozenset(), patterns=()))
    out = redactor.redact_text("Jane Doe met Bob Roe; Jane Doe left")
    assert out == "«PERSON_001» met «PERSON_002»; «PERSON_001» left"
    assert len(vault) == 2
    assert redactor.counts["PERSON"] == 3


def test_folded_builtin_type_is_suppressed_with_its_rule(
    monkeypatch: pytest.MonkeyPatch, fold_raw: bool
) -> None:
    # A model label folded into a built-in type (EMAIL_ADDRESS -> EMAIL)
    # follows that type's rule toggle: off means off for NER too.
    pipe = FakeHfPipe([("a@corp.example", "EMAIL_ADDRESS", 0.9), ("Jane", "PER", 0.9)])
    install_transformers(monkeypatch, pipe)
    ner = NerConfig(enabled=True, backend="hf", entities=("EMAIL", "PERSON"))
    (off,) = build_detectors(DetectionConfig(enabled=(), ner=ner))
    assert [d.detector_type for d in off.detect("Jane a@corp.example")] == ["PERSON"]
    _email_rule, on = build_detectors(DetectionConfig(enabled=("email",), ner=ner))
    assert sorted(d.detector_type for d in on.detect("Jane a@corp.example")) == ["EMAIL", "PERSON"]


# --- gliner ---------------------------------------------------------------------


def test_gliner_type_request_sends_its_prompt(
    monkeypatch: pytest.MonkeyPatch, fold_raw: bool
) -> None:
    model = FakeGliner([("1 Main St", "street address", 0.9)])
    install_gliner(monkeypatch, model)
    ner = NerConfig(enabled=True, backend="gliner", entities=("ADDRESS",))
    (detector,) = build_detectors(DetectionConfig(enabled=(), ner=ner))
    assert [(d.detector_type, d.value) for d in detector.detect("at 1 Main St")] == [
        ("ADDRESS", "1 Main St")
    ]
    assert model.calls == [["street address"]]


def test_gliner_raw_request_is_sent_verbatim(fold_raw: bool) -> None:
    model = FakeGliner([("555-0100", "phone number", 0.9)])
    detector = GlinerDetector(model, frozenset({"phone number"}), 1000, 0.5)
    assert [d.detector_type for d in detector.detect("call 555-0100")] == [
        "PHONE" if fold_raw else "PHONE_NUMBER"
    ]
    assert model.calls == [["phone number"]]


def test_gliner_type_request_wins_a_shared_prompt(fold_raw: bool) -> None:
    # entities = ["PHONE", "phone number"]: the prompt is sent once and its
    # label comes back as the type request's type, in both modes.
    model = FakeGliner([("555-0100", "phone number", 0.9)])
    policy = LabelPolicy(("PHONE", "phone number"), backend="gliner")
    detector = GlinerDetector(model, frozenset(), 1000, 0.5, policy=policy)
    assert [d.detector_type for d in detector.detect("call 555-0100")] == ["PHONE"]
    assert model.calls == [["phone number"]]
    assert policy.classify("phone number") == "PHONE"


def test_gliner_label_equal_to_a_prompt_is_the_requesting_type(fold_raw: bool) -> None:
    policy = LabelPolicy(("PERSON", "USERNAME", "ORG"), backend="gliner")
    assert policy.prompts == ("person", "username", "ORG")
    assert policy.classify_gliner("person") == "PERSON"
    assert policy.classify_gliner("username") == "USERNAME"
    assert policy.classify_gliner("ORG") == "ORG"
    assert policy.classify_gliner("street address") is None  # never requested


def test_gliner_without_prompts_never_calls_the_model() -> None:
    model = FakeGliner([("Jane", "person", 0.9)])
    detector = GlinerDetector(model, frozenset(), 1000, 0.5)
    assert list(detector.detect("Jane")) == []
    assert model.calls == []


@pytest.mark.parametrize(("start", "end"), [(-1, 3), (2, 2), (0, 99)])
def test_gliner_out_of_range_offsets_are_skipped(start: int, end: int) -> None:
    class Scripted:
        def predict_entities(
            self, text: str, labels: list[str], threshold: float
        ) -> list[dict[str, object]]:
            return [
                {"start": start, "end": end, "label": "person", "text": "x", "score": 0.9},
                {"start": 0, "end": 4, "label": "person", "text": "JANE", "score": 0.9},
            ]

    detector = GlinerDetector(Scripted(), frozenset({"PERSON"}), 1000, 0.5)
    assert [(d.start, d.end, d.value) for d in detector.detect("Jane Doe")] == [(0, 4, "Jane")]


# --- spaCy, Stanza, Presidio (the same policy) ------------------------------------


@pytest.mark.parametrize("backend", ["spacy", "stanza"])
def test_spacy_and_stanza_fold_per(
    monkeypatch: pytest.MonkeyPatch, backend: str, fold_raw: bool
) -> None:
    nlp = FakeSpacy([("Jane Doe", "PER", 1.0), ("Acme", "ORG", 1.0)])
    (install_spacy if backend == "spacy" else install_stanza)(monkeypatch, nlp)
    for entities, expected in (
        (("PERSON",), ["PERSON"]),
        (("PER",), ["PERSON" if fold_raw else "PER"]),
        (("PERSON", "ORG"), ["PERSON", "ORG"]),
    ):
        ner = NerConfig(enabled=True, backend=backend, entities=entities)
        (detector,) = build_detectors(DetectionConfig(enabled=(), ner=ner))
        assert [d.detector_type for d in detector.detect("Jane Doe at Acme")] == expected


# A realistic slice of presidio-analyzer's default recognizers (English).
PRESIDIO_SUPPORTED = (
    "CREDIT_CARD",
    "CRYPTO",
    "DATE_TIME",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "MEDICAL_LICENSE",
    "NRP",
    "PERSON",
    "PHONE_NUMBER",
    "URL",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "US_SSN",
)
PRESIDIO_FINDINGS = [
    ("jane@corp-example.com", "EMAIL_ADDRESS", 0.9),
    ("10.1.2.3", "IP_ADDRESS", 0.9),
    ("Jane Doe", "PERSON", 0.9),
    ("Berlin", "LOCATION", 0.9),
]
PRESIDIO_TEXT = "Jane Doe <jane@corp-example.com> from Berlin at 10.1.2.3"


def _presidio(
    monkeypatch: pytest.MonkeyPatch, entities: tuple[str, ...], *, enabled: tuple[str, ...] = ()
) -> tuple[FakeAnalyzer, list[str]]:
    analyzer = FakeAnalyzer(PRESIDIO_FINDINGS, PRESIDIO_SUPPORTED)
    install_presidio(monkeypatch, analyzer)
    ner = NerConfig(enabled=True, backend="presidio", entities=entities)
    detectors = build_detectors(DetectionConfig(enabled=enabled, ner=ner))
    found = sorted(d.detector_type for d in detectors[-1].detect(PRESIDIO_TEXT))
    return analyzer, found


def test_presidio_folds_exactly_as_before(monkeypatch: pytest.MonkeyPatch, fold_raw: bool) -> None:
    analyzer, found = _presidio(
        monkeypatch, ("EMAIL_ADDRESS", "IP_ADDRESS", "PERSON"), enabled=("email",)
    )
    assert analyzer.calls == [["EMAIL_ADDRESS", "IP_ADDRESS", "PERSON"]]
    assert found == ["EMAIL", "IP_ADDRESS", "PERSON"]  # IP_ADDRESS stays unfolded


def test_presidio_type_requests_are_translated(
    monkeypatch: pytest.MonkeyPatch, fold_raw: bool
) -> None:
    # EMAIL asks Presidio for EMAIL_ADDRESS; ADDRESS has no Presidio entity
    # and is dropped from the request (the analyzer would raise on it).
    analyzer, found = _presidio(monkeypatch, ("EMAIL", "PERSON", "ADDRESS"), enabled=("email",))
    assert analyzer.calls == [["EMAIL_ADDRESS", "PERSON"]]
    assert found == ["EMAIL", "PERSON"]


def test_presidio_is_asked_for_person_once_raw_entities_fold(
    monkeypatch: pytest.MonkeyPatch, fold_raw: bool
) -> None:
    analyzer, found = _presidio(monkeypatch, ("PER", "EMAIL"), enabled=("email",))
    if fold_raw:
        assert analyzer.calls == [["EMAIL_ADDRESS", "PERSON"]]
        assert found == ["EMAIL", "PERSON"]
    else:  # PER is no Presidio entity: asked for as written, so dropped
        assert analyzer.calls == [["EMAIL_ADDRESS"]]
        assert found == ["EMAIL"]


@pytest.mark.parametrize("entities", [("ADDRESS",), ("PER",), ("job title", "USERNAME")])
def test_presidio_supporting_no_entity_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch, entities: tuple[str, ...]
) -> None:
    from llm_redact.config import ConfigError

    monkeypatch.setattr(labels, "FOLD_RAW_REQUESTS", False)
    analyzer = FakeAnalyzer(PRESIDIO_FINDINGS, PRESIDIO_SUPPORTED)
    install_presidio(monkeypatch, analyzer)
    ner = NerConfig(enabled=True, backend="presidio", entities=entities)
    with pytest.raises(ConfigError, match="match no entity the Presidio analyzer supports"):
        build_detectors(DetectionConfig(enabled=(), ner=ner))
    assert analyzer.calls == []  # never a per-request ValueError


def test_presidio_rule_toggle_suppresses_its_folds(
    monkeypatch: pytest.MonkeyPatch, fold_raw: bool
) -> None:
    _analyzer, found = _presidio(monkeypatch, ("EMAIL", "PERSON"), enabled=())
    assert found == ["PERSON"]


# --- [detection.ner.labels] overrides ---------------------------------------------


def _parse_labels(table: object) -> tuple[tuple[str, str], ...]:
    from llm_redact.config import parse_config

    return parse_config({"detection": {"ner": {"labels": table}}}, "<test>").detection.ner.labels


def test_labels_parse_normalized_and_sorted() -> None:
    assert _parse_labels({"city": "ADDRESS", "first name": "PERSON", "B-TIME": ""}) == (
        ("CITY", "ADDRESS"),
        ("FIRST_NAME", "PERSON"),
        ("TIME", ""),
    )
    assert _parse_labels({}) == ()
    # Two spellings of one label with the same type are one entry.
    assert _parse_labels({"first name": "PERSON", "FIRST_NAME": "PERSON"}) == (
        ("FIRST_NAME", "PERSON"),
    )


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ({"CITY": "address"}, r"\[detection.ner.labels\] CITY: the type must match"),
        ({"CITY": "A" * 21}, r"CITY: the type must match .* at most 20"),
        ({"CITY": "STREET-ADDRESS"}, "CITY: the type must match"),
        ({"CITY": "1ADDRESS"}, "CITY: the type must match"),
        ({"CITY": 5}, "CITY: the type must match"),
        ({"CITY": ["ADDRESS"]}, "CITY: the type must match"),
        ({"--": "ADDRESS"}, "'--': a label needs at least one letter or digit"),
        (
            {"first name": "PERSON", "FIRST_NAME": "NAME"},
            r"'first name' and 'FIRST_NAME' name the same label \(FIRST_NAME\)",
        ),
        ("CITY", "must be a table of LABEL = TYPE"),
    ],
)
def test_invalid_labels_are_config_errors(table: object, message: str) -> None:
    from llm_redact.config import ConfigError

    with pytest.raises(ConfigError, match=message):
        _parse_labels(table)


def test_a_twenty_character_type_is_accepted() -> None:
    assert _parse_labels({"CITY": "A" * 20}) == (("CITY", "A" * 20),)


def test_override_opts_a_label_into_a_requested_type(
    monkeypatch: pytest.MonkeyPatch, fold_raw: bool
) -> None:
    # CITY is deliberately unfolded; an override folds it into ADDRESS.
    pipe = FakeHfPipe([("Springfield", "CITY", 0.9), ("1 Main St", "STREET_ADDRESS", 0.9)])
    install_transformers(monkeypatch, pipe)
    for labels_table, expected in (
        ((), ["ADDRESS"]),
        ((("CITY", "ADDRESS"),), ["ADDRESS", "ADDRESS"]),
    ):
        ner = NerConfig(enabled=True, backend="hf", entities=("ADDRESS",), labels=labels_table)
        (detector,) = build_detectors(DetectionConfig(enabled=(), ner=ner))
        found = [d.detector_type for d in detector.detect("1 Main St, Springfield")]
        assert found == expected


def test_empty_override_drops_the_label(fold_raw: bool) -> None:
    policy = LabelPolicy(("EMAIL", "PERSON"), overrides={"EMAIL": "", "PER": ""})
    assert policy.classify("EMAIL") is None
    assert policy.classify("PER") is None
    assert policy.classify("PERSON") == "PERSON"


def test_override_keeps_per_in_both_modes(monkeypatch: pytest.MonkeyPatch, fold_raw: bool) -> None:
    # The opt-out from the 2.0.0 fold: PER = "PER" keeps entities = ["PER"]
    # requesting and emitting PER.
    install_transformers(monkeypatch, FakeHfPipe([("Jane Doe", "PER", 0.9)]))
    ner = NerConfig(enabled=True, backend="hf", entities=("PER",), labels=(("PER", "PER"),))
    (detector,) = build_detectors(DetectionConfig(enabled=(), ner=ner))
    assert [d.detector_type for d in detector.detect("hi Jane Doe")] == ["PER"]
    policy = LabelPolicy(("PER",), overrides={"PER": "PER"})
    assert policy.requested == {"PER"}
    assert policy.raw_requested == frozenset()


@pytest.mark.parametrize("backend", ["spacy", "stanza", "gliner", "presidio", "hf"])
def test_every_builder_hands_the_overrides_to_its_policy(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    # Each backend's model reports the same name as CITY; the override makes
    # it an ADDRESS for every one of them.
    install_spacy(monkeypatch, FakeSpacy([("Springfield", "CITY", 1.0)]))
    install_stanza(monkeypatch, FakeSpacy([("Springfield", "CITY", 1.0)]))
    install_gliner(monkeypatch, FakeGliner([("Springfield", "CITY", 0.9)]))
    install_transformers(monkeypatch, FakeHfPipe([("Springfield", "CITY", 0.9)]))
    install_presidio(monkeypatch, FakeAnalyzer([("Springfield", "CITY", 0.9)], ("CITY", "PERSON")))
    ner = NerConfig(
        enabled=True, backend=backend, entities=("ADDRESS", "CITY"), labels=(("CITY", "ADDRESS"),)
    )
    (detector,) = build_detectors(DetectionConfig(enabled=(), ner=ner))
    assert [d.detector_type for d in detector.detect("in Springfield")] == ["ADDRESS"]


def test_an_override_target_outside_the_grammar_is_never_emitted(fold_raw: bool) -> None:
    # Programmatic configs skip the parser's check; the runtime guard holds.
    policy = LabelPolicy(("CITY",), overrides={"CITY": "city-name"})
    assert policy.classify("CITY") is None


# --- [detection.allowlist_by_type] keys for NER types ------------------------------


def _allowlist(
    keys: tuple[tuple[str, tuple[str, ...]], ...], ner: NerConfig, **detection: object
) -> "Allowlist":
    from llm_redact.detection.engine import build_allowlist

    return build_allowlist(
        DetectionConfig(allowlist_by_type=keys, ner=ner, **detection)  # type: ignore[arg-type]
    )


def test_allowlist_accepts_the_type_gliner_emits(fold_raw: bool) -> None:
    ner = NerConfig(backend="gliner", entities=("job title",))
    allow = _allowlist((("JOB_TITLE", ("Engineer",)),), ner)
    assert allow.allows_for("JOB_TITLE", "Engineer")
    # The entity as written stays valid, and now matches what GLiNER emits.
    as_written = _allowlist((("job title", ("Engineer",)),), ner)
    assert as_written.allows_for("JOB_TITLE", "Engineer")
    detector = GlinerDetector(FakeGliner([("Engineer", "job title", 0.9)]), frozenset(), 1000, 0.5)
    from llm_redact.detection.engine import detect_all

    assert detect_all([detector], "an Engineer", as_written) == []


def test_allowlist_accepts_a_type_request(fold_raw: bool) -> None:
    allow = _allowlist((("ADDRESS", ("1 Main St",)),), NerConfig(entities=("ADDRESS",)))
    assert allow.allows_for("ADDRESS", "1 Main St")


def test_allowlist_written_for_per_keeps_matching(
    monkeypatch: pytest.MonkeyPatch, fold_raw: bool
) -> None:
    from llm_redact.detection.engine import build_allowlist, detect_all

    install_transformers(monkeypatch, FakeHfPipe([("Jane Doe", "PER", 0.9), ("Bob", "PER", 0.9)]))
    config = DetectionConfig(
        enabled=(),
        allowlist_by_type=(("PER", ("Jane Doe",)),),
        ner=NerConfig(enabled=True, backend="hf", entities=("PER",)),
    )
    allow = build_allowlist(config)
    found = detect_all(build_detectors(config), "Jane Doe and Bob", allow)
    assert [(d.detector_type, d.value) for d in found] == [("PERSON" if fold_raw else "PER", "Bob")]


def test_allowlist_key_canonicalization_is_logged_without_values(
    caplog: pytest.LogCaptureFixture, fold_raw: bool
) -> None:
    import logging

    ner = NerConfig(entities=("PER", "job title"))
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        allow = _allowlist((("PER", ("Jane Doe",)), ("job title", ("Engineer",))), ner)
    assert allow.by_type == {
        ("PERSON" if fold_raw else "PER"): frozenset({"Jane Doe"}),
        "JOB_TITLE": frozenset({"Engineer"}),
    }
    (record,) = [r for r in caplog.records if "allowlist_by_type" in r.getMessage()]
    message = record.getMessage()
    assert "'job title' -> JOB_TITLE" in message
    assert ("'PER' -> PERSON" in message) is fold_raw
    assert "Jane" not in message and "Engineer" not in message


def test_allowlist_keys_of_one_type_merge(fold_raw: bool) -> None:
    ner = NerConfig(entities=("PERSON", "PER"), labels=(("PER", "PERSON"),))
    allow = _allowlist((("PER", ("Jane",)), ("PERSON", ("Bob",))), ner)
    assert allow.by_type == {"PERSON": frozenset({"Jane", "Bob"})}


def test_allowlist_rule_types_are_never_renamed(fold_raw: bool) -> None:
    # A custom rule may emit a type that is also a fold source (PASSWORD ->
    # SECRET): its allowlist must keep matching the custom rule's type.
    from llm_redact.detection.engine import CustomRule

    allow = _allowlist(
        (("PASSWORD", ("hunter2",)),),
        NerConfig(entities=("PASSWORD",)),
        custom_rules=(CustomRule(name="pw", detector_type="PASSWORD", pattern="pw:\\S+"),),
    )
    assert allow.by_type == {"PASSWORD": frozenset({"hunter2"})}


def test_allowlist_keys_valid_before_stay_valid(fold_raw: bool) -> None:
    from llm_redact.detection.deny import DenyEntry
    from llm_redact.detection.engine import CustomRule

    keys = (
        ("EMAIL", ("a@corp.example",)),
        ("TICKET", ("PROJ-1",)),
        ("DENY", ("x",)),
        ("job title", ("Engineer",)),
        ("PER", ("Jane",)),
        ("--", ("y",)),  # an entity as written that normalizes to nothing
    )
    allow = _allowlist(
        keys,
        NerConfig(entities=("job title", "PER", "--")),
        custom_rules=(CustomRule(name="t", detector_type="TICKET", pattern="PROJ-\\d+"),),
        deny_strings=(DenyEntry(value="x"),),
    )
    assert {"EMAIL", "TICKET", "DENY", "JOB_TITLE", "--"} <= set(allow.by_type)


def test_allowlist_unknown_keys_still_refused(fold_raw: bool) -> None:
    with pytest.raises(ValueError, match=r"unknown placeholder type\(s\) \['ADRESS', 'PERSON'\]"):
        _allowlist(
            (("ADRESS", ("x",)), ("PERSON", ("y",))),
            NerConfig(entities=("ADDRESS",)),
        )


def test_allowlist_accepts_override_targets(fold_raw: bool) -> None:
    ner = NerConfig(entities=("ADDRESS", "CITY"), labels=(("CITY", "LOCALITY"),))
    allow = _allowlist((("LOCALITY", ("Springfield",)), ("CITY", ("Shelbyville",))), ner)
    assert allow.by_type == {"LOCALITY": frozenset({"Springfield", "Shelbyville"})}


# --- 2.0.0 deprecation of raw entities (D15 (i)) --------------------------------------


def _deprecations(**ner: object) -> list[str]:
    from llm_redact.detection.engine import ner_warnings

    return ner_warnings(DetectionConfig(ner=NerConfig(enabled=True, **ner)))  # type: ignore[arg-type]


def test_raw_entity_that_changes_type_warns_once() -> None:
    assert _deprecations(backend="hf", entities=("PER", "PERSON", "ORG")) == [
        '[detection.ner] entities: "PER" is emitted as PER now and as PERSON from 2.0.0;'
        ' write "PERSON" to switch now, or set [detection.ner.labels] PER = "PER" to'
        " keep PER"
    ]
    # One line per entity, not per backend.
    assert len(_deprecations(backends=("hf", "spacy"), entities=("PER",))) == 1


@pytest.mark.parametrize(
    "ner",
    [
        {"backend": "hf", "entities": ("PERSON",)},  # a type request
        {"backend": "hf", "entities": ("PER",), "labels": (("PER", "PER"),)},  # an override
        {"backend": "hf", "entities": ("PER",), "labels": (("PER", "PERSON"),)},
        {"backend": "presidio", "entities": ("EMAIL_ADDRESS",)},  # already EMAIL there
        {"backend": "spacy", "entities": ("ORG", "job title")},  # no fold either way
        {"backend": "gliner", "entities": ("PHONE", "phone number")},  # the type wins
    ],
)
def test_entities_whose_type_does_not_change_do_not_warn(ner: dict[str, object]) -> None:
    assert _deprecations(**ner) == []


def test_emitted_type_names_the_backend_where_it_changes() -> None:
    # EMAIL_ADDRESS is EMAIL on presidio already, but EMAIL_ADDRESS on hf.
    (warning,) = _deprecations(backends=("presidio", "hf"), entities=("EMAIL_ADDRESS",))
    assert "emitted as EMAIL_ADDRESS now and as EMAIL from 2.0.0" in warning
    assert '[detection.ner.labels] EMAIL_ADDRESS = "EMAIL_ADDRESS"' in warning
    (gliner,) = _deprecations(backend="gliner", entities=("phone number",))
    assert '"phone number" is emitted as PHONE_NUMBER now and as PHONE' in gliner
    assert 'set [detection.ner.labels] PHONE_NUMBER = "PHONE_NUMBER"' in gliner


def test_no_deprecation_once_raw_entities_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(labels, "FOLD_RAW_REQUESTS", True)
    assert _deprecations(backend="hf", entities=("PER",)) == []


def test_no_deprecation_while_ner_is_off() -> None:
    from llm_redact.detection.engine import ner_warnings

    assert ner_warnings(DetectionConfig(ner=NerConfig(backend="hf", entities=("PER",)))) == []


def test_startup_and_rebuild_log_the_deprecation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import dataclasses
    import logging

    from llm_redact.config import Config
    from llm_redact.proxy import create_app

    install_transformers(monkeypatch, FakeHfPipe([("Jane Doe", "PER", 0.9)]))
    ner = NerConfig(enabled=True, backend="hf", entities=("PER",))
    config = Config(detection=DetectionConfig(ner=ner))
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        app = create_app(config)
    logged = [r.getMessage() for r in caplog.records if "from 2.0.0" in r.getMessage()]
    assert logged == [labels.raw_entity_deprecations(ner)[0]]
    caplog.clear()
    state = app.state.proxy
    # A reload that keeps [detection] does not rebuild, so it does not repeat it...
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        state.apply_config(config)
    assert not [r for r in caplog.records if "from 2.0.0" in r.getMessage()]
    # ...one that rebuilds the detectors does.
    changed = dataclasses.replace(
        config,
        detection=dataclasses.replace(config.detection, ner=dataclasses.replace(ner, max_chars=9)),
    )
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        state.apply_config(changed)
    assert len([r for r in caplog.records if "from 2.0.0" in r.getMessage()]) == 1


def test_doctor_shows_the_deprecation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse
    import json

    from llm_redact.doctor_cli import run_doctor

    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    config_file = tmp_path / "config.toml"
    for entities, expected in (('["PER"]', 1), ('["PERSON"]', 0)):
        config_file.write_text(
            f'[detection.ner]\nenabled = true\nbackend = "hf"\nentities = {entities}\n'
        )
        run_doctor(argparse.Namespace(config=config_file, json=True))
        rows = json.loads(capsys.readouterr().out)["checks"]
        warned = [r for r in rows if r["area"] == "ner" and "from 2.0.0" in r["message"]]
        assert len(warned) == expected
        assert all(r["level"] == "WARN" for r in warned)


# --- parts of one name or address merge (T07) -----------------------------------------


def _parts(text: str, *spans: tuple[str, str]) -> list[tuple[str, str]]:
    from llm_redact.detection.base import Detection
    from llm_redact.detection.labels import merge_adjacent_parts

    detections = []
    cursor = 0
    for surface, type_name in spans:
        start = text.index(surface, cursor)
        cursor = start + len(surface)
        detections.append(Detection(start, cursor, type_name, surface, priority=120))
    return [(d.detector_type, d.value) for d in merge_adjacent_parts(detections, text)]


@pytest.mark.parametrize(
    "text",
    ["Jane Doe", "Jane  Doe", "Jane\tDoe", "Jane Doe", "Jane  Doe", "Jane\t Doe"],
)
def test_name_parts_separated_by_blanks_merge(text: str) -> None:
    assert _parts(text, ("Jane", "PERSON"), ("Doe", "PERSON")) == [("PERSON", text)]


@pytest.mark.parametrize(
    "text",
    [
        "Jane\nDoe",
        "Jane\r\nDoe",
        "Jane, Doe",
        "Jane,Doe",
        '"Jane", "Doe"',
        '{"first": "Jane", "last": "Doe"}',
        "Jane   Doe",  # three blanks
        "Jane.Doe",
        "JaneDoe",  # no gap at all
        "Jane - Doe",
    ],
)
def test_name_parts_across_anything_else_stay_apart(text: str) -> None:
    assert _parts(text, ("Jane", "PERSON"), ("Doe", "PERSON")) == [
        ("PERSON", "Jane"),
        ("PERSON", "Doe"),
    ]


def test_three_parts_and_addresses_merge() -> None:
    assert _parts("Jane Mary Doe", ("Jane", "PERSON"), ("Mary", "PERSON"), ("Doe", "PERSON")) == [
        ("PERSON", "Jane Mary Doe")
    ]
    assert _parts("at 12 Main St", ("12", "ADDRESS"), ("Main St", "ADDRESS")) == [
        ("ADDRESS", "12 Main St")
    ]


def test_only_names_and_addresses_merge_and_only_within_one_type() -> None:
    assert _parts("Acme Corp", ("Acme", "ORG"), ("Corp", "ORG")) == [
        ("ORG", "Acme"),
        ("ORG", "Corp"),
    ]
    assert _parts("Jane Main St", ("Jane", "PERSON"), ("Main St", "ADDRESS")) == [
        ("PERSON", "Jane"),
        ("ADDRESS", "Main St"),
    ]


def test_merge_reads_parts_in_text_order() -> None:
    from llm_redact.detection.base import Detection
    from llm_redact.detection.labels import merge_adjacent_parts

    text = "Jane Doe"
    reversed_parts = [Detection(5, 8, "PERSON", "Doe"), Detection(0, 4, "PERSON", "Jane")]
    (merged,) = merge_adjacent_parts(reversed_parts, text)
    assert (merged.start, merged.end, merged.value) == (0, 8, "Jane Doe")


@pytest.mark.parametrize("backend", ["spacy", "stanza", "gliner", "presidio", "hf"])
def test_every_backend_merges_first_and_last_names(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    findings = [("Jane", "first_name", 0.9), ("Doe", "last_name", 0.9)]
    install_spacy(monkeypatch, FakeSpacy(findings))
    install_stanza(monkeypatch, FakeSpacy(findings))
    install_gliner(monkeypatch, FakeGliner(findings))
    install_transformers(monkeypatch, FakeHfPipe(findings))
    install_presidio(monkeypatch, FakeAnalyzer(findings, ("first_name", "last_name")))
    entities = ("PERSON", "first_name", "last_name") if backend == "gliner" else ("PERSON",)
    ner = NerConfig(enabled=True, backend=backend, entities=entities)
    (detector,) = build_detectors(DetectionConfig(enabled=(), ner=ner))
    found = [(d.detector_type, d.value) for d in detector.detect("hi Jane Doe, bye")]
    if backend == "gliner":  # raw requests keep their own type until 2.0.0
        assert found == [("FIRST_NAME", "Jane"), ("LAST_NAME", "Doe")]
    else:
        assert found == [("PERSON", "Jane Doe")]


def test_gliner_merges_parts_once_raw_entities_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(labels, "FOLD_RAW_REQUESTS", True)
    model = FakeGliner([("Jane", "first name", 0.9), ("Doe", "last name", 0.9)])
    detector = GlinerDetector(model, frozenset({"first name", "last name"}), 1000, 0.5)
    assert [(d.detector_type, d.value) for d in detector.detect("hi Jane Doe")] == [
        ("PERSON", "Jane Doe")
    ]


def test_merged_name_gets_one_token(fold_raw: bool) -> None:
    hf = HfDetector(
        FakeHfPipe([("Jane", "B-first_name", 0.9), ("Doe", "B-last_name", 0.9)]),
        frozenset({"PERSON"}),
        max_chars=1000,
        threshold=0.5,
    )
    vault = InMemoryVault()
    redactor = Redactor([hf], vault, Allowlist(exact=frozenset(), patterns=()))
    assert redactor.redact_text("Jane Doe; Jane Doe") == "«PERSON_001»; «PERSON_001»"
    assert len(vault) == 1


# --- entities that can never match (T08) ------------------------------------------------

# dslim/bert-base-NER's config.json id2label (CoNLL-2003 tags).
DSLIM_ID2LABEL = {
    0: "O",
    1: "B-MISC",
    2: "I-MISC",
    3: "B-PER",
    4: "I-PER",
    5: "B-ORG",
    6: "I-ORG",
    7: "B-LOC",
    8: "I-LOC",
}


def _built(monkeypatch: pytest.MonkeyPatch, **ner: object) -> tuple[DetectionConfig, list[object]]:
    install_transformers(monkeypatch, FakeHfPipe([], id2label=DSLIM_ID2LABEL))
    install_spacy(monkeypatch, FakeSpacy([], labels=("PERSON", "ORG", "GPE", "DATE")))
    install_stanza(monkeypatch, FakeSpacy([]))
    install_gliner(monkeypatch, FakeGliner([]))
    install_presidio(monkeypatch, FakeAnalyzer([], PRESIDIO_SUPPORTED))
    config = DetectionConfig(enabled=(), ner=NerConfig(enabled=True, **ner))  # type: ignore[arg-type]
    return config, list(build_detectors(config))


def _never(monkeypatch: pytest.MonkeyPatch, **ner: object) -> list[str]:
    from llm_redact.detection.engine import ner_warnings

    config, detectors = _built(monkeypatch, **ner)
    return [w for w in ner_warnings(config, detectors) if "can never match" in w]  # type: ignore[arg-type]


def test_default_hf_config_matches(monkeypatch: pytest.MonkeyPatch, fold_raw: bool) -> None:
    assert _never(monkeypatch, backend="hf") == []
    assert _never(monkeypatch, backend="hf", entities=("PERSON", "ORG", "LOC", "MISC")) == []
    # While raw entities keep their type, a raw PER claims the model's PER
    # label, so nothing is left for PERSON on this model: said, not hidden.
    both = _never(monkeypatch, backend="hf", entities=("PERSON", "PER"))
    assert [w.split('"')[1] for w in both] == ([] if fold_raw else ["PERSON"])


def test_a_typo_can_never_match(monkeypatch: pytest.MonkeyPatch, fold_raw: bool) -> None:
    assert _never(monkeypatch, backend="hf", entities=("PERSONS",)) == [
        '[detection.ner] entities: "PERSONS" can never match: no active backend emits it'
        " (hf: dslim/bert-base-NER)"
    ]


def test_multi_backend_warns_only_for_uncovered_entities(
    monkeypatch: pytest.MonkeyPatch, fold_raw: bool
) -> None:
    warnings = _never(
        monkeypatch,
        backends=("hf", "spacy", "presidio"),
        entities=("PERSON", "ORG", "GPE", "EMAIL", "ADDRESS"),
    )
    assert warnings == [
        '[detection.ner] entities: "ADDRESS" can never match: no active backend emits it'
        " (hf: dslim/bert-base-NER, spacy: en_core_web_sm, presidio: en_core_web_sm)"
    ]


@pytest.mark.parametrize("backend", ["gliner", "stanza"])
def test_backends_without_a_label_set_can_emit_anything(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    assert _never(monkeypatch, backends=("hf", backend), entities=("PERSONS", "ADDRESS")) == []


def test_presidio_entities_it_lacks_can_never_match(
    monkeypatch: pytest.MonkeyPatch, fold_raw: bool
) -> None:
    warnings = _never(monkeypatch, backend="presidio", entities=("PERSON", "ADDRESS"))
    assert [w.split('"')[1] for w in warnings] == ["ADDRESS"]


def test_a_dropped_or_ungrammatical_entity_can_never_match(
    monkeypatch: pytest.MonkeyPatch, fold_raw: bool
) -> None:
    warnings = _never(
        monkeypatch,
        backends=("gliner",),
        entities=("PERSON", "3d model"),
        labels=(("PERSON", ""),),
    )
    assert [w.split('"')[1] for w in warnings] == ["PERSON", "3d model"]


def test_unmatched_entities_are_kept_on_the_detectors(
    monkeypatch: pytest.MonkeyPatch, fold_raw: bool
) -> None:
    from llm_redact.detection.engine import ner_backends, ner_unmatched_entities

    _config, detectors = _built(monkeypatch, backends=("hf", "spacy"), entities=("PERSON", "ZIP"))
    assert ner_unmatched_entities(detectors) == ("ZIP",)  # type: ignore[arg-type]
    backends = ner_backends(detectors)  # type: ignore[arg-type]
    assert [b.unmatched_entities for b in backends] == [("ZIP",), ("ZIP",)]  # type: ignore[attr-defined]
    assert ner_unmatched_entities([]) == ()


def test_backends_that_do_not_say_are_never_reported() -> None:
    # A pipeline without id2label / get_pipe: unknown, so nothing is claimed.
    from llm_redact.detection.engine import ner_unmatched_entities

    hf = HfDetector(FakeHfPipe([]), frozenset({"PERSONS"}), 1000, 0.5)
    spacy_like = NerDetector(FakeSpacy([]), frozenset({"PERSONS"}), 1000)
    assert hf.emittable_types is None and spacy_like.emittable_types is None
    nlp_without_ner = FakeSpacy([], labels=None)
    assert NerDetector(nlp_without_ner, frozenset({"PERSON"}), 1000).emittable_types is None
    assert ner_unmatched_entities([hf, spacy_like]) == ()


def test_stand_in_backends_without_a_policy_record_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_redact.detection import ner as ner_mod
    from llm_redact.detection.base import Detection
    from llm_redact.detection.engine import ner_unmatched_entities, ner_warnings

    class Stub:
        name = "stub"

        def detect(self, text: str) -> list[Detection]:
            return []

    monkeypatch.setattr(ner_mod, "build_ner_detector", lambda config: Stub())
    config = DetectionConfig(ner=NerConfig(enabled=True, entities=("PERSONS",)))
    detectors = build_detectors(config)
    assert ner_unmatched_entities(detectors) == ()
    assert ner_warnings(config, detectors) == []


def test_startup_logs_the_never_match_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from llm_redact.config import Config
    from llm_redact.proxy import create_app

    install_transformers(monkeypatch, FakeHfPipe([], id2label=DSLIM_ID2LABEL))
    ner = NerConfig(enabled=True, backend="hf", entities=("PERSON", "PERSONS"))
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        create_app(Config(detection=DetectionConfig(ner=ner)))
    assert [r.getMessage() for r in caplog.records if "can never match" in r.getMessage()] == [
        '[detection.ner] entities: "PERSONS" can never match: no active backend emits it'
        " (hf: dslim/bert-base-NER)"
    ]
