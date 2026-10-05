"""The NER label policy (detection/labels.py): one placeholder type per value
across every backend, in both raw-entity modes (FOLD_RAW_REQUESTS False =
1.12.x, True = 2.0.0; the ``fold_raw`` fixture runs a test in each)."""

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
