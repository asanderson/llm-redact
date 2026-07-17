"""Assemble the detector list from configuration."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

from llm_redact.detection.base import Detection, Detector
from llm_redact.detection.deny import DenyDetector, DenyEntry
from llm_redact.detection.regex_rules import BUILTIN_RULES, PreparedText, RegexDetector, RegexRule


@dataclass
class TypeFilteredDetector:
    """Drops detections whose placeholder type is suppressed.

    Wraps NER backends so a type disabled at the rule level (email off in
    ``[detection] enabled``) cannot come back through an NER fold — the
    rule toggles stay the single source of truth for per-type enablement.
    (Not frozen: the Detector protocol's ``name`` member reads as settable,
    and frozen-dataclass fields do not satisfy that.)
    """

    inner: Detector
    suppressed: frozenset[str]
    name: str = "ner_type_filter"

    def detect(self, text: str) -> Iterable[Detection]:
        return [d for d in self.inner.detect(text) if d.detector_type not in self.suppressed]


DEFAULT_ALLOWLIST = frozenset({"127.0.0.1", "0.0.0.0", "255.255.255.255", "::1", "::"})


@dataclass(frozen=True)
class Allowlist:
    exact: frozenset[str] = DEFAULT_ALLOWLIST
    patterns: tuple[re.Pattern[str], ...] = ()
    # Exact values allowed only when matched as a specific detector TYPE:
    # "this is our support address, but redact every other email".
    by_type: dict[str, frozenset[str]] = field(default_factory=dict)

    def allows(self, value: str) -> bool:
        if value in self.exact:
            return True
        return any(p.search(value) for p in self.patterns)

    def allows_for(self, detector_type: str, value: str) -> bool:
        if self.allows(value):
            return True
        return value in self.by_type.get(detector_type, frozenset())


@dataclass(frozen=True)
class CustomRule:
    name: str
    detector_type: str
    pattern: str
    priority: int = 100
    # Optional named checksum/format gate (see detection/validators.py): the
    # rule fires only when the regex matches AND the validator passes.
    validator: str | None = None
    # Optional hot-path prefilter hints, mirroring the built-in rules: every
    # `required` literal must be present in the text before the rule runs, and
    # every match must start with one of `anchors`. Both are single-literal
    # forms of RegexRule's CNF/anchor machinery — a wrong hint is a silent
    # recall bug, so they are opt-in and off by default.
    required: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class NerConfig:
    # Default off: requires an extra (`ner` for spacy, `gliner` for gliner,
    # `presidio` for presidio) and adds per-string latency the regex hot
    # path doesn't have.
    enabled: bool = False
    backend: str = "spacy"  # or "gliner" / "presidio" / "stanza" / "hf"
    # Multi-backend form: when set it wins over `backend` (which stays the
    # one-element legacy spelling); every listed backend runs concurrently
    # behind the same Detector protocol, and same-span same-type hits
    # dedupe in overlap resolution.
    backends: tuple[str, ...] | None = None
    entities: tuple[str, ...] = ("PERSON",)
    max_chars: int = 20000
    # Only meaningful for backends that emit confidences (gliner, presidio,
    # hf); config loading rejects it when no such backend is active (spacy
    # and stanza emit none).
    score_threshold: float = 0.5
    # NER language (presidio wires it through the analyzer, stanza selects the
    # language model; for spacy it is implied by the model) and an optional
    # model-name override: the spaCy pipeline for spacy/presidio (default
    # en_core_web_sm), the HF model id for gliner (default
    # urchade/gliner_small-v2.1) and hf (default dslim/bert-base-NER).
    language: str = "en"
    model: str | None = None
    # Per-backend model overrides ([detection.ner.models], stored sorted
    # for canonical equality). The legacy single `model` key only applies
    # when exactly one backend is active — a spaCy pipeline name handed to
    # gliner would be nonsense.
    models: tuple[tuple[str, str], ...] = ()

    def active_backends(self) -> tuple[str, ...]:
        return self.backends if self.backends is not None else (self.backend,)

    def model_for(self, backend: str) -> str | None:
        for name, model in self.models:
            if name == backend:
                return model
        active = self.active_backends()
        return self.model if len(active) == 1 else None


@dataclass(frozen=True)
class DetectionConfig:
    enabled: tuple[str, ...] = tuple(rule.name for rule in BUILTIN_RULES)
    # [detection] languages: ISO 639-1 codes the deployment's text is in.
    # None (default) = all languages, exact historical behavior. When set,
    # language-tagged rules (national ids) with no overlapping tag are NOT
    # BUILT; untagged rules (emails, IPs, vendor tokens, credit cards,
    # IBANs, phones) always run. Stored sorted — canonical equality.
    languages: tuple[str, ...] | None = None
    allowlist: tuple[str, ...] = ()
    allowlist_patterns: tuple[str, ...] = ()
    # Per-detector-type exact allowlist, stored sorted (canonical equality,
    # like modes): (("EMAIL", ("a@corp.example", ...)), ...).
    allowlist_by_type: tuple[tuple[str, tuple[str, ...]], ...] = ()
    custom_rules: tuple[CustomRule, ...] = field(default_factory=tuple)
    ner: NerConfig = field(default_factory=NerConfig)
    # Per-rule handling, keyed by RULE NAME: "redact" (default; omit),
    # "warn" (count + log the type, leave the value in the request), or
    # "block" (reject the whole request fail-closed). Stored sorted so
    # config equality (reload's detector-reuse check) is canonical.
    modes: tuple[tuple[str, str], ...] = ()
    # User deny strings (tier 0: always redacted, win every overlap, bypass
    # the allowlist, never subject to modes). Stored sorted — canonical
    # equality, same reason as modes.
    deny_strings: tuple[DenyEntry, ...] = ()
    # [detection.mcp] exempt_servers: MCP content blocks addressed to these
    # server names/labels bypass detection (the block is stashed before the
    # sweep and restored after, so nothing in it is counted). Stored sorted.
    mcp_exempt_servers: tuple[str, ...] = ()


def build_allowlist(config: DetectionConfig) -> Allowlist:
    # A typo'd TYPE key was silently inert (the user believes the value is
    # allowlisted; it keeps being redacted). Validate against every type that
    # can actually be emitted: built-in rules, custom rules, deny entries,
    # and the configured NER entity labels.
    if config.allowlist_by_type:
        known_types = (
            {rule.detector_type for rule in BUILTIN_RULES}
            | {rule.detector_type for rule in config.custom_rules}
            | {entry.detector_type for entry in config.deny_strings}
            | set(config.ner.entities)
        )
        unknown_types = sorted(
            detector_type
            for detector_type, _values in config.allowlist_by_type
            if detector_type not in known_types
        )
        if unknown_types:
            raise ValueError(
                f"unknown placeholder type(s) {unknown_types} in"
                f" [detection.allowlist_by_type]; known types are"
                f" {sorted(known_types)}"
            )
    patterns = []
    for p in config.allowlist_patterns:
        try:
            patterns.append(re.compile(p))
        except re.error as exc:
            # ValueError so serve --check / doctor / the editor report it as
            # a named config problem, never a raw re.error traceback.
            raise ValueError(
                f"[detection] allowlist_patterns entry {p!r}: invalid regex: {exc}"
            ) from exc
    return Allowlist(
        exact=DEFAULT_ALLOWLIST | frozenset(config.allowlist),
        patterns=tuple(patterns),
        by_type={
            detector_type: frozenset(values) for detector_type, values in config.allowlist_by_type
        },
    )


def _language_active(rule: RegexRule, languages: "tuple[str, ...] | None") -> bool:
    return (
        languages is None or rule.languages is None or not set(rule.languages).isdisjoint(languages)
    )


def active_rule_names(config: DetectionConfig) -> list[str]:
    """config.enabled minus rules language-scoped out.

    The single list build_detectors instantiates and the config editor's
    effective-rule display reports — computing it twice would let the UI
    disagree with what actually runs.
    """
    known = {rule.name: rule for rule in BUILTIN_RULES}
    unknown = [name for name in config.enabled if name not in known]
    if unknown:
        raise ValueError(f"unknown detection rule(s) {unknown!r}; built-ins are {sorted(known)}")
    return [name for name in config.enabled if _language_active(known[name], config.languages)]


def build_detectors(config: DetectionConfig) -> list[Detector]:
    known = {rule.name: rule for rule in BUILTIN_RULES}
    active = active_rule_names(config)
    detectors: list[Detector] = [RegexDetector(known[name]) for name in active]
    for custom in config.custom_rules:
        validator = None
        if custom.validator is not None:
            from llm_redact.detection.validators import VALIDATORS

            validator = VALIDATORS.get(custom.validator)
            if validator is None:
                raise ValueError(
                    f"custom rule {custom.name!r}: unknown validator {custom.validator!r};"
                    f" valid names are {sorted(VALIDATORS)}"
                )
        try:
            compiled = re.compile(custom.pattern)
        except re.error as exc:
            # Same contract as the unknown-validator error above: a bad
            # custom rule is a named ValueError, not an re.error traceback.
            raise ValueError(f"custom rule {custom.name!r}: invalid pattern: {exc}") from exc
        detectors.append(
            RegexDetector(
                RegexRule(
                    name=custom.name,
                    detector_type=custom.detector_type,
                    pattern=compiled,
                    priority=custom.priority,
                    validator=validator,
                    # A user literal becomes a single-alternative CNF clause.
                    required=tuple((literal,) for literal in custom.required),
                    anchors=custom.anchors,
                )
            )
        )
    if config.deny_strings:
        detectors.append(DenyDetector(config.deny_strings))
    if config.ner.enabled:
        # A placeholder type disabled at the rule level is disabled, period
        # — NER must not reintroduce it (presidio folds EMAIL/PHONE/SSN/
        # IBAN/CREDIT_CARD into the built-in types). Rule toggles are the
        # single source of truth; entity types with no built-in rule
        # (PERSON) are never suppressed. Language scoping counts as a rule
        # toggle here: a type whose only rule is scoped out stays out.
        enabled_types = frozenset(known[name].detector_type for name in active)
        suppressed = frozenset(
            rule.detector_type for rule in BUILTIN_RULES if rule.detector_type not in enabled_types
        )
        for backend_name in config.ner.active_backends():
            # Each backend builder still sees a single-backend view with
            # its own resolved model — the builders stay untouched.
            single = replace(
                config.ner,
                backend=backend_name,
                backends=None,
                model=config.ner.model_for(backend_name),
                models=(),
            )
            # Imported only when enabled: the NER dependencies stay
            # optional and startup fails fast per backend if missing.
            if backend_name == "gliner":
                from llm_redact.detection.gliner_ner import build_gliner_detector

                inner: Detector = build_gliner_detector(single)
            elif backend_name == "presidio":
                from llm_redact.detection.presidio_ner import build_presidio_detector

                inner = build_presidio_detector(single)
            elif backend_name == "stanza":
                from llm_redact.detection.stanza_ner import build_stanza_detector

                inner = build_stanza_detector(single)
            elif backend_name == "hf":
                from llm_redact.detection.hf_ner import build_hf_detector

                inner = build_hf_detector(single)
            else:
                from llm_redact.detection.ner import build_ner_detector

                inner = build_ner_detector(single)
            detectors.append(TypeFilteredDetector(inner, suppressed) if suppressed else inner)
    return detectors


def build_modes(config: DetectionConfig) -> dict[str, str]:
    """Rule-name-keyed mode config -> detector-TYPE-keyed dispatch map.

    Detections carry the detector type, not the rule name, and several rules
    share a type (github_token and github_fine_grained_pat are both
    GITHUB_TOKEN) — so modes are configured per rule for readability but
    must resolve to one mode per type. Conflicting assignments and unknown
    rule names are hard errors. Only non-default entries are returned, so an
    empty dict keeps the hot path branch-free.
    """
    type_by_rule = {rule.name: rule.detector_type for rule in BUILTIN_RULES}
    for custom in config.custom_rules:
        type_by_rule[custom.name] = custom.detector_type

    modes_by_type: dict[str, str] = {}
    rule_by_type: dict[str, str] = {}
    unknown = [name for name, _mode in config.modes if name not in type_by_rule]
    if unknown:
        raise ValueError(
            f"unknown rule name(s) in [detection.modes]: {unknown!r};"
            f" known rules are {sorted(type_by_rule)}"
        )
    for name, mode in config.modes:
        if mode == "redact":
            continue
        detector_type = type_by_rule[name]
        existing = modes_by_type.get(detector_type)
        if existing is not None and existing != mode:
            raise ValueError(
                f"conflicting modes for detector type {detector_type}: rules"
                f" {rule_by_type[detector_type]!r} and {name!r} share that type"
                " and must use the same mode"
            )
        modes_by_type[detector_type] = mode
        rule_by_type[detector_type] = name
    return modes_by_type


def detect_all(detectors: Sequence[Detector], text: str, allowlist: Allowlist) -> list[Detection]:
    # One PreparedText per body: regex detectors share it so their
    # required-literal prefilters (and the lowered haystack behind the
    # case-insensitive ones) are computed once, not per rule.
    prepared = PreparedText(text)
    detections: list[Detection] = []
    for det in detectors:
        found = (
            det.detect_prepared(prepared) if isinstance(det, RegexDetector) else det.detect(text)
        )
        # Tier-0 (deny) detections bypass the allowlist — global AND
        # per-type: deny is the user's explicit strongest signal, so a
        # deny/allowlist contradiction resolves in favor of redaction.
        detections.extend(
            d for d in found if d.tier == 0 or not allowlist.allows_for(d.detector_type, d.value)
        )
    detections.sort(key=lambda d: (d.start, -(d.end - d.start), d.priority))
    return detections
