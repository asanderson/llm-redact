"""Assemble the detector list from configuration."""

import re
from collections.abc import Callable, Iterable, Sequence
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
    return plan_for(detectors).detect(text, allowlist)


# Texts up to this many characters take the gated path of DetectorPlan.detect;
# longer ones run every detector. Per-rule interpreter overhead (~1 us a
# rule) is what the gate saves: measured, the gate wins 9x on a 10-character
# string and breaks even around 1-2 KB, where the scans themselves dominate
# and the combined pattern search starts to cost a second scan of any text
# in which something matches.
GATED_MAX_CHARS = 1024


def _runner(det: Detector) -> Callable[[PreparedText], Iterable[Detection]]:
    """How the plan runs ``det`` on a prepared text: regex rules share the
    PreparedText (their prefilters read its cached haystacks)."""
    if isinstance(det, RegexDetector):
        return det.detect_prepared
    return lambda prepared: det.detect(prepared.text)


def _gate_group(required: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """The CNF group a rule is gated on: any one group is a necessary
    condition; the one whose shortest literal is longest skips most."""
    return max(required, key=lambda group: min(len(lit) for lit in group))


def _combinable(pattern: "re.Pattern[str]") -> str | None:
    """``pattern``'s source as an alternation member, or None when it cannot
    be one: capturing groups (a backreference would be renumbered, a group
    name could repeat) or any flag beyond the str default (a global inline
    flag must lead the whole expression)."""
    if pattern.groups or pattern.flags & ~re.UNICODE:
        return None
    member = f"(?:{pattern.pattern})"
    try:
        compiled = re.compile(member)
    except re.error:
        return None  # e.g. a leading "(?u)": legal alone, not inside a group
    return member if compiled.flags == pattern.flags else None


class DetectorPlan:
    """A detector list compiled for detection on one string at a time.

    A request body is walked string by string, and on a short string every
    rule costs about a microsecond of interpreter overhead even when none
    can match — a body of 300,000 tiny strings once kept the event loop
    busy for over half a minute. On a short text the plan first decides
    which detectors could fire at all, then runs only those, in list order,
    through the unchanged per-detector code. The output is identical by
    construction:

    * a regex rule with a ``required`` CNF is skipped only when no literal
      of one of its groups occurs in that rule's own haystack — exactly the
      texts its prefilter returns nothing for (a literal occurs only where
      its first character does, so indexing literals by first character
      and looking up the text's characters finds every one);
    * the regex rules without literals whose patterns can be alternation
      members share one combined search: an alternation matches nowhere in
      a text exactly when none of its members does, and a rule whose
      pattern matches nowhere detects nothing;
    * every other detector — deny strings, NER backends, plugin detectors,
      rules whose patterns cannot be combined — always runs.

    The differential tests run the gated path against every detector's full
    scan, and against a plan whose gate lies, to prove they would notice.
    """

    def __init__(
        self, detectors: Sequence[Detector], *, gated_max_chars: int = GATED_MAX_CHARS
    ) -> None:
        self.source = detectors
        self.detectors = tuple(detectors)
        self.gated_max_chars = gated_max_chars
        self._runners = tuple(_runner(det) for det in self.detectors)
        # first character -> literal -> indices of the rules it gates, for
        # case-sensitive literals (digit-folded haystack) and case-
        # insensitive ones (lowered haystack).
        self._cs: dict[str, dict[str, list[int]]] = {}
        self._ci: dict[str, dict[str, list[int]]] = {}
        always: list[int] = []
        members: list[str] = []
        member_indices: list[int] = []
        for index, det in enumerate(self.detectors):
            if isinstance(det, RegexDetector) and det.required:
                gate = _gate_group(det.required)
                if "" in gate:
                    always.append(index)  # an empty literal is always present
                    continue
                table = self._ci if det.rule.required_ci else self._cs
                for lit in gate:
                    table.setdefault(lit[0], {}).setdefault(lit, []).append(index)
                continue
            member = _combinable(det.rule.pattern) if isinstance(det, RegexDetector) else None
            if member is None:
                always.append(index)
            else:
                members.append(member)
                member_indices.append(index)
        self._always = frozenset(always)
        self._members = tuple(member_indices)
        self._combined = re.compile("|".join(members)) if members else None
        self._every = tuple(range(len(self.detectors)))

    def candidates(self, prepared: PreparedText) -> tuple[int, ...]:
        """Indices, in list order, of the detectors that could fire on the
        text; the others provably detect nothing in it."""
        chosen = set(self._always)
        for table, haystack in (
            (self._cs, prepared.folded if self._cs else ""),
            (self._ci, prepared.lower if self._ci else ""),
        ):
            for first in table.keys() & set(haystack):
                for lit, indices in table[first].items():
                    if lit in haystack:
                        chosen.update(indices)
        if self._combined is not None and self._combined.search(prepared.text) is not None:
            chosen.update(self._members)
        return tuple(sorted(chosen))

    def detect(self, text: str, allowlist: Allowlist) -> list[Detection]:
        # One PreparedText per string: regex detectors share it so their
        # required-literal prefilters (and the derived haystacks behind
        # them) are computed once, not per rule.
        prepared = PreparedText(text)
        order = self._every if len(text) > self.gated_max_chars else self.candidates(prepared)
        return self._run(order, prepared, allowlist)

    def detect_each(self, text: str, allowlist: Allowlist) -> list[Detection]:
        """Every detector, ungated: the reference the gated path equals."""
        return self._run(self._every, PreparedText(text), allowlist)

    def _run(
        self, order: Iterable[int], prepared: PreparedText, allowlist: Allowlist
    ) -> list[Detection]:
        detections: list[Detection] = []
        runners = self._runners
        allows = allowlist.allows_for
        for index in order:
            found = runners[index](prepared)
            # Tier-0 (deny) detections bypass the allowlist — global AND
            # per-type: deny is the user's explicit strongest signal, so a
            # deny/allowlist contradiction resolves in favor of redaction.
            detections.extend(
                d for d in found if d.tier == 0 or not allows(d.detector_type, d.value)
            )
        detections.sort(key=lambda d: (d.start, -(d.end - d.start), d.priority))
        return detections


# Plans by detector-list identity. An entry holds its list, so the id cannot
# be reused while it is cached, and the contents are compared on every
# lookup, so a list changed in place gets a fresh plan. Bounded: a reload
# builds a new list only when [detection] changed.
_PLANS: dict[int, DetectorPlan] = {}
_PLANS_MAX = 32


def plan_for(detectors: Sequence[Detector]) -> DetectorPlan:
    """The (cached) DetectorPlan for ``detectors``."""
    key = id(detectors)
    plan = _PLANS.get(key)
    if plan is not None and plan.source is detectors and plan.detectors == tuple(detectors):
        return plan
    plan = DetectorPlan(detectors)
    _PLANS.pop(key, None)
    if len(_PLANS) >= _PLANS_MAX:
        del _PLANS[next(iter(_PLANS))]  # the oldest entry
    _PLANS[key] = plan
    return plan
