"""Known-equivalent mutants: mutations that cannot change observable behavior.

Mutation testing's honest tail. A surviving mutant is either a real test gap
(fix it with an assertion) or an EQUIVALENT mutant — a source change that
provably cannot alter any observable behavior, so no test can kill it. The
`mutation` CI gate (scripts/mutation_gate.py) treats a survivor as a failure
UNLESS its id appears here with a justification: every survivor is either
killed or explicitly, reviewably declared equivalent. Nothing survives
silently, and stale entries (mutants that stop surviving) fail the gate too.

The dominant class is SQL/PRAGMA case changes: SQLite keywords, identifiers,
and PRAGMA names are case-insensitive, so `select ... from mappings` is
byte-for-byte equivalent to the engine. The rest are hand-reviewed logic
equivalences (parity-invariant arithmetic, falsy-None-for-False flags,
idempotent re-stamps, amortization-cadence shifts under an unchanged bound)
and the one deliberately unreachable defensive branch.

mutmut mutant ids are positional within a source file, so they are stable
while the mutated module is unchanged. Editing a mutated module renumbers
them — which is the point: the gate flags the drift and the equivalence
claims get re-reviewed against fresh diffs.

OSCILLATING_MUTANTS is the one nuance: mutmut selects tests per mutant from
coverage recorded during the stats run, and hypothesis-driven tests cover
different mutants on different runs — so a handful of PROVABLY equivalent
mutants (codec-case renames, redundant guards) flip between "survived" and
"killed" run to run; the sporadic "kills" are selection artifacts, not real
distinguishing inputs. The same applies to "survived"/"timeout" flips: the
per-mutant time limit is timing-based, so a slow runner can tip an
equivalent mutant's (behaviorally identical) test run over it. Those ids
stay justified in EQUIVALENT_MUTANTS and are exempted ONLY from the gate's
staleness check — an oscillating entry may legitimately be absent from one
run's survivor list. Every other entry must match the run exactly.
"""

# id -> why no test can kill it (reviewed against the mutant's diff)
EQUIVALENT_MUTANTS: dict[str, str] = {
    "llm_redact.detection.validators.x__b64url_json_object__mutmut_7": (
        "padding (-len % 4) -> (+len % 4): base64 leniently ignores surplus '=', "
        "and a VALID base64 segment never has length % 4 == 1 (the only length "
        "class where the two paddings diverge in effect). For every reachable "
        "segment the decoded bytes are identical."
    ),
    "llm_redact.detection.validators.x__entropy__mutmut_19": (
        "bits >= 3.5 -> bits > 3.5: the two differ only when the Shannon estimate "
        "is EXACTLY 3.5 bits/char. No realistic secret hits that float boundary "
        "exactly; the threshold is a soft heuristic, not a hard equality."
    ),
    "llm_redact.detection.validators.x__jwt__mutmut_11": (
        "value.split('.', 2) -> value.split('.'): _JWT_RE.fullmatch already "
        "guarantees exactly two dots, so maxsplit is never reached — identical "
        "3-element result."
    ),
    "llm_redact.detection.validators.x__jwt__mutmut_12": (
        "split('.', 2) -> rsplit('.', 2): with exactly two dots (enforced by "
        "_JWT_RE), left- and right-split give the identical partition."
    ),
    "llm_redact.detection.validators.x__jwt__mutmut_14": (
        "split('.', 2) -> split('.', 3): the string has exactly two dots, so a "
        "larger maxsplit produces the identical 3-element result."
    ),
    "llm_redact.detection.validators.x__mod97__mutmut_2": (
        "match.group(0).upper() -> .lower(): _to_mod97_number maps letters with "
        "int(c, 36), which is case-insensitive, and str.isalnum() is too. Every "
        "input yields the identical MOD-97 number. The .upper() is defensive, not "
        "load-bearing."
    ),
    "llm_redact.redactor.x__resolve_overlaps__mutmut_2": (
        "fast-path predicate all(tier != 0) -> all(tier == 0): the fast path is a "
        "pure optimization of the two-phase sweep. With only tiers {0, 1} "
        "present, the two-phase path degenerates to _sweep(all) whenever the tier "
        "set is uniform, so routing to it produces the identical result."
    ),
    "llm_redact.redactor.x__resolve_overlaps__mutmut_3": (
        "fast-path predicate all(tier != 0) -> all(tier != 1): same reasoning — "
        "only tiers {0, 1} exist, and the two-phase path already computes the "
        "correct result for every tier composition, so a different route to it is "
        "observationally identical."
    ),
    "llm_redact.redactor.x__sweep__mutmut_4": (
        "last_end = -1 -> -2: a sentinel below every real detection start (>= 0). "
        "The first comparison `d.start >= last_end` is True for both, so no valid "
        "detection stream distinguishes them."
    ),
    "llm_redact.redactor.xǁRedactorǁredact_text__mutmut_11": (
        "cursor = 0 -> cursor = None: cursor is used only as a slice start, and "
        "text[None:i] == text[0:i]. It is reassigned to an int (d.end) after the "
        "first detection, so None only ever stands in for 0 at the first slice."
    ),
    "llm_redact.redactor.xǁRedactorǁredact_text__mutmut_14": (
        "tier-0 mode literal 'redact' -> 'XXredactXX': mode is only compared to "
        "'block' and 'warn'; any other value takes the redact branch, so the "
        "exact spelling is irrelevant."
    ),
    "llm_redact.redactor.xǁRedactorǁredact_text__mutmut_15": (
        "tier-0 mode literal 'redact' -> 'REDACT': same — only 'block'/'warn' are "
        "matched; every other value redacts."
    ),
    "llm_redact.redactor.xǁRedactorǁredact_text__mutmut_19": (
        "_modes.get(type, 'redact') -> .get(type, None): the default is only used "
        "for a type absent from _modes, and None (like 'redact') is neither "
        "'block' nor 'warn', so it takes the redact branch identically."
    ),
    "llm_redact.redactor.xǁRedactorǁredact_text__mutmut_21": (
        "_modes.get(type, 'redact') -> .get(type): dropping the default makes it "
        "None, which redacts exactly as 'redact' does (see mutmut_19)."
    ),
    "llm_redact.redactor.xǁRedactorǁredact_text__mutmut_22": (
        "default 'redact' -> 'XXredactXX': a non-'block'/'warn' default redacts identically."
    ),
    "llm_redact.redactor.xǁRedactorǁredact_text__mutmut_23": (
        "default 'redact' -> 'REDACT': a non-'block'/'warn' default redacts identically."
    ),
    "llm_redact.rehydrate.x_escape_prefix_start__mutmut_24": (
        "(i - j) % 2 -> (i + j) % 2: i-j and i+j always share parity, so the "
        "literal-backslash-pair test is unchanged for every input."
    ),
    "llm_redact.rehydrate.x_substitute_tokens__mutmut_19": (
        "counts key canonical[1:-1] -> [1:-2]: the slice feeds "
        "rpartition('_')[0], and dropping the final digit of the counter never "
        "changes the text before the LAST underscore — the type key is identical "
        "for every canonical token (numbers are >= 1 digit). NOTE: oscillates "
        "between killed/survived across mutmut runs (coverage-based test "
        "selection); the justification holds either way."
    ),
    "llm_redact.rehydrate.xǁRehydratorǁrehydrate_text__mutmut_4": (
        "json_escape=False -> None: the flag is only truth-tested (`if "
        "json_escape:`), and None is falsy exactly like False."
    ),
    # build_cipher / cipher_from_key / _rdbms_backend_requires_pro: the paid
    # subsystem fail-closed ConfigErrors (open-core split). Each survivor below
    # mutates ONE segment of a multi-part message (XX-wrap or ASCII case flip)
    # while "llm-redact-pro" still appears in another segment, so the fail-closed
    # behavior AND the "error names the paid package" assertion are unchanged —
    # only the install-hint prose wording differs. Killing them would require a
    # brittle verbatim-message assertion; the wording is UX, not behavior.
    "llm_redact.vault.x_build_cipher__mutmut_5": (
        "ConfigError message wording only (XX-wrap of line 1); 'llm-redact-pro' "
        "still present, behavior unchanged."
    ),
    "llm_redact.vault.x_build_cipher__mutmut_6": (
        "ConfigError message wording only (case flip of line 1); the lowercase "
        "'llm-redact-pro' in line 2 remains, behavior unchanged."
    ),
    "llm_redact.vault.x_build_cipher__mutmut_7": (
        "ConfigError message wording only (XX-wrap of line 2); 'llm-redact-pro' "
        "still present, behavior unchanged."
    ),
    "llm_redact.vault.x_build_cipher__mutmut_8": (
        "ConfigError message wording only ('Free'->'free' in line 2); behavior unchanged."
    ),
    "llm_redact.vault.x_build_cipher__mutmut_9": (
        "ConfigError message wording only (case flip of line 2); the lowercase "
        "'llm-redact-pro' in line 1 remains, behavior unchanged."
    ),
    "llm_redact.vault.x_build_cipher__mutmut_10": (
        "ConfigError message wording only (XX-wrap of the trailing line); behavior unchanged."
    ),
    "llm_redact.vault.x_build_cipher__mutmut_11": (
        "ConfigError message wording only (case flip of the trailing line); behavior unchanged."
    ),
    "llm_redact.vault.x_cipher_from_key__mutmut_2": (
        "ConfigError message wording only (XX-wrap); 'llm-redact-pro' still "
        "present, the fail-closed raise is unchanged."
    ),
    "llm_redact.vault.x__rdbms_backend_requires_pro__mutmut_2": (
        "ConfigError message wording only (XX-wrap of the second line); "
        "'llm-redact-pro' still present and the backend name in line 1 is "
        "unaffected, behavior unchanged."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_15": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_16": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_18": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_20": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_3": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_30": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_31": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_34": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_35": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_38": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_39": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_45": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_49": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_50": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_53": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_56": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_57": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_60": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_8": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__migrate_to_v3__mutmut_9": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__open_connection__mutmut_28": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__open_connection__mutmut_29": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__open_connection__mutmut_32": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__open_connection__mutmut_33": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__open_connection__mutmut_36": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__open_connection__mutmut_37": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__open_connection__mutmut_42": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__open_connection__mutmut_43": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__open_connection__mutmut_53": (
        "version < 2 -> <= 2 for the user_version stamp: at version == 2 the "
        "extra `PRAGMA user_version = 2` rewrites the value it already has — "
        "idempotent, no observable change."
    ),
    "llm_redact.vault.x__open_connection__mutmut_54": (
        "version < 2 -> < 3 for the user_version stamp: same idempotent re-stamp "
        "— a v2 database is stamped 2 again."
    ),
    "llm_redact.vault.x__open_connection__mutmut_57": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__open_connection__mutmut_58": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__verify_key__mutmut_10": (
        "inside _verify_key's `row is None` defensive branch (pragma: no cover): "
        "key_check is always written at creation/migration, so the branch never "
        "executes and no test can reach the mutation. The branch-CONDITION "
        "mutants ARE killed (test_wrong_key_fails_at_open_even_on_empty_vault)."
    ),
    "llm_redact.vault.x__verify_key__mutmut_11": (
        "inside _verify_key's `row is None` defensive branch (pragma: no cover): "
        "key_check is always written at creation/migration, so the branch never "
        "executes and no test can reach the mutation. The branch-CONDITION "
        "mutants ARE killed (test_wrong_key_fails_at_open_even_on_empty_vault)."
    ),
    "llm_redact.vault.x__verify_key__mutmut_12": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__verify_key__mutmut_13": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__verify_key__mutmut_4": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x__verify_key__mutmut_7": (
        "inside _verify_key's `row is None` defensive branch (pragma: no cover): "
        "key_check is always written at creation/migration, so the branch never "
        "executes and no test can reach the mutation. The branch-CONDITION "
        "mutants ARE killed (test_wrong_key_fails_at_open_even_on_empty_vault)."
    ),
    "llm_redact.vault.x__verify_key__mutmut_8": (
        "inside _verify_key's `row is None` defensive branch (pragma: no cover): "
        "key_check is always written at creation/migration, so the branch never "
        "executes and no test can reach the mutation. The branch-CONDITION "
        "mutants ARE killed (test_wrong_key_fails_at_open_even_on_empty_vault)."
    ),
    "llm_redact.vault.x__verify_key__mutmut_9": (
        "inside _verify_key's `row is None` defensive branch (pragma: no cover): "
        "key_check is always written at creation/migration, so the branch never "
        "executes and no test can reach the mutation. The branch-CONDITION "
        "mutants ARE killed (test_wrong_key_fails_at_open_even_on_empty_vault)."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_11": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_12": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_20": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_21": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_23": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_24": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_3": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_34": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_35": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_38": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_39": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_45": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_49": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_52": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_55": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_56": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_59": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_8": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.x_rotate_vault_key__mutmut_9": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁget__mutmut_17": (
        "popitem(last=False) -> last=None: OrderedDict.popitem truth-tests "
        "`last`, and None is falsy exactly like False — still pops the LRU end."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁlookup_response_session__mutmut_7": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁlookup_response_session__mutmut_8": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁprune_sessions__mutmut_10": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁprune_sessions__mutmut_26": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁprune_sessions__mutmut_37": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁprune_sessions__mutmut_40": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁprune_sessions__mutmut_9": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁrecord_response_session__mutmut_10": (
        "_response_inserts += 1 -> += 2: prunes every 128th insert instead of "
        "every 256th — a strictly more frequent amortization with the same bound "
        "(pinned by test_response_session_map_stays_bounded)."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁrecord_response_session__mutmut_19": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁrecord_response_session__mutmut_20": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁrecord_response_session__mutmut_22": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁrecord_response_session__mutmut_23": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁrecord_response_session__mutmut_6": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁrecord_response_session__mutmut_7": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁsession_count__mutmut_4": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁsession_count__mutmut_5": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁsessions_summary__mutmut_4": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁsessions_summary__mutmut_5": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁsessions_summary__mutmut_7": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁsessions_summary__mutmut_8": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁtotal_entries__mutmut_4": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultManagerǁtotal_entries__mutmut_5": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁ__init____mutmut_14": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁ__init____mutmut_15": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁ__init____mutmut_24": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁ__init____mutmut_25": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁoriginal_for__mutmut_11": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁoriginal_for__mutmut_12": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁoriginal_for__mutmut_23": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁoriginal_for__mutmut_24": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_14": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_15": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_17": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_18": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_36": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_42": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_43": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_45": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_47": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_57": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_60": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_68": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_69": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_7": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_71": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_72": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_79": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_80": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_82": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_83": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    "llm_redact.vault.xǁSqliteVaultǁplaceholder_for__mutmut_97": (
        "SQL/PRAGMA case change only: SQLite keywords, identifiers, and PRAGMA "
        "names are case-insensitive, so the statement is byte-for-byte equivalent "
        "to the engine."
    ),
    # --- Round 2 (1.16.0): codec + session-router equivalents ---
    "llm_redact.eventstream.x__parse_headers__mutmut_14": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_15": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_16": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_21": (
        "codec-case: utf-8 to UTF-8; Python normalizes codec names, identical bytes."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_39": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_40": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_41": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_54": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_55": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_56": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_71": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_72": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_73": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_81": (
        "codec-case: utf-8 to UTF-8; Python normalizes codec names, identical bytes."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_87": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_88": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__parse_headers__mutmut_89": (
        "error-msg-text: EventStreamError message text change (None / XX-wrap / "
        "case); same type raised at the same condition, message never asserted."
    ),
    "llm_redact.eventstream.x__serialize_headers__mutmut_10": (
        "error-msg-text: 'header name over 255 bytes' message change; never asserted."
    ),
    "llm_redact.eventstream.x__serialize_headers__mutmut_28": (
        "codec-case: utf-8 to UTF-8; Python normalizes codec names, identical bytes."
    ),
    "llm_redact.eventstream.x__serialize_headers__mutmut_39": (
        "error-msg-text: 'unknown header value type' message to None; serialize-side "
        "raise, message never asserted."
    ),
    "llm_redact.eventstream.x__serialize_headers__mutmut_5": (
        "codec-case: utf-8 to UTF-8; Python normalizes codec names, identical bytes."
    ),
    "llm_redact.eventstream.x__serialize_headers__mutmut_8": (
        "error-msg-text: 'header name over 255 bytes' message change; never asserted."
    ),
    "llm_redact.eventstream.x__serialize_headers__mutmut_9": (
        "error-msg-text: 'header name over 255 bytes' message change; never asserted."
    ),
    "llm_redact.eventstream.x_serialize__mutmut_15": (
        "int-encoding-equiv: prelude pack >II to >ii; total and header-count are "
        "small positive ints (<< 2^31), so signed/unsigned pack byte-identically -> "
        "same CRC and downstream."
    ),
    "llm_redact.eventstream.xǁEventStreamParserǁfeed__mutmut_28": (
        "error-msg-text: 'prelude CRC mismatch' XX-wrapped; the test's match='prelude "
        "CRC' still matches, nothing else observes it."
    ),
    "llm_redact.eventstream.xǁEventStreamParserǁfeed__mutmut_49": (
        "error-msg-text: 'message CRC mismatch' XX-wrapped; match='message CRC' still matches."
    ),
    "llm_redact.multipart.x_parse__mutmut_47": (
        "unreachable-off-by-one: end<0 to end<=0; end = rest.find(delim, 2) is -1 or "
        ">=2, never 0/1, so both predicates select the same set."
    ),
    "llm_redact.multipart.x_parse__mutmut_48": (
        "unreachable-off-by-one: end<0 to end<1; end is never 0, so identical."
    ),
    "llm_redact.multipart.x_parse_boundary__mutmut_28": (
        "codec-case: encode('ascii') to 'ASCII'; codec name normalized."
    ),
    "llm_redact.placeholders.x_viable_prefix_start__mutmut_10": (
        "redundant-guard: disabling the early guillemet-close return; a closing "
        "guillemet in tail is in body and matches neither the body-char class nor "
        "the fuzzy interior grammar, so the body/interior check below already "
        "returns None. Pure optimization. NOTE: oscillates between killed/survived "
        "across mutmut runs (coverage-based test selection); the justification "
        "holds either way."
    ),
    "llm_redact.sse.x_serialize__mutmut_17": ("codec-case: .encode('utf-8') to 'UTF-8'."),
    "llm_redact.sse.xǁSSEParserǁclose__mutmut_10": (
        "unreachable-branch: the byte-buffer guard is never true at close() -- the "
        "preceding feed of a newline always drains the buffer to empty -- so these "
        "append mutants cannot be reached."
    ),
    "llm_redact.sse.xǁSSEParserǁclose__mutmut_5": (
        "unreachable-branch: the byte-buffer guard is never true at close() -- the "
        "preceding feed of a newline always drains the buffer to empty -- so these "
        "append mutants cannot be reached."
    ),
    "llm_redact.sse.xǁSSEParserǁclose__mutmut_6": (
        "unreachable-branch: the byte-buffer guard is never true at close() -- the "
        "preceding feed of a newline always drains the buffer to empty -- so these "
        "append mutants cannot be reached."
    ),
    "llm_redact.sse.xǁSSEParserǁclose__mutmut_7": (
        "unreachable-branch: the byte-buffer guard is never true at close() -- the "
        "preceding feed of a newline always drains the buffer to empty -- so these "
        "append mutants cannot be reached."
    ),
    "llm_redact.sse.xǁSSEParserǁclose__mutmut_8": (
        "unreachable-branch: the byte-buffer guard is never true at close() -- the "
        "preceding feed of a newline always drains the buffer to empty -- so these "
        "append mutants cannot be reached."
    ),
    "llm_redact.sse.xǁSSEParserǁclose__mutmut_9": (
        "unreachable-branch: the byte-buffer guard is never true at close() -- the "
        "preceding feed of a newline always drains the buffer to empty -- so these "
        "append mutants cannot be reached."
    ),
    "llm_redact.sse.xǁSSEParserǁfeed__mutmut_26": (
        "codec-case: .decode('utf-8','replace') to 'UTF-8'. NOTE: oscillates "
        "between killed/survived across mutmut runs (coverage-based test "
        "selection); the justification holds either way."
    ),
    "llm_redact.eventstream.xǁEventStreamParserǁfeed__mutmut_19": (
        "same-exception-equiv: the headers_len sanity bound (> total-16) to "
        "(> total+16); for every divergence input (headers_len in "
        "(total-16, total+16]) the byte offsets force _parse_headers onto a "
        "block that overruns the frame or ends in CRC bytes, so it raises "
        "EventStreamError just as the original length-check would. Both paths "
        "raise the SAME exception type -> identical verbatim-degrade downstream; "
        "only the never-asserted message text differs."
    ),
}

# Reviewed-equivalent ids that flip between survived/killed across mutmut runs
# (hypothesis coverage varies the per-mutant test selection), or between
# survived/timeout (the per-mutant time limit is timing-based, so a slow
# runner tips an equivalent mutant's identical-behavior test run over it).
# Exempt from the gate's STALENESS check only — they must still carry a
# justification above.
OSCILLATING_MUTANTS: frozenset[str] = frozenset(
    {
        "llm_redact.eventstream.x__parse_headers__mutmut_81",
        "llm_redact.placeholders.x_viable_prefix_start__mutmut_10",
        "llm_redact.rehydrate.x_substitute_tokens__mutmut_19",
        "llm_redact.sse.xǁSSEParserǁfeed__mutmut_26",
    }
)
