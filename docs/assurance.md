# Assurance: proving the suites have teeth

Phase 18 ([security-testing.md](security-testing.md)) proved the guarantees
are *correct*. Phase 19 ([resilience.md](resilience.md)) proved they *hold
under fault*. Both rest on a premise that green checkmarks alone cannot
establish: **that the suites would actually FAIL if a guard were silently
weakened or a checksum table corrupted.** This document is the record of
measuring that premise — mutation testing over the load-bearing core, one
falsifiable statement of "never a wrong value", differential fuzzing up
through the JSON body layer, and byte-reproducible builds.

## Mutation testing

`mutmut` (dev-only; the request path gains no dependency) generates
mutants — small semantic edits: flipped comparisons, dropped arguments,
altered constants and strings — and runs the fast suite against each. A
mutant the suite fails to kill is either a real test gap or a provably
equivalent mutation. The contract, enforced by the gating `mutation` CI
job:

> **Every mutant is killed, or its id is recorded in
> `scripts/mutation_equivalents.py` with a reviewed justification.
> Nothing survives silently — and a stale allow-list entry (a mutant that
> stopped surviving) fails the gate too, so dead justifications cannot
> accumulate.**

`scripts/mutation_gate.py` checks both directions after `mutmut run`; the
run step itself is non-fatal because mutmut exits non-zero on the reviewed
equivalents — the gate script is the pass/fail arbiter.

### Scope and baseline

The recurring scope (`[tool.mutmut]` in pyproject.toml) is the load-bearing
correctness core — the modules where a silent weakening becomes a leak or a
wrong value. Round 1 (1.12.0) covered `detection/validators.py`,
`redactor.py`, `rehydrate.py`, and `vault.py`; round 2 (1.16.0) added the
byte-level codec trust boundary and session routing: `sse.py`, `ndjson.py`,
`jsonwalk.py`, `placeholders.py`, `multipart.py`, `eventstream.py`, and
`sessions.py`. Round-1 baseline: **1100 mutants — 968 killed, 130 reviewed
equivalents** (plus one timeout, which counts as caught, and one
unreferenced-helper placeholder).

Round 2 added **140 survivors across the seven new files — 78 killed, 62
reviewed equivalents** (`ndjson.py` produced zero survivors: its
split-at-every-offset sweep already kills them all). The kills were not
cosmetic. The crown-jewel finds were in `sessions.py`'s `resolve()`
entry point — the GET-`/v1/responses/{id}` retrieval branch, the
`/v1/conversations` and Gemini batch/cache static-session guards, and the
empty-`previous_response_id` case were all untested *at the `resolve()`
call*, and several mutants there route a request to a **different vault
session** (a wrong-value leak); and in `jsonwalk.py`, where dropping
`skip_keys`/`key_overrides` in the list, dict, and `object:list` recursion
branches would redact a protected realtime `audio`/enum field or break the
Responses `arguments` JSON-source escaping. The session-router mutants are now
pinned in the pro repo (`test_sessions_pro.py`); the codec mutants stay here in
`tests/test_codec_mutation_kills.py`.

**R2 relocation (3.12.0) → R4 physical split (4.0.0).** The open-core split moved
the per-conversation session router out of the Free core into the paid package
(Free keeps only the static router). Through 3.15.0 the package was co-located in
this repo, so its mutation coverage rode the in-repo mutmut config
(`src/llm_redact_pro` a `source_path`, `sessions.py` in `only_mutate`). R4
physically relocated the package to the private `llm-redact-pro` repo: the
session-router mutation coverage (the same `resolve()`/`_canonical`/
`orphan_session_id` mutants) now runs in that repo's CI, and this repo's mutmut
`source_paths`/`only_mutate` and `scripts/mutation_equivalents.py` cover only the
Free codecs and vault.

`vault.py` stays fully mutation-covered here, including its ENCRYPTED arms (the
memory-encrypted vault, the cipher-threading manager, key rotation's rollback
path, `open_sqlite_vault`'s ownership). The tests that killed those crypto-path
mutants used the real Fernet cipher and moved to the pro repo with the rest of
the crypto suite; `tests/test_vault_crypto_free.py` reproduces exactly that
mutation coverage over the Free-side `FakeVaultCipher`, so a weakened crypto arm
still fails a Free-side test even though the paid cipher lives elsewhere.

The equivalents fall into reviewed classes (each entry in
`scripts/mutation_equivalents.py` carries its specific justification):
SQL/PRAGMA case changes (SQLite keywords, identifiers, and PRAGMA names are
case-insensitive — the dominant class in round 1), logic equivalences
(parity-invariant arithmetic, falsy-`None`-for-`False` flags, idempotent
version re-stamps, amortization-cadence shifts under an unchanged bound),
`_verify_key`'s deliberately unreachable defensive branch, and mutants of
dead parameters. Round 2 added a few narrow classes: exception/log
**message-text** changes (the raise fires at the same condition; the text
is never asserted), **codec-case** (`utf-8` vs `UTF-8` — Python normalizes
codec names), an **int-encoding equivalence** (`>II` vs `>ii` on small
positive lengths packs byte-identically), and a **json-serialize
equivalence** on `sessions._canonical` — safe *only* because no test or
caller pins the exact session-id bytes, and its justification says so, so
a future test asserting a fixed `conv-` id correctly turns those mutants
back into kills. Mutant ids are positional per file: editing a mutated
module renumbers them, the staleness check flags the drift, and the
equivalence claims get re-reviewed against fresh diffs — that churn is the
point, not a defect.

One nuance the gate encodes explicitly: mutmut selects tests per mutant
from coverage recorded during its stats run, and hypothesis-driven tests
cover different mutants on different runs — so a handful of *provably*
equivalent mutants (codec-case renames, redundant guards) flip between
"survived" and "killed" run to run; the sporadic kills are selection
artifacts, not real distinguishing inputs. "Survived"/"timeout" flips are
the same artifact in a different coat: the per-mutant time limit is
timing-based, so a slow runner can tip an equivalent mutant's (behaviorally
identical) test run over it. Those ids are listed in
`OSCILLATING_MUTANTS` (in `scripts/mutation_equivalents.py`, each still
carrying its full justification) and are exempt from the staleness check
only — every other entry must match the run exactly, and an oscillating
survivor still needs its justification to pass.

### The proxy guard chain (one-off pass)

`proxy.py` (2477 mutants) is CI-prohibitive for the recurring gate, so its
request-guard chain got a targeted, recorded pass instead: **all 111
generated mutants** of `_guarded_post_json` (the layered
host → origin → CSRF-token → content-type → size-cap guard),
`_host_allowed`, `_origin_allowed`, `_allowed_hostnames`, and
`_read_capped` **were killed by the existing security-boundary suite**
(`test_security_boundaries.py`, the B1–B12 battery) — full killing power,
zero new tests needed. The same full-file pass left 861 survivors in the
rest of proxy.py (streaming plumbing, dashboard/config rendering,
bookkeeping); they are outside the assured scope and deliberately
untriaged — extending the recurring scope there would trade meaningful
signal for hours of CI.

### What the survivor-killing actually found

The value of the method is what fell out of triaging the original 260
survivors across the scoped core (40 in validators+redactor, 220 in
rehydrate+vault) — real, load-bearing gaps the existing
suites could not see (all pinned in `tests/test_mutation_hardening.py` and
extensions to the fault/rotate suites):

- **An EMPTY encrypted vault opened with the wrong key was guarded only by
  the `key_check` comparison.** Every existing wrong-key test used a
  non-empty vault, where preload decryption fails anyway — so a mutant that
  skipped the comparison (or re-stamped `key_check` under the wrong key,
  silently re-keying the vault) passed the suite.
- **The write-through cache contract was never asserted.** Cache-skipping
  mutants hid behind the DB fallback, which reconstructs the same answers
  via the `IntegrityError` recovery path. The hot-path-off-the-DB claim is
  now proven by issuing and resolving over a deliberately closed
  connection.
- **The encrypted in-memory counter was only ever tested with one value
  per type** — a counter frozen at 1 (a wrong-value token collision) went
  unnoticed.
- Checksum-validator floors and table entries, deny-tier overlap span
  algebra, escape-holdback boundary arithmetic, streaming-channel
  constructor defaults, flush/reuse state resets, LRU eviction edges, the
  bounded response-session map, prune rollback propagation, and the
  fail-closed double-fault path (INSERT and ROLLBACK both failing).

### Running it locally

```bash
uv run mutmut run                      # mutate the scoped core, run the fast suite
uv run mutmut results                  # list non-killed mutants
uv run mutmut show <mutant-id>         # one mutant's diff
uv run python scripts/mutation_gate.py # the CI verdict: kill-or-justify, no stale entries
```

A new survivor after a code change means the change weakened an assertion
path (write a killing test) or introduced a genuinely equivalent mutation
(add it to the allow-list with a justification a reviewer can check).

## "Never a wrong value" as one statement

Resilience commitment #1 used to be proven piecemeal. It is now ONE
falsifiable property — a hypothesis `RuleBasedStateMachine`
(`TestNeverWrongValue` in `tests/test_properties.py`): across any
interleaving of `(session, type, value)` writes over multiple sessions
(with type names deliberately shared, so every session has an
`«EMAIL_001»`), any mix of own/foreign/unknown/mangled tokens in a text,
any chunking of the stream, and any truncation point — a placeholder
restores to exactly the value its own session stored, or passes through
verbatim. Writes are deterministic and collision-free against a shadow
model, per-`(session, type)` counters are exactly `1..n` after every step,
streaming equals whole-text, and a truncated stream equals the whole-text
rehydration of the prefix. Write-path faults remain the sqlite battery's
job (`test_vault_faults.py`).

## Differential fuzzing, bottom to top

| Layer | Property | Pinned by |
| --- | --- | --- |
| Byte codecs (SSE, NDJSON, eventstream, multipart) | Parsers never lose bytes, never raise foreign exceptions, chunking never changes the parse | `test_codec_fuzz.py` |
| JSON body pipeline | redact → rehydrate is identity over arbitrary JSON bodies (real detectors, seeded with values verified to fire detection) | `test_properties.py` |
| jsonwalk skip semantics | Structural keys skipped, plaintext-document `data` walked, `object:list` `data` walked — checked against an independent reference walker sharing no code with jsonwalk | `test_properties.py` |
| Streaming channels | `RehydratorPool` channels stay isolated under interleaving: each channel's streamed output equals the whole-text rehydration of just its fragments | `test_properties.py` |

## Reproducible builds

Two builds of the same tree under a pinned `SOURCE_DATE_EPOCH` produce
byte-identical wheel and sdist artifacts — checked on every CI run by the
`reproducible build` job and documented in
[dependencies.md](dependencies.md#reproducible-builds). This closes the
loop with the SBOM and build-provenance attestations: the artifacts are
independently re-derivable from the source they claim to come from.
