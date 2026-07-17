# Coding standards and style guide

How code in llm-redact is written. [CONTRIBUTING.md](CONTRIBUTING.md) covers
the mechanics (setup, the gates, the PR checklist); this document is the style
guide — the conventions a reviewer will hold a change to. The architectural
ground truth is [../CLAUDE.md](../CLAUDE.md); read it before touching the proxy
hot path.

The overriding principle: **write code that reads like the code around it.**
Match the surrounding file's naming, comment density, and idioms rather than
importing a different house style into one function.

## Formatting and linting (enforced)

- **`ruff format`** is the single formatter — no hand-formatting, no other
  tool. Run `uv run ruff format .`; CI runs `--check`.
- **`ruff check`** with the `E`, `F`, `I`, `UP`, `B`, `SIM` rule sets. Import
  order is ruff/isort's; do not hand-sort.
- **Line length is 100.** Wrap prose in comments/docstrings to roughly the same
  width.
- **`mypy` runs in `strict` mode** over `src/`. New code is fully typed —
  parameters, returns, and non-trivial locals. Optional third-party imports
  (spaCy, presidio, oracledb, …) are the only `ignore_missing_imports`
  exceptions, declared in `pyproject.toml`.

These four (`ruff check`, `ruff format --check`, `pytest`, `mypy`) plus
`python -m llm_redact.bench --check` are the gates every commit passes.

## Language and dependencies

- **Runtime dependencies stay at three: `httpx`, `starlette`, `uvicorn`.**
  Everything else the runtime needs comes from the standard library
  (`tomllib`, `argparse`, `dataclasses`, `hmac`/`hashlib`, `sqlite3`, …). A new
  capability that needs another package goes behind an **optional extra** (as
  `crypto`, `ner`, `realtime`, `otel`, and the vault drivers do) and must not
  touch the request path.
- **No pydantic, no FastAPI.** The proxy forwards unknown JSON fields verbatim;
  it must never validate or reshape a body it does not need to touch. Config
  parsing is hand-rolled over `tomllib` into dataclasses.
- **Prefer the standard library and explicit code** over a dependency or a
  clever abstraction. Vendored cryptographic primitives (Ed25519, Keccak-256)
  are pinned to their specifications' own test vectors — correctness rests on
  the published vectors, never on self-consistency.
- Target **Python 3.11+**, Linux and macOS. Windows is unsupported by design.

## Correctness invariants the code must uphold

These are not style preferences — they are the product's contract, and a change
that weakens one will be rejected regardless of how clean it looks.

- **Never break the tool.** Unrecognized traffic passes through **verbatim**.
  The proxy operates on the raw byte stream; never assume typed objects — that
  is the failure mode that broke other proxies.
- **Fail closed only where the security goal is at stake** — oversized bodies
  (413), block mode (400), disabled providers (502), bind policy, vault-key
  resolution, license gating. Everywhere else, degrade to pass-through rather
  than error.
- **Never a wrong value.** A token is only ever restored from the session that
  owns it; a miss passes through verbatim. Vault writes fail closed and roll
  back; counters are dense so a retry reissues the *same* number.
- **Never log or echo secrets.** Log lines carry paths, statuses, and detection
  counts — never header values, body content, query strings, DSNs, keys, or the
  matched secrets. **Error messages name positions or types, never values.**

## Comments and docstrings

- Comments explain **why**, not what — usually the *invariant* a line protects
  ("caches write only after COMMIT so nothing is poisoned", "flush leftovers
  become synthetic deltas *before* the stop event"). The codebase is densely
  commented at decision points; match that density where the reasoning is
  non-obvious, and stay quiet where the code speaks for itself.
- Module docstrings state the module's job and its load-bearing constraints.
- Reference code as ``file.py`` / `func()` and design docs by path; keep those
  pointers accurate when files move.

## Placeholders and the byte-level codecs

- Placeholders use **guillemets** (`«TYPE_NNN»`) precisely because they don't
  collide with code or markdown. Do not change the token format without
  updating the holdback logic, the fuzzy grammar, and the split sweeps together
  — they are a single coupled unit.
- SSE / NDJSON / eventstream / multipart are incremental **byte-level**
  parser/serializers. Untouched frames must re-serialize byte-identically;
  `feed()` never loses bytes on error. When in doubt, forward verbatim.

## Testing style

- The **load-bearing suites are the split-at-every-offset sweeps**
  (`test_rehydrate.py`, `test_sse.py`, and the eventstream/NDJSON/WS siblings):
  they cut a stream at every byte offset and assert streaming output equals
  non-streaming output. Touch `rehydrate.py`, `sse.py`, or an adapter's event
  handling → **extend a sweep**, don't add a single-case test.
- **Provider event fixtures are hand-authored**, never produced by the codec
  under test (a self-consistent bug would pass). Golden binary fixtures are
  assembled longhand.
- Integration tests run the real app in-process via `httpx.ASGITransport`;
  because it joins body parts into one chunk, chunk-boundary behavior needs a
  custom transport (see `_ChunkedUpstream`). Real sockets are the exception
  (mTLS, WS relay) — uvicorn on port 0 in a thread.
- Inject fakes for optional backends (spaCy/GLiNER/presidio models, OTel
  tracer/meter) so the base suite runs with no extras installed.
- **Property tests** (`hypothesis`, `test_properties.py`) probe the same
  invariants with random inputs — they extend, never replace, the sweeps.
- Assurance gates hold the suite honest: **mutation testing** (`mutmut` +
  `scripts/mutation_gate.py` — every surviving mutant is killed or recorded as a
  reviewed equivalent) and the **complexity-coverage gate** (every function with
  cyclomatic complexity > 1 is executed by the suite).

## Secret-shaped test data

Use **secret-SHAPED but verifiably fake** values in tests, fixtures, and docs —
vendors' canonical examples (`AKIAIOSFODNN7EXAMPLE`), alphabet runs, RFC-2606
domains (`corp.example`). Never a real credential, not even a revoked one. The
`bench/fp_corpus/` negatives and generated positives follow the same rule;
`scripts/history_sweep.py` audits history with the production detectors.

## Config, CLI, and ops surfaces

- Config edits and CLI output are **honest and value-free**: every deliberate
  coverage opt-out (warn mode, per-provider detection off, MCP exempt servers,
  language-scoped rules) is surfaced in `/status`, `doctor`, and the dashboard —
  never silent.
- The dashboard is self-contained package data with a strict CSP: **textContent
  DOM writes only**, no inline event handlers, no remote fetches.
- Reserved `/__llm-redact/*` endpoints are GET-only (with the three guarded
  POST exceptions) and provably never forwarded upstream.
