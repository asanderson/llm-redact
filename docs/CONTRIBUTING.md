# Contributing to llm-redact

Thanks for helping keep secrets out of LLM providers' hands. This guide
covers the mechanics; the **[coding standards and style guide](coding-standards.md)**
is how the code is written (formatting, typing, the correctness invariants,
testing style); and the architectural ground truth lives in
[CLAUDE.md](../CLAUDE.md) (yes, it doubles as the contributor architecture
reference — read it before touching the proxy hot path).

## Development setup

```bash
uv sync                 # installs runtime + dev dependencies
uv run llm-redact serve # run the proxy locally
uv run python scripts/fake_upstream.py --port 9999   # a fake provider for manual e2e
```

Python 3.11+ on Linux or macOS. Windows is unsupported (deliberately —
SIGHUP reload alone would sink it).

## The gates

Every commit must pass all of these; CI runs the same set:

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest
uv run mypy
uv run python -m llm_redact.bench --check
```

The bench gate enforces recall == 1.0 per detection rule against a
generated corpus AND exact false-positive counts against the vendored
`bench/fp_corpus/`. It is a functional regression gate, not a benchmark
you may skip.

## Hard rules

- **Runtime dependencies stay at three** (httpx, starlette, uvicorn). New
  capabilities that need more go behind an optional extra, like `crypto`,
  `ner`, and `realtime` do.
- **No pydantic, no FastAPI.** The proxy forwards unknown JSON fields
  verbatim; it must never validate or reshape a body it doesn't need to
  touch.
- **Never break the tool.** Unrecognized traffic passes through verbatim, to
  the provider it is positively attributed to (never a guessed one).
  Fail closed only where the security goal is at stake (oversized bodies,
  block mode, disabled providers, bind policy, vault keys, a body a
  recognized route cannot scan, a request from a web page).
- **Never log values.** Log lines carry paths, statuses, and detection
  counts — never header values, body content, query strings, or the
  matched secrets themselves. Error messages name positions or types,
  never values.

## Testing conventions

The load-bearing suites are the **split-at-every-offset sweeps**
(`test_rehydrate.py`, `test_sse.py`, and their eventstream/NDJSON/WS
siblings): they cut streams at every byte offset and assert streaming
output equals non-streaming output. When you touch `rehydrate.py`,
`sse.py`, or an adapter's event handling, **extend the sweeps** — a
single-case test proves almost nothing about chunk boundaries.

Other conventions worth knowing before writing tests:

- Integration tests run the real app in-process via `httpx.ASGITransport`
  — which joins response bodies into one chunk, so chunk-boundary
  behavior can't be pinned through it (see `_ChunkedUpstream` in
  `tests/test_provider_bedrock.py` for the workaround).
- Real sockets are the exception, not the rule: mTLS (`test_tls.py`) and
  WebSocket relay (`test_realtime_relay.py`) run uvicorn on port 0 in a
  thread because their transports can't ride ASGITransport.
- Provider event fixtures are hand-authored, never generated with the
  codec under test. Live-API drift tests (`-m live`) are deselected by
  default and double-gated on env vars + API keys.
- NER and OTel tests inject fake models/exporters so the suite runs
  without any extra installed. Tests that load a real NER model carry the
  `real_model` marker, deselected by default: they read the model from the
  local Hugging Face cache only (never a download) and skip when it or the
  model's extra is absent (`uv run pytest -m real_model`). The CI
  `ner-models` job pulls every model they load and sets
  `LLM_REDACT_TEST_REAL_MODELS_REQUIRED=1`, which makes such a skip a
  failure: a new real-model test needs its model in a config the job pulls
  (`bench/configs/` or `tests/real_model_configs/`, checked by
  `tests/test_ner_ci.py`).
- The session never touches your machine (`tests/isolation.py`, applied
  by `conftest.py` before collection): `HOME` and the XDG
  config/data/state dirs point into one throwaway directory — whatever
  your environment set, subprocesses included — and a deployment's
  `LLM_REDACT_*` variables (config file, vault DSN/key/key command,
  proxy URL, bind) are dropped; the test-control ones
  (`LLM_REDACT_TEST_*`, `LLM_REDACT_LIVE*`, the live model picks,
  `LLM_REDACT_PRO_*`, `LLM_REDACT_LICENSE_*`) are kept. The run FAILS
  when anything under the real llm-redact config/data dirs, the agent
  plugins' command dirs or the user service unit was created, changed or
  removed while it ran; a real proxy (or another checkout's test run) on
  the same machine writes there too, so
  `LLM_REDACT_TEST_ALLOW_REAL_DIR_CHANGES=1` skips that check. A test
  that needs a real-looking home builds one under `tmp_path`
  (`isolation_root` is the session's).

## Adding a detection rule

1. Write the rule in `src/llm_redact/detection/` — prefix-anchored for
   vendor tokens; grouped-display-form + checksum validator for national
   ids (bare digit runs never fire; that bar is why Dutch BSN is absent).
2. Declare `required`/`anchors` prefilter literals ONLY with generator
   coverage — a wrong literal is a silent recall bug. The per-rule
   soundness tests and the fast-vs-naive differential suite must cover it.
   The same literals gate the rule per string (`DetectorPlan`); a rule
   without literals joins one combined per-string search only if its
   pattern has no capturing group and no flag, and otherwise runs on every
   string — so prefer `(?:...)` groups. `tests/test_detector_plan.py`
   proves the gate against every detector's full scan, rule by rule.
3. Add a generator to `bench/corpus.py` (positives are generated at
   runtime, never committed).
4. Run `uv run python -m llm_redact.bench --check`. If the rule fires on
   `bench/fp_corpus/` files, either fix the rule or update
   `MANIFEST.toml` with a written justification for hits you judge
   legitimate.
5. Add the rule name to the `enabled` list in `config.example.toml` and a
   CHANGELOG entry.

## Adding an NER model or backend

Models are measured, not trusted: a model or backend lands with its facts,
its pins, its measurements and its documentation in the same change.

1. **License and lineage.** Read the model card (and its base model's, and
   the cards of the datasets it was trained on). Record facts only — the
   license identifier, "not OSI-approved", the backbone, "trained on
   <dataset> (<license>)", undisclosed training data — never a legal or
   procurement conclusion (`tests/test_model_catalog.py` rejects
   conclusion words). Known restricted lineage (non-commercial data,
   Llama-derived data) keeps a model from "vetted".
2. **Catalog entry** in `src/llm_redact/detection/model_catalog.py`: status,
   SPDX license, one neutral reason line with the card link and check date,
   lineage tags from `LINEAGE_TAGS`, the `main` commit on the check date as
   `revision` (a base model's commit as `backbone_revision` when the
   checkpoint ships no tokenizer or encoder configuration), recommended
   entities, label map or GLiNER prompt overrides, ONNX files, tagging and
   window. A new backend gets its `DEFAULT_MODELS` entry, its allowed file
   patterns in `model_files.py`, its `models` CLI and doctor coverage, and
   the air-gap rules: with downloads off it opens no connection.
3. **Bench configuration**: `bench/configs/<name>.toml` with
   `[detection.ner] enabled = true`, the backend and, unless it is the
   backend's default, the model; downloads stay off (the CI job pulls the
   model first). A `<backend>-default` name is kept for a backend's default
   model and names none (`tests/test_ner_ci.py`). Score it on the synthetic corpus, the negatives corpus and
   for latency ([ner-bench.md](ner-bench.md)), and on the published
   datasets that cover its types.
4. **Thresholds and ceilings**: record its baselines in
   `bench/ner_thresholds.toml` and `bench/ner_ceilings.toml` (dates,
   revisions and measured values in the notes; tolerances as in
   [ner-bench.md](ner-bench.md#recording-a-baseline)).
   `tests/test_ner_ci.py` requires an entry for every bench configuration.
5. **CI**: the `ner-models` job runs every `bench/configs/*.toml` with no
   edit; add the configuration's name to the matrix of the weekly
   `ner-eval` workflow (`.github/workflows/ner-eval.yml`, one job per
   configuration; `tests/test_ner_ci.py` fails until it is listed).
   A model too slow for a runner goes to `bench/configs/manual/` instead
   (measured by hand; its comment and baseline note say how).
   A real-model test of another model needs that model in a config under
   `tests/real_model_configs/`, or the job fails the skipped test; a slow
   model can still get a smoke test on a few short strings there, as
   `openai/privacy-filter` does, and a large one is removed from the model
   cache after the tests, as the job does for that model
   ([ner-bench.md](ner-bench.md), "CI").
6. **Admission.** "vetted" needs an OSI-approved weights license, no
   known restricted lineage (undisclosed training data alone does not rule
   a model out; the catalog states it), and on the bench: PERSON recall
   ≥ 0.85 and character-leak rate ≤ 0.15 on the synthetic corpus, at most
   one false positive per 50 KB on the agent-traffic negatives for its
   recommended entities, and p50 ≤ 100 ms for a 500-character string on
   the reference CPU (name the CPU). Otherwise "caution", with the numbers
   shown. A backend's default model changes only by a maintainer's
   decision.
7. **Docs**: the model-catalog tables (vetted, caution, restricted) in
   `docs/detection.md` (pinned to the catalog by
   `tests/test_model_sources.py`), `docs/ner-landscape.md` (the dated verdict and its numbers), a
   `docs/troubleshooting.md` entry for every new warning or error text,
   `config.example.toml` for any new key, and a CHANGELOG entry.
8. **Extras**: a new library goes behind an extra in `pyproject.toml`
   (with the torch floor if it loads torch weights), `uv lock`, and rows in
   `docs/dependencies.md` and `docs/SBOM.md` (both test-pinned); add a mypy
   override if the package has no type hints.
9. **Diagrams**: update `docs/diagrams/ner-pipeline.mmd` for a new path
   through a backend (windows, decoding) and `ner-bench.mmd` for a new
   dataset kind or gate, then re-render (below).

## Diagrams

The architecture diagrams in the README are Mermaid sources under
`docs/diagrams/` with their rendered PNGs committed alongside (so the
README needs no toolchain). Diagrams that show a data flow or a message
exchange also carry a `%% animate` comment, and `scripts/animate_diagrams.py`
renders an animated SVG (CSS only, so it plays when embedded as an image
and stays a complete static picture under `prefers-reduced-motion`) plus a
GIF of it; the docs pages embed the SVG, the README keeps the PNG. If you
change a `.mmd`, re-render and commit everything:

```bash
scripts/render_diagrams.sh   # needs Node + uv; see the header for sandboxed-Chromium setups
```

The agent-plugin terminal screenshots (`docs/screenshots/plugins/`,
referenced from docs/plugins.md) are rendered against fixture traffic:

```bash
uv run python scripts/capture_plugin_screenshots.py
```

The browser dashboard is part of the llm-redact-pro package, so its
screenshots and their capture script live in that repo.

## Licensing of contributions (CLA)

llm-redact is free software under the [GNU AGPL-3.0](../LICENSE), and the
same maintainer ships the proprietary `llm-redact-pro` package built on
this core — a dual-license model that requires consolidated rights in
the core. Contributions are therefore accepted under the project's
[Contributor License Agreement](CLA.md): you keep ownership of your
work and it stays available to everyone under the AGPL; the CLA
additionally lets the maintainer license it on other terms so the model
keeps working.

Sign off every commit with your real name and email:

```bash
git commit -s
```

For this project the `Signed-off-by:` line indicates agreement to the
[CLA](CLA.md) — not only the Developer Certificate of Origin — for the
changes in that pull request.

## Pull requests

- Keep commits self-contained and green (all gates above).
- Sign off every commit (`git commit -s`) — see the CLA section above.
- Add a `CHANGELOG.md` entry under `[Unreleased]` with the change that
  introduces it.
- Use secret-SHAPED but verifiably fake values in tests and docs —
  vendors' canonical examples (`AKIAIOSFODNN7EXAMPLE`), alphabet runs,
  RFC-2606 domains (`corp.example`). Never real ones, not even revoked
  ones.
- Do not report security vulnerabilities in issues or PRs — see
  [SECURITY.md](SECURITY.md) for private reporting.
