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
- **Never break the tool.** Unrecognized traffic passes through verbatim.
  Fail closed only where the security goal is at stake (oversized bodies,
  block mode, disabled providers, bind policy, vault keys).
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
  without any extra installed.

## Adding a detection rule

1. Write the rule in `src/llm_redact/detection/` — prefix-anchored for
   vendor tokens; grouped-display-form + checksum validator for national
   ids (bare digit runs never fire; that bar is why Dutch BSN is absent).
2. Declare `required`/`anchors` prefilter literals ONLY with generator
   coverage — a wrong literal is a silent recall bug. The per-rule
   soundness tests and the fast-vs-naive differential suite must cover it.
3. Add a generator to `bench/corpus.py` (positives are generated at
   runtime, never committed).
4. Run `uv run python -m llm_redact.bench --check`. If the rule fires on
   `bench/fp_corpus/` files, either fix the rule or update
   `MANIFEST.toml` with a written justification for hits you judge
   legitimate.
5. Add the rule name to the `enabled` list in `config.example.toml` and a
   CHANGELOG entry.

## Diagrams

The architecture diagrams in the README are Mermaid sources under
`docs/diagrams/` with their rendered PNGs committed alongside (so the
README needs no toolchain). If you change a `.mmd`, re-render and commit
both:

```bash
scripts/render_diagrams.sh   # needs Node; see the header for sandboxed-Chromium setups
```

Web-UI screenshots (`docs/screenshots/`, referenced from the README ops
section) are captured against fixture traffic:

```bash
uv run --with playwright python scripts/capture_screenshots.py
```

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
