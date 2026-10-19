# Open WebUI validation and documentation follow-up

> Execute the approved recommendations in this session. Review the whole branch before opening the follow-up PR.

**Goal:** Close the browser and persisted-provider test gaps in PR #120 and include the optional frontend in relevant documentation and diagrams.

**Architecture:** Extend the existing disposable Docker smoke project. A dev-only Playwright client uses the real browser frontend; an actual saved WebUI configuration row points at the synthetic upstream before restart. Check the effective connection and upstream logs to prove the environment policy wins. Keep the proxy runtime unchanged.

**Spec:** User-approved follow-ups in this conversation; supported deployment and boundaries in `docs/openwebui.md`.

## Constraints and review focus

- Single trusted local user and text chat; no core auth, runtime dependencies, or hot-path changes.
- Synthetic canaries, dummy provider keys, disposable credentials/projects/volumes only.
- A working WebSocket handshake and restored assistant answer must be observed through Caddy; an echoed user message alone is not success.
- The stale URL must be a real database row and a reachable bypass target; merely changing in-memory defaults is insufficient.
- Preserve environment policy, loopback binds, backend egress isolation, and cleanup on failure.
- Browser dependencies are development/CI tools. No claimed validation on the owner's machine or real provider without access and credentials.

## Task 1: Persisted configuration regression

- [x] Add and run failing tests for an idempotent seed helper using a real temporary SQLite database with the pinned WebUI config schema.
- [x] Implement the fixture helper, then seed the disposable container database, restart WebUI, inspect its effective provider URL, and rerun redaction/restoration checks.
- [x] Run the focused tests and lint.

## Task 2: Browser acceptance

- [x] Add and run a failing CLI contract check for opt-in browser verification.
- [x] Add `scripts/openwebui_browser_smoke.py` and `--browser`/`--artifacts-dir` to the existing smoke entry point.
- [x] Use pinned Playwright in the Docker CI job; verify UI login, actual model selection, restored assistant response, streaming request, and Socket.IO WebSocket handshake. Capture synthetic-only screenshots for diagnosis.
- [x] Run real-container/browser CI and resolve any failures before claiming the browser checks pass.

## Task 3: Documentation and diagrams

- [x] Audit and update relevant installation, architecture, data flow, security, operations, testing, and user-guide pages; preserve unrelated documentation.
- [x] Update affected Mermaid sources and committed PNG/SVG/GIF renders, plus a focused Open WebUI topology.
- [x] Record original PR #120 completion and its 24 successful CI checks. Document target-machine/provider acceptance steps and actual tested scope.

## Task 4: Delivery

- [x] Run full pytest, ruff, format, mypy, deterministic benchmarks, and relevant asset checks.
- [x] Independently review the complete diff and resolve material findings.
- [x] Push the isolated branch, open the follow-up PR, and report remaining environmental limitations explicitly. Do not merge.

## Validation record

- The fixture/helper and CLI checks failed before implementation, then all 12 feature checks passed. Final focused feature/diagram/packaging run: 39 passed.
- Full local suite: **9,608 passed, 61 skipped, 29 deselected**; ruff, formatting (395 files), mypy (111 source files), and deterministic benchmark (84 types; 20 negative-corpus files) passed.
- Real Docker/WebUI and Chromium fixture acceptance passed in [CI run 39314225475](https://github.com/asanderson/llm-redact/actions/runs/39314225475), including the seeded stale provider URL and Socket.IO handshake. The final branch also explicitly probes the direct fake-provider inventory before seeding; current-head CI status is tracked in [PR #122](https://github.com/asanderson/llm-redact/pull/122).
- Independent whole-branch review found no critical or important defects. Its completion-record housekeeping note is addressed here.
- Relevant guides and diagrams were audited; Mermaid sources and their PNG/SVG/GIF assets are committed. The original PR #120 completion record and its final 24 successful CI checks were recorded.
- No owner's machine or real provider account was accessed or validated. Follow `docs/openwebui.md#verify-your-deployment` on the intended host before sending private data. No merge is authorized by this work.
