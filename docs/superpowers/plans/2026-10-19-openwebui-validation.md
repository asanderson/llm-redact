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

- [ ] Add and run failing tests for an idempotent seed helper using a real temporary SQLite database with the pinned WebUI config schema.
- [ ] Implement the fixture helper, then seed the disposable container database, restart WebUI, inspect its effective provider URL, and rerun redaction/restoration checks.
- [ ] Run the focused tests and lint.

## Task 2: Browser acceptance

- [ ] Add and run a failing CLI contract check for opt-in browser verification.
- [ ] Add `scripts/openwebui_browser_smoke.py` and `--browser`/`--artifacts-dir` to the existing smoke entry point.
- [ ] Use pinned Playwright in the Docker CI job; verify UI login, actual model selection, restored assistant response, streaming request, and Socket.IO WebSocket handshake. Capture synthetic-only screenshots for diagnosis.
- [ ] Run real-container/browser CI and resolve any failures before claiming the browser checks pass.

## Task 3: Documentation and diagrams

- [ ] Audit and update relevant installation, architecture, data flow, security, operations, testing, and user-guide pages; preserve unrelated documentation.
- [ ] Update affected Mermaid sources and committed PNG/SVG/GIF renders, plus a focused Open WebUI topology.
- [ ] Record original PR #120 completion and its 24 successful CI checks. Document target-machine/provider acceptance steps and actual tested scope.

## Task 4: Delivery

- [ ] Run full pytest, ruff, format, mypy, deterministic benchmarks, and relevant asset checks.
- [ ] Independently review the complete diff and resolve material findings.
- [ ] Push the isolated branch, open the follow-up PR, and report remaining environmental limitations explicitly. Do not merge.
