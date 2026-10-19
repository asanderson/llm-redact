# Optional Open WebUI Implementation Plan

> **For agentic workers:** Execute inline using superpowers:executing-plans; review the complete branch before opening a PR.

**Goal:** Add an optional local text-chat frontend that sends model traffic through llm-redact.

**Architecture:** A standalone Compose profile runs the unmodified, version-pinned Open WebUI image and the existing proxy. WebUI's backend has only an internal network; the proxy also has provider egress. A fixed Caddy HTTP frontend supplies loopback browser access because Docker does not publish internal-only endpoints. A fake OpenAI-compatible upstream and an actual WebUI API smoke test prove discovery, redaction, restoration, persistence, and failure behavior without cloud keys.

**Tech Stack:** Docker Compose, existing Python/Starlette/httpx test dependencies, stdlib smoke client.

**Spec:** User-approved research proposal from the preceding conversation, reflected in docs/openwebui.md.

## Global Constraints

- Single trusted local user and text chat; no core authentication or paid-feature implementation.
- No change to runtime dependencies or the proxy hot path.
- Keep the existing demo and default startup unchanged.
- WebUI v0.12.3; preserve upstream branding/license.
- Host ports bound to loopback; disable alternate integrations and use separate persistent volumes.
- Never commit real API keys, passwords, or private prompts.

## Review Focus

- Persisted WebUI configuration must not restore an old direct-provider URL.
- A dead proxy must fail rather than use a direct-provider fallback.
- Both streaming and nonstream replies must restore split placeholder tokens.
- Raw WebUI history remains sensitive even though upstream requests are redacted.
- Container-only confinement must not be advertised as remote or multiuser authorization.

### Task 1: OpenAI-compatible fake provider

**Files:** scripts/fake_upstream.py; tests/test_openwebui.py.
**Interface:** GET /v1/models and POST /v1/chat/completions return deterministic OpenAI-shaped responses.
- [ ] Write tests for discovery, nonstream replies, SSE framing, mangle mode, and proxy round trips with synthetic canaries.
- [ ] Run the tests to establish failure before adding the handlers.
- [ ] Add only the missing fixture handlers and register the routes.
- [ ] Run focused tests.

### Task 2: Optional deployment and executable smoke test

**Files:** compose.openwebui.yaml; examples/openwebui-config.toml; examples/openwebui.env.example; examples/openwebui-demo.compose.yaml; examples/openwebui-demo-config.toml; scripts/openwebui_smoke.py; .github/workflows/ci.yml; .gitignore.
**Interface:** docker compose -f compose.openwebui.yaml --env-file .env.openwebui --profile webui up -d; smoke script manages an isolated temporary demo project.
- [ ] Add focused deployment-contract checks for optionality, loopback exposure, backend network isolation, credentials, and persisted configuration.
- [ ] Run those checks before adding the stack.
- [ ] Add the standalone stack and no-key demo overlay.
- [ ] Add a stdlib smoke client using a disposable WebUI admin, then exercise authenticated model discovery and nonstream/SSE completions through the real WebUI backend.
- [ ] Check fixture logs for upstream canaries, proxy restart persistence, proxy failure, internal-network egress, and cleanup.
- [ ] Add a container CI job to execute the smoke test; local container execution is unavailable unless an engine is discovered.

### Task 3: Documentation and final verification

**Files:** docs/openwebui.md; docs/README.md; README.md.
- [ ] Document demo and real-provider setup, first admin, supported scope, key ownership, local model addressing, troubleshooting, backups, and version/license conditions.
- [ ] Run ruff, mypy, full pytest, and deterministic benchmark gates.
- [ ] Obtain an independent whole-branch review and resolve material findings.
- [ ] Push only the feature branch, open a PR, inspect CI, and report exact tested/untested scope.
