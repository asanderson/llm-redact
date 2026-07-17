# Changelog

All notable changes to llm-redact are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

Convention: entries land in `[Unreleased]` with the change that introduces
them. A release moves `[Unreleased]` into a dated version section, bumps
`__version__` in `src/llm_redact/__init__.py` (the single source of truth),
and tags `vX.Y.Z`.

## [Unreleased]

### Added

- Interactive install script `scripts/install.sh`: detects the tools
  available on the machine (uv, pipx, pip, Homebrew, docker, podman),
  prompts for the preferred install method (`--method NAME` for
  non-interactive use), and prints every command before running it. The
  container methods pull `ghcr.io/asanderson/llm-redact:latest` and
  offer a loopback-published run.
- README Install section now shows the prebuilt-container path
  explicitly — `docker pull` / `podman pull` from
  `ghcr.io/asanderson/llm-redact` with the loopback publish spec — and
  links the install script.

## [1.0.0] - 2026-07-17

Initial public release. llm-redact was developed privately before this
debut; the public history starts here, at v1.0.0.

### Added

- **The transparent redaction proxy.** A local proxy sits between an
  agentic tool and its LLM provider: outbound requests are scanned for
  private values, each detected value is replaced with a deterministic
  `«TYPE_NNN»` placeholder whose mapping lives only in a local vault, and
  inbound responses — including streamed ones, even when a token splits
  across chunk boundaries — have the placeholders restored. Streaming is
  handled at the byte level across SSE, NDJSON, AWS eventstream, and
  WebSocket framings; unrecognized traffic passes through verbatim
  (never break the tool), and upstream faults fail closed with
  provider-shaped errors.
- **Detection**: 80+ built-in rules — vendor API keys and tokens
  (prefix-anchored), credit cards/IBANs/phones, checksum-validated
  national identifiers across 20+ countries, PGP/private-key armor, and
  checksum-vetoed crypto wallet addresses — plus user deny strings
  (always-win, tier 0), per-type allowlists, per-rule redact/warn/block
  modes, language scoping, custom rules with named validators, and
  optional person-name NER behind five interchangeable backends (spaCy,
  GLiNER, Presidio, Stanza, Hugging Face). A recall==1.0 benchmark gate
  and a real-world false-positive corpus pin detection quality in CI.
- **Providers**: Anthropic (Messages + Batches), OpenAI (Chat,
  Responses, Conversations, Files/Batches, Realtime WS, images/audio/
  video), Google Gemini (+ Live WS, context caching, batch), AWS
  Bedrock, Azure OpenAI, GCP Vertex (Gemini and Claude), Cohere, Ollama
  native, and any number of named custom OpenAI-compatible upstreams
  (vLLM, LM Studio, OpenRouter, …). Embeddings and file uploads are
  redact-only; the endpoint matrix is pinned by test in both directions.
- **Vault**: deterministic placeholder issuance — the same value always
  gets the same token within a session — with in-memory and persistent
  SQLite backends, session lifecycle CLI, and strict never-restore-
  across-sessions isolation.
- **Ops surface**: a self-contained local dashboard with a guarded
  config editor (validate → atomic write → SIGHUP hot reload), redaction
  preview dry-run, Prometheus metrics, health endpoints, a live SSE
  event feed, structured JSON logs, `doctor` diagnostics, and an honest
  posture block that surfaces every protection opt-out — warn-mode
  rules, disabled providers, MCP exemptions, language scoping — loudly.
- **Agent plugins** for Claude Code, Codex, OpenCode, and Cursor: ten
  slash commands mirroring the dashboard and config editor, with a
  proxy-presence guard and real-output screenshots in the docs.
- **Deployment**: hardened Dockerfile (multi-arch), k8s sidecar
  manifest, a Helm chart with sidecar and standalone modes plus optional
  HPA autoscaling, systemd/launchd service units, shell completions, an
  init wizard, and an env-injecting `run` wrapper. Non-loopback binds
  are fail-closed behind full mutual TLS.
- **Assurance**: split-at-every-offset streaming equivalence sweeps,
  property-based tests, differential codec fuzzing, a red-team boundary
  suite, mutation-testing gates, a complexity-coverage gate (every
  branching function executed by the suite), reproducible builds, SBOMs,
  and signed release artifacts.

### Licensing

- The core is **free and open-source software under GNU AGPL-3.0-only**,
  with nothing gated: no license keys, tiers, or seat caps anywhere in
  this repository. Contributions are accepted under `docs/CLA.md`.
- The separately-installed proprietary `llm-redact-pro` package
  (**coming soon**) will supply additional operational subsystems —
  server RDBMS vaults, vault encryption at rest, the audit log and its
  object-store sinks, OpenTelemetry export, per-conversation sessions,
  and named users. Configuring one of those without the package fails
  closed naming the feature and the package — never a silent downgrade.
