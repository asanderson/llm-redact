# llm-redact documentation index

Start with the repository [README](../README.md) (what the proxy does, the
full configuration surface, the provider matrix) — then come here to find
the right deep-dive. The packaged **user guide** (dashboard, config editor,
agent plugin commands) is a separate document that ships inside the wheel:
run `llm-redact guide`, open `/__llm-redact/guide` on a running proxy, or
read [src/llm_redact/user_guide.md](../src/llm_redact/user_guide.md).

## Getting started

| Doc | What it covers |
| --- | --- |
| [quickstart.md](quickstart.md) | Redacting your first session in five minutes: install, init, run, verify. |
| [how-it-works.md](how-it-works.md) | The mechanism end to end: the round-trip diagrams, a worked example (placeholder body, vault rows, audit row), persistence and fuzzy token restoration, and session isolation. |
| [providers.md](providers.md) | Per-provider setup: Azure/Vertex/Bedrock/Ollama/custom upstreams, embeddings, batch APIs, the realtime relay, and the opt-out switches. |
| [detection.md](detection.md) | The full detection reference: built-in rules, deny strings, per-rule modes, allowlists, and the person-name NER backends. |
| [dashboard.md](dashboard.md) | The local ops surface: dashboard, config editor, status/metrics/health endpoints, redaction preview, and the agent plugins. |
| [troubleshooting.md](troubleshooting.md) | Keyed by the exact error strings you will see, with the fix for each. |
| [plugins.md](plugins.md) | The dashboard and config editor as agent slash commands (Claude Code, Codex, OpenCode, Cursor). |

## Running it in production

| Doc | What it covers |
| --- | --- |
| [deployment.md](deployment.md) | The end-to-end guide: bind policy and mTLS, containers, health probes, the Helm chart, SIGHUP reloads, vault lifecycle, service units, log rotation. |
| [observability.md](observability.md) | Prometheus scrape/alert examples and the Grafana dashboard, mapped to the emitted metrics. |
| [resilience.md](resilience.md) | The failure-mode catalogue: what happens on every upstream fault, stream truncation, and vault write error — and the tests that pin it. |
| [api-coverage.md](api-coverage.md) | The endpoint matrix: every provider route and how the proxy treats it (pinned by test in both directions). |

The paid operational features — the server RDBMS vault, at-rest vault
encryption, the audit log and off-machine sinks, OpenTelemetry, named
users, and per-conversation sessions — will ship in the
separately-installed `llm-redact-pro` package (**coming soon** — not yet
generally available; its operator guides ship with the package).

## Licensing and teams

This repository is **free and open-source software under the GNU
AGPL-3.0** ([LICENSE](../LICENSE)); the paid `llm-redact-pro` package is
proprietary. The plain-language explanation of both — what the AGPL asks
of you (share-alike on distributed or network-served modifications,
nothing for normal self-hosted use), the condensed **tier matrix**, how
license keys resolve and expire, and the open-core package boundary —
is in [editions.md](editions.md); contributions are accepted under the
[CLA](CLA.md). The pro package is **coming soon** — its full licensing
reference and enforcement internals ship with it.

## Security

| Doc | What it covers |
| --- | --- |
| [SECURITY.md](SECURITY.md) | The security policy: how to report a vulnerability (GitHub private reporting) and what counts as one. |
| [threat-model.md](threat-model.md) | What the proxy defends against, what it deliberately does not, and why loopback is the default. |
| [security-dataflows.md](security-dataflows.md) | The request-path trust boundaries with each policy decision/enforcement point mapped to code. |
| [security-testing.md](security-testing.md) | The red-team boundary suite, canary leak harness, and differential fuzzing. |
| [security-review-3.1.md](security-review-3.1.md) | The adversarial security review record and the 3.1.1 fixes it produced. |
| [fips.md](fips.md) | FIPS 140-3 posture: approved-algorithm selection, and why validation belongs to the host. |
| [history-hygiene.md](history-hygiene.md) | Auditing all git history with the production detectors. |

## Engineering record

| Doc | What it covers |
| --- | --- |
| [assurance.md](assurance.md) | Proving the suites have teeth: mutation testing, property tests, differential fuzzing, the complexity-coverage gate. |
| [dependencies.md](dependencies.md) | What ships and why: the three runtime deps, every extra, and the vendored-code policy (pinned to pyproject by test). |
| [versioning.md](versioning.md) | SemVer policy: what counts as breaking, deprecation windows, release verification. |
| [ner-landscape.md](ner-landscape.md) | The FOSS NER landscape survey behind the optional backends. |
| [compaction-relink.md](compaction-relink.md) | The rejected design record for relinking history-compaction session forks — read before re-attempting. |

## Contributing and releasing

| Doc | What it covers |
| --- | --- |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, the gates every commit passes, testing conventions, how to add a detection rule, and the CLA sign-off. |
| [CLA.md](CLA.md) | The contributor license agreement: you keep ownership, your work stays AGPL for everyone, and the maintainer gets the relicensing rights the dual-license model needs. |
| [coding-standards.md](coding-standards.md) | The style guide: formatting, typing, the correctness invariants, comment style, and testing style a reviewer holds a change to. |
| [RELEASING.md](RELEASING.md) | The tag-driven release process: preflight checklist, cutting the tag, what the workflow produces, and one-time setup. |

Diagram sources live in [diagrams/](diagrams/) (rendered PNGs are
committed; `scripts/render_diagrams.sh` regenerates them) and the
documentation screenshots in [screenshots/](screenshots/) are captured
from fixture traffic only, never real sessions.
