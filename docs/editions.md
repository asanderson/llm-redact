# Editions and licensing

This repository is **free and open-source software** under the
[GNU Affero General Public License v3.0](../LICENSE) — see
[The licenses](#the-licenses) below for what that means in practice.

**This repository is not gated at all.** The FOSS core carries no tier
checks, no seat caps, no cloud entitlements, and needs no license key —
everything in it, from every detection rule to the cloud LLM adapters,
non-loopback (mTLS) serving, and Kubernetes deployment, works keyless.
Tiers exist only as the commercial packaging of the separately-installed,
proprietary **`llm-redact-pro`** package.

> **llm-redact-pro is coming soon** — it is not yet generally available.
> Everything in this repository is complete and unrestricted without it;
> when it ships, it will be a per-seat commercial subscription, and its
> full licensing reference and operator guides will ship with it.

## What's what

| | FOSS core (this repo) | + llm-redact-pro (Pro) | Team | Unlimited / Managed |
| --- | --- | --- | --- | --- |
| Named users (email-verified seats) | implicit single local user | 1 | 25 | unlimited |
| The entire redaction/rehydration path: every rule, mode, NER backend, deny/allow lists, the realtime relay | ✓ | ✓ | ✓ | ✓ |
| ALL provider adapters — Anthropic/OpenAI/Gemini/Ollama/Cohere/custom **and** AWS Bedrock / Azure OpenAI / GCP Vertex | ✓ | ✓ | ✓ | ✓ |
| In-memory + persistent unencrypted SQLite vault, JSON logs, dashboard/editor/preview, doctor, plugins | ✓ | ✓ | ✓ | ✓ |
| Non-loopback (mTLS) serving; Kubernetes deployment (Helm chart + HPA) | ✓ | ✓ | ✓ | ✓ |
| Server persistent vault (PostgreSQL / MySQL / Oracle / any DB-API RDBMS, incl. cloud-managed DBMS), vault encryption at rest, audit log + tamper chain + backup sinks (with batch encryption), OTel, per-conversation sessions, named users | | ✓ | ✓ | ✓ |

## How keys work

- **The core never asks for one.** A key matters only to the pro
  package's subsystems; the core resolves and displays it (status,
  dashboard, doctor) but enforces nothing.
- **Verification is entirely offline** — the proxy never phones home.
- **Resolution order**: `LLM_REDACT_LICENSE_KEY` env var → `[license] key`
  → `[license] key_file`.
- **A pro-only subsystem configured without the pro package refuses
  startup naming the feature and the package** — never a silent
  downgrade; with the package installed, its own factories honor the
  key's tier, seats, and expiry.
- **Expiry**: a valid key starts warning 30 days before expiry (startup
  log, `/status`, doctor, dashboard) and keeps its tier for a 14-day
  grace window after it.
- `llm-redact license show` decodes your key.
- **Named seats** (`llm-redact users`) live in the pro package; the
  invite→verify→key walkthrough ships with it.

## The licenses

The model is FOSS core + proprietary plugin:

- **This repository (the core) is free software: GNU AGPL-3.0.** You can
  use it for any purpose, study and audit every line, modify it, and
  redistribute it — the freedoms that matter most for a tool you trust
  with your secrets. The single obligation is **share-alike**: if you
  distribute a modified version, *or let others use a modified version
  over a network* (the AGPL's addition to the GPL), you must offer them
  your modified source under the same license. Running the unmodified
  proxy yourself — the normal case — triggers no obligations, and the
  tools that merely talk to the proxy over HTTP are not affected by its
  license.
- **`llm-redact-pro` is proprietary.** It is licensed per seat under a
  commercial subscription agreement, not open source, and may not be
  redistributed. Its vendor holds the copyright to the core and licenses
  its own code separately — the AGPL applies to the core, not to the pro
  package.
- **Contributions to the core require a [CLA](CLA.md)** so the copyright
  stays consolidated enough to keep this dual-license model working.

## The open-core boundary

The paid subsystems live in that **separately-installed `llm-redact-pro`
package**, not in the AGPL wheel — distribution control is the primary
boundary, the signed key the secondary tier gate. In practice:

- A paid config without that package **fails closed** with a
  `ConfigError` naming the feature and the package — never a silent
  downgrade.
- `doctor`, `/status`, and the dashboard each carry a
  **licensed-features package: installed / not installed** line, so the
  boundary is never a guess.
- The redaction/rehydration core stays wholly in the FOSS wheel — and
  the core contains **no tier gates at all**: what keeps the paid tiers
  paid is that their implementations ship only with a subscription,
  and the pro package's own factories honor the key they're paired
  with.

The honest threat-model framing is unchanged from the start: enforcement
is deterrence plus a legal boundary, never DRM, and verification never
phones home.
