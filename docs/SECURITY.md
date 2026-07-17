# Security Policy

llm-redact is a privacy tool: its whole job is keeping your secrets out of
LLM providers' hands. Security reports get priority attention.

## Supported versions

Only the latest release line receives fixes. There are no backports.

| Version | Supported |
|---|---|
| latest release | ✅ |
| anything older | ❌ upgrade |

## Reporting a vulnerability

Use **GitHub private vulnerability reporting**: the repository's *Security*
tab → *Report a vulnerability*. Please do not open public issues for
suspected vulnerabilities, and do not include real secrets in
reproductions — the fp-corpus style (shaped-but-fake values) works fine.

You can expect an acknowledgment within a few days. Fixes ship as a normal
release; credit is given unless you ask otherwise.

## Scope

What counts as a vulnerability is defined by the threat model —
see [threat-model.md](threat-model.md); the security-relevant
data flows, with each policy decision point and enforcement point mapped
to code, are diagrammed in
[security-dataflows.md](security-dataflows.md). How each
boundary is *tested* — the attacker-scenario → coverage map — is in
[security-testing.md](security-testing.md). Highlights:

**In scope**: secret values leaking into logs, the audit DB, metrics, or
error responses; redaction bypasses on covered body shapes; cross-session
placeholder restoration; vault files created with permissive modes;
`/__llm-redact/*` endpoints reachable cross-origin or forwarded upstream;
at-rest encryption not actually encrypting.

**Out of scope (documented limitations, not vulnerabilities)**: plaintext
secrets in process memory or swap; anything a same-UID process can do;
missing TLS on 127.0.0.1; multi-user deployments; the provider inferring
redacted content from surrounding text; length/timing side channels;
base64-encoded media contents; detection misses on value shapes the rules
deliberately exclude (bare-digit phone numbers, street addresses).

## Hardening quickstart

- Keep the proxy on `127.0.0.1` (the default). Do not bind it to other
  interfaces or add TLS termination in front — that is explicitly
  unsupported territory.
- Prefer `[vault] backend = "memory"` (default) unless you need restart
  persistence; if you use sqlite, add
  `encryption = "fernet"` (`crypto` extra) and keep `LLM_REDACT_VAULT_KEY`
  in your shell profile, not in the config file.
- Leave `[audit]` off unless you need it; it stores metadata only, but
  metadata is still a record of your traffic.
- Verify release artifacts:
  `gh attestation verify <file> --repo asanderson/llm-redact`.
