# Versioning and stability policy

llm-redact follows [Semantic Versioning](https://semver.org/). 1.0.0 is the
first stable release; from here on, the surfaces below are stable: they
only change in a MAJOR release, with a documented migration.

## The stable surface

- **Placeholder token format**: `«TYPE_NNN»` guillemet tokens, the fuzzy
  mangle grammar that restores them, and the deterministic
  same-(session, type, value) ⇒ same-token vault rule. Vault databases
  written by 1.x are readable by every later 1.x (schema migrations are
  one-way but automatic).
- **Configuration**: existing `config.toml` keys keep their meaning;
  new keys are additive with safe defaults. Unknown keys stay hard errors
  (that is the safety design, not instability). A key is removed or
  repurposed only in a MAJOR release.
- **CLI**: command names, subcommands, flags, and exit codes are stable;
  new flags/commands are additive. `serve` stays quiet on stdout;
  `lookup` remains the only command that prints secret values.
- **Local API** (`/__llm-redact/*`): existing endpoints and JSON fields
  keep their shape; new fields and endpoints are additive. The reserved
  prefix itself never changes. Anything under it remains metadata-only
  (the documented exceptions: `lookup`-equivalent config-editor GET
  returns allowlist and deny values).
- **Proxying behavior contracts**: unrecognized traffic forwards
  verbatim; oversized redactable bodies fail closed (413); block mode
  answers 400 before upstream contact; streaming output equals
  whole-text output; a framing violation degrades to verbatim
  pass-through, never corruption.
- **Runtime dependencies**: exactly three (httpx, starlette, uvicorn).
  Growing that set in a MINOR release would change the audit surface
  users signed up for — it is treated as a breaking change. Optional
  extras may evolve freely.

## What MINOR releases may do

Add detection rules (they change what gets redacted — by design, the
gates are the recall corpus and fp-corpus manifest), add providers,
add config keys/CLI flags/endpoints/fields, improve performance, and fix
bugs whose old behavior was contrary to documentation.

## Deprecation policy

Anything scheduled for removal in the next MAJOR release is first marked
deprecated in a MINOR release: the CHANGELOG announces it, and using it
logs a warning (never an error) for at least one MINOR cycle. Security
fixes are exempt from all of the above — if a behavior is found unsafe,
it changes in the next release, whatever the number says.
