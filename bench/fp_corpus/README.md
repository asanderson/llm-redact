# False-positive corpus

Real-world *negatives* for the detection precision gate. The recall
benchmark's positives are generated at runtime and never committed (they
are secret-shaped by construction); the files here contain no secrets, so
they can be vendored and scanned byte-for-byte on every run.

`MANIFEST.toml` pins the exact expected detector-type counts per file.
`python -m llm_redact.bench --check` fails on any mismatch in either
direction. When a rule change legitimately alters a count, update the
manifest in the same commit and explain why.

## Contents and provenance

| file | origin | license/status |
|---|---|---|
| `sqlite_excerpt.c` | `src/util.c` from [SQLite](https://sqlite.org) (fetched from the sqlite/sqlite GitHub mirror) | Public domain (SQLite's blessing header) |
| `rfc_excerpt.txt` | [RFC 5737](https://www.rfc-editor.org/rfc/rfc5737.txt), "IPv4 Address Blocks Reserved for Documentation" | Reproduced per the IETF Trust Legal Provisions; contains real (published) author contact details, which the manifest pins as correct detections |
| `gutenberg_prose.txt` | *Alice's Adventures in Wonderland* by Lewis Carroll, body text of [Project Gutenberg eBook #11](https://www.gutenberg.org/ebooks/11) | Public domain; Project Gutenberg header/footer and trademark removed as the PG license requires for redistributed copies |
| `synthetic_app.log` | Authored for this repo | Part of llm-redact (MIT) |
| `synthetic_config.yaml` | Authored for this repo | Part of llm-redact (MIT) |
| `synthetic_code.py` | Authored for this repo | Part of llm-redact (MIT) |
| `synthetic_k8s.yaml` | Authored for this repo | Part of llm-redact (MIT) |
| `synthetic_eu_document.txt` | Authored for this repo | Part of llm-redact (MIT) |
| `synthetic_web3.ts` | Authored for this repo | Part of llm-redact (MIT) |
| `synthetic_terraform.tf` | Authored for this repo | Part of llm-redact (MIT) |

The synthetic files deliberately probe the noisy rules: UUIDs and git SHAs
against the hex rules, ten-digit invoice ids and truncated numbers against
the phone grammar, low-entropy assignments against the generic-secret
entropy gate, Luhn-failing card shapes, SSA-invalid SSN shapes, mod-97-
failing IBAN shapes, non-JSON-header JWT shapes, `AccountKey=` with an
entropy-zero value, and vendor prefixes one character short of their
grammar. Each probe's intent is commented inline.

This directory ships in the repository only — it is not part of the wheel.
