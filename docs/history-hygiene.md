# Git history hygiene audit

Making the repository public publishes every commit ever pushed, not just
the current tree. This is the record of the pre-publication sweep; re-run
it (and update this file) immediately before any visibility flip.

## Method

`scripts/history_sweep.py` runs the production detector stack — the same
rules, validators, and overlap resolution the proxy uses — over the diff of
every commit on every ref, then classifies each unique detected value:

- **in current tree** — the value exists in the working tree today. These
  are the deliberate secret-shaped fixtures (tests, bench generators, docs
  examples) that normal code review already covers.
- **history-only** — the value was committed at some point and later
  removed. This is the set that needs eyeballs: it is exactly what an
  attacker would mine a freshly-public history for.

Supplementary greps cover what the detectors do not: every email address
ever committed, home-directory paths, and real PEM header variants.

## Findings — 2026-07-08 (118 commits, all refs)

130 unique detected values. **Verdict: GO — nothing real in history.**

- **History-only: 2**, both false positives — the private-key detector
  matching across the *source code* of the synthetic-key generators
  (`bench/corpus.py`, an early `test_detectors.py`), where the literal
  `-----BEGIN PRIVATE KEY-----` marker and base64 helper calls appear as
  adjacent diff lines. No key material.
- **In current tree: 128**, all reviewable fixtures: RFC-2606 documentation
  domains (`corp.example`, `example.com`, `.internal`), AWS's own
  documented example key (`AKIAIOSFODNN7EXAMPLE`), alphabet-run stubs
  (`sk-abcdefghijklmnopqrstuvwx`), and truncated PEM bodies far too short
  to be real keys.
- **Emails**: every address in diff content is a documentation domain, the
  `noreply@github.com` allowlist default, or the two published RFC 5737
  author contacts vendored in `bench/fp_corpus/rfc_excerpt.txt` (see its
  provenance table). Commit author/committer emails are the git identities
  that are public on any repo.
- **Paths**: no `/home/<user>` or `/Users/<user>` strings ever committed.
- **Tracked dotfiles**: `.claude/settings.json` contains a tool-permission
  allowlist entry only.

## Before a future visibility flip

- Re-run the sweep over the then-current history; update this file.
- GitHub secret scanning will pattern-match the public history. Our
  fixtures are verifiably fake but some SHAPES (e.g. `xoxb-…`) may still
  trigger scanning-partner notifications; expect and triage those, or
  align the noisiest fixtures with vendors' canonical example values
  first.
