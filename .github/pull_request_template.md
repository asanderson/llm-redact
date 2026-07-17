<!-- Never include real secrets anywhere in a PR — use secret-shaped fakes
     (vendors' canonical examples, alphabet runs, corp.example domains). -->

## What & why

## Checklist

- [ ] All gates pass locally: `ruff check` + `ruff format --check`,
      `pytest`, `mypy`, `python -m llm_redact.bench --check`
- [ ] `CHANGELOG.md` entry under `[Unreleased]` (user-visible changes)
- [ ] Touched `rehydrate.py`/`sse.py`/adapter event handling → extended the
      split-at-every-offset sweeps (not just single-case tests)
- [ ] New detection rule → generator in `bench/corpus.py`, fp-corpus gate
      run, `MANIFEST.toml` updated with justification if counts changed
- [ ] No new runtime dependencies (extras are fine)
