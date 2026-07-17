"""Detection quality benchmark: deterministic synthetic corpus + metrics.

Run with ``python -m llm_redact.bench``. The corpus is generated at runtime
(never committed — no realistic-looking secrets in the repository) from a
fixed seed, so results are reproducible and rule regressions are diffable.
"""
