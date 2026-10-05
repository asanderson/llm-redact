"""Out-of-band PII corpus tooling (dev-only; never shipped in the wheel).

``generate.py`` asks a local, Apache-2.0-licensed LLM served by Ollama for
coding-agent artifacts with invented personal data, ``audit.py`` has the
same teacher read the false-positive corpus and lists candidate misses and
false positives for a human, ``review.py`` is the hand-verification CLI that
freezes the private agent-traffic evaluation set, and ``train_student.py``
is the (skeleton) student-model recipe. See README.md.

Nothing here logs or prints a generated, audited or reviewed text value:
output is counts, ids, offsets, types and model names. Generated and
verified data are written outside every git work tree and never committed.
"""
