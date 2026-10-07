#!/usr/bin/env bash
# The environment of the real-model CI jobs (ci.yml's ner-models and
# ner-eval.yml; docs/ner-bench.md, "CI"): the hf, gliner, gliner2 and
# bench-data extras (and the dev group the tests need) exactly as uv.lock
# pins them, every wheel checked against its SHA-256, except that torch is
# the PyTorch CPU wheel of the locked version (the locked PyPI torch wheel
# brings several GB of CUDA libraries and triton no runner uses).
#
#   scripts/ner_ci_env.sh [VENV]      # default .venv (replaced)
#
# It is the one CPU-torch recipe the -ner image, the release SBOM, the
# airgap job and the offline wheelhouse use too: scripts/cpu_torch.py
# splits the hash-checked export into requirements.txt (every locked
# package but torch and its GPU-only dependencies, with the lock's hashes)
# and torch.txt (torch==<locked version>+cpu with the SHA-256 of every CPU
# wheel it records; it refuses an export whose torch version it records no
# wheel of, and a test pins its table to uv.lock's torch). Both installs run
# with --require-hashes and --no-deps (the export is the whole locked
# closure); `cpu_torch.py check` then refuses any CUDA or triton
# distribution and a torch that is not a +cpu build, and `uv pip check`
# confirms the result is consistent.
set -euo pipefail

venv="${1:-.venv}"
here="$(cd "$(dirname "$0")" && pwd)"
extras=(--extra hf --extra gliner --extra gliner2 --extra bench-data)
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv export --frozen --no-header --no-annotate --no-emit-project "${extras[@]}" \
  | python3 "$here/cpu_torch.py" requirements "$work/requirements.txt" >"$work/torch.txt"
uv venv --python 3.13 "$venv"
uv pip install --python "$venv" --require-hashes --no-deps -r "$work/torch.txt" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$venv" --require-hashes --no-deps -r "$work/requirements.txt"
uv pip install --python "$venv" --no-deps -e .
"$venv/bin/python" "$here/cpu_torch.py" check
uv pip check --python "$venv"
