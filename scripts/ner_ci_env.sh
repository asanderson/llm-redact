#!/usr/bin/env bash
# The environment of the real-model CI jobs (ci.yml's ner-models and
# ner-eval.yml; docs/ner-bench.md, "CI"): the hf, gliner, gliner2 and
# bench-data extras exactly as uv.lock pins them, every wheel checked
# against its sha256, except that torch is the PyTorch CPU wheel pinned by
# its own sha256 in scripts/ner_ci_torch_cpu.txt (the locked PyPI torch
# wheel brings several GB of CUDA libraries no runner uses).
#
#   scripts/ner_ci_env.sh [VENV]      # default .venv (replaced)
#
# The venv is CPython 3.13 on Linux x86_64, the platform the pinned torch
# wheel is built for (any other platform fails its hash check).
#
# Packages are left out of the export by NAME (--no-emit-package), so the
# lock's hashes stay in it: torch, triton and the nvidia-*/cuda-* packages
# only the CUDA wheel needs. Both installs run with --require-hashes and
# --no-deps (the export is the whole locked closure), then `uv pip check`
# confirms the result is consistent.
set -euo pipefail

venv="${1:-.venv}"
here="$(cd "$(dirname "$0")" && pwd)"
pin="$here/ner_ci_torch_cpu.txt"
extras=(--extra hf --extra gliner --extra gliner2 --extra bench-data)
requirements="$(mktemp)"
trap 'rm -f "$requirements"' EXIT

listed="$(uv export --frozen --no-header --no-annotate --no-hashes --no-emit-project "${extras[@]}")"
locked="$(sed -nE 's/^torch==([^ ;]+).*/\1/p' <<<"$listed")"
pinned="$(sed -nE 's/^torch==([^+ ;]+)\+cpu .*/\1/p' "$pin")"
if [ -z "$locked" ] || [ "$locked" != "$pinned" ]; then
  echo "uv.lock pins torch ${locked:-(none)} but scripts/ner_ci_torch_cpu.txt pins ${pinned:-(none)}+cpu: update the pin and its sha256" >&2
  exit 1
fi

omit=()
while read -r name; do
  omit+=(--no-emit-package "$name")
done < <(sed -nE 's/^(torch|triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+)==.*/\1/p' <<<"$listed")

uv export --frozen --no-header --no-annotate --no-emit-project "${extras[@]}" "${omit[@]}" >"$requirements"
uv venv --python 3.13 "$venv"
uv pip install --python "$venv" --require-hashes --no-deps -r "$requirements"
uv pip install --python "$venv" --require-hashes --no-deps -r "$pin" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$venv" --no-deps -e .
uv pip check --python "$venv"
