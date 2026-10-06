#!/bin/sh
# The offline half of the wheelhouse recipe (docs/deployment.md, "Offline
# installs"), as the CI `airgap` job runs it inside a network namespace with
# no route, from the repository root:
#
#   install_wheelhouse.sh PYTHON WHEELHOUSE VENV
#
# WHEELHOUSE holds requirements.txt and torch.txt (scripts/cpu_torch.py) and
# every wheel they name plus llm-redact-proxy's, downloaded on a connected
# machine of the same platform and Python. pip never looks past it
# (--no-index); every locked package is hash-checked.
set -eu

python=$1
wheelhouse=$2
venv=$3

"$python" -m venv "$venv"
pip="$venv/bin/pip"
"$pip" install --no-index --find-links "$wheelhouse" --no-deps -r "$wheelhouse/torch.txt"
"$pip" install --no-index --find-links "$wheelhouse" --no-deps --require-hashes \
    -r "$wheelhouse/requirements.txt"
"$pip" install --no-index --find-links "$wheelhouse" --no-deps llm-redact-proxy
"$pip" check
"$venv/bin/python" scripts/cpu_torch.py check

doctor=$(mktemp)
"$venv/bin/llm-redact" doctor --config tests/airgap/config.toml >"$doctor" 2>&1 || true
grep -E '^(PASS|WARN|FAIL) +ner:' "$doctor"
for backend in hf gliner; do
    grep -q "^PASS  ner: $backend backend importable" "$doctor" || {
        echo "FAIL: the $backend backend is not importable from the wheelhouse install" >&2
        exit 1
    }
done
echo "wheelhouse install: OK"
