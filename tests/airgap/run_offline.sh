#!/bin/sh
# The air-gapped half of the CI `airgap` job (.github/workflows/ci.yml), run
# as root inside a network namespace with no route, from the repository root:
#
#   sudo unshare --net --mount -- env HOME=... HF_HOME=<empty dir> \
#     tests/airgap/run_offline.sh PYTHON MODELS EDGE_MODELS EMPTY_DIR
#
# PYTHON is the environment's interpreter (the hf, gliner and presidio
# extras installed, and spaCy's en_core_web_sm); MODELS and EDGE_MODELS are
# the folders `llm-redact models pull --to DIR --as /models` wrote for
# tests/airgap/pull.toml and pull-edge.toml; EMPTY_DIR is an empty folder.
# Each is mounted read-only at /models in turn, as an enclave mounts carried
# folders. Every llm-redact command runs through no_network.py, which fails
# (exit 3) on any attempt made through Python's socket module to resolve a
# host or connect an IP socket — a library that falls back quietly after a
# failed connection would otherwise go unnoticed. Native code connecting by
# itself is not seen by it; the namespace with no route still stops that. HF_HOME points at an empty folder: nothing loads from a cache.
set -eu

python=$1
models=$2
edge=$3
empty=$4

run() {
    echo "+ llm-redact $*"
    "$python" tests/airgap/no_network.py "$@"
}

mount_models() {
    mkdir -p /models
    umount /models 2>/dev/null || true
    mount --bind "$1" /models
    mount -o remount,bind,ro /models
}

# The namespace really has no route.
if "$python" -c 'import socket; socket.create_connection(("1.1.1.1", 443), timeout=3)' 2>/dev/null; then
    echo "FAIL: this network namespace can reach the internet" >&2
    exit 1
fi

# The default hf and GLiNER models (the GLiNER folder carries its base
# model's tokenizer and configuration).
mount_models "$models"
run models verify --dir /models
run serve --check --config tests/airgap/config.toml

# Knowledgator's self-contained GLiNER-PII edge model.
mount_models "$edge"
run models verify --dir /models
run serve --check --config tests/airgap/edge.toml

# Presidio's email check (tldextract) on a request: the shipped Public
# Suffix List snapshot, never a fetch.
run preview --config tests/airgap/presidio.toml --text "write to Jane Roe at jane.roe@example.com"

# An empty folder where the models belong: refused with the `models pull`
# hint, still without a network attempt (exit 1, not no_network's 3).
mount_models "$empty"
out=$(mktemp)
status=0
run serve --check --config tests/airgap/config.toml >"$out" 2>&1 || status=$?
cat "$out"
if [ "$status" -ne 1 ]; then
    echo "FAIL: serve --check with an empty model folder exited $status, expected 1" >&2
    exit 1
fi
grep -q 'llm-redact models pull --to DIR' "$out" || {
    echo "FAIL: the refusal does not name llm-redact models pull --to" >&2
    exit 1
}
umount /models
echo "air-gapped start: OK"
