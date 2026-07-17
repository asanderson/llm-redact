#!/usr/bin/env bash
# Opt-in Oracle smoke for the RDBMS vault. No CI job runs this — the Oracle
# container takes minutes to initialize — so run it locally when touching
# the oracle dialect:
#
#   uv sync --extra vault-oracle
#   ./scripts/oracle_smoke.sh [docker|podman]
#
# It starts gvenzl/oracle-free, waits for readiness, runs the real-server
# battery from tests/test_vault_rdbms.py against it, and tears it down.
# The password below is a throwaway container credential, not a secret.
set -euo pipefail

engine="${1:-docker}"
name="llmr-oracle-smoke"
password="ci-oracle-pass"

cleanup() { "$engine" rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

echo "starting $name (gvenzl/oracle-free:23-slim) ..."
"$engine" run -d --name "$name" -p 127.0.0.1:1521:1521 \
  -e ORACLE_PASSWORD="$password" gvenzl/oracle-free:23-slim >/dev/null

echo -n "waiting for the database (can take a few minutes) "
for _ in $(seq 1 120); do
  if "$engine" logs "$name" 2>&1 | grep -q "DATABASE IS READY TO USE"; then
    ready=1
    break
  fi
  echo -n "."
  sleep 5
done
echo
if [ "${ready:-0}" != "1" ]; then
  echo "oracle container never became ready" >&2
  exit 1
fi

LLM_REDACT_TEST_ORACLE_DSN="oracle://system:${password}@127.0.0.1:1521/FREEPDB1" \
  uv run pytest "tests/test_vault_rdbms.py::test_battery_real_server[oracle]" -v
echo "oracle smoke passed"
