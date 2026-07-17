#!/usr/bin/env bash
# End-to-end smoke test of the container images. Engine-agnostic:
#   ./scripts/container_smoke.sh docker llm-redact:ci llm-redact-fake:ci
#   ./scripts/container_smoke.sh podman llm-redact:ci llm-redact-fake:ci
set -euo pipefail

ENGINE="${1:-docker}"
PROXY_IMAGE="${2:-llm-redact:ci}"
FAKE_IMAGE="${3:-llm-redact-fake:ci}"
NETWORK="llm-redact-smoke"
EMAIL="jane.doe@corp.example"

cleanup() {
  "$ENGINE" rm -f smoke-proxy smoke-fake >/dev/null 2>&1 || true
  "$ENGINE" network rm "$NETWORK" >/dev/null 2>&1 || true
  rm -f /tmp/llm-redact-smoke-config.toml
}
trap cleanup EXIT

"$ENGINE" network create "$NETWORK" >/dev/null

"$ENGINE" run -d --name smoke-fake --network "$NETWORK" "$FAKE_IMAGE" >/dev/null

cat > /tmp/llm-redact-smoke-config.toml <<EOF
[providers.anthropic]
upstream_base_url = "http://smoke-fake:9999"
EOF

"$ENGINE" run -d --name smoke-proxy --network "$NETWORK" \
  -p 127.0.0.1:18787:8787 \
  -v /tmp/llm-redact-smoke-config.toml:/etc/llm-redact/config.toml:ro \
  "$PROXY_IMAGE" >/dev/null

for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:18787/__llm-redact/healthz >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "--- health + readiness probes"
curl -sf http://127.0.0.1:18787/__llm-redact/healthz | grep -q '"status": *"ok"' \
  || { echo "FAIL: /healthz did not report ok"; exit 1; }
# The shipped image bundles the realtime + perf extras: WS upgrades are
# servable and uvloop is importable.
curl -sf http://127.0.0.1:18787/__llm-redact/readyz | grep -q '"realtime": *true' \
  || { echo "FAIL: /readyz reports realtime unavailable (websockets not in image)"; exit 1; }
"$ENGINE" exec smoke-proxy python -c "import uvloop" \
  || { echo "FAIL: uvloop (perf extra) not in image"; exit 1; }

echo "--- status endpoint"
curl -sf http://127.0.0.1:18787/__llm-redact/status | head -c 200
echo

echo "--- streaming round trip"
RESPONSE=$(curl -sN http://127.0.0.1:18787/v1/messages \
  -H 'content-type: application/json' -H 'x-api-key: smoke' \
  -d "{\"model\":\"m\",\"stream\":true,\"max_tokens\":10,\
\"messages\":[{\"role\":\"user\",\"content\":\"mail $EMAIL please\"}]}")

echo "$RESPONSE" | grep -q "$EMAIL" || { echo "FAIL: original not restored in response"; exit 1; }
if echo "$RESPONSE" | grep -q "«"; then
  echo "FAIL: placeholder leaked to client"; exit 1
fi

FAKE_LOGS=$("$ENGINE" logs smoke-fake 2>&1)
echo "$FAKE_LOGS" | grep -q "«EMAIL_001»" || { echo "FAIL: upstream did not receive placeholder"; exit 1; }
if echo "$FAKE_LOGS" | grep -q "$EMAIL"; then
  echo "FAIL: original leaked to upstream"; exit 1
fi

echo "container smoke test passed ($ENGINE)"
