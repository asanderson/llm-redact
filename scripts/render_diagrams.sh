#!/usr/bin/env sh
# Re-render docs/diagrams/*.mmd to the committed PNGs.
#
# Needs Node (mermaid-cli is fetched via npx) and a Chromium/Chrome for
# puppeteer. If puppeteer cannot download its own browser (sandboxed CI,
# corporate proxy), point it at an existing binary:
#   PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium scripts/render_diagrams.sh
#
# PNGs are committed so the README renders without any toolchain; run this
# whenever a .mmd source changes and commit both.
set -eu

cd "$(dirname "$0")/.." || exit 1

PUPPETEER_CONFIG=$(mktemp)
trap 'rm -f "$PUPPETEER_CONFIG"' EXIT
{
    printf '{"args": ["--no-sandbox"]'
    if [ -n "${PUPPETEER_EXECUTABLE_PATH:-}" ]; then
        printf ', "executablePath": "%s"' "$PUPPETEER_EXECUTABLE_PATH"
    fi
    printf '}'
} > "$PUPPETEER_CONFIG"

for src in docs/diagrams/*.mmd; do
    out="${src%.mmd}.png"
    # -s 2: 2x pixel density so text stays crisp; white background because
    # GitHub renders READMEs on both light and dark pages.
    npx -y @mermaid-js/mermaid-cli -p "$PUPPETEER_CONFIG" \
        -i "$src" -o "$out" -b white -s 2 --quiet
    echo "rendered $out"
done
