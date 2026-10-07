#!/usr/bin/env sh
# Re-render docs/diagrams/*.mmd to the committed PNGs, then the animated
# SVG + GIF of every diagram that opts in with a "%% animate" comment
# (scripts/animate_diagrams.py; needs uv, which fetches playwright + pillow).
#
# Needs Node (mermaid-cli is fetched via npx) and a Chromium/Chrome for
# puppeteer. If puppeteer cannot download its own browser (sandboxed CI,
# corporate proxy), point it at an existing binary:
#   PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium scripts/render_diagrams.sh
#
# PNGs are committed so the README renders without any toolchain; run this
# whenever a .mmd source changes and commit both. DIAGRAMS names another
# folder of sources (llm-redact-pro's scripts/render_diagrams.sh sets it to
# its own docs/diagrams); both the PNGs and the animations are rendered there.
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

for src in "${DIAGRAMS:-docs/diagrams}"/*.mmd; do
    out="${src%.mmd}.png"
    # -s 2: 2x pixel density so text stays crisp; white background because
    # GitHub renders READMEs on both light and dark pages.
    # Pinned to mermaid-cli 11: 12.0 changed the layout engine, and the
    # committed PNGs were drawn with 11.x.
    npx -y @mermaid-js/mermaid-cli@11 -p "$PUPPETEER_CONFIG" \
        -i "$src" -o "$out" -b white -s 2 --quiet
    echo "rendered $out"
done

# Animated versions (data flow / message order); the same Chromium serves
# mermaid-cli and the GIF frame capture. ANIMATE=0 skips them.
if [ "${ANIMATE:-1}" != "0" ]; then
    uv run --no-project --with playwright --with pillow \
        python scripts/animate_diagrams.py --diagrams "${DIAGRAMS:-docs/diagrams}"
fi
