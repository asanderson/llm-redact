#!/usr/bin/env bash
# Interactive installer for llm-redact. Detects the tools available on
# this machine (uv, pipx, pip, Homebrew, docker, podman), asks which
# install you prefer, and runs exactly that — nothing else. Every
# command is printed before it runs.
#
#   ./scripts/install.sh                  # interactive menu
#   ./scripts/install.sh --method podman  # non-interactive
#
# Methods: uv | pipx | pip | brew | docker | podman
set -euo pipefail

PYPI_PKG="llm-redact-proxy"
IMAGE="ghcr.io/asanderson/llm-redact:latest"
TAP="asanderson/llm-redact"
TAP_URL="https://github.com/asanderson/llm-redact"
METHODS=(uv pipx pip brew docker podman)

usage() {
  echo "usage: install.sh [--method uv|pipx|pip|brew|docker|podman]"
  echo "Without --method, an interactive menu asks which install you prefer."
}

case "$(uname -s)" in
  Linux | Darwin) ;;
  *)
    echo "error: llm-redact supports Linux and macOS natively; elsewhere use the container image ($IMAGE)." >&2
    exit 1
    ;;
esac

method=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      method="${2:?error: --method needs a value}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }

available() {
  case "$1" in
    pip) have python3 ;;
    *) have "$1" ;;
  esac
}

describe() {
  case "$1" in
    uv) echo "uv tool install $PYPI_PKG" ;;
    pipx) echo "pipx install $PYPI_PKG" ;;
    pip) echo "python3 -m pip install --user $PYPI_PKG" ;;
    brew) echo "brew tap $TAP && brew install $TAP/llm-redact" ;;
    docker) echo "docker pull $IMAGE" ;;
    podman) echo "podman pull $IMAGE" ;;
  esac
}

if [[ -z "$method" ]]; then
  if [[ ! -t 0 ]]; then
    echo "error: stdin is not a terminal — pass --method uv|pipx|pip|brew|docker|podman" >&2
    exit 2
  fi
  echo "How would you like to install llm-redact?"
  echo
  i=0
  for m in "${METHODS[@]}"; do
    i=$((i + 1))
    marker=""
    available "$m" || marker="   (not found on this machine)"
    printf '  %d) %-6s — %s%s\n' "$i" "$m" "$(describe "$m")" "$marker"
  done
  echo
  read -r -p "Choice [1-${#METHODS[@]}]: " choice
  if ! [[ "$choice" =~ ^[1-9][0-9]*$ ]] || ((choice > ${#METHODS[@]})); then
    echo "error: invalid choice: $choice" >&2
    exit 2
  fi
  method="${METHODS[$((choice - 1))]}"
fi

case "$method" in
  uv | pipx | pip | brew | docker | podman) ;;
  *)
    echo "error: unknown method '$method' (uv|pipx|pip|brew|docker|podman)" >&2
    exit 2
    ;;
esac
if ! available "$method"; then
  echo "error: '$method' is not available on this machine" >&2
  exit 1
fi

run() {
  printf '+ %s\n' "$*"
  "$@"
}

case "$method" in
  uv) run uv tool install "$PYPI_PKG" ;;
  pipx) run pipx install "$PYPI_PKG" ;;
  pip) run python3 -m pip install --user "$PYPI_PKG" ;;
  brew)
    run brew tap "$TAP" "$TAP_URL"
    run brew install "$TAP/llm-redact"
    ;;
  docker | podman)
    run "$method" pull "$IMAGE"
    echo
    echo "Start it publishing to LOOPBACK ONLY — never -p 8787:8787, which"
    echo "would expose the proxy (and the secrets it rehydrates) to your LAN:"
    echo
    echo "  $method run -d --name llm-redact \\"
    echo "    -p 127.0.0.1:8787:8787 -v llm-redact-data:/data \\"
    echo "    $IMAGE"
    echo
    if [[ -t 0 ]]; then
      read -r -p "Run it now? [y/N] " yn
      case "$yn" in
        [Yy]*)
          run "$method" run -d --name llm-redact \
            -p 127.0.0.1:8787:8787 -v llm-redact-data:/data "$IMAGE"
          ;;
      esac
    fi
    ;;
esac

echo
echo "Done. Next steps:"
if [[ "$method" == docker || "$method" == podman ]]; then
  cat <<'EOF'
  - point your tools at it: export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
    (OPENAI_BASE_URL / GOOGLE_GEMINI_BASE_URL / OLLAMA_HOST likewise)
  - dashboard: http://127.0.0.1:8787/
  - mount a config: -v ./config.toml:/etc/llm-redact/config.toml:ro
  - the sqlite vault lives on the llm-redact-data volume (persists restarts)
EOF
else
  cat <<'EOF'
  - llm-redact init                # starter config + env exports for your tools
  - llm-redact run -- claude ...   # or export the base-URL vars init prints
  - llm-redact service install     # run the proxy at login (launchd/systemd)
  - llm-redact doctor              # read-only preflight
EOF
fi
