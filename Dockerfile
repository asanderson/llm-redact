# Multi-stage build; compatible with docker (BuildKit) and podman >= 4.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
# --extra perf (uvloop) + realtime (websockets): the image runs on uvloop via
# uvicorn loop="auto", and CAN serve OpenAI Realtime / Gemini Live upgrades —
# without websockets uvicorn refuses every WS upgrade, so the shipped image
# would silently lack that capability.
# EXTRAS is overridable so release.yml can build the `-rdbms` image variant
# (adds vault-postgres/vault-mysql/crypto for the Helm standalone shared-vault
# path — a server DSN without those extras refuses startup).
ARG EXTRAS="--extra perf --extra realtime"
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS}
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${EXTRAS}

FROM python:3.13-slim-bookworm AS runtime
LABEL org.opencontainers.image.title="llm-redact" \
      org.opencontainers.image.description="Transparent redact/rehydrate proxy for LLM API traffic" \
      org.opencontainers.image.source="https://github.com/asanderson/llm-redact" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --uid 10001 --create-home app && mkdir -p /data && chown 10001:10001 /data
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --chown=10001:10001 scripts/fake_upstream.py /app/scripts/fake_upstream.py
# 0.0.0.0 binds only inside the container's network namespace; the host
# boundary is the publish spec. Document -p 127.0.0.1:8787:8787 — a bare
# -p 8787:8787 exposes the proxy (and its rehydrated secrets) to the LAN.
# The native (non-container) default stays 127.0.0.1. INSECURE_BIND is the
# documented hatch for exactly this confined-wider-bind case: outside a
# container, `serve` refuses any non-loopback bind without mutual TLS.
ENV PATH=/app/.venv/bin:$PATH \
    LLM_REDACT_HOST=0.0.0.0 \
    LLM_REDACT_INSECURE_BIND=1 \
    XDG_DATA_HOME=/data \
    PYTHONUNBUFFERED=1
USER 10001:10001
EXPOSE 8787
# Note: podman warns and ignores HEALTHCHECK when building OCI-format
# images; the compose.yaml healthcheck covers that path.
# Probe the DB-free /healthz (not /status, which queries the vault each call).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('LLM_REDACT_PORT','8787')+'/__llm-redact/healthz')"]
ENTRYPOINT ["llm-redact"]
CMD ["serve"]

FROM runtime AS fake-upstream
ENTRYPOINT ["python", "/app/scripts/fake_upstream.py"]
CMD ["--host", "0.0.0.0", "--port", "9999"]
