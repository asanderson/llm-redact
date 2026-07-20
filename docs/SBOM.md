# Software bill of materials

Every package llm-redact can pull in, and where the authoritative
machine-readable records live. Three layers:

- **This document** — the human-readable inventory: names, what each
  package is for, and which install path brings it in. Pinned to
  `pyproject.toml` by `tests/test_sbom_doc.py`, so a dependency change
  that forgets this page fails CI.
- **[dependencies.md](dependencies.md)** — the why-chosen record: the
  selection rationale and the alternatives rejected, per package.
- **CycloneDX SBOM** — the machine-readable artifact
  (`llm-redact-runtime.cdx.json`) built from the frozen runtime closure
  and attached to every GitHub Release, covered by the same Sigstore
  build-provenance attestation as the wheel/sdist. The container image
  additionally carries a BuildKit-generated SBOM attestation in GHCR
  beside its keyless cosign signature.

Exact pinned versions are deliberately not repeated here — they live in
[`uv.lock`](../uv.lock) (the committed resolution) and in each
release's CycloneDX asset. Regenerate the runtime closure locally with:

```bash
uv export --frozen --no-dev --no-emit-project
```

## Runtime (what `pip install llm-redact-proxy` installs)

Three direct dependencies — deliberately the whole audit surface — plus
their transitive closure:

| Package | Role |
| --- | --- |
| **httpx** | Upstream HTTP client: async, streaming bodies, transport seam for in-process fake-upstream testing. |
| **starlette** | ASGI app layer: routing, WebSockets, streaming responses — no validation machinery, bodies forward verbatim. |
| **uvicorn** | ASGI server; `loop="auto"` / WS auto-selection let the `perf` and `realtime` extras activate without code changes. |

Transitive closure (from `uv export`): `anyio`, `certifi`, `click`,
`h11`, `httpcore`, `idna`, plus `colorama` (Windows only) and
`typing-extensions` (Python < 3.13 only).

Configuration and CLI use the standard library (`tomllib`, `argparse`,
dataclasses); Prometheus metrics text, the AWS SigV4 and Azure
SharedKey signers, and the audit HMAC chain are hand-rolled on stdlib
`hmac`/`hashlib`. Vendored code: a keccak-256 implementation in
`detection/wallet_checksums.py` (hashlib ships only NIST SHA3, whose
padding differs), pinned to the published Keccak test vectors.

## Optional extras (opt-in, never installed by default)

| Extra | Packages | Purpose |
| --- | --- | --- |
| `ner` | `spacy` | Person-name NER, small-footprint English-first default backend. |
| `gliner` | `gliner` | Zero-shot NER, robust on unusual names; separate extra because it pulls torch + transformers. |
| `presidio` | `presidio-analyzer` | Microsoft's FOSS PII analyzer layered over spaCy (recognizers + context scoring). |
| `stanza` | `stanza` | Stanford Stanza NER, 60+ languages — the multilingual complement to spaCy. |
| `hf` | `transformers` | Any Hugging Face token-classification checkpoint as a detector; emits confidences. |
| `crypto` | `cryptography` | At-rest Fernet encryption for the vault (`[vault] encryption = "fernet"`). |
| `vault-postgres` | `psycopg[binary]` | PostgreSQL driver for the Pro RDBMS vault backend. |
| `vault-mysql` | `PyMySQL` | Pure-Python MySQL/MariaDB driver for the Pro RDBMS vault backend. |
| `vault-oracle` | `oracledb` | Oracle thin-mode driver (no Instant Client) for the Pro RDBMS vault backend. |
| `keyring` | `keyring` | Vault key in the OS keychain instead of an env var (`llm-redact vault set-key`). |
| `perf` | `uvloop` | Faster event loop, auto-selected by uvicorn's `loop="auto"`. |
| `realtime` | `websockets` | The realtime WS relay — serves both uvicorn's WS protocol and the upstream wss client. |
| `otel` | `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http` | Metadata-only traces/counters over OTLP/HTTP. |

The NER extras' heavyweight transitive dependencies (torch,
transformers, pydantic via presidio) never touch the request-forwarding
path — detectors only read strings and return spans.

## Development toolchain (dev group; never ships in the wheel)

`pytest`, `pytest-asyncio`, `ruff`, `mypy`, `hypothesis` (property
tests), `mutmut` (mutation assurance), `coverage` (complexity-coverage
gate), plus `cryptography` and `websockets` so the crypto and realtime
paths are always exercised by the suite.

## Verifying a release

```bash
gh release download vX.Y.Z --repo asanderson/llm-redact --dir /tmp/rel
gh attestation verify /tmp/rel/llm-redact-runtime.cdx.json \
  --repo asanderson/llm-redact
```

The same `gh attestation verify` works for the wheel and sdist;
container verification (cosign, digest-pinned) is documented in
[deployment.md](deployment.md).
