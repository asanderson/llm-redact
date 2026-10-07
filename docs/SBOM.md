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
  beside its keyless cosign signature. The `-ner` image variant (the
  `hf` and `gliner` extras on a CPU-only torch) gets the same, and the
  Release also carries `llm-redact-ner-image.cdx.json`: that image's Python
  closure (its extras' locked packages without torch's CUDA and triton
  dependencies, and torch as the PyTorch CPU index's `+cpu` build with the
  wheel digests `scripts/cpu_torch.py` records).

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
| `gliner` | `gliner`, `onnxruntime`, `torch` | Zero-shot NER, robust on unusual names; separate extra because it pulls torch + transformers. `onnxruntime` (MIT) runs ONNX weights (`[detection.ner.onnx]`); listed directly since gliner 0.2.29 made it optional. |
| `gliner2` | `gliner2`, `torch`, `transformers`, `peft`, `safetensors`, `numpy` | Fastino's GLiNER2 zero-shot extraction with character spans; lists gliner2's model dependencies itself (its own `local` extra caps transformers below 5). Its hosted-API client is never used. |
| `presidio` | `presidio-analyzer` | Microsoft's FOSS PII analyzer layered over spaCy (recognizers + context scoring). |
| `stanza` | `stanza`, `torch` | Stanford Stanza NER, 60+ languages — the multilingual complement to spaCy. |
| `hf` | `transformers`, `torch` | Any Hugging Face token-classification checkpoint as a detector, PII models included; emits confidences. `transformers>=4.56` (the float32 `dtype` the backend loads every model with), `torch>=2.6`. |
| `crypto` | `cryptography` | At-rest Fernet encryption for the vault (`[vault] encryption = "fernet"`). |
| `vault-postgres` | `psycopg[binary]` | PostgreSQL driver for the Pro RDBMS vault backend. |
| `vault-mysql` | `PyMySQL` | Pure-Python MySQL/MariaDB driver for the Pro RDBMS vault backend. |
| `vault-oracle` | `oracledb` | Oracle thin-mode driver (no Instant Client) for the Pro RDBMS vault backend. |
| `keyring` | `keyring` | Vault key in the OS keychain instead of an env var (`llm-redact vault set-key`). |
| `perf` | `uvloop` | Faster event loop, auto-selected by uvicorn's `loop="auto"`. |
| `realtime` | `websockets` | The realtime WS relay — serves both uvicorn's WS protocol and the upstream wss client. |
| `otel` | `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http` | Metadata-only traces/counters over OTLP/HTTP. |
| `extract` | `pypdf` | PDF text layers for document extraction, in the isolated extraction worker only. |
| `bench-data` | `huggingface_hub`, `pyarrow` | The NER bench's published datasets (download at a pinned revision; parquet reading). Bench only, never the proxy. |

The NER extras' heavyweight transitive dependencies (torch,
transformers, pydantic via presidio) never touch the request-forwarding
path — detectors only read strings and return spans. The `gliner`,
`gliner2`, `stanza` and `hf` extras each require `torch>=2.6` directly (the release
that fixed CVE-2025-32434); on Linux, PyPI's torch wheel also brings the
NVIDIA CUDA libraries, which a CPU-only install avoids by taking torch from
the PyTorch CPU index ([dependencies.md](dependencies.md)); the `-ner` image,
the CI air-gap job and the offline wheelhouse recipe do so with
`scripts/cpu_torch.py`, which hash-checks that wheel against the CPU index
digests it records for the locked version.

## Development toolchain (dev group; never ships in the wheel)

`pytest`, `pytest-asyncio`, `ruff`, `mypy`, `hypothesis` (property
tests), `mutmut` (mutation assurance), `coverage` (complexity-coverage
gate), plus `cryptography`, `websockets` and `pypdf` so the crypto,
realtime and document-extraction paths are always exercised by the suite,
and `pyyaml` for the k8s manifest and Helm chart render tests.

## Verifying a release

```bash
gh release download vX.Y.Z --repo asanderson/llm-redact --dir /tmp/rel
gh attestation verify /tmp/rel/llm-redact-runtime.cdx.json \
  --repo asanderson/llm-redact
```

The same `gh attestation verify` works for the wheel and sdist;
container verification (cosign, digest-pinned) is documented in
[deployment.md](deployment.md).
