# Dependencies: what ships, and why each package was chosen

The machine-readable companion is the CycloneDX SBOM attached to every
GitHub Release (`llm-redact-runtime.cdx.json`, the exported runtime
closure) and the BuildKit SBOM attestation on the container image. This
page is the human-readable half: every direct dependency and the reason
it is here. `tests/test_dependencies_doc.py` diffs this page against
pyproject.toml in both directions, so it cannot go stale silently.

## Runtime (deliberately exactly three)

| Package | Why chosen |
|---|---|
| `httpx` | The upstream HTTP client: async, streaming request/response bodies (the proxy re-frames streams byte-level), HTTP/2 capable, and its `AsyncBaseTransport` seam is what lets the whole integration suite run in-process against fake upstreams. |
| `starlette` | The ASGI app layer: routing, WebSocket support, streaming responses — without FastAPI's validation machinery, which the proxy must NOT have (unknown JSON fields are forwarded verbatim, never validated or reshaped; that rule is why pydantic is banned from the request path). |
| `uvicorn` | The ASGI server: production-grade, SIGHUP-friendly, and its `loop="auto"` / websocket protocol auto-selection is what lets the `perf` and `realtime` extras light up capabilities without any serve-code changes. |

Everything else in the request path is stdlib (`tomllib`, `argparse`,
`dataclasses`, `sqlite3`, `hashlib`/`hmac`, `zlib` for eventstream CRCs) —
a deliberate supply-chain ceiling enforced by the versioning policy and
checked by the weekly gating pip-audit job.

## Extras (opt-in; never on the default install)

| Extra | Package(s) | Why chosen |
|---|---|---|
| `ner` | `spacy` | Person-name detection. Chosen over transformer NER for footprint (tens of MB, ~1-5 ms/string on CPU) and MIT license; the `Detector` protocol keeps heavier backends optional rather than default. |
| `gliner` | `gliner`, `onnxruntime`, `torch` | Zero-shot NER, more robust on unusual names; deliberately separate from `ner` because it hard-depends on torch + transformers (gigabytes). `torch` is listed for its floor (below). `onnxruntime` (MIT) runs a model's ONNX export (`[detection.ner.onnx]`): gliner required it up to 0.2.28, and 0.2.29 moved it to gliner's own optional `onnx` extra (checked on PyPI 2026-10-06), so this extra lists it itself. |
| `gliner2` | `gliner2`, `torch`, `transformers`, `peft`, `safetensors`, `numpy` | Fastino's GLiNER2: zero-shot, schema-driven extraction that reports each entity's character span and a confidence (`score_threshold` applies). Separate from `gliner` (a different package and checkpoint format). gliner2's own `local` extra holds its model dependencies but caps transformers below 5, which the `gliner` and `hf` extras resolve to, so this extra lists them itself without that cap (checked 2026-10-06 with gliner2 2.0.0 on transformers 5.10.1); `peft` is imported by gliner2's extraction runtime. The gliner2 package also contains a client for Fastino's hosted API (`requests`); llm-redact never uses it — models run locally only. |
| `presidio` | `presidio-analyzer` | Microsoft's FOSS PII analyzer: pattern recognizers + checksums + context scoring over the same spaCy pipeline; overlapping entity types fold into the built-in placeholder names. Pulls pydantic — acceptable because extras never touch the body-forwarding path. |
| `stanza` | `stanza`, `torch` | Stanford Stanza NER for 60+ languages — the multilingual complement to the English-first spaCy default. Runs on torch; separate extra for the same reason as gliner. No per-entity confidence. |
| `hf` | `transformers`, `torch` | Any Hugging Face `token-classification` checkpoint as a detector (multilingual/domain-tuned models). transformers declares torch only as its own extra, so `hf` lists `torch` itself — without it no model can run. Same class as gliner; emits confidences so `score_threshold` applies. |
| `crypto` | `cryptography` | Vault encryption (Fernet + HKDF + HMAC index). The canonical, FIPS-aware Python crypto library; nothing hand-rolled. |
| `vault-postgres` | `psycopg` | PostgreSQL driver for the Pro RDBMS vault (psycopg 3, the maintained line; `[binary]` wheel so no libpq build). Vault point lookups only — never the body-forwarding path. |
| `vault-mysql` | `PyMySQL` | Pure-Python MySQL/MariaDB driver — no C extension, no client library, easiest install story of the MySQL drivers. |
| `vault-oracle` | `oracledb` | Oracle's own thin-mode driver: pure Python, no Instant Client required. Covers the connect-to-existing-corporate-Oracle case. The generic `backend = "dbapi"` needs no extra — it imports whatever DB-API 2.0 module the operator names. |
| `keyring` | `keyring` | OS-keychain storage for the vault key, so the key need not live in an env var. |
| `perf` | `uvloop` | Drop-in event-loop speedup; uvicorn's `loop="auto"` picks it up with zero configuration. |
| `realtime` | `websockets` | One package serves BOTH sides of the realtime relay: uvicorn's server-side WebSocket protocol (auto-enabled when importable) and the upstream wss client. Floor 15.0: the relay overrides the asyncio client's redirect hook (added in 13.1) and forwards the client's `User-Agent`, which clients before 15.0 send a second time. |
| `otel` | `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http` | Metadata-only telemetry export over OTLP/HTTP, the vendor-neutral standard; scoped SDK providers, never process globals. |
| `extract` | `pypdf` | PDF text layers for [document extraction](extraction.md) (`[extraction]`): pure Python, BSD-3-Clause, imported only in the isolated, resource-limited extraction worker process — never on the request path. The OOXML/ODF, markup and RTF readers are stdlib. |
| `bench-data` | `huggingface_hub`, `pyarrow` | The [NER bench](ner-bench.md)'s published datasets only (`python -m llm_redact.bench.ner --dataset openpii`, …): `huggingface_hub` downloads a dataset file at a pinned revision, `pyarrow` reads the parquet ones a batch at a time. Never imported by the proxy. Floor `pyarrow>=14.0.1`: 14.0.1 fixed CVE-2023-47248 (code execution when reading an untrusted IPC/Parquet file), and a downloaded dataset is exactly such a file. |

**torch (the `gliner`, `gliner2`, `stanza` and `hf` extras).** Each of the
four extras requires `torch>=2.6`: torch 2.6 fixed CVE-2025-32434, a bypass of
`torch.load(weights_only=True)`, which is how GLiNER and GLiNER2 load a
`pytorch_model.bin` checkpoint, and none of the four libraries asks for
that floor itself. `llm-redact doctor` FAILs a configured torch backend
when torch is missing or older than 2.6. PyPI's Linux torch wheel brings
the NVIDIA CUDA libraries (several GB), and `uv sync --extra hf` installs
that locked build. On a CPU-only host, install torch from the PyTorch CPU
index first; pip (or `uv pip`) then keeps it, since it satisfies the floor:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install 'llm-redact-proxy[hf]'
```

From a checkout, `scripts/cpu_torch.py` does the same with every other
package exactly as `uv.lock` pins it (hash-checked): it splits torch and its
GPU-only dependencies (`triton`, `nvidia-*`, `cuda-*`, which PyPI's Linux
torch pulls on x86_64 and aarch64 alike) out of `uv export`, so torch is
installed at its locked version from the CPU index and the rest with
`--require-hashes`, then `cpu_torch.py check` fails if a CUDA library got
in. torch is hash-checked too: `uv.lock` holds only the PyPI wheels'
digests, so the script records the CPU index's SHA-256 of every wheel of
the locked torch (`CPU_WHEELS`, a test pins it to `uv.lock`'s version) and
writes them under the `torch==<version>+cpu` pin it prints
(`requirements --macos`: the plain macOS wheel, which has no `+cpu`
build), which is installed with `--require-hashes` as well. It is the one
CPU-torch recipe: the `-ner` container image, its release SBOM, the CI
`airgap` job, the offline wheelhouse recipe (deployment.md, "Offline
installs") and `scripts/ner_ci_env.sh` (the environment of the
`ner-models` and `ner-eval` CI jobs, docs/ner-bench.md "CI") are all
built this way. A `[tool.uv.sources]`
entry pointing torch at the CPU index was not used: it would change what
`uv sync --extra hf` installs for every Linux user (a GPU host included)
and rewrite `uv.lock`, and a `pip install` from PyPI ignores it anyway.

## Development group

The dev group (`uv sync`) additionally carries the test/lint toolchain
(pytest, pytest-asyncio, ruff, mypy, hypothesis, mutmut) plus
`cryptography`, `websockets` and `pypdf` so the whole suite exercises the
crypto, realtime, document-extraction and mutation-assurance paths without
extra flags. Dev
dependencies never ship in the wheel.

## Reproducible builds

The wheel and sdist are byte-reproducible: two builds of the same tree
under a pinned `SOURCE_DATE_EPOCH` produce identical artifacts, checked
on every CI run by the `reproducible build` job. That is what makes the
release artifacts independently verifiable against the source they claim
to come from — any divergence (an embedded timestamp, nondeterministic
file ordering) fails the job.
