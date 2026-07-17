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
| `gliner` | `gliner` | Zero-shot NER, more robust on unusual names; deliberately separate from `ner` because it hard-depends on torch + transformers (gigabytes). |
| `presidio` | `presidio-analyzer` | Microsoft's FOSS PII analyzer: pattern recognizers + checksums + context scoring over the same spaCy pipeline; overlapping entity types fold into the built-in placeholder names. Pulls pydantic — acceptable because extras never touch the body-forwarding path. |
| `stanza` | `stanza` | Stanford Stanza NER for 60+ languages — the multilingual complement to the English-first spaCy default. Pulls torch; separate extra for the same reason as gliner. No per-entity confidence. |
| `hf` | `transformers` | Any Hugging Face `token-classification` checkpoint as a detector (multilingual/domain-tuned models). Pulls transformers + torch (same class as gliner); emits confidences so `score_threshold` applies. |
| `crypto` | `cryptography` | Vault encryption (Fernet + HKDF + HMAC index). The canonical, FIPS-aware Python crypto library; nothing hand-rolled. |
| `vault-postgres` | `psycopg` | PostgreSQL driver for the Pro RDBMS vault (psycopg 3, the maintained line; `[binary]` wheel so no libpq build). Vault point lookups only — never the body-forwarding path. |
| `vault-mysql` | `PyMySQL` | Pure-Python MySQL/MariaDB driver — no C extension, no client library, easiest install story of the MySQL drivers. |
| `vault-oracle` | `oracledb` | Oracle's own thin-mode driver: pure Python, no Instant Client required. Covers the connect-to-existing-corporate-Oracle case. The generic `backend = "dbapi"` needs no extra — it imports whatever DB-API 2.0 module the operator names. |
| `keyring` | `keyring` | OS-keychain storage for the vault key, so the key need not live in an env var. |
| `perf` | `uvloop` | Drop-in event-loop speedup; uvicorn's `loop="auto"` picks it up with zero configuration. |
| `realtime` | `websockets` | One package serves BOTH sides of the realtime relay: uvicorn's server-side WebSocket protocol (auto-enabled when importable) and the upstream wss client. |
| `otel` | `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http` | Metadata-only telemetry export over OTLP/HTTP, the vendor-neutral standard; scoped SDK providers, never process globals. |

## Development group

The dev group (`uv sync`) additionally carries the test/lint toolchain
(pytest, pytest-asyncio, ruff, mypy, hypothesis, mutmut) plus
`cryptography` and `websockets` so the whole suite exercises the crypto,
realtime, and mutation-assurance paths without extra flags. Dev
dependencies never ship in the wheel.

## Reproducible builds

The wheel and sdist are byte-reproducible: two builds of the same tree
under a pinned `SOURCE_DATE_EPOCH` produce identical artifacts, checked
on every CI run by the `reproducible build` job. That is what makes the
release artifacts independently verifiable against the source they claim
to come from — any divergence (an embedded timestamp, nondeterministic
file ordering) fails the job.
