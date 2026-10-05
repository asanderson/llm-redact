# Out-of-band PII corpus tooling

Dev-only scripts (not part of the `llm-redact-proxy` wheel) that build the
agent-traffic evaluation corpus no public dataset provides. A local LLM — the
*teacher* — writes coding-agent artifacts with invented personal data, a
human verifies every row, and the frozen set is scored by the NER bench
(`python -m llm_redact.bench.ner --dataset agent-eval --path FILE`,
[docs/ner-bench.md](../../docs/ner-bench.md)).

The data stays private (plan decision D7): generated, verified and frozen
files are written outside every git work tree (each script refuses a path
inside one), as mode-0600 files, by default under
`${XDG_DATA_HOME:-~/.local/share}/llm-redact/pii-corpus/`. Nothing these
scripts print or log contains a generated or audited text value: they print
counts, ids, offsets, types and model names.

| Script | Plan task | What it does |
|---|---|---|
| `generate.py` | T40 | Asks the teacher for tagged artifacts, grounds every span, writes unverified JSONL rows and a run manifest. |

## Requirements

A local [Ollama](https://ollama.com) server with an allowed teacher already
pulled (`ollama pull gemma4:e4b`); the scripts never pull a model. They talk
to `http://127.0.0.1:11434` by default and refuse any non-loopback server
unless `--allow-remote-server` is given, and then only over https (the
server receives every prompt and, for the audit, the corpus text). Requests
ignore proxy environment variables and never follow redirects.

## Generating samples (`generate.py`)

```bash
uv run python scripts/pii_corpus/generate.py --model gemma4:e4b --count 500 --seed 7
# -> ~/.local/share/llm-redact/pii-corpus/generated-gemma4-e4b-7.jsonl (+ .manifest.json)
```

| Option | Meaning |
|---|---|
| `--model NAME:TAG` | The teacher (required): an allowlisted Ollama model with a fixed size tag. |
| `--count N` | Prompts to send (default 100); dropped answers are counted, not retried. |
| `--seed N` | Seeds the prompt variations and every request (`options.seed`; temperature is always 0). |
| `--negatives F` | Share of hard-negative prompts (default 0.25). |
| `--url URL` | The Ollama server (default `http://127.0.0.1:11434`). |
| `--allow-remote-server` | Permit a non-loopback server, https only. |
| `--out PATH` | The output JSONL (default under the private data directory; refused inside a git work tree). |
| `--force` | Replace an existing output file. |

Each row is one JSON line:

```json
{"id": "gemma4-e4b-7-000012", "text": "...", "spans": [{"start": 10, "end": 17, "type": "PERSON"}],
 "teacher": "gemma4:e4b", "prompt_id": "tool-args", "seed": 7}
```

`PATH.manifest.json` records the teacher's digest and license check, the
server kind (loopback or remote), the prompt catalog's version and SHA-256,
the options, whether the run completed, and counts (written rows, spans,
repeats made gold, drops by reason). It holds no text. Exit status: 0 done,
1 a teacher error cut the run short (rows written so far are kept and the
manifest says `"complete": false`), 2 an input problem.

### Grounding

The teacher tags every invented value inline as
`<pii type="TYPE">value</pii>`. A row is kept only when every span is
grounded (`grounding.py`): the tags are well formed and name a type the
prompt asked for; each value is non-empty, unpadded, at most 200 characters
and on one line (an `ADDRESS` may span lines); one value has one type; every
other whole-token occurrence of a tagged value becomes gold too (an
untagged repeat would be scored as a miss the labels caused), and a repeat
that overlaps a different span drops the row; a hard negative carries no
tag, a positive at least one; text holding a placeholder guillemet (`«`,
`»`) is dropped. Dropped rows are counted by reason, never shown.

### Prompt catalog (`prompts.py`, catalog version 1)

One system prompt fixes the tag format, asks for invented values only (names
from many cultures) and lists the types with a one-line hint each:
`PERSON`, `ADDRESS`, `DATE_OF_BIRTH`, `PASSPORT`, `DRIVER_LICENSE`,
`USERNAME`, `ACCOUNT_NUMBER` (the canonical NER types) and `EMAIL`, `PHONE`
(regex-owned, for the bench's structured-regression check). Each user prompt
draws, from a generator seeded by the run seed and the row index:

| `prompt_id` | Artifact | Formats |
|---|---|---|
| `tool-args` | the JSON arguments of a tool call | JSON |
| `tool-result` | the JSON result a tool returns | JSON |
| `diff` | a unified git diff of source, fixtures or seed data | Python, TypeScript, Go, Java, Ruby, YAML, SQL |
| `log` | application or CI log lines | plain text, JSON lines, nginx access log, GitHub Actions log |
| `config` | a service configuration file | YAML, TOML, .env, JSON, INI |
| `commit` | a commit message with author and trailers | git log output, a message body |
| `test-fixture` | a unit test with fixture data | pytest, jest, Go testing, JUnit, RSpec |
| `sql` | SQL statements or query output | PostgreSQL, MySQL, SQLite |
| `chat` | a user's message to a coding assistant pasting data | prose, prose with a code block |
| `traceback` | an exception traceback with locals or the failing request | Python, Java, Node.js |

plus one of twelve scenarios (a CRM migration, an HR onboarding service, a
clinic scheduler, …) and, for a positive, two to four of the types above. A
hard negative (`prompt_id` `<artifact>.negative`) asks for the same kind of
artifact with no personal data at all but with two kinds of name-like
confusers: tool names that are surnames (Jenkins, Hudson), CamelCase class
names, UUIDs and hashes, file paths, bare field names, timestamps and
version numbers. A changed wording bumps `CATALOG_VERSION`.

## Licensing: why these teachers

The teacher's output becomes evaluation data and, later, possibly training
data for a student model shipped to users. A model whose terms reach its
output would pass duties to that student, so only models whose weights are
published under **Apache-2.0** may teach (plan T40). `teacher.py` enforces it
three ways:

1. **Allowlist by exact name and size tag** (checked 2026-10-05 against each
   tag's license layer in the Ollama registry and the weights' Hugging Face
   card): `gemma4` (`e2b`, `e4b`, `12b`, `26b`, `31b`), `mistral` (`7b`),
   `mistral-nemo` (`12b`), `mistral-small` (`24b`), `mistral-small3.2`
   (`24b`), `devstral` (`24b`), `magistral` (`24b`), `ministral-3` (`3b`,
   `8b`, `14b`), `mixtral` (`8x7b`, `8x22b`). A tag may carry a variant
   suffix (`gemma4:12b-it-q8_0`). An untagged or `latest` name is refused
   because it can move to another model (`mistral-small:latest` once pointed
   at the research-licensed 22B), and so is every `cloud` tag (it runs on
   Ollama's servers). `mistral-small3.1:24b` is left out: its registry
   manifest carries no license layer, so the run-time check could not pass.
2. **Denylist by name**, checked first so the reason is explicit: the Llama
   family (the Llama Community License attaches naming and use terms to
   models trained on its output), Qwen and DeepSeek (refused by name for
   lineage and procurement reasons), Gemma 1 to 3n including CodeGemma
   (Gemma Terms of Use), Codestral (Mistral AI Non-Production License) and
   the Mistral AI Research License models (`mistral-large`, `pixtral-large`,
   the original `ministral`, `mistral-small:22b`).
3. **Run-time check**: the server must report the Apache License 2.0 text
   for the model (`/api/show`'s `license`, copied from the registry
   manifest), the model must be local (not a remote or cloud model) and
   already pulled, and its digest goes into the run manifest.

To add a teacher, read its tag's license layer
(`https://registry.ollama.ai/v2/library/NAME/manifests/TAG`, the layer of
media type `application/vnd.ollama.image.license`) and its card, add it to
`ALLOWED` with the check date, and extend `tests/test_pii_corpus.py`.

## Tests

`uv run pytest tests/test_pii_corpus.py` runs everything against a fake
Ollama server (`tests/fake_ollama.py`, an `httpx.MockTransport`); no test
opens a network connection, and no test needs a real model.
