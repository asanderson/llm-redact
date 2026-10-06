# Deployment guide

How to run llm-redact in production. This ties the operational surfaces —
bind policy, vault lifecycle, health probes, and observability — to the
single security goal in [threat-model.md](threat-model.md): *private values
must not reach the provider, and the placeholder↔value mapping must never
leave the machine.* Everything below is subordinate to that.

The proxy is designed to run **next to the tool it serves, as the same
user, on loopback**. That is not a limitation to work around — it is the
trust boundary the design depends on (see the threat model's assumptions).
This guide covers the FOSS core's whole deployment surface — including
team serving over mutual TLS and Kubernetes (the sidecar manifest, the
Helm chart, and HPA autoscaling), which are ungated parts of this
repository. The paid operational subsystems — at-rest vault encryption
and key rotation, the server RDBMS vaults, the audit log and its
off-machine sinks, and OpenTelemetry export — ship in the
`llm-redact-pro` package (**coming soon**; its operator guide ships with
it).

## Choose a bind, and know what it costs

`serve` runs `validate_bind_security` before it opens the socket. The rule
is fail-closed:

| Host | Requirement | Why |
| --- | --- | --- |
| `127.0.0.1` / `::1` (default) | none | A network client of the proxy can read *rehydrated* secrets; loopback keeps that client on the same machine, where a same-UID attacker already has the vault. |
| any non-loopback host | `[tls]` **certfile + keyfile + client_ca** (full mutual TLS) | Every connecting client must present a certificate the CA signed. Without client auth, anyone who can route to the port can read restored secrets and edit detection config. |
| unresolvable hostname | treated as non-loopback | Fail closed rather than guess. |

`serve` refuses to start on a non-loopback host without the mTLS trio.
The single documented escape hatch is `LLM_REDACT_INSECURE_BIND=1`, and it
exists for exactly one confined case: a container that binds `0.0.0.0`
*inside its own network namespace* while the operator publishes it to
loopback only. Setting it anywhere else is on you.

The client-side commands on the proxy's own machine (`llm-redact status`,
`doctor`, `run`, the `plugin install` probe and `vault rotate-key`'s
liveness check) reach it at the configured `host`: a wildcard bind
(`0.0.0.0`, `::`, or empty) at loopback — `127.0.0.1`, or `[::1]` for `::`
— and an IPv6 literal in brackets (`http://[::1]:8787`). `run` exports that
same URL to the tools it wraps.

**Default deployment (recommended): loopback.** Point the tool's base URL
at `http://127.0.0.1:8787` — `http://127.0.0.1:8787/v1` for
`OPENAI_BASE_URL` — or use `llm-redact run -- <tool>`, which injects the
right env vars. Nothing else to configure.

**Remote / shared deployment** — serving clients on other hosts — means
operating a client-certificate PKI: configure the full `[tls]` trio
(`certfile`, `keyfile`, `client_ca`; the bind policy refuses a
non-loopback host without it) and query the running proxy with
`llm-redact status --ca/--cert/--key`. This is part of the FOSS core.

## Containers

The published GHCR image binds `0.0.0.0` inside the container netns and
sets `LLM_REDACT_INSECURE_BIND=1` for that confined case only. **Always
publish it to loopback:**

```bash
docker run -d --name llm-redact \
  -p 127.0.0.1:8787:8787 \
  --stop-timeout 90 \
  -v llm-redact-data:/data \
  -e ANTHROPIC_BASE_URL=… \
  ghcr.io/asanderson/llm-redact:latest
```

`-p 127.0.0.1:8787:8787` — never `-p 8787:8787`, which would expose the
proxy on every interface with client auth disabled. `--stop-timeout 90`
replaces Docker's 10 s stop timeout, which would SIGKILL the proxy during
its shutdown (the off-machine audit sinks' final flush alone may take 45 s;
docs/resilience.md, "Shutdown order"); a compose service sets
`stop_grace_period: 90s`. The image ships the
`perf` (uvloop), `realtime` (WebSocket) and `extract` (pypdf) extras, so it
runs on uvloop, can relay OpenAI Realtime / Gemini Live, and can read PDFs
for `[extraction]` (docs/extraction.md). `XDG_DATA_HOME=/data` holds
the vault and audit DB — mount a volume there for persistence. Released
images are multi-arch (amd64 + arm64).

**The `-ner` image** (`ghcr.io/asanderson/llm-redact:<version>-ner`,
`latest-ner`) is the same image plus the `hf` and `gliner` NER extras, with a
CPU-only torch on both architectures (PyPI's Linux torch wheel brings the
CUDA libraries, which no CPU host uses): about 1 GB more than the stock
image (the CI `container-ner` job prints the built size). It is built,
attested (BuildKit SBOM) and cosign-signed like the stock image, and the
release carries its closure as `llm-redact-ner-image.cdx.json`. It ships
no model: provision them as "Provisioning NER models" below describes — the
image's Hugging Face cache (`HF_HOME`) is `/data/huggingface`, on the data
volume, and `llm-redact models pull --to` folders mounted read-only at
`/models` need no cache at all. To carry the image into a network without
internet access:

```bash
docker pull ghcr.io/asanderson/llm-redact:<version>-ner        # connected side
cosign verify ghcr.io/asanderson/llm-redact@<digest> \
  --certificate-identity-regexp 'https://github.com/asanderson/llm-redact/.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
docker save ghcr.io/asanderson/llm-redact:<version>-ner | gzip > llm-redact-ner.tar.gz
# carry the file in, then on the other side:
docker load < llm-redact-ner.tar.gz
```

### Health probes

Orchestrators should probe the DB-free liveness endpoint, not `/status`
(which reads the vault on every call):

- `GET /__llm-redact/healthz` → `{"status": "ok"}` — liveness.
- `GET /__llm-redact/readyz` → `{"status": "ready", "version": …,
  "realtime": <bool>}` — readiness, including whether the WebSocket extra
  is importable.

The container `HEALTHCHECK` and the compose healthcheck already probe
`/healthz`. Under Kubernetes, wire `healthz` to the liveness probe and
`readyz` to the readiness probe. These endpoints are unauthenticated by
design (metadata only, no secrets) but are still served with the reserved-
path security headers and are provably never forwarded upstream.

Kubernetes deployment is part of the FOSS core: the hardened sidecar
manifest lives at `deploy/k8s-sidecar.yaml` and the Helm chart (sidecar
and standalone modes, optional HPA autoscaling) at
`deploy/helm/llm-redact/` — its `NOTES.txt` and `values.yaml` document
the modes and guardrails.

The chart sets the pod's `terminationGracePeriodSeconds` (value
`terminationGracePeriodSeconds`, default 90, both modes): Kubernetes'
own default of 30 s can SIGKILL the proxy while the off-machine audit
sinks are still doing their final flush (up to 45 s), before the audit
database and the vault close. 90 s covers the proxy's bounded shutdown
steps (57 s) plus 33 s for requests still running; the pod goes away as
soon as the proxy exits, so a fast stop never waits for it. Raise it if
your clients hold long streams open (the request drain has no bound of
its own); the render fails on anything but a non-negative integer, an
absent value (`helm upgrade --reuse-values` from a release made before
it existed) renders 90, and NOTES warns below 57 s. In sidecar mode the value covers the whole pod,
your tool container included. The plain manifest
`deploy/k8s-sidecar.yaml` sets the same 90 s; set it in a manifest of
your own. In standalone mode a `preStop` delay (`preStopSleepSeconds`,
default 5; 0 removes it) keeps the proxy serving until the Service stops
routing new connections to a terminating pod, so a rolling update refuses
none; it counts against the grace period. See docs/resilience.md,
"Shutdown order".

Replicas sharing one vault issue consistent tokens. The durable maps
(Responses chains, stored-object owners, Live resumption handles) are
written on a background thread after each answer, and `[vault] map_writes`
decides when the client gets the answer. With a shared database backend
(`postgresql`, `mysql`, `oracle`, `dbapi`) the default is
`"before_answer"`: an answer that recorded something waits (bounded, off
the event loop) until its write landed, so a follow-up
(`previous_response_id`, a Live resumption, a stored-object read) may reach
ANY replica — no session affinity needed. Past the bound
(`[vault] map_write_wait_seconds`, default 5 s; a database stall) the
answer is sent anyway and counted (`llm_redact_map_write_wait_timeouts_total`;
the wait is part of a request, not of the shutdown budget above); a
follow-up reaching another replica before the write landed is then refused
or sealed, never restored wrong. `map_writes = "background"` sends answers
at once (one write's latency less on those answers) and brings that window
back for every answer: use session affinity at the load balancer then
(docs/resilience.md, "Durable map writes off the event loop").

## Host names the proxy answers to (`allowed_hosts`)

Any web page in your browser can send requests to the proxy on
127.0.0.1 (see the threat model's "Requests from web pages"). Before any
upstream contact the proxy refuses — HTTP 403, WebSocket close 1008:

- a **browser request** (one carrying `Origin` or a `Sec-Fetch-*` header)
  from another origin or site, or addressed to a host name the proxy does
  not answer to (DNS rebinding); and
- a request that would **spend a credential the proxy holds**
  (`[providers.NAME] auth = "identity"`, or a routed upstream with an
  operator key or no key) addressed over plain HTTP to a host name the
  proxy does not answer to.

The proxy answers to `127.0.0.1`, `localhost`, `::1`, its bind host and,
with llm-redact-pro's access gate, the gate's public origin. When clients
reach a plain-HTTP proxy under another name AND it spends its own
credential, list those names:

```toml
allowed_hosts = ["llm-redact", "llm-redact.team.svc.cluster.local"]
```

- A top-level key (before any `[table]`), restart-only. Names only: no
  scheme, port, path or wildcard; IP literals are accepted.
- Typical names: a compose service, `host.docker.internal` (a devcontainer
  reaching a proxy on the host), a Kubernetes Service. The Helm chart's
  standalone mode lists its own Service's in-cluster names; its
  `allowedHosts` value adds more.
- Not needed for CLI tools and SDKs on their own credential (they send no
  browser markers, so any name works), for loopback clients, or for a TLS
  listener: a browser verifies the proxy's certificate against the name it
  resolved, so a rebound page cannot reach it, and clients may use any
  name the certificate covers.
- `llm-redact doctor` reports the configured names, and WARNs when a
  non-loopback plain-HTTP bind spends a proxy-held credential with none
  listed. `/status` counts refusals by kind in
  `request_origin_refusals_total` (`host`, `origin`, `fetch_site`).
- A browser-based client served from another origin (a web chat UI, a
  browser extension, an Electron renderer) is refused: it cannot be told
  apart from a malicious page — unless you list its origin in
  `allowed_origins` (below).

## Browser apps on other origins (`allowed_origins`)

By default a page on another origin — a web chat UI, a local dev server,
an Electron renderer, a browser extension — cannot use the proxy: it
cannot be told apart from a malicious page, and any page that reaches the
proxy could read your redacted values back (every rehydrating route
restores the vault's values into what the provider sends back). To serve
a browser app you trust with those values, list its origin:

```toml
allowed_origins = ["https://chat.example.com", "http://localhost:3000"]
```

- **What listing means.** A page on a listed origin is served like a local
  tool: it can read restored values back through the proxy, and spend
  every credential the proxy holds (`auth = "identity"` providers, routed
  operator keys). So can any code that runs on that origin — an XSS in
  the app, a compromised script it loads. List only origins you trust with
  your redacted values. `llm-redact doctor` WARNs with the list, `/status`
  carries the count (`allowed_origins`), and `llm-redact status` shows it
  in its posture block.
- **Exact origins.** Each entry is a web origin, `scheme://host[:port]`:
  http or https, no path, query, user info or wildcard. It is stored in
  the form browsers send (host lowercased, default port left out, IPv6 in
  brackets) and a request's `Origin` must match it exactly:
  `https://chat.example.com` admits neither `https://app.chat.example.com`
  nor `https://chat.example.com:8443`. `http` is accepted only for this
  machine (`localhost`, names under `.localhost`, loopback addresses):
  anyone on the network path can serve a page as a remote plain-HTTP
  origin. The opaque `null` origin (sandboxed frames, `file://` pages) can
  never be listed, and neither can other schemes (`chrome-extension://`,
  `app://`). A top-level key (before any `[table]`), restart-only.
- **Every other rule still holds.** The request must be addressed to a
  host name the proxy answers to (127.0.0.1, localhost, ::1, the bind
  host, `allowed_hosts`), and a request without an `Origin` header (an
  `<img>`, a `no-cors` fetch) is still refused: there is nothing to check
  against the list. The reserved `/__llm-redact/*` endpoints (the
  dashboard, `/sessions`, …) never consult it.
- **CORS stays the provider's.** For a listed origin the proxy forwards the
  CORS preflight and the request as they are and relays the provider's own
  CORS answer, so the app works only against a provider that accepts calls
  from browsers: the Anthropic and OpenAI SDKs need
  `dangerouslyAllowBrowser: true` (the Anthropic API also wants the
  `anthropic-dangerous-direct-browser-access: true` header the SDK then
  sends). An upstream's own policy still applies to its traffic: a local
  Ollama must allow the origin too (`OLLAMA_ORIGINS`). The proxy's own
  answers (a 400 block, a 413, a 502) carry no CORS headers, so the app
  sees a network error for them — `/__llm-redact/recent` and the proxy's
  log say why. Under `auth = "identity"` a preflight is refused (only the
  API routes llm-redact recognizes are forwarded), so a browser app cannot
  call those providers the usual way; a listed page can still send them
  "simple" requests, which spend the proxy's identity.
- **Realtime.** A listed origin may open a realtime WebSocket (browsers
  apply no CORS to WebSockets; the provider sees the page's `Origin`).
- **The browser may ask first.** Some browsers put a public site's
  requests to 127.0.0.1 behind a local-network-access permission prompt;
  allow it for the app's site.
- **llm-redact-pro's local connector** (`llm-redact connect`) stays
  same-origin-only in this release: point a browser app at the proxy
  itself, not at the connector.

## Service management (native installs)

`llm-redact service install` writes a per-user launchd (macOS) or systemd
(Linux) unit; `service status` / `uninstall` manage it. Let the platform
own restarts and log retention:

- **systemd**: `journalctl -u llm-redact` owns retention. The generated unit
  ships a sandbox (`NoNewPrivileges`, `ProtectSystem=strict`, empty
  `CapabilityBoundingSet`, a `@system-service` syscall filter, and
  `ReadWritePaths` scoped to the XDG data dir so the vault/audit still write).
  Review it with `service install --print-only`; heavy NER extras (torch) may
  need the syscall filter loosened.
- **containers**: cap the log driver (`--log-opt max-size=10m --log-opt
  max-file=3`), since the proxy does not rotate its own logs.

Config changes apply on **SIGHUP** without dropping in-flight requests
(`kill -HUP $(pgrep -f 'llm-redact serve')`, or `docker kill
--signal=HUP`). A request keeps the configuration it was admitted under:
its provider's upstream and authorization mode (`auth`, the authorizer
it holds) are read once, before its body, so a reload that lands while a
body is still arriving never signs a pass-through request with the
proxy's identity. Detection rules, allowlists, NER, fuzzy rehydration, note
injection, `max_body_bytes`, `max_body_strings`, upstream URLs, and the routing sections
`[upstreams]`/`[routing]`/`[prices]` (llm-redact-pro) hot-reload; vault, audit,
host, port, allowed_hosts, allowed_origins, log, TLS, OTel, users, email, extraction, and overrides changes warn
"require restart"
and keep the old value; so do sections a plugin adds, such as llm-redact-pro's
`[auth]`. A broken config file is logged and ignored — the running config
stays live. There is deliberately no HTTP reload endpoint (it would be a
CSRF-reachable mutating endpoint on loopback).

The same guarded flow is available from inside an agent: the
`/llm-redact:config-edit` plugin command reads the effective config,
edits the file, gates on `serve --check`, reloads via SIGHUP, and reads
back the coverage posture — and `/llm-redact:doctor` runs the same
read-only preflight as the CLI ([plugins.md](plugins.md)).

### Reloads and open realtime connections

The in-flight rule above covers one request. An open realtime (WebSocket)
connection lasts far longer (realtime sessions run for tens of minutes), so
a reload reaches it: a relay redacts and forwards every client frame under
the configuration it was admitted with, and a reload that changes that
configuration closes it. What a connection was admitted with is:

- its provider's `[providers.NAME]` settings (`upstream_base_url`,
  `enabled`, `detection`, `auth`, `region`);
- the upstream authorizer that opened it under `auth = "identity"` (a
  change to any identity provider's `auth`, `region` or upstream rebuilds
  them all);
- everything under `[detection]` (rules, deny strings, allowlists, modes,
  NER, languages).

When a reload (SIGHUP or the config editor) changes any of these, the proxy
closes the connection on both sides. The client gets WebSocket close code
**1012** (Service Restart: reconnect), with a reason that names what
changed and never a value. Reconnecting puts the client under the new
configuration in the same vault session, so tokens issued before the
reload still restore.

Reloads run on the proxy's event loop, and the relay checks for a reload
between reading a client frame and redacting it, so no frame it reads after
the reload is redacted or forwarded under the old configuration. The one
exception is a frame the relay had already read and handed to the
provider's socket when the reload ran. A connection still being authorized
under the proxy's identity when the reload lands is never dialled: it is
closed 1012 and recorded as a 503. One whose upstream handshake was already
under way completes it (the in-flight rule above) and is then closed before
it relays a frame. Each connection still gets its one `/recent` and audit
row when it closes.

Two settings are read per frame and apply to open connections at once:
`inject_system_note` and `max_body_strings`. `rehydration.fuzzy` applies to
connections opened after the reload. A reload that changes nothing an open
connection depends on (another provider, the body limits, note injection,
routing) leaves it open.

A client still writing frames when the proxy closes it may see a connection
reset instead of the 1012 frame, because the server closes its socket right
after sending the close frame. Treat either as "reconnect".

## Provisioning NER models

NER is opt-in (`[detection.ner] enabled = true`, docs/detection.md). The
`gliner`, `gliner2` and `hf` backends load Hugging Face models, and with
`[detection.ner] allow_download = false` — the default — the proxy never
fetches one: every model loads from local files, and a model that is not
there stops the startup with an error naming it, its revision and
`llm-redact models pull`. So provisioning is a step of its own, done before
the proxy starts with NER on (the [model supply chain
diagram](diagrams/model-supply-chain.png) shows the whole path):

- **On the proxy's machine:** run `llm-redact models pull --config PATH` as
  the user the proxy runs as, with network access to huggingface.co. It
  fetches each configured model (and a GLiNER model's base model) at its
  pinned revision into that user's Hugging Face cache (`HF_HOME`, default
  `~/.cache/huggingface`), which the proxy reads. `llm-redact models verify`
  then checks offline that every file the load reads is there, and
  `llm-redact doctor` reports the same under `models`. spaCy, Presidio and
  Stanza models are Python packages or library downloads, not Hub
  snapshots: `llm-redact models list` prints their install commands.
- **For a machine or container that cannot write a cache** (or must not
  reach the network): run `llm-redact models pull --to DIR --as /models` on
  a connected machine. It writes one self-contained folder per model plus a
  manifest of SHA-256 sums, and prints the `[detection.ner.models]` lines
  that load them from `/models`. Mount `DIR` there read-only, check it with
  `llm-redact models verify --dir /models`, and paste the printed lines into
  the configuration. A folder needs no cache, no base-model assembly and no
  network.
- **`allow_download = true`** lets the proxy's startup (`serve`,
  `serve --check`) fetch what is missing instead — pinned files only, never
  request content — which needs network access to huggingface.co and a
  writable `HF_HOME` at every start. Reloads never download: a SIGHUP or
  config-editor change that names a model not on disk is refused and the
  running configuration is kept, whatever `allow_download` says.

Where the files live under a hardened deployment:

- **systemd** (`llm-redact service install`): the unit runs with
  `ProtectHome=read-only` and writes only the llm-redact data, config and
  state directories. The Hugging Face cache stays readable, so models pulled
  beforehand load; but the unit cannot write it, so run `llm-redact models
  pull` as the user outside the unit (and leave `allow_download` off). The
  folders the `gliner` backend assembles for urchade-style models (under
  `~/.local/share/llm-redact/models/gliner/`) are inside the writable data
  directory.
- **Containers and Helm** (`readOnlyRootFilesystem: true`): the image's
  home directory, and with it the default Hugging Face cache, is on the
  read-only root filesystem. Pre-provision a models volume — the `pull --to`
  folders, mounted read-only at `/models` and named in
  `[detection.ner.models]` — or mount a pre-filled Hugging Face cache and
  point `HF_HOME` at it (a cache needs a writable `XDG_DATA_HOME` for the
  GLiNER base-model assembly; `/data` is). Either way the pod loads models
  with no network at all.

## Offline installs (no container)

For a machine without internet access and without containers, build a
wheelhouse on a connected machine of the same OS, CPU architecture and
Python minor version, from a checkout of the release you install. Every
locked package is downloaded at its `uv.lock` version and checked against
its hash; torch comes from the PyTorch CPU index at its locked version
(`scripts/cpu_torch.py` splits it and its GPU-only dependencies out of the
export):

```bash
# connected side, in the llm-redact checkout
mkdir wheelhouse
uv export --frozen --no-dev --no-emit-project --extra hf --extra gliner \
  | python3 scripts/cpu_torch.py requirements wheelhouse/requirements.txt > wheelhouse/torch.txt
python3 -m pip download --require-hashes --no-deps --only-binary=:all: \
  -r wheelhouse/requirements.txt -d wheelhouse
python3 -m pip download --no-deps --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cpu -r wheelhouse/torch.txt -d wheelhouse
uv build --wheel --out-dir wheelhouse          # llm-redact-proxy itself
```

Carry `wheelhouse/` (and `scripts/cpu_torch.py`) in, then install with no
index at all:

```bash
python3 -m venv /opt/llm-redact
pip=/opt/llm-redact/bin/pip
$pip install --no-index --find-links wheelhouse --no-deps -r wheelhouse/torch.txt
$pip install --no-index --find-links wheelhouse --no-deps --require-hashes -r wheelhouse/requirements.txt
$pip install --no-index --find-links wheelhouse --no-deps llm-redact-proxy
$pip check
/opt/llm-redact/bin/python scripts/cpu_torch.py check   # no CUDA library got in
/opt/llm-redact/bin/llm-redact doctor                     # ner: hf, gliner backends importable
```

Add `--extra presidio`, `--extra gliner2` or `--extra stanza` to the export
for those backends (spaCy and Stanza models are carried separately:
`llm-redact models list` prints their install commands). The CI `airgap`
job runs exactly these commands, the install inside a network namespace
with no route. Then provision the models as below.

## Vault lifecycle in production

The sqlite vault (`[vault] backend = "sqlite"`) is the one piece of state
whose corruption is unacceptable: a lost or reused token number would
silently rehydrate the *wrong* secret. Treat it accordingly.

- **Back it up consistently.** `llm-redact vault backup <dest>` takes a
  single-file snapshot through the SQLite online-backup API — it reads
  *through* the WAL, so it is safe against a running proxy and cannot tear
  a mapping the way `cp vault.db` can. The destination is created `0600`.
- **Verify integrity.** `llm-redact vault verify` is a read-only sweep:
  it checks that every row's token matches its own number (in
  `1..999999999`; the UNIQUE constraints already bar a reused number, so
  `MAX(n)` truly bounds what was issued) and, for encrypted vaults, that
  every ciphertext decrypts and every HMAC index matches its plaintext.
  Gaps in the numbering are noted, not failed: a request carrying tokens
  its session never issued has its new values numbered past them (the
  token floor). It prints sessions/types/counts only, never a value, and
  exits non-zero on any failure. Run it after a restore or before a key
  rotation.
- **Bound growth.** `[vault] session_ttl_days = N` prunes whole sessions
  idle longer than N days via a background task (never the active
  session, nor a session the session router marks durable, such as a
  named user's own copy of it under llm-redact-pro); `0` (default)
  disables it. For manual control,
  `llm-redact sessions prune --older-than 90d` deletes whole idle sessions
  (partial deletion could reuse a still-referenced number).
  `POST /__llm-redact/sessions/prune` (and the llm-redact-pro dashboard,
  which calls it) does the same, safe against the live process. Every
  delete **retires** the session's numbers (a small `retired_numbers` row
  per deleted session, no values, never removed): a new value in that
  session is numbered above them, so a token of the deleted session — in
  a provider's history, in another proxy instance's memory, on a realtime
  connection still open — is never restored to a different value. Several
  instances may share one vault file: each drops a session another one
  deleted from its memory within a second. A session counts as idle when
  it issued no NEW value for N days, and each instance's prune spares only
  its own static session (and the router's durable ones) — so on a shared
  vault one instance can prune another's static session that kept re-using
  known values. That costs the deleted tokens their restoration (they pass
  through verbatim), never a wrong value; set `session_ttl_days` above the
  longest such pause, or leave it `0` on a shared vault. A delete also
  removes the session's rows from the vault's Live resumption handle map
  (`handle_sessions`: digests of the Gemini / Vertex AI Live resumption
  handles llm-redact-pro records, each mapped to its session; at most
  1,024 per session and, beyond 10,000 rows, only rows of sessions holding
  no values are trimmed, oldest first), so a handle into a deleted session
  is no longer honoured.

At-rest **encryption** of the vault (`[vault] encryption = "fernet"`), **key
rotation** (`vault rotate-key`), and the **server RDBMS** backends are Pro
features of the `llm-redact-pro` package (**coming soon**; documented with
the package).

`llm-redact doctor` ties these together as a read-only preflight: it
checks config parse, the bind policy, proxy reachability and version skew,
vault/audit file permissions (`0600`/`0700`), missing extras, that the
fernet key actually **matches** the vault (not merely that it is set), and
loudly reports every coverage opt-out (warn-mode types, per-provider
detection off, MCP exempt servers, language-scoped-out rules). It exits
non-zero on any FAIL and never prints a value. Run it in a pre-deploy step.

## Observability

All telemetry is **metadata only** — types, counts, paths, durations;
never a value or a placeholder id. That contract holds across every sink.

- **Prometheus**: scrape `GET /__llm-redact/metrics`. Request-duration is
  labeled by `provider` and `streamed`, so per-provider p95 and
  stream-vs-non-stream latency are visible. Detection/rehydration counters
  are per type. `llm_redact_upstream_errors_total{provider}` counts transport
  faults the proxy failed closed as 502 (the `LlmRedactUpstreamErrors` alert
  fires on a sustained rate). Ready-to-use scrape config, alert rules, and an
  importable Grafana dashboard live in [`deploy/`](../deploy) — see
  [observability.md](observability.md).
- **Live tail**: `GET /__llm-redact/recent` (last 200 rows) and the SSE
  feed `GET /__llm-redact/events` (what the llm-redact-pro dashboard
  subscribes to) work without the audit DB.
- **JSON logs**: `[log] format = "json"` (or `serve --log-format json`)
  switches to one JSON object per line for log shippers — content is
  unchanged (paths, statuses, counts; never values or headers).

The **audit log** (`[audit]`, with the tamper-evident chain, the zero-loss
`required` mode — "no audit row, no service" — and off-machine S3/Azure
sinks) and **OpenTelemetry** export (`[otel]`) are Pro features of the
`llm-redact-pro` package (**coming soon**; documented with the package).

## Performance and platform posture

- **Event loop**: `pip install 'llm-redact-proxy[perf]'` adds uvloop;
  uvicorn picks it up automatically — no configuration.
- **FIPS 140-3**: all cryptographic uses are FIPS-approved algorithm
  selections; run `llm-redact fips-check` on the host and see
  [fips.md](fips.md) for validated-host deployment.

## Coverage honesty

Several settings deliberately let some traffic through unredacted, and the
docs must never imply otherwise. Each is surfaced in `/status`, by
`doctor`/`status` posture output — never silently:

- `warn` mode forwards the matched value (and anything a longer warn match
  overlaps) upstream — it is observation only.
- `[providers.NAME] detection = false` forwards that whole provider's
  requests unredacted (rehydration stays on).
- `[detection.mcp] exempt_servers` bypasses detection for the named MCP
  servers' content blocks.
- `[detection] languages` does not build national-id rules scoped outside
  the listed languages.

If you enable any of these, `llm-redact doctor` and `llm-redact status`
will tell you — that is the intended way to confirm your deployment's
actual coverage.
