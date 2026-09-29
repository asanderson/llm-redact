# Changelog

All notable changes to llm-redact are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

Convention: entries land in `[Unreleased]` with the change that introduces
them. A release moves `[Unreleased]` into a dated version section, bumps
`__version__` in `src/llm_redact/__init__.py` (the single source of truth),
and tags `vX.Y.Z`.

## [Unreleased]

### Security

- Tool results and documents nested under keys named like protocol fields (`data`,
  `name`, `id`, `status`, …) are now redacted; their whole subtree used to be
  skipped. Only scalar values of structural keys are skipped now, and tool-call
  arguments, tool results and documents are walked with no skips at all. This
  affected Gemini/Vertex `functionResponse`, Vertex Live `toolResponse`, Bedrock
  Converse `toolResult`, Cohere `documents` and realtime events.
- Request paths with `.`/`..` segments (including `%2E` and backslash forms) are
  refused with 400 (WebSocket: 1011) before any upstream contact.
- Under `auth = "identity"`:
  - more credential query and header variants are stripped (`$key`, `userProject`,
    `quotaUser`, `subscription-key`, `password`, `passwd`, `oauth_token`, and the
    matching headers);
  - only the `realtime` and `openai-beta.*` WebSocket subprotocols are forwarded;
  - the signed URL must be exactly the base URL plus the request path;
  - an `http://` upstream on a non-loopback host is a ConfigError;
  - a multipart body the proxy cannot parse is refused with 400 instead of being
    forwarded;
  - WebSocket refusals are recorded like HTTP ones.

### Added

- Realtime: Azure OpenAI Realtime's GA path (`/openai/v1/realtime?model=…`) and the
  Vertex AI Live API (`/ws/google.cloud.aiplatform.{v1,v1beta1}.LlmBidiService/BidiGenerateContent`,
  `[providers.vertex]`) are relayed. With llm-redact-pro, `auth = "identity"` now
  works on these WebSockets instead of being refused: client credentials in the
  upgrade headers, the query and subprotocols are stripped, and the upgrade is
  authorized with the proxy's identity on the exact documented paths only.
- Cloud-provider route coverage. Identity auth forwards only recognized routes, and
  unrecognized routes of key-authorized providers used to be forwarded unredacted:
  - Vertex AI: context caching (`projects/{p}/locations/{l}/cachedContents`; the
    create is redacted on the static session and its cache name tracked;
    get/list/patch/delete recognized), `:computeTokens`, `:embedContent`,
    `:fetchPredictOperation`, publisher-model and Model Registry metadata GETs.
  - Azure OpenAI: legacy completions, image generation/edit prompts and
    text-to-speech input, `/openai/v1/files` upload and content download, the v1
    Conversations store, Responses cancel/delete, batches (user `metadata` redacted
    and restored in the echo), and file/model/deployment listings.
  - Bedrock runtime: `count-tokens`, ApplyGuardrail (rewritten outputs restored),
    StartAsyncInvoke and the async-invoke list/get.
  - `docs/api-coverage.md` gains Vertex AI, Azure OpenAI, Bedrock and realtime
    WebSocket tables, pinned in both directions by `tests/test_api_coverage.py`.
- Object tracking (the session-ownership seam used by llm-redact-pro) also reports
  stored chat completions (`store: true`, including streamed ones) and video
  create/remix jobs; `tracks_object_ids` takes the parsed request body.

### Changed

- Under `auth = "identity"`, any query parameter whose name contains
  `authorization` is stripped as a client credential (HTTP and WebSocket).

### Fixed

- A failed upstream WebSocket dial is counted in `upstream_errors` and recorded; it
  used to leave an `[audit] required` START row with no END row.
- WebSocket close reasons are cut to the protocol's 123-byte limit, so a long refusal
  reason no longer makes the close fail silently.
- Azure JSONL file uploads get the per-line system note on chat-shaped lines (parity
  with OpenAI `/v1/files`); Azure's note is otherwise confined to chat completions.
- Vertex express-mode metadata GETs (`/v1/publishers/…`) reach the vertex upstream
  instead of the anthropic default.
- The realtime relay keeps the `upstream_base_url` path (APIM and gateway bases).
- The Azure batch list is no longer rehydrated (redact-only, like OpenAI's).
- Bedrock `count-tokens` decodes, redacts and re-encodes `input.invokeModel.body`;
  a body that cannot be decoded is refused with 400.
- A `/v1/…` request authenticated with a Google API key (`x-goog-api-key`, `?key=`)
  goes to Gemini instead of OpenAI.
- The `websockets` logger is pinned at WARNING, so it cannot log upgrade URLs.

## [1.8.0] - 2026-09-28

The core side of the proxy authenticating as ITSELF to cloud services, plus two
sign-in additions. Every credential is fetched by llm-redact-pro; the core carries
only config shapes, generic seams, fail-closed defaults and doctor/status surfaces.
A config that asks for one of them without llm-redact-pro is a startup error naming
the package (`[email]`, which does nothing without the package, stays inert).

### Security

- The proxy refuses a request target that is not a path (`400`), over HTTP and realtime alike,
  and checks that the URL it builds keeps the configured upstream's scheme, userinfo, host and
  port. A percent-encoded target such as `%2Fv1%2Fx@host:port/…` used to join onto the base URL
  as `userinfo@host` and send the request (with its credentials) to another host.
- A provider authorized with the proxy's own identity (`[providers.NAME] auth = "identity"`)
  forwards only the API routes llm-redact recognizes. Any other path is a recorded local `403`,
  so a client can't reach the rest of that cloud API as the proxy's principal, unredacted.
  Google's `x-goog-user-project` and legacy IAM selector headers are stripped along with the
  client's credentials.
- Only paths *below* `/__llm-redact/auth/` go to the access gate; the bare prefix stays behind
  dashboard admission.
- `[vault.rdbms] auth = "identity"` requires a verified server certificate off loopback
  (PostgreSQL `sslmode=verify-ca|verify-full`, MySQL `?ssl_ca=`), because the database asks for
  the token in the clear inside TLS. `LLM_REDACT_VAULT_TLS_UNVERIFIED=1` accepts an unverified
  link, surfaced as `vault.tls_unverified` in `/status`, doctor and `llm-redact status`.
  PostgreSQL identity connections always pass an explicit `sslmode` and refuse `service`,
  `hostaddr`, a TCP `host=` override, `PGSERVICE`, `PGSERVICEFILE` and `PGHOSTADDR`.
- `[audit.azure]` with `auth = "sas"` or `"identity"` requires an `https` `endpoint_url` unless the
  host is loopback (both are bearer secrets). `[audit.s3] endpoint_url` and `[audit.azure]
  endpoint_url` refuse userinfo, a query or a fragment, and the S3 one refuses a path (requests
  are signed over `/bucket/key`).
- Version skew fails closed: when a plugin is loaded, a configured sink auth mode, email OAuth or
  `implicit_tls` that it does not advertise in `Registry.config_capabilities` stops startup (and
  is a doctor FAIL), instead of an older llm-redact-pro silently using static credentials.
- An identity provider for which the plugin builds no authorizer is a startup `ConfigError`,
  never a pass-through of the client's credential.
- `serve` (and `serve --check`) warns when identity-authorized providers are served on a
  non-loopback bind with no access gate.
- `[providers.bedrock] region` keeps its 32-character cap and `[vault.rdbms] region` is matched in
  full (a trailing newline used to pass).

### Added

- **Proxy-held provider credentials:** `[providers.bedrock|vertex|azure] auth =
  "identity"` (plus a bedrock-only `region`). The proxy authorizes each request with
  its own cloud identity AFTER redaction: it strips every client credential channel
  (authorization/api-key headers, `x-amz-*`, cookies, and the `key=`,
  `access_token=` and `X-Amz-*` query parameters), hands the plugin the final URL and
  bytes, and sends exactly the bytes it authorized. A credential failure is a
  recorded, provider-shaped 502. Identity providers are never routed, and Realtime
  WebSockets to them are refused (1011). New seam: `plugin_api.UpstreamAuth` /
  `UpstreamAuthError`, `Registry.build_upstream_auth`. `/status providers_auth`, a
  `status` posture line, and `doctor` rows that WARN on a non-loopback bind without
  an access gate that requires an identity.
- **IAM database login:** `[vault.rdbms] auth = "identity"` (plus an aws-only
  `region`), for PostgreSQL and MySQL. It requires `cloud`, forbids `password_env`
  and a DSN password, and enforces TLS (PostgreSQL `sslmode` at least `require`,
  set when absent; MySQL over an SSL context with optional `?ssl_ca=` verification,
  and the cleartext auth plugin only over TLS). New seam:
  `Registry.build_db_password` / `plugin_api.DbPasswordProvider`; the RDBMS store
  asks for the password at every connect and reconnect, because tokens expire.
  `/status vault.auth`; a doctor row.
- **KMS-wrapped vault key:** `[vault.kms]` (provider `aws`, `gcp`, `azure` or
  `hashicorp`, `key_id`, and `wrapped_key_env` or `wrapped_key_file`; HashiCorp also
  takes `address`, `mount`, `auth`, `role`, `auth_mount` and
  `service_account_token_file`). It requires `encryption = "fernet"`, and the key
  then comes ONLY from the KMS: `LLM_REDACT_VAULT_KEY` or `_CMD` set alongside it is a
  startup error. New seam: `Registry.resolve_vault_key` and
  `vault_crypto.resolve_cipher`, the one cipher path shared by the proxy and the
  vault CLI (which now resolves the key from the loaded config, `--config` included).
  `/status vault.key_source` (`kms:<provider>`, `local` or null); doctor posture rows
  that never call the KMS.
- **Audit sink credentials:** `[audit.s3] auth = "keys" | "identity"` and
  `[audit.azure] auth = "key" | "sas" | "identity"` (identity is refused for
  MinIO/Ceph). Doctor checks the credentials each mode needs; `/status` reports the
  mode.
- **Email OAuth:** `[email] auth = "oauth"` with `oauth_provider = "azure" | "google"
  | "refresh_token"`, `oauth_subject`, and the refresh-grant keys (`oauth_token_url`,
  https only; `oauth_client_id`; `oauth_client_secret_env`;
  `oauth_refresh_token_env`; `oauth_scope`). Secrets are named by environment
  variable, never stored in the file, and OAuth refuses cleartext at parse time. New
  `[email] implicit_tls` (SMTPS, port 465). A value-free doctor `email` row.
- **Sign-in paths:** everything under `/__llm-redact/auth/` is dispatched to the
  access gate (`AUTH_PREFIX`), not only the fixed login/callback/logout paths, so a
  sign-in method such as llm-redact-pro's passkeys can serve its own pages and JSON
  endpoints. Host and Origin checks apply (POST included); these paths never need
  dashboard admission and are never forwarded.
- The Gemini adapter reports a created context cache's name (`cachedContents/…`) through the
  optional `SessionRouter.record_object_id` seam, as the OpenAI and Anthropic adapters already do
  for files, batches and conversations. llm-redact-pro uses it to keep a cache with the user who
  created it. Only reported outside static mode; nothing changes without a router that takes it.

### Changed

- The `vault-mysql` extra requires PyMySQL 1.2 or newer. Older releases carry on
  WITHOUT TLS when the server doesn't offer it, even with `ssl=` set, which IAM
  database authentication must never allow.

## [1.7.0] - 2026-09-28

### Security

- Session routers can veto the durable response-session map: `SessionRouter.record_response_id`
  may return `False` (a refused cross-namespace re-home, or a router that never reads the map), and
  the proxy then skips its durable write. Previously the proxy mirrored every mapping, so a
  per-user router that refused to move another user's response id into the reader's namespace was
  overruled by the durable row on its next lookup. Returning `None` keeps the historical behavior.
- The response-id map's size cap (10,000 rows) no longer trims rows of sessions that still hold
  mappings, on sqlite and RDBMS vaults alike. A per-user router reads a missing row as "that
  session was pruned" and resumes the chain in a fresh session, so busy traffic from other users
  could make an idle user's next chained turn reissue `«EMAIL_001»` for a new value while the
  provider's history still meant the old one. Only rows of sessions without mappings (pruned, or
  never redacting anything) are trimmed now; a live session's rows leave with the session.
- The in-memory vault manager no longer hands the session router a durable lookup: its "unknown"
  answer was indistinguishable from "that session was pruned", which orphaned every chained
  Responses turn under llm-redact-pro 0.11.0 and could reissue a placeholder number for a
  different value.

### Added

- Two optional seams for an access-control plugin, read via `getattr` so older plugins keep
  working:
  - `AccessGate.bind_sessions(store)` hands the gate a `plugin_api.SessionStore` over the running
    proxy's vault (`session_ids()`, `forget(ids)`), so a deleted user's vault sessions can be
    dropped through the live vault manager (whole sessions only; the configured static session is
    never forgotten). Every vault manager gains `forget_sessions`.
  - `SessionRouter.record_object_id(object_id, session_id)` receives the ids of objects a provider
    stores for later reads — uploaded files, OpenAI batches (and their output/error files), stored
    conversations, Anthropic message batches — with the session that created them, mirrored in the
    durable map unless the router returns `False`. Only called outside static mode. Adapters
    declare what they track via `tracks_object_ids` / `object_ids_from_body`.
- `llm-redact doctor` shows access-control rows from llm-redact-pro when it is installed (silent
  otherwise).

### Fixed

- The live prune fails safe on a misbehaving router: an `is_durable` that raises keeps the session
  (logged by exception type) instead of answering `POST /__llm-redact/sessions/prune` with a 500,
  and only an explicit `False` releases a session (a `None` from a buggy router used to prune it).
  The compaction-fork check follows the same contract: a raising `is_durable` no longer fails the
  request, and only an explicit `False` lets a session count as a fork.

## [1.6.0] - 2026-09-27

### Added

- Session routers may mark sessions as durable (`SessionRouter.is_durable(session_id)`, an
  optional member read via `getattr`). The live prune — the `session_ttl_days` loop and
  `POST /__llm-redact/sessions/prune` — keeps those sessions like the configured static session,
  because provider-side state (Responses chains, batches) still points into them and a recreated
  session would issue the same placeholder numbers for new values. llm-redact-pro uses it for each
  named user's copy of the static session. Routers without the member are unaffected.

### Fixed

- `compaction_forks` no longer counts a session this process has not seen yet whose vault already
  holds entries — a persisted conversation resumed after a restart owns the placeholders in its
  history — nor a session the router marks durable (llm-redact-pro's per-user copy of the static
  session has no first-message anchor for compaction to fork). The metric, the `/status` field and
  the dashboard pill now reflect only real history-compaction forks.

### Documentation

- The session docs (how-it-works, providers, api-coverage) now say that with llm-redact-pro's named
  users, the flows described as using "the static vault session" — realtime connections, batches,
  the Conversations API, Gemini context caching — use the requesting user's own copy of it.

## [1.5.0] - 2026-09-27

### Security

- `serve` now passes `proxy_headers=False` to uvicorn, so the ASGI client address is always the
  socket peer. uvicorn used to rewrite it from `X-Forwarded-For`, for any peer when
  `FORWARDED_ALLOW_IPS=*`. A client could then claim loopback or a trusted load balancer's address
  to an access gate.
- A WebSocket upgrade under a `/u/<key>/` path that no gate removed is refused without logging
  its path, the same as HTTP. The access-gate refusal line used to log it, key included.

### Added

- `plugin_api.ConfigSection` and `Registry.config_sections`: a plugin can own a top-level config
  table, such as llm-redact-pro's `[auth]`. The core parses it into `Config.extensions`, writes it
  back in `config show` and the config editor, and treats it as restart-only. Without the plugin the
  section is still an unknown key, and startup fails naming llm-redact-pro.
- Under mutual TLS, `serve` passes the verified client certificate to the app as the standard ASGI
  TLS extension (`scope["extensions"]["tls"]`), so an access gate can map certificates to users.
  Nothing in the core reads it.
- Optional access-gate members for browser sign-in and remote administration:
  - `guards_dashboard` makes the core ask the gate to admit every reserved endpoint except the
    monitoring probes. A refused browser GET is redirected only to a same-proxy
    `/__llm-redact/…` page (`Admission.redirect`); anything else is a 403.
  - `public_origin()` lets such a gate name the one `https://host` the proxy is reached at, which
    the Host and Origin checks then accept.
- New gate-only paths, answered by llm-redact-pro and a local 404 without it:
  `/__llm-redact/auth/login|callback|logout` (browser sign-in) and everything under
  `/__llm-redact/scim/v2/` (SCIM 2.0 provisioning, Host-checked but not Origin-checked).

### Changed

- `AccessGate.admit` may return an awaitable, so a gate can check a credential against a directory
  or an identity provider without blocking the event loop. Gates that answer directly keep
  working unchanged.

## [1.4.0] - 2026-09-26

### Removed

- All named-user and client-authentication code, which moves to llm-redact-pro:
  - the `/u/<key>/` and `x-llm-redact-user` key handling
  - the rule that requires a key once two users are verified
  - the `/__llm-redact/users` admin endpoints
  - the `llm-redact users` command and the verification-email sender
  - the `llm_redact.users` module
  - the `users` agent slash command

  With llm-redact-pro 0.8 installed, all of these work as before.

### Added

- A generic admission hook (`plugin_api.AccessGate`, `Registry.build_access_gate`). Before
  routing, it lets llm-redact-pro admit or refuse each HTTP request and WebSocket upgrade, and
  record which user each request belongs to.
- `plugin_api.CliCommand` and `Registry.cli_commands`, so a plugin can supply its own
  `llm-redact` subcommands, including their shell completions.
- `Registry.tool_base_url`, which lets a plugin change the base URL that `llm-redact run`
  passes to wrapped tools.

### Changed

- Every `x-llm-redact-*` request header is now dropped before forwarding, on HTTP and
  WebSocket. Before, only `x-llm-redact-user` was.
- Without llm-redact-pro, a `/u/<key>/…` path is answered locally with a 404. It is never
  forwarded or recorded, so the key cannot leak.
- A reserved `/__llm-redact/…` path reached through a stripped prefix is now answered
  locally. Before, it was forwarded to the provider.
- `/__llm-redact/users*` without llm-redact-pro now returns 404 instead of 403.
- Startup now fails with a clear error when a paid license tier or a `[users]` section
  expects access control that the installed llm-redact-pro does not provide. This happens
  with llm-redact-pro older than 0.8, or without the package.
- Startup also fails when a license key is configured and llm-redact-pro is installed but its
  plugin did not load, as happens with a pre-0.8 llm-redact-pro on this core. Before, the key
  fell back to Free and the proxy served with no access control.
- WebSocket connections are now recorded with the admitted user.
- The editions matrix (`docs/editions.md`) and the README list the llm-redact-pro Team
  deployment kit: a shared mutual-TLS team server on Docker, Podman and Kubernetes, for Team
  and above. This repository's Helm chart and container images stay keyless.
- `scripts/render_diagrams.sh` pins mermaid-cli 11, which reproduces the committed diagrams
  (12.0 changed the layout engine).

## [1.3.0] - 2026-09-25

### Removed

- The retired per-cloud license entitlement is no longer surfaced: `/status`'s `license`
  block, `llm-redact status`, and `llm-redact license show` (text and `--json`) drop the
  `clouds` field, and `ResolvedLicense.clouds` is gone. Nothing gated on it — the core
  enforces no tier. `License.clouds` (now defaulted and ignored) and the `CLOUDS` constant
  remain only so llm-redact-pro 0.4 and earlier keep working against this core.
- `llm_redact.cloud_detect` (best-effort cloud-platform detection via instance metadata
  probes, with the `LLM_REDACT_CLOUD` / `LLM_REDACT_SKIP_CLOUD_DETECT` env vars) is removed.
  It existed for the retired per-cloud license placement check and had no consumer; the
  proxy never called it.

### Changed

- The user guide's dashboard section describes the redesigned llm-redact-pro dashboard
  (sidebar views, picklists and type-ahead in the config editor, the unsaved-changes save
  bar).
- The architecture diagram draws the browser dashboard inside the llm-redact-pro box, and
  the documentation describes the current Free/Pro layout rather than how it came about.

## [1.2.0] - 2026-09-24

### Removed

- **The browser dashboard moved to llm-redact-pro.** The web page at
  `/__llm-redact/` (status pills, totals, upstreams, recent requests,
  sessions, users and routing cards), its config editor
  (`GET/POST /__llm-redact/config`) and its redaction-preview card
  (`POST /__llm-redact/preview`) are now part of the separately-installed
  llm-redact-pro package (Pro tier; 0.4 or newer). Without it those paths
  answer a local 404 that names the package (or the missing key, or a
  plugin that did not register) — still answered before routing, never
  forwarded, still hardened. Everything else stays free: the JSON
  `/status`, Prometheus `/metrics`, `/healthz`, `/readyz`, `/recent`,
  `/events`, `/sessions` (+ prune), `/audit`, `/users` and `/guide`
  endpoints, and every CLI — `llm-redact status`, `llm-redact preview`
  (the local dry run), `llm-redact config show` and the
  `/llm-redact:config-edit` plugin command. The dashboard screenshots and
  `scripts/capture_screenshots.py` moved with it.

### Added

- The dashboard seam: `plugin_api.Dashboard` / `plugin_api.DashboardHost`
  and `Registry.build_dashboard(tier)` (Free default: None). The core
  dispatches only its fixed dashboard paths (`proxy.DASHBOARD_PATHS`) to a
  registered dashboard, so a plugin can never shadow a core endpoint, and
  rebuilds it when a reload changes the resolved license tier.
  `ProxyState` satisfies `DashboardHost`: `config_file_path()`, the guard
  methods, `validate_config(candidate)` (the editor's dry run, extracted
  unchanged) and `preview(text)` (the live-pipeline dry run, extracted
  unchanged).
- `config.RESTART_ONLY_KEYS`: the one list `apply_config` pins and the
  editor shows read-only (it was duplicated in both).

### Fixed

- **A failed hot reload no longer half-applies the provider list.** When a
  reload changed the `[providers.*]` set and then failed in the routing
  factory (for example `[routing] enabled = true` without llm-redact-pro),
  the new adapter list was already installed while the old config stayed:
  a still-configured `[providers.custom.NAME]` upstream lost its adapter and
  its traffic was forwarded unredacted until the next good reload. The
  adapter list is now swapped together with the rest of the config.
- Keyless, `doctor`, `llm-redact status` and the dashboard no longer say
  "nothing gated" when llm-redact-pro is installed: that package runs the
  Free tier without a key and refuses its paid subsystems, so they now say
  its features need a key.
- The routing requires-package messages from `llm-redact routes|spend` and
  `doctor` now say `0.3 or newer`, so an installed llm-redact-pro that
  predates the routing layer is not mistaken for a missing one.

## [1.1.0] - 2026-09-22

### Added

- Routing seam for the llm-redact-pro routing layer: the
  `[upstreams]`/`[routing]`/`[prices]` config shapes, parser invariants and
  emitter (file-only, hot on SIGHUP, preserved by the config editor); the
  `plugin_api` `Router`/`RoutePlan`/`RouteDelivery` contract with
  `Registry.build_router` (the Free default fails closed naming the
  package); the policy-free routed-delivery driver in `proxy.py` (a request
  is routed only when a registered router plans it — the unrouted path is
  byte-identical); `llm_redact_routed_requests_total` /
  `llm_redact_reissues_total`; the `/status` `routing` block; the
  `routes list|test` / `spend` / `doctor --offline` parsers (dispatching to
  the pro package); the `/llm-redact:routes` and `/llm-redact:spend` plugin
  commands; the dashboard routing pill/table; the Anthropic 402/404 error
  types. Every surface reports honestly without the package;
  `[routing] enabled = true` without llm-redact-pro is a startup ConfigError
  naming it.

### Changed

- recent/events rows carry a `route` key (`null` on the unrouted path); the
  legacy delivery tail is the extracted
  `_deliver`/`_fault_response`/`_begin_audit_guarded` (behaviour-identical,
  pinned by the existing suites).

### Removed

- The routing implementation that landed in #31/#32 was reverted (#33) and
  ships in llm-redact-pro 0.3 instead (the owner's directive: routing is a
  Pro feature).

## [1.0.3] - 2026-07-29

### Added

- Plugin-first onboarding: every plugin command's proxy-presence guard
  now offers three PINNED setup tiers when the `llm-redact` CLI is
  missing — an ephemeral `uvx --from llm-redact-proxy llm-redact serve`
  run (nothing lands on PATH), a one-approval install
  (`uv tool install` / `pipx install` + `init --yes` +
  `service install`), or pointing `LLM_REDACT_PROXY_URL` at an existing
  proxy. The package name `llm-redact-proxy` is pinned verbatim in both
  directions by test (agent-improvised install commands are a
  supply-chain vector).
- Per-tool routing honesty in the guards: each tool's rendering states
  ITS routing truth — Claude Code checks `ANTHROPIC_BASE_URL` and names
  the `llm-redact run -- claude` relaunch, Codex/OpenCode check
  `OPENAI_BASE_URL` with their `run --` launch forms, and Cursor states
  plainly that its traffic is NOT protected outside custom-API-key mode
  with the base-URL override. A live proxy is never presented as a
  routed session.
- Claude Code marketplace plugin: `bin/llm-redact-posture` (read-only
  POSIX sh posture check, on the Bash tool's PATH) plus a `SessionStart`
  hook running it with `--quiet-ok` — reports CLI-missing / proxy-down /
  session-unrouted loudly, stays silent when healthy, never installs
  anything, and echoes URLs as scheme://host:port only. Behavioral
  tests drive all four states against a live loopback server
  (`tests/test_plugin_posture.py`).
- `opencode` is now a `TOOL_EXPORTS` entry: `llm-redact init --tools
  opencode` and `llm-redact run -- opencode` route OpenCode via
  `OPENAI_BASE_URL`.
- docs/plugins.md: "Plugin-first onboarding" section (the three tiers,
  the per-tool routing-honesty table, the SessionStart posture check,
  and the OpenCode JS plugin as a documented future option).

## [1.0.2] - 2026-07-19

### Added

- Privacy policy (`docs/privacy.md`): no telemetry or phone-home; what
  stays on the machine and what leaves it.

### Fixed

- Plugin command frontmatter: descriptions containing ": " rendered as
  invalid YAML, so Claude Code silently dropped ALL frontmatter fields
  (description, allowed-tools, disable-model-invocation) for the
  `recent`, `status`, and `config-edit` commands, and
  `claude plugin validate` rejected the plugin. The renderer now quotes
  YAML-unsafe values; the plugin passes official validation.

## [1.0.1] - 2026-07-17

### Added

- Interactive install script `scripts/install.sh`: detects the tools
  available on the machine (uv, pipx, pip, Homebrew, docker, podman),
  prompts for the preferred install method (`--method NAME` for
  non-interactive use), and prints every command before running it. The
  container methods pull `ghcr.io/asanderson/llm-redact:latest` and
  offer a loopback-published run.
- README Install section now shows the prebuilt-container path
  explicitly — `docker pull` / `podman pull` from
  `ghcr.io/asanderson/llm-redact` with the loopback publish spec — and
  links the install script.

## [1.0.0] - 2026-07-17

Initial public release. llm-redact was developed privately before this
debut; the public history starts here, at v1.0.0.

### Added

- **The transparent redaction proxy.** A local proxy sits between an
  agentic tool and its LLM provider: outbound requests are scanned for
  private values, each detected value is replaced with a deterministic
  `«TYPE_NNN»` placeholder whose mapping lives only in a local vault, and
  inbound responses — including streamed ones, even when a token splits
  across chunk boundaries — have the placeholders restored. Streaming is
  handled at the byte level across SSE, NDJSON, AWS eventstream, and
  WebSocket framings; unrecognized traffic passes through verbatim
  (never break the tool), and upstream faults fail closed with
  provider-shaped errors.
- **Detection**: 80+ built-in rules — vendor API keys and tokens
  (prefix-anchored), credit cards/IBANs/phones, checksum-validated
  national identifiers across 20+ countries, PGP/private-key armor, and
  checksum-vetoed crypto wallet addresses — plus user deny strings
  (always-win, tier 0), per-type allowlists, per-rule redact/warn/block
  modes, language scoping, custom rules with named validators, and
  optional person-name NER behind five interchangeable backends (spaCy,
  GLiNER, Presidio, Stanza, Hugging Face). A recall==1.0 benchmark gate
  and a real-world false-positive corpus pin detection quality in CI.
- **Providers**: Anthropic (Messages + Batches), OpenAI (Chat,
  Responses, Conversations, Files/Batches, Realtime WS, images/audio/
  video), Google Gemini (+ Live WS, context caching, batch), AWS
  Bedrock, Azure OpenAI, GCP Vertex (Gemini and Claude), Cohere, Ollama
  native, and any number of named custom OpenAI-compatible upstreams
  (vLLM, LM Studio, OpenRouter, …). Embeddings and file uploads are
  redact-only; the endpoint matrix is pinned by test in both directions.
- **Vault**: deterministic placeholder issuance — the same value always
  gets the same token within a session — with in-memory and persistent
  SQLite backends, session lifecycle CLI, and strict never-restore-
  across-sessions isolation.
- **Ops surface**: a self-contained local dashboard with a guarded
  config editor (validate → atomic write → SIGHUP hot reload), redaction
  preview dry-run, Prometheus metrics, health endpoints, a live SSE
  event feed, structured JSON logs, `doctor` diagnostics, and an honest
  posture block that surfaces every protection opt-out — warn-mode
  rules, disabled providers, MCP exemptions, language scoping — loudly.
- **Agent plugins** for Claude Code, Codex, OpenCode, and Cursor: ten
  slash commands mirroring the dashboard and config editor, with a
  proxy-presence guard and real-output screenshots in the docs.
- **Deployment**: hardened Dockerfile (multi-arch), k8s sidecar
  manifest, a Helm chart with sidecar and standalone modes plus optional
  HPA autoscaling, systemd/launchd service units, shell completions, an
  init wizard, and an env-injecting `run` wrapper. Non-loopback binds
  are fail-closed behind full mutual TLS.
- **Assurance**: split-at-every-offset streaming equivalence sweeps,
  property-based tests, differential codec fuzzing, a red-team boundary
  suite, mutation-testing gates, a complexity-coverage gate (every
  branching function executed by the suite), reproducible builds, SBOMs,
  and signed release artifacts.

### Licensing

- The core is **free and open-source software under GNU AGPL-3.0-only**,
  with nothing gated: no license keys, tiers, or seat caps anywhere in
  this repository. Contributions are accepted under `docs/CLA.md`.
- The separately-installed proprietary `llm-redact-pro` package
  (**coming soon**) will supply additional operational subsystems —
  server RDBMS vaults, vault encryption at rest, the audit log and its
  object-store sinks, OpenTelemetry export, per-conversation sessions,
  and named users. Configuring one of those without the package fails
  closed naming the feature and the package — never a silent downgrade.
