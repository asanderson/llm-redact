# Optional Open WebUI text chat

Run [Open WebUI](https://github.com/open-webui/open-webui) as an optional
local chat frontend. Its backend calls llm-redact at
`http://proxy:8787/v1`; the provider receives recognized values as
placeholders, and WebUI receives restored answers. The normal CLI,
agent plugins, Python dependencies, and default demo stay unchanged.

This recipe is for **one trusted local user and text chat**. It uses the
official, unmodified Open WebUI **v0.12.3** image and llm-redact **1.12.1**.
Keep Open WebUI's branding and notices; its
[license](https://github.com/open-webui/open-webui/blob/v0.12.3/LICENSE)
has branding conditions. Upgrading either image needs another smoke run.

## Prerequisites and first setup

Use Docker with Compose v2 and Linux containers (Linux or Docker Desktop
on macOS; on Windows, run these commands in WSL with Docker Desktop
integration). Native Windows Python serving is not supported. This
particular Compose recipe is tested with Docker; the proxy's separate
Podman support does not imply this recipe was tested with Podman Compose.
No GPU or local Python install is needed for the frontend.

From a checkout of this repository:

```bash
cp examples/openwebui.env.example .env.openwebui-demo
chmod 600 .env.openwebui-demo
openssl rand -hex 32     # copy the output into WEBUI_SECRET_KEY below
```

Edit `.env.openwebui-demo` locally:

| Variable | Demo value |
| --- | --- |
| `OPENAI_API_KEY` | `unused-demo-key` — no real provider key |
| `WEBUI_SECRET_KEY` | The random value generated above; keep it stable |
| `WEBUI_ADMIN_EMAIL` | Your local administrator login email |
| `WEBUI_ADMIN_PASSWORD` | A strong password you choose |

The env file is ignored by git. Do not commit secrets or publish its
contents. The first administrator is created on the first boot of an
empty WebUI data volume. Signup stays disabled. Changing the bootstrap
password in this file **does not reset an existing account**; change it
through WebUI's account settings. Keep the env file private after setup.

## Try the synthetic demo first

```bash
docker compose --project-name llm-redact-webui-demo \
  --env-file .env.openwebui-demo \
  -f compose.openwebui.yaml -f examples/openwebui-demo.compose.yaml \
  --profile webui up --build -d
```

Open <http://localhost:3000>, sign in using the administrator credentials,
and select **fake-openwebui**. Send only synthetic data, for example:

> Email jane.doe@corp.example; rotate AKIAIOSFODNN7EXAMPLE

The reply should repeat those originals. Inspect the fake provider:

```bash
docker compose --project-name llm-redact-webui-demo \
  --env-file .env.openwebui-demo \
  -f compose.openwebui.yaml -f examples/openwebui-demo.compose.yaml \
  --profile webui logs fake-upstream
```

Its request should contain `«EMAIL_001»` and `«AWS_KEY_001»`, with no
original canary values. The injected system note also contains example
tokens; those are not vault entries. The demo **logs request bodies**;
never use it for private information or real credentials.

Use the same Compose arguments with `down` instead of `up --build -d` to
stop the demo. `down` retains its data. Do not add `--volumes` unless you
intend to erase that project's WebUI history/accounts and vault.

## Connect a real provider

Stop the demo first if it uses the same host ports. Create a separate
private env file and a separate project, so demo data and provider data
are not mixed:

```bash
cp examples/openwebui.env.example .env.openwebui
chmod 600 .env.openwebui
# Edit all four required values as above; OPENAI_API_KEY is now your provider key.
docker compose --project-name llm-redact-webui \
  --env-file .env.openwebui -f compose.openwebui.yaml --profile webui up -d
```

The default [provider config](../examples/openwebui-config.toml) targets
OpenAI. WebUI holds the provider key in its backend and llm-redact forwards
it; this recipe does not broker a proxy-owned credential. The key is not
part of the browser's direct-connection configuration.

The WebUI connection is fixed to `http://proxy:8787/v1` and model discovery
uses `/v1/models`. Select a model available to your account. If the
provider has no discovery endpoint, add its model IDs under Admin
Settings → Connections → the existing **proxy** connection. Saved admin
configuration does not persist in this recipe; for repeatable manual
IDs add `OPENAI_API_CONFIGS` to the WebUI service environment, for example:

```yaml
OPENAI_API_CONFIGS: '{"0":{"enable":true,"model_ids":["YOUR_PROVIDER_MODEL_ID"]}}'
```

Use a real provider-supported ID. The allowlist skips discovery; it does
not validate that the provider serves the model. Keep Chat Completions
mode; other API modes and provider-specific parameter conversions need
their own verification. A proxy supporting a native provider API does
not automatically make WebUI's hostname-based provider detection work.

### Local OpenAI-compatible models

Change `upstream_base_url` in `examples/openwebui-config.toml` to the
model server's root URL, then recreate the proxy. Its incoming route
already includes `/v1`, so do not repeat it in the standard upstream URL.
For a no-key model server, put an unused dummy value in `OPENAI_API_KEY`.

Inside the proxy container, `127.0.0.1` names that container. For a model
on the host, Docker Desktop provides `host.docker.internal`; on Linux add
`extra_hosts: ["host.docker.internal:host-gateway"]` to the **proxy** service
and ensure the server listens on a host address reachable from Docker.
For a model in another container, use a network shared with the proxy
and that service's DNS name. Never weaken a model server's bind policy
without checking who can then reach it. Ollama's native API is a separate
recipe; this one uses its OpenAI-compatible surface if selected.

## Trust boundary and defaults

**WebUI handles raw prompts and restored answers.** Its chat history,
accounts, exports, uploads, caches, logs, and backups may contain private
data. Redaction of provider requests does not sanitize those stores. Use
encrypted host storage/backups and suitable retention. The core SQLite
vault is also unencrypted; see [editions](editions.md) for the optional
Pro encryption and identity features.

| Control | Behavior in this recipe |
| --- | --- |
| Optionality | Only the explicit Compose file/profile starts WebUI. |
| Network | WebUI is on an internal network; only the proxy also has provider egress. |
| Host exposure | UI `127.0.0.1:3000`; proxy `127.0.0.1:8787`. Change the numeric ports in the private env file if occupied. |
| Configuration | `ENABLE_PERSISTENT_CONFIG=false` prevents old database settings from replacing supplied defaults after restart. It is not a runtime admin lock. |
| Other model paths | Direct connections and the separate Ollama connector are disabled. |
| Extra integrations | Plugins/tools, web search, image/code features, community sharing, and generated-title/tag/follow-up tasks are disabled. |
| Identity forwarding | Off; review custom headers separately if you add any. |
| Downloads/exports | Offline mode enabled; telemetry exports disabled. No automatic model downloads are needed for text chat. |

`OFFLINE_MODE=true` is **not** a firewall. The internal Docker network is
the backend egress restriction. It does not firewall the user's browser,
authenticate arbitrary local software, or make the proxy a multiuser
service. Other trusted containers on that network can reach the proxy.
The image's wider bind is confined to its container namespace; do not
publish its port on all host interfaces. Native/remote proxy serving
still follows the [mTLS bind policy](deployment.md#choose-a-bind-and-know-what-it-costs).

File/RAG ingestion, remote extractors/embeddings/rerankers, audio/realtime,
tools, MCP servers, and terminals are **outside this recipe's supported
scope**. They can use other paths before a final chat request is scanned.
Regular-user upload/knowledge permissions are disabled, but permission
flags do not necessarily constrain an administrator. Do not interpret
the UI controls as proof every possible ingestion path is blocked.
Binary uploads that reach this proxy are refused by the example policy.

One static vault namespace serves this local instance. Different WebUI
accounts or chats are not isolated proxy namespaces. Shared/multiuser
deployment requires a verified identity mapping and isolated vaults (or
separate instances); keep authentication/user logic in llm-redact-pro,
as required by the repository's open-core boundary.

Normal WebUI backend requests rebuild their HTTP headers; do **not** add
an `allowed_origins` exception to the proxy for this recipe. Browser
direct connections are a different flow and remain disabled.

## Operations and troubleshooting

- **No models:** check the selected provider URL/key, discovery response,
  proxy health, and container logs. Add explicit model IDs if necessary.
- **401 on initial login:** bootstrap credentials apply only to an empty
  volume; verify the project name and existing account. Do not delete a
  populated volume as a password-reset shortcut.
- **404 on chat:** WebUI's base URL must end in `/v1`; the standard proxy
  upstream must not add a second `/v1`.
- **A local model is unreachable:** check container DNS, its listener,
  shared networks, and host gateway configuration.
- **Changed config seems ignored:** restart/recreate the appropriate
  container. UI admin changes are intentionally ephemeral here.
- **Missing local processing models/offline errors:** ingestion/audio are
  outside this text-chat scope; do not restore internet access just to
  make an unreviewed feature download dependencies.

Keep the `openwebui-data` and `llm-redact-data` volumes together in a
backup policy. Stop the stack for a consistent filesystem-level backup.
Before upgrading, back up both and record image digests. An image-only
rollback may fail after a WebUI database migration; restore the matching
backup too. Removing the optional stack does not affect the core wheel.

## Automated verification

The normal Python suite uses the real proxy and fake OpenAI routes to
check model inventory, JSON/SSE, mangled-token restoration, and SQLite
persistence. The container CI step additionally boots the **real WebUI**,
signs in, discovers the fake model through the proxy, submits JSON/SSE
chats, exercises a generated-title task, restarts services, checks backend
egress, and stops the proxy to prove failure without direct fallback.

Run the same container smoke test locally from the checkout:

```bash
docker build --target runtime -t llm-redact:ci .
docker build --target fake-upstream -t llm-redact-fake:ci .
python3 scripts/openwebui_smoke.py
```

The script uses only synthetic canaries and a fresh random project with
ephemeral ports and credentials. It removes **only its own** test
containers/volumes afterward; it does not load your private env file.
It exercises WebUI's HTTP backend, not browser rendering. No cloud key
or paid feature is required. The fake service's logging is for this demo
and test only.

Primary references: [provider connection](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/),
[environment settings](https://docs.openwebui.com/reference/env-configuration/),
[offline operation](https://docs.openwebui.com/tutorials/maintenance/offline-mode/),
and [Compose profiles](https://docs.docker.com/compose/how-tos/profiles/).
