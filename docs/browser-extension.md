# Browser extension: design and decision record

> **Status: Phase 0 (planning). Nothing here is implemented.** Every
> configuration key, endpoint, command, policy key and directory named
> below (`allowed_extension_origins`, `[gateway] enabled`,
> `[gateway] rehydration`, `llm-redact extension pair`, `extension/`, …)
> is a proposal until it ships; today's proxy has none of them and
> rejects the keys as unknown. The owner approved the plan on
> 2026-10-02. This page records its twelve decisions (D1–D12) and the
> Phase 0 spike register, and is the decision record for the browser
> extension.

## Architecture in brief

llm-redact protects agentic tools that reach it through base-URL
variables; browsers reach LLMs on paths it never sees. One TypeScript
codebase built with [WXT](https://github.com/wxt-dev/wxt) produces every
browser package, under Apache-2.0 in an `extension/` directory of this
repository (D5). The browser gains exactly one new, opt-in proxy
surface, and the proxy forwards no browser traffic.

| Lane | Traffic | Mechanism | Ships in |
| --- | --- | --- | --- |
| A | Consumer chat sites (claude.ai, chatgpt.com, chat.deepseek.com first) | MAIN-world hooks relay bodies and streams to the gateway's site codecs | Phase 3 |
| B | Browser-side API calls (BYOK apps, Open WebUI Direct Connections) | The same hooks, with the existing provider adapters as codecs (D12) | Phase 2 |
| C | Local LLMs and self-hosted chat UIs | llm-redact placed server-side, in front of the model server or UI backend | Phase 1 (docs) |
| D | Traffic the hooks cannot see | A TLS-intercepting forward proxy | Backlog (D9) |

**Lanes A and B share one pipeline.** A manifest content script with
`"world": "MAIN"` at `document_start` (Chrome 111, Firefox 128,
Safari 18;
[MDN BCD](https://github.com/mdn/browser-compat-data/blob/main/webextensions/manifest/content_scripts.json))
wraps `fetch`, `XMLHttpRequest`, `WebSocket`, `EventSource` and
`navigator.sendBeacon`. Requests that the site's inert routing data
matches are decompressed and relayed through a `MessageChannel` and the
isolated content script to the background, which posts them to the
proxy's opt-in **site gateway**. The codec answers send (with the
rewritten body), block (types, never values) or pass; **the page sends
the redacted request itself** through the saved native `fetch`, keeping
its TLS session, cookies and keys; and a `TransformStream` restores
response chunks according to the read-back mode (D1). The background
relay sidesteps the page's CSP
([Chrome](https://developer.chrome.com/docs/extensions/develop/concepts/content-scripts))
and, from Chrome 144.0.7512.0, Local Network Access prompts
([chromium-extensions](https://groups.google.com/a/chromium.org/g/chromium-extensions/c/pUDh8RiTjJk)).

**Content travels, not connections.** Routing whole connections fails:
the proxy is path-routed with one upstream per provider, has no
`CONNECT` or certificate authority, and refuses an intercepted
`Host: claude.ai` request (`request_origin_refusal` in
[proxy.py](../src/llm_redact/proxy.py)), while redirects lose cookies
and no extension API rewrites bodies
([Chrome webRequest](https://developer.chrome.com/docs/extensions/reference/api/webRequest)).

**Lane C needs no extension.** Pointing a self-hosted UI's backend
(Open WebUI, LibreChat) at llm-redact
([Open WebUI](https://docs.openwebui.com/troubleshooting/connection-error/)),
or serving llm-redact on `11434` with Ollama on a private port, redacts
once at the real egress. The UI's users share one vault namespace
without llm-redact-pro's named users, and enclaves should set
`OLLAMA_NO_CLOUD=1`
([Ollama FAQ](https://docs.ollama.com/faq)). KoboldAI's
`/api/v1/generate` is attributed to Ollama and forwarded unredacted
today (`providers/attribution.py`), a Phase 1 fix; llama.cpp's
`/completion` and TGI's `/generate` get codecs in Phase 2.

## Decision log

All twelve entries are dated 2026-10-02, one subsection each.
**Accepted** marks the owner's choice, **Default** a plan default
adopted with the plan, **Merged** a decision folded into another, and
**Deferred** one parked in the backlog.

### D1. Gateway read-back: three nested modes, off until consent

**Status:** Accepted, 2026-10-02.

**Context.** Restoring values in the page needs plaintext read-back, but
`/__llm-redact/` is metadata-only apart from one documented exception,
the llm-redact-pro config-editor GET
([versioning.md](versioning.md#the-stable-surface)), and sequential
token numbers let whoever reaches read-back enumerate values.

**Decision.** `[gateway] enabled` defaults to false, so existing installs
do not change; onboarding detects the disabled gateway through `/status`
and asks for consent, showing the exact config change, as the agent
plugins do ([plugins.md](plugins.md#plugin-first-onboarding)).
`[gateway] rehydration` then selects the mode, defaulting to "proxy".
The endpoints sit under the reserved prefix, never forwarded, and their
POSTs join the guarded POSTs behind the extension-origin guard (D2): a
**request rewrite** (send, block or pass), a **rehydration stream** (a
WebSocket, with a buffered variant for history) returning restored text
and highlight spans, and **resolve**, returning exact token→value pairs.

| | "off" | "proxy" (default) | "extension" |
| --- | --- | --- | --- |
| Read-back endpoints | None: rehydration and resolve answer 404 | Rehydration stream: text plus spans, so the extension never parses tokens | Resolve and the stream; redact responses also return the request's own pairs |
| Restored values live | Nowhere in the browser; users run `llm-redact lookup` | Only in the rendered page | In the page and the background's token map |
| Vault | Persistent recommended (`doctor` warns) | Persistent SQLite, idle pruning off (`doctor` fails otherwise) | As "proxy" |
| Brief proxy outage | Sends block | Sends block; nothing restores | Sends block; cached tokens restore |
| Metadata-only promise | Holds unchanged | Second `lookup`-equivalent exception | Same exception |
| Posture | "Placeholder mode" indicator | Normal | Widest exposure, flagged in `doctor` and `/status` |

- **Modes nest: off ⊂ proxy ⊂ extension.** Each mode also registers
  every lower mode's endpoints. The proxy's mode, read from `/status`, is
  authoritative; a managed-policy key such as `maxRehydrationMode` may
  only tighten toward "off", never loosen, and a capped extension uses
  the strongest endpoint its cap allows (under a "proxy" cap, the stream,
  never `resolve`). The badge shows the effective mode.
- **Exact replacement in mode "extension".** The extension embeds no
  token grammar (D5): it replaces exact strings only, with pairs from the
  redact response (values the browser itself sent) and from `resolve`. A
  segment containing `«` that does not match exactly goes to `resolve`,
  which applies the canonical and fuzzy grammar
  ([placeholders.py](../src/llm_redact/placeholders.py)) and returns
  exact pairs. The streaming holdback comes from `/status` capabilities
  (`MAX_PLACEHOLDER_LEN` is 40 today).
- **Escaped and framed encodings.** The gateway returns each pair
  pre-encoded for the channel encoding the adapter declares (inside JSON
  a token can arrive as `\u00ab…\u00bb`), and adapters with
  length-prefixed frames (Gemini, Kimi) restore proxy-side even in mode
  "extension", a tightening the nesting permits.
- **The token map** stays in background memory and `storage.session`,
  which is in-memory, cleared on restart and hidden from content scripts
  by default
  ([Chrome storage](https://developer.chrome.com/docs/extensions/reference/api/storage));
  Firefox has no `setAccessLevel`, so its content scripts never get
  access
  ([MDN BCD](https://github.com/mdn/browser-compat-data/blob/f2dd714f4923299ce2c56b7379252f78eed74417/webextensions/api/storage.json)).
  It never touches `storage.local`, content scripts or the MAIN world, is
  evicted per tab and site, and is cleared on pause, unpair and browser
  restart;
  a CI rule keeps the area closed to untrusted contexts, and `/status`
  counts resolves without values, with an optional per-minute cap. Spike
  S15 gates this mode alone.
- **Four rules hold in every mode.** (1) The core gateway serves only
  where the bind policy allows single-user plaintext serving: loopback,
  or the container hatch `LLM_REDACT_INSECURE_BIND=1` with a `127.0.0.1`
  publish (D4); on a non-loopback mutual-TLS bind it refuses unless
  llm-redact-pro per-user isolation is installed. (2) It uses a dedicated
  persistent session (D7). (3) The same release ships versioning.md's
  **second documented `lookup`-equivalent exception**, after the
  config-editor GET and for "proxy" and "extension" only, with a
  [threat-model.md](threat-model.md) section, a
  [security-dataflows.md](security-dataflows.md) gate row and per-mode
  red-team tests. (4) The residual is stated plainly: site JavaScript can
  re-send restored values from the DOM; outbound hooks re-redact known
  endpoints, and D3's scan blocks unknown ones when detectors fire.

**Consequences.** "off" is the highest-assurance mode: values never
re-enter the browser, the metadata-only promise holds, and because
`lookup` reads the vault database, it recommends a persistent vault.
"extension" trades exposure for resilience: no per-chunk round trip, but
values sit in browser memory for the session.

**Open-core placement.** Core and opt-in: the endpoints, the `[gateway]`
keys, all three modes, the `doctor` gates, documentation and tests. A
gateway on a non-loopback bind needs llm-redact-pro per-user isolation
(D2).

### D2. Admission: exact extension origins on loopback

**Status:** Accepted, 2026-10-02.

**Context.** The proxy refuses extensions today: `allowed_origins`
rejects `chrome-extension://…` because `normalize_origin` accepts http
and https only (pinned by
[test_allowed_origins.py](../tests/test_allowed_origins.py)); the threat
model calls an extension "indistinguishable from the attacker"
([threat-model.md](threat-model.md#requests-from-web-pages-the-operators-browser));
and the reserved endpoints' stricter `_origin_allowed` check fails every
`*-extension://` origin.

**Decision.** A new restart-only key, `allowed_extension_origins`, with
its own strict normalizer: the `chrome-extension`, `moz-extension` and
`safari-web-extension` schemes, exact IDs, no wildcards.
`request_origin_refusal` checks it, and the Host rule stays. Listed
origins are honored only on the gateway paths and on
`/__llm-redact/healthz`, `readyz` and `/status` (where the stricter
check is relaxed and `allowed_origins` is never consulted), and only
they get CORS answers, including on proxy-generated errors, which carry
none today
([deployment.md](deployment.md#browser-apps-on-other-origins-allowed_origins)).
The gateway serves on loopback or the container hatch only (D1). A new
`llm-redact extension pair` command records the ID the extension
displays: Chromium IDs are stable, derived from the public key
([Chrome self-hosting](https://developer.chrome.com/docs/extensions/how-to/distribute/host-extensions)),
so the config lists the store and self-hosted IDs, which policy can
pre-seed; Firefox and Safari IDs are probably per-install (S8), making
pairing mandatory there.

**Consequences.** An exact origin proves "this extension", not "this
user", and local software can forge it, which the threat model already
accepts for software running as the operator. Fleets must name both IDs
in policies, `runtime_allowed_hosts` and `allowed_extension_origins`, or
standardize on one. **Correction to v1 of the plan:** non-loopback
serving under full mutual TLS is already a keyless core feature
([editions.md](editions.md);
[deployment.md](deployment.md#choose-a-bind-and-know-what-it-costs)).
Only the remote *gateway* is Pro, because core mTLS "identifies no
user": every certificate holder would share one vault and could
enumerate everyone's browser values.

**Open-core placement.** Core, as a security floor: the key and its
normalizer, scoped CORS, `extension pair`, `doctor` and `/status` lines,
and the non-loopback refusal. llm-redact-pro, behind the `AccessGate`
seam: a pairing-secret header in the `x-llm-redact-*` family (which the
core already strips before forwarding), named users with per-user vault
sessions, and the remote gateway with the Team deployment kit. Any core
seam these need falls under the open-core placement rule in
[CLAUDE.md](../CLAUDE.md): ask the owner first.

### D3. Unredactable content: block drift and outages, scan the unknown

**Status:** Accepted, 2026-10-02.

**Context.** Site protocols drift, unknown endpoints appear, and the
proxy can be down or the extension unpaired; each could leak silently.

**Decision.**

| Case | Behavior |
| --- | --- |
| Known prompt endpoint, unrecognized shape (site changed) | Block the send; the badge shows "adapter outdated"; offer the DOM fallback: composer text redacted through the gateway on submit, replies show placeholders, copy restores originals in "proxy" and "extension" only |
| Unknown endpoint carrying text (JSON, text, form; gzip decoded) | Scan through the gateway; block only when detectors fire (types, never values); otherwise pass |
| Anti-abuse and telemetry endpoints the adapter knows | Pass untouched, as the providers' terms require |
| Proxy down or extension unpaired | Block sends on protected sites; personal installs may "pause on this site"; policy can lock fail-closed |
| Personal install switches a site to "warn" | Send unredacted after a per-send click-through; a policy lock removes the option |

**Consequences.** The scan reuses `Redactor.scan`
([redactor.py](../src/llm_redact/redactor.py)): the live detectors, no
placeholder issued, nothing written to the vault. Adapter health shows
in `/status` ("adapter outdated: sending blocked") instead of a leak,
and the DOM fallback is a safety net, not a primary path. Lane B pages
follow the same rules; the proxy's fail-open rule for unrecognized
forwarded traffic ([security-dataflows.md](security-dataflows.md)) is
untouched, because the gateway never forwards.

**Open-core placement.** Core: the gateway scan and adapter health. The
extension: blocking, the badge, "pause", "warn" and the DOM fallback,
with the fail-closed lock in managed policy. Nothing in llm-redact-pro.

### D4. Windows: the Linux proxy in a container or WSL2

**Status:** Accepted, 2026-10-02.

**Context.** The proxy targets Linux and macOS; Windows is "unsupported
by design" ([coding-standards.md](coding-standards.md)).

**Decision.** No native port. Windows users run the published image
under Podman Desktop or Docker Desktop, or a WSL2 distribution with the
normal install, over HTTP transport only (no native-messaging host, so
no Windows binary).

**Consequences.**

- **Loopback publish only.** The image binds `0.0.0.0` inside its network
  namespace under `LLM_REDACT_INSECURE_BIND=1`
  ([Dockerfile](../Dockerfile)). The gateway cannot see the publish spec
  and a LAN client can forge `Origin` and `Host`, so a bare
  `-p 8787:8787` turns read-back into a LAN service the core cannot
  detect. The Windows guide must insist on `-p 127.0.0.1:8787:8787` and
  a named vault volume, as [compose.yaml](../compose.yaml) does; that
  documentation is the core's only control, and the llm-redact-pro
  pairing secret is the real fix.
- **WSL2.** Default NAT `localhostForwarding` serves a `127.0.0.1` bind
  to Windows browsers at `localhost`; mirrored mode lets the LAN reach
  WSL directly, harmless only for a loopback bind
  ([Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/networking)).
  WSL stops idle distributions after 15 seconds by default
  ([Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)),
  so the guide must keep the proxy alive (S14).
- **Runtime choice.** Docker Desktop needs a paid subscription at
  companies with 250 or more employees or $10 million or more in annual
  revenue
  ([GCN](https://gcn.com/docker-desktop-ships-free-every-developer/21055/));
  Podman Desktop is Apache-2.0, so organizations should default to it or
  to plain WSL2. Off-store force-installs on Windows need Active
  Directory domain join
  ([Chrome](https://support.google.com/chrome/a/answer/7532015?hl=en)).
- **The OCI label fix.** The image's `org.opencontainers.image.licenses`
  label says `MIT` ([Dockerfile](../Dockerfile)), while
  [pyproject.toml](../pyproject.toml) declares `AGPL-3.0-only`, so
  license scanners, which enclaves rely on, misreport the image until
  Phase 1 fixes the label.

**Open-core placement.** Core: the Windows guide and the label fix.
llm-redact-pro: the shared team gateway, with the Team kit.

### D5. Extension license: Apache-2.0 in `extension/`

**Status:** Accepted, 2026-10-02; D11 is merged into it.

**Context.** The core is AGPL-3.0 under a CLA, while browser stores,
contributors and the App Store favor a permissive license.

**Decision.** Apache-2.0, in `extension/` of this public repository.
Tools that talk to the proxy over HTTP "are not affected by its license"
([editions.md](editions.md#the-licenses)), and the [CLA](CLA.md) lets
the maintainer sublicense contributions "under any license terms" while
promising each "remains available under the AGPL-3.0", which should hold
because Apache-2.0 code can be conveyed under AGPLv3 (an inference for
counsel; see below). The rules:

- The extension embeds no core code, no detection rules and no token
  grammar; the gateway returns restored text and spans ("proxy") or
  exact pairs ("extension").
- Every file carries an SPDX header beside a per-directory LICENSE, and
  a CI npm license allowlist excludes GPL-only dependencies.
- One repository keeps the manifest version locked to the proxy's
  `__version__` and the gateway protocol (a fourth dotted integer for
  extension-only respins) and shares CI, SBOM and provenance
  ([RELEASING.md](RELEASING.md)).
- llm-redact-pro features arrive as `/status` capabilities, never as a
  license key: a client-side check in public source can be deleted, and
  Apple's guideline 3.1.1 forbids "their own mechanisms to unlock content
  or functionality, such as license keys"
  ([App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)).
- Distribution, with D11: the Chrome Web Store, Edge Add-ons and listed
  AMO, plus an unlisted signed XPI for enclaves; Safari through Developer
  ID and notarization first, the Mac App Store optional later; iOS after
  the llm-redact-pro remote gateway.

**Consequences.** The FSF lists Apache-2.0 as GPLv3-compatible
([FSF](https://gplv3.fsf.org/wiki/index.php/Compatible_licenses)), one
way only ([ASF](https://www.apache.org/licenses/GPL-compatibility.html)),
and AGPLv3 carries the identical §7 clause ([LICENSE](../LICENSE)), so
the AGPLv3 path is **an inference for counsel to confirm**: the FSF's
list could not be retrieved, and the CLA is itself a "Draft for counsel
review".
CONTRIBUTING should state the directory's license. The permissive
license also avoids the GPL's App Store conflict
([FSF](https://www.fsf.org/blogs/licensing/more-about-the-app-store-gpl-enforcement)),
and a closed fork is low-risk, being useless without the AGPL gateway.

**Open-core placement.** Neither tier: public Apache-2.0 code beside the
AGPL core, which llm-redact-pro reaches only through `/status`
capabilities.

### D6. Launch order

**Status:** Default, 2026-10-02.

**Context.** Chat sites differ widely in feasibility and protocol
stability, and Safari needs its own transport (S7).

**Decision.** Tier 1: claude.ai, chatgpt.com and chat.deepseek.com.
Tier 2: perplexity.ai, huggingface.co/chat, gemini.google.com, grok.com
and copilot.microsoft.com. Tier 3 (Meta AI, Kimi, Poe, Le Chat, whose
protocol is unverified, and AI Studio) gets the DOM fallback or a block.
Chromium and Firefox first, macOS Safari after S7, iOS in the backlog;
floors are
Chrome 144, Firefox 140 (ESR 153 for fleets) and Safari 18.4 (26 for
the DNR backstop).

**Consequences.** Phase 3 starts with claude.ai, ChatGPT and DeepSeek.
Site codecs live in the proxy, because store review can take "up to a
few weeks"
([CWS review](https://developer.chrome.com/docs/webstore/review-process))
and Chrome bans remotely fetched logic
([CWS MV3](https://developer.chrome.com/docs/webstore/program-policies/mv3-requirements));
the extension keeps only inert routing data.

**Open-core placement.** Core: the site codecs, behind one new seam, a
host-keyed `SiteAdapter` registry.

### D7. Browser vault session

**Status:** Default, 2026-10-02; required by D1.

**Context.** Chat sites send only the new turn, so llm-redact-pro's
per-conversation sessions, keyed on the first user message, do not fit.
Placeholders persist upstream in history, titles and shared links, which
the default in-memory vault would leave unrestorable after a restart.

**Decision.** The gateway uses one dedicated, persistent static session,
apart from the agent tools' session.

**Consequences.** Sequential-token enumeration (`«EMAIL_001»`,
`«EMAIL_002»`, …) is confined to browser-originated values, and the
never-wrong-value contract holds
([coding-standards.md](coding-standards.md#correctness-invariants-the-code-must-uphold)).
Modes "proxy" and "extension" require the SQLite vault with idle-session
pruning off, and `doctor` fails otherwise.

**Open-core placement.** Core; with llm-redact-pro's named users, each
user gets a per-user session (D2).

### D8. Uploads

**Status:** Default, 2026-10-02.

**Context.** Server-side extraction bypasses prompt redaction, and the
API path forwards binary uploads unscanned by default
([providers.md](providers.md#request-bodies-llm-redact-cannot-read)).

**Decision.** Protected sites block binary uploads by default, with the
extraction inspector ([extraction.md](extraction.md)) as an opt-in;
voice stays out of scope.

**Consequences.** Every site adapter covers uploads, history reads,
generated titles and memory endpoints, the last so that restored text
never flows back into stored memory.

**Open-core placement.** Core, like the extraction inspector.

### D9. Lane D, TLS interception

**Status:** Deferred, 2026-10-02.

**Context.** Interception's only unique value is coverage of what the
hooks cannot see (worker sends, other extensions, desktop apps), against
fingerprint and cookie hazards and a poor record: 18 of 20 interception
products weakened connections
([NDSS 2017](https://www.ndss-symposium.org/ndss2017/ndss-2017-programme/security-impact-https-interception/)).

**Decision.** The lane waits in the backlog and reopens only if S3 finds
prompt paths the hooks cannot see on a priority site, or if managed
fleets need coverage of other extensions or desktop apps.

**Consequences.** S10–S13 are retired and return with the lane, as do
its design notes: a per-install CA whose name-constrained intermediate
signs only allowlisted LLM hosts, using FIPS-approved algorithms
([fips.md](fips.md)), and `curl_cffi` for browser-like fingerprints.

**Open-core placement.** Decided by the record that reopens the lane.

### D10. Firefox manifest

**Status:** Default, 2026-10-02.

**Context.** WXT builds Firefox packages as Manifest V2 by default
([WXT](https://wxt.dev/guide/essentials/target-different-browsers.html)).

**Decision.** Firefox ships Manifest V3.

**Consequences.** Manifest MAIN-world parity across browsers.

**Open-core placement.** Extension packaging only.

### D11. App Store

**Status:** Merged into D5, 2026-10-02.

**Context.** v1 asked whether the Safari build needed a GPLv3 §7 App
Store exception.

**Decision.** Developer ID first; the Mac App Store is optional later.

**Consequences.** Under Apache-2.0, no GPL App Store question remains.

**Open-core placement.** Extension distribution only.

### D12. Lane B transport: the gateway, not forwarding

**Status:** Accepted, 2026-10-02.

**Context.** BYOK apps such as
[TypingMind](https://docs.typingmind.com/general-faqs) and Open WebUI's
[Direct Connections](https://docs.openwebui.com/features/chat-conversations/direct-connections/)
call providers from the page, and CORS does not stop delivery, so
redaction must happen before the request leaves the browser. In v1, the
background re-sent these calls to the proxy's base-URL routes for
forwarding.

**Decision.** Lane B runs Lane A's pipeline. The hook sends the gateway
the method, path, content type, non-credential marker headers and body;
a provider codec redacts the body; the page sends the result through the
saved native `fetch` with its own key. **Credential headers, cookies and
`key=` query parameters never leave the page.** The codecs are the
existing provider adapters run transform-only (`matches`,
`prepare_request` and `rehydrate_body` in
[providers/base.py](../src/llm_redact/providers/base.py)). The SSE and
NDJSON parsers are already byte-level, but `_stream_rehydrated` and its
NDJSON twin consume an `httpx.Response` with the proxy's request state,
so the gateway spec must confirm that they extract cleanly.

**Consequences.**

- The proxy never sees browser API keys or cookies and forwards no
  browser traffic; the codec follows the request format, so any
  OpenAI-compatible host needs no per-host configuration.
- The page's own `fetch` waits on slow models, so Chrome's 30-second
  limit on a service worker's `fetch()`
  ([Chrome](https://developer.chrome.com/docs/extensions/develop/concepts/service-workers/lifecycle))
  stops mattering; v1's header-flush change and managed DNR redirect are
  dropped, which retires S9.
- Gemini Live's `BidiGenerateContentConstrained`
  ([Gemini Live API](https://ai.google.dev/api/live)), which the matcher
  refuses, and xAI voice, which has no route, become standalone core
  fixes; browser realtime text frames go through hooked WebSocket send
  and message plus gateway frame codecs.
- llm-redact-pro's forwarding features (routing, fallback chains,
  budgets, brokered provider keys; [editions.md](editions.md)) do not
  apply to browser API calls.
- **Hooks run only where content scripts are registered**, so BYOK and
  self-hosted origins need a per-origin opt-in: an optional host
  permission, then `scripting.registerContentScripts` with
  `world: "MAIN"` (Chrome 102, Firefox 128, Safari 16.4;
  [MDN BCD](https://github.com/mdn/browser-compat-data/blob/f2dd714f4923299ce2c56b7379252f78eed74417/webextensions/api/scripting.json)).
  Whether force-installs grant that access without a user gesture is a
  Phase 2 check.
- Managed fleets can fail closed with a `declarativeNetRequest` `block`
  rule on LLM API hosts that exempts protected origins through
  `excludedInitiatorDomains` (Chrome 101, Firefox 113, Safari 26;
  [MDN BCD](https://github.com/mdn/browser-compat-data/blob/f2dd714f4923299ce2c56b7379252f78eed74417/webextensions/api/declarativeNetRequest.json)).
  Whether Safari honors it on sites the user has not granted is
  **unverified**, no rule touches other extensions' requests, and the
  permission's install-time warning
  ([Chrome DNR](https://developer.chrome.com/docs/extensions/reference/api/declarativeNetRequest))
  makes store versus managed-only packaging a Phase 2 call.
- Host permissions shrink to chat-site hosts, the loopback gateway and
  optional per-origin grants, with no provider-API hosts.

**Open-core placement.** Core: the codec interface, realtime text-frame
codecs, the llama.cpp and TGI codecs, and the standalone fixes. Nothing
moves into llm-redact-pro.

## Spike register

The spikes test what the research could not confirm. Their harness
lives outside this repository for now, in the Phase 0 spike kit
delivered to the owner. "Container (Chromium)" work runs headless in a
container; "owner-run" work needs the owner's machines, browsers or
signed-in sessions. Answers are recorded here as spikes close; S15
decides whether mode "extension" ships, and only S3 can reopen a
decision (D9).

| # | Question | Pass condition | Where it runs | Status |
| --- | --- | --- | --- | --- |
| S1 | Which `Origin` and `Sec-Fetch-Site` do extension background fetches to `127.0.0.1` send in Chrome 155, Firefox 153 and Safari 27, and are they exempt from CORS? | Deterministic per browser, recorded against a header-echo server; drives admission (D2) | Container (Chromium 141, done); owner-run (Chrome 155, Edge, Firefox 153, Safari 27) | Partly answered (Chromium 141): a host-permitted worker or extension page is CORS-exempt with no preflight, sends `Origin: chrome-extension://<id>` on POST and WebSocket but no `Origin` on GET, and `Sec-Fetch-Site: none`; content scripts carry the page's origin and get CORS. Target browsers open |
| S2 | Do `document_start` MAIN-world hooks install before the first prompt request on claude.ai, chatgpt.com, grok.com and gemini.google.com? (Safari injects only after the per-site grant.) | Hooks win on every test load in all three engines | Container (fixtures, done); owner-run (real sites, three engines) | Mechanism passes (Chromium 141): hooks ran 0.5–8.2 ms before the first inline script on all 85 fixture loads, `registerContentScripts` too. Gap: an iframe's initial `about:blank` document before its navigation completes is never hooked. Real sites open |
| S3 | Do these sites send prompts or read streams from workers, service workers or iframes? | A per-site inventory; each path gets an adapter or a backstop | Container (fixtures, done); owner-run (real-site inventory) | Mechanism confirmed: dedicated, module and shared workers, page service workers and unmatched frames bypass the hooks; non-blocking `webRequest` sees them all, and dedicated-worker sends look exactly like the page's, so only the hooks-versus-network diff finds them. Real-site inventory open; it alone can reopen D9 |
| S4 | Can the Chrome service worker relay long streams, and survive long pre-first-token pauses, using message heartbeats? | No termination across 20 runs | Container (Chromium 141, done); owner-run (Chrome 155 headful, Firefox, Safari) | Passes on Chromium 141: 0 terminations in 23 heartbeat runs (20 across the three scenarios), no lost or duplicated events; without heartbeats the worker dies about 30 s into any longer silence. Content-script pings fail in a tab hidden over 5 minutes, so the keepalive must come from the worker. Target browsers open |
| S5 | Do background loopback fetches avoid Local Network Access prompts on Chrome 144+ and Firefox 153 without policy? | No prompt; policy needed only for page-initiated loopback | Container (approximation, done); owner-run (Chrome 144+, Edge, Firefox 153) | Approximation only (Chromium 141): page and content-script loopback fetches are already gated, while extension pages and workers are classified loopback and never gated. 141 cannot confirm the 144.0.7512.0 fix; open |
| S6 | Fresh owner-captured HAR files for Tier 1 sites: Claude's gzip and event set, ChatGPT's operation types, Grok's REST versus WebSocket | Synthetic fixtures authored for each codec | Owner-run | Open — Phase 0 |
| S7 | Safari transport: native messaging versus HTTPS loopback; the service-worker CORS bug after 18.4; iOS `webRequest`, where Apple's documentation and MDN BCD disagree | The chosen transport streams within the latency budget | Owner-run (macOS) | Open — Phase 0 |
| S8 | Which `data_collection_permissions` categories AMO wants for local-proxy transmission; whether Firefox and Safari extension origins are per-install | A written AMO answer; the pairing design confirmed | Owner-run | Open — Phase 0 |
| S9 | The managed DNR redirect variant of v1's Lane B | — | — | Retired: D12 dropped the variant |
| S10 | Upstream challenge rates for a re-originating listener | — | — | Retired: Lane D only (D9) |
| S11 | Name-constraint enforcement for a local CA | — | — | Retired: Lane D only (D9) |
| S12 | A macOS transparent proxy | — | — | Retired: Lane D only (D9) |
| S13 | `proxy.onRequest` loopback visibility | — | — | Retired: Lane D only (D9) |
| S14 | Windows container and WSL2: does the background reach the published port without a Local Network Access prompt (Chrome and Edge 144+, Firefox 153)? NAT versus mirrored networking; vault and proxy survival across restarts and WSL's idle shutdown; does the Host check accept `127.0.0.1:8787`? | All checks pass on Podman Desktop, Docker Desktop and a WSL2 distribution, with the port reachable from loopback only | Owner-run (Windows 11) | Open — Phase 0 |
| S15 | Mode "extension" map store: `storage.session` availability, access limited to trusted extension contexts, quota, and survival across background restarts in Chrome, Firefox and Safari | The map survives background restarts, stays invisible to content scripts and fits the quota in all three; blocks only mode "extension" | Container (Chromium, done); owner-run (Firefox 153, Safari 27) | Passes on Chromium: 10,485,760-byte quota with atomic failure; content scripts refused by default; the map survives idle and forced worker stops; cleared by `runtime.reload()` and browser restart, so mode "extension" must also refill after an extension update. Firefox and Safari open |

### Phase 0 findings so far

The container half of Phase 0 ran on 2 October 2026 on Chromium 141 (headless), on local
fixtures only. Its results refine the decisions without changing any of them:

- **Admission (D2).** Chrome sends no `Origin` on a host-permitted extension's GET, so
  gateway endpoints that rely on origin admission must be POST or WebSocket; the
  metadata GETs keep their existing path. Chrome's host-permitted contexts are also
  CORS-exempt, so the scoped CORS answers matter mainly for Safari and for contexts
  without host permission (S1).
- **Relay through the background**, as planned: content scripts are subject to the
  page's CORS (S1) and its Local Network Access checks (S5).
- **Keepalive from the worker.** While any stream or gateway call is active, the
  background sends a WebSocket message to the gateway about every 20 s; content-script
  heartbeats alone fail in hidden tabs, and gateway-side protocol pings do not keep the
  worker alive. The gateway keys its rehydration holdback by stream id, so a stream that
  resumes after a worker restart never emits a broken placeholder (S4).
- **Hooks chain and stay writable.** Locking the wrappers silently broke a site's own
  instrumentation in a fixture; the initial-document gap of a page-created iframe is a
  residual unless the `contentWindow` getter is wrapped (S2).
- **Workers bypass the hooks.** A `Worker`-constructor wrapper works for dedicated and
  module workers but breaks under `worker-src 'self'`, so it is a per-site fallback, not
  a default; the real-site inventory decides whether D9 reopens (S3).
- **Mode "extension" (D1).** `storage.session` fits on Chromium; the map is refilled
  after an extension reload or update as well as after a browser restart, and a CI rule
  forbids opening the area to untrusted contexts (S15).
- **Safari transport (D2, S7).** Native messaging carries no `Origin`, so if S7 picks
  it, pairing comes from the container app rather than from an extension origin.
- **Testing.** Playwright's automatic attachment keeps an extension's service worker
  alive, so lifetime tests must never attach a debugger to the worker; branded Chrome
  137+ ignores `--load-extension`, and CDP `Extensions.loadUnpacked` works instead.

Still open for the Phase 0 exit: every owner-run row above (real sites, Chrome 144+/155,
Edge, Firefox, Safari, Windows, AMO), and the S15 answer for Firefox and Safari, which
decides whether mode "extension" ships.

## Phase plan

The cheapest and safest protection comes first: Lane C needs only
documentation, Lane B lands with the gateway on public wire formats, and
Lane A follows. Air-gapped enclaves get a complete kit at the end of
Phase 2, before any store listing.

| Phase | Scope | Exit criteria |
| --- | --- | --- |
| 0. Decide and de-risk | Spikes S1–S8, S14 and S15; this decision record | A written answer for every open spike; S15 decides whether mode "extension" ships |
| 1. Foundations | Admission (`allowed_extension_origins`, scoped CORS, `extension pair`), `/status` capabilities, the KoboldAI and OCI label fixes; Lane C, air-gapped and Windows guides; the WXT skeleton with badge, onboarding, pairing and managed schema | Five posture states correct on Chrome 144+ and Firefox 140+; red-team tests show page origins still refused, with nothing reaching upstream; the Lane C guide reproduced on an offline VM with Ollama and Open WebUI; the Windows checklist passes on Podman Desktop and WSL2 |
| 2. Gateway and Lane B | The gateway's three modes; the dedicated session; provider and realtime text codecs; hooks, relay and per-origin opt-in; the DNR fail-closed option; D1 docs and tests; unlisted betas; enclave kit v1 | BYOK fixtures (direct Anthropic, OpenAI and Gemini calls) redact end to end in all three modes; tests prove the proxy never receives API keys; "off" has no read-back paths; the refusal red-team passes; proxy down means blocked, not leaked |
| 3. Lane A | The `SiteAdapter` registry; Tier 1, then Tier 2 codecs with history, titles and memory; upload blocking; D3 behaviors; the DOM fallback; public store listings | Split sweeps per codec; fixture end-to-end runs; an owner live canary per site; unknown shapes blocked; `doctor` enforces a persistent vault |
| 4. Safari and enclave kit v2 | The Safari container app, the S7 transport, device-management templates, a notarized and stapled Developer ID package | The Safari 18.4+ checklist passes; a complete Chrome and Edge, Firefox ESR 153 and Safari install on an air-gapped network makes zero outbound connections; every artifact verifies offline |
| Backlog | Lane D; iOS; the Mac App Store; the llm-redact-pro pairing secret, named users, remote gateway and Team kit | Reopened only by a new decision record |

Every codec inherits the proxy's fixture and split-sweep discipline
([coding-standards.md](coding-standards.md#testing-style)), and
admission and the gateway get red-team scenarios
([security-testing.md](security-testing.md)). Changing a decision, or
reopening a backlog item, takes a new entry in this record, approved by
the owner.
