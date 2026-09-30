# Changelog

All notable changes to llm-redact are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

Convention: entries land in `[Unreleased]` with the change that introduces
them. A release moves `[Unreleased]` into a dated version section, bumps
`__version__` in `src/llm_redact/__init__.py` (the single source of truth),
and tags `vX.Y.Z`.

## [Unreleased]

### Added
- Document extraction is now part of the free core: `[extraction]` (docs/extraction.md)
  reads binary uploads — PDFs, Office/OpenDocument files, markup, RTF — as text in an
  isolated, resource-limited worker process per file, so the proxy scans them before
  anything is sent (a value found refuses the upload; a complete clean reading sends the
  file byte-identical). It moved here from llm-redact-pro unchanged in behavior; the
  Pro-tier requirement is gone. A core config section (restart-only, emitted by `config
  show`), built through `Registry.build_upload_inspector` (a plugin may still replace
  it; an enabled section for which the registered factory builds no inspector refuses
  to start). PDFs need the new `extract` extra (`pip install
  "llm-redact-proxy[extract]"`, pypdf); doctor gains value-free `extraction` rows.
- Cloud OCR services for `[extraction]`: AWS Textract (`kind = "textract"`, SigV4 with
  the core's own signer, credentials from named env vars), Google Document AI
  (`documentai`: an access token from a named env var, or a service-account key file —
  RS256 JWT with the `crypto` extra), Azure AI Document Intelligence
  (`azure_docintel`: `:analyze` then its operation polled, bounded by the service's
  timeout). Each requires `trusted = true` and https; any failure is no reading. A
  cloud reading counts as complete only when the pages it analyzed equal the file's
  own page count (a PDF the local extractor opened; one for a single-image PNG, JPEG
  or BMP) — Azure's free tier silently reads only the first two pages.
- Convert mode, opt-in (`[extraction] convert = true` or a list of classes): an
  upload whose binary file holds values to redact, read completely, is sent as its
  REDACTED extracted text (`text/plain`, file name `.txt`) instead of being refused —
  on routes whose provider takes a text file for the upload (OpenAI/Azure/custom Files
  with purpose `assistants` or `user_data`, container files, Anthropic Files), under a
  credential the proxy holds only with `proxy_credential`. The model sees text, not the
  original file. New outcomes `converted`/`converted_refused`; a `status` posture line.
- The container image ships the `extract` extra, so `[extraction]` with its default
  formats (pdf included) runs in it.
- `plugin_api.UploadPart.extension` (the file name's lower-cased extension, `""` when
  the part's names disagree) and `plugin_api.Inspection.convert_text`; the extractors
  treat an extension naming another format than the file's bytes as incomplete.

### Changed
- An upload whose binary parts go to the upload inspector is now refused BEFORE the
  inspection — never after it — for what refuses it whatever the redaction finds: a
  provider with no upstream configured (the 502), a routing layer's local refusal, and
  a `[audit] required` write-ahead START row that cannot be committed (the 503). For
  such an upload the START row is written right before the inspection (with no
  detections; the END row carries the request's own). When the redaction then finds
  values — warn-mode values are forwarded — a second START row carrying the counts is
  committed before any upstream contact and the early row is ended (status none, no
  detections), so the record durable before contact always says what leaves (its
  failure refuses 503); such a request has two START rows. Every refusal after the
  inspection — a value found in the file, a block, the upstream authorizer, a routing
  budget, a send that fails — is its END row; a way out that records nothing still
  closes it, so START and END rows stay paired. The upstream authorizer stays after
  the inspection (it signs the final, redacted bytes). Requests without an inspector,
  and uploads with no binary part to inspect, keep the previous order. These refusals
  no longer count an inspected part as `clean_refused`: nothing is inspected.
- The stored-object check (`SessionRouter.object_access_refusal`, llm-redact-pro's named
  users) now sees the METADATA part of the Gemini API's single-request upload
  (`POST /upload/v1beta/files`, a `multipart/related` body): it is handed that part's
  JSON object — what Google reads as the create's body, a chosen `file.name` included —
  exactly as it is handed the metadata-only JSON create's body, before anything is
  sent, under the client's own key and a credential the proxy holds alike. It is read
  like a JSON body (strict UTF-8, repeated keys last-wins, at most 128 levels deep); a
  metadata part repeating a key is sent re-serialized, exactly as checked (only its JSON
  text is rewritten: an empty header block's CRLF, whitespace and a byte-order mark
  around it stay, so the part keeps its shape). Metadata the
  check cannot read (lenient or non-UTF-8 JSON, a transfer encoding, a foreign charset,
  no JSON metadata first) is refused 400 under a credential the proxy holds and, with
  the client's own key, wherever redaction applies; only with `detection = false` and
  the client's own key does it go out unchecked, as an unparseable JSON body does.
  New adapter hook `ProviderAdapter.upload_metadata_boundary`.
- Multipart part headers are read with one reading only: a header line carrying a
  bare CR or LF (or any other control but a tab) is refused like any other ambiguous
  header, since a reader accepting a bare LF as a line break ends the header block
  there and reads what follows as the part's content — a Gemini upload's metadata (a
  chosen file name) the stored-object check never saw, or file lines redaction never
  scanned. Such an upload is refused 400 wherever redaction applies and under a
  credential the proxy holds, on every multipart route. So is a part with no header
  block that does not open with an empty one (see Fixed).
- A multipart request's boundary is read only when its Content-Type has one reading:
  a repeated `boundary` parameter (`boundary=a; boundary=b`, which a reader taking
  the last one parses with `b`), a quoted value holding `;` or an escape (which a
  naive split reads differently), a control, or a boundary outside the RFC 2046
  characters now leaves the body unreadable, so it is refused 400 wherever redaction
  applies and under a credential the proxy holds, like any other multipart body the
  proxy cannot parse. A non-ASCII boundary used to be read with its non-ASCII
  characters dropped. Applies to form uploads and the Gemini API's multipart/related
  upload alike.
- The Gemini API upload's metadata part is read as JSON only when it declares
  `application/json` (parameters allowed) or no type at all: a first part declaring
  another type is metadata the check cannot read (a server parsing by the declared type
  could read a form-encoded `file.name=…` out of a strict-JSON string), and a
  `multipart/related` naming its root part with `start` (the metadata then need not be
  the first part) is a body llm-redact cannot read. Both are refused 400 wherever
  redaction applies (the metadata is redacted as the value the check reads — see
  Fixed) and under a credential the proxy holds; with `detection = false` there, the
  declared type only where a session router's stored-object check reads the upload
  (llm-redact-pro).
- File uploads (OpenAI/Azure/custom `…/files`) are read by CONTENT
  (`upload_content.classify_file`): JSONL is redacted per line as before, any other
  text file (UTF-8, or UTF-16/32 with a BOM) is redacted as one text and re-encoded as
  it came, and a BINARY file (a PDF, image or archive — a known signature or a NUL
  byte means binary even when the bytes decode) is no longer refused under the
  client's own key: it is forwarded UNSCANNED, its file name still redacted, and
  counted. Under a credential the proxy holds (cloud identity, a routed operator key)
  a binary file is still refused 400 — unless an upload inspector cleared it (see
  Added: only when the inspection allows that credential). Downloads of text files
  (`GET …/files/{id}/content`) are restored line by line; binary downloads are
  untouched.

### Added
- Upload inspection seam (`plugin_api.UploadInspector`, `UploadPart`, `Inspection`;
  `Registry.build_upload_inspector(config, tier)`, Free default None): a plugin
  (llm-redact-pro's document extractors) reads each BINARY file part of an upload as
  text, awaited before redaction on every upload route the core redacts (OpenAI,
  Azure, custom and container `…/files`, Anthropic's and the Gemini API's Files
  uploads), bounded by the core (16 parts, 4 at a time, the inspector's size limit,
  one deadline capped at 300 s; what still runs is cancelled). The extracted text is
  scanned with the live detectors without issuing placeholders
  (`Redactor.scan_text`: allowlists, modes and deny strings apply): a block-mode
  value blocks, any value that would be redacted refuses the upload (400, naming its
  types — the file cannot be rewritten), a warn-mode value is counted and forwarded.
  A COMPLETE reading that scans clean sends the file byte-identical — with the
  client's own key always, under a credential the proxy holds only when the
  inspection allows it; anything else (no text, an incomplete clean reading, a
  timeout, a fault) keeps the unscanned-binary rules. New `/status`
  `inspected_uploads_total` and `upload_inspector`,
  `llm_redact_inspected_uploads_total{provider,outcome}` (`clean` counts only parts
  of an upload handed to the upstream; a clean part of an upload refused before that
  is `clean_refused`), a `llm-redact status` posture line for clean forwards. Every
  part's headers (file names, transfer encodings, charsets, a header block every
  reader finds) and every scanned part's format (a form field that is not UTF-8 text,
  a JSONL line nesting too deep, the Gemini upload's metadata) are checked before any
  part is inspected, so an upload the redaction would refuse for its form is never
  handed to the inspector; with a rule in block mode, so is every string the
  redaction will scan (file names, form fields, text and JSONL files, the Gemini
  metadata) for a block-mode value, read-only and under its own `max_body_strings`
  count. What needs the redaction itself — `max_body_strings` without a block-mode
  rule, a sealed session, placeholder exhaustion, a vault fault — the upstream
  authorizer and a routing budget refusal can still follow the inspection; no upstream
  configured, a routing layer's local refusal and the `[audit] required` START row come
  before it (see Changed). A clean scan covers the extracted text only.
- `[detection] binary_uploads = "forward" | "refuse"` (default `"forward"`, hot):
  `"refuse"` keeps refusing binary uploads under the client's own key too. Forwarded
  binaries are surfaced: /status `unscanned_uploads_total`,
  `llm_redact_unscanned_uploads_total{provider}` (Grafana panel), a `llm-redact
  status` posture line when nonzero, and a doctor line stating the setting.
- Optional `SessionRouter.response_observer(context) -> ResponseObserver | None`
  (`plugin_api.ResponseContext`, `plugin_api.ResponseObserver`): a plugin observes
  upstream answers read-only and provider-neutrally. Told the adapter, method, path,
  status, content type, the request body as sent upstream, the session and whether a
  proxy-held credential was spent, its callable receives each JSON value of the
  answer as the PROVIDER sent it (placeholders only, each its own parse): a buffered
  JSON body once, SSE per event, NDJSON per line, AWS eventstream per frame (realtime
  is not observed). Faults are contained as bookkeeping stage `response_observer`
  (type-only log, the answer delivered unchanged); a router without the member costs
  one attribute test per answer.
- Open realtime relays and dashboard live-events streams now end when their admission
  ends. An access gate may close them at once through the optional
  `AccessGate.bind_connections(control)` (`plugin_api.ConnectionControl`:
  `close(subject=, grant=, reason=)`, thread-safe), and the core re-checks each open
  connection's optional `Admission.recheck` every `recheck_interval` seconds (gate
  member, 5..3600, default 30), failing closed: a check that raises, times out or
  answers anything but a bool, None or a string closes the connection. A WebSocket
  closes 1008 with the gate's reason (upstream 1000); an events stream ends.
  `Admission.grant` is an opaque key the core never logs or reports. New `/status`
  `connections` block, `llm_redact_connections_closed_total{cause}`, bookkeeping
  stage `recheck`. Streaming HTTP answers are not cut (each answers a request that
  was admitted).
- Optional `SessionRouter.realtime_frame_refusal(adapter_name, path, frame, *,
  identity, session_id) -> str | None`: the realtime twin of `object_access_refusal`.
  A router with it is asked, synchronously, for every client frame of a realtime
  connection that parses as JSON (text or binary; OpenAI Realtime, Azure, Gemini
  Live, Vertex Live; every mode; `detection = false` included) before the frame is
  redacted, numbered or sent, in the connection's own context. A string closes the
  connection 1008 with that fixed reason (cut to 123 bytes), recorded as a 403 —
  nothing of the frame reaches the upstream or the vault; an exception or a malformed
  answer closes it 1008 with the core's reason, counted as bookkeeping stage
  `realtime_frame` (type-only log). While a router checks frames, a frame nesting
  JSON too deep is refused on every connection, a non-JSON one under the proxy's own
  identity even with `detection = false`, and `detection = false` sends the checked
  value re-serialized. Each frame is still parsed once. A router without the member
  costs one attribute test per connection.
- Optional `SessionRouter.realtime_server_frame(adapter_name, path, frame, *,
  identity, session_id) -> None`: the server-side twin of `realtime_frame_refusal`.
  A router with it is handed, synchronously, every UPSTREAM frame of a realtime
  connection that parses as JSON within the proxy's JSON bound (all four realtime
  adapters, text or binary, every mode, `detection = false` included) as its own
  parse of the provider's bytes — placeholders, never a restored value — BEFORE the
  frame is restored or sent to the client, in the connection's own context. It is
  read-only: the return value is ignored and nothing it does to its parse reaches
  the client; an exception is contained (bookkeeping stage `realtime_server_frame`,
  type-only log) and the frame delivered. llm-redact-pro uses it to record whose
  Gemini/Vertex Live session a `sessionResumptionUpdate` handle belongs to. A router
  without the member costs one attribute test per connection and no parse.
- The Gemini API's Batch Mode beyond the create is recognized, so a credential the
  proxy holds (a routed operator key) may reach it: a batch's status
  (`GET /v1beta/batches/{id}`, the name the create answers with) and the batch list
  restore the display name and a finished batch's INLINED responses; cancel and
  delete are redact-only; `models/{m}:asyncBatchEmbedContent` is redact-only and
  tracked like `:batchGenerateContent`. No batch route carries the system note.
- The Gemini API's Files API is recognized: the single-request upload
  (`X-Goog-Upload-Protocol: multipart`, a `multipart/related` body) and the
  metadata-only create are redacted part by part: the upload's first part, the file's
  metadata, as the create's JSON body (see Fixed), every other part with the OpenAI
  Files upload's content policy (read as a file by its content; a binary file forwarded
  unscanned only with the client's own key under `binary_uploads = "forward"`,
  counted in `unscanned_uploads`); a file's metadata and the file list restore each
  `displayName`; the download (`…:download`, `/download/v1beta/…:download`) is
  restored like an OpenAI file download; delete is redact-only. A RESUMABLE upload's
  start is refused (403) under a credential the proxy holds — its upload URL would
  carry the file's data to Google unread, as the proxy's principal — and an
  `X-Goog-Upload-URL` answer header is never relayed there; data chunks and the raw
  protocol stay pass-through (the client's own key only).
- Anthropic's Files API (beta, a request carrying `anthropic-version` alone) is
  recognized: the upload uses the OpenAI Files upload unchanged, every file object's
  echoed filename is restored (upload answer, list, metadata), a file's content is
  restored like an OpenAI file download, delete is redact-only.
- Optional adapter hooks: `listing_item_id` (a listed item's id: `id`, Gemini
  `name`), `multipart_boundary` (a route reading another multipart type),
  `proxy_credential_refusal` (a recognized protocol never served with a proxy-held
  credential: recorded 403) and `capability_response_headers` (never relayed under
  one). `openai.rehydrate_text_file` is the one file-download restoration.
- OpenAI fine-tuning jobs, vector stores and code interpreter containers are
  recognized (OpenAI, both Azure families — containers on the v1 API only — and
  custom providers), so a routed operator key or cloud identity may reach them.
  Fine-tuning: job `metadata` redacted and restored in every echo, event messages
  restored, checkpoints recognized. Vector stores: `name`, `description`,
  `metadata`, file `attributes` (a caller-keyed map, now walked like `metadata`),
  search queries and filter values redacted; echoes, search results and a file's
  parsed content restored. Containers: `name` redacted and restored, container
  file uploads read by content like `/v1/files` uploads, downloads restored like a
  Files API download. New adapter hooks `verbatim_fields` and `label_fields` with
  `providers.base.prepare_route_request` as the proxy's redaction entry point:
  identifier fields the provider uses exactly as sent (a fine-tune `suffix`, which
  becomes part of the model name; training/validation file ids; W&B
  `integrations`; the file ids of a store, attach, file batch or container; a
  search filter's attribute `key`) are scanned but never rewritten — a value that
  would be redacted there refuses the request (400); label fields under a
  structural key (a store's or container's `name`) are redacted and restored.
  Created stores, containers and container files are reported to a session router
  (so is the container a Responses code interpreter call ran in), and the job,
  store and container lists are listings it attributes per item. Checkpoint
  permissions stay pass-through, and the Uploads API stays unrecognized (a part is
  an opaque byte range; documented in docs/api-coverage.md).

### Changed
- Recognized upload routes are capped by `max_body_bytes` (default 10 MiB):
  Anthropic Files and Gemini Files uploads over the cap sent with the client's own
  key used to pass through unscanned and are now refused 413 — raise
  `max_body_bytes` for larger files.

### Fixed
- An upload refused for a form field that is not UTF-8 text or a JSONL line nesting
  too deep had its binary parts handed to the upload inspector first (which may send
  them to an extraction service): those format checks now run before the inspection,
  with the header checks. The CHANGELOG and docs claimed more than that: they now state
  which refusals can still follow an inspection (see Added).
- An upload refused for a block-mode value in another part or in a file name (an
  address in a PDF's name, a text file or form field beside it) had its binary parts
  handed to the upload inspector first. With a rule in block mode, every string the
  redaction will scan is now checked for one before any part is inspected, exactly as
  the redaction reads it (a batch line's request walk included), with nothing issued or
  counted and its own `max_body_strings` count (`Redactor.blocked_type`,
  `PartsReading.require_unblocked`).
- An inspected upload part that scanned clean was counted `clean` ("forwarded after a
  clean scan") as soon as redaction returned, even when the proxy then refused the
  request without contacting the upstream (no upstream configured, the upstream
  authorizer failing, the `[audit] required` START row failing, a routed budget
  refusal); binary parts were likewise counted in `unscanned_uploads_total` and logged
  as forwarded. Both now count only once the request is handed to the upstream (a
  send that then fails in transit included); a refused upload's clean parts are
  `clean_refused`. And a request whose extracted texts ran out of `max_body_strings`
  no longer drops its parts' outcomes: the part that ran the budget out and every
  later one count `incomplete`.
- A text file upload redacted as one text was remembered for its download (restored
  raw, by digest, newest 1024) as soon as redaction returned, even when the proxy then
  refused the request without contacting the upstream (no upstream configured, the
  upstream authorizer, the `[audit] required` START row, a routed refusal): such
  refusals could evict what sent uploads recorded, and that download then came back
  restored escape-aware instead of byte-exact. It is now remembered only once the
  request is handed to the upstream.
- An upload's request row (log line, `/recent`, `/events`, audit rows and sinks, OTel)
  counted the redactions and warn-mode forwards of every OTHER request that ran while
  the upload inspector read its binary parts: the per-request count diff spanned that
  await. The window now restarts once the inspector returns, so a row counts only its
  own request (the extracted texts' warn-mode values included). Process-wide totals
  were never affected.
- The Gemini API upload's metadata part (`POST /upload/v1beta/files`, the first part of
  the `multipart/related` body) was shown to the stored-object check as a JSON body but
  redacted as a FILE: on one line it was parsed as JSON, pretty-printed it was redacted
  as raw text, so a JSON-escaped value (`\u0040` in an address, `ensure_ascii` escapes
  of a name) went out unredacted and the provider decoded it. The part is now read and
  redacted as the one JSON value the check reads (`upload_view.read_metadata_part`:
  declared `application/json` or undeclared, strict UTF-8, escapes resolved, every
  string redacted, keys never; only the JSON text's span rewritten, and only when a
  value changed or a key repeats). Metadata that reading refuses (lenient or non-UTF-8
  JSON, another declared type, a foreign charset, the media first) is refused 400
  wherever redaction applies, with or without a session router, as a JSON body the
  proxy cannot read is; it used to be redacted as a file and forwarded.
- A multipart part with no header/body separator (no CRLF CRLF) was redacted as a
  plain field with no header at all, while a reader accepting a bare LF as a line
  break finds headers in it: a file name, a `Content-Transfer-Encoding` (a
  quoted-printable address no detector matches, base64 hiding a binary file) or a
  charset the proxy never read. Such a part is now refused 400 wherever redaction
  applies (whatever the credential) and wherever the stored-object check reads the
  upload, on every multipart route and every part — the Gemini API upload's media
  too — before any part is handed to an upload inspector. An empty part, or one
  opening with an empty header block (CRLF), is still read.
- A realtime client frame holding JSON the parser refuses for anything but its syntax
  (an integer longer than Python's 4300-digit limit) counted as "not JSON", so under
  the client's own key it was relayed as sent — neither redacted nor put to the session
  router's frame check. It is now refused like a frame nesting too deep: closed 1008,
  recorded 400 (the HTTP twin's answer). Upstream frames the proxy cannot read are
  still forwarded as they came.
- An upload refused part way through no longer records its text parts in the bounded
  memory of text uploads whose downloads are restored byte-exact, so refused requests
  cannot evict what accepted uploads recorded. That memory is per process: a JSON text
  upload downloaded through another replica, another `llm-redact run` proxy or after a
  restart is restored as valid JSON with every value, but a value whose source held an
  escape comes back escaped once more (documented in docs/providers.md).
- A downloaded JSONL file restores EVERY value of its lines, under `id`, `name`,
  `type`, `data` and the other request-body structural names too, as the upload now
  redacts them (the download kept the request-body skip set, so a data file came back
  with placeholders under those keys). A batch output's tool-call `arguments` is still
  restored as JSON source.
- A downloaded text file that is one JSON document (a JSON file a model or code wrote
  around a placeholder, served as `application/json` or not) stays valid JSON: each
  restored value is JSON-escaped where it lands, keys included and formatting kept (a
  PEM key's newlines once landed raw inside a string). Raw restoration is kept for the
  files it is right for: every text upload this process redacted as one text is now
  remembered (by digest, newest 1024) and comes back byte-exact. A JSON text upload
  no longer remembered (a restart) is read like a model-written file.
- A connection re-check that swallows its own cancellation can no longer hold the
  periodic re-check pass open (and so stop every later re-check): the pass stops
  waiting at the timeout, closes the connection and abandons the check, instead of
  waiting for it to finish cancelling as `asyncio.wait_for` does.
- A buffered file download served as JSON Lines under a JSON content type
  (`application/jsonl` contains `application/json`) is restored through the
  adapter's file-download restoration instead of being left unrestored.
- The app lifespan tolerates `add_signal_handler` raising `ValueError` (uvloop off the
  main thread) alongside `NotImplementedError` and `RuntimeError`; SIGHUP reload is
  then unavailable, as on Windows.
- An uploaded text file whose first bytes happen to spell a media format's name
  (a CSV row opening `ID3,`, a note opening `RIFF` or `GIF89a`, a word `ftyp` at
  byte 4) is redacted as text again instead of being classified binary and
  forwarded unscanned. Only `%PDF-` and control-byte signatures still mark
  decodable bytes as binary; real files of the other formats fail the strict
  text decode on their own.
- With a session router that checks stored-object access, a UTF-16/32 text upload
  whose first line parsed as JSON repeating a key was rewritten in mixed encodings
  by the check's re-reading, then read as binary and forwarded unscanned. The
  check now reads file lines only as UTF-8 text and rewrites a line only in a
  UTF-8 file; an upload whose re-reading would change what a file part is gets a
  400.
- A downloaded text file (not JSON Lines) is restored as one text, the way it was
  redacted on upload: a line that parses as JSON only once it holds a placeholder
  no longer gets its restored value JSON-escaped, so the file round-trips byte for
  byte. JSON Lines files are still restored line by line as JSON.
- Holding a route's verbatim fields out of redaction (vector store search filter
  keys, file batch file ids) costs time linear in the body: a request with many
  such fields used to block the proxy's event loop for tens of seconds.
- A fine-tuning job create's reinforcement grader names (nested multi-grader
  names included) are redacted like other user-written labels and restored in
  every echo of the job; they were forwarded as sent, now under a credential the
  proxy holds too.
- A request on a recognized route that carries an HTTP method override (the
  `X-HTTP-Method-Override`, `X-HTTP-Method` or `X-Method-Override` header, or a
  `_method`/`$httpMethod`-style query parameter) is refused (400) before its body
  is read, and those headers are never forwarded on a recognized route or a
  realtime relay: an upstream honoring one ran another method than the one the
  request was redacted, restored and checked as (a file create served as the file
  list). Unrecognized pass-through traffic forwards them as sent. A Gemini file
  create's answer reports only its own file, never a `files` array.
- An access gate's awaitable re-check that ends in a cancellation (a shared lookup
  another path cancelled) now closes its connection like any failed check; it used
  to end the re-check backstop for good, leaving later revocations only a re-check
  could see unapplied. A backstop task that ended anyway is started again.
- An Anthropic Messages answer's code execution container (buffered, and streamed
  in `message_start` / `message_delta`) is reported to a session router as the
  requester's, like the files the run wrote, so a later request reusing the
  container can be checked against its creator; a container the request itself
  named is never reported.
- An uploaded JSON Lines data file (OpenAI Files of a purpose other than batch or
  fine-tuning, container files, Anthropic Files, Gemini uploads) has every value
  redacted, including values under keys such as `id`, `name`, `type` or `data`,
  which were skipped as request protocol fields and forwarded as sent. Only a
  batch line's request body and a fine-tuning example's conversation keep the
  request reading, and only those get the system note.
- With a session router that checks stored-object access, reading an upload for
  the check no longer parses every line: blank lines, prose and other lines that
  cannot hold a JSON object cost one scan, and the lines it does parse are
  counted against `max_body_strings` (an upload over it is refused 413 under a
  credential the proxy holds). A 10 MiB upload of blank lines used to block the
  proxy for about a minute.
- A downloaded file (OpenAI, Azure and custom-provider files and container files,
  Anthropic Files, Gemini downloads) is restored per file whatever Content-Type the
  provider serves it with. A JSON file served as `application/json` used to be
  walked as one JSON body, leaving placeholders in its keys and under names such
  as `id` or `data` and re-serializing the whole file; JSON Lines and event-stream
  media types took the streaming readings.
- A text file uploaded redacted as one text that reads as JSON Lines only once
  redacted (a line holding the secret was not valid JSON before, such as an
  unescaped backslash in a Windows account name) is restored on download exactly
  as it was uploaded, instead of line by line as JSON with the restored value
  JSON-escaped and the line re-spaced. The proxy remembers such uploads by a
  digest of the bytes it sent, for the newest 1024 in the running process.
- Holding a route's verbatim fields out of redaction no longer costs time per
  level of nesting for each field, and each field found counts against
  `max_body_strings` at once: a deeply nested vector store search with too many
  filter keys is refused as fast as any over-budget body, instead of blocking the
  proxy for tens of seconds first.
- Stopping the connection re-check backstop while an access re-check is pending no
  longer lets that check's later exception be logged at shutdown as an
  unretrieved task exception (with its message); its outcome is discarded, as a
  timed-out check's already was.
- A connection already closed for access (an events stream whose client stopped
  reading stays open until its socket closes) is no longer re-checked every
  interval, logging and counting a failed re-check each time, nor reported as
  open. A connection whose earlier re-check is still running after being
  abandoned is not asked again, and at most 64 abandoned re-checks may run at
  once (further awaitable re-checks count as failed without being started).
- A realtime relay revoked (by a config reload or the access gate) while the proxy
  was dialling its upstream is closed without being served: the client no longer
  receives the upstream's opening frame, and the refusal is recorded (503 for a
  reload, 403 for an access revocation). An open relay revoked by the gate no
  longer sends the client upstream frames that arrive after the revocation.

## [1.9.0] - 2026-09-29

Hardening of the request path. Web pages can no longer use the proxy; a request
goes only to a provider it is positively attributed to; a recognized route forwards
only a body the proxy read; a placeholder can no longer be given a second meaning;
and a config reload reaches open realtime connections. llm-redact-pro 0.14 requires
this release: without the new request-path seams its paid tiers and identity
providers refuse to start.

**Upgrading** (a request 1.8 forwarded may now be refused):
- There is no Anthropic default any more. A request no route matches is forwarded
  only to a provider it can be attributed to (a path family, or headers only one
  provider's clients send); anything else is a recorded `404`. `OPENAI_BASE_URL`
  must end in `/v1` (`http://127.0.0.1:8787/v1`); without it OpenAI-shaped requests
  get a `404` that names the fix. `llm-redact run`, `init` and the deploy manifests
  now export it with `/v1`.
- A path with an empty segment (`//`, often a base URL ending in `/`) is refused
  `400`. Another spelling of a recognized route (a trailing `/`, other case, or what
  a front end that normalizes paths reads as the route: `\` or `%5C` for `/`, a
  segment's `;params`, trailing spaces, tabs or dots, `%uXXXX` escapes or a second
  percent-encoding, Unicode compatibility forms) is a recorded `400`; a recognized
  route without its `/v1`, or under an extra prefix such as `/v1/v1/…`, is a
  recorded `404`.
- Requests from web pages are refused. A browser request (one carrying `Origin` or
  a `Sec-Fetch-*` header) from another origin or site, or addressed to a host name
  the proxy does not answer to, gets a recorded `403` (WebSocket: close `1008`).
  List a browser app you trust with your restored values in the new
  `allowed_origins`. A request that spends a credential the proxy holds (`auth =
  "identity"`, a routed operator key) over plain HTTP must name a host the proxy
  answers to; list aliases such as a compose service or a Kubernetes Service in the
  new `allowed_hosts`.
- Where redaction applies (`detection` on, the default; the client's own key
  included), a recognized route refuses a body it cannot read with a recorded `400`
  instead of forwarding it unredacted: non-JSON bytes, invalid UTF-8, bytes after
  the JSON value, a top-level JSON array or scalar, JSON nesting deeper than 128
  levels of objects and arrays (an uploaded JSONL line's too), whitespace only,
  multipart on a route that does not redact multipart or outside the canonical form,
  and a repeated `Content-Type`. A `Content-Encoding` other than `identity` gets
  `415` with `Accept-Encoding: identity`: send request bodies uncompressed.
- Uploads must be readable in full: a `/v1/files` upload (OpenAI, Azure, custom
  providers) of a file that is not JSONL, such as a PDF, text, CSV or image file,
  is refused `400`, and so is a form field that is not UTF-8, a multipart preamble
  or epilogue, a part header with more than one reading, a
  `Content-Transfer-Encoding` other than 7bit/8bit/binary, or a declared charset
  other than UTF-8/US-ASCII. Image-edit and video media parts are still sent as
  they are. `[providers.NAME] detection = false` restores verbatim forwarding for
  that provider (with the client's own key).
- New `max_body_strings` (default 100,000): a redactable request with more strings
  to redact (JSON strings, form fields, file names, uploaded JSONL lines) or more
  multipart parts gets the same `413` as `max_body_bytes`; a realtime client frame
  over it closes `1009`. Raise it together with `max_body_bytes` for large batch
  uploads.
- An upstream redirect (a 3xx with `Location`) is relayed only for an unrouted
  request that is pass-through or has no body, is not signed with the proxy's
  identity, carries no credential for the proxy itself and is not addressed to a
  custom upstream; anything else gets a recorded `502` naming the status.
- With llm-redact-pro routing, a route llm-redact does not recognize is refused
  `403` before its body is read when the plan would send it with an operator key
  (or none): vector stores, fine-tuning, moderations, assistants/threads, uploads,
  Anthropic and Gemini Files, stored-completion list/update/delete and Ollama model
  management need the client's own key.
- With llm-redact-pro's per-conversation sessions or named users, a Responses
  compaction (`POST /v1/responses/compact`) or input-token count
  (`/v1/responses/input_tokens`) that carries a value to redact is refused `403`
  until llm-redact-pro resolves the two routes: it reads `compact` and
  `input_tokens` as the id of a Response it has no record of. Both used to be
  forwarded unredacted; the default static session is unaffected.
- A config reload that changes an open realtime connection's provider settings,
  its authorizer or `[detection]` closes it with `1012` (reconnect).
- The `realtime` extra requires websockets 15.0 or newer.
- A server RDBMS vault creates the `llm_redact_retired` table at startup; a
  database user that may not create it gets a startup error naming the table and
  its DDL.

### Security

- **Web pages could read the vault back and spend the proxy's credentials.** A page
  in the operator's browser could send requests through the local proxy. With its
  own provider key it could have the model repeat a placeholder and read the real
  value the proxy restored, because the CORS preflight was forwarded and the
  provider's CORS answer relayed; realtime WebSockets get no CORS check at all, and
  a DNS-rebound page is same-origin. A cross-site "simple" POST, a rebound page or a
  WebSocket could also spend the proxy's own cloud identity or a routed operator
  key. Every HTTP request and WebSocket upgrade bound for an upstream is now checked
  before any credential fetch or upstream contact: browser markers (`Origin`, any
  `Sec-Fetch-*`) require a host name the proxy answers to (its loopback names, the
  bind host, `allowed_hosts`, the access gate's public origin), the request's own
  origin (port-exact) and a `Sec-Fetch-Site` of `same-origin` or `none`, or an
  `Origin` listed in `allowed_origins`; a request that would spend a credential the
  proxy holds needs such a host name even without browser markers, unless it
  arrived over TLS. Refusals are a recorded, provider-shaped `403` (WebSocket:
  close `1008`, checked first), counted by kind in `/status`
  `request_origin_refusals_total` and logged by kind only.
- **Requests went to the wrong provider.** An unrecognized request was forwarded to
  the Anthropic upstream by default: with the documented
  `OPENAI_BASE_URL=http://127.0.0.1:8787`, an OpenAI client's API key and
  unredacted prompt went to api.anthropic.com, and an Anthropic SDK's
  `GET /v1/models` went to OpenAI with its `x-api-key`. Requests are now attributed
  positively: an explicit path family first (`/v1beta/`, `/v1/projects/`,
  `/v1/publishers/`, `/openai/`, `/model/`, `/api/`, `/v2/`, `/v1/messages`, …),
  then headers only one provider's clients send (`anthropic-version`, a Google API
  key or any `x-goog-*` header, `openai-*` headers, the Cohere SDK's
  `x-fern-sdk-name`), then OpenAI for a non-Anthropic `Bearer sk-` key or an OpenAI
  resource path (`/v1/containers`, `/v1/evals`, `/v1/chatkit` and
  `/v1/organization` added). Anything else, or markers of two providers, is a
  recorded `404`.
- **Misaddressed routes were forwarded unredacted.** Trailing-slash, doubled-slash
  and case spellings of recognized routes, the spellings front ends normalize to
  them (`/v1/chat%5Ccompletions` reached api.openai.com unredacted, and so did
  `…/chat/completions;x` and `…/chat/completions%20`: IIS and API Management read
  `\` as `/`, Tomcat and Jetty drop `;params`, IIS trims trailing spaces and dots),
  a recognized route missing `/v1` or under an extra prefix, and OpenAI-compatible
  endpoints under a base path without `/v1` (custom upstreams such as
  `/custom/NAME/inference/…` or `…/api/paas/v4/…`, and the Gemini API's
  `/v1beta/openai/` surface) all reached their upstream unredacted. They are now
  refused as above, or matched on the endpoint's tail and redacted. Routing a path
  costs time linear in its length, and the misaddressing check runs only for a
  request the request-origin rule and the access gate admit: the tail search tries
  tails of at most eight segments after `/v1` (every OpenAI endpoint has four or
  fewer) and the one at the first OpenAI resource name, and an extra prefix is
  looked for up to eight segments deep.
- **Responses compaction and input-token counts were forwarded unredacted.**
  `POST /v1/responses/compact` and `POST /v1/responses/input_tokens` take a
  responses.create body, and neither was matched: the whole `input` went out in
  clear text — for a compaction (Codex CLI compacts long sessions) the entire
  conversation, restored values included. A compaction is now a chat route: its
  window is redacted, the system note joins its `instructions` (the compacted state
  must carry every token exactly), and the compacted window it answers is restored
  (the encrypted compaction item goes back as sent). The count is redact-only,
  redacted with the note like the request it counts. Both are matched on OpenAI,
  Azure (`/openai/v1/…` and `/openai/…`), custom providers and the Gemini API's
  OpenAI surface; neither creates a stored object.
- **Recognized routes forward only a body the proxy read.** Wherever redaction
  applies, and whatever `detection` says when a request spends a credential the
  proxy holds, the bodies listed under Upgrading are refused before the
  stored-object check, the session, redaction, any credential fetch, the routing
  plan's first hop, the audit START row and any upstream contact. They used to be
  forwarded verbatim, and lenient upstreams (Ollama, Express, Jackson) decoded them
  unredacted; under `auth = "identity"` or a routed operator key they were signed
  with the proxy's credential. On an identity realtime connection a non-JSON frame
  closes the connection `1008`, unsent (recorded as `400`).
- **A repeated JSON key bypassed redaction.** The parser keeps the last occurrence
  of a repeated key, so earlier ones were never scanned, and a body with nothing
  else to redact was forwarded as its original bytes, earlier occurrences
  included; an upstream that keeps the first occurrence received the unredacted
  value. A body or uploaded JSONL line that repeats a key at any depth is now
  always re-serialized from the scanned object (with `detection = false` too), and
  a Bedrock `count-tokens` blob that repeats a key is refused `400`.
- **Upload filenames and form fields are redacted.** A multipart part's `filename`
  and RFC 8187 `filename*` are redacted on every auth mode (only the value bytes
  change) and restored where the provider echoes them: the upload response, the
  file list and a file's metadata (OpenAI `/v1/files`, Azure `/openai/files` and
  `/openai/v1/files`, custom providers). Plain form fields are scanned as text, so a
  `user` field holding an email is redacted like its JSON twin. A part carrying
  only `filename*` is treated as a file, so its JSONL lines are redacted.
- **Batch `metadata` was forwarded unredacted** (OpenAI and Azure) and echoed on
  every batch read. Batch create, retrieve and cancel now redact it, and every batch
  echo, the batch list included, restores it in the request's own session (OpenAI,
  Azure and custom providers). With llm-redact-pro's named users only the reader's
  own batches are restored.
- **Values under caller-chosen keys named like protocol fields** (`id`, `name`,
  `type`, `data`, …) are redacted and restored: `metadata` and `requestMetadata`
  anywhere, Responses/Realtime `prompt.variables`, and a `:predict` body's
  `instances`/`parameters`. Tool results and documents nested under such keys
  (Gemini/Vertex `functionResponse`, Vertex Live `toolResponse`, Bedrock Converse
  `toolResult`, Cohere `documents`, realtime events) are redacted too; only scalar
  values of structural keys are skipped now.
- **Values written with non-ASCII digits were not redacted.** The prefilters tested
  ASCII literals, so nine rules (My Number, SSN, SIN, TFN, NINO, CPF, Belgian NN,
  personnummer, phone) skipped values written with full-width, Arabic-Indic or
  Devanagari digits, and case-insensitive rules missed `İ`, `ı` and `ſ`. The
  prefilters now see text as the patterns do.
- **A new value could reuse a placeholder its own request already carried.** A
  session numbers its tokens from 001, so a compacted history forked into a fresh
  session, an answer pasted from another conversation or a token from another proxy
  could be issued again for a different value, and the echo restored the wrong
  one. New values are now numbered above every placeholder the request carries
  (canonical, fuzzy-mangled and JSON-escaped forms; the whole upload; the Bedrock
  `count-tokens` blob; a running floor per realtime connection). Numbers stop at
  999999999: past that the request is refused `400`.
- **A deleted session's numbers could be issued again.** After a whole-session
  delete (the TTL prune, `POST /__llm-redact/sessions/prune`, `llm-redact sessions
  prune`, an access gate's purge), a re-created session could issue a deleted
  value's number again for a new value, and another instance's cache or an open
  realtime connection then restored one value where the other was meant. Every
  delete now retires the session's highest number (`retired_numbers`, RDBMS
  `llm_redact_retired`), new values are numbered above it, and views re-check at
  most once a second. The prune's idle check and its delete are one transaction. A
  re-check that cannot read the database keeps the view's cache (a cached token only
  ever restores its own value) and runs again a second later, counted as
  `bookkeeping_errors{stage="vault_check"}` and logged once per outage by exception
  type: restoring a cached token never needs the database.
- **A config reload did not reach open realtime connections.** A reload that
  withdrew the proxy's identity (`auth` back to `passthrough`, the provider
  disabled, its upstream moved) kept spending it on the live upstream session, and
  a reload that tightened redaction never applied to new frames. Such a reload now
  closes the connection on both sides (`1012`, reconnect) in the same step that
  swaps the configuration in; a connection still being authorized is never dialled
  (recorded `503`).
- **A reload during a body read changed the request's authorization.** `handle()`
  reads the provider's authorizer and upstream once, before the body, so a request
  admitted as pass-through is never signed.
- **Redirects are never followed with protected data.** The realtime relay no
  longer follows a WebSocket handshake redirect (a failed dial: `1011`). An HTTP
  upstream redirect is a recorded `502` naming only the status (the `Location` is
  never relayed or logged) wherever a following client would re-send a redacted
  body or a credential; a routed hop's redirect is a hop fault the router can fail
  over from.
- **A routed operator key is treated like the proxy's own identity.** A route
  llm-redact does not recognize is refused `403` before its body is read when a
  plan would send it with an operator key (or none), and the stored-object check
  runs after the routing plan and applies its identity policy to such requests
  (`RoutePlan.proxy_credential`; a plan without it counts as the proxy's). An upload
  or body the check cannot read is refused under a proxy-held credential.
- Request paths with `.`/`..` segments (including `%2E` and backslash forms) are
  refused with 400 (WebSocket: 1011) before any upstream contact.
- Under `auth = "identity"`:
  - more credential query and header variants are stripped (`$key`, `userProject`,
    `quotaUser`, `subscription-key`, `password`, `passwd`, `oauth_token`, and the
    matching headers);
  - only the `realtime` and `openai-beta.*` WebSocket subprotocols are forwarded;
  - the signed URL must be exactly the base URL plus the request path;
  - an `http://` upstream on a non-loopback host is a ConfigError.

### Added

- `allowed_origins` (top-level, restart-only, default empty): exact browser origins
  the proxy serves across origins, over HTTP and realtime WebSockets. A listed page
  can read restored values back and spend the proxy's credentials, so `doctor` warns
  with the list, `/status` counts it and `llm-redact status` shows it. Only http(s)
  origins, plain http only on this machine, never `null`; the host rule still holds,
  reserved `/__llm-redact/*` paths never consult the list, and CORS answers come
  from the provider (proxy-generated errors carry no CORS headers).
- `allowed_hosts` (top-level, restart-only, default empty): alias host names clients
  use to reach the proxy. The Helm chart's standalone mode lists its Service names;
  `doctor` reports the names and warns when a plain-HTTP non-loopback bind lends a
  credential without them.
- `max_body_strings` (top-level, hot, default 100,000; see Upgrading), in `/status`,
  `doctor`'s body-cap line, `config show` and the config-edit command.
- Session-router seams for stored-object ownership (llm-redact-pro), optional and
  read via `getattr`, so older routers keep today's behavior:
  - `object_access_refusal(adapter_name, method, path, body, *, identity)` is asked
    for every forwarded HTTP request, routed or not, after the routing plan and
    before the audit START row, redaction, any credential and any upstream contact.
    `identity` is true whenever the request spends a credential the proxy holds. A
    reason, a non-string answer or an exception becomes a recorded,
    provider-shaped `403`. An upload's JSONL lines and form fields are read for it
    before redaction, whether or not the route redacts.
  - `listing_item_session(object_id)` and the batched `listing_item_sessions`: each
    item of a 2xx OpenAI-shaped listing (files, batches, video jobs, stored chat
    completions; OpenAI, Azure and custom prefixes) that the router names is
    restored, from the provider's own bytes, in that existing session (on a vault
    with a durable map, only while the map still records the object there). A
    session that is empty or gone restores nothing; items the router's lookup fails
    for are delivered as the provider sent them (counted); listings never record
    ownership.
  - `sealed(session_id)`: a sealed session is read for rehydration but never
    written. A request that would redact a value into it gets a recorded `403` (a
    string answer is the reason) before any upstream contact, and a realtime
    connection to it is refused.
  - `RoutePlan.proxy_credential`; the vault managers' optional
    `lookup_response_sessions` and `record_object_session` (and the in-memory
    manager's `has_session`); the adapter hooks `redacts_multipart`,
    `lists_objects`/`listing_items` and `object_ids_from_event`/
    `reports_object_ids_once`.
- More stored objects reported to an ownership-tracking router, with their creator:
  stored chat completions (`store: true`, streamed ones too), video create/remix
  jobs, Anthropic Files uploads, completed OpenAI Uploads (not those of purpose
  `batch` or none, whose parts are never read), OpenAI fine-tuning jobs and a job's
  `result_files`, Gemini API files and a finished Gemini batch's output file, Gemini
  batch and Veo operations, Vertex Veo operations and Bedrock async invocations.
  Files a provider tool writes for a request (Anthropic code execution output,
  OpenAI code interpreter container files, streamed or not) are reported as the
  requester's; an id the request itself carries never is. Owner records are bounded
  apart from Responses rows (a `kind` column, 10,000 each); an RDBMS user that may
  not `ALTER` keeps the shared bound, surfaced as `/status`
  `vault.owner_bound_shared`, a `status` posture line and a `doctor` WARN.
- `[users] unrecorded_objects = "refuse" | "allow"`: the config shape of
  llm-redact-pro's unknown-owner policy (restart-only; without the package any
  non-default `[users]` refuses to start).
- `llm_redact_bookkeeping_errors_total{stage}` and `/status`
  `bookkeeping_errors_total`: faults in the proxy's own bookkeeping after the
  upstream answered (`response_id`, `object_ids`, `listing`, `delivery`), vault
  write faults before it (`vault`), and vault staleness checks that could not read
  the database (`vault_check`, contained).
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
- Newly recognized (redact-only, so a credential the proxy holds may reach them,
  and `/recent` and metrics no longer label them pass-through): `GET /v1/models`
  and `/v1/models/{id}` (Anthropic's only with `anthropic-version`), Anthropic batch
  list/poll/cancel/delete, OpenAI deletes of responses, conversations (and items),
  files and videos, `GET /v1/videos/{id}/content`, Gemini `GET /v1beta/models`, and
  Ollama `/api/tags`, `/api/ps`, `/api/version` and `/api/show`. The Gemini API's
  `/v1beta/openai/` surface is redacted and restored like OpenAI's.

### Changed

- File objects (the upload response, the file list, a file's metadata) and batch
  routes are chat routes so their echoes are restored; they get no system note.
- Stored objects are reported to a router that tracks ownership in every session
  mode, static included (Response ids stay unrecorded in static mode).
- The system note's example token is «EMAIL_000», a number the vault never issues.
- `GET` and `HEAD /` are answered locally (the ollama CLI's heartbeat), after the
  request-origin check.
- Faster redaction of short strings: each string up to 1,024 characters runs only
  the detectors that could match it, with identical detections. A 10 MiB body of
  300,000 tiny messages took 32 s of event-loop time; it now takes 1.3 s (and is
  over `max_body_strings`).
- The sqlite vault writes a request's new values in one transaction, committed
  before anything is forwarded; any refusal or fault rolls it all back and a retry
  gets the same numbers. 10,000 new values: 2.5 s to 0.3 s.
- `llm-redact vault verify` notes numbering gaps (token floors create them) instead
  of failing, and fails a row whose token disagrees with its number or that reuses
  a retired number.
- The in-memory vault manager counts and lists only sessions holding mappings, and
  a forgotten session keeps its numbering.
- The `realtime` extra requires websockets 15.0 or newer: older clients send a
  second User-Agent, and 13.0 cannot refuse handshake redirects.
- Under `auth = "identity"`, any query parameter whose name contains
  `authorization` is stripped as a client credential (HTTP and WebSocket).

### Fixed

- A vault that cannot record a request's placeholders (a failed sqlite write or
  COMMIT, an RDBMS driver error, an RDBMS allocation that kept colliding, now
  `RdbmsAllocationError`), or cannot open the session a request resolves to (a new
  session's view reads its rows), refuses the request with a recorded,
  provider-shaped `503` before any upstream contact, counted as
  `bookkeeping_errors{stage="vault"}` and logged by exception type only; a realtime
  frame, or a connection whose session cannot be opened, closes `1011`. It was an
  unrecorded bare `500`.
- A lone UTF-16 surrogate escape in a body, an answer or a stream no longer causes a
  bare `500`, a `502` or a cut stream: every re-serialization goes through one
  serializer that re-escapes it. Unchanged bodies are still forwarded
  byte-identical.
- A codice fiscale or CURP containing a non-ASCII digit crashed the request with an
  unrecorded `500`; it is now detected and redacted.
- A JSON document nested deeper than the proxy can walk no longer causes a bare,
  unrecorded `500` (`'{"messages":' + '[' * 200000`), a `502` or a cut stream.
  Every document the proxy reads may nest at most 128 levels of objects and arrays:
  a deeper request body, uploaded JSONL line or form field, or Bedrock
  `count-tokens` blob is a recorded `400` (a realtime client frame closes `1008`),
  and a deeper answer, SSE event, NDJSON or JSONL line, event-stream payload or
  realtime frame is forwarded exactly as it came, its placeholders left in place. A
  JWT whose header is too deep to parse is redacted (it was a `500`).
- Faults in post-response bookkeeping (response ids, stored objects, the listing
  restore) are contained and counted, and the answer is delivered; a fault restoring
  a buffered answer is a recorded `502`, and a stream the proxy cuts is recorded as
  `502`.
- `llm-redact status`, `doctor`, `run`, the `plugin install` probe and `vault
  rotate-key`'s liveness check dial a wildcard bind at loopback and bracket an IPv6
  literal; an IPv6-bound proxy was unreachable (`status` and `run` crashed with
  `InvalidURL`). `doctor` probes an IPv6 bind's port with an IPv6 socket, and the
  `serve` banner shows the dialed status URL.
- Gemini and Vertex streaming (SSE and the array form) and Live messages restore
  function-call arguments, generated code and grounding exactly as the buffered
  answer does.
- Bedrock base64 media (`source.bytes`) is no longer scanned.
- The multipart parser is linear; a body of many parts no longer freezes the event
  loop.
- Realtime policy closes (block mode, a refused frame) reach the client as `1008`:
  the upstream's mirrored `1000` used to arrive first. Refusals before the upstream
  dial (the access gate's `403`, a disabled or unconfigured provider's `502`, an
  `[audit] required` START failure's `503`) and failed dials are recorded, and close
  reasons are cut to the protocol's 123-byte limit.
- A malformed `Origin` on the reserved endpoints' guard chain is a `403`, not a
  `500`.
- A routing `no_route` `502` is recorded with the configured static session.
- An empty `XDG_DATA_HOME` or `XDG_CONFIG_HOME` counts as unset (it resolved to a
  relative path).
- Azure JSONL file uploads get the per-line system note on chat-shaped lines (parity
  with OpenAI `/v1/files`); Azure's note is otherwise confined to chat completions.
- Vertex express-mode metadata GETs (`/v1/publishers/…`) reach the Vertex upstream,
  and Gemini file downloads (`/download/v1beta/…`) the Gemini upstream.
- The realtime relay keeps the `upstream_base_url` path (APIM and gateway bases).
- Bedrock `count-tokens` decodes, redacts and re-encodes `input.invokeModel.body`;
  a body that cannot be decoded is refused with 400.
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
