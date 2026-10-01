# Document extraction: binary uploads read as text

A file uploaded through a Files API (OpenAI, Azure OpenAI, custom
OpenAI-compatible providers, code interpreter containers, Anthropic's and the
Gemini API's Files uploads) is read by its content. Text files are redacted.
A **binary** file — a PDF, a Word, Excel or PowerPoint file, an image —
cannot be rewritten, so without this feature the core forwards it
**unscanned** with the client's own key (`[detection] binary_uploads =
"forward"`, counted in `/status` `unscanned_uploads_total`) or refuses it
(`"refuse"`, and always under a credential the proxy holds).

`[extraction]` reads such a file as **text** first, so the proxy can scan it
with the same detectors, allowlists, modes and deny strings as everything
else. It is part of the free core (no license, no plugin); the extractors
only READ — what happens next is decided by the proxy's redaction core
(`upload_inspection.py`):

| The text read from the file | The upload |
| --- | --- |
| holds a value llm-redact would redact (or a deny string) | refused, **400**, naming the detector TYPES — the value cannot be removed from inside the file; in **convert mode** (below), with a complete reading on a route that takes a text file, the file is REPLACED by its redacted text instead |
| holds a value of a rule in `block` mode | refused, **400** (blocked) |
| holds a value of a rule in `warn` mode | counted as a warning; the value leaves inside the file (warn mode never protects) |
| is clean and the reading is **complete** | sent **byte-identical**, with the client's own key (even under `binary_uploads = "refuse"`); under a credential the proxy holds only with `proxy_credential = true` |
| is clean but the reading is **incomplete**, or nothing could be read, or the extractor failed or timed out | the unscanned-binary rules above, exactly as without `[extraction]` |

**What a clean scan covers.** Only the text the extractors read. The file
itself goes out as sent. A reading is *complete* only when every
text-bearing element of the file was read (below); anything the extractors
cannot read — a scanned page, a photo of a whiteboard, an embedded
spreadsheet — makes the reading incomplete, and the file keeps the rules for
an unscanned binary. Placeholders are never placed inside a file.

Install the `extract` extra for PDFs (`pip install
"llm-redact-proxy[extract]"`: pypdf, imported only by the isolated worker
process; the container image ships it); the OOXML/ODF, markup and RTF
readers are standard library.

## Configuration

```toml
[extraction]
enabled = true
formats = ["pdf", "ooxml", "odf", "html", "rtf"]  # local extractors (the default)
max_file_bytes = 26214400        # larger files are not read (the unscanned rules apply)
timeout_seconds = 30             # per file: the local worker (each service has its own)
request_timeout_seconds = 60     # how long the proxy waits for all files of one request
max_text_chars = 5000000         # per file; beyond it a reading is incomplete
max_inflated_bytes = 134217728   # a document package's parts, inflated, in total
worker_memory_mb = 512           # the worker's address space (Linux; on macOS files over 1/4 of it are not read)
max_workers = 2                  # extraction processes at a time
proxy_credential = false         # see below
convert = false                  # true, or a list of classes: see "Convert mode"

[[extraction.services]]          # optional: for scans (OCR) and other formats
kind = "tika"                    # tika | docling | unstructured | textract | documentai | azure_docintel
url = "http://127.0.0.1:9998"
trusted = false                  # must be true for a service off this machine (and every cloud one)
complete = false                 # true: the operator vouches it reads everything (OCR on)
token_env = "TIKA_TOKEN"         # optional credential, by environment variable NAME
formats = ["pdf", "image"]       # pdf | office | markup | rtf | image | other (default: all;
                                 # textract/documentai: pdf, image; azure_docintel: + office, markup)
timeout_seconds = 30             # this service's call, per file (default 30)
```

The cloud services and their own keys are under "Cloud OCR services" below.

A file is read by the local worker first and then, while its reading is
not complete, by each service asked for its class **one after another**,
all inside the one deadline the proxy gives a request's files
(`request_timeout_seconds`). So `request_timeout_seconds` must cover what
ONE file may take: for every class, the local `timeout_seconds` plus the
`timeout_seconds` of every service asked for that class; a configuration
where it does not (a service the deadline would always cut off before its
own timeout) refuses to start, naming the class and the sum. The defaults
fit one service per class (30 + 30 = 60); with two services for the same
class, raise `request_timeout_seconds` (at most 300) or lower their
timeouts.

That check is per file. The files of one request share the deadline: the
core inspects them four at a time (at most 16), and at most `max_workers`
local workers run at once — a file waiting for one spends its own
`timeout_seconds` waiting. A request carrying more slow files than that can
reach the deadline before its last files are read; those count as not read
(`timeout`) and keep the unscanned-binary rules (refused under `"refuse"`,
sent unscanned under `"forward"`), never sent on a reading that did not
happen. For requests carrying many files that need OCR, raise
`request_timeout_seconds`.

`[extraction]` is a core config section, free on every tier. It is
restart-only (in `RESTART_ONLY_KEYS`: a reload that changes it keeps the
running value and says so) and emitted by `llm-redact config show`; its live
state is `/status` `upload_inspector` (below). The inspector is built through
the `Registry.build_upload_inspector` seam, so a plugin may still replace it
— but an enabled `[extraction]` for which the registered factory builds no
inspector (a plugin that predates the core's section) refuses to start
rather than silently reading nothing.

It refuses to start (a `ConfigError` naming the setting) with `"pdf"` in
`formats` but pypdf missing (`pip install "llm-redact-proxy[extract]"`);
with a service's credential variable unset (named, never echoed); with a
service off loopback — or any cloud service — that is not `trusted`, or
reached over plain `http` (off loopback; a cloud service always needs
`https`), or whose URL carries credentials, a query or a fragment; with a
key file Document AI cannot sign with; with service timeouts the request
deadline cannot cover (above); and with neither local formats nor
services.

### `proxy_credential` (default `false`)

Under a credential the proxy holds — its cloud identity (`[providers.NAME]
auth = "identity"`) or a routed operator key — every client is one
principal upstream, and the proxy vouches only for what it read. The default
keeps refusing binary files there even when their text scans clean: the
scan covers the extracted text only, and an operator should opt in knowing
that. With `true`, a complete, clean reading is sent under the proxy's
credential too; an incomplete one is still refused. With `false` such an
upload is refused whatever a reading says, so its files are read by the
local extractors only (their reading still names the value types a
refusal reports) and never sent to an external service.

## The local extractors

Each file is read by a **fresh worker process** (`python -I -m
llm_redact.extract_worker`): isolated mode, an empty environment (no
credential of the proxy's reaches the process that parses hostile files),
and — set by the worker itself before it reads a byte — an address-space
limit (`worker_memory_mb`; macOS ignores it, so there the proxy never hands
the worker a file larger than a quarter of `worker_memory_mb` — counted
`memory_unenforced`, nothing read), a CPU-time limit, no file
writes, few open files and no child processes. The proxy kills the worker
at `timeout_seconds` and reaps it; its answer is read up to a size bound. A
parser wedged or blown up by a crafted file costs one killed process: the
file counts as not read.

| Format | What is read | Incomplete when |
| --- | --- | --- |
| `pdf` (pypdf) | every page's text layer; what every form XObject, tiling pattern and annotation appearance draws — a form field's and a link's too, so a field whose value holds bare digits but whose appearance shows them formatted (`123-45-6789`) is read as shown; the text marked content and structure elements stand for (`/ActualText`, `/Alt`, `/E`: what copy and paste and screen readers read) — `/ActualText` also IN PLACE of the glyphs it stands for, as poppler and Acrobat's copy read the line; each page's lines as they are SHOWN: where text of different drawings meets on one line — the page's own, a form it draws, a form field's or another annotation's appearance placed by its rectangle — it is also read joined (directly and with a space); annotation strings, rich contents and link targets; form field names and values (text streams too); the document information dictionary and XMP metadata; bookmarks | a page draws an image, or paints anything (outlines, a pattern) yet reads as no text; a form, pattern or appearance paints an inline image, or shows text that reads as nothing (no font to read it by); ANY object of the file (every one listed, reached from a page or not, object streams included) attaches a file — a file-attachment annotation, a PDF 2.0 associated file, the embedded-files list, a portfolio — or carries rich media, a movie, a sound, 3D or a screen annotation, an image or PostScript XObject (from a pattern, a glyph or an appearance too), a font whose text pypdf cannot map (no ToUnicode map and a Type3 or composite font, an encoding or glyph name pypdf does not know, or no encoding outside the standard 14 fonts), or a Type3 glyph that shows text or an inline image; a document-level script, or a JavaScript action in a file whose form fields the viewer redraws (`NeedAppearances`, or a field without an appearance: a format script may change what it shows); an XFA form; a character could not be mapped; it cannot be decrypted with an empty password |
| `ooxml` (docx, xlsx, pptx, …) and `odf` (odt, ods, odp, …) | the entry names and the zip's comments; every XML part's text and attribute values (so sheet names, formulas, comments, hyperlink targets, document properties), read three ways: joined within paragraphs — a value Word splits across runs — as the line is SHOWN — what a renderer keeps out of the line (a tracked deletion or move, a field's instruction — everything between the field's begin and its separator, whatever element holds it — a hidden run, ruby and phonetic text, a text box or drawing, an OpenDocument inline comment or note) is read on its own and the text around it joined, and a paragraph whose mark is deleted or hidden joins the next — and element by element; tabs, spaces, Word's non-breaking hyphens and symbols (`w:sym`, a symbol font's code as its character) read as the characters they show; a spreadsheet number shown through a format that joins its digits with literal text (Excel's Special formats: `000-00-0000`, `(000) 000-0000`, `00000-0000`, or text such as `"123-45-"0000`) also as the format shows it; a date or time shown with more than four digits also as its format shows it, in the workbook's 1900 or 1904 date system (so a format such as `hhm-ss-yyyy` or `"123-45-"yyyy` over a date is read as the `123-45-6789` it shows) | the package holds anything else: an image, an embedded object or file, fonts, macros, an encrypted entry (the application's own thumbnail, a rendering of the text, is not counted); an XML part inlines base64 data (`w:binData`, `office:binary-data`); a style or the document defaults hide text (Word's `vanish`, OpenDocument's `text:display`), since which runs it hides is not resolved; a field is never closed or its boundaries are out of order; a symbol's character cannot be read; a number format joins digits with literal text in a way not rendered (a text format such as `"123-45-"@`, fractions, dates mixed with digit placeholders, an era or another calendar or digit script), or — like a date format that is not written as dates are (a number twice, two numbers with no text between them, digits in its text) — sits in a conditional format or a chart, where which values it shows is not resolved; the workbook holds a second styles part (a cell's style could find another format); the workbook names its date system only after a date was shown; a date cell holds no date |
| `html` (markup that is not UTF-8 text) | the decoded source itself — tags, attributes, comments, scripts — its text with character references resolved, that text as a browser shows it (without scripts, styles and comments, so a value they split is read whole), and its raw bytes as Latin-1 (every ASCII byte where it stands, whatever the declared charset made of the bytes around it) | the charset was guessed rather than declared (it is taken only from an XML declaration at the very start or a real `<meta charset>` / `<meta http-equiv="Content-Type">` tag within the first 1024 bytes — never from a mention in a comment or a script), or declared as one that does not decode ASCII to itself (UTF-16/32, EBCDIC, UTF-7: the page was written in ASCII), or another charset mentioned in the first 2048 bytes disagrees with the declaration (the page is then still decoded as declared), or the page inlines an image, any base64 `data:` URI or an XML binary-data element |
| `rtf` (best effort) | the source (every ASCII byte), its decoded text — `\'hh` escapes and 8-bit text in the code page of the current font (`\fcharset`, `\cpg`) or the document's (`\ansicpg`), consecutive bytes decoded together (a double-byte character is two escapes); `\uN` with the `\ucN` fallback characters after it skipped, as every reader does; the non-breaking hyphen `\_` and space `\~` as a hyphen and a space — and that text as SHOWN: without hidden (`\v`) or deleted text, field instructions (marked `\*` or not), notes, headers, shapes, drawing objects, index entries and the document's tables, so the text around them joins | a picture, an embedded object or binary data; bytes the code page does not decode, or a character standing for one that could not be mapped (U+FFFD) |

**A file is judged as the format its bytes show.** A provider may open it
with another reader — picked from the declared media type or the file
name — so a reading does not vouch for a file (it is still scanned, but
not complete) when the part's declared type names a different format
(`application/pdf` over a Word file, a Word type over a PDF, `text/plain`
or `application/zip` over either; `application/octet-stream` and no type
claim nothing), when the file name's extension does (`report.docx` over a
PDF, `notes.txt` over either; any extension outside the known ones counts
as another format; `.bin`, `.dat`, `.tmp` and no extension claim nothing;
a part whose `filename` and `filename*` name different extensions counts
as another format too — the provider may go by either), or when the file also holds another format where that
format's readers look for it — whoever read it (a service configured
`complete = true` opens the file by its leading bytes too, so its
reading does not vouch for a polyglot either) and whatever its class
(an image or a legacy Office file carrying a zip or a PDF included): a PDF's header or trailer (`%PDF-`,
`startxref`, `%%EOF`) anywhere in a file that is not a PDF (pypdf, poppler,
pdf.js and MuPDF find a PDF by its trailer from the end, whatever comes
before or after it, even without a header), a zip end record near the end of one that is not a
zip, or bytes after a zip's end record, a PDF's last `%%EOF` or an RTF
document's outer group (a PDF with a Word file appended is read by
python-docx or Tika as the Word file). The core hands the extractors the
declared type and the file name's extension (`UploadPart.extension`,
lower-cased, only when it matches `[a-z0-9]{1,10}`), never the name
itself.

Document packages are read with zip-bomb limits: at most 10,000 entries,
`max_inflated_bytes` inflated in total, and no entry inflating beyond 200
times its compressed size; a document type declaration (and with it every
entity) is refused. Other archives, images, legacy Office (OLE2) files and
anything else are not read locally — only by a service.

## External services

For scans that need OCR and formats without a good local reader, the
extractors can ask services, in order, for files the local extractors did
not read completely (and only for the classes in each service's
`formats`). The first reading that counts as complete wins; every text read
along the way is scanned.

| `kind` | Request | What is read |
| --- | --- | --- |
| `tika` | `PUT {url}/rmeta/text` (the recursive endpoint), `Authorization: Bearer` from `token_env` | the text and metadata of the document and of every embedded document; any `X-TIKA:EXCEPTION` makes the reading incomplete |
| `docling` | `POST {url}/v1/convert/file` (docling-serve, text output), `X-Api-Key` from `token_env` | `document.text_content`; a status other than `success` or any error makes it incomplete |
| `unstructured` | `POST {url}/general/v0/general`, `unstructured-api-key` from `token_env` | every element's text and string metadata (link targets, HTML renderings) |
| `textract` | AWS Textract `DetectDocumentText`, SigV4-signed | every `LINE` block's text (below) |
| `documentai` | Google Document AI `:process` | `document.text`; a `document.error` makes it incomplete |
| `azure_docintel` | Azure AI Document Intelligence `:analyze`, then its operation polled | `analyzeResult.content`; any `warnings` make it incomplete |

A service **sees every file it is asked to read** — never one of an upload
spending a credential the proxy holds while `proxy_credential = false` (it
is refused whatever the service answers). One that is not on
loopback must be declared `trusted = true` and reached over `https`; doctor
flags it. A service's reading counts as **complete** only with `complete =
true` — the operator vouching that it reads every text-bearing element
(OCR on, embedded documents followed) — and only when it reports no error.
Services are called with bounded time and answer size, no redirects, and no
proxy or netrc settings from the environment; a credential is read from the
environment variable `token_env` names at each call and never logged. Any
failure — an error status, a timeout, a malformed or oversized answer, a
missing credential — is a reading that did not happen. The file name sent to
a service is derived from the file's bytes (`upload.pdf`, `upload.docx`, …),
never the client's.

### Cloud OCR services

A cloud service sends every file it is asked to read to that cloud
provider. Each one therefore requires `trusted = true` (always, not only off
loopback) and `https`; its `url` is the service's origin only (no path).
Like the self-hosted services, it is never asked for a file of an upload
spending a credential the proxy holds while `proxy_credential = false`, its
call is bounded by its own `timeout_seconds` inside the request's deadline,
redirects are never followed, and any failure — an error status, a
malformed answer, a missing credential, the deadline — is a reading that did
not happen. The credential comes only from environment variables the config
NAMES (or, for Document AI, a key file it names); none is ever in the
config, a log line, `/status` or `doctor` output.

```toml
[[extraction.services]]
kind = "textract"                # AWS Textract DetectDocumentText
region = "eu-west-1"             # required: the endpoint textract.REGION.amazonaws.com and the SigV4 scope
trusted = true
complete = true                  # Textract reads the whole image/page (OCR)
# url = "https://vpce-….textract.eu-west-1.vpce.amazonaws.com"  # optional (a VPC endpoint)
# access_key_env = "AWS_ACCESS_KEY_ID"          # the variables' NAMES (these are the defaults);
# secret_key_env = "AWS_SECRET_ACCESS_KEY"      # the session token is sent when set
# session_token_env = "AWS_SESSION_TOKEN"

[[extraction.services]]
kind = "documentai"              # Google Document AI (an OCR or layout processor)
processor = "projects/PROJECT/locations/eu/processors/PROCESSOR_ID"  # endpoint: eu-documentai.googleapis.com
trusted = true
credentials_file = "/etc/llm-redact/docai-sa.json"  # a service-account key (needs the crypto extra) …
# token_env = "DOCAI_ACCESS_TOKEN"                  # … or an OAuth access token, by variable NAME

[[extraction.services]]
kind = "azure_docintel"          # Azure AI Document Intelligence
url = "https://NAME.cognitiveservices.azure.com"
token_env = "DOCINTEL_KEY"       # the resource key, sent as Ocp-Apim-Subscription-Key
model = "prebuilt-read"          # the default
trusted = true
```

- **Textract** is called synchronously: it reads an image (PNG, JPEG, TIFF)
  or a ONE-page PDF. A longer PDF is refused by Textract and so counts as
  not read — the asynchronous API needs the file in S3, which this proxy
  does not do. The request is signed with SigV4 (the core's own signer,
  pinned by the AWS documentation vector); the access key pair is read
  from the named variables at each call.
- **Document AI** gets the file's bytes with a media type derived from them
  (`application/pdf`, `image/png`, …), never the client's claim. With
  `credentials_file`, the proxy signs a JWT with the service account's key
  (RS256, the `crypto` extra) and exchanges it at
  `https://oauth2.googleapis.com/token` — the only token endpoint a key file
  may name — caching the access token until a minute before it expires;
  the key file is read once at startup. With `token_env`, the variable
  holds an access token as is: a process cannot see a variable changed
  after it started, and Google access tokens expire (typically within an
  hour), so this form suits a supervisor that restarts the proxy with a
  fresh token; prefer `credentials_file` for a long-running proxy.
- **Document Intelligence** is asynchronous: `POST
  {url}/documentintelligence/documentModels/{model}:analyze?api-version=2024-11-30`
  answers 202 with an `Operation-Location`, which is polled (each wait its
  `Retry-After`, between 0.5 and 5 seconds) until the analysis succeeds —
  all within the service's `timeout_seconds`. The operation must be on the
  service's own `https` origin; the key is never sent anywhere else.

Whether a cloud reading counts as complete is still the operator's
declaration (`complete = true`), as for the self-hosted services — and,
for a cloud service only, it must also cover EVERY page of the file. A
cloud OCR service analyzes up to its tier's page limit and reports nothing
about the pages past it: Azure Document Intelligence's free (F0) tier
analyzes only the first two pages of a PDF or TIFF, its standard (S0) tier
at most 2,000; Document AI and Textract have their own limits. So the page
count the service answers with (`analyzeResult.pages`, `document.pages`,
Textract's `DocumentMetadata.Pages`) must equal the file's own, and the
reading is incomplete (counted `pages_unverified` in
`readings_total`) when it does not or when the file's count is unknown.
The file's count is known for a PDF the local PDF extractor opened (its
page tree as pypdf reads it — keep `pdf` in `formats`), and is one for a
PNG that is not animated, a JPEG or a BMP; a TIFF, GIF, WebP or HEIF image
(which may hold several), an Office file, markup, or a PDF the local
extractor did not open never counts as completely read by a cloud service.

## Convert mode

`convert = true` (every class) or a list of classes (`convert = ["pdf",
"office"]`; the classes of a service's `formats`) is OPT-IN. With it, an
upload whose binary file holds values to redact is not refused: the file is
REPLACED by its extracted text, redacted exactly as a text upload is —
placeholders issued, one text against `max_body_strings` — and sent as a
UTF-8 `text/plain` part whose file name ends in `.txt` (`report.pdf` →
`report.txt`, the name redacted too). A later download of that file is
restored like any text upload's (the proxy remembers the bytes it sent).
It applies only when ALL of these hold:

- the reading is **complete** — the same bar as a clean file (a partial
  reading is never passed off as the document) — and of a class in
  `convert`;
- a rule in `block` mode found nothing (a block still refuses);
- the route's provider takes a text file for this upload: OpenAI, Azure
  OpenAI and custom providers' Files uploads with purpose `assistants`
  (file search, code interpreter) or `user_data` (file inputs — whether a
  model accepts a text file there is the provider's to decide), code
  interpreter container files, and Anthropic's Files API. Never the media
  routes (`/images/edits`, `/videos`: the part IS the image), a `batch`,
  `fine-tune`, `vision` or `evals` upload, or the Gemini API's upload (its
  metadata part declares the file's media type, which the proxy does not
  rewrite) — there the upload is refused as without convert mode;
- under a credential the proxy holds, only with `proxy_credential = true`
  (as for clean files).

**What the model sees is text, not the file.** The text is the extractor's
DISPLAY reading — each PDF page's text layer and filled-in form fields, the
shown text of a document's content parts, markup without its tags, scripts
and styles, RTF as shown, or a service's plain text — so layout, tables,
images, annotations, formatting and anything else the text does not carry
are gone, and so is everything the file held beyond its text. Spreadsheets
(`xl/workbook.xml`) have no faithful display reading (cells cite their
strings by number) and are never converted — their upload is refused as
without convert mode. The scan and every refusal are still decided on the
full reading; the converted text is redacted on its own. A clean complete
file is still sent as the original, byte-identical.

## Bounds the core applies

The core waits at most `request_timeout_seconds` (capped at 300) for all the
binary files of one request, inspects at most 16 of them, four at a time,
hands over none larger than `max_file_bytes`, and scans at most
`max_body_bytes` characters of extracted text per request (each text counts
as one string against `max_body_strings`). Whatever is still running at the
deadline is cancelled (the worker killed) and counts as not read (`timeout`);
so does an inspection that fails or answers something else (`error`): the
file keeps the unscanned-binary rules, never goes out on a reading that did
not happen.

## Surfaces

- `/status`: `inspected_uploads_total` (per provider and outcome: `clean` —
  sent after a clean scan — `clean_refused` — read clean, but the upload was
  refused for another part or rule — `converted` / `converted_refused` —
  replaced by its redacted text (convert mode), and whether the upload was
  then sent — `detected`, `blocked`, `incomplete`, `not_inspected`,
  `timeout`, `error`) and `upload_inspector` — the core's bounds and the
  extractors' formats, services (kind, host, trusted, complete), convert
  classes, worker counts and `readings_total` per extractor.
- Prometheus: `llm_redact_inspected_uploads_total{provider,outcome}`.
- `llm-redact status`: a posture line when files went out after a clean
  scan of their extracted text, and one when files were converted.
- `llm-redact doctor`: `extraction` rows, offline and value-free — what keeps
  it from starting (the `extract` extra, each credential variable by name, a
  key file), each service and whether it sees files off this machine (a
  WARN), convert mode (a WARN), and what a clean scan lets through. When a
  plugin replaces the upload inspector, doctor builds it as `serve` would: one
  that builds none for the enabled section (an llm-redact-pro older than the
  core's extraction) is a FAIL, as the proxy refuses to start with it.
- Logs: per request, the count of binary parts and their outcomes; the
  extractor, a service's host and an exception's TYPE on a failure — never a
  file name, content, a value found, a token or a service's answer.

## Limits (deliberate)

- The file is never rewritten: a value found refuses the upload — unless
  convert mode replaces it with its redacted text (below). Otherwise send the
  document's text instead (it is redacted), remove the value, allowlist it,
  or use `warn` mode where forwarding it is acceptable.
- Completeness is judged by what the reader can see. A page drawn only as
  vector outlines is incomplete, but outlines drawn beside real text on the
  same page are not told apart from a drawing (a logo, a table rule), and a
  font whose ToUnicode map lies is taken at its word; white-on-white text is
  read like any other. A PDF drawing any image, anywhere, is incomplete
  whether or not the image holds text. Text in PDF objects the document no
  longer references (an earlier revision kept by an incremental save) is
  not read — a PDF reader does not show it either.
- Legacy Office files (`.doc`, `.xls`, `.ppt`) and images are read only by
  a service.
- Textract is synchronous only (images and one-page PDFs); Document AI's
  `token_env` form cannot refresh its token (use `credentials_file`).
- A cloud OCR reading vouches for a file only when it covers every page and
  the file's page count is known (above): multi-page TIFFs, Office files and
  PDFs the local extractor could not open are never cleared by one.
