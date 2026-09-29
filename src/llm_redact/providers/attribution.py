"""Which provider an UNRECOGNIZED request is addressed to — or none.

A request no adapter recognizes (pass-through) is forwarded verbatim, with
the credential the client attached, to the upstream of the provider it is
attributed to. A guess would hand one provider's credential and unredacted
content to another (the anthropic default this replaced sent an OpenAI key
and prompt to api.anthropic.com), so a request is attributed only POSITIVELY:

1. an explicit path family — ``/v1beta/…`` is the Gemini API's,
   ``/openai/…`` Azure's, ``/api/…`` Ollama's, … — whatever headers or keys
   the request carries;
2. otherwise the markers only one provider's clients send (``MARKERS``):
   ``anthropic-version`` (every Anthropic SDK request carries it), a Google
   API key (``x-goog-api-key``, a ``key=``/``$key=`` query parameter) or any
   other ``x-goog-*`` header, an ``openai-*`` header, the Cohere SDK's
   ``x-fern-sdk-name``. Markers of two providers attribute nothing;
3. otherwise OpenAI's: an ``Authorization: Bearer sk-…`` key that is not
   Anthropic's (``sk-ant-…``), or a path under one of OpenAI's own API
   prefixes (``OPENAI_PREFIXES``).

Anything else is unattributable, and the proxy answers it locally (a
recorded 404) instead of forwarding it anywhere. Prefixes match whole path
segments: ``/v1/embed`` is not under ``/v1/embeddings``, nor
``/v1/organizations`` (Anthropic's Admin API) under ``/v1/organization``
(OpenAI's).
"""

import urllib.parse
from collections.abc import Mapping

CUSTOM_ROUTE_PREFIX = "/custom/"

# Path families that belong to one provider, checked first (an explicit
# family wins over every header or key: express-mode Vertex traffic carries
# a Google API key yet is Vertex's, never the Gemini API host's).
PATH_FAMILIES: tuple[tuple[str, str], ...] = (
    # The Gemini API (its OpenAI-compatible /v1beta/openai/ surface too),
    # its resumable/multipart Files upload and its file downloads.
    ("/v1beta", "gemini"),
    ("/upload/v1beta", "gemini"),
    ("/download/v1beta", "gemini"),
    # Vertex AI (express mode sends a Google API key: still Vertex's).
    ("/v1/projects", "vertex"),
    ("/v1/publishers", "vertex"),
    ("/v1beta1", "vertex"),
    ("/openai", "azure"),
    # bedrock-runtime: the model routes, ApplyGuardrail, async invocations.
    ("/model", "bedrock"),
    ("/guardrail", "bedrock"),
    ("/async-invoke", "bedrock"),
    ("/api", "ollama"),
    ("/v2", "cohere"),
    # Anthropic-only resources (every Anthropic request also carries the
    # anthropic-version marker; these hold even for a client that omits it).
    ("/v1/messages", "anthropic"),
    ("/v1/complete", "anthropic"),
    ("/v1/organizations", "anthropic"),
)

# OpenAI's API resources under the /v1 prefix other providers share. Their
# pass-through siblings (moderations, fine-tuning, uploads, vector stores,
# assistants/threads, the admin API, …) must reach the OpenAI upstream.
OPENAI_PREFIXES: tuple[str, ...] = (
    "/v1/chat",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models",
    "/v1/responses",
    "/v1/conversations",
    "/v1/files",
    "/v1/uploads",
    "/v1/batches",
    "/v1/images",
    "/v1/audio",
    "/v1/videos",
    "/v1/moderations",
    "/v1/fine_tuning",
    "/v1/realtime",
    "/v1/vector_stores",
    "/v1/assistants",
    "/v1/threads",
    "/v1/containers",
    "/v1/evals",
    "/v1/chatkit",
    # The admin API (singular — Anthropic's is /v1/organizations).
    "/v1/organization",
)


def under(path: str, prefix: str) -> bool:
    """Whether ``path`` is ``prefix`` or lies below it, segment-wise."""
    return path == prefix or path.startswith(prefix + "/")


def path_family(path: str) -> str | None:
    """The provider whose explicit path family ``path`` is in (a custom
    provider as ``custom:NAME``), or None."""
    if path.startswith(CUSTOM_ROUTE_PREFIX):
        # Addressed to that upstream by name; an unknown name yields a key
        # with no config entry, which the proxy answers 502.
        return "custom:" + path[len(CUSTOM_ROUTE_PREFIX) :].split("/", 1)[0]
    for prefix, provider in PATH_FAMILIES:
        if under(path, prefix):
            return provider
    return None


def google_key_in_query(query: str) -> bool:
    """Whether the query carries a Google API key (``key=``/``$key=``)."""
    return any(
        urllib.parse.unquote_plus(part.split("=", 1)[0]).lower().lstrip("$") == "key"
        for part in query.split("&")
        if part
    )


def provider_markers(headers: "Mapping[str, str] | None", query: str = "") -> frozenset[str]:
    """The providers whose clients alone send one of the request's headers
    (or, for Google, its ``key=`` query parameter)."""
    found: set[str] = set()
    for name, value in (headers or {}).items():
        lowered = name.lower()
        if lowered == "anthropic-version":
            found.add("anthropic")
        elif lowered.startswith("x-goog-"):
            found.add("gemini")
        elif lowered.startswith("openai-"):
            found.add("openai")
        elif lowered == "x-fern-sdk-name" and value.lower().startswith("cohere"):
            found.add("cohere")
    if google_key_in_query(query):
        found.add("gemini")
    return frozenset(found)


def _openai_bearer(headers: "Mapping[str, str] | None") -> bool:
    """An ``Authorization: Bearer sk-…`` key that is not Anthropic's."""
    for name, value in (headers or {}).items():
        if name.lower() == "authorization":
            scheme, _, token = value.strip().partition(" ")
            token = token.strip()
            if scheme.lower() == "bearer" and token.startswith("sk-"):
                return not token.startswith("sk-ant-")
    return False


def attribute(path: str, headers: "Mapping[str, str] | None", query: str = "") -> str | None:
    """The provider an unrecognized request is addressed to, or None when it
    cannot be attributed positively (see the module docstring)."""
    family = path_family(path)
    if family is not None:
        return family
    markers = provider_markers(headers, query)
    if markers:
        return next(iter(markers)) if len(markers) == 1 else None
    if _openai_bearer(headers) or any(under(path, prefix) for prefix in OPENAI_PREFIXES):
        return "openai"
    return None


def unattributed_reason(headers: "Mapping[str, str] | None", query: str = "") -> str:
    """Why ``attribute`` found no provider, naming marker KINDS only."""
    markers = provider_markers(headers, query)
    if len(markers) > 1:
        return f"it carries markers of more than one provider ({', '.join(sorted(markers))})"
    return "no path family or provider marker names its provider"


__all__ = [
    "CUSTOM_ROUTE_PREFIX",
    "OPENAI_PREFIXES",
    "PATH_FAMILIES",
    "attribute",
    "google_key_in_query",
    "path_family",
    "provider_markers",
    "unattributed_reason",
    "under",
]
