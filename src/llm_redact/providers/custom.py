"""OpenAI-compatible surfaces served under a path prefix.

Named custom upstreams ([providers.custom.NAME]) serve the FULL OpenAI
adapter surface (chat, embeddings, files/batches hooks, /v1/responses)
under the /custom/NAME/ prefix; the proxy strips the prefix before
forwarding, so tools simply point OPENAI_BASE_URL at
http://127.0.0.1:8787/custom/NAME. Several can run side by side (vLLM + LM
Studio + OpenRouter), each independently disable-able (fail-closed 502 like
every provider). Unmatched subpaths under a configured prefix still pass
through to THAT upstream — the client addressed it explicitly. Realtime WS
custom upstreams are out of scope (documented).

The Gemini API's own OpenAI-compatible surface (``/v1beta/openai/…`` on
generativelanguage.googleapis.com) is the same shape under a fixed prefix,
served by the ``gemini`` provider: its chat, embeddings, files and batches
are redacted and restored like OpenAI's (the raw path is forwarded).
"""

from collections.abc import Iterable
from typing import Any

from llm_redact.providers.attribution import CUSTOM_ROUTE_PREFIX
from llm_redact.providers.base import ProviderAdapter, RouteKind
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.rehydrate import Rehydrator

# The methods an OpenAI endpoint tail is recognized under (matching the
# request itself stays method-exact: see _PrefixedOpenAIMixin.matches).
_ENDPOINT_METHODS = ("POST", "GET", "DELETE")
# OpenAI API resource names. A tail below one of them is a sub-resource of
# it (a container's or vector store's files, a thread's messages), never
# an endpoint under a base path: /containers/c/files/f/content is not the
# Files API's download, and restoring it as one would read a code
# interpreter's output in the wrong session. Base paths never hold one
# (``models`` is left out: Azure AI's base path is /models).
_OPENAI_RESOURCES = frozenset(
    {
        "assistants",
        "audio",
        "batches",
        "chat",
        "chatkit",
        "completions",
        "containers",
        "conversations",
        "embeddings",
        "evals",
        "files",
        "fine_tuning",
        "images",
        "moderations",
        "organization",
        "realtime",
        "responses",
        "threads",
        "uploads",
        "vector_stores",
        "videos",
    }
)


# The deepest endpoint the wrapped OpenAI adapters match has five segments
# after /v1 (/v1/vector_stores/{id}/files/{file_id}/content): a longer tail
# is never an endpoint, except one starting at an OpenAI resource name (the
# conversations route reads everything below it). Tails are tried up to
# this depth — twice the deepest, for headroom — plus the one at the first
# resource name, so the search costs a bounded number of matches whatever
# the path (tests/test_api_coverage.py pins the depth of every route).
MAX_ENDPOINT_SEGMENTS = 10


def custom_prefix(provider_key: str) -> str:
    """Route prefix for a "custom:NAME" provider key."""
    return CUSTOM_ROUTE_PREFIX + provider_key.removeprefix("custom:")


class _PrefixedOpenAIMixin:
    """An OpenAI-family adapter whose paths sit under ``prefix``.

    Every path-sensitive hook delegates with the canonical path so the
    wrapped adapter keeps reasoning in its native /v1/... namespace.
    """

    prefix: str

    def _strip(self, path: str) -> str | None:
        if path.startswith(self.prefix + "/"):
            return path[len(self.prefix) :]
        return None

    def _recognized(self, candidate: str, method: str | None, kind: RouteKind | None) -> bool:
        """Whether ``candidate`` is an endpoint of the wrapped adapter: for
        ``method`` when given (the request's own), else for any method whose
        route is of ``kind`` (a hook that knows the matched kind, not the
        method, lands on the tail the match did)."""
        wrapped = super().matches  # type: ignore[misc]
        for candidate_method in (method,) if method is not None else _ENDPOINT_METHODS:
            found = wrapped(candidate_method, candidate)
            if found is not RouteKind.NONE and (kind is None or found is kind):
                return True
        return False

    def _canonical(
        self, path: str, *, method: str | None = None, kind: RouteKind | None = None
    ) -> str:
        """The inner path in the OpenAI namespace the wrapped adapter
        matches on (exact /v1/...).

        OpenAI-compatible upstreams serve the SAME endpoints under varied
        base paths — Groq /openai/v1, OpenRouter /api/v1, Fireworks
        /inference/v1, but also Gemini /v1beta/openai, GitHub Models
        /inference, Azure AI /models, Zhipu /api/paas/v4, a Cloudflare AI
        Gateway /v1/{account}/{gateway}/openai — and some tool configs put
        /v1 in upstream_base_url so the inner path omits it entirely. The
        endpoint is the path's TAIL: the longest tail, at a segment
        boundary, that is a known OpenAI endpoint (taken as is when it
        already starts /v1/, else under /v1) for the request's method — or,
        from a hook that knows only the matched ``kind``, of that kind — and
        is not nested under an OpenAI resource (``_OPENAI_RESOURCES``), so a
        known endpoint routes under any base path (Azure AI's POST
        /models/embeddings is the embeddings endpoint, not a GET of a model
        named "embeddings"). Without a match the old rule (re-anchor at the
        last /v1/, else prepend /v1) applies, and an unknown tail still falls
        through to NONE (pass-through) via the wrapped exact matcher.

        Only tails of at most ``MAX_ENDPOINT_SEGMENTS`` segments after /v1
        are tried, and the one at the first resource name: no longer tail is
        an endpoint, and trying every tail of a path (each built and matched
        in full) made the search quadratic in the path's length."""
        inner = self._strip(path)
        if inner is None:
            return path
        segments = inner.split("/")
        count = len(segments)
        # Tails are tried longest first up to the one at the first resource
        # name: every later tail is a sub-resource of that resource.
        last = next(
            (index for index in range(1, count) if segments[index] in _OPENAI_RESOURCES),
            count - 1,
        )
        # The tail at `index` has count - index segments (a leading "v1" one
        # of them when it has one).
        first = max(1, count - 1 - MAX_ENDPOINT_SEGMENTS)
        for index in range(first, last + 1) if first <= last else (last,):
            tail = "/" + "/".join(segments[index:])
            candidate = tail if tail.startswith("/v1/") else "/v1" + tail
            if self._recognized(candidate, method, kind):
                return candidate
        marker = inner.rfind("/v1/")
        return inner[marker:] if marker != -1 else "/v1" + inner

    def matches(self, method: str, path: str) -> RouteKind:
        if self._strip(path) is None:
            return RouteKind.NONE
        canonical = self._canonical(path, method=method)
        return super().matches(method, canonical)  # type: ignore[misc,no-any-return]

    def request_model(self, method: str, path: str, parsed: Any) -> str | None:
        """The OpenAI rules on the endpoint the path's tail names — but only
        when the client put nothing between the prefix and that endpoint
        except an optional lone ``/v1``. Any other segment it chose (an
        Azure-shaped ``/openai/deployments/{d}``, a router's
        ``/{provider}/models/{m}``, even a gateway prefix such as Groq's
        ``/openai/v1``) reaches the upstream as sent and may select what it
        runs, which the core cannot know: the model is unknown (None). A base
        path, deployment or router prefix belongs in ``upstream_base_url``."""
        canonical = self._canonical(path, method=method)
        if self._strip(path) not in (canonical, canonical.removeprefix("/v1")):
            return None
        return super().request_model(method, canonical, parsed)  # type: ignore[misc,no-any-return]

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        canonical = self._canonical(path, kind=kind)
        return super().wants_system_note(kind, canonical)  # type: ignore[misc,no-any-return]

    def restores_file_download(self, method: str, path: str) -> bool:
        # The canonical path rehydrate_raw_body restores under.
        canonical = self._canonical(path, kind=RouteKind.CHAT)
        return super().restores_file_download(method, canonical)  # type: ignore[misc,no-any-return]

    def rehydrate_raw_body(self, path: str, raw: bytes, rehydrator: Rehydrator) -> bytes | None:
        # Consulted on CHAT routes only.
        canonical = self._canonical(path, kind=RouteKind.CHAT)
        return super().rehydrate_raw_body(canonical, raw, rehydrator)  # type: ignore[misc,no-any-return]


class _CustomPrefixMixin(_PrefixedOpenAIMixin):
    """A named custom upstream: /custom/NAME/..."""

    def __init__(self, custom_name: str) -> None:
        self.name = f"custom:{custom_name}"
        self.prefix = CUSTOM_ROUTE_PREFIX + custom_name


class CustomOpenAIAdapter(_CustomPrefixMixin, OpenAIAdapter):
    pass


class CustomResponsesAdapter(_CustomPrefixMixin, OpenAIResponsesAdapter):
    pass


# The Gemini API's OpenAI-compatible base (generativelanguage.googleapis.com
# /v1beta/openai/): the OpenAI SDK appends chat/completions, embeddings, …
GEMINI_OPENAI_PREFIX = "/v1beta/openai"


class GeminiOpenAIAdapter(_PrefixedOpenAIMixin, OpenAIAdapter):
    name = "gemini"
    prefix = GEMINI_OPENAI_PREFIX


class GeminiOpenAIResponsesAdapter(_PrefixedOpenAIMixin, OpenAIResponsesAdapter):
    """Only the POSTs — the create, a compaction and the input-token count:
    an answer is restored in the request's own session. A stored response
    read back by id is left alone (pass-through) — the session that created
    it is not one a later read on this prefix is resolved to, and a
    placeholder left in place is safe where a restore in another session
    is not."""

    name = "gemini"
    prefix = GEMINI_OPENAI_PREFIX

    def matches(self, method: str, path: str) -> RouteKind:
        return super().matches(method, path) if method == "POST" else RouteKind.NONE


def build_custom_adapters(provider_keys: "Iterable[str]") -> list[ProviderAdapter]:
    """Adapter instances for every "custom:NAME" key, chat surface first."""
    adapters: list[ProviderAdapter] = []
    for key in sorted(provider_keys):
        if key.startswith("custom:"):
            name = key.removeprefix("custom:")
            adapters.append(CustomOpenAIAdapter(name))
            adapters.append(CustomResponsesAdapter(name))
    return adapters


__all__ = [
    "CUSTOM_ROUTE_PREFIX",
    "GEMINI_OPENAI_PREFIX",
    "CustomOpenAIAdapter",
    "CustomResponsesAdapter",
    "GeminiOpenAIAdapter",
    "GeminiOpenAIResponsesAdapter",
    "build_custom_adapters",
    "custom_prefix",
]
