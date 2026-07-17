"""Named custom OpenAI-compatible upstreams ([providers.custom.NAME]).

Each custom provider serves the FULL OpenAI adapter surface (chat,
embeddings, files/batches hooks, /v1/responses) under the /custom/NAME/
prefix; the proxy strips the prefix before forwarding, so tools simply
point OPENAI_BASE_URL at http://127.0.0.1:8787/custom/NAME. Several can
run side by side (vLLM + LM Studio + OpenRouter), each independently
disable-able (fail-closed 502 like every provider). Unmatched subpaths
under a configured prefix still pass through to THAT upstream — the
client addressed it explicitly. Realtime WS custom upstreams are out of
scope (documented).
"""

from collections.abc import Iterable

from llm_redact.providers.base import ProviderAdapter, RouteKind
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.rehydrate import Rehydrator

CUSTOM_ROUTE_PREFIX = "/custom/"


def custom_prefix(provider_key: str) -> str:
    """Route prefix for a "custom:NAME" provider key."""
    return CUSTOM_ROUTE_PREFIX + provider_key.removeprefix("custom:")


class _CustomPrefixMixin:
    """Strips the /custom/NAME prefix around an OpenAI-family adapter.

    Every path-sensitive hook delegates with the stripped path so the
    wrapped adapter keeps reasoning in its native /v1/... namespace.
    """

    def __init__(self, custom_name: str) -> None:
        self.name = f"custom:{custom_name}"
        self.prefix = CUSTOM_ROUTE_PREFIX + custom_name

    def _strip(self, path: str) -> str | None:
        if path.startswith(self.prefix + "/"):
            return path[len(self.prefix) :]
        return None

    def _canonical(self, path: str) -> str:
        """Normalize a custom-provider inner path to the OpenAI-family
        namespace the wrapped adapter matches on (exact /v1/...).

        OpenAI-compatible upstreams serve the SAME endpoints under varied base
        paths — Groq /openai/v1, OpenRouter /api/v1, Fireworks /inference/v1 —
        and some tool configs put /v1 in upstream_base_url so the inner path
        omits it entirely. Without this, a misplaced base path matched NOTHING
        and the request was silently forwarded UNREDACTED. Re-anchor at the
        last /v1/ if present; otherwise assume the tail is already the endpoint
        and prepend /v1. Custom providers are opted-in OpenAI-compatible, so a
        path ending in a known endpoint is genuinely ours; unknown tails still
        fall through to NONE (pass-through) via the wrapped exact matcher."""
        inner = self._strip(path)
        if inner is None:
            return path
        marker = "/v1/"
        index = inner.rfind(marker)
        if index != -1:
            return inner[index:]
        return "/v1" + inner if inner.startswith("/") else "/v1/" + inner

    def matches(self, method: str, path: str) -> RouteKind:
        if self._strip(path) is None:
            return RouteKind.NONE
        return super().matches(method, self._canonical(path))  # type: ignore[misc,no-any-return]

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        return super().wants_system_note(kind, self._canonical(path))  # type: ignore[misc,no-any-return]

    def rehydrate_raw_body(self, path: str, raw: bytes, rehydrator: Rehydrator) -> bytes | None:
        canonical = self._canonical(path)
        return super().rehydrate_raw_body(canonical, raw, rehydrator)  # type: ignore[misc,no-any-return]


class CustomOpenAIAdapter(_CustomPrefixMixin, OpenAIAdapter):
    pass


class CustomResponsesAdapter(_CustomPrefixMixin, OpenAIResponsesAdapter):
    pass


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
    "CustomOpenAIAdapter",
    "CustomResponsesAdapter",
    "build_custom_adapters",
    "custom_prefix",
]
