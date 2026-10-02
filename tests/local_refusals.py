"""The local-refusal invariants every test checks (conftest's autouse
``_local_refusals_counted_once``) and the helpers the refusal tests share.

``llm_redact_local_refusals_total{kind,provider}`` counts each response the
proxy generates itself instead of forwarding, ONCE per request, under the
kind its refusal site names. The fixture wraps the app's catch-all handler
and the metrics counter for the whole suite: every HTTP request any test
sends through ``create_app`` is checked to count at most one refusal, and
every refusal counted with a row is checked against the statuses its kind
can answer — so a refusal site that counts twice (or under the wrong kind)
fails whichever test exercises it.
"""

from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
from typing import Any

from llm_redact.metrics import LOCAL_REFUSAL_KINDS

# kind -> the row statuses it may be recorded with (HTTP; a realtime row is
# 101 for a frame refused on an open relay that keeps its upgrade status —
# a block — and 500 for a relay the proxy itself failed). None: counted
# without a row (its path may hold a key).
KIND_STATUSES: dict[str, frozenset[int | None]] = {
    "access_gate": frozenset({403}),
    "audit_unavailable": frozenset({503}),
    "binary_values": frozenset({400}),
    "blocked_value": frozenset({400, 101}),
    "budget": frozenset({402}),
    "credential_protocol": frozenset({403}),
    "delivery_fault": frozenset({502, 500}),
    "disabled_provider": frozenset({502}),
    "identity_path": frozenset({None}),
    "identity_route": frozenset({403}),
    "method_override": frozenset({400}),
    "misaddressed": frozenset({400, 404}),
    "no_route": frozenset({502}),
    "no_upstream": frozenset({502}),
    "object_access": frozenset({403}),
    "override_fault": frozenset({400}),
    "override_raced": frozenset({400}),
    "placeholder_limit": frozenset({400}),
    "realtime_unavailable": frozenset({None}),
    "redirect_refused": frozenset({502}),
    "reload": frozenset({503}),
    "request_origin": frozenset({403}),
    "request_target": frozenset({400, None}),
    "route_unsupported": frozenset({404}),
    "scanned_body": frozenset({400}),
    "sealed_session": frozenset({403}),
    "too_large": frozenset({413}),
    "too_many_strings": frozenset({413}),
    "unattributed": frozenset({404, None}),
    "unchecked_body": frozenset({400}),
    "unredactable": frozenset({400}),
    "unscanned_upload": frozenset({400}),
    "unsupported_encoding": frozenset({415}),
    "upstream_auth": frozenset({502}),
    "upstream_fault": frozenset({502}),
    "vault_fault": frozenset({503}),
    "verbatim_field": frozenset({400}),
}
assert set(KIND_STATUSES) == set(LOCAL_REFUSAL_KINDS)

# The refusals the current request counted (set by the wrapped handler).
REQUEST_REFUSALS: ContextVar[list[str] | None] = ContextVar("test_request_refusals", default=None)


def refused_once(state: Any, kind: str, provider: str) -> None:
    """The proxy counted exactly one local refusal: ``kind`` for
    ``provider`` ("passthrough" when none was attributed)."""
    assert state.metrics.local_refusals == Counter({(kind, provider): 1}), dict(
        state.metrics.local_refusals
    )


def nothing_refused(state: Any) -> None:
    """The proxy generated no response of its own."""
    assert not state.metrics.local_refusals, dict(state.metrics.local_refusals)


def refused_once_scraped(text: str, kind: str, provider: str) -> None:
    """``refused_once`` read from a real server's /__llm-redact/metrics text
    (a test whose app runs in a uvicorn thread)."""
    series = [
        line for line in text.splitlines() if line.startswith("llm_redact_local_refusals_total{")
    ]
    assert series == [
        f'llm_redact_local_refusals_total{{kind="{kind}",provider="{provider}"}} 1'
    ], series
