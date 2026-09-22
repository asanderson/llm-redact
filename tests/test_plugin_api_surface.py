"""The ``plugin_api`` seam, snapshotted: every Protocol member and dataclass
``__init__`` signature, the Protocol data attributes, and ``__all__``.

The paid ``llm-redact-pro`` package binds to exactly these names and shapes
(FACTS: pro binds only to ``llm_redact.plugin_api`` and the public config
dataclasses). A change here is a deliberate edit of this snapshot — and a pro
floor bump — never an incidental one. The snapshot is LITERAL text
(``inspect.signature`` renders the ``from __future__ import annotations``
strings verbatim), so a renamed parameter, a widened type, a dropped default
or a new member all fail here, keyless and pro-free.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from llm_redact import plugin_api

# Protocol name -> (data attribute names in declaration order, {method: signature}).
PROTOCOLS: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    "Telemetry": (
        (),
        {
            "record": (
                "(self, row: 'dict[str, Any]', duration_seconds: 'float', *,"
                " traceparent: 'str | None' = None) -> 'None'"
            ),
            "shutdown": "(self) -> 'None'",
        },
    ),
    "VaultCipher": (
        (),
        {
            "mac": "(self, session_id: 'str', detector_type: 'str', original: 'str') -> 'str'",
            "encrypt": "(self, original: 'str') -> 'bytes'",
            "decrypt": "(self, token: 'bytes') -> 'str'",
            "key_check": "(self) -> 'str'",
        },
    ),
    "SessionRouter": (
        ("mode",),
        {
            "resolve": (
                "(self, adapter_name: 'str | None', method: 'str', path: 'str', body: 'Any')"
                " -> 'str'"
            ),
            "record_response_id": "(self, response_id: 'str', session_id: 'str') -> 'None'",
        },
    ),
    "RouteDelivery": (
        ("upstream", "rule", "headers"),
        {
            "observe_event": "(self, event: 'SSEEvent') -> 'SSEEvent'",
            "observe_line": "(self, line: 'bytes') -> 'bytes'",
            "wants_payload": "(self, kind: 'RouteKind') -> 'bool'",
            "observe_payload": "(self, payload: 'Any', kind: 'RouteKind') -> 'bool'",
            "mark_failed": "(self, status_class: 'str') -> 'None'",
            "row": "(self) -> 'dict[str, Any]'",
            "finish": "(self, status: 'int | None') -> 'dict[str, Any]'",
        },
    ),
    "RoutePlan": (
        ("inject_system_note", "deadline"),
        {
            "local_refusal": "(self) -> 'RouteRefusal | None'",
            "begin": (
                "(self, outbound: 'bytes', outbound_obj: 'Mapping[str, Any] | None',"
                " forward_headers: 'Sequence[tuple[str, str]]') -> 'HopRequest | RouteRefusal'"
            ),
            "decide": "(self, result: 'HopResult') -> 'HopDecision'",
            "wait": "(self, seconds: 'float') -> 'None'",
            "delivery": "(self) -> 'RouteDelivery'",
        },
    ),
    "Router": (
        (),
        {
            "local_answer": (
                "(self, method: 'str', path: 'str', headers: 'Mapping[str, str]')"
                " -> 'LocalAnswer | None'"
            ),
            "plan": "(self, inbound: 'RouteInbound') -> 'RoutePlan | RouteRefusal | None'",
            "status": "(self) -> 'dict[str, Any]'",
            "validate": "(self, config: 'Config') -> 'None'",
            "reconfigure": "(self, config: 'Config') -> 'None'",
            "close": "(self) -> 'None'",
        },
    ),
}

# Frozen dataclass name -> its generated __init__ signature.
DATACLASSES: dict[str, str] = {
    "RouteInbound": (
        "(adapter_name: 'str | None', provider_name: 'str', method: 'str', path: 'str',"
        " raw_path: 'str', query: 'str', headers: 'Mapping[str, str]', model: 'str | None')"
        " -> None"
    ),
    "RouteRefusal": (
        "(status: 'int', message: 'str', row: 'Mapping[str, Any]',"
        " headers: 'Mapping[str, str] | None' = None) -> None"
    ),
    "HopRequest": (
        "(upstream: 'str', url: 'str', headers: 'Sequence[tuple[str, str]]', body: 'bytes',"
        " reissued_from: 'str | None' = None, unavailable: 'str | None' = None) -> None"
    ),
    "HopResult": (
        "(upstream: 'str', status: 'int | None', headers: 'Mapping[str, str]',"
        " fault: 'str | None') -> None"
    ),
    "HopDecision": "(next: 'HopRequest | None', wait_seconds: 'float' = 0.0) -> None",
    "LocalAnswer": (
        "(status: 'int', body: 'Mapping[str, Any]', provider: 'str', reason: 'str') -> None"
    ),
}

ALL: tuple[str, ...] = (
    "HopDecision",
    "HopRequest",
    "HopResult",
    "LocalAnswer",
    "MAX_RESPONSE_ROWS",
    "RESPONSE_PRUNE_EVERY",
    "RouteDelivery",
    "RouteInbound",
    "RouteKind",
    "RoutePlan",
    "RouteRefusal",
    "Router",
    "SSEEvent",
    "SessionRouter",
    "Telemetry",
    "Vault",
    "VaultCipher",
    "VaultKeyError",
    "VaultManager",
)


def _public_methods(cls: type) -> set[str]:
    return {name for name, value in vars(cls).items() if callable(value) and name[0] != "_"}


@pytest.mark.parametrize("name", sorted(PROTOCOLS))
def test_protocol_members_are_snapshotted(name: str) -> None:
    cls = getattr(plugin_api, name)
    attrs, methods = PROTOCOLS[name]
    assert getattr(cls, "_is_protocol", False), f"{name} is not a Protocol"
    # Data attributes: exactly these, in this order (get_annotations never
    # falls through to a base class the way a bare __annotations__ can).
    assert tuple(inspect.get_annotations(cls)) == attrs
    # Methods: exactly these — a member added or dropped fails before any
    # signature is compared.
    assert _public_methods(cls) == set(methods)
    for method, signature in methods.items():
        assert str(inspect.signature(getattr(cls, method))) == signature, f"{name}.{method}"


@pytest.mark.parametrize("name", sorted(DATACLASSES))
def test_dataclass_signatures_are_snapshotted(name: str) -> None:
    cls = getattr(plugin_api, name)
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen  # values, never mutable handles
    assert str(inspect.signature(cls)) == DATACLASSES[name]


def test_all_is_snapshotted_and_resolves() -> None:
    assert tuple(plugin_api.__all__) == ALL
    assert list(ALL) == sorted(ALL)  # kept sorted: additions land deliberately
    for name in ALL:
        getattr(plugin_api, name)
    # Every snapshotted Protocol and dataclass is exported; the two
    # re-exports pro reaches through this module instead of core internals
    # are the real classes.
    assert set(PROTOCOLS) | set(DATACLASSES) <= set(ALL)
    from llm_redact.providers.base import RouteKind
    from llm_redact.sse import SSEEvent

    assert plugin_api.RouteKind is RouteKind
    assert plugin_api.SSEEvent is SSEEvent
