"""Scripted fakes satisfying the ``plugin_api`` routing contract.

The routing LAYER (rule selection, credentials, chains, cooldowns, budgets,
prices, model rewrite) lives in llm-redact-pro; the core keeps only the
policy-free DRIVER of a routed request. These fakes let the Free suite
execute every arm of that driver — refusals, hops, waits, re-issues, the
delivery hooks, reload and the editor dry-run — without a license key and
without the pro package, registered through the plugin registry exactly as
tests/test_audit_required.py registers its fake audit log.

``FakeRouter(scripts)`` plans from the inbound ``x-fake-route`` header:
absent → ``None`` (the legacy path), ``"refuse"`` → a 502 no_route
``RouteRefusal``, any other value → ``FakePlan(scripts[value])``. A
``FakePlan`` consumes its actions in order (``Refuse404`` from
``local_refusal``, ``Refuse402``/``Hop`` from ``begin``, ``Hop``/``Stop``
from ``decide``) and records everything the core hands it; ``FakeDelivery``
records the hooks and optionally mutates what it observes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

import llm_redact.registry as registry_mod
from llm_redact.config import Config, RoutingConfig, UpstreamConfig
from llm_redact.plugin_api import (
    HopDecision,
    HopRequest,
    HopResult,
    LocalAnswer,
    RouteInbound,
    RouteKind,
    RouteRefusal,
    SSEEvent,
)
from llm_redact.registry import Registry

ROUTE_HEADER = "x-fake-route"


# --- scripted actions ---------------------------------------------------------


@dataclass(frozen=True)
class Refuse404:
    """Returned by ``local_refusal`` (the count_tokens gate)."""

    message: str = "fake: upstream a does not implement count_tokens"


@dataclass(frozen=True)
class Refuse402:
    """Returned by ``begin`` (the budget gate); ``headers`` are stamped on the reply."""

    headers: Mapping[str, str] | None = None
    message: str = "fake: upstream a budget exhausted for this period"


@dataclass(frozen=True)
class Hop:
    """A hop the plan mints (from ``begin`` or as ``decide``'s ``next``)."""

    upstream: str
    url: str
    wait: float = 0.0
    reissued_from: str | None = None
    unavailable: str | None = None
    body: bytes | None = None


@dataclass(frozen=True)
class Stop:
    """``decide`` → ``HopDecision(None)``: deliver what is in hand."""


Action = Refuse404 | Refuse402 | Hop | Stop


def route_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "rule": "r",
        "upstream": "a",
        "hops": 1,
        "auth": "none",
        "class": "ok",
        "reissue": "no",
    }
    row.update(overrides)
    return row


# --- the delivery -------------------------------------------------------------


class FakeDelivery:
    """Records every delivery hook; ``mutate=True`` marks what it observes so a
    test can see the hook's output reach the client."""

    def __init__(
        self,
        upstream: str,
        rule: str | None,
        headers: Mapping[str, str],
        *,
        hops: int = 1,
        wants_payload: bool = True,
        mutate: bool = False,
    ) -> None:
        self.upstream = upstream
        self.rule = rule
        self.headers = dict(headers)
        self.hops = hops
        self.mutate = mutate
        self._wants_payload = wants_payload
        self.status_class = "ok"
        self.events: list[SSEEvent] = []
        self.lines: list[bytes] = []
        self.wants_payload_calls: list[RouteKind] = []
        self.payloads: list[tuple[Any, RouteKind]] = []
        self.failed: list[str] = []
        self.finished: list[int | None] = []

    def observe_event(self, event: SSEEvent) -> SSEEvent:
        self.events.append(event)
        if self.mutate and event.data:
            try:
                payload = json.loads(event.data)
            except ValueError:
                return event
            if isinstance(payload, dict):
                payload["observed"] = True
                event.data = json.dumps(payload, ensure_ascii=False)
        return event

    def observe_line(self, line: bytes) -> bytes:
        self.lines.append(line)
        if not self.mutate:
            return line
        try:
            payload = json.loads(line)
        except ValueError:
            return line
        if not isinstance(payload, dict):
            return line
        payload["observed"] = True
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def wants_payload(self, kind: RouteKind) -> bool:
        self.wants_payload_calls.append(kind)
        return self._wants_payload

    def observe_payload(self, payload: Any, kind: RouteKind) -> bool:
        self.payloads.append((payload, kind))
        if isinstance(payload, dict):
            payload["observed"] = True
        return self.mutate

    def mark_failed(self, status_class: str) -> None:
        self.failed.append(status_class)
        self.status_class = status_class

    def row(self) -> dict[str, Any]:
        return route_row(
            rule=self.rule, upstream=self.upstream, hops=self.hops, **{"class": self.status_class}
        )

    def finish(self, status: int | None) -> dict[str, Any]:
        self.finished.append(status)
        return self.row()


# --- the plan -------------------------------------------------------------------


class FakePlan:
    def __init__(
        self,
        actions: Sequence[Action],
        *,
        inject_system_note: bool = True,
        rule: str | None = "r",
        delivery_headers: Mapping[str, str] | None = None,
        wants_payload: bool = True,
        mutate: bool = False,
        deadline_seconds: float = 600.0,
    ) -> None:
        self.actions = list(actions)
        self.inject_system_note = inject_system_note
        self.deadline = time.monotonic() + deadline_seconds
        self.rule = rule
        self.delivery_headers = dict(delivery_headers or {})
        self.wants_payload_flag = wants_payload
        self.mutate = mutate
        self.begun: list[tuple[bytes, Mapping[str, Any] | None, Sequence[tuple[str, str]]]] = []
        self.results: list[HopResult] = []
        self.waits: list[float] = []
        self.hops_issued: list[HopRequest] = []
        self.delivered: FakeDelivery | None = None

    def _hop(self, action: Hop) -> HopRequest:
        outbound, _obj, forward_headers = self.begun[-1]
        hop = HopRequest(
            upstream=action.upstream,
            url=action.url,
            headers=tuple(forward_headers),
            body=action.body if action.body is not None else outbound,
            reissued_from=action.reissued_from,
            unavailable=action.unavailable,
        )
        self.hops_issued.append(hop)
        return hop

    def local_refusal(self) -> RouteRefusal | None:
        if self.actions and isinstance(self.actions[0], Refuse404):
            action = self.actions.pop(0)
            assert isinstance(action, Refuse404)
            return RouteRefusal(
                404, action.message, row=route_row(**{"class": "404"}, hops=0), headers=None
            )
        return None

    def begin(
        self,
        outbound: bytes,
        outbound_obj: Mapping[str, Any] | None,
        forward_headers: Sequence[tuple[str, str]],
    ) -> HopRequest | RouteRefusal:
        self.begun.append((outbound, outbound_obj, forward_headers))
        action = self.actions.pop(0)
        if isinstance(action, Refuse402):
            reissue = "skipped:stateful" if action.headers else "no"
            return RouteRefusal(
                402,
                action.message,
                row=route_row(**{"class": "budget_exhausted"}, reissue=reissue),
                headers=action.headers,
            )
        assert isinstance(action, Hop), f"begin() needs a Hop or Refuse402, got {action!r}"
        return self._hop(action)

    def decide(self, result: HopResult) -> HopDecision:
        self.results.append(result)
        action = self.actions.pop(0)
        if isinstance(action, Stop):
            return HopDecision(None)
        assert isinstance(action, Hop), f"decide() needs a Hop or Stop, got {action!r}"
        return HopDecision(self._hop(action), wait_seconds=action.wait)

    async def wait(self, seconds: float) -> None:
        self.waits.append(seconds)  # deliberately never sleeps

    def delivery(self) -> FakeDelivery:
        last = self.hops_issued[-1].upstream if self.hops_issued else "a"
        self.delivered = FakeDelivery(
            last,
            self.rule,
            self.delivery_headers,
            hops=len({h.upstream for h in self.hops_issued}) or 1,
            wants_payload=self.wants_payload_flag,
            mutate=self.mutate,
        )
        return self.delivered


# --- the router -----------------------------------------------------------------


class FakeRouter:
    def __init__(
        self,
        scripts: Mapping[str, Sequence[Action]] | None = None,
        *,
        local: Mapping[str, LocalAnswer] | None = None,
        status: Mapping[str, Any] | None = None,
        raise_everything: bool = False,
        plan_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        refusal_headers: Mapping[str, str] | None = None,
        validate_error: Exception | None = None,
        reconfigure_error: Exception | None = None,
    ) -> None:
        self.scripts = dict(scripts or {})
        self.local = dict(local or {})
        self._status = dict(status or {"enabled": True, "fake": True})
        self.raise_everything = raise_everything
        self.plan_kwargs = {name: dict(kwargs) for name, kwargs in (plan_kwargs or {}).items()}
        self.refusal_headers = refusal_headers
        self.validate_error = validate_error
        self.reconfigure_error = reconfigure_error
        self.local_calls: list[tuple[str, str]] = []
        self.inbounds: list[RouteInbound] = []
        self.plans: list[FakePlan] = []
        self.validated: list[Config] = []
        self.reconfigured: list[Config] = []
        self.closed = 0

    def _guard(self) -> None:
        if self.raise_everything:
            raise AssertionError("the unrouted path consulted the router")

    def local_answer(
        self, method: str, path: str, headers: Mapping[str, str]
    ) -> LocalAnswer | None:
        self._guard()
        self.local_calls.append((method, path))
        return self.local.get(path)

    def plan(self, inbound: RouteInbound) -> FakePlan | RouteRefusal | None:
        self._guard()
        self.inbounds.append(inbound)
        name = inbound.headers.get(ROUTE_HEADER)
        if name is None:
            return None
        if name == "refuse":
            return RouteRefusal(
                502,
                "fake no_route",
                row=route_row(rule=None, upstream=None, hops=0, **{"class": "no_route"}),
                headers=self.refusal_headers,
            )
        plan = FakePlan(self.scripts[name], **self.plan_kwargs.get(name, {}))
        self.plans.append(plan)
        return plan

    def status(self) -> dict[str, Any]:
        self._guard()
        return dict(self._status)

    def validate(self, config: Config) -> None:
        self._guard()
        self.validated.append(config)
        if self.validate_error is not None:
            raise self.validate_error

    def reconfigure(self, config: Config) -> None:
        self._guard()
        if self.reconfigure_error is not None:
            raise self.reconfigure_error
        self.reconfigured.append(config)

    def close(self) -> None:
        self._guard()
        self.closed += 1


# --- registry wiring --------------------------------------------------------------


def install(
    monkeypatch: pytest.MonkeyPatch,
    router: FakeRouter | Callable[[], FakeRouter],
    *,
    audit: Any | None = None,
) -> tuple[Registry, list[tuple[Config, str]]]:
    """Register a bare Registry whose ``build_router`` returns ``router`` (or a
    fresh one from a factory callable) when ``[routing]`` is enabled and None
    otherwise — the same enabled/None shape as the real factories. Returns the
    registry and the list of ``(config, tier)`` factory calls."""
    reg = Registry()
    calls: list[tuple[Config, str]] = []

    def build_router(config: Config, tier: str) -> FakeRouter | None:
        calls.append((config, tier))
        if not config.routing.enabled:
            return None
        return router() if callable(router) else router

    reg.build_router = build_router
    if audit is not None:
        reg.build_audit = lambda cfg: audit if cfg.enabled else None
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return reg, calls


def routed_config(**overrides: Any) -> Config:
    """A Config with ``[routing] enabled = true`` (the fake ignores the
    upstream table; the factory only reads ``enabled``)."""
    return Config(
        routing=RoutingConfig(
            enabled=True,
            present=True,
            default_upstreams=(("anthropic", "x"),),
            upstreams=(
                UpstreamConfig(name="x", protocol="anthropic", base_url="http://x.example"),
            ),
        ),
        **overrides,
    )


ROUTED_TOML = (
    '[upstreams.x]\nprotocol = "anthropic"\nbase_url = "http://x.example"\n'
    '[routing]\nenabled = true\ndefault_upstream = "x"\n'
)
