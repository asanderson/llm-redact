"""``llm_redact_local_refusals_total{kind,provider}``: every response the
proxy generates ITSELF instead of forwarding is counted once, under the kind
its refusal site names; an upstream's own answer never is.

Three layers:

- static guards: every ``record_request`` call states its ``refusal``
  (None included), every kind a source names is in ``LOCAL_REFUSAL_KINDS``
  and every kind is used, and every refusal-shaped helper counts;
- the enumeration: every kind is asserted, end to end, by some test in the
  suite (``refused_once(state, "<kind>", ...)`` — this file drives the
  simple ones, the suites of the complex paths assert theirs);
- conftest's autouse ``_local_refusals_counted_once`` checks EVERY request
  of the suite: at most one refusal, and a status its kind answers.
"""

from __future__ import annotations

import ast
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, DetectionConfig, ProviderConfig
from llm_redact.metrics import LOCAL_REFUSAL_KINDS, Metrics
from llm_redact.proxy import ProxyState, create_app
from local_refusals import nothing_refused, refused_once

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "llm_redact"
TESTS = ROOT / "tests"
EMAIL = "jane.doe@corp.example"
REFUSING_SOURCES = (SRC / "proxy.py", SRC / "realtime.py")


# ------------------------------------------------------------- static guards --


def _calls(path: Path, name: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == name)
            or (isinstance(node.func, ast.Name) and node.func.id == name)
        )
    ]


def _literals(node: ast.AST) -> set[str]:
    return {
        sub.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    }


@pytest.mark.parametrize("source", REFUSING_SOURCES, ids=lambda p: p.name)
def test_every_recorded_row_states_its_refusal(source: Path) -> None:
    calls = _calls(source, "record_request")
    assert calls
    for call in calls:
        keywords = {k.arg: k.value for k in call.keywords}
        assert "refusal" in keywords, f"{source.name}:{call.lineno} states no refusal="


def _named_kinds() -> set[str]:
    """Every string literal a refusal site hands the counter."""
    named: set[str] = set()
    for source in REFUSING_SOURCES:
        for call in _calls(source, "record_request"):
            for keyword in call.keywords:
                if keyword.arg == "refusal":
                    named |= _literals(keyword.value)
        for helper in ("count_local_refusal", "_record_ws_refusal", "_record_refused"):
            for call in _calls(source, helper):
                for keyword in call.keywords:
                    if keyword.arg == "kind":
                        named |= _literals(keyword.value)
                for arg in call.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        named.add(arg.value)
        for helper in ("refused_response", "_route_refusal", "_Unreadable"):
            for call in _calls(source, helper):
                for keyword in call.keywords:
                    if keyword.arg == "kind":
                        named |= _literals(keyword.value)
                if helper == "refused_response" and len(call.args) == 4:
                    named |= _literals(call.args[3])
    tree = ast.parse((SRC / "realtime.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        # The relay's frame refusals: ``refusal = "<kind>"`` (or a tuple
        # assignment with the status).
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name | ast.Tuple)]
            names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if "refusal" in names:
                named |= _literals(node.value)
    return named


def test_every_named_kind_is_a_kind_and_every_kind_is_named() -> None:
    named = _named_kinds() - {"max_body_bytes"}
    for source in REFUSING_SOURCES:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # The helpers that derive a kind (_unreadable_kind, _raced_kind,
            # _revoked_kind) return literals too.
            if isinstance(node, ast.FunctionDef) and node.name in (
                "_unreadable_kind",
                "_raced_kind",
                "_revoked_kind",
            ):
                named |= _literals(node) - {"kind"}
    named = {n for n in named if re.fullmatch(r"[a-z_]+", n)}
    assert named <= set(LOCAL_REFUSAL_KINDS), sorted(named - set(LOCAL_REFUSAL_KINDS))
    assert set(LOCAL_REFUSAL_KINDS) <= named, sorted(set(LOCAL_REFUSAL_KINDS) - named)


# A function answering a refusal must count it: by name, every helper that
# builds the proxy's own error response.
_REFUSAL_HELPER = re.compile(
    r"_.*(refused|refusal|failure|unavailable|unconfigured|too_large|fault_response)$"
)


@pytest.mark.parametrize("source", REFUSING_SOURCES, ids=lambda p: p.name)
def test_every_refusal_helper_counts(source: Path) -> None:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    helpers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and _REFUSAL_HELPER.fullmatch(node.name)
        and node.returns is not None
        and ("Response" in ast.unparse(node.returns) or ast.unparse(node.returns) == "None")
    ]
    assert helpers
    for helper in helpers:
        calls = {
            call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id
            for call in ast.walk(helper)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute | ast.Name)
        }
        if helper.name in ("_dashboard_unavailable",):
            continue
        assert calls & {"record_request", "count_local_refusal"}, (
            f"{source.name}:{helper.name} answers a refusal without counting it"
        )


def test_the_realtime_record_helpers_require_a_kind() -> None:
    import inspect

    from llm_redact import realtime

    for helper in (realtime._record_ws_refusal, realtime._record_refused):
        parameter = inspect.signature(helper).parameters["kind"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty


def test_every_kind_is_documented() -> None:
    doc = (ROOT / "docs" / "observability.md").read_text(encoding="utf-8")
    section = doc.split("## Local refusal kinds", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"^\| `([a-z_]+)`(?: / `([a-z_]+)`)?", section, re.M))
    kinds = {k for pair in documented for k in pair if k}
    assert kinds == set(LOCAL_REFUSAL_KINDS)


def test_every_kind_is_asserted_end_to_end_somewhere_in_the_suite() -> None:
    """The enumeration: each kind has a test that drives its refusal through
    the real app and asserts it was counted once, under that kind."""
    asserted: set[str] = set()
    pattern = re.compile(r"refused_once(?:_scraped)?\(\s*[^,]+,\s*\"([a-z_]+)\"", re.S)
    for test in TESTS.glob("test_*.py"):
        asserted |= set(pattern.findall(test.read_text(encoding="utf-8")))
    assert asserted == set(LOCAL_REFUSAL_KINDS), sorted(set(LOCAL_REFUSAL_KINDS) - asserted)


def test_the_metric_renders_kind_and_provider() -> None:
    metrics = Metrics("0")
    metrics.count_local_refusal("blocked_value", "openai")
    metrics.count_local_refusal("request_target", None)
    text = metrics.render(
        detections=Counter(),
        rehydrations=Counter(),
        warnings=Counter(),
        blocked=Counter(),
        vault_entries=0,
        vault_sessions=0,
    )
    assert "# TYPE llm_redact_local_refusals_total counter" in text
    assert 'llm_redact_local_refusals_total{kind="blocked_value",provider="openai"} 1' in text
    assert 'llm_redact_local_refusals_total{kind="request_target",provider="passthrough"} 1' in text


# ----------------------------------------------------------------- end to end --


class _Upstream:
    def __init__(self, response: httpx.Response | Exception | None = None) -> None:
        self.calls: list[httpx.Request] = []
        self.response = response

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        if self.response is not None:
            return self.response
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})


def _app(config: Config | None = None, upstream: _Upstream | None = None) -> Any:
    return create_app(
        config or Config(), upstream_transport=httpx.MockTransport(upstream or _Upstream())
    )


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _state(app: Any) -> ProxyState:
    state: ProxyState = app.state.proxy
    return state


def _chat(text: str = "hello") -> dict[str, Any]:
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": text}]}


def _openai(**fields: Any) -> dict[str, ProviderConfig]:
    return {**Config().providers, "openai": ProviderConfig("https://api.openai.test", **fields)}


async def _send(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    async with _client(app) as client:
        return await client.request(method, path, **kwargs)


async def test_an_upstream_answer_is_never_a_local_refusal() -> None:
    # A provider's own 4xx/5xx is the provider's answer, not the proxy's.
    upstream = _Upstream(httpx.Response(429, json={"error": {"message": "slow down"}}))
    app = _app(Config(providers=_openai()), upstream)
    response = await _send(app, "POST", "/v1/chat/completions", json=_chat())
    assert response.status_code == 429 and upstream.calls
    nothing_refused(_state(app))
    assert _state(app).metrics.requests[("openai", "429")] == 1


async def test_a_dot_segment_is_a_request_target_refusal() -> None:
    app = _app()
    response = await _send(app, "GET", "/v1/%2E%2E/models")
    assert response.status_code == 400
    refused_once(_state(app), "request_target", "passthrough")


async def test_an_unclaimed_identity_prefix_is_counted_unrecorded() -> None:
    app = _app()
    response = await _send(app, "GET", "/u/sk-test-key/v1/models")
    assert response.status_code == 404
    refused_once(_state(app), "identity_path", "passthrough")
    assert not _state(app).recent  # never recorded: the path holds a key


async def test_a_web_pages_request_is_a_request_origin_refusal() -> None:
    upstream = _Upstream()
    app = _app(Config(providers=_openai()), upstream)
    response = await _send(
        app,
        "POST",
        "/v1/chat/completions",
        json=_chat(),
        headers={"origin": "https://evil.example", "authorization": "Bearer sk-x"},
    )
    assert response.status_code == 403 and not upstream.calls
    refused_once(_state(app), "request_origin", "openai")


async def test_a_trailing_slash_is_a_misaddressed_refusal() -> None:
    upstream = _Upstream()
    app = _app(Config(providers=_openai()), upstream)
    response = await _send(
        app, "POST", "/v1/chat/completions/", json=_chat(), headers={"authorization": "Bearer sk-x"}
    )
    assert response.status_code == 400 and not upstream.calls
    refused_once(_state(app), "misaddressed", "openai")


async def test_an_unattributable_request_is_an_unattributed_refusal() -> None:
    app = _app()
    response = await _send(app, "GET", "/nothing-here")
    assert response.status_code == 404
    refused_once(_state(app), "unattributed", "passthrough")


async def test_an_unknown_custom_provider_is_a_no_upstream_refusal() -> None:
    app = _app()
    response = await _send(app, "POST", "/custom/nope/v1/chat/completions", json=_chat())
    assert response.status_code == 502
    refused_once(_state(app), "no_upstream", "custom:nope")


async def test_an_unconfigured_upstream_is_a_no_upstream_refusal() -> None:
    providers = {**Config().providers, "azure": ProviderConfig("")}
    app = _app(Config(providers=providers))
    response = await _send(
        app,
        "POST",
        "/openai/deployments/d/chat/completions?api-version=2024-10-21",
        json=_chat(),
        headers={"api-key": "k"},
    )
    assert response.status_code == 502
    refused_once(_state(app), "no_upstream", "azure")


async def test_a_disabled_provider_is_a_disabled_provider_refusal() -> None:
    upstream = _Upstream()
    app = _app(Config(providers=_openai(enabled=False)), upstream)
    response = await _send(
        app, "POST", "/v1/chat/completions", json=_chat(), headers={"authorization": "Bearer sk"}
    )
    assert response.status_code == 502 and not upstream.calls
    refused_once(_state(app), "disabled_provider", "openai")


async def test_a_method_override_is_a_method_override_refusal() -> None:
    upstream = _Upstream()
    app = _app(Config(providers=_openai()), upstream)
    response = await _send(
        app,
        "POST",
        "/v1/chat/completions",
        json=_chat(),
        headers={"authorization": "Bearer sk", "x-http-method-override": "GET"},
    )
    assert response.status_code == 400 and not upstream.calls
    refused_once(_state(app), "method_override", "openai")


async def test_a_body_over_max_body_bytes_is_too_large() -> None:
    upstream = _Upstream()
    app = _app(Config(providers=_openai(), max_body_bytes=64), upstream)
    response = await _send(app, "POST", "/v1/chat/completions", json=_chat("x" * 200))
    assert response.status_code == 413 and not upstream.calls
    refused_once(_state(app), "too_large", "openai")


async def test_a_body_over_max_body_strings_is_too_many_strings() -> None:
    upstream = _Upstream()
    app = _app(Config(providers=_openai(), max_body_strings=2), upstream)
    body = {"model": "m", "messages": [{"role": "user", "content": f"{i}"} for i in range(5)]}
    response = await _send(app, "POST", "/v1/chat/completions", json=body)
    assert response.status_code == 413 and not upstream.calls
    refused_once(_state(app), "too_many_strings", "openai")


async def test_a_body_that_is_no_json_object_is_a_scanned_body_refusal() -> None:
    upstream = _Upstream()
    app = _app(Config(providers=_openai()), upstream)
    response = await _send(app, "POST", "/v1/chat/completions", content=f"mail {EMAIL}".encode())
    assert response.status_code == 400 and not upstream.calls
    refused_once(_state(app), "scanned_body", "openai")


async def test_a_content_encoded_body_is_an_unsupported_encoding_refusal() -> None:
    upstream = _Upstream()
    app = _app(Config(providers=_openai()), upstream)
    response = await _send(
        app,
        "POST",
        "/v1/chat/completions",
        content=gzip.compress(json.dumps(_chat()).encode()),
        headers={"content-encoding": "gzip", "content-type": "application/json"},
    )
    assert response.status_code == 415 and not upstream.calls
    refused_once(_state(app), "unsupported_encoding", "openai")


async def test_a_block_mode_value_is_a_blocked_value_refusal() -> None:
    upstream = _Upstream()
    detection = DetectionConfig(modes=(("email", "block"),))
    app = _app(Config(providers=_openai(), detection=detection), upstream)
    response = await _send(app, "POST", "/v1/chat/completions", json=_chat(f"mail {EMAIL}"))
    assert response.status_code == 400 and not upstream.calls
    refused_once(_state(app), "blocked_value", "openai")


async def test_a_value_in_a_verbatim_field_is_a_verbatim_field_refusal() -> None:
    upstream = _Upstream()
    app = _app(Config(providers=_openai()), upstream)
    body = {"model": "gpt-4o-mini", "training_file": "file-abc", "suffix": EMAIL}
    response = await _send(app, "POST", "/v1/fine_tuning/jobs", json=body)
    assert response.status_code == 400 and not upstream.calls
    refused_once(_state(app), "verbatim_field", "openai")


async def test_a_binary_upload_under_refuse_is_an_unscanned_upload_refusal() -> None:
    upstream = _Upstream()
    detection = DetectionConfig(binary_uploads="refuse")
    app = _app(Config(providers=_openai(), detection=detection), upstream)
    response = await _send(
        app,
        "POST",
        "/v1/files",
        data={"purpose": "assistants"},
        files={"file": ("a.pdf", b"%PDF-1.7\n\x00\x01binary", "application/pdf")},
    )
    assert response.status_code == 400 and not upstream.calls
    refused_once(_state(app), "unscanned_upload", "openai")


async def test_an_undecodable_count_tokens_blob_is_an_unredactable_refusal() -> None:
    upstream = _Upstream()
    providers = {
        **Config().providers,
        "bedrock": ProviderConfig("https://bedrock-runtime.us-east-1.amazonaws.com"),
    }
    app = _app(Config(providers=providers), upstream)
    body = {"input": {"invokeModel": {"body": "!!! not base64 !!!"}}}
    response = await _send(
        app,
        "POST",
        "/model/anthropic.claude-3-haiku/count-tokens",
        json=body,
        headers={"authorization": "Bearer bedrock-key"},
    )
    assert response.status_code == 400 and not upstream.calls
    refused_once(_state(app), "unredactable", "bedrock")


async def test_a_transport_fault_is_an_upstream_fault_refusal() -> None:
    upstream = _Upstream(httpx.ConnectError("refused"))
    app = _app(Config(providers=_openai()), upstream)
    response = await _send(app, "POST", "/v1/chat/completions", json=_chat())
    assert response.status_code == 502
    refused_once(_state(app), "upstream_fault", "openai")


async def test_an_upstream_redirect_is_a_redirect_refusal() -> None:
    upstream = _Upstream(httpx.Response(307, headers={"location": "https://elsewhere.test/"}))
    app = _app(Config(providers=_openai()), upstream)
    response = await _send(app, "POST", "/v1/chat/completions", json=_chat(f"mail {EMAIL}"))
    assert response.status_code == 502
    refused_once(_state(app), "redirect_refused", "openai")
