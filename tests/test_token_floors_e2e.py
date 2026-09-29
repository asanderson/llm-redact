"""Token floors end to end: the real app, fake upstreams, ASGITransport.

Every session numbers its own tokens from 001, so a request can carry
tokens its session never issued. Without a floor, the session could issue
one of those very names for a new value: the upstream would read one token
with two meanings, and an echo of it would restore the NEW value where the
model meant the old one — a wrong value. These drive the request paths that
redact (JSON bodies in a forked and in the static session, a multipart
JSONL upload) and pin: a new value is numbered above every token the
request carries; the upstream never sees one token name with two meanings;
an echoed foreign token passes through verbatim while the new one restores.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from llm_redact import multipart
from llm_redact.config import Config, ProviderConfig, VaultConfig
from llm_redact.placeholders import MAX_TOKEN_NUMBER, canonicalize, token_floors
from llm_redact.proxy import create_app
from llm_redact.registry import Registry

UPSTREAM = "https://upstream.test"
BOB = "bob@corp.example"
CAROL = "carol@corp.example"
DAVE = "dave@corp.example"
EVE = "eve@corp.example"
# Every form the fuzzy rehydrator restores, loosely.
_TOKENS = re.compile("«[^«»]{1,40}»")


class Upstream:
    """Records every request; answers each route with a scripted reply."""

    def __init__(self, reply_text: str = "ok") -> None:
        self.reply_text = reply_text
        self.requests: list[httpx.Request] = []
        self.file_content = b""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/messages":
            return httpx.Response(
                200,
                json={
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.reply_text}],
                },
            )
        if path == "/v1/chat/completions":
            return httpx.Response(
                200, json={"choices": [{"message": {"content": self.reply_text}}]}
            )
        if path == "/v1/files" and request.method == "POST":
            return httpx.Response(200, json={"id": "file-1", "object": "file"})
        if path == "/v1/files/file-1/content":
            return httpx.Response(
                200, content=self.file_content, headers={"content-type": "application/jsonl"}
            )
        return httpx.Response(404, json={"error": "unexpected"})

    def json_body(self, index: int = -1) -> Any:
        return json.loads(self.requests[index].content)


def _config(tmp_path: Path, backend: str = "memory") -> Config:
    vault = (
        VaultConfig(backend="sqlite", path=str(tmp_path / "vault.db"))
        if backend == "sqlite"
        else VaultConfig()
    )
    return Config(
        providers={
            **Config().providers,
            "anthropic": ProviderConfig(UPSTREAM),
            "openai": ProviderConfig(UPSTREAM),
        },
        vault=vault,
    )


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _user_texts(body: dict[str, Any]) -> list[str]:
    """The user turns as sent (the note the proxy injects is a system turn)."""
    return [m["content"] for m in body["messages"] if m.get("role") == "user"]


def _tokens(text: str) -> set[str]:
    """Every token in ``text``, canonicalized where the grammar allows."""
    return {canonicalize(t) or t for t in _TOKENS.findall(text)}


def _assert_one_meaning_per_token(sent: str, carried: set[str], issued: dict[str, str]) -> None:
    """The upstream never reads one token name with two meanings: every
    token the proxy issued in this request is absent from what the request
    already carried, and every carried token is still there, untouched."""
    assert carried <= _tokens(sent)
    assert not carried & set(issued), f"issued {sorted(issued)} onto carried {sorted(carried)}"


# --- (a) a compaction fork in a per-conversation-like session -------------------


class ForkingRouter:
    """A per-conversation session-router double: the session is derived from
    the first message, so a compacted history (a rewritten first message)
    resolves to a NEW, empty session — the compaction fork."""

    mode = "per-conversation"

    def __init__(self) -> None:
        self.resolved: list[str] = []

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        first = json.dumps(body["messages"][0], sort_keys=True)
        session = "conv-" + hashlib.sha256(first.encode()).hexdigest()[:16]
        self.resolved.append(session)
        return session

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None


def _forking_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, upstream: Upstream, backend: str
) -> tuple[Any, ForkingRouter]:
    router = ForkingRouter()
    reg = Registry()
    reg.build_session_router = lambda config, **kw: router
    monkeypatch.setattr(registry_mod, "_registry", reg)
    app = create_app(_config(tmp_path, backend), upstream_transport=httpx.MockTransport(upstream))
    return app, router


_SUMMARY = (
    "Summary of the conversation so far: the user asked «EMAIL_001» to forward"
    " the report to «EMAIL_003»; «email_2» was copied."
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_a_compaction_fork_numbers_new_values_above_the_summarys_tokens(
    backend: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstream = Upstream(
        reply_text="Done: «EMAIL_001» and «email_2» and «EMAIL_003» now include «EMAIL_004»."
    )
    app, router = _forking_app(monkeypatch, tmp_path, upstream, backend)
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": _SUMMARY},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": f"Also loop in {BOB}, please."},
        ],
    }
    async with _client(app) as client:
        response = await client.post("/v1/messages", json=body)
    assert response.status_code == 200
    state = app.state.proxy
    (session,) = set(router.resolved)
    assert state.compaction_forks == 1  # the fork detector is unchanged

    sent = upstream.json_body()
    assert sent["messages"][2]["content"] == "Also loop in «EMAIL_004», please."
    assert sent["messages"][0]["content"] == _SUMMARY  # carried tokens untouched
    vault = state.vault_manager.get(session)
    issued = {"«EMAIL_004»": BOB}
    assert len(vault) == 1 and vault.original_for("«EMAIL_004»") == BOB
    _assert_one_meaning_per_token(
        upstream.requests[-1].content.decode(),
        {"«EMAIL_001»", "«EMAIL_002»", "«EMAIL_003»"},
        issued,
    )

    # The echo: tokens the ORIGINAL session issued stay placeholders (never
    # the new value); the fork's own token restores.
    text = response.json()["content"][0]["text"]
    assert text == f"Done: «EMAIL_001» and «email_2» and «EMAIL_003» now include {BOB}."

    # The next turn of the forked conversation: the summary (still first)
    # keeps its floor, BOB keeps his token, the next value continues.
    body["messages"] += [
        {"role": "assistant", "content": text},
        {"role": "user", "content": f"And {CAROL}."},
    ]
    async with _client(app) as client:
        await client.post("/v1/messages", json=body)
    sent = upstream.json_body()
    assert sent["messages"][2]["content"] == "Also loop in «EMAIL_004», please."
    assert sent["messages"][4]["content"] == "And «EMAIL_005»."
    assert set(router.resolved) == {session}
    assert state.compaction_forks == 1


# --- (b) the static session with a pasted foreign token ------------------------------


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_a_pasted_foreign_token_in_the_static_session(backend: str, tmp_path: Path) -> None:
    upstream = Upstream(reply_text="Writing to «EMAIL_003», as «EMAIL_002» asked.")
    app = create_app(_config(tmp_path, backend), upstream_transport=httpx.MockTransport(upstream))
    pasted = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": f"An earlier answer said «EMAIL_002» — write to {CAROL}"}
        ],
    }
    async with _client(app) as client:
        first = await client.post("/v1/chat/completions", json=pasted)
        # The floor belonged to THAT request: a token-free one continues
        # from the session's own highest number, the floor did not stick to
        # the shared static redactor.
        second = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": f"and {DAVE}"}]},
        )
    assert first.status_code == second.status_code == 200
    assert _user_texts(upstream.json_body(0)) == [
        "An earlier answer said «EMAIL_002» — write to «EMAIL_003»"
    ]
    assert _user_texts(upstream.json_body(1)) == ["and «EMAIL_004»"]
    _assert_one_meaning_per_token(
        upstream.requests[0].content.decode(), {"«EMAIL_002»"}, {"«EMAIL_003»": CAROL}
    )
    # «EMAIL_002» was never issued here: its echo stays a placeholder.
    assert first.json()["choices"][0]["message"]["content"] == (
        f"Writing to {CAROL}, as «EMAIL_002» asked."
    )
    static = app.state.proxy.vault
    assert static.original_for("«EMAIL_002»") is None
    assert static.original_for("«EMAIL_003»") == CAROL


async def test_floors_read_escaped_bodies_and_json_source_arguments(tmp_path: Path) -> None:
    # A Python-default (ensure_ascii) body escapes every guillemet, and a
    # tool call echoed in history carries JSON-SOURCE arguments whose own
    # guillemets are escaped once more — both still bound the new value.
    upstream = Upstream()
    app = create_app(_config(tmp_path), upstream_transport=httpx.MockTransport(upstream))
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "forward «EMAIL_010»"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "send",
                            "arguments": json.dumps({"to": "«EMAIL_012»"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sent"},
            {"role": "user", "content": f"now {EVE}"},
        ],
    }
    raw = json.dumps(body).encode()
    assert "«".encode() not in raw  # nothing but escapes on the wire
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions", content=raw, headers={"content-type": "application/json"}
        )
    assert response.status_code == 200
    assert _user_texts(upstream.json_body())[-1] == "now «EMAIL_013»"


async def test_a_token_free_body_skips_the_floor_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llm_redact.proxy as proxy_mod

    def never(obj: Any) -> dict[str, int]:
        raise AssertionError("the byte gate must skip the walk")

    monkeypatch.setattr(proxy_mod, "json_floors", never)
    upstream = Upstream()
    app = create_app(_config(tmp_path), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": f"hi {BOB}"}]},
        )
    assert response.status_code == 200
    assert _user_texts(upstream.json_body()) == ["hi «EMAIL_001»"]


# --- (c) a multipart JSONL upload ---------------------------------------------------


def _batch_line(custom_id: str, text: str) -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": text}]},
        },
        ensure_ascii=False,
    )


async def test_a_jsonl_upload_numbers_above_every_line_and_restores_its_output(
    tmp_path: Path,
) -> None:
    upstream = Upstream()
    app = create_app(_config(tmp_path), upstream_transport=httpx.MockTransport(upstream))
    upload = (
        _batch_line("a", f"mail {BOB}") + "\n" + _batch_line("b", "history «EMAIL_007»") + "\n"
    ).encode()
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            data={"purpose": "batch"},
            files={"file": ("in.jsonl", upload, "application/jsonl")},
        )
    assert response.status_code == 200
    sent = upstream.requests[-1]
    boundary = multipart.parse_boundary(sent.headers["content-type"])
    assert boundary is not None
    parsed = multipart.parse(sent.content, boundary)
    assert parsed is not None
    lines = [json.loads(line) for line in parsed.parts[1].content.split(b"\n") if line.strip()]
    contents = [line["body"]["messages"][-1]["content"] for line in lines]
    # The FIRST line's value is numbered above the SECOND line's token.
    assert contents == ["mail «EMAIL_008»", "history «EMAIL_007»"]

    # The batch output file echoes both tokens: the foreign one stays a
    # placeholder, the upload's own restores.
    output = {
        "custom_id": "a",
        "response": {
            "body": {"choices": [{"message": {"content": "«EMAIL_007» and «EMAIL_008»"}}]}
        },
    }
    upstream.file_content = json.dumps(output, ensure_ascii=False).encode() + b"\n"
    async with _client(app) as client:
        fetched = await client.get("/v1/files/file-1/content")
    restored = json.loads(fetched.content.splitlines()[0])
    assert restored["response"]["body"]["choices"][0]["message"]["content"] == (
        f"«EMAIL_007» and {BOB}"
    )


# --- the end of the number space ------------------------------------------------------


async def test_a_token_at_the_limit_refuses_only_a_request_that_needs_a_number(
    tmp_path: Path,
) -> None:
    upstream = Upstream()
    app = create_app(_config(tmp_path), upstream_transport=httpx.MockTransport(upstream))
    at_limit = f"«EMAIL_{MAX_TOKEN_NUMBER}»"
    async with _client(app) as client:
        refused = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": f"{at_limit} and {BOB}"}],
            },
        )
        served = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": f"{at_limit} ok"}]},
        )
    assert refused.status_code == 400
    message = refused.json()["error"]["message"]
    assert message.startswith("llm-redact: no EMAIL placeholder number")
    assert str(MAX_TOKEN_NUMBER) in message and "not forwarded" in message
    assert BOB not in refused.text
    assert served.status_code == 200 and len(upstream.requests) == 1
    rows = list(app.state.proxy.recent)
    assert [row["status"] for row in rows] == [400, 200]  # oldest first


async def test_a_multipart_upload_at_the_limit_names_the_limit_not_identity_auth(
    tmp_path: Path,
) -> None:
    upstream = Upstream()
    app = create_app(_config(tmp_path), upstream_transport=httpx.MockTransport(upstream))
    upload = (_batch_line("a", f"«EMAIL_{MAX_TOKEN_NUMBER}» and {BOB}") + "\n").encode()
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            data={"purpose": "batch"},
            files={"file": ("in.jsonl", upload, "application/jsonl")},
        )
    assert response.status_code == 400 and upstream.requests == []
    message = response.json()["error"]["message"]
    assert "placeholder number" in message and "identity" not in message


# --- sealed sessions still refuse -------------------------------------------------------


async def test_a_sealed_session_still_refuses_a_floored_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Sealing(ForkingRouter):
        def sealed(self, session_id: str) -> bool:
            return True

    router = Sealing()
    reg = Registry()
    reg.build_session_router = lambda config, **kw: router
    monkeypatch.setattr(registry_mod, "_registry", reg)
    upstream = Upstream()
    app = create_app(_config(tmp_path), upstream_transport=httpx.MockTransport(upstream))
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": f"{_SUMMARY} And {BOB}."}],
    }
    async with _client(app) as client:
        response = await client.post("/v1/messages", json=body)
    assert response.status_code == 403 and upstream.requests == []
    (session,) = set(router.resolved)
    assert len(app.state.proxy.vault_manager.get(session)) == 0


def test_the_helpers_agree_with_the_scan() -> None:
    # The e2e assertions canonicalize like the rehydrator; the scan agrees.
    assert _tokens(_SUMMARY) == {"«EMAIL_001»", "«EMAIL_002»", "«EMAIL_003»"}
    assert token_floors(_SUMMARY) == {"EMAIL": 3}
    check: Callable[..., None] = _assert_one_meaning_per_token
    with pytest.raises(AssertionError):
        check("«EMAIL_001»", {"«EMAIL_001»"}, {"«EMAIL_001»": BOB})
