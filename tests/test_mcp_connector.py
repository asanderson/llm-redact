"""MCP connector surfaces: config passes through BY DESIGN, content flows.

The provider must receive the real MCP server credential to call it on
the model's behalf — so `mcp_servers[]` (Anthropic) and
`tools[].type == "mcp"` (Responses, Realtime) bypass redaction, stripped
BEFORE the walk so nothing in them is counted. MCP call CONTENT
(arguments/output) is redacted and rehydrated like any other content.
"""

import json
from types import SimpleNamespace

from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors
from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.base import SYSTEM_NOTE
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.realtime import OpenAIRealtimeWs
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import RehydratorPool
from llm_redact.sse import SSEEvent
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"
GITHUB_PAT = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"


def _redactor(vault: InMemoryVault) -> Redactor:
    return Redactor(
        detectors=build_detectors(DetectionConfig()),
        vault=vault,
        allowlist=Allowlist(exact=frozenset(), patterns=()),
    )


def test_anthropic_mcp_servers_preserved() -> None:
    adapter = AnthropicAdapter()
    redactor = _redactor(InMemoryVault())
    body = {
        "model": "claude-sonnet-4-5",
        "mcp_servers": [
            {
                "type": "url",
                "url": "https://mcp.corp.example/sse",
                "name": "github",
                "authorization_token": GITHUB_PAT,
            }
        ],
        "messages": [{"role": "user", "content": f"email {EMAIL} about the repo"}],
    }
    prepared = adapter.prepare_request(body, redactor, inject_note=True)
    # The connector block is byte-for-byte the original — credential intact.
    assert prepared["mcp_servers"] == body["mcp_servers"]
    assert prepared["mcp_servers"][0]["authorization_token"] == GITHUB_PAT
    # Conversation content is still redacted, and the note still lands.
    assert EMAIL not in json.dumps(prepared["messages"])
    assert SYSTEM_NOTE in str(prepared.get("system", ""))
    # Nothing in the connector block was counted as a detection.
    assert redactor.counts.get("GITHUB_TOKEN", 0) == 0
    assert redactor.counts["EMAIL"] == 1


def test_responses_mcp_tools_preserved() -> None:
    adapter = OpenAIResponsesAdapter()
    redactor = _redactor(InMemoryVault())
    body = {
        "model": "gpt-4o",
        "tools": [
            {
                "type": "mcp",
                "server_label": "github",
                "server_url": "https://mcp.corp.example/sse",
                "headers": {"Authorization": f"Bearer {GITHUB_PAT}"},
            },
            {"type": "function", "name": "lookup", "description": f"contact {EMAIL}"},
        ],
        "input": f"email {EMAIL} please",
    }
    prepared = adapter.prepare_request(body, redactor, inject_note=False)
    assert prepared["tools"][0] == body["tools"][0]  # mcp entry verbatim
    assert GITHUB_PAT in prepared["tools"][0]["headers"]["Authorization"]
    # Non-MCP tools and the input are still redacted.
    assert EMAIL not in json.dumps(prepared["tools"][1])
    assert EMAIL not in json.dumps(prepared["input"])
    assert redactor.counts.get("GITHUB_TOKEN", 0) == 0


def test_responses_mcp_call_arguments_stream_split() -> None:
    """MCP arguments stream like function arguments: JSON source, split
    tokens reassembled, done event re-rehydrated — at every offset."""
    adapter = OpenAIResponsesAdapter()
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    args = json.dumps({"to": token})

    def run(split: int) -> str:
        pool = RehydratorPool(vault)
        out = ""
        for part in (args[:split], args[split:]):
            event = SSEEvent(
                event="response.mcp_call_arguments.delta",
                data=json.dumps(
                    {"type": "response.mcp_call_arguments.delta", "item_id": "m1", "delta": part}
                ),
            )
            for rewritten in adapter.rehydrate_event(event, pool):
                out += json.loads(rewritten.data).get("delta", "")
        done = SSEEvent(
            event="response.mcp_call_arguments.done",
            data=json.dumps(
                {"type": "response.mcp_call_arguments.done", "item_id": "m1", "arguments": args}
            ),
        )
        done_out = ""
        for rewritten in adapter.rehydrate_event(done, pool):
            payload = json.loads(rewritten.data)
            if payload["type"] == "response.mcp_call_arguments.done":
                done_out = payload["arguments"]
            else:  # synthetic leftover delta precedes the done event
                out += payload.get("delta", "")
        assert json.loads(done_out) == {"to": EMAIL}
        return out

    for split in range(1, len(args)):
        assert json.loads(run(split)) == {"to": EMAIL}, f"split at {split}"


def test_realtime_mcp_tools_preserved() -> None:
    adapter = OpenAIRealtimeWs()
    vault = InMemoryVault()
    ctx = SimpleNamespace(redactor=_redactor(vault))
    event = {
        "type": "session.update",
        "session": {
            "instructions": f"always cc {EMAIL}",
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github",
                    "server_url": "https://mcp.corp.example/sse",
                    "headers": {"Authorization": f"Bearer {GITHUB_PAT}"},
                }
            ],
        },
    }
    out = json.loads(adapter.redact_message(json.dumps(event), ctx))
    assert out["session"]["tools"][0] == event["session"]["tools"][0]
    assert EMAIL not in out["session"]["instructions"]
    assert ctx.redactor.counts.get("GITHUB_TOKEN", 0) == 0
