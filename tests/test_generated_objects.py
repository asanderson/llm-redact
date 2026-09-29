"""Objects the PROVIDER generates on a request's behalf are reported as that
request's creator's (``SessionRouter.record_object_id``), from the creator's
OWN answer — never from a request body, never from a read:

- Anthropic Messages: a code execution tool result lists every file its run
  wrote (``code_execution_output`` / ``bash_code_execution_output`` entries
  with a ``file_id``, downloadable through the Files API) — non-streaming,
  and streamed (the result block arrives whole in ``content_block_start``);
- OpenAI Responses: a ``container_file_citation`` annotation (and a code
  interpreter call's output files) names a file the code interpreter wrote
  into its container — non-streaming, and streamed (the annotation, the
  finished item, the finished content part and the completed response
  each repeat it; each id is reported once);
- OpenAI fine-tuning: a job create reports the job, and a job read reports
  its ``result_files`` (the router attributes them to the job's recorder).

The streamed forms are pinned by split-stream sweeps: the SSE bytes split
at every offset report exactly the ids the unsplit stream reports, once
each, and the client receives the same restored stream.

Keyless: a scripted router on a bare Registry stands in for llm-redact-pro.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from llm_redact.config import Config, ProviderConfig, VaultConfig
from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.azure_openai import AzureOpenAIAdapter, AzureResponsesAdapter
from llm_redact.providers.custom import CustomResponsesAdapter
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.proxy import RequestMeta, _stream_rehydrated, create_app
from llm_redact.registry import Registry
from llm_redact.sse import SSEEvent

UPSTREAM = "https://upstream.test"
EMAIL = "ada.private@example.org"
SESSION = "user:n1:main"


class Router:
    """Records every reported object; resolves everything to one session."""

    def __init__(self) -> None:
        self.mode = "per-user"
        self.objects: list[tuple[str, str]] = []

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return SESSION

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def record_object_id(self, object_id: str, session_id: str) -> bool:
        self.objects.append((object_id, session_id))
        return True


def _app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    router: Router,
    respond: Any,
    provider: str,
) -> Any:
    reg = Registry()
    reg.build_session_router = lambda config, **kw: router
    monkeypatch.setattr(registry_mod, "_registry", reg)
    config = Config(
        providers={**Config().providers, provider: ProviderConfig(UPSTREAM)},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    return create_app(config, upstream_transport=httpx.MockTransport(respond))


async def _send(app: Any, method: str, path: str, body: Any = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        if body is None:
            return await client.request(method, path)
        return await client.request(method, path, json=body)


def _sse(events: list[tuple[str | None, dict[str, Any]]]) -> bytes:
    out = []
    for name, payload in events:
        head = f"event: {name}\n" if name else ""
        out.append(f"{head}data: {json.dumps(payload, ensure_ascii=False)}\n\n")
    return "".join(out).encode()


# --- Anthropic Messages: code execution output files ----------------------------------


def _code_result(*file_ids: str, kind: str = "bash_code_execution") -> dict[str, Any]:
    return {
        "type": f"{kind}_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": {
            "type": f"{kind}_result",
            "stdout": "wrote the chart",
            "stderr": "",
            "return_code": 0,
            "content": [{"type": f"{kind}_output", "file_id": fid} for fid in file_ids],
        },
    }


MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "container": {"id": "container_1", "expires_at": "2026-09-29T00:00:00Z"},
    "content": [
        {"type": "text", "text": "Here is the chart."},
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "bash_code_execution"},
        _code_result("file_chart", "file_table"),
        _code_result("file_legacy", kind="code_execution"),
        # Neither a generated file: an error result, an unrelated tool's
        # result, and blocks naming files the REQUEST supplied.
        {
            "type": "code_execution_tool_result",
            "tool_use_id": "srvtoolu_2",
            "content": {"type": "code_execution_tool_result_error", "error_code": "unavailable"},
        },
        {"type": "web_search_tool_result", "tool_use_id": "t", "content": [{"file_id": "nope"}]},
        {"type": "container_upload", "file_id": "file_input"},
        {"type": "document", "source": {"type": "file", "file_id": "file_doc"}},
    ],
}


def test_anthropic_reads_the_files_a_code_execution_run_wrote() -> None:
    adapter = AnthropicAdapter()
    assert adapter.tracks_object_ids("POST", "/v1/messages")
    assert not adapter.tracks_object_ids("POST", "/v1/messages/count_tokens")
    assert not adapter.tracks_object_ids("GET", "/v1/messages")
    assert adapter.object_ids_from_body("POST", "/v1/messages", MESSAGE) == (
        "file_chart",
        "file_table",
        "file_legacy",
    )
    # The message id is no stored object; odd shapes name nothing.
    assert adapter.object_ids_from_body("POST", "/v1/messages", {"id": "msg_1"}) == ()
    assert adapter.object_ids_from_body("POST", "/v1/messages", {"content": "x"}) == ()
    assert adapter.object_ids_from_body("POST", "/v1/messages", ["x"]) == ()
    odd = {"content": [7, {"type": 3}, _code_result(), {"type": "x_tool_result", "content": []}]}
    assert adapter.object_ids_from_body("POST", "/v1/messages", odd) == ()
    bad_output = _code_result()
    bad_output["content"]["content"] = [{"type": "bash_code_execution_output", "file_id": 7}, 1]
    assert adapter.object_ids_from_body("POST", "/v1/messages", {"content": [bad_output]}) == ()
    # Batches and uploads keep reporting the object they create.
    assert adapter.object_ids_from_body("POST", "/v1/messages/batches", {"id": "msgbatch_1"}) == (
        "msgbatch_1",
    )
    # A stream names its files on the events that carry them: each is read.
    assert not adapter.reports_object_ids_once("POST", "/v1/messages")
    assert adapter.reports_object_ids_once("POST", "/v1/messages/batches")


def test_anthropic_events_name_the_files_of_the_blocks_they_start() -> None:
    adapter = AnthropicAdapter()
    start = {"type": "content_block_start", "index": 2, "content_block": _code_result("file_a")}
    event = SSEEvent(event="content_block_start", data=json.dumps(start))
    assert adapter.object_ids_from_event("POST", "/v1/messages", event) == ("file_a",)
    opening = {"type": "message_start", "message": {**MESSAGE, "content": [_code_result("f_m")]}}
    event = SSEEvent(event="message_start", data=json.dumps(opening))
    assert adapter.object_ids_from_event("POST", "/v1/messages", event) == ("f_m",)
    for data in (
        # No file id anywhere: never even parsed.
        '{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta"}}',
        # A text delta that merely MENTIONS the key, escaped inside a string.
        json.dumps({"type": "content_block_delta", "delta": {"text": '"file_id": "f"'}}),
        '{"type": "message_start", "message": "file_id", "x": {"file_id": 1}}',
        '{"type": "content_block_stop", "index": 2, "file_id": "f"}',
        'not json "file_id"',
        '["file_id"]',
        "",
    ):
        event = SSEEvent(event="x", data=data)
        assert adapter.object_ids_from_event("POST", "/v1/messages", event) == (), data


async def test_anthropic_generated_files_are_reported_with_their_creator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = Router()
    app = _app(
        monkeypatch, tmp_path, router, lambda r: httpx.Response(200, json=MESSAGE), "anthropic"
    )
    body = {"model": "claude", "max_tokens": 10, "messages": [{"role": "user", "content": "chart"}]}
    response = await _send(app, "POST", "/v1/messages", body)
    assert response.status_code == 200
    assert router.objects == [
        ("file_chart", SESSION),
        ("file_table", SESSION),
        ("file_legacy", SESSION),
    ]
    durable = app.state.proxy.vault_manager.lookup_response_session("file_chart")
    assert durable == SESSION


async def test_an_id_the_request_itself_cites_is_never_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The request uploads file_chart into the container; a result naming it
    # again did not create it — it is an existing object, echoed.
    router = Router()
    app = _app(
        monkeypatch, tmp_path, router, lambda r: httpx.Response(200, json=MESSAGE), "anthropic"
    )
    content = [
        {"type": "container_upload", "file_id": "file_chart"},
        {"type": "text", "text": "file_legacy is what I call it"},  # a mention is no citation
    ]
    body = {"model": "c", "max_tokens": 1, "messages": [{"role": "user", "content": content}]}
    assert (await _send(app, "POST", "/v1/messages", body)).status_code == 200
    assert router.objects == [("file_table", SESSION), ("file_legacy", SESSION)]


async def test_an_answer_whose_files_are_all_cited_reports_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = Router()
    answer = {"content": [_code_result("file_in")]}
    app = _app(
        monkeypatch, tmp_path, router, lambda r: httpx.Response(200, json=answer), "anthropic"
    )
    content = [{"type": "container_upload", "file_id": "file_in"}]
    body = {"model": "c", "max_tokens": 1, "messages": [{"role": "user", "content": content}]}
    assert (await _send(app, "POST", "/v1/messages", body)).status_code == 200
    assert router.objects == []


def _anthropic_stream() -> bytes:
    return _sse(
        [
            ("message_start", {"type": "message_start", "message": {**MESSAGE, "content": []}}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "for «EMAIL_001»"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 1, "content_block": _code_result("f1")},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": _code_result("f2", "f1", kind="code_execution"),
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 2}),
            ("message_stop", {"type": "message_stop"}),
        ]
    )


async def test_a_streamed_message_reports_its_generated_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = Router()
    stream = _anthropic_stream()

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

    app = _app(monkeypatch, tmp_path, router, respond, "anthropic")
    body = {
        "model": "c",
        "max_tokens": 1,
        "stream": True,
        "messages": [{"role": "user", "content": "x"}],
    }
    response = await _send(app, "POST", "/v1/messages", body)
    assert response.status_code == 200 and "message_stop" in response.text
    assert router.objects == [("f1", SESSION), ("f2", SESSION)]


# --- the generator-level split sweep ---------------------------------------------------


class _Chunks:
    """An upstream response whose body arrives in the given chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.status_code = 200
        self._chunks = chunks

    async def aiter_bytes(self) -> Any:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


async def _sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter: Any,
    path: str,
    stream: bytes,
    request_body: Any = None,
) -> tuple[list[str], bytes]:
    """Run ``stream`` through the rehydrating generator whole and split at
    every offset; every run must report the same ids, each once, and emit
    the same bytes. Returns the ids and the restored stream."""
    router = Router()
    app = _app(monkeypatch, tmp_path, router, lambda r: httpx.Response(500), "openai")
    state = app.state.proxy
    ctx = state.context_for(adapter, "POST", path, request_body)
    ctx.vault.placeholder_for("EMAIL", EMAIL)
    tracker = state.object_tracker(adapter, "POST", path, body=request_body)
    assert tracker is adapter

    async def run(chunks: list[bytes]) -> tuple[list[str], bytes]:
        router.objects.clear()
        meta = RequestMeta("POST", path, 0.0, {}, {})
        out = b""
        async for piece in _stream_rehydrated(
            _Chunks(chunks),  # type: ignore[arg-type]
            adapter,
            state,
            ctx,
            request_meta=meta,
            object_tracker=tracker,
            request_body=request_body,
        ):
            out += piece
        return [object_id for object_id, _ in router.objects], out

    whole_ids, whole = await run([stream])
    for offset in range(1, len(stream)):
        ids, out = await run([stream[:offset], stream[offset:]])
        assert (ids, out) == (whole_ids, whole), offset
    assert len(set(whole_ids)) == len(whole_ids)
    return whole_ids, whole


async def test_a_split_anthropic_stream_reports_each_file_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids, out = await _sweep(
        monkeypatch, tmp_path, AnthropicAdapter(), "/v1/messages", _anthropic_stream()
    )
    assert ids == ["f1", "f2"]
    assert EMAIL in out.decode()  # rehydration unaffected


async def test_a_split_anthropic_stream_skips_what_the_request_cites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = {
        "messages": [{"role": "user", "content": [{"type": "container_upload", "file_id": "f1"}]}]
    }
    ids, _ = await _sweep(
        monkeypatch, tmp_path, AnthropicAdapter(), "/v1/messages", _anthropic_stream(), body
    )
    assert ids == ["f2"]


# --- OpenAI Responses: container files ------------------------------------------------


def _citation(file_id: str) -> dict[str, Any]:
    return {
        "type": "container_file_citation",
        "container_id": "cntr_1",
        "file_id": file_id,
        "filename": "chart.png",
        "start_index": 0,
        "end_index": 4,
    }


RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "output": [
        {
            "type": "code_interpreter_call",
            "id": "ci_1",
            "container_id": "cntr_1",
            "code": "plot()",
            "outputs": [
                {"type": "logs", "logs": "ok"},
                {"type": "files", "files": [{"file_id": "cfile_out", "mime_type": "image/png"}]},
            ],
            "results": [
                {"type": "files", "files": [{"file_id": "cfile_legacy"}, "x", {"file_id": 7}]}
            ],
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "see chart for «EMAIL_001»",
                    "annotations": [
                        _citation("cfile_chart"),
                        _citation("cfile_out"),  # repeated: reported once
                        {"type": "url_citation", "url": "https://example.org"},
                        {"type": "file_citation", "file_id": "file-cited-input"},
                        "odd",
                    ],
                },
                "odd",
                {"type": "refusal"},
            ],
        },
        "odd",
        {"type": "message", "content": "odd"},
    ],
}


@pytest.mark.parametrize(
    ("adapter", "path"),
    [
        (OpenAIResponsesAdapter(), "/v1/responses"),
        (AzureResponsesAdapter(), "/openai/v1/responses"),
        (AzureResponsesAdapter(), "/openai/responses"),
        (CustomResponsesAdapter("lm"), "/custom/lm/api/v1/responses"),
    ],
)
def test_responses_read_the_container_files_an_answer_cites(adapter: Any, path: str) -> None:
    assert adapter.tracks_object_ids("POST", path)
    assert not adapter.tracks_object_ids("GET", path + "/resp_1")  # a read creates nothing
    assert not adapter.tracks_object_ids("POST", path + "/resp_1/cancel")
    assert adapter.object_ids_from_body("POST", path, RESPONSE) == (
        "cfile_out",
        "cfile_legacy",
        "cfile_chart",
    )
    assert adapter.object_ids_from_body("POST", path, {"output": "x"}) == ()
    assert adapter.object_ids_from_body("POST", path, ["x"]) == ()
    assert not adapter.reports_object_ids_once("POST", path)


def test_responses_events_name_the_container_files_they_carry() -> None:
    adapter = OpenAIResponsesAdapter()
    cases = [
        (
            {"type": "response.output_text.annotation.added", "annotation": _citation("cf_a")},
            ("cf_a",),
        ),
        (
            {"type": "response.content_part.done", "part": RESPONSE["output"][1]["content"][0]},
            ("cfile_chart", "cfile_out"),
        ),
        (
            {"type": "response.output_item.done", "item": RESPONSE["output"][0]},
            ("cfile_out", "cfile_legacy"),
        ),
        (
            {"type": "response.completed", "response": RESPONSE},
            ("cfile_out", "cfile_legacy", "cfile_chart"),
        ),
        ({"type": "response.output_text.delta", "delta": '"file_id": "x"'}, ()),
        ({"type": "response.output_item.added", "item": "odd", "file_id": "x"}, ()),
    ]
    for payload, expected in cases:
        event = SSEEvent(event=str(payload["type"]), data=json.dumps(payload))
        assert adapter.object_ids_from_event("POST", "/v1/responses", event) == expected, payload
    for data in ('{"type": "response.created"}', 'not json "file_id"', '["file_id"]', ""):
        event = SSEEvent(event="x", data=data)
        assert adapter.object_ids_from_event("POST", "/v1/responses", event) == ()


async def test_a_responses_answer_reports_its_container_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = Router()
    app = _app(
        monkeypatch, tmp_path, router, lambda r: httpx.Response(200, json=RESPONSE), "openai"
    )
    body = {
        "model": "gpt",
        "input": "plot it",
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
    }
    response = await _send(app, "POST", "/v1/responses", body)
    assert response.status_code == 200
    assert EMAIL not in response.text  # nothing to restore in that session: untouched
    assert router.objects == [
        ("cfile_out", SESSION),
        ("cfile_legacy", SESSION),
        ("cfile_chart", SESSION),
    ]


def _responses_stream() -> bytes:
    item = RESPONSE["output"][1]
    part = item["content"][0]
    return _sse(
        [
            ("response.created", {"type": "response.created", "response": {"id": "resp_1"}}),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "output_index": 1,
                    "content_index": 0,
                    "delta": "see chart for «EMAIL_001»",
                },
            ),
            (
                "response.output_text.annotation.added",
                {
                    "type": "response.output_text.annotation.added",
                    "item_id": "msg_1",
                    "annotation": _citation("cfile_chart"),
                },
            ),
            (
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": "msg_1",
                    "output_index": 1,
                    "content_index": 0,
                    "text": "see chart for «EMAIL_001»",
                },
            ),
            ("response.content_part.done", {"type": "response.content_part.done", "part": part}),
            ("response.output_item.done", {"type": "response.output_item.done", "item": item}),
            ("response.completed", {"type": "response.completed", "response": RESPONSE}),
        ]
    )


async def test_a_streamed_response_reports_its_container_files_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = Router()
    stream = _responses_stream()

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

    app = _app(monkeypatch, tmp_path, router, respond, "openai")
    body = {"model": "gpt", "input": "plot it", "stream": True}
    response = await _send(app, "POST", "/v1/responses", body)
    assert response.status_code == 200 and "response.completed" in response.text
    assert router.objects == [
        ("cfile_chart", SESSION),
        ("cfile_out", SESSION),
        ("cfile_legacy", SESSION),
    ]


async def test_a_split_responses_stream_reports_each_file_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids, out = await _sweep(
        monkeypatch, tmp_path, OpenAIResponsesAdapter(), "/v1/responses", _responses_stream()
    )
    assert ids == ["cfile_chart", "cfile_out", "cfile_legacy"]
    assert EMAIL in out.decode()


async def test_a_split_responses_stream_skips_what_the_request_cites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The client resent an earlier answer citing cfile_out: that file is
    # not this request's creation.
    earlier = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "old", "annotations": [_citation("cfile_out")]}
        ],
    }
    body = {"model": "gpt", "input": [earlier, {"role": "user", "content": "again"}]}
    ids, _ = await _sweep(
        monkeypatch, tmp_path, OpenAIResponsesAdapter(), "/v1/responses", _responses_stream(), body
    )
    assert ids == ["cfile_chart", "cfile_legacy"]


# --- OpenAI fine-tuning: the job and its result files ----------------------------------


@pytest.mark.parametrize(
    ("method", "path", "tracked"),
    [
        ("POST", "/v1/fine_tuning/jobs", True),
        ("POST", "/openai/fine_tuning/jobs", True),  # Azure's prefixes
        ("POST", "/openai/v1/fine_tuning/jobs", True),
        ("GET", "/v1/fine_tuning/jobs/ftjob-1", True),
        ("POST", "/v1/fine_tuning/jobs/ftjob-1/cancel", True),
        ("POST", "/v1/fine_tuning/jobs/ftjob-1/pause", True),
        ("POST", "/v1/fine_tuning/jobs/ftjob-1/resume", True),
        ("GET", "/v1/fine_tuning/jobs", False),  # the listing
        ("GET", "/v1/fine_tuning/jobs/ftjob-1/events", False),
        ("GET", "/v1/fine_tuning/jobs/ftjob-1/checkpoints", False),
        ("DELETE", "/v1/fine_tuning/jobs/ftjob-1", False),
    ],
)
def test_openai_tracks_fine_tuning_jobs(method: str, path: str, tracked: bool) -> None:
    for adapter in (OpenAIAdapter(), AzureOpenAIAdapter()):
        assert adapter.tracks_object_ids(method, path) is tracked


def test_a_fine_tuning_job_reports_itself_then_its_result_files() -> None:
    adapter = OpenAIAdapter()
    created = {
        "id": "ftjob-1",
        "object": "fine_tuning.job",
        "training_file": "file-train",
        "validation_file": "file-valid",
        "result_files": [],
    }
    assert adapter.object_ids_from_body("POST", "/v1/fine_tuning/jobs", created) == ("ftjob-1",)
    done = {**created, "status": "succeeded", "result_files": ["file-metrics", 7, ""]}
    job = "/v1/fine_tuning/jobs/ftjob-1"
    assert adapter.object_ids_from_body("GET", job, done) == ("file-metrics",)
    assert adapter.object_ids_from_body("POST", job + "/cancel", done) == ("file-metrics",)
    assert adapter.object_ids_from_body("GET", job, {**done, "result_files": None}) == ()
    assert adapter.object_ids_from_body("GET", job, ["x"]) == ()


async def test_fine_tuning_jobs_are_reported_on_the_pass_through_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = Router()
    job = {
        "id": "ftjob-9",
        "object": "fine_tuning.job",
        "training_file": "file-t",
        "result_files": [],
    }

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json=job)
        return httpx.Response(200, json={**job, "result_files": ["file-results"]})

    app = _app(monkeypatch, tmp_path, router, respond, "openai")
    created = await _send(
        app, "POST", "/v1/fine_tuning/jobs", {"model": "m", "training_file": "file-t"}
    )
    assert created.status_code == 200 and created.json() == job  # untouched
    read = await _send(app, "GET", "/v1/fine_tuning/jobs/ftjob-9")
    assert read.status_code == 200
    assert router.objects == [("ftjob-9", SESSION), ("file-results", SESSION)]
