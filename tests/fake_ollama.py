"""A fake Ollama server for the corpus-tooling tests (httpx.MockTransport;
no network). Answers /api/show, /api/tags and /api/chat; chat answers come
from a callable fed the request body, and every request is recorded."""

import json
from collections.abc import Callable
from typing import Any

import httpx

APACHE = (
    "\n                                 Apache License\n                           Version"
    " 2.0, January 2004\n                        http://www.apache.org/licenses/\n"
)
DIGEST = "a" * 64

Answer = Callable[[dict[str, Any]], str]


class FakeOllama:
    def __init__(
        self,
        answer: Answer,
        *,
        models: tuple[str, ...] = ("gemma4:e4b",),
        license_text: object = APACHE,
        show_extra: dict[str, Any] | None = None,
        chat_status: int = 200,
    ) -> None:
        self.answer = answer
        self.models = models
        self.license_text = license_text
        self.show_extra = show_extra or {}
        self.chat_status = chat_status
        self.requests: list[tuple[str, dict[str, Any] | None]] = []

    @property
    def chats(self) -> list[dict[str, Any]]:
        return [body for path, body in self.requests if path == "/api/chat" and body]

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.url.path, body))
        if request.url.path == "/api/show":
            assert body is not None
            if body["model"] not in self.models:
                return httpx.Response(404, json={"error": "model not found"})
            return httpx.Response(200, json={"license": self.license_text, **self.show_extra})
        if request.url.path == "/api/tags":
            listed = [{"name": m, "model": m, "digest": DIGEST} for m in self.models]
            return httpx.Response(200, json={"models": listed})
        if request.url.path == "/api/chat":
            assert body is not None
            if self.chat_status != 200:
                return httpx.Response(self.chat_status, text="internal error: a prompt echo")
            content = self.answer(body)
            return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})
        return httpx.Response(404)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
