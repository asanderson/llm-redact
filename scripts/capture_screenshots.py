#!/usr/bin/env python3
"""Capture the dashboard and config-editor screenshots for the docs.

Dev-only; the committed PNGs live in docs/screenshots/ (deliberately
separate from the mermaid-rendered docs/diagrams/). Regenerate with:

    uv run --with playwright python scripts/capture_screenshots.py

Playwright needs a Chromium; if it cannot download one (sandboxed CI),
point it at an existing binary:

    LLM_REDACT_CHROMIUM=/usr/bin/chromium uv run --with playwright ...

The proxy is seeded with FIXTURE traffic against an in-process fake
upstream — never run this against a proxy that has handled real secrets.
"""

import os
import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from llm_redact.config import Config, ProviderConfig
from llm_redact.proxy import create_app

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
EMAIL = "jane.doe@corp.example"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _serve(app: object, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    return server


def _fake_upstream_app() -> object:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def messages(request: Request) -> JSONResponse:
        body = await request.json()
        text = str(body.get("messages", ""))
        tokens = [w for w in text.split() if w.startswith("«")]
        echoed = " and ".join(tokens) or "nothing"
        return JSONResponse(
            {"role": "assistant", "content": [{"type": "text", "text": f"noted {echoed}"}]}
        )

    return Starlette(routes=[Route("/v1/messages", messages, methods=["POST"])])


def main() -> int:
    upstream_port = _free_port()
    proxy_port = _free_port()
    upstream = _serve(_fake_upstream_app(), upstream_port)
    proxy = _serve(
        create_app(
            Config(
                providers={
                    "anthropic": ProviderConfig(
                        upstream_base_url=f"http://127.0.0.1:{upstream_port}"
                    )
                }
            )
        ),
        proxy_port,
    )
    base = f"http://127.0.0.1:{proxy_port}"

    # Seed fixture traffic so counters, the recent table, and the session
    # summary all show real-looking numbers.
    with httpx.Client(base_url=base) as client:
        for text in (
            f"email {EMAIL} the launch checklist",
            f"rotate {AWS_KEY} before the audit",
            f"cc {EMAIL} and archive the thread",
        ):
            client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": text}],
                },
            ).raise_for_status()

    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    executable = os.environ.get("LLM_REDACT_CHROMIUM")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=executable or None)
        page = browser.new_page(viewport={"width": 1280, "height": 940})
        page.goto(f"{base}/__llm-redact/")
        page.wait_for_timeout(1200)  # one poll cycle populates the tables
        page.screenshot(path=str(OUT / "dashboard.png"))
        editor = page.locator("#config-form")
        editor.scroll_into_view_if_needed()
        page.wait_for_timeout(200)
        editor.screenshot(path=str(OUT / "config-editor.png"))

        # Redaction-preview card: type representative text, run the dry-run,
        # and capture the masked output + per-type summary.
        preview = page.locator("[data-preview]")
        preview.scroll_into_view_if_needed()
        page.fill(
            "#preview-input",
            f"Email {EMAIL} and rotate {AWS_KEY} before the launch review.",
        )
        page.click("#preview-run")
        page.wait_for_timeout(400)
        preview.screenshot(path=str(OUT / "preview.png"))
        browser.close()

    proxy.should_exit = True
    upstream.should_exit = True
    print(f"wrote dashboard.png, config-editor.png, and preview.png to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
