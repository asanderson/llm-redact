"""Real Chromium acceptance for the disposable Open WebUI smoke project.

Invoked by openwebui_smoke.py --browser; Playwright is a separate development
tool, never part of the proxy runtime. No mocked browser requests or responses.
"""

from __future__ import annotations

from pathlib import Path


def run_browser_smoke(
    base: str,
    credentials: dict[str, str],
    prompt: str,
    expected_values: tuple[str, ...],
    artifacts_dir: Path | None = None,
) -> None:
    from playwright.sync_api import expect, sync_playwright

    if artifacts_dir:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    frames: list[str] = []
    streaming_requests: list[bool] = []
    foreign_requests: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, locale="en-US")
        page.set_default_timeout(30_000)

        def websocket_opened(socket) -> None:
            if socket.url.startswith(base.replace("http", "ws", 1) + "/ws/socket.io/"):
                socket.on(
                    "framereceived",
                    lambda data: frames.append(data) if isinstance(data, str) else None,
                )

        def request_sent(request) -> None:
            if request.method == "POST" and request.url.split("?")[0].endswith("/chat/completions"):
                if not request.url.startswith(base + "/"):
                    foreign_requests.append(request.url)
                streaming_requests.append(request.post_data_json.get("stream") is True)

        page.on("websocket", websocket_opened)
        page.on("request", request_sent)
        try:
            page.goto(base + "/auth")
            page.locator("#email").fill(credentials["email"])
            page.locator("#password").fill(credentials["password"])
            page.get_by_role("button", name="Sign in", exact=True).click()
            # This is the administrator's first browser login in a fresh volume.
            page.get_by_role("button", name="Okay, Let's Go!", exact=True).click()
            page.locator('[id^="model-selector-"][id$="-button"]').first.click()
            page.get_by_role("option").filter(has_text="fake-openwebui").first.click()
            page.locator("#chat-input").fill(prompt)
            page.locator("#send-message-button").click()
            response = page.locator("#response-content-container").last
            for value in expected_values:
                expect(response).to_contain_text(value, timeout=60_000)
            expect(response).not_to_contain_text("«")
            if not streaming_requests or not all(streaming_requests) or foreign_requests:
                raise RuntimeError("Browser chat did not use the local streaming backend")
            # Socket.IO's namespace acknowledgement proves more than a TCP open.
            if not any(frame.startswith("40") for frame in frames):
                raise RuntimeError("Browser did not complete its WebSocket handshake through Caddy")
            if artifacts_dir:
                page.screenshot(path=str(artifacts_dir / "chat.png"), full_page=True)
            print(
                "PASS: browser login, model selection, streaming chat, WebSocket handshake",
                flush=True,
            )
        except Exception:
            if artifacts_dir:
                page.screenshot(path=str(artifacts_dir / "browser-failure.png"), full_page=True)
                (artifacts_dir / "browser-failure.html").write_text(
                    page.content(), encoding="utf-8"
                )
            raise
        finally:
            browser.close()
