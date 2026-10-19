"""Exercise the real optional WebUI stack without provider keys or real data.

Requires Docker Compose v2 and locally built proxy/fake images (see
docs/openwebui.md). Owns a uniquely named temporary project and removes ONLY
that project's containers/volumes. Uses stdlib; no WebUI dependency in the wheel.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EMAIL = "jane.doe@corp.example"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def read_sse(lines: Iterable[bytes]) -> str:
    text = []
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            return "".join(text)
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError("WebUI returned a streaming error")
        for choice in event.get("choices", []):
            text.append(choice.get("delta", {}).get("content", ""))
    raise RuntimeError("SSE ended without its completion marker")


def check_reply(reply: str) -> None:
    if EMAIL not in reply or AWS_KEY not in reply or "«" in reply:
        raise RuntimeError("Synthetic canaries were not restored in the WebUI reply")


def check_upstream(logs: str) -> None:
    if EMAIL in logs or AWS_KEY in logs:
        raise RuntimeError("A synthetic original leaked to the upstream")
    if "«EMAIL_001»" not in logs or "«AWS_KEY_001»" not in logs:
        raise RuntimeError("The upstream did not receive both synthetic placeholders")


def request_json(url: str, body: dict[str, Any] | None = None, token: str = "") -> Any:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        url, data=json.dumps(body).encode() if body is not None else None, headers=headers
    )
    with OPENER.open(request, timeout=30) as response:
        return json.load(response)


def wait_ready(url: str) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            request_json(url)
            return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError("A smoke service did not become ready within 180 seconds")


def completion(base: str, token: str, message: str, stream: bool) -> str:
    body = {
        "model": "fake-openwebui",
        "stream": stream,
        "messages": [{"role": "user", "content": message}],
    }
    if not stream:
        return request_json(base + "/api/chat/completions", body, token)["choices"][0]["message"][
            "content"
        ]
    request = urllib.request.Request(
        base + "/api/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    )
    with OPENER.open(request, timeout=30) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise RuntimeError("WebUI did not return an SSE response")
        return read_sse(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-image", default="llm-redact:ci")
    parser.add_argument("--fake-image", default="llm-redact-fake:ci")
    args = parser.parse_args()
    if shutil.which("docker") is None:
        parser.error("Docker with Compose v2 is required")
    project = "llm-redact-webui-smoke-" + secrets.token_hex(6)
    env = dict(os.environ)
    env.update(
        LLM_REDACT_IMAGE=args.proxy_image,
        LLM_REDACT_FAKE_IMAGE=args.fake_image,
        OPENAI_API_KEY="unused-demo-key",
        WEBUI_SECRET_KEY=secrets.token_hex(32),
        WEBUI_ADMIN_EMAIL="smoke-admin@corp.example",
        WEBUI_ADMIN_PASSWORD=secrets.token_urlsafe(32),
        OPENWEBUI_PORT="0",
        OPENWEBUI_PROXY_PORT="0",
    )
    with tempfile.TemporaryDirectory(prefix=project) as directory:
        empty_env = Path(directory) / "empty.env"
        empty_env.touch()
        command = [
            "docker",
            "compose",
            "--project-name",
            project,
            "--env-file",
            str(empty_env),
            "-f",
            str(ROOT / "compose.openwebui.yaml"),
            "-f",
            str(ROOT / "examples/openwebui-demo.compose.yaml"),
            "--profile",
            "webui",
        ]

        def compose(*arguments: str, check: bool = True) -> str:
            return subprocess.run(
                [*command, *arguments],
                env=env,
                cwd=ROOT,
                check=check,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).stdout

        def address(service: str, port: str) -> str:
            published = compose("port", service, port).strip()
            if not published.startswith("127.0.0.1:"):
                raise RuntimeError("A smoke port was not published to loopback")
            return "http://" + published

        try:
            compose("config", "--quiet")
            compose("up", "-d", "--no-build")
            base = address("webui-http", "8080")
            proxy = address("proxy", "8787")
            wait_ready(base + "/health")
            credentials = {
                "email": env["WEBUI_ADMIN_EMAIL"],
                "password": env["WEBUI_ADMIN_PASSWORD"],
            }
            token = request_json(base + "/api/v1/auths/signin", credentials)["token"]
            models = request_json(base + "/api/models", token=token)
            if not any(model["id"] == "fake-openwebui" for model in models["data"]):
                raise RuntimeError("WebUI model discovery did not traverse the proxy")
            prompt = f"Email {EMAIL}; rotate {AWS_KEY}"
            for stream in (False, True):
                check_reply(completion(base, token, prompt, stream))
            check_upstream(compose("logs", "--no-color", "fake-upstream"))
            print(
                "PASS: WebUI login, discovery, JSON/SSE restoration, upstream canaries", flush=True
            )

            # Exercise a generated model task separately; these are off in the example.
            task_config = request_json(base + "/api/v1/tasks/config", token=token)
            if task_config["ENABLE_TITLE_GENERATION"]:
                raise RuntimeError("Title generation was unexpectedly enabled")
            task_config.update(ENABLE_TITLE_GENERATION=True, TASK_MODEL_EXTERNAL="fake-openwebui")
            request_json(base + "/api/v1/tasks/config/update", task_config, token)
            title = request_json(
                base + "/api/v1/tasks/title/completions",
                {"model": "fake-openwebui", "messages": [{"role": "user", "content": prompt}]},
                token,
            )
            check_reply(title["choices"][0]["message"]["content"])
            check_upstream(compose("logs", "--no-color", "fake-upstream"))

            compose("restart", "proxy")
            wait_ready(proxy + "/__llm-redact/healthz")
            check_reply(completion(base, token, "Repeat «EMAIL_001» and «AWS_KEY_001»", False))
            compose("restart", "open-webui")
            wait_ready(base + "/health")
            token = request_json(base + "/api/v1/auths/signin", credentials)["token"]
            if request_json(base + "/api/v1/tasks/config", token=token)["ENABLE_TITLE_GENERATION"]:
                raise RuntimeError("A runtime setting overrode environment policy after restart")
            check_reply(completion(base, token, prompt, False))
            print("PASS: generated task, persistent vault, restart configuration", flush=True)

            # OFFLINE_MODE is not a firewall: verify the internal-only network too.
            compose(
                "exec",
                "-T",
                "open-webui",
                "python",
                "-c",
                "import socket\ntry:\n socket.create_connection(('1.1.1.1',443),2)\n"
                "except OSError:\n pass\nelse:\n raise SystemExit('unexpected external route')",
            )
            compose("stop", "proxy")
            try:
                completion(base, token, prompt, False)
            except urllib.error.HTTPError as error:
                if error.code < 400:
                    raise
            else:
                raise RuntimeError("WebUI succeeded with its proxy stopped")
            check_upstream(compose("logs", "--no-color", "fake-upstream"))
            print("PASS: no direct egress and no fallback when proxy is stopped", flush=True)
        except Exception as error:
            # Only this disposable synthetic stack; never print compose config/secrets.
            if isinstance(error, subprocess.CalledProcessError):
                print(error.stdout)
            print(compose("ps", "--all", check=False))
            print(compose("logs", "--no-color", "--tail", "40", check=False))
            raise
        finally:
            compose("down", "--volumes", "--remove-orphans", check=False)


if __name__ == "__main__":
    main()
