"""The acknowledged compatibility opt-out is effective, visible and reversible."""

import json
import logging
from dataclasses import replace
from typing import Any

import httpx
import pytest

from llm_redact.cli import _print_posture
from llm_redact.config import Config, ConfigError, parse_config
from llm_redact.doctor_cli import _check_posture, _Report
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"
BLOB = "gAAAAABq0eYc-AKIAIOSFODNN7EXAMPLE-opaque"


def _config(enabled: bool) -> Config:
    return parse_config(
        {
            "inject_system_note": False,
            "detection": {
                "enabled": ["email", "aws_access_key_id"],
                "encrypted_content_passthrough": enabled,
                "acknowledge_encrypted_content_risk": enabled,
            },
        },
        "test",
    )


def test_status_text_exposes_passthrough_without_claiming_full_coverage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_posture({"detection": {"encrypted_content_passthrough": True}})
    output = capsys.readouterr().out
    assert "encrypted_content" in output and "UNSCANNED" in output
    assert "all traffic redacted" not in output


def test_doctor_warns_about_acknowledged_passthrough() -> None:
    report = _Report(json_mode=True)
    _check_posture(report, _config(True))
    assert any(
        row["level"] == "WARN" and "encrypted_content" in row["message"] for row in report.rows
    )
    assert not any("no coverage opt-outs" in row["message"] for row in report.rows)


@pytest.mark.parametrize(
    "path", ["/v1/responses", "/v1/responses/compact", "/v1/responses/input_tokens"]
)
@pytest.mark.parametrize("enabled", [False, True])
async def test_actual_http_replay_and_live_status(path: str, enabled: bool) -> None:
    sent: list[dict[str, Any]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"output": []})

    app = create_app(_config(enabled), upstream_transport=httpx.MockTransport(upstream))
    body = {
        "model": "m",
        "input": [{"type": "reasoning", "encrypted_content": BLOB, "summary": [{"text": EMAIL}]}],
        "encrypted_content": EMAIL,
        "metadata": {"encrypted_content": EMAIL},
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(path, json=body)
        assert response.status_code == 200, response.text
        status = (await client.get("/__llm-redact/status")).json()
    (forwarded,) = sent
    assert (forwarded["input"][0]["encrypted_content"] == BLOB) is enabled
    assert (forwarded["encrypted_content"] == EMAIL) is enabled
    assert EMAIL not in json.dumps(forwarded["metadata"])
    assert EMAIL not in json.dumps(forwarded["input"][0]["summary"])
    assert status["detection"]["encrypted_content_passthrough"] is enabled


def test_startup_and_reload_validate_acknowledgement_and_keep_old_policy_on_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    enabled = _config(True)
    unsafe = replace(
        enabled, detection=replace(enabled.detection, acknowledge_encrypted_content_risk=False)
    )
    with pytest.raises(ConfigError, match="acknowledge_encrypted_content_risk"):
        create_app(unsafe)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        app = create_app(enabled)
    assert any("encrypted_content" in record.message for record in caplog.records)
    state = app.state.proxy
    with pytest.raises(ConfigError, match="acknowledge_encrypted_content_risk"):
        state.apply_config(unsafe)
    assert state.config == enabled
    assert state.redactor.encrypted_content_passthrough
    state.apply_config(_config(False))
    assert not state.redactor.encrypted_content_passthrough
    state.apply_config(enabled)
    assert state.redactor.encrypted_content_passthrough


@pytest.mark.parametrize("enabled", [False, True])
async def test_listing_keeps_request_policy_when_reload_happens_upstream(enabled: bool) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        state.apply_config(_config(not enabled))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "file_1", "encrypted_content": token, "text": token}],
            },
        )

    app = create_app(_config(enabled), upstream_transport=httpx.MockTransport(upstream))
    state = app.state.proxy
    token = state.redactor.redact_text(EMAIL)
    state._listing_item_session = lambda _object_id: state.config.vault.session
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get("/v1/files")
    assert response.status_code == 200, response.text
    (item,) = response.json()["data"]
    assert item["encrypted_content"] == (token if enabled else EMAIL)
    assert item["text"] == EMAIL
