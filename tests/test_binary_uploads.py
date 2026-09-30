"""[detection] binary_uploads: the knob and its honesty surfaces.

A binary file part of an upload (upload_content.classify_file) cannot be
redacted. With the client's own key it is forwarded unscanned ("forward",
the default) or refused ("refuse"); under a credential the proxy holds it is
always refused. The knob parses strictly, round-trips through the emitter,
is hot on reload, and every forwarded part is counted: /status, /metrics,
the ``llm-redact status`` posture block and a doctor line stating the knob.
"""

from __future__ import annotations

import tomllib
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ConfigError, ProviderConfig, parse_config
from llm_redact.config_write import emit_config_toml
from llm_redact.detection.engine import DetectionConfig
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"
PDF = b"%PDF-1.7\n(" + EMAIL.encode() + b")\n%%EOF\n"
HEADERS = {"content-type": "multipart/form-data; boundary=b", "authorization": "Bearer t"}


def _upload(content: bytes = PDF, filename: str = "a.pdf") -> bytes:
    return (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="'
        + filename.encode()
        + b'"\r\n\r\n'
        + content
        + b"\r\n--b--\r\n"
    )


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"id": "file-1"})


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


# --- config -----------------------------------------------------------------------------


def test_default_is_forward() -> None:
    assert Config().detection.binary_uploads == "forward"
    assert parse_config({}, "t").detection.binary_uploads == "forward"


@pytest.mark.parametrize("mode", ["forward", "refuse"])
def test_parses_and_round_trips(mode: str) -> None:
    config = parse_config({"detection": {"binary_uploads": mode}}, "t")
    assert config.detection.binary_uploads == mode
    emitted = emit_config_toml(config)
    assert f'binary_uploads = "{mode}"' in emitted
    assert parse_config(tomllib.loads(emitted), "t") == config
    # The default is annotated: those bytes go upstream unscanned.
    assert ("UNSCANNED" in emitted) is (mode == "forward")


@pytest.mark.parametrize("bad", ["Forward", "allow", "", 1, True, ["forward"]])
def test_unknown_values_are_refused(bad: Any) -> None:
    with pytest.raises(ConfigError, match=r"\[detection\] binary_uploads must be one of"):
        parse_config({"detection": {"binary_uploads": bad}}, "t")


async def test_the_knob_is_hot_on_reload() -> None:
    upstream = Upstream()
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    state = app.state.proxy
    async with _client(app) as client:
        assert (await client.post("/v1/files", content=_upload(), headers=HEADERS)).is_success
        restart = state.apply_config(Config(detection=DetectionConfig(binary_uploads="refuse")))
        assert restart == []
        refused = await client.post("/v1/files", content=_upload(), headers=HEADERS)
        state.apply_config(Config())
        again = await client.post("/v1/files", content=_upload(), headers=HEADERS)
    assert refused.status_code == 400 and again.status_code == 200
    assert len(upstream.requests) == 2
    assert state.unscanned_uploads == {"openai": 2}


# --- honesty surfaces ---------------------------------------------------------------------


async def test_counted_per_provider_in_status_and_metrics() -> None:
    upstream = Upstream()
    config = Config(
        providers={**Config().providers, "custom:lm": ProviderConfig("http://127.0.0.1:1234/v1")}
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
        assert status["unscanned_uploads_total"] == {}
        metrics = (await client.get("/__llm-redact/metrics")).text
        assert "# TYPE llm_redact_unscanned_uploads_total counter" in metrics
        for path in ("/v1/files", "/v1/files", "/custom/lm/v1/files"):
            assert (await client.post(path, content=_upload(), headers=HEADERS)).is_success
        # A text file is redacted, not counted.
        text = _upload(b"notes " + EMAIL.encode(), "notes.txt")
        assert (await client.post("/v1/files", content=text, headers=HEADERS)).is_success
        status = (await client.get("/__llm-redact/status")).json()
        metrics = (await client.get("/__llm-redact/metrics")).text
    assert status["unscanned_uploads_total"] == {"openai": 2, "custom:lm": 1}
    assert 'llm_redact_unscanned_uploads_total{provider="openai"} 2' in metrics
    assert 'llm_redact_unscanned_uploads_total{provider="custom:lm"} 1' in metrics
    assert EMAIL.encode() not in upstream.requests[-1].content


def test_status_posture_line_only_when_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.cli import _print_posture

    base: dict[str, Any] = {"warnings_total": {}, "detection": {}, "audit": {}}
    _print_posture({**base, "unscanned_uploads_total": {}})
    assert "all traffic redacted" in capsys.readouterr().out
    _print_posture({**base, "unscanned_uploads_total": {"openai": 3, "azure": 1}})
    out = capsys.readouterr().out
    assert "binary uploads: azure×1 openai×3 file part(s) forwarded UNSCANNED" in out


@pytest.mark.parametrize(
    ("mode", "text"),
    [
        ("forward", "binary file uploads (PDF, images, archives) sent with the client's own key"),
        ("refuse", 'binary_uploads = "refuse": binary file uploads are refused (400)'),
    ],
)
def test_doctor_states_the_knob_as_a_pass(mode: str, text: str) -> None:
    from llm_redact.doctor_cli import _check_posture, _Report

    report = _Report(json_mode=True)
    _check_posture(report, Config(detection=DetectionConfig(binary_uploads=mode)))
    rows = [row for row in report.rows if "binary_uploads" in row["message"]]
    assert len(rows) == 1 and rows[0]["level"] == "PASS" and text in rows[0]["message"]
    assert not report.failed


# --- a printable word is no binary signature ----------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        b"ID3,name,email\nID3001,Jane," + EMAIL.encode() + b"\n",
        b"The ftyp box notes: " + EMAIL.encode() + b"\n",
        b"RIFF notes " + EMAIL.encode(),
        b"GIF89a caption by " + EMAIL.encode(),
        b"OggS fLaC wOFF wOF2 " + EMAIL.encode(),
    ],
)
async def test_text_opening_with_a_media_word_is_redacted_not_forwarded(content: bytes) -> None:
    upstream = Upstream()
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        reply = await client.post(
            "/v1/files", content=_upload(content, "rows.csv"), headers=HEADERS
        )
        status = (await client.get("/__llm-redact/status")).json()
    assert reply.status_code == 200
    (sent,) = upstream.requests
    assert EMAIL.encode() not in sent.content
    assert "«EMAIL_001»".encode() in sent.content
    assert status["unscanned_uploads_total"] == {}
