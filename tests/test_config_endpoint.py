"""The /__llm-redact/config editor endpoint: defense stack + apply semantics."""

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from llm_redact.config import load_config
from llm_redact.proxy import CSRF_HEADER, create_app

received: dict[str, Any] = {}

BASE_TOML = """
[providers.anthropic]
upstream_base_url = "http://upstream"

[detection]
enabled = ["email", "ipv4"]

[vault]
session = "editor-test"
"""


def _fake_upstream() -> Starlette:
    async def messages(request: Request) -> JSONResponse:
        received["anthropic"] = await request.json()
        return JSONResponse({"content": [{"type": "text", "text": "ok"}], "role": "assistant"})

    return Starlette(routes=[Route("/v1/messages", messages, methods=["POST"])])


def _make_client(
    config_file: Path | None, *, base_url: str = "http://127.0.0.1:8787"
) -> httpx.AsyncClient:
    received.clear()
    config = load_config(config_file) if config_file is not None else load_config()
    app = create_app(
        config,
        upstream_transport=httpx.ASGITransport(app=_fake_upstream()),
        config_path=config_file,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(BASE_TOML)
    return path


async def _token(client: httpx.AsyncClient) -> str:
    response = await client.get("/__llm-redact/config")
    assert response.status_code == 200
    return str(response.json()["csrf_token"])


async def test_get_shape(config_file: Path) -> None:
    client = _make_client(config_file)
    payload = (await client.get("/__llm-redact/config")).json()
    assert payload["config_path"] == str(config_file)
    assert payload["config_file_exists"] is True
    assert payload["editable"]["detection"]["enabled"] == ["email", "ipv4"]
    assert "email" in payload["builtin_rules"]
    assert payload["readonly"]["vault"]["session"] == "editor-test"
    assert len(payload["csrf_token"]) > 20
    assert payload["warnings"]


async def test_post_applies_persists_and_changes_behavior(config_file: Path) -> None:
    client = _make_client(config_file)
    body = {"model": "m", "messages": [{"role": "user", "content": "mail jane@corp.example"}]}

    await client.post("/v1/messages", json=body)
    assert "jane@corp.example" not in json.dumps(received["anthropic"])  # redacted today

    token = await _token(client)
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={"config": {"detection": {"enabled": ["email"], "allowlist": ["jane@corp.example"]}}},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["applied"] is True
    assert result["restart_required"] == []
    assert result["backup"] == str(config_file) + ".bak"

    # Persisted: reloading the rewritten file yields the new values, and the
    # readonly [vault] section from the original file survived untouched.
    reloaded = load_config(config_file)
    assert reloaded.detection.allowlist == ("jane@corp.example",)
    assert reloaded.vault.session == "editor-test"
    assert (config_file.with_name("config.toml.bak")).read_text() == BASE_TOML

    # Applied live: the allowlisted email now passes through unredacted.
    await client.post("/v1/messages", json=body)
    assert "jane@corp.example" in json.dumps(received["anthropic"])


async def test_post_without_file_writes_xdg_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    client = _make_client(None)
    target = tmp_path / "xdg" / "llm-redact" / "config.toml"

    payload = (await client.get("/__llm-redact/config")).json()
    assert payload["config_path"] == str(target)
    assert payload["config_file_exists"] is False

    token = await _token(client)
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={"config": {"rehydration": {"fuzzy": False}}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["backup"] is None
    assert target.exists()
    if os.name == "posix":
        assert (target.stat().st_mode & 0o777) == 0o600
    assert load_config(target).rehydration.fuzzy is False


async def test_ner_edits_via_endpoint(config_file: Path) -> None:
    # The dashboard's NER card drives exactly this payload shape.
    client = _make_client(config_file)
    token = await _token(client)

    # Disabled NER config round-trips regardless of installed extras.
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={
            "config": {
                "detection": {
                    "ner": {
                        "enabled": False,
                        "backend": "presidio",
                        "entities": ["PERSON", "EMAIL_ADDRESS"],
                        "max_chars": 5000,
                        "score_threshold": 0.4,
                    }
                }
            }
        },
    )
    assert response.status_code == 200, response.text
    ner = load_config(config_file).detection.ner
    assert (ner.backend, ner.enabled, ner.score_threshold) == ("presidio", False, 0.4)
    assert ner.entities == ("PERSON", "EMAIL_ADDRESS")

    payload = (await client.get("/__llm-redact/config")).json()
    assert payload["editable"]["detection"]["ner"]["backend"] == "presidio"

    # score_threshold with the spacy backend is a validation error, surfaced
    # as a clean 400 (the dashboard drops the key client-side, but the
    # endpoint must hold the line for hand-written payloads).
    rejected = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={"config": {"detection": {"ner": {"backend": "spacy", "score_threshold": 0.4}}}},
    )
    assert rejected.status_code == 400
    assert "score_threshold" in rejected.json()["error"]


async def test_allowlist_by_type_via_editor(config_file: Path) -> None:
    client = _make_client(config_file)
    body = {"model": "m", "messages": [{"role": "user", "content": "mail jane@corp.example"}]}
    token = await _token(client)
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={"config": {"detection": {"allowlist_by_type": {"EMAIL": ["jane@corp.example"]}}}},
    )
    assert response.status_code == 200, response.text
    assert load_config(config_file).detection.allowlist_by_type == (
        ("EMAIL", ("jane@corp.example",)),
    )
    payload = (await client.get("/__llm-redact/config")).json()
    assert payload["editable"]["detection"]["allowlist_by_type"] == {"EMAIL": ["jane@corp.example"]}
    # Applied live: the email now passes through as its allowed type.
    await client.post("/v1/messages", json=body)
    assert "jane@corp.example" in json.dumps(received["anthropic"])


async def test_provider_disable_via_editor_fails_closed(config_file: Path) -> None:
    client = _make_client(config_file)
    body = {"model": "m", "messages": [{"role": "user", "content": "mail jane@corp.example"}]}

    # Enabled today: the request reaches the fake upstream (redacted).
    assert (await client.post("/v1/messages", json=body)).status_code == 200
    assert "anthropic" in received

    token = await _token(client)
    payload = (await client.get("/__llm-redact/config")).json()
    assert payload["editable"]["providers"]["anthropic"]["enabled"] is True
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={
            "config": {
                "providers": {
                    "anthropic": {"upstream_base_url": "http://upstream", "enabled": False}
                }
            }
        },
    )
    assert response.status_code == 200, response.text
    assert load_config(config_file).providers["anthropic"].enabled is False

    # Applied live and fail-closed: the matched route is answered by the
    # proxy, and pass-through traffic inferred to the provider is too —
    # neither reaches the upstream.
    received.clear()
    blocked = await client.post("/v1/messages", json=body)
    assert blocked.status_code == 502
    assert "disabled" in json.dumps(blocked.json())
    # Pass-through traffic inferred to the disabled provider is refused too
    # (/v1/complete has no adapter and defaults to anthropic).
    passthrough = await client.get("/v1/complete")
    assert passthrough.status_code == 502
    assert received == {}


async def test_ner_enable_without_extra_is_clean_400(config_file: Path) -> None:
    try:
        import gliner  # noqa: F401

        pytest.skip("gliner installed; missing-dependency path not testable")
    except ImportError:
        pass
    client = _make_client(config_file)
    token = await _token(client)
    # The dry-run detector build catches the missing extra BEFORE anything
    # is written or applied; the error names the install command.
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={"config": {"detection": {"ner": {"enabled": True, "backend": "gliner"}}}},
    )
    assert response.status_code == 400
    assert "uv sync --extra gliner" in response.json()["error"]
    # Nothing was persisted: the file still parses to NER disabled.
    assert load_config(config_file).detection.ner.enabled is False


async def test_hostile_host_rejected_even_for_get(config_file: Path) -> None:
    # DNS rebinding: the attacker's page reaches 127.0.0.1 but Host carries
    # their domain — the token must be unobtainable.
    client = _make_client(config_file, base_url="http://evil.example")
    assert (await client.get("/__llm-redact/config")).status_code == 403
    posted = await client.post(
        "/__llm-redact/config", headers={CSRF_HEADER: "x"}, json={"config": {}}
    )
    assert posted.status_code == 403


@pytest.mark.parametrize("origin", ["https://evil.example", "http://evil.example", "null"])
async def test_hostile_origin_rejected(config_file: Path, origin: str) -> None:
    client = _make_client(config_file)
    response = await client.get("/__llm-redact/config", headers={"origin": origin})
    assert response.status_code == 403


async def test_local_origin_accepted(config_file: Path) -> None:
    client = _make_client(config_file)
    response = await client.get("/__llm-redact/config", headers={"origin": "http://127.0.0.1:8787"})
    assert response.status_code == 200


async def test_missing_or_wrong_csrf_token(config_file: Path) -> None:
    client = _make_client(config_file)
    no_token = await client.post("/__llm-redact/config", json={"config": {}})
    assert no_token.status_code == 403
    wrong = await client.post(
        "/__llm-redact/config", headers={CSRF_HEADER: "nope"}, json={"config": {}}
    )
    assert wrong.status_code == 403


async def test_options_preflight_dies_without_cors(config_file: Path) -> None:
    client = _make_client(config_file)
    response = await client.options(
        "/__llm-redact/config",
        headers={
            "origin": "http://127.0.0.1:8787",
            "access-control-request-method": "POST",
            "access-control-request-headers": CSRF_HEADER,
        },
    )
    assert response.status_code == 405
    assert not any(name.lower().startswith("access-control-") for name in response.headers)


async def test_wrong_content_type(config_file: Path) -> None:
    client = _make_client(config_file)
    token = await _token(client)
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token, "content-type": "text/plain"},
        content=b'{"config": {}}',
    )
    assert response.status_code == 415


async def test_oversized_body_rejected(config_file: Path) -> None:
    client = _make_client(config_file)
    token = await _token(client)
    huge = {"config": {"detection": {"allowlist": ["x" * (1024 * 1024 + 10)]}}}
    response = await client.post("/__llm-redact/config", headers={CSRF_HEADER: token}, json=huge)
    assert response.status_code == 413


@pytest.mark.parametrize(
    ("edits", "expected_error_bit"),
    [
        ({"detection": {"enabled": ["not_a_rule"]}}, "not_a_rule"),
        ({"detection": {"custom_rules": [{"name": "b", "type": "T", "pattern": "("}]}}, ""),
        ({"host": "0.0.0.0"}, "restart"),
        ({"nonsense": 1}, "unknown key"),
        ({"max_body_bytes": -5}, "positive"),
        ({"detection": {"allowlist_patterns": ["("]}}, ""),
    ],
)
async def test_invalid_edits_rejected_and_nothing_written(
    config_file: Path, edits: dict[str, Any], expected_error_bit: str
) -> None:
    client = _make_client(config_file)
    token = await _token(client)
    before = config_file.read_text()
    response = await client.post(
        "/__llm-redact/config", headers={CSRF_HEADER: token}, json={"config": edits}
    )
    assert response.status_code == 400
    assert expected_error_bit in response.json()["error"]
    assert config_file.read_text() == before  # untouched
    assert not config_file.with_name("config.toml.bak").exists()


async def test_invalid_json_body(config_file: Path) -> None:
    client = _make_client(config_file)
    token = await _token(client)
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token, "content-type": "application/json"},
        content=b"not json [",
    )
    assert response.status_code == 400


async def test_corrupt_on_disk_toml_conflicts(config_file: Path) -> None:
    client = _make_client(config_file)
    token = await _token(client)
    config_file.write_text("this is not toml [ [")
    response = await client.post(
        "/__llm-redact/config", headers={CSRF_HEADER: token}, json={"config": {}}
    )
    assert response.status_code == 409


async def test_other_reserved_endpoints_stay_get_only(config_file: Path) -> None:
    client = _make_client(config_file)
    assert (await client.post("/__llm-redact/status", content=b"{}")).status_code == 405
    assert (await client.post("/__llm-redact/metrics", content=b"{}")).status_code == 405


async def test_modes_edit_hot_applies_and_persists(config_file: Path) -> None:
    client = _make_client(config_file)
    body = {"model": "m", "messages": [{"role": "user", "content": "mail jane@corp.example"}]}

    payload = (await client.get("/__llm-redact/config")).json()
    assert payload["editable"]["detection"]["modes"] == {}

    token = await _token(client)
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={"config": {"detection": {"enabled": ["email"], "modes": {"email": "block"}}}},
    )
    assert response.status_code == 200, response.text

    blocked = await client.post("/v1/messages", json=body)
    assert blocked.status_code == 400
    assert "anthropic" not in received  # applied live, fail closed

    reloaded = load_config(config_file)
    assert reloaded.detection.modes == (("email", "block"),)
    view = (await client.get("/__llm-redact/config")).json()
    assert view["editable"]["detection"]["modes"] == {"email": "block"}


@pytest.mark.parametrize(
    "modes",
    [
        {"no_such_rule": "warn"},  # unknown rule name
        {"github_token": "warn", "github_fine_grained_pat": "block"},  # shared-type conflict
        {"email": "drop"},  # invalid mode value
    ],
)
async def test_bad_modes_rejected_and_nothing_written(
    config_file: Path, modes: dict[str, str]
) -> None:
    client = _make_client(config_file)
    token = await _token(client)
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={"config": {"detection": {"modes": modes}}},
    )
    assert response.status_code == 400
    assert config_file.read_text() == BASE_TOML  # untouched
    assert not config_file.with_name("config.toml.bak").exists()


async def test_deny_strings_edit_hot_applies(config_file: Path) -> None:
    client = _make_client(config_file)
    body = {"model": "m", "messages": [{"role": "user", "content": "codename aurora here"}]}

    await client.post("/v1/messages", json=body)
    assert "aurora" in json.dumps(received["anthropic"])  # not redacted today

    payload = (await client.get("/__llm-redact/config")).json()
    assert payload["editable"]["detection"]["deny_strings"] == []

    token = await _token(client)
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={
            "config": {
                "detection": {
                    "enabled": ["email"],
                    "deny_strings": [{"value": "aurora", "case_sensitive": False, "type": "DENY"}],
                }
            }
        },
    )
    assert response.status_code == 200, response.text

    await client.post("/v1/messages", json=body)
    flat = json.dumps(received["anthropic"], ensure_ascii=False)
    assert "aurora" not in flat  # applied live
    assert "«DENY_001»" in flat

    reloaded = load_config(config_file)
    assert len(reloaded.detection.deny_strings) == 1
    view = (await client.get("/__llm-redact/config")).json()
    assert view["editable"]["detection"]["deny_strings"] == [
        {"value": "aurora", "case_sensitive": False, "type": "DENY"}
    ]


async def test_bad_deny_edit_rejected(config_file: Path) -> None:
    client = _make_client(config_file)
    token = await _token(client)
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={"config": {"detection": {"deny_strings": [{"value": ""}]}}},
    )
    assert response.status_code == 400
    assert config_file.read_text() == BASE_TOML


async def test_editor_post_roundtrips_custom_rule_gate_fields(config_file: Path) -> None:
    """2.0 rider: a POST that includes a custom rule's validator/required/
    anchors must persist them — the dashboard previously dropped all three
    on save, silently stripping the rule's checksum gate."""
    client = _make_client(config_file)
    token = await _token(client)
    rule = {
        "name": "corp_token",
        "type": "CORP_TOKEN",
        "pattern": "corp-[0-9]{12}",
        "priority": 50,
        "validator": "luhn",
        "required": ["corp-"],
        "anchors": ["corp-"],
    }
    response = await client.post(
        "/__llm-redact/config",
        json={"config": {"detection": {"enabled": ["email"], "custom_rules": [rule]}}},
        headers={"x-llm-redact-csrf": token},
    )
    assert response.status_code == 200, response.text
    written = config_file.read_text()
    assert 'validator = "luhn"' in written
    assert "corp-" in written
    fetched = (await client.get("/__llm-redact/config")).json()
    saved = fetched["editable"]["detection"]["custom_rules"][0]
    assert saved["validator"] == "luhn"
    assert saved["required"] == ["corp-"]
    assert saved["anchors"] == ["corp-"]


async def test_get_includes_config_fingerprint(config_file: Path) -> None:
    client = _make_client(config_file)
    payload = (await client.get("/__llm-redact/config")).json()
    assert isinstance(payload["config_fingerprint"], str)
    assert len(payload["config_fingerprint"]) == 64  # sha256 hex


async def test_stale_fingerprint_post_is_rejected_409(config_file: Path) -> None:
    # The editor's stale-form guard (3.3): a Save against a file that changed
    # since the form loaded (CLI edit, SIGHUP rewrite, another tab) must not
    # silently last-writer-wins over it.
    client = _make_client(config_file)
    payload = (await client.get("/__llm-redact/config")).json()
    token = str(payload["csrf_token"])
    fingerprint = str(payload["config_fingerprint"])

    # Someone edits the file behind the editor's back (top-level key first —
    # TOML top-level keys must precede any table header).
    config_file.write_text("max_body_bytes = 999999\n" + BASE_TOML)

    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={"config": {"rehydration": {"fuzzy": False}}, "fingerprint": fingerprint},
    )
    assert response.status_code == 409
    assert "changed since" in response.json()["error"]
    # The out-of-band edit survived untouched.
    assert "max_body_bytes = 999999" in config_file.read_text()

    # A fresh GET yields the new fingerprint, and the same save then applies.
    fresh = (await client.get("/__llm-redact/config")).json()
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: str(fresh["csrf_token"])},
        json={
            "config": {"rehydration": {"fuzzy": False}},
            "fingerprint": fresh["config_fingerprint"],
        },
    )
    assert response.status_code == 200, response.text


async def test_post_without_fingerprint_still_applies(config_file: Path) -> None:
    # The fingerprint is optional (older dashboards, scripts): omitting it
    # keeps the pre-3.3 last-writer-wins behavior.
    client = _make_client(config_file)
    token = await _token(client)
    response = await client.post(
        "/__llm-redact/config",
        headers={CSRF_HEADER: token},
        json={"config": {"rehydration": {"fuzzy": False}}},
    )
    assert response.status_code == 200, response.text
