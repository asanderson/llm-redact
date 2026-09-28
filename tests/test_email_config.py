"""[email] config SHAPE: SMTP transport security + the OAuth 2.0 (XOAUTH2)
keys llm-redact-pro's verification-email sender reads. Parse validation,
emitter round-trip and the value-free doctor row. The sender itself (and
every token fetch) lives in llm-redact-pro."""

import tomllib
from typing import Any

import pytest

from llm_redact.config import Config, ConfigError, EmailConfig, parse_config
from llm_redact.config_write import emit_config_toml
from llm_redact.doctor_cli import _check_email, _Report

_BASE = {"smtp_host": "smtp.example", "from_address": "llm-redact@corp.example"}
_REFRESH = {
    **_BASE,
    "auth": "oauth",
    "oauth_provider": "refresh_token",
    "oauth_token_url": "https://login.example/oauth2/token",
    "oauth_client_id": "client-123",
    "oauth_refresh_token_env": "SMTP_REFRESH_TOKEN",
}


def _email(section: dict[str, Any]) -> EmailConfig:
    return parse_config({"email": section}, "test").email


def _round_trip(config: Config) -> Config:
    return parse_config(tomllib.loads(emit_config_toml(config)), "round-trip")


def test_defaults_are_password_auth_over_starttls() -> None:
    email = _email(dict(_BASE))
    assert email.auth == "password"
    assert email.starttls is True and email.implicit_tls is False
    assert email.tls == "starttls"
    assert email.oauth_provider is None


def test_implicit_tls_turns_starttls_off_by_default() -> None:
    email = _email({**_BASE, "implicit_tls": True, "smtp_port": 465})
    assert email.implicit_tls is True and email.starttls is False
    assert email.tls == "implicit"
    assert _email({**_BASE, "starttls": False}).tls == "none"


def test_implicit_tls_and_starttls_are_exclusive() -> None:
    with pytest.raises(ConfigError, match="mutually exclusive"):
        _email({**_BASE, "implicit_tls": True, "starttls": True})


def test_implicit_tls_must_be_a_boolean() -> None:
    with pytest.raises(ConfigError, match="implicit_tls must be a boolean"):
        _email({**_BASE, "implicit_tls": "yes"})


def test_unknown_auth_mode_is_refused() -> None:
    with pytest.raises(ConfigError, match='auth must be "password" or "oauth"'):
        _email({**_BASE, "auth": "kerberos"})


def test_oauth_keys_require_oauth_mode() -> None:
    with pytest.raises(ConfigError, match='oauth_provider require auth = "oauth"'):
        _email({**_BASE, "oauth_provider": "azure"})


@pytest.mark.parametrize("provider", ["azure", "google", "refresh_token"])
def test_oauth_requires_tls(provider: str) -> None:
    section = (
        dict(_REFRESH)
        if provider == "refresh_token"
        else {**_BASE, "auth": "oauth", "oauth_provider": provider}
    )
    with pytest.raises(ConfigError, match="requires TLS"):
        _email({**section, "starttls": False})


def test_oauth_requires_a_configured_sender() -> None:
    with pytest.raises(ConfigError, match="requires smtp_host and from_address"):
        _email({"auth": "oauth", "oauth_provider": "azure"})


@pytest.mark.parametrize("provider", [None, "entra", ""])
def test_oauth_requires_a_known_provider(provider: str | None) -> None:
    section: dict[str, Any] = {**_BASE, "auth": "oauth"}
    if provider is not None:
        section["oauth_provider"] = provider
    with pytest.raises(ConfigError, match="oauth_provider"):
        _email(section)


def test_subject_is_google_only() -> None:
    google = _email(
        {**_BASE, "auth": "oauth", "oauth_provider": "google", "oauth_subject": "bot@corp.example"}
    )
    assert google.oauth_subject == "bot@corp.example"
    with pytest.raises(ConfigError, match="oauth_subject applies only"):
        _email(
            {**_BASE, "auth": "oauth", "oauth_provider": "azure", "oauth_subject": "a@corp.example"}
        )


@pytest.mark.parametrize("provider", ["azure", "google"])
def test_workload_providers_refuse_refresh_grant_keys(provider: str) -> None:
    with pytest.raises(ConfigError, match="oauth_token_url apply only"):
        _email(
            {
                **_BASE,
                "auth": "oauth",
                "oauth_provider": provider,
                "oauth_token_url": "https://login.example/token",
            }
        )


@pytest.mark.parametrize(
    "missing", ["oauth_token_url", "oauth_client_id", "oauth_refresh_token_env"]
)
def test_refresh_token_required_keys(missing: str) -> None:
    section = dict(_REFRESH)
    del section[missing]
    with pytest.raises(ConfigError, match=f"requires {missing}"):
        _email(section)


@pytest.mark.parametrize(
    "url",
    [
        "http://login.example/token",
        "https://",
        "https://user:pw@login.example/token",
        "https://login.example/token#frag",
        "https://login.example:notaport/token",
        "https://login.example/to ken",
        "login.example/token",
    ],
)
def test_token_url_must_be_plain_https(url: str) -> None:
    with pytest.raises(ConfigError, match="must be an https:// URL"):
        _email({**_REFRESH, "oauth_token_url": url})


@pytest.mark.parametrize("key", ["oauth_refresh_token_env", "oauth_client_secret_env"])
def test_secret_keys_name_env_vars_and_never_echo_the_value(key: str) -> None:
    pasted = "1//0gLiteral-Refresh.Token"
    with pytest.raises(ConfigError, match="must name an environment variable") as caught:
        _email({**_REFRESH, key: pasted})
    assert pasted not in str(caught.value)


def test_empty_string_values_are_refused() -> None:
    with pytest.raises(ConfigError, match="oauth_scope must be a non-empty string"):
        _email({**_REFRESH, "oauth_scope": ""})


def test_refresh_token_config_parses_every_key() -> None:
    email = _email(
        {
            **_REFRESH,
            "oauth_client_secret_env": "SMTP_CLIENT_SECRET",
            "oauth_scope": "https://mail.example/smtp offline_access",
            "username": "sender@corp.example",
        }
    )
    assert email.auth == "oauth" and email.oauth_provider == "refresh_token"
    assert email.oauth_token_url == "https://login.example/oauth2/token"
    assert email.oauth_client_id == "client-123"
    assert email.oauth_client_secret_env == "SMTP_CLIENT_SECRET"
    assert email.oauth_refresh_token_env == "SMTP_REFRESH_TOKEN"
    assert email.oauth_scope == "https://mail.example/smtp offline_access"


@pytest.mark.parametrize(
    "email",
    [
        EmailConfig(smtp_host="smtp.example", from_address="a@corp.example", implicit_tls=True,
                    starttls=False, smtp_port=465),
        EmailConfig(smtp_host="smtp.example", from_address="a@corp.example", auth="oauth",
                    oauth_provider="azure"),
        EmailConfig(smtp_host="smtp.example", from_address="a@corp.example", auth="oauth",
                    oauth_provider="google", oauth_subject="bot@corp.example",
                    implicit_tls=True, starttls=False),
        EmailConfig(smtp_host="smtp.example", from_address="a@corp.example", auth="oauth",
                    oauth_provider="refresh_token", oauth_token_url="https://idp.example/t",
                    oauth_client_id="cid", oauth_client_secret_env="SMTP_SECRET",
                    oauth_refresh_token_env="SMTP_RT", oauth_scope="smtp offline_access",
                    username="sender@corp.example"),
    ],
)  # fmt: skip
def test_emitter_round_trips(email: EmailConfig) -> None:
    config = Config(email=email)
    assert _round_trip(config).email == email


def test_password_config_emits_no_oauth_keys() -> None:
    text = emit_config_toml(Config(email=EmailConfig(smtp_host="h", from_address="a@b.example")))
    assert "auth =" not in text and "oauth_" not in text and "implicit_tls" not in text


# -- doctor row (no network, names only) --------------------------------------


def _doctor(email: EmailConfig) -> list[dict[str, str]]:
    report = _Report(json_mode=True)
    _check_email(report, Config(email=email))
    assert not report.failed  # email trouble breaks invites, never serving
    return report.rows


def test_doctor_is_silent_without_email() -> None:
    assert _doctor(EmailConfig()) == []


def test_doctor_password_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    base = EmailConfig(smtp_host="h", from_address="a@corp.example")
    assert _doctor(base) == [
        {"level": "PASS", "area": "email", "message": "SMTP without authentication (STARTTLS)"}
    ]
    with_user = EmailConfig(smtp_host="h", from_address="a@corp.example", username="u")
    monkeypatch.delenv("LLM_REDACT_SMTP_PASSWORD", raising=False)
    [row] = _doctor(with_user)
    assert row["level"] == "WARN" and "LLM_REDACT_SMTP_PASSWORD is not" in row["message"]
    monkeypatch.setenv("LLM_REDACT_SMTP_PASSWORD", "hunter2-secret")
    [row] = _doctor(with_user)
    assert row == {
        "level": "PASS",
        "area": "email",
        "message": "SMTP password auth (STARTTLS; LLM_REDACT_SMTP_PASSWORD set)",
    }
    cleartext = EmailConfig(
        smtp_host="h", from_address="a@corp.example", username="u", starttls=False
    )
    [row] = _doctor(cleartext)
    assert row["level"] == "WARN" and "cleartext" in row["message"]
    assert "hunter2-secret" not in row["message"]


@pytest.mark.parametrize(
    ("provider", "needle"),
    [("azure", "Microsoft Entra ID"), ("google", "domain-wide delegation")],
)
def test_doctor_workload_oauth_rows(provider: str, needle: str) -> None:
    email = EmailConfig(
        smtp_host="h",
        from_address="a@corp.example",
        auth="oauth",
        oauth_provider=provider,
        implicit_tls=True,
        starttls=False,
    )
    [row] = _doctor(email)
    assert row["level"] == "PASS"
    assert "XOAUTH2, implicit TLS" in row["message"] and needle in row["message"]


def test_doctor_refresh_token_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    email = EmailConfig(
        smtp_host="h",
        from_address="a@corp.example",
        auth="oauth",
        oauth_provider="refresh_token",
        oauth_token_url="https://login.example/oauth2/token",
        oauth_client_id="cid",
        oauth_client_secret_env="SMTP_SECRET",
        oauth_refresh_token_env="SMTP_RT",
    )
    monkeypatch.delenv("SMTP_RT", raising=False)
    monkeypatch.delenv("SMTP_SECRET", raising=False)
    [row] = _doctor(email)
    assert row["level"] == "WARN" and "SMTP_RT and SMTP_SECRET not set" in row["message"]
    monkeypatch.setenv("SMTP_RT", "rt-value-secret")
    monkeypatch.setenv("SMTP_SECRET", "client-secret-value")
    [row] = _doctor(email)
    assert row["level"] == "PASS"
    assert "refresh-token grant at login.example" in row["message"]
    assert "rt-value-secret" not in row["message"]
    assert "client-secret-value" not in row["message"]
