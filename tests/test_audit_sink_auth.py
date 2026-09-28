"""`[audit.s3] auth` / `[audit.azure] auth`: the config SHAPE (parse,
validation, emitter round trip), the auth-aware credential-presence helper,
the doctor rows, and the /status field. Credential fetching itself is
llm-redact-pro's; the core only knows which mode is configured."""

from __future__ import annotations

import tomllib
from typing import Any

import httpx
import pytest

from llm_redact.audit_s3 import (
    AZURE_STORAGE_KEY_ENV,
    AZURE_STORAGE_SAS_ENV,
    credential_env_names,
    required_credential_env,
)
from llm_redact.config import (
    AuditConfig,
    AzureAuditConfig,
    Config,
    ConfigError,
    S3AuditConfig,
    parse_config,
)
from llm_redact.config_write import emit_config_toml
from llm_redact.doctor_cli import _check_audit_azure, _check_audit_s3, _Report
from llm_redact.proxy import create_app


def _s3(**section: Any) -> S3AuditConfig:
    return parse_config({"audit": {"s3": section}}, "<t>").audit.s3


def _azure(**section: Any) -> AzureAuditConfig:
    return parse_config({"audit": {"azure": section}}, "<t>").audit.azure


def _round_trip(config: Config) -> Config:
    return parse_config(tomllib.loads(emit_config_toml(config)), "<t>")


# --- config shape -------------------------------------------------------------


def test_defaults_keep_the_historical_static_key_modes() -> None:
    assert S3AuditConfig().auth == "keys"
    assert AzureAuditConfig().auth == "key"
    assert _s3(bucket="b").auth == "keys"
    assert _azure(account="a", container="c").auth == "key"


@pytest.mark.parametrize(
    ("provider", "auth"),
    [("aws", "keys"), ("aws", "identity"), ("gcs", "identity"), ("minio", "keys")],
)
def test_s3_auth_modes_parse(provider: str, auth: str) -> None:
    extra = {"endpoint_url": "http://127.0.0.1:9000"} if provider == "minio" else {}
    parsed = _s3(enabled=True, provider=provider, bucket="b", auth=auth, **extra)
    assert (parsed.provider, parsed.auth) == (provider, auth)


@pytest.mark.parametrize("auth", ["key", "sas", "identity"])
def test_azure_auth_modes_parse(auth: str) -> None:
    assert _azure(enabled=True, account="a", container="c", auth=auth).auth == auth


@pytest.mark.parametrize("provider", ["minio", "ceph"])
def test_identity_refused_where_there_is_no_cloud_identity(provider: str) -> None:
    # Refused even while disabled: the shape itself is meaningless.
    with pytest.raises(ConfigError, match=rf"auth = 'identity'.*{provider!r}"):
        _s3(provider=provider, bucket="b", endpoint_url="http://h:9000", auth="identity")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://gw.example/minio",  # sent to /minio/b/k but signed over /b/k
        "http://user:pw@h:9000",  # userinfo: signed host != sent host
        "http://h:9000?x=1",
        "http://h:9000#f",
        "ftp://h:9000",
        "http://:9000",
        "http://h:notaport",
    ],
)
def test_s3_endpoint_url_is_scheme_host_and_port_only(endpoint: str) -> None:
    with pytest.raises(ConfigError, match=r"\[audit.s3\] endpoint_url"):
        _s3(provider="minio", bucket="b", endpoint_url=endpoint)


def test_s3_endpoint_url_keeps_plain_http_and_a_trailing_slash() -> None:
    assert _s3(provider="ceph", bucket="b", endpoint_url="http://h:9000/").endpoint_url == (
        "http://h:9000"
    )


@pytest.mark.parametrize("auth", ["sas", "identity"])
def test_azure_bearer_modes_need_https_off_loopback(auth: str) -> None:
    # A SAS or an Entra token is a bearer secret; SharedKey sends a signature.
    with pytest.raises(ConfigError, match="must be https unless the host is loopback"):
        _azure(account="a", container="c", auth=auth, endpoint_url="http://blob.example.net")
    for ok in ("https://blob.example.net", "http://127.0.0.1:10000/devstoreaccount1"):
        assert _azure(account="a", container="c", auth=auth, endpoint_url=ok).endpoint_url == ok
    assert _azure(
        account="a", container="c", auth="key", endpoint_url="http://blob.example.net"
    ).auth == ("key")


def test_azure_endpoint_url_refuses_userinfo_and_query() -> None:
    for bad in ("https://u@blob.example.net", "https://blob.example.net?sv=1"):
        with pytest.raises(ConfigError, match=r"\[audit.azure\] endpoint_url"):
            _azure(account="a", container="c", endpoint_url=bad)


def test_unknown_auth_modes_are_config_errors() -> None:
    with pytest.raises(ConfigError, match=r"\[audit.s3\] auth must be 'keys' or 'identity'"):
        _s3(auth="key")  # the Azure spelling is not an S3 mode
    with pytest.raises(ConfigError, match=r"\[audit.azure\] auth must be 'key', 'sas'"):
        _azure(auth="keys")  # nor the other way round
    with pytest.raises(ConfigError, match=r"\[audit.azure\] auth"):
        _azure(auth="shared-key")


@pytest.mark.parametrize(
    "audit",
    [
        AuditConfig(s3=S3AuditConfig(enabled=True, provider="aws", bucket="b", auth="identity")),
        AuditConfig(
            s3=S3AuditConfig(enabled=True, provider="gcs", bucket="b", region="us", auth="identity")
        ),
        AuditConfig(azure=AzureAuditConfig(enabled=True, account="a", container="c", auth="sas")),
        AuditConfig(
            azure=AzureAuditConfig(enabled=True, account="a", container="c", auth="identity")
        ),
        # Defaults stay out of the file (and still round-trip).
        AuditConfig(azure=AzureAuditConfig(enabled=True, account="a", container="c")),
    ],
)
def test_auth_round_trips_through_the_emitter(audit: AuditConfig) -> None:
    config = Config(audit=audit)
    assert _round_trip(config) == config


def test_emitter_omits_default_auth() -> None:
    text = emit_config_toml(
        Config(
            audit=AuditConfig(
                s3=S3AuditConfig(enabled=True, bucket="b"),
                azure=AzureAuditConfig(enabled=True, account="a", container="c"),
            )
        )
    )
    assert "auth =" not in text
    text = emit_config_toml(
        Config(audit=AuditConfig(azure=AzureAuditConfig(account="a", auth="identity")))
    )
    assert 'auth = "identity"' in text


# --- auth-aware credential presence -------------------------------------------


def test_required_credential_env_per_mode() -> None:
    assert required_credential_env("s3", "aws", "keys") == credential_env_names("aws")[:2]
    assert required_credential_env("s3", "gcs", "keys") == ("GCS_HMAC_ACCESS_ID", "GCS_HMAC_SECRET")
    assert required_credential_env("s3", "aws", "identity") == ()
    assert required_credential_env("s3", "gcs", "identity") == ()
    assert required_credential_env("azure", auth="key") == (AZURE_STORAGE_KEY_ENV,)
    assert required_credential_env("azure", auth="sas") == (AZURE_STORAGE_SAS_ENV,)
    assert required_credential_env("azure", auth="identity") == ()
    assert AZURE_STORAGE_SAS_ENV == "AZURE_STORAGE_SAS_TOKEN"


# --- doctor -------------------------------------------------------------------


def _messages(report: _Report) -> str:
    return " ".join(row["level"] + " " + row["message"] for row in report.rows)


def test_doctor_s3_identity_is_informational_and_needs_no_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "GCS_HMAC_ACCESS_ID"):
        monkeypatch.delenv(name, raising=False)
    for provider in ("aws", "gcs"):
        config = Config(
            audit=AuditConfig(
                s3=S3AuditConfig(enabled=True, provider=provider, bucket="b", auth="identity")
            )
        )
        report = _Report(json_mode=True)
        _check_audit_s3(report, config)
        assert not report.failed
        text = _messages(report)
        assert "workload identity" in text and "resolve at runtime" in text
        assert "not checked offline" in text


def test_doctor_s3_keys_still_checks_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    config = Config(audit=AuditConfig(s3=S3AuditConfig(enabled=True, bucket="b")))
    report = _Report(json_mode=True)
    _check_audit_s3(report, config)
    assert report.failed and "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY" in _messages(report)


@pytest.mark.parametrize(
    ("auth", "env", "label"),
    [("key", AZURE_STORAGE_KEY_ENV, "SharedKey"), ("sas", AZURE_STORAGE_SAS_ENV, "SAS-token")],
)
def test_doctor_azure_checks_the_mode_specific_env(
    monkeypatch: pytest.MonkeyPatch, auth: str, env: str, label: str
) -> None:
    monkeypatch.delenv(AZURE_STORAGE_KEY_ENV, raising=False)
    monkeypatch.delenv(AZURE_STORAGE_SAS_ENV, raising=False)
    config = Config(
        audit=AuditConfig(
            azure=AzureAuditConfig(enabled=True, account="a", container="c", auth=auth)
        )
    )
    missing = _Report(json_mode=True)
    _check_audit_azure(missing, config)
    assert missing.failed and env in _messages(missing)

    monkeypatch.setenv(env, "sv=2024&sig=canary-not-a-secret")
    present = _Report(json_mode=True)
    _check_audit_azure(present, config)
    assert not present.failed
    text = _messages(present)
    assert label in text and "canary-not-a-secret" not in text  # presence only


def test_doctor_azure_identity_is_informational(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(AZURE_STORAGE_KEY_ENV, raising=False)
    monkeypatch.delenv(AZURE_STORAGE_SAS_ENV, raising=False)
    config = Config(
        audit=AuditConfig(
            azure=AzureAuditConfig(enabled=True, account="a", container="c", auth="identity")
        )
    )
    report = _Report(json_mode=True)
    _check_audit_azure(report, config)
    assert not report.failed
    assert "Entra ID" in _messages(report) and "workload identity" in _messages(report)


# --- /status ------------------------------------------------------------------


async def test_status_reports_the_configured_auth_modes() -> None:
    app = create_app(Config())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["audit"]["s3"]["auth"] == "keys"
    assert status["audit"]["azure"]["auth"] == "key"


# --- version skew: capabilities the loaded plugin must advertise ----------------


def test_plugin_capabilities_required_names_every_non_default_shape() -> None:
    from llm_redact.config import plugin_capabilities_required

    assert plugin_capabilities_required(Config()) == []
    config = parse_config(
        {
            "audit": {
                "s3": {"enabled": True, "bucket": "b", "auth": "identity"},
                "azure": {"enabled": True, "account": "a", "container": "c", "auth": "sas"},
            },
            "email": {
                "smtp_host": "smtp.example",
                "from_address": "p@example.com",
                "implicit_tls": True,
                "smtp_port": 465,
            },
        },
        "<t>",
    )
    assert plugin_capabilities_required(config) == [
        "audit.s3.auth=identity",
        "audit.azure.auth=sas",
        "email.implicit_tls",
    ]
    # A disabled sink's mode is inert: nothing to refuse.
    disabled = parse_config({"audit": {"s3": {"bucket": "b", "auth": "identity"}}}, "<t>")
    assert plugin_capabilities_required(disabled) == []


def test_a_plugin_without_the_capability_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_redact.proxy as proxy_mod
    import llm_redact.registry as registry_mod

    email = {"smtp_host": "smtp.example", "from_address": "p@example.com"}
    config = parse_config({"email": {**email, "auth": "oauth", "oauth_provider": "azure"}}, "<t>")
    old = registry_mod.Registry()
    monkeypatch.setattr(registry_mod, "_registry", old)
    monkeypatch.setattr(proxy_mod, "loaded_plugins", lambda: ["llm_redact_pro"])
    with pytest.raises(ConfigError, match=r"does not support email\.auth=oauth"):
        create_app(config)
    old.config_capabilities.add("email.auth=oauth")
    create_app(config)
    # Without any plugin the core's own fail-closed seams apply instead.
    monkeypatch.setattr(proxy_mod, "loaded_plugins", lambda: [])
    old.config_capabilities.clear()
    create_app(config)
