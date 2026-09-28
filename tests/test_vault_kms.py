"""[vault.kms]: the KMS-wrapped vault key — the core's side of the seam.

The core owns the config SHAPE (parser + emitter round trip), the fail-closed
defaults (no llm-redact-pro, an older one, or an ambiguous local key refuse to
start), the ONE cipher resolution path the proxy and every vault CLI command
share (``vault_crypto.resolve_cipher``), and the posture surfaces (/status
``key_source``, doctor rows that never call the KMS). The unwrap itself is
llm-redact-pro's; here a fake ``resolve_vault_key`` stands in for it.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import tomllib
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from fake_cipher import FakeVaultCipher
from llm_redact.config import (
    ConfigError,
    VaultConfig,
    kms_address_problem,
    parse_config,
    parse_vault_kms,
)
from llm_redact.config_write import emit_config_toml
from llm_redact.registry import Registry
from llm_redact.vault import InMemoryVaultManager, VaultKeyError, open_sqlite_vault
from llm_redact.vault_crypto import (
    CMD_ENV_KEY,
    ENV_KEY,
    key_source,
    require_vault_key_source,
    resolve_cipher,
    resolve_vault_key,
)

AWS_KEY = "arn:aws:kms:eu-west-1:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab"
GCP_KEY = "projects/p1/locations/global/keyRings/ring/cryptoKeys/vault"
AZURE_KEY = "https://my-vault.vault.azure.net/keys/llm-redact/0123456789abcdef"
MASTER = bytes(range(32))

VALID: dict[str, dict[str, str]] = {
    "aws": {"provider": "aws", "key_id": AWS_KEY, "wrapped_key_env": "WRAPPED"},
    "aws-alias": {
        "provider": "aws",
        "key_id": "arn:aws-us-gov:kms:us-gov-west-1:123456789012:alias/llm-redact",
        "wrapped_key_file": "/etc/llm-redact/vault-key.wrapped",
    },
    "gcp": {"provider": "gcp", "key_id": GCP_KEY, "wrapped_key_env": "WRAPPED"},
    "azure": {"provider": "azure", "key_id": AZURE_KEY, "wrapped_key_env": "WRAPPED"},
    "azure-unversioned": {
        "provider": "azure",
        "key_id": "https://my-vault.vault.azure.net/keys/llm-redact",
        "wrapped_key_file": "/w",
    },
    "hashicorp-token": {"provider": "hashicorp", "key_id": "llm-redact", "wrapped_key_env": "W"},
    "hashicorp-full": {
        "provider": "hashicorp",
        "key_id": "llm-redact",
        "wrapped_key_file": "/w",
        "address": "https://vault.internal:8200/",
        "mount": "kv/transit",
        "auth": "kubernetes",
        "role": "llm-redact",
        "auth_mount": "k8s-prod",
        "service_account_token_file": "/var/run/token",
    },
    "hashicorp-loopback-http": {
        "provider": "hashicorp",
        "key_id": "k",
        "wrapped_key_env": "W",
        "address": "http://127.0.0.1:8200",
    },
}


def _vault_raw(kms: dict[str, Any]) -> dict[str, Any]:
    return {"vault": {"backend": "sqlite", "encryption": "fernet", "kms": kms}}


# --- config shape ---------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(VALID))
def test_valid_tables_parse_and_round_trip(name: str) -> None:
    config = parse_config(_vault_raw(VALID[name]), "t")
    kms = config.vault.kms
    assert kms is not None and kms.provider == VALID[name]["provider"]
    emitted = emit_config_toml(config)
    assert "[vault.kms]" in emitted
    again = parse_config(tomllib.loads(emitted), "t")
    assert again.vault == config.vault


def test_hashicorp_values_and_defaults() -> None:
    full = parse_vault_kms(VALID["hashicorp-full"])
    assert full.address == "https://vault.internal:8200"  # trailing slash dropped
    assert (full.mount, full.auth, full.role, full.auth_mount) == (
        "kv/transit",
        "kubernetes",
        "llm-redact",
        "k8s-prod",
    )
    assert full.service_account_token_file == "/var/run/token"
    plain = parse_vault_kms(VALID["hashicorp-token"])
    assert (plain.address, plain.mount, plain.auth, plain.role) == ("", "transit", "token", "")
    assert plain.service_account_token_file.endswith("serviceaccount/token")
    # Defaults are emitted as nothing (they re-parse to the same value).
    toml = emit_config_toml(parse_config(_vault_raw(VALID["hashicorp-token"]), "t"))
    assert "mount =" not in toml and "auth =" not in toml


def test_absent_table_means_no_kms() -> None:
    config = parse_config({"vault": {"encryption": "fernet"}}, "t")
    assert config.vault.kms is None
    assert "[vault.kms]" not in emit_config_toml(config)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"provider": "vault"}, "provider must be one of"),
        ({"provider": None}, "provider must be one of"),
        ({"key_id": "alias/llm-redact"}, "key or alias ARN"),
        ({"key_id": ""}, "key or alias ARN"),
        ({"wrapped_key_env": None}, "exactly one of wrapped_key_env or wrapped_key_file"),
        ({"wrapped_key_file": "/w"}, "exactly one of wrapped_key_env or wrapped_key_file"),
        ({"wrapped_key_env": "1BAD-NAME"}, "environment variable name"),
        ({"wrapped_key_env": ENV_KEY}, "plaintext key source"),
        ({"wrapped_key_env": CMD_ENV_KEY}, "plaintext key source"),
        ({"address": "https://v:8200"}, "apply only to provider = 'hashicorp'"),
        ({"mount": "transit"}, "apply only to provider = 'hashicorp'"),
        ({"unknown": "x"}, "unknown key(s)"),
    ],
)
def test_aws_table_errors(patch: dict[str, Any], message: str) -> None:
    raw: dict[str, Any] = dict(VALID["aws"])
    for key, value in patch.items():
        if value is None:
            raw.pop(key)
        else:
            raw[key] = value
    with pytest.raises(ConfigError, match=re.escape(message)):
        parse_vault_kms(raw)


@pytest.mark.parametrize(
    ("provider", "key_id"),
    [
        ("gcp", "projects/p/locations/l/keyRings/r"),
        ("gcp", GCP_KEY + "/cryptoKeyVersions/1"),
        ("azure", "https://evil.example/keys/k"),
        ("azure", "http://my-vault.vault.azure.net/keys/k"),
        ("hashicorp", "transit/keys/k"),
    ],
)
def test_key_id_shapes_are_checked(provider: str, key_id: str) -> None:
    with pytest.raises(ConfigError, match=f"key_id for provider '{provider}'"):
        parse_vault_kms({"provider": provider, "key_id": key_id, "wrapped_key_env": "W"})


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"address": "http://vault.internal:8200"}, "https unless the host is loopback"),
        ({"address": "https://v:8200/?x=1"}, "query string"),
        ({"address": "vault:8200"}, "must look like"),
        ({"auth": "approle"}, "auth must be one of"),
        ({"mount": "/transit"}, "mount must be a mount path"),
        ({"auth": "kubernetes"}, "requires role"),
        ({"role": "r"}, "apply only to auth = 'kubernetes'"),
        ({"auth_mount": "k"}, "apply only to auth = 'kubernetes'"),
        (
            {"auth": "kubernetes", "role": "r", "service_account_token_file": ""},
            "must not be empty",
        ),
        ({"auth": "kubernetes", "role": "r", "auth_mount": "a//b"}, "auth_mount must be"),
    ],
)
def test_hashicorp_table_errors(patch: dict[str, Any], message: str) -> None:
    raw = {**VALID["hashicorp-token"], **patch}
    with pytest.raises(ConfigError, match=message):
        parse_vault_kms(raw)


def test_table_type_errors() -> None:
    with pytest.raises(ConfigError, match="must be a table"):
        parse_vault_kms("aws")
    with pytest.raises(ConfigError, match="role must be a string"):
        parse_vault_kms({**VALID["hashicorp-token"], "role": 5})


def test_requires_fernet_encryption() -> None:
    with pytest.raises(ConfigError, match='requires \\[vault\\] encryption = "fernet"'):
        parse_config({"vault": {"backend": "sqlite", "kms": VALID["aws"]}}, "t")


def test_wrapping_form_needs_no_wrapped_value() -> None:
    kms = parse_vault_kms({"provider": "aws", "key_id": AWS_KEY}, require_wrapped=False)
    assert kms.wrapped_key_env == "" and kms.wrapped_key_file == ""
    with pytest.raises(ConfigError, match="exactly one"):
        parse_vault_kms({"provider": "aws", "key_id": AWS_KEY})


def test_error_messages_never_echo_values() -> None:
    secretish = "arn:aws:kms:SECRET"
    with pytest.raises(ConfigError) as info:
        parse_vault_kms({"provider": "aws", "key_id": secretish, "wrapped_key_env": "W"})
    assert secretish not in str(info.value)


@pytest.mark.parametrize(
    ("address", "ok"),
    [
        ("https://vault.example:8200", True),
        ("http://localhost:8200", True),
        ("http://[::1]:8200", True),
        ("http://10.0.0.5:8200", False),
        ("https://v#frag", False),
        ("ftp://v", False),
    ],
)
def test_kms_address_problem(address: str, ok: bool) -> None:
    assert (kms_address_problem(address) is None) is ok


# --- the seam: Free default, guard, one resolution path ---------------------------


def _kms_vault(provider: str = "aws") -> VaultConfig:
    raw = {"aws": VALID["aws"], "hashicorp": VALID["hashicorp-token"]}[provider]
    return parse_config(_vault_raw(raw), "t").vault


class _Calls:
    def __init__(self) -> None:
        self.resolved: list[VaultConfig] = []
        self.keys: list[bytes] = []


@pytest.fixture
def kms_registry(monkeypatch: pytest.MonkeyPatch) -> tuple[Registry, _Calls]:
    """A registry whose plugin 'unwraps' to MASTER and whose ciphers are fakes."""
    monkeypatch.delenv(ENV_KEY, raising=False)
    monkeypatch.delenv(CMD_ENV_KEY, raising=False)
    calls = _Calls()
    reg = Registry()

    def resolve(config: VaultConfig) -> bytes | None:
        if config.kms is None:
            return None
        calls.resolved.append(config)
        return MASTER

    def from_key(key: bytes) -> FakeVaultCipher:
        calls.keys.append(key)
        return FakeVaultCipher(key)

    reg.resolve_vault_key = resolve
    reg.cipher_from_key = from_key  # type: ignore[assignment]
    reg.build_cipher = lambda config: FakeVaultCipher(b"local" * 6 + b"xx")  # type: ignore[assignment,return-value]
    reg.build_vault_manager = lambda config: InMemoryVaultManager(cipher=resolve_cipher(config))
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return reg, calls


def test_free_default_resolver() -> None:
    assert resolve_vault_key(VaultConfig()) is None
    assert resolve_vault_key(VaultConfig(encryption="fernet")) is None
    with pytest.raises(ConfigError, match="llm-redact-pro") as info:
        resolve_vault_key(_kms_vault())
    assert "[vault.kms]" in str(info.value) and "aws" in str(info.value)
    assert Registry().resolve_vault_key is resolve_vault_key


def test_key_source_labels() -> None:
    assert key_source(VaultConfig()) is None
    assert key_source(VaultConfig(encryption="fernet")) == "local"
    assert key_source(_kms_vault("hashicorp")) == "kms:hashicorp"


def test_guard_refuses_free_registry_and_ambiguous_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_KEY, raising=False)
    monkeypatch.delenv(CMD_ENV_KEY, raising=False)
    require_vault_key_source(VaultConfig(encryption="fernet"), Registry())  # no kms: no-op
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        require_vault_key_source(_kms_vault(), Registry())
    plugin = Registry()
    plugin.resolve_vault_key = lambda config: MASTER
    require_vault_key_source(_kms_vault(), plugin)
    for name in (ENV_KEY, CMD_ENV_KEY):
        with pytest.raises(ConfigError, match=f"{name} is also set") as info:
            require_vault_key_source(_kms_vault(), plugin, {name: "anything"})
        assert "anything" not in str(info.value)
    # Whitespace-only is unset.
    require_vault_key_source(_kms_vault(), plugin, {ENV_KEY: "  "})


def test_resolve_cipher_paths(kms_registry: tuple[Registry, _Calls]) -> None:
    reg, calls = kms_registry
    local = resolve_cipher(VaultConfig(encryption="fernet"))
    assert isinstance(local, FakeVaultCipher) and calls.resolved == []
    cipher = resolve_cipher(_kms_vault())
    assert isinstance(cipher, FakeVaultCipher)
    assert calls.keys == [MASTER] and len(calls.resolved) == 1
    assert cipher.key_check() == FakeVaultCipher(MASTER).key_check()


@pytest.mark.parametrize("bad", [None, b"short", "x" * 32])
def test_resolve_cipher_refuses_unusable_keys(
    kms_registry: tuple[Registry, _Calls], bad: Any
) -> None:
    reg, calls = kms_registry
    reg.resolve_vault_key = lambda config: bad
    with pytest.raises(VaultKeyError, match="refusing to fall back"):
        resolve_cipher(_kms_vault())
    assert calls.keys == []


def test_resolve_cipher_refuses_ambiguous_env(
    kms_registry: tuple[Registry, _Calls], monkeypatch: pytest.MonkeyPatch
) -> None:
    _reg, calls = kms_registry
    monkeypatch.setenv(ENV_KEY, "k")
    with pytest.raises(ConfigError, match="two sources"):
        resolve_cipher(_kms_vault())
    assert calls.resolved == []  # refused before the KMS is asked


# --- proxy startup + /status -------------------------------------------------------


def _kms_config(tmp_path: Path) -> Any:
    raw = _vault_raw(VALID["aws"])
    raw["vault"]["backend"] = "memory"
    return parse_config(raw, "t")


def test_proxy_refuses_kms_without_the_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact.proxy import create_app

    monkeypatch.delenv(ENV_KEY, raising=False)
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        create_app(_kms_config(tmp_path))


def test_proxy_refuses_ambiguous_env_before_building_the_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kms_registry: tuple[Registry, _Calls]
) -> None:
    from llm_redact.proxy import create_app

    reg, calls = kms_registry
    monkeypatch.setenv(ENV_KEY, "k")
    with pytest.raises(ConfigError, match="also set"):
        create_app(_kms_config(tmp_path))
    assert calls.resolved == []


async def test_status_reports_key_source(
    tmp_path: Path, kms_registry: tuple[Registry, _Calls]
) -> None:
    from llm_redact.proxy import create_app

    _reg, calls = kms_registry
    app = create_app(_kms_config(tmp_path))
    assert len(calls.resolved) == 1  # unwrapped once, at startup
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        body = (await client.get("/__llm-redact/status")).json()
    assert body["vault"]["key_source"] == "kms:aws"
    assert AWS_KEY not in str(body)  # the key id is not surfaced
    plain = create_app(parse_config({}, "t"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=plain), base_url="http://127.0.0.1"
    ) as client:
        assert (await client.get("/__llm-redact/status")).json()["vault"]["key_source"] is None


def test_status_cli_prints_key_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.cli import run_status

    payload = {
        "version": "1",
        "uptime_seconds": 1.0,
        "session": "default",
        "vault": {"backend": "sqlite", "entries": 3, "key_source": "kms:gcp"},
        "detections_total": {},
        "rehydrations_total": {},
        "audit": {"enabled": False, "rows": 0},
        "rehydration": {"fuzzy": True},
        "detection": {"ner_enabled": False},
        "providers": {},
        "license": {"tier": "free", "max_users": 1},
    }

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    args = argparse.Namespace(config=None, port=8787, json=False, ca=None, cert=None, key=None)
    assert run_status(args) == 0
    assert "(3 entries, key: kms:gcp)" in capsys.readouterr().out


# --- the vault CLI goes through the same resolution ---------------------------------


def _write_config(tmp_path: Path, db: Path, kms: dict[str, str] | None = None) -> Path:
    lines = ["[vault]", 'backend = "sqlite"', f'path = "{db.as_posix()}"', 'encryption = "fernet"']
    if kms is not None:
        lines.append("[vault.kms]")
        lines += [f'{k} = "{v}"' for k, v in kms.items()]
    path = tmp_path / "config.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _seed(db: Path, key: bytes = MASTER) -> None:
    vault = open_sqlite_vault(db, "conv-a", FakeVaultCipher(key))
    vault.placeholder_for("EMAIL", "jane@corp.example")
    vault.close()


def _run(argv: list[str]) -> int:
    from llm_redact.cli import main

    with pytest.raises(SystemExit) as info:
        main(argv)
    return int(info.value.code or 0)


def test_cli_lookup_and_verify_use_the_kms_key(
    tmp_path: Path, kms_registry: tuple[Registry, _Calls], capsys: pytest.CaptureFixture[str]
) -> None:
    _reg, calls = kms_registry
    db = tmp_path / "vault.db"
    _seed(db)
    cfg = _write_config(tmp_path, db, VALID["aws"])
    assert _run(["lookup", "«EMAIL_001»", "--config", str(cfg)]) == 0
    assert "jane@corp.example" in capsys.readouterr().out
    # An explicit --db names the file; the config still names the key.
    assert (
        _run(["lookup", "--value", "jane@corp.example", "--db", str(db), "--config", str(cfg)]) == 0
    )
    assert "«EMAIL_001»" in capsys.readouterr().out
    assert _run(["vault", "verify", "--config", str(cfg)]) == 0
    assert "verify OK" in capsys.readouterr().out
    assert len(calls.resolved) == 3 and set(calls.keys) == {MASTER}


def test_cli_mismatch_names_the_kms_source(
    tmp_path: Path, kms_registry: tuple[Registry, _Calls], capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "vault.db"
    _seed(db, key=b"z" * 32)
    cfg = _write_config(tmp_path, db, VALID["aws"])
    assert _run(["lookup", "«EMAIL_001»", "--config", str(cfg)]) == 2
    out = capsys.readouterr().out
    assert "KMS-unwrapped vault key" in out and "aws" in out
    assert "jane@corp.example" not in out


def test_cli_refuses_kms_without_the_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(ENV_KEY, raising=False)
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    db = tmp_path / "vault.db"
    _seed(db)
    cfg = _write_config(tmp_path, db, VALID["aws"])
    assert _run(["vault", "verify", "--config", str(cfg)]) == 2
    assert "llm-redact-pro" in capsys.readouterr().out


def test_cli_bad_config_fails_closed(
    tmp_path: Path, kms_registry: tuple[Registry, _Calls], capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "vault.db"
    _seed(db)
    cfg = tmp_path / "config.toml"
    cfg.write_text("[vault]\nencryption = 'fernet'\n[vault.kms]\nprovider = 'nope'\n")
    assert _run(["lookup", "«EMAIL_001»", "--db", str(db), "--config", str(cfg)]) == 2
    assert "provider must be one of" in capsys.readouterr().out


def test_cli_rotate_key_under_kms(
    tmp_path: Path,
    kms_registry: tuple[Registry, _Calls],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import llm_redact.vault_cli as vault_cli
    from llm_redact.vault_crypto import NEW_ENV_KEY

    monkeypatch.setattr(vault_cli, "_proxy_reachable", lambda args: False)
    db = tmp_path / "vault.db"
    _seed(db)
    cfg = _write_config(tmp_path, db, VALID["aws"])
    import base64

    monkeypatch.setenv(NEW_ENV_KEY, base64.urlsafe_b64encode(b"n" * 32).decode())
    assert _run(["vault", "rotate-key", "--config", str(cfg), "--yes"]) == 0
    out = capsys.readouterr().out
    assert "kms-wrap" in out and "aws" in out
    # Rotated under the new key: the KMS key no longer matches (as the message says).
    conn = sqlite3.connect(db)
    check = conn.execute("SELECT value FROM vault_meta WHERE key = 'key_check'").fetchone()[0]
    conn.close()
    assert check == FakeVaultCipher(b"n" * 32).key_check()
    assert _run(["vault", "rotate-key", "--config", str(cfg), "--yes"]) == 2
    assert "current the KMS-unwrapped vault key" in capsys.readouterr().out


def test_rdbms_store_cipher_comes_from_the_same_path(
    kms_registry: tuple[Registry, _Calls], monkeypatch: pytest.MonkeyPatch
) -> None:
    import llm_redact.vault_rdbms as vault_rdbms
    from llm_redact.vault_cli import _open_rdbms_store

    seen: list[Any] = []

    class Recorder:
        def __init__(self, vault: VaultConfig, cipher: Any) -> None:
            seen.append(cipher)

    monkeypatch.setattr(vault_rdbms, "RdbmsStore", Recorder)
    raw = _vault_raw(VALID["aws"])
    raw["vault"].update(backend="postgresql", rdbms={"dsn": "postgresql://u@127.0.0.1/db"})
    _open_rdbms_store(parse_config(raw, "t"))
    assert isinstance(seen[0], FakeVaultCipher)
    assert seen[0].key_check() == FakeVaultCipher(MASTER).key_check()


# --- doctor: posture only, never the KMS --------------------------------------------


def _doctor(tmp_path: Path, kms: dict[str, str]) -> tuple[int, str]:
    import socket

    from llm_redact.doctor_cli import run_doctor

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    lines = [f"port = {port}", "[vault]", 'encryption = "fernet"', "[vault.kms]"]
    lines += [f'{k} = "{v}"' for k, v in kms.items()]
    cfg = tmp_path / "config.toml"
    cfg.write_text("\n".join(lines) + "\n")
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = run_doctor(argparse.Namespace(config=cfg))
    return code, buffer.getvalue()


@pytest.fixture
def never_unwrap(kms_registry: tuple[Registry, _Calls]) -> Registry:
    reg, _calls = kms_registry

    def boom(config: VaultConfig) -> bytes:
        raise AssertionError("doctor must never call the KMS")

    reg.resolve_vault_key = boom
    return reg


def _vault_rows(out: str) -> list[str]:
    return [line for line in out.splitlines() if "vault" in line]


def test_doctor_kms_healthy(
    tmp_path: Path, never_unwrap: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("cryptography")
    monkeypatch.setenv("WRAPPED", "AQID")
    code, out = _doctor(tmp_path, VALID["aws"])
    rows = _vault_rows(out)
    assert any("PASS" in row and "kms:aws" in row for row in rows), out
    assert not any("FAIL" in row for row in rows), out
    assert "AQID" not in out and AWS_KEY not in out


def test_doctor_kms_failures(
    tmp_path: Path, never_unwrap: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WRAPPED", raising=False)
    monkeypatch.setenv(ENV_KEY, "plaintext-key")
    code, out = _doctor(tmp_path, VALID["aws"])
    assert code == 1
    assert "WRAPPED is not set" in out
    assert f"{ENV_KEY} is also set" in out and "plaintext-key" not in out
    code, out = _doctor(
        tmp_path,
        {**VALID["gcp"], "wrapped_key_env": "", "wrapped_key_file": str(tmp_path / "absent")},
    )
    assert "does not exist" in out


def test_doctor_kms_without_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    monkeypatch.setenv("WRAPPED", "AQID")
    code, out = _doctor(tmp_path, VALID["aws"])
    assert code == 1 and "needs llm-redact-pro" in out


def test_doctor_kms_hashicorp(
    tmp_path: Path, never_unwrap: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("VAULT_ADDR", "VAULT_TOKEN", "VAULT_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("W", "vault:v1:abc")
    _code, out = _doctor(tmp_path, VALID["hashicorp-token"])
    assert "no address and VAULT_ADDR is not set" in out
    assert "set VAULT_TOKEN or VAULT_TOKEN_FILE" in out
    monkeypatch.setenv("VAULT_ADDR", "http://vault.internal:8200")
    monkeypatch.setenv("VAULT_TOKEN", "s.secret")
    _code, out = _doctor(tmp_path, VALID["hashicorp-token"])
    assert "https unless the host is loopback" in out
    assert "VAULT_TOKEN or" not in out and "s.secret" not in out
    kube = {**VALID["hashicorp-full"], "wrapped_key_file": str(tmp_path / "w")}
    (tmp_path / "w").write_text("vault:v1:abc")
    kube["service_account_token_file"] = str(tmp_path / "no-token")
    _code, out = _doctor(tmp_path, kube)
    rows = _vault_rows(out)
    assert any("WARN" in row and "no-token" in row for row in rows), out
