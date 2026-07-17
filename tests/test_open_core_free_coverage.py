"""Free-side execution coverage for code whose end-to-end tests moved to the
llm-redact-pro repo in the R4 open-core split.

These functions are Free CODE (they live in ``src/llm_redact``) but are only
reachable behind a paid feature — audit HMAC/backup key resolution, the vault
key resolvers, the Bedrock/Vertex cloud adapters, encrypted-vault rotation, the
named-user surface. Their behavioural coverage lives in the pro repo (which
boots a licensed proxy); here we drive each one directly, without a license, so
the Free suite still EXECUTES every branching function (the complexity gate) and
a Free-side regression is caught in Free CI.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from fake_cipher import FakeVaultCipher

# --- key resolvers (env / command / keyring; audit HMAC + backup) ------------


def test_audit_hmac_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_redact.audit import AUDIT_HMAC_ENV, audit_hmac_key_from_env

    monkeypatch.delenv(AUDIT_HMAC_ENV, raising=False)
    assert audit_hmac_key_from_env() is None
    monkeypatch.setenv(AUDIT_HMAC_ENV, "a passphrase")
    assert len(audit_hmac_key_from_env() or b"") == 32  # SHA-256 -> 32 bytes


def test_audit_enc_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_redact.audit_s3 import AUDIT_ENC_KEY_ENV, audit_enc_key_from_env

    monkeypatch.delenv(AUDIT_ENC_KEY_ENV, raising=False)
    assert audit_enc_key_from_env() is None
    monkeypatch.setenv(AUDIT_ENC_KEY_ENV, "a passphrase")
    assert len(audit_enc_key_from_env() or b"") == 44  # urlsafe-b64 Fernet key


def test_vault_key_from_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_redact.vault_crypto import CMD_ENV_KEY, key_from_command

    monkeypatch.delenv(CMD_ENV_KEY, raising=False)
    assert key_from_command() is None
    monkeypatch.setenv(CMD_ENV_KEY, "printf secretkey")
    assert key_from_command() == "secretkey"
    # A failing command logs by TYPE only and returns None (fail-closed).
    monkeypatch.setenv(CMD_ENV_KEY, "false")
    assert key_from_command() is None


def test_vault_key_from_keyring_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # The keyring extra is not in the dev group, so this exercises the
    # ImportError fail-closed arm (None, never a silent downgrade).
    from llm_redact.vault_crypto import key_from_keyring

    assert key_from_keyring() is None


def test_decode_master_key_validation() -> None:
    import base64

    from llm_redact.vault import VaultKeyError
    from llm_redact.vault_crypto import decode_master_key

    good = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    assert decode_master_key(good, "TEST") == b"k" * 32
    with pytest.raises(VaultKeyError, match="urlsafe base64"):
        decode_master_key("!!!not base64!!!", "TEST")
    short = base64.urlsafe_b64encode(b"k" * 16).decode("ascii")
    with pytest.raises(VaultKeyError, match="32 bytes"):
        decode_master_key(short, "TEST")


# --- config / users value objects --------------------------------------------


def test_email_config_configured() -> None:
    from llm_redact.config import EmailConfig

    assert EmailConfig().configured is False
    assert EmailConfig(smtp_host="mail.example").configured is False
    assert EmailConfig(smtp_host="mail.example", from_address="a@example").configured is True


def test_user_row_status() -> None:
    from llm_redact.users import UserRow

    base = dict(name="Ada", email="a@corp.example", invited_at="t0")
    invited = UserRow(**base, verified_at=None, revoked_at=None)
    verified = UserRow(**base, verified_at="t1", revoked_at=None)
    revoked = UserRow(**base, verified_at="t1", revoked_at="t2")
    assert invited.status == "invited"
    assert verified.status == "verified"
    assert revoked.status == "revoked"  # revoked wins over verified


def test_send_verification_email_uses_operator_smtp(monkeypatch: pytest.MonkeyPatch) -> None:
    from email.message import EmailMessage

    from llm_redact.users import send_verification_email

    sent: dict[str, object] = {}

    class _FakeSMTP:
        def __init__(self, host: str, port: int, timeout: int) -> None:
            sent["host"], sent["port"] = host, port

        def starttls(self) -> None:
            sent["starttls"] = True

        def login(self, user: str, password: str) -> None:
            sent["login"] = (user, password)

        def send_message(self, msg: EmailMessage) -> None:
            sent["body"] = msg.get_content()

        def quit(self) -> None:
            sent["quit"] = True

    monkeypatch.setenv("SMTP_PW", "hunter2")
    send_verification_email(
        smtp_host="mail.example",
        smtp_port=587,
        starttls=True,
        username="mailer",
        password_env="SMTP_PW",
        from_address="admin@corp.example",
        to_address="ada@corp.example",
        display_name="Ada",
        code="12345678",
        smtp_factory=_FakeSMTP,  # type: ignore[arg-type]
    )
    assert sent["starttls"] is True and sent["login"] == ("mailer", "hunter2")
    assert sent["quit"] is True
    # The code appears in the body; the SMTP password never does.
    assert "12345678" in str(sent["body"]) and "hunter2" not in str(sent["body"])


# --- cloud adapters (Team-gated; instantiating one needs no license) ---------


def test_vertex_wants_system_note() -> None:
    from llm_redact.providers.base import RouteKind
    from llm_redact.providers.vertex import VertexAdapter

    adapter = VertexAdapter()
    assert adapter.wants_system_note(RouteKind.CHAT, "/v1/models/x:generateContent") is True
    assert adapter.wants_system_note(RouteKind.REDACT_ONLY, "/v1/models/x:countTokens") is True
    assert adapter.wants_system_note(RouteKind.REDACT_ONLY, "/v1/models/x:predict") is False


def test_bedrock_looks_like_converse_and_note_injection() -> None:
    from llm_redact.providers.bedrock import BedrockAdapter, _looks_like_converse

    adapter = BedrockAdapter()
    converse = {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}
    assert _looks_like_converse(converse) is True
    # A Converse body gains a system note appended as a {"text": ...} block.
    noted = adapter.inject_system_note(dict(converse))
    assert noted["system"][-1]["text"]
    # A Claude-on-Bedrock (anthropic_version) body routes to the Messages path.
    claude = {"anthropic_version": "bedrock-2023-05-31", "messages": []}
    assert "system" in adapter.inject_system_note(claude)
    # A typed (OpenAI-style) block is NOT Converse and is left untouched.
    typed = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    assert _looks_like_converse(typed) is False
    assert adapter.inject_system_note(dict(typed)) == typed


def _bedrock_frame(event_type: str, payload: dict[str, object]):
    from llm_redact.eventstream import EventStreamMessage, string_header

    return EventStreamMessage(
        headers=[
            string_header(":message-type", "event"),
            string_header(":event-type", event_type),
        ],
        payload=json.dumps(payload).encode("utf-8"),
    )


def test_bedrock_stream_rehydration_paths(vault) -> None:
    from llm_redact.providers.bedrock import BedrockAdapter
    from llm_redact.rehydrate import RehydratorPool

    adapter = BedrockAdapter()
    pool = RehydratorPool(vault)

    # contentBlockStop flushes any buffered channel leftovers (none here, so it
    # is a pass-through) -> exercises _rehydrate_converse_stop.
    stop = _bedrock_frame("contentBlockStop", {"contentBlockIndex": 0})
    assert adapter.rehydrate_eventstream_message(stop, pool) == [stop]

    # invoke-with-response-stream chunk with an undecodable inner payload is
    # forwarded verbatim -> exercises _rehydrate_invoke_chunk.
    chunk = _bedrock_frame("chunk", {"bytes": "not base64!!"})
    assert adapter.rehydrate_eventstream_message(chunk, pool) == [chunk]


# --- vault key rotation (encrypted vault; Pro feature, FakeVaultCipher) -------


def test_rotate_vault_key_reencrypts(tmp_path: Path) -> None:
    from llm_redact.vault import open_sqlite_vault, rotate_vault_key

    db = tmp_path / "vault.db"
    old = FakeVaultCipher(seed=b"o" * 32)
    va = open_sqlite_vault(db, "conv-aaaa", old)
    token = va.placeholder_for("EMAIL", "jane@corp.example")
    va.close()

    conn = sqlite3.connect(db, isolation_level=None)
    new = FakeVaultCipher(seed=b"n" * 32)
    assert rotate_vault_key(conn, old, new) == 1  # one mapping re-encrypted
    conn.close()

    # Reopen under the NEW cipher: the token rehydrates to the identical value,
    # token identity preserved (never-wrong-value across rotation).
    reopened = open_sqlite_vault(db, "conv-aaaa", new)
    assert reopened.original_for(token) == "jane@corp.example"
    reopened.close()


# --- doctor vault-key-match check --------------------------------------------


def test_check_vault_key_matches_memory_backend() -> None:
    # A non-sqlite backend has nothing on local disk to compare against, so the
    # check reports PASS ("verified at open") -> exercises _check_vault_key_matches.
    from llm_redact.config import Config, VaultConfig
    from llm_redact.doctor_cli import _check_vault_key_matches, _Report

    report = _Report(json_mode=True)
    _check_vault_key_matches(
        report, Config(vault=VaultConfig(backend="memory", encryption="fernet"))
    )
    assert any(r["level"] == "PASS" and r["area"] == "vault" for r in report.rows)


# --- proxy: license-warning refresh + users endpoint (Free fail-closed) ------


def test_refresh_license_warnings_on_free_app(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_redact.config import Config
    from llm_redact.proxy import create_app

    monkeypatch.delenv("LLM_REDACT_LICENSE_KEY", raising=False)
    state = create_app(Config()).state.proxy
    state.refresh_license_warnings()  # a Free app: still resolves cleanly to free
    assert state.license.tier == "free"


async def test_users_endpoint_free_tier_403() -> None:
    from llm_redact.config import Config
    from llm_redact.proxy import create_app

    app = create_app(Config())
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")
    resp = await client.get("/__llm-redact/users")
    assert resp.status_code == 403
    assert "llm-redact-pro" in resp.json()["error"]  # users live in the pro package
    await client.aclose()


# --- vault_cli decrypt helper ------------------------------------------------


def test_vault_cli_decrypt_fails_helper() -> None:
    from llm_redact.vault_cli import _decrypt_fails

    cipher = FakeVaultCipher()
    assert _decrypt_fails(cipher, cipher.encrypt("ok")) == 0  # decrypts cleanly
    assert _decrypt_fails(cipher, b"garbage") == 1  # wrong/garbage -> 1


def test_vault_set_key_without_keyring_extra() -> None:
    # keyring is not in the dev group, so run_vault_set_key hits its
    # missing-extra guard (exit 2) -> exercises the function's entry branch.
    from llm_redact.vault_cli import run_vault_set_key

    assert run_vault_set_key(argparse.Namespace()) == 2
