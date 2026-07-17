"""sessions/lookup/gen-key CLI against real vault databases."""

import os
import sqlite3
from pathlib import Path

import pytest

from llm_redact.cli import main
from llm_redact.vault import open_sqlite_vault


def _seed_plain(path: Path) -> None:
    vault_a = open_sqlite_vault(path, "conv-aaaa")
    vault_a.placeholder_for("EMAIL", "jane@corp.example")
    vault_a.placeholder_for("EMAIL", "bob@corp.example")
    vault_a.close()
    vault_b = open_sqlite_vault(path, "conv-bbbb")
    vault_b.placeholder_for("SECRET", "hunter2hunter2")
    vault_b.close()


def _age_session(path: Path, session: str, days: int) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE mappings SET created_at ="
        " strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?) WHERE session_id = ?",
        (f"-{days} days", session),
    )
    conn.commit()
    conn.close()


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    return int(excinfo.value.code or 0)


def test_sessions_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "vault.db"
    _seed_plain(db)
    assert _run(["sessions", "list", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "conv-aaaa" in out and "conv-bbbb" in out
    assert "jane@corp.example" not in out  # metadata only


def test_sessions_prune_whole_sessions_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "vault.db"
    _seed_plain(db)
    _age_session(db, "conv-aaaa", 120)

    assert _run(["sessions", "prune", "--older-than", "90d", "--yes", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "conv-aaaa" in out and "restart" in out

    conn = sqlite3.connect(db)
    sessions = {row[0] for row in conn.execute("SELECT DISTINCT session_id FROM mappings")}
    conn.close()
    assert sessions == {"conv-bbbb"}  # recent session untouched


def test_sessions_prune_bad_cutoff(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "vault.db"
    _seed_plain(db)
    assert _run(["sessions", "prune", "--older-than", "3 weeks", "--db", str(db)]) == 2


def test_sessions_prune_nothing_stale(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "vault.db"
    _seed_plain(db)
    assert _run(["sessions", "prune", "--older-than", "90d", "--yes", "--db", str(db)]) == 0
    assert "nothing to prune" in capsys.readouterr().out


def test_lookup_token_including_mangled(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "vault.db"
    _seed_plain(db)
    assert _run(["lookup", "«EMAIL_001»", "--db", str(db)]) == 0
    assert "jane@corp.example" in capsys.readouterr().out
    # A pasted mangle canonicalizes before lookup.
    assert _run(["lookup", "«email-1»", "--db", str(db), "--session", "conv-aaaa"]) == 0
    assert "jane@corp.example" in capsys.readouterr().out
    assert _run(["lookup", "«EMAIL_999»", "--db", str(db)]) == 1


def test_lookup_by_value(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "vault.db"
    _seed_plain(db)
    assert _run(["lookup", "--value", "bob@corp.example", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "«EMAIL_002»" in out and "conv-aaaa" in out
    assert _run(["lookup", "--value", "nobody@corp.example", "--db", str(db)]) == 1


def test_lookup_requires_exactly_one_selector(tmp_path: Path) -> None:
    assert _run(["lookup", "--db", str(tmp_path / "v.db")]) == 2
    assert _run(["lookup", "«X_001»", "--value", "y", "--db", str(tmp_path / "v.db")]) == 2


def test_missing_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["sessions", "list", "--db", str(tmp_path / "absent.db")]) == 1


# ---- encrypted vaults: the CLI paths over the paid cipher live in
# tests/pro/test_cli_vault_crypto.py (seeding + lookup/verify both resolve
# the concrete cipher through the registry). ----


def test_vault_gen_key(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("cryptography")
    assert _run(["vault", "gen-key"]) == 0
    key = capsys.readouterr().out.strip()
    assert len(key) == 44


# ---- vault verify ----


def test_vault_verify_plaintext_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "vault.db"
    _seed_plain(db)
    assert _run(["vault", "verify", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "verify OK" in out
    assert "PASS counter density" in out


def test_vault_verify_detects_counter_gap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "vault.db"
    _seed_plain(db)  # conv-aaaa has EMAIL n=1 and n=2
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM mappings WHERE session_id = 'conv-aaaa' AND n = 1")
    conn.commit()
    conn.close()
    assert _run(["vault", "verify", "--db", str(db)]) == 1
    out = capsys.readouterr().out
    assert "FAIL counter density" in out and "verify FAILED" in out


def test_vault_verify_missing_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["vault", "verify", "--db", str(tmp_path / "nope.db")]) == 2
    assert "no vault database" in capsys.readouterr().out


# ---- vault backup ----


def test_vault_backup_is_a_consistent_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "vault.db"
    _seed_plain(db)
    dest = tmp_path / "backup" / "vault-snap.db"
    assert _run(["vault", "backup", str(dest), "--db", str(db)]) == 0
    assert dest.exists()
    # 0600 — the copy holds the same secrets as the source.
    if os.name == "posix":
        assert (dest.stat().st_mode & 0o777) == 0o600
    # A single-file snapshot (no -wal sidecar) that lookups can read.
    assert _run(["lookup", "«EMAIL_001»", "--db", str(dest)]) == 0
    assert "jane@corp.example" in capsys.readouterr().out


def test_vault_backup_refuses_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "vault.db"
    _seed_plain(db)
    dest = tmp_path / "snap.db"
    dest.write_bytes(b"existing")
    assert _run(["vault", "backup", str(dest), "--db", str(db)]) == 2
    assert "--force" in capsys.readouterr().out
    assert _run(["vault", "backup", str(dest), "--db", str(db), "--force"]) == 0


def test_readonly_uri_percent_encodes_query_metacharacters(tmp_path: Path) -> None:
    # Security review 3.1.1: a '?' or '#' in the vault path must be encoded so
    # it can't be parsed as the URI's query/fragment (defeating mode=ro).
    from llm_redact.vault_cli import _readonly_uri

    uri = _readonly_uri(tmp_path / "a?b#c.db")
    assert uri.endswith("?mode=ro")
    assert "a?b" not in uri and "#c" not in uri
    assert "%3F" in uri and "%23" in uri


@pytest.mark.skipif(os.name != "posix", reason="O_NOFOLLOW / symlinks are POSIX")
def test_vault_backup_refuses_to_write_through_a_symlink(tmp_path: Path) -> None:
    # Security review 3.1.1: a pre-planted (even broken) symlink at the dest
    # slips past the exists() guard; O_NOFOLLOW must refuse to write the vault
    # secrets THROUGH it onto the symlink target.
    db = tmp_path / "vault.db"
    _seed_plain(db)
    target = tmp_path / "victim.txt"
    dest = tmp_path / "snap.db"
    dest.symlink_to(target)  # broken symlink: target does not exist yet
    assert _run(["vault", "backup", str(dest), "--db", str(db)]) != 0
    assert not target.exists()  # nothing was written through the link
