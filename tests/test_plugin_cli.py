"""`llm-redact plugin`: install/uninstall/status against a fake HOME.

The installer writes only its own namespaced llm-redact-*.md files, never
silently overwrites a file whose content was modified (that needs
--force), and uninstall removes exactly the managed files.
"""

from pathlib import Path

import pytest

from llm_redact.plugin_assets import COMMANDS
from llm_redact.plugin_cli import _target_dir, install, status, uninstall


def _env(tmp_path: Path) -> dict[str, str]:
    return {"HOME": str(tmp_path)}


def test_target_dirs_honor_tool_env_overrides(tmp_path: Path) -> None:
    env = {
        "HOME": str(tmp_path),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "cc"),
        "CODEX_HOME": str(tmp_path / "cx"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
    }
    assert _target_dir("claude", env) == tmp_path / "cc" / "commands"
    assert _target_dir("codex", env) == tmp_path / "cx" / "prompts"
    assert _target_dir("opencode", env) == tmp_path / "xdg" / "opencode" / "commands"
    home_only = _env(tmp_path)
    assert _target_dir("claude", home_only) == tmp_path / ".claude" / "commands"
    assert _target_dir("codex", home_only) == tmp_path / ".codex" / "prompts"
    assert _target_dir("opencode", home_only) == tmp_path / ".config" / "opencode" / "commands"
    assert _target_dir("cursor", home_only) == tmp_path / ".cursor" / "commands"


def test_install_writes_all_commands_for_every_tool(tmp_path: Path) -> None:
    env = _env(tmp_path)
    for tool in ("claude", "codex", "opencode", "cursor"):
        assert install(tool, env, print_only=False, force=False, posture_hint=lambda: "-") == 0
        target = _target_dir(tool, env)
        names = sorted(p.name for p in target.iterdir())
        assert names == sorted(f"llm-redact-{c.name}.md" for c in COMMANDS)
        for path in target.iterdir():
            text = path.read_text(encoding="utf-8")
            if tool == "cursor":
                assert text.startswith("# ")  # plain markdown, no frontmatter
            else:
                assert text.startswith("---\ndescription: ")


def test_install_is_idempotent_and_respects_modifications(tmp_path: Path) -> None:
    env = _env(tmp_path)
    assert install("codex", env, print_only=False, force=False, posture_hint=lambda: "-") == 0
    # Second run: nothing to do, still 0.
    assert install("codex", env, print_only=False, force=False, posture_hint=lambda: "-") == 0
    # A user-modified file blocks a plain reinstall...
    victim = _target_dir("codex", env) / "llm-redact-status.md"
    victim.write_text("my custom version", encoding="utf-8")
    assert install("codex", env, print_only=False, force=False) == 1
    assert victim.read_text(encoding="utf-8") == "my custom version"
    # ...and --force restores it.
    assert install("codex", env, print_only=False, force=True, posture_hint=lambda: "-") == 0
    assert victim.read_text(encoding="utf-8").startswith("---\ndescription: ")


def test_print_only_touches_nothing(tmp_path: Path) -> None:
    env = _env(tmp_path)
    assert install("claude", env, print_only=True, force=False) == 0
    assert not _target_dir("claude", env).exists()


def test_uninstall_removes_only_managed_files(tmp_path: Path) -> None:
    env = _env(tmp_path)
    install("opencode", env, print_only=False, force=False, posture_hint=lambda: "-")
    target = _target_dir("opencode", env)
    bystander = target / "my-own-command.md"
    bystander.write_text("keep me", encoding="utf-8")
    assert uninstall("opencode", env) == 0
    assert list(target.iterdir()) == [bystander]


def test_status_reports_per_tool_state(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = _env(tmp_path)
    install("codex", env, print_only=False, force=False, posture_hint=lambda: "-")
    (_target_dir("codex", env) / "llm-redact-status.md").write_text("edited", encoding="utf-8")
    assert status(env) == 0
    out = capsys.readouterr().out
    assert "claude" in out and "not installed" in out
    assert "codex" in out and "1 stale" in out


def test_install_prints_proxy_posture_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = _env(tmp_path)
    assert (
        install("cursor", env, print_only=False, force=False, posture_hint=lambda: "PROBE-LINE")
        == 0
    )
    out = capsys.readouterr().out
    assert "PROBE-LINE" in out


# --- the proxy-setup step (3.0): --proxy-url / --install-proxy / interactive ----


def test_install_proxy_url_probes_and_prints_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.plugin_cli import proxy_setup

    probed: list[str] = []

    def probe(url: str) -> bool:
        probed.append(url)
        return True

    code = proxy_setup(
        proxy_url="https://redact.corp.example:8787/",
        install_proxy=False,
        probe=probe,
        runner=lambda cmd: pytest.fail(f"runner must not be called: {cmd}"),
        ask=None,
    )
    assert code == 0
    assert probed == ["https://redact.corp.example:8787"]  # trailing slash stripped
    out = capsys.readouterr().out
    assert "existing proxy confirmed" in out
    assert "export LLM_REDACT_PROXY_URL=https://redact.corp.example:8787" in out


def test_install_proxy_url_dead_proxy_warns_but_succeeds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from llm_redact.plugin_cli import proxy_setup

    code = proxy_setup(
        proxy_url="https://redact.corp.example:8787",
        install_proxy=False,
        probe=lambda url: False,
        runner=lambda cmd: 0,
        ask=None,
    )
    assert code == 0
    assert "nothing answering" in capsys.readouterr().out


def test_install_proxy_url_refuses_plain_http_off_loopback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from llm_redact.plugin_cli import proxy_setup

    code = proxy_setup(
        proxy_url="http://redact.corp.example:8787",
        install_proxy=False,
        probe=lambda url: True,
        runner=lambda cmd: 0,
        ask=None,
    )
    assert code == 1
    assert "loopback-only" in capsys.readouterr().out


def test_install_proxy_runs_init_and_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.plugin_cli import proxy_setup

    # No config anywhere: init --yes runs first, then service install.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    code = proxy_setup(
        proxy_url=None, install_proxy=True, probe=lambda url: False, runner=runner, ask=None
    )
    assert code == 0
    assert [cmd[-2:] for cmd in calls] == [["init", "--yes"], ["service", "install"]]
    assert "login service" in capsys.readouterr().out


def test_interactive_prompt_offers_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_redact.plugin_cli import proxy_setup

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    calls: list[list[str]] = []
    answers = iter(["i"])
    code = proxy_setup(
        proxy_url=None,
        install_proxy=False,
        probe=lambda url: False,
        runner=lambda cmd: calls.append(cmd) or 0,
        ask=lambda prompt: next(answers),
    )
    assert code == 0
    assert calls and calls[-1][-2:] == ["service", "install"]


def test_interactive_prompt_point_at_url(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.plugin_cli import proxy_setup

    answers = iter(["p", "https://redact.corp.example:8787"])
    code = proxy_setup(
        proxy_url=None,
        install_proxy=False,
        # No LOCAL proxy answers (else the prompt is rightly skipped);
        # the remote one the user names does.
        probe=lambda url: url.startswith("https://"),
        runner=lambda cmd: pytest.fail("no install expected"),
        ask=lambda prompt: next(answers),
    )
    assert code == 0
    assert "export LLM_REDACT_PROXY_URL=" in capsys.readouterr().out


def test_interactive_prompt_skipped_when_proxy_already_answers() -> None:
    from llm_redact.plugin_cli import proxy_setup

    # probe() returning True for the local default means no prompt at all —
    # ask() failing loudly proves it was never consulted.
    code = proxy_setup(
        proxy_url=None,
        install_proxy=False,
        probe=lambda url: True,
        runner=lambda cmd: 0,
        ask=lambda prompt: pytest.fail("must not prompt when a proxy answers"),
    )
    assert code == 0


def test_interactive_default_is_skip(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.plugin_cli import proxy_setup

    code = proxy_setup(
        proxy_url=None,
        install_proxy=False,
        probe=lambda url: False,
        runner=lambda cmd: pytest.fail("skip must not install"),
        ask=lambda prompt: "",
    )
    assert code == 0
