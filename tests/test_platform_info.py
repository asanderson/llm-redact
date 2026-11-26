"""Which platform the proxy runs on, and where a path really is
(platform_info.py), with the consequences: in WSL a SQLite vault on a
Windows drive (a 9p mount under /mnt/c in WSL 2, drvfs in WSL 1) is refused
at startup and FAILs in doctor, the other stores and model folders there
WARN; doctor shows WSL, its distribution and networking mode; /status says
which platform answered.

Everything runs on fakes: the kernel release, the mount table, wslinfo."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_redact import platform_info
from llm_redact.config import AuditConfig, Config, ConfigError, OverridesConfig, VaultConfig

WSL2_RELEASE = "5.15.167.4-microsoft-standard-WSL2\n"
WSL1_RELEASE = "4.4.0-22621-Microsoft\n"
LINUX_RELEASE = "6.8.0-48-generic\n"


def test_mountinfo_lines_are_read_with_their_escapes() -> None:
    text = (
        "24 1 8:32 / / rw,relatime shared:1 - ext4 /dev/sdc rw,discard\n"
        "50 24 0:47 / /mnt/c rw,noatime - 9p C:\\134 rw,dirsync,aname=drvfs;path=C:\\\n"
        "51 24 0:48 / /mnt/my\\040drive rw - drvfs D:\\134 rw\n"
        "not a mountinfo line\n"
    )
    assert platform_info.parse_mountinfo(text) == [
        platform_info.Mount("/", "ext4", "/dev/sdc"),
        platform_info.Mount("/mnt/c", "9p", "C:\\"),
        platform_info.Mount("/mnt/my drive", "drvfs", "D:\\"),
    ]


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_windows_never_reads_the_linux_files(
    platform: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        pytest.fail(f"read or ran {args}")

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(platform_info, "_read", refuse)
    monkeypatch.setattr(platform_info.os.path, "exists", refuse)
    monkeypatch.setattr(platform_info.subprocess, "run", refuse)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    assert platform_info.is_wsl() is False
    assert platform_info.mount_of(Path("x")) is None
    assert platform_info.networking_mode() is None
    assert platform_info.windows_drive_problem(Path("x")) is None


def _mountinfo(root: Path, *windows: str) -> str:
    lines = ["24 1 8:32 / / rw,relatime - ext4 /dev/sdc rw"]
    for n, point in enumerate(windows):
        lines.append(f"{50 + n} 24 0:{47 + n} / {point} rw,noatime - 9p C:\\134 rw,aname=drvfs")
    return "\n".join(lines) + "\n"


@pytest.mark.skipif(os.name != "posix", reason="mount points are POSIX paths")
def test_a_path_is_on_the_deepest_mount_holding_it(tmp_path: Path) -> None:
    root = Path(os.path.realpath(tmp_path))
    drive = root / "mnt" / "c"
    text = _mountinfo(root, str(drive))
    assert platform_info.mount_of(drive / "Users" / "a" / "vault.db", text).fstype == "9p"
    assert platform_info.mount_of(root / "home" / "vault.db", text).fstype == "ext4"
    # "/mnt/cdrom" is not under "/mnt/c".
    assert platform_info.mount_of(Path(f"{drive}drom") / "x", text).fstype == "ext4"


@pytest.mark.skipif(os.name != "posix", reason="a symlink into the drive")
def test_a_link_into_a_windows_drive_counts_as_the_drive(tmp_path: Path) -> None:
    root = Path(os.path.realpath(tmp_path))
    drive = root / "mnt" / "c"
    (drive / "data").mkdir(parents=True)
    (root / "home").mkdir()
    (root / "home" / "data").symlink_to(drive / "data")
    mount = platform_info.mount_of(
        root / "home" / "data" / "vault.db", _mountinfo(root, str(drive))
    )
    assert mount is not None and mount.point == str(drive)


@pytest.mark.parametrize(
    ("platform", "environ", "release", "wsl"),
    [
        ("linux", {"WSL_DISTRO_NAME": "Ubuntu-24.04"}, LINUX_RELEASE, True),
        ("linux", {"WSL_INTEROP": "/run/WSL/1_interop"}, LINUX_RELEASE, True),
        ("linux", {}, WSL2_RELEASE, True),
        ("linux", {}, WSL1_RELEASE, True),
        ("linux", {}, LINUX_RELEASE, False),
        ("linux", {}, None, False),
        ("darwin", {"WSL_DISTRO_NAME": "x"}, WSL2_RELEASE, False),
        ("win32", {"WSL_DISTRO_NAME": "x"}, WSL2_RELEASE, False),
    ],
)
def test_wsl_is_detected_from_its_variables_or_the_kernel(
    platform: str,
    environ: dict[str, str],
    release: str | None,
    wsl: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    assert platform_info.is_wsl(environ, release) is wsl


@pytest.mark.parametrize(
    ("platform", "wsl", "name"),
    [
        ("win32", False, "windows"),
        ("darwin", False, "macos"),
        ("linux", False, "linux"),
        ("linux", True, "wsl"),
        ("freebsd14", False, "freebsd14"),
    ],
)
def test_the_platform_name(
    platform: str, wsl: bool, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(platform_info, "is_wsl", lambda *a: wsl)
    assert platform_info.platform_name() == name


@pytest.mark.skipif(os.name != "posix", reason="a shell script stands in for wslinfo")
@pytest.mark.parametrize(
    ("script", "mode"),
    [
        ("echo mirrored", "mirrored"),
        ("echo NAT", "nat"),
        ("echo something-new", None),
        ("exit 1", None),
    ],
)
def test_the_networking_mode_comes_from_wslinfo(
    script: str, mode: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")  # it is asked on Linux only
    wslinfo = tmp_path / "wslinfo"
    wslinfo.write_text(f'#!/bin/sh\n[ "$1" = --networking-mode ] || exit 2\n{script}\n')
    wslinfo.chmod(wslinfo.stat().st_mode | stat.S_IXUSR)
    assert platform_info.networking_mode(str(wslinfo)) == mode
    assert platform_info.networking_mode(str(tmp_path / "missing")) is None


@pytest.fixture
def wsl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """WSL with a Windows drive mounted at <tmp>/mnt/c; returns the drive."""
    root = Path(os.path.realpath(tmp_path))
    drive = root / "mnt" / "c"
    drive.mkdir(parents=True)
    table = root / "mountinfo"
    table.write_text(_mountinfo(root, str(drive)), encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-24.04")
    monkeypatch.setattr(platform_info, "MOUNTINFO", table)
    monkeypatch.setattr(platform_info, "networking_mode", lambda wslinfo=None: "mirrored")
    return drive


@pytest.mark.skipif(os.name != "posix", reason="mount points are POSIX paths")
def test_a_vault_on_a_windows_drive_is_refused_at_startup(wsl: Path, tmp_path: Path) -> None:
    from llm_redact.vault import build_vault, build_vault_manager

    path = wsl / "Users" / "ada" / "vault.db"
    linux_default = Path.home() / ".local" / "share" / "llm-redact" / "vault.db"
    expected = (
        f"[vault] {path} is on a Windows drive mounted into WSL ({wsl}, 9p): SQLite cannot"
        " lock files there reliably, so the vault could lose writes and issue a token number"
        f" twice; keep it in the Linux file system (the default is {linux_default})"
    )
    for build in (build_vault, build_vault_manager):
        with pytest.raises(ConfigError) as refused:
            build(VaultConfig(backend="sqlite", path=str(path)))
        assert str(refused.value) == expected
    assert not path.parent.exists()  # refused before anything was created
    # One in the Linux file system opens as before.
    manager = build_vault_manager(VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")))
    manager.close()


@pytest.mark.skipif(os.name != "posix", reason="mount points are POSIX paths")
def test_outside_wsl_a_9p_vault_opens_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact.vault import build_vault_manager

    root = Path(os.path.realpath(tmp_path))
    table = root / "mountinfo"
    table.write_text(_mountinfo(root, str(root / "share")), encoding="utf-8")
    monkeypatch.setattr(platform_info, "MOUNTINFO", table)
    monkeypatch.setattr(platform_info, "is_wsl", lambda *a: False)
    manager = build_vault_manager(VaultConfig(backend="sqlite", path=str(root / "share" / "v.db")))
    manager.close()


def _doctor_rows(config: Config) -> list[dict[str, Any]]:
    from llm_redact.doctor_cli import _check_paths, _check_platform, _check_vault, _Report

    report = _Report(json_mode=True)
    _check_platform(report)
    _check_vault(report, config)
    _check_paths(report, config)
    return report.rows


@pytest.mark.skipif(os.name != "posix", reason="mount points are POSIX paths")
def test_doctor_fails_a_vault_and_warns_other_stores_on_a_windows_drive(
    wsl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(wsl / "data"))
    config = Config(
        vault=VaultConfig(backend="sqlite", path=str(wsl / "vault.db")),
        audit=AuditConfig(enabled=True),
        overrides=OverridesConfig(enabled=True),
    )
    rows = _doctor_rows(config)
    (platform_row,) = [row for row in rows if row["area"] == "platform"]
    assert platform_row["level"] == "PASS"
    message = platform_row["message"]
    assert "WSL" in message and "Ubuntu-24.04" in message and "mirrored" in message
    vault = [row for row in rows if row["area"] == "vault" and row["level"] == "FAIL"]
    assert len(vault) == 1 and "Windows drive" in vault[0]["message"]
    warned = [
        row["message"]
        for row in rows
        if row["level"] == "WARN" and "Windows drive" in row["message"]
    ]
    assert any("audit" in m for m in warned)
    assert any("override" in m for m in warned)
    assert any("model folders" in m for m in warned)


def test_doctor_outside_wsl_says_nothing_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform_info, "is_wsl", lambda *a: False)
    rows = _doctor_rows(Config(vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db"))))
    assert not [row for row in rows if row["area"] == "platform"]
    assert not [row for row in rows if "Windows drive" in row["message"]]


async def test_status_says_which_platform_answered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.cli import run_status
    from llm_redact.proxy import create_app

    monkeypatch.setattr(platform_info, "platform_name", lambda: "wsl")
    app = create_app(Config())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        payload = (await client.get("/__llm-redact/status")).json()
    assert payload["platform"] == "wsl"

    # `llm-redact status` prints it on its first line (and an older proxy's
    # answer, with no platform, as before).
    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.delenv("LLM_REDACT_PROXY_URL", raising=False)
    monkeypatch.setattr(httpx, "get", fake_get)
    args = argparse.Namespace(config=None, port=8787, json=False, ca=None, cert=None, key=None)
    assert run_status(args) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith(f"llm-redact {payload['version']} on wsl — up ")
    del payload["platform"]
    assert run_status(args) == 0
    assert " on " not in capsys.readouterr().out.splitlines()[0]


def _loopback_rows(config: Config) -> list[str]:
    from llm_redact.doctor_cli import _check_loopback_names, _Report

    report = _Report(json_mode=True)
    _check_loopback_names(report, config)
    assert all(row["level"] == "WARN" for row in report.rows)
    return [row["message"] for row in report.rows]


def _named(ollama: str, llm: str | None = None) -> Config:
    import dataclasses

    from llm_redact.config import DEFAULT_PROVIDERS, ProviderConfig
    from llm_redact.detection.engine import DetectionConfig, LlmNerConfig, NerConfig

    providers = dict(DEFAULT_PROVIDERS)
    providers["ollama"] = ProviderConfig(upstream_base_url=ollama)
    providers["custom:lmstudio"] = ProviderConfig(upstream_base_url="http://[::1]:1234/v1")
    detection = DetectionConfig()
    if llm is not None:
        ner = NerConfig(enabled=True, backends=("llm",), llm=LlmNerConfig(url=llm, model="m"))
        detection = dataclasses.replace(detection, ner=ner)
    return Config(providers=providers, detection=detection)


@pytest.mark.parametrize("where", ["windows", "wsl"])
def test_doctor_warns_a_localhost_upstream_on_windows_and_in_wsl(
    where: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32" if where == "windows" else "linux")
    monkeypatch.setattr(platform_info, "is_wsl", lambda *a: where == "wsl")
    messages = _loopback_rows(_named("http://localhost:11434", llm="http://localhost:11434"))
    assert len(messages) == 3
    lmstudio, ollama, llm = messages  # providers in name order, then the llm backend
    assert ollama.startswith("[providers.ollama] upstream_base_url names localhost")
    assert lmstudio.startswith("[providers.custom.lmstudio] upstream_base_url names ::1")
    assert llm.startswith("[detection.ner.llm] url names localhost")
    assert all("127.0.0.1" in message for message in messages)
    assert _loopback_rows(_named("http://127.0.0.1:11434"))[:1] == [lmstudio]


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_no_loopback_warning_elsewhere(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(platform_info, "is_wsl", lambda *a: False)
    assert _loopback_rows(_named("http://localhost:11434", llm="http://localhost:11434")) == []
