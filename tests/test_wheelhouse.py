"""The Windows wheelhouse a release attaches (scripts/build_wheelhouse.py,
.github/workflows/wheelhouse.yml, docs/air-gapped.md).

An enclave without internet installs llm-redact on Windows from it: every
locked package of the base install and the crypto, keyring, realtime and
extract extras, as wheels for x64 Windows and Python 3.11 to 3.14, checked
against uv.lock's hashes, with SHA256SUMS and the release's provenance
attestation. The build runs on a Windows runner, where pip reads the
requirements' markers as Windows does, and runs the documented install with
no index before the zip is written. These tests pin the commands, the
workflows and the docs that repeat them; the build itself runs on the
Windows runner (on a pull request that changes it, and in every release).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_wheelhouse.py"
WORKFLOWS = ROOT / ".github" / "workflows"
DOCS = ROOT / "docs" / "air-gapped.md"
WHEEL = "llm_redact_proxy-9.9.9-py3-none-any.whl"


def _load() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("build_wheelhouse_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wheelhouse = _load()


def _workflow(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return loaded


def _windows_section() -> str:
    text = DOCS.read_text(encoding="utf-8")
    start = text.index("## Native Windows: the release's wheelhouse")
    return text[start : text.index("\n## ", start + 1)]


def test_sha256sums_lists_every_file_as_sha256sum_does(tmp_path: Path) -> None:
    (tmp_path / "b.whl").write_bytes(b"bee")
    (tmp_path / "a.whl").write_bytes(b"ay")
    (tmp_path / "SHA256SUMS").write_text("stale\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    expected = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in (("a.whl", b"ay"), ("b.whl", b"bee"))
    )
    assert wheelhouse.sha256sums(tmp_path) == expected


def test_the_downloads_take_the_locked_wheels_only_each_against_its_hash() -> None:
    export = wheelhouse.export_command(("crypto", "keyring"), Path("out/requirements.txt"))
    assert export[:6] == [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-header",
    ]
    assert export[6:] == [
        "--extra",
        "crypto",
        "--extra",
        "keyring",
        "-o",
        str(Path("out/requirements.txt")),
    ]
    download = wheelhouse.download_command(Path("py"), Path("r.txt"), Path("out"))
    assert download[:4] == ["py", "-m", "pip", "download"]
    for flag in ("--require-hashes", "--no-deps", "--only-binary=:all:"):
        assert flag in download


def test_the_install_has_no_index_and_ends_with_the_startup_gate_and_doctor() -> None:
    venv, wh = Path("venv"), Path("wh")
    commands = wheelhouse.install_commands(venv, wh, WHEEL, windows=True)
    python = str(venv / "Scripts" / "python.exe")
    llm_redact = str(venv / "Scripts" / "llm-redact.exe")
    locked, project = commands[0], commands[1]
    assert locked[:4] == [python, "-m", "pip", "install"]
    for flag in ("--no-index", "--no-deps", "--require-hashes"):
        assert flag in locked
    assert locked[-2:] == ["-r", str(wh / "requirements.txt")]
    # The release's wheel by its file name: a folder holding a higher
    # version beside it changes nothing.
    assert "--no-index" in project and "--find-links" not in project
    assert project[-2:] == ["--no-deps", str(wh / WHEEL)]
    assert commands[2:] == [
        [python, "-m", "pip", "check"],
        [llm_redact, "--version"],
        [llm_redact, "serve", "--check"],
        [llm_redact, "doctor"],
    ]
    elsewhere = wheelhouse.install_commands(venv, wh, WHEEL, windows=False)
    assert elsewhere[-1] == [str(venv / "bin" / "llm-redact"), "doctor"]


def test_the_offline_install_sees_no_pip_configuration_and_no_working_proxy() -> None:
    environ = {
        "PATH": "C:\\Windows",
        "SYSTEMROOT": "C:\\Windows",
        "PIP_INDEX_URL": "https://mirror.example/simple",
        "pip_find_links": "https://mirror.example/links",
        "https_proxy": "http://proxy.example:3128",
        "NO_PROXY": "127.0.0.1",
    }
    env = wheelhouse.offline_env(environ)
    assert env == {
        "PATH": "C:\\Windows",
        "SYSTEMROOT": "C:\\Windows",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9",
        "PIP_CONFIG_FILE": os.devnull,
    }


def test_the_zip_is_named_after_the_release_and_holds_the_files_at_its_top(
    tmp_path: Path,
) -> None:
    assert wheelhouse.zip_name(Path(WHEEL)) == "llm-redact-9.9.9-wheelhouse-win_amd64.zip"
    for other in ("httpx-0.28.1-py3-none-any.whl", "llm_redact_proxy-9.9.9.tar.gz"):
        with pytest.raises(SystemExit, match="is not an llm_redact_proxy wheel"):
            wheelhouse.zip_name(Path(other))
    folder = tmp_path / "wheelhouse"
    folder.mkdir()
    (folder / "b.whl").write_bytes(b"bee")
    (folder / "SHA256SUMS").write_text("sums\n", encoding="utf-8")
    (folder / "nested").mkdir()
    wheelhouse.write_zip(folder, tmp_path / "out.zip")
    with zipfile.ZipFile(tmp_path / "out.zip") as archive:
        assert archive.namelist() == ["SHA256SUMS", "b.whl"]
        assert archive.read("b.whl") == b"bee"


def test_the_extras_and_pythons_are_the_ones_the_docs_name() -> None:
    assert wheelhouse.EXTRAS == ("crypto", "keyring", "realtime", "extract")
    assert wheelhouse.PYTHONS == ("3.11", "3.12", "3.13", "3.14")
    section = _windows_section()
    assert "crypto, keyring, realtime and extract" in section
    assert "3.11 to 3.14" in section


def test_the_docs_give_the_install_the_build_checks() -> None:
    section = _windows_section()
    for line in (
        "-m pip install --no-index --find-links . --no-deps --require-hashes -r requirements.txt",
        "-m pip install --no-index --no-deps llm_redact_proxy-X.Y.Z-py3-none-any.whl",
        "-m pip check",
        "llm-redact.exe doctor",
        "gh attestation verify llm-redact-X.Y.Z-wheelhouse-win_amd64.zip"
        " --repo asanderson/llm-redact",
        "Get-FileHash -Algorithm SHA256",
        "wsl --import",
    ):
        assert line in section or line in DOCS.read_text(encoding="utf-8"), line


def test_main_builds_checks_with_no_index_then_zips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[tuple[list[str], Any, Any]] = []

    def run(command: list[str], *, env: Any = None, cwd: Any = None) -> None:
        ran.append((list(command), env, cwd))
        if command[:2] == ["uv", "export"]:
            Path(command[-1]).write_text("pypdf==6.5 --hash=sha256:00\n", encoding="utf-8")

    monkeypatch.setattr(wheelhouse, "_run", run)
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.example/simple")
    wheel = tmp_path / WHEEL
    wheel.write_bytes(b"wheel")
    out, zips = tmp_path / "wheelhouse", tmp_path / "zips"
    argv = ["--wheel", str(wheel), "--out", str(out), "--python", "3.12", "--zip", str(zips)]
    assert wheelhouse.main(argv) == 0
    assert (out / WHEEL).read_bytes() == b"wheel"
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8")
    assert f"  {WHEEL}\n" in sums and "  requirements.txt\n" in sums
    commands = [command for command, _, _ in ran]
    assert commands[0] == wheelhouse.export_command(wheelhouse.EXTRAS, out / "requirements.txt")
    assert [c[:2] for c in commands[:3]] == [
        ["uv", "export"],
        ["uv", "venv"],
        [commands[2][0], "-m"],
    ]
    assert commands[2][2:4] == ["pip", "download"]
    assert commands[3][:2] == ["uv", "venv"]
    assert [c[-1] for c in commands[4:]] == [
        str(out.resolve() / "requirements.txt"),
        str(out.resolve() / WHEEL),
        "check",
        "--version",
        "--check",
        "doctor",
    ]
    assert all("3.12" in c for c in commands if c[:2] == ["uv", "venv"])
    for _command, env, cwd in ran[4:]:
        assert env["HTTPS_PROXY"] == "http://127.0.0.1:9" and "PIP_INDEX_URL" not in env
        assert cwd is not None and Path(cwd) != ROOT
    with zipfile.ZipFile(zips / "llm-redact-9.9.9-wheelhouse-win_amd64.zip") as archive:
        assert sorted(archive.namelist()) == sorted(p.name for p in out.iterdir())


def test_extras_given_replace_the_four(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[list[str]] = []

    def run(command: list[str], *, env: Any = None, cwd: Any = None) -> None:
        ran.append(list(command))
        if command[:2] == ["uv", "export"]:
            Path(command[-1]).write_text("", encoding="utf-8")

    monkeypatch.setattr(wheelhouse, "_run", run)
    wheel = tmp_path / WHEEL
    wheel.write_bytes(b"wheel")
    out = tmp_path / "out"
    argv = ["--wheel", str(wheel), "--out", str(out), "--python", "3.13", "--no-check"]
    assert wheelhouse.main([*argv, "--extra", "hf", "--extra", "gliner"]) == 0
    assert ran[0] == wheelhouse.export_command(("hf", "gliner"), out / "requirements.txt")
    assert len(ran) == 3  # export, the 3.13 environment, the download: --no-check


def test_another_wheel_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wheelhouse, "_run", lambda *a, **k: pytest.fail("ran a command"))
    with pytest.raises(SystemExit, match="is not an llm_redact_proxy wheel"):
        wheelhouse.main(["--wheel", str(tmp_path / "x.whl"), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_the_workflow_builds_on_windows_with_read_access_only() -> None:
    workflow = _workflow("wheelhouse.yml")
    triggers = workflow[True]  # YAML 1.1 reads the key "on" as true
    assert set(triggers) == {"workflow_call", "workflow_dispatch", "pull_request"}
    assert set(triggers["pull_request"]["paths"]) == {
        ".github/workflows/wheelhouse.yml",
        "scripts/build_wheelhouse.py",
    }
    assert workflow["permissions"] == {"contents": "read"}
    (job,) = workflow["jobs"].values()
    assert job["runs-on"] == "windows-latest"
    assert "permissions" not in job
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "python scripts/build_wheelhouse.py" in runs
    assert "--no-check" not in runs  # the install with no index always runs
    uploads = [s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert [s["with"]["name"] for s in uploads] == ["wheelhouse"]


def test_a_release_attests_and_attaches_the_wheelhouse_after_the_release_exists() -> None:
    jobs = _workflow("release.yml")["jobs"]
    assert jobs["wheelhouse"]["uses"] == "./.github/workflows/wheelhouse.yml"
    assert jobs["wheelhouse"]["needs"] == "build"
    assert jobs["wheelhouse"]["with"] == {"dist-artifact": "dist"}
    publish = jobs["publish-wheelhouse"]
    assert set(publish["needs"]) == {"build", "wheelhouse"}
    assert publish["permissions"] == {
        "contents": "write",
        "id-token": "write",
        "attestations": "write",
    }
    steps = publish["steps"]
    uses = [step.get("uses", "") for step in steps]
    assert any(u.startswith("actions/download-artifact@") for u in uses)
    assert any(u.startswith("actions/attest-build-provenance@") for u in uses)
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "gh attestation verify" in runs
    assert 'gh release upload "$GITHUB_REF_NAME"' in runs
    # The other publish jobs never wait for it: a failed wheelhouse leaves
    # PyPI and the images as they are.
    for name in ("publish-pypi", "publish-ghcr"):
        needs = jobs[name]["needs"]
        assert "wheelhouse" not in (needs if isinstance(needs, list) else [needs])


@pytest.mark.parametrize("name", ["wheelhouse.yml", "release.yml"])
def test_actions_are_pinned_and_checkouts_keep_no_credentials(name: str) -> None:
    for job in _workflow(name)["jobs"].values():
        for step in job.get("steps", []):
            uses = step.get("uses")
            if uses is None:
                continue
            assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", uses), uses
            if uses.startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False
