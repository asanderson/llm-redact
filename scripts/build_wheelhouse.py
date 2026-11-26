"""Build the Windows wheelhouse a release attaches (docs/air-gapped.md,
"Native Windows: the release's wheelhouse").

Run on a Windows machine (CI: windows-latest, .github/workflows/wheelhouse.yml),
from a checkout of the release, with uv on PATH:

    python scripts/build_wheelhouse.py --wheel dist/llm_redact_proxy-X.Y.Z-py3-none-any.whl \
        --out wheelhouse --zip .

For each supported Python it downloads, with pip in a throwaway virtual
environment of that Python, every package `uv.lock` names for the base
package and the crypto, keyring, realtime and extract extras: wheels only,
each checked against its locked hash. Running there, pip reads the
requirements' markers as Windows does (on another OS, `pip download
--platform win_amd64` still reads them for the machine it runs on and leaves
out the Windows-only packages). Then it adds the release's own wheel, the
requirements file the offline install reads, and SHA256SUMS, and runs the
install docs/air-gapped.md gives in a fresh environment of each Python:
no index, no pip configuration, every proxy variable pointing at a closed
port, then `pip check`, `serve --check` and `doctor`. A wheelhouse missing a
package fails here, not in an enclave. `--zip DIR` writes it, once checked,
as `llm-redact-<version>-wheelhouse-win_amd64.zip`. `--extra NAME` (repeated)
replaces the four extras, for a wheelhouse of your own (`--extra hf --extra
gliner` for NER, which the release's does not carry).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

PYTHONS = ("3.11", "3.12", "3.13", "3.14")
EXTRAS = ("crypto", "keyring", "realtime", "extract")
DISTRIBUTION = "llm_redact_proxy"
REQUIREMENTS = "requirements.txt"
SUMS = "SHA256SUMS"
# Port 9 (discard) has no listener on a CI runner: a command that tried to
# reach a network through any of these would fail at once.
CLOSED_PROXY = "http://127.0.0.1:9"
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


def venv_command(venv: Path, name: str, *, windows: bool | None = None) -> Path:
    """A command of a virtual environment: ``python``, ``llm-redact``."""
    if os.name == "nt" if windows is None else windows:
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def release_version(wheel: Path) -> str:
    """The version in the release wheel's file name (PEP 427)."""
    parts = wheel.name.split("-")
    if wheel.suffix != ".whl" or len(parts) < 5 or parts[0] != DISTRIBUTION:
        raise SystemExit(f"{wheel.name} is not an {DISTRIBUTION} wheel")
    return parts[1]


def zip_name(wheel: Path) -> str:
    return f"llm-redact-{release_version(wheel)}-wheelhouse-win_amd64.zip"


def sha256sums(folder: Path) -> str:
    """``sha256sum``'s lines for every file in ``folder`` but SHA256SUMS
    itself, sorted by name: what `sha256sum -c` checks, and the PowerShell
    loop in docs/air-gapped.md."""
    lines = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if path.name == SUMS or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}\n")
    return "".join(lines)


def export_command(extras: Sequence[str], output: Path) -> list[str]:
    """The locked closure of the base package and ``extras``, with hashes."""
    command = ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--no-header"]
    for extra in extras:
        command += ["--extra", extra]
    return [*command, "-o", str(output)]


def download_command(python: Path, requirements: Path, out: Path) -> list[str]:
    """Wheels only, each against its locked hash, for ``python``'s platform."""
    return [
        str(python),
        "-m",
        "pip",
        "download",
        "--disable-pip-version-check",
        "--require-hashes",
        "--no-deps",
        "--only-binary=:all:",
        "-r",
        str(requirements),
        "-d",
        str(out),
    ]


def install_commands(
    venv: Path, wheelhouse: Path, wheel_name: str, *, windows: bool | None = None
) -> list[list[str]]:
    """The offline install docs/air-gapped.md gives: no index at all, the
    locked closure against its hashes, then the release's wheel by its file
    name (never the highest version a folder happens to hold)."""
    python = str(venv_command(venv, "python", windows=windows))
    llm_redact = str(venv_command(venv, "llm-redact", windows=windows))
    pip = [python, "-m", "pip", "install", "--disable-pip-version-check", "--no-index"]
    links = ["--find-links", str(wheelhouse), "--no-deps"]
    return [
        [*pip, *links, "--require-hashes", "-r", str(wheelhouse / REQUIREMENTS)],
        [*pip, "--no-deps", str(wheelhouse / wheel_name)],
        [python, "-m", "pip", "check"],
        [llm_redact, "--version"],
        [llm_redact, "serve", "--check"],
        [llm_redact, "doctor"],
    ]


def offline_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment of the offline install: no pip configuration and no
    proxy but a closed port, so nothing it runs reaches a network unseen."""
    dropped = {*PROXY_VARIABLES, "NO_PROXY"}
    env = {
        name: value
        for name, value in environ.items()
        if not name.upper().startswith("PIP_") and name.upper() not in dropped
    }
    env.update(dict.fromkeys(PROXY_VARIABLES, CLOSED_PROXY))
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


def write_zip(folder: Path, destination: Path) -> None:
    """Every file of ``folder`` at the archive's top, sorted by name."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir(), key=lambda p: p.name):
            if path.is_file():
                archive.write(path, arcname=path.name)


def _run(
    command: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: Path | None = None
) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env, cwd=cwd)


def _venv(python: str, folder: Path) -> Path:
    _run(["uv", "venv", "--seed", "--python", python, str(folder)])
    return folder


def build(wheel: Path, out: Path, pythons: Sequence[str], extras: Sequence[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _run(export_command(extras, out / REQUIREMENTS))
    with tempfile.TemporaryDirectory(prefix="wheelhouse-") as scratch:
        for python in pythons:
            venv = _venv(python, Path(scratch) / f"download-{python}")
            _run(download_command(venv_command(venv, "python"), out / REQUIREMENTS, out))
    shutil.copy2(wheel, out / wheel.name)
    (out / SUMS).write_text(sha256sums(out), encoding="utf-8", newline="\n")


def check(out: Path, pythons: Sequence[str], wheel_name: str) -> None:
    env = offline_env(os.environ)
    with tempfile.TemporaryDirectory(prefix="wheelhouse-check-") as scratch:
        for python in pythons:
            venv = _venv(python, Path(scratch) / f"check-{python}")
            for command in install_commands(venv, out.resolve(), wheel_name):
                _run(command, env=env, cwd=Path(scratch))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--wheel", type=Path, required=True, help="the release's wheel")
    parser.add_argument("--out", type=Path, required=True, help="the wheelhouse folder")
    parser.add_argument("--python", action="append", help="a Python version (default: all)")
    parser.add_argument(
        "--extra", action="append", help="an extra to include (default: " + ", ".join(EXTRAS) + ")"
    )
    parser.add_argument("--zip", type=Path, help="write the checked wheelhouse's zip here")
    parser.add_argument("--no-check", action="store_true", help="skip the offline install")
    args = parser.parse_args(argv)
    name = zip_name(args.wheel)  # refuses another wheel before any download
    pythons = args.python or list(PYTHONS)
    build(args.wheel, args.out, pythons, args.extra or list(EXTRAS))
    if not args.no_check:
        check(args.out, pythons, args.wheel.name)
    print(f"wheelhouse: {sum(1 for _ in args.out.iterdir())} files in {args.out}")
    if args.zip is not None:
        args.zip.mkdir(parents=True, exist_ok=True)
        write_zip(args.out, args.zip / name)
        print(f"zip: {args.zip / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
