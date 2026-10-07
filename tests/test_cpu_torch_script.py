"""scripts/cpu_torch.py: the locked NER extras with a CPU-only torch.

The `-ner` image, the CI `airgap` job, the real-model CI jobs
(scripts/ner_ci_env.sh) and the offline wheelhouse recipe install torch at
its locked version from the PyTorch CPU index and every other locked
package hash-checked from the export — without torch's GPU dependencies,
which PyPI's Linux torch pulls on every architecture.
"""

from __future__ import annotations

import importlib.util
import io
import re
import sys
import tomllib
import types
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cpu_torch.py"


def _load() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("cpu_torch_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu_torch = _load()

EXPORT = """\
# a comment line is kept
filelock==3.20.0 \\
    --hash=sha256:aaaa
nvidia-cublas==13.1.1.3 ; sys_platform == 'linux' \\
    --hash=sha256:bbbb \\
    --hash=sha256:cccc
sympy==1.14.0 \\
    --hash=sha256:dddd
torch==2.13.0 \\
    --hash=sha256:eeee
triton==3.7.1 ; python_full_version < '3.15' and sys_platform == 'linux' \\
    --hash=sha256:ffff
cuda-bindings==13.3.1 ; sys_platform == 'linux' \\
    --hash=sha256:0000
nvidia_nccl_cu13==2.29.7 ; sys_platform == 'linux'
transformers==5.10.1 \\
    --hash=sha256:1111
"""


def test_split_drops_torch_and_its_gpu_dependencies_whole() -> None:
    kept, torch = cpu_torch.split(EXPORT.splitlines(keepends=True))
    assert torch == "2.13.0"
    assert "".join(kept) == (
        "# a comment line is kept\n"
        "filelock==3.20.0 \\\n"
        "    --hash=sha256:aaaa\n"
        "sympy==1.14.0 \\\n"
        "    --hash=sha256:dddd\n"
        "transformers==5.10.1 \\\n"
        "    --hash=sha256:1111\n"
    )


def test_requirements_writes_the_rest_and_prints_the_torch_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(EXPORT))
    out = tmp_path / "requirements.txt"
    assert cpu_torch.main(["requirements", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("torch==2.13.0+cpu \\\n    --hash=sha256:")
    assert printed.endswith("\n") and not printed.endswith("\\\n")
    assert "torch" not in out.read_text(encoding="utf-8").replace("transformers", "")


def _hashes(pinned: str) -> set[str]:
    return set(re.findall(r"--hash=sha256:([0-9a-f]{64})", pinned))


def test_the_torch_pin_carries_every_recorded_cpu_wheel_digest() -> None:
    wheels = cpu_torch.CPU_WHEELS["2.13.0"]
    plus_cpu = {digest for name, digest in wheels.items() if "+cpu-" in name}
    macos = {digest for name, digest in wheels.items() if "macosx" in name}
    assert plus_cpu and macos and plus_cpu | macos == set(wheels.values())
    pinned = cpu_torch.torch_requirement("2.13.0")
    assert pinned is not None and pinned.splitlines()[0] == "torch==2.13.0+cpu \\"
    assert _hashes(pinned) == plus_cpu
    mac = cpu_torch.torch_requirement("2.13.0", macos=True)
    assert mac is not None and mac.splitlines()[0] == "torch==2.13.0 \\"
    assert _hashes(mac) == macos
    # The image's wheel, downloaded and hashed when the table was recorded.
    assert (
        wheels["torch-2.13.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl"]
        == "3fbf9c9d1f3c10c2d59d04aca426dee9ccc6ceb32d255c61e93acc3b4f75fae6"
    )


def _locked_torch() -> str:
    lock = tomllib.loads((SCRIPT.parents[1] / "uv.lock").read_text(encoding="utf-8"))
    (version,) = [package["version"] for package in lock["package"] if package["name"] == "torch"]
    return str(version)


def test_the_recorded_cpu_wheels_are_the_locked_torch() -> None:
    # A torch bump in uv.lock fails here until CPU_WHEELS records the new
    # version's CPU wheels (https://download.pytorch.org/whl/cpu/torch/).
    locked = _locked_torch()
    assert set(cpu_torch.CPU_WHEELS) == {locked}
    for name, digest in cpu_torch.CPU_WHEELS[locked].items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), name
        assert re.fullmatch(
            rf"torch-{re.escape(locked)}(?:\+cpu)?-cp3(?:1[1-9])-cp3(?:1[1-9])t?-[a-z0-9_]+\.whl",
            name,
        ), name
    tags = {name.split("-", 2)[2] for name in cpu_torch.CPU_WHEELS[locked]}
    # Both architectures of the -ner image (Python 3.13) and the wheelhouse
    # recipe's Linux and macOS hosts on every supported Python.
    for python in ("cp311", "cp312", "cp313", "cp314"):
        for platform in ("manylinux_2_28_x86_64", "manylinux_2_28_aarch64"):
            assert f"{python}-{python}-{platform}.whl" in tags
        assert f"{python}-{python}-macosx_14_0_arm64.whl" in tags


def test_requirements_without_recorded_wheels_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("torch==9.9.9 \\\n    --hash=sha256:ab\n"))
    out = tmp_path / "r.txt"
    assert cpu_torch.main(["requirements", str(out)]) == 1
    assert "no CPU wheel hash recorded for torch 9.9.9" in capsys.readouterr().err
    assert not out.exists()


def test_requirements_for_macos_pins_the_plain_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(EXPORT))
    assert cpu_torch.main(["requirements", "--macos", str(tmp_path / "r.txt")]) == 0
    assert capsys.readouterr().out == cpu_torch.torch_requirement("2.13.0", macos=True)
    assert (tmp_path / "r.txt").exists()


def test_requirements_without_torch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("filelock==3.20.0\n"))
    assert cpu_torch.main(["requirements", str(tmp_path / "r.txt")]) == 1
    assert "holds no torch" in capsys.readouterr().err


def test_bad_arguments_print_the_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert cpu_torch.main([]) == 2
    assert "CPU-only torch" in capsys.readouterr().err


def _installed(monkeypatch: pytest.MonkeyPatch, dists: dict[str, str], platform: str) -> None:
    fakes = [types.SimpleNamespace(metadata={"Name": name}) for name in dists]
    monkeypatch.setattr(cpu_torch.metadata, "distributions", lambda: fakes)

    def version(name: str) -> str:
        if name not in dists:
            raise cpu_torch.metadata.PackageNotFoundError(name)
        return dists[name]

    monkeypatch.setattr(cpu_torch.metadata, "version", version)
    monkeypatch.setattr(cpu_torch.sys, "platform", platform)


@pytest.mark.parametrize(
    ("dists", "platform", "code", "said"),
    [
        ({"torch": "2.13.0+cpu", "sympy": "1.14"}, "linux", 0, "OK    CPU-only torch 2.13.0+cpu"),
        ({"torch": "2.13.0", "sympy": "1.14"}, "darwin", 0, "OK    CPU-only torch 2.13.0"),
        ({"torch": "2.13.0"}, "linux", 1, "torch 2.13.0 is not a CPU-only build"),
        ({"torch": "2.13.0+cpu", "nvidia-cublas": "13"}, "linux", 1, "nvidia-cublas"),
        ({"torch": "2.13.0+cpu", "triton": "3.7"}, "linux", 1, "installed: triton"),
        ({"sympy": "1.14"}, "linux", 1, "torch is not installed"),
    ],
)
def test_check(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dists: dict[str, str],
    platform: str,
    code: int,
    said: str,
) -> None:
    _installed(monkeypatch, dists, platform)
    assert cpu_torch.main(["check"]) == code
    captured: Any = capsys.readouterr()
    assert said in captured.out + captured.err
