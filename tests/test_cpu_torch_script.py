"""scripts/cpu_torch.py: the locked NER extras with a CPU-only torch.

The `-ner` image, the CI `airgap` job and the offline wheelhouse recipe
install torch at its locked version from the PyTorch CPU index and every
other locked package hash-checked from the export — without torch's GPU
dependencies, which PyPI's Linux torch pulls on every architecture.
"""

from __future__ import annotations

import importlib.util
import io
import sys
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
    assert capsys.readouterr().out == "torch==2.13.0\n"
    assert "torch" not in out.read_text(encoding="utf-8").replace("transformers", "")


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
