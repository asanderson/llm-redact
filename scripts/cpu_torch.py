#!/usr/bin/env python3
"""Install the locked NER extras with a CPU-only torch.

PyPI's Linux torch wheel (both x86_64 and aarch64) depends on the CUDA
libraries and triton — several GB no CPU host, CI runner or container image
uses. The PyTorch CPU index carries the same torch release without them.
uv.lock pins the PyPI wheel, so a CPU install takes torch at its LOCKED
version from the CPU index and everything else exactly as uv.lock pins it
(hash-checked), minus torch's GPU dependencies. Used by the `-ner` image
(Dockerfile), the CI `airgap` job and the offline wheelhouse recipe
(docs/deployment.md, "Offline installs"):

    uv export --frozen --no-dev --no-emit-project --extra hf --extra gliner \\
        | python scripts/cpu_torch.py requirements requirements.txt > torch.txt
    #   requirements.txt: every locked package but torch and its GPU
    #                     dependencies, with their hashes;
    #   torch.txt:        "torch==<locked version>" (install it from
    #                     https://download.pytorch.org/whl/cpu, --no-deps)
    python scripts/cpu_torch.py check   # in the installed environment

``check`` exits 1 when a CUDA or triton distribution is installed, or when a
Linux torch is not a ``+cpu`` build. Standard library only.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from importlib import metadata

# Distributions a CPU-only torch never needs: torch itself (installed from
# the CPU index instead), triton (its GPU compiler) and the CUDA runtime.
GPU_ONLY = re.compile(r"(?:torch|triton|pytorch-triton|nvidia-[a-z0-9._-]+|cuda-[a-z0-9._-]+)")
_REQUIREMENT = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def split(lines: Iterable[str]) -> tuple[list[str], str | None]:
    """The export's lines without torch's and its GPU dependencies' entries
    (an entry is its requirement line and the indented ``--hash`` lines
    under it), and torch's locked version (None when the export has no
    torch)."""
    kept: list[str] = []
    torch: str | None = None
    dropping = False
    for line in lines:
        if line[:1].isspace():  # a continuation of the entry above
            if not dropping:
                kept.append(line)
            continue
        found = _REQUIREMENT.match(line)
        name = _normalize(found.group(1)) if found else None
        dropping = name is not None and GPU_ONLY.fullmatch(name) is not None
        if name == "torch" and found is not None:
            torch = found.group(2)
        if not dropping:
            kept.append(line)
    return kept, torch


def gpu_distributions() -> list[str]:
    """The installed distributions a CPU-only environment must not hold."""
    names = {_normalize(dist.metadata["Name"] or "") for dist in metadata.distributions()}
    return sorted(name for name in names if name != "torch" and GPU_ONLY.fullmatch(name))


def check() -> int:
    problems = [f"GPU-only distribution installed: {name}" for name in gpu_distributions()]
    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        problems.append("torch is not installed")
    else:
        if sys.platform.startswith("linux") and not version.endswith("+cpu"):
            problems.append(f"torch {version} is not a CPU-only build (+cpu)")
    for problem in problems:
        print(f"FAIL  {problem}", file=sys.stderr)
    if not problems:
        print(f"OK    CPU-only torch {version}, no CUDA or triton distribution installed")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["check"] and len(argv) == 1:
        return check()
    if argv[:1] == ["requirements"] and len(argv) == 2:
        kept, torch = split(sys.stdin.read().splitlines(keepends=True))
        if torch is None:
            print("FAIL  the export holds no torch", file=sys.stderr)
            return 1
        with open(argv[1], "w", encoding="utf-8") as out:
            out.writelines(kept)
        print(f"torch=={torch}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
