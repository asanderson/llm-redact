#!/usr/bin/env python3
"""Install the locked NER extras with a CPU-only torch.

PyPI's Linux torch wheel (both x86_64 and aarch64) depends on the CUDA
libraries and triton — several GB no CPU host, CI runner or container image
uses. The PyTorch CPU index carries the same torch release without them.
uv.lock pins the PyPI wheel, so a CPU install takes torch at its LOCKED
version from the CPU index and everything else exactly as uv.lock pins it
(hash-checked), minus torch's GPU dependencies. torch is hash-checked too:
uv.lock holds only the PyPI wheels' hashes, so ``CPU_WHEELS`` below records
the CPU index's SHA-256 of every wheel of the locked version (a test pins
the table to uv.lock's torch, so a torch bump fails until it is updated).
It is the one CPU-torch recipe: the `-ner` image (Dockerfile), its release
SBOM, the CI `airgap` job, scripts/ner_ci_env.sh (the CI `ner-models` and
`ner-eval` jobs) and the offline wheelhouse recipe (docs/deployment.md,
"Offline installs") all use it:

    uv export --frozen --no-dev --no-emit-project --extra hf --extra gliner \\
        | python scripts/cpu_torch.py requirements requirements.txt > torch.txt
    #   requirements.txt: every locked package but torch and its GPU
    #                     dependencies, with their hashes;
    #   torch.txt:        "torch==<locked version>+cpu" with the CPU
    #                     wheels' hashes (install it from
    #                     https://download.pytorch.org/whl/cpu with
    #                     --no-deps --require-hashes)
    python scripts/cpu_torch.py check   # in the installed environment

``requirements --macos`` pins the plain ``torch==<locked version>`` with
its macOS wheels' hashes instead (macOS has no ``+cpu`` build; its wheel is
CPU-only). ``requirements`` exits 1 when the export holds no torch or
``CPU_WHEELS`` records no wheel of its version for the platform.

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
# The PyTorch CPU index's SHA-256 of every torch wheel of the locked version
# for the Pythons llm-redact supports (>= 3.11), as the index lists them at
# https://download.pytorch.org/whl/cpu/torch/ (checked 2026-10-06; the
# cp313 manylinux x86_64 wheel was downloaded and its digest recomputed).
# Linux and Windows wheels are the "+cpu" builds; macOS has only the plain
# one. A wheel whose digest is not listed here is refused by
# --require-hashes, so a substituted or re-uploaded wheel never installs.
_CPU_WHEEL_SHA256 = """\
6e9817dbdf5ea76789babd46e457eac5bf14ff566cf85f8addbfdff2d56601ce  torch-2.13.0+cpu-cp311-cp311-linux_s390x.whl
84453b69508ec79902f899c5ed9495acb9e2bbe9fda5f1d5d6f19e3c3842e1a7  torch-2.13.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl
6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b  torch-2.13.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl
10717d8b3b67c45a4788bf7ffc0bab1ea1e5ebbedd24466be6100102d141fac1  torch-2.13.0+cpu-cp311-cp311-win_amd64.whl
2b3d093abd919ad934c43d47e73ba63ceba7cbd7269fc2e9c1e4fc29e8fe45fa  torch-2.13.0+cpu-cp311-cp311-win_arm64.whl
ffadde149901c8afa138daa38d898264003cfcf1a3336ca5cd964b5af227d867  torch-2.13.0+cpu-cp312-cp312-linux_s390x.whl
6f307c2c32d764ffc6ff6893b801fad6d4752f3e67966cb8abf1843427c02604  torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl
4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2  torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl
a8b450c1e58e5800e5b4691dac412f8d2d65a1dc3298166f91596603a3531e6f  torch-2.13.0+cpu-cp312-cp312-win_amd64.whl
fa0762705b933624d59f6823db9ce7ec2e35b3e1e9c319c9db51fbeecfc3e319  torch-2.13.0+cpu-cp312-cp312-win_arm64.whl
966d020354f465672dc7dd10d3a5c6cd17d7eb48620aa1d265b48a1f78f06898  torch-2.13.0+cpu-cp313-cp313-linux_s390x.whl
0b8f7d0423027ae8b90c7977c627f3379f325363a08224dffad9b4b2d684a83d  torch-2.13.0+cpu-cp313-cp313-manylinux_2_28_aarch64.whl
3fbf9c9d1f3c10c2d59d04aca426dee9ccc6ceb32d255c61e93acc3b4f75fae6  torch-2.13.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl
a17ff48608634db245e17e8bb00a9558554a49aeb1e4f5fe6cd039af2a10515b  torch-2.13.0+cpu-cp313-cp313-win_amd64.whl
ac7aaf322be4777765a53bed7264a214dd81b3a1d276b93150515a3c5f75e4b0  torch-2.13.0+cpu-cp313-cp313-win_arm64.whl
dec241fef3984c0d1edadd1f58708e218d4eae881ceef7bc10cf9964d41b68b9  torch-2.13.0+cpu-cp314-cp314-linux_s390x.whl
ca021f9eb2f8345c83fa03e3a04587308afb8df71bd472670b3ece00df58621c  torch-2.13.0+cpu-cp314-cp314-manylinux_2_28_aarch64.whl
d20fa53ee744502fa4c69818a720b05ca0d37abd055d4f6e66cae155114bc691  torch-2.13.0+cpu-cp314-cp314-manylinux_2_28_x86_64.whl
e2e5134decf00e218da62318f3dc5df156231d367871918e91eba95ab0ad43ab  torch-2.13.0+cpu-cp314-cp314-win_amd64.whl
991cc14b39e751122c01f017be6448533989868731cb5eecd1006893d26787c2  torch-2.13.0+cpu-cp314-cp314t-linux_s390x.whl
7b8d26e29bceafbdaa8d63bfe7612f23875b5af2cc07e13f809c3ed890bbe1d8  torch-2.13.0+cpu-cp314-cp314t-manylinux_2_28_aarch64.whl
b222c15a0fc2ce207d1c1a59700b46c8fa6748df1f447ad11e5c870dde0933d9  torch-2.13.0+cpu-cp314-cp314t-manylinux_2_28_x86_64.whl
a43376bd094124ef626bfdd3d4c2c62eacb0b5ddc99776f4a32d4fd16f1f3420  torch-2.13.0+cpu-cp314-cp314t-win_amd64.whl
8eb5002ca81af00ae69b57540f615b58b8ae922b6d4848176b366a52bd2196e6  torch-2.13.0+cpu-cp315-cp315-manylinux_2_28_aarch64.whl
1a3a35229fdc13446b4eab50e7fcf9399ff941e89a3b761497786297a5d8dde5  torch-2.13.0+cpu-cp315-cp315-manylinux_2_28_x86_64.whl
8e109528e6bab044815daebaf71770fbaace3a66ef1c816cb55c875350f78a60  torch-2.13.0+cpu-cp315-cp315t-manylinux_2_28_aarch64.whl
222a6681467cc7f6f05cd3068dfbc603def3a1e46d1d4620c1c8cdf6178bd563  torch-2.13.0+cpu-cp315-cp315t-manylinux_2_28_x86_64.whl
e76f9bcecc52b8ff711239a2f7547d5353df95878ab232f0773c1d95928b92f8  torch-2.13.0-cp311-cp311-macosx_14_0_arm64.whl
2fe228aba290d14b9f31b049be550dbd469c3fd3013d7a19705b30454da97027  torch-2.13.0-cp312-cp312-macosx_14_0_arm64.whl
33449899ce5496c1b84b4853179d94fd102028ae1407314d9fb956bb79e70d09  torch-2.13.0-cp313-cp313-macosx_14_0_arm64.whl
d849b390e07d8d333ce8ecaf91b273c656c598379a19c9acf1318a883f6b391c  torch-2.13.0-cp314-cp314-macosx_14_0_arm64.whl
c28def70706c2f9ecc752574766e8ae4da9b810ab6676b611166761a78a9f1e1  torch-2.13.0-cp314-cp314t-macosx_14_0_arm64.whl
"""  # noqa: E501 (sha256sum lines: digest, two spaces, wheel file name)


def _by_version(listing: str) -> dict[str, dict[str, str]]:
    """{torch version: {wheel file name: SHA-256}} of a sha256sum listing
    (the version is the file name's, without a local "+cpu" label)."""
    table: dict[str, dict[str, str]] = {}
    for line in listing.splitlines():
        digest, name = line.split("  ")
        version = name.split("-")[1].split("+")[0]
        table.setdefault(version, {})[name] = digest
    return table


CPU_WHEELS = _by_version(_CPU_WHEEL_SHA256)

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


def torch_requirement(version: str, *, macos: bool = False) -> str | None:
    """torch.txt: ``torch==<version>+cpu`` and one ``--hash`` line per
    recorded ``+cpu`` wheel of it (Linux, Windows); with ``macos``, the plain
    ``torch==<version>`` and its macOS wheels (macOS has no ``+cpu`` build:
    its one wheel is CPU-only). None when no such wheel is recorded. The pin
    names the local label because uv matches hashes to the exact version
    (``torch==X`` resolving to ``X+cpu`` has, for uv, no hashes)."""
    cpu = f"-{version}+cpu-"
    digests = [
        digest
        for name, digest in sorted(CPU_WHEELS.get(version, {}).items())
        if (cpu in name) != macos
    ]
    if not digests:
        return None
    pin = f"torch=={version}" if macos else f"torch=={version}+cpu"
    return " \\\n".join([pin, *(f"    --hash=sha256:{digest}" for digest in digests)]) + "\n"


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
    macos = argv[1:2] == ["--macos"]
    if argv[:1] == ["requirements"] and len(argv) == 2 + macos:
        kept, torch = split(sys.stdin.read().splitlines(keepends=True))
        if torch is None:
            print("FAIL  the export holds no torch", file=sys.stderr)
            return 1
        pinned = torch_requirement(torch, macos=macos)
        if pinned is None:
            print(
                f"FAIL  no CPU wheel hash recorded for torch {torch}: add every wheel of it from "
                "https://download.pytorch.org/whl/cpu/torch/ to CPU_WHEELS (scripts/cpu_torch.py)",
                file=sys.stderr,
            )
            return 1
        with open(argv[-1], "w", encoding="utf-8") as out:
            out.writelines(kept)
        sys.stdout.write(pinned)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
