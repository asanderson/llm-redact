"""The manifest of a folder of portable NER models (AD11: air-gapped
enclaves).

``llm-redact models pull --to DIR`` writes one self-contained folder per
model into DIR (with its ``llm-redact-model.json`` sidecar) and, beside
them, :data:`MANIFEST_NAME`: the schema version, the llm-redact version
that wrote it and, per model, its backend, Hub id, revision, base model and
every file with its size and SHA-256. ``llm-redact models verify --dir DIR``
checks a folder against it with no network access, so a folder carried
into an enclave can be checked there before the proxy loads it. Model
bundles (llm-redact-pro) extend this manifest rather than define their own:
unknown keys are ignored.

Messages name files, folders and kinds of problems, never file content.
"""

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from llm_redact.detection.model_catalog import (
    HUB_BACKENDS,
    MODEL_ID_RE,
    REVISION_RE,
    SIDECAR_NAME,
)

MANIFEST_NAME = "llm-redact-models.json"
MANIFEST_KIND = "llm-redact-models"
SCHEMA = 1
# The most of a manifest ever read (a model has tens of files).
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_CHUNK = 1024 * 1024
# A path segment of a model folder or a file inside one: what `models pull`
# writes (Hub file names and model-id slugs), never "." or "..".
_SEGMENT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
# The backends whose models a manifest lists: the Hugging Face Hub ones.
BACKENDS = HUB_BACKENDS


def file_digest(path: Path) -> str:
    """The SHA-256 of ``path``'s bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_records(folder: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    """``{"path", "size", "sha256"}`` for each relative POSIX path in
    ``names`` under ``folder``, sorted by path."""
    records = []
    for name in sorted(names):
        path = folder / name
        records.append({"path": name, "size": path.stat().st_size, "sha256": file_digest(path)})
    return records


def folder_files(folder: Path) -> list[str]:
    """Every regular file under ``folder`` as a relative POSIX path, sorted
    (symbolic links count as the files they name)."""
    return sorted(
        path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()
    )


def safe_path(value: object, *, folder: bool = False) -> bool:
    """Whether ``value`` is a relative path of plain segments (a model
    folder's name is one segment): never absolute, never ``..``."""
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    parts = PurePosixPath(value).parts
    if folder and len(parts) != 1:
        return False
    return str(PurePosixPath(*parts)) == value and all(_SEGMENT_RE.fullmatch(p) for p in parts)


@dataclass(frozen=True)
class ManifestModel:
    """One model of a manifest."""

    backend: str
    model_id: str
    revision: str | None
    folder: str
    files: tuple[tuple[str, int, str], ...]
    backbone: str | None = None
    backbone_revision: str | None = None
    onnx: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "revision": self.revision,
            "folder": self.folder,
            "backbone": self.backbone,
            "backbone_revision": self.backbone_revision,
            "onnx": self.onnx,
            "files": [{"path": p, "size": s, "sha256": h} for p, s, h in self.files],
        }


def manifest_json(models: Iterable[ManifestModel], version: str) -> dict[str, Any]:
    """The manifest document for ``models``, written by llm-redact ``version``."""
    return {
        "manifest": MANIFEST_KIND,
        "schema": SCHEMA,
        "llm_redact": version,
        "models": [model.as_json() for model in models],
    }


class ManifestError(ValueError):
    """A manifest that cannot be read as one (names the problem only)."""


def _revision(value: object, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not REVISION_RE.fullmatch(value):
        raise ManifestError(f"{what} must be a 40-character lowercase hex commit id or null")
    return value


def _model(entry: object, index: int) -> ManifestModel:
    where = f"models[{index}]"
    if not isinstance(entry, Mapping):
        raise ManifestError(f"{where} is not an object")
    backend = entry.get("backend")
    if backend not in BACKENDS:
        raise ManifestError(f"{where}.backend must be one of {', '.join(BACKENDS)}")
    model_id = entry.get("model_id")
    if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id):
        raise ManifestError(f"{where}.model_id must be a Hugging Face model id")
    folder = entry.get("folder")
    if not safe_path(folder, folder=True):
        raise ManifestError(f"{where}.folder must be one plain folder name")
    backbone = entry.get("backbone")
    if backbone is not None and (
        not isinstance(backbone, str) or not MODEL_ID_RE.fullmatch(backbone)
    ):
        raise ManifestError(f"{where}.backbone must be a Hugging Face model id or null")
    onnx = entry.get("onnx")
    if onnx is not None and not safe_path(onnx):
        raise ManifestError(f"{where}.onnx must be a relative path inside the folder or null")
    files = entry.get("files")
    if not isinstance(files, list) or not files:
        raise ManifestError(f"{where}.files must be a non-empty list")
    records: list[tuple[str, int, str]] = []
    for position, record in enumerate(files):
        spot = f"{where}.files[{position}]"
        if not isinstance(record, Mapping):
            raise ManifestError(f"{spot} is not an object")
        path, size, sha = record.get("path"), record.get("size"), record.get("sha256")
        if not safe_path(path):
            raise ManifestError(f"{spot}.path must be a relative path inside the folder")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ManifestError(f"{spot}.size must be a byte count")
        if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
            raise ManifestError(f"{spot}.sha256 must be 64 lowercase hex characters")
        records.append((str(path), size, sha))
    if len({path for path, _, _ in records}) != len(records):
        raise ManifestError(f"{where}.files lists a path twice")
    return ManifestModel(
        backend=str(backend),
        model_id=model_id,
        revision=_revision(entry.get("revision"), f"{where}.revision"),
        folder=str(folder),
        files=tuple(records),
        backbone=backbone,
        backbone_revision=_revision(entry.get("backbone_revision"), f"{where}.backbone_revision"),
        onnx=onnx,
    )


def read_manifest(root: Path) -> list[ManifestModel]:
    """The models ``root``'s :data:`MANIFEST_NAME` lists; ManifestError
    when it is missing or is not a manifest of a schema this llm-redact
    reads."""
    from llm_redact.jsonwalk import loads_bounded

    path = root / MANIFEST_NAME
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_MANIFEST_BYTES + 1)
    except FileNotFoundError as exc:
        raise ManifestError(f"no {MANIFEST_NAME} in {root}") from exc
    except OSError as exc:
        raise ManifestError(f"{path} cannot be read ({type(exc).__name__})") from exc
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ManifestError(f"{path} is larger than {MAX_MANIFEST_BYTES} bytes")
    try:
        data = loads_bounded(raw.decode("utf-8"))
    except ValueError as exc:
        raise ManifestError(f"{path} is not a UTF-8 JSON document") from exc
    if not isinstance(data, Mapping) or data.get("manifest") != MANIFEST_KIND:
        raise ManifestError(f"{path} is not an llm-redact models manifest")
    if data.get("schema") != SCHEMA:
        raise ManifestError(
            f"{path}: schema {data.get('schema')!r} is not one this llm-redact reads"
        )
    models = data.get("models")
    if not isinstance(models, list):
        raise ManifestError(f"{path}: models must be a list")
    parsed = [_model(entry, index) for index, entry in enumerate(models)]
    if len({model.folder for model in parsed}) != len(parsed):
        raise ManifestError(f"{path}: two models share a folder")
    return parsed


@dataclass
class FolderCheck:
    """What ``models verify --dir`` found for one model folder."""

    model: ManifestModel
    problems: list[str] = field(default_factory=list)


def check_files(root: Path, model: ManifestModel) -> FolderCheck:
    """Whether ``model``'s folder under ``root`` holds exactly the files the
    manifest lists, each of its size and SHA-256 (an unlisted file is a
    problem too: a loader could read it), and a sidecar naming the same
    model and revision."""
    from llm_redact.detection.model_catalog import SidecarError, read_sidecar

    check = FolderCheck(model)
    folder = root / model.folder
    if not folder.is_dir():
        check.problems.append(f"{model.folder}: the folder is missing")
        return check
    listed = {path for path, _, _ in model.files}
    for path, size, sha in model.files:
        target = folder / path
        if not target.is_file():
            check.problems.append(f"{model.folder}/{path}: missing")
        elif target.stat().st_size != size:
            check.problems.append(f"{model.folder}/{path}: size differs from the manifest")
        elif file_digest(target) != sha:
            check.problems.append(f"{model.folder}/{path}: SHA-256 differs from the manifest")
    for path in folder_files(folder):
        if path not in listed:
            check.problems.append(f"{model.folder}/{path}: not listed in the manifest")
    if SIDECAR_NAME not in listed:
        check.problems.append(f"{model.folder}: the manifest lists no {SIDECAR_NAME}")
        return check
    try:
        identity = read_sidecar(folder)
    except SidecarError as exc:
        check.problems.append(str(exc))
        return check
    if identity is not None and (identity.model_id, identity.revision) != (
        model.model_id,
        model.revision,
    ):
        check.problems.append(
            f"{model.folder}/{SIDECAR_NAME}: names another model or revision than the manifest"
        )
    return check
